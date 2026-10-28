#!/usr/bin/env python3
"""Style DPO on top of an SFT model (sft_ablate.py --save .pt).

Pairs come straight from the labeled corpus, no sampling: for game i with style label p_i
(one random LLM band per game per step),
  chosen   = game i's focal-side moves under p_i         (the label was written for it)
  rejected = game i+1's focal-side moves under p_i       (a game with a different style)
with i+1 taken inside the length-sorted batch, so pairs have similar length. Sequence
log-prob = sum over focal moves of log p(kind)+log p(src)+log p(dst)+log p(piece)+log p(promo).
Loss = -logsigmoid(beta * [(pi_w - ref_w) - (pi_l - ref_l)]) + nll_weight * CE(chosen),
reference = frozen copy of the SFT model. Trains the same parameters SFT trained.

Eval (val, fixed subset): pair accuracy / reward margin, and sft_ablate.evaluate CE
(matched / mismatched / zero) so the style gap and plain move prediction are both tracked.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_ablate as S  # noqa: E402

LOST = None
COLS = ("kind", "src", "dst", "piece", "promo")


def move_logp(logits, rows):
    """(G, T-1) log-prob of each next move (sum over the 5 heads)."""
    tgt = rows[:, 1:]
    lp = 0.0
    for c, k in enumerate(COLS):
        lp = lp + F.log_softmax(logits[k][:, :-1].float(), -1).gather(-1, tgt[..., c:c + 1]).squeeze(-1)
    return lp


def seq_logp(logits, rows, focal, norm):
    """(G,) summed (or per-move mean) log-prob of each row's focal moves."""
    f = focal.float()
    s = (move_logp(logits, rows) * f).sum(1)
    return s / f.sum(1).clamp(min=1) if norm else s


def style_mask(d, focal, frac):
    """Per row, the focal moves with the largest |d| (top `frac` of that row's focal moves; at least 1)."""
    f = focal.bool(); a = d.abs().masked_fill(~f, -1.0)
    k = (f.sum(1).float() * frac).ceil().clamp(min=1).long()
    thr = a.sort(1, descending=True).values.gather(1, (k - 1).unsqueeze(1))
    return (a >= thr) & f


def one_band_cond(text, cond, b, n_llm, rng, band=None):
    """Condition for one random LLM band per game: (G,...) dict, plus the rolled version where
    row j carries game j-1's label (so row i+1 under the roll = game i+1 with label p_i)."""
    G = len(b["texts"])
    li = [rng.randrange(n_llm) for _ in range(G)] if band is None else [band] * G
    if "emb" in b:  # precomputed pooled label vectors
        emb = b["emb"].to(cond.gate.device, non_blocking=True).float()
        pooled = emb[torch.arange(G, device=emb.device), torch.tensor(li, device=emb.device)]
        ctx, mask = pooled.unsqueeze(1), torch.ones(G, 1, dtype=torch.bool, device=pooled.device)
    else:  # online frozen encoder (per-token models: all token states are keys)
        ctx, mask, pooled = text([b["texts"][g][li[g]] for g in range(G)])
    c = cond.prepare(ctx, mask, pooled); c["gate"] = cond.gate
    if HARD_NEG:  # row i gets the most similar OTHER label in the batch (by text embedding), not a random one
        e = F.normalize(pooled.float(), dim=-1); sim = e @ e.T; sim.fill_diagonal_(-2.0)
        perm = sim.argmax(1)
        r = {k: (v[perm] if k in ("ctx", "film", "mask") and v is not None else v) for k, v in c.items()}
    else:
        r = {k: (v.roll(1, 0) if k in ("ctx", "film", "mask") and v is not None else v) for k, v in c.items()}
    return c, r


PAIR_MODE = "game"
HARD_NEG = False
BANDS = "one"  # one: a random LLM band per game per step; long: always the longest band; all: every band of the game
GRAN = "game"  # game: preference on the summed log-prob of the game's moves; move: a separate preference per focal move
STYLE_FRAC = 0.0  # >0: DPO log-probs only over each row's style moves (largest ref |log p(own label) - log p(other label)|)


def kl_to_ref(logits, ref_logits, focal):  # focal may be any (G, T-1) move mask
    """Mean over focal moves of sum over the 5 heads of KL(ref || policy) at each next-move prediction."""
    f = focal.float()  # same alignment as seq_logp
    kl = 0.0
    for k in COLS:
        lp = F.log_softmax(logits[k][:, :-1].float(), -1); lq = F.log_softmax(ref_logits[k][:, :-1].float(), -1)
        kl = kl + (lq.exp() * (lq - lp)).sum(-1)
    return (kl * f).sum() / f.sum().clamp(min=1)


def pair_logps(chess, cond_mod, text, b, rows, focal, n_shared, n_frozen, n_llm, rng, norm, smask=None):
    """smask=None on the reference pass with STYLE_FRAC>0: computes the style mask and returns it as a 4th value.
    BANDS=all: pairs for every band, concatenated band-major (G*n_llm rows); NLL/KL use one random band's logits.
    GRAN=move: w/lw are (rows, T-1) per-move log-probs (masked to focal moves) instead of per-game sums."""
    bands = {"one": [None], "long": [n_llm - 1], "all": list(range(n_llm))}[BANDS]
    keep = rng.randrange(len(bands))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        h = S.forward_shared(chess, rows, n_shared, n_frozen)
    W, LW, LM = [], [], None
    for i, band in enumerate(bands):
        c, r = one_band_cond(text, cond_mod, b, n_llm, rng, band)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            lm = S.forward_top(chess, h, rows, n_shared, c)
            lr = S.forward_top(chess, h, rows, n_shared, r)
        pm, pr = move_logp(lm, rows), move_logp(lr, rows)
        made = STYLE_FRAC > 0 and smask is None
        if made:  # moves where this row's own label vs the rolled/hard-neg label changes the reference most
            smask = style_mask(pm - pr, focal, STYLE_FRAC)
        f = (smask if STYLE_FRAC > 0 else focal).float()
        if GRAN == "move":
            w, lw = pm * f, pr * f
        else:
            w, lw = (pm * f).sum(1), (pr * f).sum(1)             # game i | p_i ; row j under p_{j-1}
            if norm:
                w, lw = w / f.sum(1).clamp(min=1), lw / f.sum(1).clamp(min=1)
        W.append(w); LW.append(lw)
        if i == keep:
            LM = lm
        del lm, lr
    out = pair_out(torch.cat(W), torch.cat(LW), LM)
    return out + (smask,) if made else out


def pair_out(w, lw, lm):
    if PAIR_MODE == "label":                              # game i | p_{i-1}: same game, wrong label
        return w, lw, lm
    l = lw.roll(-1, 0)                                    # game i+1 | p_i
    if PAIR_MODE == "both":
        return torch.cat([w, w]), torch.cat([l, lw]), lm
    return w, l, lm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--init", required=True, help="SFT model .pt (sft_ablate --save)")
    ap.add_argument("--data", default="data/sft30m")
    ap.add_argument("--label-emb", default="data/sft30m/emb/Qwen__Qwen3-Embedding-0.6B_last_d256.f16")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-weight", type=float, default=0.2, help="CE on chosen (keeps move prediction anchored)")
    ap.add_argument("--pair-mode", choices=("game", "label", "both"), default="game",
                    help="rejected = other game under the same label (game), same game under another game's label (label), or both")
    ap.add_argument("--wins-only", action="store_true", help="DPO pairs only where the labelled game was not a loss")
    ap.add_argument("--hard-neg", action="store_true", help="label pairs: wrong label = most similar other label in the batch")
    ap.add_argument("--style-frac", type=float, default=0.0, help="DPO only on the top FRAC style moves of each row (by ref label gap)")
    ap.add_argument("--nll-rest", action="store_true", help="with --style-frac: NLL anchor only on the non-style moves")
    ap.add_argument("--loss", choices=("dpo", "ipo"), default="dpo", help="ipo: (beta*margin - 1)^2 / beta^2 style squared loss (bounded push)")
    ap.add_argument("--bands", choices=("one", "long", "all"), default="one", help="which LLM label(s) per game form the pairs")
    ap.add_argument("--granularity", choices=("game", "move"), default="game",
                    help="move: one preference per focal move (label pairs only) instead of one per game")
    ap.add_argument("--kl-rest", action="store_true", help="with --style-frac: KL guard only on the non-style moves")
    ap.add_argument("--kl-weight", type=float, default=0.0, help="KL(SFT || policy) on prompted moves (strength guard)")
    ap.add_argument("--pure", action="store_true", help="pairs only from label sets with no contradicted band")
    ap.add_argument("--labelers", default="", help="comma list of index `labeler` values (train and val); empty = all")
    ap.add_argument("--freeze-dummy", action="store_true", help="freeze cross-attn v/out biases (keeps the no-text policy fixed)")
    ap.add_argument("--norm", action="store_true", help="per-move mean log-prob instead of sum")
    ap.add_argument("--lr", type=float, default=2e-7)
    ap.add_argument("--cond-lr", type=float, default=1e-4)
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--token-budget", type=int, default=98304)
    ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--time-budget", type=float, default=3600)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-games", type=int, default=4000)
    ap.add_argument("--save-dir", default="checkpoints/dpo")
    ap.add_argument("--results", default="results/dpo.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=0.0,
                    help=">0: train for this many passes over the train split (each game once per pass, sharded over ranks); "
                         "LR schedule by games seen instead of --time-budget")
    ap.add_argument("--save-every", type=int, default=0, help="also save the model every N steps (rank 0, overwrites)")
    ap.add_argument("--milestones", default="", help="with --epochs: comma list of progress fractions (e.g. 0.25,0.5,0.75) at which "
                                                      "to evaluate on val and save a separate checkpoint <trial>_p<pct>.pt")
    a = ap.parse_args()
    RANK, WORLD = 0, 1
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:  # data parallel (srun/torchrun env): grads averaged by hand below
        torch.distributed.init_process_group("nccl")
        RANK, WORLD = torch.distributed.get_rank(), torch.distributed.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    global PAIR_MODE, HARD_NEG, LOST, STYLE_FRAC, BANDS, GRAN; PAIR_MODE = a.pair_mode; HARD_NEG = a.hard_neg; STYLE_FRAC = a.style_frac
    BANDS, GRAN = a.bands, a.granularity
    assert not (GRAN == "move" and a.pair_mode != "label"), "--granularity move needs --pair-mode label (same moves under both labels)"
    assert not (STYLE_FRAC > 0 and BANDS == "all"), "--style-frac with --bands all is not supported"
    assert not (a.hard_neg and a.pair_mode != "label"), "--hard-neg is for --pair-mode label"
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    if dev.type == "cpu":
        S.USE_FLASH = False  # no flash-attn kernel on CPU (as in sft_ablate.py)

    sd = torch.load(a.init, map_location="cpu", weights_only=False)
    sa = SimpleNamespace(**sd["args"])
    sa.data, sa.label_emb, sa.eval_games, sa.compile = a.data, a.label_emb, a.eval_games, False
    attrs = [x for x in sa.meta_labels.split(",") if x]
    data = S.SFTData(Path(a.data), sa.n_llm_labels, attrs, sa.max_rows)
    S.apply_filters(data, Path(a.data), a.pure, a.labelers, a.labelers)
    if a.wins_only:
        po = pq.read_table(Path(a.data) / "index.parquet", columns=["player_outcome"]).column("player_outcome").to_pylist()
        LOST = (np.asarray(po, dtype=object) == "loss")[data.orig]
    if a.label_emb and sa.encoder_kind == "embed":
        data.load_label_emb(Path(a.label_emb), sa.emb_dim)
    S.LABEL_W = torch.tensor([1.0] * data.n_llm + [sa.meta_weight] * len(attrs))
    chess, text, cond, _, info = S.build(sa, dev)
    if "opt" in sd and getattr(sa, "freeze_dummy", False) and "frozen_dummy" not in sd:
        # SFT training snapshot: its pinned no-text term = the pretrained value, i.e. the freshly built model's
        with torch.no_grad():
            sd["frozen_dummy"] = {}
            for i, l in enumerate(chess.layers):
                if hasattr(l, "cross_attn"):
                    ca = l.cross_attn; z = torch.zeros(1, 1, ca.v_proj.in_features, device=ca.v_proj.weight.device, dtype=ca.v_proj.weight.dtype)
                    sd["frozen_dummy"][i] = ca.out_proj(ca.v_proj(z)).detach().clone()
    sd["chess"] = {k.replace("._orig_mod", ""): v for k, v in sd["chess"].items()}
    chess.load_state_dict(sd["chess"]); cond.load_state_dict(sd["cond"])
    for i, t in sd.get("frozen_dummy", {}).items():  # SFT run with --freeze-dummy: keep its pinned no-text term
        chess.layers[int(i)].frozen_dummy = t.to(dev)
    text.n_llm = data.n_llm
    n_shared, n_frozen = info["n_shared"], info["n_frozen"]
    if a.freeze_dummy:  # the no-text path adds out_proj(v_proj(0)); pin it to the SFT value so it never moves
        with torch.no_grad():
            for l in chess.layers:
                if hasattr(l, "cross_attn") and getattr(l, "frozen_dummy", None) is None:
                    ca = l.cross_attn; z = torch.zeros(1, 1, ca.v_proj.in_features, device=ca.v_proj.weight.device, dtype=ca.v_proj.weight.dtype)
                    l.frozen_dummy = ca.out_proj(ca.v_proj(z)).detach().clone()
    ref_chess, ref_cond = copy.deepcopy(chess).requires_grad_(False).eval(), copy.deepcopy(cond).requires_grad_(False).eval()
    params_cond = [p for p in cond.parameters()] + [p for l in chess.layers for n, p in l.named_parameters()
                                                    if p.requires_grad and n.startswith(("cross_attn.", "n2."))]
    params_cond = [p for p in params_cond if p.requires_grad]
    ids_c = {id(p) for p in params_cond}
    params_body = [p for p in chess.parameters() if p.requires_grad and id(p) not in ids_c]
    groups = [{"params": params_cond, "lr": a.cond_lr}, {"params": params_body, "lr": a.lr}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)

    def evaluate():
        if RANK != 0:  # rank 0 evaluates, the others wait
            torch.distributed.barrier(); return {}
        chess.eval(); cond.eval()
        r = random.Random(0)
        ids = np.sort(np.random.default_rng(0).choice(data.split["val"], size=min(a.eval_games, len(data.split["val"])), replace=False))
        ids = ids[np.argsort(data.n_rows[ids], kind="stable")]
        acc = marg = n = 0
        with torch.no_grad():
            for s0 in range(0, len(ids), 32):
                b = data.batch(ids[s0:s0 + 32])
                rows, focal = b["rows"].to(dev), b["focal"].to(dev)
                sd0 = r.random()
                rw, rl, _, *sm = pair_logps(ref_chess, ref_cond, text, b, rows, focal, n_shared, len(chess.layers), data.n_llm, random.Random(sd0), a.norm)
                w, l, _ = pair_logps(chess, cond, text, b, rows, focal, n_shared, len(chess.layers), data.n_llm, random.Random(sd0), a.norm, smask=sm[0] if sm else None)
                m = (w - rw) - (l - rl)
                if m.dim() == 2:  # move granularity: compare per-game sums
                    w, l, m = w.sum(1), l.sum(1), m.sum(1)
                acc += float((w - l > 0).float().sum()); marg += float(m.sum()); n += len(m)
        ce = S.evaluate(sa, chess, text, cond, data, dev, n_shared, "val", data.n_labels)
        chess.train(); cond.train()
        if WORLD > 1:
            torch.distributed.barrier()
        return {"val_pair_acc": acc / n, "val_reward_margin": marg / n, **ce}

    def save(name=None):
        out = Path(a.save_dir) / f"{name or a.trial}.pt"; out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "sft-ablate-v1", "args": sd["args"], "dpo_args": vars(a), "info": info,
                    "chess": chess.state_dict(), "cond": cond.state_dict(),
                    "frozen_dummy": {i: l.frozen_dummy.cpu() for i, l in enumerate(chess.layers) if getattr(l, "frozen_dummy", None) is not None}},
                   str(out) + ".tmp")
        os.replace(str(out) + ".tmp", out)

    base = evaluate()
    if RANK == 0: print("[base]", json.dumps(base), flush=True)
    planned = a.epochs * len(data.split["train"]) if a.epochs > 0 else 0
    stream = S.Prefetcher(data, S.length_bucketed_batches(data, "train", a.games, a.token_budget,
                                                          np.random.default_rng(a.seed * 1000 + RANK),
                                                          shard=(RANK, WORLD) if planned else None))
    t_start, measured, step, hist, games_seen = time.time(), 0.0, 0, [], 0
    params_all = [p for g in groups for p in g["params"]]
    miles = sorted(float(x) for x in a.milestones.split(",") if x) if planned else []
    while step < a.max_steps and (games_seen < planned if planned else measured < a.time_budget):
        frac = games_seen / planned if planned else measured / a.time_budget
        mult = frac / a.warmup_frac if frac < a.warmup_frac else 0.5 * (1 + math.cos(math.pi * (frac - a.warmup_frac) / (1 - a.warmup_frac)))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        b = stream.get(); t0 = time.time()
        rows, focal = b["rows"].to(dev, non_blocking=True), b["focal"].to(dev, non_blocking=True)
        occ, pmask = b["occ"].to(dev, non_blocking=True), b["pmask"].to(dev, non_blocking=True)
        seed = rng.random()
        with torch.no_grad():
            rw, rl, lm_ref, *sm = pair_logps(ref_chess, ref_cond, text, b, rows, focal, n_shared, len(chess.layers), data.n_llm, random.Random(seed), a.norm)
        sm = sm[0] if sm else None
        w, l, lm = pair_logps(chess, cond, text, b, rows, focal, n_shared, n_frozen, data.n_llm, random.Random(seed), a.norm, smask=sm)
        m = (w - rw) - (l - rl)
        if GRAN == "move":  # per-move preference, averaged over focal moves (w etc. are (rows, T-1), zero off-focal)
            fm = focal.float().repeat(m.shape[0] // focal.shape[0], 1)
            loss = -(F.logsigmoid(a.beta * m) * fm).sum() / fm.sum().clamp(min=1)
            w, l, m = w.sum(1), l.sum(1), m.sum(1)
        elif a.wins_only:  # style preference only from games the described player did not lose (NLL/KL still on all)
            ok = torch.from_numpy(~LOST[b["idx"]]).to(dev).float()
            ok = torch.cat([ok, ok]) if PAIR_MODE == "both" else ok
            loss = -(F.logsigmoid(a.beta * m) * ok).sum() / ok.sum().clamp(min=1)
        elif a.loss == "ipo":  # IPO: regress the log-ratio margin to 1/(2 beta) instead of pushing it to infinity
            loss = ((m - 1 / (2 * a.beta)) ** 2).mean()
        else:
            loss = -F.logsigmoid(a.beta * m).mean()
        if a.nll_weight > 0:
            nf = (focal.bool() & ~sm) if (a.nll_rest and sm is not None) else focal
            loss = loss + a.nll_weight * S.focal_loss(lm, rows, nf, occ, pmask, sa.illegal_penalty)
        if a.kl_weight > 0:  # keep prompted play close to the SFT policy: KL(ref || pi) per head, game i under its own label
            kf = (focal.bool() & ~sm) if (a.kl_rest and sm is not None) else focal
            loss = loss + a.kl_weight * kl_to_ref(lm, lm_ref, kf)
        opt.zero_grad(set_to_none=True); loss.backward()
        if WORLD > 1:  # average grads over ranks (same parameter set everywhere), one flat all-reduce
            from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params_all]
            flat = _flatten_dense_tensors(grads); torch.distributed.all_reduce(flat); flat /= WORLD
            for p, g_ in zip(params_all, _unflatten_dense_tensors(flat, grads)):
                p.grad = g_
        if WORLD > 1:  # exact global count (batch sizes differ per rank): every rank must agree on milestones / the stop
            ng = torch.tensor([float(len(b["idx"]))], device=dev); torch.distributed.all_reduce(ng); games_seen += int(ng.item())
        else:
            games_seen += len(b["idx"])
        gn = float(torch.nn.utils.clip_grad_norm_(params_all, 1.0))
        opt.step(); torch.cuda.synchronize() if dev.type == "cuda" else None
        step += 1; measured += time.time() - t0
        if WORLD > 1 and not planned:  # time budget: ranks must agree on the clock or they stop on different steps and hang
            mt = torch.tensor([measured], device=dev); torch.distributed.all_reduce(mt, op=torch.distributed.ReduceOp.MAX); measured = float(mt.item())
        hist.append((float(loss), float((w - l > 0).float().mean()), float(m.mean())))
        if step % 25 == 0 and RANK == 0:
            h = np.mean(hist[-25:], 0)
            print(json.dumps({"step": step, "loss": round(h[0], 4), "pair_acc": round(h[1], 3), "margin": round(h[2], 3),
                              "gnorm": round(gn, 3), "lr_mult": round(mult, 3), "train_s": round(measured),
                              "games": games_seen, "games_per_s": round(games_seen / max(measured, 1e-9))}), flush=True)
        if step % a.eval_every == 0:
            ev = evaluate()
            if RANK == 0: print("[eval]", step, json.dumps(ev), flush=True)
        if a.save_every and step % a.save_every == 0 and RANK == 0:
            save()
        if miles and games_seen >= miles[0] * planned:  # same games_seen on every rank -> all ranks enter together
            pct = round(miles.pop(0) * 100); ev = evaluate()
            if RANK == 0:
                print("[milestone]", pct, json.dumps({"step": step, "games": games_seen, "train_s": round(measured), **ev}), flush=True)
                with open(a.results.replace(".jsonl", "_curve.jsonl"), "a") as f:
                    f.write(json.dumps({"trial": a.trial, "pct": pct, "step": step, "games": games_seen, "train_s": measured, **ev}) + "\n")
                save(f"{a.trial}_p{pct}")
    final = evaluate()
    if RANK != 0:
        torch.distributed.destroy_process_group(); return
    res = {"trial": a.trial, **vars(a), "world": WORLD, "steps": step, "train_seconds": measured, "wall_seconds": time.time() - t_start,
           "games_seen": games_seen, "games_per_s": games_seen / max(measured, 1e-9),
           **{f"base_{k}": v for k, v in base.items()}, **final}
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    with open(a.results, "a") as f:
        f.write(json.dumps(res) + "\n")
    print("RESULT " + json.dumps(res), flush=True)
    save()
    if WORLD > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
