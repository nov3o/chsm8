"""Preference tuning (DPO) on top of the SFT model: make the description matter more.

    python scripts/train/dpo.py configs/dpo.yaml          # chsm8 (one GPU, about 20 minutes)

The pairs come straight from the described corpus, no sampling. For game i with one of its descriptions d_i
(a random band per game and step):
    chosen    game i's moves under d_i
    rejected  game i's moves under d_{i-1}, the description of another game in the batch
Both are the same moves, so the preference only teaches what the text should change. The loss adds two anchors
that keep the model's chess: the SFT loss on the chosen side, and KL(SFT || model) on the moves under d_i.

    loss = -log sigmoid(beta * [(log pi - log ref)(i | d_i) - (log pi - log ref)(i | d_{i-1})])
           + nll_weight * SFT loss + kl_weight * KL
"""
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8 import checkpoint, config, style  # noqa: E402
from chsm8.data import Sides, length_batches, prefetch  # noqa: E402
from chsm8.model import FIELDS, loss_fn  # noqa: E402
from chsm8.text import TextEncoder  # noqa: E402

c = config.load()
ddp = "WORLD_SIZE" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
master = rank == 0
torch.backends.cuda.matmul.allow_tf32 = True
autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

repo, _, filename = c.init.partition(":")
model = checkpoint.load(repo, filename or "model.pt", device).train()
ref = copy.deepcopy(model).eval().requires_grad_(False)  # the SFT model, frozen
encoder = TextEncoder(c.encoder, device=device)
groups = style.param_groups(model, c.lr, c.text_lr)
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=c.weight_decay)
params = [p for g in groups for p in g["params"]]
dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if ddp else model

sides = Sides(c.data, columns=("labels",), load=c.load_in_ram)
train_ids = sides.split["train"]
n_labels = len(sides.get("labels", train_ids[0]))
val_ids = np.sort(np.random.default_rng(0).choice(sides.split["validation"],
                                                  min(c.eval_games, len(sides.split["validation"])), replace=False))


def batches():
    """Endless passes over this rank's training sides, in length-sorted batches (same size rule as SFT)."""
    for epoch in range(10 ** 9):
        r = np.random.default_rng([c.seed, rank, epoch])
        yield from length_batches(sides.n_rows, r.permutation(train_ids)[rank::world], c.games,
                                  c.token_budget // n_labels, r)


def load(ids):
    b = sides.batch(ids)
    return {k: v.pin_memory() if torch.is_tensor(v) and device.type == "cuda" else v for k, v in b.items()}


def pair_text(b, rng):
    """One random description per game, then each game twice: with its own and with the previous game's."""
    own = encoder([sides.get("labels", k)[rng.randrange(n_labels)] for k in b["ids"]])
    return {k: torch.stack([v, v.roll(1, 0)], 1).flatten(0, 1) for k, v in own.items()}


def kl(ref_logits, logits, focal):
    """KL(ref || model) summed over the five heads, mean over the described side's moves."""
    out = 0.0
    for k in FIELDS:
        lq, lp = F.log_softmax(ref_logits[k].float(), -1), F.log_softmax(logits[k].float(), -1)
        out = out + (lq.exp() * (lq - lp)).sum(-1)
    return (out * focal).sum() / focal.sum().clamp(min=1)


def margins(net, b, text):
    """Log-probabilities of each game's moves under (own description, other description) -> (G, 2), and logits."""
    focal = b["focal"].float()
    with autocast():
        logits = net(b["rows"][:, :-1], text, repeat=2)
    lp = style.sequence_logprob(logits, b["rows"].repeat_interleave(2, 0), focal.repeat_interleave(2, 0))
    return lp.view(-1, 2), {k: v[0::2] for k, v in logits.items()}


@torch.no_grad()
def evaluate(tag):
    if master:
        model.eval()
        acc, n, rng = 0.0, 0, random.Random(0)
        for s in range(0, len(val_ids), 32):
            b = style.to(sides.batch(val_ids[s:s + 32]), device)
            lp, _ = margins(model, b, pair_text(b, rng))
            acc, n = acc + float((lp[:, 0] > lp[:, 1]).sum()), n + len(lp)
        out = {"pair_accuracy": acc / n, **style.evaluate(model, encoder, sides, val_ids)}
        print(f"{tag} | val " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
        model.train()
    if ddp:
        dist.barrier()


evaluate("start")
step, t0, window, rng = 0, time.time(), [], random.Random(c.seed + rank)
for b in prefetch(load, batches()):
    if step == c.max_steps:
        break
    mult = style.lr_mult(step / c.max_steps, c.warmup, c.min_lr_ratio)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * mult
    b = style.to(b, device)
    text = pair_text(b, rng)
    with torch.no_grad():
        ref_lp, ref_logits = margins(ref, b, text)
    lp, logits = margins(dmodel, b, text)
    m = (lp[:, 0] - ref_lp[:, 0]) - (lp[:, 1] - ref_lp[:, 1])
    loss = -F.logsigmoid(c.beta * m).mean()
    loss = loss + c.nll_weight * loss_fn(logits, style.focal_targets(b["rows"], b["focal"]), b["occ"], b["pmask"],
                                         c.legal_penalty)
    loss = loss + c.kl_weight * kl(ref_logits, logits, b["focal"].float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
    opt.step()
    step += 1
    window.append(torch.stack([loss.detach(), (lp[:, 0] > lp[:, 1]).float().mean().detach(), m.mean().detach()]))
    if master and step % c.log_every == 0:
        l, acc, mm = torch.stack(window).mean(0).tolist()
        print(f"step {step:,}/{c.max_steps:,} | loss {l:.4f} | pair acc {acc:.3f} | margin {mm:.2f} | "
              f"lr x{mult:.3f} | {time.time() - t0:.0f}s", flush=True)
        window = []
    if step % c.eval_every == 0:
        evaluate(f"step {step:,}")

evaluate("final")
if master:
    os.makedirs(c.out, exist_ok=True)
    checkpoint.save(model, os.path.join(c.out, "model.pt"))
    print(f"done: {os.path.join(c.out, 'model.pt')}", flush=True)
if ddp:
    dist.destroy_process_group()
