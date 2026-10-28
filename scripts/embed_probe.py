#!/usr/bin/env python3
"""Does a text embedder keep fine details (player, Elo, opening, speed) when they come wrapped in templates?

For each attribute we render many prompts with the real meta_templates (random template per prompt), alone ("bare")
and appended to a real LLM style description ("mixed", the detail is a small part of a long sentence). Metrics:
  xtpl_acc  nearest neighbour must use a DIFFERENT template; accuracy that it has the same value
            (template-invariant retrieval of the detail; chance = 1/#values)
  probe_acc logistic regression on held-out templates (linear readability)
  elo: also within-100 accuracy of the cross-template neighbour.
    python3 scripts/embed_probe.py --encoders 'Qwen/Qwen3-Embedding-0.6B:last' ... --out results/embed_probe.jsonl
"""
import argparse, json, random, re, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import meta_templates as M  # noqa: E402

INSTRUCT = "Instruct: Represent this chess playing-style request, keeping player names, ratings, openings and time controls\nQuery: "


def items(rng, styles, n_per):
    """(attr, value, text, template_key) tuples."""
    out = []
    players = sorted(set(pq.read_table(ROOT / "data/sft_players/index.parquet", columns=["player"]).column("player").to_pylist()))
    for p in players:
        for _ in range(n_per):
            name = rng.choice(M.player_names(p)); t = rng.choice(M.PLAYER)
            out.append(("player", p, t.format(p=name), t))
    for elo in range(1200, 2900, 100):
        for _ in range(n_per):
            if rng.random() < 0.7:
                t = rng.choice(M.ELO_NUM); out.append(("elo", elo, t.format(elo=elo), t))
            else:
                t = rng.choice(M.ELO_TITLE[M.elo_bucket(elo)]); out.append(("elo", elo, t, "title:" + t))
    for sp in ("bullet", "blitz", "rapid", "classical"):
        base, inc = {"bullet": (60, 0), "blitz": (180, 2), "rapid": (600, 5), "classical": (1800, 20)}[sp]
        for _ in range(n_per * 3):
            txt = M.render_tc({"tc_base": base, "tc_increment": inc}, rng)
            out.append(("speed", sp, txt, re.sub(r"\d+(\.\d+)?|bullet|blitz|rapid|classical", "#", txt)))
    ops = [o for o in OPENINGS]
    for o in ops:
        for _ in range(n_per):
            t = rng.choice(M.OPENING); name = o
            if rng.random() < 0.3:  # "the Sicilian" as users often say
                name = re.sub(r" (Defense|Opening|Game|System)$", "", o)
            out.append(("opening", o, t.format(o=name), t))
    mixed = [(a, v, f"{rng.choice(styles)}, {t}", k) for a, v, t, k in out]
    return out, mixed


OPENINGS = ["Sicilian Defense", "French Defense", "Caro-Kann Defense", "Ruy Lopez", "Italian Game", "Queen's Gambit Declined",
            "King's Indian Defense", "Nimzo-Indian Defense", "English Opening", "Scandinavian Defense", "Pirc Defense",
            "Grunfeld Defense", "Slav Defense", "Dutch Defense", "London System", "Scotch Game", "Alekhine Defense",
            "Benoni Defense", "Catalan Opening", "Petrov's Defense"]


def load(name, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if "t5gemma" in name:
        from transformers import T5GemmaEncoderModel
        m = T5GemmaEncoderModel.from_pretrained(name, dtype=torch.bfloat16, is_encoder_decoder=False)
    else:
        m = AutoModel.from_pretrained(name, dtype=torch.bfloat16, trust_remote_code=True)
    return tok, m.to(device).eval()


@torch.no_grad()
def embed(tok, m, texts, pool, device, bs=256):
    vs = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], padding=True, truncation=True, max_length=192, return_tensors="pt").to(device)
        h = m(**enc).last_hidden_state.float(); mask = enc["attention_mask"].bool()
        v = (h[:, 0] if pool == "cls" else h[torch.arange(h.shape[0], device=device), mask.sum(1) - 1] if pool == "last"
             else (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))
        vs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(vs).numpy()


def metrics(E, its):
    res = {}
    for attr in ("player", "elo", "speed", "opening"):
        idx = [i for i, x in enumerate(its) if x[0] == attr]
        X = E[idx]; vals = [its[i][1] for i in idx]; keys = [its[i][3] for i in idx]
        uv = sorted(set(vals), key=str); y = np.array([uv.index(v) for v in vals]); k = np.array([hash(x) for x in keys])
        S = X @ X.T; S[k[:, None] == k[None, :]] = -9  # neighbour must use a different template
        nn = S.argmax(1)
        r = {"xtpl_acc": float((y[nn] == y).mean()), "chance": 1 / len(uv)}
        if attr == "elo":
            ev = np.array(vals); r["xtpl_within100"] = float((np.abs(ev[nn] - ev) <= 100).mean())
        # linear probe, split by template so test templates are unseen
        ukeys = sorted(set(keys)); rng = np.random.default_rng(0); test_t = set(rng.choice(len(ukeys), len(ukeys) // 3, replace=False))
        te = np.array([ukeys.index(x) in test_t for x in keys])
        if te.sum() and (~te).sum():
            Xt = torch.tensor(X); W = torch.zeros(X.shape[1], len(uv), requires_grad=True); bb = torch.zeros(len(uv), requires_grad=True)
            opt = torch.optim.LBFGS([W, bb], max_iter=300); yt = torch.tensor(y); tr = torch.tensor(~te)
            def clo():
                opt.zero_grad(); l = torch.nn.functional.cross_entropy(Xt[tr] @ W * 10 + bb, yt[tr]) + 1e-3 * W.pow(2).sum(); l.backward(); return l
            opt.step(clo)
            r["probe_acc"] = float(((Xt[~tr] @ W * 10 + bb).argmax(1) == yt[~tr]).float().mean())
        res[attr] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True, help="name:pool[:instruct]")
    ap.add_argument("--n-per", type=int, default=24)
    ap.add_argument("--out", default="results/embed_probe.jsonl")
    a = ap.parse_args()
    rng = random.Random(0)
    lab = pq.read_table(ROOT / "data/sft30m/index.parquet", columns=["labels"]).slice(0, 20000).column("labels").to_pylist()
    styles = [l[3] for l in lab if l and len(l) > 3]  # medium-band real descriptions
    bare, mixed = items(rng, styles, a.n_per)
    dev = torch.device("cuda")
    for spec in a.encoders:
        name, pool, *ins = spec.split(":")
        pre = INSTRUCT if ins else ""
        try:
            tok, m = load(name, dev)
        except Exception as e:
            print(f"[skip] {spec}: {e}", flush=True); continue
        row = {"encoder": spec}
        for tag, its in (("bare", bare), ("mixed", mixed)):
            row[tag] = metrics(embed(tok, m, [pre + x[2] for x in its], pool, dev), its)
        print(json.dumps(row), flush=True)
        with open(ROOT / a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
