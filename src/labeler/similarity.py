#!/usr/bin/env python3
"""Semantic similarity of generated labels to the teacher's labels for the same game.

For each sample file and band: mean cosine(student label, teacher label of the SAME game), and
a floor: mean cosine(student label, teacher label of a DIFFERENT game, same band). The gap
(same - different) is the game-specific part of the similarity; the floor is shared style.

usage: similarity.py TRIAL [TRIAL ...]   (reads results/labeler/samples_<TRIAL>.jsonl)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

R = Path("results/labeler")
EMB = "BAAI/bge-base-en-v1.5"
BANDS = ("terse_1", "terse_2", "short", "medium", "long")


@torch.no_grad()
def embed(texts, tok, model, dev):
    out = []
    for i in range(0, len(texts), 256):
        b = tok(texts[i: i + 256], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state[:, 0]  # bge uses the CLS vector
        out.append(F.normalize(h.float(), dim=-1).cpu())
    return torch.cat(out)


def main():
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(EMB)
    model = AutoModel.from_pretrained(EMB).to(dev).eval()
    for t in sys.argv[1:]:
        rows = [json.loads(l) for l in open(R / f"samples_{t}.jsonl")]
        rows = [r for r in rows if all(r["pred"].get(b) for b in BANDS)]
        res = {"trial": t, "games": len(rows)}
        same_all, diff_all = [], []
        for b in BANDS:
            s = embed([r["pred"][b] for r in rows], tok, model, dev)
            te = embed([r["teacher"][b] for r in rows], tok, model, dev)
            same = (s * te).sum(-1)
            diff = (s * te.roll(1, 0)).sum(-1)  # teacher label of the neighbouring (different) game
            res[b] = {"same_game": round(float(same.mean()), 4), "other_game": round(float(diff.mean()), 4)}
            same_all.append(same.mean())
            diff_all.append(diff.mean())
        res["same_game"] = round(float(torch.stack(same_all).mean()), 4)
        res["other_game"] = round(float(torch.stack(diff_all).mean()), 4)
        res["game_specific_gap"] = round(res["same_game"] - res["other_game"], 4)
        print(json.dumps(res), flush=True)
        with open(R / "similarity.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
