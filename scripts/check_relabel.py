#!/usr/bin/env python3
"""Contradiction check (labeler grounding rules, same as label_correct) on the relabelled Apertus records.
Writes labels/distilled_relabel_apertus/check.jsonl ({key, correct:[5 x 0/1]}) and prints a summary."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, "src/labeler")
import repair_labels as R  # noqa: E402
I = Path(".")
BANDS = ["terse_1", "terse_2", "short", "medium", "long"]

def one(i):
    user = {}
    for l in open(I / f"data/relabel_apertus96/shard_{i:03d}.jsonl"):
        d = json.loads(l); user[d["key"]] = d["user"]
    out, nbad, nrec, nset = [], [0] * 5, 0, 0
    for l in open(I / f"labels/distilled_relabel_apertus/labels_{i:03d}.jsonl"):
        d = json.loads(l); lab = d.get("labels") or {}
        if not all((lab.get(b) or "").strip() for b in BANDS):
            out.append({"key": d["key"], "correct": None}); continue
        bad = set(R.bad_bands(user[d["key"]], lab))
        c = [0 if b in bad else 1 for b in BANDS]
        out.append({"key": d["key"], "correct": c}); nrec += 1; nset += bool(bad)
        for j in range(5): nbad[j] += 1 - c[j]
    return out, nbad, nrec, nset

if __name__ == "__main__":
    with Pool(48) as p:
        res = p.map(one, range(96))
    nbad, nrec, nset, nmiss = [0] * 5, 0, 0, 0
    with open(I / "labels/distilled_relabel_apertus/check.jsonl", "w") as f:
        for out, b, r, s in res:
            for o in out:
                f.write(json.dumps(o) + "\n"); nmiss += o["correct"] is None
            nbad = [x + y for x, y in zip(nbad, b)]; nrec += r; nset += s
    print(json.dumps({"records": nrec, "incomplete": nmiss, "sets_with_any_bad_band": nset, "set_bad_rate": nset / nrec,
                      "bad_band_rate": sum(nbad) / (5 * nrec), "bad_per_band": dict(zip(BANDS, nbad))}))
