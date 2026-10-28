"""Data for the label-diversity figure (run from repo root, CPU only, about 1 GB of RAM).

Share of distinct label strings per band in random samples of the 30M label sets, for the Gemma-4-31B teacher and the
distilled Gemma-3-1B labeler, plus a power-law projection (unique = k * n^beta, fitted on samples >= 100k) to 31M.
Writes results/label_uniqueness.json; make_figs.py label_uniqueness draws it.

PYTHONPATH=.pydeps_plot .venv_data/bin/python3.12 techreport/scripts/label_uniqueness_data.py
"""
import array, gzip, hashlib, json

import numpy as np

SRC = "labels/chsm8_labels_30m.jsonl.gz"
OUT = "results/label_uniqueness.json"
BANDS = ["terse_1", "terse_2", "short", "medium", "long"]
LABELERS = ["gemma-4-31b", "gemma-3-1b-distill"]
SIZES = [10_000, 31_000, 100_000, 310_000, 1_000_000, 3_100_000, 10_000_000]


def h(s):
    return int.from_bytes(hashlib.blake2b(s.strip().lower().encode(), digest_size=8).digest(), "little")


hashes = {l: array.array("Q") for l in LABELERS}
with gzip.open(SRC, "rt") as f:
    for i, line in enumerate(f):
        r = json.loads(line)
        if r["labeler"] in hashes:
            hashes[r["labeler"]].extend(h(r["labels"][b]) for b in BANDS)
        if i % 5_000_000 == 0:
            print(i, {l: len(a) // 5 for l, a in hashes.items()}, flush=True)

rng = np.random.default_rng(0)
out = {}
for l, a in hashes.items():
    a = np.frombuffer(a, dtype=np.uint64).reshape(-1, 5)
    a = a[rng.permutation(len(a))]
    out[l] = {"sets": len(a)}
    for j, b in enumerate(BANDS):
        ns = [n for n in SIZES if n <= len(a)]
        us = [int(len(np.unique(a[:n, j]))) for n in ns]
        x = np.log([n for n in ns if n >= 100_000]); y = np.log([u for n, u in zip(ns, us) if n >= 100_000])
        beta, logk = np.polyfit(x, y, 1)
        out[l][b] = {"n": ns, "unique": us, "beta": float(beta), "proj_31m": float(np.exp(logk) * 31e6 ** beta)}
    print(l, {b: round(100 * out[l][b]["proj_31m"] / 31e6, 1) for b in BANDS}, flush=True)
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
