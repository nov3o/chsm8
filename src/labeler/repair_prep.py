#!/usr/bin/env python3
"""Repair inputs: split the Gemma distill set into 24 shards and write the repair manifest
(distilled 96 + gemma 24 + apertus 24 tasks).  usage: repair_prep.py"""
import hashlib
from pathlib import Path

I = Path(".")
g = I / "data/repair_gemma"
g.mkdir(parents=True, exist_ok=True)
fs = [open(g / f"shard_{i:03d}.jsonl", "w") for i in range(24)]
for split in ("train", "val", "test"):
    for l in open(I / f"data/labeler_gemma_full/{split}.jsonl"):
        k = l[l.index('"key": "') + 8:].split('"', 1)[0]
        fs[int.from_bytes(hashlib.blake2b(k.encode(), digest_size=4).digest(), "little") % 24].write(l)
for f in fs:
    f.close()
out = I / "labels/repaired"
out.mkdir(parents=True, exist_ok=True)
lines = [f"{I}/data/unlabeled/shard_{i:03d}.jsonl|{I}/labels/distilled_gemma3_1b/labels_{i:03d}.jsonl|{out}/distilled_{i:03d}.jsonl"
         for i in range(96)]
lines += [f"{g}/shard_{i:03d}.jsonl|{out}/gemma_{i:03d}.jsonl" for i in range(24)]
lines += [f"{I}/data/relabel_apertus/shard_{i:03d}.jsonl|{out}/apertus_{i:03d}.jsonl" for i in range(24)]
(I / "data/repair_manifest.txt").write_text("\n".join(lines) + "\n")
print("manifest", len(lines), "tasks", flush=True)
