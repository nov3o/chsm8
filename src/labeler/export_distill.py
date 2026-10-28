#!/usr/bin/env python3
"""Distill set for the HF release: data/labeler_gemma_full/{train,val,test}.jsonl -> parquet shards with
the per-band `correct` mask and `clean` flag (grounding.RULES vs the measured card).
usage: export_distill.py SRC_DIR OUT_DIR"""
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounding as G  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "data").mkdir(parents=True, exist_ok=True)
ROWS = 250_000
for split in ("train", "validation", "test"):
    f = src / f"{'val' if split == 'validation' else split}.jsonl"
    buf, part, n, n_clean = [], 0, 0, 0

    def flush():
        global part
        pq.write_table(pa.Table.from_pylist(buf), out / "data" / f"{split}-{part:05d}.parquet", compression="zstd")
        part += 1
        buf.clear()

    for l in open(f):
        r = json.loads(l)
        fa = G.facts(r["user"])
        correct = [0 if any(re.search(rx, (r["labels"][b] or "").lower()) and cond(fa) for _, rx, cond in G.RULES) else 1
                   for b in G.BANDS]
        buf.append({"key": r["key"], "month": r["month"], "row_idx": r["row_idx"], "side": r["side"], "elo": r.get("elo"),
                    "user": r["user"], "labels": {b: r["labels"][b] for b in G.BANDS},
                    "correct": correct, "clean": all(correct)})
        n += 1
        n_clean += all(correct)
        if len(buf) == ROWS:
            flush()
    if buf:
        flush()
    print(f"{split}: {n} rows, {n_clean / n:.1%} clean, {part} files", flush=True)
