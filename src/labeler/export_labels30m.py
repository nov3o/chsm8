#!/usr/bin/env python3
"""labels/chsm8_labels_30m.jsonl.gz -> HF layout data/<month>/part-00000.parquet (normalized: joins on
(month, row_idx) to LegumMagister/chsm8-pre-training; prompts rebuilt by scripts/build_prompts.py).
usage: export_labels30m.py IN.jsonl.gz OUT_DIR"""
import collections
import gzip
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

src, out = sys.argv[1], Path(sys.argv[2])
by_month = collections.defaultdict(list)
c = collections.Counter()
with gzip.open(src, "rt") as f:
    for l in f:
        r = json.loads(l)
        m, i, s = r["key"].split(":")
        by_month[m].append({"month": m, "row_idx": int(i), "side": s, "labeler": r["labeler"],
                            "labels": r["labels"], "correct": r["correct"], "clean": all(r["correct"])})
        c[r["labeler"]] += 1
for m, rows in sorted(by_month.items()):
    rows.sort(key=lambda r: (r["row_idx"], r["side"]))
    (out / "data" / m).mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "data" / m / "part-00000.parquet", compression="zstd")
print(json.dumps({"rows": sum(c.values()), "months": len(by_month), **c}), flush=True)
