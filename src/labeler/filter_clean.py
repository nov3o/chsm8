#!/usr/bin/env python3
"""Clean distill set: keep only training label sets with no band the card contradicts (grounding.RULES).
val/test are copied unfiltered so scores stay comparable with the original run.
usage: filter_clean.py SRC_DIR DST_DIR"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounding as G  # noqa: E402

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
kept = dropped = 0
with open(dst / "train.jsonl", "w") as fo:
    for l in open(src / "train.jsonl"):
        r = json.loads(l)
        f = G.facts(r["user"])
        if any(re.search(rx, (r["labels"][b] or "").lower()) and cond(f) for b in G.BANDS for _, rx, cond in G.RULES):
            dropped += 1
            continue
        fo.write(l)
        kept += 1
for s in ("val", "test"):
    shutil.copy(src / f"{s}.jsonl", dst / f"{s}.jsonl")
print(f"kept {kept} dropped {dropped} ({dropped / (kept + dropped):.1%})", flush=True)
