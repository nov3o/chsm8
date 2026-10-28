#!/usr/bin/env python3
"""chsm8-distillation (Hugging Face) -> data/labeler/{train,val,test}.jsonl, the files src/labeler/train.py reads.

    python prep/labeler.py                 # 2.40M clean teacher sets
    python prep/labeler.py --max-rows 2000 # small test

Each line: {key, split, month, row_idx, side, labeler, elo, system, user, labels{terse_1..long}}, as in our run.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = {"train": "train", "validation": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="", help="local copy of chsm8-distillation (default: download)")
    ap.add_argument("--out", default="data/labeler")
    ap.add_argument("--max-rows", type=int, default=0, help="per split, 0 = all")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    src = a.src or snapshot_download("LegumMagister/chsm8-distillation", repo_type="dataset",  # --max-rows: first shards only
                                     allow_patterns=["system_prompt.txt", "data/*-00000.parquet" if a.max_rows else "data/*"])
    system = Path(src, "system_prompt.txt").read_text().rstrip("\n")
    os.makedirs(a.out, exist_ok=True)
    for hf, split in SPLITS.items():
        t = pq.ParquetDataset(sorted(map(str, Path(src, "data").glob(f"{hf}-*.parquet")))).read()
        if a.max_rows:
            t = t.slice(0, a.max_rows)
        with open(f"{a.out}/{split}.jsonl", "w") as f:
            for r in t.to_pylist():
                f.write(json.dumps({"key": r["key"], "split": split, "month": r["month"], "row_idx": r["row_idx"],
                                    "side": r["side"], "labeler": "gemma", "elo": r["elo"], "system": system,
                                    "user": r["user"], "labels": r["labels"]}) + "\n")
        print(f"{a.out}/{split}.jsonl: {len(t):,}")


if __name__ == "__main__":
    main()
