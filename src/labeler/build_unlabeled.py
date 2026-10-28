#!/usr/bin/env python3
"""Inputs for every record of data/4/player_game_index.parquet that has no teacher label yet.

Writes N_SHARDS jsonl shards of {key, month, row_idx, side, elo, user} (user = the exact
teacher prompt, same builder as build_data.py) for label_shards.py.
usage: build_unlabeled.py --out DIR [--shards 96] [--workers 256] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data as BD  # noqa: E402

INDEX = BD.ROOT / "data/4/player_game_index.parquet"


def build_month(args):
    month, rows = args
    for r in rows:
        r.update(split=None, labeler=None, labels=None)
    return [{k: v for k, v in r.items() if k in ("key", "month", "row_idx", "side", "elo", "user", "error")}
            for r in BD.build_month((month, rows))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=str(BD.ROOT / "labels/labels_clean_2026-09-29.jsonl.gz"))
    ap.add_argument("--shards", type=int, default=96)
    ap.add_argument("--workers", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--select", choices=["unlabeled", "apertus"], default="unlabeled",
                    help="unlabeled: records with no teacher label; apertus: records labeled by Apertus (to relabel)")
    a = ap.parse_args()

    done, apertus = set(), {}
    with gzip.open(a.labels, "rt") as f:
        for line in f:
            d = json.loads(line)
            k = f"{d['month']}:{d['row_idx']}:{d['side']}"
            if d["labeler"] == "apertus":
                apertus[k] = d["labels"] if a.select == "apertus" else None
            else:
                done.add(k)
    if a.select == "unlabeled":
        done |= set(apertus)
    print("already labeled", len(done) + (0 if a.select == "unlabeled" else len(apertus)), "apertus", len(apertus), flush=True)

    by_month, n = defaultdict(list), 0
    for batch in pq.ParquetFile(INDEX).iter_batches(batch_size=200_000):
        for r in batch.to_pylist():
            key = f"{r['month']}:{r['row_idx']}:{r['side']}"
            if key in done or (a.select == "apertus" and key not in apertus):
                continue
            by_month[r["month"]].append({"key": key, "month": r["month"], "row_idx": r["row_idx"],
                                         "side": r["side"], "elo": r.get("elo")})
            n += 1
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    print("unlabeled", n, "records over", len(by_month), "months", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [open(out / f"shard_{i:03d}.jsonl", "w") for i in range(a.shards)]
    written = errors = 0
    with Pool(a.workers) as pool:
        # split big months so the pool stays balanced
        jobs = [(m, rows[i:i + 20000]) for m, rows in by_month.items() for i in range(0, len(rows), 20000)]
        for rows in pool.imap_unordered(build_month, jobs):
            for r in rows:
                if "error" in r:
                    errors += 1
                    continue
                if a.select == "apertus":
                    r["labels"] = apertus[r["key"]]  # keep the original labels for patching / comparison
                s = int.from_bytes(hashlib.blake2b(r["key"].encode(), digest_size=4).digest(), "little") % a.shards
                files[s].write(json.dumps(r) + "\n")
                written += 1
            if written and written % 1_000_000 < len(rows):
                print(f"written {written} errors {errors}", flush=True)
    for f in files:
        f.close()
    print(f"done: written {written} errors {errors}", flush=True)


if __name__ == "__main__":
    main()
