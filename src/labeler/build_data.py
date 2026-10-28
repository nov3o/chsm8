#!/usr/bin/env python3
"""Join teacher labels with the exact s10-mini context the teacher saw.

Output rows: {key, month, row_idx, side, labeler, system, user, labels}.
Split is by game (hash of month|row_idx), so both sides of a game land in the
same split. Selection is a deterministic hash bucket, not the file order
(the labels file is player-ordered and starts with a long apertus-only run).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prompt-ablation"))
sys.path.insert(0, str(ROOT / "src"))

import types  # noqa: E402

# production_label imports autolabel.providers (API clients) at module level;
# we only need its game decoding, so stub the client module out.
sys.modules.setdefault("autolabel.providers", types.ModuleType("autolabel.providers"))
import production_label as P  # noqa: E402
from autolabel import corpus  # noqa: E402

DATA_ROOTS = [ROOT / "data/2b"]


def data_root(month: str) -> Path:
    return next((r for r in DATA_ROOTS if (r / month / "_SUCCESS").exists()), DATA_ROOTS[0])


def bucket(month: str, row_idx: int) -> int:
    h = hashlib.blake2b(f"{month}|{row_idx}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % 10000


def split_of(b: int, n_val: int, n_train: int) -> str | None:
    if b < n_val:
        return "val"
    if b < 2 * n_val:
        return "test"
    if b < 2 * n_val + n_train:
        return "train"
    return None


def build_month(args):
    month, rows = args
    P.DATA_2B = data_root(month)
    cache = P.MonthCache()
    out = []
    for r in sorted(rows, key=lambda r: r["row_idx"]):
        try:
            meta, moves, local = cache.get(month, r["row_idx"])
            g = P.decode_game(meta, moves, local, month, r["side"])
            ctx = corpus.build_context(g, r["side"], card=P.VARIANT.card, moves=P.VARIANT.moves,
                                       anonymise=P.VARIANT.anonymise, reveal_result=P.VARIANT.reveal_result)
        except Exception as e:  # corrupt/missing game: skip, count
            out.append({"error": f"{type(e).__name__}: {e}", "key": r["key"]})
            continue
        msgs = P.VARIANT.messages(ctx)
        out.append({"key": r["key"], "split": r["split"], "month": month, "row_idx": r["row_idx"],
                    "side": r["side"], "labeler": r["labeler"], "elo": r.get("elo"),
                    "system": msgs[0]["content"], "user": msgs[1]["content"], "labels": r["labels"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "labels/labels_clean_2026-09-29.jsonl.gz"))
    ap.add_argument("--out", default=str(ROOT / "data/labeler"))
    ap.add_argument("--val-buckets", type=int, default=5)      # 0.05% of games each for val/test
    ap.add_argument("--train-buckets", type=int, default=600)  # ~6% of games -> ~470k records
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--labeler", default="", help="keep only this teacher (e.g. gemma)")
    a = ap.parse_args()

    by_month = defaultdict(list)
    with gzip.open(a.labels, "rt") as f:
        for line in f:
            d = json.loads(line)
            if a.labeler and d.get("labeler") != a.labeler:
                continue
            s = split_of(bucket(d["month"], d["row_idx"]), a.val_buckets, a.train_buckets)
            if s is None:
                continue
            d["split"] = s
            d["key"] = f"{d['month']}:{d['row_idx']}:{d['side']}"
            by_month[d["month"]].append(d)
    print("selected", sum(map(len, by_month.values())), "records over", len(by_month), "months", flush=True)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: open(out_dir / f"{s}.jsonl", "w") for s in ("train", "val", "test")}
    counts, errors = defaultdict(int), 0
    with Pool(a.workers) as pool:
        for rows in pool.imap_unordered(build_month, sorted(by_month.items(), key=lambda kv: -len(kv[1]))):
            for r in rows:
                if "error" in r:
                    errors += 1
                    continue
                files[r["split"]].write(json.dumps(r) + "\n")
                counts[(r["split"], r["labeler"])] += 1
    for f in files.values():
        f.close()
    print("counts", dict(counts), "errors", errors, flush=True)


if __name__ == "__main__":
    main()
