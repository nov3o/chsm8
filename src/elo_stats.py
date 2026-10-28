#!/usr/bin/env python3
"""Exact streaming Elo statistics for tier-2 metadata.

Reports both player appearances and move-token weighted distributions.  The
latter is the relevant distribution when focal move losses receive Elo weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def summarize(hist: np.ndarray) -> dict:
    values = np.arange(len(hist), dtype=np.float64)
    total = float(hist.sum())
    mean = float((hist * values).sum() / total)
    std = float(np.sqrt((hist * (values - mean) ** 2).sum() / total))
    cumulative = np.cumsum(hist)
    def quantile(q: float) -> int:
        return int(np.searchsorted(cumulative, q * total, side="left"))
    return {"n": int(total), "mean": mean, "std_population": std,
            "min": int(np.flatnonzero(hist)[0]), "p01": quantile(.01),
            "p05": quantile(.05), "p25": quantile(.25), "median": quantile(.50),
            "p75": quantile(.75), "p95": quantile(.95), "p99": quantile(.99),
            "max": int(np.flatnonzero(hist)[-1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-rows", type=int, default=1_000_000)
    ap.add_argument("--alpha", type=float, default=.30)
    ap.add_argument("--elo-ref", type=float, default=1800.)
    ap.add_argument("--min-weight", type=float, default=.6)
    ap.add_argument("--max-weight", type=float, default=2.5)
    args = ap.parse_args()
    files = sorted(args.root.glob("*/meta.parquet"))
    if not files:
        raise SystemExit(f"no shard metadata under {args.root}")
    appearances = np.zeros(65536, dtype=np.int64)
    move_tokens = np.zeros(65536, dtype=np.int64)
    games = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=args.batch_rows,
                                     columns=["white_elo", "black_elo", "n_plies"]):
            we = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            be = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            plies = batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            appearances += np.bincount(we, minlength=65536)
            appearances += np.bincount(be, minlength=65536)
            move_tokens += np.bincount(we, weights=(plies + 1) // 2, minlength=65536).astype(np.int64)
            move_tokens += np.bincount(be, weights=plies // 2, minlength=65536).astype(np.int64)
            games += len(we)
    raw = np.clip(10 ** (args.alpha * (np.arange(65536) - args.elo_ref) / 400),
                  args.min_weight, args.max_weight)
    normalizer = float((raw * move_tokens).sum() / move_tokens.sum())
    out = {"root": str(args.root), "shards": len(files), "games": games,
           "player_appearances": summarize(appearances),
           "move_tokens": summarize(move_tokens),
           "elo_weighting": {"formula": "clip(10^(alpha*(elo-elo_ref)/400), min, max) / Z",
                             "alpha": args.alpha, "elo_ref": args.elo_ref,
                             "min_weight": args.min_weight, "max_weight": args.max_weight,
                             "Z_move_token_arithmetic_mean": normalizer}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
