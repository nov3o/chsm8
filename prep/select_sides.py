#!/usr/bin/env python3
"""chsm8-pre-training (1B games) -> data/4/player_game_index.parquet, the game sides to describe.

    python prep/select_sides.py                       # all months: 30,085,219 sides
    python prep/select_sides.py --months 2014-01      # nano

A side is kept when its player is rated >= 2200 and the opponent >= 1800; then each player keeps at most 100 of their
sides over all months, sampled uniformly. This is our rule: over all of chsm8-pre-training it gives 308,465,941 eligible
sides from 871,497 players and exactly 30,085,219 after the cap, the size of chsm8-post-training. Players with <= 100
sides (672,227 of them) match chsm8-post-training row for row. Our original sample of the 199,270 capped players was
unseeded, so with --seed you get the same counts but a different 100 for them; chsm8-post-training holds our exact
list (recipes/6_label.sh rebuilds the index from it).

Columns: month, row_idx, player, elo, side, as src/labeler/build_unlabeled.py reads them.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def eligible(args):
    src, month, min_elo, min_opp = args
    t = pq.ParquetDataset(sorted(map(str, Path(src, "data", month).glob("*.parquet")))).read(
        columns=["row_idx", "white", "black", "white_elo", "black_elo"])
    out = []
    for side, other in (("white", "black"), ("black", "white")):
        x = t.filter(pc.and_(pc.greater_equal(t[f"{side}_elo"], min_elo), pc.greater_equal(t[f"{other}_elo"], min_opp)))
        out.append(pa.table({"month": pa.array([month] * len(x), pa.large_string()), "row_idx": x["row_idx"],
                             "player": pc.cast(x[side], pa.large_string()), "elo": pc.cast(x[f"{side}_elo"], pa.uint16()),
                             "side": pa.array([side] * len(x), pa.large_string())}))
    return pa.concat_tables(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="", help="local copy of chsm8-pre-training (default: download)")
    ap.add_argument("--out", default="data/4/player_game_index.parquet")
    ap.add_argument("--months", nargs="*", default=[], help="default: all")
    ap.add_argument("--min-elo", type=int, default=2200)
    ap.add_argument("--min-opponent-elo", type=int, default=1800)
    ap.add_argument("--per-player", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    src = a.src or snapshot_download("LegumMagister/chsm8-pre-training", repo_type="dataset",
                                     allow_patterns=[f"data/{m}/*" for m in a.months] or ["data/*"])
    months = a.months or sorted(p.name for p in Path(src, "data").iterdir() if p.is_dir())
    with Pool(a.workers) as p:
        t = pa.concat_tables(p.map(eligible, [(src, m, a.min_elo, a.min_opponent_elo) for m in months]))
    print(f"{len(t):,} eligible sides", flush=True)
    # uniform sample of per-player sides: sort by (player, random key), keep the first per-player of each run
    t = t.append_column("_k", pa.array(np.random.default_rng(a.seed).random(len(t))))
    t = t.take(pc.sort_indices(t, [("player", "ascending"), ("_k", "ascending")]))
    player = t["player"].combine_chunks()
    start = np.flatnonzero(np.r_[True, pc.not_equal(player[1:], player[:-1]).to_numpy(zero_copy_only=False)])
    rank = np.arange(len(t)) - np.repeat(start, np.diff(np.r_[start, len(t)]))
    t = t.filter(pa.array(rank < a.per_player)).drop(["_k"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pq.write_table(t.select(["month", "row_idx", "player", "elo", "side"]), a.out)
    print(f"{a.out}: {len(t):,} sides, {len(start):,} players")


if __name__ == "__main__":
    main()
