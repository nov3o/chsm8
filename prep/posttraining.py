#!/usr/bin/env python3
"""chsm8-post-training (labels) + chsm8-pre-training (moves) -> the game-side parquet our pipeline was built from.

    python prep/posttraining.py                       # all 30.1M sides
    python prep/posttraining.py --first-files --months 2014-01   # nano: one month of the first files

Writes data/chsm8-post-training-unlabeled/data/part-NNNNN.parquet: one row per described side with the game's moves
(joined on (month, row_idx)), its metadata, the five labels, the labeler, the per-band check and the split. Our two
builders read exactly this:

    python src/build_sft_corpus.py data/chsm8-post-training-unlabeled/data --out data/sft30m_v21   # SFT / DPO
    python src/build_games_packed.py --source pool --out data/games30m                            # player-base
"""
import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BANDS = ["terse_1", "terse_2", "short", "medium", "long"]
SPLITS = {"train": "train", "validation": "val", "test": "test"}
GAME = ["white", "black", "white_elo", "black_elo", "white_title", "black_title", "result", "termination",
        "time_control", "tc_base", "tc_increment", "eco", "opening", "n_plies", "moves"]
OUTCOME = {("1-0", "white"): "win", ("0-1", "black"): "win", ("1-0", "black"): "loss", ("0-1", "white"): "loss"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/chsm8-post-training-unlabeled/data")
    ap.add_argument("--post", default="", help="local copy of chsm8-post-training (default: download)")
    ap.add_argument("--pre", default="", help="local copy of chsm8-pre-training (default: download needed months)")
    ap.add_argument("--first-files", action="store_true", help="download only the first file of each split (nano)")
    ap.add_argument("--months", nargs="*", default=[], help="only sides of these months (default: all)")
    ap.add_argument("--max-sides", type=int, default=0, help="first N sides of each split, 0 = all")
    ap.add_argument("--rows-per-file", type=int, default=1_000_000)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    post = a.post or snapshot_download("LegumMagister/chsm8-post-training", repo_type="dataset",
                                    allow_patterns=["data/*/part-00000.parquet" if a.first_files else "data/*"])
    parts = []
    for hf_split, split in SPLITS.items():
        t = pq.read_table(Path(post, "data", hf_split), columns=["month", "row_idx", "side", "labels", "labeler", "correct"])
        t = t.append_column("_i", pa.array(range(len(t))))
        if a.months:
            t = t.filter(pc.is_in(t["month"], pa.array(a.months)))
        if a.max_sides:
            t = t.slice(0, a.max_sides)
        labels = t["labels"].combine_chunks().to_pylist()
        t = t.drop(["labels", "correct"]).append_column(
            "labels", pa.array([[d[b] for b in BANDS] for d in labels], pa.list_(pa.large_string())))
        correct = pq.read_table(Path(post, "data", hf_split), columns=["correct"])["correct"]
        t = t.append_column("label_correct", pc.cast(pc.take(correct, t["_i"]), pa.list_(pa.int8()))).drop(["_i"])
        parts.append(t.append_column("split", pa.array([split] * len(t))))
    sides = pa.concat_tables(parts)
    months = sorted(set(sides["month"].to_pylist()))
    pre = a.pre or snapshot_download("LegumMagister/chsm8-pre-training", repo_type="dataset",
                                     allow_patterns=[f"data/{m}/*" for m in months])
    os.makedirs(a.out, exist_ok=True)
    buf, n_file, n_rows = [], 0, 0

    def flush():
        nonlocal buf, n_file
        if buf:
            pq.write_table(pa.concat_tables(buf), f"{a.out}/part-{n_file:05d}.parquet")
            buf, n_file = [], n_file + 1

    for m in months:
        s = sides.filter(pc.equal(sides["month"], m))
        need = pa.array(sorted(set(s["row_idx"].to_pylist())), pa.int32())
        games = pa.concat_tables([pq.read_table(f, columns=["row_idx"] + GAME) for f in sorted(Path(pre, "data", m).glob("*.parquet"))])
        games = games.filter(pc.is_in(games["row_idx"], need)).to_pydict()
        at = {r: i for i, r in enumerate(games["row_idx"])}
        cols = {k: [] for k in ("player", "player_elo", "opponent", "opponent_elo", "player_outcome", *GAME)}
        for r, side in zip(s["row_idx"].to_pylist(), s["side"].to_pylist()):
            g = {k: games[k][at[r]] for k in GAME}
            other = "black" if side == "white" else "white"
            for k in GAME:
                cols[k].append(g[k])
            cols["player"].append(g[side]); cols["player_elo"].append(g[f"{side}_elo"])
            cols["opponent"].append(g[other]); cols["opponent_elo"].append(g[f"{other}_elo"])
            cols["player_outcome"].append(OUTCOME.get((g["result"], side), "draw"))
        t = s.select(["month", "row_idx", "side"])
        for k, v in cols.items():
            t = t.append_column(k, pa.array(v))
        for k in ("labels", "labeler", "label_correct", "split"):
            t = t.append_column(k, s[k])
        buf.append(t); n_rows += len(t)
        if sum(len(x) for x in buf) >= a.rows_per_file:
            flush()
        print(f"{m}: {len(t):,} sides", flush=True)
    flush()
    print(f"{a.out}: {n_rows:,} sides in {n_file} files")


if __name__ == "__main__":
    main()
