"""chsm8-top-players (Hugging Face) -> data for the player models and for the player games of SFT.

    python scripts/data/prepare_players.py --packed data/players --sides data/player_sides

    --packed  one packed corpus per player, data/players/<name>/{train,val,test}, with a focal mask on the player's
              own moves (the opponent's moves are context), for scripts/train/pretrain.py configs/player.yaml
    --sides   all players' games as one game-sides corpus with a `player` column, for scripts/train/sft.py

The dataset keeps the newest games of each player (up to 1,000 each) as validation and test (column `holdout`).
--players limits the players (file names in the dataset, e.g. carlsen kasparov).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8.data import Packer, SidesWriter  # noqa: E402
from chsm8.tokenizer import KIND_MOVE, encode  # noqa: E402

REPO = "LegumMagister/chsm8-top-players"
SPLIT = {"train": "train", "val": "validation", "test": "test"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", default="", help="output: one packed corpus per player")
    ap.add_argument("--sides", default="", help="output: game-sides corpus of all players")
    ap.add_argument("--players", nargs="*", help="default: all")
    ap.add_argument("--source", default="", help="local copy of the dataset (default: download from the Hub)")
    ap.add_argument("--max-rows", type=int, default=512)
    a = ap.parse_args()
    assert a.packed or a.sides, "nothing to do: give --packed and/or --sides"
    src = a.source
    if not src:
        from huggingface_hub import snapshot_download
        src = snapshot_download(REPO, repo_type="dataset", allow_patterns=["data/*"])
    files = sorted(Path(src, "data").glob("*.parquet"))
    if a.players:
        files = [f for f in files if f.stem in a.players]
    sides = SidesWriter(a.sides) if a.sides else None
    index = {k: [] for k in ("split", "side", "player", "row_off", "n_rows")}
    for f in files:
        t = pq.read_table(f, columns=["player", "player_side", "holdout", "moves"]).to_pydict()
        packers = {s: Packer(f"{a.packed}/{f.stem}/{s}/games", 512, focal=True) for s in SPLIT} if a.packed else {}
        skipped = 0
        for player, side, split, moves in zip(t["player"], t["player_side"], t["holdout"], t["moves"]):
            try:
                rows, occ, pmask = encode(" ".join(moves))
            except ValueError:
                skipped += 1
                continue
            if len(rows) > a.max_rows:
                skipped += 1
                continue
            mine = 0 if side == "white" else 1  # row k >= 1 is ply k-1; white plays the even plies
            focal = [int(r[0] == KIND_MOVE and (k - 1) % 2 == mine) for k, r in enumerate(rows)]
            if packers:
                packers[split].add(rows, occ, pmask, focal)
            if sides:
                off, n = sides.add_game(rows, occ, pmask)
                for k, v in zip(index, (SPLIT[split], side, player, off, n)):
                    index[k].append(v)
        if packers:
            n = {s: p.close()[1] for s, p in packers.items()}
            Path(a.packed, f.stem, "info.json").write_text(json.dumps({"seq_len": 512, "player": t["player"][0],
                                                                       **{f"{s}_games": v for s, v in n.items()}}))
        print(f"{f.stem}: {len(t['moves']):,} games, {skipped} skipped", flush=True)
    if sides:
        sides.close(pa.table({**{k: pa.array(index[k]) for k in ("split", "side", "player")},
                              "row_off": pa.array(index["row_off"], pa.int64()),
                              "n_rows": pa.array(index["n_rows"], pa.int32())}))
        print(f"{a.sides}: {len(index['split']):,} player games")


if __name__ == "__main__":
    main()
