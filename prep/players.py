#!/usr/bin/env python3
"""chsm8-top-players (Hugging Face) -> the per-player JSONL our player pipeline was built from.

    python prep/players.py

Writes, for each of the 14 players:
    data/top_players/games/<key>.jsonl   one game per line (LAN moves list, player_side, date, url, ...)
    data/top_players/accounts.json       key -> {"name": ...}
<key> is the FIDE id for active players and the name for historical ones (fischer, kasparov, karpov, polgar_j), the
same keys as data/players/<key> in our runs. Then, exactly as we ran them:

    python src/build_games_packed.py --source players --inputs data/top_players/games/<key>.jsonl \
        --out data/players/<key> --holdout 2000 --repeat-to 20000          # player fine-tunes
    python src/build_player_corpus.py --out data/sft_players                 # player prompts in SFT

build_games_packed holds out the newest games (min(2000, 20%)); that is exactly the `holdout` column of the dataset.
Our SFT used 99 players; only these 14 are public, so data/sft_players is the 14-player version.
"""
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

HISTORICAL = {"fischer": "fischer", "kasparov": "kasparov", "karpov": "karpov", "polgar_judit": "polgar_j"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="", help="local copy of chsm8-top-players (default: download)")
    ap.add_argument("--out", default="data/top_players")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    src = a.src or snapshot_download("LegumMagister/chsm8-top-players", repo_type="dataset", allow_patterns=["data/*"])
    os.makedirs(f"{a.out}/games", exist_ok=True)
    accounts = {}
    for d in sorted(Path(src, "data").glob("*.parquet")):
        rows = pq.read_table(d).to_pylist()
        if not rows:
            continue
        key = HISTORICAL.get(d.stem) or str(rows[0]["player_fide_id"])
        accounts[key] = {"name": rows[0]["player"]}
        with open(f"{a.out}/games/{key}.jsonl", "w") as f:
            for r in sorted(rows, key=lambda r: r["date"]):
                moves = r["moves"].split() if isinstance(r["moves"], str) else list(r["moves"])
                f.write(json.dumps({"white": r["white"], "black": r["black"], "white_elo": r["white_elo"],
                                    "black_elo": r["black_elo"], "result": r["result"], "date": r["date"],
                                    "event": r["event"], "tc": r["time_control"], "source": r["source"],
                                    "time_class": r["time_class"], "url": r["url"], "moves": moves,
                                    "player_side": r["player_side"], "player_fide_id": key}) + "\n")
        print(f"{d.stem} -> {key}: {len(rows):,} games")
    Path(a.out, "accounts.json").write_text(json.dumps(accounts, indent=1))


if __name__ == "__main__":
    main()
