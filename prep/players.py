#!/usr/bin/env python3
"""chsm8-top-players (Hugging Face) -> the per-player JSONL our player pipelines were built from.

    python prep/players.py              # the 14 player-model corpora
    python prep/players.py --sft        # the 99-player crawl behind SFT's "play like {player}" prompts

The dataset has one split per player (99 of the FIDE top 100 + 4 historical players) and two corpora, marked by columns:
    holdout     train/val/test for the 14 player models (null = not in that corpus)
    sft_split   train/val/test for SFT (null = not in that corpus), game_idx = the crawl order SFT's corpus was built in

Writes data/top_players/games/<key>.jsonl and data/top_players/accounts.json. <key> is the FIDE id, or the name for the
historical players (fischer, kasparov, karpov, polgar_j): the keys of data/players/<key> in our runs. Then, as we ran them:

    python src/build_games_packed.py --source players --inputs data/top_players/games/<key>.jsonl \
        --out data/players/<key> --holdout 2000 --repeat-to 20000          # after the default mode
    python src/build_player_corpus.py --out data/sft_players                 # after --sft

build_games_packed holds out the newest games (min(2000, 20%)): exactly the `holdout` column. Both outputs rebuild our
corpora byte for byte.
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
    ap.add_argument("--sft", action="store_true", help="the 99-player SFT crawl instead of the 14 player-model corpora")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    src = a.src or snapshot_download("LegumMagister/chsm8-top-players", repo_type="dataset", allow_patterns=["data/*"])
    os.makedirs(f"{a.out}/games", exist_ok=True)
    col = "sft_split" if a.sft else "holdout"
    accounts, n = {}, 0
    for d in sorted(Path(src, "data").glob("*.parquet")):
        rows = [r for r in pq.read_table(d).to_pylist() if r[col] is not None]
        if not rows:
            continue
        key = HISTORICAL.get(d.stem) or rows[0]["player_fide_id"]
        rows.sort(key=(lambda r: r["game_idx"]) if a.sft else (lambda r: r["date"]))
        accounts[key] = {"name": rows[0]["player"]}
        with open(f"{a.out}/games/{key}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"white": r["white"], "black": r["black"], "white_elo": r["white_elo"],
                                    "black_elo": r["black_elo"], "result": r["result"], "date": r["date"],
                                    "event": r["event"], "tc": r["time_control"], "source": r["source"],
                                    "time_class": r["time_class"], "url": r["url"], "moves": list(r["moves"]),
                                    "player_fide_id": key, "player_side": r["player_side"]}) + "\n")
        n += len(rows)
    Path(a.out, "accounts.json").write_text(json.dumps(accounts, indent=1))
    print(f"{len(accounts)} players, {n:,} games -> {a.out}/games")


if __name__ == "__main__":
    main()
