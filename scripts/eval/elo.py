"""Elo against Stockfish, our protocol: Stockfish 17.1 at UCI_Elo 2000, 0.05 s per move, 1 thread, start position,
colours alternate, greedy legal moves, at most 240 plies (unfinished games are left out).

    python scripts/eval/elo.py --model LegumMagister/chsm8 --stockfish /path/to/stockfish --games 1000
    python scripts/eval/elo.py --model LegumMagister/chsm8 --prompt "attack the king at any cost" --games 200

Elo = engine Elo + 400 log10(S / (1 - S)) for the score S, with a 95% interval. Games run in parallel threads, one
engine each; --pgn saves them.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8.play import Player, elo, play_stockfish  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LegumMagister/chsm8", help="Hugging Face repo, or repo:filename")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--engine-elo", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=0.05)
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--pgn", default="")
    a = ap.parse_args()
    repo, _, filename = a.model.partition(":")
    player = Player.from_pretrained(repo, filename or "model.pt")

    def one(i):
        return play_stockfish(player, a.stockfish, a.engine_elo, a.seconds, i % 2 == 0, a.prompt, a.max_plies)
    with ThreadPoolExecutor(a.threads) as pool:
        games = list(pool.map(one, range(a.games)))
    if a.pgn:
        with open(a.pgn, "w") as f:
            f.write("\n\n".join(str(g) for g in games) + "\n")
    print(json.dumps({"model": a.model, "prompt": a.prompt, "engine_elo": a.engine_elo, **elo(games, a.engine_elo)}))


if __name__ == "__main__":
    main()
