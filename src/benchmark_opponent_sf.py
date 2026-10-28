#!/usr/bin/env python3
"""Another chess model vs Stockfish, with exactly the protocol of benchmark_play.py (CHSM8's Elo runs), so both land on
the same scale: start position, colours alternate, Stockfish 17.1 with Threads=1, Hash=16, UCI_LimitStrength and
--engine-elo, --engine-time seconds per move, max --max-plies plies (then unfinished = draw in summarize_elo1k.py).
The opponent plays its own greedy choice through the adapters in benchmark_versus.py.

    python src/benchmark_opponent_sf.py --opponent maia2 --opponent-path models/opponents/maia2_blitz/original/model.pt \
        --name maia2-blitz --engine external/stockfish-sf17.1/src/stockfish --games 500 --out results/opp_sf/maia2/shard0
"""
import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_versus as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    V.add_opponent_args(ap)
    ap.add_argument("--name", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--engine-elo", type=int, default=2000)
    ap.add_argument("--engine-time", type=float, default=0.05)
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--first-game", type=int, default=0, help="game index offset, so shards alternate colours alike")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--parallel-games", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.device
    opp = V.build_opponent(a, d)
    results, pgns = [], []
    for start in range(0, a.games, a.parallel_games):
        idx = list(range(a.first_game + start, a.first_game + min(start + a.parallel_games, a.games)))
        engines = []
        try:
            for _ in idx:
                e = chess.engine.SimpleEngine.popen_uci(a.engine)
                e.configure({"Threads": 1, "Hash": 16})
                if a.engine_elo:
                    e.configure({"UCI_LimitStrength": True, "UCI_Elo": a.engine_elo})
                engines.append(e)
            st = [{"gi": g, "color": chess.WHITE if g % 2 == 0 else chess.BLACK, "board": chess.Board()} for g in idx]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(idx)) as pool:
                while True:
                    live = [(k, s) for k, s in enumerate(st) if not s["board"].is_game_over(claim_draw=True)
                            and len(s["board"].move_stack) < a.max_plies]
                    if not live:
                        break
                    mine = [(k, s) for k, s in live if s["board"].turn == s["color"]]
                    theirs = [(k, s) for k, s in live if s["board"].turn != s["color"]]
                    if mine:
                        mv = opp.choose([(k, s["board"]) for k, s in mine])
                        for k, s in mine:
                            s["board"].push(mv[k])
                    if theirs:
                        played = pool.map(lambda ks: engines[ks[0]].play(ks[1]["board"], chess.engine.Limit(time=a.engine_time)).move, theirs)
                        for (k, s), m in zip(theirs, played):
                            s["board"].push(m)
        finally:
            for e in engines:
                e.quit()
        for s in st:
            b = s["board"]
            res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "*"
            won = (res == "1-0") == (s["color"] == chess.WHITE) and res in ("1-0", "0-1")
            lost = res in ("1-0", "0-1") and not won
            outcome = "win" if won else "loss" if lost else "draw" if res != "*" else "unfinished"
            results.append({"game": s["gi"] + 1, "model_color": "white" if s["color"] else "black", "result": res,
                            "outcome": outcome, "plies": len(b.move_stack)})
            g = chess.pgn.Game.from_board(b)
            g.headers.update({"White": a.name if s["color"] else "Stockfish", "Black": "Stockfish" if s["color"] else a.name,
                              "Result": res, "Round": str(s["gi"] + 1)})
            pgns.append(str(g))
        print(f"{len(results)}/{a.games} games", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"model": a.name, "checkpoint": a.opponent_path, "opponent_path": a.opponent_path, "engine": a.engine, "engine_time_s": a.engine_time,
               "engine_elo": a.engine_elo, "max_plies": a.max_plies, "games": results,
               "wins": sum(r["outcome"] == "win" for r in results), "draws": sum(r["outcome"] == "draw" for r in results),
               "losses": sum(r["outcome"] == "loss" for r in results),
               "unfinished": sum(r["outcome"] == "unfinished" for r in results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "games.pgn").write_text("\n\n".join(pgns) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
    # Results are written; skip interpreter shutdown, where engine pools (python-chess event-loop threads, e.g.
    # Matilda's Stockfish processes) could wait on each other forever. Child engines get EOF and exit.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
