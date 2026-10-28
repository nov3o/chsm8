"""Time and memory per move for one player: a CHSM8 checkpoint, a public opponent, or Stockfish.

For each batch size B, B self-play games advance together one ply at a time (one call per ply for all B boards), from
four first moves; every call is timed after 3 warmup calls. Reports the median wall time of a call and per move
(call / B), and peak memory: GPU memory used on this job's GPU (sampled with nvidia-smi, minus the idle level) and peak
host RSS of this process and its children (engines).

  python src/time_per_move.py --chsm8 checkpoints/pt100/chsm8_latest.pt --name chsm8-pt --batches 1,8,64,256
  python src/time_per_move.py --opponent maia3 --opponent-path models/opponents/maia3_79m/maia3-79m.pt --name maia3-79m
  python src/time_per_move.py --stockfish external/stockfish-sf17.1/src/stockfish --name stockfish --batches 1
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factored_tokenizer import ROW_BOS, move_to_row  # noqa: E402

FIRST = ["e2e4", "d2d4", "c2c4", "g1f3"]


class GpuMem(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        self.base = self.read(); self.peak = self.base; self.stop = False

    def read(self):
        try:
            out = subprocess.run(["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 stdout=subprocess.PIPE, universal_newlines=True, timeout=10).stdout
            return int(out.strip().splitlines()[0])
        except Exception:
            return 0

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, self.read()); time.sleep(0.2)


def rss_mb():
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return s / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chsm8")
    ap.add_argument("--stockfish")
    ap.add_argument("--sf-time", type=float, default=0.05)
    ap.add_argument("--opponent")
    ap.add_argument("--opponent-path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--batches", default="1,8,64,256")
    ap.add_argument("--plies", type=int, default=30, help="timed plies per game")
    ap.add_argument("--out", default="results/time_per_move2.jsonl")
    a, rest = ap.parse_known_args()
    mem = GpuMem(); mem.start()
    device = "cuda:0"
    if a.chsm8:
        from benchmark_play import choose_model_moves_batched, load_checkpoint
        model, cfg = load_checkpoint(a.chsm8, device)[:2]
        def moves(gs):
            r = choose_model_moves_batched(model, cfg, device, [(i, {"rows": g["rows"], "board": g["b"]}) for i, g in gs])
            return {i: mv for i, (mv, _) in r.items()}
    elif a.stockfish:
        eng = chess.engine.SimpleEngine.popen_uci(a.stockfish)
        eng.configure({"Threads": 1, "UCI_LimitStrength": True, "UCI_Elo": 2000})
        def moves(gs):
            return {i: eng.play(g["b"], chess.engine.Limit(time=a.sf_time)).move for i, g in gs}
    else:
        from benchmark_versus import add_opponent_args, build_opponent
        op = argparse.ArgumentParser(); add_opponent_args(op)
        opp = build_opponent(op.parse_args(["--opponent", a.opponent, "--opponent-path", a.opponent_path] + rest), device)
        def moves(gs):
            return opp.choose([(i, g["b"]) for i, g in gs])
    res = dict(name=a.name, gpu_idle_mb=mem.base, by_batch={})
    for B in [int(x) for x in a.batches.split(",")]:
        games = []
        for i in range(B):
            b, rows = chess.Board(), [ROW_BOS]
            m = chess.Move.from_uci(FIRST[i % 4]); rows.append(move_to_row(m, b)); b.push(m)
            games.append({"b": b, "rows": rows})
        times = []
        try:
            for _ in range(a.plies):
                live = [(i, g) for i, g in enumerate(games) if not g["b"].is_game_over()]
                if not live:
                    break
                t = time.perf_counter(); mv = moves(live); times.append(time.perf_counter() - t)
                for i, g in live:
                    g["rows"].append(move_to_row(mv[i], g["b"])); g["b"].push(mv[i])
            t = np.array(times[3:]) * 1000
            res["by_batch"][B] = dict(call_ms=float(np.median(t)), per_move_ms=float(np.median(t)) / B,
                                     gpu_peak_mb=mem.peak - mem.base, host_rss_mb=rss_mb())
        except Exception as e:  # e.g. out of memory at this batch size
            res["by_batch"][B] = dict(error=repr(e)[:200])
        print(json.dumps({B: res["by_batch"][B]}), flush=True)
    mem.stop = True
    if a.stockfish:
        eng.quit()
    print(json.dumps(res), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    os._exit(0)


if __name__ == "__main__":
    main()
