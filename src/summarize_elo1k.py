#!/usr/bin/env python3
"""Merge the shard summaries of one 1,000-game Stockfish run (slurm/elo1k.sbatch) into an Elo.

Elo = engine_elo + 400*log10(S/(1-S)) for score S = (W + D/2)/N, 95% CI from the per-game
standard error (same formula as reports/model_comparison.md). Writes <dir>/elo.json.
    python src/summarize_elo1k.py results/elo1k/<tag> [...]
"""
import glob
import json
import math
import os
import sys


def elo_of(score, engine_elo):
    score = min(max(score, 1e-4), 1 - 1e-4)
    return engine_elo + 400 * math.log10(score / (1 - score))


def summarize(run_dir):
    shards = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "shard*", "summary.json")))]
    if not shards:
        raise SystemExit(f"no shard summaries in {run_dir}")
    games = [g for s in shards for g in s["games"] if g.get("outcome") in ("win", "draw", "loss")]
    w = sum(g["outcome"] == "win" for g in games)
    d = sum(g["outcome"] == "draw" for g in games)
    l = sum(g["outcome"] == "loss" for g in games)
    n = w + d + l
    per = [1.0 if g["outcome"] == "win" else 0.5 if g["outcome"] == "draw" else 0.0 for g in games]
    s = sum(per) / n
    se = math.sqrt(sum((x - s) ** 2 for x in per) / (n - 1) / n)
    engine = shards[0]["engine_elo"]
    out = {"tag": os.path.basename(run_dir.rstrip("/")), "checkpoint": shards[0]["checkpoint"],
           "games": n, "wins": w, "draws": d, "losses": l,
           "unfinished": sum(sh.get("unfinished", 0) for sh in shards), "score": round(s, 4),
           "elo": round(elo_of(s, engine)), "elo_ci95": [round(elo_of(s - 1.96 * se, engine)),
                                                         round(elo_of(s + 1.96 * se, engine))],
           "engine_elo": engine, "engine_time_s": shards[0]["engine_time_s"]}
    json.dump(out, open(os.path.join(run_dir, "elo.json"), "w"), indent=1)
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        summarize(d)
