#!/usr/bin/env python3
"""Do "play like {player}" prompts make the text-conditioned model play like that player?

Test games: the 10 current players that have a player fine-tune, games that no model trained on, i.e. in the val/test
split of the SFT player corpus (data/sft_players, split by game url) AND newer than the fine-tune's holdout date.
For every game the player's own moves are scored by

  ce          factored cross-entropy of the move played (nats per move)
  match       top-1: the model's best legal move (sum of factored log-probs) is the move played

for the text-conditioned models under three prompts (one random template and alias per game, same template for all
three): the right player's name, another top player's name (drawn at random from the top-100 corpus), and no text;
and, as references, the player's own fine-tune, chsm8-player-base and chsm8-pt (no text).

    python scripts/eval_player_prompts.py --out results/player_prompts.jsonl
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from benchmark_play import load_checkpoint  # noqa: E402
from eval_sft import SFTPolicy  # noqa: E402
from factored_tokenizer import KIND_MOVE, PROMOTIONS, move_to_row  # noqa: E402
from meta_templates import render_player  # noqa: E402
from model import build_cu_seqlens, make_dummy_context  # noqa: E402

CAP = 513
STYLED = {"sft": "checkpoints/sft_v21/sft_v21.pt", "dpo": "checkpoints/chsm8_final/model.pt"}
REFS = {"player-base": "checkpoints/games/g100_drop_lr5e-05/latest.pt", "pt": "checkpoints/pt100/chsm8_latest.pt"}


def all_player_games(per_player, seed=0, top=0, part=(0, 1)):
    """Every top player of the SFT corpus: up to per_player games from its val/test split (never trained on by SFT/DPO)."""
    t = pq.read_table(ROOT / "data/sft_players/index.parquet",
                      columns=["player_fide_id", "player", "side", "split", "row_off", "n_rows"]).to_pylist()
    rows = np.memmap(ROOT / "data/sft_players/rows.i16", dtype=np.int16, mode="r").reshape(-1, 5)
    by = {}
    for g in t:
        if g["split"] != "train":
            by.setdefault(g["player_fide_id"], []).append(g)
    rng = random.Random(seed); games = []
    keys = sorted(by)
    if top:  # the `top` players with the most held-out games, all of their games; `part` = (i, n) slice of those players
        keys = sorted(sorted(by, key=lambda k: -len(by[k]))[:top])[part[0]::part[1]]
    for key in keys:
        for g in rng.sample(by[key], min(per_player, len(by[key]))):
            r = np.array(rows[g["row_off"]:g["row_off"] + min(g["n_rows"], CAP)], dtype=np.int64)
            games.append({"key": key, "player": g["player"], "side": g["side"], "rows": r})
    return games, sorted({r["player"] for r in t})


def test_games():
    t = pq.read_table(ROOT / "data/sft_players/index.parquet",
                      columns=["player_fide_id", "player", "side", "split", "date", "row_off", "n_rows"]).to_pylist()
    rows = np.memmap(ROOT / "data/sft_players/rows.i16", dtype=np.int16, mode="r").reshape(-1, 5)
    names = {r["player_fide_id"]: r["player"] for r in t}
    games = []
    for d in sorted(glob.glob(str(ROOT / "data/players/*"))):
        key = Path(d).name
        hold = json.load(open(f"{d}/summary.json")).get("holdout_date_from")
        if key not in names or not hold:
            continue
        for g in t:
            if g["player_fide_id"] == key and g["split"] != "train" and g["date"] > hold:
                r = np.array(rows[g["row_off"]:g["row_off"] + min(g["n_rows"], CAP)], dtype=np.int64)
                games.append({"key": key, "player": g["player"], "side": g["side"], "rows": r})
    return games, sorted({r["player"] for r in t})


def score(logits, games, b0):
    """Per game: (sum ce, n moves, hits) over the player's own moves."""
    lp = {k: F.log_softmax(v.float(), -1).cpu().numpy() for k, v in logits.items() if k != "kind"}
    out = []
    for bi, g in enumerate(games):
        r = g["rows"]; board = chess.Board(); me = chess.WHITE if g["side"] == "white" else chess.BLACK
        ce = n = hit = prob = hit3 = 0.0
        for t in range(1, len(r)):
            kind, src, dst, piece, promo = r[t]
            if kind != KIND_MOVE:
                break
            played = chess.Move(int(src), int(dst), PROMOTIONS[int(promo)])
            if board.turn == me:
                p = t - 1
                mv_s = []
                for mv in board.legal_moves:
                    _, a, bq, pc, pr = move_to_row(mv, board)
                    mv_s.append((lp["src"][bi, p, a] + lp["dst"][bi, p, bq] + lp["piece"][bi, p, pc] + lp["promo"][bi, p, pr], mv))
                sc = np.array([x for x, _ in mv_s], dtype=np.float64); pr_ = np.exp(sc - sc.max()); pr_ /= pr_.sum()
                order = np.argsort(-sc); legal = [m for _, m in mv_s]
                ip = legal.index(played)
                ce -= lp["src"][bi, p, src] + lp["dst"][bi, p, dst] + lp["piece"][bi, p, piece] + lp["promo"][bi, p, promo]
                n += 1; hit += order[0] == ip; hit3 += ip in order[:3]; prob += pr_[ip]
            board.push(played)
        out.append((float(ce), n, float(hit), float(prob), float(hit3)))
    return out


def batches(games, bs):
    for b0 in range(0, len(games), bs):
        gs = games[b0:b0 + bs]
        T = max(len(g["rows"]) for g in gs)
        x = np.zeros((len(gs), T, 5), dtype=np.int64)
        for i, g in enumerate(gs):
            x[i, :len(g["rows"])] = g["rows"]
        yield b0, gs, torch.from_numpy(x).cuda()


@torch.no_grad()
def run_styled(path, games, prompts, bs):
    pol = SFTPolicy(path, "", torch.device("cuda"), prompts=prompts) if any(prompts) else SFTPolicy(path, "", torch.device("cuda"))
    res = []
    for b0, gs, x in batches(games, bs):
        logits = pol(x, list(range(b0, b0 + len(gs)))) if any(prompts) else pol(x)
        res += score(logits, gs, b0)
    return res


@torch.no_grad()
def run_plain(path, games, bs):
    model, cfg, _ = load_checkpoint(path, "cuda")
    res = []
    for b0, gs, x in batches(games, bs):
        enc = make_dummy_context(x.shape[0], cfg.d_model, device="cuda") if cfg.use_cross_attention else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, enc, **dict(zip(("cu_seqlens", "max_seqlen"), build_cu_seqlens(x))))
        res += score(logits, gs, b0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/player_prompts.jsonl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of conditions to run (debug)")
    ap.add_argument("--all-players", type=int, default=0, help="N > 0: all top players, N held-out games each")
    ap.add_argument("--top-players", type=int, default=0, help="N > 0: the N players with most held-out games, all games")
    ap.add_argument("--part", default="0/1", help="i/n: this process takes every n-th of those players, from i")
    a = ap.parse_args()
    if a.top_players:
        i, n = map(int, a.part.split("/"))
        games, pool = all_player_games(10 ** 9, top=a.top_players, part=(i, n))
    else:
        games, pool = all_player_games(a.all_players) if a.all_players else test_games()
    rng = random.Random(a.seed)
    right, wrong = [], []
    for g in games:
        s = rng.getstate()
        right.append(render_player(g["player"], rng))
        other = rng.choice([p for p in pool if p != g["player"]])
        rng.setstate(s)
        wrong.append(render_player(other, rng))
    print(f"{len(games)} games, {len(set(g['key'] for g in games))} players; e.g. {right[0]!r} / {wrong[0]!r}", flush=True)
    conds = []
    for m, p in STYLED.items():
        conds += [(f"{m}:right", lambda p=p: run_styled(p, games, right, a.bs)),
                  (f"{m}:wrong", lambda p=p: run_styled(p, games, wrong, a.bs)),
                  (f"{m}:none", lambda p=p: run_styled(p, games, [""] * len(games), a.bs))]
    for m, p in REFS.items():
        conds.append((m, lambda p=p: run_plain(p, games, a.bs)))
    conds.append(("player-ft", None))
    only = set(a.only.split(",")) if a.only else None
    with open(ROOT / a.out, "a") as fo:
        for name, fn in conds:
            if only and name not in only:
                continue
            if name == "player-ft":
                res = [None] * len(games)
                for key in sorted(set(g["key"] for g in games)):
                    idx = [i for i, g in enumerate(games) if g["key"] == key]
                    for i, r in zip(idx, run_plain(f"checkpoints/players/final_{key}/latest.pt", [games[i] for i in idx], a.bs)):
                        res[i] = r
            else:
                res = fn()
            per = {}
            for g, (ce, n, hit, pb, h3) in zip(games, res):
                d = per.setdefault(g["key"], [0.0, 0, 0.0, 0, 0.0, 0.0, g["player"]])
                d[0] += ce; d[1] += n; d[2] += hit; d[3] += 1; d[4] += pb; d[5] += h3
            ce, n, hit, pb, h3 = (sum(r[i] for r in res) for i in range(5))
            game_match = [r[2] / r[1] for r in res if r[1]]
            rec = {"cond": name, "games": len(games), "moves": int(n), "ce": ce / n, "match": hit / n, "prob": pb / n,
                   "top3": h3 / n,
                   "match_sd_games": float(np.std(game_match)),
                   "players": {k: {"name": v[6], "games": v[3], "moves": v[1], "ce": v[0] / v[1], "match": v[2] / v[1],
                                  "prob": v[4] / v[1], "top3": v[5] / v[1]} for k, v in per.items()}}
            print(json.dumps({k: v for k, v in rec.items() if k != "players"}), flush=True)
            fo.write(json.dumps(rec) + "\n"); fo.flush()


if __name__ == "__main__":
    main()
