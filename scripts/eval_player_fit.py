#!/usr/bin/env python3
"""How well a checkpoint imitates one player, on that player's held-out games.

Reads a player pack from src/build_games_packed.py --source players (<pack>/<split>/packed_*),
runs the model once per packed row, and scores only the focal player's moves (_side.u1 & 4):

  focal_ce      factored CE (the training objective, no illegal penalty) on the player's moves
  move_match    fraction of the player's moves where the model's top *legal* move (same scoring
                as src/benchmark_play.py: sum of factored log-probs) is exactly the move played
  move_match_top3

Prints one JSON line; --out appends it to a JSONL file.

    python scripts/eval_player_fit.py CKPT data/players/2016192 --split val --out results/player_fit.jsonl
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from benchmark_play import load_checkpoint  # noqa: E402
from factored_tokenizer import KIND_BOS, KIND_MOVE, PROMOTIONS, move_to_row  # noqa: E402
from model import build_cu_seqlens, factored_cross_entropy, make_dummy_context  # noqa: E402
from train_pretrain import build_block_causal_mask  # noqa: E402


def games_in_row(row):
    """Split one packed row into [(start, end)) spans of individual games (BOS ... EOS)."""
    starts = np.flatnonzero(row[:, 0] == KIND_BOS)
    ends = list(starts[1:]) + [len(row)]
    return list(zip(starts, ends))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("pack")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_checkpoint(args.checkpoint, DEV)
    shard = sorted(glob.glob(f"{args.pack}/{args.split}/packed_*.i16"))[0]
    cap = cfg.seq_len + 1
    rows = np.fromfile(shard, dtype="<i2").reshape(-1, cap, 5).astype(np.int64)
    side = np.fromfile(shard[:-4] + "_side.u1", dtype="u1").reshape(-1, cap)

    ce_sum = ce_n = 0.0
    hit1 = hit3 = n_moves = 0
    for b0 in range(0, len(rows), args.batch):
        xb = torch.from_numpy(rows[b0:b0 + args.batch]).to(DEV)
        x, y = xb[:, :-1], xb[:, 1:].clone()
        focal = torch.from_numpy((side[b0:b0 + args.batch, 1:] & 4) != 0).to(DEV)
        y[~focal] = 0  # non-focal targets -> PAD (ignored by every head)
        enc = make_dummy_context(x.shape[0], cfg.d_model, device=DEV) if cfg.use_cross_attention else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = model(x, enc, **(dict(zip(("cu_seqlens", "max_seqlen"), build_cu_seqlens(x))) if DEV == "cuda"
                                      else dict(mask=build_block_causal_mask(x))))  # CPU: dense block-causal (as train_pretrain.py)
        n_f = int(focal.sum())
        ce_sum += factored_cross_entropy(logits, y).item() * n_f
        ce_n += n_f
        lp = {k: F.log_softmax(v.float(), -1).cpu().numpy() for k, v in logits.items() if k != "kind"}
        for bi in range(x.shape[0]):
            full = rows[b0 + bi]
            for s, e in games_in_row(full):
                board = chess.Board()
                for t in range(s + 1, e):
                    kind, src, dst, piece, promo = full[t]
                    if kind != KIND_MOVE:
                        break
                    played = chess.Move(int(src), int(dst), PROMOTIONS[int(promo)])
                    if side[b0 + bi, t] & 4:
                        p = t - 1  # prediction for row t is made at position t-1
                        scores = []
                        for mv in board.legal_moves:
                            _, a, bq, pc, pr = move_to_row(mv, board)
                            scores.append((lp["src"][bi, p, a] + lp["dst"][bi, p, bq] + lp["piece"][bi, p, pc]
                                           + lp["promo"][bi, p, pr], mv))
                        scores.sort(key=lambda z: -z[0])
                        top = [mv for _, mv in scores[:3]]
                        hit1 += top[0] == played
                        hit3 += played in top
                        n_moves += 1
                    board.push(played)
    res = {"tag": args.tag, "checkpoint": args.checkpoint, "pack": args.pack, "split": args.split,
           "focal_ce": ce_sum / max(ce_n, 1), "move_match": hit1 / max(n_moves, 1),
           "move_match_top3": hit3 / max(n_moves, 1), "focal_moves": n_moves}
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
