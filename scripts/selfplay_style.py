#!/usr/bin/env python3
"""Style evaluation by self-play: a text-conditioned model (one prompt per game) against a fixed no-text opponent,
all games advancing one ply at a time in GPU batches. No engine, so no CPU bottleneck: 1,000 games take about
one game length of batched forward passes.

Each book opening line is played twice with colours swapped (as in benchmark_versus.py); both sides play their most
likely legal move. The styled side is called "model" in the PGN, so src/summarize_sft_eval.behaviour() measures it.

    python scripts/selfplay_style.py checkpoints/chsm8_final/model.pt --prompt random --out results/selfplay/dpo_random
    --prompt none | random | "<text>"     random: one test-split description per game (results/selfplay/prompts_random.txt)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from benchmark_play import ROW_BOS, choose_model_moves_batched, load_checkpoint, move_to_row  # noqa: E402
from eval_sft import SFTPolicy  # noqa: E402

RANDOM = ROOT / "results/selfplay/prompts_random.txt"


def random_prompts(n, seed=0):
    """n descriptions drawn uniformly from the test split of the post-training corpus, one random band each."""
    if RANDOM.exists() and sum(1 for _ in open(RANDOM)) >= n:
        return [l.rstrip("\n") for l in open(RANDOM)][:n]
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ROOT / "data/sft30m_v21/index.parquet")
    rng = random.Random(seed)
    groups = rng.sample(range(pf.num_row_groups), min(pf.num_row_groups, 4 * n))
    out = []
    for g in groups:
        t = [r for r in pf.read_row_group(g, columns=["split", "labels"]).to_pylist() if r["split"] == "test"]
        if t:
            labels = t[rng.randrange(len(t))]["labels"]
            out.append(" ".join(str(rng.choice(list(labels))).split()))
        if len(out) == n:
            break
    RANDOM.parent.mkdir(parents=True, exist_ok=True)
    RANDOM.write_text("\n".join(out) + "\n")
    return out


def game_metrics(s):
    """Behaviour of the styled side over its own moves after the book (book moves are not the model's)."""
    b = chess.Board(); me = chess.WHITE if s["styled_white"] else chess.BLACK
    own = capt = checks = pawn = 0; castled = ooo = early_q = False; q40 = float("nan")
    for ply, mv in enumerate(s["board"].move_stack):
        mine = b.turn == me and ply >= s["book_plies"]
        if mine:
            own += 1; capt += b.is_capture(mv); pawn += b.piece_type_at(mv.from_square) == chess.PAWN
            if b.is_castling(mv):
                castled = True; ooo = b.is_queenside_castling(mv)
            if b.piece_type_at(mv.from_square) == chess.QUEEN and own <= 5:
                early_q = True
        b.push(mv)
        if mine and b.is_check():
            checks += 1
        if ply == 39:
            q40 = float(not b.pieces(chess.QUEEN, chess.WHITE) and not b.pieces(chess.QUEEN, chess.BLACK))
    own = max(own, 1)
    return {"plies": len(s["board"].move_stack), "capt": capt / own, "checks": checks / own, "pawn": pawn / own,
            "q_off40": q40, "castled": float(castled), "ooo_all": float(castled and ooo),
            "ooo": float(ooo) if castled else float("nan"), "early_q": float(early_q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="text-conditioned SFT/DPO checkpoint (the styled side)")
    ap.add_argument("--opponent", default="checkpoints/games/g100_drop_lr5e-05/latest.pt", help="no-text opponent")
    ap.add_argument("--prompt", default="none")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--openings", default=str(ROOT / "external/chess_gpt_eval/logs/openings.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.prompt == "random":
        prompts = random_prompts(a.games)
    elif a.prompt == "none":
        prompts = [""] * a.games
    else:
        prompts = [a.prompt] * a.games
    styled = SFTPolicy(a.checkpoint, "", dev, prompts=prompts) if any(prompts) else SFTPolicy(a.checkpoint, "", dev)
    scfg = argparse.Namespace(seq_len=512, d_model=0, use_cross_attention=False)
    opp, ocfg, _ = load_checkpoint(a.opponent, dev)
    ocfg.seq_len = 512

    class Wrap(torch.nn.Module):  # pretrained-format model: plain call with the dummy context built inside the chooser
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x, enc=None):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return {k: v.float() for k, v in self.m(x, enc).items()}

    oppw = Wrap(opp)
    book = [l.strip() for l in Path(a.openings).read_text().splitlines()[1:] if l.strip()]
    states = []
    for i in range(a.games):
        b = chess.Board(); rows = [ROW_BOS]
        for tok in book[(i // 2) % len(book)].split():
            if tok[0].isdigit() and "." in tok:
                tok = tok.split(".", 1)[1]
                if not tok:
                    continue
            mv = b.parse_san(tok); rows.append(move_to_row(mv, b)); b.push(mv)
        states.append({"game_i": i, "board": b, "rows": rows, "styled_white": i % 2 == 0, "book_plies": len(b.move_stack)})
    while True:
        live = [(i, s) for i, s in enumerate(states)
                if not s["board"].is_game_over(claim_draw=True) and len(s["board"].move_stack) < a.max_plies]
        if not live:
            break
        mine = [(i, s) for i, s in live if (s["board"].turn == chess.WHITE) == s["styled_white"]]
        theirs = [(i, s) for i, s in live if (s["board"].turn == chess.WHITE) != s["styled_white"]]
        chosen = {}
        with torch.no_grad():
            if mine:
                chosen.update(choose_model_moves_batched(styled, scfg, dev, mine))
            if theirs:
                chosen.update(choose_model_moves_batched(oppw, ocfg, dev, theirs))
        for i, (mv, _) in chosen.items():
            s = states[i]; s["rows"].append(move_to_row(mv, s["board"])); s["board"].push(mv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pgns, rec = [], []
    for s in states:
        b = s["board"]
        res = b.result(claim_draw=True) if b.is_game_over(claim_draw=True) else "1/2-1/2"
        g = chess.pgn.Game.from_board(b)
        g.headers.update({"White": "model" if s["styled_white"] else "opponent",
                          "Black": "opponent" if s["styled_white"] else "model", "Result": res, "Round": str(s["game_i"] + 1)})
        pgns.append(str(g))
        pts = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
        rec.append(pts if s["styled_white"] else 1 - pts)
    (out / "games.pgn").write_text("\n\n".join(pgns) + "\n")
    summ = {"checkpoint": a.checkpoint, "opponent": a.opponent, "prompt": a.prompt, "games": a.games,
            "score": float(np.mean(rec)), "games_detail": []}
    per = [game_metrics(st) for st in states]
    beh = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
    summ["behaviour"] = beh
    (out / "per_game.jsonl").write_text("".join(json.dumps(p) + "\n" for p in per))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"prompt": a.prompt[:60], "score": summ["score"], **{k: round(v, 4) for k, v in beh.items()}}))


if __name__ == "__main__":
    main()
