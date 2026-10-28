#!/usr/bin/env python3
"""Play a factored-pretraining checkpoint against a UCI engine.

The model always chooses its highest-scoring *legal* chess move.  This is a
real playing test, not next-token loss: games alternate colours, are recorded
as PGN, and report W/D/L plus Stockfish's centipawn change on model moves.
"""
import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import chess.engine
import chess.pgn
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factored_tokenizer import ROW_BOS, move_to_row  # noqa: E402
from model import make_dummy_context  # noqa: E402
from train_pretrain import build_model  # noqa: E402


@torch.no_grad()
def choose_model_move(model, cfg, device, rows, board):
    legal = list(board.legal_moves)
    if not legal:
        return None, None
    seq = rows[-cfg.seq_len:]
    x = torch.tensor([seq], dtype=torch.long, device=device)
    enc = make_dummy_context(1, cfg.d_model, device=device) if cfg.use_cross_attention else None
    logits = model(x, enc)
    src = F.log_softmax(logits["src"][0, -1], dim=-1)
    dst = F.log_softmax(logits["dst"][0, -1], dim=-1)
    piece = F.log_softmax(logits["piece"][0, -1], dim=-1)
    promo = F.log_softmax(logits["promo"][0, -1], dim=-1)
    best_move, best_score = None, float("-inf")
    for mv in legal:
        _, a, b, p, q = move_to_row(mv, board)
        score = float(src[a] + dst[b] + piece[p] + promo[q])
        if score > best_score:
            best_move, best_score = mv, score
    return best_move, best_score


@torch.no_grad()
def choose_model_moves_batched(model, cfg, device, states):
    """Choose legal moves for many same-length histories in GPU batches."""
    chosen = {}
    by_length = {}
    for i, state in states:
        by_length.setdefault(len(state["rows"][-cfg.seq_len:]), []).append((i, state))
    for group in by_length.values():
        x = torch.tensor([s["rows"][-cfg.seq_len:] for _, s in group], dtype=torch.long, device=device)
        enc = make_dummy_context(len(group), cfg.d_model, device=device) if cfg.use_cross_attention else None
        if getattr(model, "per_game", False):  # per-game prompts (eval_sft --prompts-file): pass the game indices
            enc = [state["game_i"] for _, state in group]
        logits = model(x, enc)
        for batch_i, (i, state) in enumerate(group):
            src = F.log_softmax(logits["src"][batch_i, -1], dim=-1)
            dst = F.log_softmax(logits["dst"][batch_i, -1], dim=-1)
            piece = F.log_softmax(logits["piece"][batch_i, -1], dim=-1)
            promo = F.log_softmax(logits["promo"][batch_i, -1], dim=-1)
            best_move, best_score = None, float("-inf")
            for mv in state["board"].legal_moves:
                _, a, b, p, q = move_to_row(mv, state["board"])
                score = float(src[a] + dst[b] + piece[p] + promo[q])
                if score > best_score:
                    best_move, best_score = mv, score
            chosen[i] = (best_move, best_score)
    return chosen


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["cfg"])
    cfg.setdefault("fast_dummy_cross_attn", False)
    cfg.setdefault("grad_checkpoint", False)
    cfg.setdefault("architecture", "classic")
    cfg.setdefault("llama_qk_norm", True)
    cfg.setdefault("torch_compile", False)
    args = SimpleNamespace(**cfg)
    model, model_cfg = build_model(args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args, ckpt


def cp(info, pov):
    return info["score"].pov(pov).score(mate_score=10_000)


def outcome_score(outcome):
    """Integer match score requested by the competitive benchmark protocol."""
    return {"win": 1, "draw": 0, "loss": -1}.get(outcome)


def score_summary(results):
    """Return total and color-split +1/0/-1 scores for completed games."""
    completed = [x for x in results if x["outcome"] in {"win", "draw", "loss"}]
    by_color = {}
    for color in ("white", "black"):
        games = [x for x in completed if x["model_color"] == color]
        by_color[color] = {
            "games": len(games),
            "wins": sum(x["outcome"] == "win" for x in games),
            "draws": sum(x["outcome"] == "draw" for x in games),
            "losses": sum(x["outcome"] == "loss" for x in games),
            "score_plus1_draw0_minus1": sum(x["score_plus1_draw0_minus1"] for x in games),
        }
    return {
        "score_convention": "win=+1, draw=0, loss=-1",
        "score_plus1_draw0_minus1": sum(x["score_plus1_draw0_minus1"] for x in completed),
        "by_model_color": by_color,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--engine", required=True, help="UCI engine executable, normally Stockfish")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--engine-time", type=float, default=0.10, help="seconds per Stockfish move/eval")
    ap.add_argument("--engine-elo", type=int, default=1800,
                    help="Stockfish UCI_Elo opponent strength; set 0 for unrestricted strength")
    ap.add_argument("--max-plies", type=int, default=240)
    ap.add_argument("--parallel-games", type=int, default=1,
                    help="independent one-thread engines; model moves are batched on the GPU")
    ap.add_argument("--out", default="results/playing_benchmark")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("no CUDA: playing on CPU (slow)", flush=True)
    model, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, all_games = [], []
    if args.parallel_games > 1:
        return run_parallel(model, cfg, device, ckpt, args, out, results, all_games)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 256})
    if args.engine_elo:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.engine_elo})
    try:
        for game_i in range(args.games):
            model_color = chess.WHITE if game_i % 2 == 0 else chess.BLACK
            board, rows = chess.Board(), [ROW_BOS]
            pgn = chess.pgn.Game()
            pgn.headers["White"] = "model" if model_color else "Stockfish"
            pgn.headers["Black"] = "Stockfish" if model_color else "model"
            pgn.headers["Round"] = str(game_i + 1)
            node = pgn
            losses, model_logps = [], []
            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < args.max_plies:
                before_board = board.copy(stack=False)
                if board.turn == model_color:
                    before = cp(engine.analyse(board, chess.engine.Limit(time=args.engine_time)), model_color)
                    move, logp = choose_model_move(model, cfg, device, rows, board)
                    if move is None:
                        break
                    board.push(move)
                    after = cp(engine.analyse(board, chess.engine.Limit(time=args.engine_time)), model_color)
                    losses.append(before - after)
                    model_logps.append(logp)
                else:
                    move = engine.play(board, chess.engine.Limit(time=args.engine_time)).move
                    board.push(move)
                rows.append(move_to_row(move, before_board))
                node = node.add_variation(move)
            result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
            pgn.headers["Result"] = result
            won = (result == "1-0" and model_color == chess.WHITE) or (result == "0-1" and model_color == chess.BLACK)
            lost = (result == "0-1" and model_color == chess.WHITE) or (result == "1-0" and model_color == chess.BLACK)
            outcome = "win" if won else "loss" if lost else "draw" if result != "*" else "unfinished"
            results.append({"game": game_i + 1, "model_color": "white" if model_color else "black", "result": result,
                            "outcome": outcome, "score_plus1_draw0_minus1": outcome_score(outcome),
                            "plies": len(board.move_stack), "mean_cp_loss": sum(losses) / len(losses) if losses else None,
                            "mean_model_logp": sum(model_logps) / len(model_logps) if model_logps else None})
            all_games.append(str(pgn))
            print(results[-1], flush=True)
    finally:
        engine.quit()
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": ckpt.get("step"), "engine": args.engine,
               "engine_time_s": args.engine_time, "engine_elo": args.engine_elo or "unrestricted", "games": results,
               "wins": sum(x["outcome"] == "win" for x in results), "draws": sum(x["outcome"] == "draw" for x in results),
               "losses": sum(x["outcome"] == "loss" for x in results),
               "unfinished": sum(x["outcome"] == "unfinished" for x in results)}
    summary.update(score_summary(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "games.pgn").write_text("\n\n".join(all_games) + "\n")
    print(json.dumps(summary, indent=2))


def run_parallel(model, cfg, device, ckpt, args, out, results, all_games):
    """Run independent games concurrently; never share a UCI engine between threads."""
    states = []
    engines = []
    try:
        for game_i in range(args.games):
            engine = chess.engine.SimpleEngine.popen_uci(args.engine)
            engine.configure({"Threads": 1, "Hash": 16})
            if args.engine_elo:
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.engine_elo})
            model_color = chess.WHITE if game_i % 2 == 0 else chess.BLACK
            game = chess.pgn.Game()
            game.headers["White"] = "model" if model_color else "Stockfish"
            game.headers["Black"] = "Stockfish" if model_color else "model"
            game.headers["Round"] = str(game_i + 1)
            states.append({"game_i": game_i, "model_color": model_color, "board": chess.Board(),
                           "rows": [ROW_BOS], "pgn": game, "node": game, "losses": [],
                           "model_logps": [], "done": False})
            engines.append(engine)

        def finish(i):
            s = states[i]
            if s["done"]:
                return
            board = s["board"]
            result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
            s["pgn"].headers["Result"] = result
            won = (result == "1-0" and s["model_color"] == chess.WHITE) or (result == "0-1" and s["model_color"] == chess.BLACK)
            lost = (result == "0-1" and s["model_color"] == chess.WHITE) or (result == "1-0" and s["model_color"] == chess.BLACK)
            outcome = "win" if won else "loss" if lost else "draw" if result != "*" else "unfinished"
            results.append({"game": s["game_i"] + 1, "model_color": "white" if s["model_color"] else "black",
                            "result": result, "outcome": outcome, "score_plus1_draw0_minus1": outcome_score(outcome),
                            "plies": len(board.move_stack), "mean_cp_loss": sum(s["losses"]) / len(s["losses"]) if s["losses"] else None,
                            "mean_model_logp": sum(s["model_logps"]) / len(s["model_logps"]) if s["model_logps"] else None})
            all_games.append(str(s["pgn"]))
            s["done"] = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_games) as pool:
            while True:
                active = [(i, s) for i, s in enumerate(states) if not s["done"]]
                if not active:
                    break
                for i, s in active:
                    if s["board"].is_game_over(claim_draw=True) or len(s["board"].move_stack) >= args.max_plies:
                        finish(i)
                active = [(i, s) for i, s in enumerate(states) if not s["done"]]
                model_turn = [(i, s) for i, s in active if s["board"].turn == s["model_color"]]
                engine_turn = [(i, s) for i, s in active if s["board"].turn != s["model_color"]]
                if model_turn:
                    before = list(pool.map(lambda pair: cp(engines[pair[0]].analyse(pair[1]["board"], chess.engine.Limit(time=args.engine_time)), pair[1]["model_color"]), model_turn))
                    chosen = choose_model_moves_batched(model, cfg, device, model_turn)
                    moved = []
                    for (i, s), pre in zip(model_turn, before):
                        move, logp = chosen[i]
                        if move is None:
                            finish(i); continue
                        old = s["board"].copy(stack=False)
                        s["board"].push(move); s["rows"].append(move_to_row(move, old)); s["node"] = s["node"].add_variation(move)
                        s["model_logps"].append(logp); moved.append((i, s, pre))
                    after = list(pool.map(lambda item: cp(engines[item[0]].analyse(item[1]["board"], chess.engine.Limit(time=args.engine_time)), item[1]["model_color"]), moved))
                    for (_, s, pre), post in zip(moved, after): s["losses"].append(pre - post)
                if engine_turn:
                    moves = list(pool.map(lambda pair: engines[pair[0]].play(pair[1]["board"], chess.engine.Limit(time=args.engine_time)).move, engine_turn))
                    for (i, s), move in zip(engine_turn, moves):
                        old = s["board"].copy(stack=False)
                        s["board"].push(move); s["rows"].append(move_to_row(move, old)); s["node"] = s["node"].add_variation(move)
        results.sort(key=lambda x: x["game"])
    finally:
        for engine in engines:
            engine.quit()
    summary = {"checkpoint": args.checkpoint, "checkpoint_step": ckpt.get("step"), "engine": args.engine,
               "engine_time_s": args.engine_time, "engine_elo": args.engine_elo or "unrestricted", "games": results,
               "wins": sum(x["outcome"] == "win" for x in results), "draws": sum(x["outcome"] == "draw" for x in results),
               "losses": sum(x["outcome"] == "loss" for x in results), "unfinished": sum(x["outcome"] == "unfinished" for x in results)}
    summary.update(score_summary(results))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "games.pgn").write_text("\n\n".join(all_games) + "\n")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__":
    main()
