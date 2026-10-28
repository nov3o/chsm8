#!/usr/bin/env python3
"""Batched move server for DeepMind's searchless chess transformers (Ruoss et al., 2024): the 9M, 136M or 270M
action-value model, chosen by the checkpoint directory's name.

Runs in its own JAX environment (.pydeps_jax) so JAX and PyTorch never share a process. Protocol, one JSON object per
line on stdin/stdout:
    in:  {"games": [["e2e4", "e7e5", ...], ...]}     moves from the start position, one list per game
    out: {"moves": ["g1f3", ...]}
Each move is chosen exactly as the authors' ActionValueEngine without temperature: the legal move with the highest
expected win probability, with moves that allow a repetition draw scored 0.5.

    PYTHONPATH=.pydeps_jax python3 scripts/searchless_server.py models/opponents/searchless_270m/270M
"""
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external"))
class _Stub(types.ModuleType):
    """Stands in for training-only dependencies (apache_beam, grain) of the authors' constants module."""
    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if f"{self.__name__}.{attr}" in sys.modules:
            return sys.modules[f"{self.__name__}.{attr}"]
        return type(attr, (), {"__init__": lambda self, *a, **k: None, "__call__": lambda self, *a, **k: None})


for name in ("apache_beam", "apache_beam.coders", "grain", "grain.python"):
    sys.modules.setdefault(name, _Stub(name))

import chess  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402
from jax import random as jrandom  # noqa: E402

from searchless_chess.src import tokenizer, transformer, utils  # noqa: E402

BATCH = 2048  # largest sequences per jitted call; smaller calls are padded to a power of two (few compilations)


SIZES = {"9M": (8, 256), "136M": (8, 1024), "270M": (16, 1024)}  # (layers, width), the authors' engines/constants.py


def load(ckpt_dir):
    layers, width = SIZES[Path(ckpt_dir).resolve().name]
    cfg = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS, output_size=128, pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2, num_heads=8, num_layers=layers, embedding_dim=width,
        apply_post_ln=True, apply_qk_layernorm=False, use_causal_mask=False)
    predictor = transformer.build_transformer_predictor(config=cfg)
    init = predictor.initial_params(rng=jrandom.PRNGKey(1), targets=np.ones((1, 1), dtype=np.uint32))
    params = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).restore(
        Path(ckpt_dir).resolve() / "6400000" / "params", restore_args=ocp.checkpoint_utils.construct_restore_args(init))
    fn = jax.jit(predictor.predict)
    _, values = utils.get_uniform_buckets_edges_values(128)

    def predict(seqs):
        out = []
        for i in range(0, len(seqs), BATCH):
            chunk = seqs[i:i + BATCH]
            size = max(64, 1 << (len(chunk) - 1).bit_length())
            pad = np.pad(chunk, ((0, size - len(chunk)), (0, 0)))
            out.append(np.asarray(fn(params=params, targets=pad, rng=None))[:len(chunk), -1])
        return np.concatenate(out)
    return predict, values


def ordered_legal(board):  # the authors' engine.get_ordered_legal_moves
    return sorted(board.legal_moves, key=lambda m: m.uci())


def main():
    predict, values = load(sys.argv[1])
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        games = json.loads(line)["games"]
        boards, seqs, spans = [], [], []
        for moves in games:
            b = chess.Board()
            for u in moves:
                b.push_uci(u)
            legal = ordered_legal(b)
            fen = tokenizer.tokenize(b.fen()).astype(np.int32)
            start = len(seqs)
            for mv in legal:
                seqs.append(np.concatenate([fen, [utils.MOVE_TO_ACTION[mv.uci()], 0]]))
            boards.append((b, legal)); spans.append((start, len(seqs)))
        logp = predict(np.stack(seqs).astype(np.int32))
        win = np.exp(logp) @ values
        out = []
        for (b, legal), (s, e) in zip(boards, spans):
            w = win[s:e].copy()
            for i, mv in enumerate(legal):
                b.push(mv)
                if b.is_fivefold_repetition() or b.can_claim_threefold_repetition():
                    w[i] = 0.5
                b.pop()
            out.append(legal[int(np.argmax(w))].uci())
        print(json.dumps({"moves": out}), flush=True)


if __name__ == "__main__":
    main()
