"""What SFT and DPO share: which weights train, the text of a batch, the focal-move targets, evaluation.

A batch is G games of one side each (data.Sides.batch). In SFT each game comes with all its descriptions (five for
a described side, one prompt for a player game); the model runs the bottom layers once per game and the text layers
once per (game, description), see model.CHSM8.forward(repeat=...).
"""
import random

import numpy as np
import torch

from chsm8 import players
from chsm8.model import loss_fn, move_logprobs
from chsm8.tokenizer import KIND_PAD


def param_groups(model, lr, text_lr):
    """SFT and DPO train the whole decoder except the input embeddings and the no-text constants. The text path
    (projection, gates, cross-attention and its norm) trains with its own, larger learning rate."""
    model.requires_grad_(False)
    text, body = [], []
    for name, p in model.named_parameters():
        if name.startswith(("text_proj.", "gate")) or ".cross_attn." in name or ".n2." in name:
            text.append(p)
        elif name.startswith(("layers.", "norm.", "head.")) and not name.endswith(".no_text"):
            body.append(p)
    for p in text + body:
        p.requires_grad_(True)
    return [{"params": text, "lr": text_lr, "base_lr": text_lr}, {"params": body, "lr": lr, "base_lr": lr}]


def lr_mult(progress, warmup, min_ratio=0.0):
    import math
    if progress < warmup:
        return progress / warmup
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / (1 - warmup))))


def focal_targets(rows, focal):
    """Next-row targets with every target that is not a move of the described side turned into PAD (ignored)."""
    y = rows[:, 1:].clone()
    y[..., 0] = torch.where(focal, y[..., 0], torch.full_like(y[..., 0], KIND_PAD))
    return y


def texts(sides, ids, rng):
    """Descriptions of each game: its five labels, or one player prompt. -> list (per game) of lists of strings."""
    if "labels" in sides.cols:
        return [sides.get("labels", k) for k in ids]
    return [[players.render(sides.get("player", k), rng)] for k in ids]


def encode(encoder, per_game, dropout=0.0, device="cpu"):
    """Encode descriptions game-major (g0 d0, g0 d1, ..., g1 d0, ...). dropout: probability that a (game,
    description) row gets no text at all, so the model keeps a good no-prompt policy."""
    flat = [t for g in per_game for t in g]
    text = encoder(flat)
    if dropout:
        text["keep"] = torch.rand(len(flat), device=device) >= dropout
    return text


def to(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, encoder, sides, ids, games_per_batch=32):
    """Mean focal-move loss (no legality penalty) of a fixed set of game sides:
        matched     with their own descriptions
        mismatched  with the descriptions of the next game in the batch (how much the text matters)
        no_text     without text (the model with an empty prompt)"""
    model.eval()
    device = next(model.parameters()).device
    ids = np.asarray(ids)[np.argsort(sides.n_rows[ids], kind="stable")]
    sums, n = {"matched": 0.0, "mismatched": 0.0, "no_text": 0.0}, 0
    for s in range(0, len(ids), games_per_batch):
        b = to(sides.batch(ids[s:s + games_per_batch]), device)
        per_game = texts(sides, b["ids"], random.Random(0))
        k = len(per_game[0])
        R = lambda t: t.repeat_interleave(k, 0)
        x, y = b["rows"][:, :-1], focal_targets(b["rows"], b["focal"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            text = encode(encoder, per_game)
            sums["matched"] += float(loss_fn(model(x, text, repeat=k), R(y)))
            rolled = {key: v.roll(k, 0) for key, v in text.items()}  # each game gets the previous game's texts
            sums["mismatched"] += float(loss_fn(model(x, rolled, repeat=k), R(y)))
            sums["no_text"] += float(loss_fn(model(x), y))
        n += 1
    model.train()
    return {key: v / max(n, 1) for key, v in sums.items()}


def sequence_logprob(logits, rows, focal):
    """(B,) log-probability of the described side's moves: the sum over its moves of log p(row), all five heads."""
    return (move_logprobs(logits, rows[:, 1:]) * focal).sum(1)
