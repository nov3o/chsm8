#!/usr/bin/env python3
"""Small, reproducible architecture search for text-conditioned chess style.

This intentionally operates on ``data/style_smoke`` rather than the full
label corpus.  It has two independent modes:

* ``clip`` trains a game/text contrastive reward model and reports held-out
  retrieval, the useful gate before any preference optimization.
* ``sft`` trains *all* chess weights from the production checkpoint with a
  description encoded into the existing cross-attention slots, and applies
  move loss only to the labeled player's plies.

The runner is not a long-training recipe.  Its output is a compact JSONL
record per trial, allowing architecture choices to be compared without
confounding corpus quality or exposing held-out games to either objective.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from factored_tokenizer import KIND_MOVE, KIND_PAD, N_PROMOTIONS
from model import (DecoderConfig, FactoredChessDecoder, make_dummy_context,
                   legal_component_masks_from_bits)

ROOT = Path(__file__).resolve().parents[1]
BANDS = ("terse_1", "terse_2", "short", "medium", "long")
LEGAL_DTYPE = np.dtype([("occ", "<u8"), ("pmask", "u1")])


def split_for(example_id: int, seed: int) -> str:
    # Stable game-level split. Both sides would share an id only in a corpus
    # containing both labels; grouping by month,row_idx below handles that too.
    value = hash(f"{seed}:{example_id}") % 100
    return "train" if value < 90 else ("val" if value < 95 else "test")


def stable_split(month: str, row_idx: int, seed: int) -> str:
    # Python's hash is process-salted; use a simple deterministic byte sum.
    raw = f"{seed}:{month}:{row_idx}".encode()
    value = int.from_bytes(__import__("hashlib").blake2b(raw, digest_size=8).digest(), "little") % 100
    return "train" if value < 90 else ("val" if value < 95 else "test")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\s]", text.lower())


class Vocab:
    def __init__(self, texts: list[str], max_words: int = 8192):
        counts: dict[str, int] = {}
        for text in texts:
            for word in tokens(text):
                counts[word] = counts.get(word, 0) + 1
        ranked = sorted(counts, key=lambda x: (-counts[x], x))[:max_words - 2]
        self.ids = {word: i + 2 for i, word in enumerate(ranked)}

    def encode(self, text: str, limit: int) -> list[int]:
        return [self.ids.get(word, 1) for word in tokens(text)[:limit]] or [1]

    def __len__(self) -> int:
        return len(self.ids) + 2


class TextTower(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, variant: str):
        super().__init__()
        self.variant, self.max_len = variant, max_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        if variant == "transformer":
            layer = nn.TransformerEncoderLayer(d_model, max(1, d_model // 64), d_model * 2,
                                               batch_first=True, activation="gelu", norm_first=True)
            self.body = nn.TransformerEncoder(layer, num_layers=2)
        else:
            self.body = None
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = ids.ne(0)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.embed(ids) + self.pos(pos)
        if self.body is not None:
            x = self.body(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return x, pooled


class StyleData:
    def __init__(self, root: Path, seed: int, max_rows: int):
        index = pq.read_table(root / "index.parquet").to_pylist()
        raw = np.memmap(root / "rows.i16", dtype="<i2", mode="r").reshape(-1, 5)
        legal_path = root / "legal.bin"
        legal = (np.memmap(legal_path, dtype=LEGAL_DTYPE, mode="r")
                 if legal_path.exists() else None)
        self.items = []
        for record in index:
            if int(record["n_rows"]) > max_rows:
                continue
            entry = dict(record)
            entry["split"] = stable_split(str(entry["month"]), int(entry["row_idx"]), seed)
            off, n = int(entry["row_off"]), int(entry["n_rows"])
            entry["rows"] = np.array(raw[off:off + n], dtype=np.int64, copy=True)
            entry["legal"] = (np.array(legal[off:off + n], copy=True) if legal is not None
                              else np.zeros(n, dtype=LEGAL_DTYPE))
            self.items.append(entry)
        if not self.items:
            raise RuntimeError(f"no games under --max-rows={max_rows}")
        self.splits = {name: [x for x in self.items if x["split"] == name]
                       for name in ("train", "val", "test")}
        if not self.splits["train"] or not self.splits["val"]:
            raise RuntimeError("pilot split lacks train or val records; build a larger smoke corpus")


def batchify(items: list[dict], vocab: Vocab, device: torch.device, max_text: int,
             band_offset: int = 0, expand_bands: bool = False,
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a list of game-side records into model-ready tensors.

    With ``expand_bands=True`` every item is repeated once per label band
    (``BANDS``, currently 5), so a batch of N games becomes N*len(BANDS) rows
    in one pass: the same move sequence paired with each of its distinct
    style texts, giving len(BANDS) updates per game per step instead of 1.
    ``band_offset`` still selects a single rotating band per item otherwise,
    used by the plain per-step training loop and by eval, which only need
    one band per game.
    """
    if expand_bands:
        expanded_items = [item for item in items for _ in BANDS]
        band_index = [i % len(BANDS) for i in range(len(expanded_items))]
        items = expanded_items
    else:
        band_index = [(i + band_offset) % len(BANDS) for i in range(len(items))]
    width = max(len(x["rows"]) for x in items)
    rows = np.zeros((len(items), width, 5), dtype=np.int64)
    # Build uint64 then reinterpret as int64.  The occupancy high bit is a
    # chess square, not a numeric magnitude; torch's signed bit operations
    # preserve its exact pattern (the same contract as train_pretrain.py).
    occ = np.zeros((len(items), width), dtype=np.uint64)
    pmask = np.zeros((len(items), width), dtype=np.uint64)
    focal = np.zeros((len(items), width - 1), dtype=np.bool_)
    text_ids: list[list[int]] = []
    for i, item in enumerate(items):
        game = item["rows"]
        rows[i, :len(game)] = game
        occ[i, :len(game)] = item["legal"]["occ"]
        pmask[i, :len(game)] = item["legal"]["pmask"]
        # Target j is row j+1. White plays ply 0 (target j=1), black ply 1.
        j = np.arange(len(game) - 1)
        wanted_parity = 0 if item["side"] == "white" else 1
        valid = (game[j + 1, 0] == KIND_MOVE) & ((j % 2) == wanted_parity)
        focal[i, :len(j)] = valid
        text = item[BANDS[band_index[i]]]
        text_ids.append(vocab.encode(text, max_text))
    text_width = max(len(x) for x in text_ids)
    text = np.zeros((len(items), text_width), dtype=np.int64)
    for i, seq in enumerate(text_ids):
        text[i, :len(seq)] = seq
    return (torch.as_tensor(rows, device=device), torch.as_tensor(text, device=device),
            torch.as_tensor(focal, device=device),
            torch.as_tensor(occ.view(np.int64), device=device),
            torch.as_tensor(pmask.view(np.int64), device=device),
            torch.as_tensor(band_index, dtype=torch.long, device=device))


def focal_pool(hidden: torch.Tensor, rows: torch.Tensor, side: list[str]) -> torch.Tensor:
    # States at a move row encode the preceding game context, including the
    # opponent, and are pooled only for the described player's own plies.
    positions = torch.arange(rows.shape[1], device=rows.device).unsqueeze(0)
    parity = torch.tensor([0 if x == "white" else 1 for x in side], device=rows.device).unsqueeze(1)
    mask = rows[..., 0].eq(KIND_MOVE) & (((positions - 1) % 2) == parity)
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def focal_ce(logits: dict[str, torch.Tensor], target: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    # Every retained target is a move, so predict kind plus all factored heads.
    losses = []
    for key, column in (("kind", 0), ("src", 1), ("dst", 2), ("piece", 3), ("promo", 4)):
        y = target[..., column].masked_fill(~focal, -100)
        losses.append(F.cross_entropy(logits[key].float().reshape(-1, logits[key].shape[-1]),
                                      y.reshape(-1), ignore_index=-100))
    return sum(losses)


def focal_logprob(logits: dict[str, torch.Tensor], target: torch.Tensor, focal: torch.Tensor) -> torch.Tensor:
    """Mean joint factor log-probability per focal move, one value/example."""
    total = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
    for key, column in (("kind", 0), ("src", 1), ("dst", 2), ("piece", 3), ("promo", 4)):
        logp = F.log_softmax(logits[key].float(), dim=-1)
        total += logp.gather(-1, target[..., column].unsqueeze(-1)).squeeze(-1).masked_fill(~focal, 0).sum(1)
    return total / focal.sum(1).clamp_min(1)


def focal_logprob_per_token(logits: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Joint factor log-probability at every position, unmasked/unpooled.

    Unlike ``focal_logprob``, this keeps one value per (example, position)
    instead of collapsing a whole trajectory into a single mean - the caller
    applies the focal mask and any further reduction (e.g. per-move
    contrastive margins want a distinct gradient at every labelled ply, not
    one pooled scalar per game).
    """
    total = torch.zeros(target.shape[:2], device=target.device, dtype=torch.float32)
    for key, column in (("kind", 0), ("src", 1), ("dst", 2), ("piece", 3), ("promo", 4)):
        logp = F.log_softmax(logits[key].float(), dim=-1)
        total = total + logp.gather(-1, target[..., column].unsqueeze(-1)).squeeze(-1)
    return total


def per_move_margin_loss(matched_logits: dict[str, torch.Tensor], wrong_logits: dict[str, torch.Tensor],
                         target: torch.Tensor, focal: torch.Tensor, beta: float) -> torch.Tensor:
    """A distinct contrastive loss contribution at every labelled ply.

    ``--conditional-beta`` pools each game's focal log-probability into one
    scalar before comparing matched vs. mismatched context, so a whole
    trajectory gets a single bit of contrastive signal that a few
    good/bad moves can wash out.  This instead compares matched vs.
    mismatched log-probability move-by-move, so every labelled ply
    contributes its own gradient.
    """
    own_t = focal_logprob_per_token(matched_logits, target)
    wrong_t = focal_logprob_per_token(wrong_logits, target)
    diff = (own_t - wrong_t)[focal]
    if diff.numel() == 0:
        return torch.zeros((), device=target.device, dtype=torch.float32)
    return -F.logsigmoid(beta * diff).mean()


def wrong_context_roll(enc: torch.Tensor, expand_bands: bool) -> torch.Tensor:
    """Roll the batch of encoder contexts to build a mismatched-text control.

    With ``--expand-bands`` the batch is laid out as len(BANDS) consecutive
    rows per game (game0-band0..4, game1-band0..4, ...); rolling by 1 would
    land on a *different band describing the same game*, which is not a
    true negative since all bands describe the same real trajectory.  Roll
    by len(BANDS) instead so the control always lands on a different game.
    """
    return enc.roll(len(BANDS) if expand_bands else 1, dims=0)


def focal_illegal_mass(logits: dict[str, torch.Tensor], target: torch.Tensor,
                       own_occ: torch.Tensor, piece_mask_bits: torch.Tensor,
                       focal: torch.Tensor) -> torch.Tensor:
    """Soft structural-illegal mass, restricted to labelled-player moves.

    This is the SFT analogue of the production pretraining penalty.  It keeps
    the regular CE over all classes, then adds differentiable pressure against
    own-square destinations, unavailable source squares/piece types, and
    invalid promotion class.  Exact move legality remains enforced at rollout
    time by whole-move legal sampling.
    """
    src_allowed, dst_allowed, piece_allowed = legal_component_masks_from_bits(own_occ, piece_mask_bits)
    piece_t, dst_t = target[..., 3], target[..., 2]
    must_promote = (piece_t == 0) & ((dst_t < 8) | (dst_t >= 56))
    promo_allowed = torch.zeros(*piece_t.shape, N_PROMOTIONS, dtype=torch.bool, device=target.device)
    promo_allowed[..., 0] = ~must_promote
    promo_allowed[..., 1:] = must_promote.unsqueeze(-1)
    denom = focal.sum().clamp_min(1)
    total = torch.zeros((), device=target.device)
    for name, allowed in (("src", src_allowed), ("dst", dst_allowed),
                          ("piece", piece_allowed), ("promo", promo_allowed)):
        mass = (F.softmax(logits[name].float(), dim=-1) * (~allowed).float()).sum(-1)
        total = total + mass.masked_select(focal).sum() / denom
    return total


def move_text_nce_loss(hidden: torch.Tensor, focal: torch.Tensor, text_embed: torch.Tensor,
                       move_proj: nn.Module, text_proj: nn.Module, temperature: float) -> torch.Tensor:
    """Per-move InfoNCE: project each focal ply's own hidden state and every
    row's pooled text into one joint space, then classify each move latent
    against *every row's* text in the batch (softmax over all candidates,
    not a single matched-vs-one-mismatched margin like per_move_margin_loss).

    With --expand-bands, a batch of N games x len(BANDS) rows gives an
    N*len(BANDS) x N*len(BANDS) candidate matrix per focal position -- e.g.
    32 games x 5 bands = 160 rows, 160^2 = 25,600 pairs, cheap relative to
    the decoder forward pass itself. This is the direct "align each move's
    own representation with its describing text's representation" loss,
    at the representation level rather than through move likelihood.
    """
    move = F.normalize(move_proj(hidden), dim=-1)             # (B, T, J)
    text = F.normalize(text_proj(text_embed), dim=-1)         # (B, J)
    logits = torch.einsum("btj,nj->btn", move, text) / temperature  # (B, T, B)
    labels = torch.arange(hidden.shape[0], device=hidden.device).view(-1, 1).expand(-1, hidden.shape[1])
    return F.cross_entropy(logits[focal], labels[focal])


def config_from_checkpoint(path: Path, device: torch.device) -> tuple[FactoredChessDecoder, DecoderConfig, dict | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    style_payload = payload if payload.get("format") == "style-autoresearch-v1" else None
    old = payload["cfg"]
    get = (lambda k, default=None: old.get(k, default)) if isinstance(old, dict) else (lambda k, default=None: getattr(old, k, default))
    cfg = DecoderConfig(vocab_size=1, d_model=int(get("d_model")), n_heads=int(get("n_heads")),
                        n_layers=int(get("depth", get("n_layers"))), d_ff=int(get("d_ff")),
                        max_seq_len=int(get("seq_len", get("max_seq_len", 512))),
                        dropout=0.0, use_cross_attention=True, architecture=str(get("architecture", "llama")),
                        llama_qk_norm=bool(get("llama_qk_norm", True)))
    chess = FactoredChessDecoder(cfg)
    state = payload["chess"] if style_payload else payload.get("model_state_dict", payload.get("model", payload))
    chess.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return chess.to(device), cfg, style_payload


@torch.no_grad()
def validate_clip(chess, text_tower, game_proj, text_proj, data, vocab, device, max_text, batch_size):
    chess.eval(); text_tower.eval(); game_proj.eval(); text_proj.eval()
    games, texts = [], []
    val = data.splits["val"]
    for start in range(0, len(val), batch_size):
        chunk = val[start:start + batch_size]
        rows, text, _, _, _, _ = batchify(chunk, vocab, device, max_text, band_offset=2)
        # Retrieval must not condition its chess tower on the candidate text:
        # retain the production checkpoint's zero dummy context instead.
        h = chess.hidden_states(rows, enc=make_dummy_context(len(chunk), chess.norm.weight.numel(), device=device))
        game = F.normalize(game_proj(focal_pool(h, rows, [x["side"] for x in chunk])), dim=-1)
        _, pooled = text_tower(text)
        texts.append(F.normalize(text_proj(pooled), dim=-1)); games.append(game)
    game, text = torch.cat(games), torch.cat(texts)
    ranks = (game @ text.T).argsort(dim=-1, descending=True).eq(torch.arange(len(game), device=device)[:, None]).nonzero()[:, 1] + 1
    return {"recall_at_1": float((ranks <= 1).float().mean()), "recall_at_10": float((ranks <= 10).float().mean()),
            "median_rank": float(ranks.float().median()), "n_val": int(len(game))}


@torch.no_grad()
def validate_sft(chess, text_tower, data, vocab, device, max_text, batch_size, context_scale=1.0,
                 split="val", style_probe=None):
    chess.eval(); text_tower.eval()
    if style_probe is not None:
        style_probe.eval()
    values, mismatched_values, illegal_values, zero_values = [], [], [], []
    probe_correct = probe_total = 0
    flip_vs_mismatched = flip_vs_zero = flip_total = 0

    def move_identity(logits: dict[str, torch.Tensor]) -> torch.Tensor:
        # (src, dst, promo) argmax stacked into one comparable key per position --
        # kind is trivially "move" at focal positions and piece is redundant with
        # src, so this is the smallest tuple that actually identifies a distinct move.
        return torch.stack([logits["src"].argmax(-1), logits["dst"].argmax(-1), logits["promo"].argmax(-1)], dim=-1)

    for start in range(0, len(data.splits[split]), batch_size):
        chunk = data.splits[split][start:start + batch_size]
        rows, text, focal, occ, pmask, band_idx = batchify(chunk, vocab, device, max_text, band_offset=3)
        enc, _ = text_tower(text)
        matched_hidden = chess.hidden_states(rows[:, :-1], enc=enc * context_scale)
        matched_logits = chess.head(matched_hidden)
        values.append(focal_ce(matched_logits, rows[:, 1:], focal).item())
        illegal_values.append(focal_illegal_mass(matched_logits, rows[:, 1:], occ[:, 1:], pmask[:, 1:], focal).item())
        matched_move = move_identity(matched_logits)
        # A deterministic wrong-description control.  It catches improvements
        # caused by a generic learned encoder bias rather than player-style
        # information: a usable conditioner should score its own text better.
        if len(chunk) > 1:
            wrong_enc = enc.roll(1, dims=0)
            wrong_logits = chess(rows[:, :-1], enc=wrong_enc * context_scale)
            mismatched_values.append(focal_ce(wrong_logits, rows[:, 1:], focal).item())
            # Counterfactual move-flip rate: does the model's actual *decision*
            # (argmax move) change under a different text, not just its log-prob?
            # A conditioner that only nudges probabilities without ever flipping
            # the greedy choice is not really steering generation.
            flipped = (matched_move != move_identity(wrong_logits)).any(-1) & focal
            flip_vs_mismatched += int(flipped.sum())
        # Third arm: true zero (unconditioned) context, same dummy used at
        # pretraining time -- distinguishes "beats a wrong description" from
        # "beats having no description at all," which a generic learned bias
        # in the encoder pathway could satisfy without real conditioning.
        zero_enc = make_dummy_context(len(chunk), enc.shape[-1], device=device)
        zero_logits = chess(rows[:, :-1], enc=zero_enc * context_scale)
        zero_values.append(focal_ce(zero_logits, rows[:, 1:], focal).item())
        flipped_zero = (matched_move != move_identity(zero_logits)).any(-1) & focal
        flip_vs_zero += int(flipped_zero.sum())
        flip_total += int(focal.sum())
        if style_probe is not None:
            probe_logits = style_probe(matched_hidden)
            pred = probe_logits.argmax(-1)
            correct = (pred == band_idx.view(-1, 1)) & focal
            probe_correct += int(correct.sum()); probe_total += int(focal.sum())
    suffix = "val" if split == "val" else split
    result = {f"focal_{suffix}_loss": float(np.mean(values)),
              f"focal_{suffix}_illegal_mass": float(np.mean(illegal_values)),
              f"n_{suffix}": len(data.splits[split]),
              f"zero_context_focal_{suffix}_loss": float(np.mean(zero_values)),
              f"matched_minus_zero_{suffix}_loss": float(np.mean(values)) - float(np.mean(zero_values)),
              f"move_flip_rate_vs_zero_{suffix}": flip_vs_zero / max(flip_total, 1)}
    if mismatched_values:
        result[f"mismatched_text_focal_{suffix}_loss"] = float(np.mean(mismatched_values))
        result[f"matched_minus_mismatched_{suffix}_loss"] = (
            result[f"focal_{suffix}_loss"] - result[f"mismatched_text_focal_{suffix}_loss"]
        )
        result[f"move_flip_rate_vs_mismatched_{suffix}"] = flip_vs_mismatched / max(flip_total, 1)
    if style_probe is not None:
        result[f"style_probe_accuracy_{suffix}"] = probe_correct / max(probe_total, 1)
    return result


@torch.no_grad()
def validate_preference(chess, text_tower, data, vocab, device, max_text, batch_size,
                        context_scale=1.0):
    chess.eval(); text_tower.eval()
    margins, nll = [], []
    for start in range(0, len(data.splits["val"]), batch_size):
        chunk = data.splits["val"][start:start + batch_size]
        rows, positive, focal, _, _, _ = batchify(chunk, vocab, device, max_text, band_offset=3)
        negative = positive.roll(1, dims=0)
        pos_enc, _ = text_tower(positive); neg_enc, _ = text_tower(negative)
        pos = focal_logprob(chess(rows[:, :-1], enc=pos_enc * context_scale), rows[:, 1:], focal)
        neg = focal_logprob(chess(rows[:, :-1], enc=neg_enc * context_scale), rows[:, 1:], focal)
        margins.append(pos - neg); nll.append(-pos)
    margin, nll = torch.cat(margins), torch.cat(nll)
    return {"preference_pair_accuracy": float((margin > 0).float().mean()),
            "preference_margin": float(margin.mean()), "positive_focal_nll": float(nll.mean()), "n_val": len(margin)}


def run(args):
    if args.expand_bands and args.accumulate_bands:
        raise ValueError("--expand-bands and --accumulate-bands are mutually exclusive; "
                          "both give len(BANDS) updates/game/step, trading memory for micro-step wall time")
    if args.accumulate_bands and args.mode != "sft":
        raise ValueError("--accumulate-bands is only implemented for --mode sft")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Architecture/training seeds must not silently redefine held-out games.
    # Keep a separately explicit split seed for comparable autoresearch trials.
    data = StyleData(Path(args.data), args.split_seed, args.max_rows)
    train_text = [item[band] for item in data.splits["train"] for band in BANDS]
    vocab = Vocab(train_text, args.vocab_size)
    # A pretrained text tower is tied to its word-ID map.  Reuse that map on
    # a larger corpus rather than silently assigning the same embedding rows
    # to different words.  New words become UNK until a larger CLIP tower is
    # trained deliberately.
    text_init_payload = None
    if args.text_init:
        text_init_payload = torch.load(args.text_init, map_location="cpu", weights_only=False)
        if text_init_payload.get("format") != "style-autoresearch-v1":
            raise RuntimeError("--text-init must be a style-autoresearch artifact")
        vocab.ids = text_init_payload["vocab"]
    chess, cfg, style_payload = config_from_checkpoint(Path(args.checkpoint), device)
    if style_payload is not None:
        # An SFT artifact is likewise tied to its saved word IDs.  This lets a
        # selected 20k artifact be evaluated on the untouched 20k test split.
        if args.text_init and vocab.ids != style_payload["vocab"]:
            raise RuntimeError("--checkpoint and --text-init use different vocabularies")
        vocab.ids = style_payload["vocab"]
    # Full SFT is intentionally full-model trainable. CLIP variants choose the
    # chess adaptation policy explicitly, avoiding a misleading shared setup.
    if args.mode == "clip" and args.chess_tune == "frozen":
        for p in chess.parameters(): p.requires_grad = False
    elif args.mode == "clip" and args.chess_tune == "last":
        for p in chess.parameters(): p.requires_grad = False
        if not 1 <= args.unfreeze_last_layers <= len(chess.layers):
            raise ValueError(f"--unfreeze-last-layers must be within [1, {len(chess.layers)}]")
        for layer in chess.layers[-args.unfreeze_last_layers:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in chess.norm.parameters(): p.requires_grad = True
    elif args.mode == "sft" and args.chess_tune in ("text", "cross"):
        for p in chess.parameters(): p.requires_grad = False
        if args.chess_tune == "cross":
            for layer in chess.layers:
                if layer.use_cross_attention:
                    for p in layer.cross_attn.parameters(): p.requires_grad = True
                    for p in layer.n2.parameters(): p.requires_grad = True
    text_tower = TextTower(len(vocab), cfg.d_model, args.max_text, args.text_tower).to(device)
    context_scale = nn.Parameter(torch.zeros((), device=device))
    reference_chess = reference_text = reference_context_scale = None
    game_proj = nn.Linear(cfg.d_model, args.joint_dim, bias=False).to(device)
    text_proj = nn.Linear(cfg.d_model, args.joint_dim, bias=False).to(device)
    style_probe = nn.Linear(cfg.d_model, len(BANDS)).to(device) if args.style_probe_weight else None
    if style_payload is not None:
        if vocab.ids != style_payload["vocab"]:
            raise RuntimeError("style checkpoint vocabulary differs from current immutable pilot split")
        text_tower.load_state_dict(style_payload["text_tower"], strict=True)
        game_proj.load_state_dict(style_payload["game_proj"], strict=True)
        text_proj.load_state_dict(style_payload["text_proj"], strict=True)
        # v1 artifacts predating this field are still valid zero-context
        # baselines; new artifacts preserve the learned conditioning strength.
        context_scale.data.copy_(torch.as_tensor(style_payload.get("context_scale", 0.0),
                                                  device=device, dtype=context_scale.dtype))
    if args.text_init:
        text_tower.load_state_dict(text_init_payload["text_tower"], strict=True)
    if args.freeze_text:
        for p in text_tower.parameters():
            p.requires_grad = False
    if args.mode == "preference":
        # Copy only *after* loading the SFT artifact.  This makes the DPO
        # reference an immutable copy of the actual SFT policy, not a random
        # newly-initialized text tower.
        reference_chess = copy.deepcopy(chess).eval()
        reference_text = copy.deepcopy(text_tower).eval()
        reference_context_scale = context_scale.detach().clone()
        for module in (reference_chess, reference_text):
            for param in module.parameters(): param.requires_grad = False
    modules = [chess, text_tower, game_proj, text_proj] + ([style_probe] if style_probe is not None else [])
    params = [p for module in modules for p in module.parameters() if p.requires_grad] + [context_scale]
    # The pretrained chess backbone and newly introduced modules should not
    # share a learning rate.  Pretraining used a zero dummy context and froze
    # every cross-attention/n2 weight, so that route is effectively new at SFT
    # time; CLIP's projection heads are likewise new.  The remaining chess
    # weights are an already-trained policy.
    if args.text_lr is None and args.cross_lr is None and args.head_lr is None:
        opt_params = params
    else:
        text_params = [p for p in text_tower.parameters() if p.requires_grad]
        head_params = [p for module in ((game_proj, text_proj) + ((style_probe,) if style_probe is not None else ()))
                       for p in module.parameters() if p.requires_grad]
        # Cross-attention and its preceding norm were frozen throughout
        # pretraining with a zero dummy context.  They are functionally new
        # conditioning weights; every other chess parameter is the pretrained
        # decoder backbone.  Keep those groups explicit for heterogeneous-LR
        # post-training instead of conflating them with the text encoder.
        cross_params = [p for name, p in chess.named_parameters()
                        if p.requires_grad and (".cross_attn." in name or ".n2." in name)]
        text_ids = {id(p) for p in text_params}
        cross_ids = {id(p) for p in cross_params}
        head_ids = {id(p) for p in head_params}
        base_params = [p for p in params if id(p) not in text_ids and id(p) not in cross_ids
                       and id(p) not in head_ids
                       and p is not context_scale]
        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": args.lr})
        if cross_params or context_scale.requires_grad:
            groups.append({"params": cross_params + [context_scale],
                           "lr": args.cross_lr if args.cross_lr is not None else args.lr})
        if text_params:
            groups.append({"params": text_params, "lr": args.text_lr if args.text_lr is not None else args.lr})
        if head_params:
            groups.append({"params": head_params, "lr": args.head_lr if args.head_lr is not None else args.lr})
        opt_params = groups
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start = time.time(); measured_s = 0.0; losses = []; step = 0
    # A detached queue adds in-batch-style negatives without trading away the
    # update frequency that made b64 beat b128.  It is intentionally limited
    # to the training objective; held-out retrieval remains the same global
    # all-pairs metric for every candidate.
    game_queue = text_queue = None
    # Karpathy's useful invariant: architecture candidates receive identical
    # *training wall time*, not identical steps.  Startup/cache/autocast
    # warmup is excluded so it cannot decide an experiment.
    while (not args.baseline_only and not args.eval_only and step < args.max_steps
           and (step <= args.warmup_steps or measured_s < args.time_budget)):
        step += 1
        step_start = time.time()
        chess.train(); text_tower.train(); game_proj.train(); text_proj.train()
        sample = random.sample(data.splits["train"], min(args.batch_size, len(data.splits["train"])))
        if args.accumulate_bands:
            # Same len(BANDS) updates per game per step as --expand-bands, run as
            # len(BANDS) sequential single-band micro-batches accumulated into one
            # optimizer step instead of one batch*len(BANDS) forward pass. Total
            # FLOPs are identical; peak activation memory stays at the single-band
            # size, since the decoder never sees more than `batch_size` rows at once.
            opt.zero_grad(set_to_none=True)
            band_losses = []
            for band in range(len(BANDS)):
                b_rows, b_text, b_focal, b_occ, b_pmask, b_band = batchify(
                    sample, vocab, device, args.max_text, band_offset=band)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    b_enc, _ = text_tower(b_text)
                    b_logits = chess(b_rows[:, :-1], enc=b_enc * context_scale)
                    band_loss = focal_ce(b_logits, b_rows[:, 1:], b_focal)
                    if args.illegal_penalty_weight:
                        band_loss = band_loss + args.illegal_penalty_weight * focal_illegal_mass(
                            b_logits, b_rows[:, 1:], b_occ[:, 1:], b_pmask[:, 1:], b_focal)
                    if args.per_move_beta:
                        # accumulate-bands never mixes bands in one batch, so a
                        # plain roll-by-1 already lands on a different game.
                        wrong_enc = wrong_context_roll(b_enc, expand_bands=False)
                        wrong_logits = chess(b_rows[:, :-1], enc=wrong_enc * context_scale)
                        band_loss = band_loss + per_move_margin_loss(
                            b_logits, wrong_logits, b_rows[:, 1:], b_focal, args.per_move_beta)
                if not torch.isfinite(band_loss):
                    raise RuntimeError(f"non-finite loss at step {step} band {band}: {float(band_loss)}")
                (band_loss / len(BANDS)).backward()
                band_losses.append(float(band_loss.detach()))
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip); opt.step()
            loss_value = sum(band_losses) / len(band_losses)
        else:
            rows, text, focal, occ, pmask, band_idx = batchify(sample, vocab, device, args.max_text, band_offset=step,
                                                      expand_bands=args.expand_bands)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                enc, pooled = text_tower(text)
                if args.mode == "sft" and args.cfg_dropout:
                    # Classifier-free-guidance-style training: force a real gap between
                    # P(move) and P(move|text) by literally training on the zero dummy
                    # context part of the time, per row, so the model cannot satisfy CE
                    # by learning to ignore the conditioning pathway entirely.
                    drop = torch.rand(enc.shape[0], device=device) < args.cfg_dropout
                    enc = torch.where(drop.view(-1, 1, 1), torch.zeros_like(enc), enc)
                if args.mode == "clip":
                    hidden = chess.hidden_states(
                        rows, enc=make_dummy_context(len(sample), cfg.d_model, device=device)
                    )
                    game = F.normalize(game_proj(focal_pool(hidden, rows, [x["side"] for x in sample])), dim=-1)
                    description = F.normalize(text_proj(pooled), dim=-1)
                    logits_g = game @ description.T
                    logits_t = description @ game.T
                    if game_queue is not None:
                        logits_g = torch.cat((logits_g, game @ text_queue.T), dim=1)
                        logits_t = torch.cat((logits_t, description @ game_queue.T), dim=1)
                    logits_g = logits_g / args.temperature
                    logits_t = logits_t / args.temperature
                    labels = torch.arange(len(sample), device=device)
                    loss = (F.cross_entropy(logits_g, labels) + F.cross_entropy(logits_t, labels)) / 2
                    if args.queue_size:
                        with torch.no_grad():
                            game_queue = game.detach() if game_queue is None else torch.cat((game_queue, game.detach()), 0)
                            text_queue = description.detach() if text_queue is None else torch.cat((text_queue, description.detach()), 0)
                            game_queue, text_queue = game_queue[-args.queue_size:], text_queue[-args.queue_size:]
                elif args.mode == "sft":
                    # The text sequence is the true encoder context, replacing the
                    # zero dummy context used during pretraining.
                    matched_hidden = chess.hidden_states(rows[:, :-1], enc=enc * context_scale)
                    matched_logits = chess.head(matched_hidden)
                    loss = focal_ce(matched_logits, rows[:, 1:], focal)
                    if args.illegal_penalty_weight:
                        loss = loss + args.illegal_penalty_weight * focal_illegal_mass(
                            matched_logits, rows[:, 1:], occ[:, 1:], pmask[:, 1:], focal)
                    if style_probe is not None:
                        # Can the *label band itself* be recovered from the decoder's own
                        # hidden state at each focal ply? A direct, generation-quality-
                        # independent check that conditioning info is actually present in
                        # the representation, not just correlated with lower CE.
                        probe_logits = style_probe(matched_hidden)
                        probe_target = band_idx.view(-1, 1).expand(-1, focal.shape[1]).masked_fill(~focal, -100)
                        loss = loss + args.style_probe_weight * F.cross_entropy(
                            probe_logits.reshape(-1, len(BANDS)), probe_target.reshape(-1), ignore_index=-100)
                    if args.move_text_nce_weight:
                        loss = loss + args.move_text_nce_weight * move_text_nce_loss(
                            matched_hidden, focal, pooled, game_proj, text_proj, args.temperature)
                    wrong_logits = None
                    if args.conditional_beta or args.per_move_beta:
                        # Conditional likelihood ranking is supervised by the
                        # paired style label itself: the observed player's moves
                        # should be more probable under their description than
                        # under another player's description.  Unlike simply
                        # maximizing CE, this supplies a direct gradient to use
                        # the encoder context while retaining the ordinary
                        # predictive objective.  With --expand-bands the batch
                        # is len(BANDS) consecutive rows per game, so the roll
                        # must skip a whole game's worth of rows, not just 1.
                        wrong_enc = wrong_context_roll(enc, expand_bands=args.expand_bands)
                        wrong_logits = chess(rows[:, :-1], enc=wrong_enc * context_scale)
                    if args.conditional_beta:
                        own = focal_logprob(matched_logits, rows[:, 1:], focal)
                        wrong = focal_logprob(wrong_logits, rows[:, 1:], focal)
                        loss = loss + args.conditional_beta * -F.logsigmoid(own - wrong).mean()
                    if args.per_move_beta:
                        loss = loss + per_move_margin_loss(
                            matched_logits, wrong_logits, rows[:, 1:], focal, args.per_move_beta)
                else:
                    negative = text.roll(1, dims=0)
                    pos_enc, _ = text_tower(text)
                    neg_enc, _ = text_tower(negative)
                    pos = focal_logprob(chess(rows[:, :-1], enc=pos_enc * context_scale), rows[:, 1:], focal)
                    neg = focal_logprob(chess(rows[:, :-1], enc=neg_enc * context_scale), rows[:, 1:], focal)
                    with torch.no_grad():
                        ref_pos_enc, _ = reference_text(text)
                        ref_neg_enc, _ = reference_text(negative)
                        ref_pos = focal_logprob(reference_chess(rows[:, :-1], enc=ref_pos_enc * reference_context_scale), rows[:, 1:], focal)
                        ref_neg = focal_logprob(reference_chess(rows[:, :-1], enc=ref_neg_enc * reference_context_scale), rows[:, 1:], focal)
                    loss = -F.logsigmoid(args.beta * ((pos - neg) - (ref_pos - ref_neg))).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip); opt.step()
            loss_value = float(loss.detach())
        if step > args.warmup_steps:
            torch.cuda.synchronize() if device.type == "cuda" else None
            measured_s += time.time() - step_start
        losses.append(loss_value)
        if step % args.log_every == 0:
            print(json.dumps({"step": step, "loss": losses[-1], "training_seconds": measured_s}, sort_keys=True), flush=True)
    metrics = (validate_clip(chess, text_tower, game_proj, text_proj, data, vocab, device, args.max_text, args.batch_size)
               if args.mode == "clip" else (validate_sft(chess, text_tower, data, vocab, device, args.max_text, args.batch_size,
                                                         context_scale, args.eval_split, style_probe=style_probe)
               if args.mode == "sft" else validate_preference(chess, text_tower, data, vocab, device, args.max_text, args.batch_size, context_scale)))
    result = {"mode": args.mode, "trial": args.trial, "checkpoint": str(args.checkpoint),
              "chess_tune": args.chess_tune, "text_tower": args.text_tower, "steps": step,
              "batch_size": args.batch_size, "unfreeze_last_layers": args.unfreeze_last_layers,
              "lr": args.lr, "text_lr": args.text_lr,
              "cross_lr": args.cross_lr, "head_lr": args.head_lr,
              "joint_dim": args.joint_dim, "queue_size": args.queue_size,
              "illegal_penalty_weight": args.illegal_penalty_weight,
              "conditional_beta": args.conditional_beta,
              "per_move_beta": args.per_move_beta,
              "cfg_dropout": args.cfg_dropout,
              "style_probe_weight": args.style_probe_weight,
              "move_text_nce_weight": args.move_text_nce_weight,
              "train_loss_last": losses[-1] if losses else None,
              "train_loss_mean": float(np.mean(losses[-20:])) if losses else None,
              "training_seconds": measured_s, "elapsed_s": time.time() - start,
              "eval_only": args.eval_only,
              "data": {x: len(data.splits[x]) for x in data.splits},
              "split_seed": args.split_seed, "text_init": args.text_init,
              "freeze_text": args.freeze_text, "vocab": len(vocab), "cfg": asdict(cfg), **metrics}
    if args.save_dir and not args.baseline_only and not args.eval_only:
        save_path = Path(args.save_dir) / f"{args.trial}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "style-autoresearch-v1", "result": result, "cfg": asdict(cfg),
                    "vocab": vocab.ids, "chess": chess.state_dict(), "text_tower": text_tower.state_dict(),
                    "game_proj": game_proj.state_dict(), "text_proj": text_proj.state_dict(),
                    "context_scale": float(context_scale.detach())}, save_path)
        result["artifact"] = str(save_path)
    out = Path(args.results); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle: handle.write(json.dumps(result, sort_keys=True) + "\n")
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("clip", "sft", "preference"), required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--data", default=str(ROOT / "data/style_smoke"))
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/full_llama_d12w960ff3520_qknorm_cosine5pct/milestones/pct35.00_step418274.pt"))
    p.add_argument("--results", default=str(ROOT / "results/style_autoresearch.jsonl"))
    p.add_argument("--save-dir", default=str(ROOT / "checkpoints/style_autoresearch"))
    p.add_argument("--time-budget", type=float, default=300,
                   help="measured train seconds; excludes --warmup-steps")
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--baseline-only", action="store_true",
                   help="evaluate the unmodified checkpoint with exactly zero text context")
    p.add_argument("--eval-only", action="store_true",
                   help="evaluate a style artifact exactly as loaded; performs zero optimizer updates")
    p.add_argument("--eval-split", choices=("val", "test"), default="val",
                   help="immutable split used for the final metric")
    p.add_argument("--max-steps", type=int, default=100000,
                   help="safety ceiling if a step is implausibly fast")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--max-text", type=int, default=96)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--joint-dim", type=int, default=512)
    p.add_argument("--text-tower", choices=("bag", "transformer"), default="bag")
    p.add_argument("--text-init", default=None,
                   help="optional CLIP/SFT artifact supplying a compatible text tower")
    p.add_argument("--freeze-text", action="store_true",
                   help="freeze text tower after loading --text-init")
    p.add_argument("--chess-tune", choices=("frozen", "last", "full", "text", "cross"), default="last")
    p.add_argument("--unfreeze-last-layers", type=int, default=1,
                   help="number of pretrained chess blocks to tune when --chess-tune=last")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--text-lr", type=float, default=None,
                   help="optional LR for the new text-conditioning encoder")
    p.add_argument("--cross-lr", type=float, default=None,
                   help="optional LR for previously unused cross-attention/n2 weights")
    p.add_argument("--head-lr", type=float, default=None,
                   help="optional LR for newly initialized CLIP projection heads")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--queue-size", type=int, default=0,
                   help="detached prior-batch negatives for CLIP; 0 disables the queue")
    p.add_argument("--illegal-penalty-weight", type=float, default=0.0,
                   help="SFT soft structural-illegal-mass penalty; production pretraining uses 0.5")
    p.add_argument("--expand-bands", action="store_true",
                   help="Pair every sampled game with all len(BANDS) label texts in one batch/step "
                        "(len(BANDS) updates per game per step) instead of one rotating band per game. "
                        "Materializes a batch_size*len(BANDS) tensor: same total FLOPs as --accumulate-bands "
                        "but len(BANDS)x the peak activation memory in exchange for one optimizer step's "
                        "worth of parallelism instead of len(BANDS) sequential micro-steps.")
    p.add_argument("--accumulate-bands", action="store_true",
                   help="sft-only. Same len(BANDS) updates per game per step as --expand-bands, but as "
                        "len(BANDS) sequential single-band micro-batches with gradients accumulated into "
                        "one optimizer step, so peak activation memory stays at batch_size (not "
                        "batch_size*len(BANDS)). Mutually exclusive with --expand-bands.")
    p.add_argument("--beta", type=float, default=0.1, help="DPO inverse-temperature for --mode preference")
    p.add_argument("--conditional-beta", type=float, default=0.0,
                   help="SFT likelihood-ranking weight for own text vs in-batch wrong text "
                        "(one pooled margin per game/trajectory)")
    p.add_argument("--per-move-beta", type=float, default=0.0,
                   help="SFT per-move contrastive margin weight: same matched-vs-wrong-context "
                        "idea as --conditional-beta, but a distinct loss contribution at every "
                        "labelled ply instead of one pooled scalar per game. Supported by both "
                        "--expand-bands and --accumulate-bands (unlike --conditional-beta, which "
                        "--accumulate-bands does not apply).")
    p.add_argument("--cfg-dropout", type=float, default=0.0,
                   help="classifier-free-guidance-style training: with this probability, replace "
                        "a given row's encoder context with the zero dummy context (per-row "
                        "Bernoulli, applied before matched_logits/wrong-context construction). "
                        "Forces the model to represent a genuine P(move) != P(move|text), rather "
                        "than letting the text pathway go unused while plain CE is still "
                        "satisfied by chess quality alone. 0.0 = disabled (matches prior runs).")
    p.add_argument("--move-text-nce-weight", type=float, default=0.0,
                   help="per-move InfoNCE weight: projects each focal ply's hidden state and "
                        "every row's pooled text into joint space, classifies each move latent "
                        "against every row's text in the batch (softmax over all candidates, not "
                        "just one mismatched pair). Reuses --joint-dim/--temperature. 0.0 = "
                        "disabled.")
    p.add_argument("--style-probe-weight", type=float, default=0.0,
                   help="auxiliary per-position linear-probe loss: predicts which of the 5 label "
                        "bands is in effect at each focal ply directly from the decoder's hidden "
                        "state, added to the main loss with this weight. Cheap, and gives an "
                        "interpretable diagnostic (probe accuracy) independent of generation "
                        "quality for whether style information is actually recoverable from the "
                        "representation, not just correlated with lower CE. 0.0 = disabled.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260802)
    p.add_argument("--split-seed", type=int, default=20260802,
                   help="immutable game-level train/validation/test split seed")
    run(p.parse_args())


if __name__ == "__main__":
    main()
