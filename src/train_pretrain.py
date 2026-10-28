#!/usr/bin/env python3
"""
train_pretrain.py — full-scale pretraining of the winning ablation config
(src/model.py:FactoredChessDecoder, frozen enc-dec + dummy context, i.e.
cross-attention present but frozen — see TRAINING.md section 1 for why this
beats the decoder-only (no-cross-attention) config that RESULTS.md's "FINAL
WRAP-UP" section documents as the winner: a later sweep run
(xattn1_d10_w512_dff2560_lr15e3, val_loss 3.651) actually beat it but
predates that write-up) on tier-3 data, across multiple GPUs via torchrun +
DistributedDataParallel.

This is the real-run counterpart of external/autoresearch/train_chess.py
(the ablation harness): same token representation, same factored loss, same
wall-clock-progress-driven LR schedule (see TOKEN_TRAINING.md), but built to
run for hours/days on a Slurm allocation instead of a fixed 55-minute
ablation budget, which needs three things the ablation harness didn't:
multi-GPU DDP, checkpoint/resume (a wall-clock LR schedule needs cumulative
elapsed time preserved across restarts, not reset to zero), and a data
loader that streams from disk instead of loading the whole corpus into RAM
(tier 3 is far too large for that — see TRAINING.md section 2).

Modes:
  (default)     real training run, wall-clock time-budget driven
  --smoke-test  synthetic in-memory data, tiny model, a few hundred steps —
                verifies the whole pipeline (model, loss, checkpoint
                save/load, LR schedule, divergence guard) without GPUs or
                real tier-3 data on disk. Run this before anything else.
  --lr-sweep    short fixed-step runs at each candidate LR (comma-separated),
                one per rank — run before committing a real time budget,
                see TRAINING.md section 1 (LR is unvalidated at untested
                widths, and this architecture is LR-sensitive).

Usage:
    python train_pretrain.py --smoke-test
    torchrun --standalone --nproc_per_node=4 train_pretrain.py \
        --time-budget 10800 --checkpoint-dir checkpoints/tier3_d10w512 --resume
    # (all architecture/optimizer flags default to the winning config above; override only to deviate)
"""
import argparse
import glob
import json
import math
import os
import queue
import random
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from factored_tokenizer import KIND_BOS, KIND_MOVE, ROW_BOS, ROW_EOS  # noqa: E402
# Keep this tiny on-disk contract local.  Importing build_tier3 also imports
# PyArrow, which is needed to construct tier-3 parquet but not to load a
# checkpoint for playing evaluation on the lean inference container.
LEGAL_DTYPE = np.dtype([("occ", "<u8"), ("pmask", "u1")])

# Measured directly from src/build_tier3.py's own printed TOTAL line for the full 162-month
# conversion (2026-07-29): 1,068,263,794 games, 732GB of int16. Used only to turn a measured
# tokens/sec into a human "how long would one full pass over tier-3 take" estimate at the end of
# a run (see run_training's DONE summary) -- not used anywhere in the training loop itself.
TIER3_TOTAL_ROWS = 73_195_039_502


def local_iso_now() -> str:
    """ISO-8601 wall time with the node's local UTC offset, to timestamp discoveries in logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_block_causal_mask(rows):
    """Per-batch-row block-diagonal causal mask, so packed games can't attend
    across each other's boundaries.

    Best-fit bin packing (Tier3Dataset.make_loader / SyntheticDataset) puts
    several games back-to-back in one sequence for full utilization (see
    TOKEN_TRAINING.md section 2, TRAINING.md section 3). A plain causal mask
    (nn.Transformer.generate_square_subsequent_mask, what
    external/autoresearch/train_chess.py's ablation harness uses) only
    blocks *future* positions — it does NOT stop a token in game 2 from
    attending back into game 1's tokens. At ablation scale (55-minute runs)
    that noise was never worth addressing; at real-run scale, over many more
    steps, it's cheap to fix and there's no reason to leave it in: every BOS
    row marks a fresh game, so segment id = cumulative BOS count, and a
    position may only attend to same-segment, non-future positions.
    """
    import torch

    B, T, _ = rows.shape
    seg_id = torch.cumsum((rows[..., 0] == KIND_BOS).long(), dim=1)  # (B,T)
    same_seg = seg_id.unsqueeze(2) == seg_id.unsqueeze(1)            # (B,T,T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=rows.device))
    allow = same_seg & causal.unsqueeze(0)                          # (B,T,T)
    # bool mask, True = ALLOWED to attend (model.py:MultiHeadAttention -> F.scaled_dot_product_attention
    # convention). No repeat_interleave over n_heads: SDPA broadcasts a (B,1,T,T) mask across heads
    # itself, so this never materializes a per-head copy -- unlike nn.MultiheadAttention's mask-present
    # path, which used to force a real (B*n_heads,T,T) tensor (4GiB+ already at B=512,n_heads=8,T=512).
    return allow.unsqueeze(1)                                       # (B, 1, T, T), bool


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "3"),
                     help="tier-3 root: <data-dir>/<month>/{rows.i16,meta.parquet} (see TRAINING.md section 2)")
    ap.add_argument("--prepacked-dir", type=str, default=None,
                     help="if set, use PrePackedDataset (see src/build_packed_test.py) instead of "
                          "Tier3Dataset -- throughput-test path with no per-step bin-packing search")
    ap.add_argument("--prepacked-local-shards", type=int, default=0,
                    help="debug throughput option: give each DDP rank this many distinct packed files "
                         "instead of randomly reading every file (0 keeps the full-corpus sampler)")
    # architecture — defaults match the ablation winner (8L/512d); override for the real target size
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2560)
    ap.add_argument("--architecture", choices=("classic", "llama"), default="classic",
                    help="Transformer body: classic learned-position/LN/GELU (current baseline), or "
                         "llama RoPE/RMSNorm/SwiGLU. The latter is a structural architecture comparison.")
    ap.add_argument("--llama-qk-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="use per-head RMSNorm on Llama Q/K (on by default); disable only for a "
                         "controlled throughput/quality ablation")
    ap.add_argument("--torch-compile", action="store_true",
                    help="compile the decoder with TorchInductor before DDP. FlashAttention-3 remains a "
                         "custom call boundary while surrounding norm/RoPE/MLP operations can fuse.")
    ap.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                    default="reduce-overhead", help="TorchInductor mode used with --torch-compile")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                     help="autocast dtype for the forward/loss pass (fp32 = no autocast, full precision "
                          "throughout). GH200 has much higher bf16 throughput than fp32, and bf16 needs no "
                          "GradScaler (its exponent range matches fp32's) -- fp16 is included only as a "
                          "throughput/stability comparison point, needs a GradScaler to avoid gradient "
                          "underflow, added below when selected.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--use-cross-attention", action=argparse.BooleanOptionalAction, default=True,
                     help="True (default) = frozen enc-dec + dummy context, matching the actual best "
                          "ablation config found (xattn1_d10_w512_dff2560_lr15e3, val_loss 3.651 — beats "
                          "the decoder-only (no-cross-attn) config documented as the RESULTS.md 'final' winner, "
                          "see TRAINING.md section 1). False = decoder-only (no cross-attention).")
    ap.add_argument("--fast-dummy-cross-attn", action="store_true",
                    help="exact fast path for the frozen length-1 all-zero dummy cross-attention used in "
                         "pretraining; preserves architecture/checkpoints while caching out_proj(v_proj(0))")
    ap.add_argument("--legal-mask-loss", action="store_true",
                     help="mask src/dst/piece/promo logits to structurally-legal classes before computing "
                          "their cross-entropy (see model.py:factored_cross_entropy, src/build_tier3.py's "
                          "legal.bin). Requires --data-dir to contain legal.bin (built by a tier-3 rebuild); "
                          "refused under --smoke-test, where there's no real chess semantics to mask against.")
    ap.add_argument("--use-flash-attn", action=argparse.BooleanOptionalAction, default=True,
                     help="real FlashAttention-3 varlen kernel for self-attention (model.py:build_cu_seqlens, "
                          "kernels-community/flash-attn3) instead of a dense block-causal mask + SDPA -- only "
                          "applies for cuda + bf16/fp16 (flash-attn has no fp32 kernel); off/ineligible falls "
                          "back to the dense-mask path.")
    ap.add_argument("--legal-penalty-weight", type=float, default=0.0,
                     help="alternative/combinable way of penalizing illegal moves (see "
                          "model.py:factored_cross_entropy): instead of (or alongside) hard-masking logits, "
                          "add weight * mean illegal-class softmax probability mass at move positions to the "
                          "loss. 0 = off. Requires the same legality bits as --legal-mask-loss.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                     help="recompute each decoder layer's activations during backward instead of storing them "
                          "for all n_layers -- trades ~1 extra forward pass of compute for O(1)-in-depth "
                          "activation memory, letting batch size grow past whatever it was capped at by FFN "
                          "activation memory (the current ceiling above batch=256, see REPORT.md).")
    # optimization — defaults match xattn1_d10_w512_dff2560_lr15e3 (val_loss 3.651, 47.7M total/37.2M
    # trainable params — see TRAINING.md section 1)
    ap.add_argument("--batch-size", type=int, default=64, help="per-GPU (per-rank) batch size")
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lr-elo", action="store_true",
                    help="reserve the per-move Elo loss-weighting hook. Currently an explicit identity "
                         "map until a scaling law and packed mover-Elo sidecar are approved; does not "
                         "change active training behavior.")
    ap.add_argument("--elo-linear-strength", type=float, default=0.0)
    ap.add_argument("--elo-exponential-alpha", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--fused-adamw", action="store_true",
                    help="use PyTorch's fused AdamW CUDA implementation (default keeps historical AdamW)")
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--warmdown-ratio", type=float, default=0.5)
    ap.add_argument("--lr-schedule-by-positions", action="store_true",
                    help="index warmup/warmdown progress by globally exposed positions divided by "
                         "--max-train-positions; default preserves the historical wall-clock schedule")
    ap.add_argument("--lr-schedule", choices=("linear", "cosine"), default="linear",
                    help="linear = warmup/stable/linear-warmdown; cosine = warmup then cosine decay")
    ap.add_argument("--cosine-min-lr-ratio", type=float, default=0.1,
                    help="final LR divided by peak LR for --lr-schedule cosine")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ddp-stop-sync-interval", type=int, default=10,
                    help="broadcast DDP wall-clock stop state every N steps instead of every step; "
                         "position caps remain checked exactly and normal training collectives keep ranks aligned")
    ap.add_argument("--time-budget", type=float, default=3300.0, help="seconds; wall-clock training budget")
    ap.add_argument("--max-train-positions", type=int, default=0,
                    help="hard cap on globally exposed packed token positions (0 = disabled); use the "
                         "train-row count times seq-len to guarantee at most one corpus pass")
    ap.add_argument("--stop-after-positions", type=int, default=0,
                    help="optional hard exposure stop independent of --max-train-positions; use this "
                         "to make an early checkpoint while retaining the full-run position LR schedule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-seed-offset", type=int, default=0,
                    help="add this to each rank's data-loader seed. Use a distinct value for each "
                         "resumed wall-clock segment so a new process does not replay the same shuffled "
                         "tier-3 stream from its beginning.")
    # data loading
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--buffer-size", type=int, default=1000, help="games held in the packing buffer per rank")
    ap.add_argument("--no-shuffle-months", action="store_true",
                     help="debugging only — see TRAINING.md section 3 on why cross-month shuffling matters")
    ap.add_argument("--no-shuffle-data", action="store_true",
                    help="consume the tier-3 game index in chronological order. The bounded packing buffer "
                         "may still choose among nearby games to fill a fixed-length row efficiently.")
    ap.add_argument("--no-prefetch", action="store_true",
                     help="disable the background-thread batch prefetcher (_PrefetchLoader) that overlaps "
                          "the next step's data load with the current step's GPU compute -- on by default, "
                          "off only for an apples-to-apples throughput comparison against the un-prefetched "
                          "baseline (see REPORT.md).")
    ap.add_argument("--prepacked-without-replacement", action="store_true",
                    help="for the prepacked TRAIN split, shuffle each rank's disjoint row shard and consume "
                         "it once before reshuffling; the default retains historical randint-with-replacement "
                         "sampling. Validation remains on the historical sampler for matched evaluation.")
    ap.add_argument("--prepacked-sequential", action="store_true",
                    help="with --prepacked-without-replacement, consume each rank's packed rows in on-disk "
                         "index order rather than a random permutation. Resume continues after the saved "
                         "global packed-row offset.")
    ap.add_argument("--prepacked-madvise-sequential", action=argparse.BooleanOptionalAction, default=False,
                    help="tell Linux/Lustre that sequential prepacked streams will not be reread soon; "
                         "only applies with --prepacked-sequential; disabled by default because the four-rank "
                         "strided Lustre reader benchmarked faster without it")
    # bookkeeping
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--results-dir", type=str, default=None,
                     help="if set, write <run_name>.json here on completion and append a row to "
                          "<results-dir>/results.tsv (ablation/sweep bookkeeping, mirrors "
                          "external/autoresearch/train_chess.py's per-run JSON)")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/default")
    ap.add_argument("--checkpoint-every", type=float, default=600.0, help="seconds")
    ap.add_argument("--keep-checkpoints", type=int, default=6,
                    help="numbered checkpoints to retain besides latest.pt; six preserves every "
                         "300-second gate of the standard 1,800-second autoresearch run for "
                         "post-hoc matched-token evaluation")
    ap.add_argument("--milestone-checkpoint-percent", type=float, default=0.02,
                    help="write unpruned, optimizer-free evaluation snapshots every fraction of "
                         "--max-train-positions (default: 0.02 = every 2%%; set 0 to disable)")
    ap.add_argument("--milestone-checkpoint-extra-percent", type=float, default=0.05,
                    help="also save exact playing-evaluation milestones at this fraction of the full horizon "
                         "(default: 0.05 = every 5%%)")
    ap.add_argument("--milestone-optimizer-state", action="store_true",
                    help="also save optimizer_state_dict (Adam m/v) inside every milestone snapshot, so a "
                         "future rollback to any milestone can resume with real momenta instead of a cold "
                         "AdamW restart. Costs ~1.3GB extra per milestone at this run's ~166M trainable "
                         "params (roughly doubling milestone size) -- off by default since most milestones "
                         "are only ever used for read-only evaluation, not as a resume point.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-checkpoint", type=str, default=None)
    ap.add_argument("--init-checkpoint-restore-progress", action="store_true",
                    help="with --init-checkpoint, also seed step/total_training_time/"
                         "total_tokens_processed from the init checkpoint's saved metadata (so the "
                         "position-based LR schedule and --prepacked-sequential loader continue from "
                         "where it left off) instead of starting the schedule/loader from zero. Use "
                         "this for a weights-plus-optimizer 'transplant' resume -- e.g. splicing a "
                         "freshly-primed optimizer state onto an evaluation milestone's weights -- "
                         "not for a from-scratch architecture-transfer init.")
    ap.add_argument("--xattn-mode", choices=("keep", "drop", "sattn"), default="keep",
                    help="text-free continuation: keep the dummy cross-attention, drop it (its constant "
                         "output becomes a bias), or replace it with a gated second self-attention "
                         "(model.convert_xattn_state_dict)")
    ap.add_argument("--sattn-init", choices=("xattn", "self", "random"), default="xattn",
                    help="--xattn-mode sattn: initialise the new self-attention from the old cross-attention "
                         "projections, from the layer's own self-attention, or randomly")
    ap.add_argument("--new-lr", type=float, default=0.0,
                    help="peak LR for parameters that do not exist in --init-checkpoint (sattn2, gates); "
                         "0 = same as --lr. Same schedule shape as --lr")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="LoRA adapters of this rank on --lora-targets Linears (src/lora.py); base weights frozen")
    ap.add_argument("--lora-alpha", type=float, default=0.0, help="LoRA alpha (0 = 2 x rank)")
    ap.add_argument("--lora-targets", default="attn,mlp", help="comma list of attn, mlp")
    ap.add_argument("--loss-side", choices=("all", "focal"), default="all",
                    help="focal: loss (train AND val) only on moves marked as the focal player's in the "
                         "packed *_side.u1 sidecar (src/build_games_packed.py --source players)")
    ap.add_argument("--eval-every-frac", type=float, default=0.0,
                    help="evaluate every this fraction of training progress (e.g. 0.1), instead of --eval-every")
    ap.add_argument("--eval-fixed", action="store_true",
                    help="every evaluation uses the same val rows (a fresh seeded val loader each time)")
    ap.add_argument("--eval-every", type=float, default=600.0, help="seconds")
    ap.add_argument("--eval-steps", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=50)
    # special modes
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--lr-sweep", type=str, default=None, help="comma-separated LRs, one assigned per rank")
    ap.add_argument("--sweep-steps", type=int, default=1000)
    return ap.parse_args()


# --------------------------------------------------------------------- #
# Tier-3 streaming dataset (see TRAINING.md section 2-3 for the contract
# and the reasoning behind memmap + cross-month shuffling)
# --------------------------------------------------------------------- #

class Tier3Dataset:
    """Discovers <data_dir>/<month>/{rows.i16,meta.parquet}, memory-maps each
    month's rows.i16 (never loads row data into RAM — only the offset
    index, which is tractable even at tier-3's full scale, see
    TRAINING.md section 2), and exposes a BOS-aligned best-fit packing
    generator per (rank, split) matching prepare_chess.make_dataloader's
    algorithm exactly, just sourcing a game's rows via a memmap slice
    instead of an index into a preloaded array."""

    def __init__(self, data_dir: str, val_fraction: float, seq_len: int, seed: int, shuffle_months: bool = True):
        import pyarrow.parquet as pq

        self.seq_len = seq_len
        months = sorted(
            m for m in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, m)) and os.path.exists(os.path.join(data_dir, m, "_SUCCESS"))
        )
        assert months, (
            f"No converted tier-3 months found under {data_dir} (each needs a _SUCCESS marker from "
            f"src/build_tier3.py). Run that first — see TRAINING.md section 2 for the contract."
        )

        # legal.bin (see src/build_tier3.py) is optional -- older tier-3 builds don't have it.
        # All-or-nothing across months: a dataset that's missing it for some months but not
        # others would silently produce wrong (zero) masks for part of the corpus.
        self.has_legal = os.path.exists(os.path.join(data_dir, months[0], "legal.bin"))
        if self.has_legal:
            missing = [m for m in months if not os.path.exists(os.path.join(data_dir, m, "legal.bin"))]
            assert not missing, f"legal.bin present for some months but missing for {missing[:5]}... — inconsistent tier-3 build"

        self.mmaps = []  # month index -> np.memmap, shape (n_rows_in_month, 5)
        self.legal_mmaps = []  # month index -> np.memmap, structured LEGAL_DTYPE, shape (n_rows_in_month,)
        month_ids: List[np.ndarray] = []
        row_offs: List[np.ndarray] = []
        n_rows_l: List[np.ndarray] = []
        for mi, m in enumerate(months):
            d = os.path.join(data_dir, m)
            mm = np.memmap(os.path.join(d, "rows.i16"), dtype="<i2", mode="r").reshape(-1, 5)
            self.mmaps.append(mm)
            if self.has_legal:
                self.legal_mmaps.append(np.memmap(os.path.join(d, "legal.bin"), dtype=LEGAL_DTYPE, mode="r"))
            t = pq.read_table(os.path.join(d, "meta.parquet"), columns=["row_off", "n_rows"])
            off = t.column("row_off").to_numpy()
            n_rows = t.column("n_rows").to_numpy()
            keep = (n_rows > 0) & (n_rows <= seq_len + 1)  # must fit a training row (T+1 for the shift)
            month_ids.append(np.full(keep.sum(), mi, dtype=np.int32))
            row_offs.append(off[keep])
            n_rows_l.append(n_rows[keep])

        self.month_ids = np.concatenate(month_ids)
        self.row_offs = np.concatenate(row_offs)
        self.n_rows = np.concatenate(n_rows_l)
        n_games = len(self.month_ids)

        rng = random.Random(seed)
        perm = list(range(n_games))
        if shuffle_months:
            # Cross-month shuffle, not per-month — tier 3 spans 2013-2026,
            # and rating/opening-prep distributions drift across 13 years
            # (see TRAINING.md section 3). A sequential-by-month reader
            # would show the model a non-stationary curriculum by accident.
            rng.shuffle(perm)
        n_val = max(1, int(n_games * val_fraction))
        self.val_idx = np.asarray(perm[:n_val], dtype=np.int64)
        self.train_idx = np.asarray(perm[n_val:], dtype=np.int64)
        print(f"Tier3Dataset: {len(months)} months, {n_games:,} usable games "
              f"({len(self.train_idx):,} train / {len(self.val_idx):,} val), "
              f"legal.bin={'present' if self.has_legal else 'absent'}", flush=True)

    def game_rows(self, game_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        mi = int(self.month_ids[game_idx])
        off = int(self.row_offs[game_idx])
        n = int(self.n_rows[game_idx])
        rows = self.mmaps[mi][off:off + n]  # (n, 5) view, no copy
        legal = self.legal_mmaps[mi][off:off + n] if self.has_legal else None  # (n,) structured view, no copy
        return rows, legal

    def shard(self, idx: np.ndarray, rank: int, world_size: int) -> np.ndarray:
        return idx[rank::world_size]

    def make_loader(self, idx: np.ndarray, batch_size: int, seed: int, buffer_size: int = 1000,
                    shuffle: bool = True):
        """Same BOS-aligned best-fit packing algorithm as
        prepare_chess.make_dataloader — see TOKEN_TRAINING.md section 2 and
        TRAINING.md section 3. Yields (inputs, targets, occ, pmask, epoch):
        inputs/targets are (B,T,5) int64; occ/pmask are (B,T) int64 (bit
        patterns, see src/build_tier3.py's legal.bin and
        model.py:legal_component_masks_from_bits) — all-zero (i.e. a no-op
        when unused) if this dataset has no legal.bin. All numpy, converted
        to torch tensors by the caller, so this class has no torch
        dependency and stays testable under --smoke-test without a
        torch/GPU environment."""
        assert len(idx), "empty game-index shard — shard too small for world_size, or corpus too small"
        row_capacity = self.seq_len + 1
        rng = random.Random(seed)
        order = list(idx)
        if shuffle:
            rng.shuffle(order)
        pos_in_order = 0
        epoch = 1

        def next_game() -> Tuple[List[Tuple[int, int, int, int, int]], List[Tuple[int, int]]]:
            nonlocal pos_in_order, epoch, order
            if pos_in_order >= len(order):
                order = list(idx)
                if shuffle:
                    rng.shuffle(order)
                pos_in_order = 0
                epoch += 1
            g = order[pos_in_order]
            pos_in_order += 1
            rows, legal = self.game_rows(g)
            rows_list = rows.tolist()
            if legal is not None:
                legal_list = list(zip(legal["occ"].tolist(), legal["pmask"].tolist()))
            else:
                legal_list = [(0, 0)] * len(rows_list)
            return rows_list, legal_list

        doc_buffer = [next_game() for _ in range(min(buffer_size, len(idx)))]

        while True:
            batch = np.zeros((batch_size, row_capacity, 5), dtype=np.int64)
            # uint64 (not int64) during packing -- own_occ can exceed int64's positive range
            # (e.g. a lone piece on square 63 gives bit pattern 2**63, one more than int64 max);
            # reinterpreted (not converted) to int64 once, after the full array is built (see
            # below) -- torch has no reliably-supported uint64 dtype, but bitwise ops only ever
            # care about the bit pattern, never magnitude, so reinterpretation is exactly right.
            batch_occ = np.zeros((batch_size, row_capacity), dtype=np.uint64)
            batch_pmask = np.zeros((batch_size, row_capacity), dtype=np.uint64)
            for row_idx in range(batch_size):
                p = 0
                while p < row_capacity:
                    if not doc_buffer:
                        doc_buffer.append(next_game())
                    remaining = row_capacity - p
                    best_i, best_len = -1, 0
                    for i, (rows_d, _) in enumerate(doc_buffer):
                        if len(rows_d) <= remaining and len(rows_d) > best_len:
                            best_i, best_len = i, len(rows_d)
                    if best_i >= 0:
                        rows_d, legal_d = doc_buffer.pop(best_i)
                    else:
                        shortest = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i][0]))
                        rows_d, legal_d = doc_buffer.pop(shortest)
                        rows_d, legal_d = rows_d[:remaining], legal_d[:remaining]
                    batch[row_idx, p:p + len(rows_d)] = rows_d
                    occs, pmasks = zip(*legal_d)
                    batch_occ[row_idx, p:p + len(rows_d)] = occs
                    batch_pmask[row_idx, p:p + len(rows_d)] = pmasks
                    p += len(rows_d)
                    if len(doc_buffer) < buffer_size // 2:
                        doc_buffer.append(next_game())
            occ_i64 = batch_occ.view(np.int64)      # bit-exact reinterpretation, not a value cast
            pmask_i64 = batch_pmask.view(np.int64)  # (pmask never exceeds 6 bits, but kept same dtype)
            yield batch[:, :-1], batch[:, 1:], occ_i64[:, 1:], pmask_i64[:, 1:], np.zeros(batch.shape[:2], dtype=np.uint16)[:, 1:], epoch


class _PrefetchLoader:
    """Wraps a batch generator, running its __next__ on a background thread so the
    next step's memmap reads overlap with the current step's GPU compute instead of
    fully serializing with it (see REPORT.md -- data/3p's random reads across a
    1.4TB/162-shard corpus can't stay cached, and the raw generator had zero
    overlap: `next(train_loader)` ran synchronously in the same thread as
    forward/backward, so every cold read stalled the GPU completely instead of
    hiding behind compute time)."""

    def __init__(self, gen, queue_depth: int = 3):
        self._q = queue.Queue(maxsize=queue_depth)
        self._err = None
        self._thread = threading.Thread(target=self._worker, args=(gen,), daemon=True)
        self._thread.start()

    def _worker(self, gen):
        try:
            for item in gen:
                self._q.put(item)
        except Exception as e:  # surfaced to the consumer on next __next__, not swallowed
            self._err = e
            self._q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None and self._err is not None:
            raise self._err
        return item


class PrePackedDataset:
    """Reads ALREADY-packed (n_rows, seq_len+1, 5) rows -- no per-step
    bin-packing search, batches are pure numpy fancy-indexing (the fix
    documented in REPORT.md).

    Two on-disk layouts supported:
      1. Single-pool throughput probe (src/build_packed_test.py): one
         `<data_dir>/packed.i16`, no legal.bin -- occ/pmask stay all-zero
         (no-op), split 99/1 train/val arbitrarily since the pool is
         synthetic anyway.
      2. Real packed corpus (src/build_tier3_packed.py): `<data_dir>/{train,val}/
         packed_NNNNN.i16` shards, one file set per worker, plus optional
         sibling `packed_NNNNN_occ.bin` / `packed_NNNNN_pmask.bin` (uint64 /
         uint8, same row layout) carrying the legal-move bitmask through to
         the packed format -- this is what makes --legal-mask-loss usable
         together with --prepacked-dir; before this, PrePackedDataset always
         returned zeroed occ/pmask regardless of what build_tier3_packed.py
         had written to disk.
    """

    def __init__(self, data_dir: str, seq_len: int, loss_side: str = "all"):
        self.seq_len = seq_len
        self.row_capacity = seq_len + 1
        self.loss_side = loss_side
        if os.path.exists(os.path.join(data_dir, "packed.i16")):
            self._legacy_single_pool(data_dir)
        else:
            self._multi_shard(data_dir)

    def _legacy_single_pool(self, data_dir: str) -> None:
        row_capacity = self.row_capacity
        mm = np.memmap(os.path.join(data_dir, "packed.i16"), dtype="<i2", mode="r")
        n_rows = mm.size // (row_capacity * 5)
        packed = mm.reshape(n_rows, row_capacity, 5)
        n_train = int(n_rows * 0.99)
        self._splits = {
            "train": {"mmaps": [packed[:n_train]], "occ": [], "pmask": [],
                      "boundaries": np.array([0, n_train]), "n_rows": n_train, "has_legal": False},
            "val": {"mmaps": [packed[n_train:]], "occ": [], "pmask": [],
                    "boundaries": np.array([0, n_rows - n_train]), "n_rows": n_rows - n_train, "has_legal": False},
        }
        self.train_idx = {"split": "train", "idx": np.arange(n_train)}
        self.val_idx = {"split": "val", "idx": np.arange(n_rows - n_train)}
        print(f"PrePackedDataset (legacy single-pool): {n_rows:,} packed rows "
              f"({n_train:,} train / {n_rows - n_train:,} val)", flush=True)

    def _multi_shard(self, data_dir: str) -> None:
        row_capacity = self.row_capacity
        self._splits = {}
        for split in ("train", "val"):
            split_dir = os.path.join(data_dir, split)
            assert os.path.exists(os.path.join(split_dir, "_SUCCESS")), \
                f"no _SUCCESS marker in {split_dir} -- run src/build_tier3_packed.py first"
            shard_paths = sorted(glob.glob(os.path.join(split_dir, "packed_*.i16")))
            assert shard_paths, f"no packed_*.i16 shards under {split_dir}"
            has_legal = os.path.exists(shard_paths[0][:-4] + "_occ.bin")
            has_elo = os.path.exists(shard_paths[0][:-4] + "_elo.u16")
            has_side = os.path.exists(shard_paths[0][:-4] + "_side.u1")
            if self.loss_side == "focal":
                assert has_side, f"--loss-side focal needs *_side.u1 sidecars in {split_dir}"
            sides = []
            mmaps, occs, pmasks, elos, counts = [], [], [], [], []
            for p in shard_paths:
                mm = np.memmap(p, dtype="<i2", mode="r").reshape(-1, row_capacity, 5)
                mmaps.append(mm)
                counts.append(mm.shape[0])
                if has_legal:
                    base = p[:-4]
                    occs.append(np.memmap(base + "_occ.bin", dtype="<u8", mode="r").reshape(-1, row_capacity))
                    pmasks.append(np.memmap(base + "_pmask.bin", dtype="u1", mode="r").reshape(-1, row_capacity))
                if has_elo:
                    elos.append(np.memmap(base + "_elo.u16", dtype="<u2", mode="r").reshape(-1, row_capacity))
                if has_side:
                    sides.append(np.memmap(p[:-4] + "_side.u1", dtype="u1", mode="r").reshape(-1, row_capacity))
            boundaries = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=boundaries[1:])
            self._splits[split] = {"mmaps": mmaps, "occ": occs, "pmask": pmasks, "elo": elos, "side": sides,
                                    "boundaries": boundaries, "n_rows": int(boundaries[-1]), "has_legal": has_legal, "has_elo": has_elo}
        self.train_idx = {"split": "train", "idx": np.arange(self._splits["train"]["n_rows"])}
        self.val_idx = {"split": "val", "idx": np.arange(self._splits["val"]["n_rows"])}
        print(f"PrePackedDataset (multi-shard): {self._splits['train']['n_rows']:,} train / "
              f"{self._splits['val']['n_rows']:,} val packed rows, "
              f"legal={'present' if self._splits['train']['has_legal'] else 'absent'}", flush=True)

    def shard(self, tagged_idx: dict, rank: int, world_size: int) -> dict:
        return {"split": tagged_idx["split"], "idx": tagged_idx["idx"][rank::world_size]}

    def local_shard_indices(self, split: str, rank: int, world_size: int, n_shards: int) -> np.ndarray:
        """Rows from a small, disjoint per-rank shard set for an I/O-locality A/B."""
        data = self._splits[split]
        total = len(data["mmaps"])
        if n_shards <= 0 or n_shards * world_size > total:
            raise ValueError(f"need 1..{total // world_size} local shards per rank, got {n_shards}")
        shard_ids = rank + np.arange(n_shards, dtype=np.int64) * world_size
        return np.concatenate([np.arange(data["boundaries"][s], data["boundaries"][s + 1], dtype=np.int64)
                               for s in shard_ids])

    def madvise_sequential(self, split: str) -> int:
        """Apply a non-semantic OS streaming hint to a packed split's mmaps.

        On the 1.4TB Lustre corpus, retaining already-consumed pages can evict
        useful future pages.  MADV_SEQUENTIAL asks the kernel to use read-ahead
        and release old pages more aggressively.  It changes neither samples
        nor their order and is deliberately opt-in only for sequential runs.
        """
        import mmap
        advised = 0
        data = self._splits[split]
        for arr in (*data["mmaps"], *data["occ"], *data["pmask"]):
            try:
                arr._mmap.madvise(mmap.MADV_SEQUENTIAL)
                advised += 1
            except (AttributeError, OSError):
                pass
        return advised

    def make_loader(self, tagged_idx: dict, batch_size: int, seed: int, buffer_size: int = 0,
                    without_replacement: bool = False, sequential: bool = False,
                    start_global_row: int = 0, rank: int = 0, world_size: int = 1):
        data = self._splits[tagged_idx["split"]]
        idx = tagged_idx["idx"]
        assert len(idx), "empty shard"
        boundaries = data["boundaries"]
        row_capacity = self.row_capacity
        rng = np.random.default_rng(seed)
        epoch = 1
        if sequential and not without_replacement:
            raise ValueError("--prepacked-sequential requires --prepacked-without-replacement")
        order = (np.arange(len(idx), dtype=np.int64) if sequential else
                 (rng.permutation(len(idx)) if without_replacement else None))
        # ``idx`` is DDP-strided (rank::world_size). Convert the global count
        # already exposed by a prior process into this rank's local cursor.
        # This applies to both on-disk sequential order and a seeded random
        # permutation: regenerating the latter from the same seed and seeking
        # to this cursor resumes one exact no-replacement corpus pass instead
        # of replaying its initial rows after every wall-clock segment.
        pos = (max(0, start_global_row + world_size - 1 - rank) // world_size
               if without_replacement else 0)
        while True:
            if without_replacement:
                # Rank shards are disjoint (see shard()), so consuming a rank-local permutation means
                # every global packed row is exposed at most once per pass. All ranks have shard sizes
                # differing by at most one and therefore the same ceil(n / batch_size) step count; their
                # final batches may differ by one row, which DDP supports without repeating/padding data.
                if pos >= len(order):
                    epoch += 1
                    order = np.arange(len(idx), dtype=np.int64) if sequential else rng.permutation(len(idx))
                    pos = 0
                take = order[pos:min(pos + batch_size, len(order))]
                pos += len(take)
                global_idx = idx[take]
            else:
                global_idx = idx[rng.integers(0, len(idx), size=batch_size)]
            shard_idx = np.searchsorted(boundaries, global_idx, side="right") - 1
            local_idx = global_idx - boundaries[shard_idx]
            n_batch = len(global_idx)
            batch = np.empty((n_batch, row_capacity, 5), dtype=np.int64)
            occ = np.zeros((n_batch, row_capacity), dtype=np.uint64)
            pmask = np.zeros((n_batch, row_capacity), dtype=np.uint8)
            elo = np.zeros((n_batch, row_capacity), dtype=np.uint16)
            not_focal = np.zeros((n_batch, row_capacity - 1), dtype=bool) if self.loss_side == "focal" else None
            for s in np.unique(shard_idx):
                sel = shard_idx == s
                locs = local_idx[sel]
                batch[sel] = data["mmaps"][s][locs]
                if data["has_legal"]:
                    occ[sel] = data["occ"][s][locs]
                    pmask[sel] = data["pmask"][s][locs]
                if data.get("has_elo", False): elo[sel] = data["elo"][s][locs]
                if self.loss_side == "focal":
                    not_focal[sel] = (data["side"][s][locs][:, 1:] & 4) == 0
            occ_i64 = occ.view(np.int64)       # bit-exact reinterpretation, see Tier3Dataset.make_loader
            pmask_i64 = pmask.astype(np.int64)  # pmask never exceeds 6 bits, plain widen is fine
            targets = batch[:, 1:]
            if not_focal is not None:
                # Targets that are not the focal player's moves become PAD rows (every factored head
                # ignores KIND_PAD targets). A copy: inputs batch[:, :-1] share this memory and must
                # keep the opponent's moves as context.
                targets = targets.copy()
                targets[not_focal] = 0
            yield batch[:, :-1], targets, occ_i64[:, 1:], pmask_i64[:, 1:], elo[:, 1:], epoch


class SyntheticDataset:
    """--smoke-test stand-in for Tier3Dataset: random-but-well-formed games
    (BOS + N legal-shaped rows + EOS, values in-range but NOT legally
    consistent chess — this only tests the training loop's plumbing, not
    move legality) so the whole script can be exercised without tier-3 data
    or a GPU."""

    def __init__(self, seq_len: int, seed: int, n_games: int = 2000):
        rng = random.Random(seed)
        self.games = []
        for _ in range(n_games):
            n_moves = rng.randint(5, min(80, seq_len - 2))
            rows = [ROW_BOS]
            for _ in range(n_moves):
                rows.append((KIND_MOVE, rng.randrange(64), rng.randrange(64), rng.randrange(6), rng.randrange(5)))
            rows.append(ROW_EOS)
            self.games.append(rows)
        self.seq_len = seq_len

    def make_loader(self, batch_size: int, seed: int):
        # Cycles through self.games without replacement per pass (not rng.choice-with-replacement,
        # which would never complete a "pass" and so could never increment epoch) -- this dataset's
        # whole point under --smoke-test is exercising the same epoch-tracking mechanism as
        # Tier3Dataset, on a small enough corpus that a run actually completes several epochs
        # within the smoke test's short time budget (see run_training's DONE summary).
        rng = random.Random(seed)
        row_capacity = self.seq_len + 1
        order = list(range(len(self.games)))
        rng.shuffle(order)
        pos = 0
        epoch = 1

        def next_game():
            nonlocal pos, epoch, order
            if pos >= len(order):
                order = list(range(len(self.games)))
                rng.shuffle(order)
                pos = 0
                epoch += 1
            g = self.games[order[pos]]
            pos += 1
            return g

        while True:
            batch = np.zeros((batch_size, row_capacity, 5), dtype=np.int64)
            for row_idx in range(batch_size):
                p = 0
                while p < row_capacity:
                    doc = next_game()[:row_capacity - p]
                    batch[row_idx, p:p + len(doc)] = doc
                    p += len(doc)
            # No legal.bin equivalent for synthetic (non-chess) data -- zero occ/pmask, same
            # no-op shape Tier3Dataset yields when has_legal is False. main() refuses to combine
            # --smoke-test with --legal-mask-loss (see below) since masking against all-zero
            # occupancy would mask out every class, not "no restriction".
            zeros = np.zeros((batch_size, row_capacity), dtype=np.int64)
            yield batch[:, :-1], batch[:, 1:], zeros[:, 1:], zeros[:, 1:], zeros[:, 1:], epoch


# --------------------------------------------------------------------- #
# LR schedule — identical shape to external/autoresearch/train_chess.py,
# see TOKEN_TRAINING.md section 4
# --------------------------------------------------------------------- #

def lr_at(progress: float, base_lr: float, warmup_ratio: float, warmdown_ratio: float,
          schedule: str = "linear", cosine_min_lr_ratio: float = 0.1) -> float:
    if warmup_ratio > 0 and progress < warmup_ratio:
        return base_lr * (progress / warmup_ratio)
    if schedule == "cosine":
        if not 0.0 <= cosine_min_lr_ratio <= 1.0:
            raise ValueError("--cosine-min-lr-ratio must be in [0, 1]")
        decay_progress = (progress - warmup_ratio) / max(1.0 - warmup_ratio, 1e-12)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, decay_progress))))
        return base_lr * (cosine_min_lr_ratio + (1.0 - cosine_min_lr_ratio) * cosine)
    warmdown_start = 1.0 - warmdown_ratio
    if progress > warmdown_start:
        decay_frac = (progress - warmdown_start) / warmdown_ratio
        return base_lr * max(0.0, 1.0 - decay_frac)
    return base_lr


# --------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------- #

def save_checkpoint(path: str, model, optimizer, step: int, total_training_time: float,
                    total_tokens_processed: int, cfg: dict):
    import torch
    raw_model = model.module if hasattr(model, "module") else model
    # Compiled modules wrap the original in ``_orig_mod``.  Persisting that
    # wrapper would prefix every state key and make normal/compiled resumes
    # incompatible despite identical parameters.
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "total_training_time": total_training_time,
        "total_tokens_processed": total_tokens_processed,
        "cfg": cfg,
    }, path + ".tmp")
    os.replace(path + ".tmp", path)  # atomic on the same filesystem — never leaves a half-written latest.pt


def save_evaluation_snapshot(path: str, model, step: int, total_training_time: float,
                             total_tokens_processed: int, cfg: dict, optimizer=None):
    """Compact immutable checkpoint for benchmarks/UI. Optimizer state (Adam m/v)
    is omitted by default -- pass --milestone-optimizer-state to include it so
    this milestone can also serve as a real resume point, not just a read-only
    evaluation snapshot."""
    import torch
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    payload = {
        "model_state_dict": raw_model.state_dict(),
        "step": step,
        "total_training_time": total_training_time,
        "total_tokens_processed": total_tokens_processed,
        "cfg": cfg,
        "checkpoint_kind": "evaluation_milestone",
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_checkpoint(path: str, model, optimizer, device: str) -> Tuple[int, float, int]:
    import torch
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model = getattr(raw_model, "_orig_mod", raw_model)
    raw_model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    # Checkpoints created before position-based deployment support have no
    # token counter. They remain loadable, but cannot preserve a token LR clock.
    return ckpt["step"], ckpt["total_training_time"], ckpt.get("total_tokens_processed", 0)


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def build_model(args, device):
    import torch
    from model import DecoderConfig, FactoredChessDecoder, freeze_encoder_and_cross_attention

    cfg = DecoderConfig(
        vocab_size=1, d_model=args.d_model, n_heads=args.n_heads, n_layers=args.depth,
        d_ff=args.d_ff, max_seq_len=args.seq_len, dropout=args.dropout,
        use_cross_attention=args.use_cross_attention, grad_checkpoint=args.grad_checkpoint,
        fast_dummy_cross_attention=args.fast_dummy_cross_attn,
        architecture=args.architecture,
        llama_qk_norm=args.llama_qk_norm,
        xattn_mode=getattr(args, "xattn_mode", "keep"),
    )
    if cfg.xattn_mode != "keep" and (args.use_cross_attention or args.architecture != "llama"):
        raise ValueError("--xattn-mode drop/sattn needs --architecture llama --no-use-cross-attention")
    if args.fast_dummy_cross_attn and (not args.use_cross_attention or args.dropout != 0.0):
        raise ValueError("--fast-dummy-cross-attn requires --use-cross-attention and --dropout 0")
    model = FactoredChessDecoder(cfg).to(device)
    if args.use_cross_attention:
        # "frozen enc-dec + dummy context" — no real encoder exists yet (no NL conditioning in this
        # phase, see ABLATIONS.md), so only cross-attn's weights need freezing, not an encoder module.
        freeze_encoder_and_cross_attention(model, encoder=None)
    if getattr(args, "lora_rank", 0):
        # Before compile/DDP and before the init checkpoint loads (LoRA keys are simply absent there).
        from lora import apply_lora
        apply_lora(model, args.lora_rank, args.lora_alpha or 2.0 * args.lora_rank, args.lora_targets)
    if args.torch_compile:
        # Keep graph breaks allowed: FA3's varlen extension is intentionally a
        # custom boundary, while Inductor still fuses the eager bookkeeping
        # surrounding it.  Batch and sequence shapes are fixed in deployment;
        # keeping Dynamo static also avoids a PyTorch 2.8 DDP dynamic-shape
        # partitioner bug triggered by symbolic attention dimensions.
        model = torch.compile(model, mode=args.compile_mode, dynamic=False)
    return model, cfg


def run_training(args, rank: int, world_size: int, local_rank: int, lr_override: Optional[float] = None,
                  fixed_steps: Optional[int] = None):
    import torch
    import torch.distributed as dist
    from model import factored_cross_entropy, make_dummy_context, build_cu_seqlens

    is_smoke = args.smoke_test
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed + rank)

    model, cfg = build_model(args, device)
    # DDP must be constructed after freeze_encoder_and_cross_attention (in build_model) sets
    # requires_grad=False on frozen params, so its reducer buckets never include them.
    if world_size > 1 and not is_smoke:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lr = lr_override if lr_override is not None else args.lr
    run_name = args.run_name or f"d{args.depth}_w{args.d_model}_lr{lr:g}"
    if rank == 0:
        print(f"[{run_name}] params: {n_trainable:,} trainable / {n_total:,} total "
              f"device={device} world_size={world_size}", flush=True)
        if args.lr_elo:
            print(f"[{run_name}] --lr-elo enabled: identity hook only; packed data has no mover-Elo "
                  f"sidecar until a concrete scaling law is selected", flush=True)

    # Parameters absent from the pretrained checkpoint (the gated second self-attention) get their
    # own peak LR (--new-lr): the best-SFT recipe trains new conditioning modules much faster than
    # the pretrained decoder. Both groups follow the same schedule shape via lr_scale.
    new_names = ("sattn2.", "xgate") if args.xattn_mode == "sattn" else ()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    base_params = [p for n, p in named if not any(k in n for k in new_names)]
    new_params = [p for n, p in named if any(k in n for k in new_names)]
    new_scale = (args.new_lr / lr) if (args.new_lr and new_params) else 1.0
    groups = [{"params": base_params, "lr_scale": 1.0}]
    if new_params:
        groups.append({"params": new_params, "lr_scale": new_scale})
    optimizer = torch.optim.AdamW(
        groups, lr=lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_adamw,
    )
    if rank == 0 and new_params:
        print(f"[{run_name}] new-param group: {sum(p.numel() for p in new_params):,} params at "
              f"lr x{new_scale:g}", flush=True)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        from model import convert_xattn_state_dict
        init_sd = convert_xattn_state_dict(init["model_state_dict"], args.xattn_mode, args.depth,
                                           args.sattn_init, args.n_heads)
        if args.sattn_init == "random" and any(".cross_attn." in k for k in init["model_state_dict"]):
            # keep the fresh sattn2 init: drop the copied projections (only when converting now;
            # an already-converted checkpoint's sattn2 is trained and must load)
            init_sd = {k: v for k, v in init_sd.items() if ".sattn2." not in k or "_norm." in k}
        missing, unexpected = raw_model.load_state_dict(
            init_sd, strict=args.sattn_init != "random" and not args.lora_rank)
        missing = [k for k in missing if "lora_" not in k]  # LoRA adapters are new by design
        if rank == 0 and (missing or unexpected):
            print(f"[{run_name}] init load: missing={missing} unexpected={unexpected}", flush=True)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    amp_enabled = device.startswith("cuda") and args.dtype != "fp32"
    # fp16's narrow exponent range (unlike bf16, which matches fp32's) can underflow small
    # gradients to zero -- GradScaler rescales the loss before backward to keep them representable.
    # No-op (scale stays 1, never skips a step) when dtype != fp16.
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16" and device.startswith("cuda")))
    enc_context_fn = (
        (lambda b: make_dummy_context(b, args.d_model, device=device)) if args.use_cross_attention else None
    )
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    resume_positions_for_loader = 0
    init_progress = None
    if args.init_checkpoint and args.init_checkpoint_restore_progress:
        init_progress = (int(init["step"]), float(init["total_training_time"]),
                         int(init["total_tokens_processed"]))
        resume_positions_for_loader = init_progress[2]
    if args.resume and os.path.exists(ckpt_path):
        # The actual model/optimizer restore happens below.  Read just enough
        # metadata now to construct a sequential packed loader at its true
        # continuation point instead of replaying its first rows.
        resume_positions_for_loader = int(torch.load(
            ckpt_path, map_location="cpu", weights_only=False,
        ).get("total_tokens_processed", 0))

    if is_smoke:
        ds = SyntheticDataset(args.seq_len, args.seed)
        train_loader = ds.make_loader(args.batch_size, args.seed + rank)
        val_loader = ds.make_loader(args.batch_size, args.seed + 999 + rank)
    elif args.prepacked_dir:
        ds = PrePackedDataset(args.prepacked_dir, args.seq_len, args.loss_side)
        assert ds._splits["train"]["has_legal"] or not (args.legal_mask_loss or args.legal_penalty_weight > 0), (
            f"--legal-mask-loss/--legal-penalty-weight given but {args.prepacked_dir} has no packed "
            f"occ/pmask sidecar files (built by an updated src/build_tier3_packed.py) -- masking/penalizing "
            f"against all-zero occupancy would silently zero out every class rather than 'no restriction'."
        )
        if args.lr_elo:
            assert ds._splits["train"].get("has_elo", False), "--lr-elo requires packed *_elo.u16 sidecars"
        train_shard = ({"split": "train", "idx": ds.local_shard_indices("train", rank, world_size,
                                                                    args.prepacked_local_shards)}
                       if args.prepacked_local_shards else ds.shard(ds.train_idx, rank, world_size))
        if args.prepacked_sequential and args.prepacked_madvise_sequential:
            advised = ds.madvise_sequential("train")
            if rank == 0:
                print(f"PrePackedDataset: MADV_SEQUENTIAL applied to {advised} train mappings", flush=True)
        val_shard = ds.shard(ds.val_idx, rank, world_size) if len(ds.val_idx["idx"]) >= world_size else ds.val_idx
        train_loader = ds.make_loader(train_shard, args.batch_size, args.seed + rank + args.data_seed_offset,
                                      without_replacement=args.prepacked_without_replacement,
                                      sequential=args.prepacked_sequential,
                                      start_global_row=resume_positions_for_loader // args.seq_len,
                                      rank=rank, world_size=world_size)
        val_loader = ds.make_loader(val_shard, args.batch_size, args.seed + 999 + rank)
        if args.eval_fixed:
            # Same val rows at every evaluation (and in every run with the same seed/world size).
            val_loader_fn = lambda: ds.make_loader(val_shard, args.batch_size, args.seed + 999 + rank)
    else:
        ds = Tier3Dataset(args.data_dir, args.val_fraction, args.seq_len, args.seed,
                           shuffle_months=not args.no_shuffle_months)
        assert ds.has_legal or not (args.legal_mask_loss or args.legal_penalty_weight > 0), (
            f"--legal-mask-loss/--legal-penalty-weight given but {args.data_dir} has no legal.bin (built by "
            f"an updated src/build_tier3.py) -- masking/penalizing against all-zero occupancy would silently "
            f"zero out every class rather than 'no restriction'."
        )
        train_shard = ds.shard(ds.train_idx, rank, world_size)
        val_shard = ds.shard(ds.val_idx, rank, world_size) if len(ds.val_idx) >= world_size else ds.val_idx
        train_loader = ds.make_loader(train_shard, args.batch_size, args.seed + rank + args.data_seed_offset,
                                      args.buffer_size)
        val_loader = ds.make_loader(val_shard, args.batch_size, args.seed + 999 + rank,
                                     min(args.buffer_size, len(val_shard)))

    if not args.eval_fixed:
        val_loader_fn = None
    if not is_smoke and not args.no_prefetch:
        train_loader = _PrefetchLoader(train_loader)
        val_loader = _PrefetchLoader(val_loader)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    step = 0
    total_training_time = 0.0
    total_tokens_processed = 0
    if args.resume and os.path.exists(ckpt_path):
        step, total_training_time, total_tokens_processed = load_checkpoint(ckpt_path, model, optimizer, device)
        if rank == 0:
            print(f"[{run_name}] resumed from {ckpt_path}: step={step} elapsed={total_training_time:.0f}s "
                  f"positions={total_tokens_processed:,}", flush=True)
    elif init_progress is not None:
        step, total_training_time, total_tokens_processed = init_progress
        if rank == 0:
            print(f"[{run_name}] init-checkpoint progress restored from {args.init_checkpoint}: "
                  f"step={step} elapsed={total_training_time:.0f}s positions={total_tokens_processed:,} "
                  f"(fresh optimizer state)", flush=True)
    # Keep process-local reporting separate from cumulative checkpoint state.
    # On a resumed segment, dividing all historical positions by this
    # process's wall time inflates throughput/MFU by the number of segments.
    segment_start_tokens = total_tokens_processed

    if any(x < 0 or x >= 1 for x in (args.milestone_checkpoint_percent,
                                      args.milestone_checkpoint_extra_percent)):
        raise ValueError("milestone checkpoint percentages must be in [0, 1)")
    milestone_intervals = sorted({int(args.max_train_positions * x)
                                  for x in (args.milestone_checkpoint_percent,
                                            args.milestone_checkpoint_extra_percent)
                                  if args.max_train_positions > 0 and x > 0})
    position_stop = args.stop_after_positions or args.max_train_positions
    milestone_dir = os.path.join(args.checkpoint_dir, "milestones")
    next_milestones = {interval: total_tokens_processed // interval + 1 for interval in milestone_intervals}
    if rank == 0 and milestone_intervals:
        os.makedirs(milestone_dir, exist_ok=True)

    last_ckpt_t = time.time()
    last_eval_t = time.time()
    running_loss = torch.zeros((), device=device)
    # A resumed allocation may begin between two log boundaries.  Track the
    # actual number of accumulated steps so its first partial window is not
    # incorrectly divided by --log-every.
    loss_window_steps = 0
    tokens_this_window = 0
    window_t0 = time.time()
    run_wall_clock_t0 = time.time()  # real wall time for this call, unlike total_training_time
    # (which excludes the first-10-step warmup and stops advancing once the budget is hit) --
    # used only for the DONE summary's tokens/sec and full-corpus-epoch-time estimate below.
    epoch = 1
    sync_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # Persist the fixed-time learning curve, not only the terminal score.  The
    # autoresearch decision is chiefly matched-token efficiency, so a final
    # scalar cannot recover the crucial early/mid-training comparison when a
    # run's logs or numbered checkpoints are later pruned.
    eval_history = []
    next_eval_frac = args.eval_every_frac if args.eval_every_frac > 0 else float("inf")

    # A resumed run already has a nonzero budget clock. Reconstruct its origin
    # before entering the loop; otherwise `step > 10` tries to subtract None
    # on the very first resumed optimizer step.
    budget_clock_t0 = (time.time() - total_training_time) if total_training_time > 0 else None
    while True:
        x_np, y_np, occ_np, pmask_np, elo_np, epoch = next(train_loader)
        if position_stop > 0:
            # Normal DDP batches have the configured, equal local size, so the
            # global exposure is known on CPU.  Avoid an all-reduce + .item()
            # synchronization on every single step; only the possible final
            # short batch needs an exact collective count.
            local_upcoming = x_np.shape[0] * x_np.shape[1]
            upcoming = local_upcoming * world_size
            if total_tokens_processed + upcoming > position_stop:
                exact_upcoming = torch.tensor(local_upcoming, dtype=torch.int64, device=device)
                if world_size > 1:
                    dist.all_reduce(exact_upcoming, op=dist.ReduceOp.SUM)
                upcoming = int(exact_upcoming.item())
            if total_tokens_processed + upcoming > position_stop:
                break
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        y = torch.from_numpy(y_np).to(device, non_blocking=True)
        move_weights = None
        if args.lr_elo:
            elo = torch.from_numpy(elo_np).to(device, non_blocking=True).float()
            clipped = elo.clamp(1800, 2800)
            if args.elo_linear_strength:
                raw = 1.0 + args.elo_linear_strength * (clipped - 2023.4421575515967) / 165.99576460568153
            elif args.elo_exponential_alpha:
                raw = 10.0 ** (args.elo_exponential_alpha * (clipped - 2000.0) / 400.0)
            else:
                raise ValueError("--lr-elo needs --elo-linear-strength or --elo-exponential-alpha")
            is_move = y[..., 0].eq(KIND_MOVE)
            move_weights = raw / raw[is_move].mean().clamp(min=1e-8)

        if args.lr_schedule_by_positions:
            progress = min(1.0, total_tokens_processed / args.max_train_positions)
        else:
            progress = min(1.0, total_training_time / args.time_budget) if args.time_budget > 0 else 0.0
        cur_lr = lr_at(progress, lr, args.warmup_ratio, args.warmdown_ratio,
                       args.lr_schedule, args.cosine_min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr * g.get("lr_scale", 1.0)

        enc = enc_context_fn(x.shape[0]) if enc_context_fn else None
        # Real flash-attn (varlen, see model.py:build_cu_seqlens) needs bf16/fp16 + cuda -- no fp32
        # kernel exists. Falls back to the dense block-causal mask + SDPA otherwise (fp32, or
        # --no-use-flash-attn for an explicit throughput A/B).
        use_flash = args.use_flash_attn and amp_enabled and args.dtype in ("bf16", "fp16")
        fwd_kwargs = (dict(zip(("cu_seqlens", "max_seqlen"), build_cu_seqlens(x))) if use_flash
                      else dict(mask=build_block_causal_mask(x)))
        # --dtype selects the autocast precision (see CLI help above): bf16 (default) needs no
        # GradScaler since its exponent range matches fp32's; fp32 disables autocast entirely
        # (amp_enabled=False); fp16 needs the scaler below to avoid gradient underflow. Master
        # weights/optimizer state stay fp32 regardless (autocast only wraps the forward-pass ops).
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x, enc, **fwd_kwargs)
            if args.legal_mask_loss or args.legal_penalty_weight > 0:
                occ = torch.from_numpy(occ_np).to(device, non_blocking=True)
                pmask = torch.from_numpy(pmask_np).to(device, non_blocking=True)
                loss = factored_cross_entropy(logits, y, own_occ=occ, piece_mask_bits=pmask,
                                               hard_mask=args.legal_mask_loss,
                                               illegal_penalty_weight=args.legal_penalty_weight,
                                               move_weights=move_weights)
            else:
                loss = factored_cross_entropy(logits, y, move_weights=move_weights)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Do not synchronize on loss.item() and cuda.synchronize() every step:
        # it serializes CPU submission, DDP, and the background data prefetcher.
        # Accumulate the detached scalar on GPU and materialize it at logging
        # cadence, where timing/logging necessarily needs a coherent value.
        if rank == 0:
            running_loss.add_(loss.detach())
            loss_window_steps += 1
        if step == 10:
            budget_clock_t0 = time.time()
        elif step > 10:
            total_training_time = time.time() - budget_clock_t0

        step_tokens = (upcoming if args.max_train_positions > 0 else
                       x.shape[0] * x.shape[1] * world_size)  # global sequence positions/step ("tokens"
        # in this project's usage means positions, not the 5 (kind,src,dst,piece,promo) columns each holds)
        tokens_this_window += step_tokens
        total_tokens_processed += step_tokens

        # These are intentionally separate from resume files: checkpoint pruning
        # must never erase a fixed-token evaluation point.  Adam state is omitted
        # by default (benchmarks only need model/config, and it keeps 50
        # two-percent snapshots practical on disk over a full corpus pass) --
        # pass --milestone-optimizer-state to include it when a milestone should
        # also be a viable resume point.
        due_milestones = [(interval, next_index) for interval, next_index in next_milestones.items()
                          if total_tokens_processed >= next_index * interval]
        for interval, _ in due_milestones:
            next_milestones[interval] += 1
        if rank == 0:
            # 10%, 20%, ... occur in both cadences; save only one copy.
            for target_positions in sorted({interval * next_index for interval, next_index in due_milestones}):
                pct = 100.0 * target_positions / args.max_train_positions
                milestone_path = os.path.join(milestone_dir, f"pct{pct:05.2f}_step{step}.pt")
                save_evaluation_snapshot(milestone_path, model, step, total_training_time,
                                         total_tokens_processed, vars(args),
                                         optimizer=optimizer if args.milestone_optimizer_state else None)
                print(f"[{run_name}] immutable evaluation milestone: {pct:.2f}% -> {milestone_path}", flush=True)

        if rank == 0 and step % args.log_every == 0:
            avg_loss = running_loss.item() / max(loss_window_steps, 1)
            running_loss.zero_()
            loss_window_steps = 0
            # Fast-fail at the same logging cadence.  DDP makes rank 0's
            # gradient path representative; this avoids a forced GPU sync on
            # every step solely for a diagnostic that has never triggered in a
            # stable run.
            if step > 10 and (math.isnan(avg_loss) or math.isinf(avg_loss) or avg_loss > 100):
                print(f"[{run_name}] DIVERGED at step={step}: avg_loss={avg_loss} — aborting", flush=True)
                sys.exit(1)
            elapsed_window = time.time() - window_t0
            toks_per_sec = tokens_this_window / max(elapsed_window, 1e-6)
            tokens_this_window = 0
            window_t0 = time.time()
            print(f"[{run_name}] step={step} epoch={epoch} loss={avg_loss:.4f} lr={cur_lr:.2e} "
                  f"tok/s={toks_per_sec:.0f} elapsed={total_training_time:.0f}s/{args.time_budget:.0f}s",
                  flush=True)

        if fixed_steps is None:
            now = time.time()
            if rank == 0 and now - last_ckpt_t >= args.checkpoint_every:
                save_checkpoint(ckpt_path, model, optimizer, step, total_training_time,
                                total_tokens_processed, vars(args))
                numbered = os.path.join(args.checkpoint_dir, f"step{step}.pt")
                save_checkpoint(numbered, model, optimizer, step, total_training_time,
                                total_tokens_processed, vars(args))
                _prune_old_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
                print(f"[{run_name}] checkpoint saved: step={step} -> {ckpt_path}", flush=True)
                last_ckpt_t = now
            if args.eval_every_frac > 0:
                # Progress is identical on every rank (same token count / broadcast clock), but the
                # time-based clock can drift by a step, so rank 0 still decides and broadcasts.
                due = progress >= next_eval_frac - 1e-9
            else:
                due = now - last_eval_t >= args.eval_every
            if world_size > 1:
                if rank == 0:
                    sync_flag.fill_(int(due))
                dist.broadcast(sync_flag, src=0)
                do_eval = bool(sync_flag.item())
            else:
                do_eval = due
            if do_eval and args.eval_every_frac > 0:
                while next_eval_frac <= progress + 1e-9:
                    next_eval_frac += args.eval_every_frac
            if do_eval:
                val_loss = evaluate(model, val_loader_fn() if val_loader_fn else val_loader, args.eval_steps,
                                    device, args.n_heads, enc_context_fn, args.dtype, args.use_flash_attn)
                if rank == 0:
                    eval_wall = time.time() - run_wall_clock_t0
                    eval_toks_per_sec = (total_tokens_processed - segment_start_tokens) / max(eval_wall, 1e-6)
                    eval_mfu = eval_toks_per_sec * (6 * n_total) / 4e15 * 100
                    eval_history.append({
                        "progress": round(progress, 4),
                        "step": step,
                        "positions": total_tokens_processed,
                        "val_loss": val_loss,
                        "wall_clock_s": eval_wall,
                        "tokens_per_sec": eval_toks_per_sec,
                        "mfu_percent": eval_mfu,
                    })
                    print(f"[{run_name}] EVAL step={step} val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20)):.2f} "
                          f"positions={total_tokens_processed:,} avg_tok/s={eval_toks_per_sec:.0f} "
                          f"mfu={eval_mfu:.2f}% elapsed={total_training_time:.0f}s at={local_iso_now()}",
                          flush=True)
                last_eval_t = now

        step += 1
        if fixed_steps is not None:
            if step >= fixed_steps:
                break
        elif step > 10:
            # All DDP ranks must leave the loop on the same step. Per-rank wall times
            # differ slightly (and can differ substantially after an I/O stall), so
            # independent budget checks make early ranks enter final evaluation while
            # late ranks are still training, deadlocking mismatched collectives.
            # DDP backward already locks ranks to the same step.  Synchronize
            # the wall-clock decision periodically rather than issuing a tiny
            # control collective each training step; exact token caps above
            # remain checked before every optimizer update.
            sync_stop_now = (step % args.ddp_stop_sync_interval == 0 or
                             (position_stop > 0 and
                              total_tokens_processed + args.batch_size * args.seq_len * world_size >=
                              position_stop))
            if world_size > 1 and sync_stop_now:
                if rank == 0:
                    sync_flag.fill_(int(total_training_time >= args.time_budget or
                                        (position_stop > 0 and
                                         total_tokens_processed >= position_stop)))
                dist.broadcast(sync_flag, src=0)
                should_stop = bool(sync_flag.item())
            elif world_size == 1:
                should_stop = (total_training_time >= args.time_budget or
                               (position_stop > 0 and
                                total_tokens_processed >= position_stop))
            else:
                should_stop = False
            if should_stop:
                break

    val_loss = evaluate(model, val_loader_fn() if val_loader_fn else val_loader, args.eval_steps, device,
                        args.n_heads, enc_context_fn, args.dtype, args.use_flash_attn)
    if rank == 0:
        eval_history.append({"progress": 1.0, "step": step, "positions": total_tokens_processed,
                             "val_loss": val_loss, "final": True})
        run_wall_clock = time.time() - run_wall_clock_t0
        segment_positions = total_tokens_processed - segment_start_tokens
        avg_toks_per_sec = segment_positions / max(run_wall_clock, 1e-6)
        cumulative_toks_per_sec = total_tokens_processed / max(total_training_time, 1e-6)
        final_mfu = avg_toks_per_sec * (6 * n_total) / 4e15 * 100
        completed_at = local_iso_now()
        print(f"[{run_name}] DONE step={step} epoch={epoch} val_loss={val_loss:.4f} "
              f"val_ppl={math.exp(min(val_loss, 20)):.2f} wall_clock={run_wall_clock:.1f}s "
              f"tokens={total_tokens_processed:,} segment_tokens={segment_positions:,} "
              f"avg_tok/s={avg_toks_per_sec:.0f} cumulative_tok/s={cumulative_toks_per_sec:.0f} "
              f"mfu={final_mfu:.2f}% at={completed_at}", flush=True)
        if not is_smoke:
            # Sustained-throughput estimate for a full pass over the real tier-3 corpus, so a
            # short run (dry run or otherwise) tells you "how long would the whole thing take" --
            # not meaningful for --smoke-test, which uses a toy 2-layer/64d model, not the real
            # architecture, so its throughput doesn't reflect the real run's cost per token.
            est_epoch_s = TIER3_TOTAL_ROWS / max(avg_toks_per_sec, 1e-6)
            print(f"[{run_name}] at this throughput, one full pass over tier-3's "
                  f"{TIER3_TOTAL_ROWS:,} rows would take ~{est_epoch_s / 3600:.1f}h "
                  f"({est_epoch_s / 86400:.2f} days)", flush=True)
        if fixed_steps is None:
            save_checkpoint(ckpt_path, model, optimizer, step, total_training_time,
                            total_tokens_processed, vars(args))
        if args.results_dir:
            os.makedirs(args.results_dir, exist_ok=True)
            result = {
                "run_name": run_name, "architecture": args.architecture,
                "llama_qk_norm": args.llama_qk_norm,
                "depth": args.depth, "d_model": args.d_model,
                "n_heads": args.n_heads, "d_ff": args.d_ff, "dropout": args.dropout,
                "use_cross_attention": args.use_cross_attention,
                "xattn_mode": args.xattn_mode, "sattn_init": args.sattn_init, "new_lr": args.new_lr,
                "loss_side": args.loss_side, "init_checkpoint": args.init_checkpoint,
                "warmup_ratio": args.warmup_ratio, "lr_schedule": args.lr_schedule,
                "cosine_min_lr_ratio": args.cosine_min_lr_ratio,
                "fast_dummy_cross_attn": args.fast_dummy_cross_attn,
                "fused_adamw": args.fused_adamw,
                "torch_compile": args.torch_compile,
                "compile_mode": args.compile_mode if args.torch_compile else None,
                "batch_size": args.batch_size, "lr": lr, "dtype": args.dtype,
                "legal_mask_loss": args.legal_mask_loss, "legal_penalty_weight": args.legal_penalty_weight,
                "prepacked_dir": args.prepacked_dir,
                "prepacked_local_shards": args.prepacked_local_shards,
                "prepacked_without_replacement": args.prepacked_without_replacement,
                "max_train_positions": args.max_train_positions,
                "lr_schedule_by_positions": args.lr_schedule_by_positions,
                "time_budget": args.time_budget, "steps": step, "wall_clock_s": run_wall_clock,
                "n_trainable_params": n_trainable, "n_total_params": n_total,
                "tokens_per_sec": avg_toks_per_sec, "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20)),
                "total_positions": total_tokens_processed,
                "mfu_percent": final_mfu,
                "completed_at": completed_at,
                "eval_history": eval_history,
            }
            with open(os.path.join(args.results_dir, f"{run_name}.json"), "w") as f:
                json.dump(result, f, indent=2)
            tsv_path = os.path.join(args.results_dir, "results.tsv")
            write_header = not os.path.exists(tsv_path)
            with open(tsv_path, "a") as f:
                if write_header:
                    f.write("run_name\tval_loss\ttokens_per_sec\tbatch_size\tdtype\tlegal_mask_loss\t"
                            "legal_penalty_weight\tlr\tdescription\tcompleted_at\n")
                f.write(f"{run_name}\t{val_loss:.4f}\t{avg_toks_per_sec:.0f}\t{args.batch_size}\t{args.dtype}\t"
                        f"{args.legal_mask_loss}\t{args.legal_penalty_weight}\t{lr:g}\t\t{completed_at}\n")
    return val_loss


def _prune_old_checkpoints(checkpoint_dir: str, keep: int) -> None:
    numbered = sorted(glob.glob(os.path.join(checkpoint_dir, "step*.pt")),
                       key=lambda p: int(os.path.basename(p)[4:-3]))
    for p in numbered[:-keep] if keep > 0 else numbered:
        try:
            os.remove(p)
        except OSError:
            pass


def evaluate(model, val_loader, eval_steps: int, device: str, n_heads: int, enc_context_fn=None,
             dtype: str = "bf16", use_flash_attn: bool = True) -> float:
    """Always UNMASKED, regardless of --legal-mask-loss during training: this is the fixed
    yardstick used to compare configs (including the planned legal-mask-loss on/off A/B runs, see
    TRAINING.md). Applying legality masking at eval time too would trivially lower the metric for
    any masked run (fewer competing classes in the softmax) independent of whether the model
    actually predicts chess better, making the two runs' val_loss numbers incomparable.

    `dtype` matches the training run's --dtype (see parse_args) so eval isn't silently run at a
    different precision than training -- eval loss is the comparison metric across dtype configs,
    so this needs to reflect what each config actually trained/infers at, not a fixed default."""
    import torch
    from model import factored_cross_entropy, build_cu_seqlens

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    amp_enabled = device.startswith("cuda") and dtype != "fp32"
    use_flash = use_flash_attn and amp_enabled and dtype in ("bf16", "fp16")
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        for _ in range(eval_steps):
            x_np, y_np, _occ_np, _pmask_np, _elo_np, _ = next(val_loader)
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)
            enc = enc_context_fn(x.shape[0]) if enc_context_fn else None
            fwd_kwargs = (dict(zip(("cu_seqlens", "max_seqlen"), build_cu_seqlens(x))) if use_flash
                          else dict(mask=build_block_causal_mask(x)))
            logits = model(x, enc, **fwd_kwargs)
            total += factored_cross_entropy(logits, y).item()
    if was_training:
        model.train()
    return total / eval_steps


def main():
    args = parse_args()

    if args.lr_schedule_by_positions and args.max_train_positions <= 0:
        raise ValueError("--lr-schedule-by-positions requires a positive --max-train-positions")

    if args.smoke_test:
        assert not args.legal_mask_loss and args.legal_penalty_weight == 0, (
            "--legal-mask-loss/--legal-penalty-weight + --smoke-test: SyntheticDataset has no real chess "
            "legality to mask/penalize against (its occ/pmask are always zero, which would zero out every "
            "class, not 'no restriction') -- test these against real tier-3 data instead."
        )
        print("[smoke-test] running a short synthetic-data training loop, no GPU/tier-3 data required")
        args.time_budget = min(args.time_budget, 30.0)
        args.checkpoint_every = 10.0
        args.eval_every = 10.0
        args.depth, args.d_model, args.n_heads, args.d_ff = 2, 64, 2, 128
        args.batch_size = 4
        args.seq_len = 32
        args.checkpoint_dir = args.checkpoint_dir or "checkpoints/smoke_test"
        val = run_training(args, rank=0, world_size=1, local_rank=0)
        # exercise resume explicitly, since that's the one thing a single short run wouldn't otherwise touch
        args.resume = True
        run_training(args, rank=0, world_size=1, local_rank=0)
        print(f"[smoke-test] OK (val_loss={val:.4f})")
        return

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.lr_sweep:
        # Each rank runs its own short, independent, non-distributed probe (like the ablation
        # phase's CUDA_VISIBLE_DEVICES-per-GPU sweeps) — no process-group needed, ranks never
        # communicate. torchrun still sets LOCAL_RANK, which is all that's needed to pick a GPU.
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        lrs = [float(x) for x in args.lr_sweep.split(",")]
        my_lr = lrs[rank % len(lrs)]
        print(f"[lr-sweep] rank{rank} testing lr={my_lr:g} for {args.sweep_steps} steps")
        run_training(args, rank=0, world_size=1, local_rank=local_rank, lr_override=my_lr,
                      fixed_steps=args.sweep_steps)
        return

    if world_size > 1:
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    run_training(args, rank=rank, world_size=world_size, local_rank=local_rank)


if __name__ == "__main__":
    main()
