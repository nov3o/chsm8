"""Style-conditioned SFT ablations under a fixed wall-clock budget
(karpathy/autoresearch protocol: every trial gets the same *training*
seconds; startup, warmup steps and eval are excluded).

Model:  frozen pretrained text model  ->  trainable projection  ->  chess
        decoder (pt-100%) conditioned in its last K layers.

Text input (--encoder-kind):
  token  tokenize, run a frozen LM (decoder or encoder), cross-attend to its
         last-layer hidden state of every token
  embed  one pooled vector from a frozen sentence-embedding model
         (BERT-style), cross-attended as a length-1 context

Conditioning (--cond):
  xattn  the checkpoint's own cross-attention (frozen + fed a zero dummy
         context during pretraining), behind a per-layer tanh gate
         initialised at 0, so step 0 is exactly the pretrained policy
  film   adaLN/FiLM: pooled text -> per-layer (scale, shift) on the FFN's
         normalised input, zero-initialised
  both   xattn + film

Loss: the pretraining objective (factored CE + 0.5 * soft structural
illegal-mass penalty, no hard mask) restricted to the *labelled side's*
moves. Opponent moves are context only: their targets are turned into PAD,
so they contribute to neither CE nor the penalty.

Every optimizer step sees every label of every sampled game (the update is
the mean over all labels):
  --band-mode expand      one forward with G*n_labels rows
  --band-mode accumulate  n_labels micro-batches of G rows, grads summed
Both give the same update; they differ only in speed and memory.

Speed: layers below the first conditioned layer do not depend on the text,
so they run once per game and their output is shared by all of that game's
labels (and by every accumulate micro-batch). Frozen layers run under
no_grad. With K = --cond-layers < depth this skips most of the decoder's
compute for the 5x label expansion.
"""
import argparse
import json
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from factored_tokenizer import KIND_MOVE, KIND_PAD  # noqa: E402
from model import RMSNorm, build_cu_seqlens, factored_cross_entropy, MultiHeadAttention  # noqa: E402
from style_autoresearch import config_from_checkpoint  # noqa: E402
import meta_templates  # noqa: E402

META_COLS = ("player_elo", "opponent_elo", "tc_base", "tc_increment", "opening", "side_moves", "captures")

ROOT = Path(__file__).resolve().parents[1]
LEGAL_DTYPE = np.dtype([("occ", "<u8"), ("pmask", "u1")])


# ------------------------------------------------------------------ data

class SFTData:
    """Memmapped corpus from build_sft_corpus.py. Only index arrays live in RAM;
    label strings stay in Arrow and are fetched per batch."""

    def __init__(self, root: Path, n_llm: int, meta_attrs: list[str], max_rows: int):
        cols = ["split", "side", "row_off", "n_rows", "labels"] + (list(META_COLS) if meta_attrs else [])
        table = pq.read_table(root / "index.parquet", columns=cols)
        # >2 GB of label text overflows 32-bit string offsets on combine_chunks
        big = {"labels": pa.list_(pa.large_string()), "opening": pa.large_string()}
        for name, typ in big.items():
            if name in table.column_names:
                table = table.set_column(table.column_names.index(name), name, table.column(name).cast(typ))
        n = table.column("n_rows").to_numpy()
        n_lab = np.asarray(table.column("labels").combine_chunks().value_lengths())
        keep = (n <= max_rows) & (n_lab >= n_llm)
        if meta_attrs:  # every metadata field must be present to render its labels
            for c in META_COLS:
                keep &= ~np.asarray(table.column(c).is_null())
        pick = lambda c: table.column(c).combine_chunks().filter(keep)
        self.labels = pick("labels")
        self.orig = np.flatnonzero(keep)  # filtered id -> index.parquet record id
        self.n_records = len(keep)
        self.emb = self.emb_ids = None
        self.meta = {c: pick(c).to_pylist() for c in META_COLS} if meta_attrs else None
        self.row_off = table.column("row_off").to_numpy()[keep]
        self.n_rows = n[keep]
        self.white = (np.asarray(pick("side")) == "white")
        split = np.asarray(pick("split"))
        self.split = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
        # Load fully into RAM (~10 GB): random per-game memmap reads from Lustre
        # cost ~40 ms/game and capped every trial at ~30 games/s.
        t0 = time.time()
        self.rows = np.fromfile(root / "rows.i16", dtype="<i2").reshape(-1, 5)
        legal = np.fromfile(root / "legal.bin", dtype=LEGAL_DTYPE)
        self.occ = np.ascontiguousarray(legal["occ"]); self.pmask = np.ascontiguousarray(legal["pmask"])
        del legal
        print(f"[data] rows+legal in RAM in {time.time() - t0:.0f}s", flush=True)
        self.n_llm, self.meta_attrs = n_llm, list(meta_attrs)
        self.attr_idx = [meta_templates.ATTRS.index(a) for a in self.meta_attrs]
        self.n_labels = n_llm + len(self.meta_attrs)
        # few/normal/many-trades buckets = train-split terciles of captures per own move
        tr = self.split["train"]
        self.cap_thresh = (meta_templates.capture_thresholds(np.asarray(self.meta["captures"])[tr],
                                                             np.asarray(self.meta["side_moves"])[tr])
                           if meta_attrs else None)

    def restrict(self, n_train: int, n_eval: int, seed: int = 1234):
        """Fixed random subset per split (same ids for every trial/rank): n_train train, n_eval val and test."""
        rng = np.random.default_rng(seed)
        for s, n in (("train", n_train), ("val", n_eval), ("test", n_eval)):
            if n and len(self.split[s]) > n:
                self.split[s] = np.sort(rng.choice(self.split[s], n, replace=False))

    def load_label_emb(self, path: Path, dim: int):
        t0 = time.time()
        ids = Path(str(path) + ".ids.npy")  # compact file: rows only for these record ids (sorted, index.parquet space)
        self.emb_ids = np.load(ids) if ids.exists() else None
        n = len(self.emb_ids) if self.emb_ids is not None else self.n_records
        self.emb = np.memmap(path, dtype=np.float16, mode="r").reshape(n, -1, dim)  # page cache shared across ranks
        assert self.emb.shape[1] >= self.n_llm
        print(f"[data] label embeddings {self.emb.shape} in RAM in {time.time() - t0:.0f}s", flush=True)

    def texts_for(self, k: int, rng: random.Random) -> list[str]:
        out = [self.labels[k][b].as_py() for b in range(self.n_llm)]
        if self.meta_attrs:
            rendered = meta_templates.render({c: self.meta[c][k] for c in META_COLS}, rng, self.cap_thresh)
            out += [rendered[i] for i in self.attr_idx]
        return out

    def batch(self, idx: np.ndarray, rng: random.Random | None = None) -> dict:
        idx = np.asarray(idx)
        T = int(self.n_rows[idx].max())
        G = len(idx)
        rows = np.zeros((G, T, 5), dtype=np.int64)
        occ = np.zeros((G, T), dtype=np.uint64)
        pmask = np.zeros((G, T), dtype=np.uint64)
        for i, k in enumerate(idx):
            o, n = int(self.row_off[k]), int(self.n_rows[k])
            rows[i, :n] = self.rows[o:o + n]
            occ[i, :n] = self.occ[o:o + n]; pmask[i, :n] = self.pmask[o:o + n]
        # Target j is row j+1; white plays even j, black odd j.
        j = np.arange(T - 1)
        parity = np.where(self.white[idx], 0, 1)[:, None]
        focal = (rows[:, 1:, 0] == KIND_MOVE) & ((j[None, :] % 2) == parity)
        # eval (rng=None): wording fixed per game, identical across trials
        texts = [self.texts_for(int(k), rng or random.Random(int(k))) for k in idx]
        out = {"rows": torch.from_numpy(rows), "focal": torch.from_numpy(focal),
               "occ": torch.from_numpy(occ.view(np.int64)), "pmask": torch.from_numpy(pmask.view(np.int64)),
               "texts": texts, "n_focal": int(focal.sum()), "n_rows": int(self.n_rows[idx].sum()), "idx": np.asarray(idx)}
        if self.emb is not None:
            pos = self.orig[idx] if self.emb_ids is None else np.searchsorted(self.emb_ids, self.orig[idx])
            out["emb"] = torch.from_numpy(np.ascontiguousarray(self.emb[pos, :self.n_llm]))
        return out


class PlayerData(SFTData):
    """Top-player corpus (build_player_corpus.py). Each game gets k distinct wordings of
    "play like {p}" (meta_templates.render_player; names via player_names()), no metadata labels."""

    def __init__(self, root: Path, k: int, max_rows: int):
        t = pq.read_table(root / "index.parquet", columns=["split", "side", "row_off", "n_rows", "player"])
        n = t.column("n_rows").to_numpy()
        keep = n <= max_rows
        col = lambda c: np.asarray(t.column(c).to_pylist(), dtype=object)[keep]
        self.player = col("player")
        self.orig = np.flatnonzero(keep); self.n_records = len(keep)
        self.labels = self.meta = self.emb = self.cap_thresh = None
        self.row_off = t.column("row_off").to_numpy()[keep]; self.n_rows = n[keep]
        self.white = col("side") == "white"
        split = col("split")
        self.split = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
        self.rows = np.fromfile(root / "rows.i16", dtype="<i2").reshape(-1, 5)
        legal = np.fromfile(root / "legal.bin", dtype=LEGAL_DTYPE)
        self.occ = np.ascontiguousarray(legal["occ"]); self.pmask = np.ascontiguousarray(legal["pmask"])
        del legal
        self.n_llm, self.meta_attrs, self.attr_idx, self.n_labels = 0, [], [], k

    def texts_for(self, k: int, rng: random.Random) -> list[str]:
        out = []
        for _ in range(50 * self.n_labels):  # distinct wordings
            t = meta_templates.render_player(self.player[k], rng)
            if t not in out:
                out.append(t)
            if len(out) == self.n_labels:
                break
        return out

    def batch(self, idx, rng=None) -> dict:
        out = super().batch(idx, rng)
        out["player"] = True
        return out


def player_share(args, frac: float) -> float:
    """Share of games from the player corpus at progress frac: flat a until 0.8, then a linear
    ramp to peak*a; a is set so the mean over the run is --player-share."""
    a = args.player_share / (0.9 + 0.1 * args.player_peak)
    return a if frac < 0.8 else a * (1 + (args.player_peak - 1) * (frac - 0.8) / 0.2)


def player_schedule(data, args, rng, rank, world):
    """Deterministic curriculum: every train player game of this rank exactly once, placed at a
    progress position u = F^-1((i+.5)/n) of the pdf (flat until 0.8, linear ramp to --player-peak x).
    Returns (batches in position order, position of each batch's last game)."""
    ids = np.concatenate([rng.permutation(data.split["train"])[rank::world] for _ in range(max(1, args.player_epochs))])
    if args.player_max_games > 0:  # tests only
        ids = ids[:args.player_max_games // world]
    x = np.linspace(0, 1, 100001)
    pdf = np.where(x < 0.8, 1.0, 1 + (args.player_peak - 1) * (x - 0.8) / 0.2)
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(x))]); cdf /= cdf[-1]
    u = np.interp((np.arange(len(ids)) + 0.5) / len(ids), cdf, x)
    batches, pos, start = [], [], 0
    for i in range(1, len(ids) + 1):
        g = i - start
        if i == len(ids) or g >= args.games or (g + 1) * int(max(data.n_rows[ids[start:i + 1]])) * data.n_labels > args.token_budget:
            batches.append(ids[start:i]); pos.append(u[i - 1]); start = i
    return batches, np.array(pos)


def length_bucketed_batches(data: SFTData, split: str, max_games: int, token_budget: int,
                            rng: np.random.Generator, pool_size: int = 16384, shard=None):
    """Endless train stream: draw a pool, sort by length, cut into batches
    whose padded size (games * max_len * n_labels label-row tokens) stays
    within token_budget, shuffle batch order. Keeps GPU memory bounded and
    steps large (few-kernel-launch overhead) without a length curriculum."""
    ids = data.split[split]
    if shard is not None:  # one pass without replacement, this rank's 1/WORLD slice; reshuffled per epoch
        ids = ids[shard[0]::shard[1]]
    while True:
      order = rng.permutation(ids) if shard is not None else None
      for s0 in range(0, len(ids), pool_size) if shard is not None else iter(int, 1):
        pool = order[s0:s0 + pool_size] if shard is not None else ids[rng.integers(0, len(ids), size=pool_size)]
        pool = pool[np.argsort(data.n_rows[pool], kind="stable")]
        batches, start = [], 0
        for i in range(len(pool)):
            g = i - start + 1
            if g > max_games or g * int(data.n_rows[pool[i]]) * data.n_labels > token_budget:
                batches.append(pool[start:i]); start = i
        batches.append(pool[start:])
        rng.shuffle(batches)
        yield from batches


class Prefetcher:
    def __init__(self, data: SFTData, it, depth: int = 6, workers: int = 3):
        self.q = queue.Queue(maxsize=depth)
        lock = threading.Lock()

        def work(seed):
            rng = random.Random(seed)
            while True:
                with lock:
                    idx = next(it, None)
                if idx is None:  # finite stream (exact player schedule) exhausted
                    return
                b = data.batch(idx, rng)
                if torch.cuda.is_available():
                    for k in ("rows", "focal", "occ", "pmask", "emb"):
                        if k in b:
                            b[k] = b[k].pin_memory()
                self.q.put(b)
        for w in range(workers):
            threading.Thread(target=work, args=(1000 + w,), daemon=True).start()

    def get(self) -> dict:
        return self.q.get()


# ------------------------------------------------------------------ text

class FrozenText(nn.Module):
    def __init__(self, name: str, kind: str, pool: str, max_len: int, device, mrl_dim: int = 0):
        super().__init__()
        self.mrl_dim, self.cache, self.n_llm, self.prefix = mrl_dim, {}, None, ""  # prefix: instruction for instruct embedders
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        mtype = AutoConfig.from_pretrained(name).model_type
        if mtype == "t5gemma":  # encoder half only (e.g. google/t5gemma-2b-2b-*, t5gemma-9b-2b-*)
            from transformers import T5GemmaEncoderModel
            self.model = T5GemmaEncoderModel.from_pretrained(name, dtype=torch.bfloat16, is_encoder_decoder=False)
        elif mtype in ("t5", "mt5", "umt5"):
            from transformers import T5EncoderModel
            self.model = T5EncoderModel.from_pretrained(name, dtype=torch.bfloat16)
        else:
            self.model = AutoModel.from_pretrained(name, dtype=torch.bfloat16)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.kind, self.pool, self.max_len, self.device = kind, pool, max_len, device
        cfg = self.model.config
        self.dim = getattr(cfg, "hidden_size", None) or getattr(getattr(cfg, "encoder", None), "hidden_size", None) or cfg.d_model
        if mrl_dim:
            assert kind == "embed", "--emb-dim (Matryoshka truncation) only applies to --encoder-kind embed"
            self.dim = mrl_dim

    @torch.no_grad()
    def forward(self, texts: list[str]):
        texts = [self.prefix + t for t in texts] if self.prefix else texts
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state                   # (B, L, D) bf16
        mask = enc["attention_mask"].bool()
        if self.pool == "cls":
            pooled = h[:, 0]
        elif self.pool == "last":
            pooled = h[torch.arange(h.shape[0], device=h.device), mask.sum(1) - 1]
        else:
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        if self.mrl_dim:  # same function as precompute_label_emb.embed
            pooled = F.normalize(F.normalize(pooled.float(), dim=-1)[:, :self.mrl_dim], dim=-1)
        if self.kind == "embed":
            return pooled.unsqueeze(1), torch.ones(h.shape[0], 1, dtype=torch.bool, device=h.device), pooled
        return h, mask, pooled


    @torch.no_grad()
    def pooled_cached(self, texts: list[str]) -> torch.Tensor:
        """(B, dim) pooled vectors; each distinct string is encoded once."""
        miss = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for s in range(0, len(miss), 1024):
            chunk = miss[s:s + 1024]
            _, _, pooled = self(chunk)
            for t, v in zip(chunk, pooled.to(torch.float16)):
                self.cache[t] = v
        return torch.stack([self.cache[t] for t in texts]).float()


class Conditioner(nn.Module):
    """All new trainable conditioning weights except the checkpoint's own
    cross-attention: projection into d_model, per-layer gates, FiLM heads."""

    def __init__(self, enc_dim: int, d_model: int, n_cond: int, cond: str):
        super().__init__()
        self.use_xattn, self.use_film = cond in ("xattn", "both"), cond in ("film", "both")
        self.proj = nn.Sequential(nn.Linear(enc_dim, d_model), RMSNorm(d_model))
        self.gate = nn.Parameter(torch.zeros(n_cond))
        if self.use_film:
            self.film = nn.Linear(enc_dim, n_cond * 2 * d_model)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.n_cond, self.d = n_cond, d_model

    def prepare(self, ctx, mask, pooled):
        out = {"ctx": self.proj(ctx.float()) if self.use_xattn else None,
               "mask": mask[:, None, None, :]}
        if self.use_film:
            out["film"] = self.film(pooled.float()).view(-1, self.n_cond, 2, self.d)
        return out


# ------------------------------------------------------------------ model

USE_FLASH = True
LABEL_W = None  # (n_labels,) CE weight per label slot
RANK, WORLD = 0, 1  # torchrun data parallel


def attn_args(rows):
    if USE_FLASH:
        return build_cu_seqlens(rows) + (None,)
    T = rows.shape[1]
    return None, None, torch.ones(T, T, dtype=torch.bool, device=rows.device).tril()


def rope_for(chess, rows):
    B, T = rows.shape[:2]
    pos = torch.arange(T, device=rows.device).expand(B, T)  # one game per row, BOS at 0
    dev = rows.device.type
    dtype = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else torch.float32
    return pos, MultiHeadAttention.rope_factors(pos, chess.layers[0].self_attn.head_dim, dtype)


def layer_fwd(layer, x, rows, pos, rope, attn, cond=None, ci=None, drop=None):
    cu, maxlen, causal = attn
    n1 = layer.n1(x)
    x = x + layer.self_attn(n1, n1, n1, attn_mask=causal, cu_seqlens=cu, max_seqlen=maxlen,
                            position_ids=pos, rope_factors=rope)
    ca = layer.cross_attn
    # The pretrained policy's cross-attn output under its zero dummy context.
    dummy = getattr(layer, "frozen_dummy", None)  # set by --freeze-dummy: the no-text path stays exactly the start policy
    if dummy is None:
        dummy = ca.out_proj(ca.v_proj(torch.zeros(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)))
    x = x + dummy.to(x.dtype)
    h = layer.n3(x)
    if cond is not None:
        if cond["ctx"] is not None:
            a = ca(layer.n2(x), cond["ctx"].to(x.dtype), cond["ctx"].to(x.dtype), attn_mask=cond["mask"])
            g = torch.tanh(cond["gate"][ci])
            if drop is not None:
                a = a * (~drop).view(-1, 1, 1).to(a.dtype)
            x = x + g * a
            h = layer.n3(x)
        if "film" in cond:
            f = cond["film"][:, ci]
            if drop is not None:
                f = f * (~drop).view(-1, 1, 1).to(f.dtype)
            h = h * (1 + f[:, 0, None, :].to(h.dtype)) + f[:, 1, None, :].to(h.dtype)
    return x + layer.ff(h)


def forward_shared(chess, rows, n_shared, n_frozen):
    """Text-independent bottom of the decoder, once per game."""
    x = chess.embed(rows)
    pos, rope = rope_for(chess, rows)
    attn = attn_args(rows)
    grad = torch.is_grad_enabled()
    for i in range(n_shared):
        with torch.set_grad_enabled(grad and i >= n_frozen):
            x = layer_fwd(chess.layers[i], x, rows, pos, rope, attn)
    return x


def forward_top(chess, x, rows, n_shared, cond, drop=None):
    pos, rope = rope_for(chess, rows)
    attn = attn_args(rows)
    for ci, i in enumerate(range(n_shared, len(chess.layers))):
        x = layer_fwd(chess.layers[i], x, rows, pos, rope, attn, cond, ci, drop)
    return chess.head(chess.norm(x))


def focal_loss(logits, rows, focal, occ, pmask, penalty, row_weight=None):
    """row_weight: optional (B,) per-label-row CE weight (weighted mean over focal moves)."""
    tgt = rows[:, 1:].clone()
    tgt[..., 0] = torch.where(focal, tgt[..., 0], torch.full_like(tgt[..., 0], KIND_PAD))
    logits = {k: v[:, :-1] for k, v in logits.items()}
    w = None if row_weight is None else row_weight.view(-1, 1).expand_as(focal).float().contiguous()
    return factored_cross_entropy(logits, tgt, own_occ=occ[:, 1:], piece_mask_bits=pmask[:, 1:],
                                  hard_mask=False, illegal_penalty_weight=penalty, move_weights=w)


# ------------------------------------------------------------------ run

def build(args, device):
    chess, cfg, _ = config_from_checkpoint(Path(args.checkpoint), device)
    L = len(chess.layers)
    K = L if args.cond_layers <= 0 else min(args.cond_layers, L)
    N = min(args.unfreeze_last, L)
    n_shared = L - K
    n_frozen = L - max(K, N)  # layers that need no grad at all
    text = FrozenText(args.encoder, args.encoder_kind, args.pool, args.max_text, device, args.emb_dim)
    text.prefix = getattr(args, "text_prefix", "").replace("\\n", "\n")
    cond = Conditioner(text.dim, cfg.d_model, K, args.cond).to(device)

    chess.requires_grad_(False)
    body, cross = [], []
    for i, layer in enumerate(chess.layers):
        if i >= L - N:
            for name, p in layer.named_parameters():
                if not name.startswith(("cross_attn.", "n2.")):
                    p.requires_grad_(True); body.append(p)
        if i >= n_shared and cond.use_xattn:
            for p in list(layer.cross_attn.parameters()) + list(layer.n2.parameters()):
                p.requires_grad_(True); cross.append(p)
    if N > 0 or args.train_head:
        for p in list(chess.norm.parameters()) + list(chess.head.parameters()):
            p.requires_grad_(True); body.append(p)
    groups = [{"params": list(cond.parameters()) + cross, "lr": args.cond_lr, "name": "cond"}]
    if body:
        groups.append({"params": body, "lr": args.lr, "name": "body"})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=device.type == "cuda")
    info = {"depth": L, "cond_layers": K, "unfreeze_last": N, "n_shared": n_shared, "n_frozen": n_frozen,
            "encoder_dim": text.dim,
            "trainable_cond": sum(p.numel() for p in groups[0]["params"]),
            "trainable_body": sum(p.numel() for p in body)}
    return chess, text, cond, opt, info


def lr_mult(args, frac: float) -> float:
    if frac < args.warmup_frac:
        return frac / max(args.warmup_frac, 1e-9)
    if args.schedule == "constant":
        return 1.0
    p = (frac - args.warmup_frac) / max(1 - args.warmup_frac, 1e-9)
    return args.final_lr_frac + (1 - args.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def encode_labels(text, cond, batch, label_ids):
    texts_by_game = batch["texts"]
    if "emb" in batch or (batch.get("player") and text.kind == "embed"):  # precomputed LLM labels; metadata/player labels cached
        emb = batch["emb"].to(cond.gate.device, non_blocking=True).float() if "emb" in batch else None
        cols = [emb[:, li] if emb is not None and li < text.n_llm
                else text.pooled_cached([g[li] for g in texts_by_game]) for li in label_ids]
        pooled = torch.stack(cols, 1).reshape(-1, cols[0].shape[-1])  # game-major
        ctx, mask = pooled.unsqueeze(1), torch.ones(pooled.shape[0], 1, dtype=torch.bool, device=pooled.device)
    else:
        flat = [g[b] for g in texts_by_game for b in label_ids]  # game-major: g0b0..g0bk, g1b0..
        ctx, mask, pooled = text(flat)
    out = cond.prepare(ctx, mask, pooled)
    out["gate"] = cond.gate
    return out


def train_step(args, chess, text, cond, opt, b, device, n_shared, n_frozen, n_labels, label_w=None):
    label_w = LABEL_W if label_w is None else label_w
    rows = b["rows"].to(device, non_blocking=True)
    focal = b["focal"].to(device, non_blocking=True)
    occ = b["occ"].to(device, non_blocking=True); pmask = b["pmask"].to(device, non_blocking=True)
    G = rows.shape[0]
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        shared = forward_shared(chess, rows, n_shared, n_frozen)
        shared_needs_grad = shared.requires_grad
        if args.band_mode == "expand":
            # --labels-per-step k: a random subset of k label slots this step (0 = all)
            ids = (sorted(random.sample(range(n_labels), args.labels_per_step))
                   if 0 < args.labels_per_step < n_labels else list(range(n_labels)))
            k = len(ids)
            c = encode_labels(text, cond, b, ids)
            R = lambda t: t.repeat_interleave(k, 0)
            drop = (torch.rand(G * k, device=device) < args.cfg_dropout) if args.cfg_dropout else None
            logits = forward_top(chess, R(shared), R(rows), n_shared, c, drop)
            w = label_w[ids].to(device).repeat(G)
            loss = focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), args.illegal_penalty, w)
            loss.backward()
            loss_value = float(loss.detach())
        else:
            total = 0.0
            for li in range(n_labels):
                c = encode_labels(text, cond, b, [li])
                drop = (torch.rand(G, device=device) < args.cfg_dropout) if args.cfg_dropout else None
                logits = forward_top(chess, shared, rows, n_shared, c, drop)
                loss = focal_loss(logits, rows, focal, occ, pmask, args.illegal_penalty) * float(label_w[li] / label_w.sum())
                loss.backward(retain_graph=shared_needs_grad and li < n_labels - 1)
                total += float(loss.detach())
            loss_value = total
    params = [p for g in opt.param_groups for p in g["params"]]
    if WORLD > 1:  # data parallel: average grads (same param set on every rank)
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
        flat = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(flat)
        flat /= WORLD
        for p, g in zip(params, _unflatten_dense_tensors(flat, grads)):
            p.grad = g
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
    opt.step()
    return loss_value, gnorm


@torch.no_grad()
def evaluate(args, chess, text, cond, data, device, n_shared, split, n_labels):
    chess.eval()
    ids = data.split[split]
    rng = np.random.default_rng(args.eval_seed)  # fixed subset, identical for every trial
    ids = np.sort(rng.choice(ids, size=min(len(ids), args.eval_games), replace=False))
    ids = ids[np.argsort(data.n_rows[ids], kind="stable")]
    sums = {"matched": 0.0, "matched_ce": 0.0, "matched_ce_llm": 0.0, "matched_ce_meta": 0.0,
            "mismatched_ce": 0.0, "zero_ce": 0.0}
    llm_mask = torch.tensor([1.0] * data.n_llm + [0.0] * (n_labels - data.n_llm), device=device)
    n_batches = 0
    for s in range(0, len(ids), args.eval_games_per_batch):
        b = data.batch(ids[s:s + args.eval_games_per_batch])
        rows, focal = b["rows"].to(device), b["focal"].to(device)
        occ, pmask = b["occ"].to(device), b["pmask"].to(device)
        G = rows.shape[0]
        R = lambda t: t.repeat_interleave(n_labels, 0)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            shared = forward_shared(chess, rows, n_shared, len(chess.layers))
            c = encode_labels(text, cond, b, list(range(n_labels)))
            logits = forward_top(chess, R(shared), R(rows), n_shared, c)
            sums["matched"] += float(focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), args.illegal_penalty))
            sums["matched_ce"] += float(focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0))
            if data.n_llm:
                sums["matched_ce_llm"] += float(focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0, llm_mask.repeat(G)))
            if n_labels > 1:
                for bi in range(n_labels):  # per slot: terse_1, terse_2, short, medium, long, then metadata attrs
                    onehot = torch.zeros(n_labels, device=device); onehot[bi] = 1.0
                    sums[f"b{bi}_matched_ce"] = sums.get(f"b{bi}_matched_ce", 0.0) + float(
                        focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0, onehot.repeat(G)))
            if n_labels > data.n_llm:
                sums["matched_ce_meta"] += float(focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0, (1 - llm_mask).repeat(G)))
            if G > 1:  # diagnostic: another game's labels (roll by one whole game)
                wrong = {k: (v.roll(n_labels, 0) if k in ("ctx", "film") and v is not None else v) for k, v in c.items()}
                wrong["mask"] = c["mask"].roll(n_labels, 0)
                logits = forward_top(chess, R(shared), R(rows), n_shared, wrong)
                sums["mismatched_ce"] += float(focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0))
                for bi in range(n_labels):
                    onehot = torch.zeros(n_labels, device=device); onehot[bi] = 1.0
                    sums[f"b{bi}_mismatched_ce"] = sums.get(f"b{bi}_mismatched_ce", 0.0) + float(
                        focal_loss(logits, R(rows), R(focal), R(occ), R(pmask), 0.0, onehot.repeat(G)))
            logits = forward_top(chess, shared, rows, n_shared, None)  # pretrained policy path, no text
            sums["zero_ce"] += float(focal_loss(logits, rows, focal, occ, pmask, 0.0))
        n_batches += 1
    chess.train()
    out = {f"{split}_{k}": v / max(n_batches, 1) for k, v in sums.items()}
    out[f"{split}_matched_minus_mismatched"] = out[f"{split}_matched_ce"] - out[f"{split}_mismatched_ce"]
    out[f"{split}_matched_minus_zero"] = out[f"{split}_matched_ce"] - out[f"{split}_zero_ce"]
    out[f"{split}_games"] = int(len(ids))
    return out


def apply_filters(data, root, pure=False, labelers="", eval_labelers=""):
    """--pure / --labelers / --eval-labelers record filters (shared with dpo_sft.py)."""
    if pure:  # only label sets whose 5 bands all passed the contradiction check (label_correct all 1)
        lc = pq.read_table(root / "index.parquet", columns=["label_correct"]).column("label_correct").to_pylist()
        ok = np.array([bool(x) and all(v == 1 for v in x) for x in lc])[data.orig]
        for ss in data.split:
            data.split[ss] = data.split[ss][ok[data.split[ss]]]
    if labelers or eval_labelers:  # keep only records from these labeler families (train / val+test)
        lab = np.asarray(pq.read_table(root / "index.parquet", columns=["labeler"]).column("labeler").to_pylist(), dtype=object)[data.orig]
        for ss, want in (("train", labelers), ("val", eval_labelers), ("test", eval_labelers)):
            if want:
                data.split[ss] = data.split[ss][np.isin(lab[data.split[ss]], want.split(","))]
        print(f"[labelers] train {len(data.split['train'])} val {len(data.split['val'])}", flush=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run(args):
    global USE_FLASH, LABEL_W, RANK, WORLD
    USE_FLASH = not args.no_flash
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        torch.distributed.init_process_group("nccl")
        RANK, WORLD = torch.distributed.get_rank(), torch.distributed.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(args.seed); random.seed(args.seed)  # same init on every rank
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        USE_FLASH = False
    attrs = [a for a in args.meta_labels.split(",") if a]
    data = SFTData(Path(args.data), args.n_llm_labels, attrs, args.max_rows)
    apply_filters(data, Path(args.data), args.pure, args.labelers, args.eval_labelers)
    if args.subset_train or args.subset_eval:
        data.restrict(args.subset_train, args.subset_eval)
    if args.dump_subset:  # record ids (index.parquet space) of all split members, for embedding a compact label file
        np.save(args.dump_subset, np.unique(np.concatenate([data.orig[v] for v in data.split.values()])))
        print(f"[subset] wrote {args.dump_subset}", flush=True)
        return
    n_labels = data.n_labels
    if args.label_emb:
        assert args.emb_dim and args.encoder_kind == "embed", "--label-emb needs --emb-dim and --encoder-kind embed"
        data.load_label_emb(Path(args.label_emb), args.emb_dim)
    if args.label_weights:
        LABEL_W = torch.tensor([float(x) for x in args.label_weights.split(",")])
    else:
        LABEL_W = torch.tensor([1.0] * data.n_llm + [args.meta_weight] * len(attrs))
    assert len(LABEL_W) == n_labels, f"--label-weights needs {n_labels} weights"
    W_START = LABEL_W.clone()
    W_END = torch.tensor([float(x) for x in args.label_weights_end.split(",")]) if args.label_weights_end else None
    assert W_END is None or len(W_END) == n_labels, f"--label-weights-end needs {n_labels} weights"
    chess, text, cond, opt, info = build(args, device)
    if args.init_from:  # evaluate / continue from a saved SFT model (--max-steps 0 = eval only)
        sd = torch.load(args.init_from, map_location=device, weights_only=False)
        chess.load_state_dict(sd["chess"]); cond.load_state_dict(sd["cond"])
        for i, t in sd.get("frozen_dummy", {}).items():
            chess.layers[int(i)].frozen_dummy = t.to(device)
        print(f"[init] {args.init_from}", flush=True)
    if args.freeze_dummy:  # no-text path = the starting policy forever (cross-attn out_proj(v_proj(0)) pinned)
        with torch.no_grad():
            for l in chess.layers:
                if hasattr(l, "cross_attn") and getattr(l, "frozen_dummy", None) is None:
                    ca = l.cross_attn; z = torch.zeros(1, 1, ca.v_proj.in_features, device=ca.v_proj.weight.device, dtype=ca.v_proj.weight.dtype)
                    l.frozen_dummy = ca.out_proj(ca.v_proj(z)).detach().clone()
    text.n_llm = data.n_llm
    if args.compile:
        # Fuse the launch-heavy elementwise parts (RMSNorm, SwiGLU); attention
        # stays eager (FA3 is not traceable). dynamic=True: variable batch/len.
        for layer in chess.layers:
            for name in ("n1", "n2", "n3", "ff"):
                if hasattr(layer, name):
                    setattr(layer, name, torch.compile(getattr(layer, name), dynamic=True))
        chess.norm = torch.compile(chess.norm, dynamic=True)
    n_shared, n_frozen = info["n_shared"], info["n_frozen"]
    info["capture_thresholds"] = data.cap_thresh
    print(json.dumps({"trial": args.trial, **info, "train_games": len(data.split["train"])}), flush=True)

    base = (evaluate(args, chess, text, cond, data, device, n_shared, "val", n_labels)
            if args.eval_baseline and RANK == 0 else {})
    pbase = {}
    exact = args.planned_games != 0
    if args.planned_games < 0:  # -1 = exactly one pass over the normal train split
        args.planned_games = len(data.split["train"])
    stream = Prefetcher(data, length_bucketed_batches(data, "train", args.games, args.token_budget,
                                                      np.random.default_rng(args.seed * 1000 + RANK),
                                                      shard=(RANK, WORLD) if exact else None))
    pdata = pstream = None; p_games = n_games = 0; p_target = 0.0; p_next = 0  # integral of ps/(1-ps) d(normal games)
    if args.player_data:
        pdata = PlayerData(Path(args.player_data), args.player_labels, args.max_rows)
        if exact:
            pb, ppos = player_schedule(pdata, args, np.random.default_rng(args.seed), RANK, WORLD)
            pstream = Prefetcher(pdata, iter(pb)); p_next = 0
            print(f"[players] exact schedule: {len(pb)} batches, {sum(map(len, pb))} games/rank; "
                  f"per 10%: {np.histogram(ppos, bins=10, range=(0, 1), weights=[len(b) for b in pb])[0].astype(int).tolist()}", flush=True)
        else:
            pstream = Prefetcher(pdata, length_bucketed_batches(pdata, "train", args.games, args.token_budget,
                                                                np.random.default_rng(args.seed * 1000 + RANK + 7)))
        player_w = torch.full((args.player_labels,), args.player_weight)
        print(f"[players] {len(pdata.split['train'])} train games, k={args.player_labels}", flush=True)
    plan = None
    if args.plan:  # pre-packed run (build_sft_plan.py): exact batches and step kinds, nothing sampled here
        z = np.load(Path(args.plan) / "plan.npz")
        ids, boff, kinds, sb = z["ids"], z["boff"], z["step_kind"], z["step_batches"]
        assert sb.shape[1] == WORLD, f"plan built for world {sb.shape[1]}, running {WORLD}"
        mine = [ids[boff[b]:boff[b + 1]] for b in sb[:, RANK]]
        plan = {"kind": kinds, "n": [m for m, k in zip(mine, kinds) if k == 0], "p": [m for m, k in zip(mine, kinds) if k == 1]}
        print(f"[plan] {len(kinds)} steps ({int(kinds.sum())} player), rank {RANK}: "
              f"{sum(map(len, plan['n']))} normal + {sum(map(len, plan['p']))} player games", flush=True)
    kinds = []  # 1 = player step, for the per-kind loss in the log
    measured = 0.0; step = 0; losses = []; games = focal_moves = rows_seen = 0
    ck_path = Path(args.save_dir) / f"{args.trial}.snap.pt"; next_ck = args.ckpt_every
    if args.resume and ck_path.exists():
        sd = torch.load(ck_path, map_location=device, weights_only=False)
        chess.load_state_dict(sd["chess"]); cond.load_state_dict(sd["cond"]); opt.load_state_dict(sd["opt"])
        measured, step, p_games, n_games = sd["measured"], sd["step"], sd["p_games"], sd["n_games"]; p_target = sd.get("p_target", float(p_games))
        if pdata is not None and exact:  # skip player batches already trained
            p_next = sd.get("p_next", 0); pstream = Prefetcher(pdata, iter(pb[p_next:]))
        next_ck = measured + args.ckpt_every
        print(f"[resume] {ck_path} step {step} train_s {measured:.0f}", flush=True)
    if plan is not None:  # streams start where the plan stands at `step` (0, or the resumed step)
        k0 = plan["kind"][:step]
        stream = Prefetcher(data, iter(plan["n"][int((k0 == 0).sum()):]))
        if pdata is not None:
            pstream = Prefetcher(pdata, iter(plan["p"][int((k0 == 1).sum()):]))
    t_start = time.time()
    while step < args.max_steps and (step <= args.warmup_steps or measured < args.time_budget):
        if WORLD > 1 and step > args.warmup_steps:  # rank 0's clock decides, so all ranks stop together
            m = torch.tensor([measured], device=device); torch.distributed.broadcast(m, 0); measured = float(m)
            if measured >= args.time_budget:
                break
        if plan is not None and step >= len(plan["kind"]):
            break  # plan done (same step count on every rank)
        if args.max_games > 0:  # fixed data budget; rank 0 decides (per-rank game counts differ)
            done = torch.tensor([float(n_games * WORLD >= args.max_games)], device=device)
            if WORLD > 1:
                torch.distributed.broadcast(done, 0)
            if bool(done):
                break
        mult = lr_mult(args, min(step / len(plan["kind"]) if plan is not None else
                                 n_games * WORLD / args.planned_games if args.planned_games > 0
                                 else measured / args.time_budget, 1.0))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        pfrac = (min(n_games * WORLD / args.planned_games, 1.0) if args.planned_games > 0  # progress by games, not time:
                 else min(measured / args.time_budget, 1.0))  # player steps are ~9x cheaper, so time skews the mix
        ps = player_share(args, pfrac) if pdata is not None else 0.0
        if plan is not None:
            is_player = bool(plan["kind"][step])
        elif exact:
            prog = n_games * WORLD / args.planned_games
            stop = prog >= 1.0 and (pdata is None or p_next >= len(pb))
            is_player = pdata is not None and p_next < len(pb) and (ppos[p_next] <= prog or prog >= 1.0)
            if WORLD > 1:  # rank 0 decides so all ranks take the same kind of step and stop together
                d = torch.tensor([float(stop), float(is_player)], device=device); torch.distributed.broadcast(d, 0)
                stop, is_player = bool(d[0]), bool(d[1])
            if stop:
                break
            p_next += is_player
        else:
            is_player = pdata is not None and p_games < p_target
        b = pstream.get() if is_player else stream.get()
        if is_player:
            p_games += len(b["texts"])
        else:
            n_games += len(b["texts"]); p_target += ps / (1 - ps) * len(b["texts"])
        if W_END is not None:  # curriculum: per-label weights move linearly from --label-weights to --label-weights-end
            LABEL_W = W_START * (1 - pfrac) + W_END * pfrac
        sync(device); t0 = time.time()
        loss, gnorm = train_step(args, chess, text, cond, opt, b, device, n_shared, n_frozen,
                                 args.player_labels if is_player else n_labels, player_w if is_player else None)
        sync(device)
        step += 1
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step > args.warmup_steps:
            measured += time.time() - t0
            nl = args.player_labels if is_player else n_labels
            k_used = args.labels_per_step if 0 < args.labels_per_step < nl else nl
            games += len(b["texts"]) * WORLD; focal_moves += b["n_focal"] * k_used * WORLD; rows_seen += b["n_rows"] * k_used * WORLD
        losses.append(loss); kinds.append(int(is_player))
        if args.ckpt_every > 0 and measured >= next_ck:
            next_ck = measured + args.ckpt_every
            if RANK == 0:
                ck_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"chess": chess.state_dict(), "cond": cond.state_dict(), "opt": opt.state_dict(),
                            "measured": measured, "step": step, "p_games": p_games, "n_games": n_games, "p_target": p_target,
                            "p_next": p_next if pdata is not None and exact else 0,
                            "args": vars(args)}, str(ck_path) + ".tmp")
                os.replace(str(ck_path) + ".tmp", ck_path)
                print(f"[ckpt] step {step} train_s {measured:.0f}", flush=True)
        if args.val_every and step % args.val_every == 0:  # periodic val curve (rank 0; others wait)
            if RANK == 0:
                va = argparse.Namespace(**{**vars(args), "eval_games": args.val_games})
                vm = evaluate(va, chess, text, cond, data, device, n_shared, "val", n_labels)
                if pdata is not None:
                    vm.update({"p" + k: v for k, v in evaluate(va, chess, text, cond, pdata, device, n_shared, "val", args.player_labels).items()})
                print("[val] " + json.dumps({"step": step, "plan_frac": round(step / len(plan["kind"]), 5) if plan is not None else None,
                                             **{k: round(v, 5) for k, v in vm.items() if isinstance(v, float)}}), flush=True)
            if WORLD > 1:
                torch.distributed.barrier()
        if step % args.log_every == 0 and RANK == 0:
            print(json.dumps({"step": step, "loss": float(np.mean(losses[-args.log_every:])), "gnorm": gnorm,
                              "lr_mult": mult, "train_s": round(measured, 1),
                              "rows_per_s": round(rows_seen / max(measured, 1e-9)),
                              "n_games": n_games * WORLD, "p_games": p_games * WORLD,
                              "p_steps": int(sum(kinds[-args.log_every:])),
                              "loss_n": (float(np.mean([l for l, k in zip(losses[-args.log_every:], kinds[-args.log_every:]) if not k]))
                                         if not all(kinds[-args.log_every:]) else None),
                              "loss_p": (float(np.mean([l for l, k in zip(losses[-args.log_every:], kinds[-args.log_every:]) if k]))
                                         if any(kinds[-args.log_every:]) else None),
                              "plan_frac": round(step / len(plan["kind"]), 5) if plan is not None else None,
                              "gates": [round(float(torch.tanh(x)), 4) for x in cond.gate.detach()]}), flush=True)
    if RANK != 0:
        torch.distributed.barrier(); return
    metrics = evaluate(args, chess, text, cond, data, device, n_shared, "val", n_labels)
    if pdata is not None:
        pm = evaluate(args, chess, text, cond, pdata, device, n_shared, "val", 1)
        metrics.update({"p" + k: v for k, v in pm.items()})
        metrics.update(player_games_seen=p_games * WORLD, normal_games_seen=n_games * WORLD)
    if args.eval_test:
        metrics.update(evaluate(args, chess, text, cond, data, device, n_shared, "test", n_labels))
    result = {"trial": args.trial, **{k: v for k, v in vars(args).items() if k != "trial"}, **info,
              "steps": step, "train_seconds": measured,
              "peak_mem_gb": (torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else None, "wall_seconds": time.time() - t_start,
              "world_size": WORLD, "records_per_s": games / max(measured, 1e-9), "games_seen": games, "label_rows_per_s": rows_seen / max(measured, 1e-9),
              "focal_moves_per_s": focal_moves / max(measured, 1e-9),
              "train_loss_last50": float(np.mean(losses[-50:])) if losses else None,
              **{f"baseline_{k}": v for k, v in base.items()}, **pbase, **metrics}
    out = Path(args.results); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("RESULT " + json.dumps(result), flush=True)
    if args.save:
        path = Path(args.save_dir) / f"{args.trial}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "sft-ablate-v1", "args": vars(args), "info": info,
                    "chess": {k.replace("._orig_mod", ""): v for k, v in chess.state_dict().items()}, "cond": cond.state_dict(),
                    "frozen_dummy": {i: l.frozen_dummy.cpu() for i, l in enumerate(chess.layers) if getattr(l, "frozen_dummy", None) is not None}}, path)
    if WORLD > 1:
        torch.distributed.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trial", required=True)
    p.add_argument("--data", default=str(ROOT / "data/sft"))
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/pt100/chsm8_latest.pt"))
    p.add_argument("--results", default=str(ROOT / "results/sft_ablate.jsonl"))
    p.add_argument("--save", action="store_true"); p.add_argument("--save-dir", default=str(ROOT / "checkpoints/sft_ablate"))
    # text side
    p.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--encoder-kind", choices=("token", "embed"), default="embed")
    p.add_argument("--pool", choices=("cls", "mean", "last"), default="cls",
                   help="pooled vector for embed/FiLM: cls (BERT/bge), mean (MiniLM/mpnet), last (decoder LMs)")
    p.add_argument("--max-text", type=int, default=128)
    p.add_argument("--emb-dim", type=int, default=0,
                   help="embed mode: L2-normalise, Matryoshka-truncate to this dim, renormalise (0 = raw pooled)")
    p.add_argument("--label-emb", default="",
                   help="precomputed LLM-label embeddings (src/precompute_label_emb.py); metadata labels are encoded online + cached")
    p.add_argument("--n-llm-labels", type=int, default=5, help="LLM style labels per game (terse_1..long)")
    p.add_argument("--labelers", default="", help="comma list of index `labeler` values kept in train; empty = all")
    p.add_argument("--eval-labelers", default="", help="same for val/test")
    p.add_argument("--text-prefix", default="", help="prepended to every online-encoded text (instruct embedders); \\n = newline")
    p.add_argument("--freeze-dummy", action="store_true", help="pin the no-text cross-attn term to its starting value")
    p.add_argument("--player-epochs", type=int, default=1, help="exact schedule: passes over the player corpus")
    p.add_argument("--pure", action="store_true", help="drop label sets with any contradicted band (label_correct)")
    p.add_argument("--max-games", type=int, default=0, help="stop after this many games (all ranks); 0 = off")
    p.add_argument("--meta-labels", default=",".join(meta_templates.ATTRS),
                   help="synthesized metadata labels, one per attribute (src/meta_templates.py); empty = none")
    p.add_argument("--player-data", default="", help="top-player corpus (build_player_corpus.py); empty = off")
    p.add_argument("--player-labels", type=int, default=1, help="distinct 'play like {p}' wordings per player game")
    p.add_argument("--player-weight", type=float, default=1.0, help="CE weight of player labels")
    p.add_argument("--player-share", type=float, default=1 / 31, help="mean share of games from the player corpus")
    p.add_argument("--player-max-games", type=int, default=0, help="cap player games (tests)")
    p.add_argument("--planned-games", type=int, default=0, help="normal games in the run; player schedule follows their count (0 = time)")
    p.add_argument("--player-peak", type=float, default=10.0, help="end-of-run share / flat share (ramp over last 20%%)")
    p.add_argument("--meta-weight", type=float, default=0.5, help="CE weight of each metadata label vs 1.0 for LLM labels")
    p.add_argument("--label-weights", default="", help="explicit per-slot weights (overrides --meta-weight)")
    p.add_argument("--label-weights-end", default="", help="curriculum: weights at the end of the run (linear from --label-weights)")
    # conditioning / what trains
    p.add_argument("--cond", choices=("xattn", "film", "both"), default="xattn")
    p.add_argument("--cond-layers", type=int, default=0, help="condition only the top K layers (0 = all)")
    p.add_argument("--unfreeze-last", type=int, default=12, help="train self-attn/MLP of the top N layers (+final norm/head)")
    p.add_argument("--train-head", action="store_true")
    p.add_argument("--band-mode", choices=("expand", "accumulate"), default="expand")
    p.add_argument("--labels-per-step", type=int, default=0,
                   help="expand mode: random subset of k label slots per step (0 = all labels every step)")
    p.add_argument("--cfg-dropout", type=float, default=0.0, help="per-row prob of zeroing the text path")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-6, help="unfrozen pretrained decoder weights")
    p.add_argument("--cond-lr", type=float, default=1e-4, help="projection, gates, FiLM, cross-attention")
    p.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--final-lr-frac", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--games", type=int, default=512, help="max games per step (x n_labels label-rows)")
    p.add_argument("--token-budget", type=int, default=49152,
                   help="max padded label-row tokens per step (games x max_len x n_labels); bounds memory")
    p.add_argument("--illegal-penalty", type=float, default=0.5, help="same as pretraining --legal-penalty-weight")
    p.add_argument("--max-rows", type=int, default=512)
    p.add_argument("--subset-train", type=int, default=0, help="train on a fixed random subset of N train records")
    p.add_argument("--subset-eval", type=int, default=0, help="fixed random subset of N val and N test records")
    p.add_argument("--dump-subset", default="", help="write the subset record ids (.npy) and exit")
    # budget / eval
    p.add_argument("--time-budget", type=float, default=600)
    p.add_argument("--warmup-steps", type=int, default=5, help="untimed compile/cache steps")
    p.add_argument("--max-steps", type=int, default=10**9)
    p.add_argument("--eval-games", type=int, default=4000)
    p.add_argument("--eval-games-per-batch", type=int, default=32)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--eval-baseline", action="store_true", help="also eval step-0 (== pretrained policy)")
    p.add_argument("--eval-test", action="store_true", help="final report only; never for selection")
    p.add_argument("--compile", action="store_true", help="torch.compile norms + SwiGLU blocks")
    p.add_argument("--no-flash", action="store_true", help="dense causal SDPA instead of FA3 varlen")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--val-every", type=int, default=0, help="val CE every N steps (0 = only at the end)")
    p.add_argument("--val-games", type=int, default=1000, help="games per periodic val (same fixed subset each time)")
    p.add_argument("--plan", default="", help="dir with plan.npz from build_sft_plan.py (overrides sampling/schedule)")
    p.add_argument("--init-from", default="", help="SFT model (.pt from --save) to start from / evaluate")
    p.add_argument("--ckpt-every", type=float, default=0, help="snapshot every N train seconds (0 = off)")
    p.add_argument("--resume", action="store_true", help="resume from <save-dir>/<trial>.snap.pt if present")
    run(p.parse_args())


if __name__ == "__main__":
    main()
