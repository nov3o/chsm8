"""
Chess encoder-decoder model.

Encoder: NL text → context embeddings (two variants, see EncoderVariant).
Decoder: chess token sequence, cross-attends to encoder output at every layer.
Legal-move masking applied during generation.
"""

from dataclasses import dataclass
from typing import Literal

import _env_fixes  # noqa: F401 — must run before any `transformers` import, see that file's docstring
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoModel, AutoTokenizer

from tokenizer import BOS_ID, EOS_ID, ChessTokenizer
from factored_tokenizer import (
    KIND_BOS, KIND_MOVE, KIND_PAD, N_KINDS, N_PIECE_TYPES, N_PROMOTIONS, N_SQUARES,
)

EncoderVariant = Literal["embed", "token"]

# ------------------------------------------------------------------ #
# Real flash attention (FlashAttention-3), not just SDPA's automatic backend
# dispatch -- SDPA with a per-example (B,1,T,T) bool mask (our packed-rows
# block-causal case) does NOT reliably land on a flash kernel, only
# memory-efficient/math (see build_cu_seqlens below for the varlen path that
# actually does). Loaded lazily (needs CUDA + network on first call to fetch
# the prebuilt kernel binary, cached under ~/.cache/huggingface/hub after).
# `kernels-community/flash-attn3` (not `varunneal/flash-attention-3`, what
# external/autoresearch/train.py uses) -- varunneal's repo has no
# torch28+aarch64 build variant, i.e. no GH200 support; kernels-community's
# does (torch28-cxx11-cu129-aarch64-linux), verified numerically identical
# to SDPA causal attention (max abs diff 0.0, bf16) on this stack.
# ------------------------------------------------------------------ #
_FA3_MODULE = None


def _get_fa3():
    global _FA3_MODULE
    if _FA3_MODULE is None:
        from kernels import get_kernel
        _FA3_MODULE = get_kernel("kernels-community/flash-attn3").flash_attn_interface
    return _FA3_MODULE


@torch.compiler.disable
def _fa3_varlen_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
    """Opaque FA3 boundary for torch.compile.

    The GH200 FA3 wheel exposes a CUDA extension without a FakeTensor/meta
    implementation.  Letting Dynamo trace into it fails before execution;
    disabling just this call preserves FA3 itself and lets Inductor compile
    the surrounding Llama norm/RoPE/MLP graph.
    """
    out = _get_fa3().flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True,
    )
    return out[0] if isinstance(out, tuple) else out


def build_cu_seqlens(rows: torch.Tensor):
    """rows: (B,T,5) token rows (kind,a,b,c,d). Returns (cu_seqlens, max_seqlen)
    for flash_attn_varlen_func: cu_seqlens marks document boundaries (each
    KIND_BOS starts a new document, matching build_block_causal_mask's segment
    definition) in the FLATTENED (B*T,...) token space that varlen expects
    (q/k/v are reshaped (B,T,H,hd) -> (B*T,H,hd) row-major, so a row's start
    is index b*T there), so trailing ROW_PAD after a row's last document is
    folded into that document's segment -- identical semantics to the dense
    block-causal mask, just without ever materializing a (B,1,T,T) tensor.

    Fully vectorized (one nonzero() + one max().item(), no per-row Python
    loop/`.tolist()`) -- an earlier per-row-loop version forced a GPU->CPU
    sync every single row every step and was measured *slower* than the
    dense-mask SDPA path it was meant to beat (58k vs 119k tok/s at matched
    batch size), entirely from that overhead, not from the flash kernel
    itself."""
    B, T = rows.shape[0], rows.shape[1]
    device = rows.device
    flat_is_start = (rows[..., 0] == KIND_BOS).reshape(-1).clone()
    flat_is_start[::T] = True  # every row start is a hard boundary regardless of BOS (packing
    # invariant: a row's first content is always a fresh document's start -- forced explicitly
    # so an all-PAD row can't get silently merged into the previous row's segment)
    boundary_idx = torch.nonzero(flat_is_start, as_tuple=True)[0]
    cu_seqlens = torch.cat([
        boundary_idx.to(torch.int32),
        torch.full((1,), B * T, dtype=torch.int32, device=device),
    ])
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return cu_seqlens, max_seqlen


# ------------------------------------------------------------------ #
# Encoders
# ------------------------------------------------------------------ #

class EmbedEncoder(nn.Module):
    """
    Single-vector encoder: frozen sentence-embedding model → linear projection.
    Output shape: (B, 1, d_model)  — one context token per prompt.
    """
    def __init__(self, model_name: str, d_model: int):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        for p in self.lm.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.lm.config.hidden_size, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over sequence
        hidden = out.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)  # (B, H)
        return self.proj(pooled).unsqueeze(1)  # (B, 1, d_model)


class TokenEncoder(nn.Module):
    """
    Sequence encoder: frozen/LoRA LM → full sequence of hidden states projected to d_model.
    Output shape: (B, L_enc, d_model)
    """
    def __init__(self, model_name: str, d_model: int, lora: bool = False):
        super().__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        if not lora:
            for p in self.lm.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(self.lm.config.hidden_size, d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state)  # (B, L, d_model)


# ------------------------------------------------------------------ #
# Chess Decoder
# ------------------------------------------------------------------ #

@dataclass
class DecoderConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    use_cross_attention: bool = True  # False = decoder-only (no cross-attention) ablation variant
    grad_checkpoint: bool = False  # recompute each layer's activations during backward instead of storing them
    fast_dummy_cross_attention: bool = False  # exact shortcut for frozen length-1 all-zero context
    architecture: str = "classic"  # classic = learned-pos/LN/GELU; llama = RoPE/RMSNorm/SwiGLU
    llama_qk_norm: bool = True  # optional per-head Q/K RMSNorm; ablate for Llama throughput
    # What to do with the pretrained (frozen, dummy-context) cross-attention when no text is used
    # (llama only; see convert_xattn_state_dict):
    #   keep  = unchanged cross-attention path (needs use_cross_attention=True)
    #   drop  = remove it; its constant output out_proj(v_proj(0)) becomes a per-layer bias
    #   sattn = drop + a second causal self-attention in its place, behind a zero-init tanh gate
    xattn_mode: str = "keep"


class RMSNorm(nn.Module):
    """Bias-free RMSNorm used by Llama-family decoder blocks."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        # Explicit cast avoids bf16 activations being promoted back to fp32 by
        # the fp32 master copy of ``weight`` under autocast.
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight.to(dtype=x.dtype)


class SwiGLU(nn.Module):
    """Gated MLP: preserves ``d_ff`` as its intermediate width."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """Same math as nn.MultiheadAttention(batch_first=True), but built on
    F.scaled_dot_product_attention directly instead of that module's own
    (non-flash) internal kernel. Matters because our per-batch-row block
    causal mask (build_block_causal_mask) is a real per-example mask, not
    the plain is_causal=True case -- nn.MultiheadAttention's mask-present
    path materializes the full (B*n_heads, T, T) QK^T score matrix in
    float32 (4GiB+ already at B=512, n_heads=8, T=512), which both wastes
    memory and forces PyTorch off the fused/flash kernels. SDPA takes the
    mask as (B, 1, T, T) and broadcasts it across heads itself -- no
    repeat_interleave, no materialized per-head copy -- and dispatches to a
    memory-efficient or flash backend when the inputs allow it.

    attn_mask convention: bool, True = ALLOWED to attend (SDPA's convention
    -- opposite of nn.MultiheadAttention's, which used True = blocked).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool = False,
                 use_qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.use_rope = use_rope
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else None

    @staticmethod
    def rope_factors(position_ids: torch.Tensor, head_dim: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Build RoPE cos/sin once for a full decoder forward.

        Packed rows share their BOS-reset position map across every decoder
        layer.  Previously each layer built this same table twice (once for Q
        and again for K): 40 inv-frequency/trig launches for a 20-layer
        model.  Passing the table down is mathematically identical and keeps
        the fp32 angle calculation used by the original implementation.
        """
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head dimension, got {head_dim}")
        pos = position_ids.to(dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim
        ))
        angles = (pos.unsqueeze(-1) * inv_freq).to(dtype=dtype)
        return angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Standard RoPE over the last (head-dimension) axis of (B,T,H,D)."""
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None, cu_seqlens: torch.Tensor = None,
                max_seqlen: int = None, position_ids: torch.Tensor = None,
                rope_factors: tuple[torch.Tensor, torch.Tensor] = None) -> torch.Tensor:
        B, Tq, D = query.shape
        Tk = key.shape[1]
        q = self.q_proj(query).view(B, Tq, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, Tk, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, Tk, self.n_heads, self.head_dim)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            # RoPE is only meaningful for self-attention. Cross-attention has no shared
            # position axis between chess rows and the one-token dummy context.
            if Tq == Tk:
                if rope_factors is None:
                    if position_ids is None:
                        position_ids = torch.arange(Tq, device=q.device).expand(B, Tq)
                    rope_factors = self.rope_factors(position_ids, self.head_dim, q.dtype)
                q, k = self._apply_rope(q, *rope_factors), self._apply_rope(k, *rope_factors)
        if cu_seqlens is not None:
            # Real FlashAttention-3 varlen kernel, not SDPA -- see build_cu_seqlens: q/k/v are
            # (B,T,H,hd) here, flattened to (B*T,H,hd) row-major to match cu_seqlens' layout.
            # Only valid for self-attention (Tq==Tk, causal, no dropout support needed at our
            # dropout=0 winning config) -- callers only pass this for the self-attn path.
            assert Tq == Tk, "cu_seqlens/varlen flash path is self-attention only"
            qf = q.reshape(B * Tq, self.n_heads, self.head_dim)
            kf = k.reshape(B * Tk, self.n_heads, self.head_dim)
            vf = v.reshape(B * Tk, self.n_heads, self.head_dim)
            out = _fa3_varlen_eager(qf, kf, vf, cu_seqlens, max_seqlen)
            out = out.reshape(B, Tq, D)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            ).transpose(1, 2).contiguous().view(B, Tq, D)
        return self.out_proj(out)


class ChessDecoderLayer(nn.Module):
    def __init__(self, cfg: DecoderConfig, use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            self.n2 = nn.LayerNorm(cfg.d_model)
            self.fast_dummy_cross_attention = cfg.fast_dummy_cross_attention
            self._dummy_cross_attn_constant = None  # runtime cache; intentionally absent from state_dict
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model)
        )
        self.n1 = nn.LayerNorm(cfg.d_model)
        self.n3 = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, T, d)
        enc: torch.Tensor,         # (B, L_enc, d) — ignored if use_cross_attention=False (decoder-only)
        causal_mask: torch.Tensor, # (T, T) or (B,1,T,T) bool — ignored when cu_seqlens is given
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
    ) -> torch.Tensor:
        # causal self-attention -- real flash-attn varlen path when cu_seqlens is given (see
        # build_cu_seqlens/MultiHeadAttention.forward), dense-mask SDPA otherwise
        a = self.self_attn(x, x, x, attn_mask=causal_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        x = self.n1(x + self.drop(a))
        # cross-attention over encoder context (skipped entirely for the decoder-only ablation)
        if self.use_cross_attention:
            if self.fast_dummy_cross_attention:
                # With one all-zero encoder token, attention softmax has one element and is exactly 1.
                # q_proj/k_proj and SDPA therefore cannot affect the result: every query receives
                # out_proj(v_proj(0)). Cross-attention is frozen in this pretraining mode, so compute
                # that constant once in the active autocast dtype and reuse it without changing any
                # parameter or checkpoint key. This flag is only accepted with dropout=0 by the trainer.
                cached = self._dummy_cross_attn_constant
                if cached is None or cached.device != x.device:
                    with torch.no_grad():
                        cached = self.cross_attn.out_proj(self.cross_attn.v_proj(enc[:1])).detach()
                    self._dummy_cross_attn_constant = cached
                a = cached.expand(x.shape[0], x.shape[1], -1)
            else:
                a = self.cross_attn(x, enc, enc)
            x = self.n2(x + self.drop(a))
        # feed-forward
        x = self.n3(x + self.drop(self.ff(x)))
        return x


class LlamaChessDecoderLayer(nn.Module):
    """Llama-style pre-norm block, retaining the optional frozen dummy cross-attention.

    This is intentionally a whole-body alternative to ``ChessDecoderLayer`` rather
    than a one-off activation ablation: position handling, normalization, and MLP
    are all changed together.  The attention interface remains identical, so the
    packed FlashAttention path and factorized chess I/O are tested unchanged.
    """

    def __init__(self, cfg: DecoderConfig, use_cross_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, use_rope=True,
            use_qk_norm=cfg.llama_qk_norm,
        )
        self.n1 = RMSNorm(cfg.d_model)
        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout)
            self.n2 = RMSNorm(cfg.d_model)
            self.fast_dummy_cross_attention = cfg.fast_dummy_cross_attention
            self._dummy_cross_attn_constant = None
        self.xattn_mode = cfg.xattn_mode
        if cfg.xattn_mode in ("drop", "sattn"):
            assert not use_cross_attention, f"xattn_mode={cfg.xattn_mode} replaces cross-attention"
            self.xbias = nn.Parameter(torch.zeros(cfg.d_model))
        if cfg.xattn_mode == "sattn":
            self.sattn2 = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, use_rope=True,
                                             use_qk_norm=cfg.llama_qk_norm)
            self.n2 = RMSNorm(cfg.d_model)
            self.xgate = nn.Parameter(torch.zeros(1))  # tanh(0) = 0: step 0 == pretrained policy
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)
        self.n3 = RMSNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, enc: torch.Tensor, causal_mask: torch.Tensor,
        cu_seqlens: torch.Tensor = None, max_seqlen: int = None,
        position_ids: torch.Tensor = None, rope_factors: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        # Q, K, and V all consume the same pre-attention normalized state.
        # Computing RMSNorm separately for each was mathematically identical
        # but tripled this bandwidth-bound operation in every Llama layer.
        n1 = self.n1(x)
        x = x + self.drop(self.self_attn(
            n1, n1, n1, attn_mask=causal_mask,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, position_ids=position_ids,
            rope_factors=rope_factors,
        ))
        if self.use_cross_attention:
            if self.fast_dummy_cross_attention:
                cached = self._dummy_cross_attn_constant
                if cached is None or cached.device != x.device or cached.dtype != x.dtype:
                    with torch.no_grad():
                        cached = self.cross_attn.out_proj(self.cross_attn.v_proj(enc[:1])).detach()
                    self._dummy_cross_attn_constant = cached
                a = cached.expand(x.shape[0], x.shape[1], -1)
            else:
                n = self.n2(x)
                a = self.cross_attn(n, enc, enc)
            x = x + self.drop(a)
        elif self.xattn_mode in ("drop", "sattn"):
            x = x + self.xbias
            if self.xattn_mode == "sattn":
                n2 = self.n2(x)
                x = x + torch.tanh(self.xgate).to(x.dtype) * self.drop(self.sattn2(
                    n2, n2, n2, attn_mask=causal_mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                    position_ids=position_ids, rope_factors=rope_factors,
                ))
        x = x + self.drop(self.ff(self.n3(x)))
        return x


class ChessDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos   = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList(
            [ChessDecoderLayer(cfg, use_cross_attention=cfg.use_cross_attention) for _ in range(cfg.n_layers)]
        )
        self.norm  = nn.LayerNorm(cfg.d_model)
        self.head  = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.drop  = nn.Dropout(cfg.dropout)
        self.max_seq_len = cfg.max_seq_len
        self.use_cross_attention = cfg.use_cross_attention

    def forward(self, input_ids: torch.Tensor, enc: torch.Tensor = None) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.embed(input_ids) + self.pos(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        for layer in self.layers:
            x = layer(x, enc, mask)
        return self.head(self.norm(x))  # (B, T, vocab_size)


# ------------------------------------------------------------------ #
# Factorized action tokenization (see src/factored_tokenizer.py)
#
# Each move is (src_square, dst_square, piece_type, promotion), embedded/
# predicted via small per-component tables/heads instead of one big
# ~123k-entry vocab embedding/softmax. Rows are (kind, a, b, c, d); for
# kind == KIND_MOVE, (a,b,c,d) = (src,dst,piece,promo), else they're unused
# (zeros) and only the small `kind_emb`/`kind_head` path is exercised.
# ------------------------------------------------------------------ #


class FactoredEmbedding(nn.Module):
    """Replaces ChessDecoder.embed: sums 4 small component embeddings for
    move rows, uses a separate small table for special (PAD/BOS/EOS) rows."""

    def __init__(self, d_model: int):
        super().__init__()
        self.kind_emb = nn.Embedding(N_KINDS, d_model)   # only used for non-move rows
        self.src_emb = nn.Embedding(N_SQUARES, d_model)
        self.dst_emb = nn.Embedding(N_SQUARES, d_model)
        self.piece_emb = nn.Embedding(N_PIECE_TYPES, d_model)
        self.promo_emb = nn.Embedding(N_PROMOTIONS, d_model)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        # rows: (B, T, 5) long — columns (kind, src, dst, piece, promo)
        kind, src, dst, piece, promo = rows.unbind(-1)
        is_move = (kind == KIND_MOVE).unsqueeze(-1)
        move_emb = self.src_emb(src) + self.dst_emb(dst) + self.piece_emb(piece) + self.promo_emb(promo)
        special_emb = self.kind_emb(kind)
        return torch.where(is_move, move_emb, special_emb)


class FactoredHead(nn.Module):
    """Replaces ChessDecoder.head: 5 small linear heads instead of one big one."""

    def __init__(self, d_model: int):
        super().__init__()
        self.kind_head = nn.Linear(d_model, N_KINDS)
        self.src_head = nn.Linear(d_model, N_SQUARES)
        self.dst_head = nn.Linear(d_model, N_SQUARES)
        self.piece_head = nn.Linear(d_model, N_PIECE_TYPES)
        self.promo_head = nn.Linear(d_model, N_PROMOTIONS)

    def forward(self, x: torch.Tensor) -> dict:
        return {
            "kind": self.kind_head(x),
            "src": self.src_head(x),
            "dst": self.dst_head(x),
            "piece": self.piece_head(x),
            "promo": self.promo_head(x),
        }


def legal_component_masks_from_bits(own_occ: torch.Tensor, piece_mask_bits: torch.Tensor):
    """
    Unpacks the bit-packed per-position legality info produced by
    src/build_tier3.py's `legal.bin` (own_occ: 64-bit bitboard of the
    mover's own squares immediately before this row's move; piece_mask_bits:
    6-bit mask of piece types the mover has on the board) into boolean
    per-class masks for the src/dst/piece factored heads.

    own_occ, piece_mask_bits: (B, T) int64 tensors, bit-packed (own_occ's
    high bit is stored via two's-complement in a signed int64 -- correct
    for bitwise ops regardless of sign, never used for magnitude comparison).

    Returns (src_allowed, dst_allowed, piece_allowed): (B,T,64), (B,T,64),
    (B,T,6) boolean tensors, True = structurally legal. dst_allowed is the
    bitwise complement of own_occ (can't land on your own piece) -- same
    coarse semantics as src/factored_tokenizer.py's `legal_component_masks`,
    just precomputed at tier-3 build time instead of derived live via
    python-chess at generation time.
    """
    squares = torch.arange(N_SQUARES, device=own_occ.device)
    src_allowed = ((own_occ.unsqueeze(-1) >> squares) & 1).bool()      # (B,T,64)
    dst_allowed = ~src_allowed
    pieces = torch.arange(N_PIECE_TYPES, device=piece_mask_bits.device)
    piece_allowed = ((piece_mask_bits.unsqueeze(-1) >> pieces) & 1).bool()  # (B,T,6)
    return src_allowed, dst_allowed, piece_allowed


def factored_cross_entropy(
    logits: dict, targets: torch.Tensor,
    own_occ: torch.Tensor = None, piece_mask_bits: torch.Tensor = None,
    hard_mask: bool = True, illegal_penalty_weight: float = 0.0,
    move_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    targets: (B, T, 5) long, same row format as FactoredEmbedding's input.
    Loss = kind CE (all non-pad positions) + src/dst/piece/promo CE
    (only positions where the target kind is KIND_MOVE — enforced via
    ignore_index=-100 on the other positions, the standard torch pattern).

    own_occ/piece_mask_bits (both optional, both required together): (B, T)
    int64 bit-packed legality info aligned with `targets` (see
    src/train_pretrain.py's Tier3Dataset, src/build_tier3.py's `legal.bin`).
    When given, two independent (combinable) mechanisms penalize illegal
    src/dst/piece choices — two different "ways of penalizing illegal moves"
    to ablate against each other, not just an on/off switch:

    1. `hard_mask=True` (the original mechanism): masks src/dst/piece logits
       to structurally-legal classes before computing their cross-entropy —
       the model is never even asked to distinguish among illegal options,
       so it gets zero gradient signal about them at all.
    2. `illegal_penalty_weight > 0` (new, soft mechanism): leaves logits
       unmasked (the model still competes probability mass across ALL
       classes, legal or not, in the CE term), and adds an auxiliary loss
       term equal to the mean softmax probability mass landing on illegal
       classes at move positions. Unlike hard masking, this lets legal and
       illegal classes shape each other's logits during training (softmax is
       still over the full class set) while still pushing illegal-class
       probability toward zero — a strictly weaker, tunable-strength
       penalty rather than a structural constraint.

    Set hard_mask=False, illegal_penalty_weight=0 with own_occ given to get
    "no chess-legality penalty at all despite having the bits" (a no-op,
    matches --legal-mask-loss off) for a clean 3-way axis: off / hard-mask /
    soft-penalty. `kind` is never masked or penalized. `promo` legality
    (mandatory-promotion-on-back-rank) is derived the same way in both
    mechanisms, from `targets` alone (no tier-3 storage needed for this one).
    """
    import torch.nn.functional as F

    kind_t, src_t, dst_t, piece_t, promo_t = targets.unbind(-1)
    is_move = kind_t == KIND_MOVE

    src_allowed = dst_allowed = piece_allowed = promo_allowed = None
    if own_occ is not None:
        src_allowed, dst_allowed, piece_allowed = legal_component_masks_from_bits(own_occ, piece_mask_bits)
        PAWN_IDX = 0  # factored_tokenizer.PIECE_TYPES[0] == chess.PAWN
        must_promote = (piece_t == PAWN_IDX) & ((dst_t < 8) | (dst_t >= 56))  # (B,T)
        promo_allowed = torch.zeros(*piece_t.shape, N_PROMOTIONS, dtype=torch.bool, device=piece_t.device)
        promo_allowed[..., 0] = ~must_promote           # "no promotion" only legal if not forced to promote
        promo_allowed[..., 1:] = must_promote.unsqueeze(-1)  # any of N/B/R/Q legal iff forced to promote

    def reduce_ce(logit, target, active, weights=None):
        per = F.cross_entropy(logit.reshape(-1, logit.shape[-1]), target.reshape(-1),
                              ignore_index=-100, reduction="none").reshape_as(target)
        if weights is None:
            return per[active].mean()
        w = torch.where(active, weights, torch.zeros_like(weights))
        return (per * w).sum() / w.sum().clamp(min=1.0)

    kind_target = torch.where(kind_t == KIND_PAD, torch.full_like(kind_t, -100), kind_t)
    kind_active = kind_target.ne(-100)
    kind_weights = None if move_weights is None else torch.where(is_move, move_weights, torch.ones_like(move_weights))
    loss = reduce_ce(logits["kind"], kind_target, kind_active, kind_weights)
    n_move = is_move.sum().clamp(min=1)
    for name, n_classes, t, allowed in (
        ("src", N_SQUARES, src_t, src_allowed), ("dst", N_SQUARES, dst_t, dst_allowed),
        ("piece", N_PIECE_TYPES, piece_t, piece_allowed), ("promo", N_PROMOTIONS, promo_t, promo_allowed),
    ):
        logit = logits[name]
        if allowed is not None and illegal_penalty_weight > 0:
            # Soft penalty: mean illegal-class probability mass at move positions, added
            # un-masked (computed from the same logits the CE term below uses).
            probs = F.softmax(logit.float(), dim=-1)
            illegal_mass = (probs * (~allowed).float()).sum(-1)  # (B,T)
            illegal_mass = torch.where(is_move, illegal_mass, torch.zeros_like(illegal_mass))
            loss = loss + illegal_penalty_weight * (illegal_mass.sum() / n_move)
        if allowed is not None and hard_mask:
            # A large finite negative (not literal -inf) avoids a real gotcha: an all-masked row
            # (e.g. at non-move positions, where own_occ/piece_mask_bits are zero by construction)
            # would make log_softmax produce NaN, and even though ignore_index excludes it from the
            # loss VALUE, gradients flowing back through log_softmax at that row can still come out
            # NaN and contaminate the (shared-across-positions) head weights. torch.where below
            # additionally only applies the mask at is_move positions at all, belt-and-suspenders.
            neg = torch.finfo(logit.dtype).min
            masked_logit = logit.masked_fill(~allowed, neg)
            logit = torch.where(is_move.unsqueeze(-1), masked_logit, logit)
        t_masked = torch.where(is_move, t, torch.full_like(t, -100))
        loss = loss + reduce_ce(logit, t_masked, is_move, move_weights)
    return loss


class FactoredChessDecoder(nn.Module):
    """Same transformer body as ChessDecoder (reuses ChessDecoderLayer), but
    with the embedding matrix and output head fully replaced by the
    factorized (src,dst,piece,promo) versions above."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.embed = FactoredEmbedding(cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.architecture == "classic" else None
        layer_cls = ChessDecoderLayer if cfg.architecture == "classic" else LlamaChessDecoderLayer
        self.layers = nn.ModuleList(
            [layer_cls(cfg, use_cross_attention=cfg.use_cross_attention) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model) if cfg.architecture == "classic" else RMSNorm(cfg.d_model)
        self.head = FactoredHead(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.max_seq_len = cfg.max_seq_len
        self.use_cross_attention = cfg.use_cross_attention
        self.grad_checkpoint = cfg.grad_checkpoint

    def hidden_states(self, rows: torch.Tensor, enc: torch.Tensor = None, mask: torch.Tensor = None,
                      cu_seqlens: torch.Tensor = None, max_seqlen: int = None) -> torch.Tensor:
        """Return contextual factored-row states before the prediction heads.

        This is deliberately the same body used by :meth:`forward`.  Style
        retrieval pools these states for just the focal player's plies, while
        text-conditioned SFT still calls ``forward`` for the normal heads.
        Keeping this public avoids a second, subtly divergent chess encoder.
        """
        B, T, _ = rows.shape
        pos = torch.arange(T, device=rows.device)
        x = self.drop(self.embed(rows) + self.pos(pos)) if self.pos is not None else self.drop(self.embed(rows))
        # `cu_seqlens`/`max_seqlen` (see build_cu_seqlens) route self-attention through the real
        # FlashAttention-3 varlen kernel instead of a dense mask — the caller (train_pretrain.py)
        # only passes these for cuda + bf16/fp16, where flash-attn applies; falls back to `mask`
        # (dense block-causal, or the old whole-sequence causal default) otherwise, so every
        # existing caller (external/autoresearch/train_chess.py, fp32 configs, --smoke-test) is
        # unaffected.
        if cu_seqlens is None and mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        rope_position_ids = None
        rope_factors = None
        if self.pos is None:
            # A packed row contains several independent games.  Rotary phases must
            # restart at every game BOS exactly as the block-causal/varlen attention
            # boundary does; letting phase continue across games leaks arbitrary
            # packing position into the representation.
            starts = rows[..., 0].eq(KIND_BOS).clone()
            starts[:, 0] = True
            token_idx = torch.arange(T, device=rows.device).expand(B, T)
            last_start = torch.cummax(torch.where(starts, token_idx, torch.zeros_like(token_idx)), dim=1).values
            rope_position_ids = token_idx - last_start
            # Embedding outputs remain fp32 under autocast, while the Q/K
            # projections below are bf16.  The factors must use the latter:
            # float32 cos/sin would promote Q/K back to fp32 and FA3 rightly
            # rejects that dtype.
            rope_dtype = (torch.get_autocast_dtype("cuda")
                          if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype)
            rope_factors = MultiHeadAttention.rope_factors(
                rope_position_ids, self.layers[0].self_attn.head_dim, rope_dtype,
            )
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                if rope_position_ids is None:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, enc, mask, cu_seqlens, max_seqlen, use_reentrant=False,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        lambda x_: layer(x_, enc, mask, cu_seqlens, max_seqlen, rope_position_ids, rope_factors), x,
                        use_reentrant=False,
                    )
            else:
                if rope_position_ids is None:
                    x = layer(x, enc, mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                else:
                    x = layer(x, enc, mask, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                              position_ids=rope_position_ids, rope_factors=rope_factors)
        return self.norm(x)

    def forward(self, rows: torch.Tensor, enc: torch.Tensor = None, mask: torch.Tensor = None,
                cu_seqlens: torch.Tensor = None, max_seqlen: int = None) -> dict:
        return self.head(self.hidden_states(rows, enc=enc, mask=mask,
                                            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen))


def convert_xattn_state_dict(sd: dict, xattn_mode: str, n_layers: int, sattn_init: str = "xattn",
                             n_heads: int = 15) -> dict:
    """Map a pretrained (dummy-context cross-attention) Llama state dict onto xattn_mode drop/sattn.

    The pretrained cross-attention only ever saw one all-zero context token, so its output is the
    constant out_proj(v_proj(0)) = W_o b_v + b_o at every position. That constant becomes `xbias`,
    so the converted model computes exactly the pretrained function at step 0 (sattn's gate is 0).
    sattn_init: "xattn" copies the cross-attention's q/k/v/out projections into the new
    self-attention (its q/k were never trained: they had no effect), "self" copies the layer's own
    self-attention, "random" keeps the fresh init.
    """
    if xattn_mode == "keep":
        return sd
    if not any(".cross_attn." in k for k in sd):
        return sd  # already converted (e.g. a checkpoint from a drop/sattn run)
    out = {}
    for k, v in sd.items():
        if ".cross_attn." in k or ".n2." in k:
            continue
        out[k] = v
    for i in range(n_layers):
        pre = f"layers.{i}."
        w_o, b_o = sd[pre + "cross_attn.out_proj.weight"], sd[pre + "cross_attn.out_proj.bias"]
        b_v = sd[pre + "cross_attn.v_proj.bias"]
        # Reproduce pretraining's bf16 autocast arithmetic (fast_dummy_cross_attention computed
        # this constant under bf16), not the exact fp32 value: the pretrained weights were fitted
        # around the rounded constant, and the fp32 one measurably changes the loss.
        bf = torch.bfloat16
        const = (w_o.to(bf).float() @ b_v.to(bf).float() + b_o.to(bf).float()).to(bf)
        out[pre + "xbias"] = const.to(w_o.dtype)
        if xattn_mode == "sattn":
            out[pre + "xgate"] = torch.zeros(1, dtype=w_o.dtype)
            out[pre + "n2.weight"] = sd.get(pre + "n2.weight", torch.ones_like(b_o))
            src = {"xattn": "cross_attn", "self": "self_attn"}.get(sattn_init)
            if src:
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    for t in ("weight", "bias"):
                        out[f"{pre}sattn2.{proj}.{t}"] = sd[f"{pre}{src}.{proj}.{t}"]
            head_dim = w_o.shape[0] // n_heads
            for nrm in ("q_norm", "k_norm"):
                key = f"{pre}{src}.{nrm}.weight" if src else None
                # cross-attention had no Q/K norm: start the new one at identity
                out[f"{pre}sattn2.{nrm}.weight"] = sd[key] if key in sd else torch.ones(head_dim, dtype=w_o.dtype)
    return out


def make_dummy_context(batch_size: int, d_model: int, device=None, learned: bool = False) -> torch.Tensor:
    """Constant (B, 1, d_model) encoder context for the frozen-enc-dec
    pretraining phase — 'we pass some dummy encoder inputs'. Zeros by
    default; pass a learned nn.Parameter broadcast to this shape instead if
    `learned=True` is wired up by the caller (kept simple here)."""
    return torch.zeros(batch_size, 1, d_model, device=device)


def freeze_encoder_and_cross_attention(decoder: "ChessDecoder | FactoredChessDecoder", encoder: nn.Module = None) -> None:
    """Freezes encoder params (if given) and every layer's cross-attention
    weights, so only decoder self-attn/FF/embedding/head weights train —
    the 'frozen enc-dec, dummy context' pretraining ablation variant."""
    if encoder is not None:
        for p in encoder.parameters():
            p.requires_grad = False
    for layer in decoder.layers:
        if layer.use_cross_attention:
            for p in layer.cross_attn.parameters():
                p.requires_grad = False
            for p in layer.n2.parameters():
                p.requires_grad = False


# ------------------------------------------------------------------ #
# "Time-proven" decoder bodies, same factorized I/O bolted on
#
# Both classes swap in FactoredEmbedding/FactoredHead exactly like
# FactoredChessDecoder above, but replace the transformer body (self-attn
# stack) with a body from an established architecture instead of our
# hand-rolled ChessDecoderLayer:
#   - FactoredT5Decoder: HF T5Stack (relative-position attention, gated-relu
#     FF), trained from scratch — no pretrained weights make sense here
#     since T5's own vocab/embeddings are English-text-specific and we
#     never touch them (bypassed entirely via inputs_embeds).
#   - FactoredPretrainedDecoder: wraps a *pretrained* HF causal-LM body
#     (e.g. Qwen2.5-0.5B) via inputs_embeds -> last_hidden_state, bypassing
#     its own embedding/lm_head. "Mutilated" in the sense the user coined:
#     gut the vocab-specific input/output layers, keep the pretrained
#     transformer body. Decoder-only models like Qwen have no
#     cross-attention slot at all, so this variant is always the
#     decoder-only (no cross-attn) kind, never the frozen-enc-dec kind.
# ------------------------------------------------------------------ #

class FactoredT5Decoder(nn.Module):
    """T5Stack body (from scratch) + factorized embedding/head."""

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        from transformers import T5Config
        from transformers.models.t5.modeling_t5 import T5Stack

        self.embed = FactoredEmbedding(cfg.d_model)
        t5cfg = T5Config(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            num_layers=cfg.n_layers,
            num_heads=cfg.n_heads,
            is_decoder=True,
            is_encoder_decoder=False,
            use_cache=False,
            vocab_size=8,  # unused: we always call with inputs_embeds, never input_ids
            relative_attention_num_buckets=32,
            dropout_rate=cfg.dropout,
        )
        self.body = T5Stack(t5cfg)
        self.body.embed_tokens.weight.requires_grad = False  # dead weight: we always call with
        # inputs_embeds, never input_ids, so T5Stack's own embedding table is never touched
        self.head = FactoredHead(cfg.d_model)
        self.use_cross_attention = cfg.use_cross_attention

    def forward(self, rows: torch.Tensor, enc: torch.Tensor = None) -> dict:
        x = self.embed(rows)
        enc_kwargs = {"encoder_hidden_states": enc} if (self.use_cross_attention and enc is not None) else {}
        out = self.body(inputs_embeds=x, **enc_kwargs)
        return self.head(out.last_hidden_state)


class FactoredPretrainedDecoder(nn.Module):
    """Pretrained HF decoder-only body (e.g. Qwen2.5-0.5B) with its own
    embedding/lm_head bypassed, factorized embedding/head bolted on instead.
    No cross-attention slot exists on plain causal-LM bodies, so this is
    always a decoder-only (no cross-attention) variant."""

    def __init__(self, model_name: str, freeze_body: bool = False):
        super().__init__()
        from transformers import AutoModel

        self.body = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        d_model = self.body.config.hidden_size
        self.embed = FactoredEmbedding(d_model)
        self.head = FactoredHead(d_model)
        self.use_cross_attention = False
        self.body.embed_tokens.weight.requires_grad = False  # dead weight — always inputs_embeds
        if freeze_body:
            for p in self.body.parameters():
                p.requires_grad = False

    def forward(self, rows: torch.Tensor, enc: torch.Tensor = None) -> dict:
        x = self.embed(rows)
        out = self.body(inputs_embeds=x)
        return self.head(out.last_hidden_state)


class FactoredPretrainedEncoderDecoder(nn.Module):
    """Real pretrained encoder-decoder body (e.g. T5) — the actual 'bar to
    compare against', not a from-scratch architecture-only comparison like
    FactoredT5Decoder. Bypasses the shared embedding/lm_head entirely (both
    are English-vocab-specific); keeps the pretrained encoder AND decoder
    stacks, wired through our factorized embedding/head.

    The encoder (when used) gets a fixed constant (B,1,d_model) input, not
    real NL text (we have none in this phase) — same 'dummy context' idea as
    make_dummy_context, except here the encoder is a real pretrained stack
    actually processing it, not skipped.

    Freezing modes, matching the "pretrain: freeze encoder-proj matrices; a
    later full-size step trains everything, probably with heterogeneous LR"
    plan:
      - freeze_encoder=True (default False): freeze only the encoder body —
        matches the custom model's "frozen enc-dec + dummy context" variant
        (decoder trains, encoder doesn't). This is what "freeze matrices
        related to encoder proj" means, not freezing everything.
      - freeze_body=True: freezes BOTH encoder and decoder (only the tiny
        factorized embed/head train) — a stricter, separate ablation point
        ("does the pretrained body help at all with near-zero adaptation"),
        kept for backward compatibility with earlier runs.
      - use_encoder=False: skip the encoder entirely (not just freeze it) —
        pretrained decoder weights only, zero cross-attention, 100% of
        forward/backward compute spent on the decoder. This is the fair,
        apples-to-apples comparison against Qwen (which is 100% decoder by
        construction, unlike a kept-but-frozen encoder that still burns
        compute on a fixed dummy input for zero conditioning value — see
        RESULTS.md's methodology note on this).
    """

    def __init__(self, model_name: str, freeze_body: bool = False,
                 freeze_encoder: bool = False, use_encoder: bool = True):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM

        full = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
        self.use_encoder = use_encoder
        self.encoder = full.encoder if use_encoder else None
        self.decoder = full.decoder
        d_model = full.config.d_model if hasattr(full.config, "d_model") else full.config.hidden_size
        self.embed = FactoredEmbedding(d_model)
        self.head = FactoredHead(d_model)
        if use_encoder:
            self.register_buffer("dummy_input", torch.zeros(1, 1, d_model))  # fixed, not learned —
            # see make_dummy_context() for the same pattern on the custom model. A *learned* dummy
            # input here caused a real bug: gradients on this single small param blew up to ~1e28,
            # which squares to inf in the global grad-norm computation, silently zeroing every
            # update after clip_grad_norm_ (why the T5-pretrained baseline runs got stuck near-random).
        self.use_cross_attention = use_encoder

        # Both encoder's and decoder's own embed_tokens tables are dead weight here (we always
        # call with inputs_embeds, never input_ids) — freeze so they're not carried as live
        # optimizer state for nothing.
        self.decoder.embed_tokens.weight.requires_grad = False
        if use_encoder:
            self.encoder.embed_tokens.weight.requires_grad = False
        else:
            # decoder's cross-attention (EncDecAttention) blocks can never receive gradient when
            # the encoder is skipped entirely — freeze them explicitly rather than leaving them as
            # live-but-untouched optimizer state.
            for block in self.decoder.block:
                cross_attn_layer = block.layer[1]
                for p in cross_attn_layer.parameters():
                    p.requires_grad = False

        if freeze_body:
            for p in self.decoder.parameters():
                p.requires_grad = False
            if use_encoder:
                for p in self.encoder.parameters():
                    p.requires_grad = False
        elif freeze_encoder and use_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, rows: torch.Tensor, enc: torch.Tensor = None) -> dict:
        x = self.embed(rows)
        if not self.use_encoder:
            dec_out = self.decoder(inputs_embeds=x)  # T5 decoder cleanly skips cross-attn when
            # no encoder_hidden_states is given (confirmed) — zero encoder compute, unlike a
            # kept-but-frozen encoder which still costs a full forward/backward pass every step.
            return self.head(dec_out.last_hidden_state)
        B = rows.shape[0]
        enc_input = enc if enc is not None else self.dummy_input.expand(B, -1, -1)  # honor an
        # externally-supplied context (e.g. real NL embeddings later) if given, else fall back to
        # the fixed dummy — this used to unconditionally ignore `enc` entirely, a real bug caught
        # by a sensitivity check (varying `enc` produced byte-identical output).
        enc_out = self.encoder(inputs_embeds=enc_input)
        dec_out = self.decoder(inputs_embeds=x, encoder_hidden_states=enc_out.last_hidden_state)
        return self.head(dec_out.last_hidden_state)


# ------------------------------------------------------------------ #
# Full encoder-decoder
# ------------------------------------------------------------------ #

class ChessEncoderDecoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: ChessDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        enc_input_ids: torch.Tensor,
        enc_attention_mask: torch.Tensor,
        dec_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.encoder(enc_input_ids, enc_attention_mask)
        return self.decoder(dec_input_ids, ctx)

    @torch.no_grad()
    def generate(
        self,
        enc_input_ids: torch.Tensor,
        enc_attention_mask: torch.Tensor,
        chess_tokenizer: ChessTokenizer,
        mode: str = "plain",
        max_moves: int = 100,
        temperature: float = 1.0,
    ) -> list[str]:
        """Greedy generation with legal-move masking."""
        self.eval()
        ctx = self.encoder(enc_input_ids, enc_attention_mask)
        board = chess.Board()

        ids = [BOS_ID]
        # skip move-number / color tokens when tracking board — only SAN tokens advance the board
        san_tokens = {v: k for k, v in chess_tokenizer.tok2id.items()
                      if not k.startswith("<") and not k[0].isdigit()}

        for _ in range(max_moves * (2 if mode == "plain" else 3)):
            inp = torch.tensor([ids], device=enc_input_ids.device)
            logits = self.decoder(inp, ctx)[0, -1]  # (vocab_size,)

            # inject legal-move mask only when it's a move token's turn
            # (in numbered/colored modes, alternate with metadata tokens — simple heuristic)
            logits = logits + chess_tokenizer.legal_move_mask(board).to(logits.device)

            if temperature != 1.0:
                logits = logits / temperature
            next_id = logits.argmax().item()

            if next_id == EOS_ID or board.is_game_over():
                break

            ids.append(next_id)
            tok = chess_tokenizer.id2tok[next_id]
            if tok in board.legal_moves.__class__.__name__ or (
                not tok.startswith("<") and not tok[0].isdigit()
            ):
                try:
                    board.push_san(tok)
                except Exception:
                    break  # illegal — shouldn't happen with mask, but guard

        return chess_tokenizer.decode_ids(ids[1:])  # strip <bos>
