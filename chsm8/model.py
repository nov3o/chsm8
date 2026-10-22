"""The CHSM8 model: a Llama-style decoder over factorized chess moves, optionally conditioned on text.

Every position of a game is one row (kind, src, dst, piece, promo), see tokenizer.py. A row is embedded as the
sum of small per-field tables, and the output has one small linear head per field.

    rows (B, T, 5) -> embed -> 12 x [self-attention, (text cross-attention), SwiGLU] -> norm -> 5 heads

Text conditioning (SFT, DPO): the top `text_layers` layers cross-attend to the token states of a frozen text
encoder (EmbeddingGemma-300m). Each cross-attention output is scaled by tanh(gate), gates start at 0, so a fresh
text model computes exactly the pretrained model. Layers below the first text layer never see the text, so they run
once per game even when a game is paired with several descriptions (`repeat`).

`no_text` is a constant vector added in every layer. Pretraining kept a frozen cross-attention block fed one all-zero
context token, as a scaffold for later text conditioning; that block's output is the constant out_proj(v_proj(0)),
so we store just the constant. It is frozen during pretraining, SFT and DPO, and trained for the games-only models.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chsm8.tokenizer import KIND_BOS, KIND_MOVE, KIND_PAD

FIELDS = {"kind": 4, "src": 64, "dst": 64, "piece": 6, "promo": 5}  # classes per head


@dataclass
class Config:
    n_layers: int = 12
    d_model: int = 960
    n_heads: int = 15
    d_ff: int = 3520
    text_layers: int = 0    # top layers with text cross-attention (6 for chsm8-sft / chsm8, 0 otherwise)
    text_dim: int = 768     # width of the text encoder's token states (EmbeddingGemma-300m)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight.to(x.dtype)


def rope(positions, head_dim, dtype):
    """cos/sin tables (B, T, 1, head_dim/2) for interleaved rotary embeddings."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    angles = (positions.float().unsqueeze(-1) * inv_freq).to(dtype)
    return angles.cos().unsqueeze(-2), angles.sin().unsqueeze(-2)


def apply_rope(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


_FA3 = None


def flash_attn():
    """FlashAttention-3 varlen kernel from the `kernels` hub if it is installed and we are on a GPU, else None."""
    global _FA3
    if _FA3 is None:
        try:
            from kernels import get_kernel
            _FA3 = get_kernel("kernels-community/flash-attn3").flash_attn_interface
        except Exception:
            _FA3 = False
    return _FA3 or None


class Attention(nn.Module):
    def __init__(self, d, n_heads, qk_norm):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))
        self.n_heads, self.head_dim = n_heads, d // n_heads
        if qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)
        self.qk_norm = qk_norm

    def forward(self, x, ctx=None, ctx_mask=None, rope_cs=None, docs=None):
        """Causal self-attention over the documents of each row (ctx=None), or cross-attention to ctx."""
        B, T, D = x.shape
        kv = x if ctx is None else ctx
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(kv).view(B, kv.shape[1], self.n_heads, self.head_dim)
        v = self.v_proj(kv).view(B, kv.shape[1], self.n_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if ctx is not None:
            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                 attn_mask=ctx_mask[:, None, None, :])
            return self.out_proj(out.transpose(1, 2).reshape(B, T, D))
        q, k = apply_rope(q, *rope_cs), apply_rope(k, *rope_cs)
        fa3 = flash_attn() if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) else None
        if fa3 is not None:  # documents as variable-length sequences, no (T, T) mask
            cu, max_len = docs["cu_seqlens"], docs["max_len"]
            out = _fa3_varlen(fa3, q.reshape(B * T, self.n_heads, self.head_dim),
                              k.reshape(B * T, self.n_heads, self.head_dim),
                              v.reshape(B * T, self.n_heads, self.head_dim), cu, max_len)
            return self.out_proj(out.reshape(B, T, D))
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=docs["mask"])
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


@torch.compiler.disable
def _fa3_varlen(fa3, q, k, v, cu, max_len):
    out = fa3.flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True)
    return out[0] if isinstance(out, tuple) else out


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: Config, text: bool):
        super().__init__()
        self.n1, self.self_attn = RMSNorm(cfg.d_model), Attention(cfg.d_model, cfg.n_heads, qk_norm=True)
        self.no_text = nn.Parameter(torch.zeros(cfg.d_model))
        if text:
            self.n2, self.cross_attn = RMSNorm(cfg.d_model), Attention(cfg.d_model, cfg.n_heads, qk_norm=False)
        self.n3, self.ff = RMSNorm(cfg.d_model), SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, rope_cs, docs, text=None, gate=None):
        n1 = self.n1(x)
        x = x + self.self_attn(n1, rope_cs=rope_cs, docs=docs)
        x = x + self.no_text.to(x.dtype)
        if text is not None:
            a = self.cross_attn(self.n2(x), text["states"].to(x.dtype), text["mask"])
            if text.get("keep") is not None:  # description dropout: these rows get no text
                a = a * text["keep"].view(-1, 1, 1).to(a.dtype)
            x = x + torch.tanh(gate) * a
        return x + self.ff(self.n3(x))


class Embedding(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.kind_emb, self.src_emb, self.dst_emb = nn.Embedding(4, d), nn.Embedding(64, d), nn.Embedding(64, d)
        self.piece_emb, self.promo_emb = nn.Embedding(6, d), nn.Embedding(5, d)

    def forward(self, rows):
        kind, src, dst, piece, promo = rows.unbind(-1)
        move = self.src_emb(src) + self.dst_emb(dst) + self.piece_emb(piece) + self.promo_emb(promo)
        return torch.where((kind == KIND_MOVE).unsqueeze(-1), move, self.kind_emb(kind))


def documents(rows):
    """Each BOS starts a new game. Returns per-position RoPE positions (reset at every BOS) and what attention
    needs to keep games apart: cu_seqlens for the flash kernel, a block-causal mask for SDPA."""
    B, T = rows.shape[:2]
    start = rows[..., 0].eq(KIND_BOS)
    start[:, 0] = True  # every row starts a new document (trailing PAD joins the row's last game)
    idx = torch.arange(T, device=rows.device).expand(B, T)
    positions = idx - torch.cummax(torch.where(start, idx, torch.zeros_like(idx)), dim=1).values
    doc = start.cumsum(1)
    out = {"positions": positions}
    if rows.is_cuda:
        flat = torch.nonzero(start.reshape(-1), as_tuple=True)[0].int()
        out["cu_seqlens"] = torch.cat([flat, flat.new_tensor([B * T])])
        out["max_len"] = int((out["cu_seqlens"][1:] - out["cu_seqlens"][:-1]).max())
    causal = torch.ones(T, T, dtype=torch.bool, device=rows.device).tril()
    out["mask"] = (causal & (doc[:, :, None] == doc[:, None, :])).unsqueeze(1)  # (B, 1, T, T); only used by SDPA
    return out


class CHSM8(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.d_model)
        first_text = cfg.n_layers - cfg.text_layers
        self.layers = nn.ModuleList(Layer(cfg, text=i >= first_text) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.ModuleDict({k: nn.Linear(cfg.d_model, n) for k, n in FIELDS.items()})
        if cfg.text_layers:
            self.text_proj = nn.Sequential(nn.Linear(cfg.text_dim, cfg.d_model), RMSNorm(cfg.d_model))
            self.gate = nn.Parameter(torch.zeros(cfg.text_layers))

    def forward(self, rows, text=None, repeat=1):
        """rows (B, T, 5) long -> logits {field: (B*repeat, T, classes)}.

        text: None, or {"states": (B*repeat, L, text_dim), "mask": (B*repeat, L) bool, "keep": optional (B*repeat,)}
        from TextEncoder. repeat: descriptions per game; the text-free bottom layers run once per game and their
        output is repeated before the first text layer.
        """
        first_text = self.cfg.n_layers - self.cfg.text_layers
        x = self.embed(rows)
        docs = documents(rows)
        dtype = torch.get_autocast_dtype("cuda") if x.is_cuda and torch.is_autocast_enabled("cuda") else x.dtype
        rope_cs = rope(docs["positions"], self.layers[0].self_attn.head_dim, dtype)
        if text is not None:
            text = {**text, "states": self.text_proj(text["states"].float())}
        for i, layer in enumerate(self.layers):
            if i == first_text and repeat > 1:
                x, rows = x.repeat_interleave(repeat, 0), rows.repeat_interleave(repeat, 0)
                docs = documents(rows)
                rope_cs = tuple(t.repeat_interleave(repeat, 0) for t in rope_cs)
            use_text = text is not None and i >= first_text
            x = layer(x, rope_cs, docs, text if use_text else None, self.gate[i - first_text] if use_text else None)
        h = self.norm(x)
        return {k: head(h) for k, head in self.head.items()}

    @classmethod
    def from_pretrained(cls, repo="LegumMagister/chsm8", filename="model.pt", device="cpu"):
        """Load any released CHSM8 checkpoint from the Hugging Face Hub (see checkpoint.py)."""
        from chsm8.checkpoint import load
        return load(repo, filename, device)


def loss_fn(logits, targets, occ=None, pmask=None, penalty=0.0):
    """Training loss, the same for every stage.

    targets (B, T, 5): the row to predict at each position; PAD targets are ignored (padding, or moves of the side we
    do not train on). Loss = cross-entropy of `kind` over all non-PAD targets + the four field cross-entropies over
    MOVE targets. With occ/pmask (the mover's pieces before each target move, see tokenizer.legal_bits) and
    penalty > 0, the mean probability the src/dst/piece/promo heads put on impossible classes is added (x penalty):
    the model learns to avoid illegal choices without masking its softmax.
    """
    kind_t, src_t, dst_t, piece_t, promo_t = targets.unbind(-1)
    is_move = kind_t == KIND_MOVE
    kind_t = torch.where(kind_t == KIND_PAD, -100, kind_t)
    loss = F.cross_entropy(logits["kind"].flatten(0, 1).float(), kind_t.flatten(), ignore_index=-100)
    allowed = {}
    if occ is not None and penalty > 0:
        own = ((occ.unsqueeze(-1) >> torch.arange(64, device=occ.device)) & 1).bool()
        must_promote = (piece_t == 0) & ((dst_t < 8) | (dst_t >= 56))  # a pawn reaching the last rank
        allowed = {"src": own, "dst": ~own,
                   "piece": ((pmask.unsqueeze(-1) >> torch.arange(6, device=pmask.device)) & 1).bool(),
                   "promo": torch.cat([~must_promote.unsqueeze(-1), must_promote.unsqueeze(-1).expand(-1, -1, 4)], -1)}
    n_move = is_move.sum().clamp(min=1)
    for name, t in (("src", src_t), ("dst", dst_t), ("piece", piece_t), ("promo", promo_t)):
        logit = logits[name]
        if name in allowed:
            illegal = (F.softmax(logit.float(), -1) * (~allowed[name]).float()).sum(-1)
            loss = loss + penalty * torch.where(is_move, illegal, 0.0).sum() / n_move
        t = torch.where(is_move, t, -100)
        loss = loss + F.cross_entropy(logit.flatten(0, 1).float(), t.flatten(), ignore_index=-100)
    return loss


def move_logprobs(logits, targets):
    """(B, T) log p of each target row: the sum over the five heads (what play and DPO use)."""
    lp = 0.0
    for c, k in enumerate(FIELDS):
        lp = lp + F.log_softmax(logits[k].float(), -1).gather(-1, targets[..., c:c + 1]).squeeze(-1)
    return lp


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def init_weights(model: CHSM8, seed=0):
    """Fresh pretraining init: PyTorch defaults, plus the no_text constant a frozen, never-trained cross-attention
    block would add (out_proj(v_proj(0)) = W_out b_v + b_out with default nn.Linear init), as in our runs."""
    g = torch.Generator().manual_seed(seed)
    d = model.cfg.d_model
    with torch.no_grad():
        for layer in model.layers:
            bound = 1 / math.sqrt(d)
            w_out = torch.empty(d, d).uniform_(-bound, bound, generator=g)
            b_v, b_out = (torch.empty(d).uniform_(-bound, bound, generator=g) for _ in range(2))
            layer.no_text.copy_((w_out.bfloat16().float() @ b_v.bfloat16().float() + b_out.bfloat16().float()).bfloat16())
    return model
