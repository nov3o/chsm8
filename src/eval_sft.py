#!/usr/bin/env python3
"""Play a style-conditioned SFT model (sft_ablate --save .pt) against Stockfish.

Same protocol as benchmark_play.py (legal argmax move, W/D/L, centipawn loss, PGN), with the
decoder conditioned on --prompt through the trained Conditioner. --prompt "" = no text
(pretrained dummy-context path), the unconditioned baseline of the same weights.
CE on val/test (matched / mismatched / zero labels, players): run
  sft_ablate.py --init-from <ckpt> --max-steps 0 --warmup-steps 0 --eval-test [--player-data ...]
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_play  # noqa: E402
import sft_ablate as S  # noqa: E402


def F_pad(t, T):
    """Pad dim 1 of (N, t, ...) to length T with zeros / False."""
    if t.shape[1] == T:
        return t
    return torch.cat([t, torch.zeros((t.shape[0], T - t.shape[1], *t.shape[2:]), dtype=t.dtype, device=t.device)], 1)


class SFTPolicy(torch.nn.Module):
    """guide_w != 1 (test-time guidance): logp = A + w * (B - A), B = this model with the prompt, A = reference: `guide_ref`
    checkpoint (default this model) with no prompt (guide_ref_prompt=False, classifier-free guidance) or the same prompt.
    w > 1 amplifies the prompt (or the fine-tune's change), 0 < w < 1 moves back toward the reference."""
    def __init__(self, ckpt, prompt, device, guide_w=1.0, guide_ref="", guide_ref_prompt=False, prompts=None):
        super().__init__()
        self.per_game = bool(prompts)  # prompts: one text per game (game i uses prompts[i % len]); overrides prompt
        self.guide_w = guide_w
        self.ref = None
        if guide_w != 1.0 and prompt:
            self.ref = SFTPolicy(guide_ref, prompt if guide_ref_prompt else "", device) if guide_ref else "self"
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = SimpleNamespace(**sd["args"])
        self.chess, self.text, self.cond, _, info = S.build(a, device)
        self.chess.load_state_dict(sd["chess"]); self.cond.load_state_dict(sd["cond"])
        for i, t in sd.get("frozen_dummy", {}).items():  # dpo_sft --freeze-dummy: pinned no-text cross-attn term
            self.chess.layers[int(i)].frozen_dummy = t.to(next(self.chess.parameters()).device)
        self.chess.eval(); self.cond.eval()
        self.n_shared, self.n_frozen = info["n_shared"], info["n_frozen"]
        self.c = None
        if prompts:
            with torch.no_grad():
                cs = [self.text(prompts[i:i + 256]) for i in range(0, len(prompts), 256)]  # chunks, padded to one length
                T = max(c[0].shape[1] for c in cs)
                ctx = torch.cat([F_pad(c[0], T) for c in cs]); mask = torch.cat([F_pad(c[1], T) for c in cs])
                pooled = torch.cat([c[2] for c in cs]) if cs[0][2] is not None else None
                self.call = self.cond.prepare(ctx, mask, pooled); self.n_prompts = len(prompts)
        elif prompt:
            with torch.no_grad():
                ctx, mask, pooled = self.text([prompt])
                self.c = self.cond.prepare(ctx, mask, pooled); self.c["gate"] = self.cond.gate

    @torch.no_grad()
    def forward(self, x, enc=None):
        B = x.shape[0]
        if self.per_game:  # enc = game indices of the batch rows -> each row gets its own prompt's context
            idx = torch.tensor([g % self.n_prompts for g in enc], device=x.device)
            c = {k: (v[idx] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == self.n_prompts else v)
                 for k, v in self.call.items()}
            c["gate"] = self.cond.gate
        else:
            c = None if self.c is None else {k: (v.expand(B, *v.shape[1:]) if k != "gate" and v is not None else v)
                                              for k, v in self.c.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = S.forward_shared(self.chess, x, self.n_shared, self.n_frozen)
            out = S.forward_top(self.chess, h, x, self.n_shared, c)
            if self.ref == "self":
                ref = S.forward_top(self.chess, h, x, self.n_shared, None)
        out = {k: v.float() for k, v in out.items()}
        if self.ref is None:
            return out
        ref = {k: v.float() for k, v in ref.items()} if self.ref == "self" else self.ref(x)
        F = torch.nn.functional
        return {k: F.log_softmax(ref[k], -1) + self.guide_w * (F.log_softmax(out[k], -1) - F.log_softmax(ref[k], -1)) for k in out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--prompts-file", default="", help="one prompt per line; game i is played with line i (overrides --prompt)")
    ap.add_argument("--guide-w", type=float, default=1.0, help="test-time guidance weight (1 = off)")
    ap.add_argument("--guide-ref", default="", help="reference checkpoint for guidance (default: the same model)")
    ap.add_argument("--guide-ref-prompt", action="store_true", help="reference also gets the prompt (default: no text)")
    a, rest = ap.parse_known_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        S.USE_FLASH = False  # no flash-attn kernel on CPU (as in sft_ablate.py)

    def load(path, dev):
        prompts = [l.rstrip("\n") for l in open(a.prompts_file) if l.strip()] if a.prompts_file else None
        pol = SFTPolicy(path, a.prompt, torch.device(dev), a.guide_w, a.guide_ref, a.guide_ref_prompt, prompts)
        cfg = SimpleNamespace(seq_len=a.seq_len, d_model=0, use_cross_attention=False)
        return pol, cfg, {"step": None, "prompt": a.prompt}

    benchmark_play.load_checkpoint = load
    sys.argv = [sys.argv[0]] + rest
    benchmark_play.main()


if __name__ == "__main__":
    main()
