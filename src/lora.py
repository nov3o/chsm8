"""LoRA for the chess decoder, added in place on existing nn.Linear modules.

Each targeted Linear keeps its own `weight`/`bias` (so pretrained state-dict keys are unchanged)
and gains `lora_A` (r x in) and `lora_B` (out x r, zero-init) parameters:
    y = x W^T + b + (alpha / r) * (x A^T) B^T
B = 0 at init, so the adapted model starts exactly at the base model. With apply_lora(freeze=True)
everything except the LoRA parameters is frozen.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = {
    "attn": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def _lora_forward(self, x):
    y = F.linear(x, self.weight, self.bias)
    return y + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.lora_scale


def apply_lora(model: nn.Module, rank: int, alpha: float, targets: str = "attn,mlp", freeze: bool = True) -> int:
    names = {n for t in targets.split(",") for n in TARGETS[t.strip()]}
    n_added = 0
    for mod_name, mod in model.named_modules():
        leaf = mod_name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Linear) and leaf in names and not hasattr(mod, "lora_A"):
            dev = mod.weight.device
            mod.lora_A = nn.Parameter(torch.empty(rank, mod.in_features, device=dev))
            nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))
            mod.lora_B = nn.Parameter(torch.zeros(mod.out_features, rank, device=dev))
            mod.lora_scale = alpha / rank
            mod.forward = _lora_forward.__get__(mod, nn.Linear)
            n_added += mod.lora_A.numel() + mod.lora_B.numel()
    if freeze:
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n
    return n_added
