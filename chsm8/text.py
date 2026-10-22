"""The frozen text encoder: a style description -> the token states the model cross-attends to."""
import torch
import torch.nn as nn

ENCODER = "google/embeddinggemma-300m"


class TextEncoder(nn.Module):
    """EmbeddingGemma-300m in bf16, never trained. Returns all its last-layer token states (B, L, 768) and the
    padding mask (B, L), for descriptions of up to 128 tokens."""

    def __init__(self, name=ENCODER, max_len=128, device="cpu"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(name, dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
        self.max_len = max_len

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward(self, texts):
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        return {"states": self.model(**enc).last_hidden_state, "mask": enc["attention_mask"].bool()}
