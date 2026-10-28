#!/usr/bin/env python3
"""Minimal reader for packed records (pack.py / shuffle.py output).

Shows how CHSM8 consumes the data: memmap every shard, draw record indices,
and split each ``(seq_len+1, 5)`` record into input rows ``[:-1]`` and target
rows ``[1:]``. Targets whose kind is PAD are ignored by the loss. With the
legality sidecars, logits can be masked before the softmax:

    src   : squares not in occ[t]            -> -inf   (no own piece there)
    dst   : squares in occ[t]                -> -inf   (own piece already there)
    piece : piece types not in pmask[t]      -> -inf   (no such piece left)

where ``t`` indexes the *target* position (the move being predicted).

    python packed_dataset.py data_3p --split val --batch 4     # prints a sample batch

Only numpy is required; ``torch_batches`` wraps it for PyTorch if installed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from chsm8_format import KIND_MOVE, PIECE_NAMES


class PackedRecords:
    def __init__(self, root, split="train"):
        d = Path(root) / split
        self.info = json.loads((d / "_SUCCESS").read_text())
        cap = self.info["seq_len"] + 1
        self.rows, self.occ, self.pmask = [], [], []
        for p in sorted(d.glob("packed_*.i16")):
            self.rows.append(np.memmap(p, dtype="<i2", mode="r").reshape(-1, cap, 5))
            if self.info["has_legal"]:
                self.occ.append(np.memmap(p.with_name(p.stem + "_occ.bin"), dtype="<u8", mode="r").reshape(-1, cap))
                self.pmask.append(np.memmap(p.with_name(p.stem + "_pmask.bin"), dtype="u1", mode="r").reshape(-1, cap))
        self.bounds = np.cumsum([0] + [len(r) for r in self.rows])

    def __len__(self):
        return int(self.bounds[-1])

    def get(self, i):
        s = int(np.searchsorted(self.bounds, i, side="right") - 1)
        j = i - self.bounds[s]
        rec = {"rows": np.asarray(self.rows[s][j])}
        if self.occ:
            rec["occ"], rec["pmask"] = np.asarray(self.occ[s][j]), np.asarray(self.pmask[s][j])
        return rec

    def batches(self, batch_size, seed=0, shuffle=True):
        """Yield dicts of numpy arrays: x (B,T,5), y (B,T,5) [, occ (B,T), pmask (B,T) for targets]."""
        order = np.random.default_rng(seed).permutation(len(self)) if shuffle else np.arange(len(self))
        for lo in range(0, len(order) - batch_size + 1, batch_size):
            recs = [self.get(int(i)) for i in order[lo:lo + batch_size]]
            rows = np.stack([r["rows"] for r in recs])
            out = {"x": rows[:, :-1], "y": rows[:, 1:]}
            if self.occ:
                out["occ"] = np.stack([r["occ"] for r in recs])[:, 1:]
                out["pmask"] = np.stack([r["pmask"] for r in recs])[:, 1:]
            yield out


def legal_masks(occ, pmask):
    """Boolean masks (True = allowed) for the src/dst/piece heads, shape (..., 64/64/6)."""
    bits = ((occ[..., None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).astype(bool)
    piece = ((pmask[..., None] >> np.arange(6, dtype=np.uint8)) & 1).astype(bool)
    return {"src": bits, "dst": ~bits, "piece": piece}


def torch_batches(records, batch_size, device="cpu", seed=0):
    import torch
    for b in records.batches(batch_size, seed=seed):
        yield {k: torch.from_numpy(v.astype(np.int64 if v.dtype != np.uint64 else np.int64)).to(device)
               for k, v in b.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    ds = PackedRecords(a.root, a.split)
    print(f"{a.split}: {len(ds):,} records, info={ds.info}")
    b = next(ds.batches(a.batch))
    print({k: v.shape for k, v in b.items()})
    y = b["y"][0]
    mv = np.flatnonzero(y[:, 0] == KIND_MOVE)[:5]
    for t in mv:
        _, s, d, p, q = y[t]
        line = f"target {t}: src={s} dst={d} piece={PIECE_NAMES[p]} promo={q}"
        if "occ" in b:
            m = legal_masks(b["occ"][0, t], b["pmask"][0, t])
            assert m["src"][s] and m["dst"][d] and m["piece"][p], "target move violates its own legality mask"
            line += f"  legal src squares={int(m['src'].sum())}"
        print(line)
