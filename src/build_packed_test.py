#!/usr/bin/env python3
"""
Throughput-test data: a small pool of already-packed (seq_len+1, 5) rows,
filled with random-but-well-formed (kind, src, dst, piece, promo) tuples --
NOT real games, just valid-range values so the model's embeddings/heads
don't index out of range. Content is irrelevant here; this only exists to
measure how fast the training loop can go once per-step bin-packing is
removed (see PrePackedDataset in train_pretrain.py), before committing to
repacking the whole real tier-3 corpus.

One shared pool works for every batch-size test -- batch size is just how
many pool rows get sliced per step, not a property of the pool itself.

Usage:
    python src/build_packed_test.py --out data/packed_test --seq-len 512 --n-rows 50000
"""
import argparse
import os

import numpy as np

from factored_tokenizer import KIND_MOVE, N_SQUARES, N_PIECE_TYPES, N_PROMOTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/packed_test")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-rows", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    row_capacity = args.seq_len + 1

    packed = np.empty((args.n_rows, row_capacity, 5), dtype=np.int16)
    packed[..., 0] = KIND_MOVE
    packed[..., 1] = rng.integers(0, N_SQUARES, size=(args.n_rows, row_capacity), dtype=np.int16)
    packed[..., 2] = rng.integers(0, N_SQUARES, size=(args.n_rows, row_capacity), dtype=np.int16)
    packed[..., 3] = rng.integers(0, N_PIECE_TYPES, size=(args.n_rows, row_capacity), dtype=np.int16)
    packed[..., 4] = rng.integers(0, N_PROMOTIONS, size=(args.n_rows, row_capacity), dtype=np.int16)

    out_path = os.path.join(args.out, "packed.i16")
    packed.tofile(out_path)
    with open(os.path.join(args.out, "_SUCCESS"), "w") as f:
        f.write(f"n_rows={args.n_rows} seq_len={args.seq_len}\n")

    print(f"wrote {out_path}: {packed.nbytes/1e6:.1f}MB, {args.n_rows:,} rows x {row_capacity} tokens "
          f"= {args.n_rows*row_capacity:,} tokens")


if __name__ == "__main__":
    main()
