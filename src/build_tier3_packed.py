#!/usr/bin/env python3
"""
build_tier3_packed.py — offline greedy packing of the whole tier-3 corpus
into fixed (seq_len+1, 5) rows, once, so training-time batches become pure
memmap slicing instead of a per-step Python bin-packing search (see
PrePackedDataset in train_pretrain.py, REPORT.md — that per-step search,
not GPU compute, was the actual pretraining throughput bottleneck: ~4x
speedup measured once removed).

Two deliberate differences from Tier3Dataset.make_loader's runtime packer:

1. Plain greedy fill, not best-fit-over-a-1000-game-buffer. The buffer/
   best-fit search existed there to get good packing density from a
   streaming generator with bounded lookahead. Here we're doing one linear
   pass over an already-globally-shuffled game order, so the "next game in
   line" is already random — just greedily take it if it fits, else start
   a new row. Simpler, no buffer bookkeeping, and just as good in
   expectation given random game order.

2. NEVER truncates a game. When the next game doesn't fit the remaining
   space in a row, that space is padded (ROW_PAD, seq_len_cost, no
   truncation) and the game that didn't fit carries over whole to start the
   next row. Every game the model ever sees is complete. ROW_PAD positions
   cost zero loss (factored_cross_entropy ignore_index=KIND_PAD / -100 on
   the other four heads), so this only costs some wasted compute on pad
   positions, not correctness — measured overhead is roughly
   1/avg_games_per_row (~14-17% at seq_len=512, ~6-7 games/row).

data/3 now holds both rows.i16 and legal.bin per month (data/3legal was a
full duplicate of rows.i16 differing only by legal.bin's presence — folded
into data/3 and removed), so both are read from the same --rows-dir here
by default; the packed output combines them into one shard set too.

Parallelized by simple index-range chunking: shuffle the full game index
once (single seed, cheap — it's just permuting int arrays), split into
`--workers` contiguous slices, pack each slice independently in a worker
process (each opens its own memmaps — no shared state, no giant in-RAM
array). Output is one directory of numbered shards, not per-month like
tier-3 itself, since packing deliberately mixes games from different
months together (see Tier3Dataset's cross-month shuffle rationale).

Usage:
    python src/build_tier3_packed.py --rows-dir data/3 --legal-dir data/3 \
        --out data/3p --seq-len 512 --val-fraction 0.01 --workers 128
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import pyarrow.parquet as pq

from factored_tokenizer import KIND_PAD
from build_tier3 import LEGAL_DTYPE

ROW_PAD_TUPLE = (KIND_PAD, 0, 0, 0, 0)


def discover_months(rows_dir: str):
    return sorted(
        m for m in os.listdir(rows_dir)
        if os.path.isdir(os.path.join(rows_dir, m)) and os.path.exists(os.path.join(rows_dir, m, "_SUCCESS"))
    )


def build_game_index(rows_dir: str, legal_dir: str, seq_len: int, max_months: int = 0,
                      month_select: str = "first"):
    """Same contract as Tier3Dataset.__init__ (train_pretrain.py), standalone
    (no torch import) since this is a data-build tool, not a training one."""
    months = discover_months(rows_dir)
    assert months, f"no converted tier-3 months under {rows_dir}"
    if max_months > 0:
        # Lichess volume grew ~1500-3000x from 2013 to now (2013-01: 3.5K games vs 2026-06:
        # 10.8M) -- "first" months are not remotely representative of corpus density, use
        # "last" for a quick representative/high-density subset.
        months = months[:max_months] if month_select == "first" else months[-max_months:]
    has_legal = legal_dir is not None
    if has_legal:
        missing = [m for m in months if not os.path.exists(os.path.join(legal_dir, m, "legal.bin"))]
        assert not missing, f"--legal-dir given but legal.bin missing for {missing[:5]}..."

    month_ids, row_offs, n_rows_l = [], [], []
    month_boundaries = [0]  # month_boundaries[mi]:month_boundaries[mi+1] = that month's slice of the index
    for mi, m in enumerate(months):
        t = pq.read_table(os.path.join(rows_dir, m, "meta.parquet"), columns=["row_off", "n_rows"])
        off = t.column("row_off").to_numpy()
        n_rows = t.column("n_rows").to_numpy()
        keep = (n_rows > 0) & (n_rows <= seq_len + 1)
        month_ids.append(np.full(keep.sum(), mi, dtype=np.int32))
        row_offs.append(off[keep])
        n_rows_l.append(n_rows[keep])
        month_boundaries.append(month_boundaries[-1] + int(keep.sum()))

    month_ids = np.concatenate(month_ids)
    row_offs = np.concatenate(row_offs)
    n_rows = np.concatenate(n_rows_l)
    print(f"build_game_index: {len(months)} months, {len(month_ids):,} usable games, "
          f"legal={'present' if has_legal else 'absent'}", flush=True)
    return months, month_ids, row_offs, n_rows, has_legal, np.array(month_boundaries, dtype=np.int64)


def pack_chunk(args):
    # month_ids/row_offs/n_rows here are already this worker's slice, not the
    # full-corpus arrays: multiprocessing.Pool pickles every task tuple through
    # an IPC queue regardless of fork vs spawn, so passing the full ~1.068B-game
    # arrays into every one of 162 tasks meant pickling/transmitting the same
    # ~17GB blob 162 times over. That's what actually stalled the full-scale
    # run (workers sitting on futex_wait_queue, not doing I/O) after the
    # sequential-read fix alone was verified fast on smaller (16-month) subsets
    # where the full arrays were still small enough not to matter.
    (worker_id, out_dir, split, rows_dir, legal_dir, months, month_ids, row_offs, n_rows,
     seq_len, has_legal) = args
    row_capacity = seq_len + 1

    mmaps = {}
    legal_mmaps = {}

    def get_mm(mi):
        if mi not in mmaps:
            mmaps[mi] = np.memmap(os.path.join(rows_dir, months[mi], "rows.i16"), dtype="<i2", mode="r").reshape(-1, 5)
            if has_legal:
                legal_mmaps[mi] = np.memmap(os.path.join(legal_dir, months[mi], "legal.bin"), dtype=LEGAL_DTYPE, mode="r")
        return mmaps[mi], legal_mmaps.get(mi)

    # Stream each finished row straight to disk instead of buffering the whole
    # chunk in a Python list: for train (99x val's game count) that list would
    # hold ~7GB+ per worker, ~1.5TB aggregate across 200 workers -- more RAM
    # than the node has. This is exactly what killed the earlier attempt (no
    # traceback in the log = OOM kill, not a hang).
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    shard_id = str(worker_id)
    rows_f = open(os.path.join(split_dir, f"packed_{shard_id}.i16"), "wb", buffering=8 << 20)
    occ_f = open(os.path.join(split_dir, f"packed_{shard_id}_occ.bin"), "wb", buffering=8 << 20) if has_legal else None
    pmask_f = open(os.path.join(split_dir, f"packed_{shard_id}_pmask.bin"), "wb", buffering=8 << 20) if has_legal else None

    cur_rows = np.full((row_capacity, 5), ROW_PAD_TUPLE, dtype=np.int16)
    cur_occ = np.zeros(row_capacity, dtype=np.uint64) if has_legal else None
    cur_pmask = np.zeros(row_capacity, dtype=np.uint8) if has_legal else None
    p = 0
    n_rows_out = 0
    n_games_packed = 0

    def flush_row():
        nonlocal cur_rows, cur_occ, cur_pmask, p, n_rows_out
        rows_f.write(cur_rows.tobytes())
        if has_legal:
            occ_f.write(cur_occ.tobytes())
            pmask_f.write(cur_pmask.tobytes())
        n_rows_out += 1
        cur_rows = np.full((row_capacity, 5), ROW_PAD_TUPLE, dtype=np.int16)
        cur_occ = np.zeros(row_capacity, dtype=np.uint64) if has_legal else None
        cur_pmask = np.zeros(row_capacity, dtype=np.uint8) if has_legal else None
        p = 0

    for mi, off, n in zip(month_ids, row_offs, n_rows):
        mi = int(mi); off = int(off); n = int(n)
        mm, legal_mm = get_mm(mi)
        if n > row_capacity - p:
            flush_row()
        cur_rows[p:p + n] = mm[off:off + n]
        if has_legal:
            legal_slice = legal_mm[off:off + n]
            cur_occ[p:p + n] = legal_slice["occ"]
            cur_pmask[p:p + n] = legal_slice["pmask"]
        p += n
        n_games_packed += 1
        if p == row_capacity:
            flush_row()
    if p > 0:
        flush_row()

    rows_f.close()
    if has_legal:
        occ_f.close()
        pmask_f.close()
    return worker_id, n_rows_out, n_games_packed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-dir", type=str, default="data/3")
    ap.add_argument("--legal-dir", type=str, default="data/3",
                     help="set to '' to skip packing legality bitmasks (data/3 now holds "
                          "both rows.i16 and legal.bin -- data/3legal was consolidated into it)")
    ap.add_argument("--out", type=str, default="data/3p")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-months", type=int, default=0,
                     help="only pack N discovered months (0 = all) -- for producing a small "
                          "subset quickly, e.g. enough for a short training run")
    ap.add_argument("--month-select", choices=["first", "last"], default="last",
                     help="with --max-months, take the first or last N months. Default 'last': "
                          "Lichess volume grew ~1500-3000x from 2013 to now, so 'first' months "
                          "are not remotely representative of corpus density.")
    ap.add_argument("--shuffle-window-months", type=int, default=1,
                     help="shuffle games only within groups of this many consecutive months, "
                          "not globally across all 162 -- a full global shuffle turns every game "
                          "read into a scattered random-offset access across the whole 687GB+ "
                          "corpus, which is catastrophically slow on Lustre (small-random-read "
                          "latency, not bandwidth, dominates). Default 1 (pack within a single "
                          "month only, maximally sequential I/O -- one file at a time): training "
                          "reads random ROWS from across the whole packed corpus regardless of "
                          "which games got packed together within a row, so cross-era mixing at "
                          "the batch level already happens without needing it at pack time too; "
                          "this window only controls which games can co-occur inside one row, and "
                          "even a single month already packs densely (~91% real tokens measured).")
    ap.add_argument("--shard-by-month", action="store_true",
                    help="write one variable-size packed shard per calendar month, named packed_YYYY-MM, "
                         "instead of opaque equal-worker chunks")
    args = ap.parse_args()

    legal_dir = args.legal_dir or None
    months, month_ids, row_offs, n_rows, has_legal, month_boundaries = build_game_index(
        args.rows_dir, legal_dir, args.seq_len, args.max_months, args.month_select)
    n_games = len(month_ids)

    # No shuffling of read order at all, at any window size: measured directly (see
    # commit message / session notes), a single thread reading the exact same file
    # in on-disk (row_off) order hits ~162k games/sec; reading a random subset of
    # the *same* file drops to ~14k games/sec -- an 11x penalty from Lustre's
    # random-read cost alone, before any multi-worker contention on top of it. A
    # full global shuffle was already known to be worse ("10+ minutes to pack just
    # the 1% val split"), but per-month windowed shuffling is not a fix for that,
    # only a smaller instance of the same problem -- it still permutes every
    # game's read offset within the window, which is exactly what's expensive.
    # Games don't need to be read in random order for training to be random:
    # training reads random ROWS from across the whole packed corpus regardless of
    # which games got packed together at pack time (see make_dataloader), so val
    # is just each month's trailing val_fraction slice and train is the rest --
    # both are contiguous ranges in on-disk order, so every worker's chunk is a
    # small number of sequential byte ranges instead of a scatter of random ones.
    order = np.arange(n_games, dtype=np.int64)
    val_chunks, train_chunks = [], []
    n_month_groups = len(months) // args.shuffle_window_months + (1 if len(months) % args.shuffle_window_months else 0)
    for gi in range(n_month_groups):
        m_start = gi * args.shuffle_window_months
        m_end = min(m_start + args.shuffle_window_months, len(months))
        start, end = int(month_boundaries[m_start]), int(month_boundaries[m_end])
        window = order[start:end]
        n_val_w = max(0, int(len(window) * args.val_fraction))
        if n_val_w:
            val_chunks.append(window[-n_val_w:])
            train_chunks.append(window[:-n_val_w])
        else:
            train_chunks.append(window)
    val_order = np.concatenate(val_chunks)
    train_order = np.concatenate(train_chunks)

    os.makedirs(args.out, exist_ok=True)

    if args.shard_by_month and args.shuffle_window_months != 1:
        raise ValueError("--shard-by-month requires --shuffle-window-months=1")

    for split, order in (("val", val_order), ("train", train_order)):
        if args.shard_by_month:
            per_month = val_chunks if split == "val" else train_chunks
            chunks = [(months[mi], chunk) for mi, chunk in enumerate(per_month) if len(chunk)]
        else:
            n_workers = max(1, min(args.workers, len(order)))
            chunks = [(f"{wi:05d}", chunk)
                      for wi, chunk in enumerate(np.array_split(order, n_workers)) if len(chunk)]
        # Slice month_ids/row_offs/n_rows down to each worker's games BEFORE
        # building the task tuple -- see pack_chunk's docstring comment for why
        # this matters at full-corpus scale.
        tasks = [
            (shard_id, args.out, split, args.rows_dir, legal_dir, months,
             month_ids[chunk], row_offs[chunk], n_rows[chunk],
             args.seq_len, has_legal)
            for shard_id, chunk in chunks
        ]
        print(f"=== packing {split}: {len(order):,} games, {len(tasks)} workers ===", flush=True)
        total_rows = 0
        total_games = 0
        with mp.Pool(len(tasks)) as pool:
            for wi, n_rows_out, n_games_packed in pool.imap_unordered(pack_chunk, tasks):
                total_rows += n_rows_out
                total_games += n_games_packed
                print(f"[{split}] worker {wi}: {n_rows_out:,} rows from {n_games_packed:,} games", flush=True)
        with open(os.path.join(args.out, split, "_SUCCESS"), "w") as f:
            f.write(f"n_shards={len(tasks)} n_rows={total_rows} n_games={total_games} "
                     f"seq_len={args.seq_len} has_legal={has_legal}\n")
        print(f"=== {split} done: {total_rows:,} packed rows, {total_games:,} games ===", flush=True)

    print("[build_tier3_packed] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
