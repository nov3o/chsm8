#!/usr/bin/env python3
"""Factorized rows (tier 3) -> fixed-length packed training records (tier 3p).

Greedy packing: games are appended, in on-disk order, into records of
``seq_len + 1`` rows (the +1 gives next-token input/target pairs of length
``seq_len``). A game is never split or truncated: when the next game does not
fit, the rest of the record is padded with PAD rows (kind=0, ignored by the
loss) and the game starts the next record. Games longer than ``seq_len + 1``
rows are skipped (at seq_len=512 that is <0.01% of games).

Each month's trailing ``--val-fraction`` of games becomes validation, the rest
training; both keep on-disk order, which keeps reads sequential (random reads
on Lustre-type filesystems were measured ~11x slower). Shuffle afterwards with
``shuffle.py`` if you want a sequential read to see a random mix of eras.

Output:

    <out>/{train,val}/packed_<YYYY-MM>_<part>.i16        (R, seq_len+1, 5) int16
    <out>/{train,val}/packed_<YYYY-MM>_<part>_occ.bin    (R, seq_len+1) uint64   [legal only]
    <out>/{train,val}/packed_<YYYY-MM>_<part>_pmask.bin  (R, seq_len+1) uint8    [legal only]
    <out>/{train,val}/_SUCCESS                           n_rows, n_games, seq_len, has_legal

    python pack.py data_3 -o data_3p --seq-len 512 --workers 64
    python pack.py data_3 -o data_3p_1024 --seq-len 1024 --no-legal --val-fraction 0.005
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from chsm8_format import KIND_PAD, LEGAL_DTYPE


def list_months(root):
    return sorted(m for m in os.listdir(root) if os.path.isfile(os.path.join(root, m, "_SUCCESS")))


def usable_games(root, month, cap):
    n = pq.read_table(os.path.join(root, month, "meta.parquet"), columns=["n_rows"])["n_rows"].to_numpy()
    return np.flatnonzero((n > 0) & (n <= cap))


def make_tasks(root, months, workers, val_fraction, cap):
    """Split each month into contiguous train parts, proportional to size."""
    sizes = {m: len(usable_games(root, m, cap)) for m in months}
    n_train = {m: sizes[m] - int(sizes[m] * val_fraction) for m in months}
    total = max(1, sum(n_train.values()))
    tasks = []
    for m in months:
        parts = max(1, round(workers * n_train[m] / total))
        for p in range(parts):
            a, z = n_train[m] * p // parts, n_train[m] * (p + 1) // parts
            if z > a:
                tasks.append((m, "train", p, a, z))
        if sizes[m] > n_train[m]:
            tasks.append((m, "val", 0, n_train[m], sizes[m]))
    return tasks


def pack_task(args):
    root, out, cap, with_legal, month, split, part, a, z = args
    meta = pq.read_table(os.path.join(root, month, "meta.parquet"), columns=["row_off", "n_rows"])
    off, nrows = meta["row_off"].to_numpy(), meta["n_rows"].to_numpy()
    games = usable_games(root, month, cap)[a:z]
    rows = np.memmap(os.path.join(root, month, "rows.i16"), dtype="<i2", mode="r").reshape(-1, 5)
    legal = np.memmap(os.path.join(root, month, "legal.bin"), dtype=LEGAL_DTYPE, mode="r") if with_legal else None
    d = Path(out) / split
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"packed_{month}_{part:03d}"
    fr = open(f"{stem}.i16", "wb", buffering=32 << 20)
    fo = open(f"{stem}_occ.bin", "wb", buffering=32 << 20) if with_legal else None
    fp = open(f"{stem}_pmask.bin", "wb", buffering=32 << 20) if with_legal else None
    r = np.zeros((cap, 5), dtype="<i2")
    occ = np.zeros(cap, dtype="<u8")
    pm = np.zeros(cap, dtype="u1")
    n_out = pos = 0

    def flush():
        nonlocal n_out, pos
        fr.write(r.tobytes())
        if with_legal:
            fo.write(occ.tobytes()); fp.write(pm.tobytes())
        r.fill(0); r[:, 0] = KIND_PAD; occ.fill(0); pm.fill(0)
        n_out += 1
        pos = 0

    for g in games:
        o, n = int(off[g]), int(nrows[g])
        if n > cap - pos:
            flush()
        r[pos:pos + n] = rows[o:o + n]
        if with_legal:
            seg = legal[o:o + n]
            occ[pos:pos + n], pm[pos:pos + n] = seg["occ"], seg["pmask"]
        pos += n
        if pos == cap:
            flush()
    if pos:
        flush()
    for f in (fr, fo, fp):
        if f:
            f.close()
    return split, month, part, n_out, len(games)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="tier-3 directory from to_factorized.py")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--seq-len", type=int, default=512, help="model context; records hold seq_len+1 rows")
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--no-legal", action="store_true", help="do not write occ/pmask sidecars")
    ap.add_argument("--months", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if os.path.exists(a.outdir) and os.listdir(a.outdir):
        raise SystemExit(f"refusing to write into non-empty {a.outdir}")
    months = a.months or list_months(a.indir)
    with_legal = not a.no_legal
    if with_legal and not all(os.path.exists(os.path.join(a.indir, m, "legal.bin")) for m in months):
        raise SystemExit("input has no legal.bin (built with --no-legal); pass --no-legal here too")
    cap = a.seq_len + 1
    tasks = make_tasks(a.indir, months, a.workers, a.val_fraction, cap)
    tasks.sort(key=lambda t: t[4] - t[3], reverse=True)
    totals = {"train": [0, 0], "val": [0, 0]}
    with Pool(max(1, min(a.workers, len(tasks)))) as pool:
        jobs = [(a.indir, a.outdir, cap, with_legal, *t) for t in tasks]
        for i, (split, month, part, n_out, n_games) in enumerate(pool.imap_unordered(pack_task, jobs), 1):
            totals[split][0] += n_out; totals[split][1] += n_games
            print(f"[{i}/{len(jobs)}] {split}/{month}/{part:03d}: {n_out:,} records, {n_games:,} games", flush=True)
    for split, (n_rows, n_games) in totals.items():
        Path(a.outdir, split).mkdir(parents=True, exist_ok=True)
        Path(a.outdir, split, "_SUCCESS").write_text(json.dumps({
            "n_rows": n_rows, "n_games": n_games, "seq_len": a.seq_len, "has_legal": with_legal,
            "positions": n_rows * a.seq_len}) + "\n")
        print(f"{split}: {n_rows:,} records ({n_rows * a.seq_len:,} positions), {n_games:,} games")


if __name__ == "__main__":
    main()
