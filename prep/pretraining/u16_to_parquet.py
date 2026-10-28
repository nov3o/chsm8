#!/usr/bin/env python3
"""Binary tier-2b layout -> the published Parquet layout.

This is the script that produced ``data/`` in this repository, kept for
provenance and so the conversion can be reproduced or re-run on a re-filtered
tier-2b build. Input, per month:

    <in>/<YYYY-MM>/moves.u16      flat uint16 move codes (chsm8_format.py)
    <in>/<YYYY-MM>/meta.parquet   one row per game (move_off, n_plies, ...)

Output, per month (one file per --rows-per-file games, in original order):

    <out>/<YYYY-MM>/part-00000.parquet, part-00001.parquet, ...

The ``row_idx`` column is the game's row number in the month's tier-2b
``meta.parquet`` (0-based), so it is a stable join key: the post-training
datasets (chsm8-post-training-*) reference games by ``(month, row_idx)``.

    python u16_to_parquet.py data_u16 -o data --workers 64
    python u16_to_parquet.py data_u16 -o data --months 2013-01 2013-02
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chsm8_format import RESULT_STRINGS, decode_moves_to_strings

SCHEMA = pa.schema([
    ("month", pa.string()),
    ("row_idx", pa.int32()),
    ("white", pa.string()),
    ("black", pa.string()),
    ("white_elo", pa.int16()),
    ("black_elo", pa.int16()),
    ("white_title", pa.string()),
    ("black_title", pa.string()),
    ("result", pa.string()),
    ("termination", pa.string()),
    ("time_control", pa.string()),
    ("tc_base", pa.int32()),
    ("tc_increment", pa.int32()),
    ("eco", pa.string()),
    ("opening", pa.string()),
    ("n_plies", pa.int32()),
    ("moves", pa.large_string()),
])


def _plain(col):
    """Dictionary-encoded Arrow column -> plain string column, '' -> null."""
    if pa.types.is_dictionary(col.type):
        col = col.cast(pa.string())
    return pa.compute.if_else(pa.compute.equal(col, ""), pa.scalar(None, pa.string()), col)


def convert_month(args):
    src, dst, month, rows_per_file, row_group = args
    t0 = time.time()
    meta = pq.read_table(os.path.join(src, "meta.parquet")).combine_chunks()
    moves = np.memmap(os.path.join(src, "moves.u16"), dtype="<u2", mode="r")
    off = meta.column("move_off").to_numpy().astype(np.int64)
    npl = meta.column("n_plies").to_numpy().astype(np.int64)
    n = len(off)
    # tier-2b stores games contiguously in order; the vectorized decoder relies on it
    assert n == 0 or (off[0] == 0 and (off[1:] == off[:-1] + npl[:-1]).all()), f"{month}: non-contiguous moves"
    assert n == 0 or off[-1] + npl[-1] == moves.size, f"{month}: moves.u16 size mismatch"

    res = meta.column("result").to_numpy()
    tc_base = meta.column("tc_base").to_numpy().astype(np.int32)
    tc_inc = meta.column("tc_incr").to_numpy().astype(np.int32)
    cols = {
        "white": _plain(meta.column("white")), "black": _plain(meta.column("black")),
        "white_title": _plain(meta.column("white_title")), "black_title": _plain(meta.column("black_title")),
        "termination": _plain(meta.column("termination")),
        "eco": _plain(meta.column("eco")), "opening": _plain(meta.column("opening")),
    }
    os.makedirs(dst, exist_ok=True)
    n_files = 0
    for part, a in enumerate(range(0, n, rows_per_file)):
        z = min(n, a + rows_per_file)
        codes = moves[off[a]:off[z - 1] + npl[z - 1]]
        offsets, data = decode_moves_to_strings(codes, npl[a:z])
        moves_arr = pa.LargeStringArray.from_buffers(z - a, pa.py_buffer(offsets), pa.py_buffer(data))
        table = pa.table({
            "month": pa.array([month] * (z - a), pa.string()),
            "row_idx": pa.array(np.arange(a, z, dtype=np.int32)),
            "white": cols["white"][a:z], "black": cols["black"][a:z],
            "white_elo": pa.array(meta.column("white_elo").to_numpy()[a:z].astype(np.int16)),
            "black_elo": pa.array(meta.column("black_elo").to_numpy()[a:z].astype(np.int16)),
            "white_title": cols["white_title"][a:z], "black_title": cols["black_title"][a:z],
            "result": pa.array(np.asarray(RESULT_STRINGS, dtype=object)[res[a:z]], pa.string()),
            "termination": cols["termination"][a:z],
            "time_control": pa.array([f"{b}+{i}" for b, i in zip(tc_base[a:z], tc_inc[a:z])], pa.string()),
            "tc_base": pa.array(tc_base[a:z]), "tc_increment": pa.array(tc_inc[a:z]),
            "eco": cols["eco"][a:z], "opening": cols["opening"][a:z],
            "n_plies": pa.array(npl[a:z].astype(np.int32)),
            "moves": moves_arr,
        }, schema=SCHEMA)
        tmp = os.path.join(dst, f".part-{part:05d}.parquet.tmp")
        pq.write_table(table, tmp, compression="zstd", row_group_size=row_group,
                       use_dictionary=["month", "white_title", "black_title", "result",
                                       "termination", "time_control", "eco", "opening"])
        os.replace(tmp, os.path.join(dst, f"part-{part:05d}.parquet"))
        n_files += 1
    return month, n, int(npl.sum()), n_files, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="tier-2b directory with <YYYY-MM>/{moves.u16,meta.parquet}")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--months", nargs="*", help="subset of months (default: all)")
    ap.add_argument("--rows-per-file", type=int, default=2_000_000)
    ap.add_argument("--row-group", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    months = sorted(a.months or [m for m in os.listdir(a.indir)
                                  if os.path.isfile(os.path.join(a.indir, m, "meta.parquet"))])
    jobs = [(os.path.join(a.indir, m), os.path.join(a.outdir, m), m, a.rows_per_file, a.row_group)
            for m in months]
    jobs.sort(key=lambda j: os.path.getsize(os.path.join(j[0], "moves.u16")), reverse=True)
    games = plies = 0
    with Pool(min(a.workers, len(jobs))) as pool:
        for month, n, p, files, sec in pool.imap_unordered(convert_month, jobs):
            games += n; plies += p
            print(f"[{month}] {n:,} games, {p:,} plies, {files} files, {sec}s", flush=True)
    print(f"TOTAL {len(jobs)} months, {games:,} games, {plies:,} plies")


if __name__ == "__main__":
    main()
