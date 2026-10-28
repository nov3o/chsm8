#!/usr/bin/env python3
"""Published Parquet layout -> compact binary tier-2b layout.

The binary layout is ~2 bytes per ply and is what the rest of the training
toolkit (``to_factorized.py`` -> ``pack.py``) reads, via numpy memmaps.

Output, per month:

    <out>/<YYYY-MM>/moves.u16      flat little-endian uint16 move codes
    <out>/<YYYY-MM>/meta.parquet   one row per game: move_off (uint64, in plies),
                                   n_plies (uint16) and all metadata columns
    <out>/<YYYY-MM>/_SUCCESS

Optional filters let you build a narrower corpus directly:

    python parquet_to_u16.py data -o data_u16 --workers 32
    python parquet_to_u16.py data -o data_u16_2200 --min-elo 2200 --months 2025-01 2025-02
    python parquet_to_u16.py data -o data_u16_rapid --min-tc-seconds 480   # base + 40*increment

The conversion is exact: on the unfiltered corpus the output ``moves.u16`` is
byte-identical to the original tier-2b file (see verify_roundtrip.py).
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from chsm8_format import RESULT_CODES, uci_to_code

# 4-char UCI -> code lookup keyed by the 4 ASCII bytes; promotions add 1..4 << 12
_PROMO = {ord("n"): 1, ord("b"): 2, ord("r"): 3, ord("q"): 4}


def strings_to_codes(moves: pa.Array):
    """Vectorized: Arrow string array of space-separated UCI -> (codes, n_plies)."""
    moves = moves.cast(pa.large_string())
    offsets = np.frombuffer(moves.buffers()[1], dtype=np.int64)[moves.offset:moves.offset + len(moves) + 1]
    data = np.frombuffer(moves.buffers()[2], dtype=np.uint8)[offsets[0]:offsets[-1]]
    offsets = offsets - offsets[0]
    # Games are stored back to back with no separator, so a token starts after a
    # space *or* at a game's first byte, and ends at the next space *or* game end.
    is_space = data == 32
    game_start = np.zeros(data.size, dtype=bool)
    game_start[offsets[:-1][offsets[:-1] < data.size]] = True
    after_space = np.concatenate(([False], is_space[:-1]))
    starts = np.flatnonzero((game_start | after_space) & ~is_space)
    bounds = np.union1d(np.flatnonzero(is_space), offsets[1:])
    tok_end = bounds[np.searchsorted(bounds, starts, side="right")]
    tok_len = tok_end - starts
    if not np.isin(tok_len, (4, 5)).all():
        bad = starts[~np.isin(tok_len, (4, 5))][0]
        raise ValueError(f"malformed UCI token at byte {bad}: {bytes(data[bad:bad + 8])!r}")
    f = (data[starts] - 97) + 8 * (data[starts + 1] - 49)
    t = (data[starts + 2] - 97) + 8 * (data[starts + 3] - 49)
    p = np.zeros(starts.size, dtype=np.int64)
    five = tok_len == 5
    p[five] = np.vectorize(_PROMO.get)(data[starts[five] + 4]) if five.any() else 0
    codes = (f.astype(np.int64) | (t.astype(np.int64) << 6) | (p << 12)).astype("<u2")
    game_of_tok = np.searchsorted(offsets, starts, side="right") - 1
    n_plies = np.bincount(game_of_tok, minlength=len(moves))
    return codes, n_plies


def convert_month(args):
    src_dir, dst_dir, min_elo, min_tc = args
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(src_dir, "*.parquet")))
    os.makedirs(dst_dir, exist_ok=True)
    out = open(os.path.join(dst_dir, "moves.u16"), "wb", buffering=16 << 20)
    metas, cursor = [], 0
    for f in files:
        t = pq.read_table(f)
        keep = None
        if min_elo:
            keep = pc.greater_equal(pc.min_element_wise(t["white_elo"], t["black_elo"]), min_elo)
        if min_tc:
            k2 = pc.greater_equal(pc.add(t["tc_base"], pc.multiply(t["tc_increment"], 40)), min_tc)
            keep = k2 if keep is None else pc.and_(keep, k2)
        if keep is not None:
            t = t.filter(keep)
        if t.num_rows == 0:
            continue
        codes, npl = strings_to_codes(t["moves"].combine_chunks())
        out.write(codes.tobytes())
        move_off = cursor + np.concatenate(([0], np.cumsum(npl)[:-1]))
        cursor += int(npl.sum())
        metas.append(pa.table({
            "move_off": pa.array(move_off.astype(np.uint64)),
            "n_plies": pa.array(npl.astype(np.uint16)),
            "white_elo": t["white_elo"].cast(pa.uint16()),
            "black_elo": t["black_elo"].cast(pa.uint16()),
            "result": pa.array(np.array([RESULT_CODES[r] for r in t["result"].to_pylist()], dtype=np.uint8)),
            "termination": t["termination"], "tc_base": t["tc_base"].cast(pa.uint16()),
            "tc_incr": t["tc_increment"].cast(pa.uint16()), "eco": t["eco"], "opening": t["opening"],
            "white": t["white"], "black": t["black"],
            "white_title": t["white_title"], "black_title": t["black_title"],
            "source_row_idx": t["row_idx"],
        }))
    out.close()
    meta = pa.concat_tables(metas) if metas else None
    if meta is not None:
        pq.write_table(meta, os.path.join(dst_dir, "meta.parquet"), compression="zstd")
    with open(os.path.join(dst_dir, "_SUCCESS"), "w") as fh:
        fh.write(f"games={0 if meta is None else meta.num_rows} plies={cursor}\n")
    return os.path.basename(dst_dir), 0 if meta is None else meta.num_rows, cursor, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="Parquet dataset root with <YYYY-MM>/part-*.parquet")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--months", nargs="*", help="subset of months (default: all)")
    ap.add_argument("--min-elo", type=int, default=0, help="keep games with min(white_elo, black_elo) >= this")
    ap.add_argument("--min-tc-seconds", type=int, default=0, help="keep games with base + 40*increment >= this")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    months = sorted(a.months or [m for m in os.listdir(a.indir) if glob.glob(os.path.join(a.indir, m, "*.parquet"))])
    jobs = [(os.path.join(a.indir, m), os.path.join(a.outdir, m), a.min_elo, a.min_tc_seconds) for m in months]
    games = plies = 0
    with Pool(max(1, min(a.workers, len(jobs)))) as pool:
        for month, n, p, sec in pool.imap_unordered(convert_month, jobs):
            games += n; plies += p
            print(f"[{month}] {n:,} games, {p:,} plies, {sec}s", flush=True)
    print(f"TOTAL {len(jobs)} months, {games:,} games, {plies:,} plies ({plies * 2 / 1e9:.1f} GB moves.u16)")


if __name__ == "__main__":
    main()
