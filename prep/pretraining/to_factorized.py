#!/usr/bin/env python3
"""Binary tier-2b layout -> factorized training rows (tier 3), per month.

CHSM8 does not train on a flat move vocabulary. Every ply becomes a 5-column
int16 row ``(kind, src, dst, piece, promo)`` and every game is wrapped in
BOS/EOS rows. ``piece`` is not stored in the uint16 move code, so this script
replays each game on a lightweight 64-square board to recover it.

The same replay also yields the per-row legality sidecar used by CHSM8's
legality-aware loss (optional, ``--no-legal`` skips it):

    occ    uint64 bitboard of the squares the side to move occupied right
           before the move. ``occ`` masks impossible source squares;
           ``~occ`` masks destinations occupied by the mover's own pieces.
    pmask  6-bit mask, bit i set = mover still has a piece of type i
           (pawn, knight, bishop, rook, queen, king).

Output, per month (mirrors the input layout):

    <out>/<YYYY-MM>/rows.i16      (N, 5) int16, kind in {PAD=0,BOS=1,EOS=2,MOVE=3}
    <out>/<YYYY-MM>/legal.bin     (N,) packed struct {occ: <u8, pmask: u1}, 9 bytes/row
    <out>/<YYYY-MM>/meta.parquet  per game: row_off (rows, not bytes), n_rows
                                  (incl. BOS+EOS), white_elo, black_elo
    <out>/<YYYY-MM>/_SUCCESS

    python to_factorized.py data_u16 -o data_3 --workers 64
    python to_factorized.py data_u16 -o data_3_nolegal --no-legal

Input is the output of ``parquet_to_u16.py``. Only numpy + pyarrow needed.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chsm8_format import KIND_BOS, KIND_EOS, KIND_MOVE, LEGAL_DTYPE

ROW_BOS = (KIND_BOS, 0, 0, 0, 0)
ROW_EOS = (KIND_EOS, 0, 0, 0, 0)
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
EMPTY = -1
_BACK_RANK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]
WHITE_START_OCC = (1 << 16) - 1
BLACK_START_OCC = ((1 << 16) - 1) << 48


def _initial_board():
    board = [EMPTY] * 64
    for f in range(8):
        board[f], board[8 + f] = _BACK_RANK[f], PAWN
        board[48 + f], board[56 + f] = PAWN, _BACK_RANK[f]
    return board


def _piece_mask(occ, board):
    mask = 0
    while occ:
        sq = (occ & -occ).bit_length() - 1
        mask |= 1 << board[sq]
        occ &= occ - 1
    return mask


def replay_game(codes):
    """uint16 move codes of one legal game -> (rows, legal), BOS..EOS aligned.

    No legality checking: the moves were validated with python-chess when the
    corpus was built. Only piece bookkeeping (captures, castling rook hop,
    en passant, promotion) is needed to know which piece moves."""
    board = _initial_board()
    white_occ, black_occ = WHITE_START_OCC, BLACK_START_OCC
    rows, legal = [ROW_BOS], [(0, 0)]
    white_to_move = True
    for code in codes.tolist():
        src, dst, promo = code & 0x3F, (code >> 6) & 0x3F, (code >> 12) & 0x7
        piece = board[src]
        mover, opp = (white_occ, black_occ) if white_to_move else (black_occ, white_occ)
        if piece == EMPTY or not mover & (1 << src):
            raise ValueError("corrupt game: from-square not occupied by side to move")
        legal.append((mover, _piece_mask(mover, board)))
        mover = (mover & ~(1 << src)) | (1 << dst)
        opp &= ~(1 << dst)
        if piece == KING and abs(dst - src) == 2:                       # castling
            rf, rt = (src + 3, src + 1) if dst > src else (src - 4, src - 1)
            board[rt], board[rf] = ROOK, EMPTY
            mover = (mover & ~(1 << rf)) | (1 << rt)
        elif piece == PAWN and (dst & 7) != (src & 7) and board[dst] == EMPTY:  # en passant
            ep = (src & ~7) | (dst & 7)
            board[ep] = EMPTY
            opp &= ~(1 << ep)
        rows.append((KIND_MOVE, src, dst, piece, promo))
        board[dst] = promo if promo else piece
        board[src] = EMPTY
        if white_to_move:
            white_occ, black_occ = mover, opp
        else:
            black_occ, white_occ = mover, opp
        white_to_move = not white_to_move
    rows.append(ROW_EOS)
    legal.append((0, 0))
    return rows, legal


def convert_month(args):
    src_dir, dst_dir, with_legal = args
    t0 = time.time()
    meta = pq.read_table(os.path.join(src_dir, "meta.parquet"),
                         columns=["move_off", "n_plies", "white_elo", "black_elo"])
    moves = np.memmap(os.path.join(src_dir, "moves.u16"), dtype="<u2", mode="r")
    off = meta["move_off"].to_numpy().astype(np.int64)
    npl = meta["n_plies"].to_numpy().astype(np.int64)
    os.makedirs(dst_dir, exist_ok=True)
    f_rows = open(os.path.join(dst_dir, "rows.i16"), "wb", buffering=8 << 20)
    f_legal = open(os.path.join(dst_dir, "legal.bin"), "wb", buffering=8 << 20) if with_legal else None
    row_off = np.zeros(len(off), dtype=np.uint64)
    n_rows = np.zeros(len(off), dtype=np.uint32)
    cursor = dropped = 0
    for i in range(len(off)):
        row_off[i] = cursor
        if npl[i] == 0:
            dropped += 1
            continue
        try:
            rows, legal = replay_game(moves[off[i]:off[i] + npl[i]])
        except ValueError:
            dropped += 1
            continue
        f_rows.write(np.asarray(rows, dtype="<i2").tobytes())
        if f_legal:
            arr = np.empty(len(legal), dtype=LEGAL_DTYPE)
            arr["occ"] = [x[0] for x in legal]
            arr["pmask"] = [x[1] for x in legal]
            f_legal.write(arr.tobytes())
        n_rows[i] = len(rows)
        cursor += len(rows)
    f_rows.close()
    if f_legal:
        f_legal.close()
    keep = n_rows > 0
    pq.write_table(pa.table({
        "row_off": pa.array(row_off[keep]), "n_rows": pa.array(n_rows[keep]),
        "white_elo": meta["white_elo"].filter(pa.array(keep)),
        "black_elo": meta["black_elo"].filter(pa.array(keep)),
    }), os.path.join(dst_dir, "meta.parquet"), compression="zstd")
    with open(os.path.join(dst_dir, "_SUCCESS"), "w") as f:
        f.write(f"games={int(keep.sum())} dropped={dropped} rows={cursor} has_legal={with_legal}\n")
    return os.path.basename(dst_dir), int(keep.sum()), dropped, cursor, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="binary layout from parquet_to_u16.py (<YYYY-MM>/moves.u16 + meta.parquet)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--months", nargs="*", help="subset of months (default: all)")
    ap.add_argument("--no-legal", action="store_true", help="skip the legal.bin sidecar")
    ap.add_argument("--workers", type=int, default=8, help="months converted in parallel")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    months = sorted(a.months or [m for m in os.listdir(a.indir)
                                  if os.path.isfile(os.path.join(a.indir, m, "meta.parquet"))])
    jobs = [(os.path.join(a.indir, m), os.path.join(a.outdir, m), not a.no_legal) for m in months
            if a.overwrite or not os.path.exists(os.path.join(a.outdir, m, "_SUCCESS"))]
    # biggest months first so they don't become the tail of the run
    jobs.sort(key=lambda j: os.path.getsize(os.path.join(j[0], "moves.u16")), reverse=True)
    tot = [0, 0, 0]
    with Pool(max(1, min(a.workers, len(jobs)))) as pool:
        for month, games, dropped, rows, sec in pool.imap_unordered(convert_month, jobs):
            tot[0] += games; tot[1] += dropped; tot[2] += rows
            print(f"[{month}] {games:,} games ({dropped:,} dropped), {rows:,} rows, {sec}s", flush=True)
    per_row = 10 + (0 if a.no_legal else LEGAL_DTYPE.itemsize)
    print(f"TOTAL {len(jobs)} months, {tot[0]:,} games, {tot[1]:,} dropped, {tot[2]:,} rows, "
          f"{tot[2] * per_row / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
