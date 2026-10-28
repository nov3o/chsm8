"""Tier 2b -> tier 3: attach piece type, wrap each game in BOS/EOS, write the
factorized (kind,src,dst,piece,promo) row format the decoder trains on, plus
per-row structural legality masks for training-time masked loss.

See TRAINING.md section 2 for the row-format contract, and section on
"legality-aware training loss" for why the masks below exist: penalizing a
model for choosing among illegal src/piece/dst options wastes gradient
signal on structurally-impossible classes, and doing this at training time
(not just generation time, where src/tokenizer.py's masking already lives)
requires knowing, for every position, the board state right before that
position's move. That's exactly what this replay pass already computes and
then discards -- persisting it is cheap relative to a full rebuild
(measured ~20 min for all 162 months on this hardware).

Why this needs a replay pass at all (not just a reshape of tier 2b): tier
2b's moves.u16 stores only (from_square, to_square, promotion) per ply
(src/move_codec.py) -- piece type is deliberately not stored there, since
it's recoverable by replay and storing it would cost bytes for free
information. The factored decoder needs piece type on every row
(src/model.py's FactoredEmbedding/FactoredHead both key on it), so tier 3
has to do that replay once, here, rather than at training-time random-access
speed (which would be far too slow at ~78B-ply scale, and would add real
per-step overhead to the training loop if done inline there instead).

The replay tracks a bare 64-square piece-type array plus two 64-bit
occupancy bitboards (one per color), NOT a python-chess Board -- no
legality checking needed (tier 2b's moves already passed python-chess's
legality check once, in build_2a.py's SAN replay), so this only needs cheap
array/bitmask bookkeeping: whose piece is on the from-square (-> the row's
`piece` column and the mover's piece-type mask), which squares the mover
occupies (-> src/dst legality masks), plus the three move kinds that
displace a second square (castling's rook hop, en passant's captured pawn,
promotion's piece-type change on the destination square).

Output, per month, in <outdir>/<month>/ (mirrors the 2a/2b layout):
  rows.i16      flat int16, shape (N, 5), columns (kind,src,dst,piece,promo)
                -- exactly src/factored_tokenizer.py's row format
                (KIND_PAD=0, KIND_BOS=1, KIND_EOS=2, KIND_MOVE=3)
  legal.bin     flat structured array, shape (N,), dtype [('occ','<u8'),
                ('pmask','u1')] (9 bytes/row, tightly packed, no alignment
                padding) -- index-aligned 1:1 with rows.i16 (same row_off/
                n_rows offsets from meta.parquet, since those are row
                counts, not byte offsets). For a MOVE row, `occ` is the
                64-bit bitboard of squares the mover (the side making THIS
                row's move) occupied immediately before this move -- this
                is simultaneously the correct src-legality mask (mover can
                only move from a square it occupies) and, bitwise-negated,
                the correct coarse dst-legality mask (mover can't land on
                its own piece) -- see src/factored_tokenizer.py's existing
                `legal_component_masks()`, whose semantics this matches
                exactly, just precomputed here instead of derived live via
                python-chess at generation time. `pmask` is a 6-bit mask
                (bit i set = mover has >=1 piece of PIECE_TYPES[i] on the
                board) matching factored_tokenizer.py's `piece_mask`. Zero
                for non-MOVE rows (BOS/EOS/PAD), matching how src/dst/piece/
                promo are already zero for those rows in rows.i16.
  meta.parquet  one row per game: row_off (uint64, offset into rows.i16/
                legal.bin in ROWS not bytes), n_rows (uint32, includes
                BOS+EOS), white_elo, black_elo (passthrough)

  python3 src/build_tier3.py data/2b -o data/3
  python3 src/build_tier3.py data/2b -o data/3 --months 2026-06 --workers 32
"""

import argparse
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --- factored row format (kept import-free from factored_tokenizer.py / chess
# so this script has zero non-numpy/pyarrow dependencies -- see module
# docstring: the whole point of the custom replay is to avoid python-chess's
# per-move overhead at this scale) ---
KIND_PAD, KIND_BOS, KIND_EOS, KIND_MOVE = 0, 1, 2, 3
ROW_BOS = (KIND_BOS, 0, 0, 0, 0)
ROW_EOS = (KIND_EOS, 0, 0, 0, 0)

# board-array piece codes, same order/index as factored_tokenizer.PIECE_TYPES
# (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING) -> 0..5; -1 = empty square.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(6)
EMPTY = -1

# move_codec.py's promotion bit-code order (0=none,1=N,2=B,3=R,4=Q) is
# already identical to factored_tokenizer's PROMOTION_TO_IDX order, and a
# non-zero promo code (1..4) is already identical to this module's own
# KNIGHT..QUEEN piece codes -- both facts are asserted by build_tier3_test
# below, so no remapping table is needed in the hot loop.

_START_BACK_RANK = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]

LEGAL_DTYPE = np.dtype([("occ", "<u8"), ("pmask", "u1")])  # itemsize 9, tightly packed
ROW_LEGAL_ZERO = (0, 0)  # BOS/EOS/PAD rows: no mover, masks unused (mirrors ROW_BOS/EOS's zeros)

WHITE_START_OCC = (1 << 16) - 1                       # squares 0-15: white's back rank + pawns
BLACK_START_OCC = ((1 << 16) - 1) << 48               # squares 48-63: black's pawns + back rank


def _initial_board() -> List[int]:
    board = [EMPTY] * 64
    for f in range(8):
        board[f] = _START_BACK_RANK[f]                 # rank 1, white
        board[8 + f] = PAWN                             # rank 2, white pawns
        board[48 + f] = PAWN                            # rank 7, black pawns
        board[56 + f] = _START_BACK_RANK[f]             # rank 8, black
    return board


def _piece_mask_of(occ: int, board: List[int]) -> int:
    """6-bit mask of piece types present among the squares set in `occ`."""
    mask = 0
    o = occ
    while o:
        sq = (o & -o).bit_length() - 1  # index of lowest set bit
        mask |= 1 << board[sq]
        o &= o - 1                      # clear lowest set bit
    return mask


def replay_game(codes: np.ndarray) -> Tuple[List[Tuple[int, int, int, int, int]], List[Tuple[int, int]]]:
    """codes: 1D uint16 array, one move-code per ply (move_codec.py's
    encoding), already-legal (extracted via a real python-chess replay in
    build_2a.py). Returns (rows, legal) -- BOS + one entry per ply + EOS in
    both, index-aligned. `legal[i]` is (own_occ, piece_mask) for `rows[i]`;
    zero for non-MOVE rows."""
    board = _initial_board()
    white_occ, black_occ = WHITE_START_OCC, BLACK_START_OCC
    rows = [ROW_BOS]
    legal = [ROW_LEGAL_ZERO]
    white_to_move = True
    for code in codes.tolist():
        src = code & 0x3F
        dst = (code >> 6) & 0x3F
        promo = (code >> 12) & 0x7  # 0..4

        piece = board[src]
        assert piece != EMPTY, "no piece at from-square -- corrupt tier-2b input"

        mover_occ, opp_occ = (white_occ, black_occ) if white_to_move else (black_occ, white_occ)
        src_bit = 1 << src
        assert mover_occ & src_bit, "from-square not occupied by side to move -- corrupt tier-2b input"
        own_occ_before = mover_occ  # snapshot BEFORE this move, for this row's legality mask
        piece_mask = _piece_mask_of(mover_occ, board)

        dst_bit = 1 << dst
        mover_occ = (mover_occ & ~src_bit) | dst_bit
        if opp_occ & dst_bit:                                    # normal capture
            opp_occ &= ~dst_bit

        if piece == KING and abs(dst - src) == 2:                # castling: hop the rook too
            if dst == src + 2:
                rook_from, rook_to = src + 3, src + 1
            else:
                rook_from, rook_to = src - 4, src - 1
            board[rook_to] = ROOK
            board[rook_from] = EMPTY
            mover_occ = (mover_occ & ~(1 << rook_from)) | (1 << rook_to)
        elif piece == PAWN and (dst % 8) != (src % 8) and board[dst] == EMPTY:
            # diagonal pawn move onto an empty square = en passant; captured
            # pawn sits on the mover's rank, destination's file.
            ep_sq = (src // 8) * 8 + (dst % 8)
            board[ep_sq] = EMPTY
            opp_occ &= ~(1 << ep_sq)

        rows.append((KIND_MOVE, src, dst, piece, promo))
        legal.append((own_occ_before, piece_mask))
        board[dst] = promo if promo != 0 else piece
        board[src] = EMPTY

        # Both mover_occ (moved-to/captured-from) AND opp_occ (a captured piece's bit cleared,
        # via a normal capture or en passant above) must persist -- opp_occ is a local copy that
        # was mutated above and must be written back to whichever side it actually belongs to, or
        # captured pieces silently "come back" on the opponent's next turn.
        if white_to_move:
            white_occ, black_occ = mover_occ, opp_occ
        else:
            black_occ, white_occ = mover_occ, opp_occ
        white_to_move = not white_to_move

    rows.append(ROW_EOS)
    legal.append(ROW_LEGAL_ZERO)
    return rows, legal


def convert_month(src_dir: str, dst_dir: str) -> Dict[str, int]:
    os.makedirs(dst_dir, exist_ok=True)
    t0 = time.time()
    table = pq.read_table(os.path.join(src_dir, "meta.parquet"),
                           columns=["move_off", "n_plies", "white_elo", "black_elo"])
    moves = np.memmap(os.path.join(src_dir, "moves.u16"), dtype="<u2", mode="r")
    off = table.column("move_off").to_numpy()
    npl = table.column("n_plies").to_numpy()
    we = table.column("white_elo").to_numpy()
    be = table.column("black_elo").to_numpy()

    out_rows = open(os.path.join(dst_dir, "rows.i16"), "wb", buffering=8 << 20)
    out_legal = open(os.path.join(dst_dir, "legal.bin"), "wb", buffering=8 << 20)
    row_off = np.empty(len(off), dtype=np.uint64)
    n_rows = np.empty(len(off), dtype=np.uint32)
    cursor = 0
    n_dropped = 0
    for i in range(len(off)):
        k = int(npl[i])
        if k == 0:
            n_dropped += 1
            row_off[i] = cursor
            n_rows[i] = 0
            continue
        o = int(off[i])
        codes = moves[o:o + k]
        try:
            rows, legal = replay_game(codes)
        except AssertionError:
            n_dropped += 1
            row_off[i] = cursor
            n_rows[i] = 0
            continue
        out_rows.write(np.asarray(rows, dtype=np.int16).tobytes())
        legal_arr = np.asarray(legal, dtype=np.uint64)  # (n,2): [occ, pmask] before narrowing pmask
        legal_struct = np.empty(len(legal), dtype=LEGAL_DTYPE)
        legal_struct["occ"] = legal_arr[:, 0]
        legal_struct["pmask"] = legal_arr[:, 1]
        out_legal.write(legal_struct.tobytes())
        row_off[i] = cursor
        n_rows[i] = len(rows)
        cursor += len(rows)
    out_rows.close()
    out_legal.close()

    keep = n_rows > 0
    pq.write_table(
        pa.table({
            "row_off": pa.array(row_off[keep], type=pa.uint64()),
            "n_rows": pa.array(n_rows[keep], type=pa.uint32()),
            "white_elo": pa.array(we[keep], type=pa.uint16()),
            "black_elo": pa.array(be[keep], type=pa.uint16()),
        }),
        os.path.join(dst_dir, "meta.parquet"), compression="zstd",
    )
    with open(os.path.join(dst_dir, "_SUCCESS"), "w") as f:
        f.write("ok\n")
    return {"games": len(off), "dropped": n_dropped, "rows": cursor, "seconds": round(time.time() - t0, 1)}


def _convert_one(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, int]]:
    month, src_dir, dst_dir = args
    return month, convert_month(src_dir, dst_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="tier 2b directory (contains <month>/ subdirs)")
    ap.add_argument("-o", "--outdir", default="data/3")
    ap.add_argument("--months", nargs="*", help="subset of months; default all")
    ap.add_argument("--workers", type=int, default=8, help="months converted in parallel")
    ap.add_argument("--overwrite", action="store_true", help="reconvert months that already have a _SUCCESS marker")
    a = ap.parse_args()

    months = sorted(a.months or [m for m in os.listdir(a.indir)
                                  if os.path.isdir(os.path.join(a.indir, m)) and not m.startswith(".")])
    jobs = []
    for m in months:
        src_dir = os.path.join(a.indir, m)
        dst_dir = os.path.join(a.outdir, m)
        if not os.path.exists(os.path.join(src_dir, "meta.parquet")):
            print(f"[{m}] no meta.parquet in tier-2b, skipping")
            continue
        if not a.overwrite and os.path.exists(os.path.join(dst_dir, "_SUCCESS")):
            print(f"[{m}] already converted, skipping (--overwrite to redo)")
            continue
        jobs.append((m, src_dir, dst_dir))

    # Largest tier-2b input first (moves.u16 size as a proxy for month size, since Lichess's
    # corpus grows ~monotonically over time -- chronological dispatch order used to mean the
    # biggest, slowest-to-convert months were queued last, so the final stretch of the whole
    # run was dominated by exactly the jobs that take longest, with fewer-and-fewer other jobs
    # left to keep all workers busy alongside them. Descending-size dispatch keeps the pool
    # saturated more evenly throughout, since large jobs get the most wall-clock time to overlap
    # with everything else instead of being stragglers at the end.
    jobs.sort(key=lambda j: os.path.getsize(os.path.join(j[1], "moves.u16")), reverse=True)

    print(f"Converting {len(jobs)} months with {a.workers} workers (largest first)")
    tot = {"games": 0, "dropped": 0, "rows": 0}
    with Pool(a.workers) as pool:
        for month, c in pool.imap_unordered(_convert_one, jobs):
            for k in tot:
                tot[k] += c[k]
            print(f"[{month}] {c['games']:,} games ({c['dropped']:,} dropped), "
                  f"{c['rows']:,} rows, {c['seconds']}s", flush=True)

    print(f"\nTOTAL {tot['games']:,} games ({tot['dropped']:,} dropped), "
          f"{tot['rows']:,} rows, {tot['rows'] * (5 * 2 + LEGAL_DTYPE.itemsize) / 1e9:.1f} GB "
          f"(rows.i16 + legal.bin combined)")


if __name__ == "__main__":
    main()
