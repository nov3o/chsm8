"""Shared constants and codecs for the CHSM8 chess datasets.

Everything the other scripts in this folder need to agree on lives here:

* the compact uint16 move code used by the binary ``moves.u16`` layout,
* the UCI-string lookup tables used by the Parquet layout,
* the factorized ``(kind, src, dst, piece, promo)`` row format that the CHSM8
  decoder trains on, and its per-row legality sidecar.

Only numpy is required (python-chess is optional, used for SAN/LAN export).

uint16 move code
----------------
    bits  0-5    from_square  (0-63, a1=0 ... h8=63, python-chess numbering)
    bits  6-11   to_square    (0-63)
    bits 12-14   promotion    (0 = none, 1 = N, 2 = B, 3 = R, 4 = Q)

Max code is 20479. Castling is stored as the king move (e1g1, e1c1, e8g8,
e8c8), exactly as UCI writes it, so every move round-trips through UCI.
"""

import numpy as np

# ---- uint16 <-> UCI ---------------------------------------------------------

N_CODES = 20480
PROMO_CHARS = ["", "n", "b", "r", "q"]
FILES = "abcdefgh"


def square_name(sq: int) -> str:
    return FILES[sq & 7] + str((sq >> 3) + 1)


def code_to_uci(code: int) -> str:
    code = int(code)
    return (square_name(code & 0x3F) + square_name((code >> 6) & 0x3F)
            + PROMO_CHARS[(code >> 12) & 0x7])


def uci_to_code(uci: str) -> int:
    f = (ord(uci[0]) - 97) + 8 * (ord(uci[1]) - 49)
    t = (ord(uci[2]) - 97) + 8 * (ord(uci[3]) - 49)
    p = PROMO_CHARS.index(uci[4]) if len(uci) == 5 else 0
    return f | (t << 6) | (p << 12)


def _build_luts():
    """Byte lookup tables: code -> UCI bytes, padded to 5, plus lengths."""
    chars = np.zeros((N_CODES, 5), dtype=np.uint8)
    lens = np.zeros(N_CODES, dtype=np.int64)
    for code in range(N_CODES):
        promo = (code >> 12) & 0x7
        if promo > 4:
            continue
        s = code_to_uci(code).encode()
        chars[code, :len(s)] = np.frombuffer(s, dtype=np.uint8)
        lens[code] = len(s)
    return chars, lens


UCI_CHARS, UCI_LENS = _build_luts()
# reverse table for parsing: 4-char key "e2e4" -> code, promotions handled apart
_SQ_BYTES = np.array([[ord(FILES[s & 7]), ord(str((s >> 3) + 1))] for s in range(64)], dtype=np.uint8)


def decode_moves_to_strings(codes: np.ndarray, n_plies: np.ndarray):
    """Vectorized: flat uint16 codes + per-game ply counts -> (offsets, bytes).

    Returns the int64 offsets (len = n_games + 1) and uint8 data buffers of an
    Arrow string array holding one space-separated UCI string per game.
    Every game must have at least one ply.
    """
    codes = np.asarray(codes, dtype=np.int64)
    n_plies = np.asarray(n_plies, dtype=np.int64)
    assert codes.size == int(n_plies.sum()), "codes/n_plies mismatch"
    assert (n_plies > 0).all(), "empty game"
    lens = UCI_LENS[codes] + 1                       # every move followed by a space ...
    last = np.cumsum(n_plies) - 1
    lens[last] -= 1                                  # ... except each game's last move
    chars = np.full((codes.size, 6), 32, dtype=np.uint8)  # pad with spaces
    chars[:, :5] = UCI_CHARS[codes]
    # a 4-char move needs its space in column 4, not column 5
    four = UCI_LENS[codes] == 4
    chars[four, 4] = 32
    data = chars[np.arange(6)[None, :] < lens[:, None]]
    move_end = np.cumsum(lens)
    offsets = np.zeros(n_plies.size + 1, dtype=np.int64)
    offsets[1:] = move_end[last]
    return offsets, data


def encode_strings_to_moves(strings):
    """Inverse of decode_moves_to_strings for an iterable of UCI game strings.

    Returns (flat uint16 codes, int64 n_plies)."""
    out, n = [], []
    for s in strings:
        toks = s.split()
        n.append(len(toks))
        out.extend(uci_to_code(t) for t in toks)
    return np.asarray(out, dtype="<u2"), np.asarray(n, dtype=np.int64)


# ---- result codes -------------------------------------------------------------

RESULT_STRINGS = ["1-0", "0-1", "1/2-1/2"]          # uint8 code in the binary layout
RESULT_CODES = {s: i for i, s in enumerate(RESULT_STRINGS)}

# ---- factorized row format (what CHSM8 actually trains on) --------------------

KIND_PAD, KIND_BOS, KIND_EOS, KIND_MOVE = 0, 1, 2, 3
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]   # piece index 0..5
PROMO_NAMES = ["none", "knight", "bishop", "rook", "queen"]           # promo index 0..4
ROW_DTYPE = np.dtype("<i2")            # rows are (N, 5) int16: kind, src, dst, piece, promo
LEGAL_DTYPE = np.dtype([("occ", "<u8"), ("pmask", "u1")])  # 9 bytes/row, packed
