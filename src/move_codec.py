"""uint16 move encoding shared by the 2a scan and the 2b filter.

A chess move is stored as a single uint16:

    bits  0-5    from_square  (0-63)
    bits  6-11   to_square    (0-63)
    bits 12-14   promotion    (0 = none, 1 = N, 2 = B, 3 = R, 4 = Q)

Max encodable value is 20479, so ids >= 20480 are free for special tokens.

Piece type is deliberately *not* stored. It is always recoverable by replaying
the move sequence from the initial position, and the factorized decoder needs
it as a derived quantity anyway. Storing it would cost bytes for no information.

Castling is encoded from the underlying king move (e1->g1), not from the "O-O"
text, so it round-trips like any other move.
"""

import chess

PROMO_TO_CODE = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
CODE_TO_PROMO = {v: k for k, v in PROMO_TO_CODE.items()}

MAX_MOVE_ID = 20479
TOK_EOS = 65535          # game separator, written by the tier-3 tokenizer only


def encode(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | (PROMO_TO_CODE[move.promotion] << 12)


def decode(code: int) -> chess.Move:
    return chess.Move(code & 0x3F, (code >> 6) & 0x3F, CODE_TO_PROMO[(code >> 12) & 0x7])
