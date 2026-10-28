"""
Factorized chess move tokenization.

Each move is a tuple (src_square, dst_square, piece_type, promotion) built
directly from a `chess.Move` object during SAN replay — NOT by parsing move
text. This matters for castling: its LAN/SAN text is "O-O"/"O-O-O", but the
underlying Move still has real from_square/to_square (e.g. e1->g1 for white
kingside castle), which is what we actually want to tokenize.

Combinatorial space: 64 x 64 x 6 x 5 (mostly unreachable combinations, which
is fine — no dense lookup table is built over it; see model.py's
FactoredEmbedding/FactoredHead, which use small per-component tables/heads
instead of one big one).

Special tokens (PAD/BOS/EOS) live in a separate small "kind" dimension
rather than being folded into the (src,dst,piece,promo) space — see the
5-column row format below.

Row format — every position is a 5-int row: (kind, a, b, c, d)
    kind == KIND_PAD  : padding, ignored everywhere (a=b=c=d=0)
    kind == KIND_BOS  : start of game (a=b=c=d=0)
    kind == KIND_EOS  : end of game (a=b=c=d=0)
    kind == KIND_MOVE : a=src_square, b=dst_square, c=piece_type, d=promotion
"""

from typing import List, Optional, Tuple

import chess
import torch

N_SQUARES = 64
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_TO_IDX = {pt: i for i, pt in enumerate(PIECE_TYPES)}
N_PIECE_TYPES = len(PIECE_TYPES)  # 6

# promotion index 0 = "no promotion"; 1-4 = N/B/R/Q
PROMOTIONS = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
PROMOTION_TO_IDX = {p: i for i, p in enumerate(PROMOTIONS)}
N_PROMOTIONS = len(PROMOTIONS)  # 5

KIND_PAD, KIND_BOS, KIND_EOS, KIND_MOVE = range(4)
N_KINDS = 4

ROW_PAD = (KIND_PAD, 0, 0, 0, 0)
ROW_BOS = (KIND_BOS, 0, 0, 0, 0)
ROW_EOS = (KIND_EOS, 0, 0, 0, 0)


def move_to_row(move: chess.Move, board: chess.Board) -> Tuple[int, int, int, int, int]:
    """Build a (kind, src, dst, piece, promo) row from a Move + the board state
    right before the move (needed to resolve which piece is moving)."""
    piece = board.piece_at(move.from_square)
    assert piece is not None, f"No piece at {chess.square_name(move.from_square)} for move {move}"
    piece_idx = PIECE_TO_IDX[piece.piece_type]
    promo_idx = PROMOTION_TO_IDX[move.promotion]
    return (KIND_MOVE, move.from_square, move.to_square, piece_idx, promo_idx)


def encode_game(san_moves: List[str], add_bos: bool = True, add_eos: bool = True) -> List[Tuple[int, int, int, int, int]]:
    """Replay SAN moves on a fresh board, return a list of 5-int rows.
    Raises on illegal/unparseable moves (mirrors tokenizer.py's ChessTokenizer)."""
    rows = []
    if add_bos:
        rows.append(ROW_BOS)
    board = chess.Board()
    for san in san_moves:
        move = board.parse_san(san)
        rows.append(move_to_row(move, board))
        board.push(move)
    if add_eos:
        rows.append(ROW_EOS)
    return rows


def row_to_san(row: Tuple[int, int, int, int, int], board: chess.Board) -> Optional[str]:
    """Inverse of move_to_row, for decoding/generation. Returns None for non-move rows."""
    kind, src, dst, piece_idx, promo_idx = row
    if kind != KIND_MOVE:
        return None
    promotion = PROMOTIONS[promo_idx]
    move = chess.Move(src, dst, promotion=promotion)
    if move not in board.legal_moves:
        # e.g. castling is stored as the king's own from/to (e1->g1), which
        # chess.Move already reconstructs correctly and chess.Board matches
        # against legal_moves by (from,to,promotion) — this branch means the
        # move is genuinely illegal in this position (model mistake).
        return None
    return board.san(move)


def legal_component_masks(board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Coarse structural masks for the side to move, as boolean tensors
    (True = allowed). Used as additive -inf masks on the factorized logits.

    src_mask   (64,): squares occupied by the side-to-move's own pieces
    piece_mask (6,) : piece types the side-to-move has >=1 of on the board
    dst_mask   (64,): squares NOT occupied by the side-to-move's own pieces
                      (coarse — doesn't encode per-piece movement geometry;
                      real full legality stays a generation-time concern,
                      same as tokenizer.py's legal_move_mask)
    """
    own = board.occupied_co[board.turn]
    src_mask = torch.zeros(N_SQUARES, dtype=torch.bool)
    dst_mask = torch.ones(N_SQUARES, dtype=torch.bool)
    piece_mask = torch.zeros(N_PIECE_TYPES, dtype=torch.bool)

    for sq in chess.SQUARES:
        if own & chess.BB_SQUARES[sq]:
            src_mask[sq] = True
            dst_mask[sq] = False
            piece = board.piece_at(sq)
            if piece is not None:
                piece_mask[PIECE_TO_IDX[piece.piece_type]] = True

    return src_mask, dst_mask, piece_mask
