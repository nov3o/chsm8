"""Chess games -> CHSM8 rows, with the released CHSM8 tokenizer (chsm8_tok, Rust).

    pip install "git+https://huggingface.co/LegumMagister/chsm8-tokenizer"    # needs Rust: https://rustup.rs

chsm8_tok reads any notation (SAN, LAN, UCI, PGN with numbers, comments, variations; mixed within a game), checks
every move for legality and returns uint16 move tokens:

    token = from | to << 6 | promo << 12     squares a1=0 ... h8=63, promo 0 none, 1 n, 2 b, 3 r, 4 q

The model reads one row of 5 integers per position:

    (kind, src, dst, piece, promo)
    kind   0 PAD, 1 BOS (start of game), 2 EOS (end of game), 3 MOVE
    piece  moving piece: 0 pawn, 1 knight, 2 bishop, 3 rook, 4 queen, 5 king

Piece is not in the token (the position determines it), so `rows()` replays the tokens on a small board to add it.
The same replay gives the two legality bits training uses per row: `occ`, the 64-bit bitboard of squares the side
to move occupies before the move, and `pmask`, the set of piece types it still has (model.loss_fn).
"""
import chsm8_tok

KIND_PAD, KIND_BOS, KIND_EOS, KIND_MOVE = 0, 1, 2, 3
BOS, EOS, PAD = (KIND_BOS, 0, 0, 0, 0), (KIND_EOS, 0, 0, 0, 0), (KIND_PAD, 0, 0, 0, 0)

tokenize = chsm8_tok.tokenize          # any notation -> [uint16 tokens]
to_uci = chsm8_tok.to_uci              # any notation -> ["e2e4", ...]
token_to_uci = chsm8_tok.token_to_uci  # 1804 -> "e2e4"


def analyze(game):
    """Game so far (any notation) -> (rows of the game with BOS, [(uci, san, row) for every legal next move])."""
    return chsm8_tok.analyze(game)


_BACK = [3, 1, 2, 4, 5, 2, 1, 3]  # a1..h1: rook knight bishop queen king bishop knight rook


def rows(tokens, bos=True, eos=True):
    """Tokens of one legal game -> (rows, occ, pmask), aligned row by row (BOS/EOS rows get occ = pmask = 0).

    The board only does the bookkeeping needed to know which piece moves: captures, the castling rook, en passant,
    promotion. Legality was already checked by chsm8_tok (or when our corpora were built)."""
    board = [-1] * 64
    for f in range(8):
        board[f], board[8 + f], board[48 + f], board[56 + f] = _BACK[f], 0, 0, _BACK[f]
    own, opp = (1 << 16) - 1, ((1 << 16) - 1) << 48  # squares of the side to move, of the other side
    out, occ, pmask = ([BOS], [0], [0]) if bos else ([], [], [])
    for t in tokens:
        src, dst, promo = t & 63, (t >> 6) & 63, (t >> 12) & 7
        piece = board[src]
        if piece < 0 or not own >> src & 1:
            raise ValueError(f"corrupt game: {token_to_uci(t)} does not start on a piece of the side to move")
        bits, m = 0, own
        while m:
            sq = (m & -m).bit_length() - 1
            bits |= 1 << board[sq]
            m &= m - 1
        out.append((KIND_MOVE, src, dst, piece, promo)); occ.append(own); pmask.append(bits)
        own = (own & ~(1 << src)) | (1 << dst)
        opp &= ~(1 << dst)
        if piece == 5 and abs(dst - src) == 2:  # castling: the rook hops over the king
            rf, rt = (src + 3, src + 1) if dst > src else (src - 4, src - 1)
            board[rt], board[rf] = 3, -1
            own = (own & ~(1 << rf)) | (1 << rt)
        elif piece == 0 and (dst & 7) != (src & 7) and board[dst] < 0:  # en passant
            ep = (src & ~7) | (dst & 7)
            board[ep] = -1
            opp &= ~(1 << ep)
        board[dst], board[src] = (promo if promo else piece), -1
        own, opp = opp, own
    if eos:
        out.append(EOS); occ.append(0); pmask.append(0)
    return out, occ, pmask


def encode(game, bos=True, eos=True):
    """A game in any notation -> (rows, occ, pmask)."""
    return rows(tokenize(game), bos, eos)


def uci_tokens(uci_moves: str):
    """Fast path for our datasets, whose moves are space-separated UCI: no parsing or legality check."""
    promo = "nbrq"
    out = []
    for m in uci_moves.split():
        t = (ord(m[0]) - 97) + 8 * (ord(m[1]) - 49) | ((ord(m[2]) - 97) + 8 * (ord(m[3]) - 49)) << 6
        out.append(t | (promo.index(m[4]) + 1) << 12 if len(m) == 5 else t)
    return out
