"""Deterministic, engine-free game features from a SAN move list.

These serve two purposes at once:

1. **Prompt material.** Mid-size LLMs are poor at inferring style from a raw
   move list; handing them a computed feature card is the single biggest
   quality lever available. The card is rendered by :func:`render_card`.
2. **Ground truth.** Every feature is a number we can correlate against style
   tags extracted from the generated label text. That correlation is what
   turns "does this label sound good" into "does this label carry real,
   learnable signal about the board" — the fitness function for the loop.

Everything here is pure python-chess. No engine, no network.

Features are computed per side, because a label describes one player.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass, field

import chess

VAL = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Home squares of the non-pawn pieces, for "kept the guard at home" measurement.
HOME = {
    chess.WHITE: {
        chess.A1: chess.ROOK, chess.B1: chess.KNIGHT, chess.C1: chess.BISHOP,
        chess.D1: chess.QUEEN, chess.F1: chess.BISHOP, chess.G1: chess.KNIGHT,
        chess.H1: chess.ROOK,
    },
    chess.BLACK: {
        chess.A8: chess.ROOK, chess.B8: chess.KNIGHT, chess.C8: chess.BISHOP,
        chess.D8: chess.QUEEN, chess.F8: chess.BISHOP, chess.G8: chess.KNIGHT,
        chess.H8: chess.ROOK,
    },
}

CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
BIG_CENTER = CENTER + [chess.C4, chess.C5, chess.F4, chess.F5]

_MOVENUM_RE = re.compile(r"\b\d+\.(?:\.\.)?")
_RESULT_RE = re.compile(r"^(1-0|0-1|1/2-1/2|\*)$")


def parse_san(moves_text: str) -> list[str]:
    """Tokenise a SAN mainline.

    Tolerates both ``1. e4 e5 2. Nf3`` and the YottaChess dump's redundant
    ``1. Nf3 1... e6 2. e4 2... Nf6`` form, plus glued ``1.e4``.
    """
    if not moves_text:
        return []
    text = _MOVENUM_RE.sub(" ", moves_text.replace("\n", " "))
    out = []
    for tok in text.split():
        tok = tok.strip()
        if not tok or _RESULT_RE.match(tok) or tok in {"...", "."}:
            continue
        if tok.startswith("{") or tok.startswith("$"):
            continue
        out.append(tok)
    return out


# -- static exchange evaluation -------------------------------------------

def _smallest_attacker_move(board: chess.Board, sq: int) -> chess.Move | None:
    """Cheapest legal capture onto ``sq`` for the side to move (None if none)."""
    best, best_val = None, math.inf
    for m in board.legal_moves:
        if m.to_square != sq or not board.is_capture(m):
            continue
        p = board.piece_at(m.from_square)
        if p is None:
            continue
        if VAL[p.piece_type] < best_val:
            best, best_val = m, VAL[p.piece_type]
    return best


def _see_rec(board: chess.Board, sq: int, depth: int = 0) -> float:
    if depth > 12:
        return 0.0
    m = _smallest_attacker_move(board, sq)
    if m is None:
        return 0.0
    victim = board.piece_at(sq)
    vv = VAL[victim.piece_type] if victim else 0.0
    board.push(m)
    val = max(0.0, vv - _see_rec(board, sq, depth + 1))
    board.pop()
    return val


def see(board: chess.Board, move: chess.Move) -> float:
    """Static exchange evaluation of a capture, in pawns, for the mover.

    Negative means the capture loses material even after every recapture —
    i.e. a genuine sacrifice rather than a trade. Uses legal moves, so pins
    are handled correctly.
    """
    b = board.copy(stack=False)
    if b.is_en_passant(move):
        vv = VAL[chess.PAWN]
    else:
        p = b.piece_at(move.to_square)
        vv = VAL[p.piece_type] if p else 0.0
    if move.promotion:
        vv += VAL[move.promotion] - VAL[chess.PAWN]
    b.push(move)
    return vv - _see_rec(b, move.to_square)


def material(board: chess.Board, color: bool) -> float:
    return sum(
        VAL[pt] * len(board.pieces(pt, color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def king_ring(board: chess.Board, color: bool) -> list[int]:
    """The enemy-facing zone: the king's square plus its neighbours."""
    ksq = board.king(color)
    if ksq is None:
        return []
    return [ksq] + list(chess.SquareSet(chess.BB_KING_ATTACKS[ksq]))


@dataclass
class SideFeatures:
    """Everything measured about one player's conduct in one game."""

    color: str = "white"
    # tempo / development
    development_ply: int = 0          # ply when the 3rd non-pawn piece left home
    castled: str = "none"             # "kingside" | "queenside" | "none"
    castle_ply: int = 0               # 0 if never
    early_queen_ply: int = 0          # first queen move, 0 if never moved
    home_pieces_mean: float = 0.0     # mean # of non-pawn pieces still on home squares
    # contact / violence
    captures: int = 0
    capture_rate: float = 0.0         # captures per own move
    checks: int = 0
    check_rate: float = 0.0
    see_sacs: int = 0                 # captures that lose >=1.5 pawns after recaptures
    sac_material: float = 0.0         # total pawns invested in those
    max_deficit: float = 0.0          # worst material deficit endured, in pawns
    material_volatility: float = 0.0  # stdev of the material curve
    # space / structure
    center_occupancy: float = 0.0     # mean own units on d4/e4/d5/e5
    center_attack: float = 0.0        # mean own attackers of the four central squares
    pawn_move_rate: float = 0.0
    storm_pushes: int = 0             # pawn pushes toward the enemy king's flank, rank>=5
    promotions: int = 0
    # king play
    king_pressure: float = 0.0        # mean own attackers on the enemy king ring
    king_exposure: float = 0.0        # mean enemy attackers on own king ring
    opposite_castling: bool = False
    # posture
    mobility: float = 0.0             # mean legal moves available on own turn
    retreat_rate: float = 0.0         # fraction of own moves that go backwards
    trades_initiated: int = 0         # captures with SEE ~ 0 (even swaps)
    # phase
    queens_off_ply: int = 0           # 0 if queens stayed on
    endgame_frac: float = 0.0         # fraction of plies played with <=12 units total


@dataclass
class GameFeatures:
    game_id: str = ""
    n_plies: int = 0
    n_moves: int = 0
    result: str = ""
    white_elo: int | None = None
    black_elo: int | None = None
    opening: str = ""
    eco: str = ""
    time_control: str = ""
    time_class: str = ""
    decisive: bool = True
    white: SideFeatures = field(default_factory=SideFeatures)
    black: SideFeatures = field(default_factory=SideFeatures)
    parse_error: str = ""

    def side(self, color: str) -> SideFeatures:
        return self.white if color == "white" else self.black

    def flat(self, color: str) -> dict[str, float]:
        """Numeric feature dict for one side, for correlation analysis."""
        sf = asdict(self.side(color))
        out: dict[str, float] = {}
        for k, v in sf.items():
            if isinstance(v, bool):
                out[k] = float(v)
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        out["castled_kingside"] = float(self.side(color).castled == "kingside")
        out["castled_queenside"] = float(self.side(color).castled == "queenside")
        out["never_castled"] = float(self.side(color).castled == "none")
        out["n_moves"] = float(self.n_moves)
        elo = self.white_elo if color == "white" else self.black_elo
        if elo:
            out["elo"] = float(elo)
        return out


def classify_time_control(tc: str) -> str:
    """Lichess-style bucket from a ``base+inc`` string."""
    m = re.match(r"^(\d+)\+(\d+)$", (tc or "").strip())
    if not m:
        return "unknown"
    base, inc = int(m.group(1)), int(m.group(2))
    est = base + 40 * inc
    if est < 29:
        return "ultrabullet"
    if est < 179:
        return "bullet"
    if est < 479:
        return "blitz"
    if est < 1499:
        return "rapid"
    return "classical"


# -- LAN parsing (the raw Lichess corpus, src/extract_lan_moves.py's output) --
#
# LAN tokens carry both squares explicitly ("Ng1-f3", "e5xd4", "c7-c8=Q"),
# unlike SAN which is context-dependent and needs board.parse_san to
# disambiguate. That means a LAN token can be turned into a chess.Move
# directly from its own text, with no need for python-chess's SAN resolver —
# faster, and immune to the rare SAN ambiguities that resolver has to guess at.
_LAN_RE = re.compile(
    r"^(?:O-O-O|O-O)[+#]?$|^[KQRBN]?([a-h][1-8])[-x]([a-h][1-8])(?:=([QRBN]))?[+#]?$"
)
_PROMO = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


def parse_lan_move(board: chess.Board, tok: str) -> chess.Move:
    """One LAN token -> a chess.Move, using the from/to squares it names."""
    bare = tok.strip().rstrip("+#")
    if bare in ("O-O", "O-O-O"):
        rank = 0 if board.turn == chess.WHITE else 7
        king_from = chess.square(4, rank)
        king_to = chess.square(2, rank) if bare == "O-O-O" else chess.square(6, rank)
        return chess.Move(king_from, king_to)
    m = _LAN_RE.match(tok.strip())
    if not m:
        raise ValueError(f"unparseable LAN token: {tok!r}")
    frm, to, promo = m.groups()
    return chess.Move(chess.parse_square(frm), chess.parse_square(to),
                      promotion=_PROMO.get(promo))


def extract(
    moves_text: str | list[str],
    *,
    game_id: str = "",
    result: str = "",
    white_elo: int | None = None,
    black_elo: int | None = None,
    opening: str = "",
    eco: str = "",
    time_control: str = "",
    max_plies: int = 400,
    notation: str = "san",
) -> GameFeatures:
    """Replay a mainline and measure both sides.

    ``notation='san'`` (default) expects a SAN string, as produced by the
    YottaChess/pgn_to_csv corpora. ``notation='lan'`` expects a list of LAN
    tokens, as produced by ``src/extract_lan_moves.py`` for the main 1.48M
    -game Lichess corpus.
    """
    gf = GameFeatures(
        game_id=game_id, result=result, white_elo=white_elo, black_elo=black_elo,
        opening=opening, eco=eco, time_control=time_control,
        time_class=classify_time_control(time_control),
        decisive="1/2" not in (result or "") and "draw" not in (result or "").lower(),
    )
    gf.white.color, gf.black.color = "white", "black"

    if notation == "lan":
        tokens = list(moves_text)[:max_plies]
    else:
        tokens = parse_san(moves_text)[:max_plies]
    board = chess.Board()

    acc: dict[bool, dict] = {
        c: {
            "moves": 0, "pawn_moves": 0, "retreats": 0, "mobility": [],
            "home": [], "center_occ": [], "center_att": [],
            "kpress": [], "kexp": [], "moved_from_home": set(),
            "material": [],
        }
        for c in (chess.WHITE, chess.BLACK)
    }
    queens_off_ply = 0
    endgame_plies = 0
    ply = 0

    for tok in tokens:
        try:
            move = parse_lan_move(board, tok) if notation == "lan" else board.parse_san(tok)
            if move not in board.legal_moves:
                raise ValueError(f"{move.uci()} is not legal here")
        except (ValueError, AssertionError) as e:
            gf.parse_error = f"ply {ply} tok {tok!r}: {e}"
            break

        mover = board.turn
        opp = not mover
        sf = gf.white if mover == chess.WHITE else gf.black
        a = acc[mover]
        ply += 1
        a["moves"] += 1

        piece = board.piece_at(move.from_square)
        assert piece is not None

        # -- pre-move measurements (position the mover is choosing from) --
        a["mobility"].append(board.legal_moves.count())
        a["home"].append(sum(
            1 for sq, pt in HOME[mover].items()
            if (p := board.piece_at(sq)) and p.color == mover and p.piece_type == pt
        ))
        a["center_occ"].append(sum(
            1 for sq in CENTER if (p := board.piece_at(sq)) and p.color == mover
        ))
        a["center_att"].append(sum(len(board.attackers(mover, sq)) for sq in CENTER))
        ring = king_ring(board, opp)
        a["kpress"].append(sum(len(board.attackers(mover, sq)) for sq in ring))
        own_ring = king_ring(board, mover)
        a["kexp"].append(sum(len(board.attackers(opp, sq)) for sq in own_ring))

        # -- the move itself --
        if board.is_capture(move):
            sf.captures += 1
            s = see(board, move)
            if s <= -1.5:
                sf.see_sacs += 1
                sf.sac_material += -s
            elif abs(s) < 0.6:
                sf.trades_initiated += 1

        if board.is_castling(move):
            sf.castled = ("kingside" if chess.square_file(move.to_square) > 4
                          else "queenside")
            sf.castle_ply = ply

        if piece.piece_type == chess.PAWN:
            a["pawn_moves"] += 1
            # A storm push: a pawn advancing to the 5th rank or beyond, on the
            # flank where the enemy king currently sits.
            ksq = board.king(opp)
            if ksq is not None:
                rank = chess.square_rank(move.to_square)
                fwd = rank if mover == chess.WHITE else 7 - rank
                same_flank = abs(chess.square_file(move.to_square)
                                 - chess.square_file(ksq)) <= 2
                if fwd >= 4 and same_flank:
                    sf.storm_pushes += 1
            if move.promotion:
                sf.promotions += 1
        else:
            if move.from_square in HOME[mover]:
                a["moved_from_home"].add(move.from_square)
                if len(a["moved_from_home"]) == 3 and sf.development_ply == 0:
                    sf.development_ply = ply
            if piece.piece_type == chess.QUEEN and sf.early_queen_ply == 0:
                sf.early_queen_ply = ply
            # Backwards move = toward one's own first rank.
            df = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
            if (df < 0) if mover == chess.WHITE else (df > 0):
                a["retreats"] += 1

        board.push(move)

        if board.is_check():
            sf.checks += 1

        for c in (chess.WHITE, chess.BLACK):
            acc[c]["material"].append(material(board, c))

        if queens_off_ply == 0 and not board.pieces(chess.QUEEN, chess.WHITE) \
                and not board.pieces(chess.QUEEN, chess.BLACK):
            queens_off_ply = ply
        if chess.popcount(board.occupied) <= 12:
            endgame_plies += 1

    gf.n_plies = ply
    gf.n_moves = (ply + 1) // 2

    for color, sf in ((chess.WHITE, gf.white), (chess.BLACK, gf.black)):
        a = acc[color]
        n = max(1, a["moves"])
        opp_mat = acc[not color]["material"]
        own_mat = a["material"]
        diffs = [o - p for o, p in zip(own_mat, opp_mat)] or [0.0]

        sf.capture_rate = sf.captures / n
        sf.check_rate = sf.checks / n
        sf.pawn_move_rate = a["pawn_moves"] / n
        sf.retreat_rate = a["retreats"] / n
        sf.mobility = statistics.fmean(a["mobility"]) if a["mobility"] else 0.0
        sf.home_pieces_mean = statistics.fmean(a["home"]) if a["home"] else 0.0
        sf.center_occupancy = statistics.fmean(a["center_occ"]) if a["center_occ"] else 0.0
        sf.center_attack = statistics.fmean(a["center_att"]) if a["center_att"] else 0.0
        sf.king_pressure = statistics.fmean(a["kpress"]) if a["kpress"] else 0.0
        sf.king_exposure = statistics.fmean(a["kexp"]) if a["kexp"] else 0.0
        sf.max_deficit = max(0.0, -min(diffs))
        sf.material_volatility = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        sf.queens_off_ply = queens_off_ply
        sf.endgame_frac = endgame_plies / max(1, ply)
        if sf.development_ply == 0:
            sf.development_ply = ply  # never finished developing

    opp_cast = (gf.white.castled != "none" and gf.black.castled != "none"
                and gf.white.castled != gf.black.castled)
    gf.white.opposite_castling = gf.black.opposite_castling = opp_cast
    return gf


# -- rendering -------------------------------------------------------------

def _band(v: float, cuts: tuple[float, ...], names: tuple[str, ...]) -> str:
    for c, nm in zip(cuts, names):
        if v < c:
            return nm
    return names[-1]


def render_card(gf: GameFeatures, color: str, *, style: str = "full") -> str:
    """Human-readable feature card for the labelling prompt.

    ``style='full'`` gives raw numbers plus qualitative bands; ``'qual'`` gives
    only the bands (cheaper, and stops weak models parroting digits into the
    label); ``'none'`` returns an empty string.
    """
    if style == "none":
        return ""
    sf = gf.side(color)
    opp = gf.side("black" if color == "white" else "white")
    me, them = color, ("black" if color == "white" else "white")

    q: list[str] = []
    q.append(f"development: {_band(sf.development_ply, (12, 20, 30), ('very fast', 'brisk', 'slow', 'very slow'))}"
             f" (3rd piece out at ply {sf.development_ply})")
    q.append(f"king: {sf.castled}" + (f" on ply {sf.castle_ply}" if sf.castle_ply else "")
             + (", opposite wings to the opponent" if sf.opposite_castling else ""))
    q.append(f"pieces left at home (mean of 7): {sf.home_pieces_mean:.1f}"
             f" — {_band(sf.home_pieces_mean, (2.0, 3.5, 5.0), ('fully mobilised', 'well developed', 'partly home', 'largely undeveloped'))}")
    q.append(f"captures: {sf.captures} ({sf.capture_rate:.2f}/move,"
             f" {_band(sf.capture_rate, (0.12, 0.20, 0.30), ('low contact', 'normal', 'high contact', 'very bloody'))})")
    q.append(f"checks given: {sf.checks} ({sf.check_rate:.2f}/move)")
    q.append(f"sacrifices (captures losing material after recaptures): {sf.see_sacs},"
             f" total invested {sf.sac_material:.1f} pawns")
    q.append(f"worst material deficit endured: {sf.max_deficit:.1f} pawns;"
             f" swing volatility {sf.material_volatility:.2f}")
    q.append(f"central presence: {sf.center_occupancy:.2f} units on d4/e4/d5/e5,"
             f" {sf.center_attack:.2f} attackers of those squares"
             f" — {_band(sf.center_attack, (2.0, 3.5, 5.0), ('cedes the centre', 'contests it', 'controls it', 'dominates it'))}")
    q.append(f"pawn moves: {sf.pawn_move_rate:.2f}/move; pawn pushes at the enemy king: {sf.storm_pushes}")
    q.append(f"pressure on {them}'s king ring: {sf.king_pressure:.2f} attackers/ply"
             f" — {_band(sf.king_pressure, (1.0, 2.0, 3.5), ('none', 'mild', 'real', 'heavy'))}")
    q.append(f"pressure suffered on {me}'s own king: {sf.king_exposure:.2f} attackers/ply")
    q.append(f"mobility: {sf.mobility:.1f} legal moves/turn; retreats {sf.retreat_rate:.2f}/move")
    q.append(f"even trades initiated: {sf.trades_initiated}")
    q.append("queens: " + (f"traded on ply {sf.queens_off_ply}" if sf.queens_off_ply
                           else "stayed on the board"))
    q.append(f"endgame share: {sf.endgame_frac:.0%} of plies with 12 or fewer units")

    if style == "qual":
        q = [re.sub(r"\s*\(?[-+]?\d[\d.,/%]*\s*(pawns|units|attackers/ply|legal moves/turn)?\)?", "", line)
             for line in q]

    head = [f"You are looking at the {me.upper()} player."]
    if gf.n_moves:
        head.append(f"Game length: {gf.n_moves} moves.")
    if gf.opening:
        # Name the opening without attributing it to a side. Opening names
        # describe the game, not one player's choice: handed "Sicilian
        # Defense" while describing White, models write "play the Sicilian as
        # White", which is wrong and would inject that error into a large
        # share of labels at corpus scale.
        head.append(
            f"Opening reached: {gf.opening}" + (f" ({gf.eco})" if gf.eco else "")
            + f" — this names the whole game, not {me}'s own system;"
            " only refer to it if it fits the side you are describing."
        )
    if gf.time_class and gf.time_class != "unknown":
        head.append(f"Time control: {gf.time_control} ({gf.time_class}).")
    elo = gf.white_elo if color == "white" else gf.black_elo
    oelo = gf.black_elo if color == "white" else gf.white_elo
    if elo and oelo:
        head.append(f"Rating: {elo} vs {oelo}.")

    return " ".join(head) + "\n" + "\n".join(f"- {line}" for line in q)
