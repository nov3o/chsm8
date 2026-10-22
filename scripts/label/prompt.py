"""One side of one game -> the prompt the teacher (and the distilled labeler) sees.

The prompt has the moves in SAN and a card of measurements computed from the board with python-chess (no engine):
development, castling, captures, sacrifices (static exchange evaluation), central control, king pressure, mobility,
trades, endgame share. The teacher gets SYSTEM plus user(); the distilled labeler gets only context(), which is the
part of the user message that changes from game to game.

    from prompt import context, user, SYSTEM
    ctx = context(uci_moves, "white", white_elo=2242, black_elo=1915, tc_base=240, tc_increment=0,
                  opening="French Defense: Rubinstein Variation", eco="C10")
"""
import math
import re
import statistics
from dataclasses import dataclass

import chess

SYSTEM = (
    "An engine plays from a written instruction. Given one side's game, write the instruction that produced this "
    "play. Imperative, style only — no moves, squares, move numbers, results, or names.\n\n"
    "Reject generic chess advice that fits any game — 'control the center', 'develop your pieces', 'put pressure on "
    "the king' say nothing about THIS side. Name what makes this side's play unusual: reckless or overcautious, fast "
    "or slow to develop, hoarding material or throwing it away, calm or chaotic, one-sided attack or balanced play.")

SPEC = """Return ONLY a JSON array. No prose before or after, no code fence.
It must contain exactly 5 objects, distributed over these 4 length bands:

  2 x "terse"  (6-12 words) — one blunt clause, lowercase is fine, no punctuation polish — how an impatient user actually types
  1 x "short"  (13-20 words) — one plain sentence naming the main thing to do, nothing more
  1 x "medium"  (22-35 words) — one or two short sentences, still just one continuous thought — not a list of topics
  1 x "long"  (35-55 words) — the supremum: as far as an unusually chatty user would ever go. Still typed in one breath, not composed — one or two things they care about, never a checklist of temperament + opening + centre + king + material. If it would read like a coach's notes instead of a text message, it is too long or too structured

Each object is {"band": "<band name>", "text": "<the instruction>"}.
Where a band asks for more than one entry, they must differ in wording and in which aspect they emphasise — do not paraphrase yourself.
Most of these are short on purpose. Do not pad a short band to sound more complete; a real user would not.

No moves, squares, move numbers, or any number/percentage/count/time-control
from the measurements (captures, plies, attackers-per-turn, bullet/blitz,
etc.) — say what they mean, don't quote them.
Bad: "push g4 g5", "castle at move 12", "bullet 60+0", "15 captures", "28% of plies".
Good: "storm the kingside with pawns", "castle late", "play fast", "trade constantly".
Nothing invented beyond what's shown."""

BANDS = ("terse_1", "terse_2", "short", "medium", "long")

VAL = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.25, chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0}
HOME = {c: {chess.square(f, 0 if c else 7): pt for f, pt in zip((0, 1, 2, 3, 5, 6, 7), (4, 2, 3, 5, 3, 2, 4))}
        for c in (chess.WHITE, chess.BLACK)}  # non-pawn pieces on their start squares (no king)
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]


# ------------------------------------------------------------------ measurements

def _cheapest_capture(board, sq):
    best, best_val = None, math.inf
    for m in board.legal_moves:
        if m.to_square == sq and board.is_capture(m):
            p = board.piece_at(m.from_square)
            if p is not None and VAL[p.piece_type] < best_val:
                best, best_val = m, VAL[p.piece_type]
    return best


def _exchange(board, sq, depth=0):
    m = _cheapest_capture(board, sq) if depth <= 12 else None
    if m is None:
        return 0.0
    victim = board.piece_at(sq)
    board.push(m)
    val = max(0.0, (VAL[victim.piece_type] if victim else 0.0) - _exchange(board, sq, depth + 1))
    board.pop()
    return val


def see(board, move):
    """Static exchange evaluation of a capture, in pawns, for the mover: < 0 loses material after all recaptures."""
    b = board.copy(stack=False)
    p = b.piece_at(move.to_square)
    gain = VAL[chess.PAWN] if b.is_en_passant(move) else (VAL[p.piece_type] if p else 0.0)
    if move.promotion:
        gain += VAL[move.promotion] - VAL[chess.PAWN]
    b.push(move)
    return gain - _exchange(b, move.to_square)


def material(board, color):
    return sum(VAL[pt] * len(board.pieces(pt, color)) for pt in range(1, 6))


def king_ring(board, color):
    k = board.king(color)
    return [] if k is None else [k] + list(chess.SquareSet(chess.BB_KING_ATTACKS[k]))


@dataclass
class Side:
    development_ply: int = 0
    castled: str = "none"
    castle_ply: int = 0
    home_pieces_mean: float = 0.0
    captures: int = 0
    capture_rate: float = 0.0
    checks: int = 0
    check_rate: float = 0.0
    see_sacs: int = 0
    sac_material: float = 0.0
    max_deficit: float = 0.0
    material_volatility: float = 0.0
    center_occupancy: float = 0.0
    center_attack: float = 0.0
    pawn_move_rate: float = 0.0
    storm_pushes: int = 0
    king_pressure: float = 0.0
    king_exposure: float = 0.0
    opposite_castling: bool = False
    mobility: float = 0.0
    retreat_rate: float = 0.0
    trades_initiated: int = 0
    queens_off_ply: int = 0
    endgame_frac: float = 0.0


def measure(moves, max_plies=400):
    """[chess.Move] of a legal game -> {chess.WHITE: Side, chess.BLACK: Side}, n_plies."""
    board = chess.Board()
    sides = {chess.WHITE: Side(), chess.BLACK: Side()}
    acc = {c: {k: [] for k in ("mobility", "home", "center_occ", "center_att", "kpress", "kexp", "material")}
           | {"moves": 0, "pawn_moves": 0, "retreats": 0, "left_home": set()} for c in sides}
    queens_off = endgame = ply = 0
    for move in moves[:max_plies]:
        mover, opp = board.turn, not board.turn
        sf, a = sides[mover], acc[mover]
        ply += 1
        a["moves"] += 1
        piece = board.piece_at(move.from_square)
        # the position the mover chooses from
        a["mobility"].append(board.legal_moves.count())
        a["home"].append(sum(1 for sq, pt in HOME[mover].items()
                             if (p := board.piece_at(sq)) and p.color == mover and p.piece_type == pt))
        a["center_occ"].append(sum(1 for sq in CENTER if (p := board.piece_at(sq)) and p.color == mover))
        a["center_att"].append(sum(len(board.attackers(mover, sq)) for sq in CENTER))
        a["kpress"].append(sum(len(board.attackers(mover, sq)) for sq in king_ring(board, opp)))
        a["kexp"].append(sum(len(board.attackers(opp, sq)) for sq in king_ring(board, mover)))
        # the move
        if board.is_capture(move):
            sf.captures += 1
            s = see(board, move)
            if s <= -1.5:
                sf.see_sacs += 1
                sf.sac_material += -s
            elif abs(s) < 0.6:
                sf.trades_initiated += 1
        if board.is_castling(move):
            sf.castled = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
            sf.castle_ply = ply
        if piece.piece_type == chess.PAWN:
            a["pawn_moves"] += 1
            k = board.king(opp)
            fwd = chess.square_rank(move.to_square) if mover else 7 - chess.square_rank(move.to_square)
            if k is not None and fwd >= 4 and abs(chess.square_file(move.to_square) - chess.square_file(k)) <= 2:
                sf.storm_pushes += 1
        else:
            if move.from_square in HOME[mover]:
                a["left_home"].add(move.from_square)
                if len(a["left_home"]) == 3 and sf.development_ply == 0:
                    sf.development_ply = ply
            dr = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
            a["retreats"] += (dr < 0) if mover else (dr > 0)
        board.push(move)
        sf.checks += board.is_check()
        for c in sides:
            acc[c]["material"].append(material(board, c))
        if not queens_off and not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            queens_off = ply
        endgame += chess.popcount(board.occupied) <= 12
    mean = lambda xs: statistics.fmean(xs) if xs else 0.0
    for c, sf in sides.items():
        a, n = acc[c], max(1, acc[c]["moves"])
        diffs = [o - p for o, p in zip(a["material"], acc[not c]["material"])] or [0.0]
        sf.capture_rate, sf.check_rate = sf.captures / n, sf.checks / n
        sf.pawn_move_rate, sf.retreat_rate = a["pawn_moves"] / n, a["retreats"] / n
        sf.mobility, sf.home_pieces_mean = mean(a["mobility"]), mean(a["home"])
        sf.center_occupancy, sf.center_attack = mean(a["center_occ"]), mean(a["center_att"])
        sf.king_pressure, sf.king_exposure = mean(a["kpress"]), mean(a["kexp"])
        sf.max_deficit = max(0.0, -min(diffs))
        sf.material_volatility = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        sf.queens_off_ply, sf.endgame_frac = queens_off, endgame / max(1, ply)
        sf.development_ply = sf.development_ply or ply  # never finished developing
    w, b = sides[chess.WHITE].castled, sides[chess.BLACK].castled
    sides[chess.WHITE].opposite_castling = sides[chess.BLACK].opposite_castling = "none" not in (w, b) and w != b
    return sides, ply


def time_class(base, inc):
    est = base + 40 * inc
    return next(n for c, n in ((29, "ultrabullet"), (179, "bullet"), (479, "blitz"), (1499, "rapid"),
                               (math.inf, "classical")) if est < c)


def _band(v, cuts, names):
    return next((n for c, n in zip(cuts, names) if v < c), names[-1])


def card(sf, color, n_moves, elo, opp_elo, opening, eco, tc_base, tc_increment):
    them = "black" if color == "white" else "white"
    q = [
        f"development: {_band(sf.development_ply, (12, 20, 30), ('very fast', 'brisk', 'slow', 'very slow'))}"
        f" (3rd piece out at ply {sf.development_ply})",
        f"king: {sf.castled}" + (f" on ply {sf.castle_ply}" if sf.castle_ply else "")
        + (", opposite wings to the opponent" if sf.opposite_castling else ""),
        f"pieces left at home (mean of 7): {sf.home_pieces_mean:.1f} — "
        + _band(sf.home_pieces_mean, (2.0, 3.5, 5.0),
                ("fully mobilised", "well developed", "partly home", "largely undeveloped")),
        f"captures: {sf.captures} ({sf.capture_rate:.2f}/move, "
        + _band(sf.capture_rate, (0.12, 0.20, 0.30), ("low contact", "normal", "high contact", "very bloody")) + ")",
        f"checks given: {sf.checks} ({sf.check_rate:.2f}/move)",
        f"sacrifices (captures losing material after recaptures): {sf.see_sacs}, "
        f"total invested {sf.sac_material:.1f} pawns",
        f"worst material deficit endured: {sf.max_deficit:.1f} pawns; swing volatility {sf.material_volatility:.2f}",
        f"central presence: {sf.center_occupancy:.2f} units on d4/e4/d5/e5, {sf.center_attack:.2f} attackers of "
        "those squares — " + _band(sf.center_attack, (2.0, 3.5, 5.0),
                                   ("cedes the centre", "contests it", "controls it", "dominates it")),
        f"pawn moves: {sf.pawn_move_rate:.2f}/move; pawn pushes at the enemy king: {sf.storm_pushes}",
        f"pressure on {them}'s king ring: {sf.king_pressure:.2f} attackers/ply — "
        + _band(sf.king_pressure, (1.0, 2.0, 3.5), ("none", "mild", "real", "heavy")),
        f"pressure suffered on {color}'s own king: {sf.king_exposure:.2f} attackers/ply",
        f"mobility: {sf.mobility:.1f} legal moves/turn; retreats {sf.retreat_rate:.2f}/move",
        f"even trades initiated: {sf.trades_initiated}",
        "queens: " + (f"traded on ply {sf.queens_off_ply}" if sf.queens_off_ply else "stayed on the board"),
        f"endgame share: {sf.endgame_frac:.0%} of plies with 12 or fewer units",
    ]
    head = [f"You are looking at the {color.upper()} player."]
    if n_moves:
        head.append(f"Game length: {n_moves} moves.")
    if opening:  # names the game, not one side's choice; the card says so to keep labels from misattributing it
        head.append(f"Opening reached: {opening}" + (f" ({eco})" if eco else "") + f" — this names the whole game, "
                    f"not {color}'s own system; only refer to it if it fits the side you are describing.")
    if tc_base is not None and tc_increment is not None:
        head.append(f"Time control: {tc_base}+{tc_increment} ({time_class(tc_base, tc_increment)}).")
    if elo and opp_elo:
        head.append(f"Rating: {elo} vs {opp_elo}.")
    return " ".join(head) + "\n" + "\n".join(f"- {line}" for line in q)


# ------------------------------------------------------------------ prompts

MAX_PLIES = 80  # the move list stops after move 40; the card still measures the whole game


def context(uci_moves, side, white_elo=None, black_elo=None, tc_base=None, tc_increment=None, opening="", eco=""):
    """The game-specific part of the prompt: which side, the moves in SAN, the measured card."""
    board, moves, san = chess.Board(), [], []
    for u in uci_moves.split() if isinstance(uci_moves, str) else uci_moves:
        m = chess.Move.from_uci(u)
        san.append(board.san(m))
        board.push(m)
        moves.append(m)
    sides, n_plies = measure(moves)
    text = " ".join(f"{i // 2 + 1}. {s}" if i % 2 == 0 else s for i, s in enumerate(san[:MAX_PLIES]))
    if len(san) > MAX_PLIES:
        text += " ..."
    white = side == "white"
    c = card(sides[white], side, (n_plies + 1) // 2, white_elo if white else black_elo,
             black_elo if white else white_elo, opening, eco, tc_base, tc_increment)
    return (f"You are looking at the {side.upper()} player.\n\n"
            f"MOVES (Moves are given in standard algebraic notation from the start of the game.)\n{text}\n\n"
            f"MEASURED FROM THE BOARD\n{c}")


def user(ctx):
    """The teacher's user message around a context()."""
    return f"GAME\n{ctx}\n\n{SPEC}"


def parse_teacher(text):
    """Teacher reply (a JSON array of {band, text}) -> {terse_1, terse_2, short, medium, long}, or {} if malformed."""
    import json
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return {}
    got = {}
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not str(it.get("text", "")).strip():
            continue
        band = str(it.get("band", "")).strip().lower()
        key = ("terse_2" if "terse_1" in got else "terse_1") if band == "terse" else band
        if key in BANDS and key not in got:
            got[key] = " ".join(str(it["text"]).split())
    return got if len(got) == 5 else {}
