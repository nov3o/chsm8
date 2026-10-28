"""Catch labels that assert something the board says is false.

`fidelity` in metrics.py is a rank correlation: it tells you whether labels
that talk about attacking tend to come from attacking games. That is the right
population-level question, but it is blind to the individual label that says
"sacrifice material for the initiative" about a game where nobody sacrificed
anything. Those are the errors that make a label actively harmful as a
training pair, and averaging cannot see them.

This module is a deterministic contradiction check. Each rule pairs a phrase
the label might use with the board measurement that would have to hold for it
to be true. Rules only fire when the evidence is unambiguous, because a false
accusation here would send the optimiser chasing a phantom — the thresholds
are deliberately loose and the check is one-sided.

Output feeds two places: a per-prompt error rate in reports, and concrete
"this label says X, the board says Y" examples handed to the optimiser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import corpus, features as F


# Words that flip the meaning of a phrase that follows them. Without this,
# "refusing to trade queens" fires the trade-queens rule and "prioritise king
# safety over rapid development" fires the fast-development rule -- and a
# false accusation sends the optimiser chasing a problem that is not there.
# A few words may sit between the negator and the phrase ("refusing TO trade
# queens"), but punctuation ends the clause and so ends the negation's reach.
NEGATORS = re.compile(
    r"\b(?:not|never|no|don'?t|do not|avoid|avoiding|refus\w*|without|"
    r"rather than|instead of|reluctant to|unwilling to|resist\w*|decline\w*|"
    r"rarely|seldom|hardly|over)\b[\w\s'-]{0,18}$", re.I
)
NEGATION_WINDOW = 34


def _negated(text: str, start: int) -> bool:
    return bool(NEGATORS.search(text[max(0, start - NEGATION_WINDOW):start]))


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    # (side features, whole-game features) -> None if consistent, else the
    # board fact that contradicts the claim.
    check: Callable[[F.SideFeatures, F.GameFeatures], str | None]


def _r(name: str, pattern: str, check) -> Rule:
    return Rule(name, re.compile(pattern, re.I), check)


RULES: list[Rule] = [
    _r("claims_sacrifice",
       r"\b(?:sacrific\w+|sac\b|gambit away|give up (?:a |your )?(?:piece|material|pawn)|"
       r"invest material|material is secondary|don'?t worry about material)\b",
       lambda s, g: (f"no sacrifice occurred (0 capture-sacs, worst deficit "
                     f"{s.max_deficit:.1f} pawns)")
       if s.see_sacs == 0 and s.max_deficit < 1.5 else None),

    _r("claims_king_attack",
       r"\b(?:attack (?:the|his|her|their) king|storm (?:the|his|their) king|"
       r"pawn storm|mating attack|hunt the king|throw .{0,20}at (?:the|his|their) king|"
       r"assault on the king)\b",
       lambda s, g: (f"little king pressure ({s.king_pressure:.1f} attackers/ply, "
                     f"{s.storm_pushes} storm pushes, {s.checks} checks)")
       if s.king_pressure < 1.0 and s.storm_pushes == 0 and s.checks <= 1 else None),

    _r("claims_castle_kingside",
       r"\bcastle (?:short|kingside)\b",
       lambda s, g: f"castled {s.castled}" if s.castled != "kingside" else None),

    _r("claims_castle_queenside",
       r"\bcastle (?:long|queenside)\b",
       lambda s, g: f"castled {s.castled}" if s.castled != "queenside" else None),

    _r("claims_opposite_castling",
       r"\bopposite (?:side|wing)s? castl\w+|castle (?:on the )?opposite\b",
       lambda s, g: "both sides castled the same way or one did not castle"
       if not s.opposite_castling else None),

    _r("claims_queen_trade",
       r"\btrade (?:the )?queens|queens? off|exchange queens|swap queens\b",
       lambda s, g: "queens stayed on the board all game"
       if s.queens_off_ply == 0 else None),

    _r("claims_queens_on",
       r"\bkeep (?:the )?queens on|queens? stay on|avoid (?:the )?queen trade\b",
       lambda s, g: f"queens came off on ply {s.queens_off_ply}"
       if 0 < s.queens_off_ply < max(20, int(g.n_plies * 0.6)) else None),

    _r("claims_pieces_home",
       r"\b(?:keep (?:your )?pieces (?:at )?home|leave .{0,15} on (?:their|the) "
       r"(?:initial|starting|home) squares?|don'?t develop|stay undeveloped|"
       r"keep the guard\b)",
       lambda s, g: f"fully mobilised (mean {s.home_pieces_mean:.1f}/7 pieces at home)"
       if s.home_pieces_mean < 2.5 else None),

    _r("claims_fast_development",
       r"\b(?:develop (?:very )?(?:fast|quickly|rapidly)|rapid development|"
       r"get (?:your )?pieces out (?:fast|quickly)|develop like lightning)\b",
       lambda s, g: f"slow development (3rd piece out only on ply {s.development_ply})"
       if s.development_ply > 24 else None),

    _r("claims_centre_control",
       r"\b(?:control (?:the )?cent(?:re|er)|take (?:the )?cent(?:re|er)|"
       r"dominate the cent(?:re|er)|occupy the cent(?:re|er))\b",
       lambda s, g: f"ceded the centre ({s.center_attack:.1f} attackers of d4/e4/d5/e5)"
       if s.center_attack < 1.8 else None),

    _r("claims_endgame",
       r"\b(?:steer (?:in)?to (?:an? )?endgame|simplify into (?:an? )?endgame|"
       r"technical endgame|convert (?:in )?the endgame|grind (?:it )?out)\b",
       lambda s, g: "the game never reached an endgame"
       if s.endgame_frac < 0.05 else None),

    # Only deliberate sacrifices count here. A large material deficit is just
    # as likely to mean the player was outplayed, and the label is an
    # instruction about intent, not a prediction of the result.
    _r("claims_material_greed",
       r"\b(?:hold (?:on to )?material|hoard material|don'?t give up material|"
       r"grab (?:the )?(?:pawn|material)|stay (?:up|ahead) on material|"
       r"never sacrifice)\b",
       lambda s, g: f"sacrificed material deliberately ({s.see_sacs} capture-sacs, "
                    f"{s.sac_material:.1f} pawns invested)"
       if s.see_sacs >= 2 else None),

    _r("claims_trading",
       r"\b(?:trade (?:often|freely|a lot|everything)|keep trading|"
       r"simplify (?:at every|whenever)|exchange (?:often|freely))\b",
       lambda s, g: f"very few exchanges ({s.captures} captures, {s.capture_rate:.2f}/move)"
       if s.capture_rate < 0.10 else None),

    _r("claims_avoid_trades",
       r"\b(?:avoid (?:trades|exchanges|simplif\w+)|keep pieces on|don'?t trade|"
       r"refuse (?:trades|exchanges))\b",
       lambda s, g: f"traded heavily ({s.captures} captures, {s.capture_rate:.2f}/move)"
       if s.capture_rate > 0.28 else None),
]


@dataclass
class Contradiction:
    rule: str
    label: str
    board_says: str
    game_id: str
    color: str


def check_label(text: str, game: corpus.Game, color: str) -> list[Contradiction]:
    sf = game.feat.side(color)
    out = []
    for rule in RULES:
        m = rule.pattern.search(text)
        if not m or _negated(text, m.start()):
            continue
        verdict = rule.check(sf, game.feat)
        if verdict:
            out.append(Contradiction(rule.name, text, verdict, game.game_id, color))
    return out


def audit(rows, games: dict[str, corpus.Game]) -> tuple[float, dict[str, int], list[Contradiction]]:
    """Return (contradictions per label, counts by rule, the examples)."""
    found: list[Contradiction] = []
    n = 0
    for r in rows:
        if not r.ok:
            continue
        g = games.get(r.game_id)
        if g is None:
            continue
        for _band, text in r.labels:
            n += 1
            found.extend(check_label(text, g, r.color))
    counts: dict[str, int] = {}
    for c in found:
        counts[c.rule] = counts.get(c.rule, 0) + 1
    return (len(found) / max(1, n), dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            found)


def format_examples(found: list[Contradiction], limit: int = 8) -> str:
    if not found:
        return ""
    # One example per rule first, so the optimiser sees the variety of errors
    # rather than eight instances of the same one.
    seen: set[str] = set()
    picked: list[Contradiction] = []
    for c in found:
        if c.rule not in seen:
            seen.add(c.rule)
            picked.append(c)
    picked += [c for c in found if c not in picked]
    lines = ["",
             "LABELS THAT STATE SOMETHING THE BOARD CONTRADICTS.",
             "Each is a training pair that would teach the wrong association:"]
    for c in picked[:limit]:
        lines.append(f"  [{c.rule}] {c.label}")
        lines.append(f"      board: {c.board_says}")
    return "\n".join(lines)
