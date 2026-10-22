"""The grounding check: does a label claim something the game's measured card contradicts?

Five rules, each a phrase in the label plus a fact of the card that rules it out (a sacrifice with no sacrifices,
a queen trade with queens on the board, ...). chsm8-distillation keeps only label sets with no contradicted band.

    from grounding import contradicted
    contradicted(user_or_context, {"terse_1": ..., ...})   # -> names of the broken rules, [] if clean
"""
import re

BANDS = ("terse_1", "terse_2", "short", "medium", "long")


def facts(card):
    m = re.search(r"sacrifices \(captures.*?\): (\d+)", card)
    p = re.search(r"pressure on \w+'s king ring: [\d.]+ attackers/ply — (\w+)", card)
    k = re.search(r"king: (\w+)", card)
    d = re.search(r"development: ([\w ]+) \(", card)
    return {"sac": int(m.group(1)) if m else None, "queens_traded": "queens: traded" in card,
            "press": p.group(1) if p else None, "castle": k.group(1) if k else None,
            "dev": d.group(1) if d else None}


RULES = [
    ("sacrifice", r"\bsacrific", lambda f: f["sac"] == 0),
    ("queen trade", r"trade (the |your )?queens?|swap (the |your )?queens?|queens? off|queenless",
     lambda f: not f["queens_traded"]),
    ("king attack", r"attack (the|his|her|their|black.s|white.s|enemy|opponent.s) king|king ?hunt|storm the king|"
                    r"crash the king|mate attack", lambda f: f["press"] == "none"),
    ("slow development", r"slow(ly)? (to )?develop|delay(ed)? development|leave pieces (at )?home",
     lambda f: f["dev"] in ("very fast", "brisk")),
    ("never castle", r"never castle|don.t castle|skip castling|leave (your )?king in the cent",
     lambda f: f["castle"] in ("kingside", "queenside")),
]


def contradicted(card, labels):
    f = facts(card)
    return [name for b in BANDS for name, rx, cond in RULES
            if re.search(rx, (labels.get(b) or "").lower()) and cond(f)]
