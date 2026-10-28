#!/usr/bin/env python3
"""Rule-based grounding check: how often labels claim something the game's own
measured-feature card contradicts. Same rules for teachers and students.

usage: grounding.py TRIAL [TRIAL ...]      (samples_<TRIAL>.jsonl in results/labeler)
"""
import collections
import json
import os
import re
import sys

R = "runs/labeler/"
VAL = os.environ.get("GROUND_SPLIT", "data/labeler/test.jsonl")
BANDS = ("terse_1", "terse_2", "short", "medium", "long")


def facts(u: str) -> dict:
    m = re.search(r"sacrifices \(captures.*?\): (\d+)", u)
    p = re.search(r"pressure on \w+'s king ring: [\d.]+ attackers/ply — (\w+)", u)
    k = re.search(r"king: (\w+)", u)
    d = re.search(r"development: ([\w ]+) \(", u)
    return {"sac": int(m.group(1)) if m else None, "queens_traded": "queens: traded" in u,
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


def score(label_sets):
    """label_sets: list of (card_text, {band: text}). Returns (label rate, game rate, Counter)."""
    c, n, bad = collections.Counter(), 0, 0
    for card, labs in label_sets:
        f, hit = facts(card), False
        for b in BANDS:
            t = (labs.get(b) or "").lower()
            n += 1
            for name, rx, cond in RULES:
                if re.search(rx, t) and cond(f):
                    c[name] += 1
                    hit = True
        bad += hit
    return sum(c.values()) / max(1, n), bad / max(1, len(label_sets)), c


def main():
    val = {json.loads(l)["key"]: json.loads(l) for l in open(VAL)}
    trials = sys.argv[1:]
    samples = {t: {json.loads(l)["key"]: json.loads(l) for l in open(R + f"samples_{t}.jsonl")} for t in trials}
    keys = sorted(set.intersection(*[set(s) for s in samples.values()]))
    rows = [("teacher (mix)", [(val[k]["user"], val[k]["labels"]) for k in keys]),
            ("teacher=gemma", [(val[k]["user"], val[k]["labels"]) for k in keys if val[k]["labeler"] == "gemma"]),
            ("teacher=apertus", [(val[k]["user"], val[k]["labels"]) for k in keys if val[k]["labeler"] == "apertus"])]
    rows += [(t, [(val[k]["user"], samples[t][k]["pred"]) for k in keys]) for t in trials]
    print(f"| source | games | labels contradicted | games with any | breakdown |\n|---|---|---|---|---|")
    for name, sets in rows:
        lr, gr, c = score(sets)
        print(f"| {name} | {len(sets)} | {lr:.1%} | {gr:.0%} | " + ", ".join(f"{k} {v}" for k, v in c.most_common()) + " |")


if __name__ == "__main__":
    main()
