"""The thing being optimised: a labelling *variant*.

A variant is not just a system prompt. It is the whole recipe for turning one
(game, side) into a set of candidate user-prompts:

* what the model is told to do (``system``),
* how the game is presented (``card`` / ``moves`` / ``anonymise``),
* and what shape of output is demanded (``rungs``).

All three matter, and the loop mutates all three. Bolting a better sentence
onto a prompt that never shows the model a feature card is wasted effort.

**Length rungs.** Real users are terse and unimaginative. A label set that is
five polished paragraphs trains a CLIP model that only recognises polished
paragraphs, and then someone types "attack" and it falls over. Each variant
therefore emits a *ladder* of lengths, from a two-word tag to a full paragraph.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace

# name -> (word budget, description shown to the model)
#
# `long` is a SUPREMUM, not a target: the most words an unusually chatty user
# would ever bother typing into a text box, not a strategy brief. Earlier
# versions described it as covering "temperament, opening approach, the centre
# and king, attitude to material" — a four-part checklist — and models duly
# wrote four-clause structured guides that read like a coach's notes, not
# something a person typed. The description below asks for one continuous
# thought instead, and the word budget is cut accordingly (was 60-90).
#
# `micro` (1-2 words) and `tag` (3-5 words) existed as generation bands but are
# deliberately excluded from the default ladder: per-band retrieval showed them
# carrying the least signal (tag R@1 4.0x chance vs terse 4.7x, short 5.1x,
# medium 7.0x, long 7.3x — monotonic in length), and the decision was made to
# bet on the downstream text encoder generalising to short queries from its own
# pretrained vocabulary rather than spending generation budget on examples that
# collide across games by design. They remain in RUNGS so a variant can opt
# back into them via `rungs=`/`band_counts=` if that bet needs revisiting.
RUNGS: dict[str, tuple[str, str]] = {
    "micro": ("1-2 words", "the absolute minimum — a single word or two, nothing else — e.g. 'aggressive', 'solid', 'attack', 'defensive'"),
    "tag": ("3-5 words", "a bare style tag, the kind of thing someone types when they cannot be bothered — e.g. 'aggressive attacking chess', 'solid and defensive'"),
    "terse": ("6-12 words", "one blunt clause, lowercase is fine, no punctuation polish — how an impatient user actually types"),
    "short": ("13-20 words", "one plain sentence naming the main thing to do, nothing more"),
    "medium": ("22-35 words", "one or two short sentences, still just one continuous thought — not a list of topics"),
    "long": ("35-55 words", "the supremum: as far as an unusually chatty user would ever go. Still typed in one breath, not composed — one or two things they care about, never a checklist of temperament + opening + centre + king + material. If it would read like a coach's notes instead of a text message, it is too long or too structured"),
}

DEFAULT_RUNGS = ["terse", "short", "medium", "long"]

# How many labels to ask for in each band.
DEFAULT_BAND_COUNTS: dict[str, int] = {
    "terse": 2, "short": 1, "medium": 1, "long": 1,
}


def band_counts_for(rungs: list[str], n_per_rung: int,
                    counts: dict[str, int] | None) -> dict[str, int]:
    """Resolve per-band label counts, falling back to a flat ``n_per_rung``."""
    if counts:
        return {r: int(counts[r]) for r in rungs if counts.get(r)}
    return {r: n_per_rung for r in rungs}


def output_spec(rungs: list[str], n_per_rung: int,
                counts: dict[str, int] | None = None) -> str:
    """The output-format block appended to every user message."""
    resolved = band_counts_for(rungs, n_per_rung, counts)
    total = sum(resolved.values())
    lines = [
        "Return ONLY a JSON array. No prose before or after, no code fence.",
        f"It must contain exactly {total} objects, distributed over these"
        f" {len(resolved)} length bands:",
        "",
    ]
    for r, n in resolved.items():
        budget, desc = RUNGS[r]
        lines.append(f'  {n} x "{r}"  ({budget}) — {desc}')
    lines += [
        "",
        'Each object is {"band": "<band name>", "text": "<the instruction>"}.',
        "Where a band asks for more than one entry, they must differ in wording"
        " and in which aspect they emphasise — do not paraphrase yourself.",
        "Most of these are short on purpose. Do not pad a short band to sound"
        " more complete; a real user would not.",
    ]
    return "\n".join(lines)


@dataclass
class Variant:
    """One point in the (prompt x presentation x format) search space."""

    vid: str
    system: str
    user_template: str  # must contain {context} and {spec}
    card: str = "full"          # full | qual | none
    moves: str = "opening"      # full | opening | none
    anonymise: bool = True
    reveal_result: bool = False
    rungs: list[str] = field(default_factory=lambda: list(DEFAULT_RUNGS))
    n_per_rung: int = 2
    band_counts: dict[str, int] | None = field(
        default_factory=lambda: dict(DEFAULT_BAND_COUNTS))
    generation: int = 0
    parent: str | None = None
    notes: str = ""

    @property
    def counts(self) -> dict[str, int]:
        return band_counts_for(self.rungs, self.n_per_rung, self.band_counts)

    @property
    def n_labels(self) -> int:
        return sum(self.counts.values())

    def messages(self, context: str) -> list[dict]:
        user = self.user_template.format(
            context=context,
            spec=output_spec(self.rungs, self.n_per_rung, self.band_counts),
        )
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user},
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def child(self, vid: str, **changes) -> "Variant":
        return replace(self, vid=vid, parent=self.vid,
                       generation=self.generation + 1, **changes)


# -- output parsing --------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_labels(text: str, rungs: list[str]) -> list[tuple[str, str]]:
    """Extract ``(band, label)`` pairs from a model response, leniently.

    Weak models wrap JSON in fences, prepend commentary, or emit a numbered
    list instead. All three are recovered rather than thrown away — discarding
    them would bias the bake-off toward models that are merely good at format
    compliance rather than at describing chess.
    """
    if not text:
        return []
    out: list[tuple[str, str]] = []

    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    candidates.append(text)

    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            # {"tag": [...], "short": [...]} form
            for band, vals in data.items():
                if band in rungs and isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, str) and v.strip():
                            out.append((band, v.strip()))
            if out:
                return out
            data = list(data.values())
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    band = str(item.get("band") or item.get("length") or "").strip()
                    txt = item.get("text") or item.get("label") or item.get("prompt")
                    if isinstance(txt, str) and txt.strip():
                        out.append((band if band in rungs else "unknown", txt.strip()))
                elif isinstance(item, str) and item.strip():
                    out.append(("unknown", item.strip()))
            if out:
                return out

    # Truncated JSON: the response hit the token cap mid-array. Recover every
    # complete object rather than discarding the whole set — otherwise the
    # bake-off punishes verbose models for length, not for quality.
    for m in re.finditer(r'\{[^{}]*"(?:text|label|prompt)"\s*:\s*"(?:[^"\\]|\\.)*"[^{}]*\}', text):
        try:
            item = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        band = str(item.get("band") or item.get("length") or "").strip()
        txt = item.get("text") or item.get("label") or item.get("prompt")
        if isinstance(txt, str) and txt.strip():
            out.append((band if band in rungs else "unknown", txt.strip()))
    if out:
        return out

    # Last resort: numbered or bulleted lines.
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.{4,})$", line)
        if m:
            out.append(("unknown", m.group(1).strip().strip('"')))
    return out


def assign_bands(pairs: list[tuple[str, str]], rungs: list[str]) -> list[tuple[str, str]]:
    """Backfill ``unknown`` bands by word count, so length metrics still work."""
    bounds = [(r, _mid_words(r)) for r in rungs]
    fixed = []
    for band, txt in pairs:
        if band in rungs:
            fixed.append((band, txt))
            continue
        n = len(txt.split())
        best = min(bounds, key=lambda b: abs(b[1] - n))[0]
        fixed.append((best, txt))
    return fixed


def _mid_words(rung: str) -> float:
    budget = RUNGS[rung][0]
    nums = [int(x) for x in re.findall(r"\d+", budget)]
    return sum(nums) / len(nums) if nums else 30.0


# -- seed population -------------------------------------------------------

# GROUNDING is shared by every seed via _COMMON_TAIL, not bolted onto one
# variant's system prompt, because the failure it targets isn't specific to
# any one seed's voice: a model naming a more specific opening sub-variation
# than it was actually given ("English Attack" for a Najdorf game whose card
# says only "Najdorf Variation" and whose moves are 6.Bg5, not the English
# Attack's 6.Be3) is a general grounding failure, not a wording problem one
# rewrite of one seed would fix. The instruction is deliberately generic
# rather than opening-specific, since the same failure mode applies to any
# detail a model might add from general chess knowledge instead of from what
# it was actually shown — a tactical pattern, a named plan, a structure.
GROUNDING = """
GROUNDING — this applies to everything above, not only the opening name.
Use only what you were actually given: the moves shown and the measurements
in "MEASURED FROM THE BOARD" (if present). Do not add a more specific detail
than what you were given because it is typical or well-known for a similar
game — e.g. if the opening line is named only "Najdorf Variation," do not
call it "English Attack" or any other specific sub-line unless the actual
moves you were shown match that sub-line's defining moves. The same applies
to named plans, structures, or tactical patterns: if you cannot point to
where in the moves or measurements it comes from, leave it out. A generic
but accurate label beats a specific but invented one.
"""

_COMMON_TAIL = """
GAME
{context}

{spec}
""" + GROUNDING

# Deliberately terse counterpart to _COMMON_TAIL: same GAME/{spec} slots, but
# the grounding rule compressed to one line instead of the full paragraph.
# Used by the s10-mini seed, which is optimised for prompt+response brevity
# and inference speed on small free models rather than peak label quality.
_MINI_TAIL = """
GAME
{context}

{spec}

No moves, squares, move numbers, or any number/percentage/count/time-control
from the measurements (captures, plies, attackers-per-turn, bullet/blitz,
etc.) — say what they mean, don't quote them.
Bad: "push g4 g5", "castle at move 12", "bullet 60+0", "15 captures", "28% of plies".
Good: "storm the kingside with pawns", "castle late", "play fast", "trade constantly".
Nothing invented beyond what's shown.
"""

SEEDS: list[Variant] = [
    Variant(
        vid="s1-invert",
        notes="Direct inversion framing: recover the instruction that caused this play.",
        system=(
            "You reverse-engineer instructions. You are shown how one player conducted a "
            "single game of chess. Your job is to recover the instruction that player was "
            "following — the prompt that, if handed to a strong chess engine told to obey "
            "it, would make it play in this manner.\n\n"
            "Write instructions, not reports. Address the player in the second person or "
            "the imperative. Describe a policy that holds across a whole game, not a "
            "commentary on what already happened.\n\n"
            "Hard rules:\n"
            "- Never mention specific moves, squares, move numbers, or the result.\n"
            "- Never mention player names, ratings, or the fact that a game is being analysed.\n"
            "- Do not say 'this player' or 'in this game'. Say 'play', 'attack', 'keep', 'trade'.\n"
            "- Be concrete about actions: attack the king, take the centre, keep pieces at "
            "home, trade queens early, push the flank pawns, hold material."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s2-coach",
        notes="Pre-game brief framing; naturally produces plans and opening intent.",
        system=(
            "You are a coach writing the pre-game brief you handed your student before "
            "this game — the plan they then carried out. Write it as instructions to them.\n\n"
            "Cover, as space allows: how much risk to take, what to do with the centre, "
            "whether to castle and which way, whether to keep queens on, how much material "
            "you may spend for an attack, and how quickly to bring pieces out. Naming an "
            "opening or a pawn structure is fine and often useful.\n\n"
            "Never reference specific moves, squares, move numbers, results, or players. "
            "It is a brief written before the game, so nothing in it may reveal what happened."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s3-user",
        notes="Simulates a real product user typing at a chess AI. Targets the terse end.",
        system=(
            "A chess AI takes a plain-English instruction and then plays a game in that "
            "style. You are shown a game. Write what a USER would have typed to get the "
            "AI to play the way this side played.\n\n"
            "Write like a real user, not like a chess author. Most users are blunt and "
            "vague: 'play aggressive', 'attack the king', 'don't lose material', 'boring "
            "solid chess'. Longer entries can be more demanding and specific, but they "
            "should still sound typed into a box, not published.\n\n"
            "Users cannot see the game — they are describing the style they want in advance. "
            "So: no moves, no squares, no move numbers, no result, no player names, and "
            "nothing that only makes sense after the fact."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s4-card-only",
        notes="Ablation: measured features only, no move list. Tests whether moves add anything.",
        card="full",
        moves="none",
        system=(
            "You turn board measurements into playing instructions. You are given "
            "quantitative measurements of how one side conducted a game of chess. "
            "Recover the instruction that would produce play with those measurements.\n\n"
            "Write imperatives addressed to a player. Translate each measurement into the "
            "intent behind it — heavy king-ring pressure and pawn pushes mean 'launch a "
            "pawn storm at their king'; many pieces left at home and few captures mean "
            "'keep your position compact and wait'.\n\n"
            "Never quote a number. Never mention moves, squares, results, or players."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s5-moves-only",
        notes="Ablation: raw moves, no feature card. Baseline for the card's value.",
        card="none",
        moves="full",
        system=(
            "You are shown a chess game. Recover the instruction one side appears to have "
            "been following — the prompt that would make an engine play this way.\n\n"
            "Write imperatives addressed to a player, describing a policy for a whole game. "
            "Be concrete about actions: attacking the king, taking or ceding the centre, "
            "developing fast or slow, trading or avoiding trades, spending material or "
            "hoarding it.\n\n"
            "Never mention specific moves, squares, move numbers, the result, or players."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s6-contrastive",
        notes="Explicitly demands discriminativeness — the failure mode that kills CLIP.",
        system=(
            "You write style instructions for a chess engine. You will be shown how one "
            "side played a game; write the instructions that would reproduce that style.\n\n"
            "The instructions must DISTINGUISH this player from a default one. A phrase that "
            "would fit nine games out of ten — 'play good chess', 'develop your pieces', "
            "'control the centre and castle' — is worthless. Every clause must be one that "
            "some other reasonable game would violate. If this side did something unusual "
            "(left the king in the middle, hoarded material, traded down immediately, never "
            "advanced a pawn past the fourth rank), that belongs in the instruction.\n\n"
            "Imperatives only. No moves, squares, move numbers, results, or player names."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s10-mini",
        notes="Minimal prompt for a swarm of small/free SwissAI models: shortest "
              "system text, trimmed grounding, no thinking budget wasted on a long "
              "brief. Optimised for prompt+response brevity and speed, not peak quality.",
        system=(
            "An engine plays from a written instruction. Given one side's game, write "
            "the instruction that produced this play. Imperative, style only — no "
            "moves, squares, move numbers, results, or names.\n\n"
            "Reject generic chess advice that fits any game — 'control the center', "
            "'develop your pieces', 'put pressure on the king' say nothing about THIS "
            "side. Name what makes this side's play unusual: reckless or overcautious, "
            "fast or slow to develop, hoarding material or throwing it away, calm or "
            "chaotic, one-sided attack or balanced play."
        ),
        user_template=_MINI_TAIL.strip(),
    ),
]



# -- second-round seeds -----------------------------------------------------
#
# Both encode a result from the first evolution run rather than a guess.
#
# Run 1 showed every optimiser-proposed child switching the feature card from
# numeric to qualitative in order to stop models quoting measurements back
# into the label ("high mobility (around 28+ legal moves per turn)"). It
# worked, and it took the factual anchoring with it: child fidelity 0.24-0.38
# against 0.56-0.65 for the seeds. The fix belongs in the instruction, not in
# hiding the numbers.
#
# The run also showed distinctiveness and fidelity trading off: the child with
# the best distinctiveness of anything measured (0.468) had the worst fidelity
# (0.236). s8 keeps that child's decision-rule structure but restores the
# numeric card, to test whether the two can be had at once.

EXTRA_SEEDS: list[Variant] = [
    Variant(
        vid="s7-no-numbers",
        notes="s6 discriminativeness + explicit ban on quoting measurements, numeric card kept.",
        card="full",
        moves="opening",
        system=(
            "You write the instruction a person would type into a chess AI to make it "
            "play the way one side played. You are given measurements taken from the "
            "board, plus the moves.\n\n"
            "The measurements are for YOUR reading only. Never quote a number, a "
            "statistic, a rate, or a count in what you write. Translate each measurement "
            "into the intent behind it and then write only the intent. 'Pressure on the "
            "enemy king ring 3.4 attackers per ply' becomes 'keep piling pieces onto his "
            "king'. If a sentence would make a user ask 'how do you know that number', "
            "it is wrong.\n\n"
            "The instruction must DISTINGUISH this side from a default player. A clause "
            "that fits nine games in ten — 'develop your pieces', 'control the centre and "
            "castle' — is worthless. Every clause should be one that some other "
            "reasonable game would violate. If this side did something unusual (left the "
            "king in the middle, hoarded material, traded down at once, never pushed a "
            "pawn past the fourth rank), that is the part worth writing.\n\n"
            "Write imperatives addressed to a player, describing a policy for a whole "
            "game. Never mention specific moves, squares, move numbers, the result, or "
            "player names. Naming an opening or a pawn structure is fine."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
    Variant(
        vid="s8-decision-rules",
        notes="Decision-rule framing that maximised distinctiveness, with the numeric card restored.",
        card="full",
        moves="opening",
        system=(
            "You recover the standing orders one side was playing under. Express them as "
            "the decisions that player had already made before the game started — the "
            "choices that, taken together, produce this and only this kind of game.\n\n"
            "Think in terms of the commitments a player makes: how fast to bring pieces "
            "out and whether to finish; which way to castle, or whether to bother; "
            "whether the centre is to be occupied, contested or handed over; whether "
            "queens stay on; the price in material they will pay for an attack, and "
            "whether they will pay it at all; whether they meet a threat by defending or "
            "by hitting back somewhere else.\n\n"
            "State the commitments this side actually made, and say them in a way that "
            "rules out the opposite choice. Vague good-chess advice is a wasted clause.\n\n"
            "The measurements given to you are for your reading only — never quote a "
            "number. Write imperatives to a player. No moves, squares, move numbers, "
            "results, or player names."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
]


# -- third-round seed --------------------------------------------------------
#
# s3-user won validation both rounds (0.632 held-out) but its fidelity_detail
# showed it: development_speed +0.06, center_control +0.09, piece_activity
# -0.08 (wrong sign), keep_pieces_home -0.02 (wrong sign). The "real user"
# framing reliably reaches for aggression/king-attack/material vocabulary and
# skips the quieter structural axes — a user asked to describe a boring,
# well-developed, centre-heavy game still tends to write about attacking.
#
# s9 keeps s3's voice (it is what won grounding, promptability and cleanliness)
# and s7's no-digit-quoting instruction, and adds an explicit requirement to
# cover the specific axes s3 was blind to.
#
# RESULT (manual round 3, 14 held-out games, grok-4.5): this made things worse,
# not better — fitness 0.581 vs 0.622 (s3-user) / 0.623 (s7-no-numbers), and
# distinctiveness collapsed to 0.074 against 0.19-0.28 for the other two.
# Inspecting the output showed why: forcing every label to address development
# speed and centre control produces near-identical boilerplate ("fast
# development", "contest the centre") across games with opposite actual
# styles — a queen-trading grind and a pawn-sacrificing gambit both got "fast
# develop... control centre... stay active". The hypothesis was backwards:
# those axes correlated weakly in s3 precisely BECAUSE most games at this
# rating range develop reasonably fast and contest the centre somewhat — they
# are not very discriminating in the first place, so mandating coverage of
# them injects generic filler rather than signal. Kept as a documented
# negative result, not part of the recommended pool.

EXTRA_SEEDS_2: list[Variant] = [
    Variant(
        vid="s9-user-axes",
        notes=("NEGATIVE RESULT (see comment above): forcing coverage of weakly "
               "correlated axes injected boilerplate and cratered distinctiveness "
               "(0.074 vs 0.19-0.28 for s3/s7). Do not use; kept for the finding."),
        card="full",
        moves="opening",
        system=(
            "A chess AI takes a plain-English instruction and then plays a game in that "
            "style. You are shown one side's measurements and moves. Write what a USER "
            "would have typed to get the AI to play the way this side played.\n\n"
            "Write like a real user, not like a chess author. Most users are blunt and "
            "vague: 'play aggressive', 'attack the king', 'boring solid chess'. Longer "
            "entries can be more demanding and specific, but should still sound typed "
            "into a box, not published.\n\n"
            "The measurements are for your reading only — never quote a number, a rate, "
            "or a count.\n\n"
            "Before you settle on the main theme (attack, sacrifice, material), check "
            "these four axes against the measurements and mention any where this side is "
            "clearly NOT average, even if that is less exciting than an attack:\n"
            "  - development speed: fast and eager to get pieces out, or slow and "
            "content to leave them home?\n"
            "  - centre: fighting for it with pieces, ceding it, or blocking it with "
            "pawns?\n"
            "  - piece activity: pieces posted on active squares making threats, or "
            "sitting passively?\n"
            "  - how many pieces stayed on their starting squares for a long time?\n"
            "A label is allowed to be about attacking AND about one of these — 'develop "
            "fast then attack the king' is better than 'attack the king' alone when both "
            "are true.\n\n"
            "Users cannot see the game — they are describing the style they want in "
            "advance. No moves, no squares, no move numbers, no result, no player names, "
            "nothing that only makes sense after the fact."
        ),
        user_template=_COMMON_TAIL.strip(),
    ),
]


def seed_population(include_extra: bool = False) -> list[Variant]:
    """The starting population.

    ``include_extra`` adds the second-round seeds derived from run 1's results.
    Off by default so the original six stay comparable across runs.
    """
    pool = SEEDS + EXTRA_SEEDS + EXTRA_SEEDS_2 if include_extra else SEEDS
    return [Variant.from_dict(v.to_dict()) for v in pool]
