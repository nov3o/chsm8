"""Automatic quality metrics for a set of generated labels.

This replaces the human pairwise-voting UI. A human comparing two paragraphs
judges fluency, which is not the thing that matters — a beautifully written
label that fits every game teaches a CLIP model nothing. What matters is
whether the text carries recoverable, game-specific, style-shaped signal.

Seven measurements, combined into one fitness score:

``yield``
    Did the model return parseable labels of roughly the demanded lengths?
    A gate, not a virtue: a model that cannot follow the format is unusable
    at corpus scale regardless of how good its prose is.
``distinct`` (M1)
    Embed every label. Does a label retrieve *its own game's other labels*
    ahead of every other game's? Measured across length bands, so the model
    cannot win by matching paragraph length. This is the single best proxy
    for "CLIP can learn something from this" and it costs a fraction of a
    cent per thousand labels.
``ground`` (M2)
    Show a judge one label and four candidate games; can it pick the right
    one? Catches fluent hallucination that embeddings happily encode.
``clean`` (M3)
    Hard leakage: move text, squares, move numbers, results, player names,
    past-tense narration. None of it exists when a user types a prompt, and
    all of it is a shortcut CLIP would take instead of learning style.
    Opening names and structures are counted separately as *soft* content and
    deliberately not penalised — they may well help.
``prompty`` (M4)
    Does it read as an instruction a user would type, rather than a report?
``fidelity`` (M5)
    Extract a style-tag vector from the label text alone, then correlate each
    tag against the board feature it should track, across the sample. This is
    the deepest metric: it asks whether the words are anchored to measurable
    reality rather than being plausible chess noise.
``truth`` (M6)
    Individual labels that assert something the board flatly contradicts —
    "sacrifice for the initiative" about a game with no sacrifice, "castle
    short" when the player castled long. ``fidelity`` is a population
    correlation and averages straight past these; they are the labels that
    would actively teach a wrong association. See ``contradictions.py``.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats

from . import contradictions, corpus, features as F, genome, providers
from .generate import LabelSet

EMBED_MODEL = "openai:text-embedding-3-small"

# -- M3: leakage patterns --------------------------------------------------

HARD_PATTERNS: dict[str, re.Pattern] = {
    # Transcribed moves: Nf3, Bxc6, exd5, Qh4+, O-O. Unlike a bare square name
    # this is the game being read back, and CLIP would latch straight onto it.
    "move_text": re.compile(
        r"\b(?:[KQRBN][a-h1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8]|O-O(?:-O)?|0-0(?:-0)?)\b"
    ),
    "move_number": re.compile(r"\b(?:move|ply)\s+\d+|\b\d+\s*\.\s*[A-Za-z]"),
    # Only phrases that reveal a finished game. Bare "victory"/"defeat"/"win"
    # are legitimate in a directive ("play for the win"), so they are not here.
    "outcome": re.compile(
        r"\b(?:resigned|was checkmated|the result|final position|"
        r"(?:won|lost|drew)\s+(?:the|this)\s+(?:game|match)|ended in a draw)\b", re.I
    ),
    "narration": re.compile(
        r"\b(?:this game|this player|in the game|the game (?:was|ended)|throughout the game|"
        r"ultimately|eventually|proved to be|demonstrated|displayed|showcased|"
        r"his opponent|her opponent|their opponent played)\b", re.I
    ),
    "meta": re.compile(
        r"\b(?:the (?:white|black) player|as (?:white|black),|the annotator|"
        r"based on the (?:measurements|features|data)|the measurements? (?:show|indicate))\b", re.I
    ),
}

_SQUARE_RE = re.compile(r"\b[a-h][1-8]\b")


def _square_spam(text: str) -> int:
    """Three or more distinct squares in one label reads as transcription.

    Naming a square or two is normal chess English — "fight for d4", "put a
    knight on f5" — and a user could easily type it, so it is soft content.
    A label that lists half the board is reciting the game instead of
    describing a style, and that is hard leakage.
    """
    return 1 if len(set(_SQUARE_RE.findall(text))) >= 3 else 0


SOFT_PATTERNS: dict[str, re.Pattern] = {
    # Ordinary chess vocabulary, reported so its effect can be ablated but
    # deliberately not penalised — naming squares or openings may well help.
    "square_ref": _SQUARE_RE,
    "opening_name": re.compile(
        r"\b(?:Sicilian|French|Caro-Kann|Ruy Lopez|Italian|Scotch|Petroff|Pirc|Modern|"
        r"Alekhine|Scandinavian|Nimzo-?Indian|King'?s Indian|Queen'?s Indian|Gr[uü]nfeld|"
        r"Benoni|Benko|Slav|Semi-Slav|Catalan|London|Tromp\w*|English Opening|Bird'?s|"
        r"R[eé]ti|Nimzo-?Larsen|Vienna|Danish|Evans|Najdorf|Dragon|Sveshnikov|Taimanov|"
        r"Sozin|Fianchetto Variation|Stonewall|Colle|Queen'?s Gambit|King'?s Gambit)\b", re.I
    ),
    "eco_code": re.compile(r"\b[A-E]\d{2}\b"),
    "structure": re.compile(
        r"\b(?:isolated queen'?s pawn|IQP|hanging pawns|pawn chain|minority attack|"
        r"backward pawn|passed pawn|doubled pawns|fianchetto|closed centre|open file)\b", re.I
    ),
}

IMPERATIVE_START = re.compile(
    r"^\s*(?:play|attack|defend|keep|hold|push|trade|avoid|develop|castle|grab|take|"
    r"control|open|close|sacrifice|sac|go|aim|build|apply|pressure|exchange|simplify|"
    r"stay|be|don'?t|do not|never|always|prioriti[sz]e|focus|target|storm|advance|"
    r"restrain|maintain|seize|fight|win|create|force|bring|put|leave)\b", re.I
)


def leakage(text: str, game: corpus.Game | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """Count hard violations and soft content markers in one label."""
    hard = {k: len(p.findall(text)) for k, p in HARD_PATTERNS.items()}
    hard["square_spam"] = _square_spam(text)
    soft = {k: len(p.findall(text)) for k, p in SOFT_PATTERNS.items()}
    if game is not None:
        names = set()
        for nm in (game.white, game.black):
            for tok in re.split(r"[\s_\-.]+", nm or ""):
                if len(tok) >= 4 and tok.isalpha():
                    names.add(tok.lower())
        hits = sum(1 for tok in re.findall(r"[A-Za-z]{4,}", text)
                   if tok.lower() in names)
        hard["player_name"] = hits
        if game.opening:
            op_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", game.opening)}
            op_tokens -= {"attack", "defense", "defence", "variation", "opening",
                          "system", "gambit", "game", "line", "with", "main"}
            soft["opening_match"] = sum(
                1 for tok in re.findall(r"[A-Za-z]{4,}", text)
                if tok.lower() in op_tokens
            )
    return hard, soft


# -- M5: style tags --------------------------------------------------------

# tag -> (feature name, sign). Sign is the direction the feature should move
# as the tag value rises.
TAG_FEATURES: dict[str, list[tuple[str, int]]] = {
    "aggression": [("king_pressure", +1), ("check_rate", +1), ("storm_pushes", +1)],
    "sacrifice_willingness": [("see_sacs", +1), ("sac_material", +1), ("max_deficit", +1)],
    "king_attack": [("king_pressure", +1), ("storm_pushes", +1)],
    "material_greed": [("sac_material", -1), ("max_deficit", -1)],
    "positional_maneuvering": [("capture_rate", -1), ("n_moves", +1), ("mobility", +1)],
    "defensive_solidity": [("retreat_rate", +1), ("storm_pushes", -1), ("king_pressure", -1)],
    "development_speed": [("development_ply", -1), ("home_pieces_mean", -1)],
    "center_control": [("center_attack", +1), ("center_occupancy", +1)],
    "pawn_storming": [("storm_pushes", +1), ("pawn_move_rate", +1)],
    "trade_seeking": [("trades_initiated", +1), ("endgame_frac", +1)],
    "endgame_orientation": [("endgame_frac", +1), ("n_moves", +1)],
    "risk_taking": [("material_volatility", +1), ("max_deficit", +1), ("king_exposure", +1)],
    "piece_activity": [("mobility", +1), ("center_attack", +1)],
    "keep_pieces_home": [("home_pieces_mean", +1), ("development_ply", +1)],
}
TAGS = list(TAG_FEATURES)

TAG_SYSTEM = (
    "You read a short instruction written for a chess player and score what it asks for. "
    "Score ONLY what the text actually asks for; if the text says nothing about an axis, "
    "use null — do not guess, and do not use your own chess opinions to fill gaps.\n\n"
    "Return ONLY a JSON object mapping each key below to a number in [0,1] or null.\n"
    + "\n".join(f"  {t}" for t in TAGS)
    + "\n\n0 means the instruction asks for the opposite, 0.5 neutral, 1 strongly asks for it. "
    "null means not addressed."
)


def parse_tags(text: str) -> dict[str, float | None]:
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    out: dict[str, float | None] = {}
    for t in TAGS:
        v = data.get(t)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[t] = max(0.0, min(1.0, float(v)))
        else:
            out[t] = None
    return out


# -- result container ------------------------------------------------------

@dataclass
class Scores:
    vid: str
    model: str
    n_units: int = 0
    n_labels: int = 0
    # gate
    parse_rate: float = 0.0
    band_compliance: float = 0.0
    dup_rate: float = 0.0
    mean_words: float = 0.0
    yield_score: float = 0.0
    # M1
    r1_cross_band: float = 0.0
    r1_same_band: float = 0.0
    intra_inter_gap: float = 0.0
    distinct_score: float = 0.0
    tag_collision_rate: float = 0.0  # informational, deliberately unpenalised
    # Near-identical labels written for *different* games. This is what a low
    # distinctiveness score actually looks like, and the optimiser needs to see
    # it as text — abstract numbers give it nothing to attack.
    confusions: list = field(default_factory=list)  # (cos, text_a, text_b)
    contradictions: list = field(default_factory=list, repr=False)
    # M2
    ground_acc: float = 0.0
    ground_n: int = 0
    # M3
    hard_rate: float = 0.0          # violations per label
    hard_breakdown: dict = field(default_factory=dict)
    soft_rate: float = 0.0
    soft_breakdown: dict = field(default_factory=dict)
    clean_score: float = 0.0
    # M6: individual labels that assert something the board contradicts.
    # fidelity is a population correlation and cannot see these.
    contradiction_rate: float = 0.0
    contradiction_counts: dict = field(default_factory=dict)
    truth_score: float = 0.0
    # M4
    imperative_rate: float = 0.0
    prompty_score: float = 0.0
    prompty_n: int = 0
    # M5
    fidelity: float = 0.0
    fidelity_detail: dict = field(default_factory=dict)
    tag_coverage: float = 0.0
    # composite
    fitness: float = 0.0
    latency_ms: float = 0.0
    notes: str = ""

    def row(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, dict)}


WEIGHTS = {
    "distinct_score": 0.26,
    "fidelity": 0.18,
    "ground_acc_norm": 0.18,
    "truth_score": 0.12,
    "prompty_score": 0.13,
    "clean_score": 0.13,
}


# -- helpers ---------------------------------------------------------------

def _unit_key(ls: LabelSet) -> tuple[str, str]:
    return (ls.game_id, ls.color)


def _cos_matrix(vecs: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vecs, axis=1, keepdims=True)
    n[n == 0] = 1.0
    v = vecs / n
    return v @ v.T


def _band_of(band: str, rungs: list[str]) -> str:
    return band if band in rungs else "unknown"


# -- the evaluator ---------------------------------------------------------

class Evaluator:
    def __init__(
        self,
        client: providers.Client,
        games: Sequence[corpus.Game],
        *,
        judge: str = "openai:gpt-5.2",
        tagger: str = "openai:gpt-5-mini",
        embed_model: str = EMBED_MODEL,
        # Judge calls dominate the cost of a generation: these counts are
        # multiplied by (variants x models). Kept modest by default; fidelity
        # needs the most samples because it is a rank correlation.
        ground_samples: int = 14,
        prompty_samples: int = 14,
        tag_samples: int = 56,
        seed: int = 0,
    ):
        self.client = client
        self.games = {g.game_id: g for g in games}
        self.judge = judge
        self.tagger = tagger
        self.embed_model = embed_model
        self.ground_samples = ground_samples
        self.prompty_samples = prompty_samples
        self.tag_samples = tag_samples
        self.rng = random.Random(seed)

    # -- gate ---------------------------------------------------------
    def _yield(self, sets: list[LabelSet], variant: genome.Variant, s: Scores) -> None:
        want = variant.n_labels
        s.n_units = len(sets)
        ok = [x for x in sets if x.ok]
        s.parse_rate = len(ok) / max(1, len(sets))
        all_labels = [(b, t) for x in ok for b, t in x.labels]
        s.n_labels = len(all_labels)
        if not all_labels:
            return
        s.mean_words = float(np.mean([len(t.split()) for _, t in all_labels]))
        s.latency_ms = float(np.mean([x.latency_ms for x in sets])) if sets else 0.0

        # band compliance: does each label's word count sit in its band's range?
        good = 0
        for band, text in all_labels:
            if band not in genome.RUNGS:
                continue
            lo, hi = (int(x) for x in re.findall(r"\d+", genome.RUNGS[band][0]))
            n = len(text.split())
            if lo * 0.6 <= n <= hi * 1.7:
                good += 1
        s.band_compliance = good / len(all_labels)

        # near-duplicate labels inside a unit are wasted training pairs
        dups = 0
        tot = 0
        for x in ok:
            seen = set()
            for _, t in x.labels:
                k = " ".join(sorted(re.findall(r"[a-z]{3,}", t.lower())))[:200]
                tot += 1
                if k in seen:
                    dups += 1
                seen.add(k)
        s.dup_rate = dups / max(1, tot)

        count_ok = float(np.mean([
            min(1.0, len(x.labels) / want) for x in ok
        ])) if ok else 0.0
        s.yield_score = (
            0.4 * s.parse_rate + 0.3 * count_ok
            + 0.2 * s.band_compliance + 0.1 * (1 - s.dup_rate)
        )

    # -- M1 -----------------------------------------------------------
    async def _distinct(self, sets: list[LabelSet], s: Scores) -> None:
        items: list[tuple[tuple[str, str], str, str]] = []  # (unit, band, text)
        for x in sets:
            if not x.ok:
                continue
            for band, text in x.labels:
                items.append((_unit_key(x), band, text))
        if len(items) < 8:
            return
        # Cap the embedding bill on very large sweeps.
        if len(items) > 3000:
            self.rng.shuffle(items)
            items = items[:3000]

        vecs = np.asarray(await self.client.embed([t for _, _, t in items],
                                                  model=self.embed_model))
        sim = _cos_matrix(vecs)
        np.fill_diagonal(sim, -2.0)
        # Dense integer ids rather than hashes: equality here decides whether a
        # retrieval counts as a hit, so a collision would silently inflate the
        # score. Also keeps results reproducible across PYTHONHASHSEED.
        uid = {u: i for i, u in enumerate(dict.fromkeys(u for u, _, _ in items))}
        bid = {b: i for i, b in enumerate(dict.fromkeys(b for _, b, _ in items))}
        units = np.array([uid[u] for u, _, _ in items])
        bands = np.array([bid[b] for _, b, _ in items])

        same_unit = units[:, None] == units[None, :]
        same_band = bands[:, None] == bands[None, :]

        # Cross-band retrieval: candidates must come from a different band, so
        # a model cannot win by matching paragraph length.
        #
        # Queries are restricted to the substantive bands. A two-word tag like
        # "attack the king" SHOULD match many games — that many-to-one mapping
        # is what a real user typing three words means, not a defect — and
        # scoring tag collisions as failures penalises precisely the terse
        # register we want. Tag collapse is reported separately instead.
        substantive = {b for b in ("short", "medium", "long") if b in bid}
        hits_cb, n_cb, hits_sb, n_sb = 0, 0, 0, 0
        for i in range(len(items)):
            if items[i][1] not in substantive:
                continue
            cb = ~same_band[i]
            cb[i] = False
            if cb.any() and same_unit[i][cb].any():
                j = int(np.argmax(np.where(cb, sim[i], -3.0)))
                hits_cb += int(same_unit[i, j])
                n_cb += 1
            sb = same_band[i].copy()
            sb[i] = False
            if sb.any() and same_unit[i][sb].any():
                j = int(np.argmax(np.where(sb, sim[i], -3.0)))
                hits_sb += int(same_unit[i, j])
                n_sb += 1
        s.r1_cross_band = hits_cb / max(1, n_cb)
        s.r1_same_band = hits_sb / max(1, n_sb)

        # Informational: how often two different games get a byte-identical
        # short label. Expected to be non-zero and not penalised.
        tag_items = [(u, t) for u, b, t in items if b in ("micro", "tag", "terse")]
        if tag_items:
            byunit: dict[str, set] = defaultdict(set)
            for u, t in tag_items:
                byunit[t.strip().lower()].add(u)
            shared = sum(1 for t, us in byunit.items() if len(us) > 1)
            s.tag_collision_rate = shared / max(1, len(byunit))

        mask = ~np.eye(len(items), dtype=bool)
        intra = sim[same_unit & mask]
        inter = sim[(~same_unit) & mask]
        s.intra_inter_gap = float(intra.mean() - inter.mean()) if intra.size and inter.size else 0.0

        # Collect the worst offenders: the most similar label pairs that belong
        # to different games. Fed verbatim to the optimiser so it can see the
        # collapse instead of being told a number.
        # Only substantive-band pairs count as confusions: two games sharing
        # the tag "attack the king" is expected, not a bug to hand the
        # optimiser.
        subs = np.array([items[i][1] in substantive for i in range(len(items))])
        cross = np.where(~same_unit & mask & subs[:, None] & subs[None, :], sim, -3.0)
        flat = np.argsort(cross, axis=None)[::-1]
        seen_pairs: set[tuple[int, int]] = set()
        for idx in flat[:400]:
            i, j = divmod(int(idx), len(items))
            if cross[i, j] < 0.6 or (min(i, j), max(i, j)) in seen_pairs:
                continue
            seen_pairs.add((min(i, j), max(i, j)))
            s.confusions.append((round(float(cross[i, j]), 3),
                                 items[i][2], items[j][2]))
            if len(s.confusions) >= 6:
                break

        # Chance for cross-band R@1 is roughly 1/(number of units).
        chance = 1.0 / max(2, len(uid))
        s.distinct_score = float(np.clip(
            0.75 * (s.r1_cross_band - chance) / (1 - chance)
            + 0.25 * np.clip(s.intra_inter_gap / 0.25, 0, 1),
            0, 1))

    # -- M2 -----------------------------------------------------------
    async def _ground(self, sets: list[LabelSet], s: Scores) -> None:
        pool = [(x, b, t) for x in sets if x.ok for b, t in x.labels
                if len(t.split()) >= 12]  # tags are too short to identify a game
        if not pool:
            return
        self.rng.shuffle(pool)
        pool = pool[:self.ground_samples]
        gids = list(self.games)
        jobs, truth = [], []
        for x, _band, text in pool:
            distract = [g for g in self.rng.sample(gids, min(6, len(gids)))
                        if g != x.game_id][:3]
            if len(distract) < 3:
                continue
            opts = distract + [x.game_id]
            self.rng.shuffle(opts)
            truth.append(opts.index(x.game_id))
            cards = "\n\n".join(
                f"OPTION {i + 1}\n" +
                F.render_card(self.games[g].feat, x.color, style="qual")
                for i, g in enumerate(opts)
            )
            msg = [
                {"role": "system", "content":
                 "You match a style instruction to the game it was written from. "
                 "Answer with a single digit 1-4 and nothing else."},
                {"role": "user", "content":
                 f"INSTRUCTION\n{text}\n\n{cards}\n\n"
                 "Which option's play best matches the instruction? Reply with one digit."},
            ]
            jobs.append((self.judge, msg, {"max_tokens": 2000}))
        if not jobs:
            return
        outs = await self.client.chat_many(jobs)
        hits, n = 0, 0
        for out, t in zip(outs, truth):
            if not out:
                continue
            m = re.search(r"[1-4]", out)
            if not m:
                continue
            n += 1
            hits += int(int(m.group()) - 1 == t)
        s.ground_n = n
        s.ground_acc = hits / max(1, n)

    # -- M3 -----------------------------------------------------------
    def _clean(self, sets: list[LabelSet], s: Scores) -> None:
        hard_tot: dict[str, int] = defaultdict(int)
        soft_tot: dict[str, int] = defaultdict(int)
        n = 0
        for x in sets:
            if not x.ok:
                continue
            g = self.games.get(x.game_id)
            for _band, text in x.labels:
                n += 1
                h, sf = leakage(text, g)
                for k, v in h.items():
                    hard_tot[k] += v
                for k, v in sf.items():
                    soft_tot[k] += v
        if not n:
            return
        s.hard_breakdown = {k: round(v / n, 4) for k, v in sorted(hard_tot.items())}
        s.soft_breakdown = {k: round(v / n, 4) for k, v in sorted(soft_tot.items())}
        s.hard_rate = sum(hard_tot.values()) / n
        s.soft_rate = sum(soft_tot.values()) / n
        s.clean_score = float(math.exp(-2.0 * s.hard_rate))

    # -- M6 -----------------------------------------------------------
    def _truth(self, sets: list[LabelSet], s: Scores) -> None:
        rate, counts, found = contradictions.audit(sets, self.games)
        s.contradiction_rate = rate
        s.contradiction_counts = counts
        s.contradictions = found[:8]
        # ~1 contradiction per 100 labels scores 0.74; 1 per 30 scores 0.37.
        s.truth_score = float(math.exp(-30.0 * rate))

    # -- M4 -----------------------------------------------------------
    async def _prompty(self, sets: list[LabelSet], s: Scores) -> None:
        pool = [t for x in sets if x.ok for _, t in x.labels]
        if not pool:
            return
        s.imperative_rate = float(np.mean([
            bool(IMPERATIVE_START.match(t)) for t in pool
        ]))
        self.rng.shuffle(pool)
        sub = pool[:self.prompty_samples]
        jobs = [(
            self.judge,
            [{"role": "system", "content":
              "You judge whether a piece of text works as a PROMPT a person would type "
              "into a chess AI to steer how it plays.\n"
              "Score 0-10 on: is it an instruction for future play (good) rather than a "
              "report on a finished game (bad); could a user have written it without "
              "having seen any game; is it specific enough to actually change how an "
              "engine plays. Natural terseness is fine and should not be penalised.\n"
              "Reply with one integer 0-10 and nothing else."},
             {"role": "user", "content": t}],
            {"max_tokens": 2000},
        ) for t in sub]
        outs = await self.client.chat_many(jobs)
        vals = []
        for o in outs:
            if not o:
                continue
            m = re.search(r"\b(10|\d)\b", o)
            if m:
                vals.append(int(m.group()) / 10.0)
        s.prompty_n = len(vals)
        s.prompty_score = float(np.mean(vals)) if vals else 0.0

    # -- M5 -----------------------------------------------------------
    async def _fidelity(self, sets: list[LabelSet], s: Scores) -> None:
        pool = [(x, t) for x in sets if x.ok for _, t in x.labels
                if len(t.split()) >= 8]
        if len(pool) < 12:
            return
        self.rng.shuffle(pool)
        pool = pool[:self.tag_samples]
        jobs = [(self.tagger,
                 [{"role": "system", "content": TAG_SYSTEM},
                  {"role": "user", "content": t}],
                 {"max_tokens": 2000}) for _x, t in pool]
        outs = await self.client.chat_many(jobs)

        rows: list[tuple[dict[str, float | None], dict[str, float]]] = []
        for (x, _t), out in zip(pool, outs):
            if not out:
                continue
            tags = parse_tags(out)
            if not tags:
                continue
            g = self.games.get(x.game_id)
            if g is None:
                continue
            rows.append((tags, g.feat.flat(x.color)))
        if len(rows) < 10:
            return

        covered = 0
        detail: dict[str, float] = {}
        rhos: list[float] = []
        for tag, feats in TAG_FEATURES.items():
            xs = [(r[0].get(tag), r[1]) for r in rows]
            present = [(v, f) for v, f in xs if v is not None]
            if len(present) < 8:
                continue
            covered += 1
            best = 0.0
            for fname, sign in feats:
                a = [v for v, f in present if fname in f]
                b = [f[fname] for v, f in present if fname in f]
                if len(a) < 8 or len(set(b)) < 3 or len(set(a)) < 2:
                    continue
                rho = stats.spearmanr(a, b).statistic
                if rho is None or math.isnan(rho):
                    continue
                signed = sign * rho
                if abs(signed) > abs(best):
                    best = signed
            if best:
                detail[tag] = round(best, 3)
                rhos.append(best)
        s.tag_coverage = covered / len(TAG_FEATURES)
        s.fidelity_detail = dict(sorted(detail.items(), key=lambda kv: -kv[1]))
        # Reward correctly-signed correlation only; wrong-signed tags are
        # actively misleading and score zero, not negative-then-averaged-away.
        s.fidelity = float(np.clip(np.mean([max(0.0, r) for r in rhos]) / 0.5, 0, 1)) if rhos else 0.0

    # -- driver -------------------------------------------------------
    async def score(
        self,
        sets: list[LabelSet],
        variant: genome.Variant,
        model: str,
        *,
        cheap: bool = False,
    ) -> Scores:
        """Score one (variant, model) cell.

        ``cheap=True`` runs only the free/near-free metrics (yield, leakage,
        embeddings) — used for the model bake-off, where paying a judge to
        evaluate an obviously broken model is waste.
        """
        s = Scores(vid=variant.vid, model=model)
        self._yield(sets, variant, s)
        self._clean(sets, s)
        self._truth(sets, s)
        if s.n_labels == 0:
            s.notes = "no parseable labels"
            return s
        await self._distinct(sets, s)
        if not cheap:
            await asyncio.gather(
                self._ground(sets, s),
                self._prompty(sets, s),
                self._fidelity(sets, s),
            )
        s.fitness = fitness(s, cheap=cheap)
        return s


def fitness(s: Scores, *, cheap: bool = False) -> float:
    """Weighted composite, gated by format yield."""
    if cheap:
        core = (0.55 * s.distinct_score + 0.25 * s.clean_score
                + 0.20 * s.truth_score)
        return round(float(s.yield_score * core), 4)
    ground_norm = float(np.clip((s.ground_acc - 0.25) / 0.75, 0, 1))
    parts = {
        "distinct_score": s.distinct_score,
        "fidelity": s.fidelity,
        "ground_acc_norm": ground_norm,
        "truth_score": s.truth_score,
        "prompty_score": s.prompty_score,
        "clean_score": s.clean_score,
    }
    core = sum(WEIGHTS[k] * v for k, v in parts.items())
    # Yield gates rather than dominates: 0.5 yield halves an otherwise great
    # prompt, because you cannot label ten million games with a 50% parse rate.
    return round(float(core * (0.35 + 0.65 * s.yield_score)), 4)
