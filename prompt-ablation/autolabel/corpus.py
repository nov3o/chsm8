"""Game loading, sampling and prompt-context construction.

Handles both CSV shapes present in ``data/``:

* the YottaChess dump (``2026-03_games.csv``) — top-100 players, has Elo,
  Opening and TimeControl, and a redundant ``1. Nf3 1... e6`` move format;
* the ``pgn_to_csv.py`` output (``games.csv``) — classical PGN headers.

Sampling is stratified by game length and decisiveness so a small evaluation
set still spans quiet grinds and 20-move slaughters. A prompt scored only on
sharp games will look great and generalise badly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import features as F

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


@dataclass
class Game:
    game_id: str
    moves_san: str
    white: str = ""
    black: str = ""
    result: str = ""
    white_elo: int | None = None
    black_elo: int | None = None
    opening: str = ""
    eco: str = ""
    time_control: str = ""
    date: str = ""
    event: str = ""
    _feat: F.GameFeatures | None = field(default=None, repr=False, compare=False)

    @property
    def feat(self) -> F.GameFeatures:
        if self._feat is None:
            self._feat = F.extract(
                self.moves_san,
                game_id=self.game_id,
                result=self.result,
                white_elo=self.white_elo,
                black_elo=self.black_elo,
                opening=self.opening,
                eco=self.eco,
                time_control=self.time_control,
            )
        return self._feat

    def player(self, color: str) -> str:
        return self.white if color == "white" else self.black

    def outcome_for(self, color: str) -> str:
        r = (self.result or "").lower()
        if "draw" in r or "1/2" in r:
            return "draw"
        won = ("white won" in r or r.strip() == "1-0")
        if color == "black":
            won = ("black won" in r or r.strip() == "0-1")
        return "won" if won else "lost"


def _int(v) -> int | None:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[Game]:
    """Load either CSV dialect, detected from the header row."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = set(rows[0])
    games: list[Game] = []

    if {"Moves", "White", "Black"} <= cols:  # YottaChess dump
        for i, r in enumerate(rows):
            gid = hashlib.sha1(
                f"{r.get('White')}|{r.get('Black')}|{r.get('UTCDate')}|"
                f"{r.get('UTCTime')}|{i}".encode()
            ).hexdigest()[:12]
            games.append(Game(
                game_id=gid,
                moves_san=r.get("Moves", ""),
                white=r.get("White", ""),
                black=r.get("Black", ""),
                result=r.get("Result", ""),
                white_elo=_int(r.get("WhiteElo")),
                black_elo=_int(r.get("BlackElo")),
                opening=r.get("Opening", ""),
                time_control=r.get("TimeControl", ""),
                date=r.get("UTCDate", ""),
            ))
    elif {"moves_san", "game_id"} <= cols:  # pgn_to_csv output
        for r in rows:
            games.append(Game(
                game_id=r["game_id"],
                moves_san=r.get("moves_san", ""),
                white=r.get("white", ""),
                black=r.get("black", ""),
                result=r.get("result", ""),
                opening="",
                eco=r.get("eco", ""),
                date=r.get("date", ""),
                event=r.get("event", ""),
            ))
    else:
        raise ValueError(f"unrecognised CSV columns in {path}: {sorted(cols)}")
    return games


def load(paths: list[Path] | None = None) -> list[Game]:
    paths = paths or [DATA / "2026-03_games.csv"]
    out: list[Game] = []
    for p in paths:
        out.extend(load_csv(Path(p)))
    return out


def sample(
    games: list[Game],
    n: int,
    seed: int = 0,
    *,
    min_moves: int = 15,
    max_moves: int = 140,
    stratify: bool = True,
) -> list[Game]:
    """Pick ``n`` games spanning short/medium/long and decisive/drawn.

    Games that fail to replay cleanly are dropped — a malformed mainline would
    otherwise silently poison the feature ground truth.
    """
    rng = random.Random(seed)
    pool = [g for g in games if g.moves_san]
    rng.shuffle(pool)

    buckets: dict[tuple, list[Game]] = {}
    checked = 0
    for g in pool:
        if checked > max(400, n * 12):
            break
        checked += 1
        nm = len(F.parse_san(g.moves_san)) // 2
        if not (min_moves <= nm <= max_moves):
            continue
        if g.feat.parse_error:
            continue
        if not stratify:
            buckets.setdefault(("all",), []).append(g)
        else:
            length = "short" if nm < 30 else ("mid" if nm < 60 else "long")
            buckets.setdefault((length, g.feat.decisive), []).append(g)
        if sum(len(v) for v in buckets.values()) >= n * 3:
            break

    keys = sorted(buckets, key=str)
    out: list[Game] = []
    i = 0
    while len(out) < n and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            out.append(buckets[k].pop())
        i += 1
    return out[:n]


# -- context construction --------------------------------------------------

MOVE_WINDOW_HELP = (
    "Moves are given in standard algebraic notation from the start of the game."
)


def move_text(g: Game, *, mode: str = "full", max_plies: int = 80) -> str:
    """Render the mainline for the prompt.

    ``mode``: ``full`` (whole game), ``opening`` (first ``max_plies`` plies),
    ``none`` (omit — feature card only).
    """
    if mode == "none":
        return ""
    toks = F.parse_san(g.moves_san)
    if mode == "opening":
        toks = toks[:max_plies]
    parts = []
    for i, t in enumerate(toks):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(t)
    text = " ".join(parts)
    if mode == "opening" and len(F.parse_san(g.moves_san)) > max_plies:
        text += " ..."
    return text


def build_context(
    g: Game,
    color: str,
    *,
    card: str = "full",
    moves: str = "full",
    anonymise: bool = True,
    reveal_result: bool = False,
) -> str:
    """Assemble what the labelling model sees for one (game, side).

    ``anonymise`` strips player names — otherwise strong models write biography
    of the famous player instead of describing the play, which is exactly the
    failure visible in the committed ``responses.csv``.
    """
    blocks: list[str] = []

    # Unconditional and first, independent of card/moves settings. render_card
    # also states this (needed there too, since metrics.py's grounding judge
    # calls it directly without going through build_context), but that line
    # only exists when card != "none" — the one ablation variant that turns
    # the card off (s5-moves-only) previously had NO side declaration
    # anywhere in the message, meaning the model had to infer which side it
    # was describing from context alone. This line is the one thing that must
    # always be present, in every configuration, for every call.
    blocks.append(f"You are looking at the {color.upper()} player.")

    head = []
    if not anonymise:
        head.append(f'White: {g.white or "?"}   Black: {g.black or "?"}')
    if reveal_result:
        head.append(f"Result: {g.result} ({color} {g.outcome_for(color)})")
    if head:
        blocks.append("\n".join(head))

    mt = move_text(g, mode=moves)
    if mt:
        blocks.append(f"MOVES ({MOVE_WINDOW_HELP})\n{mt}")

    cardtext = F.render_card(g.feat, color, style=card)
    if cardtext:
        blocks.append("MEASURED FROM THE BOARD\n" + cardtext)

    return "\n\n".join(blocks)
