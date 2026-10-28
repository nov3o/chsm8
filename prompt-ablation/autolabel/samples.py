"""Dump generated labels as plain readable text for human eyeballing.

The metrics decide what wins, but a human should still be able to look at a
game and the labels written for it without reading JSONL. Nothing here is
part of the optimisation loop — it exists so you can sanity-check that the
automatic scores are measuring something you agree with.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from . import corpus, features as F
from .generate import read_jsonl


def _short(model: str) -> str:
    return model.split(":", 1)[-1].split("/")[-1].replace("-Instruct", "")


def dump(
    labels_path: Path,
    *,
    n_games: int = 4,
    colors: tuple[str, ...] = ("white",),
    models: list[str] | None = None,
    vid: str | None = None,
    seed: int = 0,
    show_moves: bool = True,
    blind: bool = False,
) -> str:
    rows = read_jsonl(labels_path)
    if vid:
        rows = [r for r in rows if r.vid == vid]
    if models:
        rows = [r for r in rows if any(m in r.model for m in models)]
    rows = [r for r in rows if r.ok and r.color in colors]
    if not rows:
        return "no matching labels"

    games = {g.game_id: g for g in corpus.load()}
    by_unit: dict[tuple[str, str], list] = defaultdict(list)
    for r in rows:
        by_unit[(r.game_id, r.color)].append(r)

    rng = random.Random(seed)
    units = sorted(by_unit)
    rng.shuffle(units)
    units = units[:n_games]

    out: list[str] = []
    for gid, color in units:
        g = games.get(gid)
        out.append("=" * 78)
        if g is None:
            out.append(f"GAME {gid}  ({color})")
        else:
            f = g.feat
            sf = f.side(color)
            out.append(f"GAME {gid}   describing: {color.upper()}")
            meta = [f"{f.n_moves} moves", g.result or "?"]
            if g.opening:
                meta.append(g.opening)
            if g.white_elo and g.black_elo:
                meta.append(f"{g.white_elo} vs {g.black_elo}")
            if f.time_class != "unknown":
                meta.append(f.time_class)
            out.append("  " + " | ".join(meta))
            out.append(
                f"  board: castled {sf.castled}"
                + (", opposite wings" if sf.opposite_castling else "")
                + f" | {sf.captures} captures | {sf.checks} checks"
                + f" | {sf.see_sacs} sacs ({sf.sac_material:.1f}p)"
                + f" | king pressure {sf.king_pressure:.1f}"
                + f" | pieces home {sf.home_pieces_mean:.1f}/7"
                + f" | centre {sf.center_attack:.1f}"
            )
            if show_moves:
                mv = corpus.move_text(g, mode="opening", max_plies=30)
                out.append(f"  moves: {mv}")
        out.append("")

        for i, r in enumerate(sorted(by_unit[(gid, color)], key=lambda x: x.model)):
            name = f"MODEL {chr(65 + i)}" if blind else _short(r.model)
            out.append(f"  --- {name} ---")
            for _band, text in r.labels:
                text = re.sub(r"\s+", " ", text).strip()
                out.append(f"    {text}")
            out.append("")
    if blind:
        out.append("-" * 78)
        out.append("(models blinded; key withheld so your rating is not anchored)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autolabel.samples", description=__doc__)
    ap.add_argument("labels", help="path to a labels.jsonl produced by a run")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--colors", default="white")
    ap.add_argument("--models", default="", help="comma-separated substrings to keep")
    ap.add_argument("--vid", default="", help="restrict to one prompt variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blind", action="store_true", help="hide model names")
    ap.add_argument("--no-moves", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    text = dump(
        Path(args.labels),
        n_games=args.games,
        colors=tuple(args.colors.split(",")),
        models=[m for m in args.models.split(",") if m] or None,
        vid=args.vid or None,
        seed=args.seed,
        show_moves=not args.no_moves,
        blind=args.blind,
    )
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
