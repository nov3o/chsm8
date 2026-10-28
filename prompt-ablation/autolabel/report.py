"""Markdown rendering for screen / generation / validation results."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .generate import LabelSet
from .genome import Variant
from .metrics import Scores


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    out = ["| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def _short(model: str) -> str:
    m = model.split(":", 1)[-1].split("/")[-1]
    return m[:32]


def screen_report(scored: Sequence[Scores], rows: Sequence[LabelSet]) -> str:
    lat: dict[str, list[float]] = defaultdict(list)
    fails: dict[str, int] = defaultdict(int)
    for r in rows:
        lat[r.model].append(r.latency_ms)
        if not r.ok:
            fails[r.model] += 1
    n_per_model = defaultdict(int)
    for r in rows:
        n_per_model[r.model] += 1

    table = _table(
        ["model", "fit", "yield", "parse", "distinct", "R@1", "clean", "hard/lbl",
         "words", "dupes", "lat_s", "fails"],
        [[
            _short(s.model), f"{s.fitness:.3f}", f"{s.yield_score:.2f}",
            f"{s.parse_rate:.2f}", f"{s.distinct_score:.3f}", f"{s.r1_cross_band:.2f}",
            f"{s.clean_score:.2f}", f"{s.hard_rate:.2f}", f"{s.mean_words:.0f}",
            f"{s.dup_rate:.2f}",
            f"{(sum(lat[s.model]) / max(1, len(lat[s.model]))) / 1000:.1f}",
            f"{fails[s.model]}/{n_per_model[s.model]}",
        ] for s in scored],
    )
    return f"\n### Model bake-off (cheap metrics only)\n\n{table}\n"


def generation_report(
    gen: int,
    ranked: Sequence[Scores],
    cells: dict[tuple[str, str], Scores],
    vmap: dict[str, Variant],
) -> str:
    rows = []
    for s in ranked:
        v = vmap.get(s.vid)
        rows.append([
            s.vid, f"{s.fitness:.3f}", f"{s.distinct_score:.3f}", f"{s.fidelity:.3f}",
            f"{s.ground_acc:.2f}", f"{s.truth_score:.2f}",
            f"{s.prompty_score:.2f}", f"{s.clean_score:.2f}",
            f"{s.yield_score:.2f}", f"{s.mean_words:.0f}",
            f"{v.card}/{v.moves}" if v else "?",
        ])
    table = _table(
        ["variant", "FIT", "distinct", "fidelity", "ground", "truth", "prompty",
         "clean", "yield", "words", "card/moves"],
        rows,
    )
    best = ranked[0] if ranked else None
    extra = ""
    if best:
        per_model = sorted(
            [(m, c.fitness) for (vid, m), c in cells.items() if vid == best.vid],
            key=lambda t: -t[1])
        extra = ("\n\nBest variant per model: "
                 + ", ".join(f"{_short(m)} {f:.3f}" for m, f in per_model))
        extra += (f"\nTag/terse collision rate (expected, unpenalised): "
                  f"{best.tag_collision_rate:.2f}")
        if best.confusions:
            cos, a, b = best.confusions[0]
            extra += (f"\nWorst confusion (cos {cos}) between two different games:"
                      f"\n    A: {a[:110]}\n    B: {b[:110]}")
        if best.fidelity_detail:
            extra += ("\nTop tag/board correlations: "
                      + ", ".join(f"{k} {v:+.2f}"
                                  for k, v in list(best.fidelity_detail.items())[:6]))
        hard = {k: v for k, v in best.hard_breakdown.items() if v}
        contra = {k: v for k, v in best.contradiction_counts.items() if v}
        extra += ("\nContradictions/label: "
                  f"{best.contradiction_rate:.4f}"
                  + (" — " + ", ".join(f"{k} {v}" for k, v in contra.items()) if contra else ""))
        extra += "\nRemaining hard leakage: " + (
            ", ".join(f"{k} {v}" for k, v in hard.items()) if hard else "none")
    return f"\n### Generation {gen}\n\n{table}{extra}\n"


def validation_report(
    agg: Sequence[Scores],
    cells: dict[tuple[str, str], Scores],
    vmap: dict[str, Variant],
    rows: Sequence[LabelSet],
) -> str:
    head = generation_report(0, agg, cells, vmap).replace(
        "### Generation 0", "### Held-out validation")
    if not agg:
        return head
    best_vid = agg[0].vid
    samples = [r for r in rows if r.vid == best_vid and r.ok][:2]
    blocks = []
    for r in samples:
        lines = "\n".join(f"  [{b:>7}] {t}" for b, t in r.labels)
        blocks.append(f"game {r.game_id} / {r.color} / {_short(r.model)}\n{lines}")
    v = vmap.get(best_vid)
    sysblock = f"\n\n#### Winning system prompt (`{best_vid}`)\n\n```\n{v.system}\n```\n" if v else ""
    return head + sysblock + "\n#### Sample output\n\n```\n" + "\n\n".join(blocks) + "\n```\n"
