"""Model bake-off and prompt evolution.

Two stages, run in order:

``screen_models``
    Fan the seed prompts over every candidate model on a handful of games and
    score with the free metrics only. Models that cannot emit parseable output,
    or whose labels are interchangeable mush, are dropped here — before any
    judge tokens get spent on them. This is the "if some models are completely
    dumb, dismiss those and go on" step.

``evolve``
    Hill-climb the prompt genome against the surviving models. Each generation:
    generate labels, score with the full metric suite, hand the worst failure
    cases to an optimiser LLM, take its rewrites plus the elites into the next
    generation. Games are resampled every generation so a prompt cannot win by
    fitting one particular set of games, and the final population is re-scored
    on a held-out set it was never optimised against.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from . import contradictions, corpus, generate, genome, metrics, providers, report

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

CANDIDATE_MODELS = [
    # paid, strong — the quality ceiling to measure the cheap ones against
    "openai:gpt-5.2",
    "openai:gpt-5-mini",
    "openai:gpt-5-nano",
    "xai:grok-4.5",
    "xai:grok-4.20-0309-non-reasoning",
    # free on the cluster — the ones that could actually label millions of games
    "swissai:meta-llama/Llama-3.3-70B-Instruct",
    "swissai:Qwen/Qwen3-32B",
    "swissai:Qwen/Qwen3.5-27B",
    "swissai:zai-org/GLM-4.7-Flash",
    "swissai:MiniMaxAI/MiniMax-M2.5",
    "swissai:google/gemma-4-31B-it",
    "swissai:mistralai/Ministral-3-14B-Instruct-2512",
]


# -- stage 1: model bake-off ----------------------------------------------

async def screen_models(
    client: providers.Client,
    *,
    candidates: Sequence[str] | None = None,
    n_games: int = 6,
    variants: Sequence[genome.Variant] | None = None,
    keep: int = 4,
    min_fitness: float = 0.05,
    seed: int = 11,
    out_dir: Path | None = None,
) -> tuple[list[str], list[metrics.Scores]]:
    """Rank candidate models on cheap metrics; return the survivors."""
    cands = providers.resolve(list(candidates or CANDIDATE_MODELS))
    variants = list(variants or genome.seed_population()[:3])
    games = corpus.sample(corpus.load(), n_games, seed=seed)
    print(f"[screen] {len(cands)} models x {len(variants)} variants x "
          f"{len(games)} games x 2 sides")

    rows = await generate.generate(client, variants, cands, games)
    ev = metrics.Evaluator(client, games, seed=seed)

    by_model: dict[str, list[generate.LabelSet]] = defaultdict(list)
    for r in rows:
        by_model[r.model].append(r)

    # Score each model on its pooled output across all seed variants: we are
    # asking "can this model do the job at all", not "which prompt suits it".
    pooled = genome.Variant(vid="screen", system="", user_template="",
                            rungs=variants[0].rungs,
                            n_per_rung=variants[0].n_per_rung)
    scored = await asyncio.gather(*(
        ev.score(by_model[m], pooled, m, cheap=True) for m in by_model
    ))
    scored = sorted(scored, key=lambda s: -s.fitness)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        generate.write_jsonl(rows, out_dir / "labels.jsonl")
        (out_dir / "scores.json").write_text(
            json.dumps([s.__dict__ for s in scored], indent=2, default=str))
        (out_dir / "report.md").write_text(report.screen_report(scored, rows))

    survivors = [s.model for s in scored if s.fitness >= min_fitness][:keep]
    dropped = [s.model for s in scored if s.model not in survivors]
    print(report.screen_report(scored, rows))
    if dropped:
        print(f"[screen] dropped: {', '.join(m.split(':', 1)[1] for m in dropped)}")
    return survivors, scored


# -- stage 2: prompt evolution --------------------------------------------

OPTIMISER_SYSTEM = """\
You improve prompts. A prompt here is a recipe that turns one side of one chess
game into a set of natural-language instructions — the instructions that would
have made an engine play that way. Those instructions become training labels for
a contrastive game-text model, and at inference a human types something similar
to steer how the engine plays.

You will be given the current prompt, its measured scores, and real examples of
its output including its worst failures. Propose concrete rewrites.

WHAT THE SCORES MEAN
- truth: individual labels that state something the board contradicts, e.g.
  "sacrifice for the initiative" about a game with no sacrifice, or "castle
  short" when the player castled long. Every one is a training pair that
  teaches a wrong association. Usually caused by the prompt inviting the model
  to embellish beyond what it was shown.
- distinct: do labels for different games differ? Embedding retrieval accuracy,
  normalised against chance. LOW means the prompt produces interchangeable
  generic chess prose, which teaches the downstream model nothing. This is the
  most important metric.
- fidelity: style words extracted from the label text correlate with what the
  board actually shows (sacrifices, king pressure, centre control, pieces left
  at home, development speed). LOW means the labels are plausible-sounding but
  unanchored — the model is writing chess-flavoured noise.
- ground: a judge shown the label and four candidate games picks the right one.
  LOW means the label does not identify its game.
- prompty: does it read like an instruction a user would type, rather than a
  report on a finished game? LOW means the output narrates the past.
- clean: freedom from things a user could never have written — concrete moves,
  square names, move numbers, results, player names, past-tense commentary.
  LOW means the labels leak game identity and the downstream model will take
  that shortcut instead of learning style.
- yield: parseable output, right number of labels, right lengths, no duplicates.

LEVERS YOU CAN PULL
- system: rewrite freely. Be directive and specific. Naming the failure you are
  fixing inside the instruction usually works better than general exhortation.
- card: "full" (numeric board measurements), "qual" (the same, qualitative
  bands only, no digits), "none" (no measurements).
- moves: "full" (whole game), "opening" (first 80 plies), "none".
- n_per_rung: how many labels per length band (1-3).

Note the interaction, which has been measured rather than assumed: card="full"
raises fidelity but tempts weak models into quoting measurements back into the
label. Downgrading to card="qual" stops the quoting and roughly halves fidelity
(0.24-0.38 against 0.56-0.65), because the numbers are what anchors a label to
its game. Keep the numeric card and forbid quoting in the instruction instead.

Return ONLY a JSON array of objects, no prose, no code fence:
[{"tag": "<short-kebab-name>", "rationale": "<one line: which metric, why>",
  "system": "<full replacement system prompt>",
  "card": "full|qual|none", "moves": "full|opening|none", "n_per_rung": 2}]
"""


def _failure_examples(
    rows: Sequence[generate.LabelSet],
    games: dict[str, corpus.Game],
    limit: int = 10,
) -> str:
    """The most instructive bad output: leaky labels first, then generic ones."""
    leaky: list[tuple[int, str, str]] = []
    plain: list[str] = []
    for r in rows:
        if not r.ok:
            plain.append(f"[UNPARSEABLE from {r.model.split(':')[-1]}] {r.error}")
            continue
        g = games.get(r.game_id)
        for band, text in r.labels:
            hard, _ = metrics.leakage(text, g)
            n = sum(hard.values())
            if n:
                worst = max(hard.items(), key=lambda kv: kv[1])[0]
                leaky.append((n, worst, f"[{band}] {text}"))
            else:
                plain.append(f"[{band}] {text}")
    out: list[str] = []
    leaky.sort(key=lambda t: -t[0])
    if leaky:
        out.append("LABELS WITH HARD LEAKAGE (these must stop):")
        out += [f"  ({w}) {t}" for _n, w, t in leaky[:limit]]
    if plain:
        out.append("")
        out.append("A RANDOM SAMPLE OF CLEAN LABELS (judge these for genericness):")
        step = max(1, len(plain) // limit)
        out += [f"  {t}" for t in plain[::step][:limit]]
    return "\n".join(out)


def _confusion_examples(score: metrics.Scores, limit: int = 5) -> str:
    """Show collapse as text.

    Across three generations the optimiser reliably improved leakage and
    promptability — the metrics it was given concrete failing examples for —
    while degrading distinctiveness and fidelity, which it only ever saw as
    abstract numbers. These are the pairs of near-identical labels written for
    *different* games, which is what a low distinctiveness score actually is.
    """
    if not score.confusions:
        return ""
    lines = ["",
             "PAIRS OF LABELS THIS PROMPT WROTE FOR **DIFFERENT** GAMES.",
             "They are nearly identical, so a contrastive model cannot tell the",
             "two games apart. This is the single biggest problem to fix:"]
    for cos, a, b in score.confusions[:limit]:
        lines.append(f"  cosine {cos}")
        lines.append(f"    game A: {a}")
        lines.append(f"    game B: {b}")
    return "\n".join(lines)


async def propose(
    client: providers.Client,
    parent: genome.Variant,
    score: metrics.Scores,
    rows: Sequence[generate.LabelSet],
    games: dict[str, corpus.Game],
    *,
    optimiser: str,
    n: int = 2,
    gen: int = 1,
) -> list[genome.Variant]:
    """Ask the optimiser LLM for ``n`` rewrites of one variant."""
    detail = ", ".join(f"{k}={v:+.2f}" for k, v in
                       list(score.fidelity_detail.items())[:8]) or "none measured"
    hard = ", ".join(f"{k}={v}" for k, v in score.hard_breakdown.items() if v) or "none"
    contra = ", ".join(f"{k}={v}" for k, v in score.contradiction_counts.items()) or "none"
    soft = ", ".join(f"{k}={v}" for k, v in score.soft_breakdown.items() if v) or "none"
    user = f"""\
CURRENT PROMPT (id {parent.vid}, generation {parent.generation})
presentation: card={parent.card} moves={parent.moves} n_per_rung={parent.n_per_rung}

system:
\"\"\"
{parent.system}
\"\"\"

MEASURED SCORES (0-1, higher is better)
  fitness   {score.fitness:.3f}
  distinct  {score.distinct_score:.3f}   (cross-band retrieval R@1 {score.r1_cross_band:.3f}, intra-inter gap {score.intra_inter_gap:+.3f})
  fidelity  {score.fidelity:.3f}   (tag coverage {score.tag_coverage:.2f}; per-tag correlations: {detail})
  ground    {score.ground_acc:.3f}   (4-way, chance 0.25, n={score.ground_n})
  prompty   {score.prompty_score:.3f}   (imperative openings {score.imperative_rate:.2f})
  clean     {score.clean_score:.3f}   (hard violations/label: {hard})
  truth     {score.truth_score:.3f}   ({score.contradiction_rate:.4f} contradictions/label: {contra})
  yield     {score.yield_score:.3f}   (parse {score.parse_rate:.2f}, band fit {score.band_compliance:.2f}, dupes {score.dup_rate:.2f}, mean {score.mean_words:.0f} words)
  soft content (NOT penalised, informational): {soft}

{_failure_examples(rows, games)}
{_confusion_examples(score)}
{contradictions.format_examples(score.contradictions)}

Propose exactly {n} rewrites. Make them differ from each other in strategy —
do not submit {n} variations of one idea. Target the lowest-scoring metrics.

Two warnings from previous runs, both measured:

- Do NOT switch the card to "qual" or "none" in order to stop the model
  quoting measurements. Every attempt to do that has cut fidelity roughly in
  half (0.24-0.38 against 0.56-0.65) because the numbers are what anchors the
  label to the actual game. Ban quoting in the instruction and keep the card.
- Rewrites that pile on prohibitions reliably raise `clean` and `prompty`
  while pushing `distinct` and `fidelity` down, and lose overall. A longer
  list of rules is not an improvement. If a rewrite does not plausibly make
  labels for two DIFFERENT games more different from each other, it is not
  worth proposing.
"""
    try:
        text = await client.chat(
            optimiser,
            [{"role": "system", "content": OPTIMISER_SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=6000, temperature=1.0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [optimise] {parent.vid}: optimiser failed ({e})")
        return []

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        props = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

    out: list[genome.Variant] = []
    for i, p in enumerate(props[:n]):
        if not isinstance(p, dict) or not isinstance(p.get("system"), str):
            continue
        tag = re.sub(r"[^a-z0-9-]", "", str(p.get("tag", f"m{i}")).lower())[:24] or f"m{i}"
        out.append(parent.child(
            vid=f"g{gen}-{parent.vid.split('-', 1)[-1][:8]}-{tag}",
            system=p["system"].strip(),
            card=p.get("card") if p.get("card") in ("full", "qual", "none") else parent.card,
            moves=p.get("moves") if p.get("moves") in ("full", "opening", "none") else parent.moves,
            n_per_rung=int(p["n_per_rung"]) if str(p.get("n_per_rung", "")).isdigit()
            and 1 <= int(p["n_per_rung"]) <= 3 else parent.n_per_rung,
            notes=str(p.get("rationale", ""))[:300],
        ))
    return out


async def evolve(
    client: providers.Client,
    models: Sequence[str],
    *,
    generations: int = 4,
    n_games: int = 8,
    population: Sequence[genome.Variant] | None = None,
    elites: int = 3,
    children_per_elite: int = 2,
    optimiser: str = "openai:gpt-5.2",
    judge: str = "openai:gpt-5.2",
    tagger: str = "openai:gpt-5-mini",
    run_dir: Path | None = None,
    seed: int = 100,
) -> tuple[list[genome.Variant], list[metrics.Scores]]:
    """Run the optimisation loop. Returns the final population and its scores."""
    run_dir = run_dir or (RUNS / time.strftime("evolve-%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    pop = list(population or genome.seed_population())
    all_games = corpus.load()
    history: list[dict] = []
    best_overall: tuple[float, genome.Variant, metrics.Scores] | None = None

    for gen in range(1, generations + 1):
        gdir = run_dir / f"gen{gen}"
        gdir.mkdir(exist_ok=True)
        # Fresh games every generation: otherwise the optimiser fits the sample.
        games = corpus.sample(all_games, n_games, seed=seed + gen * 977)
        gmap = {g.game_id: g for g in games}
        print(f"\n{'=' * 70}\n[gen {gen}/{generations}] {len(pop)} variants, "
              f"{len(models)} models, {len(games)} games\n{'=' * 70}")

        rows = await generate.generate(client, pop, models, games)
        generate.write_jsonl(rows, gdir / "labels.jsonl")

        ev = metrics.Evaluator(client, games, judge=judge, tagger=tagger, seed=seed + gen)
        cells: dict[tuple[str, str], list[generate.LabelSet]] = defaultdict(list)
        for r in rows:
            cells[(r.vid, r.model)].append(r)
        vmap = {v.vid: v for v in pop}

        n_judge = len(cells) * (ev.ground_samples + ev.prompty_samples + ev.tag_samples)
        print(f"  [score] {len(cells)} (variant, model) cells, "
              f"up to ~{n_judge} judge calls")
        cell_scores = await asyncio.gather(*(
            ev.score(v, vmap[vid], model) for (vid, model), v in cells.items()
        ))
        by_cell = dict(zip(cells.keys(), cell_scores))
        print(f"  [score] done — {client.failure_summary()}")

        # A variant's fitness is its mean across models: we want a prompt that
        # works, not one that happens to suit a single model's quirks.
        agg: dict[str, metrics.Scores] = {}
        for v in pop:
            mine = [by_cell[(v.vid, m)] for m in models if (v.vid, m) in by_cell]
            if not mine:
                continue
            agg[v.vid] = _mean_scores(v.vid, mine)

        ranked = sorted(agg.values(), key=lambda s: -s.fitness)
        print(report.generation_report(gen, ranked, by_cell, vmap))
        (gdir / "scores.json").write_text(json.dumps(
            {"aggregate": [s.__dict__ for s in ranked],
             "cells": {f"{k[0]}|{k[1]}": v.__dict__ for k, v in by_cell.items()}},
            indent=2, default=str))
        (gdir / "population.json").write_text(json.dumps(
            [v.to_dict() for v in pop], indent=2))
        (gdir / "report.md").write_text(
            report.generation_report(gen, ranked, by_cell, vmap))
        history.append({"gen": gen, "best": ranked[0].fitness if ranked else 0.0,
                        "best_vid": ranked[0].vid if ranked else "",
                        "mean": float(np.mean([s.fitness for s in ranked])) if ranked else 0.0})

        if ranked and (best_overall is None or ranked[0].fitness > best_overall[0]):
            best_overall = (ranked[0].fitness, vmap[ranked[0].vid], ranked[0])

        if gen == generations:
            pop = [vmap[s.vid] for s in ranked]
            break

        # Breed: elites survive, each spawns children targeting its weak metrics.
        keep = [vmap[s.vid] for s in ranked[:elites]]
        rows_by_vid: dict[str, list[generate.LabelSet]] = defaultdict(list)
        for r in rows:
            rows_by_vid[r.vid].append(r)
        print(f"  [optimise] breeding from {', '.join(v.vid for v in keep)}")
        broods = await asyncio.gather(*(
            propose(client, v, agg[v.vid], rows_by_vid[v.vid], gmap,
                    optimiser=optimiser, n=children_per_elite, gen=gen + 1)
            for v in keep
        ))
        children = [c for brood in broods for c in brood]
        print(f"  [optimise] {len(children)} children: "
              + ", ".join(c.vid for c in children))
        pop = keep + children

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = [agg[v.vid] for v in pop if v.vid in agg]
    (run_dir / "best.json").write_text(json.dumps(
        {"variant": pop[0].to_dict() if pop else None,
         "scores": final[0].__dict__ if final else None,
         "history": history}, indent=2, default=str))
    return pop, final


def _mean_scores(vid: str, cells: Sequence[metrics.Scores]) -> metrics.Scores:
    """Average a variant's per-model cells into one score row."""
    out = metrics.Scores(vid=vid, model=f"mean of {len(cells)}")
    numeric = [k for k, v in cells[0].__dict__.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for k in numeric:
        setattr(out, k, float(np.mean([getattr(c, k) for c in cells])))
    out.n_units = int(sum(c.n_units for c in cells))
    out.n_labels = int(sum(c.n_labels for c in cells))
    merged: dict[str, list[float]] = defaultdict(list)
    for c in cells:
        for k, v in c.fidelity_detail.items():
            merged[k].append(v)
    out.fidelity_detail = {k: round(float(np.mean(v)), 3)
                           for k, v in sorted(merged.items(),
                                              key=lambda kv: -np.mean(kv[1]))}
    for field_name in ("hard_breakdown", "soft_breakdown"):
        acc: dict[str, list[float]] = defaultdict(list)
        for c in cells:
            for k, v in getattr(c, field_name).items():
                acc[k].append(v)
        setattr(out, field_name,
                {k: round(float(np.mean(v)), 4) for k, v in sorted(acc.items())})
    return out


async def validate(
    client: providers.Client,
    variants: Sequence[genome.Variant],
    models: Sequence[str],
    *,
    n_games: int = 12,
    seed: int = 424242,
    judge: str = "openai:gpt-5.2",
    tagger: str = "openai:gpt-5-mini",
    out_dir: Path | None = None,
) -> list[metrics.Scores]:
    """Re-score variants on games never used during optimisation."""
    games = corpus.sample(corpus.load(), n_games, seed=seed)
    print(f"[validate] {len(variants)} variants x {len(models)} models "
          f"x {len(games)} held-out games")
    rows = await generate.generate(client, variants, models, games)
    ev = metrics.Evaluator(client, games, judge=judge, tagger=tagger, seed=seed)
    cells: dict[tuple[str, str], list[generate.LabelSet]] = defaultdict(list)
    for r in rows:
        cells[(r.vid, r.model)].append(r)
    vmap = {v.vid: v for v in variants}
    scored = await asyncio.gather(*(
        ev.score(v, vmap[vid], model) for (vid, model), v in cells.items()
    ))
    by_cell = dict(zip(cells.keys(), scored))
    agg = []
    for v in variants:
        mine = [by_cell[(v.vid, m)] for m in models if (v.vid, m) in by_cell]
        if mine:
            agg.append(_mean_scores(v.vid, mine))
    agg.sort(key=lambda s: -s.fitness)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        generate.write_jsonl(rows, out_dir / "labels.jsonl")
        (out_dir / "scores.json").write_text(
            json.dumps([s.__dict__ for s in agg], indent=2, default=str))
        (out_dir / "report.md").write_text(
            report.validation_report(agg, by_cell, vmap, rows))
    print(report.validation_report(agg, by_cell, vmap, rows))
    return agg
