"""CLI: reverse-engineer the labelling prompt.

    python -m autolabel.run screen                 # which models can do this at all
    python -m autolabel.run evolve --generations 4 # optimise the prompt
    python -m autolabel.run validate --run runs/evolve-...
    python -m autolabel.run label  --run runs/evolve-... --n 500 --out labels.jsonl
    python -m autolabel.run inspect --run runs/evolve-...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import corpus, generate, genome, loop, metrics, providers, report

RUNS = loop.RUNS


def _client(args) -> providers.Client:
    return providers.Client(concurrency=args.concurrency, use_cache=not args.no_cache)


async def cmd_screen(args) -> int:
    client = _client(args)
    out = RUNS / (args.name or "screen")
    survivors, _ = await loop.screen_models(
        client, n_games=args.games, keep=args.keep, out_dir=out,
        candidates=args.models.split(",") if args.models else None,
    )
    (out / "survivors.json").write_text(json.dumps(survivors, indent=2))
    print(f"\nSurvivors -> {out / 'survivors.json'}")
    print("  " + "\n  ".join(survivors))
    await client.aclose()
    return 0


async def cmd_evolve(args) -> int:
    client = _client(args)
    if args.models:
        models = providers.resolve(args.models.split(","))
    else:
        sfile = RUNS / "screen" / "survivors.json"
        if sfile.exists():
            models = providers.resolve(json.loads(sfile.read_text()))
            print(f"[evolve] using {len(models)} survivors from {sfile}")
        else:
            print("[evolve] no screen results; running a quick bake-off first")
            models, _ = await loop.screen_models(
                client, n_games=4, keep=args.keep, out_dir=RUNS / "screen")
    if not models:
        print("no usable models", file=sys.stderr)
        return 1

    run_dir = RUNS / args.name if args.name else None
    pop, scores = await loop.evolve(
        client, models,
        population=genome.seed_population(include_extra=args.extra_seeds),
        generations=args.generations, n_games=args.games,
        elites=args.elites, children_per_elite=args.children,
        optimiser=args.optimiser, judge=args.judge, tagger=args.tagger,
        run_dir=run_dir,
    )
    print(f"\nBest variant: {pop[0].vid if pop else 'none'}")
    await client.aclose()
    return 0


def _load_best(run: Path) -> genome.Variant:
    best = json.loads((run / "best.json").read_text())
    if best.get("variant"):
        return genome.Variant.from_dict(best["variant"])
    gens = sorted(run.glob("gen*/population.json"))
    pop = json.loads(gens[-1].read_text())
    return genome.Variant.from_dict(pop[0])


async def cmd_validate(args) -> int:
    client = _client(args)
    run = Path(args.run)
    pops = sorted(run.glob("gen*/population.json"))
    if not pops:
        print(f"no populations under {run}", file=sys.stderr)
        return 1
    variants = [genome.Variant.from_dict(d)
                for d in json.loads(pops[-1].read_text())][:args.top]
    models = providers.resolve(args.models.split(",")) if args.models else \
        providers.resolve(json.loads((RUNS / "screen" / "survivors.json").read_text()))
    await loop.validate(client, variants, models, n_games=args.games,
                        judge=args.judge, tagger=args.tagger,
                        out_dir=run / "validation")
    await client.aclose()
    return 0


async def cmd_label(args) -> int:
    """Produce the actual training labels with the winning recipe."""
    client = _client(args)
    variant = _load_best(Path(args.run))
    models = providers.resolve([args.model]) if args.model else \
        providers.resolve(json.loads((RUNS / "screen" / "survivors.json").read_text()))[:1]
    games = corpus.sample(corpus.load(), args.n, seed=args.seed)
    print(f"[label] variant={variant.vid} model={models[0]} games={len(games)}")
    rows = await generate.generate(client, [variant], models, games)
    out = Path(args.out)
    generate.write_jsonl(rows, out)
    ok = sum(1 for r in rows if r.ok)
    n_lab = sum(len(r.labels) for r in rows)
    print(f"[label] {ok}/{len(rows)} sides labelled, {n_lab} labels -> {out}")
    await client.aclose()
    return 0


async def cmd_rescore(args) -> int:
    """Re-apply the metrics to labels already on disk.

    Generation is the expensive, non-deterministic part. When a metric is
    recalibrated — as the leakage rules were — this replays the scoring over
    stored labels instead of paying to regenerate them.
    """
    client = _client(args)
    path = Path(args.labels)
    rows = generate.read_jsonl(path)
    if not rows:
        print(f"no labels in {path}", file=sys.stderr)
        return 1
    gids = {r.game_id for r in rows}
    games = [g for g in corpus.load() if g.game_id in gids]

    pop_file = path.parent / "population.json"
    vmap = {}
    if pop_file.exists():
        vmap = {v["vid"]: genome.Variant.from_dict(v)
                for v in json.loads(pop_file.read_text())}

    ev = metrics.Evaluator(client, games, judge=args.judge, tagger=args.tagger)
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        cells[(r.vid, r.model) if args.by == "cell" else ("pooled", r.model)].append(r)

    fallback = genome.Variant(vid="pooled", system="", user_template="")
    scored = await asyncio.gather(*(
        ev.score(v, vmap.get(vid, fallback), model, cheap=args.cheap)
        for (vid, model), v in cells.items()
    ))
    scored = sorted(scored, key=lambda s: -s.fitness)
    print(report.screen_report(scored, rows))
    for s in scored:
        hard = {k: v for k, v in s.hard_breakdown.items() if v}
        if hard:
            print(f"  {s.model.split('/')[-1][:30]:<32} hard={hard}")
    if args.out:
        Path(args.out).write_text(json.dumps([s.__dict__ for s in scored],
                                             indent=2, default=str))
        print(f"\nwrote {args.out}")
    await client.aclose()
    return 0


async def cmd_inspect(args) -> int:
    run = Path(args.run)
    for p in sorted(run.glob("gen*/report.md")) + sorted(run.glob("*/report.md")):
        print(p.read_text())
    hist = run / "history.json"
    if hist.exists():
        print("\n### Fitness history\n")
        for h in json.loads(hist.read_text()):
            print(f"  gen {h['gen']}: best {h['best']:.3f} ({h['best_vid']}), "
                  f"mean {h['mean']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="autolabel.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--judge", default="openai:gpt-5.2")
    ap.add_argument("--tagger", default="openai:gpt-5-mini")
    ap.add_argument("--optimiser", default="openai:gpt-5.2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="rank candidate labelling models")
    s.add_argument("--games", type=int, default=6)
    s.add_argument("--keep", type=int, default=4)
    s.add_argument("--models", default="", help="comma-separated override")
    s.add_argument("--name", default="screen")
    s.set_defaults(fn=cmd_screen)

    e = sub.add_parser("evolve", help="optimise the labelling prompt")
    e.add_argument("--generations", type=int, default=4)
    e.add_argument("--games", type=int, default=8)
    e.add_argument("--elites", type=int, default=3)
    e.add_argument("--children", type=int, default=2)
    e.add_argument("--keep", type=int, default=3)
    e.add_argument("--models", default="")
    e.add_argument("--name", default="")
    e.add_argument("--extra-seeds", action="store_true",
                   help="include the second-round seeds derived from run 1")
    e.set_defaults(fn=cmd_evolve)

    v = sub.add_parser("validate", help="re-score on held-out games")
    v.add_argument("--run", required=True)
    v.add_argument("--games", type=int, default=12)
    v.add_argument("--top", type=int, default=4)
    v.add_argument("--models", default="")
    v.set_defaults(fn=cmd_validate)

    l = sub.add_parser("label", help="generate training labels with the winner")
    l.add_argument("--run", required=True)
    l.add_argument("--n", type=int, default=100)
    l.add_argument("--out", default="data/labels.jsonl")
    l.add_argument("--model", default="")
    l.add_argument("--seed", type=int, default=2024)
    l.set_defaults(fn=cmd_label)

    rs = sub.add_parser("rescore", help="re-apply metrics to stored labels")
    rs.add_argument("labels", help="path to a labels.jsonl")
    rs.add_argument("--by", choices=["cell", "model"], default="model")
    rs.add_argument("--cheap", action="store_true", help="skip judge-backed metrics")
    rs.add_argument("--out", default="")
    rs.set_defaults(fn=cmd_rescore)

    i = sub.add_parser("inspect", help="print reports from a run")
    i.add_argument("--run", required=True)
    i.set_defaults(fn=cmd_inspect)

    args = ap.parse_args(argv)
    # Runs are long and usually launched with output redirected, where Python
    # block-buffers stdout and a healthy run looks indistinguishable from a
    # hung one for minutes at a time.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
