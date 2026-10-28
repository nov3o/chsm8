#!/usr/bin/env python3
"""Unblind and aggregate subagent judge verdicts.

usage: judge_aggregate.py <judge_out dir> <key.json>
Prints, per source: hallucinated labels per 100 labels, games with any hallucination,
mean specificity / vibe (1-5), mean rank, win rate vs the teacher, and lists hallucinations.
"""
import collections
import json
import sys
from pathlib import Path

out_dir, key_path = Path(sys.argv[1]), Path(sys.argv[2])
key = json.load(open(key_path))
verdicts = {}
for f in sorted(out_dir.glob("*.json")):
    verdicts.update(json.load(open(f))["games"])

agg = collections.defaultdict(lambda: {"hall": 0, "games_hall": 0, "spec": [], "vibe": [], "rank": [], "n": 0,
                                       "beats_teacher": 0, "ties": 0, "examples": []})
for g, v in verdicts.items():
    m = key[g]["map"]  # letter -> source
    rank_of = {m[l]: i + 1 for i, l in enumerate(v["ranking"])}
    for letter, s in v["sets"].items():
        src = m[letter]
        a = agg[src]
        h = s.get("hallucinations", [])
        a["n"] += 1
        a["hall"] += len(h)
        a["games_hall"] += bool(h)
        a["spec"].append(s["specificity"])
        a["vibe"].append(s["vibe"])
        a["rank"].append(rank_of[src])
        if src != "teacher" and "teacher" in rank_of:
            a["beats_teacher"] += rank_of[src] < rank_of["teacher"]
        for x in h:
            a["examples"].append(f"{g} [{x.get('band')}] {x.get('claim')} — {x.get('why')}")

mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
print(f"{len(verdicts)} games judged\n")
print("| source | hallucinations / 100 labels | games with any | specificity | vibe | mean rank | ranked above teacher |")
print("|---|---:|---:|---:|---:|---:|---:|")
for src, a in sorted(agg.items(), key=lambda kv: mean(kv[1]["rank"])):
    bt = "—" if src == "teacher" else f"{a['beats_teacher'] / a['n']:.0%}"
    print(f"| {src} | {100 * a['hall'] / (5 * a['n']):.1f} | {a['games_hall'] / a['n']:.0%} | {mean(a['spec']):.2f} | "
          f"{mean(a['vibe']):.2f} | {mean(a['rank']):.2f} | {bt} |")
print()
for src, a in agg.items():
    print(f"## {src}: {len(a['examples'])} hallucinations")
    for e in a["examples"][:6]:
        print("  -", e)
