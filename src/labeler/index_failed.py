#!/usr/bin/env python3
"""Index label sets that fail the grounding rules: per source, count sets by number of bad bands and
which rules fire, and write {key, bad} for every failing set (patch if 1 bad band, replace if 2+).
usage: index_failed.py NAME OUT.jsonl FILE [FILE ...]   (rows need key, user, labels)"""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounding as G  # noqa: E402

name, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
nbad, rules, n = collections.Counter(), collections.Counter(), 0
with open(out, "w") as fo:
    for fn in files:
        for l in open(fn):
            r = json.loads(l)
            f = G.facts(r["user"])
            bad = []
            for b in G.BANDS:
                hit = [nm for nm, rx, cond in G.RULES if re.search(rx, (r["labels"][b] or "").lower()) and cond(f)]
                rules.update(hit)
                if hit:
                    bad.append(b)
            n += 1
            nbad[min(len(bad), 2)] += 1
            if bad:
                fo.write(json.dumps({"key": r["key"], "bad": bad}) + "\n")
print(json.dumps({"source": name, "sets": n, "ok": nbad[0], "patch_1bad": nbad[1], "replace_2plus": nbad[2],
                  "fail_rate": round((nbad[1] + nbad[2]) / n, 4), "rules": dict(rules.most_common())}), flush=True)
