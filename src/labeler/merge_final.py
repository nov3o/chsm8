#!/usr/bin/env python3
"""Final 30M label corpus from the repaired shards (needs all 144 .done markers).
labeler: <base> | <base>-patched (one band regenerated) | gemma-3-1b-distill (whole set regenerated),
with base = gemma-4-31b / apertus-70b / gemma-3-1b-distill. `correct` is the per-band rule-check mask.
usage: merge_final.py OUT.jsonl.gz"""
import collections
import glob
import gzip
import json
import sys

R = "labels/repaired/"
BASE = {"distilled": "gemma-3-1b-distill", "gemma": "gemma-4-31b", "apertus": "apertus-70b"}
BANDS = ("terse_1", "terse_2", "short", "medium", "long")
done = glob.glob(R + "*.jsonl.done")
if len(done) != 144:
    sys.exit(f"only {len(done)}/144 repair tasks done; not merging")
# second pass (fix_loop.py, clean labeler) overrides the rows it re-fixed
fix = {}
for f in glob.glob("labels/fix/fix_w*.jsonl"):
    for l in open(f):
        d = json.loads(l)
        fix[d["key"]] = d
print("fix rows", len(fix), flush=True)
c, seen = collections.Counter(), set()
with gzip.open(sys.argv[1], "wt") as fo:
    for f in sorted(glob.glob(R + "*.jsonl")):
        base = BASE[f.rsplit("/", 1)[1].split("_")[0]]
        for l in open(f):
            r = json.loads(l)
            r = fix.pop(r["key"], r)
            if r["key"] in seen:
                c["dup"] += 1
                continue
            seen.add(r["key"])
            lab = {"orig": base, "patch": base + "-patched", "replace": "gemma-3-1b-distill"}[r["src"]]
            c[lab] += 1
            c["clean"] += not r["bad_after"]
            fo.write(json.dumps({"key": r["key"], "labels": r["labels"], "labeler": lab,
                                 "correct": [int(b not in r["bad_after"]) for b in BANDS]}) + "\n")
print(json.dumps({"total": len(seen), **c}), flush=True)
