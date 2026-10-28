#!/usr/bin/env python3
"""CPU: collect the records still failing the rule check after repair -> data/fix_todo.jsonl
({key, user, labels, src, bad}), the small index the fix loop (fix_loop.py) works on."""
import json
import sys
from multiprocessing import Pool

I = ""


def one(line):
    t = line.strip().split("|")
    bad = {}
    for l in open(t[-1]):
        r = json.loads(l)
        if r["bad_after"]:
            bad[r["key"]] = r
    out = []
    for l in open(t[0]):
        if bad:
            u = json.loads(l)
            r = bad.pop(u["key"], None)
            if r:
                out.append(json.dumps({"key": r["key"], "user": u["user"], "labels": r["labels"], "src": r["src"],
                                       "bad": r["bad_after"]}))
    return out


if __name__ == "__main__":
    tasks = [l for l in open(I + "data/repair_manifest.txt") if l.strip()]
    n = 0
    with open(I + "data/fix_todo.jsonl", "w") as fo, Pool(48) as p:
        for out in p.imap_unordered(one, tasks):
            for s in out:
                fo.write(s + "\n")
            n += len(out)
    print("todo", n, file=sys.stderr)
