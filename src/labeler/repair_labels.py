#!/usr/bin/env python3
"""Repair label sets that fail the grounding check, with the distilled labeler (vLLM, one process per GPU).

Per record, the bands whose claims the card contradicts (grounding.RULES) are found, then:
  0 bad bands  -> kept as is
  1 bad band   -> patch: prompt = game + the earlier (good) bands + "<band>: ", sample N candidates for
                  that one line, keep the first that passes; later bands stay as they were
  2+ bad bands -> replace: sample N whole label sets, keep the first with no bad band (else the fewest)
Tasks come from a manifest of lines "input.jsonl[|labels.jsonl]|output.jsonl"; input rows need key + user,
and labels either in the row or joined by key from labels.jsonl. Task i goes to worker i % n-workers;
an output with a .done marker is skipped, so resubmitting resumes.
usage: repair_labels.py --ckpt DIR --manifest FILE --worker K --n-workers N
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounding as G  # noqa: E402
import train as T  # noqa: E402


def bad_bands(user: str, labels: dict) -> list[str]:
    f = G.facts(user)
    return [b for b in T.BANDS if any(re.search(rx, (labels.get(b) or "").lower()) and cond(f) for _, rx, cond in G.RULES)]


def claim(out: str, stale_s: int = 7200) -> bool:
    """Atomically claim a task output (O_EXCL), so any number of jobs can share one manifest.
    A claim older than stale_s without .done is from a dead job and may be taken over."""
    if Path(out + ".done").exists():
        return False
    c = out + ".claim"
    try:
        os.close(os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        return True
    except FileExistsError:
        if time.time() - os.path.getmtime(c) > stale_s:
            os.utime(c)
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--n", type=int, default=4, help="candidates per repair")
    ap.add_argument("--temperature", type=float, default=0.7)
    a = ap.parse_args()

    tasks = [l.strip().split("|") for l in open(a.manifest) if l.strip()]
    # own tasks first, then everyone else's: claims stop two jobs doing the same task
    order = [t for i, t in enumerate(tasks) if i % a.n_workers == a.worker] + [t for i, t in enumerate(tasks) if i % a.n_workers != a.worker]
    mine = [t for t in order if not Path(t[-1] + ".done").exists()]
    print(f"worker {a.worker}/{a.n_workers}: {len(mine)} tasks", flush=True)
    if not mine:
        return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    cfg = types.SimpleNamespace(input="compact", tag=0, special_tokens=0, max_in=2048)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    llm = LLM(model=a.ckpt, dtype="bfloat16", max_model_len=2560, gpu_memory_utilization=0.9,
              max_num_seqs=1024, max_num_batched_tokens=32768, enable_prefix_caching=True)
    sp_line = SamplingParams(n=a.n, temperature=a.temperature, max_tokens=120, stop=["\n"])
    sp_set = SamplingParams(n=a.n, temperature=a.temperature, max_tokens=320)
    enc = lambda text: {"prompt_token_ids": tok(text, add_special_tokens=True)["input_ids"][-2300:]}  # noqa: E731

    for t in mine:
        if not all(Path(x).exists() for x in t[:-1]) or (len(t) == 3 and not Path(t[1].replace(".jsonl", ".done")).exists()):
            print("inputs not ready, skipped:", t[0], flush=True)  # a resubmit picks it up
            continue
        if not claim(t[-1]):
            continue
        t0 = time.time()
        rows = [json.loads(l) for l in open(t[0])]
        if len(t) == 3:
            lab = {}
            for l in open(t[1]):
                d = json.loads(l)
                lab[d["key"]] = d["labels"]
            rows = [dict(r, labels=lab[r["key"]]) for r in rows if r["key"] in lab]
        patch, replace = [], []
        for i, r in enumerate(rows):
            r["bad_before"] = bad_bands(r["user"], r["labels"])
            r["src"] = "orig"
            if len(r["bad_before"]) == 1:
                b = r["bad_before"][0]
                head = "".join(f"{T.BAND_NAME[x]}: {r['labels'][x].strip()}\n" for x in T.BANDS[: T.BANDS.index(b)])
                patch.append((i, b, T.prompt_text(r, cfg, None) + head + f"{T.BAND_NAME[b]}:"))
            elif len(r["bad_before"]) > 1:
                replace.append((i, T.prompt_text(r, cfg, None)))
        f = {}
        if patch:
            for (i, b, _), o in zip(patch, llm.generate([enc(p) for *_, p in patch], sp_line, use_tqdm=False)):
                cands = [c.text.strip() for c in o.outputs if c.text.strip()]
                r = rows[i]
                ok = [c for c in cands if not bad_bands(r["user"], {b: c})]
                if ok:  # if every candidate fails too, keep the original line
                    r["labels"] = dict(r["labels"], **{b: ok[0]})
                    r["src"] = "patch"
        if replace:
            for (i, _), o in zip(replace, llm.generate([enc(p) for _, p in replace], sp_set, use_tqdm=False)):
                r = rows[i]
                sets = [s for s in (T.parse_joint(c.text) for c in o.outputs) if all(s.get(b) for b in T.BANDS)]
                if sets:
                    best = min(sets, key=lambda s: len(bad_bands(r["user"], s)))
                    if len(bad_bands(r["user"], best)) < len(r["bad_before"]):
                        r["labels"], r["src"] = {b: best[b] for b in T.BANDS}, "replace"
        cnt = {"rows": len(rows), "patch_tried": len(patch), "replace_tried": len(replace)}
        tmp = Path(t[-1] + ".tmp")
        with open(tmp, "w") as fo:
            for r in rows:
                after = bad_bands(r["user"], r["labels"])
                cnt[r["src"]] = cnt.get(r["src"], 0) + 1
                cnt["bad_bands_before"] = cnt.get("bad_bands_before", 0) + len(r["bad_before"])
                cnt["bad_bands_after"] = cnt.get("bad_bands_after", 0) + len(after)
                fo.write(json.dumps({"key": r["key"], "labels": r["labels"], "src": r["src"],
                                     "bad_before": r["bad_before"], "bad_after": after}) + "\n")
        tmp.rename(t[-1])
        cnt["sec"] = round(time.time() - t0)
        Path(t[-1] + ".done").write_text(json.dumps(cnt))
        print(t[-1], json.dumps(cnt), flush=True)


if __name__ == "__main__":
    main()
