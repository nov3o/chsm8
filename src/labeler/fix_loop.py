#!/usr/bin/env python3
"""GPU fix loop on data/fix_todo.jsonl with the clean labeler: up to --rounds rounds, each round regenerates
the first still-failing band of every row (prefix = game + earlier bands, --n candidates, first one passing
the rule check is spliced in) and re-checks. Rows with 3+ bad bands first try a whole new set.
Writes labels/fix/fix_wK.jsonl {key, labels, src, bad_after} + .done."""
import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grounding as G  # noqa: E402
import train as T  # noqa: E402
from repair_labels import bad_bands  # noqa: E402

I = ""
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=I + "results/labeler/ckpt_full_gemma3_1b_clean")
ap.add_argument("--worker", type=int, required=True)
ap.add_argument("--n-workers", type=int, required=True)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--sub", type=int, default=0, help="rerun one worker's rows split over --n-sub GPUs")
ap.add_argument("--n-sub", type=int, default=1)
a = ap.parse_args()
out = Path(I + f"labels/fix/fix_w{a.worker}" + (f"s{a.sub}" if a.n_sub > 1 else "") + ".jsonl")
mark = Path(str(out) + (".ok" if a.n_sub > 1 else ".done"))
out.parent.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for i, l in enumerate(open(I + "data/fix_todo.jsonl")) if i % a.n_workers == a.worker]
rows = rows[a.sub::a.n_sub]
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

cfg = types.SimpleNamespace(input="compact", tag=0, special_tokens=0, max_in=2048)
tok = AutoTokenizer.from_pretrained(a.ckpt)
llm = LLM(model=a.ckpt, dtype="bfloat16", max_model_len=2560, gpu_memory_utilization=0.9,
          max_num_seqs=1024, max_num_batched_tokens=32768, enable_prefix_caching=True)
sp_line = SamplingParams(n=a.n, temperature=0.8, max_tokens=120, stop=["\n"])
sp_set = SamplingParams(n=a.n, temperature=0.8, max_tokens=320)
enc = lambda text: {"prompt_token_ids": tok(text, add_special_tokens=True)["input_ids"][-2300:]}  # noqa: E731
for r in rows:
    r["bad_now"] = bad_bands(r["user"], r["labels"])
print(f"worker {a.worker}: {len(rows)} rows, {sum(map(len, (r['bad_now'] for r in rows)))} bad bands", flush=True)
t0 = time.time()
multi = [r for r in rows if len(r["bad_now"]) >= 3]
for r, o in zip(multi, llm.generate([enc(T.prompt_text(r, cfg, None)) for r in multi], sp_set, use_tqdm=False)):
    sets = [s for s in (T.parse_joint(c.text) for c in o.outputs) if all(s.get(b) for b in T.BANDS)]
    if sets:
        best = min(sets, key=lambda s: len(bad_bands(r["user"], s)))
        if len(bad_bands(r["user"], best)) < len(r["bad_now"]):
            r["labels"], r["src"], r["bad_now"] = {b: best[b] for b in T.BANDS}, "replace", bad_bands(r["user"], best)
for k in range(a.rounds):
    todo = [r for r in rows if r["bad_now"]]
    if not todo:
        break
    ps = []
    for r in todo:
        b = r["bad_now"][0]
        head = "".join(f"{T.BAND_NAME[x]}: {r['labels'][x].strip()}\n" for x in T.BANDS[: T.BANDS.index(b)])
        ps.append(enc(T.prompt_text(r, cfg, None) + head + f"{T.BAND_NAME[b]}:"))
    stuck = 0
    for r, o in zip(todo, llm.generate(ps, sp_line, use_tqdm=False)):
        b = r["bad_now"][0]
        ok = [c.text.strip() for c in o.outputs if c.text.strip() and not bad_bands(r["user"], {b: c.text.strip()})]
        if ok:
            r["labels"] = dict(r["labels"], **{b: ok[0]})
            r["src"] = "patch" if r["src"] == "orig" else r["src"]
        else:
            stuck += 1
        r["bad_now"] = sorted(bad_bands(r["user"], r["labels"]), key=lambda x: (x == b and not ok, T.BANDS.index(x)))
    left = sum(len(r["bad_now"]) for r in rows)
    print(f"round {k + 1}: tried {len(todo)} stuck {stuck} bad bands left {left} ({time.time() - t0:.0f}s)", flush=True)
with open(str(out) + ".tmp", "w") as fo:
    for r in rows:
        fo.write(json.dumps({"key": r["key"], "labels": r["labels"], "src": r["src"], "bad_after": r["bad_now"]}) + "\n")
Path(str(out) + ".tmp").rename(out)
mark.write_text(json.dumps({"rows": len(rows), "bad_left": sum(len(r["bad_now"]) for r in rows),
                                                 "sec": round(time.time() - t0)}))
