#!/usr/bin/env python3
"""Second, independent Gemma-4-31B labelling of test games: the teacher-vs-teacher ceiling.

Uses the exact production messages (s10-mini system + user, as stored in test.jsonl), thinking
off, and the model's own default sampling (production sent no sampling params to the Gemma
route). Output rows use the samples_*.jsonl format with pred = new Gemma labels and
teacher = the original Gemma labels, so similarity.py / grounding.py score it directly.

usage: teacher_resample.py [--n 384] [--tp 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prompt-ablation"))
from autolabel import genome  # noqa: E402

BANDS = ("terse_1", "terse_2", "short", "medium", "long")
VARIANT = next(v for v in genome.SEEDS if v.vid == "s10-mini")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--data", default="data/labeler_gemma/test.jsonl")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default="results/labeler/samples_teacher_gemma_resample.jsonl")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(a.data)][: a.n]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=4096, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]
    sp = llm.get_default_sampling_params()
    sp.max_tokens = 512
    print("sampling:", sp, flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"generated {len(outs)} in {time.time() - t0:.0f}s", flush=True)
    ok = 0
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            try:
                pairs = genome.assign_bands(genome.parse_labels(text, VARIANT.rungs), VARIANT.rungs)
                pred = dict(zip(BANDS, [t for _, t in pairs])) if len(pairs) == 5 else {}
            except Exception:  # noqa: BLE001
                pred = {}
            ok += len(pred) == 5
            f.write(json.dumps({"key": r["key"], "labeler": "gemma", "pred": pred, "teacher": r["labels"], "raw": text}) + "\n")
    print(f"parsed {ok}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
