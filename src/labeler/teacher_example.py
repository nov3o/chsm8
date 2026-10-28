#!/usr/bin/env python3
"""Gemma-4-31B teacher (RedHatAI FP8-block) label for ONE game side, with the exact production messages.

system = the s10-mini system prompt shared by every record of data/labeler_gemma/test.jsonl;
user = the "user" field of the key's record in data/relabel_apertus/shard_000.jsonl.
Thinking off (chat template enable_thinking=False, also the template default); sampling = the
model's generation_config defaults (production sent no sampling params to the Gemma route).

usage: teacher_example.py [--key 2014-01:3847:white] [--n 3] [--tp 4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prompt-ablation"))
from autolabel import genome  # noqa: E402

BANDS = ("terse_1", "terse_2", "short", "medium", "long")
ORDER = {"terse": 0, "short": 1, "medium": 2, "long": 3}
RUNGS = ["terse", "short", "medium", "long"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def to_bands(text: str) -> dict:
    pairs = genome.assign_bands(genome.parse_labels(text, RUNGS), RUNGS)
    if len(pairs) != 5:
        return {}
    pairs = sorted(pairs, key=lambda p: ORDER.get(p[0], 9))  # stable: the two terse keep output order
    if [b for b, _ in pairs] != ["terse", "terse", "short", "medium", "long"]:
        return {}
    return dict(zip(BANDS, [t for _, t in pairs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="2014-01:3847:white")
    ap.add_argument("--model", default=str(ROOT / "models/gemma-4-31b-it-fp8-block"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "results/labeling/example_2014-01_3847_white_gemma.json"))
    a = ap.parse_args()

    systems = {json.loads(l)["system"] for l in open(ROOT / "data/labeler_gemma/test.jsonl")}
    assert len(systems) == 1, f"{len(systems)} distinct system prompts"
    system = systems.pop()
    needle = f'"{a.key}"'
    rec = None
    with open(ROOT / "data/relabel_apertus/shard_000.jsonl") as f:
        for line in f:
            if needle in line and json.loads(line)["key"] == a.key:
                rec = json.loads(line)
                break
    assert rec and rec["key"] == a.key, a.key
    user = rec["user"]

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=4096, gpu_memory_utilization=0.9,
              seed=a.seed, limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    d = llm.get_default_sampling_params()
    print("default sampling:", d, flush=True)
    sp = SamplingParams(n=a.n, temperature=d.temperature, top_p=d.top_p, top_k=d.top_k, min_p=d.min_p,
                        repetition_penalty=d.repetition_penalty, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    out = llm.generate([prompt], sp)[0]
    print(f"generated in {time.time() - t0:.1f}s", flush=True)
    samples = []
    for o in out.outputs:
        labels = to_bands(o.text)
        samples.append({"raw": o.text, "finish_reason": o.finish_reason, "n_tokens": len(o.token_ids),
                        "parsed_ok": len(labels) == 5, "labels": labels})
    primary = next((s for s in samples if s["parsed_ok"]), samples[0])
    res = {
        "key": a.key, "labeler": "gemma-4-31B-it (teacher)", "model": "RedHatAI/gemma-4-31B-it-FP8-block",
        "model_path": a.model, "vllm_version": vllm_version, "tensor_parallel_size": a.tp,
        "prompt": {"variant": "s10-mini", "system_source": "data/labeler_gemma/test.jsonl (all records share it)",
                   "user_source": "data/relabel_apertus/shard_000.jsonl", "system_sha256": sha(system),
                   "user_sha256": sha(user), "rendered_prompt_sha256": sha(prompt),
                   "chat_template_kwargs": {"enable_thinking": False}, "system": system, "user": user},
        "sampling": {"source": "model generation_config defaults (llm.get_default_sampling_params); "
                               "production sent no sampling params to the Gemma route",
                     "temperature": sp.temperature, "top_p": sp.top_p, "top_k": sp.top_k, "min_p": sp.min_p,
                     "repetition_penalty": sp.repetition_penalty, "max_tokens": sp.max_tokens,
                     "seed": a.seed, "n": a.n},
        "primary_sample_index": samples.index(primary),
        "labels": primary["labels"],
        "samples": samples,
        "apertus_labels": rec.get("labels"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"labels": res["labels"], "parsed": [s["parsed_ok"] for s in samples]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
