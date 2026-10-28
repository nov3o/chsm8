#!/usr/bin/env python3
"""vLLM throughput + output check for a fully fine-tuned decoder labeler (e.g. Gemma-3-1B).

Labels every game of a test split on one GPU with greedy decoding, reports games/s, and
writes samples_<name>.jsonl in the same format as train.py so grounding.py can score them.

usage: bench_vllm.py --ckpt <results/labeler/ckpt_TRIAL> --name <out name> [--special-tokens 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="data/labeler_gemma/test.jsonl")
    ap.add_argument("--special-tokens", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--speed-proxy", type=int, default=0,
                    help="N>0: ignore EOS and emit exactly N tokens per game (speed only, for an untuned base model)")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the prompt set to lengthen the timed run")
    ap.add_argument("--n-rows", type=int, default=0, help=">0: label this many distinct games from --data")
    ap.add_argument("--max-batched-tokens", type=int, default=0)
    ap.add_argument("--prefix-cache", type=int, default=-1, help="-1 auto (on unless repeating)")
    ap.add_argument("--quant", default=None, help="e.g. fp8")
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--sort", type=int, default=0, help="submit prompts sorted by length")
    ap.add_argument("--tag", default="", help="label for this config in bench_vllm.jsonl")
    ap.add_argument("--no-samples", type=int, default=0)
    ap.add_argument("--out", default="results/labeler")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    cfg = types.SimpleNamespace(input="compact", tag=0, special_tokens=a.special_tokens, max_in=2048)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    rows = T.load(Path(a.data), a.n_rows or None) * a.repeat
    prompts = [{"prompt_token_ids": tok(T.prompt_text(r, cfg, None), add_special_tokens=True)["input_ids"][-2048:]}
               for r in rows]
    if a.sort:
        order = sorted(range(len(rows)), key=lambda i: len(prompts[i]["prompt_token_ids"]))
        rows, prompts = [rows[i] for i in order], [prompts[i] for i in order]
    extra = {"max_num_batched_tokens": a.max_batched_tokens} if a.max_batched_tokens else {}
    if a.quant:
        extra["quantization"] = a.quant

    t0 = time.time()
    pc = (a.repeat == 1) if a.prefix_cache < 0 else bool(a.prefix_cache)
    llm = LLM(model=a.ckpt, dtype="bfloat16", max_model_len=2560, gpu_memory_utilization=a.gpu_mem,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=pc, enforce_eager=bool(a.eager), **extra)
    load_s = time.time() - t0
    sp = (SamplingParams(temperature=0.0, max_tokens=a.speed_proxy, ignore_eos=True) if a.speed_proxy
          else SamplingParams(temperature=0.0, max_tokens=320, skip_special_tokens=False))
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    samples = []
    for r, o in zip(rows, outs):
        ids = list(o.outputs[0].token_ids)
        if a.special_tokens == 2:
            pred = T.parse_reserved_ids(ids, tok)
        else:
            pred = T.parse_joint(tok.decode(ids, skip_special_tokens=True))
        samples.append({"key": r["key"], "labeler": r["labeler"], "pred": pred, "teacher": r["labels"]})
    ok = sum(all(s["pred"].get(b) for b in T.BANDS) for s in samples) / len(samples)
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    import os
    res = {"name": a.name, "tag": a.tag, "cfg": {"seqs": a.max_num_seqs, "mbt": a.max_batched_tokens, "prefix": pc,
           "quant": a.quant, "eager": a.eager, "sort": a.sort, "attn": os.environ.get("VLLM_ATTENTION_BACKEND", "auto")}, "ckpt": a.ckpt, "engine": "vllm", "games": len(rows), "load_s": round(load_s, 1),
           "gen_s": round(gen_s, 1), "games_per_s": round(len(rows) / gen_s, 1),
           "out_tokens_per_s": round(n_tok / gen_s), "parse_ok": round(ok, 4)}
    print(json.dumps(res), flush=True)
    with open(Path(a.out) / "bench_vllm.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    if a.no_samples:
        return
    with open(Path(a.out) / f"samples_{a.name}.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
