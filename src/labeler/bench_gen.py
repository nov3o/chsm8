#!/usr/bin/env python3
"""Inference-speed benchmark for a trained T5Gemma labeler (games/s on one GPU).

Variants, each measured on the same test games and checked for identical-ish output:
  hf_b32        HF generate, LoRA unmerged, batch 32 (what the training eval did)
  merged_bN     LoRA merged into the weights, length-sorted batches of N
  static_bN     + static KV cache and torch.compile of the decoder step
  (next, if large batches leave a gap: a continuous-batching loop that refills
   finished slots immediately, since labels end at different lengths)

usage: bench_gen.py --ckpt <results/labeler/ckpt_TRIAL> --base <hf model dir> [--n 2048]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def load(base, ckpt, merge):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    m = AutoModelForSeq2SeqLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    m = PeftModel.from_pretrained(m, ckpt)
    if merge:
        m = m.merge_and_unload()
    m.eval()
    m.config.use_cache = True
    return m, tok


@torch.no_grad()
def run_hf(m, tok, rows, a, bs, sort, **gen_kw):
    """Batched HF generate; returns (preds, seconds)."""
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["user"])) if sort else list(range(len(rows)))
    preds = [None] * len(rows)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(0, len(order), bs):
        idx = order[i: i + bs]
        chunk = [rows[j] for j in idx]
        if a.output == "parallel":
            p, _ = T.gen_parallel(m, tok, chunk, a, "cuda")
        else:
            tok.padding_side = "right"
            enc = tok([T.prompt_text(r, a, None) for r in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_in).to("cuda")
            g = m.generate(**enc, max_new_tokens=a.max_out, do_sample=False, pad_token_id=tok.pad_token_id,
                           **gen_kw)
            if a.special_tokens == 2:
                p = [T.parse_reserved_ids(row.tolist(), tok) for row in g]
            else:
                texts = tok.batch_decode(g, skip_special_tokens=True)
                p = [T.parse_joint(t) for t in texts]
        for j, x in zip(idx, p):
            preds[j] = x
    torch.cuda.synchronize()
    return preds, time.time() - t0


BUCKETS = (256, 512, 768, 1024, 1536, 2048)


@torch.no_grad()
def run_bucketed(m, tok, rows, a, bs, **gen_kw):
    """Fixed shapes for compiled decoding: inputs padded to one of BUCKETS tokens and every
    batch filled to exactly `bs` rows (short last batch padded with repeats, dropped after)."""
    tok.padding_side = "right"
    prompts = [T.prompt_text(r, a, None) for r in rows]
    lens = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
    bucket_of = [next(b for b in BUCKETS if n <= b) if n <= BUCKETS[-1] else BUCKETS[-1] for n in lens]
    preds = [None] * len(rows)
    torch.cuda.synchronize()
    t0 = time.time()
    for bl in BUCKETS:
        idx = sorted((i for i in range(len(rows)) if bucket_of[i] == bl), key=lambda i: lens[i])
        for k in range(0, len(idx), bs):
            chunk = idx[k: k + bs]
            fill = chunk + [chunk[-1]] * (bs - len(chunk))
            enc = tok([prompts[i] for i in fill], return_tensors="pt", padding="max_length", truncation=True,
                      max_length=bl).to("cuda")
            g = m.generate(**enc, max_new_tokens=a.max_out, do_sample=False, pad_token_id=tok.pad_token_id, **gen_kw)
            for j, i in enumerate(chunk):
                preds[i] = (T.parse_reserved_ids(g[j].tolist(), tok) if a.special_tokens == 2
                            else T.parse_joint(tok.decode(g[j], skip_special_tokens=True)))
    torch.cuda.synchronize()
    return preds, time.time() - t0


def agreement(p, ref):
    same = sum(all(x.get(b) == y.get(b) for b in T.BANDS) for x, y in zip(p, ref))
    ok = sum(all(x.get(b) for b in T.BANDS) for x in p)
    return same / len(p), ok / len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", default="data/labeler_gemma/test.jsonl")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--special-tokens", type=int, default=0)
    ap.add_argument("--output", default="joint")
    ap.add_argument("--batches", default="64,128,256,512")
    ap.add_argument("--mode", choices=["all", "static"], default="all")
    ap.add_argument("--out", default="results/labeler/bench_gen.jsonl")
    a = ap.parse_args()
    a.input, a.tag, a.max_in, a.max_out, a.arch = "compact", 0, 2048, 320, "seq2seq"
    rows = T.load(Path(a.data), a.n)
    res = []

    def log(name, preds, dt, ref):
        agree, ok = agreement(preds, ref) if ref else (1.0, sum(all(x.get(b) for b in T.BANDS) for x in preds) / len(preds))
        r = {"variant": name, "ckpt": a.ckpt, "games": len(preds), "sec": round(dt, 1), "games_per_s": round(len(preds) / dt, 2),
             "parse_ok": round(ok, 3), "same_as_baseline": round(agree, 3), "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(r), flush=True)
        res.append(r)
        torch.cuda.reset_peak_memory_stats()

    if a.mode == "static":
        m, tok = load(a.base, a.ckpt, merge=True)
        ref, dt = run_hf(m, tok, rows, a, 256, sort=True)
        log("merged_sorted_b256", ref, dt, None)
        for bs in (256, 512):
            try:
                p, dt = run_bucketed(m, tok, rows, a, bs)
                log(f"bucketed_eager_b{bs}", p, dt, ref)
                p, dt = run_bucketed(m, tok, rows, a, bs, cache_implementation="static")
                log(f"bucketed_static_compile_b{bs}_first_pass", p, dt, ref)
                p, dt = run_bucketed(m, tok, rows, a, bs, cache_implementation="static")
                log(f"bucketed_static_compile_b{bs}_warm", p, dt, ref)
            except Exception as e:  # noqa: BLE001
                print(f"bucketed b{bs} failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                torch.cuda.empty_cache()
        with open(a.out, "a") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
        return

    m, tok = load(a.base, a.ckpt, merge=False)
    base_rows = rows[:256]
    ref, dt = run_hf(m, tok, base_rows, a, 32, sort=False)
    log("hf_unmerged_b32", ref, dt, None)
    del m
    torch.cuda.empty_cache()

    m, tok = load(a.base, a.ckpt, merge=True)
    p, dt = run_hf(m, tok, base_rows, a, 32, sort=False)
    log("merged_b32", p, dt, ref)
    full_ref = None
    for bs in map(int, a.batches.split(",")):
        try:
            p, dt = run_hf(m, tok, rows, a, bs, sort=True)
            log(f"merged_sorted_b{bs}", p, dt, None)
            full_ref = full_ref or p
        except torch.OutOfMemoryError:
            print(f"OOM at batch {bs}", flush=True)
            torch.cuda.empty_cache()
            break
    # static cache + compiled decoder step (may be unsupported for this arch; recorded either way)
    for bs in (256,):
        try:
            p, dt = run_hf(m, tok, rows, a, bs, sort=True, cache_implementation="static",
                           compile_config=None)
            log(f"static_b{bs}_warm0", p, dt, full_ref)
            p, dt = run_hf(m, tok, rows, a, bs, sort=True, cache_implementation="static")
            log(f"static_b{bs}", p, dt, full_ref)
        except Exception as e:  # noqa: BLE001
            print(f"static cache unsupported: {type(e).__name__}: {str(e)[:300]}", flush=True)
    with open(a.out, "a") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
