#!/usr/bin/env python3
"""Production labeling: one process per GPU, vLLM offline API, shard-level resume.

Takes every shard_NNN.jsonl in --in whose index % --n-workers == --worker, skips shards that
already have a .done marker, and writes labels_NNN.jsonl of {key, labels, ok}.
usage: label_shards.py --ckpt DIR --in DIR --out DIR --worker K --n-workers N
"""
from __future__ import annotations

import argparse
import os
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def claim(out: Path, idx: str, stale_s: int = 5400) -> bool:
    """Atomically claim a shard (O_EXCL). A claim older than stale_s without a .done
    marker is from a dead process and may be taken over."""
    if (out / f"labels_{idx}.done").exists():
        return False
    c = out / f"labels_{idx}.claim"
    try:
        os.close(os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        return True
    except FileExistsError:
        if time.time() - c.stat().st_mtime > stale_s and not (out / f"labels_{idx}.done").exists():
            c.touch()
            return True
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--pool", default="",
                    help="comma list of worker ids whose shards this process may steal (claim-based; "
                         "any number of processes can share a pool). --worker is then ignored.")
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-batched-tokens", type=int, default=32768)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(a.inp).glob("shard_*.jsonl"))
    owners = {int(w) for w in a.pool.split(",")} if a.pool else {a.worker}
    mine = [s for s in shards if int(s.stem.split("_")[1]) % a.n_workers in owners
            and not (out / f"labels_{s.stem.split('_')[1]}.done").exists()]
    if a.pool:  # spread processes over the pool so claims rarely collide
        import random
        random.Random(f"{os.uname().nodename}-{os.getpid()}").shuffle(mine)
    print(f"worker {a.worker}/{a.n_workers}: {len(mine)} shards to do", flush=True)
    if not mine:
        return
    cfg = types.SimpleNamespace(input="compact", tag=0, special_tokens=0, max_in=2048)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    llm = LLM(model=a.ckpt, dtype="bfloat16", max_model_len=2560, gpu_memory_utilization=0.9,
              max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_batched_tokens, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=320)
    for s in mine:
        idx = s.stem.split("_")[1]
        if not claim(out, idx):  # always: rescue + chase jobs may share shards
            continue
        t0 = time.time()
        rows = [json.loads(l) for l in open(s)]
        prompts = [{"prompt_token_ids": tok(T.prompt_text(r, cfg, None), add_special_tokens=True)["input_ids"][-2048:]}
                   for r in rows]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        n_ok = 0
        tmp = out / f"labels_{idx}.jsonl.tmp.{os.uname().nodename}.{os.getpid()}"
        with open(tmp, "w") as f:
            for r, o in zip(rows, outs):
                pred = T.parse_joint(o.outputs[0].text)
                ok = all(pred.get(b) for b in T.BANDS)
                n_ok += ok
                f.write(json.dumps({"key": r["key"], "labels": {b: pred.get(b, "") for b in T.BANDS}, "ok": ok}) + "\n")
        tmp.rename(out / f"labels_{idx}.jsonl")
        (out / f"labels_{idx}.done").write_text(json.dumps({"rows": len(rows), "ok": n_ok, "sec": round(time.time() - t0)}))
        print(f"shard {idx}: {len(rows)} rows, {n_ok/len(rows):.4f} parsed, {len(rows)/(time.time()-t0):.0f} games/s", flush=True)


if __name__ == "__main__":
    main()
