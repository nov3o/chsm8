#!/usr/bin/env python3
"""Teacher-label likelihood of saved labeler checkpoints on one shared test split.

Reports nats per UTF-8 byte of the Gemma teacher's labels (overall and per band), the same
metric as train.py's val loss, so every finalist is scored on identical games.

usage: score_ckpt.py SPEC [SPEC ...]
  SPEC = name=<trial>,ckpt=<dir>[,base=<hf dir>][,arch=causal|seq2seq][,tok=0|2][,output=joint|parallel]
  base is required for LoRA adapters (the ckpt dir then holds adapter weights only).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

DATA = Path("data/labeler_gemma/test.jsonl")
OUT = Path("results/labeler/likelihood_test.jsonl")


def load(spec):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    Model = AutoModelForSeq2SeqLM if spec["arch"] == "seq2seq" else AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(spec["ckpt"])
    if "base" in spec:
        from peft import PeftModel
        m = Model.from_pretrained(spec["base"], dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        m = PeftModel.from_pretrained(m, spec["ckpt"])
    else:
        m = Model.from_pretrained(spec["ckpt"], dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return m.eval(), tok


def main():
    rows = T.load(DATA, None)
    for arg in sys.argv[1:]:
        spec = dict(kv.split("=", 1) for kv in arg.split(","))
        spec.setdefault("arch", "causal")
        a = types.SimpleNamespace(input="compact", tag=0, special_tokens=int(spec.get("tok", 0)),
                                  max_in=2048, max_out=320, arch=spec["arch"], output=spec.get("output", "joint"),
                                  eval_tokens=32768)
        m, tok = load(spec)
        enc = T.Encoder(tok, a)
        ex = [enc.encode_parallel(r) for r in rows] if a.output == "parallel" else [enc.encode(r, None) for r in rows]
        v, per = T.evaluate(m, ex, a, tok.pad_token_id, "cuda")
        res = {"name": spec["name"], "games": len(rows), "test_nats_per_byte": round(v, 4),
               "per_band": {b: round(x, 4) for b, x in per.items()}}
        print(json.dumps(res), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
