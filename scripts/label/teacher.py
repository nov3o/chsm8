"""Label games with the teacher, Gemma-4-31B-it, using vLLM: the labels chsm8-distillation and chsm8-game-labels
were made from. The distilled labeler (label.py) is the cheap way to label many games; this is the expensive one.

    pip install vllm
    python scripts/label/teacher.py --games games.parquet --out teacher.jsonl --tp 4     # 4 GPUs, one model

--games: a parquet file with the columns of chsm8-pre-training; both sides of every game are labeled, unless the
file has a `side` column. The teacher sees SYSTEM and the user message (prompt.py), thinking off, the model's default
sampling settings, and answers with a JSON array of five labels. Output, one line per side:
    {"row", "side", "user", "labels", "ok": all five bands parsed, "clean": ok and no band contradicted (grounding.py)}
Train the labeler on the clean sets (train_labeler.py reads them from chsm8-distillation).
"""
import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from grounding import contradicted  # noqa: E402
from prompt import SYSTEM, context, parse_teacher, user  # noqa: E402

BANDS = ("terse_1", "terse_2", "short", "medium", "long")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel GPUs")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    from vllm import LLM

    t = pq.read_table(a.games).to_pylist()[:a.max_games or None]
    jobs = [(i, s) for i, g in enumerate(t) for s in ([g["side"]] if "side" in g else ("white", "black"))]
    users = [user(context(t[i]["moves"], s, t[i]["white_elo"], t[i]["black_elo"], t[i]["tc_base"],
                          t[i]["tc_increment"], t[i].get("opening") or "", t[i].get("eco") or "")) for i, s in jobs]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, max_model_len=4096, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False) for u in users]
    sp = llm.get_default_sampling_params()
    sp.max_tokens = 512
    outs = llm.generate(prompts, sp)
    n_ok = n_clean = 0
    with open(a.out, "w") as f:
        for (i, side), u, o in zip(jobs, users, outs):
            labels = parse_teacher(o.outputs[0].text)
            ok = len(labels) == 5
            clean = ok and not contradicted(u, labels)
            n_ok, n_clean = n_ok + ok, n_clean + clean
            f.write(json.dumps({"row": i, "side": side, "user": u, "labels": {b: labels.get(b, "") for b in BANDS},
                                "ok": ok, "clean": clean}) + "\n")
    print(f"{len(jobs):,} sides: {n_ok / max(len(jobs), 1):.1%} parsed, {n_clean / max(len(jobs), 1):.1%} clean -> {a.out}")


if __name__ == "__main__":
    main()
