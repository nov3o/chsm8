"""Label games with the distilled labeler (chsm8-labeler-gemma3-1b) using vLLM: five style descriptions per side.

    pip install vllm
    python scripts/label/label.py --games games.parquet --out labels.jsonl          # one GPU; we ran one process per GPU

--games is a parquet file with the columns of chsm8-pre-training (moves in UCI, white_elo, black_elo, tc_base,
tc_increment, opening, eco). Both sides of every game are labeled, unless the file has a `side` column. The output
has one line per side: {"row": row number in --games, "side", "labels": {terse_1, terse_2, short, medium, long},
"ok": all five bands parsed}. Greedy decoding, as for chsm8-post-training.
"""
import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from prompt import context  # noqa: E402
from train_labeler import BANDS, parse, prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="LegumMagister/chsm8-labeler-gemma3-1b")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t = pq.read_table(a.games).to_pylist()[:a.max_games or None]
    jobs = [(i, s) for i, g in enumerate(t) for s in ([g["side"]] if "side" in g else ("white", "black"))]
    tok = AutoTokenizer.from_pretrained(a.model)

    def ids(i, side):
        g = t[i]
        ctx = context(g["moves"], side, g["white_elo"], g["black_elo"], g["tc_base"], g["tc_increment"],
                      g.get("opening") or "", g.get("eco") or "")
        return {"prompt_token_ids": tok(prompt(ctx), add_special_tokens=True)["input_ids"][-2048:]}

    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=2560, gpu_memory_utilization=0.9,
              max_num_seqs=a.max_num_seqs, enable_prefix_caching=True)
    outs = llm.generate([ids(i, s) for i, s in jobs], SamplingParams(temperature=0.0, max_tokens=320))
    n_ok = 0
    with open(a.out, "w") as f:
        for (i, side), o in zip(jobs, outs):
            got = parse(o.outputs[0].text)
            ok = all(got.get(b) for b in BANDS)
            n_ok += ok
            f.write(json.dumps({"row": i, "side": side, "labels": {b: got.get(b, "") for b in BANDS}, "ok": ok}) + "\n")
    print(f"{len(jobs):,} sides, {n_ok / max(len(jobs), 1):.2%} with all five bands -> {a.out}")


if __name__ == "__main__":
    main()
