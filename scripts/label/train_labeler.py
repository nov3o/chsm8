"""Distill the teacher into a small labeler: full fine-tune of Gemma-3-1B-it on chsm8-distillation.

    torchrun --nproc-per-node 4 scripts/label/train_labeler.py --out runs/labeler     # we used 4 GH200, ~10 hours

chsm8-distillation holds 2.4M teacher label sets (Gemma-4-31B-it) whose five bands all passed the grounding check,
each with the prompt the teacher saw. The labeler reads only the game-specific part of that prompt (prompt.context:
moves and measured card) and writes the five bands:

    <context>

    ### STYLE LABELS
    terse: ...
    terse: ...
    short: ...
    medium: ...
    long: ...<eos>

One epoch, lr 5e-5 with 200 warmup steps and cosine decay to 10%, batches of at most 16,384 padded tokens per GPU,
loss on the label tokens only.
"""
import argparse
import math
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

BANDS = ("terse_1", "terse_2", "short", "medium", "long")
HEADER = "### STYLE LABELS\n"


def compact(user):
    """The teacher's user message -> the game-specific part (what prompt.context() builds)."""
    return user.split("GAME\n", 1)[-1].split("\n\nReturn ONLY", 1)[0].strip()


def prompt(context):
    return context + "\n\n" + HEADER


def target(labels):
    return "".join(f"{b.split('_')[0]}: {labels[b].strip()}\n" for b in BANDS)


def parse(text):
    """Labeler output -> {terse_1, terse_2, short, medium, long} (missing bands left out)."""
    import re
    got = {}
    for m in re.finditer(r"\b(terse|short|medium|long)\s*:\s*(.*?)(?=\b(?:terse|short|medium|long)\s*:|$)", text, re.S):
        b, body = m.group(1), " ".join(m.group(2).split())
        key = ("terse_2" if "terse_1" in got else "terse_1") if b == "terse" else b
        if body:
            got.setdefault(key, body)
    return got


def batches(examples, max_tokens, rng):
    """Shuffled, length-sorted within pools of 512, cut so that rows x longest row <= max_tokens."""
    order = rng.sample(range(len(examples)), len(examples))
    for i in range(0, len(order), 512):
        pool = sorted(order[i:i + 512], key=lambda j: len(examples[j][0]) + len(examples[j][1]))
        cur, longest = [], 0
        for j in pool:
            n = len(examples[j][0]) + len(examples[j][1])
            if cur and max(longest, n) * (len(cur) + 1) > max_tokens:
                yield cur
                cur, longest = [], 0
            cur.append(j)
            longest = max(longest, n)
        if cur:
            yield cur


def collate(rows, pad, device):
    """Left-padded so all targets end at the last position: the model then computes logits only for the last
    Lt + 1 positions (logits_to_keep), not for the long prompts."""
    L, Lt = max(len(p) + len(t) for p, t in rows), max(len(t) for _, t in rows)
    ids = torch.full((len(rows), L), pad)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    lab = torch.full((len(rows), Lt), -100)
    for i, (p, t) in enumerate(rows):
        ids[i, L - len(p) - len(t):] = torch.tensor(p + t)
        att[i, L - len(p) - len(t):] = 1
        lab[i, Lt - len(t):] = torch.tensor(t)
    pos = (att.cumsum(1) - 1).clamp(min=0)
    return dict(input_ids=ids.to(device), attention_mask=att.to(device), position_ids=pos.to(device),
                logits_to_keep=Lt + 1), lab.to(device)


def token_nll(model, inputs, lab):
    logits = model(**inputs).logits[:, :-1].float()  # the last position predicts past EOS
    return F.cross_entropy(logits.flatten(0, 1), lab.flatten().clamp(min=0), reduction="none").view_as(lab) * (lab >= 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--data", default="LegumMagister/chsm8-distillation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--max-in", type=int, default=2048)
    ap.add_argument("--max-out", type=int, default=320)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    device = torch.device("cuda", torch.cuda.current_device())
    tok = AutoTokenizer.from_pretrained(a.model)
    ds = load_dataset(a.data, split="train")
    if a.max_examples:
        ds = ds.select(range(a.max_examples))
    ds = ds.shard(world, rank)

    def encode(r):
        p = tok(prompt(compact(r["user"])), add_special_tokens=True)["input_ids"][-a.max_in:]
        t = (tok(target(r["labels"]), add_special_tokens=False)["input_ids"] + [tok.eos_token_id])[:a.max_out]
        return p, t
    t0 = time.time()
    examples = [encode(r) for r in ds]
    if rank == 0:
        print(f"{len(examples):,} examples per GPU tokenized in {time.time() - t0:.0f}s", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    rng = random.Random(a.seed + rank)
    epoch_batches = list(batches(examples, a.batch_tokens, rng))
    n = torch.tensor(len(epoch_batches), device=device)
    if ddp:  # every rank takes the same number of steps
        dist.all_reduce(n, op=dist.ReduceOp.MIN)
    total = int(int(n) * a.epochs)
    step, ema, t0 = 0, None, time.time()
    while step < total:
        for idx in epoch_batches[:int(n)]:
            if step == total:
                break
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups:
                g["lr"] = lr
            inputs, lab = collate([examples[j] for j in idx], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = token_nll(dmodel, inputs, lab).sum() / (lab >= 0).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            ema = float(loss) if ema is None else 0.98 * ema + 0.02 * float(loss)
            if rank == 0 and step % 25 == 0:
                print(f"step {step:,}/{total:,} | loss {ema:.4f} | lr {lr:.2e} | {time.time() - t0:.0f}s", flush=True)
        epoch_batches = list(batches(examples, a.batch_tokens, rng))
    if rank == 0:
        model.save_pretrained(a.out)
        tok.save_pretrained(a.out)
        print(f"done: {a.out}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
