#!/usr/bin/env python3
"""Fixed-wall-clock fine-tune of a student labeler on teacher (apertus/gemma) labels.

Knobs ablated:
  --arch      causal | seq2seq         decoder-only vs encoder-decoder (T5Gemma)
  --lora-r    0 = full fine-tune, else LoRA rank on all linear layers
  --output    joint  = all 5 labels in one sequence (one call per record)
              single = one label per sequence, band named in the prompt
  --input     teacher = exact s10-mini system+user prompt the teacher saw
              compact = moves + measured card only (drops fixed instructions)
  --tag       condition on teacher family ("labeler: gemma") so one student
              can reproduce either voice

Primary metric: val nats per UTF-8 byte of label text (tokenizer-independent),
overall and per band, always summed over all 5 labels of a record so joint and
single are comparable. Secondary: greedy-generation checks on held-out records.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

BANDS = ("terse_1", "terse_2", "short", "medium", "long")
BAND_NAME = {"terse_1": "terse", "terse_2": "terse", "short": "short", "medium": "medium", "long": "long"}
WORD_RANGE = {"terse": (6, 12), "short": (13, 20), "medium": (22, 35), "long": (35, 55)}
HEADER = "### STYLE LABELS\n"
# --special-tokens 1: compact delimiters instead of plain-text scaffolding.
# Each new embedding starts at the mean embedding of its plain-word meaning.
SPECIAL = {"<|labels|>": "style labels", "<|labeler:apertus|>": "apertus", "<|labeler:gemma|>": "gemma",
           "<|terse|>": "terse", "<|short|>": "short", "<|medium|>": "medium", "<|long|>": "long",
           "<|end|>": "end"}
# --special-tokens 2: reuse the tokenizer's reserved <unusedN> tokens (no new
# embeddings). Each label is <band token> text <end token>, one token per band
# so terse_1/terse_2 are distinct; outputs are read from token ids, not parsed.
RESERVED = {b: f"<unused{i}>" for i, b in enumerate(BANDS)}
LABEL_END = "<unused5>"


# ---------------------------------------------------------------- data

def compact(user: str) -> str:
    body = user.split("GAME\n", 1)[-1]
    return body.split("\n\nReturn ONLY", 1)[0].strip()


def prompt_text(r: dict, a, band: str | None) -> str:
    inp = (r["system"] + "\n\n" + r["user"]) if a.input == "teacher" else compact(r["user"])
    lines = [inp, ""]
    if a.special_tokens == 1:
        tail = f"<|labeler:{r['labeler']}|>" if a.tag else ""
        if band is not None:
            tail += f"<|{BAND_NAME[band]}|>" + ("<|terse|>" if band == "terse_2" else "")
        return "\n".join(lines) + "\n" + tail + "<|labels|>"
    if a.tag:
        lines.append(f"labeler: {r['labeler']}")
    if band is not None:
        lines.append(f"band: {BAND_NAME[band]}" + (" (second)" if band == "terse_2" else ""))
    return "\n".join(lines) + "\n" + HEADER


def target_segments(r: dict, a, band: str | None) -> list[tuple[str, str]]:
    """[(band, text)] where text is appended verbatim; last segment gets EOS."""
    if band is not None:
        return [(band, r["labels"][band].strip())]
    if a.special_tokens == 2:
        return [(b, f"{RESERVED[b]}{r['labels'][b].strip()}{LABEL_END}") for b in BANDS]
    if a.special_tokens == 1:
        return [(b, f"<|{BAND_NAME[b]}|>{r['labels'][b].strip()}") for b in BANDS]
    return [(b, f"{BAND_NAME[b]}: {r['labels'][b].strip()}\n") for b in BANDS]


def load(path: Path, n: int | None, rank: int = 0, world: int = 1) -> list[dict]:
    """First n rows (all if None); with world > 1 each rank keeps every world-th of them."""
    rows = []
    with open(path) as f:
        for i, l in enumerate(f):
            if n is not None and i >= n:
                break
            if i % world == rank:
                rows.append(json.loads(l))
    return rows


class Encoder:
    def __init__(self, tok, a):
        self.tok, self.a = tok, a
        self.eos = tok.convert_tokens_to_ids("<|end|>") if a.special_tokens == 1 else tok.eos_token_id

    def encode(self, r: dict, band: str | None):
        a, tok = self.a, self.tok
        p = tok(prompt_text(r, a, band), add_special_tokens=True)["input_ids"][-a.max_in:]
        tgt, seg = [], []
        for b, text in target_segments(r, a, band):
            ids = tok(text, add_special_tokens=False)["input_ids"]
            tgt += ids
            seg += [BANDS.index(b)] * len(ids)
        tgt.append(self.eos)
        seg.append(seg[-1])
        nbytes = [0] * len(BANDS)
        for b, text in target_segments(r, a, band):
            nbytes[BANDS.index(b)] += len(r["labels"][b].strip().encode())
        return p, tgt[: a.max_out], seg[: a.max_out], nbytes

    def encode_parallel(self, r: dict):
        """One encoder input, five decoder targets <band>text<end> (no EOS needed)."""
        a, tok = self.a, self.tok
        p = tok(prompt_text(r, a, None), add_special_tokens=True)["input_ids"][-a.max_in:]
        tgts = [tok(text, add_special_tokens=False)["input_ids"][:110] for _, text in target_segments(r, a, None)]
        nbytes = [len(r["labels"][b].strip().encode()) for b in BANDS]
        return p, tgts, list(range(len(BANDS))), nbytes


def collate(batch, arch: str, pad: int, device):
    if arch == "parallel":
        # encoder rows = records; decoder rows = records x 5 bands (row i*5+b)
        Lp = max(len(p) for p, *_ in batch)
        Lt = max(len(t) for _, ts, _, _ in batch for t in ts)
        ids = torch.full((len(batch), Lp), pad, dtype=torch.long)
        att = torch.zeros((len(batch), Lp), dtype=torch.long)
        lab = torch.full((len(batch) * 5, Lt), -100, dtype=torch.long)
        segm = torch.full((len(batch) * 5, Lt), -1, dtype=torch.long)
        for i, (p, ts, _, _) in enumerate(batch):
            ids[i, : len(p)] = torch.tensor(p)
            att[i, : len(p)] = 1
            for b, t in enumerate(ts):
                lab[i * 5 + b, : len(t)] = torch.tensor(t)
                segm[i * 5 + b, : len(t)] = b
        return (dict(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.clamp(min=0).to(device),
                     n_rep=5), lab.to(device), segm.to(device))
    if arch == "causal":
        # left-pad so every target sits in the last Lt positions; the model then
        # only materialises logits there (full-vocab logits over 16k tokens OOM)
        L = max(len(p) + len(t) for p, t, _, _ in batch)
        Lt = max(len(t) for _, t, _, _ in batch)
        ids = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        lab = torch.full((len(batch), Lt), -100, dtype=torch.long)
        segm = torch.full((len(batch), Lt), -1, dtype=torch.long)
        for i, (p, t, s, _) in enumerate(batch):
            n = len(p) + len(t)
            ids[i, L - n:] = torch.tensor(p + t)
            att[i, L - n:] = 1
            lab[i, Lt - len(t):] = torch.tensor(t)
            segm[i, Lt - len(t):] = torch.tensor(s)
        pos = (att.cumsum(1) - 1).clamp(min=0)
        return (dict(input_ids=ids.to(device), attention_mask=att.to(device), position_ids=pos.to(device),
                     logits_to_keep=Lt + 1), lab.to(device), segm.to(device))
    Lp = max(len(p) for p, *_ in batch)
    Lt = max(len(t) for _, t, _, _ in batch)
    ids = torch.full((len(batch), Lp), pad, dtype=torch.long)
    att = torch.zeros((len(batch), Lp), dtype=torch.long)
    lab = torch.full((len(batch), Lt), -100, dtype=torch.long)
    segm = torch.full((len(batch), Lt), -1, dtype=torch.long)
    for i, (p, t, s, _) in enumerate(batch):
        ids[i, : len(p)] = torch.tensor(p)
        att[i, : len(p)] = 1
        lab[i, : len(t)] = torch.tensor(t)
        segm[i, : len(t)] = torch.tensor(s)
    return dict(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.clamp(min=0).to(device)), lab.to(device), segm.to(device)


def token_nll(model, inputs, lab, arch):
    n_rep = inputs.pop("n_rep", None)
    if n_rep:  # run the encoder once, share its states across the 5 band decoders
        from transformers.modeling_outputs import BaseModelOutput
        h = model.get_encoder()(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        out = model(encoder_outputs=BaseModelOutput(last_hidden_state=h.repeat_interleave(n_rep, 0)),
                    attention_mask=inputs["attention_mask"].repeat_interleave(n_rep, 0), labels=inputs["labels"])
        logits = out.logits[:, : lab.shape[1]].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1).clamp(min=0),
                               reduction="none").view_as(lab) * (lab != -100)
    out = model(**inputs)
    # causal: logits cover the last Lt+1 positions; drop the final one (predicts past EOS)
    logits = (out.logits[:, :-1] if arch == "causal" else out.logits)[:, : lab.shape[1]].float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1).clamp(min=0),
                           reduction="none").view_as(lab) * (lab != -100)


def batches(examples, max_tokens: int, rng: random.Random):
    """Length-bucketed batches under a padded-token budget."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    for i in range(0, len(order), 512):
        pool = sorted(order[i : i + 512], key=lambda j: len(examples[j][0]) + len(examples[j][1]))
        cur, cur_max = [], 0
        for j in pool:
            n = len(examples[j][0]) + len(examples[j][1])
            if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
                yield [examples[k] for k in cur]
                cur, cur_max = [], 0
            cur.append(j)
            cur_max = max(cur_max, n)
        if cur:
            yield [examples[k] for k in cur]


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, val_ex, a, pad, device):
    model.eval()
    nll = torch.zeros(len(BANDS), dtype=torch.float64)
    nb = torch.zeros(len(BANDS), dtype=torch.float64)
    for batch in batches(val_ex, a.eval_tokens, random.Random(0)):
        inputs, lab, segm = collate(batch, "parallel" if a.output == "parallel" else a.arch, pad, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tl = token_nll(model, inputs, lab, a.arch)
        for b in range(len(BANDS)):
            nll[b] += tl[segm == b].sum().double().cpu()
        # EOS token belongs to the last segment; bytes counted once per label
        for *_, nbytes in batch:
            nb += torch.tensor(nbytes, dtype=torch.float64)
    model.train()
    per = (nll / nb.clamp(min=1)).tolist()
    return float(nll.sum() / nb.sum()), dict(zip(BANDS, per))


LEAK = re.compile(r"\b[a-h][1-8]\b|\bmove \d+|\d+\s*(%|percent|captures|plies|moves)|\b[KQRBN]x?[a-h][1-8]\b|O-O")


def parse_joint(text: str) -> dict:
    """Split on band markers, not newlines (T5 tokenizers drop \n)."""
    for name in ("terse", "short", "medium", "long"):
        text = text.replace(f"<|{name}|>", f" {name}: ")
    text = text.split("<|end|>")[0]
    got = {}
    for m in re.finditer(r"\b(terse|short|medium|long)\s*:\s*(.*?)(?=\b(?:terse|short|medium|long)\s*:|$)", text, re.S):
        b, body = m.group(1), m.group(2).strip()
        if not body:
            continue
        key = ("terse_2" if "terse_1" in got else "terse_1") if b == "terse" else b
        got.setdefault(key, " ".join(body.split()))
    return got


def parse_reserved_ids(ids: list[int], tok) -> dict:
    """Split generated ids on reserved band / end tokens; no text parsing."""
    band_of = {tok.convert_tokens_to_ids(t): b for b, t in RESERVED.items()}
    end = tok.convert_tokens_to_ids(LABEL_END)
    got, cur, buf = {}, None, []
    for i in ids + [end]:
        if i in band_of or i == end or i == tok.eos_token_id:
            if cur is not None and buf and cur not in got:
                got[cur] = tok.decode(buf, skip_special_tokens=True).strip()
            cur, buf = band_of.get(i), []
            if i == tok.eos_token_id:
                break
        elif cur is not None:
            buf.append(i)
    return got


@torch.no_grad()
def gen_parallel(model, tok, chunk, a, device):
    """Encode each game once, decode the 5 labels as 5 parallel short sequences."""
    from transformers.modeling_outputs import BaseModelOutput
    enc = tok([prompt_text(r, a, None) for r in chunk], return_tensors="pt", padding=True,
              truncation=True, max_length=a.max_in).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.get_encoder()(**enc).last_hidden_state.repeat_interleave(5, 0)
        start = (model.generation_config.decoder_start_token_id if model.generation_config.decoder_start_token_id
                 is not None else tok.bos_token_id)
        band_ids = [tok.convert_tokens_to_ids(RESERVED[b]) for b in BANDS]
        dec = torch.tensor([[start, band_ids[j]] for _ in chunk for j in range(5)], device=device)
        g = model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=h),
                           attention_mask=enc["attention_mask"].repeat_interleave(5, 0), decoder_input_ids=dec,
                           max_new_tokens=110, do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=[tok.convert_tokens_to_ids(LABEL_END), tok.eos_token_id])
    preds = []
    for k in range(len(chunk)):
        preds.append({b: parse_reserved_ids(g[k * 5 + j, 1:].tolist(), tok).get(b, "") for j, b in enumerate(BANDS)})
    return preds, int((g[:, 2:] != tok.pad_token_id).sum())


@torch.no_grad()
def gen_eval(model, tok, rows, a, device):
    model.eval()
    tok.padding_side = "left"
    out_rows, t0, gen_tokens = [], time.time(), 0
    for i in range(0, len(rows), a.gen_bs):
        chunk = rows[i : i + a.gen_bs]
        if a.output == "parallel":
            preds, n_tok = gen_parallel(model, tok, chunk, a, device)
            gen_tokens += n_tok
            for r, p in zip(chunk, preds):
                out_rows.append({"key": r["key"], "labeler": r["labeler"], "pred": p, "teacher": r["labels"]})
            continue
        jobs = [(r, None) for r in chunk] if a.output == "joint" else [(r, b) for r in chunk for b in BANDS]
        enc = tok([prompt_text(r, a, b) for r, b in jobs], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.max_in).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**enc, max_new_tokens=a.max_out if a.output == "joint" else 110,
                               do_sample=False, pad_token_id=tok.pad_token_id,
                               eos_token_id=[tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|end|>")] if a.special_tokens == 1 else []))
        if a.arch == "causal":
            g = g[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(g, skip_special_tokens=False)
        for sp in tok.all_special_tokens:  # our delimiters are added as normal tokens, so they survive
            texts = [t.replace(sp, "") for t in texts]
        gen_tokens += int((g != tok.pad_token_id).sum())
        if a.special_tokens == 2 and a.output == "joint":
            preds = [parse_reserved_ids(row.tolist(), tok) for row in g]
        elif a.output == "joint":
            preds = [parse_joint(t) for t in texts]
        else:
            preds = [{b: texts[k * 5 + j].strip() for j, b in enumerate(BANDS)} for k in range(len(chunk))]
        for r, p in zip(chunk, preds):
            out_rows.append({"key": r["key"], "labeler": r["labeler"], "pred": p, "teacher": r["labels"]})
    dt = time.time() - t0
    model.train()
    tok.padding_side = "right"
    complete = [o for o in out_rows if all(b in o["pred"] and o["pred"][b] for b in BANDS)]
    in_range, leak, n = 0, 0, 0
    for o in complete:
        for b in BANDS:
            w = len(o["pred"][b].split())
            lo, hi = WORD_RANGE[BAND_NAME[b]]
            in_range += lo - 2 <= w <= hi + 5
            leak += bool(LEAK.search(o["pred"][b]))
            n += 1
    # teacher's own band compliance on the same records, as the reference
    t_in = sum(WORD_RANGE[BAND_NAME[b]][0] - 2 <= len(o["teacher"][b].split()) <= WORD_RANGE[BAND_NAME[b]][1] + 5
               for o in out_rows for b in BANDS) / max(1, 5 * len(out_rows))
    return {"gen_parse_ok": len(complete) / max(1, len(out_rows)),
            "gen_band_len_ok": in_range / max(1, n), "teacher_band_len_ok": t_in,
            "gen_leak_rate": leak / max(1, n), "gen_records_per_s": len(rows) / dt,
            "gen_tokens_per_s": gen_tokens / dt}, out_rows


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", choices=["causal", "seq2seq"], default="causal")
    ap.add_argument("--data", default="data/labeler")
    ap.add_argument("--out", default="runs/labeler")
    ap.add_argument("--minutes", type=float, default=50)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--output", choices=["joint", "single", "parallel"], default="joint")
    ap.add_argument("--input", choices=["teacher", "compact"], default="compact")
    ap.add_argument("--tag", type=int, default=1)
    ap.add_argument("--max-in", type=int, default=2048)
    ap.add_argument("--max-out", type=int, default=320)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--eval-tokens", type=int, default=32768)
    ap.add_argument("--special-tokens", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=150000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-gen", type=int, default=128)
    ap.add_argument("--gen-split", choices=["val", "test"], default="val")
    ap.add_argument("--gen-bs", type=int, default=32)
    ap.add_argument("--eval-every-min", type=float, default=10)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--save", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=0, help=">0: step-based schedule over this many epochs (ignores --minutes)")
    ap.add_argument("--save-every-min", type=float, default=0, help=">0: rank 0 saves ckpt_<trial> this often")
    a = ap.parse_args()

    # multi-GPU: one process per GPU (srun sets RANK/WORLD_SIZE), rows sharded by rank, DDP gradients
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        torch.distributed.init_process_group("nccl")
    main_proc = rank == 0
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed + rank)
    device = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    Model = AutoModelForCausalLM if a.arch == "causal" else AutoModelForSeq2SeqLM
    model = Model.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    new_ids = []
    if a.output == "parallel":
        assert a.arch == "seq2seq" and a.special_tokens == 2, "parallel decoding needs seq2seq + reserved tokens"
    if a.special_tokens == 1:
        tok.add_tokens(list(SPECIAL))  # non-special: survive decode, parsed explicitly
        new_ids = tok.convert_tokens_to_ids(list(SPECIAL))
        old = model.get_input_embeddings().weight.shape[0]
        if max(new_ids) >= old:
            model.resize_token_embeddings(len(tok), mean_resizing=False)
        emb = model.get_input_embeddings().weight
        out_emb = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else None
        with torch.no_grad():
            for t, i in zip(SPECIAL.values(), new_ids):
                src = tok(t, add_special_tokens=False)["input_ids"]
                emb[i] = emb[src].mean(0)
                if out_emb is not None and out_emb.data_ptr() != emb.data_ptr():
                    out_emb[i] = out_emb[src].mean(0)
    if a.lora_r > 0:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        extra = {"trainable_token_indices": new_ids} if new_ids else {}
        model = get_peft_model(model, LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha or a.lora_r, lora_dropout=0.0, **extra,
            target_modules="all-linear", task_type="CAUSAL_LM" if a.arch == "causal" else "SEQ_2_SEQ_LM"))
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    raw = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)

    enc = Encoder(tok, a)
    train_rows = load(Path(a.data) / "train.jsonl", a.n_train or None, rank, world)
    val_rows = load(Path(a.data) / "val.jsonl", a.n_val)
    t0 = time.time()
    if a.output == "parallel":
        train_ex = [enc.encode_parallel(r) for r in train_rows]
        val_ex = [enc.encode_parallel(r) for r in val_rows]
    elif a.output == "joint":
        train_ex = [enc.encode(r, None) for r in train_rows]
        val_ex = [enc.encode(r, None) for r in val_rows]
    else:  # one sampled band per record for training (prompt cost is paid per label)
        train_ex = [enc.encode(r, rng.choice(BANDS)) for r in train_rows]
        val_ex = [enc.encode(r, b) for r in val_rows for b in BANDS]
    if main_proc: print(f"[{a.trial}] tokenized {len(train_ex)} train in {time.time()-t0:.0f}s; "
          f"mean prompt {sum(len(e[0]) for e in train_ex)/len(train_ex):.0f} tok, "
          f"mean target {sum(len(e[1]) if a.output != 'parallel' else sum(map(len, e[1])) for e in train_ex)/len(train_ex):.0f} tok; "
          f"params {n_params/1e9:.2f}B trainable {n_train_params/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr,
                            weight_decay=a.wd, betas=(0.9, 0.95), fused=True)
    budget = a.minutes * 60
    history = []
    step, seen, tok_seen, train_time = 0, 0, 0, 0.0
    next_eval = a.eval_every_min * 60
    ema = None
    model.train()
    done = False
    epoch = 0
    total_steps = eval_steps = 0
    if a.epochs:
        ep_batches = list(batches(train_ex, a.batch_tokens, rng))
        n_ep = torch.tensor(len(ep_batches), device=device)
        if world > 1:  # every rank must take the same number of steps
            torch.distributed.all_reduce(n_ep, op=torch.distributed.ReduceOp.MIN)
        total_steps = int(int(n_ep) * a.epochs)
        eval_steps = max(1, total_steps // 12)
        if main_proc: print(f"[{a.trial}] {total_steps} steps x {world} ranks ({int(n_ep)} per epoch)", flush=True)
    last_save = time.time()
    while not done:
        it = (ep_batches if epoch == 0 else list(batches(train_ex, a.batch_tokens, rng)))[: int(n_ep)] if a.epochs \
            else batches(train_ex, a.batch_tokens, rng)
        for batch in it:
            ts = time.time()
            # warmup then cosine on progress: steps if --epochs, else wall clock (budget is the schedule)
            frac = step / total_steps if a.epochs else train_time / budget
            lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0))))
            for g in opt.param_groups:
                g["lr"] = lr
            inputs, lab, _ = collate(batch, "parallel" if a.output == "parallel" else a.arch, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tl = token_nll(model, inputs, lab, a.arch)
            loss = tl.sum() / (lab != -100).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            train_time += time.time() - ts
            step += 1
            seen += len(batch)
            tok_seen += int(inputs["attention_mask"].sum())
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 25 == 0 and main_proc:
                print(f"[{a.trial}] step {step} ex {seen} ep {epoch} loss {ema:.4f} lr {lr:.2e} "
                      f"{tok_seen/train_time:.0f} tok/s {seen/train_time:.2f} ex/s "
                      f"mem {torch.cuda.max_memory_allocated()/2**30:.0f}G", flush=True)
            finished = step >= total_steps if a.epochs else train_time >= budget
            if (step % eval_steps == 0 if a.epochs else train_time >= next_eval) or finished:
                if main_proc:
                    v, per = evaluate(raw, val_ex, a, tok.pad_token_id, device)
                    history.append({"train_min": train_time / 60, "examples": seen * world, "val_nats_per_byte": v})
                    print(f"[{a.trial}] EVAL {train_time/60:.1f}min ex {seen * world} val {v:.4f} " +
                          " ".join(f"{b}={x:.3f}" for b, x in per.items()), flush=True)
                next_eval += a.eval_every_min * 60
            if main_proc and a.save_every_min and time.time() - last_save > a.save_every_min * 60:
                raw.save_pretrained(out / f"ckpt_{a.trial}")
                tok.save_pretrained(out / f"ckpt_{a.trial}")
                print(f"[{a.trial}] saved ckpt at step {step}", flush=True)
                last_save = time.time()
            if finished:
                done = True
                break
        epoch += 1

    if world > 1:
        torch.distributed.barrier()
        if not main_proc:
            torch.distributed.destroy_process_group()
            return
    model = raw
    v, per = evaluate(model, val_ex, a, tok.pad_token_id, device)
    if a.grad_ckpt:
        model.config.use_cache = True
    gen_rows = val_rows if a.gen_split == "val" else load(Path(a.data) / "test.jsonl", None)
    gen, samples = gen_eval(model, tok, gen_rows[: a.n_gen], a, device)
    res = {"trial": a.trial, "model": a.model, "arch": a.arch, "lora_r": a.lora_r, "lr": a.lr,
           "output": a.output, "input": a.input, "tag": a.tag, "batch_tokens": a.batch_tokens,
           "minutes": a.minutes, "params_b": n_params / 1e9, "trainable_m": n_train_params / 1e6,
           "steps": step, "examples": seen * world, "epochs": seen / len(train_ex), "world": world,
           "train_tok_per_s": tok_seen / train_time, "val_nats_per_byte": v, "val_per_band": per,
           "history": history, **gen, "wall_min": (time.time() - t_start) / 60}
    print(f"[{a.trial}] FINAL " + json.dumps(res), flush=True)
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    with open(out / f"samples_{a.trial}.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    if a.save:
        model.save_pretrained(out / f"ckpt_{a.trial}")
        tok.save_pretrained(out / f"ckpt_{a.trial}")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
