"""Supervised fine-tuning with text: teach a games-only CHSM8 to follow style descriptions and player prompts.

    torchrun --nproc-per-node 4 scripts/train/sft.py configs/sft.yaml       # chsm8-sft (we used 2 nodes x 4 GPUs)
    python scripts/train/sft.py configs/nano_sft.yaml                       # the nano model, any machine

The top `text_layers` layers get cross-attention to a frozen text encoder (text.py). Every described game side is
trained with all its descriptions at once (the bottom layers run once per game). One pass over the training sides;
the player games (`player_data`, `player_passes` passes) are spread evenly over the run. Loss: the pretraining loss
on the described side's moves only; the opponent's moves are context.
"""
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8 import checkpoint, config, style  # noqa: E402
from chsm8.data import Sides, length_batches, prefetch  # noqa: E402
from chsm8.model import loss_fn, n_params  # noqa: E402
from chsm8.text import TextEncoder  # noqa: E402

c = config.load()
ddp = "WORLD_SIZE" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
master = rank == 0
torch.manual_seed(c.seed)  # same fresh text weights on every rank
torch.backends.cuda.matmul.allow_tf32 = True

# ---- model and text encoder
repo, _, filename = c.init.partition(":")
# bf16_constant=False: our SFT re-computed the pretraining scaffold's constant in fp32 when it pinned it
model = checkpoint.load(repo, filename or "model.pt", device, text_layers=c.text_layers, bf16_constant=False).train()
encoder = TextEncoder(c.encoder, device=device)
groups = style.param_groups(model, c.lr, c.text_lr)
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=c.weight_decay, fused=device.type == "cuda")
params = [p for g in groups for p in g["params"]]
if master:
    print(f"{n_params(model) / 1e6:.1f}M parameters, {sum(p.numel() for p in groups[0]['params']) / 1e6:.1f}M text "
          f"path + {sum(p.numel() for p in groups[1]['params']) / 1e6:.1f}M decoder trained, {world} GPU(s)", flush=True)

# ---- data: a fixed schedule of batches, the same length on every rank
rng = np.random.default_rng(c.seed)
sides = Sides(c.data, columns=("labels",), load=c.load_in_ram)
train_ids = rng.permutation(sides.split["train"])
if c.max_games:
    train_ids = train_ids[:c.max_games]
n_labels = len(sides.get("labels", train_ids[0]))
mine = np.sort(train_ids[rank::world])
schedule = [("sides", b) for b in length_batches(sides.n_rows, mine, c.games, c.token_budget // n_labels,
                                                 np.random.default_rng([c.seed, rank]))]
players = None
if c.player_data:
    players = Sides(c.player_data, columns=("player",), load=True)
    pb = [("players", b) for p in range(c.player_passes)
          for b in length_batches(players.n_rows, np.sort(players.split["train"][rank::world]), c.games,
                                  c.token_budget, np.random.default_rng([c.seed, rank, p]))]
counts = torch.tensor([len(schedule), len(pb) if players else 0], device=device)
if ddp:
    dist.all_reduce(counts, op=dist.ReduceOp.MIN)
n_sides, n_players = (int(x) for x in counts)
schedule = schedule[:n_sides]
if players:  # player batch j goes in at progress (j + 0.5) / n_players
    keys = np.concatenate([np.arange(n_sides), (np.arange(n_players) + 0.5) / n_players * n_sides - 1e-9])
    merged = schedule + pb[:n_players]
    schedule = [merged[i] for i in np.argsort(keys, kind="stable")]
total_steps = len(schedule)
val_ids = np.sort(np.random.default_rng(0).choice(sides.split["validation"],
                                                  min(c.eval_games, len(sides.split["validation"])), replace=False))
if master:
    print(f"{len(train_ids):,} training sides, {n_labels} descriptions each"
          + (f", {len(players.split['train']):,} player games x {c.player_passes}" if players else "")
          + f": {total_steps:,} steps", flush=True)

step = 0
os.makedirs(c.out, exist_ok=True)
last = os.path.join(c.out, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step = ck["step"]
    if master:
        print(f"resumed at step {step:,}", flush=True)
dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if ddp else model
text_rng = random.Random(c.seed * 1000 + rank)


def load(item):
    src = sides if item[0] == "sides" else players
    b = src.batch(item[1])
    b["texts"] = style.texts(src, b["ids"], text_rng)
    if device.type == "cuda":
        b = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in b.items()}
    return b


def evaluate(tag):
    if master:
        out = style.evaluate(model, encoder, sides, val_ids)
        if players:
            p = style.evaluate(model, encoder, players, players.split["validation"][:c.eval_games])
            out.update({f"player_{k}": v for k, v in p.items()})
        print(f"{tag} | val " + " | ".join(f"{k} {v:.4f}" for k, v in out.items()), flush=True)
    if ddp:
        dist.barrier()


def save(path, **extra):
    if master:
        checkpoint.save(model, path + ".tmp", **extra)
        os.replace(path + ".tmp", path)


t0, window = time.time(), []
for b in prefetch(load, schedule[step:]):
    mult = style.lr_mult(step / total_steps, c.warmup, c.min_lr_ratio)
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * mult
    b = style.to(b, device)
    k = len(b["texts"][0])
    R = lambda t: t.repeat_interleave(k, 0)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        text = style.encode(encoder, b["texts"], c.description_dropout, device)
        logits = dmodel(b["rows"][:, :-1], text, repeat=k)
        loss = loss_fn(logits, R(style.focal_targets(b["rows"], b["focal"])), R(b["occ"]), R(b["pmask"]),
                       c.legal_penalty)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
    opt.step()
    step += 1
    window.append(loss.detach())
    if master and step % c.log_every == 0:
        gates = " ".join(f"{float(torch.tanh(g)):+.3f}" for g in model.gate.detach())
        print(f"step {step:,}/{total_steps:,} | loss {float(torch.stack(window).mean()):.4f} | lr x{mult:.3f} | "
              f"gates {gates} | {time.time() - t0:.0f}s", flush=True)
        window = []
    if step % c.eval_every == 0:
        evaluate(f"step {step:,}")
    if step % c.save_every == 0:
        save(last, optimizer=opt.state_dict(), step=step)

evaluate("final")
if master:
    test = style.evaluate(model, encoder, sides, np.sort(np.random.default_rng(0).choice(
        sides.split["test"], min(c.eval_games, len(sides.split["test"])), replace=False)))
    print("final | test " + " | ".join(f"{k} {v:.4f}" for k, v in test.items()), flush=True)
    save(os.path.join(c.out, "model.pt"))
    print(f"done: {os.path.join(c.out, 'model.pt')}", flush=True)
if ddp:
    dist.destroy_process_group()
