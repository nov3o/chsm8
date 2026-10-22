"""Train CHSM8 on packed games, with no text: pretraining, the games-only continuation, and the player models.

    torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/pt.yaml            # chsm8-pt
    torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/player_base.yaml   # chsm8-player-base
    torchrun --nproc-per-node 4 scripts/train/pretrain.py configs/player.yaml --data=data/players/carlsen
    python scripts/train/pretrain.py configs/nano.yaml                                # a few minutes, any machine

One optimizer step = batch_size records of seq_len positions per GPU. The learning rate follows the number of
positions seen: linear warmup, then cosine decay to min_lr_ratio x the peak. Interrupted runs resume from
<out>/last.pt.
"""
import math
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8 import checkpoint, config  # noqa: E402
from chsm8.data import PackedData  # noqa: E402
from chsm8.model import CHSM8, Config, init_weights, loss_fn, n_params  # noqa: E402

c = config.load()
ddp = "WORLD_SIZE" in os.environ
if ddp:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    rank, world = 0, 1
device = "cuda" if torch.cuda.is_available() else "cpu"
master = rank == 0
torch.manual_seed(c.seed + rank)
torch.backends.cuda.matmul.allow_tf32 = True
autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

# ---- model
if c.init:  # continue from a released or saved checkpoint, e.g. LegumMagister/chsm8-pt
    repo, _, filename = c.init.partition(":")
    model = checkpoint.load(repo, filename or "model.pt", device).train()
else:
    model = init_weights(CHSM8(Config(n_layers=c.n_layers, d_model=c.d_model, n_heads=c.n_heads, d_ff=c.d_ff)),
                         seed=c.seed).to(device)
for layer in model.layers:
    layer.no_text.requires_grad_(c.train_no_text)  # pretraining keeps the scaffold's constant frozen
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=c.lr, betas=(0.9, 0.95), weight_decay=c.weight_decay, fused=device == "cuda")
if master:
    print(f"{n_params(model) / 1e6:.1f}M parameters, {sum(p.numel() for p in params) / 1e6:.1f}M trained, "
          f"{world} GPU(s)", flush=True)

# ---- data and budget
train, val = PackedData(c.data, "train"), PackedData(c.data, "val")
positions_per_step = c.batch_size * train.positions_per_record * world
if c.passes:  # budget in passes over the data instead of positions
    c.max_positions = int(c.passes * train.n_records * train.positions_per_record)
total_steps = math.ceil(c.max_positions / positions_per_step)

step, seen = 0, 0
os.makedirs(c.out, exist_ok=True)
last = os.path.join(c.out, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); step, seen = ck["step"], ck["seen"]
    if master:
        print(f"resumed at step {step:,} ({seen:,} positions)", flush=True)
batches = train.batches(c.batch_size, seed=c.seed, rank=rank, world=world,
                        start=step * c.batch_size * world, focal_only=c.focal_only)
dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()]) if ddp \
    else model
if c.compile:
    dmodel = torch.compile(dmodel)


def lr_at(progress):
    if progress < c.warmup:
        return c.lr * progress / c.warmup
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - c.warmup) / (1 - c.warmup))))
    return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * cos)


def batch_to(b):
    return [t.to(device, non_blocking=True) for t in b]


@torch.no_grad()
def evaluate():
    """Mean loss (without the legality penalty) on the first eval_batches batches of the validation split."""
    model.eval()
    it, losses = val.batches(c.batch_size, seed=0, rank=rank, world=world, focal_only=c.focal_only), []
    for _ in range(c.eval_batches):
        x, y, occ, pmask = batch_to(next(it))
        with autocast:
            losses.append(float(loss_fn(model(x), y)))
    model.train()
    return sum(losses) / len(losses)


def save(path):
    if master:
        checkpoint.save(model, path + ".tmp", optimizer=opt.state_dict(), step=step, seen=seen)
        os.replace(path + ".tmp", path)


t0, window = time.time(), []
while step < total_steps:
    lr = lr_at(seen / c.max_positions)
    for g in opt.param_groups:
        g["lr"] = lr
    x, y, occ, pmask = batch_to(next(batches))
    with autocast:
        loss = loss_fn(dmodel(x), y, occ, pmask, c.legal_penalty)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, c.grad_clip)
    opt.step()
    step, seen = step + 1, seen + positions_per_step
    window.append(loss.detach())
    if master and step % c.log_every == 0:
        dt = time.time() - t0
        print(f"step {step:,}/{total_steps:,} | loss {float(torch.stack(window).mean()):.4f} | lr {lr:.2e} | "
              f"{len(window) * positions_per_step / dt / 1e3:.0f}k positions/s", flush=True)
        t0, window = time.time(), []
    if step % c.eval_every == 0 or step == total_steps:
        v = evaluate()
        if master:
            print(f"step {step:,} | val loss {v:.4f}", flush=True)
    if step % c.save_every == 0:
        save(last)

save(last)
if master:
    checkpoint.save(model, os.path.join(c.out, "model.pt"))
    print(f"done: {os.path.join(c.out, 'model.pt')}", flush=True)
if ddp:
    dist.destroy_process_group()
