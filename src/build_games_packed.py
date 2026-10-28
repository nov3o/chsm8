#!/usr/bin/env python3
"""Pack plain games (no text labels) into the pretraining PrePackedDataset
format, for text-free continued training from the pretrained checkpoint.

Output layout (what train_pretrain.PrePackedDataset reads):
  <out>/{train,val,test}/packed_NNNNN.i16        (n, seq_len+1, 5) factored rows
                         packed_NNNNN_occ.bin    (n, seq_len+1) uint64  own-piece occupancy
                         packed_NNNNN_pmask.bin  (n, seq_len+1) uint8   piece-type mask
                         packed_NNNNN_side.u1    (n, seq_len+1) uint8   who played the move in
                                                 this row: 1 white, 2 black, 0 BOS/EOS/PAD;
                                                 +4 when that side is the game's *focal*
                                                 player (only set for --source players)
                         _SUCCESS
Games are packed greedily and never truncated (same rule as build_tier3_packed.py).

Sources
  pool     data/chsm8-post-training-unlabeled (30.1M game-sides). A game that
           appears once per side is packed ONCE (whole-sequence training). Split = the
           same deterministic 90/5/5 game hash as build_sft_corpus.py, so val/test games
           are the SFT val/test games.
  players  JSONL shards from src/crawl_top_players.py (LAN moves + player_side). One
           record per (player, game); the focal side is marked (+4) in _side.u1 so
           training can restrict loss to that player's moves. Split: the --holdout
           newest games of each file go to val/test (half each), the rest to train;
           --repeat-to N upsamples train (whole-game repetition) to N games.

Usage
  python src/build_games_packed.py --source pool --out data/games30m --workers 256
  python src/build_games_packed.py --source players --inputs data/top_players_top10/games/2016192.jsonl \
      --out data/player_2016192 --holdout 2000 --repeat-to 20000
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_corpus import stable_split, to_codes  # noqa: E402
from build_tier3 import LEGAL_DTYPE, replay_game  # noqa: E402
ROW_PAD = (0, 0, 0, 0, 0)  # factored_tokenizer.ROW_PAD (KIND_PAD=0); not imported: it pulls in torch

SIDE_WHITE, SIDE_BLACK, FOCAL = 1, 2, 4
POOL = "data/chsm8-post-training-unlabeled/data"


class ShardWriter:
    """Greedy packer for one output shard (one split, one worker)."""

    def __init__(self, split_dir: str, shard: str, seq_len: int):
        os.makedirs(split_dir, exist_ok=True)
        base = os.path.join(split_dir, f"packed_{shard}")
        self.f = {k: open(base + ext, "wb", buffering=8 << 20)
                  for k, ext in (("rows", ".i16"), ("occ", "_occ.bin"), ("pmask", "_pmask.bin"), ("side", "_side.u1"))}
        self.cap = seq_len + 1
        self.n_rows = self.n_games = 0
        self._new()

    def _new(self):
        self.rows = np.empty((self.cap, 5), dtype=np.int16)
        self.rows[:] = ROW_PAD
        self.occ = np.zeros(self.cap, dtype=np.uint64)
        self.pmask = np.zeros(self.cap, dtype=np.uint8)
        self.side = np.zeros(self.cap, dtype=np.uint8)
        self.p = 0

    def _flush(self):
        for k, a in (("rows", self.rows), ("occ", self.occ), ("pmask", self.pmask), ("side", self.side)):
            self.f[k].write(a.tobytes())
        self.n_rows += 1
        self._new()

    def add(self, rows: np.ndarray, legal: np.ndarray, side: np.ndarray):
        n = len(rows)
        if n > self.cap - self.p:
            self._flush()
        s = slice(self.p, self.p + n)
        self.rows[s], self.occ[s], self.pmask[s], self.side[s] = rows, legal["occ"], legal["pmask"], side
        self.p += n
        self.n_games += 1
        if self.p == self.cap:
            self._flush()

    def close(self):
        if self.p:
            self._flush()
        for fh in self.f.values():
            fh.close()
        return self.n_rows, self.n_games


def game_arrays(codes: np.ndarray, focal: str | None):
    rows, legal = replay_game(codes)
    rows = np.asarray(rows, dtype=np.int16)
    legal = np.asarray(legal, dtype=LEGAL_DTYPE)
    side = np.zeros(len(rows), dtype=np.uint8)
    n_moves = len(codes)  # rows = BOS + moves + EOS
    side[1:1 + n_moves:2] = SIDE_WHITE
    side[2:1 + n_moves:2] = SIDE_BLACK
    if focal:
        side[side == (SIDE_WHITE if focal == "white" else SIDE_BLACK)] += FOCAL
    return rows, legal, side


# ------------------------------------------------------------------ pool
def pool_jobs():
    files = sorted(glob.glob(os.path.join(POOL, "*.parquet")))
    return [(f, g) for f in files for g in range(pq.ParquetFile(f).num_row_groups)]


def pool_keep_masks(jobs, workers):
    """First occurrence of each (month, row_idx) game across the whole pool."""
    with mp.Pool(workers) as pool:
        all_keys = pool.map(_keys_job, jobs)
    seen, masks = set(), []
    for ks in all_keys:
        m = np.zeros(len(ks), dtype=bool)
        for i, k in enumerate(ks):
            if k not in seen:
                seen.add(k)
                m[i] = True
        masks.append(m)
    return masks


def _keys_job(job):
    t = pq.ParquetFile(job[0]).read_row_group(job[1], columns=["month", "row_idx"])
    return [f"{m}:{r}" for m, r in zip(t.column(0).to_pylist(), t.column(1).to_pylist())]


def pool_worker(task):
    wid, (path, rg), keep, out, seq_len, max_rows, seed = task
    t = pq.ParquetFile(path).read_row_group(rg, columns=["month", "row_idx", "moves"]).to_pydict()
    writers = {s: ShardWriter(os.path.join(out, s), f"{wid:05d}", seq_len) for s in ("train", "val", "test")}
    skipped = 0
    for i in np.flatnonzero(keep):
        try:
            codes, _ = to_codes(t["moves"][i].split())
            rows, legal, side = game_arrays(codes, None)
        except Exception:
            skipped += 1
            continue
        if len(rows) > max_rows or len(codes) == 0:
            skipped += 1
            continue
        writers[stable_split(t["month"][i], int(t["row_idx"][i]), seed)].add(rows, legal, side)
    return wid, {s: w.close() for s, w in writers.items()}, skipped


# ------------------------------------------------------------------ players
def lan_to_codes(lan_moves):
    """Crawler LAN ("e2-e4", "Ng1xf3+", "O-O") -> move codes, by matching board.lan()."""
    import chess
    import move_codec
    board, codes = chess.Board(), []
    for text in lan_moves:
        want = text.rstrip("+#")
        mv = next((m for m in board.legal_moves if board.lan(m).rstrip("+#") == want), None)
        if mv is None:
            raise ValueError(f"illegal LAN {text!r}")
        codes.append(move_codec.encode(mv))
        board.push(mv)
    return np.asarray(codes, dtype=np.uint16)


def players_build(args):
    recs = []
    for path in args.inputs:
        for line in open(path):
            r = json.loads(line)
            recs.append(r)
    recs.sort(key=lambda r: r["date"])  # oldest -> newest; newest go to val/test
    n_hold = min(args.holdout, len(recs) // 5)
    hold, train = recs[len(recs) - n_hold:], recs[:len(recs) - n_hold]
    val, test = hold[0::2], hold[1::2]
    rng = np.random.default_rng(args.seed)
    if args.repeat_to and len(train) < args.repeat_to:
        extra = rng.choice(len(train), args.repeat_to - len(train), replace=len(train) < args.repeat_to - len(train))
        train = train + [train[i] for i in extra]
    rng.shuffle(train)
    stats = {}
    for split, rs in (("train", train), ("val", val), ("test", test)):
        w = ShardWriter(os.path.join(args.out, split), "00000", args.seq_len)
        bad = 0
        for r in rs:
            try:
                rows, legal, side = game_arrays(lan_to_codes(r["moves"]), r["player_side"])
            except Exception:
                bad += 1
                continue
            if len(rows) > args.seq_len + 1:
                bad += 1
                continue
            w.add(rows, legal, side)
        n_rows, n_games = w.close()
        write_success(args.out, split, 1, n_rows, n_games, args.seq_len)
        stats[split] = {"games": n_games, "packed_rows": n_rows, "skipped": bad}
    stats["unique_train_games"] = len({(r["date"], " ".join(r["moves"])) for r in train})
    stats["holdout_date_from"] = hold[0]["date"] if hold else None
    return stats


def write_success(out, split, n_shards, n_rows, n_games, seq_len):
    with open(os.path.join(out, split, "_SUCCESS"), "w") as f:
        f.write(f"n_shards={n_shards} n_rows={n_rows} n_games={n_games} seq_len={seq_len} has_legal=True has_side=True\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("pool", "players"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--workers", type=int, default=min(256, os.cpu_count() or 1))
    ap.add_argument("--split-seed", type=int, default=20260802, help="same as build_sft_corpus.py")
    ap.add_argument("--holdout", type=int, default=2000, help="players: newest games held out (val+test)")
    ap.add_argument("--repeat-to", type=int, default=0, help="players: upsample train to this many games")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.source == "players":
        stats = players_build(args)
    else:
        jobs = pool_jobs()
        print(f"[pool] {len(jobs)} row groups; dedup pass", flush=True)
        masks = pool_keep_masks(jobs, args.workers)
        print(f"[pool] {sum(len(m) for m in masks):,} game-sides -> {sum(int(m.sum()) for m in masks):,} unique games",
              flush=True)
        tasks = [(i, job, masks[i], args.out, args.seq_len, args.seq_len + 1, args.split_seed)
                 for i, job in enumerate(jobs)]
        totals = {s: [0, 0] for s in ("train", "val", "test")}
        skipped = 0
        with mp.Pool(args.workers) as pool:
            for k, (wid, res, sk) in enumerate(pool.imap_unordered(pool_worker, tasks)):
                skipped += sk
                for s, (r, g) in res.items():
                    totals[s][0] += r
                    totals[s][1] += g
                if k % 50 == 0:
                    print(f"[pool] {k + 1}/{len(tasks)} row groups, train games {totals['train'][1]:,}", flush=True)
        for s, (r, g) in totals.items():
            write_success(args.out, s, len(tasks), r, g, args.seq_len)
        stats = {s: {"games": g, "packed_rows": r} for s, (r, g) in totals.items()}
        stats["skipped"] = skipped
    stats["tokens_train"] = stats["train"]["packed_rows"] * (args.seq_len + 1)
    Path(args.out, "summary.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
