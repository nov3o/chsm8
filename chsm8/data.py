"""Training data on disk, and batches from it.

Two formats, both plain numpy files that are memory-mapped, never parsed at train time:

Packed records (pretraining, games-only continuation, player fine-tunes)
    <dir>/{train,val,test}/<name>.rows.i16    (R, seq_len+1, 5) int16   rows of whole games, PAD-filled
                           <name>.occ.u64     (R, seq_len+1)   uint64   legality bits per row (tokenizer.legal_bits)
                           <name>.pmask.u8    (R, seq_len+1)   uint8
                           <name>.focal.u8    (R, seq_len+1)   uint8    optional: 1 = a move of the player we fit
    Games are packed greedily in order and never split; a record holds seq_len+1 rows so that inputs and targets
    are both seq_len long.

Game sides (SFT, DPO): one game per record, plus which side is described and its descriptions
    <dir>/rows.i16, <dir>/occ.u64, <dir>/pmask.u8   all games' rows concatenated
    <dir>/index.parquet                            one row per described side: split, side, row_off, n_rows,
                                                   labels (list of strings), labeler, player (or null)
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from chsm8.tokenizer import KIND_MOVE


# ------------------------------------------------------------------ packed records

class Packer:
    """Writes one packed shard: add() whole games, close() at the end."""

    def __init__(self, path_stem, seq_len=512, focal=False):
        os.makedirs(os.path.dirname(path_stem), exist_ok=True)
        self.cap, self.focal = seq_len + 1, focal
        exts = ["rows.i16", "occ.u64", "pmask.u8"] + (["focal.u8"] if focal else [])
        self.files = [open(f"{path_stem}.{e}", "wb", buffering=16 << 20) for e in exts]
        self.n_records = self.n_games = self.pos = 0
        self._new()

    def _new(self):
        self.buf = [np.zeros((self.cap, 5), "<i2"), np.zeros(self.cap, "<u8"), np.zeros(self.cap, "u1")]
        if self.focal:
            self.buf.append(np.zeros(self.cap, "u1"))
        self.pos = 0

    def _flush(self):
        for f, b in zip(self.files, self.buf):
            f.write(b.tobytes())
        self.n_records += 1
        self._new()

    def add(self, rows, occ, pmask, focal=None):
        """Returns False (and skips the game) if it is longer than one record."""
        n = len(rows)
        if n > self.cap:
            return False
        if n > self.cap - self.pos:
            self._flush()
        s = slice(self.pos, self.pos + n)
        self.buf[0][s], self.buf[1][s], self.buf[2][s] = rows, occ, pmask
        if self.focal:
            self.buf[3][s] = focal
        self.pos += n
        self.n_games += 1
        if self.pos == self.cap:
            self._flush()
        return True

    def close(self):
        if self.pos:
            self._flush()
        for f in self.files:
            f.close()
        return self.n_records, self.n_games


class PackedData:
    """All shards of one split, memory-mapped. Yields (inputs, targets, occ, pmask) as torch tensors, where
    targets are the inputs shifted by one and occ/pmask are aligned with the targets."""

    def __init__(self, root, split="train"):
        stems = sorted(str(p)[:-len(".rows.i16")] for p in Path(root, split).glob("*.rows.i16"))
        assert stems, f"no packed shards in {root}/{split}"
        self.cap = json.loads(Path(root, "info.json").read_text())["seq_len"] + 1  # written by the prepare scripts
        self.shards = []
        for s in stems:
            rows = np.memmap(s + ".rows.i16", dtype="<i2", mode="r").reshape(-1, self.cap, 5)
            occ = np.memmap(s + ".occ.u64", dtype="<u8", mode="r").reshape(-1, self.cap)
            pmask = np.memmap(s + ".pmask.u8", dtype="u1", mode="r").reshape(-1, self.cap)
            focal = (np.memmap(s + ".focal.u8", dtype="u1", mode="r").reshape(-1, self.cap)
                     if os.path.exists(s + ".focal.u8") else None)
            self.shards.append((rows, occ, pmask, focal))
        self.bounds = np.cumsum([0] + [len(s[0]) for s in self.shards])
        self.n_records = int(self.bounds[-1])
        self.positions_per_record = self.cap - 1

    def get(self, idx, focal_only=False):
        idx = np.asarray(idx)
        shard = np.searchsorted(self.bounds, idx, side="right") - 1
        rows = np.empty((len(idx), self.cap, 5), np.int64)
        occ = np.zeros((len(idx), self.cap), np.uint64)
        pmask = np.zeros((len(idx), self.cap), np.int64)
        focal = np.ones((len(idx), self.cap), bool)
        for s in np.unique(shard):
            sel, local = shard == s, idx[shard == s] - self.bounds[s]
            r, o, p, f = self.shards[s]
            rows[sel], occ[sel], pmask[sel] = r[local], o[local], p[local]
            if focal_only:
                focal[sel] = f[local].astype(bool)
        x, y = rows[:, :-1], rows[:, 1:].copy()
        if focal_only:
            y[~focal[:, 1:]] = 0  # the other side's moves are context only: PAD targets are ignored by the loss
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(occ[:, 1:].view(np.int64)), torch.from_numpy(pmask[:, 1:]))

    def batches(self, batch_size, seed=0, rank=0, world=1, start=0, focal_only=False):
        """Endless: passes over this rank's records without replacement, each pass in a new random order.
        `start` = records already consumed by all ranks together (to resume mid-run)."""
        mine = np.arange(rank, self.n_records, world)
        epoch, pos = divmod(start // world, len(mine))
        while True:
            order = np.random.default_rng([seed, epoch]).permutation(mine)
            while pos < len(order):
                yield self.get(order[pos:pos + batch_size], focal_only)
                pos += batch_size
            epoch, pos = epoch + 1, 0


# ------------------------------------------------------------------ game sides

class SidesWriter:
    def __init__(self, root):
        os.makedirs(root, exist_ok=True)
        self.root = Path(root)
        self.f = [open(self.root / e, "wb", buffering=16 << 20) for e in ("rows.i16", "occ.u64", "pmask.u8")]
        self.offset, self.index = 0, []

    def add_game(self, rows, occ, pmask):
        """Write one game's rows; returns (row_off, n_rows) for the index records of its described sides."""
        for f, a, dt in zip(self.f, (rows, occ, pmask), ("<i2", "<u8", "u1")):
            f.write(np.asarray(a, dtype=dt).tobytes())
        off, self.offset = self.offset, self.offset + len(rows)
        return off, len(rows)

    def close(self, index_table):
        import pyarrow.parquet as pq
        for f in self.f:
            f.close()
        pq.write_table(index_table, self.root / "index.parquet", compression="zstd")


class Sides:
    """Game sides with their descriptions. Rows stay memory-mapped (load=True copies them into RAM, which is much
    faster on network filesystems), index columns live in numpy arrays."""

    def __init__(self, root, columns=("labels",), load=False):
        import pyarrow as pa
        import pyarrow.parquet as pq
        root = Path(root)
        t = pq.read_table(root / "index.parquet", columns=["split", "side", "row_off", "n_rows", *columns])
        self.split = {s: np.flatnonzero(np.asarray(t["split"]) == s) for s in ("train", "validation", "test")}
        self.white = np.asarray(t["side"]) == "white"
        self.row_off, self.n_rows = t["row_off"].to_numpy(), t["n_rows"].to_numpy()
        # text columns stay in Arrow (27M x 5 descriptions as Python strings would need tens of GB); `large` types
        # because the descriptions are more than 2 GB of text
        large = {"labels": pa.list_(pa.large_string()), "player": pa.large_string()}
        self.cols = {c: (t[c].cast(large[c]) if c in large else t[c]).combine_chunks() for c in columns}
        read = np.fromfile if load else (lambda p, dtype: np.memmap(p, dtype=dtype, mode="r"))
        self.rows = read(root / "rows.i16", dtype="<i2").reshape(-1, 5)
        self.occ = read(root / "occ.u64", dtype="<u8")
        self.pmask = read(root / "pmask.u8", dtype="u1")

    def get(self, column, k):
        return self.cols[column][int(k)].as_py()

    def keep(self, mask):
        """Restrict every split to records where mask is True."""
        self.split = {s: ids[mask[ids]] for s, ids in self.split.items()}

    def batch(self, ids):
        """Right-padded games: rows (G, T, 5), focal (G, T-1) = target t is a move of the described side,
        occ/pmask aligned with the targets."""
        T = int(self.n_rows[ids].max())
        rows = np.zeros((len(ids), T, 5), np.int64)
        occ = np.zeros((len(ids), T), np.uint64)
        pmask = np.zeros((len(ids), T), np.int64)
        for i, k in enumerate(ids):
            o, n = int(self.row_off[k]), int(self.n_rows[k])
            rows[i, :n], occ[i, :n], pmask[i, :n] = self.rows[o:o + n], self.occ[o:o + n], self.pmask[o:o + n]
        ply = np.arange(T - 1)[None, :]  # target t+1 is ply t: white plays even t, black odd t
        focal = (rows[:, 1:, 0] == KIND_MOVE) & (ply % 2 == np.where(self.white[ids], 0, 1)[:, None])
        return {"ids": np.asarray(ids), "rows": torch.from_numpy(rows), "focal": torch.from_numpy(focal),
                "occ": torch.from_numpy(occ[:, 1:].view(np.int64)), "pmask": torch.from_numpy(pmask[:, 1:])}


def length_batches(lengths, ids, max_games, token_budget, rng, pool=16384):
    """One pass over ids, in batches of games of similar length: draw a pool, sort it by length, cut it into
    batches with games x longest game <= token_budget, shuffle the batch order."""
    order = rng.permutation(ids)
    for p0 in range(0, len(order), pool):
        part = order[p0:p0 + pool]
        part = part[np.argsort(lengths[part], kind="stable")]
        out, start = [], 0
        for i in range(len(part)):
            if i - start + 1 > max_games or (i - start + 1) * int(lengths[part[i]]) > token_budget:
                out.append(part[start:i]); start = i
        out.append(part[start:])
        rng.shuffle(out)
        yield from out


def prefetch(fn, items, depth=4):
    """fn(item) for each item, computed ahead in a background thread (batch assembly overlaps the GPU step)."""
    import queue
    import threading
    q, done = queue.Queue(maxsize=depth), object()

    def work():
        for it in items:
            q.put(fn(it))
        q.put(done)
    threading.Thread(target=work, daemon=True).start()
    while (b := q.get()) is not done:
        yield b
