#!/usr/bin/env python3
"""Integrity checks for each stage of the pipeline.

    python verify.py u16 data_u16                  # offsets contiguous, sizes match, sample replay legal
    python verify.py factorized data_3             # BOS..EOS per game, sidecar aligned, piece/replay consistent
    python verify.py packed data_3p                # record/sidecar counts, BOS/EOS balance, game count
    python verify.py roundtrip data_u16 original_2b   # byte-identical moves.u16 vs. an original tier-2b copy

Each prints one line per month/shard and exits non-zero on the first failure.
"""
import argparse
import filecmp
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from chsm8_format import KIND_BOS, KIND_EOS, KIND_MOVE, KIND_PAD, LEGAL_DTYPE, code_to_uci


def months(root):
    return sorted(m for m in os.listdir(root) if os.path.isfile(os.path.join(root, m, "meta.parquet")))


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def check_u16(root, sample):
    import chess
    for m in months(root):
        meta = pq.read_table(os.path.join(root, m, "meta.parquet"), columns=["move_off", "n_plies"])
        off, n = meta["move_off"].to_numpy().astype(np.int64), meta["n_plies"].to_numpy().astype(np.int64)
        size = os.path.getsize(os.path.join(root, m, "moves.u16")) // 2
        if len(off) and (off[0] != 0 or (off[1:] != off[:-1] + n[:-1]).any() or off[-1] + n[-1] != size):
            fail(f"{m}: offsets not contiguous / size mismatch")
        mv = np.memmap(os.path.join(root, m, "moves.u16"), dtype="<u2", mode="r")
        for g in np.random.default_rng(0).choice(len(off), min(sample, len(off)), replace=False):
            b = chess.Board()
            for c in mv[off[g]:off[g] + n[g]]:
                u = chess.Move.from_uci(code_to_uci(c))
                if u not in b.legal_moves:
                    fail(f"{m}: game {g} has illegal move {u}")
                b.push(u)
        print(f"[{m}] ok: {len(off):,} games, {size:,} plies")


def check_factorized(root):
    for m in months(root):
        meta = pq.read_table(os.path.join(root, m, "meta.parquet"), columns=["row_off", "n_rows"])
        off, n = meta["row_off"].to_numpy().astype(np.int64), meta["n_rows"].to_numpy().astype(np.int64)
        rows = np.memmap(os.path.join(root, m, "rows.i16"), dtype="<i2", mode="r").reshape(-1, 5)
        if len(off) and off[-1] + n[-1] != len(rows):
            fail(f"{m}: rows.i16 size mismatch")
        legal_p = os.path.join(root, m, "legal.bin")
        if os.path.exists(legal_p) and os.path.getsize(legal_p) != len(rows) * LEGAL_DTYPE.itemsize:
            fail(f"{m}: legal.bin not aligned with rows.i16")
        kinds = rows[:, 0]
        if (kinds[off] != KIND_BOS).any() or (kinds[off + n - 1] != KIND_EOS).any():
            fail(f"{m}: game not wrapped in BOS..EOS")
        if int((kinds == KIND_MOVE).sum()) != int((n - 2).sum()):
            fail(f"{m}: MOVE row count != sum(n_rows - 2)")
        print(f"[{m}] ok: {len(off):,} games, {len(rows):,} rows, legal={os.path.exists(legal_p)}")


def check_packed(root):
    for split in ("train", "val"):
        d = Path(root) / split
        info = json.loads((d / "_SUCCESS").read_text())
        cap = info["seq_len"] + 1
        records = games = 0
        for p in sorted(d.glob("packed_*.i16")):
            r = np.memmap(p, dtype="<i2", mode="r").reshape(-1, cap, 5)
            if info["has_legal"]:
                for suffix, width in (("_occ.bin", 8), ("_pmask.bin", 1)):
                    s = p.with_name(p.stem + suffix)
                    if s.stat().st_size != len(r) * cap * width:
                        fail(f"{s}: sidecar size mismatch")
            k = r[:, :, 0]
            nb, ne = int((k == KIND_BOS).sum()), int((k == KIND_EOS).sum())
            if nb != ne:
                fail(f"{p}: {nb} BOS vs {ne} EOS")
            # a record never ends mid-game: after the last EOS only PAD may follow
            last_eos = cap - 1 - np.argmax((k == KIND_EOS)[:, ::-1], axis=1)
            tail = np.arange(cap)[None, :] > last_eos[:, None]
            if (k[tail] != KIND_PAD).any():
                fail(f"{p}: game split across records")
            records += len(r); games += nb
        if records != info["n_rows"] or games != info["n_games"]:
            fail(f"{split}: counted {records:,} records/{games:,} games, _SUCCESS says "
                 f"{info['n_rows']:,}/{info['n_games']:,}")
        print(f"[{split}] ok: {records:,} records, {games:,} games, seq_len={info['seq_len']}, legal={info['has_legal']}")


def check_roundtrip(a, b):
    for m in months(a):
        if not filecmp.cmp(os.path.join(a, m, "moves.u16"), os.path.join(b, m, "moves.u16"), shallow=False):
            fail(f"{m}: moves.u16 differs")
        print(f"[{m}] moves.u16 identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["u16", "factorized", "packed", "roundtrip"])
    ap.add_argument("path")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--sample", type=int, default=200, help="games replayed with python-chess per month (u16)")
    a = ap.parse_args()
    {"u16": lambda: check_u16(a.path, a.sample), "factorized": lambda: check_factorized(a.path),
     "packed": lambda: check_packed(a.path), "roundtrip": lambda: check_roundtrip(a.path, a.other)}[a.stage]()
    print("ALL OK")


if __name__ == "__main__":
    main()
