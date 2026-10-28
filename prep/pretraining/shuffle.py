#!/usr/bin/env python3
"""Optional: globally shuffle packed training records (tier 3p -> 3p-shuffled).

``pack.py`` output is in chronological order (2013 -> 2026). Reading it
sequentially is an unintended old-to-new curriculum; reading it randomly is
slow on network filesystems. This external bucket shuffle gives both: it scans
the source sequentially, hashes every record into one of ``--buckets`` files,
then permutes each bucket in memory. Every output shard is a uniform random
sample of the whole corpus, so a trainer can read shards sequentially.

Records and their occ/pmask sidecars always move together. Validation is not
shuffled; it is copied (or symlinked with --link-val) unchanged.

    python shuffle.py data_3p -o data_3p_shuffled --buckets 128 --seed 42
    python shuffle.py data_3p -o data_3p_shuffled --stage /fast/tmp/stage   # staging on a faster disk

Disk: needs one extra copy of the training split for staging (deleted at the
end unless --keep-stage).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def splitmix64(x):
    x = x.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def sidecars(path: Path):
    return path.with_name(path.stem + "_occ.bin"), path.with_name(path.stem + "_pmask.bin")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="packed directory from pack.py (has train/ and val/)")
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--buckets", type=int, default=128, help="number of output shards")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", help="staging directory (default: <outdir>/.stage)")
    ap.add_argument("--chunk-records", type=int, default=16384)
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--link-val", action="store_true", help="symlink val/ instead of copying it")
    a = ap.parse_args()

    src = Path(a.indir)
    info = json.loads((src / "train" / "_SUCCESS").read_text())
    cap, legal = info["seq_len"] + 1, info["has_legal"]
    rec_bytes = [cap * 5 * 2] + ([cap * 8, cap] if legal else [])
    dtypes = [("<i2", (cap, 5))] + ([("<u8", (cap,)), ("u1", (cap,))] if legal else [])
    out = Path(a.outdir)
    stage = Path(a.stage) if a.stage else out / ".stage"
    (out / "train").mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)

    shards = sorted((src / "train").glob("packed_*.i16"))
    files = [[p, *sidecars(p)][:len(rec_bytes)] for p in shards]
    names = lambda b: [stage / f"bucket_{b:05d}{s}" for s in (".i16", "_occ.bin", "_pmask.bin")][:len(rec_bytes)]
    handles = [[open(p, "wb", buffering=0) for p in names(b)] for b in range(a.buckets)]
    seen = 0
    for i, trio in enumerate(files):
        n = trio[0].stat().st_size // rec_bytes[0]
        maps = [np.memmap(f, dtype=dt, mode="r", shape=(n, *shape)) for f, (dt, shape) in zip(trio, dtypes)]
        for lo in range(0, n, a.chunk_records):
            hi = min(n, lo + a.chunk_records)
            ids = np.arange(seen + lo, seen + hi, dtype=np.uint64)
            bucket = (splitmix64(ids ^ np.uint64(a.seed)) % np.uint64(a.buckets)).astype(np.int64)
            order = np.argsort(bucket, kind="stable")
            cuts = np.r_[0, np.flatnonzero(np.diff(bucket[order])) + 1, hi - lo]
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                take = order[c0:c1]
                for h, m in zip(handles[int(bucket[order[c0]])], maps):
                    h.write(m[lo:hi][take].tobytes())
        seen += n
        print(f"[scatter {i + 1}/{len(files)}] {trio[0].name}: {n:,} records", flush=True)
    for hs in handles:
        for h in hs:
            h.close()

    total = 0
    for b in range(a.buckets):
        parts = names(b)
        n = parts[0].stat().st_size // rec_bytes[0]
        if n == 0:  # possible when only a few months are prepared; leave no shard for an empty bucket
            print(f"[materialize {b + 1}/{a.buckets}] empty, skipped", flush=True)
            continue
        perm = np.random.default_rng(a.seed + b).permutation(n)
        for p, (dt, shape), suffix in zip(parts, dtypes, (".i16", "_occ.bin", "_pmask.bin")):
            m = np.memmap(p, dtype=dt, mode="r", shape=(n, *shape))
            with open(out / "train" / f"packed_{b:05d}{suffix}", "wb", buffering=8 << 20) as f:
                for lo in range(0, n, a.chunk_records):
                    f.write(m[perm[lo:lo + a.chunk_records]].tobytes())
            del m
        total += n
        print(f"[materialize {b + 1}/{a.buckets}] {n:,} records", flush=True)
    if total != info["n_rows"]:
        raise SystemExit(f"record count mismatch: {total:,} vs {info['n_rows']:,}")
    (out / "train" / "_SUCCESS").write_text(json.dumps({**info, "shuffled": True, "seed": a.seed,
                                                        "buckets": a.buckets}) + "\n")
    if (out / "val").exists() or (out / "val").is_symlink():
        pass
    elif a.link_val:
        os.symlink((src / "val").resolve(), out / "val")
    else:
        shutil.copytree(src / "val", out / "val")
    if not a.keep_stage:
        shutil.rmtree(stage)
    print(f"done: {total:,} records in {a.buckets} shuffled shards")


if __name__ == "__main__":
    main()
