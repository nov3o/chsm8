"""chsm8-pre-training (Hugging Face) -> packed training records for scripts/train/pretrain.py.

    python scripts/data/prepare_pretraining.py --out data/pt --workers 64                    # all 162 months, ~1.5 TB
    python scripts/data/prepare_pretraining.py --out data/nano --months 2026-06 --max-games 20000

Each game (moves in UCI) becomes rows with BOS/EOS plus legality bits (chsm8/tokenizer.py). Games are packed into
records of seq_len+1 rows, whole games only; the few games longer than one record are skipped. In every month the
last --val-fraction of games is validation. The trainer reads records in a random order, so the output can stay in
chronological order. One worker per month; the largest months take about half an hour.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8.data import Packer  # noqa: E402
from chsm8.tokenizer import rows, uci_tokens  # noqa: E402

REPO = "LegumMagister/chsm8-pre-training"


def month_job(job):
    files, out, month, seq_len, val_fraction, max_games = job
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if max_games:
        n = min(n, max_games)
    n_val = int(n * val_fraction)
    packers = {s: Packer(f"{out}/{s}/{month}", seq_len) for s in ("train", "val")}
    i, skipped = 0, 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(columns=["moves"], batch_size=65536):
            for moves in batch.column(0).to_pylist():
                if i >= n:
                    break
                r, occ, pmask = rows(uci_tokens(moves))
                skipped += not packers["train" if i < n - n_val else "val"].add(r, occ, pmask)
                i += 1
    stats = {s: p.close() for s, p in packers.items()}
    return month, stats, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--months", nargs="*", help="e.g. 2026-05 2026-06 (default: all)")
    ap.add_argument("--max-games", type=int, default=0, help="per month, 0 = all (for small test sets)")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--val-fraction", type=float, default=0.01)
    ap.add_argument("--source", default="", help="local copy of the dataset (default: download from the Hub)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    src = a.source
    if not src:
        from huggingface_hub import snapshot_download
        pattern = [f"data/{m}/*" for m in a.months] if a.months else ["data/*"]
        src = snapshot_download(REPO, repo_type="dataset", allow_patterns=pattern)
    months = sorted(a.months or os.listdir(Path(src, "data")))
    jobs = [(sorted(map(str, Path(src, "data", m).glob("*.parquet"))), a.out, m, a.seq_len, a.val_fraction,
             a.max_games) for m in months]
    jobs.sort(key=lambda j: -sum(os.path.getsize(f) for f in j[0]))  # largest first
    total = {"train": [0, 0], "val": [0, 0]}
    with Pool(min(a.workers, len(jobs))) as pool:
        for month, stats, skipped in pool.imap_unordered(month_job, jobs):
            for s, (records, games) in stats.items():
                total[s][0] += records; total[s][1] += games
            print(f"{month}: {stats['train'][1]:,} train / {stats['val'][1]:,} val games, {skipped} too long",
                  flush=True)
    info = {"seq_len": a.seq_len, **{f"{s}_records": r for s, (r, g) in total.items()},
            **{f"{s}_games": g for s, (r, g) in total.items()}}
    Path(a.out, "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
