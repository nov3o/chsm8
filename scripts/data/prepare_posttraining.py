"""chsm8-post-training + chsm8-pre-training (Hugging Face) -> the SFT/DPO corpus and the games-only corpus.

    python scripts/data/prepare_posttraining.py --sides data/sides --games data/games --workers 64

chsm8-post-training holds the descriptions (30.1M described game sides, five bands each, with train / validation /
test splits); the moves come from chsm8-pre-training, joined on (month, row_idx). Two outputs:

    --sides   one record per described side, for scripts/train/sft.py and dpo.py (format: chsm8/data.py, Sides)
    --games   every game once, packed, for the games-only continuation (chsm8-player-base); train = the games of
              the train split, val = those of the validation split

Games longer than 512 rows are left out of both (0.02% of sides). --max-sides keeps the first N sides of each split
(for a small test corpus).
"""
import argparse
import os
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from chsm8.data import Packer, SidesWriter  # noqa: E402
from chsm8.tokenizer import rows, uci_tokens  # noqa: E402

BANDS = ["terse_1", "terse_2", "short", "medium", "long"]
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def month_job(job):
    month, sides_path, pre_files, tmp, games_out, max_rows = job
    sides = pq.read_table(sides_path).to_pydict()
    need = set(sides["row_idx"])
    moves = {}
    for f in pre_files:
        t = pq.read_table(f, columns=["row_idx", "moves"])
        t = t.filter(pc.is_in(t["row_idx"], pa.array(sorted(need), pa.int32())))
        moves.update(zip(t["row_idx"].to_pylist(), t["moves"].to_pylist()))
    w = SidesWriter(f"{tmp}/{month}")
    packers = {s: Packer(f"{games_out}/{s}/{month}", 512) for s in ("train", "val")} if games_out else {}
    where, index = {}, []
    for i in range(len(sides["row_idx"])):
        g = sides["row_idx"][i]
        if g not in where:
            r, occ, pmask = rows(uci_tokens(moves[g]))
            where[g] = w.add_game(r, occ, pmask) if len(r) <= max_rows else None
            split = {"train": "train", "validation": "val"}.get(sides["split"][i])
            if where[g] and split in packers:
                packers[split].add(r, occ, pmask)
        if where[g]:
            index.append((i, *where[g]))
    for p in packers.values():
        p.close()
    keep = [i for i, _, _ in index]
    table = pa.table({**{k: pa.array([sides[k][i] for i in keep]) for k in sides},
                      "row_off": pa.array([o for _, o, _ in index], pa.int64()),
                      "n_rows": pa.array([n for _, _, n in index], pa.int32())})
    w.close(table)
    return month, len(keep), len(sides["row_idx"]) - len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", required=True, help="output: SFT/DPO corpus")
    ap.add_argument("--games", default="", help="output: packed games-only corpus (optional)")
    ap.add_argument("--post", default="", help="local copy of chsm8-post-training (default: download)")
    ap.add_argument("--pre", default="", help="local copy of chsm8-pre-training (default: download needed months)")
    ap.add_argument("--max-sides", type=int, default=0, help="per split, 0 = all")
    ap.add_argument("--max-rows", type=int, default=512)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    post = a.post or snapshot_download("LegumMagister/chsm8-post-training", repo_type="dataset", allow_patterns=["data/*"])
    parts = []
    for split in SPLITS:
        t = pq.read_table(Path(post, "data", split), columns=["month", "row_idx", "side", "labels", "labeler"])
        if a.max_sides:
            t = t.slice(0, a.max_sides)
        labels = t["labels"].combine_chunks()
        t = t.drop(["labels"]).append_column("labels", pa.array(
            [[d[b] for b in BANDS] for d in labels.to_pylist()], pa.list_(pa.string())))
        parts.append(t.append_column("split", pa.array([split] * len(t))))
    table = pa.concat_tables(parts)
    months = sorted(set(table["month"].to_pylist()))
    pre = a.pre or snapshot_download("LegumMagister/chsm8-pre-training", repo_type="dataset",
                                     allow_patterns=[f"data/{m}/*" for m in months])
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(a.sides)) or ".")
    jobs = []
    for m in months:
        path = f"{tmp}/{m}.sides.parquet"
        pq.write_table(table.filter(pc.equal(table["month"], m)), path)
        jobs.append((m, path, sorted(map(str, Path(pre, "data", m).glob("*.parquet"))), tmp, a.games, a.max_rows))
    del table
    with Pool(min(a.workers, len(jobs))) as pool:
        for m, kept, dropped in pool.imap_unordered(month_job, jobs):
            print(f"{m}: {kept:,} sides ({dropped} games too long)", flush=True)
    # concatenate the months into one corpus
    out = SidesWriter(a.sides)
    tables, offset = [], 0
    for m in months:
        d = Path(tmp, m)
        for f, name in zip(out.f, ("rows.i16", "occ.u64", "pmask.u8")):
            f.write((d / name).read_bytes())
        t = pq.read_table(d / "index.parquet")
        tables.append(t.set_column(t.column_names.index("row_off"), "row_off", pc.add(t["row_off"], offset)))
        offset += (d / "rows.i16").stat().st_size // 10
    out.close(pa.concat_tables(tables))
    if a.games:
        Path(a.games, "info.json").write_text('{"seq_len": 512}\n')
    os.system(f"rm -rf {tmp}")
    print(f"{a.sides}: {sum(len(t) for t in tables):,} sides")


if __name__ == "__main__":
    main()
