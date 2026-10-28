"""Tier 2a -> tier 2b: apply the approved filters. Cheap, local, re-runnable.

Everything here is a decision we might want to revisit, which is exactly why it
lives in this pass and not in the 2.51 TB scan. Re-running with different
thresholds costs minutes over our own output instead of another archive scan.

Approved filters:
  - Termination not in {Time forfeit, Abandoned, Rules infraction}
  - min(white_elo, black_elo) >= 1800
  - truncate each game at trunc_ply (the first ply with < 15s on the mover's
    clock, as recorded by build_2a)

Output mirrors the 2a layout, so downstream code reads either identically.

  python src/build_2b.py data/2a -o data/2b
  python src/build_2b.py data/2a -o data/2b-elo1500 --min-elo 1500 --keep-time-forfeit
"""

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DROP_TERMINATIONS = {"Time forfeit", "Abandoned", "Rules infraction"}
MOVE_BUF_BYTES = 8 << 20


def filter_month(src: str, dst: str, min_elo: int, drop_terms: set, truncate: bool) -> Dict[str, int]:
    os.makedirs(dst, exist_ok=True)
    t0 = time.time()
    table = pq.read_table(os.path.join(src, "meta.parquet"))
    moves = np.memmap(os.path.join(src, "moves.u16"), dtype="<u2", mode="r")

    term = table.column("termination").to_pylist()
    we = table.column("white_elo").to_numpy()
    be = table.column("black_elo").to_numpy()
    off = table.column("move_off").to_numpy()
    npl = table.column("n_plies").to_numpy()
    trp = table.column("trunc_ply").to_numpy()

    keep = (np.minimum(we, be) >= min_elo)
    if drop_terms:
        keep &= np.fromiter((t not in drop_terms for t in term), dtype=bool, count=len(term))
    # A game truncated to zero plies carries no signal.
    new_len = np.where(truncate, np.minimum(trp, npl), npl).astype(np.uint16)
    keep &= new_len > 0
    idx = np.flatnonzero(keep)

    out_f = open(os.path.join(dst, "moves.u16"), "wb", buffering=MOVE_BUF_BYTES)
    new_off = np.empty(len(idx), dtype=np.uint64)
    cursor = 0
    for j, i in enumerate(idx):
        k = int(new_len[i])
        out_f.write(moves[off[i]:off[i] + k].tobytes())
        new_off[j] = cursor
        cursor += k
    out_f.close()

    cols = {name: table.column(name).take(pa.array(idx)) for name in table.schema.names}
    cols["move_off"] = pa.array(new_off, type=pa.uint64())
    cols["n_plies"] = pa.array(new_len[idx], type=pa.uint16())
    cols["trunc_ply"] = cols["n_plies"]
    pq.write_table(pa.table(cols, schema=table.schema),
                   os.path.join(dst, "meta.parquet"), compression="zstd")

    return {"in_games": len(term), "out_games": len(idx),
            "in_plies": int(npl.sum()), "out_plies": int(cursor),
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", help="tier 2a directory (contains <month>/ subdirs)")
    ap.add_argument("-o", "--outdir", default="data/2b")
    ap.add_argument("--min-elo", type=int, default=1800,
                    help="floor on min(white_elo, black_elo); default 1800")
    ap.add_argument("--keep-time-forfeit", action="store_true",
                    help="keep games that ended on the clock (see DATASET.md open question)")
    ap.add_argument("--no-truncate", action="store_true",
                    help="ignore trunc_ply and keep full games")
    ap.add_argument("--months", nargs="*", help="subset of months; default all")
    a = ap.parse_args()

    drops = set(DROP_TERMINATIONS)
    if a.keep_time_forfeit:
        drops.discard("Time forfeit")

    months = sorted(a.months or [m for m in os.listdir(a.indir)
                                 if os.path.isdir(os.path.join(a.indir, m))])
    tot = {"in_games": 0, "out_games": 0, "in_plies": 0, "out_plies": 0}
    for m in months:
        src = os.path.join(a.indir, m)
        if not os.path.exists(os.path.join(src, "meta.parquet")):
            print(f"[{m}] no meta.parquet, skipping")
            continue
        c = filter_month(src, os.path.join(a.outdir, m), a.min_elo, drops, not a.no_truncate)
        for k in tot:
            tot[k] += c[k]
        print(f"[{m}] {c['in_games']:,} -> {c['out_games']:,} games "
              f"({100*c['out_games']/max(c['in_games'],1):.2f}%), "
              f"{c['in_plies']:,} -> {c['out_plies']:,} plies, {c['seconds']}s", flush=True)
        # Mirrors build_2a.py's convention: a streaming wrapper needs a marker
        # that only appears after the month's writes are fully flushed, so it
        # can tell "done" from "crashed mid-write" on restart.
        with open(os.path.join(a.outdir, m, "_SUCCESS"), "w") as f:
            f.write(repr(c) + "\n")

    print(f"\nTOTAL {tot['in_games']:,} -> {tot['out_games']:,} games "
          f"({100*tot['out_games']/max(tot['in_games'],1):.2f}%), "
          f"{tot['in_plies']:,} -> {tot['out_plies']:,} plies "
          f"({100*tot['out_plies']/max(tot['in_plies'],1):.2f}%), "
          f"{tot['out_plies']*2/1e9:.1f} GB of uint16")


if __name__ == "__main__":
    main()
