"""Build the SFT corpus from the labeled post-training dataset
(HF `LegumMagister/posttrain-labeled`, ~7.5M labeled game-sides).

The pretraining tier-3/3p data is no longer on disk, so this rebuilds
everything SFT needs straight from the labeled records:

  rows.i16       (N_rows, 5) int16 factored rows (kind,src,dst,piece,promo),
                 BOS + one row per ply + EOS per game, concatenated
  legal.bin      (N_rows,) LEGAL_DTYPE (own_occ, piece_mask), index-aligned
                 with rows.i16 -- the exact bits pretraining's
                 --legal-penalty-weight used, recomputed by
                 build_tier3.replay_game
  index.parquet  one row per labeled game-side: month, row_idx, side, split,
                 row_off, n_rows, labels (list<string>), plus every other
                 scalar metadata column of the source passed through
                 untouched (for eval slicing only; SFT never reads them)

Split: the record's own `split` column if it has one, otherwise the same
deterministic game-level 90/5/5 hash style_autoresearch.stable_split uses,
so both sides of a game always land in the same split.

Input: a local directory (or HF snapshot) of .parquet / .jsonl(.zst) files.
Also stores the labelled side's move/capture/check counts, which
src/meta_templates.py turns into synthesized metadata labels at train time.
Moves are read from the first present of: moves_u16 (move_codec ints),
moves_uci, moves_san, moves (SAN or UCI strings). Labels from `labels`
(dict or list) or from band columns (terse_1, terse_2, short, medium, long).

Usage:
    python3 src/build_sft_corpus.py /path/to/posttrain-labeled --out data/sft --workers 64
    python3 src/build_sft_corpus.py LegumMagister/posttrain-labeled --out data/sft   # needs HF login
"""
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tier3 import LEGAL_DTYPE, replay_game  # noqa: E402
import move_codec  # noqa: E402

BANDS = ("terse_1", "terse_2", "short", "medium", "long")
MOVE_FIELDS = ("moves_u16", "moves_uci", "moves_san", "moves")
SKIP_META = {"labels", "system", "user", "prompt", "raw", "response", *MOVE_FIELDS, *BANDS,
             *(f"style_{b}" for b in BANDS)}


ALLOW_UNLABELED = False  # --allow-unlabeled: keep records without labels (attached later by key)


def stable_split(month: str, row_idx: int, seed: int) -> str:
    raw = f"{seed}:{month}:{row_idx}".encode()
    value = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % 100
    return "train" if value < 90 else ("val" if value < 95 else "test")


def to_codes(moves) -> tuple[np.ndarray, dict]:
    """Move codes plus per-side capture/check counts (for the metadata label)."""
    import chess
    stats = {"white": [0, 0, 0], "black": [0, 0, 0]}  # moves, captures, checks
    if len(moves) == 0:
        return np.zeros(0, dtype=np.uint16), stats
    if isinstance(moves[0], (int, np.integer)):
        moves = [move_codec.decode(int(c)).uci() for c in moves]
    board, codes = chess.Board(), []
    for text in moves:
        try:
            move = chess.Move.from_uci(text)
            if move not in board.legal_moves:
                raise ValueError
        except ValueError:
            move = board.parse_san(text)
        st = stats["white" if board.turn else "black"]
        st[0] += 1
        st[1] += board.is_capture(move)
        st[2] += board.gives_check(move)
        codes.append(move_codec.encode(move))
        board.push(move)
    return np.asarray(codes, dtype=np.uint16), stats


def extract(record: dict) -> tuple[str, int, str, list[str], np.ndarray] | None:
    month, row_idx, side = record.get("month"), record.get("row_idx"), record.get("side")
    if (month is None or row_idx is None or side is None) and record.get("key"):
        month, row_idx, side = str(record["key"]).split(":")
    labels = record.get("labels")
    if isinstance(labels, dict):
        labels = [labels.get(b) for b in BANDS]
    elif labels is None:
        labels = [record.get(b, record.get(f"style_{b}")) for b in BANDS]
    labels = [str(x).strip() for x in (labels or []) if isinstance(x, str) and x.strip()]
    moves = next((record[f] for f in MOVE_FIELDS if record.get(f) is not None), None)
    if (len(labels) != len(BANDS) and not ALLOW_UNLABELED) or moves is None or side not in ("white", "black"):
        return None
    if isinstance(moves, str):
        moves = moves.split()
    codes, stats = to_codes(list(moves))
    n, caps, checks = stats[side]
    if n == 0:
        return None
    # labels = the 5 LLM bands (BANDS order); metadata labels are rendered at
    # train time from these counts + the passthrough columns (meta_templates.py)
    return str(month), int(row_idx), side, labels, codes, {"side_moves": n, "captures": caps, "checks": checks}


def iter_file(path: Path, row_group: int = -1):
    name = path.name
    if name.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        groups = range(pf.num_row_groups) if row_group < 0 else [row_group]
        for g in groups:
            yield from pf.read_row_group(g).to_pylist()
    elif name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
        if name.endswith(".zst"):
            import zstandard
            handle = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(path.open("rb")))
        else:
            handle = path.open()
        with handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def process_file(job: tuple[str, int, str, int, int]) -> dict:
    path, row_group, tmp_dir, max_rows, split_seed = job
    stem = hashlib.md5(f"{path}:{row_group}".encode()).hexdigest()[:12]
    rows_path, legal_path = os.path.join(tmp_dir, f"{stem}.rows"), os.path.join(tmp_dir, f"{stem}.legal")
    meta_rows, cum, skipped = [], 0, 0
    with open(rows_path, "wb", buffering=16 << 20) as fr, open(legal_path, "wb", buffering=16 << 20) as fl:
        for record in iter_file(Path(path), row_group):
            got = extract(record)
            if got is None:
                skipped += 1
                continue
            month, row_idx, side, labels, codes, counts = got
            rows, legal = replay_game(codes)
            if len(rows) > max_rows:
                skipped += 1
                continue
            fr.write(np.asarray(rows, dtype="<i2").tobytes())
            fl.write(np.array(legal, dtype=LEGAL_DTYPE).tobytes())
            meta = {k: v for k, v in record.items()
                    if k not in SKIP_META and (v is None or isinstance(v, (str, int, float, bool)))}
            meta.update(counts)
            meta.update(month=month, row_idx=row_idx, side=side, labels=labels,
                        split=str(record.get("split") or stable_split(month, row_idx, split_seed)),
                        row_off=cum, n_rows=len(rows))
            meta_rows.append(meta)
            cum += len(rows)
    return {"path": path, "rg": row_group, "rows": rows_path, "legal": legal_path, "meta": meta_rows,
            "n_rows": cum, "skipped": skipped}


def resolve_input(src: str) -> list[Path]:
    root = Path(src)
    if not root.exists():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(src, repo_type="dataset"))
    files = sorted(p for p in root.rglob("*")
                   if p.name.endswith((".parquet", ".jsonl", ".jsonl.zst")))
    if not files:
        raise SystemExit(f"no .parquet/.jsonl files under {root}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="local dir or HF dataset repo id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--max-rows", type=int, default=512, help="drop games longer than this many rows (BOS+plies+EOS)")
    ap.add_argument("--split-seed", type=int, default=20260802)
    ap.add_argument("--allow-unlabeled", action="store_true",
                    help="keep records without the 5 LLM labels (labels=[]); attach them later by key")
    args = ap.parse_args()
    global ALLOW_UNLABELED
    ALLOW_UNLABELED = args.allow_unlabeled  # set before the fork so workers inherit it

    files = resolve_input(args.input)
    out = Path(args.out); tmp = out / "_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    print(f"[build] {len(files)} input files -> {out}", flush=True)
    jobs = [(str(f), g, str(tmp), args.max_rows, args.split_seed) for f in files
            for g in (range(pq.ParquetFile(f).num_row_groups) if f.name.endswith(".parquet") else [-1])]
    metas, offset, skipped = [], 0, 0
    with open(out / "rows.i16", "wb", buffering=64 << 20) as fr, \
            open(out / "legal.bin", "wb", buffering=64 << 20) as fl, \
            mp.Pool(args.workers) as pool:
        # imap (ordered) keeps the concatenation deterministic across reruns.
        for part in pool.imap(process_file, jobs):
            for name, handle in (("rows", fr), ("legal", fl)):
                with open(part[name], "rb") as src:
                    while chunk := src.read(64 << 20):
                        handle.write(chunk)
                os.remove(part[name])
            for m in part["meta"]:
                m["row_off"] += offset
            metas.extend(part["meta"]); offset += part["n_rows"]; skipped += part["skipped"]
            print(f"[build] {Path(part['path']).name}#{part['rg']}: {len(part['meta'])} kept, "
                  f"{part['skipped']} skipped (total {len(metas)})", flush=True)
    pq.write_table(pa.Table.from_pylist(metas), out / "index.parquet")
    tmp.rmdir()
    splits = {s: sum(m["split"] == s for m in metas) for s in ("train", "val", "test")}
    n_labels = np.bincount([len(m["labels"]) for m in metas])
    summary = {"records": len(metas), "rows": offset, "skipped": skipped, "splits": splits,
               "labels_per_record_hist": {i: int(c) for i, c in enumerate(n_labels) if c},
               "metadata_columns": sorted(set().union(*(m.keys() for m in metas[:1000])))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
