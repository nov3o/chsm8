"""Tier 1 -> tier 2a: scan one Lichess .pgn.zst into uint16 moves + metadata.

This is the expensive pass. It reads the compressed archive exactly once, so it
applies only the *cheap, uncontroversial* filters and records everything else as
metadata. Decisions still under discussion - the Elo floor, the time-forfeit
drop, the clock-truncation threshold - are deliberately NOT applied here. They
become a cheap local pass (build_2b.py) over our own output instead of another
2.51 TB rescan.

Applied here:
  - time control: base + 40*incr >= 180s  (blitz/rapid/classical, no bullet)
  - rated, non-bot, Standard variant, decisive result, parseable Elos
  - drop empty games and games whose SAN fails to replay

Recorded, NOT filtered:
  - termination (so "Time forfeit" stays revisitable)
  - both Elos (so the 1800 floor stays revisitable)
  - trunc_ply: first ply where the mover had < 15s left, or n_plies if never
    (so the truncation threshold stays revisitable - the clocks themselves are
    parsed here and then discarded, they never land on disk)

Output, per month, in <outdir>/<month>/:
  moves.u16      little-endian uint16, all games concatenated
  meta.parquet   one row per game, with move_off/n_plies indexing into moves.u16

Time control is decided by ARITHMETIC on the TimeControl header, never by the
Event label. Lichess relabelled the >=8min band when it introduced Rapid in
2017-04, so "Rated Classical game" means different things before and after that
date; the arithmetic does not. See DATASET.md.

  python src/build_2a.py data/archive/lichess_db_standard_rated_2019-03.pgn.zst -o data/2a
"""

import argparse
import io
import itertools
import os
import re
import subprocess
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import chess
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move_codec import encode  # noqa: E402

MIN_SECONDS = 180                 # blitz floor; excludes bullet and ultrabullet
CLK_TRUNC_SECONDS = 15            # recorded, not applied
BATCH_GAMES = 100_000             # parquet row-group size
MOVE_BUF_BYTES = 8 << 20

TAG_RE = re.compile(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]\s*$')
TC_RE = re.compile(r"^(\d+)\+(\d+)$")
RATED_RE = re.compile(r"\brated\b", re.IGNORECASE)
CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")
ANNOTATION_RE = re.compile(r"\{[^}]*\}")
NAG_RE = re.compile(r"\$\d+")
MOVE_NUM_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_RE = re.compile(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$")

VALID_RESULTS = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}
DROP_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}

REASONS = ["no_timecontrol", "too_fast", "casual", "bot", "unfinished",
           "variant", "no_elo", "empty", "illegal"]


def open_zst(path: str) -> Iterator[str]:
    """Stream decoded lines from .pgn.zst (or plain .pgn)."""
    if not path.endswith(".zst"):
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from f
        return
    try:
        import zstandard
        with open(path, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            yield from io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    except ImportError:
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        try:
            yield from io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        finally:
            proc.stdout.close()
            proc.wait()


def split_games(lines: Iterator[str]) -> Iterator[str]:
    """Yield one raw text block per game.

    This runs in the reader process and is deliberately as dumb as possible -
    a single startswith per line. Header regexes and SAN replay both happen in
    the workers. When header parsing lived here the reader capped the whole
    file at ~10k games/s no matter how many workers were attached.
    """
    buf: List[str] = []
    for line in lines:
        if line.startswith("[Event ") and buf:
            yield "".join(buf)
            buf = [line]
        else:
            buf.append(line)
    if buf:
        yield "".join(buf)


def parse_block(block: str) -> Tuple[Dict[str, str], str]:
    """Raw game text -> (headers, move text). Runs in a worker."""
    headers: Dict[str, str] = {}
    moves: List[str] = []
    for line in block.split("\n"):
        if line.startswith("["):
            m = TAG_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
                continue
        s = line.strip()
        if s:
            moves.append(s)
    return headers, " ".join(moves)


def time_control_ok(tc: str) -> Optional[Tuple[int, int]]:
    """(base, incr) if this is a >=180s game, else None.

    Arithmetic only - the Event label is never consulted, because its meaning
    changed in 2017-04.
    """
    m = TC_RE.match(tc)
    if not m:
        return None
    base, incr = int(m.group(1)), int(m.group(2))
    return (base, incr) if base + 40 * incr >= MIN_SECONDS else None


def clean_san(raw: str) -> List[str]:
    s = ANNOTATION_RE.sub("", raw)     # must run AFTER clocks are extracted
    s = RESULT_RE.sub("", s)
    s = NAG_RE.sub("", s)
    s = MOVE_NUM_RE.sub("", s)
    return [t.rstrip("!?") for t in s.split() if t.rstrip("!?") not in DROP_TOKENS and t.rstrip("!?")]


def truncation_ply(raw: str, n_plies: int) -> int:
    """First ply index whose mover had < CLK_TRUNC_SECONDS left; n_plies if never.

    Games before 2017-04 carry no [%clk] and get n_plies (kept whole).
    """
    for i, (h, m, s) in enumerate(CLK_RE.findall(raw)):
        if int(h) * 3600 + int(m) * 60 + int(s) < CLK_TRUNC_SECONDS:
            return min(i, n_plies)
    return n_plies


SCHEMA = pa.schema([
    ("move_off", pa.uint64()), ("n_plies", pa.uint16()), ("trunc_ply", pa.uint16()),
    ("white_elo", pa.uint16()), ("black_elo", pa.uint16()),
    ("result", pa.uint8()), ("termination", pa.dictionary(pa.int32(), pa.string())),
    ("tc_base", pa.uint16()), ("tc_incr", pa.uint16()),
    ("eco", pa.dictionary(pa.int32(), pa.string())),
    ("opening", pa.dictionary(pa.int32(), pa.string())),
    ("white", pa.dictionary(pa.int32(), pa.string())),
    ("black", pa.dictionary(pa.int32(), pa.string())),
    ("white_title", pa.dictionary(pa.int32(), pa.string())),
    ("black_title", pa.dictionary(pa.int32(), pa.string())),
])
# Everything except move_off, which only the writer can assign.
ROW_FIELDS = [f.name for f in SCHEMA][1:]


class MetaWriter:
    """Batched parquet writer. Strings are dictionary-encoded, so repeated
    usernames / ECO codes / opening names cost almost nothing."""

    def __init__(self, path: str):
        self.writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        self.cols: Dict[str, list] = {f.name: [] for f in SCHEMA}
        self.n = 0

    def add_row(self, move_off: int, row: tuple) -> None:
        self.cols["move_off"].append(move_off)
        for name, v in zip(ROW_FIELDS, row):
            self.cols[name].append(v)
        self.n += 1
        if self.n % BATCH_GAMES == 0:
            self.flush()

    def flush(self) -> None:
        if not self.cols["move_off"]:
            return
        self.writer.write_table(pa.table(
            {f.name: pa.array(self.cols[f.name], type=f.type) for f in SCHEMA},
            schema=SCHEMA))
        for v in self.cols.values():
            v.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def process_game(h: Dict[str, str], raw: str, counts: Dict[str, int]):
    """Filter + replay one game. Returns (move_bytes, meta_row) or None."""
    counts["seen"] += 1

    tc = time_control_ok(h.get("TimeControl", ""))
    if tc is None:
        counts["no_timecontrol" if not TC_RE.match(h.get("TimeControl", "")) else "too_fast"] += 1
        return None
    if not RATED_RE.search(h.get("Event", "")):
        counts["casual"] += 1
        return None
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        counts["bot"] += 1
        return None
    res = VALID_RESULTS.get(h.get("Result", ""))
    if res is None:
        counts["unfinished"] += 1
        return None
    if h.get("Variant", "Standard").lower() != "standard":
        counts["variant"] += 1
        return None
    try:
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        counts["no_elo"] += 1
        return None

    san = clean_san(raw)
    if not san:
        counts["empty"] += 1
        return None

    board = chess.Board()
    codes = []
    try:
        for tok in san:
            mv = board.parse_san(tok)
            codes.append(encode(mv))
            board.push(mv)
    except (ValueError, AssertionError, IndexError):
        counts["illegal"] += 1
        return None

    n = len(codes)
    counts["kept"] += 1
    counts["plies"] += n
    return (np.asarray(codes, dtype="<u2").tobytes(),
            (n, truncation_ply(raw, n), min(we, 65535), min(be, 65535), res,
             h.get("Termination", "Normal"), min(tc[0], 65535), min(tc[1], 65535),
             h.get("ECO", "?"), h.get("Opening", "?"), h.get("White", "?"),
             h.get("Black", "?"), h.get("WhiteTitle", ""), h.get("BlackTitle", "")))


def _worker(batch):
    """Pool entry point: a list of raw game blocks -> (results, counts)."""
    counts = {k: 0 for k in REASONS}
    counts.update(seen=0, kept=0, plies=0)
    out = []
    for block in batch:
        h, raw = parse_block(block)
        r = process_game(h, raw, counts)
        if r:
            out.append(r)
    return out, counts


def _batched(it, size):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build(path: str, outdir: str, workers: int = 1) -> Dict[str, int]:
    """Scan one month.

    SAN replay dominates the cost, so large months need a worker pool: the
    biggest file in the archive is 108M games, which is ~25h single-process.
    The reader stays in the main process (splitting text is cheap relative to
    replaying it) and hands batches to workers. Results are consumed in order,
    so move_off assignment stays deterministic.
    """
    month = os.path.basename(path).replace("lichess_db_standard_rated_", "").split(".")[0]
    dest = os.path.join(outdir, month)
    os.makedirs(dest, exist_ok=True)

    totals = {k: 0 for k in REASONS}
    totals.update(seen=0, kept=0, plies=0)
    t0 = time.time()

    moves_f = open(os.path.join(dest, "moves.u16"), "wb", buffering=MOVE_BUF_BYTES)
    meta = MetaWriter(os.path.join(dest, "meta.parquet"))
    off = 0

    def consume(results, counts):
        nonlocal off
        for k, v in counts.items():
            totals[k] += v
        for mv_bytes, row in results:
            moves_f.write(mv_bytes)
            meta.add_row(off, row)
            off += row[0]

    games = split_games(open_zst(path))
    if workers <= 1:
        for batch in _batched(games, BATCH_GAMES // 100):
            consume(*_worker(batch))
    else:
        import multiprocessing as mp
        # Bounded windows keep in-flight text off the heap; Pool.imap would
        # otherwise run arbitrarily far ahead of the writer.
        window = workers * 4
        with mp.Pool(workers) as pool:
            batches = _batched(games, BATCH_GAMES // 100)
            while True:
                chunk = list(itertools.islice(batches, window))
                if not chunk:
                    break
                for results, counts in pool.imap(_worker, chunk):
                    consume(results, counts)

    moves_f.close()
    meta.close()
    totals["seconds"] = round(time.time() - t0, 1)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".pgn.zst (or .pgn) for one month")
    ap.add_argument("-o", "--outdir", default="data/2a")
    ap.add_argument("-w", "--workers", type=int, default=1,
                    help="SAN-replay worker processes; large months need >1")
    ap.add_argument("--force", action="store_true", help="rebuild even if output exists")
    a = ap.parse_args()

    month = os.path.basename(a.input).replace("lichess_db_standard_rated_", "").split(".")[0]
    done = os.path.join(a.outdir, month, "_SUCCESS")
    if os.path.exists(done) and not a.force:
        print(f"[{month}] already built, skipping")
        return

    c = build(a.input, a.outdir, a.workers)
    rate = c["seen"] / c["seconds"] if c["seconds"] else 0
    dropped = ", ".join(f"{r}={c[r]:,}" for r in REASONS if c[r])
    print(f"[{month}] {c['seen']:,} scanned, {c['kept']:,} kept ({100*c['kept']/max(c['seen'],1):.2f}%), "
          f"{c['plies']:,} plies, dropped: {dropped or 'none'}, "
          f"{c['seconds']:,}s ({rate:,.0f} games/s)", flush=True)
    with open(done, "w") as f:
        f.write(repr(c) + "\n")


if __name__ == "__main__":
    main()
