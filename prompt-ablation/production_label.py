"""Production labeling run over data/4/player_game_index.parquet.

Deterministically splits the 30,085,219 game-occurrences 1:2.5 between
Apertus-v1.5-70B and Gemma-4-31B (both free, CSCS-hosted), decodes real SAN
moves from data/2b's moves.u16 binary format, generates 5-label sets with the
s10-mini prompt, and writes one JSON row per occurrence:

    {month, row_idx, player, elo, side, model,
     labels: {terse_1, terse_2, short, medium, long}, ok, error}

Resumable: progress is checkpointed to disk after every 1,024-row chunk per
logical model. A chunk is marked complete only when every call parsed as a
five-element pack; incomplete chunks are retried on the next run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import chess
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autolabel import corpus, genome, providers  # noqa: E402
from move_codec import decode as decode_move  # noqa: E402

DATA_2B = ROOT / "data" / "2b"
INDEX_PATH = ROOT / "data" / "4" / "player_game_index.parquet"
OUT_DIR = Path(__file__).resolve().parent / "runs" / "production"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_ROUTE_STATE = OUT_DIR / "production_active_routes.json"

APERTUS_CANDIDATES = (
    "swissai:swiss-ai/Apertus-v1.5-70B",
    "swissai:swiss-ai/Apertus-v1.5-70B-thinking",
    "swissai:apertus-ai/Apertus-v1.5-70B",
)
DEFAULT_MODELS = {
    # ``apertus`` is a logical corpus share, not one fixed deployment. At
    # startup every candidate below is health-checked and live rows are spread
    # across just that available pool.
    "apertus": APERTUS_CANDIDATES,
    "gemma": "swissai:google/gemma-4-31B-it-bdoan",
}
FIREWORKS_MODEL = os.environ.get("FIREWORKS_LABEL_MODEL", "").strip()
# Set FIREWORKS_LABEL_MODEL to a Fireworks serverless or dedicated deployment
# ID to send the whole corpus to one provider, e.g.
# accounts/fireworks/models/gpt-oss-120b. Do not start this script until the
# preflight has measured parsing, output length, and sustainable rate.
MODELS = ({"fireworks": f"fireworks:{FIREWORKS_MODEL}"}
          if FIREWORKS_MODEL else DEFAULT_MODELS)
CONCURRENCY = ({"fireworks": int(os.environ.get("FIREWORKS_LABEL_CONCURRENCY", "32"))}
               # 4,000 was not usable: this program sends 64-row waves and
               # it just made all direct routes retry together. 64 is a real
               # concurrent wave; increase only after a live server proves it
               # can sustain more.
               if FIREWORKS_MODEL else {"apertus": 64, "gemma": 64})
CHUNK_SIZE = 1024
# Each healthy model gets a 192-request wave. Thus adding a live replica really
# adds request capacity (2 routes -> 384 concurrent calls; 3 -> 576) instead
# of merely splitting a fixed global trickle between more GPU nodes.
SEND_BATCH_SIZE_PER_ROUTE = 192
# Five requested labels normally use ~150-250 output tokens. Leaving this at
# 512 lets occasional runaway descriptions monopolise an entire concurrent
# wave; 384 retains room for the five requested bands while bounding that tail.
APERTUS_MAX_TOKENS = 384
PARQUET_READ_BATCH = 100_000

VARIANT_ID = os.environ.get("LABEL_PROMPT_VARIANT", "s10-mini")
VARIANT = next(v for v in genome.SEEDS if v.vid == VARIANT_ID)
BAND_ORDER = ["terse", "terse", "short", "medium", "long"]
BAND_KEYS = ["terse_1", "terse_2", "short", "medium", "long"]


def assign_model(month: str, row_idx: int, side: str) -> str:
    """Deterministic 1:2.5 apertus:gemma split via a stable hash (2/7 vs 5/7)."""
    h = hashlib.blake2b(f"{month}|{row_idx}|{side}".encode(), digest_size=8).digest()
    bucket = int.from_bytes(h, "little") % 7
    return "apertus" if bucket < 2 else "gemma"


class MonthCache:
    """Read only needed meta parquet row groups, not whole multi-million-row months."""

    def __init__(self):
        self._files: dict[str, tuple[pq.ParquetFile, np.memmap, list[int]]] = {}
        self._groups: OrderedDict[tuple[str, int], dict] = OrderedDict()
        # A row group can itself be large.  Retaining dozens of groups makes a
        # player-ordered chunk span enough months to OOM before its first API
        # request; a small LRU trades a little rereading for bounded memory.
        self._max_groups = 4

    def get(self, month: str, row_idx: int):
        if month not in self._files:
            meta_path = DATA_2B / month / "meta.parquet"
            moves_path = DATA_2B / month / "moves.u16"
            pf = pq.ParquetFile(meta_path)
            starts, total = [], 0
            for group in range(pf.num_row_groups):
                starts.append(total)
                total += pf.metadata.row_group(group).num_rows
            moves = np.memmap(moves_path, dtype="<u2", mode="r")
            self._files[month] = (pf, moves, starts)
        pf, moves, starts = self._files[month]
        group = max(i for i, start in enumerate(starts) if start <= row_idx)
        key = (month, group)
        if key not in self._groups:
            table = pf.read_row_group(group, columns=[
                "move_off", "n_plies", "white_elo", "black_elo", "tc_base", "tc_incr", "eco", "opening",
            ])
            self._groups[key] = {
                "move_off": table.column("move_off").to_numpy(),
                "n_plies": table.column("n_plies").to_numpy(),
                "white_elo": table.column("white_elo").to_numpy(),
                "black_elo": table.column("black_elo").to_numpy(),
                "tc_base": table.column("tc_base").to_numpy(),
                "tc_incr": table.column("tc_incr").to_numpy(),
                "eco": table.column("eco"), "opening": table.column("opening"),
            }
            if len(self._groups) > self._max_groups:
                self._groups.popitem(last=False)
        self._groups.move_to_end(key)
        return self._groups[key], moves, row_idx - starts[group]


def decode_game(meta: dict, moves: np.memmap, row_idx: int, month: str, side: str) -> corpus.Game:
    off = int(meta["move_off"][row_idx])
    n = int(meta["n_plies"][row_idx])
    codes = moves[off:off + n]
    board = chess.Board()
    sans = []
    for code in codes:
        mv = decode_move(int(code))
        sans.append(board.san(mv))
        board.push(mv)
    # result is unused: VARIANT.reveal_result=False means it never reaches the
    # prompt, so it's not worth guessing the exact uint8 code mapping.
    tc = f"{int(meta['tc_base'][row_idx])}+{int(meta['tc_incr'][row_idx])}"
    return corpus.Game(
        game_id=f"{month}_{row_idx}",
        moves_san=" ".join(sans),
        result="",
        white_elo=int(meta["white_elo"][row_idx]),
        black_elo=int(meta["black_elo"][row_idx]),
        opening=str(meta["opening"][row_idx].as_py() or ""),
        eco=str(meta["eco"][row_idx].as_py() or ""),
        time_control=tc,
    )


def chunk_state_path(model_key: str) -> Path:
    return OUT_DIR / f"{model_key}.chunks.json"


def load_done_chunks(model_key: str) -> set[int]:
    """Load durable completed-chunk markers; seed them from legacy output."""
    path = chunk_state_path(model_key)
    if path.exists():
        return set(json.loads(path.read_text()).get("done", []))
    out = OUT_DIR / f"{model_key}.jsonl"
    # The previous writer appended in deterministic assignment order.  Full
    # 1024-row prefixes can therefore be safely carried into the new scheme.
    existing = sum(1 for _ in out.open(encoding="utf-8")) if out.exists() else 0
    return set(range(existing // CHUNK_SIZE))


def save_done_chunks(model_key: str, done: set[int]) -> None:
    path = chunk_state_path(model_key)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"chunk_size": CHUNK_SIZE, "done": sorted(done)}))
    tmp.replace(path)


async def process_batch(client, model_key, specs, rows, month_cache, out_f, pbar):
    jobs_meta = []
    msgs_list = []
    # The index is player-ordered, so a chunk alternates between months.  Decode
    # month-by-month inside the immutable chunk to reuse each parquet row group
    # instead of repeatedly evicting/reloading it.
    for r in sorted(rows, key=lambda item: item["month"]):
        meta, moves, local_idx = month_cache.get(r["month"], r["row_idx"])
        g = decode_game(meta, moves, local_idx, r["month"], r["side"])
        ctx = corpus.build_context(
            g, r["side"], card=VARIANT.card, moves=VARIANT.moves,
            anonymise=VARIANT.anonymise, reveal_result=VARIANT.reveal_result,
        )
        msgs = VARIANT.messages(ctx)
        jobs_meta.append(r)
        msgs_list.append(msgs)

    params = {}
    if model_key == "fireworks":
        # Gemma 4 otherwise spends thousands of tokens in hidden reasoning.
        # A schema prevents its occasional trailing-comma JSON failures.
        params = {
            "max_tokens": 512,
            "reasoning_effort": "none",
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "chess_style_labels",
                    "schema": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "band": {"type": "string", "enum": ["terse", "short", "medium", "long"]},
                                "text": {"type": "string", "minLength": 1},
                            },
                            "required": ["band", "text"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        }
    elif model_key == "apertus":
        # The thinking deployment is useful only if its hidden chain of
        # thought is disabled. This setting is harmless on the non-thinking
        # Apertus routes and keeps all three outputs comparable.
        params = {"max_tokens": APERTUS_MAX_TOKENS,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}

    async def one(r, msgs):
        # Route selection is deterministic within the currently healthy pool;
        # an unavailable replica never receives traffic. The resulting model
        # is recorded on every row, so mixed Apertus provenance stays visible.
        route = int.from_bytes(
            hashlib.blake2b(f"{r['month']}|{r['row_idx']}|{r['side']}".encode(),
                            digest_size=8).digest(), "little") % len(specs)
        spec = specs[route]
        try:
            text = await client.chat(spec, msgs, **params)
            pairs = genome.parse_labels(text, VARIANT.rungs)
            labels = genome.assign_bands(pairs, VARIANT.rungs)
            texts = [t for _, t in labels]
            ok = len(texts) == 5
            row = {
                "month": r["month"], "row_idx": r["row_idx"], "player": r["player"],
                "elo": r["elo"], "side": r["side"], "model": spec,
                "labels": dict(zip(BAND_KEYS, texts + [None] * (5 - len(texts)))),
                "ok": ok, "error": "" if ok else f"expected 5 labels, got {len(texts)}",
            }
        except Exception as e:  # noqa: BLE001
            row = {
                "month": r["month"], "row_idx": r["row_idx"], "player": r["player"],
                "elo": r["elo"], "side": r["side"], "model": spec,
                "labels": {k: None for k in BAND_KEYS},
                "ok": False, "error": f"{type(e).__name__}: {e}"[:300],
            }
        # Written as each call finishes (not after the whole batch) so progress
        # is visible continuously rather than jumping once per ~10-minute
        # batch. asyncio is single-threaded and this write has no `await`
        # inside it, so concurrent calls can't interleave mid-line. Tradeoff:
        # on a crash mid-batch, rows already written here can exceed the
        # batch-level checkpoint, so a resume may reprocess (and duplicate)
        # up to one batch's worth of already-written rows — acceptable for a
        # $0-cost run; dedupe downstream on (month, row_idx, side, model) if
        # it ever matters.
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        pbar.update(1)
        return row

    return await asyncio.gather(*(one(r, m) for r, m in zip(jobs_meta, msgs_list)))


def iter_chunks(model_key: str, limit: int | None):
    """Stream only this model's rows from parquet, never materialising 30M rows."""
    parquet = pq.ParquetFile(INDEX_PATH)
    pending: list[dict] = []
    seen = 0
    for batch in parquet.iter_batches(batch_size=PARQUET_READ_BATCH):
        for row in batch.to_pylist():
            if limit is not None and seen >= limit:
                break
            seen += 1
            assigned = "fireworks" if FIREWORKS_MODEL else assign_model(
                row["month"], row["row_idx"], row["side"],
            )
            if assigned != model_key:
                continue
            pending.append(row)
            if len(pending) == CHUNK_SIZE:
                yield pending
                pending = []
        if limit is not None and seen >= limit:
            break
    if pending:
        yield pending


async def run_worker(model_key: str, limit: int | None, position: int):
    specs = MODELS[model_key]
    if isinstance(specs, str):
        specs = (specs,)
    done = load_done_chunks(model_key)
    route_count = len(specs)
    wave_size = (SEND_BATCH_SIZE_PER_ROUTE * route_count
                 if model_key == "apertus" else SEND_BATCH_SIZE_PER_ROUTE)
    client = providers.Client(
        concurrency=max(CONCURRENCY[model_key], wave_size),
        max_retries=6 if model_key == "fireworks" else 4,
        use_cache=False,
    )
    out_path = OUT_DIR / f"{model_key}.jsonl"
    pbar = tqdm(desc=model_key, position=position, unit="labels")
    month_cache = MonthCache()  # persists across all batches for this worker

    with open(out_path, "a", encoding="utf-8") as out_f:
        for chunk_id, batch in enumerate(iter_chunks(model_key, limit)):
            if chunk_id in done:
                pbar.update(len(batch))
                continue
            t0 = time.perf_counter()
            # A 1024-row chunk is the durable resume unit.  Preparing all
            # 1024 game records first delays the initial request because the
            # player-ordered index spans many months, so submit it in small
            # batches and make visible progress immediately.
            chunk_rows = []
            for start in range(0, len(batch), wave_size):
                chunk_rows.extend(await process_batch(
                    client, model_key, specs, batch[start:start + wave_size],
                    month_cache, out_f, pbar,
                ))
            # Do not make failures permanent. Partial rows remain visible for
            # diagnosis; this chunk deliberately has no durable marker and is
            # retried (with downstream dedupe by month/row/side/model).
            if all(row["ok"] for row in chunk_rows):
                done.add(chunk_id)
                save_done_chunks(model_key, done)
            else:
                failed = sum(not row["ok"] for row in chunk_rows)
                print(f"{model_key}: chunk {chunk_id} has {failed} failed calls; will retry", flush=True)
            dt = time.perf_counter() - t0
            pbar.set_postfix(rate=f"{len(batch)/dt:.1f} rows/s", errs=client.stats.get("fail", 0))

    await client.aclose()
    pbar.close()


async def healthy_models() -> list[str]:
    """Probe routes once so unavailable replicas get no corpus traffic."""
    active: list[str] = []
    for key, wanted in list(MODELS.items()):
        candidates = (wanted,) if isinstance(wanted, str) else wanted
        # Resolve against /models first: slurm jobs may have a routable suffix.
        candidates = tuple(providers.resolve(candidates, verbose=False))
        healthy: list[str] = []
        for spec in candidates:
            client = providers.Client(concurrency=1, max_retries=1, use_cache=False)
            try:
                params = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                          if key == "apertus" else {})
                await client.chat(spec, [{"role": "user", "content": "Reply only: ok"}], **params)
                healthy.append(spec)
            except Exception as error:  # noqa: BLE001
                print(f"skipping unavailable {spec}: {type(error).__name__}: {error}", flush=True)
            finally:
                await client.aclose()
        if healthy:
            MODELS[key] = tuple(healthy)
            active.append(key)
            print(f"{key}: {len(healthy)} live route(s): " + ", ".join(healthy), flush=True)
    # This is the producer's own authoritative startup snapshot.  The
    # dashboard must not guess activity from old JSONL rows or a hard-coded
    # expected replica count.
    tmp = ACTIVE_ROUTE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "updated_at": time.time(),
        "routes": {key: list(MODELS[key]) for key in active},
    }, indent=2, sort_keys=True))
    tmp.replace(ACTIVE_ROUTE_STATE)
    return active


async def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("streaming data/4/player_game_index.parquet in 1024-row resumable chunks ...", flush=True)

    if FIREWORKS_MODEL:
        print(f"fireworks: model={FIREWORKS_MODEL}, "
              f"concurrency={CONCURRENCY['fireworks']}", flush=True)
        await run_worker("fireworks", limit, 0)
    else:
        active = await healthy_models()
        print("active label routes: " + (", ".join(active) or "none"), flush=True)
        if not active:
            raise RuntimeError("no label endpoint passed startup health check")
        print("apertus/gemma split is deterministic (2/7 vs 5/7); completed chunks are skipped.", flush=True)
        await asyncio.gather(*(run_worker(key, limit, position)
                               for position, key in enumerate(active)))


if __name__ == "__main__":
    asyncio.run(main())
