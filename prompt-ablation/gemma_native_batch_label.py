#!/usr/bin/env python3
"""Resumable high-throughput Gemma 4 labeler using vLLM's native batch API.

One 1,536-row durable chunk is divided across four 384-conversation native
vLLM batch calls.  This is deliberately separate from ``gemma.jsonl``: that
file is the earlier public-route run and contains historical failed rows.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prompt-ablation"))
import production_label as pl  # noqa: E402
from autolabel import corpus, genome  # noqa: E402

OUT_DIR = ROOT / "prompt-ablation" / "runs" / "production"
BATCH_PER_REPLICA = 384
BAND_KEYS = ["terse_1", "terse_2", "short", "medium", "long"]
_WORKER_CACHE: pl.MonthCache | None = None


def record_key(row: dict) -> tuple[str, int, str]:
    return str(row["month"]), int(row["row_idx"]), str(row["side"])


def valid_keys(path: Path) -> set[tuple[str, int, str]]:
    """Existing valid packs are the source of truth for a resumable import."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("ok"):
                    keys.add(record_key(row))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return keys


def import_valid_legacy(source: Path, output: Path, seen: set[tuple[str, int, str]]) -> int:
    """Copy only valid historic Gemma packs once; failures are deliberately skipped."""
    copied = 0
    with source.open(encoding="utf-8") as inp, output.open("a", encoding="utf-8") as out:
        for line in inp:
            try:
                row = json.loads(line)
                key = record_key(row)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if not row.get("ok") or key in seen:
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(key)
            copied += 1
    return copied


def build_message(row: dict) -> tuple[dict, list[dict]]:
    """CPU-heavy SAN decoding and prompt formatting, safe in a worker process."""
    global _WORKER_CACHE
    if _WORKER_CACHE is None:
        _WORKER_CACHE = pl.MonthCache()
    meta, moves, local_idx = _WORKER_CACHE.get(row["month"], row["row_idx"])
    game = pl.decode_game(meta, moves, local_idx, row["month"], row["side"])
    context = corpus.build_context(
        game, row["side"], card=pl.VARIANT.card, moves=pl.VARIANT.moves,
        anonymise=pl.VARIANT.anonymise, reveal_result=pl.VARIANT.reveal_result,
    )
    return row, pl.VARIANT.messages(context)


def make_messages(rows: list[dict], pool: concurrent.futures.ProcessPoolExecutor | None) -> list[tuple[dict, list[dict]]]:
    ordered = sorted(rows, key=lambda x: x["month"])
    if pool is None:
        return [build_message(row) for row in ordered]
    # map preserves corpus order, while 16 independent caches turn the Python
    # chess/SAN loop from a one-core prefill gap into a parallel CPU stage.
    return list(pool.map(build_message, ordered, chunksize=8))


def iter_gemma_waves(wave: int, limit: int | None):
    """Stream the deterministic Gemma share in full, four-replica waves."""
    parquet = pq.ParquetFile(pl.INDEX_PATH)
    pending: list[dict] = []
    seen = 0
    for batch in parquet.iter_batches(batch_size=pl.PARQUET_READ_BATCH):
        for row in batch.to_pylist():
            if limit is not None and seen >= limit:
                break
            seen += 1
            if pl.assign_model(row["month"], row["row_idx"], row["side"]) != "gemma":
                continue
            pending.append(row)
            if len(pending) == wave:
                yield pending
                pending = []
        if limit is not None and seen >= limit:
            break
    if pending:
        yield pending


def call_batch(endpoint: str, model: str, jobs: list[tuple[dict, list[dict]]],
               temperature: float, top_p: float, max_tokens: int) -> list[str]:
    payload = {
        "model": model,
        "messages": [messages for _, messages in jobs],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                endpoint.rstrip("/") + "/v1/chat/completions/batch",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(request, timeout=900) as response:
                choices = json.load(response)["choices"]
            if len(choices) != len(jobs):
                raise RuntimeError(f"returned {len(choices)} choices for {len(jobs)} requests")
            return [choice["message"]["content"] for choice in choices]
        except Exception as exc:  # retry transient scheduling/network failures
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{endpoint}: batch failed after retries: {last_error}")


def to_record(row: dict, text: str, model: str) -> dict:
    pairs = genome.assign_bands(genome.parse_labels(text, pl.VARIANT.rungs), pl.VARIANT.rungs)
    texts = [value for _, value in pairs]
    ok = len(texts) == 5 and all(value.strip() for value in texts)
    return {
        "month": row["month"], "row_idx": row["row_idx"], "player": row["player"],
        "elo": row["elo"], "side": row["side"], "model": f"local:{model}",
        "labels": dict(zip(BAND_KEYS, texts + [None] * (5 - len(texts)))),
        "ok": ok, "error": "" if ok else f"expected 5 labels, got {len(texts)}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints", required=True,
                        help="comma-separated private vLLM endpoints; append |model for a per-endpoint alias")
    parser.add_argument("--model", default="gemma4-31b-standard-x4")
    parser.add_argument("--batch-per-replica", type=int, default=BATCH_PER_REPLICA,
                        help="native requests submitted to each replica per wave")
    parser.add_argument("--message-workers", type=int,
                        default=min(16, max(1, os.cpu_count() or 1)),
                        help="processes used to decode games and construct prompts")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=384,
                        help="per-pack completion cap; keep 384 for production until a real-corpus 256-token truncation audit passes")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "gemma31_tp4_x4.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="source rows, for a smoke test")
    parser.add_argument("--max-waves", type=int, default=None,
                        help="stop after this many submitted waves (benchmark helper)")
    parser.add_argument("--import-valid-from", type=Path,
                        help="one-time import of valid historic packs before resuming")
    args = parser.parse_args()
    endpoints = []
    for item in args.endpoints.split(","):
        item = item.strip()
        if not item:
            continue
        endpoint, separator, endpoint_model = item.partition("|")
        endpoints.append((endpoint.rstrip("/"), endpoint_model or args.model))
    if not endpoints:
        parser.error("at least one endpoint is required")
    if args.batch_per_replica < 1:
        parser.error("--batch-per-replica must be positive")
    wave = args.batch_per_replica * len(endpoints)
    state_path = args.output.with_suffix(".chunks.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # A different replica count changes the wave boundaries.  Existing valid
    # keys still prevent duplicate requests, but old chunk ids must not skip a
    # different slice of the corpus.
    done = set(state.get("done", [])) if state.get("chunk_size") == wave else set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = valid_keys(args.output)
    # Emit the live configuration before optional historic import work.  The
    # dashboard can then distinguish a new writer from stale prior-rate logs.
    print(json.dumps({"event": "start", "endpoints": [url for url, _ in endpoints], "wave": wave,
                      "output": str(args.output), "done_chunks": len(done)}), flush=True)
    if args.import_valid_from:
        copied = import_valid_legacy(args.import_valid_from, args.output, seen)
        print(json.dumps({"event": "legacy_import", "copied_valid_packs": copied,
                          "seen_valid_packs": len(seen)}), flush=True)
    if args.message_workers < 0:
        parser.error("--message-workers must be non-negative")

    def persist_state() -> None:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"chunk_size": wave, "done": sorted(done)}))
        tmp.replace(state_path)

    def submit(rows: list[dict], chunk_ids: list[int], pool) -> None:
        """Write one coalesced native batch and checkpoint every source chunk.

        Historic imports leave a sparse pattern (often 384 missing records in
        each 1536-record source chunk).  Coalescing those fragments is vital:
        otherwise only one of four replicas receives work during resume.
        """
        started = time.perf_counter()
        jobs = make_messages(rows, pool)
        # Resume fragments can overshoot the nominal wave (e.g. two sparse
        # 2,304-row source chunks contribute 3,072 missing packs).  Balance
        # that larger wave across every replica instead of sending a second
        # 384-pack request to the first few endpoints.
        per_endpoint = math.ceil(len(jobs) / len(endpoints))
        partitions = [jobs[i:i + per_endpoint]
                      for i in range(0, len(jobs), per_endpoint)]
        # A serving job can expire independently of the other TP4 replicas.
        # Do not let that turn a healthy two- or three-node labeling pool into
        # a total outage: preserve successful partitions and replay only the
        # failed partition through a survivor.  If every endpoint fails, the
        # exception remains fatal so systemd can retry after the pool returns.
        outputs: list[list[str] | None] = [None] * len(partitions)
        failed_partitions: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(partitions)) as request_pool:
            futures = {
                request_pool.submit(call_batch, endpoints[i % len(endpoints)][0],
                                    endpoints[i % len(endpoints)][1], part,
                                    args.temperature, args.top_p, args.max_tokens): i
                for i, part in enumerate(partitions)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    outputs[index] = future.result()
                except Exception as exc:
                    failed_partitions.append(index)
                    print(json.dumps({"event": "replica_partition_failed",
                                      "partition": index, "error": str(exc)}), flush=True)
        if failed_partitions:
            survivors = [index for index, output in enumerate(outputs) if output is not None]
            if not survivors:
                raise RuntimeError("all Gemma replica partitions failed")
            print(json.dumps({"event": "replay_failed_partitions",
                              "count": len(failed_partitions), "survivors": len(survivors)}), flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(failed_partitions)) as replay_pool:
                replay = {
                    replay_pool.submit(call_batch,
                                       endpoints[survivors[offset % len(survivors)] % len(endpoints)][0],
                                       endpoints[survivors[offset % len(survivors)] % len(endpoints)][1],
                                       partitions[index], args.temperature, args.top_p, args.max_tokens): index
                    for offset, index in enumerate(failed_partitions)
                }
                for future in concurrent.futures.as_completed(replay):
                    outputs[replay[future]] = future.result()
        assert all(output is not None for output in outputs)
        records = []
        for part, texts in zip(partitions, outputs, strict=True):
            records.extend(to_record(row, text, args.model)
                           for (row, _), text in zip(part, texts, strict=True))
        # A rare malformed model response should not throw away a whole
        # 2,304-pack wave. Retry only those individual requests, rotating them
        # across replicas.  This keeps the checkpoint atomic while preventing
        # systemd from repeatedly redoing good work because of one bad pack.
        failed_positions = [i for i, record in enumerate(records) if not record["ok"]]
        for attempt in range(1, 4):
            if not failed_positions:
                break
            print(json.dumps({"event": "retry_invalid_packs", "attempt": attempt,
                              "count": len(failed_positions)}), flush=True)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(failed_positions), len(endpoints))) as retry_pool:
                futures = {
                    retry_pool.submit(call_batch,
                                      endpoints[position % len(endpoints)][0],
                                      endpoints[position % len(endpoints)][1],
                                      [jobs[position]], args.temperature, args.top_p, args.max_tokens): position
                    for position in failed_positions
                }
                for future in concurrent.futures.as_completed(futures):
                    position = futures[future]
                    text = future.result()[0]
                    row, _ = jobs[position]
                    records[position] = to_record(row, text, args.model)
            failed_positions = [i for i, record in enumerate(records) if not record["ok"]]
        if failed_positions:
            raise RuntimeError(
                f"invalid pack in source chunks {chunk_ids}: {len(failed_positions)} failures after retries")
        with args.output.open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        seen.update(record_key(record) for record in records)
        done.update(chunk_ids)
        persist_state()
        elapsed = time.perf_counter() - started
        print(json.dumps({"event": "chunk_complete", "chunks": chunk_ids,
                          "rows": len(records), "packs_per_s": len(records) / elapsed,
                          "completed_chunks": len(done)}), flush=True)

    executor = (concurrent.futures.ProcessPoolExecutor(max_workers=args.message_workers)
                if args.message_workers else None)

    try:
        pending_rows: list[dict] = []
        pending_chunks: list[int] = []
        submitted_waves = 0
        for chunk_id, rows in enumerate(iter_gemma_waves(wave, args.limit)):
            if chunk_id in done:
                continue
            # A historic import can cover only part of this wave. Ask the model
            # solely for the missing rows; a completed wave remains checkpointed.
            rows = [row for row in rows if record_key(row) not in seen]
            if not rows:
                done.add(chunk_id)
                continue
            pending_rows.extend(rows)
            pending_chunks.append(chunk_id)
            # Full coalesced waves keep all replicas busy; a short final wave is
            # sent only after the source stream is exhausted.
            if len(pending_rows) >= wave:
                submit(pending_rows, pending_chunks, executor)
                submitted_waves += 1
                pending_rows, pending_chunks = [], []
                if args.max_waves is not None and submitted_waves >= args.max_waves:
                    break
        if pending_rows:
            submit(pending_rows, pending_chunks, executor)
        elif done:
            persist_state()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
