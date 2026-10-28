"""Fan a variant out over (model x game x side) and collect label sets."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from . import corpus, genome, providers


@dataclass
class LabelSet:
    """All labels one model produced for one side of one game under one variant."""

    game_id: str
    color: str
    vid: str
    model: str
    labels: list[tuple[str, str]] = field(default_factory=list)  # (band, text)
    latency_ms: float = 0.0
    ok: bool = True
    error: str = ""
    raw_chars: int = 0

    @property
    def key(self) -> str:
        return f"{self.vid}|{self.model}|{self.game_id}|{self.color}"

    @property
    def texts(self) -> list[str]:
        return [t for _, t in self.labels]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSet":
        d = dict(d)
        d["labels"] = [tuple(x) for x in d.get("labels", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


GEN_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 2200,
    "extra_body": {"top_k": 40},
}


async def generate(
    client: providers.Client,
    variants: Sequence[genome.Variant],
    models: Sequence[str],
    games: Sequence[corpus.Game],
    colors: Sequence[str] = ("white", "black"),
    *,
    params: dict | None = None,
    progress: bool = True,
) -> list[LabelSet]:
    """Run every (variant, model, game, side) combination concurrently."""
    params = {**GEN_PARAMS, **(params or {})}
    jobs: list[tuple] = []
    meta: list[tuple] = []
    for v in variants:
        for g in games:
            for color in colors:
                ctx = corpus.build_context(
                    g, color, card=v.card, moves=v.moves,
                    anonymise=v.anonymise, reveal_result=v.reveal_result,
                )
                msgs = v.messages(ctx)
                for m in models:
                    jobs.append((m, msgs, params))
                    meta.append((v, m, g, color))

    total = len(jobs)
    if progress:
        print(f"  [generate] {total} calls "
              f"({len(variants)} variants x {len(models)} models x "
              f"{len(games)} games x {len(colors)} sides)")

    done = 0
    results: list[LabelSet] = []
    lock = asyncio.Lock()

    async def one(idx: int) -> LabelSet:
        nonlocal done
        spec, msgs, prm = jobs[idx]
        v, model, g, color = meta[idx]
        t0 = time.perf_counter()
        ls = LabelSet(game_id=g.game_id, color=color, vid=v.vid, model=model)
        try:
            text = await client.chat(spec, msgs, **prm)
            ls.raw_chars = len(text)
            pairs = genome.parse_labels(text, v.rungs)
            ls.labels = genome.assign_bands(pairs, v.rungs)
            ls.ok = bool(ls.labels)
            if not ls.ok:
                ls.error = "unparseable response"
        except Exception as e:  # noqa: BLE001
            ls.ok = False
            ls.error = f"{type(e).__name__}: {e}"[:300]
        ls.latency_ms = (time.perf_counter() - t0) * 1000.0
        async with lock:
            done += 1
            if progress and (done % max(1, total // 20) == 0 or done == total):
                print(f"  [generate] {done}/{total}", flush=True)
        return ls

    results = list(await asyncio.gather(*(one(i) for i in range(total))))
    return results


def write_jsonl(rows: Sequence[LabelSet], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[LabelSet]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [LabelSet.from_dict(json.loads(line)) for line in f if line.strip()]
