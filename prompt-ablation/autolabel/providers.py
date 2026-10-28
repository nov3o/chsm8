"""Unified async access to SwissAI / OpenAI / xAI / Fireworks, with disk caching.

Every call is keyed by (model spec, messages, params) and cached on disk, so
re-running the loop after a crash — or re-scoring labels you already paid for —
costs nothing. This matters: the optimizer re-reads earlier generations often.

Model specs are strings of the form ``provider:model_id``::

    swissai:swiss-ai/Apertus-v1.5-70B
    openai:gpt-5.2
    xai:grok-4.5
    fireworks:accounts/fireworks/models/gpt-oss-120b

SwissAI models are user-launched slurm jobs with an ``expires_at``; they come
and go. Never hardcode one — call :func:`discover` and intersect with what you
want.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import httpx
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "runs" / "_cache"

PROVIDERS = {
    "swissai": "https://api.swissai.svc.cscs.ch/v1",
    "openai": "https://api.openai.com/v1",
    "xai": "https://api.x.ai/v1",
    # Fireworks exposes an OpenAI-compatible inference endpoint.  Keep this
    # separate from its management/batch API base URL.
    "fireworks": "https://api.fireworks.ai/inference/v1",
}

_KEY_ENV = {
    "swissai": ("CSCS_SERVING_API", "SWISSAI_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "xai": ("GROK_API_KEY", "XAI_API_KEY"),
    "fireworks": ("FIREWORKS_API_KEY",),
}


def _load_keys() -> dict[str, str]:
    """Env vars win; fall back to a gitignored ``keys.py`` next to the package."""
    out: dict[str, str] = {}
    fallback = {}
    for path in (ROOT / "keys.py", ROOT.parent.parent.parent / "keys.py"):
        if path.exists():
            ns: dict = {}
            exec(compile(path.read_text(), str(path), "exec"), ns)  # noqa: S102
            fallback = {k: v for k, v in ns.items() if k.endswith("_API_KEY")}
            break
    for provider, names in _KEY_ENV.items():
        for name in names:
            if os.environ.get(name):
                out[provider] = os.environ[name]
                break
        else:
            for name in names:
                if name in fallback:
                    out[provider] = fallback[name]
                    break
    return out


KEYS = _load_keys()

THINK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning traces that open-weight thinking models leak into content."""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    # Some models emit an unopened trailing </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

    @classmethod
    def parse(cls, spec: str) -> "ModelSpec":
        if ":" not in spec:
            raise ValueError(f"model spec must be 'provider:model', got {spec!r}")
        provider, model = spec.split(":", 1)
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; have {list(PROVIDERS)}")
        return cls(provider, model)

    def __str__(self) -> str:  # noqa: D105
        return f"{self.provider}:{self.model}"

    @property
    def short(self) -> str:
        """Compact label for filenames and tables."""
        return self.model.split("/")[-1]


class ModelUnavailable(RuntimeError):
    """Raised when a model is gone (expired slurm job, 404) rather than erroring."""


def _sanitize(spec: ModelSpec, params: dict) -> dict:
    """Per-family parameter fixups.

    The GPT-5 family rejects ``temperature``/``top_p``/``max_tokens`` on the
    chat completions endpoint and wants ``max_completion_tokens`` instead.
    vLLM-backed SwissAI servers accept ``top_k`` only inside ``extra_body``.
    """
    p = dict(params)
    if spec.provider == "openai" and spec.model.startswith(("gpt-5", "o3", "o4")):
        p.pop("temperature", None)
        p.pop("top_p", None)
        p.pop("presence_penalty", None)
        if "max_tokens" in p:
            p["max_completion_tokens"] = p.pop("max_tokens")
        # Reasoning tokens are drawn from the same budget as the visible answer.
        # At the default effort a gpt-5 model will happily spend 2k tokens
        # thinking and return an empty string, which looks exactly like an
        # outage. Cap the thinking and leave real headroom for the answer.
        p.setdefault("reasoning_effort", "low")
        p["max_completion_tokens"] = max(p.get("max_completion_tokens", 0), 8000)
        extra = p.pop("extra_body", {}) or {}
        extra.pop("top_k", None)
        if extra:
            p["extra_body"] = extra
    if spec.provider == "xai":
        p.pop("presence_penalty", None)
        # Grok 4.20 rejects both OpenAI-style repetition penalties.  Keep the
        # provider adapter strict so Fireworks-shaped batch payloads can be
        # replayed against xAI without wasting requests on 400s.
        p.pop("frequency_penalty", None)
        extra = p.pop("extra_body", {}) or {}
        extra.pop("top_k", None)
        extra.pop("chat_template_kwargs", None)
        if extra:
            p["extra_body"] = extra
    if spec.provider == "fireworks" and "top_k" in p:
        extra = p.pop("extra_body", {}) or {}
        extra["top_k"] = p.pop("top_k")
        p["extra_body"] = extra
    return p


def _cache_key(kind: str, spec: str, payload: object) -> Path:
    blob = json.dumps([kind, spec, payload], sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(blob.encode()).hexdigest()[:40]
    return CACHE_DIR / kind / f"{h}.json"


@dataclass
class Client:
    """Async fan-out client with a global concurrency cap and a disk cache."""

    concurrency: int = 24
    timeout: float = 300.0
    max_retries: int = 4
    use_cache: bool = True
    _sems: dict[str, asyncio.Semaphore] = field(default_factory=dict, repr=False)
    _clients: dict[str, AsyncOpenAI] = field(default_factory=dict, repr=False)
    stats: dict[str, int] = field(default_factory=lambda: {"hit": 0, "miss": 0, "fail": 0})
    errors: dict[str, int] = field(default_factory=dict, repr=False)

    def failure_summary(self) -> str:
        """One line naming what actually went wrong, for end-of-phase logging.

        Silently swallowing errors and reporting a zero metric makes a broken
        provider look like a bad prompt.
        """
        if not self.errors:
            return f"calls: {self.stats['miss']} live, {self.stats['hit']} cached, no failures"
        top = sorted(self.errors.items(), key=lambda kv: -kv[1])[:4]
        return (f"calls: {self.stats['miss']} live, {self.stats['hit']} cached, "
                f"{self.stats['fail']} FAILED — " + "; ".join(f"{k} x{v}" for k, v in top))

    def _client(self, provider: str) -> AsyncOpenAI:
        if provider not in self._clients:
            key = KEYS.get(provider)
            if not key:
                raise ModelUnavailable(f"no API key configured for provider {provider!r}")
            self._clients[provider] = AsyncOpenAI(
                api_key=key,
                base_url=PROVIDERS[provider],
                timeout=httpx.Timeout(self.timeout, connect=20.0),
                max_retries=0,  # we do our own, with jitter
            )
        return self._clients[provider]

    def _sem(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._sems:
            self._sems[provider] = asyncio.Semaphore(self.concurrency)
        return self._sems[provider]

    # -- chat ------------------------------------------------------------

    async def chat(
        self,
        spec: str | ModelSpec,
        messages: Sequence[dict],
        **params,
    ) -> str:
        """One chat completion. Returns stripped text, or raises ModelUnavailable."""
        ms = spec if isinstance(spec, ModelSpec) else ModelSpec.parse(spec)
        params = _sanitize(ms, params)
        cache_path = _cache_key("chat", str(ms), [list(messages), params])
        if self.use_cache and cache_path.exists():
            self.stats["hit"] += 1
            return json.loads(cache_path.read_text())["text"]

        last: Exception | None = None
        for attempt in range(self.max_retries):
            # The semaphore is acquired per attempt, not around the whole retry
            # loop: holding a concurrency slot while sleeping off a backoff
            # starves every other caller and can stall a whole run behind a
            # handful of failing requests.
            try:
                async with self._sem(ms.provider):
                    resp = await self._client(ms.provider).chat.completions.create(
                        model=ms.model, messages=list(messages), **params
                    )
                if not resp.choices:
                    raise RuntimeError("empty choices")
                text = strip_thinking(resp.choices[0].message.content or "")
                if not text:
                    raise RuntimeError("empty completion after stripping thinking")
                self.stats["miss"] += 1
                if self.use_cache:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps({
                        "spec": str(ms), "text": text, "ts": int(time.time()),
                    }, ensure_ascii=False))
                return text
            except Exception as e:  # noqa: BLE001 - provider errors are untyped
                last = e
                msg = str(e).lower()
                fatal = any(s in msg for s in (
                    "not found", "does not exist", "no such model",
                    "invalid_api_key", "incorrect api key", "unauthorized",
                ))
                if fatal:
                    break
                if "rate limit" in msg or "429" in msg:
                    self.stats["ratelimit"] = self.stats.get("ratelimit", 0) + 1
                await asyncio.sleep(min(30.0, 2.0 ** attempt) * (0.6 + random.random()))
        self.stats["fail"] += 1
        self.errors[f"{ms}: {type(last).__name__}"] = (
            self.errors.get(f"{ms}: {type(last).__name__}", 0) + 1)
        raise ModelUnavailable(f"{ms}: {type(last).__name__}: {last}")

    async def chat_many(
        self,
        jobs: Iterable[tuple[str | ModelSpec, Sequence[dict], dict]],
        on_error: str = "none",
    ) -> list[str | None]:
        """Run many chats concurrently. ``on_error='none'`` yields None per failure."""
        async def one(spec, messages, params):
            try:
                return await self.chat(spec, messages, **params)
            except Exception:  # noqa: BLE001
                if on_error == "raise":
                    raise
                return None

        return await asyncio.gather(*(one(s, m, p) for s, m, p in jobs))

    # -- embeddings ------------------------------------------------------

    async def embed(
        self,
        texts: Sequence[str],
        model: str = "openai:text-embedding-3-small",
        batch: int = 256,
    ) -> list[list[float]]:
        """Embed texts, caching per individual string so partial reuse works."""
        ms = ModelSpec.parse(model)
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            path = _cache_key("embed", str(ms), t)
            if self.use_cache and path.exists():
                out[i] = json.loads(path.read_text())["v"]
                self.stats["hit"] += 1
            else:
                todo.append(i)

        for start in range(0, len(todo), batch):
            idxs = todo[start:start + batch]
            payload = [texts[i] or " " for i in idxs]
            async with self._sem(ms.provider):
                resp = await self._client(ms.provider).embeddings.create(
                    model=ms.model, input=payload
                )
            for i, item in zip(idxs, resp.data):
                out[i] = item.embedding
                self.stats["miss"] += 1
                if self.use_cache:
                    path = _cache_key("embed", str(ms), texts[i])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"v": item.embedding}))
        return [v for v in out if v is not None]

    async def aclose(self) -> None:
        for c in self._clients.values():
            await c.close()


# -- discovery ------------------------------------------------------------

def discover(provider: str = "swissai", ready_only: bool = True) -> list[str]:
    """List model ids a provider is currently serving, as full ``provider:model`` specs."""
    key = KEYS.get(provider)
    if not key:
        return []
    try:
        r = httpx.get(
            f"{PROVIDERS[provider]}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()["data"]
    except Exception:  # noqa: BLE001
        return []
    # Only the `id` field is routable. SwissAI also advertises a friendly
    # `labels.served_model_name` (e.g. meta-llama/Llama-3.3-70B-Instruct) but
    # requesting that alias returns 503 "No provider found" — the router keys
    # on the id, which for some jobs is the on-disk model path.
    ids = []
    for m in data:
        if ready_only and m.get("status") not in (None, "ready"):
            continue
        if m.get("id"):
            ids.append(f"{provider}:{m['id']}")
    return sorted(dict.fromkeys(ids))


def _alias_index(live: Sequence[str]) -> dict[str, str]:
    """Map every plausible way of naming a live model to its routable spec."""
    idx: dict[str, str] = {}
    for spec in live:
        model = spec.split(":", 1)[1]
        idx.setdefault(model, spec)
        parts = model.split("/")
        # ".../models/meta-llama/Llama-3.3-70B-Instruct" -> "meta-llama/Llama-..."
        if len(parts) >= 2:
            idx.setdefault("/".join(parts[-2:]), spec)
        idx.setdefault(parts[-1], spec)
        # Strip the "-<launcher>" suffix SwissAI appends to served names.
        idx.setdefault(re.sub(r"-[a-z]{3,10}$", "", parts[-1]), spec)
    return idx


def resolve(wanted: Sequence[str], verbose: bool = True) -> list[str]:
    """Filter a wishlist down to specs that are live, rewriting aliases.

    Non-SwissAI specs pass through — their catalogues are stable. SwissAI
    specs are matched against the live slurm jobs by full id, by
    ``org/model`` suffix, or by bare model name, and rewritten to whatever is
    actually routable right now.
    """
    live = discover("swissai")
    idx = _alias_index(live)
    kept, dropped = [], []
    for spec in wanted:
        if not spec.startswith("swissai:"):
            kept.append(spec)
            continue
        model = spec.split(":", 1)[1]
        hit = idx.get(model) or idx.get(model.split("/")[-1])
        if hit:
            kept.append(hit)
        else:
            dropped.append(spec)
    if verbose and dropped:
        print(f"  [providers] {len(dropped)} SwissAI model(s) not live, skipping: "
              + ", ".join(s.split(":", 1)[1] for s in dropped))
    return list(dict.fromkeys(kept))
