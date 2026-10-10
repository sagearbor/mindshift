"""On-disk LLM response cache.

Caches LLM API responses to avoid re-spending API credits on unchanged inputs.
Cache key = SHA-256(model + system_prompt + user_prompt). Responses are stored
as JSON under ``tests/fixtures/llm_cache/{key}.json``.

Usage:
    cache = LLMResponseCache(llm_client)
    result = cache.complete(system="...", user="...", max_tokens=256)

Set ``REFRESH_LLM_CACHE=1`` to force a real call even on a cache hit.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator

try:
    import spend_ledger  # server/ on sys.path (how the server and tests import)
except ImportError:  # imported as server.llm_cache from the repo root
    from server import spend_ledger  # type: ignore

# Default cache location: <repo-root>/tests/fixtures/llm_cache
CACHE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "llm_cache"


class LLMCacheMiss(RuntimeError):
    """An OFFLINE cache (no client) was asked for a prompt it never saw.
    Raised, never answered with a made-up response: the caller reports the
    turn as unavailable exactly as it would a real provider failure."""


class LLMResponseCache:
    """Wraps an LLMClient to cache its responses on disk.

    Beyond plain ``complete`` (the original use), it also serves the realtime
    pipeline, which streams (``stream_complete``) and runs every call in a
    worker thread:

    * ``stream_complete`` records the chunks with their arrival times; a hit
      yields the same chunks. ``complete`` and ``stream_complete`` share one
      entry per prompt.
    * ``llm_client=None`` (+ ``model``) is an OFFLINE cache: hits only, a miss
      raises :class:`LLMCacheMiss`.
    * ``replay_latency=True`` sleeps on a hit to reproduce the recorded
      first-chunk and total latency, so an offline replay of a live session
      keeps realistic timing (blocking sleep: callers already use a thread).
    """

    def __init__(
        self,
        llm_client,
        cache_dir: Path | None = None,
        *,
        model: str | None = None,
        replay_latency: bool = False,
    ):
        self._client = llm_client
        self._model = model
        if llm_client is None and not model:
            raise ValueError("an offline LLMResponseCache needs model=")
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._refresh = os.getenv("REFRESH_LLM_CACHE", "").strip() == "1"
        self._replay_latency = replay_latency
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "offline_misses": 0}
        self.log: list[dict] = []

    @property
    def model(self) -> str:
        return self._model or self._client.model

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _cache_key(model: str, system: str, user: str) -> str:
        """SHA-256 of model + system + user prompt."""
        payload = f"{model}\n---\n{system}\n---\n{user}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _note(self, kind: str, key: str, hit: bool, latency_ms: float | None, system: str) -> None:
        with self._lock:
            self.stats["hits" if hit else "misses"] += 1
            self.log.append({"kind": kind, "key": key[:12], "hit": hit, "latency_ms": latency_ms,
                             "system": str(system)[:60]})

    def _lookup(self, system: str, user: str) -> tuple[str, Path, dict | None]:
        key = self._cache_key(self.model, str(system), str(user))
        path = self._cache_path(key)
        if path.exists() and not self._refresh:
            return key, path, json.loads(path.read_text())
        return key, path, None

    def _offline_miss(self, key: str, system: str) -> LLMCacheMiss:
        with self._lock:
            self.stats["offline_misses"] += 1
            self.stats["misses"] += 1
            self.log.append({"kind": "miss", "key": key[:12], "hit": False, "latency_ms": None,
                             "system": str(system)[:60]})
        return LLMCacheMiss(f"offline LLM cache has no entry {key[:12]} (prompt changed since it was recorded)")

    def _write(self, path: Path, system: str, user: str, response: str, **extra) -> None:
        path.write_text(json.dumps(
            {"model": self.model, "system": str(system), "user": str(user), "response": response, **extra},
            indent=2,
        ))

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_schema: dict | None = None,
    ) -> str:
        """Return a cached response, or call the real LLM and cache the result."""
        key, path, data = self._lookup(system, user)
        if data is not None:
            if self._replay_latency and data.get("latency_ms"):
                time.sleep(float(data["latency_ms"]) / 1000.0)
            self._note("complete", key, True, data.get("latency_ms"), system)
            return data["response"]
        if self._client is None:
            raise self._offline_miss(key, system)

        # Cache miss or forced refresh — call the real LLM (inside the spend
        # ledger's cap when the offline tooling enabled it).
        spend_ledger.reserve("anthropic", spend_ledger.llm_cost(system, user, max_tokens=max_tokens))
        kwargs = {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        t0 = time.monotonic()
        response = self._client.complete(**kwargs)
        latency = round((time.monotonic() - t0) * 1000.0, 1)
        self._write(path, system, user, response, latency_ms=latency)
        spend_ledger.record("anthropic", spend_ledger.llm_cost(system, user, out_text=response), key[:12])
        self._note("complete", key, False, latency, system)
        return response

    def stream_complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_schema: dict | None = None,
    ) -> Iterator[str]:
        """Yield the completion as chunks (recorded ones on a hit). Nothing
        is looked up until iteration starts, like the real client."""
        key, path, data = self._lookup(system, user)
        if data is not None:
            return self._replay(key, data, system)
        if self._client is None:
            miss = self._offline_miss(key, system)

            def _raise() -> Iterator[str]:
                raise miss
                yield  # pragma: no cover — makes this a generator

            return _raise()
        return self._record(key, path, system, user, temperature, max_tokens, response_schema)

    def _replay(self, key: str, data: dict, system: str) -> Iterator[str]:
        chunks = data.get("chunks") or [[data.get("latency_ms") or 0.0, data["response"]]]
        self._note("stream", key, True, data.get("latency_ms"), system)
        t0 = time.monotonic()
        for at_ms, text in chunks:
            if self._replay_latency:
                wait = float(at_ms) / 1000.0 - (time.monotonic() - t0)
                if wait > 0:
                    time.sleep(wait)
            yield text

    def _record(self, key, path, system, user, temperature, max_tokens, response_schema) -> Iterator[str]:
        spend_ledger.reserve("anthropic", spend_ledger.llm_cost(system, user, max_tokens=max_tokens))
        kwargs = {"system": system, "user": user, "temperature": temperature, "max_tokens": max_tokens}
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        t0 = time.monotonic()
        chunks: list[list] = []
        streamer = getattr(self._client, "stream_complete", None)
        source = streamer(**kwargs) if callable(streamer) else iter([self._client.complete(**kwargs)])
        for text in source:
            chunks.append([round((time.monotonic() - t0) * 1000.0, 1), text])
            yield text
        latency = round((time.monotonic() - t0) * 1000.0, 1)
        self._write(path, system, user, "".join(c[1] for c in chunks), chunks=chunks, latency_ms=latency,
                    first_chunk_ms=chunks[0][0] if chunks else None)
        spend_ledger.record("anthropic", spend_ledger.llm_cost(system, user, out_text="".join(c[1] for c in chunks)),
                            key[:12])
        self._note("stream", key, False, latency, system)
