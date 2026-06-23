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
from pathlib import Path

# Default cache location: <repo-root>/tests/fixtures/llm_cache
CACHE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "llm_cache"


class LLMResponseCache:
    """Wraps an LLMClient to cache its responses on disk."""

    def __init__(self, llm_client, cache_dir: Path | None = None):
        self._client = llm_client
        self._cache_dir = cache_dir or CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._refresh = os.getenv("REFRESH_LLM_CACHE", "").strip() == "1"

    @staticmethod
    def _cache_key(model: str, system: str, user: str) -> str:
        """SHA-256 of model + system + user prompt."""
        payload = f"{model}\n---\n{system}\n---\n{user}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Return a cached response, or call the real LLM and cache the result."""
        model = self._client.model
        key = self._cache_key(model, system, user)
        path = self._cache_path(key)

        # Cache hit (and not forcing a refresh).
        if path.exists() and not self._refresh:
            data = json.loads(path.read_text())
            return data["response"]

        # Cache miss or forced refresh — call the real LLM.
        response = self._client.complete(
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        path.write_text(json.dumps(
            {"model": model, "system": system, "user": user, "response": response},
            indent=2,
        ))
        return response
