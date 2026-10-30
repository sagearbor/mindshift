"""Tests for the on-disk LLM response cache (server/llm_cache.py)."""

from unittest.mock import MagicMock


from llm_cache import LLMResponseCache


def _client(response: str = "hello", model: str = "claude-3-haiku-20240307"):
    client = MagicMock()
    client.model = model
    client.complete.return_value = response
    return client


def test_cache_miss_calls_client_and_writes_file(tmp_path):
    client = _client("first answer")
    cache = LLMResponseCache(client, cache_dir=tmp_path)

    result = cache.complete(system="sys", user="usr")

    assert result == "first answer"
    client.complete.assert_called_once()
    # Exactly one cache file was written.
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_hit_does_not_call_client_again(tmp_path):
    client = _client("cached answer")
    cache = LLMResponseCache(client, cache_dir=tmp_path)

    first = cache.complete(system="sys", user="usr")
    second = cache.complete(system="sys", user="usr")

    assert first == second == "cached answer"
    # Underlying client invoked only on the miss, not the hit.
    client.complete.assert_called_once()


def test_different_prompts_use_different_keys(tmp_path):
    client = _client("answer")
    cache = LLMResponseCache(client, cache_dir=tmp_path)

    cache.complete(system="sys", user="a")
    cache.complete(system="sys", user="b")

    assert client.complete.call_count == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_refresh_env_forces_recall(tmp_path, monkeypatch):
    client = _client("answer")
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    cache.complete(system="sys", user="usr")  # populate cache

    monkeypatch.setenv("REFRESH_LLM_CACHE", "1")
    refreshed = LLMResponseCache(client, cache_dir=tmp_path)
    refreshed.complete(system="sys", user="usr")

    # Second instance ignored the cached file and called the client again.
    assert client.complete.call_count == 2


def test_cache_key_is_stable_for_same_inputs(tmp_path):
    key1 = LLMResponseCache._cache_key("model-x", "sys", "usr")
    key2 = LLMResponseCache._cache_key("model-x", "sys", "usr")
    key3 = LLMResponseCache._cache_key("model-y", "sys", "usr")

    assert key1 == key2
    assert key1 != key3


# ---------------------------------------------------------------------------
# Streaming, offline replay and recorded latency (recording-replay fixtures)
# ---------------------------------------------------------------------------

import time  # noqa: E402

import pytest  # noqa: E402

from llm_cache import LLMCacheMiss  # noqa: E402


class _StreamingClient:
    model = "claude-x"

    def __init__(self, chunks, delay=0.0):
        self.chunks = chunks
        self.delay = delay
        self.stream_calls = 0
        self.complete_calls = 0

    def complete(self, system, user, temperature=0.7, max_tokens=512, response_schema=None):
        self.complete_calls += 1
        return "".join(self.chunks)

    def stream_complete(self, system, user, temperature=0.7, max_tokens=512, response_schema=None):
        self.stream_calls += 1
        for c in self.chunks:
            time.sleep(self.delay)
            yield c


def test_stream_miss_records_chunks_and_latency_then_hits(tmp_path):
    client = _StreamingClient(['{"sugg', 'estions": ["a"]}'], delay=0.02)
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    out = "".join(cache.stream_complete(system="s", user="u"))
    assert out == '{"suggestions": ["a"]}'
    entry = next(tmp_path.glob("*.json")).read_text()
    assert '"chunks"' in entry and '"latency_ms"' in entry and '"first_chunk_ms"' in entry
    again = list(cache.stream_complete(system="s", user="u"))
    assert "".join(again) == out and len(again) == 2
    assert client.stream_calls == 1
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1


def test_complete_and_stream_share_one_entry(tmp_path):
    client = _StreamingClient(["ab", "c"])
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    assert cache.complete(system="s", user="u") == "abc"
    assert "".join(cache.stream_complete(system="s", user="u")) == "abc"
    assert client.stream_calls == 0


def test_offline_without_client_raises_on_miss_and_serves_hits(tmp_path):
    LLMResponseCache(_StreamingClient(["x"]), cache_dir=tmp_path).complete(system="s", user="u")
    off = LLMResponseCache(None, cache_dir=tmp_path, model="claude-x")
    assert off.model == "claude-x"
    assert off.complete(system="s", user="u") == "x"
    with pytest.raises(LLMCacheMiss):
        off.complete(system="s", user="other")
    with pytest.raises(LLMCacheMiss):
        list(off.stream_complete(system="s", user="other"))
    assert off.stats["offline_misses"] == 2


def test_replay_latency_reproduces_the_recorded_timing(tmp_path):
    client = _StreamingClient(["a", "b"], delay=0.08)
    LLMResponseCache(client, cache_dir=tmp_path).stream_complete  # noqa: B018
    "".join(LLMResponseCache(client, cache_dir=tmp_path).stream_complete(system="s", user="u"))
    replay = LLMResponseCache(None, cache_dir=tmp_path, model="claude-x", replay_latency=True)
    t0 = time.monotonic()
    first_at = None
    for _ in replay.stream_complete(system="s", user="u"):
        if first_at is None:
            first_at = time.monotonic() - t0
    total = time.monotonic() - t0
    assert first_at >= 0.06 and total >= 0.14
    fast = LLMResponseCache(None, cache_dir=tmp_path, model="claude-x")
    t0 = time.monotonic()
    "".join(fast.stream_complete(system="s", user="u"))
    assert time.monotonic() - t0 < 0.05


def test_cached_prefix_prompt_keys_like_its_text(tmp_path):
    from llm_client import CachedPrefixPrompt

    client = _StreamingClient(["x"])
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    cache.complete(system=CachedPrefixPrompt("pre ", "body"), user="u")
    cache.complete(system="pre body", user="u")
    assert client.complete_calls == 1
