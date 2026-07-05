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
