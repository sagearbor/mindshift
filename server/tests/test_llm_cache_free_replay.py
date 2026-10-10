"""Free-replay additions to server/llm_cache.py: the offline MISS LOG (the
exact prompt of every offline miss, so it can be answered later without a
paid call) and per-SOURCE answer counts (real API / recorded cache /
cc-subagent stand-in)."""

import json
from unittest.mock import MagicMock

import pytest

from llm_cache import LLMCacheMiss, LLMResponseCache

MODEL = "claude-haiku-4-5"


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_offline_complete_miss_is_logged_with_the_exact_prompt_and_still_raises(tmp_path):
    log = tmp_path / "misses.jsonl"
    cache = LLMResponseCache(None, cache_dir=tmp_path / "c", model=MODEL, miss_log=log)
    schema = {"type": "object"}
    with pytest.raises(LLMCacheMiss):
        cache.complete(system="SYS", user="USR", temperature=0.3, max_tokens=99, response_schema=schema)
    [row] = _rows(log)
    assert row["key"] == LLMResponseCache._cache_key(MODEL, "SYS", "USR")
    assert row["model"] == MODEL and row["system"] == "SYS" and row["user"] == "USR"
    assert row["temperature"] == 0.3 and row["max_tokens"] == 99 and row["response_schema"] == schema
    assert row["kind"] == "complete"
    assert row["cache_dir"] == str(tmp_path / "c")


def test_offline_stream_miss_is_logged_on_iteration(tmp_path):
    log = tmp_path / "misses.jsonl"
    cache = LLMResponseCache(None, cache_dir=tmp_path / "c", model=MODEL, miss_log=log)
    gen = cache.stream_complete(system="S", user="U", max_tokens=200)
    with pytest.raises(LLMCacheMiss):
        list(gen)
    [row] = _rows(log)
    assert row["kind"] == "stream" and row["max_tokens"] == 200 and row["user"] == "U"


def test_the_same_prompt_is_logged_once(tmp_path):
    log = tmp_path / "misses.jsonl"
    cache = LLMResponseCache(None, cache_dir=tmp_path / "c", model=MODEL, miss_log=log)
    for _ in range(3):
        with pytest.raises(LLMCacheMiss):
            cache.complete(system="S", user="U")
    assert len(_rows(log)) == 1
    assert cache.stats["offline_misses"] == 3


def test_miss_log_from_env(tmp_path, monkeypatch):
    log = tmp_path / "env.jsonl"
    monkeypatch.setenv("MINDSHIFT_LLM_MISS_LOG", str(log))
    cache = LLMResponseCache(None, cache_dir=tmp_path / "c", model=MODEL)
    with pytest.raises(LLMCacheMiss):
        cache.complete(system="S", user="U")
    assert len(_rows(log)) == 1


def test_no_miss_log_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MINDSHIFT_LLM_MISS_LOG", raising=False)
    cache = LLMResponseCache(None, cache_dir=tmp_path / "c", model=MODEL)
    with pytest.raises(LLMCacheMiss):
        cache.complete(system="S", user="U")
    assert not list(tmp_path.glob("*.jsonl"))


def test_sources_count_api_cache_and_cc_subagent(tmp_path):
    client = MagicMock()
    client.model = MODEL
    client.complete.return_value = "real"
    del client.stream_complete
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    cache.complete(system="S", user="U1")          # real call
    cache.complete(system="S", user="U1")          # recorded cache hit
    # an entry a CC subagent answered (scripts/recreplay/answer_misses.py)
    key = LLMResponseCache._cache_key(MODEL, "S", "U2")
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "model": MODEL + "@cc-subagent:sonnet", "provenance": "cc-subagent:sonnet",
        "system": "S", "user": "U2", "response": "stand-in", "latency_ms": 900.0, "latency_simulated": True,
    }))
    assert cache.complete(system="S", user="U2") == "stand-in"
    assert list(cache.stream_complete(system="S", user="U2")) == ["stand-in"]
    assert cache.stats["sources"] == {"api": 1, "cache": 1, "cc-subagent": 2}
    assert cache.stats["cc_models"] == {"sonnet": 2}


def test_real_calls_are_written_with_api_provenance(tmp_path):
    client = MagicMock()
    client.model = MODEL
    client.complete.return_value = "real"
    cache = LLMResponseCache(client, cache_dir=tmp_path)
    cache.complete(system="S", user="U")
    [f] = tmp_path.glob("*.json")
    assert json.loads(f.read_text())["provenance"] == "api"
