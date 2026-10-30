"""The shared paid-call ledger refuses calls past its cap and records spend."""
import json

import pytest

import spend_ledger
from llm_cache import LLMResponseCache


class _FakeClient:
    model = "claude-haiku-4-5"

    def __init__(self):
        self.calls = 0

    def complete(self, **kw):
        self.calls += 1
        return "ok" * 50


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "spend.json"
    monkeypatch.setenv("MINDSHIFT_SPEND_LEDGER", str(p))
    monkeypatch.setenv("MINDSHIFT_SPEND_LEDGER_ON", "1")
    return p


def test_miss_is_recorded_and_hit_is_free(ledger, tmp_path):
    client = _FakeClient()
    cache = LLMResponseCache(client, cache_dir=tmp_path / "c")
    cache.complete(system="s" * 400, user="u" * 400)
    cache.complete(system="s" * 400, user="u" * 400)
    data = json.loads(ledger.read_text())
    assert client.calls == 1
    assert len(data["calls"]) == 1 and data["total_usd"] > 0


def test_cap_refuses_before_calling(ledger, tmp_path, monkeypatch):
    ledger.write_text(json.dumps({"cap_usd": 0.0001, "total_usd": 0.0, "calls": []}))
    client = _FakeClient()
    cache = LLMResponseCache(client, cache_dir=tmp_path / "c")
    with pytest.raises(spend_ledger.SpendCapExceeded):
        cache.complete(system="s" * 4000, user="u" * 4000, max_tokens=512)
    assert client.calls == 0


def test_env_cannot_raise_the_file_cap(ledger, monkeypatch):
    ledger.write_text(json.dumps({"cap_usd": 1.0, "total_usd": 0.99, "calls": []}))
    monkeypatch.setenv("MINDSHIFT_SPEND_CAP_USD", "100")
    with pytest.raises(spend_ledger.SpendCapExceeded):
        spend_ledger.reserve("anthropic", 0.05)


def test_disabled_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("MINDSHIFT_SPEND_LEDGER", str(tmp_path / "x.json"))
    monkeypatch.delenv("MINDSHIFT_SPEND_LEDGER_ON", raising=False)
    spend_ledger.reserve("anthropic", 999)
    spend_ledger.record("anthropic", 1.0)
    assert not (tmp_path / "x.json").exists()
