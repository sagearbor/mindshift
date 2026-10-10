"""scripts/recreplay/answer_misses.py: an offline replay's miss log ->
batch files a Claude Code subagent answers ($0, Max plan) -> cache entries
labelled ``cc-subagent:<model>`` (never passed off as the coach model), with
SIMULATED latency drawn from recorded real calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from recreplay import answer_misses  # noqa: E402
from llm_cache import LLMCacheMiss, LLMResponseCache  # noqa: E402

MODEL = "claude-haiku-4-5"
SYS = "You are a coach. Reply with JSON: {\"suggestions\": [...], \"importance\": n}"


def _miss(tmp_path, users, system=SYS):
    """Run an offline cache over ``users``; return (cache_dir, miss_log)."""
    cache_dir, log = tmp_path / "cache", tmp_path / "misses.jsonl"
    cache = LLMResponseCache(None, cache_dir=cache_dir, model=MODEL, miss_log=log)
    for u in users:
        with pytest.raises(LLMCacheMiss):
            list(cache.stream_complete(system=system, user=u, max_tokens=300))
    return cache_dir, log


def _real_entry(cache_dir: Path, user: str, latency: float, chunks):
    key = LLMResponseCache._cache_key(MODEL, "other", user)
    (cache_dir / f"{key}.json").write_text(json.dumps({
        "model": MODEL, "system": "other", "user": user, "response": "".join(c[1] for c in chunks),
        "chunks": chunks, "latency_ms": latency, "first_chunk_ms": chunks[0][0]}))


def _answer_all(batch_dir: Path, make):
    for b in sorted(batch_dir.glob("batch_*.md")):
        n = b.stem.split("_")[1]
        ids = [r["id"] for r in answer_misses.read_manifest(batch_dir) if r["batch"] == int(n)]
        (batch_dir / f"answers_{n}.jsonl").write_text(
            "".join(json.dumps({"id": i, "response": make(i)}) + "\n" for i in ids))


def test_export_writes_batches_with_the_exact_prompts(tmp_path):
    _, log = _miss(tmp_path, ["turn one", "turn two", "turn three"])
    out = tmp_path / "batches"
    summary = answer_misses.export([log], out, batch_size=2)
    assert summary["prompts"] == 3 and summary["batches"] == 2
    text = (out / "batch_00.md").read_text()
    assert SYS in text and "turn one" in text and "turn two" in text
    assert "answers_00.jsonl" in text  # tells the subagent where to write
    rows = answer_misses.read_manifest(out)
    assert {r["user"] for r in rows} == {"turn one", "turn two", "turn three"}
    assert all(r["key"] == LLMResponseCache._cache_key(MODEL, SYS, r["user"]) for r in rows)


def test_export_skips_prompts_already_answered(tmp_path):
    cache_dir, log = _miss(tmp_path, ["a", "b"])
    key = LLMResponseCache._cache_key(MODEL, SYS, "a")
    (cache_dir / f"{key}.json").write_text(json.dumps({"response": "x"}))
    assert answer_misses.export([log], tmp_path / "out")["prompts"] == 1


def test_ingest_writes_labelled_entries_with_simulated_latency_and_offline_replay_hits(tmp_path):
    cache_dir, log = _miss(tmp_path, ["turn one", "turn two"])
    _real_entry(cache_dir, "recorded", 1066.0, [[600.0, '{"suggestions": ["Hi'], [1066.0, '"], "importance": 40}']])
    out = tmp_path / "batches"
    answer_misses.export([log], out)
    _answer_all(out, lambda i: json.dumps({"suggestions": [f"line {i}"], "importance": 55}))
    summary = answer_misses.ingest(out, answered_by="sonnet")
    assert summary["written"] == 2 and summary["rejected"] == []
    key = LLMResponseCache._cache_key(MODEL, SYS, "turn one")
    entry = json.loads((cache_dir / f"{key}.json").read_text())
    assert entry["provenance"] == "cc-subagent:sonnet"
    assert entry["model"] == f"{MODEL}@cc-subagent:sonnet"
    assert entry["latency_simulated"] is True
    assert entry["latency_ms"] == 1066.0 and entry["chunks"][-1][0] == 1066.0
    assert "".join(c[1] for c in entry["chunks"]) == entry["response"]
    # the offline replay now hits, and counts it as a stand-in answer
    cache = LLMResponseCache(None, cache_dir=cache_dir, model=MODEL)
    got = "".join(cache.stream_complete(system=SYS, user="turn one", max_tokens=300))
    assert json.loads(got)["importance"] == 55
    assert cache.stats["sources"]["cc-subagent"] == 1 and cache.stats["offline_misses"] == 0


def test_ingest_rejects_non_json_unknown_ids_and_never_overwrites_real_entries(tmp_path):
    cache_dir, log = _miss(tmp_path, ["one", "two", "three"])
    out = tmp_path / "batches"
    answer_misses.export([log], out)
    rows = {r["user"]: r for r in answer_misses.read_manifest(out)}
    # a real answer for "three" lands after the export (e.g. a paid run)
    real = cache_dir / f"{rows['three']['key']}.json"
    real.write_text(json.dumps({"model": MODEL, "response": "REAL", "latency_ms": 900.0}))
    (out / "answers_00.jsonl").write_text("\n".join([
        json.dumps({"id": rows["one"]["id"], "response": "not json at all"}),
        json.dumps({"id": "deadbeef", "response": "{}"}),
        json.dumps({"id": rows["two"]["id"], "response": "```json\n{\"suggestions\": [\"ok\"], \"importance\": 9}\n```"}),
        json.dumps({"id": rows["three"]["id"], "response": "{\"suggestions\": [], \"importance\": 1}"}),
    ]) + "\n")
    summary = answer_misses.ingest(out, answered_by="sonnet", latency_ms_fallback=1000.0)
    reasons = {r["id"]: r["reason"] for r in summary["rejected"]}
    assert "json" in reasons[rows["one"]["id"]].lower()
    assert "unknown" in reasons["deadbeef"]
    assert "real" in reasons[rows["three"]["id"]]
    assert json.loads(real.read_text())["response"] == "REAL"
    assert summary["written"] == 1
    two = json.loads((cache_dir / f"{rows['two']['key']}.json").read_text())
    assert json.loads(two["response"])["importance"] == 9  # fences stripped
    assert two["latency_source"].startswith("recorded:")  # timing borrowed from the real entry
    assert summary["unanswered"] == 0  # every prompt got an answer line (2 were rejected)


def test_ingest_needs_a_latency_source(tmp_path):
    _, log = _miss(tmp_path, ["one"])
    out = tmp_path / "batches"
    answer_misses.export([log], out)
    _answer_all(out, lambda i: "{}")
    with pytest.raises(answer_misses.NoLatencySource):
        answer_misses.ingest(out, answered_by="sonnet")
    assert answer_misses.ingest(out, answered_by="sonnet", latency_ms_fallback=1066.0)["written"] == 1
    [entry] = [json.loads(p.read_text()) for p in (tmp_path / "cache").glob("*.json")]
    assert entry["latency_source"] == "fallback:1066ms" and entry["latency_simulated"] is True


def test_sources_summary_line():
    s = answer_misses.sources_line({"sources": {"api": 0, "cache": 3, "cc-subagent": 5}, "offline_misses": 0,
                                    "cc_models": {"sonnet": 5}}, {"turns": 0, "fires": 0})
    assert "real API 0" in s and "cache 3" in s and "cc-subagent 5 (sonnet)" in s and "rules 0" in s
    assert "simulated" in s
