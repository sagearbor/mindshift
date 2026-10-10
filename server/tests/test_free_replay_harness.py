"""The free-replay plumbing in the recording-replay harness: an OFFLINE
server run writes its cache misses to a miss log (and still reports each
missed turn honestly), ``coach="rules"`` runs the no-LLM coach, and every
run reports where its coach answers came from."""

from __future__ import annotations

import json
import os

import numpy as np

from recreplay import answer_misses, server_run

MODEL = "claude-haiku-4-5"


def _turn(uid, text, start, end, is_self):
    return {"type": "turn_local", "session_id": "x", "turn_uid": uid, "speaker": "You" if is_self else "S2",
            "speaker_person_id": None, "speaker_match_score": None, "is_self": is_self, "text": text,
            "start_time": start, "end_time": end, "transcript_source": "on-device", "prosody": None,
            "text_tone": None, "suggestion": None, "suggestion_source": None, "tts_source": "on-device"}


def _pcm(seconds=4):
    return (np.random.default_rng(0).standard_normal(16000 * seconds) * 800).astype("<i2")


def test_offline_run_logs_misses_and_reports_sources(tmp_path):
    log = tmp_path / "misses.jsonl"
    cfg = server_run.ServerConfig(speed=4.0, offline=True, llm_cache_dir=tmp_path / "cache", llm_model=MODEL,
                                  miss_log=log, stop_timeout_s=30)
    run = server_run.run_server(_pcm(), [_turn("t-0", "I failed the quiz today.", 0.5, 1.6, False)], [2.0], cfg)
    assert run.error is None, run.error
    errors = [e["event"] for e in run.events if e["event"].get("type") == "suggestion_error"]
    assert errors and errors[0]["reason"] == "LLMCacheMiss"
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert rows and rows[0]["model"] == MODEL and "I failed the quiz today." in rows[0]["user"]
    assert run.llm["sources"] == {"api": 0, "cache": 0, "cc-subagent": 0}
    assert run.llm["offline_misses"] >= 1
    assert run.coach == "llm"


def test_rules_coach_run_needs_no_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("MINDSHIFT_COACH_RULES", raising=False)

    class Exploding:
        model = "none"

        def complete(self, **_):
            raise AssertionError("rules mode called the LLM")

        def stream_complete(self, **_):
            raise AssertionError("rules mode called the LLM")

    cfg = server_run.ServerConfig(speed=4.0, llm_client=Exploding(), coach="rules", stop_timeout_s=30)
    turns = [_turn("t-0", "and so on and so on", 0.5, 46.5, True), _turn("t-1", "Right.", 47.0, 48.5, False)]
    run = server_run.run_server(_pcm(), turns, [2.0, 3.0], cfg)
    assert run.error is None, run.error
    finals = [e["event"] for e in run.events if e["event"].get("type") == "suggestion"]
    assert [f["suggestion_source"] for f in finals] == ["rules"]
    assert not [e for e in run.events if e["event"].get("type") == "suggestion_error"]
    assert run.rules == {"turns": 2, "fires": 1, "by_rule": {"floor": 1}}
    assert run.coach == "rules"
    assert "MINDSHIFT_COACH_RULES" not in os.environ  # restored
    line = answer_misses.sources_line(run.llm, run.rules)
    assert "rules 1 fires / 2 turns" in line


def test_run_bundle_flags_every_answer_source(tmp_path):
    from recreplay import pipeline

    log = tmp_path / "m.jsonl"
    log.write_text("{}\n")
    srv = {"llm": {"sources": {"api": 0, "cache": 4, "cc-subagent": 7}, "cc_models": {"sonnet": 7},
                   "offline_misses": 0}, "rules": None}
    s = pipeline.llm_sources(srv, log)
    assert (s["api"], s["cache"], s["cc_subagent"], s["rules_fires"], s["paid_calls"]) == (0, 4, 7, 0, 0)
    assert s["miss_log"] == str(log)
    assert "simulated" in s["note"] and "cc-subagent 7 (sonnet)" in s["line"]
    r = pipeline.llm_sources({"llm": {}, "rules": {"turns": 9, "fires": 2}}, None)
    assert r["rules_fires"] == 2 and r["rules_turns"] == 9 and "note" not in r
