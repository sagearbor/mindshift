"""Debug save of live sessions with ZERO phone turns (owner-approved for now).

At the family dinner the phone's on-device loop produced no turns, so
POST /sessions/live was refused with 422 (``turns`` had min_length=1) and
the session was lost — including the server's own cloud transcript, the
one record of what went wrong. Behind ``LIVE_DEBUG_SAVE_EMPTY_SESSIONS``
(env ``MINDSHIFT_LIVE_DEBUG_SAVE_EMPTY_SESSIONS``, default on) such a POST
is stored, using the server's own transcript of that session when it has
one. With the flag off the old 422 is back.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import audio_pipeline
import main as main_module
from audio_pipeline import SessionContext
from main import app
from models.audio import Utterance
from routers import sessions as sessions_router
from test_sessions_live import SESSION_ID, FakeLiveStore, _body

UID = "test-user"  # conftest's default X-Test-Uid


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(audio_pipeline, "LIVE_DEBUG_SAVE_EMPTY_SESSIONS", True)
    audio_pipeline._DEBUG_TRANSCRIPTS.clear()
    store = FakeLiveStore()
    app.state.recordings_store = store
    main_module._rate_limiter.reset()
    llm = MagicMock()
    llm.complete.return_value = "not json"
    app.state.llm_client = llm
    try:
        yield TestClient(app), store
    finally:
        del app.state.recordings_store
        sessions_router.BACKGROUND_TASKS.clear()
        audio_pipeline._DEBUG_TRANSCRIPTS.clear()


def _stored(store):
    (rec,) = store._by_uid[UID].values()
    return rec


def _empty(**extra):
    return _body(turns=[], analyze=False, reflect=False, **extra)


def test_zero_turns_is_stored_not_422(env):
    client, store = env
    res = client.post("/sessions/live", json=_empty())
    assert res.status_code == 201, res.text
    assert res.json()["turn_count"] == 0
    rec = _stored(store)
    assert rec["meta"]["debug_no_phone_turns"] is True
    assert rec["meta"]["debug_transcript_source"] == "none"
    assert rec["turns"] == []


def test_zero_turns_keeps_the_servers_own_transcript(env):
    client, store = env
    ctx = SessionContext(session_id=SESSION_ID, uid=UID)
    audio_pipeline._debug_register_session(ctx)
    for i, (spk, text) in enumerate([
        ("Speaker A", "We had a quiz in math today."),
        ("Speaker B", "How did it go?"),
    ]):
        audio_pipeline._remember_utterance(ctx, Utterance(
            session_id=SESSION_ID, speaker=spk, text=text,
            start_time=i * 3.0, end_time=i * 3.0 + 2.0,
        ))
    res = client.post("/sessions/live", json=_empty())
    assert res.status_code == 201, res.text
    assert res.json()["turn_count"] == 2
    rec = _stored(store)
    assert rec["meta"]["debug_transcript_source"] == "server"
    assert [t["text"] for t in rec["turns"]] == ["We had a quiz in math today.", "How did it go?"]
    assert {t["speaker"] for t in rec["turns"]} == {"Speaker A", "Speaker B"}


def test_server_transcript_is_scoped_to_the_uid(env):
    client, store = env
    other = SessionContext(session_id=SESSION_ID, uid="someone-else")
    audio_pipeline._debug_register_session(other)
    audio_pipeline._remember_utterance(other, Utterance(
        session_id=SESSION_ID, speaker="Speaker A", text="secret", start_time=0, end_time=1,
    ))
    res = client.post("/sessions/live", json=_empty())
    assert res.status_code == 201
    assert "secret" not in json.dumps(_stored(store))


def test_session_context_kept_only_on_a_debug_save(env):
    client, store = env
    res = client.post("/sessions/live", json=_empty(session_context="Dinner with my son."))
    assert res.status_code == 201
    assert _stored(store)["meta"]["debug_session_context"] == "Dinner with my son."


def test_flag_off_restores_the_422(env, monkeypatch):
    client, store = env
    monkeypatch.setattr(audio_pipeline, "LIVE_DEBUG_SAVE_EMPTY_SESSIONS", False)
    res = client.post("/sessions/live", json=_empty())
    assert res.status_code == 422
    assert "turn" in res.text
    assert store.save_calls == 0


def test_flag_off_registers_nothing(monkeypatch):
    monkeypatch.setattr(audio_pipeline, "LIVE_DEBUG_SAVE_EMPTY_SESSIONS", False)
    audio_pipeline._DEBUG_TRANSCRIPTS.clear()
    audio_pipeline._debug_register_session(SessionContext(session_id="s", uid="u"))
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}
