"""Debug save of live sessions with ZERO phone turns.

At the family dinner the phone's on-device loop produced no turns, so
POST /sessions/live was refused with 422 (``turns`` had min_length=1) and
the session was lost — including the server's own cloud transcript, the
one record of what went wrong.

Since 2026-10-18 this is PER-SESSION OPT-IN (the phone's Settings ->
Diagnostics -> "Debug session capture" switch, off by default): the WS
config frame carries ``debug_capture: true`` and the POST body carries
``debug_capture: true``. Only then is the server's transcript kept in
memory and a zero-turn POST stored. A normal user's zero-turn POST is the
422 again. ``LIVE_DEBUG_SAVE_EMPTY_SESSIONS`` (env
``MINDSHIFT_LIVE_DEBUG_SAVE_EMPTY_SESSIONS``, default on) stays as a
server-wide kill switch: "0" = never, whatever the phone asks.
"""

from __future__ import annotations

import asyncio
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
    """A zero-turn POST from a phone with debug capture ON (the default
    here; pass debug_capture=False for a normal user)."""
    extra.setdefault("debug_capture", True)
    return _body(turns=[], analyze=False, reflect=False, **extra)


def _debug_ctx(uid=UID):
    return SessionContext(session_id=SESSION_ID, uid=uid, debug_capture=True)


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
    ctx = _debug_ctx()
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
    other = _debug_ctx(uid="someone-else")
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
    audio_pipeline._debug_register_session(
        SessionContext(session_id="s", uid="u", debug_capture=True))
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}


# --- per-session opt-in (2026-10-18) ------------------------------------------


def test_normal_users_zero_turn_post_is_422(env):
    """No `debug_capture` in the body (every normal user) -> the 422."""
    client, store = env
    res = client.post("/sessions/live", json=_empty(debug_capture=False))
    assert res.status_code == 422
    assert "turn" in res.text
    body = _empty()
    del body["debug_capture"]
    assert client.post("/sessions/live", json=body).status_code == 422
    assert store.save_calls == 0


def test_debug_capture_must_be_a_bool(env):
    client, store = env
    res = client.post("/sessions/live", json=_empty(debug_capture="please"))
    assert res.status_code == 422
    assert store.save_calls == 0


def test_session_without_debug_capture_registers_nothing(env):
    """Default config (no `debug_capture`): no transcript kept in memory."""
    ctx = SessionContext(session_id=SESSION_ID, uid=UID)
    assert ctx.debug_capture is False
    audio_pipeline._debug_register_session(ctx)
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}
    assert ctx.debug_transcript is None


def test_debug_capture_session_registers(env):
    ctx = _debug_ctx()
    audio_pipeline._debug_register_session(ctx)
    assert (UID, SESSION_ID) in audio_pipeline._DEBUG_TRANSCRIPTS
    assert ctx.debug_transcript is not None


def test_config_frame_debug_capture_parsing(env):
    ctx = SessionContext(session_id=SESSION_ID, uid=UID)
    asyncio.run(audio_pipeline._apply_config(ctx, {"empathy_slider": 40}))
    assert ctx.debug_capture is False
    # Wrong types are ignored (validated-or-ignored, like report_latency).
    asyncio.run(audio_pipeline._apply_config(ctx, {"debug_capture": "true"}))
    assert ctx.debug_capture is False
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}
    asyncio.run(audio_pipeline._apply_config(ctx, {"debug_capture": True}))
    assert ctx.debug_capture is True
    # An authenticated session that opts in by a later config frame is
    # registered at once (the opening frame registers after auth).
    assert (UID, SESSION_ID) in audio_pipeline._DEBUG_TRANSCRIPTS


def test_config_frame_debug_capture_before_auth_waits(env):
    """Pre-auth (no uid yet): the flag is recorded; registration happens
    after auth (handle_audio_stream calls _debug_register_session)."""
    ctx = SessionContext(session_id=SESSION_ID)
    asyncio.run(audio_pipeline._apply_config(ctx, {"debug_capture": True}))
    assert ctx.debug_capture is True
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}


def test_config_frame_debug_capture_respects_kill_switch(env, monkeypatch):
    monkeypatch.setattr(audio_pipeline, "LIVE_DEBUG_SAVE_EMPTY_SESSIONS", False)
    ctx = SessionContext(session_id=SESSION_ID, uid=UID)
    asyncio.run(audio_pipeline._apply_config(ctx, {"debug_capture": True}))
    audio_pipeline._debug_register_session(ctx)
    assert audio_pipeline._DEBUG_TRANSCRIPTS == {}


def test_non_empty_post_ignores_debug_capture(env):
    """`debug_capture` changes nothing for a normal (turns > 0) session."""
    client, store = env
    res = client.post(
        "/sessions/live", json=_body(analyze=False, reflect=False, debug_capture=True),
    )
    assert res.status_code == 201, res.text
    assert "debug_no_phone_turns" not in (_stored(store).get("meta") or {})
