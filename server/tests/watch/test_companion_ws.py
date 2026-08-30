# Tier B (2026-08-30): companion sockets — the watch's no-mic, phone-listens
# mode. The socket authenticates like any live-session WS, announces itself
# with `{"type":"companion"}`, sends JSON heartbeats only (never PCM), receives
# relayed vector_event/nudge frames via watch/relay.py, and persists NOTHING —
# neither on `end` nor on an abrupt disconnect.
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from models.audio import TurnLocalEvent, TurnProsody, TurnTextTone
from watch import relay
from watch.models import EnrollmentBaseline
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app


@pytest.fixture(autouse=True)
def _clean_registry():
    """Module-level relay registry must never leak between tests."""
    relay._registry.clear()
    yield
    relay._registry.clear()


def _hostile_turn() -> TurnLocalEvent:
    return TurnLocalEvent(
        session_id="phone-1", speaker="Speaker A", is_self=True,
        text="I said I'm fine.", start_time=1.0, end_time=2.0,
        transcript_source="on-device",
        prosody=TurnProsody(rms_dbfs=-30.0),
        text_tone=TurnTextTone(frustration=78, label="frustrated"),
    )


def test_companion_hello_is_acked_and_heartbeats_are_silently_tolerated():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/companion-1?account=alice") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        assert json.loads(ws.receive_text()) == {"type": "companion_ack"}

        # A heartbeat produces NO frame at all: the next thing received must be
        # the error for the bogus frame sent right after it — if the heartbeat
        # had produced anything (an ack, an error), it would arrive first.
        ws.send_text(json.dumps({"type": "heartbeat"}))
        ws.send_text(json.dumps({"type": "bogus"}))
        assert json.loads(ws.receive_text()) == {"type": "error", "detail": "unknown_type"}


def test_companion_abrupt_disconnect_persists_nothing():
    # The exact junk-doc case this mode exists to avoid: an end-less companion
    # socket that never sent PCM drops (pocket dead zone) — no `not_analyzed`
    # live session may be saved for it.
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/companion-2?account=bob") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        assert json.loads(ws.receive_text()) == {"type": "companion_ack"}
        ws.send_text(json.dumps({"type": "heartbeat"}))
        # No "end" — the context manager just closes the socket.

    assert asyncio.run(store.get_live_session("companion-2")) is None


def test_companion_end_persists_nothing_and_reports_status_companion():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/companion-3?account=carol") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        assert json.loads(ws.receive_text()) == {"type": "companion_ack"}
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved == {
            "type": "live_session_saved",
            "live_session_id": "companion-3",
            "status": "companion",
        }

    assert asyncio.run(store.get_live_session("companion-3")) is None


def test_phone_turn_nudges_a_companion_socket_and_still_persists_nothing():
    """The whole point of Tier B: the phone hears a hostile self turn; the
    relay escalates the account's OPEN companion socket; the wrist gets
    vector_event + nudge frames — and no live-session doc ever exists."""
    store = MemoryLiveSessionStore()
    asyncio.run(store.put_baseline(
        EnrollmentBaseline(account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x"),
    ))
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/companion-4?account=alice") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        assert json.loads(ws.receive_text()) == {"type": "companion_ack"}
        # Registered for the relay exactly like a mic session — registration
        # happens at accept, before (and regardless of) any PCM.
        assert relay.live_session_for("alice") is not None

        relay.push_turn_local("alice", _hostile_turn())
        vector_event = json.loads(ws.receive_text())
        nudge = json.loads(ws.receive_text())
        assert vector_event["type"] == "vector_event"
        assert vector_event["vector"] == "aggressive_tone" and vector_event["level"] == 2
        assert nudge == {"type": "nudge", "channel": "A", "level": 2, "t": 0.0, "vectors": ["aggressive_tone"]}
        # No "end" — abrupt disconnect, the companion's ordinary way out.

    assert relay.live_session_for("alice") is None, "unregistered on the way out"
    assert asyncio.run(store.get_live_session("companion-4")) is None
