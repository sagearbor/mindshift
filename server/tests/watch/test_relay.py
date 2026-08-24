"""server/watch/relay.py — the phone->watch turn relay (Track 1).

Three layers, cheapest first:
1. pure conversion (`turn_local_to_vector_events`, `tone_level`) — the
   golden vectors in test_tone_escalation_vectors.py cover the rungs; here
   we only pin the pieces those can't express (the running-median fallback,
   provenance in `detail`).
2. `push_turn_local` against a hand-built LiveWatchSession with a fake
   `emit`, inside one asyncio loop — registry semantics, the is_self guard,
   the no-live-watch no-op, the "empty relay never ticks the policy" rule.
3. end to end through the real WS handler with Starlette's TestClient: a
   watch connects, the PHONE reports a calm-volume hostile turn from another
   thread, and a `nudge` frame comes down the watch's socket — the whole
   point of the module.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi.testclient import TestClient

from models.audio import ToneFlagEvent, TurnLocalEvent, TurnProsody, TurnTextTone
from nudge_policy import NudgePolicy
from server.tests.watch.test_vectors import pcm
from watch import relay
from watch.models import EnrollmentBaseline, VectorEvent, VectorSubscription
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app
from watch.vectors import VectorEngine


def _turn(**overrides) -> TurnLocalEvent:
    base = dict(
        session_id="phone-1", speaker="Speaker A", is_self=True, text="I said I'm fine.",
        start_time=12.5, end_time=14.25, transcript_source="on-device",
    )
    base.update(overrides)
    return TurnLocalEvent(**base)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Module-level registry must never leak between tests."""
    relay._registry.clear()
    yield
    relay._registry.clear()


# ------------------------------------------------------------- pure layer --

def test_detail_carries_phone_provenance():
    ev = _turn(prosody=TurnProsody(rms_dbfs=-14.0), text_tone=TurnTextTone(frustration=80, label="defensive"))
    events = relay.turn_local_to_vector_events(ev, t=3.0, baseline_rms_db=-30.0)
    assert [e.vector for e in events] == ["yelling", "aggressive_tone"]
    assert events[0].detail.startswith("phone turn:") and "16.0 dB" in events[0].detail
    assert "defensive" in events[1].detail and events[1].value == 2.0
    assert all(e.t == 3.0 for e in events)


def test_tone_level_reads_both_sources_and_ignores_unconfident_flags():
    assert relay.tone_level(None, None) == 0
    assert relay.tone_level(TurnTextTone(frustration=None, defensiveness=None)) == 0
    assert relay.tone_level(TurnTextTone(sarcasm=100, sadness=100, warmth=0)) == 0, "only frustration/defensiveness escalate"
    assert relay.tone_level(TurnTextTone(frustration=55)) == 1
    assert relay.tone_level(TurnTextTone(defensiveness=70)) == 2
    assert relay.tone_level(TurnTextTone(frustration=85)) == 3
    flag = ToneFlagEvent(session_id="s", speaker="Speaker A", start_time=0, end_time=1, source="audio",
                         scores={"frustration": 90.0}, label="furious", confidence=0.49)
    assert relay.tone_level(None, flag) == 0
    assert relay.tone_level(None, flag.model_copy(update={"confidence": 0.5})) == 3
    assert relay.tone_level(TurnTextTone(frustration=60), flag.model_copy(update={"confidence": 0.9})) == 3


# ------------------------------------------------------ session + registry --

class _Recorder:
    """Stands in for ws.py's emit closure: records calls and runs a real
    NudgePolicy so the test can see nudges, not just events."""

    def __init__(self, subs=None):
        self.calls: list[tuple[list[VectorEvent], float]] = []
        self.policy = NudgePolicy(subs or [VectorSubscription(vector="yelling"), VectorSubscription(vector="aggressive_tone")])
        self.nudges = []

    async def emit(self, events, t):
        self.calls.append((events, t))
        self.nudges.extend(self.policy.on_events(events, t))


def _session(engine: VectorEngine, recorder: _Recorder, account="alice") -> relay.LiveWatchSession:
    return relay.LiveWatchSession(
        account_id=account, live_session_id="ls-1", engine=engine, emit=recorder.emit,
        loop=asyncio.get_running_loop(),
    )


async def _settle():
    """Let tasks scheduled on this loop run."""
    for _ in range(3):
        await asyncio.sleep(0)


def test_push_turn_local_without_live_watch_is_a_logged_noop(caplog):
    with caplog.at_level(logging.DEBUG, logger="watch.relay"):
        relay.push_turn_local("nobody", _turn(text_tone=TurnTextTone(frustration=99)))
    assert "no live watch session for nobody" in caplog.text


def test_push_turn_local_escalates_live_session_from_tone_alone():
    async def run():
        engine = VectorEngine(EnrollmentBaseline(account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x"))
        engine.t = 7.0
        rec = _Recorder()
        session = _session(engine, rec)
        relay.register_live_session(session)

        relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-30.0), text_tone=TurnTextTone(frustration=78)))
        await _settle()
        assert len(rec.calls) == 1
        events, t = rec.calls[0]
        assert t == 7.0, "runs on the WATCH session's stream clock, not the phone's start_time"
        assert [(e.vector, e.level) for e in events] == [("aggressive_tone", 2)]
        assert [(n.channel, n.level, n.vectors) for n in rec.nudges] == [("A", 2, ["aggressive_tone"])]

    asyncio.run(run())


def test_push_turn_local_ignores_other_and_unknown_speakers():
    async def run():
        engine = VectorEngine(EnrollmentBaseline(account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x"))
        rec = _Recorder()
        relay.register_live_session(_session(engine, rec))
        hostile = dict(prosody=TurnProsody(rms_dbfs=-5.0), text_tone=TurnTextTone(frustration=99, defensiveness=99))
        relay.push_turn_local("alice", _turn(is_self=False, **hostile))
        relay.push_turn_local("alice", _turn(is_self=None, **hostile))
        await _settle()
        assert rec.calls == []
        assert rec.policy.current() == {"A": 0, "B": 0}

    asyncio.run(run())


def test_empty_relay_never_ticks_the_policy():
    """A calm self turn must not become a cooldown tick: the policy's clock
    belongs to the watch's own 1 s windows."""
    async def run():
        engine = VectorEngine(EnrollmentBaseline(account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x"))
        rec = _Recorder()
        relay.register_live_session(_session(engine, rec))
        # Escalate via the watch's own path first (so there is something to decay).
        rec.policy.on_events([VectorEvent(vector="yelling", level=2, t=1.0, value=11.0)], 1.0)
        engine.t = 100.0  # far past cooldown on the stream clock
        relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-30.0), text_tone=TurnTextTone(frustration=5)))
        await _settle()
        assert rec.calls == [], "nothing over threshold -> no emit at all"
        assert rec.policy.current()["A"] == 2, "and therefore no de-escalation from the phone's cadence"

    asyncio.run(run())


def test_running_median_fallback_uses_phone_history_not_watch_history():
    async def run():
        engine = VectorEngine(None)  # not enrolled
        rec = _Recorder()
        session = _session(engine, rec)
        relay.register_live_session(session)

        # First turn: nothing to compare against -> can't yell even at -10 dBFS.
        relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-10.0)))
        await _settle()
        assert rec.calls == []
        assert session.phone_baseline_rms_db() == -10.0, "…but it seeds the phone-side median"
        assert len(engine._rms_db_history) == 0, "the watch engine's own history is untouched (dB path byte-identical)"

        # Three quiet turns pull the median down to conversational level…
        for _ in range(3):
            relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-30.0)))
        await _settle()
        assert rec.calls == []
        # …so a -14 dBFS turn is now 16 over the (-30) median -> yelling 3.
        relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-14.0)))
        await _settle()
        assert len(rec.calls) == 1
        assert [(e.vector, e.level) for e in rec.calls[0][0]] == [("yelling", 3)]

        # Silence-floor turns never enter the history (push_pcm's rule).
        before = list(session._phone_rms_history)
        relay.push_turn_local("alice", _turn(prosody=TurnProsody(rms_dbfs=-60.0)))
        await _settle()
        assert list(session._phone_rms_history) == before

    asyncio.run(run())


def test_unregister_only_removes_the_same_session():
    async def run():
        engine = VectorEngine(None)
        rec = _Recorder()
        older = _session(engine, rec)
        newer = _session(engine, rec)
        relay.register_live_session(older)
        relay.register_live_session(newer)
        assert relay.live_session_for("alice") is newer
        relay.unregister_live_session(older)  # stale teardown must not clobber the live one
        assert relay.live_session_for("alice") is newer
        relay.unregister_live_session(newer)
        assert relay.live_session_for("alice") is None

    asyncio.run(run())


# ------------------------------------------------------------- end to end --

def test_phone_turn_nudges_a_live_watch_socket_from_another_thread():
    """The real thing: watch WS open on the TestClient's loop thread; the
    phone pipeline (this test thread) calls push_turn_local; the watch
    receives vector_event + nudge frames; and the persisted live session
    carries the relayed events with phone provenance."""
    store = MemoryLiveSessionStore()
    asyncio.run(store.put_baseline(EnrollmentBaseline(account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x")))
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e-relay?account=alice") as ws:
        assert relay.live_session_for("alice") is not None, "registered while the socket is open"

        # No watch PCM window before the relay call on purpose: TestClient's
        # send_bytes returns before the app thread has necessarily advanced
        # engine.t, so asserting a post-window clock here would be racy.
        # The "runs on the watch stream clock" rule is pinned deterministically
        # by test_push_turn_local_escalates_live_session_from_tone_alone.
        relay.push_turn_local(
            "alice",
            _turn(prosody=TurnProsody(rms_dbfs=-30.0), text_tone=TurnTextTone(frustration=78, label="frustrated")),
        )
        vector_event = json.loads(ws.receive_text())
        nudge = json.loads(ws.receive_text())
        assert vector_event["type"] == "vector_event"
        assert vector_event["vector"] == "aggressive_tone" and vector_event["level"] == 2
        assert vector_event["t"] == 0.0, "stamped with the watch stream clock (no window yet)"
        assert nudge == {"type": "nudge", "channel": "A", "level": 2, "t": 0.0, "vectors": ["aggressive_tone"]}

        # The watch mic path still works exactly as before alongside it.
        ws.send_bytes(pcm(0.4))
        loud = json.loads(ws.receive_text())
        assert loud["type"] == "vector_event" and loud["vector"] == "yelling"
        louder = json.loads(ws.receive_text())
        assert louder["type"] == "nudge" and louder["level"] == 3, "yelling 3 out-escalates the tone-2 the relay set"

        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved"

    assert relay.live_session_for("alice") is None, "unregistered on the way out"
    ls = asyncio.run(store.get_live_session("e-relay"))
    assert ls is not None
    assert [(e.vector, e.level) for e in ls.vector_events][:1] == [("aggressive_tone", 2)]
    assert ls.vector_events[0].detail.startswith("phone turn tone: frustrated")
    assert ls.nudge_events and ls.nudge_events[0].vectors == ["aggressive_tone"]

    # After the socket is gone, the same call is a no-op, not an error.
    relay.push_turn_local("alice", _turn(text_tone=TurnTextTone(frustration=99)))
