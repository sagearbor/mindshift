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


# ----------------------------------------------------------- valence veto --
# The +6/+10/+14 dB ladder measures AROUSAL, so it fires on 44.5% of HAPPY
# speech (CREMA-D). The veto lets the server's own valence reading suppress a
# loudness nudge that reads as PLEASANT. It must only ever subtract a buzz,
# must leave the tone lane alone, and must be inert unless switched on.
# Rationale + calibration: docs/decisions/2026-09-17-valence-veto.md.

def _audio_flag(valence: float | None, **overrides) -> ToneFlagEvent:
    """A ToneFlagEvent shaped like the one audio_pipeline._enrich_tone builds
    from the dimensional backend — note confidence 0.0, which is what
    tone_id._dims_to_result really emits."""
    scores: dict[str, float] = {"arousal": 0.7, "dominance": 0.6}
    if valence is not None:
        scores["valence"] = valence
    base = dict(session_id="s", speaker="Speaker A", start_time=0, end_time=1,
                source="audio", scores=scores, label="unscored", confidence=0.0)
    base.update(overrides)
    return ToneFlagEvent(**base)


def _loud_turn() -> TurnLocalEvent:
    """+16 dB over baseline — comfortably past the top rung."""
    return _turn(prosody=TurnProsody(rms_dbfs=-14.0))


def _vectors(turn, flag):
    return relay.turn_local_to_vector_events(turn, t=3.0, baseline_rms_db=-30.0, tone_flag=flag)


def test_valence_veto_is_off_unless_the_env_var_says_otherwise(monkeypatch):
    """Production safety: with the flag unset, a loud PLEASANT turn still
    buzzes exactly as it does today. If this fails, shipping changed
    behaviour for every wearer without anyone flipping anything."""
    monkeypatch.delenv(relay.VALENCE_GATE_ENV, raising=False)
    assert relay.valence_gate_enabled() is False
    pleasant = _audio_flag(0.95)
    assert relay.valence_veto(pleasant) == (False, None)
    assert [e.vector for e in _vectors(_loud_turn(), pleasant)] == ["yelling"]
    for value in ("", "off", "0", "false", "no", "dark", "maybe"):
        monkeypatch.setenv(relay.VALENCE_GATE_ENV, value)
        assert relay.valence_gate_enabled() is False, f"{value!r} must not enable the veto"
    for value in ("1", "on", "true", "YES", " On "):
        monkeypatch.setenv(relay.VALENCE_GATE_ENV, value)
        assert relay.valence_gate_enabled() is True, f"{value!r} should enable the veto"


def test_valence_veto_suppresses_a_loud_but_pleasant_turn(monkeypatch):
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    vetoed, valence = relay.valence_veto(_audio_flag(0.80))
    assert vetoed is True and valence == 0.80
    assert _vectors(_loud_turn(), _audio_flag(0.80)) == [], "laughing must not buzz"


def test_valence_veto_lets_a_loud_unpleasant_turn_through(monkeypatch):
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    angry = _audio_flag(0.20)
    assert relay.valence_veto(angry) == (False, 0.20)
    assert [e.vector for e in _vectors(_loud_turn(), angry)] == ["yelling"]


def test_valence_veto_boundary_is_inclusive_of_the_threshold(monkeypatch):
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    at = relay.VALENCE_VETO_MAX
    assert relay.valence_veto(_audio_flag(at))[0] is False, "at the threshold still buzzes"
    assert relay.valence_veto(_audio_flag(at + 0.001))[0] is True


def test_valence_veto_never_silences_the_tone_lane(monkeypatch):
    """The words are the better signal. A pleasant-sounding turn whose TEXT
    reads as hostile must still reach the wrist on aggressive_tone."""
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    turn = _turn(prosody=TurnProsody(rms_dbfs=-14.0),
                 text_tone=TurnTextTone(frustration=90, label="defensive"))
    assert [e.vector for e in _vectors(turn, _audio_flag(0.95))] == ["aggressive_tone"]


def test_valence_veto_applies_at_every_rung_including_the_loudest(monkeypatch):
    """Loud+happy concentrates at the TOP rung (93% of >=+14 dB happy clips),
    so exempting level 3 would give up most of the benefit."""
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    for rms, expected_level in ((-22.0, 1), (-18.0, 2), (-14.0, 3)):
        turn = _turn(prosody=TurnProsody(rms_dbfs=rms))
        assert [e.level for e in _vectors(turn, None)] == [expected_level], "rung sanity"
        assert _vectors(turn, _audio_flag(0.90)) == [], f"level {expected_level} must be vetoable"


def test_valence_veto_ignores_the_confidence_floor(monkeypatch):
    """Regression guard. The dimensional backend reports confidence 0.0 by
    construction, so reusing TONE_FLAG_MIN_CONFIDENCE here would veto nothing
    ever and the feature would look enabled while doing nothing."""
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    assert _audio_flag(0.90).confidence == 0.0
    assert relay.valence_veto(_audio_flag(0.90))[0] is True


@pytest.mark.parametrize("flag,why", [
    (None, "no tone flag at all (MINDSHIFT_TONE_AUDIO below 'on')"),
    (_audio_flag(None), "categorical backend — no valence key"),
    (_audio_flag(0.90, source="text"), "the text lane never carries valence"),
])
def test_valence_veto_fails_open(monkeypatch, flag, why):
    """A missing signal is not evidence that a turn was pleasant. Every
    degraded path must leave today's behaviour untouched."""
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    assert relay.valence_veto(flag)[0] is False, why
    assert [e.vector for e in _vectors(_loud_turn(), flag)] == ["yelling"], why


def test_valence_veto_survives_a_junk_score(monkeypatch):
    monkeypatch.setenv(relay.VALENCE_GATE_ENV, "on")
    flag = _audio_flag(0.9)
    flag.scores["valence"] = float("nan")
    # NaN compares False against everything — must not crash, must not veto.
    assert relay.valence_veto(flag)[0] is False
    assert [e.vector for e in _vectors(_loud_turn(), flag)] == ["yelling"]


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


# ------------------------------------------------------- positives (praise) --
#
# Before push_positive existed the relay carried VectorEvents only, so a wearer
# with the phone in a pocket felt every complaint and no praise: the wrist was a
# pure complaint channel. These cases pin the fix AND its limits — praise must
# reach the wrist without ever entering the escalation machinery.

class _PositiveRecorder(_Recorder):
    def __init__(self, subs=None):
        super().__init__(subs)
        self.positives: list[tuple[str, float]] = []

    async def send_positive(self, code, t):
        self.positives.append((code, t))


def _positive_session(engine, recorder, account="alice"):
    return relay.LiveWatchSession(
        account_id=account, live_session_id="ls-1", engine=engine, emit=recorder.emit,
        send_positive=recorder.send_positive, loop=asyncio.get_running_loop(),
    )


def test_push_positive_reaches_the_wrist_without_touching_the_policy():
    async def run():
        engine = VectorEngine(None)
        rec = _PositiveRecorder()
        relay.register_live_session(_positive_session(engine, rec))

        assert relay.push_positive("alice", "E", 41.5) is True
        await _settle()
        assert rec.positives == [("E", 41.5)]
        # The whole point: no vector event, no nudge, no level.
        assert rec.calls == []
        assert rec.nudges == []
        assert rec.policy.current() == {"A": 0, "B": 0}

    asyncio.run(run())


def test_push_positive_refuses_anything_that_is_not_praise():
    """A bad code must die at the relay rather than travel. H is an ALERT
    (relaying it here would launder a complaint as praise) and K is silent by
    contract (buzzing to say nothing happened is the definition of a nag)."""
    async def run():
        rec = _PositiveRecorder()
        relay.register_live_session(_positive_session(VectorEngine(None), rec))
        for code in ("H", "C", "A", "P", "K", "", "zzz", "e"):
            assert relay.push_positive("alice", code, 1.0) is False, code
        await _settle()
        assert rec.positives == []

    asyncio.run(run())


def test_push_positive_without_a_live_watch_is_a_noop():
    assert relay.push_positive("nobody", "E", 1.0) is False


def test_push_positive_on_a_session_that_cannot_send_is_a_noop():
    """A socket opened before this path existed has no send_positive. It must
    report False, not fall back to emit — falling back would escalate."""
    async def run():
        rec = _PositiveRecorder()
        relay.register_live_session(_session(VectorEngine(None), rec))  # no send_positive
        assert relay.push_positive("alice", "E", 1.0) is False
        await _settle()
        assert rec.calls == [] and rec.positives == []

    asyncio.run(run())


def test_positive_frame_reaches_a_live_watch_socket_from_another_thread():
    """End to end: watch WS open on the app's loop; the phone pipeline (this
    thread) delivers a positive; a `positive` frame comes down the wire and
    NOTHING is persisted as a nudge."""
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/e-praise?account=alice") as ws:
        assert relay.push_positive("alice", "R", 12.0) is True
        frame = json.loads(ws.receive_text())
        assert frame == {"type": "positive", "code": "R", "t": 12.0}

        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "live_session_saved"

    ls = asyncio.run(store.get_live_session("e-praise"))
    assert ls is not None
    assert ls.vector_events == [] and ls.nudge_events == [], "praise is not an escalation"


# ------------------------------------------------- companion heart rate --
#
# The wearer's own coached sessions are the ONLY place heart rate and speech
# are ever observed together — no public emotion corpus carries both — so this
# is the only data that can ever test whether HR adds anything over loudness
# (docs/plans/2026-09-10-heat-rubric-and-buzz-dose.md). These cases pin that it
# is actually collected, and that collecting it did not resurrect the junk-doc
# problem the companion path exists to avoid.

def _companion_ws(client, sid="hr-1", account="alice"):
    return client.websocket_connect(f"/ws/live-session/{sid}?account={account}")


def test_companion_persists_its_heart_rate_under_a_derived_id():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with _companion_ws(client) as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        for bpm in (72.0, 74.0, 130.0):
            ws.send_text(json.dumps({"type": "hr", "bpm": bpm, "t": 1.0}))
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        while saved["type"] != "live_session_saved":
            saved = json.loads(ws.receive_text())
    assert saved["status"] == "companion_hr"

    # NOT under the socket's own id: a companion reuses one id per day and its
    # socket drops often, so writing there would have each reconnect overwrite
    # the last and lose most of the day.
    assert asyncio.run(store.get_live_session("hr-1")) is None
    docs = [d for d in asyncio.run(_all_sessions(store)) if d.id.startswith("hr-1-hr-")]
    assert len(docs) == 1
    doc = docs[0]
    assert doc.status == "companion_hr"
    assert doc.series["hr_bpm"] == [72.0, 74.0, 130.0], "every sample, not just the spike"
    assert len(doc.series["hr_t"]) == 3, "each stamped on the server stream clock"
    assert doc.pcm_b64 == "", "a companion never carries audio"


def test_companion_with_no_heart_rate_still_persists_nothing():
    """The junk-doc rule survives. An all-day wrist socket that collected
    nothing must not mint a document every time it drops."""
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with _companion_ws(client, sid="hr-empty") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        ws.send_text(json.dumps({"type": "end"}))
        saved = json.loads(ws.receive_text())
        while saved["type"] != "live_session_saved":
            saved = json.loads(ws.receive_text())
    assert saved["status"] == "companion"
    assert asyncio.run(_all_sessions(store)) == []


def test_companion_heart_rate_survives_an_abrupt_disconnect():
    """The path most companion sockets actually take. A clean "end" is the
    exception — screen-off churn and pocket dead zones are the rule — so
    dropping HR here would have lost most of what the feature collects."""
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with _companion_ws(client, sid="hr-drop") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        ws.send_text(json.dumps({"type": "hr", "bpm": 99.0, "t": 1.0}))
        # no "end" — just go away
    docs = [d for d in asyncio.run(_all_sessions(store)) if d.id.startswith("hr-drop-hr-")]
    assert len(docs) == 1 and docs[0].series["hr_bpm"] == [99.0]
    assert docs[0].status == "companion_hr", "never not_analyzed — there was no audio to analyse"


def test_a_normal_mic_session_also_keeps_the_raw_series():
    """Not companion-specific: the negatives matter on the mic path too. An
    hr_spike VectorEvent is only emitted over +15 bpm, so the vector log alone
    is the positives and cannot be evaluated against anything."""
    store = MemoryLiveSessionStore()
    asyncio.run(store.put_baseline(EnrollmentBaseline(
        account_id="alice", rms_db=-30.0, f0_median=120.0, updated_at="x")))
    client = TestClient(create_watch_test_app(store=store, allow_legacy=True))
    with client.websocket_connect("/ws/live-session/hr-mic?account=alice") as ws:
        ws.send_bytes(pcm(0.1))
        ws.send_text(json.dumps({"type": "hr", "bpm": 70.0, "t": 1.0}))
        ws.send_text(json.dumps({"type": "end"}))
        while json.loads(ws.receive_text())["type"] != "live_session_saved":
            pass
    doc = asyncio.run(store.get_live_session("hr-mic"))
    assert doc is not None and doc.series["hr_bpm"] == [70.0]
    assert not any(e.vector == "hr_spike" for e in doc.vector_events), \
        "70 bpm is not a spike — which is exactly the sample the series exists to keep"


async def _all_sessions(store) -> list:
    """Every persisted live session, however the store spells its internals."""
    if hasattr(store, "_live_sessions"):
        return list(store._live_sessions.values())
    raise AssertionError("MemoryLiveSessionStore shape changed — update this helper")
