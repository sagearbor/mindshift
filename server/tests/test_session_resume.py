"""Session resume (server/session_resume.py) — a live session or in-app call
survives a 10–30 s network drop.

Three failures a real dropped call used to cause, each pinned here:

* the turns the phone finalized while the socket was down never reached the
  cloud coach (or the other members of a call) — now they are buffered on the
  phone and flushed after ``resume``, de-duplicated server-side by
  ``turn_uid`` so a turn that was in flight when the socket died is processed
  exactly once;
* the server's session clock restarted at 0 on the replacement socket while
  the phone's capture clock kept counting — now ``last_local_time``
  re-anchors the PCM ring buffer (and a rebound call participant KEEPS the
  clock offset PR #167 resets for a genuinely new socket);
* nothing replayed the merged call transcript the reconnecting phone missed —
  now ``since_seq`` does, bounded, render-only.

Providers are the suite's usual doubles on ``app.state``; auth is conftest's
keyless harness.
"""

from __future__ import annotations

import asyncio
import json
import types

import pytest
from starlette.testclient import TestClient

import audio_pipeline
import calls
import main
import session_resume
from audio_pipeline import PcmRingBuffer
from main import app

from tests.test_audio_pipeline import (  # noqa: E402 — the DI doubles/helpers
    MOCK_LLM_JSON, NUDGE_LLM_JSON, SELF_DOC, FakeSpeakerId, FakeTTS,
    FakeVoiceprintStore, StoppableTranscriber, _clear_overrides, open_ws,
    recv_until,
)
from tests.test_calls import FakeStore  # noqa: E402 — the calls suite's store double

SID = "resume-session"
UID = "test-user"          # conftest maps FAKE_ID_TOKEN → this uid
OTHER_TOKEN = "tok-user-a"
OTHER_UID = "user-a"


# ---------------------------------------------------------------------------
# The pure state: what survives a dropped socket
# ---------------------------------------------------------------------------

class TestResumeState:
    def test_a_repeat_is_recognised_a_new_id_is_not(self):
        state = session_resume.ResumeState(session_id=SID, uid=UID)
        assert state.remember("t-1") is True
        assert state.remember("t-2") is True
        assert state.remember("t-1") is False          # the re-sent one
        assert state.known_turns == 2
        assert state.turns_seen == 2 and state.duplicates == 1

    def test_a_turn_without_an_id_is_always_new(self):
        """A client that predates turn_uid must never have a turn silently
        swallowed — it just gets no de-duplication."""
        state = session_resume.ResumeState(session_id=SID, uid=UID)
        assert [state.remember(None) for _ in range(3)] == [True, True, True]
        assert state.known_turns == 0 and state.turns_seen == 3

    def test_remembered_ids_are_bounded_oldest_first(self, monkeypatch):
        monkeypatch.setattr(session_resume, "RESUME_SEEN_UIDS_MAX", 3)
        state = session_resume.ResumeState(session_id=SID, uid=UID)
        for i in range(5):
            state.remember(f"t-{i}")
        assert state.known_turns == 3
        assert not state.seen("t-0") and not state.seen("t-1")
        assert state.seen("t-2") and state.seen("t-4")
        # An id pushed out is "new" again — the honest consequence of the
        # bound, and unreachable in practice: a client's own offline queue is
        # smaller and is flushed in one go.
        assert state.remember("t-0") is True

    def test_expiry_is_measured_from_the_last_touch(self):
        state = session_resume.ResumeState(session_id=SID, uid=UID)
        state.touched_at = -1000.0
        assert state.expired()
        state.remember("t-1")     # touches
        assert not state.expired()


class TestResumeRegistry:
    @pytest.fixture(autouse=True)
    def clean(self):
        session_resume.registry.reset()
        yield
        session_resume.registry.reset()

    def test_acquire_returns_the_same_state_for_the_same_session(self):
        first = session_resume.registry.acquire(SID, UID)
        first.remember("t-1")
        assert session_resume.registry.acquire(SID, UID) is first
        assert session_resume.registry.acquire(SID, UID).seen("t-1")

    def test_state_is_never_handed_to_another_account(self):
        """A session id is client-chosen, so two accounts can pick the same
        one. The second gets an isolated state — it can neither read the
        first's turn ids nor clobber them."""
        mine = session_resume.registry.acquire(SID, UID)
        mine.remember("t-1")
        assert session_resume.registry.get(SID, "someone-else") is None
        theirs, known = session_resume.registry.attach(SID, "someone-else")
        assert known is False
        assert not theirs.seen("t-1") and theirs.uid == "someone-else"
        assert session_resume.registry.get(SID, UID) is mine     # untouched

    def test_a_second_connection_is_what_makes_a_resume_findable(self):
        _, known = session_resume.registry.attach(SID, UID)
        assert known is False                       # first socket of a session
        _, known = session_resume.registry.attach(SID, UID)
        assert known is True                        # the reconnect

    def test_expired_state_is_swept_and_a_resume_starts_fresh(self):
        state = session_resume.registry.acquire(SID, UID)
        state.remember("t-1")
        state.touched_at = -1000.0
        assert session_resume.registry.get(SID, UID) is None
        assert len(session_resume.registry) == 0

    def test_registry_is_bounded_evicting_the_least_recently_touched(self, monkeypatch):
        monkeypatch.setattr(session_resume, "RESUME_MAX_SESSIONS", 2)
        a = session_resume.registry.acquire("s-a", UID)
        session_resume.registry.acquire("s-b", UID)
        a.touched_at -= 10.0                       # least recently used
        session_resume.registry.acquire("s-c", UID)
        assert len(session_resume.registry) == 2
        assert session_resume.registry.get("s-a", UID) is None
        assert session_resume.registry.get("s-c", UID) is not None

    def test_forget_drops_one_session(self):
        session_resume.registry.acquire(SID, UID)
        assert session_resume.registry.forget(SID) is True
        assert session_resume.registry.forget(SID) is False


class TestFrameValidation:
    def test_turn_uid_shape_is_bounded(self):
        assert session_resume.clean_turn_uid("t-abc_1:2.3") == "t-abc_1:2.3"
        assert session_resume.clean_turn_uid("  t-1  ") == "t-1"
        assert session_resume.clean_turn_uid("") is None
        assert session_resume.clean_turn_uid("bad id") is None       # space
        assert session_resume.clean_turn_uid("bad\nid") is None      # control char
        assert session_resume.clean_turn_uid("x" * 65) is None
        assert session_resume.clean_turn_uid(7) is None

    def test_since_seq_is_a_non_negative_int(self):
        assert session_resume.clean_since_seq(12) == 12
        assert session_resume.clean_since_seq(0) == 0
        assert session_resume.clean_since_seq(-3) == 0
        assert session_resume.clean_since_seq(True) == 0     # bool is not a seq
        assert session_resume.clean_since_seq("12") == 0
        assert session_resume.clean_since_seq(None) == 0

    def test_last_local_time_rejects_what_cannot_address_the_ring_buffer(self):
        assert session_resume.clean_local_time(30) == 30.0
        assert session_resume.clean_local_time(30.25) == 30.25
        assert session_resume.clean_local_time(0) == 0.0
        assert session_resume.clean_local_time(-1) is None
        assert session_resume.clean_local_time(float("inf")) is None
        assert session_resume.clean_local_time(float("nan")) is None
        assert session_resume.clean_local_time(session_resume.LAST_LOCAL_TIME_MAX_S + 1) is None
        assert session_resume.clean_local_time("30") is None


# ---------------------------------------------------------------------------
# The clock: re-anchoring the ring buffer
# ---------------------------------------------------------------------------

class TestRingBufferRebase:
    def test_rebase_moves_the_origin_so_later_audio_lands_at_the_phones_time(self):
        ring = PcmRingBuffer(seconds=10.0, sample_rate=100)
        ring.append(b"\x01\x02" * 100)             # 1 s at [0, 1)
        assert ring.seconds_received == 1.0
        ring.rebase(30.0)                          # the drop cost 29 s
        assert ring.seconds_received == 30.0
        ring.append(b"\x03\x04" * 150)             # 1.5 s at [30, 31.5)
        assert ring.seconds_received == 31.5
        assert ring.slice(30.0, 31.5) == b"\x03\x04" * 150
        # The outage is a HOLE, not a shift: the pre-drop audio is gone and a
        # slice over the gap comes back empty rather than wrong.
        assert ring.slice(0.0, 1.0) == b""
        assert ring.slice(20.0, 25.0) == b""

    def test_rebase_never_goes_backwards(self):
        """A client may say the timeline moved on, never rewrite audio the
        server already placed."""
        ring = PcmRingBuffer(seconds=10.0, sample_rate=100)
        ring.append(b"\x01\x02" * 500)             # 5 s
        ring.rebase(2.0)
        assert ring.seconds_received == 5.0
        assert ring.slice(0.0, 5.0) == b"\x01\x02" * 500


# ---------------------------------------------------------------------------
# The merged call transcript: replay, de-duplication, the clock offset
# ---------------------------------------------------------------------------

class RecordingEndpoint(calls.CallEndpoint):
    """A member's socket, reduced to what the call touches."""

    def __init__(self, uid: str) -> None:
        self.uid = uid
        self.session_id = f"sess-{uid}"
        self.sent: list[dict] = []
        self.remote: list[dict] = []
        self.detached = 0

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def on_remote_turn(self, turn: dict, *, display_name: str) -> None:
        self.remote.append({**turn, "display_name": display_name})

    def set_peer_name(self, label: str, display_name: str) -> None:
        pass

    def detach(self) -> None:
        self.detached += 1


def _make_call() -> calls.Call:
    call = calls.registry.create(UID, host_email="host@example.test")
    calls.registry.join(call, OTHER_UID, join_code=call.join_code, email="peer@example.test")
    return call


def _turn(sid: str, text: str, *, start: float, end: float, uid: str | None = None) -> dict:
    return {
        "type": "turn_local", "session_id": sid, "speaker": "Speaker Q", "text": text,
        "start_time": start, "end_time": end, "transcript_source": "on-device",
        **({"turn_uid": uid} if uid else {}),
    }


class TestCallReplayAndDedupe:
    @pytest.fixture(autouse=True)
    def clean(self):
        calls.registry.reset()
        yield
        calls.registry.reset()

    def test_turns_since_is_what_the_viewer_missed(self):
        call = _make_call()
        host, peer = RecordingEndpoint(UID), RecordingEndpoint(OTHER_UID)

        async def run():
            await call.bind(UID, host)
            await call.bind(OTHER_UID, peer)
            await call.push_turn(UID, _turn("a", "one", start=0, end=1))
            await call.push_turn(OTHER_UID, _turn("b", "two", start=0, end=1))
            await call.push_turn(UID, _turn("a", "three", start=2, end=3))

        asyncio.run(run())
        # The peer missed everything after seq 1: the host's turns only —
        # its own words were never delivered to it in the first place.
        rows, dropped = call.turns_since(OTHER_UID, 1)
        assert [r["text"] for r in rows] == ["three"] and dropped == 0
        rows, _ = call.turns_since(OTHER_UID, 0)
        assert [r["text"] for r in rows] == ["one", "three"]
        # Caught up: nothing to replay.
        assert call.turns_since(OTHER_UID, 3) == ([], 0)

    def test_replay_is_bounded_dropping_the_oldest(self):
        call = _make_call()
        host, peer = RecordingEndpoint(UID), RecordingEndpoint(OTHER_UID)

        async def run():
            await call.bind(UID, host)
            await call.bind(OTHER_UID, peer)
            for i in range(10):
                await call.push_turn(UID, _turn("a", f"turn {i}", start=i, end=i + 1))

        asyncio.run(run())
        rows, dropped = call.turns_since(OTHER_UID, 0, limit=4)
        assert [r["text"] for r in rows] == ["turn 6", "turn 7", "turn 8", "turn 9"]
        assert dropped == 6

    def test_a_re_sent_turn_is_ignored_not_merged_twice(self):
        call = _make_call()
        host, peer = RecordingEndpoint(UID), RecordingEndpoint(OTHER_UID)

        async def run():
            await call.bind(UID, host)
            await call.bind(OTHER_UID, peer)
            first = await call.push_turn(UID, _turn("a", "hello", start=0, end=1, uid="t-1"))
            again = await call.push_turn(UID, _turn("a", "hello", start=0, end=1, uid="t-1"))
            return first, again

        first, again = asyncio.run(run())
        assert first["seq"] == again["seq"] == 1
        assert len(call.turns) == 1
        # …and the peer was delivered to exactly once.
        assert [t["text"] for t in peer.remote] == ["hello"]

    def test_a_resumed_bind_keeps_the_clock_offset(self, monkeypatch):
        """PR #167 re-fixes a rebound member's sender→call-timeline offset
        because a new socket meant a new capture clock. A RESUMED socket is
        the same clock — re-fixing it would land the rest of its turns in the
        past."""
        call = _make_call()
        host, peer = RecordingEndpoint(UID), RecordingEndpoint(OTHER_UID)

        async def run():
            await call.bind(UID, host)
            await call.bind(OTHER_UID, peer)
            monkeypatch.setattr(call, "_t0", call._t0 - 20.0)   # 20 s into the call
            await call.push_turn(UID, _turn("a", "before", start=5, end=6))
            fixed = call.participant(UID).offset_s
            # Drop + reconnect: same phone, same capture clock.
            resumed = RecordingEndpoint(UID)
            await call.bind(UID, resumed, resume=True)
            assert call.participant(UID).offset_s == fixed
            await call.push_turn(UID, _turn("a", "after", start=9, end=10))
            # …and a genuinely NEW socket still re-fixes it.
            await call.bind(UID, RecordingEndpoint(UID))
            assert call.participant(UID).offset_s is None
            return fixed

        fixed = asyncio.run(run())
        before, after = call.turns[0], call.turns[1]
        assert before["start_time"] == round(5 + fixed, 3)
        assert after["start_time"] == round(9 + fixed, 3)
        assert after["start_time"] > before["end_time"]   # the call moved forward


# ---------------------------------------------------------------------------
# The wire: a solo session drops and resumes
# ---------------------------------------------------------------------------

class RoutingLLM:
    """Plain object (no MagicMock → the non-streaming ``complete`` path):
    a nudge for the speaker's own turns, suggestions for everyone else's.
    ``prompts`` is what reached the model, which is how these tests prove a
    replayed turn was NOT coached a second time."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, **_) -> str:
        self.prompts.append(user)
        if "real-time delivery coach" in system:
            return NUDGE_LLM_JSON
        return MOCK_LLM_JSON


@pytest.fixture
def resume_env(monkeypatch):
    """Local-first providers + a clean resume registry per test."""
    _clear_overrides()
    session_resume.registry.reset()
    calls.registry.reset()
    monkeypatch.setattr(audio_pipeline, "tone_id", None)
    monkeypatch.setattr(audio_pipeline, "speaker_id", None)
    monkeypatch.setattr(audio_pipeline, "watch_relay", None)
    monkeypatch.setattr(audio_pipeline, "SLICE_GRACE_S", 0.0)
    llm = RoutingLLM()
    app.state.llm_client = llm
    app.state.transcriber_factory = lambda: StoppableTranscriber()
    app.state.tts_client = FakeTTS()
    yield types.SimpleNamespace(client=TestClient(app), llm=llm)
    _clear_overrides()
    session_resume.registry.reset()
    calls.registry.reset()


def _resume(ws, *, since_seq: int = 0, last_local_time: float | None = None,
            session_id: str = SID) -> dict:
    ws.send_text(json.dumps({
        "type": "resume", "session_id": session_id, "since_seq": since_seq,
        "last_local_time": last_local_time,
    }))
    ack, _ = recv_until(ws, lambda m: m.get("type") == "resume_ack")
    return ack


class TestSoloSessionResume:
    def test_a_re_sent_turn_is_coached_exactly_once_across_the_drop(self, resume_env):
        """The drop's real shape: turn 1 got through, turn 2 was in flight
        when the socket died (so the phone re-sends it), turn 3 was finalized
        while it was down. The coach must see 2 and 3 exactly once each."""
        client = resume_env.client
        with open_ws(client, f"/ws/session/{SID}") as ws:
            ws.send_text(json.dumps(_turn(SID, "one", start=0, end=1, uid="t-1")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            # …and turn 2 arrives, but its answer never makes it back.
            ws.send_text(json.dumps(_turn(SID, "two", start=1, end=2, uid="t-2")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
        # The socket died. The phone reconnects and replays what it is not
        # sure the server got, plus what it buffered while down.
        with open_ws(client, f"/ws/session/{SID}") as ws:
            ack = _resume(ws, last_local_time=12.0)
            assert ack["resumed"] is True and ack["known_turns"] == 2
            ws.send_text(json.dumps(_turn(SID, "two", start=1, end=2, uid="t-2")))
            ws.send_text(json.dumps(_turn(SID, "three", start=12, end=13, uid="t-3")))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "suggestion"
            assert resp["utterance_text"] == "three"      # "two" was ignored
            ws.send_text(json.dumps({"type": "stop"}))
            recv_until(ws, lambda m: m.get("type") == "session_complete")
        assert [p for p in resume_env.llm.prompts if "two" in p] == [
            p for p in resume_env.llm.prompts if "two" in p
        ][:1], "the re-sent turn reached the LLM twice"
        assert sum('"three"' in p for p in resume_env.llm.prompts) == 1

    def test_a_turn_without_an_id_is_never_swallowed(self, resume_env):
        """No turn_uid = no de-duplication, not a dropped turn."""
        client = resume_env.client
        with open_ws(client, f"/ws/session/{SID}") as ws:
            ws.send_text(json.dumps(_turn(SID, "hello", start=0, end=1)))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
        with open_ws(client, f"/ws/session/{SID}") as ws:
            _resume(ws)
            ws.send_text(json.dumps(_turn(SID, "hello", start=0, end=1)))
            assert json.loads(ws.receive_text())["type"] == "suggestion"

    def test_a_solo_resume_has_nothing_to_replay(self, resume_env):
        """The server keeps no transcript for a solo session (the phone holds
        it), so resume is the clock and the de-duplication — and says so
        instead of inventing a replay."""
        with open_ws(resume_env.client, f"/ws/session/{SID}") as ws:
            ack = _resume(ws, since_seq=5, last_local_time=3.0)
            assert ack == {
                "type": "resume_ack", "session_id": SID, "resumed": False,
                "since_seq": 5, "last_local_time": 3.0, "known_turns": 0,
            }
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_resume_for_an_unknown_session_is_answered_honestly(self, resume_env):
        """A server restart (or a TTL sweep) loses the state. The client is
        told ``resumed: false`` rather than left to guess; the session still
        works, and the clock is still re-anchored."""
        with open_ws(resume_env.client, f"/ws/session/{SID}") as ws:
            ack = _resume(ws, last_local_time=8.0)
            assert ack["resumed"] is False and ack["known_turns"] == 0
            assert ack["last_local_time"] == 8.0
            ws.send_text(json.dumps(_turn(SID, "still works", start=8, end=9, uid="t-9")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"

    def test_state_from_another_account_is_never_adopted(self, resume_env):
        client = resume_env.client
        with open_ws(client, f"/ws/session/{SID}") as ws:
            ws.send_text(json.dumps(_turn(SID, "mine", start=0, end=1, uid="t-1")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
        with open_ws(client, f"/ws/session/{SID}", token=OTHER_TOKEN) as ws:
            ack = _resume(ws)
            assert ack["resumed"] is False and ack["known_turns"] == 0
            # …and the other account's turn id is NOT de-duplicated away.
            ws.send_text(json.dumps(_turn(SID, "mine", start=0, end=1, uid="t-1")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"

    def test_a_graceful_stop_leaves_nothing_to_resume(self, resume_env):
        with open_ws(resume_env.client, f"/ws/session/{SID}") as ws:
            ws.send_text(json.dumps(_turn(SID, "one", start=0, end=1, uid="t-1")))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_text(json.dumps({"type": "stop"}))
            recv_until(ws, lambda m: m.get("type") == "session_complete")
        assert session_resume.registry.get(SID, UID) is None

    def test_a_mismatched_session_id_is_rejected(self, resume_env):
        with open_ws(resume_env.client, f"/ws/session/{SID}") as ws:
            ws.send_text(json.dumps({"type": "resume", "session_id": "another"}))
            assert json.loads(ws.receive_text())["error"] == "resume session_id mismatch"
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"


class TestResumedClockAnchorsEnrichment:
    """The re-anchor is not bookkeeping: without it every turn the phone
    reports after a drop addresses audio the ring buffer does not hold, and
    the server-side enrichment silently stops producing anything."""

    def _run(self, client, *, resume: bool) -> list[dict]:
        """One turn reported at 30 s on the phone's clock, with only the last
        1.5 s of its audio on this (replacement) socket. Drained through a
        graceful stop, which waits for in-flight enrichment — so "no verdict"
        is a real absence, not a race."""
        frame = b"\x01\x00" * 1600                       # 100 ms of PCM
        with open_ws(client, f"/ws/session/{SID}") as ws:
            if resume:
                _resume(ws, last_local_time=30.0)
            for _ in range(15):
                ws.send_bytes(frame)
            ws.send_text(json.dumps(
                _turn(SID, "after the drop", start=30.0, end=31.4, uid="t-r")
            ))
            ws.send_text(json.dumps({"type": "stop"}))
            _, seen = recv_until(
                ws, lambda m: m.get("type") == "session_complete", limit=12,
            )
            return seen

    def test_a_resumed_session_still_recovers_the_turns_audio(self, resume_env, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "speaker_id", FakeSpeakerId())
        app.state.recordings_store = FakeVoiceprintStore([SELF_DOC])
        seen = self._run(resume_env.client, resume=True)
        identity = [e for e in seen if e.get("type") == "speaker_identity"]
        assert identity and identity[0]["person_id"] == "self"

    def test_without_the_re_anchor_the_same_turn_recovers_nothing(self, resume_env, monkeypatch):
        """The bug this fixes, pinned: same frames, same turn, no resume —
        the slice is empty and there is no verdict at all."""
        monkeypatch.setattr(audio_pipeline, "speaker_id", FakeSpeakerId())
        app.state.recordings_store = FakeVoiceprintStore([SELF_DOC])
        seen = self._run(resume_env.client, resume=False)
        assert not [e for e in seen if e.get("type") == "speaker_identity"]


# ---------------------------------------------------------------------------
# The wire: a call member drops and resumes
# ---------------------------------------------------------------------------

@pytest.fixture
def call_env(resume_env, monkeypatch):
    monkeypatch.setattr(calls, "ANALYZE_ON_END", False)
    monkeypatch.setattr(calls, "REFLECT_ON_END", False)
    monkeypatch.setattr(main, "resolve_uid_by_email", lambda e: None)
    monkeypatch.setattr(main, "resolve_email_by_uid", lambda u: f"{u}@example.test")
    app.state.recordings_store = FakeStore()
    from anyio.from_thread import start_blocking_portal
    client = TestClient(app)
    with start_blocking_portal(**client.async_backend) as portal:
        client.portal = portal
        yield types.SimpleNamespace(client=client, llm=resume_env.llm)


class TestCallSessionResume:
    def test_a_member_that_drops_gets_the_missed_turns_replayed(self, call_env):
        """The whole story on the wire: the peer's socket dies, the host keeps
        talking, the peer comes back, resumes, and its screen is caught up —
        without those turns being coached a second time for anyone."""
        client = call_env.client
        call = calls.registry.create(UID, host_email="host@example.test")
        calls.registry.join(call, OTHER_UID, join_code=call.join_code, email="peer@example.test")
        call_id = call.call_id

        with open_ws(client, "/ws/session/host-sess") as host:
            host.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
            recv_until(host, lambda m: m.get("type") == "call_state")

            with open_ws(client, "/ws/session/peer-sess", token=OTHER_TOKEN) as peer:
                peer.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
                recv_until(peer, lambda m: m.get("type") == "call_state")
                host.send_text(json.dumps(
                    _turn("host-sess", "before the drop", start=0, end=1, uid="h-1")))
                seen, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
                assert seen["text"] == "before the drop" and seen["seq"] == 1
                assert "replay" not in seen          # the live wire is unchanged
            # …the peer's network dies here. The host talks on, and is
            # coached on its own turns as usual (drained here so the LLM is
            # quiet before the reconnect below).
            host.send_text(json.dumps(
                _turn("host-sess", "while you were gone", start=2, end=3, uid="h-2")))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            host.send_text(json.dumps(
                _turn("host-sess", "still here?", start=4, end=5, uid="h-3")))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            before_replay = list(call_env.llm.prompts)
            # The peer comes back on a new socket and resumes at what it had.
            with open_ws(client, "/ws/session/peer-sess", token=OTHER_TOKEN) as peer:
                ack = _resume(ws=peer, since_seq=1, last_local_time=6.0,
                              session_id="peer-sess")
                # Same session id, same account: the server remembers it.
                assert ack["resumed"] is True
                peer.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
                summary, _ = recv_until(peer, lambda m: m.get("type") == "resume_replay")
                assert summary["replayed"] == 2 and summary["dropped"] == 0
                assert summary["since_seq"] == 1 and summary["call_id"] == call_id
                replayed = [json.loads(peer.receive_text()) for _ in range(2)]
                assert [t["text"] for t in replayed] == ["while you were gone", "still here?"]
                assert all(t["type"] == "transcript" and t["replay"] is True for t in replayed)
                assert [t["seq"] for t in replayed] == [2, 3]
                # Replay RENDERS; it does not re-coach. A turn from twenty
                # seconds ago is history, and the coach's answer to it would
                # answer a question the conversation has already moved past.
                assert call_env.llm.prompts == before_replay
                # The peer is caught up and coached normally from here on.
                host.send_text(json.dumps(
                    _turn("host-sess", "and now?", start=6, end=7, uid="h-4")))
                live, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
                assert live["text"] == "and now?" and "replay" not in live
                coached, _ = recv_until(peer, lambda m: m.get("type") == "suggestion")
                assert coached["utterance_text"] == "and now?"

    def test_the_turns_a_member_buffered_offline_are_merged_on_resume(self, call_env):
        client = call_env.client
        call = calls.registry.create(UID, host_email="host@example.test")
        calls.registry.join(call, OTHER_UID, join_code=call.join_code, email="peer@example.test")
        call_id = call.call_id

        with open_ws(client, "/ws/session/host-sess") as host:
            host.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
            recv_until(host, lambda m: m.get("type") == "call_state")
            with open_ws(client, "/ws/session/peer-sess", token=OTHER_TOKEN) as peer:
                peer.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
                recv_until(peer, lambda m: m.get("type") == "call_state")
                peer.send_text(json.dumps(
                    _turn("peer-sess", "I said this", start=0, end=1, uid="p-1")))
                got, _ = recv_until(host, lambda m: m.get("type") == "transcript")
                assert got["text"] == "I said this"
            # Dropped mid-flush: the phone is not sure p-1 landed, and it
            # finalized p-2 while it was down. It sends BOTH on resume.
            with open_ws(client, "/ws/session/peer-sess", token=OTHER_TOKEN) as peer:
                _resume(ws=peer, since_seq=1, last_local_time=4.0, session_id="peer-sess")
                peer.send_text(json.dumps({"type": "call_join", "call_id": call_id}))
                recv_until(peer, lambda m: m.get("type") == "resume_replay")
                peer.send_text(json.dumps(
                    _turn("peer-sess", "I said this", start=0, end=1, uid="p-1")))
                peer.send_text(json.dumps(
                    _turn("peer-sess", "and this while offline", start=2, end=3, uid="p-2")))
                got, _ = recv_until(host, lambda m: m.get("type") == "transcript", limit=8)
                assert got["text"] == "and this while offline"
        # One row for the re-sent turn, one for the buffered one.
        assert [t["text"] for t in call.turns] == ["I said this", "and this while offline"]
