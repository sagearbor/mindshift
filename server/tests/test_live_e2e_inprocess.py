"""scripts/live_e2e.py driven in-process, so the phone-shaped end-to-end
client can't rot.

The SAME functions the CLI uses (``live_e2e.run_e2e`` and everything under
it: the Firebase-token config frame, PCM streaming on a scaled clock,
``turn_local`` per scene turn, ``POST /sessions/live`` → poll → reflect →
detail → growth → share → the therapist's ``GET /sessions``) run against a
REAL uvicorn on 127.0.0.1 serving ``main.app`` — real WebSocket and HTTP
over the loopback, not a TestClient — with the external providers replaced
at their existing DI seams:

* transcriber / TTS / LLM via ``app.state`` (the audio_pipeline seams),
* the recordings store via ``app.state.recordings_store`` (an in-memory
  fake with the share + voiceprint surface routers/sessions.py and
  main.py read),
* audio tone / speaker-ID at the ``audio_pipeline.tone_id`` /
  ``audio_pipeline.speaker_id`` module attributes (label-echo doubles that
  exercise the tone_flag / speaker_identity WIRE path — the real ECAPA path
  on these scenes is pinned by test_diarize_scenes.py and exercised by the
  production run),
* auth through the suite's existing keyless harness (conftest's fake
  ``verify_id_token`` for the WS handshake + the ``X-Test-Uid`` dependency
  override for REST) — nothing is weakened; the client just presents the
  suite's fake token/uid the way it would present a real one.

Two scenes: the 2-voice self-escalation arc and the rapid 3-voice family
scene. The assertion is the script's own: no ❌ in the report.

``--with-watch`` (the phone -> server -> wrist path) runs here too, against
the SAME ``main.app``: the real pairing routers (``/me/pair/*`` on the
process's MemoryPairingStore, the real Firebase-then-DeviceToken verifier
chain with only ``firebase_admin.auth.verify_id_token`` faked the way
conftest fakes the main app's), the real ``/ws/live-session`` handler, the
real ``watch.relay`` (``audio_pipeline.watch_relay`` restored from the
None the other tests use), the real shared ``NudgePolicy``. Only the
watch's post-session Whisper spawn is stubbed (``ws._spawn_live_session_analysis``
— the suite's watch conftest disables it via env, but main's routers were
already built by the time that runs).
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pytest
import uvicorn

import audio_pipeline
import calls
import live_e2e
import live_sessions
import main
from main import app
from routers import sessions as sessions_router

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Provider doubles
# ---------------------------------------------------------------------------

class NullTranscriber:
    """Deepgram stand-in for a LOCAL-FIRST session: the audio still streams
    (and lands in the PCM ring buffer) but nothing is finalized server-side
    — every turn comes from the phone's turn_local."""

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes):
        return []

    async def finish(self):
        return []

    async def close(self) -> None:
        pass


class FirstTurnRaceTranscriber:
    """Deepgram stand-in that WINS the first-turn race, the way production
    does (3-way call e2e, 2026-08-25): it finalizes a member's OPENING
    utterance before that phone's first ``turn_local`` has latched the
    session local-first, so the cloud copy — no ``text_tone``, no sender
    clock, Deepgram's own wording — is what the other viewers render first.

    It hears only what this member's phone sends (``call_side_pcm`` silences
    everyone else), so the first scene turn whose window carries sound is
    this member's own opening line. It is emitted a fraction of a second
    into that window, i.e. seconds of scene time before the phone reports
    it, and nothing is emitted after: once the session is local-first the
    pipeline drops this transcriber's segments anyway (a member with silent
    audio — the observer — never emits at all).

    The fix under test is ``calls.Call.push_turn`` replacing that row in
    place and RE-RELAYING it tagged ``replaces_seq``; without it every
    viewer keeps a tone-less line for that turn.
    """

    SAMPLE_RATE = 16000
    # How far into the utterance it finalizes. Scene turns run ~4-6 s, so
    # this beats the phone's turn_local by seconds of scene time.
    LEAD_S = 0.15

    def __init__(self, turns: list[dict]) -> None:
        self._turns = turns
        self._heard = np.zeros(0, dtype=np.int16)
        self._done = False

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes):
        if self._done:
            return []
        self._heard = np.concatenate([self._heard, np.frombuffer(audio_bytes, dtype=np.int16)])
        heard_s = self._heard.shape[0] / self.SAMPLE_RATE
        for t in self._turns:
            start, end = float(t["start_time"]), float(t["end_time"])
            judge_at = min(end, start + self.LEAD_S)
            if heard_s < judge_at:
                return []  # not enough audio to tell whose turn this is yet
            window = self._heard[int(start * self.SAMPLE_RATE):int(judge_at * self.SAMPLE_RATE)]
            if not np.any(window):
                continue  # somebody else's turn: silence on this member's stream
            self._done = True
            # Deepgram-shaped: lower-cased and unpunctuated, so a row that
            # never gets corrected stands out in the merged transcript.
            return [audio_pipeline.TranscriptSegment(
                t["text"].lower().replace(".", "").replace(",", ""), start, end, speaker=1,
            )]
        return []

    async def finish(self):
        return []

    async def close(self) -> None:
        pass


class FakeTTS:
    async def synthesize(self, text: str):
        return "ZmFrZS1hdWRpbw=="


class RoutingLLM:
    """One LLM double for every prompt the run touches, told apart by the
    system prompt exactly the way test_sessions_live does: cloud
    suggestions (STREAMED, so partial previews flow), self-turn nudges, the
    batch analysis, and the reflection."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _answer(self, system: str, user: str) -> str:
        if system.startswith(live_sessions.REFLECT_SYSTEM_PROMPT):
            m = re.search(r"Reflect on \(YOU\) turn indexes: ([0-9, ]+)", user)
            idx = [int(x) for x in m.group(1).split(",")] if m else []
            return json.dumps({"reflections": [
                {"turn_index": i, "could_have_said": f"Could have said, turn {i}.",
                 "why": "Names the feeling.", "tone_read": "warm"}
                for i in idx
            ]})
        if system.startswith(main.ANALYZE_SYSTEM_PROMPT):
            n = int(user.split("Conversation (")[1].split(" turns")[0])
            speakers = sorted(set(re.findall(r"^\d+\. \[([^\]]+)\]", user, flags=re.M)))
            return json.dumps({
                "per_turn": [{"heat": 15 + (7 * i) % 60, "markers": [], "trigger_phrase": None} for i in range(n)],
                "requests": [], "narrative": "They found their way back.", "speaker_names": {},
                "report_cards": {
                    sp: {"score": 64, "headline": f"{sp} stayed present", "did_well": "Kept talking.", "work_on": "Slow down."}
                    for sp in speakers
                },
            })
        if "real-time delivery coach" in system:
            return json.dumps({"nudge": "ease up", "importance": 70})
        return json.dumps({
            "suggestions": ["I hear you.", "That sounds hard.", "Tell me more."],
            "importance": 80,
        })

    def complete(self, system: str, user: str, max_tokens: int = 512, **_) -> str:
        self.calls.append(system[:40])
        return self._answer(system, user)

    def stream_complete(self, system: str, user: str, **_):
        self.calls.append("stream:" + system[:40])
        text = self._answer(system, user)
        for i in range(0, len(text), 9):
            yield text[i:i + 9]


class EchoToneId:
    """tone_id.py double in 'on' mode: every recovered slice is classified
    (fixed label) and SURFACED as a tone_flag."""

    MIN_TURN_SECONDS = 1.0
    MAX_TURN_SECONDS = 30.0

    class ToneUnavailable(RuntimeError):
        pass

    def __init__(self) -> None:
        self.calls = 0

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def surface_allowed(self) -> bool:
        return True

    def classify_pcm(self, pcm, sr):
        self.calls += 1
        return {"label": "neutral", "scores": {"neutral": 0.7, "angry": 0.2, "sad": 0.05, "happy": 0.05},
                "confidence": 0.7, "model": "fake"}


class LabelEchoSpeakerId:
    """speaker_id.py double: the enrolled 'self' print matches the slice iff
    the phone labelled the turn with the scene's self speaker. Exercises the
    identity WIRE path (SpeakerIdentityEvent per turn) deterministically."""

    MATCH_THRESHOLD = 0.5
    MIN_MATCH_SECONDS = 1.0

    def __init__(self, self_speaker: str) -> None:
        self.self_speaker = self_speaker
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def identify_speakers_multi(self, pcm, sr, turns, voiceprints, *, threshold=None, people=None):
        self.calls += 1
        speaker = turns[0]["speaker"]
        score = 0.91 if speaker == self.self_speaker else 0.17
        matched = "self" if score >= self.MATCH_THRESHOLD else None
        return {
            "matched": {speaker: matched} if matched else {},
            "speakers": {speaker: {
                "scores": {"self": score}, "matched_person_id": matched,
                "is_self": bool(matched), "display_name": "You" if matched else None,
            }},
        }


class MemoryStore:
    """In-memory recordings store: the read/write, share and voiceprint
    surface that routers/sessions.py, main's recording reads, /growth and
    the WS enrichment path use."""

    def __init__(self) -> None:
        self._by_uid: dict[str, dict[str, dict]] = {}
        self._shares: dict[str, dict[str, dict]] = {}
        self._links: dict[str, dict] = {}
        self.voiceprints: dict[str, list[dict]] = {}

    async def save_live_session(self, uid, recording_id, *, meta, turns, analysis):
        slot = self._by_uid.setdefault(uid, {})
        written = dict(meta)
        existing = slot.get(recording_id)
        if existing:
            for key in ("manual_speaker_labels", "shares"):
                if key in existing["meta"] and key not in written:
                    written[key] = existing["meta"][key]
        slot[recording_id] = {"meta": written, "turns": turns, "analysis": analysis}
        return written

    async def update_analysis(self, uid, recording_id, analysis):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return False
        r["analysis"] = analysis
        return True

    async def list_recordings(self, uid):
        out = [{**r["meta"], "has_analysis": r["analysis"] is not None} for r in self._by_uid.get(uid, {}).values()]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, recording_id):
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def recording_exists(self, uid, recording_id):
        return recording_id in self._by_uid.get(uid, {})

    async def delete_recording(self, uid, recording_id):
        return self._by_uid.get(uid, {}).pop(recording_id, None) is not None

    async def open_media_stream(self, uid, recording_id, range_header):
        return None

    async def list_voiceprints(self, uid):
        return list(self.voiceprints.get(uid, []))

    async def add_share(self, owner_uid, recording_id, *, recipient_uid, recipient_email, owner_email):
        rec = self._by_uid.get(owner_uid, {}).get(recording_id)
        if rec is None:
            return None
        created = datetime.now(timezone.utc).isoformat()
        shares = [s for s in rec["meta"].get("shares", []) if s.get("uid") != recipient_uid]
        shares.append({"uid": recipient_uid, "email": recipient_email, "created_at": created})
        rec["meta"]["shares"] = shares
        self._shares.setdefault(recipient_uid, {})[recording_id] = {
            "owner_uid": owner_uid, "recording_id": recording_id,
            "owner_email": owner_email, "created_at": created,
        }
        return shares

    async def find_share(self, recipient_uid, recording_id):
        return self._shares.get(recipient_uid, {}).get(recording_id)

    async def list_shared_with(self, recipient_uid):
        out = []
        for rid, grant in self._shares.get(recipient_uid, {}).items():
            rec = self._by_uid.get(grant["owner_uid"], {}).get(rid)
            if rec is None:
                continue
            meta = {**rec["meta"], "has_analysis": rec["analysis"] is not None,
                    "owner_email": grant["owner_email"], "shared": True}
            meta.pop("shares", None)
            out.append(meta)
        return out

    # -- therapist link (the --call run links so the call episode auto-shares) --
    async def read_therapist_link(self, patient_uid):
        return self._links.get(patient_uid)

    async def write_therapist_link(self, patient_uid, link):
        self._links[patient_uid] = dict(link)

    async def delete_therapist_link(self, patient_uid):
        return self._links.pop(patient_uid, None) is not None

    async def list_therapist_patients(self, therapist_uid):
        return [link for link in self._links.values() if link.get("therapist_uid") == therapist_uid]


# ---------------------------------------------------------------------------
# A real server on the loopback
# ---------------------------------------------------------------------------

PATIENT_UID, THERAPIST_UID, PEER_UID = "user-a", "user-b", "test-user"
PATIENT_EMAIL, THERAPIST_EMAIL = "patient-e2e@example.test", "therapist-e2e@example.test"
PEER_EMAIL = "dad-e2e@example.test"   # the second participant of the three-way call
ACCOUNTS = {PATIENT_EMAIL: PATIENT_UID, THERAPIST_EMAIL: THERAPIST_UID, PEER_EMAIL: PEER_UID}
UID_TO_EMAIL = {v: k for k, v in ACCOUNTS.items()}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    """uvicorn serving ``main.app`` in a daemon thread (lifespan off: the
    providers are injected by the test, exactly as every other suite does
    on ``app.state``)."""
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)


def _clear_state() -> None:
    for attr in ("transcriber_factory", "tts_client", "diarizer_factory", "monotonic_clock", "recordings_store"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
def e2e_env(monkeypatch):
    """Providers + auth for one run; returns the store so the test can seed
    a voiceprint and inspect what was persisted."""
    _clear_state()
    calls.registry.reset()
    store = MemoryStore()
    app.state.recordings_store = store
    app.state.llm_client = RoutingLLM()
    app.state.transcriber_factory = lambda: NullTranscriber()
    app.state.tts_client = FakeTTS()
    sessions_router._REFLECT_LOCKS.clear()
    main._rate_limiter.reset()
    monkeypatch.setattr(main, "resolve_uid_by_email", lambda email: ACCOUNTS.get(email.strip().lower()))
    monkeypatch.setattr(main, "resolve_email_by_uid", lambda uid: UID_TO_EMAIL.get(uid))
    monkeypatch.setattr(audio_pipeline, "watch_relay", None)
    monkeypatch.setattr(audio_pipeline, "tone_id", EchoToneId())
    yield store
    _clear_state()
    calls.registry.reset()


def _account(uid: str, email: str, token: str) -> live_e2e.Account:
    # conftest FAKE_TOKENS maps the token → uid for the WS handshake; the
    # X-Test-Uid header is the REST suite's dependency override.
    return live_e2e.Account(email=email, ws_token=token, headers={"X-Test-Uid": uid}, uid=uid)


async def _run_scene(scene_name: str, base_url: str, store: MemoryStore, monkeypatch) -> live_e2e.Report:
    scene = live_e2e.load_scene(scene_name)
    monkeypatch.setattr(audio_pipeline, "speaker_id", LabelEchoSpeakerId(scene.self_speaker))
    # The account "enrolled" earlier: one self voiceprint (the embedding is
    # opaque to the label-echo double; the store shape is what matters).
    store.voiceprints[PATIENT_UID] = [
        {"person_id": "self", "display_name": "You", "is_self": True, "embedding": [1.0, 0.0]},
    ]
    patient = _account(PATIENT_UID, PATIENT_EMAIL, "tok-user-a")
    therapist = _account(THERAPIST_UID, THERAPIST_EMAIL, "tok-user-b")
    return await live_e2e.run_e2e(
        base_url=base_url, patient=patient, therapist=therapist, scene=scene,
        speed=25.0, mode="earpiece", enroll=False, analysis_timeout_s=30.0,
        session_id=f"e2e-{scene.name}-{int(time.time() * 1000)}",
    )


@pytest.mark.parametrize("scene_name", ["scene_couple_escalation", "scene_family3"])
async def test_live_e2e_inprocess(scene_name, live_server, e2e_env, monkeypatch):
    report = await _run_scene(scene_name, live_server, e2e_env, monkeypatch)
    text = live_e2e.format_report(report)
    print("\n" + text)
    assert not report.failures, text

    scene = live_e2e.load_scene(scene_name)
    data = report.data
    # The whole live protocol was exercised, not just the REST tail.
    assert data["ws"]["turn_locals"] == len(scene.turns)
    assert data["ws"]["close_code"] == 1000
    assert data["latency_summary"]["total"]["n"] >= 1
    assert "llm_first_partial" in data["latency_summary"]          # the LLM streamed
    assert data["suggestions"]["partial"] >= 1                     # …and previews reached the client
    assert data["suggestions"]["sources"] == ["cloud"]
    assert data["suggestions"]["errors"] == 0
    # Enrichment on the phone's turns: an identity verdict per turn (the
    # ring buffer recovered every slice) that agrees with the scene.
    assert data["identity"]["verdicts"] == len(scene.turns)
    assert data["identity"]["agree"] == len(scene.turns)
    assert data["tone_flags"]["count"] == len(scene.turns)
    # The stored episode carries the scene's self escalations and the
    # therapist sees the same numbers through the share.
    assert data["detail"]["escalation_turns"] == scene.expected_self_escalations
    assert data["therapist_row"]["escalation_turns"] == scene.expected_self_escalations
    assert data["therapist_row"]["couldHaveSaid"] == len(scene.self_turn_indexes)
    assert data["growth_point"]["my_score"] == 64


# ---------------------------------------------------------------------------
# --with-watch: phone -> server -> wrist, in-process
# ---------------------------------------------------------------------------

WATCH_TOKENS = {"tok-user-a": PATIENT_UID, "tok-user-b": THERAPIST_UID}


@pytest.fixture
def watch_env(e2e_env, monkeypatch):
    """On top of e2e_env: the real relay, the watch domain's Firebase
    verifier faked to the suite's tokens (so POST /me/pair/claim can
    authenticate the patient), no Whisper spawn on the watch's `end`."""
    import firebase_admin.auth as fb_auth

    from watch import relay
    from watch.routers import ws as watch_ws

    def _verify(token: str) -> dict:
        try:
            return {"uid": WATCH_TOKENS[token]}
        except KeyError:
            raise ValueError("invalid test token")

    monkeypatch.setattr(fb_auth, "verify_id_token", _verify)
    monkeypatch.setattr(audio_pipeline, "watch_relay", relay)
    monkeypatch.setattr(watch_ws, "_spawn_live_session_analysis", lambda *a, **k: None)
    relay._registry.clear()
    yield e2e_env
    relay._registry.clear()


@pytest.mark.parametrize("watch_auth", ["token", "account"])
async def test_live_e2e_inprocess_with_watch(watch_auth, live_server, watch_env, monkeypatch):
    """The wrist hears the phone: pair, hold the watch socket open through
    the couple scene, and get exactly the frames the shared relay +
    NudgePolicy predict — nudge A1 after self turn 4 (tense_rising, tone
    level 1), A3 after turn 6 (shout_angry, tone 3), a sustaining
    aggressive_tone=3 vector_event and NO new nudge after turn 8
    (cold_contempt: same level, hysteresis), nothing for Speaker B."""
    scene_name = "scene_couple_escalation"
    scene = live_e2e.load_scene(scene_name)
    monkeypatch.setattr(audio_pipeline, "speaker_id", LabelEchoSpeakerId(scene.self_speaker))
    watch_env.voiceprints[PATIENT_UID] = [
        {"person_id": "self", "display_name": "You", "is_self": True, "embedding": [1.0, 0.0]},
    ]
    patient = _account(PATIENT_UID, PATIENT_EMAIL, "tok-user-a")
    patient.headers["Authorization"] = "Bearer tok-user-a"   # the watch routers verify a real bearer
    therapist = _account(THERAPIST_UID, THERAPIST_EMAIL, "tok-user-b")
    report = await live_e2e.run_e2e(
        base_url=live_server, patient=patient, therapist=therapist, scene=scene,
        speed=25.0, mode="earpiece", enroll=False, analysis_timeout_s=30.0, cleanup=True,
        session_id=f"e2e-{scene.name}-{int(time.time() * 1000)}",
        with_watch=True, watch_auth=watch_auth, watch_settle_s=0.3,
    )
    text = live_e2e.format_report(report)
    print("\n" + text)
    assert not report.failures, text

    w = report.data["watch"]
    assert w["paired"] and w["auth_mode"] == watch_auth
    assert [e["turn_index"] for e in w["expected"]] == [4, 6, 8]
    assert w["groups"] == [
        ([("aggressive_tone", 1)], [("A", 1)]),
        ([("aggressive_tone", 3)], [("A", 3)]),
        ([("aggressive_tone", 3)], []),
    ]
    assert [t["turn_index"] for t in w["timing"]] == [4, 6, 8]
    assert all(0 < t["ms"] < 10_000 for t in w["timing"])
    assert [r["ok"] for r in w["spec"]] == [True, True, True]
    assert w["saved"]["status"] == "captured" and w["close_code"] == 1000
    assert w["frames"] == 3 + 2 + 1   # vector_events + nudges + live_session_saved
    names = [c.name for c in report.checks]
    for name in ("watch pairing", "watch ws", "watch nudges", "watch nudge timing", "watch scene spec", "watch session persisted"):
        assert name in names, name
    cleanup = next(c for c in report.checks if c.name == "cleanup").detail
    assert "watch live session delete 204" in cleanup and "unpair watch 200 count=1" in cleanup


class TestWatchHelpers:
    def test_expected_watch_relay_matches_the_scene_spec_through_the_policy(self):
        scene = live_e2e.load_scene("scene_couple_escalation")
        turns = live_e2e.build_turn_locals(scene, "s")
        expected = live_e2e.expected_watch_relay(turns)
        assert [e["turn_index"] for e in expected] == [n["after_turn_index"] for n in scene.meta["expected_nudges"]]
        assert [e["events"] for e in expected] == [[("aggressive_tone", 1)], [("aggressive_tone", 3)], [("aggressive_tone", 3)]]
        assert [e["nudges"] for e in expected] == [[("A", 1)], [("A", 3)], []]
        assert [e["level_after"] for e in expected] == [1, 3, 3]
        # An enrolled account whose baseline sits well under the scene's
        # loudness yells too: both lanes, max wins, `vectors` names the winner.
        loud = live_e2e.expected_watch_relay(turns, baseline_rms_db=-40.0)
        assert loud[0]["events"][0][0] == "yelling" and loud[0]["nudges"] == [("A", 3)]

    def test_group_watch_frames_splits_per_turn(self):
        f = lambda i, **kw: (float(i), kw)  # noqa: E731
        frames = [
            f(1, type="vector_event", vector="yelling", level=2, t=0.0),
            f(2, type="vector_event", vector="aggressive_tone", level=1, t=0.0),
            f(3, type="nudge", channel="A", level=2, t=0.0, vectors=["yelling"]),
            f(4, type="vector_event", vector="aggressive_tone", level=3, t=0.0),
            f(5, type="nudge", channel="A", level=3, t=0.0, vectors=["aggressive_tone"]),
            f(6, type="nudge", channel="A", level=2, t=30.0, vectors=[]),          # bare de-escalation (watch-clock tick)
            f(7, type="vector_event", vector="aggressive_tone", level=3, t=30.0),  # escalates again
            f(8, type="nudge", channel="A", level=3, t=30.0, vectors=["aggressive_tone"]),
            f(9, type="vector_event", vector="aggressive_tone", level=3, t=30.0),  # sustain: no nudge
            f(10, type="vector_event", vector="aggressive_tone", level=3, t=30.0),  # next turn, same vector
            f(11, type="live_session_saved", live_session_id="x"),
        ]
        groups = live_e2e.group_watch_frames(frames)
        assert [(g["events"], g["nudges"]) for g in groups] == [
            ([("yelling", 2), ("aggressive_tone", 1)], [("A", 2)]),
            ([("aggressive_tone", 3)], [("A", 3)]),
            ([], [("A", 2)]),
            ([("aggressive_tone", 3)], [("A", 3)]),
            ([("aggressive_tone", 3)], []),
            ([("aggressive_tone", 3)], []),
        ]
        assert [g["at"] for g in groups] == [1.0, 4.0, 6.0, 7.0, 9.0, 10.0]
        assert live_e2e._channel_a_after(groups, 1) == 3 and live_e2e._channel_a_after(groups, 2) == 2

    def test_watch_ws_url_and_spec_levels(self):
        assert live_e2e.watch_ws_url("https://x.run.app/", "ls", token="tok") == "wss://x.run.app/ws/live-session/ls?token=tok"
        assert live_e2e.watch_ws_url("http://127.0.0.1:8000", "ls", account="u") == "ws://127.0.0.1:8000/ws/live-session/ls?account=u"
        assert live_e2e._spec_level_ok("mild", 1) and live_e2e._spec_level_ok("strong", 3)
        assert not live_e2e._spec_level_ok("strong", 2) and not live_e2e._spec_level_ok("bogus", 3)


async def test_call_e2e_inprocess(live_server, e2e_env, monkeypatch):
    """``--call``: the patient hosts, the therapist joins by code, both
    phones bind + signal over their own sockets, speak their halves of the
    couple scene concurrently, and each ends with a mode=call episode of
    the merged transcript — the patient's auto-shared to the therapist
    through the link the run creates."""
    scene = live_e2e.load_scene("scene_couple_escalation")
    monkeypatch.setattr(audio_pipeline, "speaker_id", None)
    patient = _account(PATIENT_UID, PATIENT_EMAIL, "tok-user-a")
    therapist = _account(THERAPIST_UID, THERAPIST_EMAIL, "tok-user-b")
    # 4x: the scene's 0.4 s turn gaps become 100 ms of wall clock — enough
    # for the two sockets' turn_locals to arrive in scene order.
    report = await live_e2e.run_call_e2e(
        base_url=live_server, patient=patient, therapist=therapist, scene=scene,
        speed=4.0, analysis_timeout_s=30.0,
        session_id=f"e2e-call-{int(time.time() * 1000)}",
    )
    text = live_e2e.format_report(report)
    print("\n" + text)
    assert not report.failures, text
    data = report.data
    n_self, n_other = len(scene.self_turn_indexes), len(scene.turns) - len(scene.self_turn_indexes)
    assert data["ws_host"]["turn_locals"] == n_self and data["ws_joiner"]["turn_locals"] == n_other
    assert data["ws_host"]["close_code"] == 1000 and data["ws_joiner"]["close_code"] == 1000
    assert data["signaling"] == {"offer_ok": True, "answer_ok": True, "host_error": None, "joiner_error": None}
    # Each phone saw exactly the other's turns, named by the join display names.
    assert data["merged_host"] == {"count": n_other, "expected": n_other, "in_order": True, "names": ["Therapist"]}
    assert data["merged_joiner"] == {"count": n_self, "expected": n_self, "in_order": True, "names": ["Patient"]}
    # Both sides were coached on the other's turns (nudges on their own are
    # subject to latest-wins at 4x, so only the presence of responses is pinned).
    assert data["coaching_host"]["responses"] >= 1 and data["coaching_joiner"]["responses"] >= 1
    assert data["coaching_host"]["errors"] == 0 and data["coaching_joiner"]["errors"] == 0
    assert data["episodes"]["shared_with"] == [THERAPIST_EMAIL]
    assert data["detail"]["turns"] == len(scene.turns) and data["detail"]["in_order"]
    assert data["detail"]["escalation_turns"] == scene.expected_self_escalations
    assert data["detail"]["peer_label"] == "Therapist"
    assert data["growth_point"]["mode"] == "call"
    assert data["therapist_rows"]["shared"]["role"] == PATIENT_EMAIL
    assert data["therapist_rows"]["own"]["role"] == "You"
    # The stored episodes: one per participant, same merged turns, opposite selves.
    host_rec = e2e_env._by_uid[PATIENT_UID][data["episodes"]["host"]]
    joiner_rec = e2e_env._by_uid[THERAPIST_UID][data["episodes"]["joiner"]]
    assert [t["text"] for t in host_rec["turns"]] == [t["text"] for t in joiner_rec["turns"]] == [t["text"] for t in scene.turns]
    assert host_rec["analysis"]["live"]["self_speaker"] == "Speaker A"
    assert joiner_rec["analysis"]["live"]["self_speaker"] == "Speaker B"
    assert [t["call_seq"] for t in host_rec["turns"]] == list(range(1, len(scene.turns) + 1))
    assert joiner_rec["meta"]["title"] == "Call with Patient"


async def test_call_e2e_inprocess_three_way(live_server, e2e_env, monkeypatch):
    """``--call --participants 3``: the patient hosts (Sage, A), a second
    account joins over REST as the invited participant (Dad, B), the
    therapist joins on her socket with the code as the observer (Mom, C).
    Full-mesh signaling, every phone's own turns merged for the others with
    relative names, coaching per participant only (Mom gets read-only
    ``for_uid`` copies, never a suggestion of her own), hang-up host → Dad →
    Mom so HER socket ends the call, two episodes both granted to her.

    The server's transcriber here BEATS each phone to that member's opening
    line (FirstTurnRaceTranscriber) — last night's production failure — so
    the run also proves the correction path: the cloud row is replaced in
    place and re-relayed tagged ``replaces_seq``, and every viewer ends up
    with one line per turn, all of them carrying the sender's tone."""
    scene = live_e2e.load_scene("scene_couple_escalation")
    monkeypatch.setattr(audio_pipeline, "speaker_id", None)
    app.state.transcriber_factory = lambda: FirstTurnRaceTranscriber(scene.turns)
    patient = _account(PATIENT_UID, PATIENT_EMAIL, "tok-user-a")
    peer = _account(PEER_UID, PEER_EMAIL, "fake-id-token")
    therapist = _account(THERAPIST_UID, THERAPIST_EMAIL, "tok-user-b")
    report = await live_e2e.run_call_e2e_three_way(
        base_url=live_server, patient=patient, peer=peer, therapist=therapist, scene=scene,
        speed=4.0, analysis_timeout_s=30.0, cleanup=True,
        session_id=f"e2e-call3-{int(time.time() * 1000)}",
    )
    text = live_e2e.format_report(report)
    print("\n" + text)
    assert not report.failures, text
    data = report.data
    n_a, n_b, n_c = len(scene.self_turn_indexes), len(scene.turns) - len(scene.self_turn_indexes), len(live_e2e.THERAPIST_LINES)
    assert (data["ws_host"]["turn_locals"], data["ws_peer"]["turn_locals"], data["ws_therapist"]["turn_locals"]) == (n_a, n_b, n_c)
    assert (data["ws_host"]["label"], data["ws_peer"]["label"], data["ws_therapist"]["label"]) == ("Speaker A", "Speaker B", "Speaker C")
    assert all(data[f"ws_{r}"]["close_code"] == 1000 for r in ("host", "peer", "therapist"))
    assert data["call_state"]["problems"] == [] and all(data["call_state"]["transitions"].values())
    # Full mesh: two addressed offers in, one deliberate unaddressed error.
    assert data["signaling"] == {"problems": [], "missing_to_error": live_e2e.MISSING_TO_ERROR,
                                 "delivered": {"host": 2, "peer": 2, "therapist": 2}}
    # Each viewer saw exactly the others' turns, named relative to itself, in
    # scene order — and NOT ONE of them tone-less. A rendered row with no
    # text_tone is the server transcriber's copy of a member's own words that
    # the member's phone also reported: the first-turn race (Deepgram
    # finalizing before the phone's first turn_local latches the session
    # local-first) that calls.Call.push_turn corrects by re-relaying the row
    # tagged replaces_seq. `frames` counts what arrived, `count` what a
    # client renders after folding by seq: a correction must never add a line.
    for role, expected in (("host", n_b + n_c), ("peer", n_a + n_c), ("therapist", n_a + n_b)):
        merged = data[f"merged_{role}"]
        assert merged["toneless"] == 0, (role, merged)
        assert merged["count"] == merged["expected"] == expected
        assert merged["frames"] == expected + merged["corrected"]
    # Both coached members lost the race on their opening line, so every
    # viewer of them got exactly one correction — and rendered one line.
    assert data["merged_host"]["corrected"] == 1      # Dad's opener (Mom is silent)
    assert data["merged_peer"]["corrected"] == 1      # the host's opener
    assert data["merged_therapist"]["corrected"] == 2  # both participants'
    assert data["merged_host"]["names"] == ["Dad", "Mom (therapist)"] and data["merged_host"]["in_order"]
    assert data["merged_peer"]["names"] == ["Mom (therapist)", "Sage"] and data["merged_peer"]["in_order"]
    assert data["merged_therapist"]["names"] == ["Dad", "Sage"] and data["merged_therapist"]["in_order"]
    # Coaching: participants only (never tagged); the observer gets copies for both and nothing of her own.
    for r in ("host", "peer"):
        assert data[f"coaching_{r}"]["errors"] == 0 and data[f"coaching_{r}"]["tagged"] == 0
        assert data[f"coaching_{r}"]["about"]["Speaker B" if r == "host" else "Speaker A"] >= 1
    ther = data["coaching_therapist"]
    assert ther["own_suggestions"] == 0 and ther["foreign"] == 0
    assert ther["copies"]["host"] >= 1 and ther["copies"]["peer"] >= 1
    assert ("suggestion", "response") in ther["copy_kinds"]["host"] and ("tone_flag", None) in ther["copy_kinds"]["host"]
    # Relay latency over the loopback: every delivery timed, well under a second.
    d = data["delivery"]
    assert d["n"] == d["expected"] == 2 * (n_a + n_b + n_c) and d["p95_ms"] < 1000
    # The last socket ended it; exactly two episodes, one per participant.
    assert data["call_ended"]["reason"] == "all participants left" and data["call_ended"]["ended_by"] == THERAPIST_UID
    assert data["call_ended"]["episode_id"] is None
    assert sorted(data["call_ended"]["episodes"]) == sorted([PATIENT_UID, PEER_UID])
    assert data["call_ended"]["turn_count"] == n_a + n_b + n_c
    assert data["episodes"]["shared_with"] == [THERAPIST_EMAIL]
    assert data["detail_host"]["labels"] == {"Speaker A": "You", "Speaker B": "Dad", "Speaker C": "Mom (therapist)"}
    assert data["detail_host"]["title"] == "Call with Dad and Mom (therapist)"
    assert data["detail_peer"]["labels"] == {"Speaker B": "You", "Speaker A": "Sage", "Speaker C": "Mom (therapist)"}
    assert data["detail_peer"]["title"] == "Call with Sage and Mom (therapist)"
    assert data["growth_point"]["mode"] == "call"
    assert data["therapist_rows"]["host"]["role"] == PATIENT_EMAIL and data["therapist_rows"]["peer"]["role"] == PEER_EMAIL
    assert data["therapist_rows"]["own_call_rows"] == 0 and data["therapist_rows"]["total"] == 2
    # The observer never got an episode; cleanup removed both participants'.
    assert THERAPIST_UID not in e2e_env._by_uid
    assert e2e_env._by_uid[PATIENT_UID] == {} and e2e_env._by_uid[PEER_UID] == {}
    cleanup = next(c for c in report.checks if c.name == "cleanup").detail
    assert "delete 204" in cleanup and "rows left for the call: 0" in cleanup


class TestPureHelpers:
    def test_text_tone_table_matches_live_sessions_rules(self):
        """Every angry scripted emotion must read as an escalation to the
        server, every non-angry one must not — otherwise the report's
        escalation checks are testing the mapping, not the server."""
        for emotion, tone in live_e2e.TEXT_TONE_BY_EMOTION.items():
            coarse = "angry" if tone["label"] == "angry" else None
            assert live_sessions.is_escalated(tone) is (coarse == "angry"), emotion
        for scene_name in live_e2e.list_scenes():
            meta = live_e2e.load_scene(scene_name).meta
            for t in meta["turns"]:
                tone = live_e2e.text_tone_for(t)
                assert live_sessions.is_escalated(tone) is (t["emotion_coarse"] == "angry"), (scene_name, t["scripted_emotion"])
                assert t["scripted_emotion"] in live_e2e.TEXT_TONE_BY_EMOTION, (scene_name, t["scripted_emotion"])

    def test_turn_locals_are_valid_wire_events(self):
        from models.audio import TurnLocalEvent

        scene = live_e2e.load_scene("scene_family3")
        events = live_e2e.build_turn_locals(scene, "sess-1")
        assert len(events) == len(scene.turns) == 15
        for ev, t in zip(events, scene.turns):
            model = TurnLocalEvent.model_validate(ev)
            assert model.is_self is scene.is_self(t["speaker"])
            assert (model.speaker_person_id == "self") is model.is_self
            assert model.prosody is not None and model.prosody.rms_dbfs is not None
            assert model.prosody.speech_rate is not None and model.prosody.speech_rate > 0
            assert model.suggestion is None and model.suggestion_source is None
        # Timeline reconstruction matches the WAV length like the ladder's.
        assert abs(events[-1]["end_time"] - scene.duration_s) < 0.01

    def test_self_voice_wav_is_only_self_turns(self):
        scene = live_e2e.load_scene("scene_couple_escalation")
        wav = live_e2e.self_voice_wav(scene, max_seconds=8.0)
        assert wav[:4] == b"RIFF"
        assert abs((len(wav) - 44) / (2 * scene.sr) - 8.0) < 0.01

    def test_ws_url_and_scene_listing(self):
        assert live_e2e.ws_url("https://x.run.app/", "s") == "wss://x.run.app/ws/session/s"
        assert live_e2e.ws_url("http://127.0.0.1:8000", "s") == "ws://127.0.0.1:8000/ws/session/s"
        assert {"scene_couple_escalation", "scene_family3", "scene_meeting4"} <= set(live_e2e.list_scenes())

    def test_report_marks_and_exit_semantics(self):
        r = live_e2e.Report(scene="s", base_url="u", speed=1.0, mode="earpiece")
        r.add("a", True, "fine")
        r.add("b", None, "skipped")
        assert not r.failures and len(r.warnings) == 1
        assert "RESULT: PASS" in live_e2e.format_report(r)
        r.add("c", False, "broken")
        assert [c.name for c in r.failures] == ["c"]
        assert "❌ c: broken" in live_e2e.format_report(r) and "RESULT: FAIL" in live_e2e.format_report(r)


def test_first_response_timing_prefers_partial():
    run = live_e2e.WsRun(session_id="s")
    run.sent_turns.append((10.0, {"text": "hello", "is_self": False}))
    run.events.append((10.4, {"type": "suggestion", "utterance_text": "hello", "partial": True}))
    run.events.append((11.0, {"type": "suggestion", "utterance_text": "hello", "partial": False}))
    run.events.append((9.0, {"type": "suggestion", "utterance_text": "hello", "partial": False}))  # before send: ignored
    t = live_e2e.first_response_ms(run)
    assert t["per_turn"][0]["first_partial_ms"] == 400.0
    assert t["per_turn"][0]["first_final_ms"] == 1000.0
    assert t["first_p50_ms"] == 400.0 and t["turns_with_response"] == 1
    assert np.isclose(t["partial_p50_ms"], 400.0)
