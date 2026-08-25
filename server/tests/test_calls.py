"""In-app calls (server/calls.py + routers/calls.py + the call frames on the
realtime WebSocket).

Two coached participants — the host (slot A, "Speaker A") and the joiner
(slot B, "Speaker B") — and optionally one observing therapist (slot C,
"Speaker C"), each on their own ``/ws/session/{id}`` socket, bound to one
call with ``call_join``. The server relays WebRTC signaling between them
(a full mesh with three), merges every phone's ``turn_local`` reports into
one shared transcript (pushed to the others as ``transcript`` events with
the right relative labels), coaches EACH participant on the merged context
(nudges on their own turns, suggestions about everyone else's), gives the
therapist a read-only ``for_uid``-tagged copy of that coaching and never
coaches her, and at the end persists one episode per participant through
the live-session ingest (mode ``call``) with auto-share via the therapist
link plus a direct grant to the therapist who was on the call.

Providers are the suite's usual doubles on ``app.state``; the store is an
in-memory double (recordings + shares + therapist links, the surface ingest,
auto-share and the dashboard read);
auth is conftest's keyless harness (``tok-user-a`` → ``user-a`` on the
socket, ``X-Test-Uid`` on REST).
"""

from __future__ import annotations

import json
import re
import types
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from starlette.testclient import TestClient

import audio_pipeline
import calls
import main
from main import app

from tests.test_audio_pipeline import (  # noqa: E402 — the DI doubles/helpers
    MOCK_LLM_JSON, NUDGE_LLM_JSON, FakeTTS, StoppableTranscriber, TranscriptSegment,
    _clear_overrides, open_ws, recv_until,
)

HOST, PEER, THIRD = "user-a", "user-b", "test-user"
HOST_TOKEN, PEER_TOKEN, THIRD_TOKEN = "tok-user-a", "tok-user-b", "fake-id-token"
EMAILS = {HOST: "sage@example.test", PEER: "mom@example.test", THIRD: "third@example.test"}
UIDS = {v: k for k, v in EMAILS.items()}
HOST_SID = "call-host-session"
PEER_SID = "call-peer-session"


def _h(uid: str) -> dict:
    return {"X-Test-Uid": uid}


class FakeStore:
    """In-memory recordings store: what ingest_live, therapist auto-share,
    the in-call therapist grant and GET /sessions touch."""

    def __init__(self) -> None:
        self._by_uid: dict = {}      # {uid: {rid: {meta, turns, analysis}}}
        self._index: dict = {}       # {recipient: {rid: grant}}
        self._links: dict = {}       # {patient_uid: link}

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

    async def list_voiceprints(self, uid):
        return []

    async def list_recordings(self, uid):
        out = [{**r["meta"], "has_analysis": r["analysis"] is not None}
               for r in self._by_uid.get(uid, {}).values()]
        out.sort(key=lambda m: m["created_at"], reverse=True)
        return out

    async def get_recording(self, uid, rid):
        r = self._by_uid.get(uid, {}).get(rid)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def recording_exists(self, uid, rid):
        return rid in self._by_uid.get(uid, {})

    async def add_share(self, owner_uid, rid, *, recipient_uid, recipient_email, owner_email):
        r = self._by_uid.get(owner_uid, {}).get(rid)
        if r is None:
            return None
        created = datetime.now(timezone.utc).isoformat()
        shares = [s for s in (r["meta"].get("shares") or []) if s["uid"] != recipient_uid]
        shares.append({"uid": recipient_uid, "email": recipient_email, "created_at": created})
        r["meta"]["shares"] = shares
        self._index.setdefault(recipient_uid, {})[rid] = {
            "owner_uid": owner_uid, "recording_id": rid,
            "owner_email": owner_email, "created_at": created,
        }
        return shares

    async def find_share(self, recipient_uid, rid):
        return self._index.get(recipient_uid, {}).get(rid)

    async def list_shared_with(self, recipient_uid):
        out = []
        for rid, grant in self._index.get(recipient_uid, {}).items():
            r = self._by_uid.get(grant["owner_uid"], {}).get(rid)
            if r is None:
                continue
            meta = {**r["meta"], "has_analysis": r["analysis"] is not None,
                    "owner_email": grant["owner_email"], "shared": True}
            meta.pop("shares", None)
            out.append(meta)
        return out

    async def read_therapist_link(self, patient_uid):
        return self._links.get(patient_uid)

    async def write_therapist_link(self, patient_uid, link):
        self._links[patient_uid] = dict(link)

    async def delete_therapist_link(self, patient_uid):
        return self._links.pop(patient_uid, None) is not None

    async def list_therapist_patients(self, therapist_uid):
        return [l for l in self._links.values() if l.get("therapist_uid") == therapist_uid]


class RoutingLLM:
    """Plain object (no MagicMock → the non-streaming complete() path),
    answering by system prompt: nudges for self turns, suggestions for the
    peer's, and the batch analysis/reflection the persisted episode runs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **_) -> str:
        self.calls.append((system, user))
        if "real-time delivery coach" in system:
            return NUDGE_LLM_JSON
        if system.startswith(main.ANALYZE_SYSTEM_PROMPT):
            n = int(user.split("Conversation (")[1].split(" turns")[0])
            speakers = sorted(set(re.findall(r"^\d+\. \[([^\]]+)\]", user, flags=re.M)))
            return json.dumps({
                "per_turn": [{"heat": 20, "markers": [], "trigger_phrase": None} for _ in range(n)],
                "requests": [], "narrative": "ok", "speaker_names": {},
                "report_cards": {sp: {"score": 60, "headline": sp, "did_well": "x", "work_on": "y"} for sp in speakers},
            })
        if "Reflect on (YOU) turn indexes" in user:
            m = re.search(r"Reflect on \(YOU\) turn indexes: ([0-9, ]+)", user)
            idx = [int(x) for x in m.group(1).split(",")] if m else []
            return json.dumps({"reflections": [
                {"turn_index": i, "could_have_said": "…", "why": "…", "tone_read": "warm"} for i in idx
            ]})
        return MOCK_LLM_JSON

    @property
    def user_prompts(self) -> list[str]:
        return [u for _, u in self.calls]


@pytest.fixture
def env(monkeypatch):
    _clear_overrides()
    calls.registry.reset()
    monkeypatch.setattr(audio_pipeline, "tone_id", None)
    monkeypatch.setattr(audio_pipeline, "speaker_id", None)
    monkeypatch.setattr(audio_pipeline, "watch_relay", None)
    monkeypatch.setattr(audio_pipeline, "SLICE_GRACE_S", 0.0)
    # The two LLM passes on the persisted episodes run as background tasks
    # on the socket's loop; off by default here (one test turns them on).
    monkeypatch.setattr(calls, "ANALYZE_ON_END", False)
    monkeypatch.setattr(calls, "REFLECT_ON_END", False)
    monkeypatch.setattr(main, "resolve_uid_by_email", lambda e: UIDS.get(e.strip().lower()))
    monkeypatch.setattr(main, "resolve_email_by_uid", lambda u: EMAILS.get(u))
    main._rate_limiter.reset()
    llm = RoutingLLM()
    store = FakeStore()
    app.state.llm_client = llm
    app.state.transcriber_factory = lambda: StoppableTranscriber()
    app.state.tts_client = FakeTTS()
    app.state.recordings_store = store
    # ONE event loop for every socket of a test (as in production, where
    # uvicorn runs them all on one loop): a bare TestClient gives each
    # websocket_connect its own portal thread, and the call relay then
    # awaits locks/queues across loops. `with TestClient(app)` would share
    # a portal but also run main's lifespan, which replaces the doubles
    # above — so the portal is attached by hand.
    from anyio.from_thread import start_blocking_portal
    client = TestClient(app)
    with start_blocking_portal(**client.async_backend) as portal:
        client.portal = portal
        yield types.SimpleNamespace(client=client, store=store, llm=llm)
    _clear_overrides()
    calls.registry.reset()


def _create(env, uid: str = HOST, **body) -> dict:
    res = env.client.post("/calls", json=body, headers=_h(uid))
    assert res.status_code == 201, res.text
    return res.json()


def _join(env, call_id: str, uid: str, **body) -> tuple[int, dict]:
    res = env.client.post(f"/calls/{call_id}/join", json=body, headers=_h(uid))
    return res.status_code, res.json()


def _turn(sid: str, text: str, *, speaker: str = "Speaker Q", start: float = 0.0, end: float = 1.0, **extra) -> dict:
    return {
        "type": "turn_local", "session_id": sid, "speaker": speaker, "text": text,
        "start_time": start, "end_time": end, "transcript_source": "on-device", **extra,
    }


def _bind(ws, call_id: str, **extra) -> dict:
    """Send call_join and return the call_state that answers it."""
    ws.send_text(json.dumps({"type": "call_join", "call_id": call_id, **extra}))
    state, _ = recv_until(ws, lambda m: m.get("type") == "call_state")
    return state


def _by_uid(state: dict) -> dict[str, dict]:
    return {p["uid"]: p for p in state["participants"]}


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

class TestCallRest:
    def test_create_returns_code_url_ice_and_host_slot(self, env):
        body = _create(env, invitee_email="Mom@Example.test", display_name="  Sage  ")
        assert re.fullmatch(audio_pipeline.UUID_PATTERN, body["call_id"])
        assert len(body["join_code"]) == calls.JOIN_CODE_LEN
        assert all(c in calls.JOIN_CODE_ALPHABET for c in body["join_code"])
        assert body["join_url"].endswith("/call/" + body["join_code"])
        assert body["status"] == "open" and body["host_uid"] == HOST
        assert body["invitee_uid"] == PEER and body["invitee_email"] == "mom@example.test"
        assert body["invitee"] == {"uid": PEER, "email": "mom@example.test"}
        assert body["self_label"] == "Speaker A" and body["peer_label"] == "Speaker B"
        assert body["ice_servers"] == [{"urls": [calls.DEFAULT_STUN_URL], "username": None, "credential": None}]
        [me] = body["participants"]
        assert me == {
            "uid": HOST, "slot": "A", "label": "Speaker A", "role": "participant", "display_name": "You",
            "is_self": True, "connected": False, "joined_at": me["joined_at"],
        }
        assert body["max_participants"] == 3 and body["self_role"] == "participant"
        assert body["therapist_label"] == "Speaker C" and body["therapist_uid"] is None
        assert body["episode_id"] is None and body["turn_count"] == 0
        assert datetime.fromisoformat(body["expires_at"]) > datetime.now(timezone.utc)

    def test_invitee_resolution_errors(self, env):
        res = env.client.post("/calls", json={"invitee_email": "nobody@example.test"}, headers=_h(HOST))
        assert res.status_code == 404
        res = env.client.post("/calls", json={"invitee_email": EMAILS[HOST]}, headers=_h(HOST))
        assert res.status_code == 400
        res = env.client.post("/calls", json={"invitee_email": "not-an-email"}, headers=_h(HOST))
        assert res.status_code == 422

    def test_turn_servers_from_env(self, env, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_TURN_URLS", "turn:relay.example:3478?transport=udp, turns:relay.example:5349")
        monkeypatch.setenv("MINDSHIFT_TURN_USERNAME", "u")
        monkeypatch.setenv("MINDSHIFT_TURN_CREDENTIAL", "p")
        body = _create(env)
        assert body["ice_servers"][1] == {
            "urls": ["turn:relay.example:3478?transport=udp", "turns:relay.example:5349"],
            "username": "u", "credential": "p",
        }

    def test_invitee_joins_without_code_anyone_else_needs_it(self, env):
        created = _create(env, invitee_email=EMAILS[PEER])
        cid, code = created["call_id"], created["join_code"]
        # The invitee sees the call before joining, a stranger does not.
        assert env.client.get(f"/calls/{cid}", headers=_h(PEER)).status_code == 200
        assert env.client.get(f"/calls/{cid}", headers=_h(THIRD)).status_code == 404
        status, body = _join(env, cid, PEER)
        assert status == 200, body
        assert body["status"] == "active" and body["self_label"] == "Speaker B"
        assert _by_uid(body)[HOST]["display_name"] == EMAILS[HOST]
        assert _by_uid(body)[HOST]["is_self"] is False and _by_uid(body)[PEER]["is_self"] is True
        # Full: a third account with the right code is refused.
        status, body = _join(env, cid, THIRD, join_code=code)
        assert status == 409
        # Re-join is idempotent (a name refresh).
        status, body = _join(env, cid, PEER, display_name="Mom")
        assert status == 200 and _by_uid(body)[PEER]["display_name"] == "You"
        assert _by_uid(env.client.get(f"/calls/{cid}", headers=_h(HOST)).json())[PEER]["display_name"] == "Mom"

    def test_open_call_join_by_code(self, env):
        created = _create(env)
        cid, code = created["call_id"], created["join_code"]
        assert _join(env, cid, PEER)[0] == 403
        assert _join(env, cid, PEER, join_code="XXXXXX")[0] == 403
        # Lower-case, spaced, dashed — the code is typed from a text.
        pretty = code[:3].lower() + "-" + code[3:]
        status, body = _join(env, cid, PEER, join_code=pretty)
        assert status == 200 and body["status"] == "active"
        # Join by code alone (the invite link only carries the code).
        res = env.client.post("/calls/join", json={"join_code": "nope00"}, headers=_h(THIRD))
        assert res.status_code == 404
        res = env.client.post("/calls/join", json={"join_code": code}, headers=_h(PEER))
        assert res.status_code == 200 and res.json()["call_id"] == cid

    def test_end_is_participant_only_and_idempotent(self, env):
        created = _create(env, invitee_email=EMAILS[PEER])
        cid = created["call_id"]
        # The invitee can SEE the call but has not joined → cannot end it.
        assert env.client.post(f"/calls/{cid}/end", headers=_h(PEER)).status_code == 403
        assert env.client.post(f"/calls/{cid}/end", headers=_h(THIRD)).status_code == 404
        res = env.client.post(f"/calls/{cid}/end", headers=_h(HOST))
        assert res.status_code == 200
        assert res.json()["status"] == "ended" and res.json()["end_reason"] == "ended"
        assert env.client.post(f"/calls/{cid}/end", headers=_h(HOST)).json()["status"] == "ended"
        # Nobody joins an ended call.
        assert _join(env, cid, PEER)[0] == 410
        # Nothing was said → no episode for anyone.
        assert env.store._by_uid == {}

    def test_expired_open_call(self, env):
        created = _create(env, invitee_email=EMAILS[PEER], ttl_minutes=1)
        call = calls.registry.get(created["call_id"])
        call.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        status, body = _join(env, created["call_id"], PEER)
        assert status == 410 and "expired" in body["detail"]
        assert env.client.get(f"/calls/{created['call_id']}", headers=_h(HOST)).json()["end_reason"] == "expired"

    def test_unknown_or_malformed_ids(self, env):
        assert env.client.get(f"/calls/{uuid.uuid4()}", headers=_h(HOST)).status_code == 404
        assert env.client.get("/calls/not-a-uuid", headers=_h(HOST)).status_code == 422

    def test_calls_require_auth(self, env):
        override = app.dependency_overrides.pop(main.get_current_uid, None)
        try:
            assert env.client.post("/calls", json={}).status_code == 401
        finally:
            if override is not None:
                app.dependency_overrides[main.get_current_uid] = override


# ---------------------------------------------------------------------------
# Binding + signaling over the session WebSocket
# ---------------------------------------------------------------------------

class TestCallSignaling:
    def test_call_join_binds_and_broadcasts_state(self, env):
        created = _create(env, invitee_email=EMAILS[PEER], display_name="Sage")
        cid = created["call_id"]
        _join(env, cid, PEER, display_name="Mom")
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host:
            first = _bind(host, cid)
            assert first["type"] == "call_state" and first["call_id"] == cid
            assert first["self_label"] == "Speaker A" and first["peer_label"] == "Speaker B"
            assert _by_uid(first)[HOST]["connected"] is True
            assert _by_uid(first)[PEER] == {
                "uid": PEER, "slot": "B", "label": "Speaker B", "role": "participant", "display_name": "Mom",
                "is_self": False, "connected": False, "joined_at": _by_uid(first)[PEER]["joined_at"],
            }
            assert first["ice_servers"][0]["urls"] == [calls.DEFAULT_STUN_URL]
            with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
                peer_state = _bind(peer, cid)
                assert _by_uid(peer_state)[HOST]["display_name"] == "Sage"
                assert _by_uid(peer_state)[HOST]["is_self"] is False
                assert _by_uid(peer_state)[PEER]["is_self"] is True
                assert all(p["connected"] for p in peer_state["participants"])
                # The host is told the peer arrived.
                second, _ = recv_until(host, lambda m: m.get("type") == "call_state")
                assert _by_uid(second)[PEER]["connected"] is True
        # REST agrees, and the socket bindings are gone.
        state = env.client.get(f"/calls/{cid}", headers=_h(HOST)).json()
        assert state["status"] == "ended"  # both left → the (empty) call ended
        assert all(not p["connected"] for p in state["participants"])

    def test_call_join_can_admit_with_the_code(self, env):
        created = _create(env)
        cid, code = created["call_id"], created["join_code"]
        with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            peer.send_text(json.dumps({"type": "call_join", "call_id": cid}))
            assert json.loads(peer.receive_text()) == {"error": "call_join: join code does not match"}
            state = _bind(peer, cid, join_code=code.lower())
            assert state["self_label"] == "Speaker B" and state["status"] == "active"

    def test_call_join_errors_are_frames_not_closes(self, env):
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as ws:
            ws.send_text(json.dumps({"type": "call_join", "call_id": "nope"}))
            assert json.loads(ws.receive_text()) == {"error": "call_join: invalid call_id"}
            ws.send_text(json.dumps({"type": "call_join", "call_id": str(uuid.uuid4())}))
            assert json.loads(ws.receive_text()) == {"error": "call_join: no such call"}
            # A signal outside any call is refused, and the session lives on.
            ws.send_text(json.dumps({"type": "rtc_signal", "call_id": str(uuid.uuid4()), "payload": {"sdp": "x"}}))
            assert json.loads(ws.receive_text()) == {"error": "rtc_signal: not in that call"}
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 10}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_rtc_signal_relayed_both_ways(self, env):
        created = _create(env, invitee_email=EMAILS[PEER])
        cid = created["call_id"]
        _join(env, cid, PEER)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            # Nobody to deliver to yet — say so rather than dropping silently.
            host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"sdp": "offer-0"}}))
            assert json.loads(host.receive_text()) == {"error": "rtc_signal: peer not connected"}
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            offer = {"sdp": "v=0 offer", "type": "offer"}
            host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": offer}))
            got, _ = recv_until(peer, lambda m: m.get("type") == "rtc_signal")
            assert got == {"type": "rtc_signal", "call_id": cid, "from": HOST, "payload": offer}
            answer = {"sdp": "v=0 answer", "type": "answer"}
            peer.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "to": HOST, "payload": answer}))
            got, _ = recv_until(host, lambda m: m.get("type") == "rtc_signal")
            assert got == {"type": "rtc_signal", "call_id": cid, "from": PEER, "payload": answer}
            candidate = {"candidate": "candidate:1 1 udp 2 10.0.0.1 5000 typ host", "sdpMid": "0"}
            peer.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": candidate}))
            assert recv_until(host, lambda m: m.get("type") == "rtc_signal")[0]["payload"] == candidate
            # Malformed payloads are refused with a reason.
            peer.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": "sdp"}))
            assert json.loads(peer.receive_text())["error"].startswith("rtc_signal: payload must be")
            peer.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "to": PEER, "payload": {"x": 1}}))
            assert json.loads(peer.receive_text()) == {"error": "rtc_signal: peer has not joined"}
            peer.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"blob": "x" * (calls.RTC_PAYLOAD_MAX_BYTES + 1)}}))
            assert json.loads(peer.receive_text()) == {"error": "rtc_signal: payload too large"}


# ---------------------------------------------------------------------------
# The merged transcript + per-participant coaching
# ---------------------------------------------------------------------------

def _open_pair(env, *, host_name: str | None = None, peer_name: str | None = None):
    """A bound host + peer; returns (call_id, host_ws_ctx, peer_ws_ctx) —
    the caller enters both context managers."""
    created = _create(env, invitee_email=EMAILS[PEER], display_name=host_name)
    cid = created["call_id"]
    _join(env, cid, PEER, display_name=peer_name)
    return cid


class TestMergedTranscript:
    def test_turn_reaches_peer_with_slot_labels_and_both_are_coached(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            # The host's phone labelled its own voice "Speaker Q" and had no
            # verdict — structurally it is Speaker A and self.
            host.send_text(json.dumps(_turn(
                HOST_SID, "You never call me back.", speaker="Speaker Q", start=3.0, end=4.5,
                text_tone={"frustration": 70, "label": "angry"}, prosody={"rms_dbfs": -12.0},
            )))
            nudge, seen = recv_until(host, lambda m: m.get("type") == "suggestion")
            assert nudge["kind"] == "nudge" and nudge["speaker"] == "Speaker A"
            assert nudge["suggestions"] == ["ease up"]
            assert not [m for m in seen if m.get("type") == "transcript"]  # no echo of your own turn
            transcript, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
            assert transcript == {
                "type": "transcript", "session_id": PEER_SID, "speaker": "Speaker A",
                "display_name": EMAILS[HOST], "role": "participant", "text": "You never call me back.",
                "start_time": transcript["start_time"], "end_time": transcript["end_time"],
                "call_id": cid, "participant_uid": HOST, "is_self": False, "seq": 1,
                "local_start_time": 3.0, "local_end_time": 4.5,
                "text_tone": {"warmth": None, "defensiveness": None, "sarcasm": None, "sadness": None,
                              "frustration": 70, "label": "angry"},
                "prosody": {"rms_dbfs": -12.0, "pitch_hz": None, "speech_rate": None},
            }
            # Re-based onto the call timeline: same length; the offset is the
            # sender's clock lag behind the call clock (never negative).
            assert transcript["end_time"] - transcript["start_time"] == pytest.approx(1.5, abs=1e-3)
            assert transcript["start_time"] >= 3.0
            response, _ = recv_until(peer, lambda m: m.get("type") == "suggestion")
            assert response["kind"] == "response" and response["speaker"] == "Speaker A"
            assert response["utterance_text"] == "You never call me back."
            assert len(response["suggestions"]) == 3
            # And back: the peer's turn is Speaker B for the host.
            peer.send_text(json.dumps(_turn(PEER_SID, "I do call.", speaker="Speaker A", start=0.0, end=0.8)))
            t2, _ = recv_until(host, lambda m: m.get("type") == "transcript")
            assert t2["speaker"] == "Speaker B" and t2["display_name"] == EMAILS[PEER] and t2["seq"] == 2
            r2, _ = recv_until(host, lambda m: m.get("type") == "suggestion")
            assert r2["kind"] == "response" and r2["speaker"] == "Speaker B"
            recv_until(peer, lambda m: m.get("type") == "suggestion" and m["kind"] == "nudge")
        # The coach was told who spoke and got the sender's tone hints.
        prompts = env.llm.user_prompts
        assert any(p.startswith(f'Transcript turn from {EMAILS[HOST]}: "You never call me back."') and "frustration 70/100" in p and "loudness -12.0 dBFS" in p for p in prompts)
        assert any(p.startswith(f'Transcript turn from {EMAILS[PEER]}: "I do call."') for p in prompts)
        call = calls.registry.get(cid)
        assert [t["participant_uid"] for t in call.turns] == [HOST, PEER]
        assert [t["speaker"] for t in call.turns] == ["Speaker A", "Speaker B"]
        assert call.turns[0]["local_start_time"] == 3.0 and call.turns[0]["text_tone"]["frustration"] == 70

    def test_speaker_label_names_the_peer_call_wide(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            # The peer names the host's voice — and claims it is themself
            # (nonsense in a call): the name sticks, the self claim does not.
            peer.send_text(json.dumps({
                "type": "speaker_label", "speaker": "Speaker A", "display_name": "Sage",
                "person_id": "sage", "is_self": True,
            }))
            ack = json.loads(peer.receive_text())
            assert ack["type"] == "speaker_label_ack" and ack["display_name"] == "Sage"
            state, _ = recv_until(peer, lambda m: m.get("type") == "call_state")
            assert _by_uid(state)[HOST]["display_name"] == "Sage"
            # Naming is per viewer: the host still sees itself as "You" and
            # its own state carries the peer's default name.
            host_state, _ = recv_until(host, lambda m: m.get("type") == "call_state")
            assert _by_uid(host_state)[HOST]["display_name"] == "You"
            assert _by_uid(host_state)[PEER]["display_name"] == EMAILS[PEER]
            host.send_text(json.dumps(_turn(HOST_SID, "Hi Mom.")))
            transcript, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
            assert transcript["display_name"] == "Sage"
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            # The peer's own turns are still ITS slot label / self.
            peer.send_text(json.dumps(_turn(PEER_SID, "Hi honey.")))
            own, _ = recv_until(peer, lambda m: m.get("type") == "suggestion")
            assert own["kind"] == "nudge" and own["speaker"] == "Speaker B"
            # A real name on your OWN label is self-declared: the other side sees it.
            host.send_text(json.dumps({"type": "speaker_label", "speaker": "Speaker A", "display_name": "Dad", "is_self": True}))
            recv_until(host, lambda m: m.get("type") == "speaker_label_ack")
            recv_until(host, lambda m: m.get("type") == "call_state")
            # …but the peer's own naming wins over the self-declared one.
            state, _ = recv_until(peer, lambda m: m.get("type") == "call_state")
            assert _by_uid(state)[HOST]["display_name"] == "Sage"
        assert any(p.startswith('Transcript turn from Sage: "Hi Mom."') for p in env.llm.user_prompts)
        call = calls.registry.get(cid)
        assert call.names == {PEER: {HOST: "Sage"}}
        assert call.participant(HOST).declared_name == "Dad"

    def test_server_stt_fallback_turn_is_the_participants_own(self, env):
        """A participant whose phone has no on-device STT streams audio; the
        server's transcriber segment is structurally THEIR turn."""
        cid = _open_pair(env)
        app.state.transcriber_factory = lambda: StoppableTranscriber(
            live=[TranscriptSegment("Server heard this.", 1.0, 2.0, speaker=1)],
        )
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            host.send_bytes(b"\x00" * 3200)
            echo, _ = recv_until(host, lambda m: m.get("type") == "transcript")
            assert echo["speaker"] == "Speaker A"  # not the diarizer's "Speaker B"
            nudge, _ = recv_until(host, lambda m: m.get("type") == "suggestion")
            assert nudge["kind"] == "nudge"
            remote, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
            assert remote["speaker"] == "Speaker A" and remote["text"] == "Server heard this."
            assert remote["local_start_time"] == 1.0 and remote["local_end_time"] == 2.0
        call = calls.registry.get(cid)
        assert call.turns[0]["transcript_source"] == "cloud"


# ---------------------------------------------------------------------------
# Ending: episodes, sharing, disconnects
# ---------------------------------------------------------------------------

def _seed_link(store: FakeStore, patient: str, therapist: str) -> None:
    import therapist_links
    store._links[patient] = therapist_links.new_link(
        patient_uid=patient, patient_email=EMAILS[patient],
        therapist_uid=therapist, therapist_email=EMAILS[therapist],
    )


class TestCallEnd:
    def test_both_stop_persists_one_episode_per_participant(self, env):
        _seed_link(env.store, HOST, PEER)  # the peer IS the host's therapist
        cid = _open_pair(env, peer_name="Mom")
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            host.send_text(json.dumps(_turn(HOST_SID, "One.", start=0.0, end=1.0, text_tone={"frustration": 80, "label": "angry"})))
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            peer.send_text(json.dumps(_turn(PEER_SID, "Two.", start=0.0, end=1.0)))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            host.send_text(json.dumps(_turn(HOST_SID, "Three.", start=2.0, end=3.0)))
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            peer.send_text(json.dumps(_turn(PEER_SID, "Four.", start=2.0, end=3.0)))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            # The peer hangs up first: the host is told, the call goes on.
            peer.send_text(json.dumps({"type": "stop"}))
            done, seen = recv_until(peer, lambda m: m.get("type") == "session_complete")
            assert not [m for m in seen if m.get("type") == "call_ended"]
            assert done["call"] == {"call_id": cid, "status": "active", "episode_id": None}
            state, _ = recv_until(host, lambda m: m.get("type") == "call_state")
            assert _by_uid(state)[PEER]["connected"] is False and state["status"] == "active"
            # The host keeps coaching solo …
            host.send_text(json.dumps(_turn(HOST_SID, "Five, alone.", start=4.0, end=5.0)))
            assert recv_until(host, lambda m: m.get("type") == "suggestion")[0]["kind"] == "nudge"
            # … and its stop ends the call: call_ended (with its episode) precedes session_complete.
            host.send_text(json.dumps({"type": "stop"}))
            ended, seen = recv_until(host, lambda m: m.get("type") == "call_ended")
            done = json.loads(host.receive_text())
            assert done["type"] == "session_complete"
        assert ended["reason"] == "all participants left" and ended["ended_by"] == HOST
        assert ended["turn_count"] == 5 and ended["shared_with"] == [EMAILS[PEER]]
        assert done["call"] == {"call_id": cid, "status": "ended", "episode_id": ended["episode_id"]}

        host_rec = env.store._by_uid[HOST][ended["episode_id"]]
        [peer_rid] = list(env.store._by_uid[PEER])
        peer_rec = env.store._by_uid[PEER][peer_rid]
        for rec, own_label in ((host_rec, "Speaker A"), (peer_rec, "Speaker B")):
            meta, turns, analysis = rec["meta"], rec["turns"], rec["analysis"]
            assert meta["mode"] == "call" and meta["session_id"] == f"call-{cid}"
            assert meta["source"]["type"] == "live" and meta["media_type"] == "none"
            assert [t["text"] for t in turns] == ["One.", "Two.", "Three.", "Four.", "Five, alone."]
            assert [t["speaker"] for t in turns] == ["Speaker A", "Speaker B", "Speaker A", "Speaker B", "Speaker A"]
            assert [t["is_self"] for t in turns] == [sp == own_label for sp in (t["speaker"] for t in turns)]
            assert [t["call_seq"] for t in turns] == [1, 2, 3, 4, 5]
            assert [t["participant_uid"] for t in turns] == [HOST, PEER, HOST, PEER, HOST]
            assert turns[2]["local_start_time"] == 2.0 and turns[2]["start_time"] >= 2.0
            assert all((t["speaker_person_id"] == "self") is t["is_self"] for t in turns)
            assert analysis["live"]["mode"] == "call"
            assert analysis["live"]["self_speaker"] == own_label
            assert analysis["speaker_labels"][own_label]["display_label"] == "You"
        assert host_rec["meta"]["title"] == "Call with Mom"
        assert host_rec["meta"]["manual_speaker_labels"] == {"Speaker A": "You", "Speaker B": "Mom"}
        assert host_rec["meta"]["manual_speaker_people"] == {"Speaker A": "self"}
        assert peer_rec["meta"]["title"] == f"Call with {EMAILS[HOST]}"
        assert peer_rec["meta"]["manual_speaker_labels"] == {"Speaker B": "You", "Speaker A": EMAILS[HOST]}
        # The host's angry turn is a self escalation on ITS episode only.
        assert host_rec["analysis"]["live"]["tone_summary"]["self"]["escalation_turns"] == [0]
        assert peer_rec["analysis"]["live"]["tone_summary"]["self"]["escalation_turns"] == []
        # Auto-share through the therapist link: the peer sees the host's episode.
        assert [s["uid"] for s in host_rec["meta"]["shares"]] == [PEER]
        assert "shares" not in peer_rec["meta"]
        # REST tells each participant its own episode.
        assert env.client.get(f"/calls/{cid}", headers=_h(HOST)).json()["episode_id"] == ended["episode_id"]
        assert env.client.get(f"/calls/{cid}", headers=_h(PEER)).json()["episode_id"] == peer_rid
        assert ended["episode_id"] != peer_rid

    def test_rest_end_notifies_both_sockets_and_solo_coaching_continues(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            host.send_text(json.dumps(_turn(HOST_SID, "Before the end.")))
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            recv_until(host, lambda m: m.get("type") == "suggestion")
            res = env.client.post(f"/calls/{cid}/end", headers=_h(PEER))
            assert res.status_code == 200 and res.json()["status"] == "ended"
            for ws, uid in ((host, HOST), (peer, PEER)):
                ended, _ = recv_until(ws, lambda m: m.get("type") == "call_ended")
                assert ended["reason"] == "ended" and ended["ended_by"] == PEER
                assert ended["episode_id"] in env.store._by_uid[uid]
            # After the call each session is solo again: the host's turn is
            # not relayed, still coached (self_speaker stayed "Speaker A").
            host.send_text(json.dumps(_turn(HOST_SID, "After.", speaker="Speaker A")))
            assert recv_until(host, lambda m: m.get("type") == "suggestion")[0]["kind"] == "nudge"
            peer.send_text(json.dumps({"type": "config", "empathy_slider": 5}))
            assert json.loads(peer.receive_text())["type"] == "config_ack"  # nothing was relayed
            host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"x": 1}}))
            assert json.loads(host.receive_text()) == {"error": "rtc_signal: not in that call"}
        assert len(calls.registry.get(cid).turns) == 1

    def test_abrupt_disconnect_updates_peer_then_last_leaver_ends(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(peer, cid)
            with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host:
                _bind(host, cid)
                recv_until(peer, lambda m: m.get("type") == "call_state")
                host.send_text(json.dumps(_turn(HOST_SID, "Host said this.")))
                recv_until(peer, lambda m: m.get("type") == "suggestion")
                recv_until(host, lambda m: m.get("type") == "suggestion")
            # The host's socket dropped with no stop.
            state, _ = recv_until(peer, lambda m: m.get("type") == "call_state")
            assert _by_uid(state)[HOST]["connected"] is False and state["status"] == "active"
            peer.send_text(json.dumps(_turn(PEER_SID, "Still here.")))
            assert recv_until(peer, lambda m: m.get("type") == "suggestion")[0]["kind"] == "nudge"
            peer.send_text(json.dumps({"type": "stop"}))
            ended, _ = recv_until(peer, lambda m: m.get("type") == "call_ended")
            assert ended["ended_by"] == PEER and ended["turn_count"] == 2
        # Both episodes carry both turns, including the host's from before it dropped.
        for uid in (HOST, PEER):
            [rec] = env.store._by_uid[uid].values()
            assert [t["text"] for t in rec["turns"]] == ["Host said this.", "Still here."]

    def test_reconnect_replaces_the_socket(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(peer, cid)
            with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host1, \
                    open_ws(env.client, f"/ws/session/{HOST_SID}-2", token=HOST_TOKEN) as host2:
                _bind(host1, cid)
                recv_until(peer, lambda m: m.get("type") == "call_state")
                _bind(host2, cid)
                peer.send_text(json.dumps(_turn(PEER_SID, "To the new phone.")))
                got, _ = recv_until(host2, lambda m: m.get("type") == "transcript")
                assert got["text"] == "To the new phone."
                # The replaced socket is detached: its later turn is solo.
                host1.send_text(json.dumps(_turn(HOST_SID, "Stale.")))
                recv_until(host1, lambda m: m.get("type") == "suggestion")
        assert [t["text"] for t in calls.registry.get(cid).turns] == ["To the new phone."]

    def test_no_store_ends_without_episodes(self, env):
        delattr(app.state, "recordings_store")
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host:
            _bind(host, cid)
            host.send_text(json.dumps(_turn(HOST_SID, "Nobody stores this.")))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            host.send_text(json.dumps({"type": "stop"}))
            ended, _ = recv_until(host, lambda m: m.get("type") == "call_ended")
        assert ended["episode_id"] is None and ended["turn_count"] == 1

    def test_analysis_and_reflection_are_scheduled_on_the_episodes(self, env, monkeypatch):
        monkeypatch.setattr(calls, "ANALYZE_ON_END", True)
        monkeypatch.setattr(calls, "REFLECT_ON_END", True)
        from routers import sessions as sessions_router
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            for i in range(4):
                ws, other, sid = (host, peer, HOST_SID) if i % 2 == 0 else (peer, host, PEER_SID)
                ws.send_text(json.dumps(_turn(sid, f"Turn {i}.", start=float(i), end=i + 0.5)))
                recv_until(other, lambda m: m.get("type") == "suggestion")
                recv_until(ws, lambda m: m.get("type") == "suggestion")
            res = env.client.post(f"/calls/{cid}/end", headers=_h(HOST))
            assert res.status_code == 200
            # The two LLM passes run as background tasks on the socket's
            # loop — wait for them while the sockets (and the loop) are open.
            import time
            for _ in range(500):
                if not sessions_router.BACKGROUND_TASKS:
                    break
                time.sleep(0.02)
            assert not sessions_router.BACKGROUND_TASKS
        for uid in (HOST, PEER):
            [rec] = env.store._by_uid[uid].values()
            live = rec["analysis"]["live"]
            assert live["analysis_status"] == "full", live
            assert len(live["could_have_said"]) == 2  # two own turns each


# ---------------------------------------------------------------------------
# Three-way: two participants + an observing therapist
# ---------------------------------------------------------------------------

THER, THER_TOKEN, THER_SID = THIRD, THIRD_TOKEN, "call-ther-session"


def _drain_state(ws, n_members: int) -> dict:
    """The call_state that shows every member connected."""
    state, _ = recv_until(
        ws, lambda m: m.get("type") == "call_state" and len(m.get("participants") or []) == n_members
        and all(p["connected"] for p in m["participants"]),
    )
    return state


def _open_three(env):
    created = _create(env, invitee_email=EMAILS[PEER], display_name="Sage")
    cid, code = created["call_id"], created["join_code"]
    assert _join(env, cid, PEER, display_name="Dad")[0] == 200
    status, body = _join(env, cid, THER, join_code=code, role="therapist", display_name="Mom")
    assert status == 200, body
    return cid


class TestThreeWay:
    def test_role_seats_and_caps_are_enforced(self, env):
        created = _create(env, invitee_email=EMAILS[PEER], display_name="Sage")
        cid, code = created["call_id"], created["join_code"]
        _join(env, cid, PEER, display_name="Dad")
        status, body = _join(env, cid, THER, join_code=code, role="therapist", display_name="Mom")
        assert status == 200, body
        assert body["status"] == "active" and body["self_role"] == "therapist"
        assert body["self_label"] == "Speaker C" and body["peer_label"] == "Speaker A"
        assert body["therapist_uid"] == THER and body["therapist_label"] == "Speaker C"
        rows = _by_uid(body)
        assert rows[HOST]["display_name"] == "Sage" and rows[HOST]["role"] == "participant"
        assert rows[PEER]["display_name"] == "Dad" and rows[PEER]["slot"] == "B"
        assert rows[THER]["display_name"] == "You" and rows[THER]["role"] == "therapist"
        # Everyone else sees her as the therapist.
        host_view = env.client.get(f"/calls/{cid}", headers=_h(HOST)).json()
        assert _by_uid(host_view)[THER]["display_name"] == "Mom (therapist)"
        # Seats: a fourth member of either role is refused; the call is full.
        assert _join(env, cid, "user-x", join_code=code, role="participant")[0] == 409
        assert _join(env, cid, "user-x", join_code=code, role="therapist")[0] == 409
        # A second therapist while a participant seat is free: the role's seat is taken.
        created2 = _create(env, uid=PEER, display_name="Dad")
        cid2, code2 = created2["call_id"], created2["join_code"]
        assert _join(env, cid2, THER, join_code=code2, role="therapist")[0] == 200
        status, body = _join(env, cid2, "user-x", join_code=code2, role="therapist")
        assert status == 409 and "therapist" in body["detail"]
        assert _join(env, cid2, "user-x", join_code=code2, role="bogus")[0] == 422
        assert _join(env, cid2, HOST, join_code=code2)[0] == 200  # the last participant seat
        # A two-member call has no room for a therapist.
        created3 = _create(env, uid=HOST, max_participants=2)
        cid3, code3 = created3["call_id"], created3["join_code"]
        assert created3["max_participants"] == 2
        assert _join(env, cid3, THER, join_code=code3, role="therapist")[0] == 200
        assert _join(env, cid3, PEER, join_code=code3)[0] == 409
        assert env.client.post("/calls", json={"max_participants": 4}, headers=_h(HOST)).status_code == 422
        assert env.client.post("/calls", json={"max_participants": 1}, headers=_h(HOST)).status_code == 422

    def test_mesh_relay_merged_transcript_and_readonly_fan_out(self, env):
        cid = _open_three(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer, \
                open_ws(env.client, f"/ws/session/{THER_SID}", token=THER_TOKEN) as ther:
            _bind(host, cid)
            _bind(peer, cid)
            ther_state = _bind(ther, cid, role="therapist")
            _drain_state(host, 3)
            _drain_state(peer, 3)
            assert ther_state["self_role"] == "therapist" and ther_state["self_label"] == "Speaker C"
            assert _by_uid(ther_state)[HOST]["display_name"] == "Sage"
            # --- signaling: full mesh, `to` required -------------------------
            host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"sdp": "x"}}))
            assert json.loads(host.receive_text()) == {"error": "rtc_signal: 'to' is required in a call with more than two members"}
            for src, dst_ws, dst_uid, src_uid in ((host, ther, THER, HOST), (ther, peer, PEER, THER), (peer, host, HOST, PEER)):
                offer = {"type": "offer", "sdp": f"from {src_uid}"}
                src.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "to": dst_uid, "payload": offer}))
                got, _ = recv_until(dst_ws, lambda m: m.get("type") == "rtc_signal")
                assert got == {"type": "rtc_signal", "call_id": cid, "from": src_uid, "payload": offer}
            # --- the host speaks: both others see it, only the peer is coached --
            host.send_text(json.dumps(_turn(HOST_SID, "Dad, you never listen.", text_tone={"frustration": 75, "label": "angry"})))
            t_peer, _ = recv_until(peer, lambda m: m.get("type") == "transcript")
            assert t_peer["speaker"] == "Speaker A" and t_peer["display_name"] == "Sage" and t_peer["role"] == "participant"
            t_ther, _ = recv_until(ther, lambda m: m.get("type") == "transcript")
            assert t_ther["speaker"] == "Speaker A" and t_ther["display_name"] == "Sage"
            assert t_ther["text_tone"]["frustration"] == 75
            nudge, _ = recv_until(host, lambda m: m.get("type") == "suggestion")
            assert nudge["kind"] == "nudge" and "for_uid" not in nudge
            response, _ = recv_until(peer, lambda m: m.get("type") == "suggestion")
            assert response["kind"] == "response" and response["speaker"] == "Speaker A" and "for_uid" not in response
            # The therapist gets BOTH coaching events read-only, tagged.
            copies = []
            for _ in range(2):
                c, _ = recv_until(ther, lambda m: m.get("type") == "suggestion")
                copies.append(c)
            by_uid = {c["for_uid"]: c for c in copies}
            assert by_uid[HOST]["kind"] == "nudge" and by_uid[HOST]["suggestions"] == ["ease up"]
            assert by_uid[PEER]["kind"] == "response" and by_uid[PEER]["utterance_text"] == "Dad, you never listen."
            # --- the therapist speaks: merged for everyone, coached for both participants, never for her --
            ther.send_text(json.dumps(_turn(THER_SID, "Let's slow down.", speaker="Speaker A")))
            for ws in (host, peer):
                t, _ = recv_until(ws, lambda m: m.get("type") == "transcript")
                assert t["speaker"] == "Speaker C" and t["display_name"] == "Mom (therapist)" and t["role"] == "therapist"
                s, _ = recv_until(ws, lambda m: m.get("type") == "suggestion" and "for_uid" not in m)
                assert s["kind"] == "response" and s["speaker"] == "Speaker C"
            # Her socket sees the two participants' suggestions about her turn, and nothing of her own.
            seen = []
            for _ in range(2):
                c, _ = recv_until(ther, lambda m: m.get("type") == "suggestion")
                seen.append(c)
            assert sorted(c["for_uid"] for c in seen) == sorted([HOST, PEER])
            assert all(c["utterance_text"] == "Let's slow down." for c in seen)
            ther.send_text(json.dumps({"type": "config", "empathy_slider": 42}))
            assert json.loads(ther.receive_text())["type"] == "config_ack"  # nothing else was queued for her
            # --- the peer names the therapist; the host's naming is separate ----
            peer.send_text(json.dumps({"type": "speaker_label", "speaker": "Speaker C", "display_name": "Linda"}))
            recv_until(peer, lambda m: m.get("type") == "speaker_label_ack")
            st, _ = recv_until(peer, lambda m: m.get("type") == "call_state")
            assert _by_uid(st)[THER]["display_name"] == "Linda (therapist)"
            st, _ = recv_until(host, lambda m: m.get("type") == "call_state")
            assert _by_uid(st)[THER]["display_name"] == "Mom (therapist)"
            recv_until(ther, lambda m: m.get("type") == "call_state")
        prompts = env.llm.user_prompts
        assert any(p.startswith('Transcript turn from Mom (therapist): "Let\'s slow down."') for p in prompts)
        call = calls.registry.get(cid)
        assert [(t["speaker"], t["role"]) for t in call.turns] == [("Speaker A", "participant"), ("Speaker C", "therapist")]

    def test_end_persists_participant_episodes_shared_with_the_in_call_therapist(self, env):
        cid = _open_three(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer, \
                open_ws(env.client, f"/ws/session/{THER_SID}", token=THER_TOKEN) as ther:
            _bind(host, cid)
            _bind(peer, cid)
            _bind(ther, cid, role="therapist")
            _drain_state(host, 3)
            _drain_state(peer, 3)
            host.send_text(json.dumps(_turn(HOST_SID, "One.", start=0.0, end=1.0)))
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            recv_until(host, lambda m: m.get("type") == "suggestion")
            ther.send_text(json.dumps(_turn(THER_SID, "Two.", start=0.0, end=1.0)))
            recv_until(peer, lambda m: m.get("type") == "suggestion" and "for_uid" not in m)
            recv_until(host, lambda m: m.get("type") == "suggestion" and "for_uid" not in m)
            peer.send_text(json.dumps(_turn(PEER_SID, "Three.", start=2.0, end=3.0)))
            recv_until(host, lambda m: m.get("type") == "suggestion" and "for_uid" not in m)
            recv_until(peer, lambda m: m.get("type") == "suggestion" and "for_uid" not in m)
            res = env.client.post(f"/calls/{cid}/end", headers=_h(HOST))
            assert res.status_code == 200
            ended_host, _ = recv_until(host, lambda m: m.get("type") == "call_ended")
            ended_peer, _ = recv_until(peer, lambda m: m.get("type") == "call_ended")
            ended_ther, _ = recv_until(ther, lambda m: m.get("type") == "call_ended")
        assert ended_ther["episode_id"] is None
        assert ended_ther["episodes"] == {HOST: ended_host["episode_id"], PEER: ended_peer["episode_id"]}
        assert ended_host["shared_with"] == [EMAILS[THER]] and ended_peer["shared_with"] == [EMAILS[THER]]
        assert THER not in env.store._by_uid  # the observer gets no episode of her own
        for uid, own_label, ended in ((HOST, "Speaker A", ended_host), (PEER, "Speaker B", ended_peer)):
            rec = env.store._by_uid[uid][ended["episode_id"]]
            turns = rec["turns"]
            assert [t["speaker"] for t in turns] == ["Speaker A", "Speaker C", "Speaker B"]
            assert [t["is_self"] for t in turns] == [sp == own_label for sp in (t["speaker"] for t in turns)]
            assert rec["analysis"]["live"]["self_speaker"] == own_label and rec["meta"]["mode"] == "call"
            assert rec["meta"]["manual_speaker_labels"]["Speaker C"] == "Mom (therapist)"
            # Shared directly with the therapist who was on the call — no link needed.
            assert [s["uid"] for s in rec["meta"]["shares"]] == [THER]
        assert env.store._by_uid[HOST][ended_host["episode_id"]]["meta"]["title"] == "Call with Dad and Mom (therapist)"
        assert env.store._by_uid[HOST][ended_host["episode_id"]]["meta"]["manual_speaker_labels"]["Speaker B"] == "Dad"
        # Her dashboard lists both participants' episodes.
        res = env.client.get("/sessions", headers=_h(THER))
        rows = res.json()["sessions"]
        assert sorted(r["role"] for r in rows) == sorted([EMAILS[HOST], EMAILS[PEER]])
        assert all(r["shared"] and r["mode"] == "call" for r in rows)

    def test_link_and_in_call_therapist_share_once(self, env):
        _seed_link(env.store, HOST, THER)  # the therapist on the call IS the linked one
        cid = _open_three(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{THER_SID}", token=THER_TOKEN) as ther:
            _bind(host, cid)
            _bind(ther, cid)
            recv_until(host, lambda m: m.get("type") == "call_state" and _by_uid(m)[THER]["connected"])
            host.send_text(json.dumps(_turn(HOST_SID, "Only me and Mom.")))
            recv_until(host, lambda m: m.get("type") == "suggestion")
            # The host hangs up; the observer is still on, so the call goes on…
            host.send_text(json.dumps({"type": "stop"}))
            done, seen = recv_until(host, lambda m: m.get("type") == "session_complete")
            assert done["call"]["status"] == "active" and not [m for m in seen if m.get("type") == "call_ended"]
            recv_until(ther, lambda m: m.get("type") == "call_state" and not _by_uid(m)[HOST]["connected"])
            # …until she leaves too: the last socket out ends it.
            ther.send_text(json.dumps({"type": "stop"}))
            ended, _ = recv_until(ther, lambda m: m.get("type") == "call_ended")
        assert ended["episode_id"] is None and ended["ended_by"] == THER
        host_ep = ended["episodes"][HOST]
        rec = env.store._by_uid[HOST][host_ep]
        # Linked AND on the call: granted once, not twice.
        assert [s["uid"] for s in rec["meta"]["shares"]] == [THER]
        assert env.client.get(f"/calls/{cid}", headers=_h(HOST)).json()["shared_with"] == [EMAILS[THER]]
        assert PEER in env.store._by_uid and len(env.store._by_uid[PEER]) == 1  # Dad's episode too (he joined, never connected)


# ---------------------------------------------------------------------------
# Pure model behaviour
# ---------------------------------------------------------------------------

class TestCallModel:
    def test_turns_for_views_and_caps(self):
        now = datetime.now(timezone.utc)
        call = calls.Call(
            call_id="c", host_uid="a", join_code="ABCDEF", created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=1)).isoformat(),
        )
        call.add_participant("a", email=None, display_name=None)
        call.add_participant("b", email="b@x", display_name=None)
        call.turns = [
            {"seq": 1, "participant_uid": "a", "slot": "A", "speaker": "Speaker A", "text": "x" * 10,
             "start_time": 0.0, "end_time": 1.0, "local_start_time": 0.0, "local_end_time": 1.0,
             "transcript_source": "on-device", "speaker_match_score": 0.9},
            {"seq": 2, "participant_uid": "b", "slot": "B", "speaker": "Speaker B", "text": "y" * 10,
             "start_time": 1.0, "end_time": 2.0, "local_start_time": 5.0, "local_end_time": 6.0,
             "transcript_source": "on-device", "speaker_match_score": 0.8},
        ]
        a_view = call.turns_for("a", "s")
        assert [(t["is_self"], t["speaker_person_id"], t["speaker_match_score"]) for t in a_view] == [
            (True, "self", 0.9), (False, None, None),
        ]
        b_view = call.turns_for("b", "s")
        assert [(t["is_self"], t["speaker_person_id"]) for t in b_view] == [(False, None), (True, "self")]
        assert b_view[1]["local_start_time"] == 5.0 and b_view[1]["start_time"] == 1.0
        assert call.display_name_for("a", "b") == "b@x" and call.display_name_for("b", "a") == "Speaker A"
        assert call.display_name_for("a", "a") == "You"
        # Character cap drops the OLDEST turns first.
        call.turns[0]["text"] = "x" * calls.CALL_MAX_TRANSCRIPT_CHARS
        assert [t["call_seq"] for t in call.turns_for("a", "s")] == [2]

    def test_join_code_helpers(self):
        code = calls.new_join_code()
        assert calls.normalize_join_code(" " + code.lower()[:3] + "-" + code[3:] + " ") == code
        assert calls.normalize_join_code("ABC") is None
        assert calls.normalize_join_code("ABCDE0") is None  # 0 is not in the alphabet
        assert calls.normalize_join_code(123) is None
        assert calls.clean_display_name("  Mom   Smith ") == "Mom Smith"
        assert calls.clean_display_name("   ") is None and calls.clean_display_name(3) is None
        assert len(calls.clean_display_name("x" * 100)) == calls.DISPLAY_NAME_MAX

    def test_registry_sweeps_expired_and_retained(self, monkeypatch):
        reg = calls.CallRegistry()
        call = reg.create("a")
        call.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        assert reg.get(call.call_id).status == "ended" and call.end_reason == "expired"
        monkeypatch.setattr(calls, "CALL_RETENTION_MINUTES", 0)
        call._ended_wall -= 1.0
        assert reg.get(call.call_id) is None and reg.by_code(call.join_code) is None
        assert len(reg) == 0

    def test_join_url_and_ice_defaults(self, monkeypatch):
        monkeypatch.delenv("MINDSHIFT_TURN_URLS", raising=False)
        assert calls.ice_servers() == [{"urls": [calls.DEFAULT_STUN_URL]}]
        monkeypatch.setenv("MINDSHIFT_CALL_JOIN_BASE", "mindshift://call/")
        assert calls.join_url("ABCDEF") == "mindshift://call/ABCDEF"


# ---------------------------------------------------------------------------
# Adversarial review (multi-tenant): registry growth, roles, brute force,
# relay flooding, reconnect clocks
# ---------------------------------------------------------------------------

class TestRegistryBounds:
    def test_active_call_nobody_ever_connected_expires_at_ttl(self, env):
        """Host creates, the invitee joins over REST, no socket ever binds:
        the call went ACTIVE and must still expire at its TTL — otherwise it
        lives in the registry forever."""
        created = _create(env, invitee_email=EMAILS[PEER], ttl_minutes=1)
        assert _join(env, created["call_id"], PEER)[0] == 200
        call = calls.registry.get(created["call_id"])
        assert call.status == "active"
        call.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        assert calls.registry.get(created["call_id"]).status == "ended"
        assert call.end_reason == "expired"
        assert _join(env, created["call_id"], THIRD, join_code=created["join_code"])[0] == 410

    def test_active_call_with_a_socket_never_expires_by_clock(self, env):
        created = _create(env, invitee_email=EMAILS[PEER], ttl_minutes=1)
        cid = created["call_id"]
        _join(env, cid, PEER)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host:
            _bind(host, cid)
            call = calls.registry.get(cid)
            call.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            assert calls.registry.get(cid).status == "active"

    def test_one_account_cannot_fill_the_registry(self, env, monkeypatch):
        """Retained ENDED calls must not count against everyone else, and one
        host has a cap on un-ended calls — a single tenant can't 503 the
        rest by creating calls in a loop."""
        monkeypatch.setattr(calls, "MAX_CALLS", 3)
        monkeypatch.setattr(calls, "MAX_OPEN_CALLS_PER_HOST", 2)
        c1 = _create(env)
        c2 = _create(env)
        res = env.client.post("/calls", json={}, headers=_h(HOST))
        assert res.status_code == 429, res.text  # the host's own cap
        # Ending one frees the seat.
        env.client.post(f"/calls/{c1['call_id']}/end", headers=_h(HOST))
        c3 = _create(env)
        # Another tenant: the retained ended call (c1) is evicted before a
        # live call is refused; two live calls (c2, c3) + this one = MAX_CALLS.
        c4 = _create(env, uid=PEER)
        assert calls.registry.get(c1["call_id"]) is None
        assert {c["call_id"] for c in (c2, c3, c4)} <= set(calls.registry._calls)
        # Only live calls are left and the global cap holds for a third tenant.
        assert env.client.post("/calls", json={}, headers=_h(THIRD)).status_code == 503


class TestRoleBoundaries:
    def test_observer_cannot_end_the_call_and_participants_do_not_see_each_others_episode_ids(self, env):
        cid = _open_three(env)
        # The therapist is a member, not a coached participant: she may leave,
        # but she cannot hang up for everyone.
        res = env.client.post(f"/calls/{cid}/end", headers=_h(THER))
        assert res.status_code == 403, res.text
        assert calls.registry.get(cid).status == "active"
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer, \
                open_ws(env.client, f"/ws/session/{THER_SID}", token=THER_TOKEN) as ther:
            _bind(host, cid)
            _bind(peer, cid)
            _bind(ther, cid)
            _drain_state(host, 3)
            _drain_state(peer, 3)
            host.send_text(json.dumps(_turn(HOST_SID, "One.")))
            recv_until(peer, lambda m: m.get("type") == "suggestion")
            recv_until(host, lambda m: m.get("type") == "suggestion")
            assert env.client.post(f"/calls/{cid}/end", headers=_h(PEER)).status_code == 200
            ended_host, _ = recv_until(host, lambda m: m.get("type") == "call_ended")
            ended_peer, _ = recv_until(peer, lambda m: m.get("type") == "call_ended")
            ended_ther, _ = recv_until(ther, lambda m: m.get("type") == "call_ended")
        # Each participant learns ITS episode; the map of everyone's is the observer's view only.
        assert ended_host["episode_id"] in env.store._by_uid[HOST]
        assert ended_peer["episode_id"] in env.store._by_uid[PEER]
        assert "episodes" not in ended_host and "episodes" not in ended_peer
        assert ended_ther["episodes"] == {HOST: ended_host["episode_id"], PEER: ended_peer["episode_id"]}


class TestJoinCodeBruteForce:
    def test_wrong_code_is_refused_before_any_account_lookup_and_capped_per_socket(self, env, monkeypatch):
        created = _create(env)
        cid, code = created["call_id"], created["join_code"]
        resolved: list[str] = []
        monkeypatch.setattr(main, "resolve_email_by_uid", lambda u: resolved.append(u) or EMAILS.get(u))
        # REST: the code is checked before the (Firebase) email lookup — a
        # guess must not cost an upstream call.
        assert _join(env, cid, PEER, join_code="AAAAAA")[0] == 403
        assert resolved == []
        with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            for _ in range(calls.JOIN_ATTEMPTS_MAX):
                peer.send_text(json.dumps({"type": "call_join", "call_id": cid, "join_code": "AAAAAA"}))
                assert json.loads(peer.receive_text()) == {"error": "call_join: join code does not match"}
            assert resolved == []
            # The per-socket cap: even the right code is refused on this socket now.
            peer.send_text(json.dumps({"type": "call_join", "call_id": cid, "join_code": code}))
            assert json.loads(peer.receive_text()) == {"error": "call_join: too many failed attempts"}
            # …and the session itself lives on.
            peer.send_text(json.dumps({"type": "config", "empathy_slider": 10}))
            assert json.loads(peer.receive_text())["type"] == "config_ack"
        assert calls.registry.get(cid).participant(PEER) is None
        # A fresh socket (a new auth handshake) with the right code still joins.
        with open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            assert _bind(peer, cid, join_code=code)["self_label"] == "Speaker B"
        assert resolved == [PEER]

    def test_too_many_wrong_guesses_burn_the_code_call_wide(self):
        reg = calls.CallRegistry()
        call = reg.create("host", invitee_uid="invitee")
        for _ in range(calls.JOIN_CODE_FAILURES_MAX):
            with pytest.raises(calls.CallError) as exc:
                reg.join(call, "stranger", join_code="AAAAAA")
            assert exc.value.status == 403
        # The right code no longer admits anyone — the host starts a new call.
        with pytest.raises(calls.CallError) as exc:
            reg.join(call, "stranger", join_code=call.join_code)
        assert exc.value.status == 403
        # The named invitee never needed the code and is unaffected.
        assert reg.join(call, "invitee").slot == "B"


def _drain_until_ack(ws) -> list[dict]:
    """Everything queued on ``ws`` up to a config_ack we ask for."""
    ws.send_text(json.dumps({"type": "config", "empathy_slider": 1}))
    seen: list[dict] = []
    while True:
        msg = json.loads(ws.receive_text())
        if msg.get("type") == "config_ack":
            return seen
        seen.append(msg)


class TestRelayFlood:
    def test_rtc_signal_flood_is_bounded_per_socket(self, env):
        cid = _open_pair(env)
        with open_ws(env.client, f"/ws/session/{HOST_SID}", token=HOST_TOKEN) as host, \
                open_ws(env.client, f"/ws/session/{PEER_SID}", token=PEER_TOKEN) as peer:
            _bind(host, cid)
            _bind(peer, cid)
            recv_until(host, lambda m: m.get("type") == "call_state")
            n = calls.RTC_SIGNAL_BURST * 3
            for i in range(n):
                host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"candidate": i}}))
            refused = [m for m in _drain_until_ack(host) if m.get("error") == "rtc_signal: too many signals"]
            relayed = [m for m in _drain_until_ack(peer) if m.get("type") == "rtc_signal"]
            # The burst goes through (ICE gathering is bursty), the rest is
            # refused with a reason — never silently, never all of it.
            slack = 10
            assert calls.RTC_SIGNAL_BURST <= len(relayed) <= calls.RTC_SIGNAL_BURST + slack
            assert len(refused) == n - len(relayed)
            # Order is preserved for what was delivered.
            assert [m["payload"]["candidate"] for m in relayed] == list(range(len(relayed)))
            # The session is still a session: a later single signal is fine.
            host.send_text(json.dumps({"type": "rtc_signal", "call_id": cid, "payload": {"sdp": "later"}}))
            later = _drain_until_ack(host)
            assert later == [] or all(m.get("error") == "rtc_signal: too many signals" for m in later)

    def test_token_bucket(self):
        clock = [0.0]
        bucket = calls.TokenBucket(rate_per_s=2.0, burst=3, clock=lambda: clock[0])
        assert [bucket.allow() for _ in range(4)] == [True, True, True, False]
        clock[0] += 0.5  # one token back
        assert [bucket.allow() for _ in range(2)] == [True, False]
        clock[0] += 100.0  # refills to the burst, never beyond
        assert [bucket.allow() for _ in range(4)] == [True, True, True, False]
