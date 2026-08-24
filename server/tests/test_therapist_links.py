"""Tests for the two-sided therapist link (routers/therapist.py +
server/therapist_links.py) and its one side effect — auto-sharing at ingest.

GCS/Firebase/LLM are never touched: an in-memory :class:`FakeStore` (the
``app.state.recordings_store`` DI style) implements the recording, share,
live-session and therapist-link surfaces the endpoints use; ``main``'s
email↔uid resolvers are monkeypatched; the LLM is a MagicMock. The real
``RecordingsStore`` link/note methods are exercised against a fake bucket
at the bottom so the GCS layout is covered too.
"""

import io
import json
import uuid
import wave
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import main
import recordings_store
import therapist_links
from main import app, init_db
from routers import sessions as sessions_router

pytestmark = pytest.mark.anyio

ACCOUNTS = {
    "sage@example.com": "user-sage",
    "mom@example.com": "user-mom",
    "other@example.com": "user-other",
}
UID_TO_EMAIL = {v: k for k, v in ACCOUNTS.items()}
PATIENT, THERAPIST, STRANGER = "user-sage", "user-mom", "user-other"


def _h(uid):
    return {"X-Test-Uid": uid}


# ---------------------------------------------------------------------------
# In-memory fake store
# ---------------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        self._by_uid: dict = {}      # {uid: {rid: {meta, turns, analysis}}}
        self._index: dict = {}       # {recipient: {rid: grant}}
        self._links: dict = {}       # {patient_uid: link}
        self._notes: dict = {}       # {(uid, rid): note}
        self.fail_link_reads = False

    # -- recordings --
    def seed(self, uid, *, rid=None, title="Kitchen talk"):
        rid = rid or str(uuid.uuid4())
        self._by_uid.setdefault(uid, {})[rid] = {
            "meta": {
                "id": rid, "created_at": datetime.now(timezone.utc).isoformat(),
                "filename": f"{title}.m4a", "title": title, "media_type": "audio",
                "duration_seconds": 12.0,
                "source": {"type": "upload", "url": None, "original_filename": None},
            },
            "turns": [
                {"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 1.0},
                {"speaker": "Speaker B", "text": "hey", "start_time": 1.0, "end_time": 2.0},
            ],
            "analysis": {"per_turn": [
                {"index": 0, "speaker": "Speaker A", "heat": 10, "markers": [], "is_spike": False, "trigger_phrase": None},
                {"index": 1, "speaker": "Speaker B", "heat": 12, "markers": [], "is_spike": False, "trigger_phrase": None},
            ], "speaker_labels": {}},
        }
        return rid

    async def save_recording(self, uid, *, audio_m4a, video_360p, original_filename,
                             original_content_type, original_bytes, duration_seconds,
                             turns, analysis, source=None, title=None, storage_note=None):
        rid = str(uuid.uuid4())
        self._by_uid.setdefault(uid, {})[rid] = {
            "meta": {
                "id": rid, "created_at": datetime.now(timezone.utc).isoformat(),
                "filename": original_filename or "recording", "title": title or "clip",
                "media_type": "audio", "duration_seconds": duration_seconds,
                "source": source or {"type": "upload", "url": None, "original_filename": None},
            },
            "turns": turns, "analysis": analysis,
        }
        return rid

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

    async def open_media_stream(self, uid, rid, range_header):
        return None

    # -- sharing --
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

    async def remove_share(self, owner_uid, rid, recipient_uid):
        r = self._by_uid.get(owner_uid, {}).get(rid)
        if r is None:
            return False
        r["meta"]["shares"] = [s for s in (r["meta"].get("shares") or []) if s["uid"] != recipient_uid]
        self._index.get(recipient_uid, {}).pop(rid, None)
        return True

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

    # -- therapist link --
    async def read_therapist_link(self, patient_uid):
        if self.fail_link_reads:
            raise RuntimeError("simulated outage")
        return self._links.get(patient_uid)

    async def write_therapist_link(self, patient_uid, link):
        self._links[patient_uid] = dict(link)

    async def delete_therapist_link(self, patient_uid):
        return self._links.pop(patient_uid, None) is not None

    async def list_therapist_patients(self, therapist_uid):
        out = [l for l in self._links.values() if l.get("therapist_uid") == therapist_uid]
        out.sort(key=lambda l: l.get("created_at") or "")
        return out

    # -- notes --
    async def read_therapist_note(self, uid, rid):
        return self._notes.get((uid, rid))

    async def write_therapist_note(self, uid, rid, note):
        self._notes[(uid, rid)] = dict(note)

    async def delete_therapist_note(self, uid, rid):
        return self._notes.pop((uid, rid), None) is not None

    async def list_therapist_notes(self, uid):
        return {rid: n for (u, rid), n in self._notes.items() if u == uid}


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture
def store(monkeypatch):
    fake = FakeStore()
    app.state.recordings_store = fake
    monkeypatch.setattr(main, "resolve_uid_by_email", lambda e: ACCOUNTS.get(e.strip().lower()))
    monkeypatch.setattr(main, "resolve_email_by_uid", lambda u: UID_TO_EMAIL.get(u))
    sessions_router._REFLECT_LOCKS.clear()
    yield fake
    del app.state.recordings_store


async def _link(client, email="mom@example.com", uid=PATIENT):
    return await client.put("/therapist/link", json={"email": email}, headers=_h(uid))


# ---------------------------------------------------------------------------
# Patient side
# ---------------------------------------------------------------------------

class TestLink:
    async def test_unlinked_by_default(self, client, store):
        res = await client.get("/therapist/link", headers=_h(PATIENT))
        assert res.status_code == 200
        assert res.json() == {"linked": False}

    async def test_link_by_email_is_pending_with_auto_share_on(self, client, store):
        res = await _link(client, "Mom@Example.com ")
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["linked"] is True
        assert body["therapist_email"] == "mom@example.com"
        assert body["status"] == "pending"
        assert body["auto_share"] is True
        assert body["created_at"] and body["accepted_at"] is None
        assert "therapist_uid" not in body  # the patient never sees uids
        # The stored link carries both identities for ingest.
        link = store._links[PATIENT]
        assert link["therapist_uid"] == THERAPIST
        assert link["patient_email"] == "sage@example.com"
        # Readable back.
        again = await client.get("/therapist/link", headers=_h(PATIENT))
        assert again.json()["therapist_email"] == "mom@example.com"

    async def test_link_errors(self, client, store):
        assert (await _link(client, "nobody@example.com")).status_code == 404
        assert (await _link(client, "not-an-email")).status_code == 422
        me = await _link(client, "sage@example.com")
        assert me.status_code == 400 and "own therapist" in me.json()["detail"]

    async def test_relink_same_therapist_keeps_acceptance(self, client, store):
        await _link(client)
        acc = await client.post(f"/therapist/patients/{PATIENT}/accept", headers=_h(THERAPIST))
        assert acc.status_code == 200 and acc.json()["status"] == "accepted"
        again = await _link(client)
        assert again.json()["status"] == "accepted"
        # A DIFFERENT therapist replaces the link and starts over as pending.
        other = await _link(client, "other@example.com")
        assert other.json()["status"] == "pending"
        assert other.json()["therapist_email"] == "other@example.com"
        assert (await client.get("/therapist/patients", headers=_h(THERAPIST))).json()["patients"] == []

    async def test_toggle_auto_share_and_unlink(self, client, store):
        assert (await client.patch("/therapist/link", json={"auto_share": False}, headers=_h(PATIENT))).status_code == 404
        await _link(client)
        res = await client.patch("/therapist/link", json={"auto_share": False}, headers=_h(PATIENT))
        assert res.status_code == 200 and res.json()["auto_share"] is False
        assert store._links[PATIENT]["auto_share"] is False
        assert (await client.delete("/therapist/link", headers=_h(PATIENT))).status_code == 204
        assert (await client.get("/therapist/link", headers=_h(PATIENT))).json() == {"linked": False}
        # Idempotent.
        assert (await client.delete("/therapist/link", headers=_h(PATIENT))).status_code == 204

    async def test_storage_disabled_is_503(self, client):
        assert (await client.get("/therapist/link", headers=_h(PATIENT))).status_code == 503
        assert (await _link(client)).status_code == 503

    async def test_requires_auth(self, client, store):
        app.dependency_overrides.pop(main.get_current_uid, None)
        try:
            res = await client.get("/therapist/link")
            assert res.status_code == 401
        finally:
            from conftest import _test_uid_override
            app.dependency_overrides[main.get_current_uid] = _test_uid_override


# ---------------------------------------------------------------------------
# Therapist side
# ---------------------------------------------------------------------------

class TestPatients:
    async def test_list_accept_decline(self, client, store):
        assert (await client.get("/therapist/patients", headers=_h(THERAPIST))).json() == {"patients": []}
        await _link(client)
        rows = (await client.get("/therapist/patients", headers=_h(THERAPIST))).json()["patients"]
        assert len(rows) == 1
        assert rows[0]["patient_uid"] == PATIENT
        assert rows[0]["patient_email"] == "sage@example.com"
        assert rows[0]["status"] == "pending"
        # A stranger sees nothing and can't accept.
        assert (await client.get("/therapist/patients", headers=_h(STRANGER))).json()["patients"] == []
        assert (await client.post(f"/therapist/patients/{PATIENT}/accept", headers=_h(STRANGER))).status_code == 404
        acc = await client.post(f"/therapist/patients/{PATIENT}/accept", headers=_h(THERAPIST))
        assert acc.status_code == 200
        assert acc.json()["status"] == "accepted" and acc.json()["accepted_at"]
        # The patient sees the acceptance.
        assert (await client.get("/therapist/link", headers=_h(PATIENT))).json()["status"] == "accepted"
        # Decline removes the link entirely.
        assert (await client.post(f"/therapist/patients/{PATIENT}/decline", headers=_h(THERAPIST))).status_code == 204
        assert (await client.get("/therapist/link", headers=_h(PATIENT))).json() == {"linked": False}
        assert (await client.post(f"/therapist/patients/{PATIENT}/decline", headers=_h(THERAPIST))).status_code == 404


# ---------------------------------------------------------------------------
# Notes (viewer-private)
# ---------------------------------------------------------------------------

class TestNotes:
    async def test_notes_follow_visibility_and_stay_private(self, client, store):
        rid = store.seed(PATIENT)
        # Not visible to the therapist yet → 404 (never confirmed).
        assert (await client.get(f"/therapist/notes/{rid}", headers=_h(THERAPIST))).status_code == 404
        await store.add_share(PATIENT, rid, recipient_uid=THERAPIST,
                              recipient_email="mom@example.com", owner_email="sage@example.com")
        empty = await client.get(f"/therapist/notes/{rid}", headers=_h(THERAPIST))
        assert empty.status_code == 200
        assert empty.json() == {"episode_id": rid, "text": "", "updated_at": None}
        put = await client.put(f"/therapist/notes/{rid}", json={"text": "  Defensive when Mom brings up calls. "}, headers=_h(THERAPIST))
        assert put.status_code == 200
        assert put.json()["text"] == "Defensive when Mom brings up calls."
        assert put.json()["updated_at"]
        # Private: the owner reads THEIR OWN (empty) note, never the therapist's.
        own = await client.get(f"/therapist/notes/{rid}", headers=_h(PATIENT))
        assert own.json()["text"] == ""
        # Blank text clears the note.
        cleared = await client.put(f"/therapist/notes/{rid}", json={"text": "   "}, headers=_h(THERAPIST))
        assert cleared.json()["text"] == "" and cleared.json()["updated_at"] is None
        assert store._notes == {}
        # Too long is a 422; a stranger is a 404.
        assert (await client.put(f"/therapist/notes/{rid}", json={"text": "x" * 5001}, headers=_h(THERAPIST))).status_code == 422
        assert (await client.put(f"/therapist/notes/{rid}", json={"text": "hi"}, headers=_h(STRANGER))).status_code == 404
        assert (await client.delete(f"/therapist/notes/{rid}", headers=_h(THERAPIST))).status_code == 204


# ---------------------------------------------------------------------------
# Auto-share at ingest
# ---------------------------------------------------------------------------

SESSION_ID = "3a2b1c9e-5a4d-4e8f-9c1a-2b3c4d5e6f71"


def _turn(i, speaker, text, *, is_self=None):
    start = i * 3.0
    return {
        "type": "turn_local", "session_id": SESSION_ID, "speaker": speaker,
        "speaker_person_id": None, "speaker_match_score": None,
        "is_self": is_self, "text": text, "start_time": start,
        "end_time": start + 2.5, "transcript_source": "on-device",
        "prosody": {"rms_dbfs": -20.0, "pitch_hz": None, "speech_rate": 3.1},
        "text_tone": None, "suggestion": None, "suggestion_source": None,
    }


def _live_body():
    return {
        "session_id": SESSION_ID, "started_at": "2026-08-24T18:05:00+00:00",
        "ended_at": "2026-08-24T18:07:00+00:00", "mode": "earpiece",
        "turns": [
            _turn(0, "Speaker A", "Hey Mom.", is_self=True),
            _turn(1, "Speaker B", "You never call."),
        ],
        "analyze": False, "reflect": False,
    }


class TestAutoShareLive:
    async def test_live_session_is_granted_to_linked_therapist(self, client, store):
        await _link(client)
        res = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert res.status_code == 201, res.text
        rid = res.json()["episode_id"]
        assert res.json()["shared_with"] == ["mom@example.com"]
        grant = await store.find_share(THERAPIST, rid)
        assert grant and grant["owner_uid"] == PATIENT and grant["owner_email"] == "sage@example.com"
        # The therapist's dashboard lists it under the patient's email —
        # the EXISTING share mechanism, nothing new.
        rows = (await client.get("/sessions", headers=_h(THERAPIST))).json()["sessions"]
        assert [r["id"] for r in rows] == [rid]
        assert rows[0]["patient"] == "sage@example.com" and rows[0]["shared"] is True
        # The therapist can read it (read-only) and keep a private note on it.
        assert (await client.get(f"/recordings/{rid}", headers=_h(THERAPIST))).status_code == 200
        assert (await client.put(f"/therapist/notes/{rid}", json={"text": "n"}, headers=_h(THERAPIST))).status_code == 200
        # Re-POST (idempotent ingest) keeps the grant and reports it again.
        again = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert again.json()["created"] is False and again.json()["shared_with"] == ["mom@example.com"]
        assert len(store._by_uid[PATIENT][rid]["meta"]["shares"]) == 1

    async def test_pending_link_still_shares_but_decline_stops_it(self, client, store):
        await _link(client)
        # Pending (not yet accepted) still shares — the patient chose it.
        res = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert res.json()["shared_with"] == ["mom@example.com"]
        await client.post(f"/therapist/patients/{PATIENT}/decline", headers=_h(THERAPIST))
        body = {**_live_body(), "session_id": "3a2b1c9e-5a4d-4e8f-9c1a-2b3c4d5e6f72"}
        body["turns"] = [{**t, "session_id": body["session_id"]} for t in body["turns"]]
        res2 = await client.post("/sessions/live", json=body, headers=_h(PATIENT))
        assert res2.status_code == 201 and res2.json()["shared_with"] == []
        assert await store.find_share(THERAPIST, res2.json()["episode_id"]) is None

    async def test_auto_share_off_and_no_link(self, client, store):
        res = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert res.status_code == 201 and res.json()["shared_with"] == []
        await _link(client)
        await client.patch("/therapist/link", json={"auto_share": False}, headers=_h(PATIENT))
        again = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert again.json()["shared_with"] == []
        assert await store.find_share(THERAPIST, again.json()["episode_id"]) is None

    async def test_link_read_failure_never_fails_ingest(self, client, store):
        await _link(client)
        store.fail_link_reads = True
        res = await client.post("/sessions/live", json=_live_body(), headers=_h(PATIENT))
        assert res.status_code == 201 and res.json()["shared_with"] == []


# --- upload ingest hook (the /analyze/upload persistence path) --------------

SR = 16000


def _wav_bytes() -> bytes:
    t = np.arange(SR * 2) / SR
    pcm = (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


MOCK_TURNS = [
    {"speaker": "Speaker A", "text": "Hey, can we talk?", "start_time": 0.0, "end_time": 1.0},
    {"speaker": "Speaker B", "text": "Sure.", "start_time": 1.0, "end_time": 2.0},
    {"speaker": "Speaker A", "text": "You never stick to it.", "start_time": 2.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "That is not fair.", "start_time": 3.0, "end_time": 4.0},
    {"speaker": "Speaker A", "text": "Okay. I hear you.", "start_time": 4.0, "end_time": 5.0},
    {"speaker": "Speaker B", "text": "Thanks.", "start_time": 5.0, "end_time": 6.0},
]


def _analyze_llm():
    m = MagicMock()
    m.complete.return_value = json.dumps({
        "per_turn": [{"heat": 20 + i * 3, "markers": [], "trigger_phrase": None} for i in range(len(MOCK_TURNS))],
        "requests": [], "narrative": "n",
        "report_cards": {sp: {"score": 70, "headline": "h", "did_well": "d", "work_on": "w"}
                         for sp in ("Speaker A", "Speaker B")},
    })
    return m


class TestAutoShareUpload:
    async def test_stored_upload_is_granted_to_linked_therapist(self, client, store, monkeypatch):
        monkeypatch.setattr(
            main, "build_derivatives",
            lambda data, **kw: audio_ingest.Derivatives(
                audio_m4a=b"FAKE-M4A", video_360p=None, has_video=False, video_note=None,
            ),
        )
        await _link(client)
        with patch("main.transcribe_upload", return_value=(MOCK_TURNS, None)), \
             patch("main.get_llm_client", return_value=_analyze_llm()):
            resp = await client.post(
                "/analyze/upload",
                files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
                data={"consent": "true", "store": "true"},
                headers=_h(PATIENT),
            )
        assert resp.status_code == 200, resp.text
        rid = resp.json()["recording_id"]
        assert resp.json()["stored"] is True
        grant = await store.find_share(THERAPIST, rid)
        assert grant and grant["owner_uid"] == PATIENT
        rows = (await client.get("/sessions", headers=_h(THERAPIST))).json()["sessions"]
        assert rid in [r["id"] for r in rows]


# ---------------------------------------------------------------------------
# Pure helpers + the real store's GCS layout
# ---------------------------------------------------------------------------

def test_should_auto_share():
    assert therapist_links.should_auto_share(None) is False
    assert therapist_links.should_auto_share({"therapist_uid": "t"}) is True
    assert therapist_links.should_auto_share({"therapist_uid": "t", "auto_share": False}) is False
    assert therapist_links.should_auto_share({"auto_share": True}) is False


async def test_auto_share_recording_tolerates_a_store_without_links():
    class Bare:
        pass
    assert await therapist_links.auto_share_recording(Bare(), "u", "r") == []


class _FakeBlob:
    def __init__(self, bucket, name):
        self._bucket, self.name = bucket, name

    def exists(self):
        return self.name in self._bucket.objects

    def download_as_bytes(self):
        return self._bucket.objects[self.name]

    def upload_from_string(self, data, content_type=None):
        self._bucket.objects[self.name] = data.encode() if isinstance(data, str) else data

    def delete(self):
        del self._bucket.objects[self.name]


class _FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def blob(self, name):
        return _FakeBlob(self, name)

    def list_blobs(self, prefix=""):
        return [_FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


async def test_real_store_link_layout_and_reverse_index():
    bucket = _FakeBucket()
    st = recordings_store.RecordingsStore(bucket)
    assert await st.read_therapist_link("p1") is None
    assert await st.list_therapist_patients("t1") == []
    link = therapist_links.new_link(patient_uid="p1", patient_email="p@x", therapist_uid="t1", therapist_email="t@x")
    await st.write_therapist_link("p1", link)
    assert set(bucket.objects) == {"therapist_links/p1/link.json", "therapist_patients/t1/p1.json"}
    assert (await st.read_therapist_link("p1"))["therapist_uid"] == "t1"
    assert [l["patient_uid"] for l in await st.list_therapist_patients("t1")] == ["p1"]
    # Re-pointing to another therapist drops the old reverse-index entry.
    link2 = therapist_links.new_link(patient_uid="p1", patient_email="p@x", therapist_uid="t2", therapist_email="t2@x")
    await st.write_therapist_link("p1", link2)
    assert "therapist_patients/t1/p1.json" not in bucket.objects
    assert "therapist_patients/t2/p1.json" in bucket.objects
    assert await st.list_therapist_patients("t1") == []
    assert await st.delete_therapist_link("p1") is True
    assert bucket.objects == {}
    assert await st.delete_therapist_link("p1") is False


async def test_real_store_notes():
    bucket = _FakeBucket()
    st = recordings_store.RecordingsStore(bucket)
    assert await st.read_therapist_note("t1", "e1") is None
    await st.write_therapist_note("t1", "e1", {"text": "hi", "updated_at": "now"})
    assert set(bucket.objects) == {"therapist_notes/t1/e1.json"}
    assert (await st.read_therapist_note("t1", "e1"))["text"] == "hi"
    assert await st.list_therapist_notes("t1") == {"e1": {"text": "hi", "updated_at": "now"}}
    assert await st.list_therapist_notes("t2") == {}
    assert await st.delete_therapist_note("t1", "e1") is True
    assert await st.delete_therapist_note("t1", "e1") is False
