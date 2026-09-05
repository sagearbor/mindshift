"""Endpoint tests for attaching a live session's mic audio after the fact:
POST /sessions/{id}/audio (direct, multipart) and the chunked
/uploads/{id}/complete with ``attach_to_recording_id`` — both converge on
routers.sessions.attach_live_audio + recordings_store.attach_audio.

GCS / ffmpeg / LLM are never touched on the HTTP tests: an in-memory fake
store (test_sessions_live's FakeLiveStore, extended with attach_audio /
get_audio_bytes + the chunked-upload session methods) is injected at
``app.state.recordings_store``, and ``audio_ingest.build_derivatives`` is
patched to a deterministic Derivatives so no transcode runs. The WAV itself
is REAL (stdlib-parsed) so the duration is decoded, not guessed. The store
layer (attach → meta flip, re-POST preservation, delete) is exercised
against the real RecordingsStore over test_account_deletion's fake bucket.
One un-patched test runs real ffmpeg when it is installed.
"""

import io
import json
import uuid
import wave
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import main
import recordings_store
from main import app, init_db
from routers import sessions as sessions_router
from tests.test_account_deletion import _FakeBucket
from tests.test_chunked_upload import _patched_upload
from tests.test_sessions_live import FakeLiveStore, _body

pytestmark = pytest.mark.anyio

SR = 16000
OTHER_UID = "other-user"


# ---------------------------------------------------------------------------
# Audio fixture — a real 6 s mono 16 kHz WAV (same builder as test_reanalyze)
# ---------------------------------------------------------------------------

def _wav_bytes(pcm: np.ndarray, sr: int = SR) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _sine(freq: float, seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


_AMPS = [0.1, 0.2, 0.5, 0.15, 0.3, 0.08]
FIXTURE_WAV = _wav_bytes(
    np.concatenate([_sine(180.0, 1.0, a) for a in _AMPS]).astype(np.float32)
)
FIXTURE_SECONDS = 6.0

FAKE_M4A = b"FAKE-M4A-FROM-LIVE-SESSION-" * 40
FAKE_M4A_2 = b"SECOND-ATTACH-M4A-" * 40

# A small chunk size so FIXTURE_WAV (~192KB) spans several parts.
SMALL_CHUNK = 64 * 1024


def _fake_derivatives(audio: bytes = FAKE_M4A):
    return audio_ingest.Derivatives(
        audio_m4a=audio, video_360p=None, has_video=False, video_note=None,
    )


# ---------------------------------------------------------------------------
# Fake store — FakeLiveStore + attach/audio + chunked-upload sessions
# ---------------------------------------------------------------------------

class FakeAudioStore(FakeLiveStore):
    def __init__(self):
        super().__init__()
        self._audio: dict[tuple[str, str], bytes] = {}
        self.attach_calls: list[dict] = []
        self._uploads: dict[str, dict[str, dict]] = {}
        self.cleanup_calls: list[tuple[str, str]] = []
        self.job_states: list = []

    # Mirrors RecordingsStore._save_live_session_sync's preserve list,
    # attached-audio fields included.
    async def save_live_session(self, uid, recording_id, *, meta, turns, analysis):
        self.save_calls += 1
        slot = self._by_uid.setdefault(uid, {})
        written = dict(meta)
        existing = slot.get(recording_id)
        if existing:
            old = existing["meta"]
            for key in ("manual_speaker_labels", "manual_speaker_people", "shares"):
                if key in old and key not in written:
                    written[key] = old[key]
            if not written.get("title") and old.get("title"):
                written["title"] = old["title"]
            if old.get("media_type") not in (None, "none"):
                for key in recordings_store._ATTACHED_AUDIO_META_KEYS:
                    if key in old:
                        written[key] = old[key]
        slot[recording_id] = {"meta": written, "turns": turns, "analysis": analysis}
        return written

    async def attach_audio(
        self, uid, recording_id, *, audio_m4a, duration_seconds=None, original_bytes=0,
    ):
        self.attach_calls.append({
            "uid": uid, "recording_id": recording_id, "audio_m4a": audio_m4a,
            "duration_seconds": duration_seconds, "original_bytes": original_bytes,
        })
        r = self._by_uid.get(uid, {}).get(recording_id)
        if r is None:
            return None
        self._audio[(uid, recording_id)] = audio_m4a
        meta = r["meta"]
        meta["media_type"] = "audio"
        meta["stored_variants"] = ["audio.m4a"]
        meta["size_bytes"] = len(audio_m4a)
        meta["original_bytes"] = original_bytes
        meta["storage_note"] = None
        if duration_seconds is not None:
            meta["duration_seconds"] = duration_seconds
        meta["audio_attached_at"] = datetime.now(timezone.utc).isoformat()
        return meta

    async def get_audio_bytes(self, uid, recording_id):
        return self._audio.get((uid, recording_id))

    def add_upload_recording(self, uid, recording_id):
        """A stored UPLOAD (not a live session) — audio can't be attached."""
        self._by_uid.setdefault(uid, {})[recording_id] = {
            "meta": {
                "id": recording_id, "created_at": "2026-08-20T10:00:00+00:00",
                "filename": "clip.m4a", "title": "clip", "media_type": "audio",
                "duration_seconds": 30.0, "size_bytes": 100,
                "stored_variants": ["audio.m4a"], "storage_note": None,
                "source": {"type": "upload", "url": None, "original_filename": "clip.m4a"},
            },
            "turns": [], "analysis": None,
        }
        self._audio[(uid, recording_id)] = b"upload-audio"

    # -- chunked upload sessions (FakeUploadStore's surface) --
    async def write_upload_manifest(self, uid, upload_id, manifest):
        self._uploads.setdefault(uid, {})[upload_id] = {"manifest": manifest, "parts": {}}

    async def read_upload_manifest(self, uid, upload_id):
        sess = self._uploads.get(uid, {}).get(upload_id)
        return None if sess is None else sess["manifest"]

    async def write_upload_part(self, uid, upload_id, index, data):
        self._uploads[uid][upload_id]["parts"][index] = data

    async def get_upload_part_sizes(self, uid, upload_id):
        sess = self._uploads.get(uid, {}).get(upload_id)
        return {} if sess is None else {i: len(d) for i, d in sess["parts"].items()}

    async def assemble_upload(self, uid, upload_id, expected_chunks):
        parts = self._uploads[uid][upload_id]["parts"]
        return b"".join(parts[i] for i in range(expected_chunks))

    async def cleanup_upload(self, uid, upload_id):
        self.cleanup_calls.append((uid, upload_id))
        self._uploads.get(uid, {}).pop(upload_id, None)

    async def write_job_state(self, uid, job_id, state):
        self.job_states.append((uid, job_id, state))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def store():
    fake = FakeAudioStore()
    app.state.recordings_store = fake
    sessions_router._REFLECT_LOCKS.clear()
    sessions_router._REFLECT_LOCK_USERS.clear()
    main._rate_limiter.reset()
    yield fake
    del app.state.recordings_store


@pytest.fixture
def fake_ffmpeg():
    """Patch the derivative transcode so no ffmpeg runs; records every call."""
    calls: list[dict] = []

    def build(data, **kw):
        calls.append({"data": data, **kw})
        return _fake_derivatives()

    with patch.object(audio_ingest, "build_derivatives", side_effect=build):
        yield calls


async def _ingest(client, store, uid="test-user") -> str:
    res = await client.post(
        "/sessions/live", json=_body(analyze=False, reflect=False),
        headers={"X-Test-Uid": uid},
    )
    assert res.status_code == 201, res.text
    rid = res.json()["episode_id"]
    rec = await store.get_recording(uid, rid)
    assert rec["media_type"] == "none" and rec["stored_variants"] == []
    return rid


async def _attach(client, rid, data=FIXTURE_WAV, uid="test-user"):
    return await client.post(
        f"/sessions/{rid}/audio",
        files={"file": ("session.wav", data, "audio/wav")},
        headers={"X-Test-Uid": uid},
    )


def _assert_attach_body(body, rid, size):
    assert body == {
        "recording_id": rid,
        "media_type": "audio",
        "duration_seconds": pytest.approx(FIXTURE_SECONDS),
        "size_bytes": size,
        "stored_variants": ["audio.m4a"],
    }


# ---------------------------------------------------------------------------
# POST /sessions/{id}/audio
# ---------------------------------------------------------------------------

class TestAttachDirect:
    async def test_happy_path_flips_meta_and_stores_m4a(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        res = await _attach(client, rid)
        assert res.status_code == 200, res.text
        _assert_attach_body(res.json(), rid, len(FAKE_M4A))

        # The ORIGINAL bytes went to the transcoder, as a mic capture.
        assert len(fake_ffmpeg) == 1
        assert fake_ffmpeg[0]["data"] == FIXTURE_WAV
        assert fake_ffmpeg[0]["expect_video"] is False

        # The derivative — never the WAV — is what is stored.
        assert await store.get_audio_bytes("test-user", rid) == FAKE_M4A
        rec = await store.get_recording("test-user", rid)
        assert rec["media_type"] == "audio"
        assert rec["stored_variants"] == ["audio.m4a"]
        assert rec["size_bytes"] == len(FAKE_M4A)
        assert rec["original_bytes"] == len(FIXTURE_WAV)
        assert rec["storage_note"] is None
        assert rec["duration_seconds"] == pytest.approx(FIXTURE_SECONDS)
        datetime.fromisoformat(rec["audio_attached_at"])  # a real timestamp
        # Still a live episode.
        assert rec["source"]["type"] == "live"

        # Downstream reads see an audio recording now.
        detail = await client.get(f"/recordings/{rid}")
        assert detail.status_code == 200
        assert detail.json()["media_type"] == "audio"
        assert detail.json()["duration_seconds"] == pytest.approx(FIXTURE_SECONDS)
        rows = (await client.get("/sessions")).json()["sessions"]
        assert next(s for s in rows if s["id"] == rid)["hasAudio"] is True

    async def test_unknown_recording_404(self, client, store, fake_ffmpeg):
        res = await _attach(client, str(uuid.uuid4()))
        assert res.status_code == 404
        assert fake_ffmpeg == [] and store.attach_calls == []

    async def test_foreign_uid_404(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        res = await _attach(client, rid, uid=OTHER_UID)
        assert res.status_code == 404
        assert store.attach_calls == []
        rec = await store.get_recording("test-user", rid)
        assert rec["media_type"] == "none"

    async def test_upload_recording_404(self, client, store, fake_ffmpeg):
        rid = str(uuid.uuid4())
        store.add_upload_recording("test-user", rid)
        res = await _attach(client, rid)
        assert res.status_code == 404
        assert store.attach_calls == []
        assert await store.get_audio_bytes("test-user", rid) == b"upload-audio"

    async def test_over_direct_cap_413(self, client, store, fake_ffmpeg, monkeypatch):
        monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
        rid = await _ingest(client, store)
        res = await _attach(client, rid)
        assert res.status_code == 413
        detail = res.json()["detail"]
        assert "too large" in detail and "/uploads/start" in detail
        assert fake_ffmpeg == [] and store.attach_calls == []
        assert (await store.get_recording("test-user", rid))["media_type"] == "none"

    async def test_undecodable_422(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        res = await _attach(client, rid, data=b"definitely not audio bytes")
        assert res.status_code == 422, res.text
        assert fake_ffmpeg == [] and store.attach_calls == []
        rec = await store.get_recording("test-user", rid)
        assert rec["media_type"] == "none" and "audio_attached_at" not in rec

    async def test_empty_file_422(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        res = await _attach(client, rid, data=b"")
        assert res.status_code == 422
        assert store.attach_calls == []

    async def test_reattach_overwrites_idempotently(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        assert (await _attach(client, rid)).status_code == 200
        first = await store.get_recording("test-user", rid)

        with patch.object(
            audio_ingest, "build_derivatives",
            return_value=_fake_derivatives(FAKE_M4A_2),
        ):
            res = await _attach(client, rid)
        assert res.status_code == 200, res.text  # never a 409
        _assert_attach_body(res.json(), rid, len(FAKE_M4A_2))
        assert await store.get_audio_bytes("test-user", rid) == FAKE_M4A_2
        second = await store.get_recording("test-user", rid)
        assert second["size_bytes"] == len(FAKE_M4A_2)
        assert second["audio_attached_at"] >= first["audio_attached_at"]

    async def test_repost_session_keeps_attached_audio(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        assert (await _attach(client, rid)).status_code == 200
        attached = await store.get_recording("test-user", rid)

        # The phone re-sends the same session (its meta says "no audio").
        res = await client.post(
            "/sessions/live", json=_body(analyze=False, reflect=False),
        )
        assert res.status_code == 201 and res.json()["created"] is False
        rec = await store.get_recording("test-user", rid)
        for key in recordings_store._ATTACHED_AUDIO_META_KEYS:
            assert rec[key] == attached[key], key
        assert rec["media_type"] == "audio"
        assert await store.get_audio_bytes("test-user", rid) == FAKE_M4A
        assert (await client.get(f"/recordings/{rid}")).json()["media_type"] == "audio"

    async def test_storage_disabled_503(self, client, fake_ffmpeg):
        main._rate_limiter.reset()
        if hasattr(app.state, "recordings_store"):
            del app.state.recordings_store
        res = await _attach(client, str(uuid.uuid4()))
        assert res.status_code == 503

    async def test_real_ffmpeg_transcodes_to_m4a(self, client, store):
        """Un-patched: the real ffmpeg derivative (skipped when it is absent)."""
        try:
            audio_ingest._ffmpeg_exe()
        except audio_ingest.TranscodeError:
            pytest.skip("ffmpeg (imageio-ffmpeg) not installed")
        rid = await _ingest(client, store)
        res = await _attach(client, rid)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["duration_seconds"] == pytest.approx(FIXTURE_SECONDS)
        m4a = await store.get_audio_bytes("test-user", rid)
        assert m4a and b"ftyp" in m4a[:16]  # an MP4/M4A container
        assert body["size_bytes"] == len(m4a)
        assert len(m4a) < len(FIXTURE_WAV)  # compressed, never the WAV


# ---------------------------------------------------------------------------
# Chunked: /uploads/start → PUT chunks → /uploads/{id}/complete {attach_to_recording_id}
# ---------------------------------------------------------------------------

def _chunks(data: bytes, size: int) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


async def _upload_all(client, data, uid="test-user"):
    resp = await client.post(
        "/uploads/start",
        json={
            "filename": "session.wav", "content_type": "audio/wav",
            "total_bytes": len(data), "consent": False, "store": True,
        },
        headers={"X-Test-Uid": uid},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["expected_chunks"] > 1  # genuinely multi-part
    for i, chunk in enumerate(_chunks(data, body["chunk_bytes"])):
        r = await client.put(
            f"/uploads/{body['upload_id']}/chunks/{i}", content=chunk,
            headers={"X-Test-Uid": uid},
        )
        assert r.status_code == 200, r.text
    return body["upload_id"]


async def _complete(client, upload_id, json_body=None, uid="test-user"):
    kwargs = {"headers": {"X-Test-Uid": uid}}
    if json_body is not None:
        kwargs["json"] = json_body
    return await client.post(f"/uploads/{upload_id}/complete", **kwargs)


class TestAttachChunked:
    @pytest.fixture(autouse=True)
    def _small_chunks(self, monkeypatch):
        monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", SMALL_CHUNK)

    async def test_complete_with_attach_attaches_without_a_job(
        self, client, store, fake_ffmpeg,
    ):
        rid = await _ingest(client, store)
        upload_id = await _upload_all(client, FIXTURE_WAV)
        jobs_before = set(main._JOB_TASKS)
        saves_before = store.save_calls

        res = await _complete(client, upload_id, {"attach_to_recording_id": rid})
        assert res.status_code == 200, res.text  # 200, not 202: no job
        _assert_attach_body(res.json(), rid, len(FAKE_M4A))

        # The reassembled WAV — byte-identical — reached the transcoder once.
        assert len(fake_ffmpeg) == 1
        assert fake_ffmpeg[0]["data"] == FIXTURE_WAV
        assert await store.get_audio_bytes("test-user", rid) == FAKE_M4A
        rec = await store.get_recording("test-user", rid)
        assert rec["media_type"] == "audio"
        assert rec["original_bytes"] == len(FIXTURE_WAV)

        # No analysis job, no job state, no new recording; parts cleaned up.
        assert set(main._JOB_TASKS) == jobs_before
        assert store.job_states == []
        assert store.save_calls == saves_before
        assert list(store._by_uid["test-user"]) == [rid]
        assert ("test-user", upload_id) in store.cleanup_calls
        assert store._uploads.get("test-user", {}).get(upload_id) is None

    async def test_complete_with_attach_unknown_recording_404(
        self, client, store, fake_ffmpeg,
    ):
        upload_id = await _upload_all(client, FIXTURE_WAV)
        res = await _complete(
            client, upload_id, {"attach_to_recording_id": str(uuid.uuid4())},
        )
        assert res.status_code == 404
        assert fake_ffmpeg == [] and store.attach_calls == []
        # Refused bytes are still cleaned up (the phone re-uploads on retry).
        assert ("test-user", upload_id) in store.cleanup_calls

    async def test_complete_with_attach_foreign_recording_404(
        self, client, store, fake_ffmpeg,
    ):
        rid = await _ingest(client, store)  # test-user's episode
        upload_id = await _upload_all(client, FIXTURE_WAV, uid=OTHER_UID)
        res = await _complete(
            client, upload_id, {"attach_to_recording_id": rid}, uid=OTHER_UID,
        )
        assert res.status_code == 404
        assert (await store.get_recording("test-user", rid))["media_type"] == "none"

    async def test_complete_with_attach_undecodable_422(self, client, store, fake_ffmpeg):
        rid = await _ingest(client, store)
        garbage = b"\x00\xff" * (SMALL_CHUNK)  # two chunks of non-audio
        upload_id = await _upload_all(client, garbage)
        res = await _complete(client, upload_id, {"attach_to_recording_id": rid})
        assert res.status_code == 422, res.text
        assert store.attach_calls == []
        assert ("test-user", upload_id) in store.cleanup_calls

    async def test_complete_with_attach_rejects_non_uuid_422(self, client, store, fake_ffmpeg):
        upload_id = await _upload_all(client, FIXTURE_WAV)
        res = await _complete(client, upload_id, {"attach_to_recording_id": "nope"})
        assert res.status_code == 422

    async def test_complete_without_attach_is_unchanged(self, client, store, fake_ffmpeg):
        """No body, ``{}`` and an explicit null all take the ORIGINAL path:
        the full analysis response, no attach."""
        for json_body in (None, {}, {"attach_to_recording_id": None}):
            upload_id = await _upload_all(client, FIXTURE_WAV)
            p1, p2 = _patched_upload()
            with p1, p2:
                res = await _complete(client, upload_id, json_body)
            assert res.status_code == 200, res.text
            data = res.json()
            assert "per_turn" in data and "turns" in data  # AnalyzeUploadResponse
            assert data["stored"] is False
            assert data["storage_note"] == "consent not given"
            assert ("test-user", upload_id) in store.cleanup_calls
        assert store.attach_calls == [] and fake_ffmpeg == []


# ---------------------------------------------------------------------------
# Store layer — the real RecordingsStore over a fake bucket
# ---------------------------------------------------------------------------

LIVE_META = {
    "id": "rid", "created_at": "2026-08-24T18:05:00+00:00", "ingested_at": "x",
    "filename": "live-session", "title": "Live session · earpiece",
    "media_type": "none", "duration_seconds": 120.0, "size_bytes": 0,
    "stored_variants": [], "storage_note": "live session — no audio kept",
    "original_bytes": 0, "original_filename": None, "original_content_type": None,
    "source": {"type": "live", "url": None, "original_filename": None},
    "mode": "earpiece", "session_id": "s1", "ended_at": "2026-08-24T18:07:00+00:00",
}
TURNS = [{"speaker": "Speaker A", "text": "hi", "start_time": 0.0, "end_time": 1.0}]


def _meta_of(bucket, uid, rid) -> dict:
    return json.loads(bucket.objects[f"recordings/{uid}/{rid}/meta.json"])


class TestStore:
    @pytest.fixture
    def bucket(self):
        return _FakeBucket()

    @pytest.fixture
    def real_store(self, bucket):
        return recordings_store.RecordingsStore(bucket)

    async def test_attach_missing_recording_is_none_and_writes_nothing(
        self, real_store, bucket,
    ):
        assert await real_store.attach_audio("u", "rid", audio_m4a=b"x") is None
        assert bucket.objects == {}

    async def test_attach_writes_blob_and_flips_meta(self, real_store, bucket):
        await real_store.save_live_session(
            "u", "rid", meta=dict(LIVE_META), turns=TURNS, analysis={"live": {}},
        )
        meta = await real_store.attach_audio(
            "u", "rid", audio_m4a=FAKE_M4A, duration_seconds=6.5, original_bytes=999,
        )
        assert bucket.objects["recordings/u/rid/audio.m4a"] == FAKE_M4A
        assert meta == _meta_of(bucket, "u", "rid")
        assert meta["media_type"] == "audio"
        assert meta["stored_variants"] == ["audio.m4a"]
        assert meta["size_bytes"] == len(FAKE_M4A)
        assert meta["original_bytes"] == 999
        assert meta["storage_note"] is None
        assert meta["duration_seconds"] == 6.5
        datetime.fromisoformat(meta["audio_attached_at"])
        # Everything else untouched.
        assert meta["source"] == LIVE_META["source"]
        assert meta["title"] == LIVE_META["title"]
        assert await real_store.get_audio_bytes("u", "rid") == FAKE_M4A
        assert await real_store.get_audio_bytes("other", "rid") is None

        # Re-attach overwrites; a missing duration keeps the existing one.
        meta2 = await real_store.attach_audio("u", "rid", audio_m4a=FAKE_M4A_2)
        assert bucket.objects["recordings/u/rid/audio.m4a"] == FAKE_M4A_2
        assert meta2["size_bytes"] == len(FAKE_M4A_2)
        assert meta2["duration_seconds"] == 6.5

    async def test_resave_live_session_preserves_attached_audio(self, real_store, bucket):
        await real_store.save_live_session(
            "u", "rid", meta=dict(LIVE_META), turns=TURNS, analysis={"live": {}},
        )
        attached = await real_store.attach_audio(
            "u", "rid", audio_m4a=FAKE_M4A, duration_seconds=6.5, original_bytes=999,
        )
        await real_store.update_manual_speaker_labels("u", "rid", {"Speaker A": "Me"})

        # A re-POST: the phone's meta still says "none" and a new title.
        written = await real_store.save_live_session(
            "u", "rid", meta={**LIVE_META, "title": "Renamed"},
            turns=TURNS, analysis={"live": {"x": 1}},
        )
        assert written == _meta_of(bucket, "u", "rid")
        for key in recordings_store._ATTACHED_AUDIO_META_KEYS:
            assert written[key] == attached[key], key
        assert written["duration_seconds"] == 6.5
        assert written["title"] == "Renamed"  # the new meta's own fields still win
        assert written["manual_speaker_labels"] == {"Speaker A": "Me"}
        assert bucket.objects["recordings/u/rid/audio.m4a"] == FAKE_M4A

    async def test_resave_without_attached_audio_takes_new_meta(self, real_store, bucket):
        await real_store.save_live_session(
            "u", "rid", meta=dict(LIVE_META), turns=TURNS, analysis={},
        )
        written = await real_store.save_live_session(
            "u", "rid", meta={**LIVE_META, "duration_seconds": 130.0},
            turns=TURNS, analysis={},
        )
        assert written["media_type"] == "none"
        assert written["duration_seconds"] == 130.0
        assert "audio_attached_at" not in written

    async def test_delete_removes_attached_audio(self, real_store, bucket):
        await real_store.save_live_session(
            "u", "rid", meta=dict(LIVE_META), turns=TURNS, analysis={},
        )
        await real_store.attach_audio("u", "rid", audio_m4a=FAKE_M4A)
        assert "recordings/u/rid/audio.m4a" in bucket.objects
        assert await real_store.delete_recording("u", "rid") is True
        assert bucket.names("recordings/u/rid/") == []
        assert await real_store.get_audio_bytes("u", "rid") is None
