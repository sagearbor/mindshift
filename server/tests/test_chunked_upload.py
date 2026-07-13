"""Fully-mocked tests for the chunked upload session endpoints.

The whole flow — POST /uploads/start → PUT /uploads/{id}/chunks/{i} →
POST /uploads/{id}/complete (and DELETE /uploads/{id}) — is exercised against an
in-memory :class:`FakeUploadStore` injected via ``app.state.recordings_store``
(the repo's DI style, mirroring test_recordings). GCS is never touched. Deepgram
and the LLM are mocked exactly as in test_analyze_upload/test_recordings, so the
analysis at complete() is deterministic and key-free.

The chunk size is monkeypatched small (``main.UPLOAD_CHUNK_BYTES``) so a tiny WAV
fixture spans several parts — the multi-chunk assembly is genuinely tested, not
faked with a single part.
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
from main import app, init_db

SR = 16000

# Deterministic ffmpeg-derivative stand-in (build_derivatives is patched in the
# `store` fixture so complete() never invokes real ffmpeg).
FAKE_AUDIO_M4A = b"FAKE-M4A-AUDIO-DERIVATIVE-" * 20


# ---------------------------------------------------------------------------
# In-memory fake store — recording methods (for the store=true path) + the new
# upload-session methods. Interface matches RecordingsStore exactly.
# ---------------------------------------------------------------------------

class FakeUploadStore:
    def __init__(self):
        # recordings: {uid: {recording_id: {...}}}
        self._recordings: dict[str, dict[str, dict]] = {}
        self.save_calls: list[dict] = []
        # uploads: {uid: {upload_id: {"manifest": dict, "parts": {index: bytes}}}}
        self._uploads: dict[str, dict[str, dict]] = {}
        self.cleanup_calls: list[tuple[str, str]] = []

    # -- recording persistence (only used on the store=true complete path) --
    async def save_recording(
        self, uid, *, audio_m4a, video_360p, original_filename,
        original_content_type, original_bytes, duration_seconds, turns,
        analysis, source=None, title=None, storage_note=None,
    ):
        recording_id = str(uuid.uuid4())
        meta = {
            "id": recording_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "filename": original_filename or "recording",
            "media_type": "video" if video_360p is not None else "audio",
            "duration_seconds": duration_seconds,
            "size_bytes": len(audio_m4a) + (len(video_360p) if video_360p else 0),
            "source": source,
        }
        self._recordings.setdefault(uid, {})[recording_id] = {
            "meta": meta, "turns": turns, "analysis": analysis,
            "audio_m4a": audio_m4a, "video_360p": video_360p,
        }
        self.save_calls.append({"uid": uid, "recording_id": recording_id,
                                "audio_m4a": audio_m4a, "video_360p": video_360p,
                                "source": source})
        return recording_id

    # -- chunked upload sessions -------------------------------------------
    async def write_upload_manifest(self, uid, upload_id, manifest):
        self._uploads.setdefault(uid, {})[upload_id] = {
            "manifest": manifest, "parts": {},
        }

    async def read_upload_manifest(self, uid, upload_id):
        sess = self._uploads.get(uid, {}).get(upload_id)
        return None if sess is None else sess["manifest"]

    async def write_upload_part(self, uid, upload_id, index, data):
        # uid-scoped: only the owner has a session, so this KeyErrors for a
        # foreign uid — but the endpoint always checks the manifest first, so a
        # foreign PUT is a 404 before it reaches here.
        self._uploads[uid][upload_id]["parts"][index] = data

    async def get_upload_part_sizes(self, uid, upload_id):
        sess = self._uploads.get(uid, {}).get(upload_id)
        if sess is None:
            return {}
        return {i: len(d) for i, d in sess["parts"].items()}

    async def assemble_upload(self, uid, upload_id, expected_chunks):
        parts = self._uploads[uid][upload_id]["parts"]
        return b"".join(parts[i] for i in range(expected_chunks))

    async def cleanup_upload(self, uid, upload_id):
        self.cleanup_calls.append((uid, upload_id))
        self._uploads.get(uid, {}).pop(upload_id, None)


# ---------------------------------------------------------------------------
# Fixtures — audio + mocks (mirrors test_analyze_upload)
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

MOCK_TURNS = [
    {"speaker": "Speaker A", "text": "Hey, can we talk about the schedule?",
     "start_time": 0.0, "end_time": 1.0},
    {"speaker": "Speaker B", "text": "Sure, what about it.",
     "start_time": 1.0, "end_time": 2.0},
    {"speaker": "Speaker A", "text": "You never stick to what we agree.",
     "start_time": 2.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "That is not fair and you know it.",
     "start_time": 3.0, "end_time": 4.0},
    {"speaker": "Speaker A", "text": "Okay. I hear you. Let me try again.",
     "start_time": 4.0, "end_time": 5.0},
    {"speaker": "Speaker B", "text": "Thanks. I appreciate that.",
     "start_time": 5.0, "end_time": 6.0},
]
_SPEAKERS = ["Speaker A", "Speaker B"]

# A small chunk size so FIXTURE_WAV (~192KB) spans several parts.
SMALL_CHUNK = 64 * 1024


def _analyze_llm_json(n_turns: int) -> str:
    return json.dumps({
        "per_turn": [
            {"heat": 20 + i * 3, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both keep showing up and trying to reconnect.",
        "report_cards": {
            sp: {
                "score": 70,
                "headline": f"{sp} stayed engaged",
                "did_well": "Kept trying to reconnect.",
                "work_on": "Pause before answering criticism.",
            }
            for sp in _SPEAKERS
        },
    })


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


def _patched_upload():
    return (
        patch("main.transcribe_prerecorded", return_value=MOCK_TURNS),
        patch("main.get_llm_client",
              return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))),
    )


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def store(monkeypatch):
    """Inject a fake store, shrink the chunk size so the tiny fixture is
    genuinely multi-part, and patch build_derivatives to deterministic bytes so
    complete()'s persistence path never invokes real ffmpeg."""
    monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", SMALL_CHUNK)
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=False,
            video_note=None,
        ),
    )
    fake = FakeUploadStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


def _chunks(data: bytes, size: int = SMALL_CHUNK) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


async def _start(client, data, uid="test-user", **overrides):
    body = {
        "filename": "clip.wav",
        "content_type": "audio/wav",
        "total_bytes": len(data),
        "consent": False,
        "store": True,
        **overrides,
    }
    return await client.post(
        "/uploads/start", json=body, headers={"X-Test-Uid": uid},
    )


async def _put_chunk(client, upload_id, index, body, uid="test-user"):
    return await client.put(
        f"/uploads/{upload_id}/chunks/{index}",
        content=body,
        headers={"X-Test-Uid": uid},
    )


async def _upload_all(client, data, uid="test-user", **start_overrides):
    """start → PUT every chunk. Returns (upload_id, expected_chunks)."""
    resp = await _start(client, data, uid=uid, **start_overrides)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    upload_id = body["upload_id"]
    for i, chunk in enumerate(_chunks(data, body["chunk_bytes"])):
        r = await _put_chunk(client, upload_id, i, chunk, uid=uid)
        assert r.status_code == 200, r.text
    return upload_id, body["expected_chunks"]


# ---------------------------------------------------------------------------
# Happy path — start / chunk / complete returns the full analysis, parts cleaned
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_happy_path_full_analysis_and_cleanup(client, store):
    upload_id, expected = await _upload_all(client, FIXTURE_WAV)
    assert expected > 1  # genuinely multi-part

    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Full AnalyzeUploadResponse shape, identical to the direct path.
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert set(data["per_speaker"]) == {"Speaker A", "Speaker B"}
    assert set(data["report_cards"]) == {"Speaker A", "Speaker B"}
    assert [t["speaker"] for t in data["turns"]] == [t["speaker"] for t in MOCK_TURNS]
    # Real WAV reassembled → prosody ran → voice labels present, no degrade note.
    assert data["voice_analysis"] is None
    for pt in data["per_turn"]:
        assert pt["voice"] is not None

    # Parts + manifest cleaned up after complete.
    assert ("test-user", upload_id) in store.cleanup_calls
    assert store._uploads.get("test-user", {}).get(upload_id) is None
    # consent defaulted False → not persisted as a recording.
    assert data["stored"] is False
    assert data["storage_note"] == "consent not given"


@pytest.mark.anyio
async def test_chunked_complete_honors_consent_true_persists(client, store):
    upload_id, _ = await _upload_all(client, FIXTURE_WAV, consent=True, store=True)
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is True
    assert data["recording_id"]
    assert data["storage_note"] is None
    # The reassembled bytes were transcoded to a derivative and persisted; the
    # chunked path is an upload, so source.type is "upload" with no url.
    assert len(store.save_calls) == 1
    call = store.save_calls[0]
    assert call["audio_m4a"] == FAKE_AUDIO_M4A
    assert call["source"] == {
        "type": "upload", "url": None, "original_filename": "clip.wav",
    }


@pytest.mark.anyio
async def test_chunked_complete_honors_consent_false(client, store):
    upload_id, _ = await _upload_all(client, FIXTURE_WAV, consent=False, store=True)
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stored"] is False
    assert data["storage_note"] == "consent not given"
    assert store.save_calls == []  # nothing persisted


# ---------------------------------------------------------------------------
# Missing chunk → 400 listing the missing indexes
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_missing_chunk_lists_indexes(client, store):
    resp = await _start(client, FIXTURE_WAV)
    body = resp.json()
    upload_id = body["upload_id"]
    chunks = _chunks(FIXTURE_WAV, body["chunk_bytes"])
    assert len(chunks) >= 3
    # Deliberately skip the middle chunk (index 1).
    for i, chunk in enumerate(chunks):
        if i == 1:
            continue
        await _put_chunk(client, upload_id, i, chunk)

    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "missing chunk" in detail.lower()
    assert "1" in detail


# ---------------------------------------------------------------------------
# Oversize total → 413 at start (no bytes uploaded)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_oversize_total_413(client, store):
    resp = await _start(
        client, b"", total_bytes=main.MAX_CHUNKED_UPLOAD_BYTES + 1,
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Oversize chunk → 413 at PUT (chunk size + slack narrowed for the test)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_oversize_chunk_413(client, store, monkeypatch):
    monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", 1024)
    monkeypatch.setattr(main, "CHUNK_SLACK_BYTES", 0)
    resp = await _start(client, b"x" * 4096, total_bytes=4096)
    upload_id = resp.json()["upload_id"]
    # 2000 bytes > 1024-byte limit (slack 0) → 413.
    r = await _put_chunk(client, upload_id, 0, b"y" * 2000)
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Index >= expected_chunks → 409
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_index_out_of_range_409(client, store):
    resp = await _start(client, FIXTURE_WAV)
    body = resp.json()
    upload_id = body["upload_id"]
    r = await _put_chunk(client, upload_id, body["expected_chunks"], b"late")
    assert r.status_code == 409
    assert "out of range" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Idempotent re-PUT — a re-sent index overwrites the previous bytes
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_reput_overwrites(client, store):
    resp = await _start(client, FIXTURE_WAV)
    body = resp.json()
    upload_id = body["upload_id"]
    chunks = _chunks(FIXTURE_WAV, body["chunk_bytes"])
    # First send chunk 0 as WRONG bytes (same length), then overwrite it.
    wrong = b"\x00" * len(chunks[0])
    r1 = await _put_chunk(client, upload_id, 0, wrong)
    assert r1.status_code == 200
    r2 = await _put_chunk(client, upload_id, 0, chunks[0])  # overwrite
    assert r2.status_code == 200
    for i in range(1, len(chunks)):
        await _put_chunk(client, upload_id, i, chunks[i])

    # Assembled bytes are the CORRECT fixture again (overwrite took effect).
    assembled = await store.assemble_upload(
        "test-user", upload_id, body["expected_chunks"],
    )
    assert assembled == FIXTURE_WAV

    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Cross-uid isolation — user-b cannot PUT to or complete user-a's upload
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_cross_uid_404(client, store):
    resp = await _start(client, FIXTURE_WAV, uid="user-a")
    body = resp.json()
    upload_id = body["upload_id"]
    # user-b cannot PUT a chunk to it...
    r_put = await _put_chunk(client, upload_id, 0, b"x", uid="user-b")
    assert r_put.status_code == 404
    # ...nor complete it.
    p1, p2 = _patched_upload()
    with p1, p2:
        r_complete = await client.post(
            f"/uploads/{upload_id}/complete",
            headers={"X-Test-Uid": "user-b"},
        )
    assert r_complete.status_code == 404


@pytest.mark.anyio
async def test_chunked_complete_unknown_upload_404(client, store):
    missing = str(uuid.uuid4())
    p1, p2 = _patched_upload()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{missing}/complete",
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Abort (DELETE) → 204 + cleanup; idempotent for an unknown id
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_abort_deletes_and_is_idempotent(client, store):
    upload_id, _ = await _upload_all(client, FIXTURE_WAV)
    resp = await client.delete(
        f"/uploads/{upload_id}", headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 204
    assert store._uploads.get("test-user", {}).get(upload_id) is None
    # Aborting an unknown id is still a clean 204 (never confirms existence).
    again = await client.delete(
        f"/uploads/{uuid.uuid4()}", headers={"X-Test-Uid": "test-user"},
    )
    assert again.status_code == 204


# ---------------------------------------------------------------------------
# Storage disabled → 503 on every /uploads endpoint
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_storage_disabled_503(client):
    # No store fixture → app.state has no recordings_store → disabled.
    uid = {"X-Test-Uid": "test-user"}
    fake_id = str(uuid.uuid4())
    start = await client.post(
        "/uploads/start",
        json={"total_bytes": 1000, "filename": "a.wav", "content_type": "audio/wav"},
        headers=uid,
    )
    assert start.status_code == 503
    assert "storage" in start.json()["detail"].lower()
    assert (await client.put(
        f"/uploads/{fake_id}/chunks/0", content=b"x", headers=uid,
    )).status_code == 503
    p1, p2 = _patched_upload()
    with p1, p2:
        assert (await client.post(
            f"/uploads/{fake_id}/complete", headers=uid,
        )).status_code == 503
    assert (await client.delete(
        f"/uploads/{fake_id}", headers=uid,
    )).status_code == 503


# ---------------------------------------------------------------------------
# Auth — dropping the override hits the real dependency → 401
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_chunked_start_requires_auth_401(client, store, monkeypatch):
    from auth import get_current_uid
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post(
        "/uploads/start",
        json={"total_bytes": 1000, "filename": "a.wav", "content_type": "audio/wav"},
    )
    assert resp.status_code == 401
