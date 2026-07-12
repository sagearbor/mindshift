"""Tests for the submit-and-poll async analysis jobs.

POST /analyze/link/jobs and POST /uploads/{id}/complete/jobs return 202 {job_id}
immediately and run the SAME pipeline as their synchronous siblings as an
in-process background task, recording staged progress the client polls via
GET /analyze/jobs/{job_id}. Everything is fully mocked (fetch_link, Deepgram,
LLM, ffmpeg derivatives) against an in-memory fake store injected at
``app.state.recordings_store`` — GCS is never touched — mirroring
test_chunked_upload / test_analyze_link.
"""

import asyncio
import io
import json
import wave
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import link_fetch
import main
from main import app, init_db

SR = 16000
FAKE_AUDIO_M4A = b"FAKE-M4A-AUDIO-DERIVATIVE-" * 20


# ---------------------------------------------------------------------------
# Audio fixture (mirrors the other upload suites)
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


def _analyze_llm_json(n_turns: int) -> str:
    return json.dumps({
        "per_turn": [
            {"heat": 20 + i * 3, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both keep showing up and trying to reconnect.",
        "report_cards": {
            sp: {"score": 70, "headline": f"{sp} stayed engaged",
                 "did_well": "Kept trying to reconnect.",
                 "work_on": "Pause before answering criticism."}
            for sp in _SPEAKERS
        },
    })


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


# ---------------------------------------------------------------------------
# In-memory fake store — recording + upload + job methods (interface matches
# RecordingsStore). Records the ordered status history per job so the queued→
# …→done transition sequence can be asserted deterministically.
# ---------------------------------------------------------------------------

class FakeJobStore:
    def __init__(self):
        self._recordings: dict = {}
        self.save_calls: list = []
        self._uploads: dict = {}
        self.cleanup_calls: list = []
        # jobs: {uid: {job_id: state}}; history: {job_id: [status, ...]}
        self._jobs: dict = {}
        self.status_history: dict = {}

    # -- recording persistence (store=true path) ---------------------------
    async def save_recording(
        self, uid, *, audio_m4a, video_360p, original_filename,
        original_content_type, original_bytes, duration_seconds, turns,
        analysis, source=None,
    ):
        import uuid
        rid = str(uuid.uuid4())
        self._recordings.setdefault(uid, {})[rid] = {"source": source}
        self.save_calls.append({"uid": uid, "recording_id": rid,
                                "source": source})
        return rid

    # -- chunked upload sessions -------------------------------------------
    async def write_upload_manifest(self, uid, upload_id, manifest):
        self._uploads.setdefault(uid, {})[upload_id] = {
            "manifest": manifest, "parts": {},
        }

    async def read_upload_manifest(self, uid, upload_id):
        sess = self._uploads.get(uid, {}).get(upload_id)
        return None if sess is None else sess["manifest"]

    async def write_upload_part(self, uid, upload_id, index, data):
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

    # -- async jobs --------------------------------------------------------
    async def write_job_state(self, uid, job_id, state):
        self._jobs.setdefault(uid, {})[job_id] = dict(state)
        self.status_history.setdefault(job_id, []).append(state["status"])

    async def read_job_state(self, uid, job_id):
        state = self._jobs.get(uid, {}).get(job_id)
        return None if state is None else dict(state)

    async def delete_job(self, uid, job_id):
        self._jobs.get(uid, {}).pop(job_id, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    """Storage-enabled fake + deterministic ffmpeg derivatives (never real ffmpeg).
    Small chunk size so the tiny WAV genuinely spans several parts for the
    complete/jobs path."""
    monkeypatch.setattr(main, "UPLOAD_CHUNK_BYTES", 64 * 1024)
    monkeypatch.setattr(
        main, "build_derivatives",
        lambda data, **kw: audio_ingest.Derivatives(
            audio_m4a=FAKE_AUDIO_M4A, video_360p=None, has_video=False,
            video_note=None,
        ),
    )
    fake = FakeJobStore()
    app.state.recordings_store = fake
    yield fake
    del app.state.recordings_store


def _patched_pipeline():
    return (
        patch("main.transcribe_prerecorded", return_value=MOCK_TURNS),
        patch("main.get_llm_client",
              return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))),
    )


async def _drain_jobs():
    """Await every spawned background job task to completion."""
    for _ in range(200):
        tasks = [t for t in main._JOB_TASKS if not t.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


async def _get_job(client, job_id, uid="test-user"):
    return await client.get(
        f"/analyze/jobs/{job_id}", headers={"X-Test-Uid": uid},
    )


# ---------------------------------------------------------------------------
# /analyze/link/jobs — happy path: queued → … → done with the full result
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_link_job_happy_path_polls_to_done(client, store):
    p1, p2 = _patched_pipeline()
    with p1, p2, patch(
        "main.link_fetch.fetch_link",
        return_value=(FIXTURE_WAV, "clip.wav", "audio/wav"),
    ):
        resp = await client.post(
            "/analyze/link/jobs",
            json={"url": "https://example.com/clip.wav", "consent": False},
            headers={"X-Test-Uid": "test-user"},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        # Immediately pollable — a valid state, result withheld until done.
        first = await _get_job(client, job_id)
        assert first.status_code == 200
        assert first.json()["status"] in {
            "queued", "downloading", "transcribing", "analyzing", "storing",
            "done",
        }

        await _drain_jobs()
        done = await _get_job(client, job_id)

    assert done.status_code == 200
    body = done.json()
    assert body["status"] == "done"
    assert body["error"] is None
    # The full AnalyzeUploadResponse is carried only when done.
    result = body["result"]
    assert result is not None
    assert len(result["per_turn"]) == len(MOCK_TURNS)
    assert set(result["report_cards"]) == {"Speaker A", "Speaker B"}
    assert [t["speaker"] for t in result["turns"]] == \
        [t["speaker"] for t in MOCK_TURNS]
    # consent False → not persisted, honest note.
    assert result["stored"] is False
    assert result["storage_note"] == "consent not given"

    # The persisted transitions started at queued and ended at done, passing
    # through the pipeline stages — never an eternal spinner.
    history = store.status_history[job_id]
    assert history[0] == "queued"
    assert history[-1] == "done"
    assert "transcribing" in history
    assert "analyzing" in history
    # duration_seconds was recorded once the audio decoded (enables client ETA).
    assert body["duration_seconds"] is not None and body["duration_seconds"] > 0


# ---------------------------------------------------------------------------
# Failure writes an honest error (the same detail the sync path would 4xx with)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_link_job_failure_writes_honest_error(client, store):
    def _boom(url, **kw):
        raise link_fetch.LinkError(
            422, "link resolves to a private/internal address — not allowed",
        )

    with patch("main.link_fetch.fetch_link", _boom):
        resp = await client.post(
            "/analyze/link/jobs",
            json={"url": "https://internal.example.com/x"},
            headers={"X-Test-Uid": "test-user"},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        await _drain_jobs()
        done = await _get_job(client, job_id)

    body = done.json()
    assert body["status"] == "failed"
    assert body["result"] is None
    # Verbatim honest detail, not a generic message.
    assert "private" in body["error"].lower()


# ---------------------------------------------------------------------------
# Stalled computation — a non-terminal state that stopped advancing
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_stalled_state_reported_on_read(client, store):
    import uuid
    stale = datetime.now(timezone.utc) - timedelta(seconds=main.JOB_STALL_SECONDS + 10)
    iso = stale.isoformat()
    # A valid uuid-shaped id is required by the path pattern.
    job_id = str(uuid.uuid4())
    await store.write_job_state("test-user", job_id, {
        "status": "transcribing", "created_at": iso, "updated_at": iso,
        "stage_started_at": iso, "progress_note": "38 MB to transcribe",
        "duration_seconds": None, "error": None, "result": None,
    })

    resp = await _get_job(client, job_id)
    assert resp.status_code == 200
    body = resp.json()
    # Computed on read — never stored as "stalled".
    assert body["status"] == "stalled"
    assert "stalled" in body["progress_note"].lower()
    assert store._jobs["test-user"][job_id]["status"] == "transcribing"


# ---------------------------------------------------------------------------
# uid isolation + unknown job → 404
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_job_uid_isolation(client, store):
    with patch("main.link_fetch.fetch_link",
               return_value=(FIXTURE_WAV, "clip.wav", "audio/wav")):
        p1, p2 = _patched_pipeline()
        with p1, p2:
            resp = await client.post(
                "/analyze/link/jobs",
                json={"url": "https://example.com/clip.wav"},
                headers={"X-Test-Uid": "user-a"},
            )
            job_id = resp.json()["job_id"]
            await _drain_jobs()

    # Owner sees it; a different uid gets a 404 (never another user's job).
    assert (await _get_job(client, job_id, uid="user-a")).status_code == 200
    assert (await _get_job(client, job_id, uid="user-b")).status_code == 404


@pytest.mark.anyio
async def test_unknown_job_404(client, store):
    import uuid
    resp = await _get_job(client, str(uuid.uuid4()))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TTL — a terminal state older than 24h is lazily deleted on read
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_terminal_job_past_ttl_deleted_on_read(client, store):
    import uuid
    old = datetime.now(timezone.utc) - timedelta(seconds=main.JOB_TTL_SECONDS + 60)
    iso = old.isoformat()
    job_id = str(uuid.uuid4())
    await store.write_job_state("test-user", job_id, {
        "status": "done", "created_at": iso, "updated_at": iso,
        "stage_started_at": iso, "progress_note": None,
        "duration_seconds": 6.0, "error": None, "result": {"x": 1},
    })
    resp = await _get_job(client, job_id)
    assert resp.status_code == 404
    # Lazily cleaned up.
    assert store._jobs.get("test-user", {}).get(job_id) is None


# ---------------------------------------------------------------------------
# Storage disabled → 503 on job POSTs, but the SYNC endpoints still work
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_link_job_503_when_storage_disabled(client):
    # No `store` fixture → app.state has no recordings_store → storage disabled.
    resp = await client.post(
        "/analyze/link/jobs",
        json={"url": "https://example.com/clip.wav"},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503
    assert "storage" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_complete_job_503_when_storage_disabled(client):
    import uuid
    resp = await client.post(
        f"/uploads/{uuid.uuid4()}/complete/jobs",
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_get_job_503_when_storage_disabled(client):
    import uuid
    resp = await _get_job(client, str(uuid.uuid4()))
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_sync_link_still_works_with_storage_disabled(client):
    """The OLD synchronous endpoint must keep working when jobs are unavailable
    (it is what old clients use, and its process-and-discard path needs no store)."""
    with patch("main.link_fetch.fetch_link",
               return_value=(FIXTURE_WAV, "clip.wav", "audio/wav")):
        p1, p2 = _patched_pipeline()
        with p1, p2:
            resp = await client.post(
                "/analyze/link",
                json={"url": "https://example.com/clip.wav", "consent": True},
                headers={"X-Test-Uid": "test-user"},
            )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert data["stored"] is False
    # Consent given, but storage is off — the honest reason (not "consent").
    assert data["storage_note"] == "storage not enabled"


# ---------------------------------------------------------------------------
# /uploads/{id}/complete/jobs — happy path + synchronous part validation
# ---------------------------------------------------------------------------

def _chunks(data: bytes, size: int) -> list:
    return [data[i:i + size] for i in range(0, len(data), size)]


async def _upload_all(client, data, uid="test-user", **overrides):
    body = {"filename": "clip.wav", "content_type": "audio/wav",
            "total_bytes": len(data), "consent": False, "store": True,
            **overrides}
    resp = await client.post(
        "/uploads/start", json=body, headers={"X-Test-Uid": uid},
    )
    assert resp.status_code == 200, resp.text
    b = resp.json()
    upload_id = b["upload_id"]
    for i, chunk in enumerate(_chunks(data, b["chunk_bytes"])):
        r = await client.put(
            f"/uploads/{upload_id}/chunks/{i}", content=chunk,
            headers={"X-Test-Uid": uid},
        )
        assert r.status_code == 200, r.text
    return upload_id, b["expected_chunks"]


@pytest.mark.anyio
async def test_complete_job_happy_path_and_cleanup(client, store):
    upload_id, expected = await _upload_all(client, FIXTURE_WAV)
    assert expected > 1  # genuinely multi-part

    p1, p2 = _patched_pipeline()
    with p1, p2:
        resp = await client.post(
            f"/uploads/{upload_id}/complete/jobs",
            headers={"X-Test-Uid": "test-user"},
        )
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]
        await _drain_jobs()
        done = await _get_job(client, job_id)

    body = done.json()
    assert body["status"] == "done"
    result = body["result"]
    assert len(result["per_turn"]) == len(MOCK_TURNS)
    # Parts cleaned up after the job finished.
    assert ("test-user", upload_id) in store.cleanup_calls


@pytest.mark.anyio
async def test_complete_job_validates_missing_parts_synchronously(client, store):
    # Start + upload all but one chunk, then request the JOB — the missing-part
    # 400 must come back SYNCHRONOUSLY (fail fast, no job spawned).
    body = {"filename": "clip.wav", "content_type": "audio/wav",
            "total_bytes": len(FIXTURE_WAV), "consent": False, "store": True}
    resp = await client.post(
        "/uploads/start", json=body, headers={"X-Test-Uid": "test-user"},
    )
    b = resp.json()
    upload_id = b["upload_id"]
    chunks = _chunks(FIXTURE_WAV, b["chunk_bytes"])
    assert len(chunks) >= 3
    for i, chunk in enumerate(chunks):
        if i == 1:
            continue  # skip the middle chunk
        await client.put(
            f"/uploads/{upload_id}/chunks/{i}", content=chunk,
            headers={"X-Test-Uid": "test-user"},
        )

    resp = await client.post(
        f"/uploads/{upload_id}/complete/jobs",
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 400
    assert "missing chunk" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_complete_job_unknown_upload_404(client, store):
    import uuid
    resp = await client.post(
        f"/uploads/{uuid.uuid4()}/complete/jobs",
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 404
