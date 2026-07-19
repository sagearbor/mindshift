"""Tests for POST /recordings/{id}/reanalyze — the submit-and-poll re-analysis job.

Re-analysis re-runs the CURRENT full pipeline over a recording's STORED audio
(audio.m4a), then overwrites analysis.json + turns.json in place and stamps
meta.reanalyzed_at — the id/title/source and derivatives are preserved. Like the
other job suites (test_analyze_jobs), everything is mocked against an in-memory
fake store injected at ``app.state.recordings_store`` — GCS/Deepgram/LLM are
never touched. The stored audio is a real tiny WAV so the decode/prosody stages
run for real; transcription + analysis are patched.
"""

import asyncio
import io
import wave
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import main
from main import app, init_db

SR = 16000


# ---------------------------------------------------------------------------
# Audio fixture (mirrors the other upload/job suites)
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
GOOD_UUID = "11111111-1111-4111-8111-111111111111"


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
# In-memory fake store — the recording read/audio/overwrite + job methods used
# by the reanalyze endpoint (interface matches RecordingsStore).
# ---------------------------------------------------------------------------

class FakeReanalyzeStore:
    def __init__(self):
        # {uid: {rid: {"meta":..., "turns":..., "analysis":..., "audio":bytes}}}
        self._recordings: dict = {}
        self.overwrite_calls: list = []
        self._jobs: dict = {}
        self.status_history: dict = {}

    def seed(self, uid, rid, *, audio, title="My chat", source=None,
             turns=None, analysis=None):
        self._recordings.setdefault(uid, {})[rid] = {
            "meta": {
                "id": rid, "created_at": "2026-01-01T00:00:00+00:00",
                "filename": "audio.m4a", "title": title, "media_type": "audio",
                "duration_seconds": 6.0, "storage_note": None,
                "source": source or {"type": "upload", "url": None,
                                     "original_filename": "audio.m4a"},
            },
            "turns": turns if turns is not None else [],
            "analysis": analysis,
            "audio": audio,
        }

    # -- recording reads ---------------------------------------------------
    async def get_recording(self, uid, recording_id):
        r = self._recordings.get(uid, {}).get(recording_id)
        if r is None:
            return None
        return {**r["meta"], "turns": r["turns"], "analysis": r["analysis"]}

    async def get_audio_bytes(self, uid, recording_id):
        r = self._recordings.get(uid, {}).get(recording_id)
        return None if r is None else r["audio"]

    async def overwrite_analysis(self, uid, recording_id, *, turns, analysis,
                                 reanalyzed_at):
        r = self._recordings.get(uid, {}).get(recording_id)
        if r is None:
            return None
        r["turns"] = turns
        r["analysis"] = analysis
        r["meta"]["reanalyzed_at"] = reanalyzed_at
        self.overwrite_calls.append({
            "uid": uid, "recording_id": recording_id, "turns": turns,
            "analysis": analysis, "reanalyzed_at": reanalyzed_at,
        })
        return dict(r["meta"])

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
def store():
    fake = FakeReanalyzeStore()
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
    for _ in range(200):
        tasks = [t for t in main._JOB_TASKS if not t.done()]
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


async def _get_job(client, job_id, uid="test-user"):
    return await client.get(
        f"/analyze/jobs/{job_id}", headers={"X-Test-Uid": uid},
    )


async def _post_reanalyze(client, rid, uid="test-user"):
    return await client.post(
        f"/recordings/{rid}/reanalyze", headers={"X-Test-Uid": uid},
    )


# ---------------------------------------------------------------------------
# Happy path — job runs to done, overwrites in place, preserves metadata
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_reanalyze_happy_path_overwrites_and_stamps(client, store):
    src = {"type": "link", "url": "https://example.com/x", "original_filename": "x"}
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV, title="Sunday talk",
               source=src, turns=[{"speaker": "old", "text": "stale"}],
               analysis={"narrative": "old", "word_metrics": None})

    p1, p2 = _patched_pipeline()
    with p1, p2:
        resp = await _post_reanalyze(client, GOOD_UUID)
        assert resp.status_code == 202, resp.text
        job_id = resp.json()["job_id"]

        await _drain_jobs()
        done = await _get_job(client, job_id)

    body = done.json()
    assert body["status"] == "done"
    assert body["error"] is None
    result = body["result"]
    assert len(result["per_turn"]) == len(MOCK_TURNS)
    # Re-analysis honestly reports the recording it (re)stored into.
    assert result["stored"] is True
    assert result["recording_id"] == GOOD_UUID
    # The fresh transcript + the new word_metrics field landed in the result.
    assert [t["speaker"] for t in result["turns"]] == \
        [t["speaker"] for t in MOCK_TURNS]
    assert result["word_metrics"] is not None
    assert set(result["word_metrics"]["speakers"]) == set(_SPEAKERS)
    # Title is preserved (no new title requested for an already-titled recording).
    assert result["title"] == "Sunday talk"

    # Persisted in place: turns/analysis overwritten, reanalyzed_at stamped,
    # id/title/source untouched.
    assert len(store.overwrite_calls) == 1
    call = store.overwrite_calls[0]
    assert call["recording_id"] == GOOD_UUID
    assert [t["speaker"] for t in call["turns"]] == [t["speaker"] for t in MOCK_TURNS]
    assert call["analysis"]["word_metrics"] is not None
    stored = store._recordings["test-user"][GOOD_UUID]
    assert stored["meta"]["reanalyzed_at"] == call["reanalyzed_at"]
    assert stored["meta"]["title"] == "Sunday talk"
    assert stored["meta"]["source"] == src


@pytest.mark.anyio
async def test_reanalyze_job_lifecycle_passes_pipeline_stages(client, store):
    store.seed("test-user", GOOD_UUID, audio=FIXTURE_WAV)
    p1, p2 = _patched_pipeline()
    with p1, p2:
        resp = await _post_reanalyze(client, GOOD_UUID)
        job_id = resp.json()["job_id"]
        await _drain_jobs()

    history = store.status_history[job_id]
    assert history[0] == "queued"
    assert history[-1] == "done"
    assert "transcribing" in history
    assert "analyzing" in history
    assert "storing" in history


# ---------------------------------------------------------------------------
# Errors — 404 unknown/foreign, 422 no audio, 503 storage off
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_reanalyze_unknown_recording_404(client, store):
    resp = await _post_reanalyze(client, GOOD_UUID)
    assert resp.status_code == 404
    assert store.status_history == {}  # no job spawned


@pytest.mark.anyio
async def test_reanalyze_foreign_recording_404(client, store):
    # Seeded for user-a; user-b must not be able to re-analyze it.
    store.seed("user-a", GOOD_UUID, audio=FIXTURE_WAV)
    resp = await _post_reanalyze(client, GOOD_UUID, uid="user-b")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_reanalyze_no_stored_audio_422(client, store):
    # Recording exists but its audio derivative is absent (analyze-only / partial).
    store.seed("test-user", GOOD_UUID, audio=None)
    resp = await _post_reanalyze(client, GOOD_UUID)
    assert resp.status_code == 422
    assert "audio" in resp.json()["detail"].lower()
    assert store.status_history == {}


@pytest.mark.anyio
async def test_reanalyze_503_when_storage_disabled(client):
    # No `store` fixture → app.state has no recordings_store → storage disabled.
    resp = await client.post(
        f"/recordings/{GOOD_UUID}/reanalyze",
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503
    assert "storage" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_reanalyze_bad_uuid_422(client, store):
    # The path pattern rejects a non-uuid id before the handler runs.
    resp = await client.post(
        "/recordings/not-a-uuid/reanalyze",
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_reanalyze_requires_auth_401(client, store, monkeypatch):
    from auth import get_current_uid
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post(f"/recordings/{GOOD_UUID}/reanalyze")
    assert resp.status_code == 401
