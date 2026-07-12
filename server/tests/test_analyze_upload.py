"""Fully-mocked tests for POST /analyze/upload.

Deepgram (transcription) and the LLM are both mocked, and the audio fixture is a
tiny stdlib-WAV built from numpy — so the whole endpoint is deterministic and
key-free. Covers: multipart happy path (with real prosody labels), the
decode-fail honest-degrade path, the no-key 503, the oversize 413, and auth 401.

The /analyze regression suite (test_dynamics.py) guards that the shared-helper
refactor did not change the text path.
"""

import io
import json
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import main
from auth import get_current_uid
from main import app, init_db

SR = 16000


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Fixtures: a tiny stdlib WAV + aligned mock transcript / mock LLM output
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


# Six alternating one-second turns; amplitudes vary so energy labels differ.
_AMPS = [0.1, 0.2, 0.5, 0.15, 0.3, 0.08]
FIXTURE_PCM = np.concatenate([_sine(180.0, 1.0, a) for a in _AMPS]).astype(np.float32)
FIXTURE_WAV = _wav_bytes(FIXTURE_PCM)

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
    per_turn = [
        {"heat": 20 + i * 3, "markers": [], "trigger_phrase": None}
        for i in range(n_turns)
    ]
    return json.dumps({
        "per_turn": per_turn,
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


# ---------------------------------------------------------------------------
# Happy path — transcript echoed, per-turn voice labels present, note absent
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_happy_path_with_voice_labels(client):
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
            data={"context": "Argument about chores"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Core AnalyzeResponse shape is intact.
    assert len(data["per_turn"]) == len(MOCK_TURNS)
    assert set(data["per_speaker"]) == {"Speaker A", "Speaker B"}
    assert data["narrative"]
    assert set(data["report_cards"]) == {"Speaker A", "Speaker B"}

    # The transcript the client never had is echoed back.
    assert [t["speaker"] for t in data["turns"]] == [t["speaker"] for t in MOCK_TURNS]
    assert data["turns"][0]["text"] == MOCK_TURNS[0]["text"]
    assert data["turns"][0]["start_time"] == 0.0

    # Every per-turn entry carries voice labels (prosody ran on real WAV).
    for pt in data["per_turn"]:
        assert pt["voice"] is not None
        assert pt["voice"]["energy_label"] in {"quiet", "normal", "loud"}
        assert pt["voice"]["rate_label"] in {"slow", "normal", "fast"}
    # Loudest turn (idx 2, amp 0.5) is louder than the quietest (idx 5, amp 0.08).
    energies = [pt["voice"]["energy_label"] for pt in data["per_turn"]]
    assert energies[2] == "loud"
    assert energies[5] == "quiet"

    # Prosody succeeded → no degrade note.
    assert data["voice_analysis"] is None


@pytest.mark.anyio
async def test_upload_prompt_gets_voice_annotation_and_addendum(client):
    """When voice labels exist the LLM prompt gains the voice addendum and each
    numbered turn line carries a [voice: …] cue."""
    mock = _mock_llm(_analyze_llm_json(len(MOCK_TURNS)))
    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.get_llm_client", return_value=mock):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 200
    system = mock.complete.call_args.kwargs["system"]
    user = mock.complete.call_args.kwargs["user"]
    assert "voice annotation" in system  # addendum present
    assert "[voice:" in user             # per-turn cue present


# ---------------------------------------------------------------------------
# Honest degrade — transcription OK but audio undecodable → voice null + note
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_decode_failure_degrades_honestly(client):
    def _boom(_data, _filename):
        raise audio_ingest.AudioDecodeError("could not decode this file")

    with patch("main.transcribe_prerecorded", return_value=MOCK_TURNS), \
         patch("main.decode_to_pcm", side_effect=_boom), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(MOCK_TURNS)))):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.mp4", b"not really audio", "video/mp4")},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Analysis still ran (transcript existed) but every turn's voice is null.
    assert all(pt["voice"] is None for pt in data["per_turn"])
    assert data["voice_analysis"].startswith("unavailable:")
    # Transcript is still returned.
    assert len(data["turns"]) == len(MOCK_TURNS)


# ---------------------------------------------------------------------------
# Honest unavailability — no DEEPGRAM_API_KEY → 503 (never a mock transcript)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_no_key_is_503(client, monkeypatch):
    # Force the key absent so the REAL transcribe_prerecorded reports unavailable
    # before any network call.
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    resp = await client.post(
        "/analyze/upload",
        files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# No speech found → 422
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_no_speech_is_422(client):
    def _empty(_data, _ct):
        raise audio_ingest.NoSpeechFound("no speech found in this recording")

    with patch("main.transcribe_prerecorded", side_effect=_empty):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    assert resp.status_code == 422
    assert "no speech" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Too few speakers/turns from transcription → 422 (reuses AnalyzeRequest bounds)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_single_speaker_transcript_succeeds(client):
    solo = [
        {"speaker": "Speaker A", "text": "just me talking",
         "start_time": float(i), "end_time": float(i) + 0.5}
        for i in range(5)
    ]
    payload = json.dumps({
        "per_turn": [
            {"heat": 20, "markers": [], "trigger_phrase": None}
            for _ in range(5)
        ],
        "requests": [],
        "narrative": "One voice, honestly analyzed.",
        "report_cards": {"Speaker A": {
            "score": 70, "headline": "Solo",
            "did_well": "spoke", "work_on": "pausing",
        }},
    })
    with patch("main.transcribe_prerecorded", return_value=solo), \
         patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        )
    # A merged-diarization / monologue recording is analyzable now — this is
    # exactly the "one person performing two voices" real-world case.
    assert resp.status_code == 200
    assert set(resp.json()["per_speaker"]) == {"Speaker A"}


# ---------------------------------------------------------------------------
# Caps — oversize file → 413 (cap monkeypatched small to keep the test fast)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_oversize_is_413(client, monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 16)
    resp = await client.post(
        "/analyze/upload",
        files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
    )
    assert resp.status_code == 413
    assert "too large" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Auth — drop the override to hit the real dependency → 401
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_upload_requires_auth_401(client, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post(
        "/analyze/upload",
        files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
    )
    assert resp.status_code == 401
