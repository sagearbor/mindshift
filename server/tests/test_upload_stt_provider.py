"""Provider selection + honest fallback for the prerecorded-upload STT path.

``MINDSHIFT_UPLOAD_STT`` picks the transcription backend for POST
/analyze/upload (and the link/chunked paths that share the helper):

* ``deepgram`` (default) — the existing vendor path, UNCHANGED. When Deepgram
  is unavailable (missing key, HTTP failure) and faster-whisper IS installed,
  the upload falls back to local Whisper and the switch is surfaced honestly
  in the response's ``transcription_note`` (never silent). Without
  faster-whisper installed the existing 503 behavior is untouched.
* ``whisper`` — local faster-whisper. A whisper failure is an HONEST error;
  it never silently falls back to the vendor (the point is de-vendoring).

Whisper produces NO diarization: every turn is "Speaker A" with word timings,
and the pipeline's local ECAPA cross-check (diarize_local) attributes speakers.

Everything here is mocked — no network, and no faster-whisper install needed.
"""

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import audio_ingest
import whisper_transcriber
from audio_ingest import (
    NoSpeechFound,
    TranscriptionUnavailable,
    transcribe_prerecorded_whisper,
    transcribe_upload,
)
from audio_pipeline import TranscriberUnavailable

SR = 16000

TURNS = [
    {"speaker": "Speaker A", "text": "Hey, can we talk about the schedule?",
     "start_time": 0.0, "end_time": 1.0},
    {"speaker": "Speaker B", "text": "Sure, what about it.",
     "start_time": 1.0, "end_time": 2.0},
    {"speaker": "Speaker A", "text": "You never stick to what we agree.",
     "start_time": 2.0, "end_time": 3.0},
    {"speaker": "Speaker B", "text": "That is not fair and you know it.",
     "start_time": 3.0, "end_time": 4.0},
]


@pytest.fixture(autouse=True)
def _no_upload_stt_env(monkeypatch):
    """Each test opts in to MINDSHIFT_UPLOAD_STT explicitly."""
    monkeypatch.delenv("MINDSHIFT_UPLOAD_STT", raising=False)


# ---------------------------------------------------------------------------
# transcribe_upload — provider selection
# ---------------------------------------------------------------------------

def test_default_provider_is_deepgram(monkeypatch):
    called = {}

    def fake_deepgram(data, content_type):
        called["deepgram"] = (data, content_type)
        return TURNS

    monkeypatch.setattr(audio_ingest, "transcribe_prerecorded", fake_deepgram)
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda *a, **k: pytest.fail("whisper must not run on the default path"),
    )
    turns, note = transcribe_upload(b"bytes", "audio/wav", "a.wav")
    assert turns == TURNS
    assert note is None
    assert called["deepgram"] == (b"bytes", "audio/wav")


def test_env_switches_to_whisper(monkeypatch):
    called = {}

    def fake_whisper(data, content_type=None, filename=""):
        called["whisper"] = (data, content_type, filename)
        return TURNS

    monkeypatch.setenv("MINDSHIFT_UPLOAD_STT", "whisper")
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: pytest.fail("deepgram must not run when whisper is primary"),
    )
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper", fake_whisper,
    )
    turns, note = transcribe_upload(b"bytes", "audio/wav", "a.wav")
    assert turns == TURNS
    assert note is None
    assert called["whisper"][0] == b"bytes"


def test_provider_env_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_UPLOAD_STT", "  DeepGram ")
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded", lambda *a, **k: TURNS,
    )
    turns, note = transcribe_upload(b"x", None)
    assert turns == TURNS and note is None


def test_unknown_provider_is_honest_config_error(monkeypatch):
    monkeypatch.setenv("MINDSHIFT_UPLOAD_STT", "espnet")
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: pytest.fail("must not guess a provider"),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_upload(b"x", None)
    assert "MINDSHIFT_UPLOAD_STT" in str(exc.value)


# ---------------------------------------------------------------------------
# transcribe_upload — deepgram-primary fallback semantics
# ---------------------------------------------------------------------------

def test_deepgram_failure_falls_back_to_whisper_with_note(monkeypatch):
    def failing_deepgram(data, content_type):
        raise TranscriptionUnavailable("transcription request failed: boom")

    monkeypatch.setattr(audio_ingest, "transcribe_prerecorded", failing_deepgram)
    monkeypatch.setattr(audio_ingest, "_whisper_installed", lambda: True)
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda data, content_type=None, filename="": TURNS,
    )
    turns, note = transcribe_upload(b"bytes", "audio/wav", "a.wav")
    assert turns == TURNS
    assert note is not None
    assert "whisper" in note.lower()
    # The note carries the REAL reason the primary was skipped — never silent.
    assert "boom" in note


def test_deepgram_failure_without_whisper_keeps_503_behavior(monkeypatch):
    """No faster-whisper installed → the original TranscriptionUnavailable
    propagates untouched (endpoint keeps returning its honest 503)."""
    original = TranscriptionUnavailable("transcription not configured")
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(audio_ingest, "_whisper_installed", lambda: False)
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda *a, **k: pytest.fail("whisper is not installed — must not be called"),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_upload(b"x", None)
    assert exc.value is original


def test_deepgram_no_speech_does_not_fall_back(monkeypatch):
    """NoSpeechFound means Deepgram WORKED and heard nothing — re-running the
    audio through whisper would be second-guessing a healthy provider."""
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: (_ for _ in ()).throw(NoSpeechFound("no speech")),
    )
    monkeypatch.setattr(audio_ingest, "_whisper_installed", lambda: True)
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda *a, **k: pytest.fail("no-speech must not trigger the fallback"),
    )
    with pytest.raises(NoSpeechFound):
        transcribe_upload(b"x", None)


def test_whisper_primary_never_falls_back_to_deepgram(monkeypatch):
    """De-vendoring: with whisper as PRIMARY, a whisper failure is an honest
    error — Deepgram is never silently substituted."""
    monkeypatch.setenv("MINDSHIFT_UPLOAD_STT", "whisper")
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda *a, **k: (_ for _ in ()).throw(
            TranscriptionUnavailable("faster-whisper not installed")),
    )
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: pytest.fail("whisper primary must not fall back to deepgram"),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_upload(b"x", None)
    assert "whisper" in str(exc.value).lower()


def test_fallback_failure_reports_both_reasons(monkeypatch):
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded",
        lambda *a, **k: (_ for _ in ()).throw(
            TranscriptionUnavailable("deepgram down")),
    )
    monkeypatch.setattr(audio_ingest, "_whisper_installed", lambda: True)
    monkeypatch.setattr(
        audio_ingest, "transcribe_prerecorded_whisper",
        lambda *a, **k: (_ for _ in ()).throw(
            TranscriptionUnavailable("model load failed")),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_upload(b"x", None)
    msg = str(exc.value)
    assert "deepgram down" in msg and "model load failed" in msg


# ---------------------------------------------------------------------------
# transcribe_prerecorded_whisper — segment → turn mapping
# ---------------------------------------------------------------------------

def _word(text, start, end):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(text, start, end, words=None):
    return SimpleNamespace(text=text, start=start, end=end, words=words)


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"audio": audio, "kwargs": kwargs})
        info = SimpleNamespace(language="en", duration=len(audio) / SR)
        return iter(self._segments), info


def _pcm(seconds=1.0, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def test_whisper_segments_map_to_single_speaker_turns(monkeypatch):
    model = _FakeModel([
        _segment(" Hey, can we talk?", 0.0, 1.4, words=[
            _word(" Hey,", 0.0, 0.4),
            _word(" can", 0.4, 0.7),
            _word(" we", 0.7, 0.9),
            _word(" talk?", 0.9, 1.4),
        ]),
        _segment(" Sure.", 1.9, 2.4, words=[_word(" Sure.", 1.9, 2.4)]),
    ])
    monkeypatch.setattr(
        audio_ingest, "decode_to_pcm", lambda data, filename: (_pcm(3.0), SR),
    )
    turns = transcribe_prerecorded_whisper(b"bytes", "audio/wav", model=model)

    # Whisper has NO diarization: every turn is honestly "Speaker A"; speaker
    # attribution is diarize_local's job downstream.
    assert [t["speaker"] for t in turns] == ["Speaker A", "Speaker A"]
    assert turns[0]["text"] == "Hey, can we talk?"
    assert turns[0]["start_time"] == 0.0
    assert turns[0]["end_time"] == 1.4
    # Word timings ride along in the same internal shape as the Deepgram path,
    # with whisper's leading-space word text stripped.
    assert turns[0]["words"] == [
        {"word": "Hey,", "start_time": 0.0, "end_time": 0.4},
        {"word": "can", "start_time": 0.4, "end_time": 0.7},
        {"word": "we", "start_time": 0.7, "end_time": 0.9},
        {"word": "talk?", "start_time": 0.9, "end_time": 1.4},
    ]
    assert turns[1]["text"] == "Sure."
    # word_timestamps must be requested — diarize_local needs the timings.
    assert model.calls[0]["kwargs"].get("word_timestamps") is True


def test_whisper_segment_without_words_omits_words_key(monkeypatch):
    model = _FakeModel([_segment(" Hello there.", 0.0, 1.0, words=None)])
    monkeypatch.setattr(
        audio_ingest, "decode_to_pcm", lambda data, filename: (_pcm(), SR),
    )
    turns = transcribe_prerecorded_whisper(b"bytes", None, model=model)
    assert turns[0]["text"] == "Hello there."
    assert "words" not in turns[0]


def test_whisper_empty_segments_raise_no_speech(monkeypatch):
    model = _FakeModel([_segment("   ", 0.0, 1.0), _segment("", 1.0, 2.0)])
    monkeypatch.setattr(
        audio_ingest, "decode_to_pcm", lambda data, filename: (_pcm(), SR),
    )
    with pytest.raises(NoSpeechFound):
        transcribe_prerecorded_whisper(b"bytes", None, model=model)


def test_whisper_resamples_non_16k_input(monkeypatch):
    """decode_to_pcm's stdlib-WAV path returns the NATIVE rate — whisper
    expects 16 kHz, so a 8 kHz decode must be resampled before the model."""
    model = _FakeModel([_segment(" Hi.", 0.0, 1.0)])
    monkeypatch.setattr(
        audio_ingest, "decode_to_pcm",
        lambda data, filename: (_pcm(1.0, sr=8000), 8000),
    )
    transcribe_prerecorded_whisper(b"bytes", None, model=model)
    audio = model.calls[0]["audio"]
    assert audio.dtype == np.float32
    assert abs(len(audio) - SR) <= 1  # 1 s of audio → ~16000 samples


def test_whisper_unavailable_maps_to_transcription_unavailable(monkeypatch):
    monkeypatch.setattr(
        whisper_transcriber, "load_shared_model",
        lambda *a, **k: (_ for _ in ()).throw(
            TranscriberUnavailable("faster-whisper not installed")),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_prerecorded_whisper(b"bytes", None)
    assert "faster-whisper" in str(exc.value)


def test_whisper_model_failure_is_honest_error(monkeypatch):
    class _Boom:
        def transcribe(self, audio, **kwargs):
            raise RuntimeError("ctranslate2 exploded")

    monkeypatch.setattr(
        audio_ingest, "decode_to_pcm", lambda data, filename: (_pcm(), SR),
    )
    with pytest.raises(TranscriptionUnavailable) as exc:
        transcribe_prerecorded_whisper(b"bytes", None, model=_Boom())
    assert "ctranslate2 exploded" in str(exc.value)


# ---------------------------------------------------------------------------
# Endpoint — the fallback note surfaces in the response (never silent)
# ---------------------------------------------------------------------------

import main  # noqa: E402
from main import app, init_db  # noqa: E402


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


def _wav_bytes(pcm: np.ndarray, sr: int = SR) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


FIXTURE_WAV = _wav_bytes(np.concatenate([_pcm(1.0)] * 4))


def _analyze_llm_json(n_turns: int) -> str:
    speakers = ["Speaker A", "Speaker B"]
    return json.dumps({
        "per_turn": [
            {"heat": 20 + i, "markers": [], "trigger_phrase": None}
            for i in range(n_turns)
        ],
        "requests": [],
        "narrative": "You both keep trying.",
        "report_cards": {
            sp: {"score": 70, "headline": f"{sp} engaged",
                 "did_well": "Kept trying.", "work_on": "Pause first."}
            for sp in speakers
        },
    })


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


@pytest.mark.anyio
async def test_upload_response_carries_transcription_note(client):
    note = "transcribed with local Whisper (small) — Deepgram was unavailable"
    with patch("main.transcribe_upload", return_value=(TURNS, note)), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(TURNS)))):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["transcription_note"] == note


@pytest.mark.anyio
async def test_upload_response_note_is_null_on_primary_success(client):
    with patch("main.transcribe_upload", return_value=(TURNS, None)), \
         patch("main.get_llm_client",
               return_value=_mock_llm(_analyze_llm_json(len(TURNS)))):
        resp = await client.post(
            "/analyze/upload",
            files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
            headers={"X-Test-Uid": "test-user"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["transcription_note"] is None


@pytest.mark.anyio
async def test_no_key_and_no_whisper_still_503(client, monkeypatch):
    """The pre-existing honest 503 (no Deepgram key) is unchanged when
    faster-whisper is not installed."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setattr(audio_ingest, "_whisper_installed", lambda: False)
    resp = await client.post(
        "/analyze/upload",
        files={"file": ("clip.wav", FIXTURE_WAV, "audio/wav")},
        headers={"X-Test-Uid": "test-user"},
    )
    assert resp.status_code == 503
    assert "transcription" in resp.json()["detail"].lower()
