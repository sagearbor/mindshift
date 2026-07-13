"""Bug 1 — transcribe_prerecorded downmixes to 16 kHz mono before Deepgram.

A real two-speaker phone recording proved nova-3's diarizer COLLAPSES both voices
into a single speaker on 48 kHz input, yet splits them correctly at 16 kHz. So
transcribe_prerecorded now sends a 16 kHz mono WAV downmix (and falls back to the
raw container only when the downmix is unavailable — never fabricating audio).

These tests mock Deepgram's HTTP entirely (no network) and assert what bytes we
actually put on the wire.
"""

import io
import wave

import numpy as np

import audio_ingest


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


_TWO_SPEAKER_PAYLOAD = {
    "results": {
        "utterances": [
            {"speaker": 0, "transcript": "You wanted the cat.",
             "start": 0.0, "end": 1.0},
            {"speaker": 1, "transcript": "I am allergic to cats.",
             "start": 1.0, "end": 2.0},
        ]
    }
}


def _read_wav(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getframerate(), wf.getsampwidth()


def test_transcribe_sends_16k_mono_wav(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
    # Avoid real ffmpeg: pretend decode yields 16 kHz mono PCM (what the ffmpeg
    # path always produces).
    pcm = (0.2 * np.sin(np.linspace(0, 40, 16000))).astype(np.float32)
    monkeypatch.setattr(
        audio_ingest, "_decode_via_ffmpeg", lambda data, filename="": (pcm, 16000),
    )
    captured = {}

    def _fake_post(url, *, params, headers, content, timeout):
        captured["content"] = content
        captured["content_type"] = headers["Content-Type"]
        return _FakeResp(_TWO_SPEAKER_PAYLOAD)

    monkeypatch.setattr(audio_ingest.httpx, "post", _fake_post)

    turns = audio_ingest.transcribe_prerecorded(b"raw-video-bytes", "video/mp4")

    # We sent a 16 kHz mono 16-bit WAV, NOT the raw 122MB video container.
    assert captured["content_type"] == "audio/wav"
    assert _read_wav(captured["content"]) == (1, 16000, 2)
    # Distinct Deepgram speaker indices map to distinct labels (the fix's payoff).
    assert [t["speaker"] for t in turns] == ["Speaker A", "Speaker B"]


def test_transcribe_falls_back_to_raw_when_downmix_unavailable(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    def _boom(data, filename=""):
        raise audio_ingest.AudioDecodeError("ffmpeg unavailable")

    monkeypatch.setattr(audio_ingest, "_decode_via_ffmpeg", _boom)
    captured = {}

    def _fake_post(url, *, params, headers, content, timeout):
        captured["content"] = content
        captured["content_type"] = headers["Content-Type"]
        return _FakeResp(_TWO_SPEAKER_PAYLOAD)

    monkeypatch.setattr(audio_ingest.httpx, "post", _fake_post)

    turns = audio_ingest.transcribe_prerecorded(b"RAW-BYTES", "audio/mpeg")

    # Downmix unavailable → raw container bytes + original content-type (Deepgram
    # decodes it itself — honest backstop, never fabricated).
    assert captured["content"] == b"RAW-BYTES"
    assert captured["content_type"] == "audio/mpeg"
    assert len(turns) == 2


def test_pcm_to_wav16_roundtrips_mono_16bit():
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    wav = audio_ingest._pcm_to_wav16(pcm, 16000)
    assert _read_wav(wav) == (1, 16000, 2)
