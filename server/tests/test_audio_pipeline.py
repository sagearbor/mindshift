"""Tests for the M2 real-time audio pipeline (WebSocket, transcription, diarization, TTS).

The pipeline depends on external speech providers (Deepgram for transcription,
a TTS service for earpiece audio). Those are credential-gated and report
themselves *unavailable* when not configured — the pipeline never fabricates
transcripts or audio. To exercise the pipeline logic without live providers,
these tests inject the test doubles defined below via ``app.state``.
"""

import json
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from main import app

MOCK_LLM_JSON = json.dumps({
    "suggestions": [
        "I hear what you're saying.",
        "That sounds really frustrating.",
        "Tell me more about that.",
    ],
    "tone_score": {
        "warmth": 60,
        "defensiveness": 30,
        "sarcasm": 10,
        "constructiveness": 55,
        "overall": 65,
    },
})

FAKE_TRANSCRIPT = "I just feel like you never listen to me."


# ---------------------------------------------------------------------------
# Test doubles — stand in for the real speech providers at the DI boundary.
# ---------------------------------------------------------------------------

class FakeTranscriber:
    """Available transcriber that yields a fixed transcript per chunk."""

    def __init__(self, transcript: str = FAKE_TRANSCRIPT) -> None:
        self._transcript = transcript
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def stream(self, audio_bytes: bytes) -> str | None:
        if not self._connected:
            raise RuntimeError("not connected")
        return self._transcript

    async def close(self) -> None:
        self._connected = False


class FakeTTS:
    def __init__(self, audio_b64: str | None = "ZmFrZS1hdWRpbw==") -> None:
        self._audio = audio_b64

    async def synthesize(self, text: str) -> str | None:
        return self._audio


def _clear_overrides() -> None:
    for attr in ("transcriber_factory", "tts_client", "diarizer_factory"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


@pytest.fixture
def fake_ws():
    """TestClient with an available (fake) transcriber + TTS injected."""
    _clear_overrides()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    app.state.transcriber_factory = lambda: FakeTranscriber()
    app.state.tts_client = FakeTTS()
    try:
        yield TestClient(app)
    finally:
        _clear_overrides()


@pytest.fixture
def unavailable_ws(monkeypatch):
    """TestClient with no transcriber configured — real default reports unavailable."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    _clear_overrides()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    try:
        yield TestClient(app)
    finally:
        _clear_overrides()


# ---------------------------------------------------------------------------
# Connection / disconnection
# ---------------------------------------------------------------------------

class TestWebSocketConnection:
    def test_connect_and_disconnect(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/test-session-1") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 75}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"

    def test_connect_different_sessions(self, fake_ws):
        for sid in ("session-a", "session-b"):
            with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
                ws.send_text(json.dumps({"type": "config"}))
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "config_ack"


# ---------------------------------------------------------------------------
# Utterance → suggestion flow (with an available transcriber)
# ---------------------------------------------------------------------------

class TestUtteranceSuggestionFlow:
    def test_audio_chunk_produces_suggestion(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/flow-1") as ws:
            ws.send_bytes(b"\x00\x01\x02\x03" * 100)
            resp = json.loads(ws.receive_text())

            assert resp["type"] == "suggestion"
            assert resp["session_id"] == "flow-1"
            assert len(resp["suggestions"]) == 3
            assert resp["speaker"] in ("Speaker A", "Speaker B")
            assert resp["utterance_text"] == FAKE_TRANSCRIPT
            assert resp["empathy_slider"] == 50  # default
            assert resp["audio_b64"] is not None  # injected TTS produced audio

    def test_empathy_slider_affects_suggestion(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/flow-2") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 10}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

            ws.send_bytes(b"\xff" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 10

    def test_multiple_chunks_produce_multiple_suggestions(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/flow-3") as ws:
            for i in range(3):
                ws.send_bytes(bytes([i]) * 50)
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "suggestion"
                assert resp["session_id"] == "flow-3"

    def test_llm_called_with_empathy_prompt(self, fake_ws):
        mock_llm = app.state.llm_client
        mock_llm.complete.reset_mock()

        with fake_ws.websocket_connect("/ws/session/flow-4") as ws:
            ws.send_bytes(b"\x00" * 50)
            ws.receive_text()

        assert mock_llm.complete.called
        call_kwargs = mock_llm.complete.call_args
        assert "system" in call_kwargs.kwargs
        assert "user" in call_kwargs.kwargs

    def test_tts_unavailable_yields_null_audio(self):
        """When TTS is unavailable, audio_b64 is None — not fabricated bytes."""
        _clear_overrides()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = mock_llm
        app.state.transcriber_factory = lambda: FakeTranscriber()
        app.state.tts_client = FakeTTS(audio_b64=None)  # unavailable TTS
        try:
            with TestClient(app).websocket_connect("/ws/session/flow-5") as ws:
                ws.send_bytes(b"\x00" * 50)
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "suggestion"
                assert resp["audio_b64"] is None
        finally:
            _clear_overrides()


# ---------------------------------------------------------------------------
# Honest behaviour when transcription is not configured
# ---------------------------------------------------------------------------

class TestTranscriptionUnavailable:
    def test_unavailable_event_on_connect(self, unavailable_ws):
        """With no DEEPGRAM_API_KEY, the server announces transcription is unavailable."""
        with unavailable_ws.websocket_connect("/ws/session/unavail-1") as ws:
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "transcription_unavailable"
            assert "reason" in resp

    def test_audio_does_not_fabricate_transcript(self, unavailable_ws):
        """Audio chunks must never yield a fabricated suggestion when unavailable."""
        with unavailable_ws.websocket_connect("/ws/session/unavail-2") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["type"] != "suggestion"

    def test_config_still_works_when_unavailable(self, unavailable_ws):
        with unavailable_ws.websocket_connect("/ws/session/unavail-3") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 80}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------

class TestSpeakerDiarization:
    def test_alternating_speakers(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/diar-1") as ws:
            speakers = []
            for _ in range(4):
                ws.send_bytes(b"\x00" * 50)
                resp = json.loads(ws.receive_text())
                speakers.append(resp["speaker"])
            assert speakers == ["Speaker A", "Speaker B", "Speaker A", "Speaker B"]

    def test_speaker_labels_from_config(self):
        from audio_pipeline import SpeakerDiarizer
        from models.audio import DiarizationConfig

        config = DiarizationConfig(labels=["Alice", "Bob", "Carol"], num_speakers=3)
        diarizer = SpeakerDiarizer(config)
        labels = [diarizer.assign_speaker() for _ in range(6)]
        assert labels == ["Alice", "Bob", "Carol", "Alice", "Bob", "Carol"]

    def test_diarizer_reset(self):
        from audio_pipeline import SpeakerDiarizer

        diarizer = SpeakerDiarizer()
        assert diarizer.assign_speaker() == "Speaker A"
        assert diarizer.assign_speaker() == "Speaker B"
        diarizer.reset()
        assert diarizer.assign_speaker() == "Speaker A"


# ---------------------------------------------------------------------------
# Graceful handling of bad audio chunks
# ---------------------------------------------------------------------------

class TestBadAudioHandling:
    def test_empty_audio_chunk_ignored(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/bad-1") as ws:
            ws.send_bytes(b"")  # empty
            ws.send_bytes(b"\x01\x02\x03")  # real chunk
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "suggestion"

    def test_invalid_json_text_returns_error(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/bad-2") as ws:
            ws.send_text("this is not json")
            resp = json.loads(ws.receive_text())
            assert resp.get("error") == "invalid JSON"

    def test_unknown_message_type_returns_error(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/bad-3") as ws:
            ws.send_text(json.dumps({"type": "foobar"}))
            resp = json.loads(ws.receive_text())
            assert "unknown type" in resp.get("error", "")


# ---------------------------------------------------------------------------
# Config messages
# ---------------------------------------------------------------------------

class TestConfigMessages:
    def test_config_updates_empathy_slider(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/cfg-1") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 90}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 90

    def test_config_updates_role(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/cfg-2") as ws:
            ws.send_text(json.dumps({"type": "config", "role": "Wife"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_config_ignores_invalid_slider(self, fake_ws):
        with fake_ws.websocket_connect("/ws/session/cfg-3") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 200}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 50  # unchanged


# ---------------------------------------------------------------------------
# Provider credential gating — providers report unavailable, never fabricate
# ---------------------------------------------------------------------------

class TestProviderGating:
    @pytest.mark.anyio
    async def test_deepgram_connect_without_key_unavailable(self, monkeypatch):
        from audio_pipeline import DeepgramTranscriber, TranscriberUnavailable

        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        t = DeepgramTranscriber()
        with pytest.raises(TranscriberUnavailable):
            await t.connect()

    @pytest.mark.anyio
    async def test_tts_without_key_returns_none(self, monkeypatch):
        from audio_pipeline import TTSClient

        monkeypatch.delenv("TTS_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        tts = TTSClient()
        assert await tts.synthesize("Hello world") is None
