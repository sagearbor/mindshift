"""Tests for the M2 real-time audio pipeline (WebSocket, Deepgram stub, TTS, diarization)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from main import app, init_db

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


@pytest.fixture
def ws_client():
    """Synchronous TestClient for WebSocket testing."""
    # Attach a mock LLM client to app state
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    return TestClient(app)


# ---------------------------------------------------------------------------
# Connection / disconnection
# ---------------------------------------------------------------------------

class TestWebSocketConnection:
    def test_connect_and_disconnect(self, ws_client):
        """Client can connect to the WebSocket and cleanly disconnect."""
        with ws_client.websocket_connect("/ws/session/test-session-1") as ws:
            # Connection is open — send a config message to verify
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 75}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"
        # Disconnect is implicit when context manager exits

    def test_connect_different_sessions(self, ws_client):
        """Multiple sessions can be addressed by different IDs."""
        with ws_client.websocket_connect("/ws/session/session-a") as ws:
            ws.send_text(json.dumps({"type": "config"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"

        with ws_client.websocket_connect("/ws/session/session-b") as ws:
            ws.send_text(json.dumps({"type": "config"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"


# ---------------------------------------------------------------------------
# Utterance → suggestion flow
# ---------------------------------------------------------------------------

class TestUtteranceSuggestionFlow:
    def test_audio_chunk_produces_suggestion(self, ws_client):
        """Sending a binary audio chunk should yield a SuggestionEvent JSON."""
        with ws_client.websocket_connect("/ws/session/flow-1") as ws:
            ws.send_bytes(b"\x00\x01\x02\x03" * 100)
            resp = json.loads(ws.receive_text())

            assert resp["type"] == "suggestion"
            assert resp["session_id"] == "flow-1"
            assert len(resp["suggestions"]) == 3
            assert resp["speaker"] in ("Speaker A", "Speaker B")
            assert "utterance_text" in resp
            assert resp["empathy_slider"] == 50  # default
            assert resp["audio_b64"] is not None  # TTS stub returns something

    def test_empathy_slider_affects_suggestion(self, ws_client):
        """Changing empathy_slider via config should be reflected in subsequent events."""
        with ws_client.websocket_connect("/ws/session/flow-2") as ws:
            # Set slider to assertive
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 10}))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "config_ack"

            # Send audio
            ws.send_bytes(b"\xff" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 10

    def test_multiple_chunks_produce_multiple_suggestions(self, ws_client):
        """Each audio chunk should produce its own suggestion event."""
        with ws_client.websocket_connect("/ws/session/flow-3") as ws:
            for i in range(3):
                ws.send_bytes(bytes([i]) * 50)
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "suggestion"
                assert resp["session_id"] == "flow-3"

    def test_llm_called_with_empathy_prompt(self, ws_client):
        """LLM.complete() should be called with empathy system prompt."""
        mock_llm = app.state.llm_client
        mock_llm.complete.reset_mock()

        with ws_client.websocket_connect("/ws/session/flow-4") as ws:
            ws.send_bytes(b"\x00" * 50)
            ws.receive_text()

        assert mock_llm.complete.called
        call_kwargs = mock_llm.complete.call_args
        assert "system" in call_kwargs.kwargs
        assert "user" in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------

class TestSpeakerDiarization:
    def test_alternating_speakers(self, ws_client):
        """Speakers should alternate between Speaker A and Speaker B."""
        with ws_client.websocket_connect("/ws/session/diar-1") as ws:
            speakers = []
            for _ in range(4):
                ws.send_bytes(b"\x00" * 50)
                resp = json.loads(ws.receive_text())
                speakers.append(resp["speaker"])

            assert speakers == ["Speaker A", "Speaker B", "Speaker A", "Speaker B"]

    def test_speaker_labels_from_config(self):
        """SpeakerDiarizer should use labels from DiarizationConfig."""
        from audio_pipeline import SpeakerDiarizer
        from models.audio import DiarizationConfig

        config = DiarizationConfig(labels=["Alice", "Bob", "Carol"], num_speakers=3)
        diarizer = SpeakerDiarizer(config)

        labels = [diarizer.assign_speaker() for _ in range(6)]
        assert labels == ["Alice", "Bob", "Carol", "Alice", "Bob", "Carol"]

    def test_diarizer_reset(self):
        """Reset should restart speaker assignment."""
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
    def test_empty_audio_chunk_ignored(self, ws_client):
        """Empty binary frames should be silently ignored (no suggestion)."""
        with ws_client.websocket_connect("/ws/session/bad-1") as ws:
            ws.send_bytes(b"")  # empty
            # Send a real chunk to verify connection still works
            ws.send_bytes(b"\x01\x02\x03")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "suggestion"

    def test_invalid_json_text_returns_error(self, ws_client):
        """Non-JSON text messages should return an error, not crash."""
        with ws_client.websocket_connect("/ws/session/bad-2") as ws:
            ws.send_text("this is not json")
            resp = json.loads(ws.receive_text())
            assert "error" in resp
            assert resp["error"] == "invalid JSON"

    def test_unknown_message_type_returns_error(self, ws_client):
        """Unknown message types should return an error."""
        with ws_client.websocket_connect("/ws/session/bad-3") as ws:
            ws.send_text(json.dumps({"type": "foobar"}))
            resp = json.loads(ws.receive_text())
            assert "error" in resp
            assert "unknown type" in resp["error"]


# ---------------------------------------------------------------------------
# Config messages
# ---------------------------------------------------------------------------

class TestConfigMessages:
    def test_config_updates_empathy_slider(self, ws_client):
        with ws_client.websocket_connect("/ws/session/cfg-1") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 90}))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "config_ack"

            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 90

    def test_config_updates_role(self, ws_client):
        with ws_client.websocket_connect("/ws/session/cfg-2") as ws:
            ws.send_text(json.dumps({"type": "config", "role": "Wife"}))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "config_ack"

    def test_config_ignores_invalid_slider(self, ws_client):
        """Out-of-range empathy_slider values should be ignored."""
        with ws_client.websocket_connect("/ws/session/cfg-3") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 200}))
            ack = json.loads(ws.receive_text())
            assert ack["type"] == "config_ack"

            # Default should still be 50
            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 50


# ---------------------------------------------------------------------------
# Deepgram transcriber stub unit tests
# ---------------------------------------------------------------------------

class TestDeepgramTranscriber:
    @pytest.mark.anyio
    async def test_connect_and_close(self):
        from audio_pipeline import DeepgramTranscriber
        t = DeepgramTranscriber()
        assert not t.is_connected
        await t.connect()
        assert t.is_connected
        await t.close()
        assert not t.is_connected

    @pytest.mark.anyio
    async def test_stream_returns_transcription(self):
        from audio_pipeline import DeepgramTranscriber
        t = DeepgramTranscriber()
        await t.connect()
        result = await t.stream(b"\x00" * 100)
        assert result is not None
        assert "Mock transcription" in result
        await t.close()

    @pytest.mark.anyio
    async def test_stream_without_connect_raises(self):
        from audio_pipeline import DeepgramTranscriber
        t = DeepgramTranscriber()
        with pytest.raises(RuntimeError, match="not connected"):
            await t.stream(b"\x00")


# ---------------------------------------------------------------------------
# TTS stub unit tests
# ---------------------------------------------------------------------------

class TestTTSClient:
    @pytest.mark.anyio
    async def test_synthesize_returns_base64(self):
        from audio_pipeline import TTSClient
        tts = TTSClient()
        result = await tts.synthesize("Hello world")
        assert isinstance(result, str)
        # Should be valid base64
        import base64
        decoded = base64.b64decode(result)
        assert b"Hello world" in decoded
