"""Tests for the M2 real-time audio pipeline (WebSocket, transcription, diarization, TTS).

The pipeline depends on external speech providers (Deepgram for transcription,
a TTS service for earpiece audio). Those are credential-gated and report
themselves *unavailable* when not configured — the pipeline never fabricates
transcripts or audio. To exercise the pipeline logic without live providers,
these tests inject the test doubles defined below via ``app.state``.
"""

import asyncio
import contextlib
import hashlib
import json
import threading
import time
import types
import uuid
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import audio_pipeline
from audio_pipeline import TranscriberUnavailable, TranscriptSegment
from main import app

# Auth: every WS now requires a verified Firebase token in the first config
# frame (see conftest FAKE_TOKENS / _server_test_auth). open_ws performs that
# handshake and consumes the config_ack, so each test body proceeds exactly as
# it did before auth existed.
FAKE_ID_TOKEN = "fake-id-token"  # conftest maps this → uid "test-user"


@contextlib.contextmanager
def open_ws(client, path, *, token=FAKE_ID_TOKEN, headers=None):
    """Open a WS, complete the Firebase auth handshake, yield the authed socket.

    The auth ``config_ack`` is consumed here; the caller sends its own audio /
    config / stop frames just as before.
    """
    kwargs = {"headers": headers} if headers is not None else {}
    with client.websocket_connect(path, **kwargs) as ws:
        ws.send_text(json.dumps({"type": "config", "id_token": token}))
        ack = json.loads(ws.receive_text())
        assert ack["type"] == "config_ack", ack
        yield ws

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


class ClosableFakeTTS(FakeTTS):
    """FakeTTS that records whether the endpoint closed it."""

    def __init__(self, audio_b64: str | None = "ZmFrZS1hdWRpbw==") -> None:
        super().__init__(audio_b64)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class StoppableTranscriber:
    """Double with the graceful-finish contract: ``stream()`` yields the queued
    live segments once; ``finish()`` flushes the buffered final segments."""

    def __init__(
        self,
        live: list[TranscriptSegment] | None = None,
        final: list[TranscriptSegment] | None = None,
    ) -> None:
        self._live = list(live or [])
        self._final = list(final or [])
        self.finish_calls = 0
        self.closed = False

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        segments, self._live = self._live, []
        return segments

    async def finish(self) -> list[TranscriptSegment]:
        self.finish_calls += 1
        segments, self._final = self._final, []
        return segments

    async def close(self) -> None:
        self.closed = True


class RecordingSegmentTranscriber:
    """Double that records every audio frame; the queued segments come back on
    the first ``stream()`` call."""

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self._segments = list(segments)
        self.frames: list[bytes] = []

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        self.frames.append(audio_bytes)
        segments, self._segments = self._segments, []
        return segments

    async def close(self) -> None:
        pass


class DyingTranscriber:
    """Connects fine, then every stream() call reports the backend as lost."""

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes):
        raise TranscriberUnavailable("mid-stream death")

    async def close(self) -> None:
        pass


class NeverConnectsTranscriber:
    """connect() always fails — models a config-broken backend (no key, etc.)."""

    async def connect(self) -> None:
        raise TranscriberUnavailable("backend is down")

    async def close(self) -> None:
        pass


class BlockingLLM:
    """LLM double whose complete() blocks until the test releases it."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.started = threading.Event()
        self.release = threading.Event()

    def complete(self, system: str, user: str) -> str:
        self.started.set()
        assert self.release.wait(timeout=10), "test never released the LLM"
        return self._response


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
        with open_ws(fake_ws, "/ws/session/fe671ae6-ab15-55a0-a52a-a420dbb8f518") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 75}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"

    def test_connect_different_sessions(self, fake_ws):
        for sid in ("f13e554a-934c-536f-bc6e-5d24c3c8b63a", "44a700b7-7f37-533b-966f-94ee1cdad404"):
            with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
                ws.send_text(json.dumps({"type": "config"}))
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "config_ack"


# ---------------------------------------------------------------------------
# P0-1: WebSocket Origin allowlist (cross-site WS hijacking / credit theft)
# ---------------------------------------------------------------------------

class TestWebSocketOriginCheck:
    def test_no_origin_connects(self, fake_ws):
        """Native mobile clients send no Origin header — always allowed."""
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_disallowed_origin_rejected_4403(self, fake_ws, monkeypatch):
        """A browser Origin not in the allowlist is rejected before accept()."""
        monkeypatch.setattr(audio_pipeline, "ALLOWED_ORIGINS", frozenset())
        sid = str(uuid.uuid4())
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with fake_ws.websocket_connect(
                f"/ws/session/{sid}", headers={"origin": "https://evil.example"}
            ) as ws:
                ws.receive_text()
        assert excinfo.value.code == 4403

    def test_allowlisted_origin_connects(self, fake_ws, monkeypatch):
        """An Origin present in MINDSHIFT_ALLOWED_ORIGINS connects normally."""
        monkeypatch.setattr(
            audio_pipeline, "ALLOWED_ORIGINS", frozenset({"https://app.example"})
        )
        sid = str(uuid.uuid4())
        with open_ws(
            fake_ws, f"/ws/session/{sid}", headers={"origin": "https://app.example"}
        ) as ws:
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_same_origin_connects(self, fake_ws, monkeypatch):
        """A same-origin client (Origin host == server Host) is allowed even
        with an empty allowlist — this is how the React Native app arrives."""
        monkeypatch.setattr(audio_pipeline, "ALLOWED_ORIGINS", frozenset())
        sid = str(uuid.uuid4())
        # Starlette's TestClient serves under Host 'testserver'.
        with open_ws(
            fake_ws, f"/ws/session/{sid}", headers={"origin": "http://testserver"}
        ) as ws:
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"


# ---------------------------------------------------------------------------
# P2-7: WebSocket session_id must be a UUID
# ---------------------------------------------------------------------------

class TestWebSocketSessionIdValidation:
    def test_unsafe_session_id_rejected_4403(self, fake_ws):
        # A session id with characters outside [A-Za-z0-9_-] is rejected.
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with open_ws(fake_ws, "/ws/session/bad.id.with.dots") as ws:
                ws.receive_text()
        assert excinfo.value.code == 4403

    def test_overlong_session_id_rejected_4403(self, fake_ws):
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with open_ws(fake_ws, "/ws/session/" + "a" * 65) as ws:
                ws.receive_text()
        assert excinfo.value.code == 4403

    def test_app_style_session_id_connects(self, fake_ws):
        # The real mobile client sends "live-<timestamp>" — must be accepted.
        with open_ws(fake_ws, "/ws/session/live-1783392818146") as ws:
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"


# ---------------------------------------------------------------------------
# Utterance → suggestion flow (with an available transcriber)
# ---------------------------------------------------------------------------

class TestUtteranceSuggestionFlow:
    def test_audio_chunk_produces_suggestion(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/ba80d20c-e237-5290-99d8-fc64759ab9db") as ws:
            ws.send_bytes(b"\x00\x01\x02\x03" * 100)
            resp = json.loads(ws.receive_text())

            assert resp["type"] == "suggestion"
            assert resp["session_id"] == "ba80d20c-e237-5290-99d8-fc64759ab9db"
            assert len(resp["suggestions"]) == 3
            assert resp["speaker"] in ("Speaker A", "Speaker B")
            assert resp["utterance_text"] == FAKE_TRANSCRIPT
            assert resp["empathy_slider"] == 50  # default
            assert resp["audio_b64"] is not None  # injected TTS produced audio

    def test_empathy_slider_affects_suggestion(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/71c97c3e-a88e-56b9-a46b-5ad735d20295") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 10}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

            ws.send_bytes(b"\xff" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 10

    def test_multiple_chunks_produce_multiple_suggestions(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/245b20ae-77dc-588d-85c4-c199bbadbaeb") as ws:
            for i in range(3):
                ws.send_bytes(bytes([i]) * 50)
                resp = json.loads(ws.receive_text())
                assert resp["type"] == "suggestion"
                assert resp["session_id"] == "245b20ae-77dc-588d-85c4-c199bbadbaeb"

    def test_llm_called_with_empathy_prompt(self, fake_ws):
        mock_llm = app.state.llm_client
        mock_llm.complete.reset_mock()

        with open_ws(fake_ws, "/ws/session/0c7b8bd6-c8dd-5dc3-9210-c0af33cddc7b") as ws:
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
            with open_ws(TestClient(app), "/ws/session/9a2ee749-c067-5e8b-bbe0-82a094cb5d6a") as ws:
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
        with open_ws(unavailable_ws, "/ws/session/74b1265a-dd00-5431-b6ca-35c8b986e290") as ws:
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "transcription_unavailable"
            assert "reason" in resp

    def test_audio_does_not_fabricate_transcript(self, unavailable_ws):
        """Audio chunks must never yield a fabricated suggestion when unavailable.

        The unavailable notice is sent ONCE (on entering the state); binary
        frames afterwards are ignored silently — no suggestion, no re-send
        flood. The next reply on the wire is the config ack.
        """
        with open_ws(unavailable_ws, "/ws/session/d78849a1-e1be-502e-94e7-200bb5414c71") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_bytes(b"\x00" * 50)
            ws.send_bytes(b"\x00" * 50)
            ws.send_text(json.dumps({"type": "config"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"  # nothing sent for the frames

    def test_config_still_works_when_unavailable(self, unavailable_ws):
        with open_ws(unavailable_ws, "/ws/session/787db2b8-6c84-5f6b-be8b-f69a9e4f8149") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 80}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"


# ---------------------------------------------------------------------------
# Speaker diarization
# ---------------------------------------------------------------------------

class TestSpeakerDiarization:
    def test_alternating_speakers(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/cdd59ad9-9e74-5783-8d2c-150fb3182f9a") as ws:
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


class TestSpeakerLabelAssigner:
    """F6: diarized speaker ints map to stable, never-merged labels; None
    falls back sensibly depending on whether diarization has been seen."""

    def _assigner(self, config=None):
        from audio_pipeline import SpeakerDiarizer, SpeakerLabelAssigner

        return SpeakerLabelAssigner(SpeakerDiarizer(config))

    def test_ints_map_positionally_without_modulo_merging(self):
        a = self._assigner()
        assert a.label_for(0) == "Speaker A"
        assert a.label_for(1) == "Speaker B"
        # Index 2 gets its OWN generated label — never merged back into A.
        assert a.label_for(2) == "Speaker C"
        assert a.label_for(3) == "Speaker D"
        assert a.label_for(2) == "Speaker C"  # stable on repeat

    def test_none_after_diarized_speaker_continues_most_recent(self):
        a = self._assigner()
        assert a.label_for(1) == "Speaker B"
        assert a.label_for(None) == "Speaker B"  # continuation assumption
        assert a.label_for(0) == "Speaker A"
        assert a.label_for(None) == "Speaker A"

    def test_none_without_any_diarization_uses_alternation(self):
        a = self._assigner()
        assert [a.label_for(None) for _ in range(4)] == [
            "Speaker A", "Speaker B", "Speaker A", "Speaker B",
        ]

    def test_custom_labels_then_generated_overflow(self):
        from models.audio import DiarizationConfig

        config = DiarizationConfig(labels=["Alice", "Bob", "Carol"], num_speakers=3)
        a = self._assigner(config)
        assert a.label_for(0) == "Alice"
        assert a.label_for(2) == "Carol"
        assert a.label_for(3) == "Speaker D"

    def test_generated_labels_extend_past_z(self):
        from audio_pipeline import _generated_speaker_label

        assert _generated_speaker_label(2) == "Speaker C"
        assert _generated_speaker_label(25) == "Speaker Z"
        assert _generated_speaker_label(26) == "Speaker AA"


# ---------------------------------------------------------------------------
# Graceful handling of bad audio chunks
# ---------------------------------------------------------------------------

class TestBadAudioHandling:
    def test_empty_audio_chunk_ignored(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/48ea3fad-753d-5df7-aa35-00f4daf8958a") as ws:
            ws.send_bytes(b"")  # empty
            ws.send_bytes(b"\x01\x02\x03")  # real chunk
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "suggestion"

    def test_invalid_json_text_returns_error(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/89b993dd-e75f-5291-88e8-8677f21f0509") as ws:
            ws.send_text("this is not json")
            resp = json.loads(ws.receive_text())
            assert resp.get("error") == "invalid JSON"

    def test_unknown_message_type_returns_error(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/564a4fe1-642f-5ee3-a650-27f1466fc408") as ws:
            ws.send_text(json.dumps({"type": "foobar"}))
            resp = json.loads(ws.receive_text())
            assert "unknown type" in resp.get("error", "")


# ---------------------------------------------------------------------------
# Config messages
# ---------------------------------------------------------------------------

class TestConfigMessages:
    def test_config_updates_empathy_slider(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/6c29fdf8-51e6-531e-a44a-f2112a05f38a") as ws:
            ws.send_text(json.dumps({"type": "config", "empathy_slider": 90}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = json.loads(ws.receive_text())
            assert resp["empathy_slider"] == 90

    def test_config_updates_role(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/9efc8eec-7259-5354-868d-7c319ca9bd74") as ws:
            ws.send_text(json.dumps({"type": "config", "role": "Wife"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"

    def test_config_ignores_invalid_slider(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/5969eeeb-0c99-5c2f-9bb7-586a81d1342a") as ws:
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

        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        monkeypatch.delenv("TTS_API_KEY", raising=False)
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        tts = TTSClient()
        assert await tts.synthesize("Hello world") is None


# ---------------------------------------------------------------------------
# Graceful stop protocol — {"type": "stop"} → flush → session_complete
# ---------------------------------------------------------------------------

def _inject(transcriber, tts=None):
    """Install doubles on app.state; caller must _clear_overrides() after."""
    _clear_overrides()
    mock_llm = MagicMock()
    mock_llm.complete.return_value = MOCK_LLM_JSON
    app.state.llm_client = mock_llm
    app.state.transcriber_factory = lambda: transcriber
    app.state.tts_client = tts if tts is not None else FakeTTS()
    return TestClient(app)


class TestGracefulStop:
    def test_stop_flushes_final_utterances_before_session_complete(self):
        """F2: every segment drained by finish() flows through the suggestion
        pipeline and is sent BEFORE session_complete; then the server closes
        with code 1000."""
        t = StoppableTranscriber(
            live=[TranscriptSegment("Live one.", 0.0, 1.0, speaker=0)],
            final=[
                TranscriptSegment("Final one.", 2.0, 3.0, speaker=1),
                TranscriptSegment("Final two.", 3.5, 4.0, speaker=0),
            ],
        )
        client = _inject(t)
        try:
            with open_ws(client, "/ws/session/54b398cb-b43e-596c-966a-f7e17da1d6c0") as ws:
                ws.send_bytes(b"\x00" * 50)
                first = json.loads(ws.receive_text())
                ws.send_text(json.dumps({"type": "stop"}))
                second = json.loads(ws.receive_text())
                third = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    ws.receive_text()
        finally:
            _clear_overrides()

        assert first["utterance_text"] == "Live one."
        assert [second["type"], third["type"]] == ["suggestion", "suggestion"]
        assert [second["utterance_text"], third["utterance_text"]] == [
            "Final one.", "Final two.",
        ]
        assert done == {"type": "session_complete"}
        assert excinfo.value.code == 1000
        assert t.finish_calls >= 1

    def test_stop_with_no_prior_audio_still_completes(self):
        t = StoppableTranscriber(
            final=[TranscriptSegment("Only final.", 0.0, 1.0, speaker=0)],
        )
        client = _inject(t)
        try:
            with open_ws(client, "/ws/session/74542016-975a-5488-a0a4-7a75117e82b1") as ws:
                ws.send_text(json.dumps({"type": "stop"}))
                suggestion = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert suggestion["type"] == "suggestion"
        assert suggestion["utterance_text"] == "Only final."
        assert done == {"type": "session_complete"}

    def test_stop_when_transcription_unavailable_completes_cleanly(self, unavailable_ws):
        """stop must work even when the (real, unconnected) transcriber never
        came up — finish() on it is a safe no-op."""
        with open_ws(unavailable_ws, "/ws/session/e56c87ae-89a0-526a-8256-5ef31eb110ea") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_text(json.dumps({"type": "stop"}))
            assert json.loads(ws.receive_text()) == {"type": "session_complete"}


# ---------------------------------------------------------------------------
# Suggestion worker — LLM/TTS latency must not stall the audio receive loop
# ---------------------------------------------------------------------------

class TestSuggestionWorker:
    def test_llm_latency_does_not_stall_audio_receive_loop(self):
        """F4: while a suggestion is being generated (LLM blocked), further
        audio frames are still consumed and forwarded to the transcriber."""
        llm = BlockingLLM(MOCK_LLM_JSON)
        t = RecordingSegmentTranscriber(
            [TranscriptSegment("Blocks the LLM.", 0.0, 1.0, speaker=0)],
        )
        client = _inject(t)
        app.state.llm_client = llm  # replace the MagicMock with the blocker
        try:
            with open_ws(client, "/ws/session/60f4a89a-537d-5139-a70b-cb9eaec86327") as ws:
                ws.send_bytes(b"\x01" * 50)
                assert llm.started.wait(timeout=5)  # worker is now inside the LLM
                ws.send_bytes(b"\x02" * 50)
                ws.send_bytes(b"\x03" * 50)
                deadline = time.monotonic() + 5
                while len(t.frames) < 3 and time.monotonic() < deadline:
                    time.sleep(0.01)
                frames_while_llm_blocked = len(t.frames)
                llm.release.set()
                resp = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        # All three frames reached the transcriber BEFORE the LLM completed.
        assert frames_while_llm_blocked == 3
        assert resp["type"] == "suggestion"
        assert resp["utterance_text"] == "Blocks the LLM."

    def test_suggestion_events_preserve_segment_order(self):
        """A slow first suggestion must not let later (fast) ones overtake it."""
        calls: list[str] = []

        def slow_first(system: str, user: str) -> str:
            calls.append(user)
            if len(calls) == 1:
                time.sleep(0.2)
            return MOCK_LLM_JSON

        t = RecordingSegmentTranscriber([
            TranscriptSegment("First.", 0.0, 1.0, speaker=0),
            TranscriptSegment("Second.", 1.0, 2.0, speaker=1),
            TranscriptSegment("Third.", 2.0, 3.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client.complete.side_effect = slow_first
        try:
            with open_ws(client, "/ws/session/9762d144-4d01-5060-8f11-2d7dba4a761f") as ws:
                ws.send_bytes(b"\x00" * 50)
                events = [json.loads(ws.receive_text()) for _ in range(3)]
        finally:
            _clear_overrides()

        assert [e["utterance_text"] for e in events] == ["First.", "Second.", "Third."]


# ---------------------------------------------------------------------------
# Unavailable notice is sent once, not per frame (F7)
# ---------------------------------------------------------------------------

class TestUnavailableNoticeOnce:
    def test_midstream_failure_notice_sent_once(self, monkeypatch):
        """After a mid-stream failure (and exhausted reconnects — P1-1) the
        client is told once; further binary frames are ignored silently (no
        per-frame re-send flood).

        Re-pinned for P1-1: a previously-connected transcriber that drops now
        triggers reconnect attempts first, so the factory here serves dead
        replacements — the unavailable latch happens only after they exhaust.
        """
        monkeypatch.setattr(
            audio_pipeline, "TRANSCRIBER_RECONNECT_BACKOFFS_S", (0.0, 0.0, 0.0)
        )
        factory_calls: list[int] = []

        def factory():
            factory_calls.append(1)
            # First transcriber connects then dies; every replacement is dead.
            if len(factory_calls) == 1:
                return DyingTranscriber()
            return NeverConnectsTranscriber()

        client = _inject(DyingTranscriber())
        app.state.transcriber_factory = factory
        try:
            with open_ws(client, "/ws/session/2512291a-796d-5f21-b1db-75a6261e6aa6") as ws:
                ws.send_bytes(b"\x00" * 50)
                ws.send_bytes(b"\x00" * 50)
                ws.send_bytes(b"\x00" * 50)
                ws.send_text(json.dumps({"type": "config"}))
                first = json.loads(ws.receive_text())
                second = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert first["type"] == "transcription_unavailable"
        # Exactly one notice — the very next message is already the config ack.
        assert second["type"] == "config_ack"
        # Initial connect + exactly 3 reconnect attempts, then the latch.
        assert len(factory_calls) == 4


# ---------------------------------------------------------------------------
# TTS client ownership (F9) — injected/shared instances are never closed
# ---------------------------------------------------------------------------

class TestTTSOwnership:
    def test_injected_tts_client_is_not_closed_by_endpoint(self):
        tts = ClosableFakeTTS()
        t = StoppableTranscriber(
            live=[TranscriptSegment("Hello.", 0.0, 1.0, speaker=0)],
        )
        client = _inject(t, tts=tts)
        try:
            with open_ws(client, "/ws/session/838378a4-1e3c-50b3-9f2e-71c3ad374969") as ws:
                ws.send_bytes(b"\x00" * 50)
                assert json.loads(ws.receive_text())["type"] == "suggestion"
        finally:
            _clear_overrides()

        assert tts.closed is False  # shared instance must survive the session


# ---------------------------------------------------------------------------
# Suggestion failures are reported, never silent, never fabricated (P0-2, P0-3)
# ---------------------------------------------------------------------------

class TestSuggestionErrorHonesty:
    def test_llm_exception_sends_suggestion_error(self):
        """P0-2: an LLM failure must produce a suggestion_error event — the
        client is told WHICH utterance yielded nothing and why (class name
        only; the raw message could carry key fragments)."""
        client = _inject(FakeTranscriber())
        app.state.llm_client.complete.side_effect = RuntimeError(
            "401 invalid x-api-key sk-ant-SECRET"
        )
        try:
            with open_ws(client, "/ws/session/a2358e57-1418-5997-8e8d-7026163bc9f5") as ws:
                ws.send_bytes(b"\x00" * 50)
                raw = ws.receive_text()
                resp = json.loads(raw)
                # Session survives the failure — control channel still works.
                ws.send_text(json.dumps({"type": "config"}))
                ack = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion_error"
        assert resp["utterance_text"] == FAKE_TRANSCRIPT
        assert resp["reason"] == "RuntimeError"
        assert "SECRET" not in raw  # exception message never hits the wire
        assert ack["type"] == "config_ack"

    def test_unparseable_llm_output_is_error_not_fabrication(self):
        """P0-3: unparseable LLM output must NOT become an 'I hear you — …'
        fake suggestion (which would even get TTS-spoken); the client gets an
        honest suggestion_error with reason llm_parse_error."""
        client = _inject(FakeTranscriber())
        app.state.llm_client.complete.return_value = "Sorry, no JSON today."
        try:
            with open_ws(client, "/ws/session/537c414a-a7df-570c-a344-5203b036cc62") as ws:
                ws.send_bytes(b"\x00" * 50)
                resp = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion_error"
        assert resp["reason"] == "llm_parse_error"
        assert "suggestions" not in resp
        assert "I hear you" not in json.dumps(resp)

    def test_wrong_shape_json_is_parse_error(self):
        """Valid JSON that isn't the expected object shape is also honest."""
        client = _inject(FakeTranscriber())
        app.state.llm_client.complete.return_value = json.dumps(["a", "list"])
        try:
            with open_ws(client, "/ws/session/38705c3f-22f9-5ab1-93e5-23097150ea63") as ws:
                ws.send_bytes(b"\x00" * 50)
                resp = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion_error"
        assert resp["reason"] == "llm_parse_error"


# ---------------------------------------------------------------------------
# Mid-session transcriber reconnect (P1-1)
# ---------------------------------------------------------------------------

class TestTranscriberReconnect:
    def test_midsession_drop_reconnects_and_restores(self, monkeypatch):
        """A previously-live transcriber that drops is replaced via the
        factory; the client hears transcription_restored and transcription
        continues on the replacement."""
        monkeypatch.setattr(
            audio_pipeline, "TRANSCRIBER_RECONNECT_BACKOFFS_S", (0.0, 0.0, 0.0)
        )
        factory_calls: list[int] = []
        healthy = RecordingSegmentTranscriber(
            [TranscriptSegment("After the blip.", 0.0, 1.0, speaker=0)],
        )

        def factory():
            factory_calls.append(1)
            return DyingTranscriber() if len(factory_calls) == 1 else healthy

        client = _inject(DyingTranscriber())
        app.state.transcriber_factory = factory
        try:
            with open_ws(client, "/ws/session/a56b5f93-1eab-5de5-bd6a-934d454ca97d") as ws:
                ws.send_bytes(b"\x00" * 50)  # dies → reconnect → restored
                restored = json.loads(ws.receive_text())
                ws.send_bytes(b"\x01" * 50)  # flows to the replacement
                suggestion = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert restored == {"type": "transcription_restored"}
        assert suggestion["type"] == "suggestion"
        assert suggestion["utterance_text"] == "After the blip."
        assert len(factory_calls) == 2  # initial + one successful reconnect
        assert healthy.frames  # audio really reached the replacement

    def test_initial_connect_failure_is_not_retried(self, monkeypatch):
        """A transcriber that NEVER connected failed for a config reason (no
        key, missing package) — reconnecting cannot fix that, so the endpoint
        must not spin the factory."""
        monkeypatch.setattr(
            audio_pipeline, "TRANSCRIBER_RECONNECT_BACKOFFS_S", (0.0, 0.0, 0.0)
        )
        factory_calls: list[int] = []

        def factory():
            factory_calls.append(1)
            return NeverConnectsTranscriber()

        client = _inject(NeverConnectsTranscriber())
        app.state.transcriber_factory = factory
        try:
            with open_ws(client, "/ws/session/af6be8a7-3f67-5301-94e1-8bfb6bd959df") as ws:
                first = json.loads(ws.receive_text())
                ws.send_bytes(b"\x00" * 50)  # ignored — latched unavailable
                ws.send_text(json.dumps({"type": "config"}))
                second = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert first["type"] == "transcription_unavailable"
        assert second["type"] == "config_ack"
        assert len(factory_calls) == 1  # no retries for a config failure


# ---------------------------------------------------------------------------
# Worker task lifecycle on immediate disconnect (P1-7)
# ---------------------------------------------------------------------------

class _GoneClientWS:
    """Minimal WebSocket stand-in: accepts, then behaves like a client that
    disconnected immediately — every send fails (as Starlette's does after a
    disconnect) and receive() reports the disconnect."""

    def __init__(self, state) -> None:
        self.app = types.SimpleNamespace(state=state)
        # No Origin header (native-client shape) — the P0-1 check reads this.
        self.headers: dict = {}

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        raise RuntimeError('Cannot call "send" once a close message has been sent.')

    async def receive(self) -> dict:
        return {"type": "websocket.disconnect"}

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        pass


class TestWorkerTaskLifecycle:
    @pytest.mark.anyio
    async def test_immediate_disconnect_does_not_leak_worker_task(self):
        """P1-7: when the client is gone before the initial unavailable-notify
        goes out, the endpoint must still tear down its suggestion worker —
        previously the notify raised before the protecting try, leaking one
        pending task per occurrence (and propagating the send error)."""
        from audio_pipeline import audio_ws_endpoint

        state = types.SimpleNamespace(
            llm_client=MagicMock(),
            transcriber_factory=lambda: NeverConnectsTranscriber(),
            tts_client=FakeTTS(),
        )
        before = asyncio.all_tasks()
        # Must return cleanly (no send error escaping) …
        await audio_ws_endpoint(_GoneClientWS(state), "21e2655a-1523-51e2-836b-b9ecfa8ceaec")
        # … and leave no pending background task behind.
        leaked = [t for t in asyncio.all_tasks() - before if not t.done()]
        assert leaked == []


# ---------------------------------------------------------------------------
# Graceful stop drain is bounded (P1-8)
# ---------------------------------------------------------------------------

class TestStopDrainTimeout:
    def test_hung_llm_does_not_stall_stop(self, monkeypatch):
        """A hung LLM call must not hold the client's stop hostage: after the
        drain timeout the server closes out with an honest pending_dropped
        count instead of a bare session_complete."""
        monkeypatch.setattr(audio_pipeline, "STOP_DRAIN_TIMEOUT_S", 0.2)
        llm = BlockingLLM(MOCK_LLM_JSON)
        t = StoppableTranscriber(
            live=[TranscriptSegment("Never finishes.", 0.0, 1.0, speaker=0)],
        )
        client = _inject(t)
        app.state.llm_client = llm
        try:
            with open_ws(client, "/ws/session/80d8585e-c7a7-52ec-951f-96b7d1a718ad") as ws:
                ws.send_bytes(b"\x00" * 50)
                assert llm.started.wait(timeout=5)  # worker is inside the LLM
                ws.send_text(json.dumps({"type": "stop"}))
                done = json.loads(ws.receive_text())
                llm.release.set()  # unblock the worker thread for teardown
        finally:
            _clear_overrides()

        assert done == {"type": "session_complete", "pending_dropped": 1}

    def test_fast_drain_keeps_bare_session_complete(self):
        """No timeout → the pre-existing exact payload is preserved."""
        t = StoppableTranscriber(
            final=[TranscriptSegment("Quick.", 0.0, 1.0, speaker=0)],
        )
        client = _inject(t)
        try:
            with open_ws(client, "/ws/session/06057ec0-a9eb-55ad-a1fe-ed066a0b3397") as ws:
                ws.send_text(json.dumps({"type": "stop"}))
                suggestion = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert suggestion["type"] == "suggestion"
        assert done == {"type": "session_complete"}  # no pending_dropped key


# ---------------------------------------------------------------------------
# In-memory utterance buffer is bounded (P1-9) + PII-safe logging (P1-4)
# ---------------------------------------------------------------------------

class TestMemoryAndLogging:
    def test_utterance_buffer_is_capped(self):
        from audio_pipeline import SessionContext, _remember_utterance
        from models.audio import Utterance

        ctx = SessionContext(session_id="cap-buf")
        for i in range(audio_pipeline.UTTERANCE_BUFFER_MAX + 1):
            _remember_utterance(ctx, Utterance(
                session_id="cap-buf", speaker="Speaker A", text=f"utterance {i}",
                start_time=float(i), end_time=float(i) + 0.5,
            ))

        assert len(ctx.utterances) == audio_pipeline.UTTERANCE_BUFFER_KEEP
        # The newest entries are the ones retained.
        assert ctx.utterances[-1].text == (
            f"utterance {audio_pipeline.UTTERANCE_BUFFER_MAX}"
        )

    def test_redact_never_contains_the_text(self):
        from audio_pipeline import _redact

        secret = "I told my therapist something deeply private"
        out = _redact(secret)
        for word in secret.split():
            assert word not in out
        # exact length is bucketed (not advertised) to avoid narrowing short phrases
        assert f"len={len(secret)}" not in out
        assert out == _redact(secret)  # stable digest → log lines correlate
        # salted HMAC, not a bare sha256 of the text (dictionary-attack resistant)
        assert hashlib.sha256(secret.encode()).hexdigest()[:12] not in out


# ---------------------------------------------------------------------------
# Session + utterance caps (P2-1)
# ---------------------------------------------------------------------------

class TestSessionCaps:
    def test_session_cap_rejects_with_1013(self, fake_ws, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "_session_slots", asyncio.Semaphore(1))
        with open_ws(fake_ws, "/ws/session/2ba12c1f-d5da-559d-b21c-9e9a5dd99cb2") as ws1:
            ws1.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws1.receive_text())["type"] == "config_ack"
            # Second concurrent session: honest 1013 "try again later". The cap
            # rejects it right after accept — before the auth handshake — so it
            # stays a raw connect (open_ws would try to auth on a closing socket).
            with fake_ws.websocket_connect(
                "/ws/session/d8e3cd62-25b7-54b9-a94c-5180a2086f45"
            ) as ws2:
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    ws2.receive_text()
                assert excinfo.value.code == 1013
        # The slot frees when the first session ends — new sessions connect.
        with open_ws(fake_ws, "/ws/session/26439e23-62b6-5469-b904-b985496c0654") as ws3:
            ws3.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws3.receive_text())["type"] == "config_ack"

    def test_utterance_cap_sends_limit_reached_once(self, fake_ws, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "MAX_UTTERANCES", 2)
        with open_ws(fake_ws, "/ws/session/206ef3fb-6502-54d8-b7c1-555711c5f449") as ws:
            # Send/receive in lockstep: limit_reached goes out from the receive
            # loop while suggestions come from the worker, so firing all frames
            # at once could interleave the two streams.
            for i in range(2):  # FakeTranscriber yields one utterance per frame
                ws.send_bytes(bytes([i]) * 50)
                assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_bytes(b"\x02" * 50)  # over budget → notified once
            assert json.loads(ws.receive_text())["type"] == "limit_reached"
            ws.send_bytes(b"\x03" * 50)  # still over budget → silence
            ws.send_text(json.dumps({"type": "config"}))
            # No second limit_reached, no suggestion — next event is the ack.
            assert json.loads(ws.receive_text())["type"] == "config_ack"


# ---------------------------------------------------------------------------
# WS input validation (P2-3)
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_oversized_audio_frame_rejected(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/d0dfee8f-90a3-538b-86c9-ce2c196e1f08") as ws:
            ws.send_bytes(b"\x00" * (audio_pipeline.MAX_AUDIO_FRAME_BYTES + 1))
            resp = json.loads(ws.receive_text())
            assert "audio frame too large" in resp["error"]
            # A contract-sized frame afterwards still flows normally.
            ws.send_bytes(b"\x00" * 3200)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

    def test_role_is_clamped_to_100_chars(self, fake_ws):
        long_role = "R" * 300
        with open_ws(fake_ws, "/ws/session/7356ce8d-30af-5360-a2d8-f2e8bb114856") as ws:
            ws.send_text(json.dumps({"type": "config", "role": long_role}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert "R" * 100 in system
        assert "R" * 101 not in system

    def test_non_string_role_is_ignored(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/b0692156-a25f-5471-b63b-794314ae2f9c") as ws:
            ws.send_text(json.dumps({"type": "config", "role": ["not", "a", "str"]}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert "Husband" in system  # default role survived the bad config


# ---------------------------------------------------------------------------
# Voice profile over the WebSocket (net-new relationship/participant plumbing)
# ---------------------------------------------------------------------------

def _ensure_schema() -> None:
    """Create the DB schema in the shared temp DB (order-independent)."""
    from main import init_db
    asyncio.run(init_db())


class TestVoiceProfileWS:
    def _seed_profile(self, client) -> str:
        """Create a relationship + participant and PUT a voice profile; return
        the relationship id. Uses the same app/DB the WS session reads."""
        # The fake_ws TestClient is created without running lifespan, so ensure
        # the schema exists regardless of test ordering (this shared temp DB may
        # be untouched when the audio tests run first).
        _ensure_schema()
        rel = client.post("/relationships", json={
            "type": "couple",
            "name": "WS Marriage",
            "participants": [
                {"id": "alex", "role": "husband", "display_name": "Alex"},
                {"id": "jordan", "role": "wife", "display_name": "Jordan"},
            ],
        })
        assert rel.status_code == 201
        rel_id = rel.json()["id"]
        put = client.put(
            f"/relationships/{rel_id}/participants/alex/voice-profile",
            json={
                "pairs": [{
                    "suggestion": "I understand you're frustrated.",
                    "rephrase": "Okay — I get it, let's just figure it out.",
                }],
                "style_notes": "short, dry",
            },
        )
        assert put.status_code == 200
        return rel_id

    def test_ws_config_loads_and_applies_profile(self, fake_ws):
        rel_id = self._seed_profile(fake_ws)
        app.state.llm_client.complete.reset_mock()
        with open_ws(
            fake_ws, "/ws/session/2b8c1e4a-0000-4000-8000-000000000001"
        ) as ws:
            ws.send_text(json.dumps({
                "type": "config",
                "relationship_id": rel_id,
                "from_participant_id": "alex",
            }))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert "Okay — I get it, let's just figure it out." in system
        assert "Style notes: short, dry" in system

    def test_ws_without_profile_prompt_unchanged(self, fake_ws):
        """No relationship/participant in config → today's exact prompt."""
        from main import empathy_system_prompt

        app.state.llm_client.complete.reset_mock()
        with open_ws(
            fake_ws, "/ws/session/2b8c1e4a-0000-4000-8000-000000000002"
        ) as ws:
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert system == empathy_system_prompt(50, "Husband")

    def test_ws_unknown_profile_falls_back_cleanly(self, fake_ws):
        """A relationship/participant with no stored profile → no block, no error."""
        from main import empathy_system_prompt

        _ensure_schema()
        rel = fake_ws.post("/relationships", json={
            "type": "couple",
            "name": "No Profile",
            "participants": [
                {"id": "alex", "role": "husband", "display_name": "Alex"},
                {"id": "jordan", "role": "wife", "display_name": "Jordan"},
            ],
        })
        rel_id = rel.json()["id"]
        app.state.llm_client.complete.reset_mock()
        with open_ws(
            fake_ws, "/ws/session/2b8c1e4a-0000-4000-8000-000000000003"
        ) as ws:
            ws.send_text(json.dumps({
                "type": "config",
                "relationship_id": rel_id,
                "from_participant_id": "alex",
            }))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert system == empathy_system_prompt(50, "Husband")


# ---------------------------------------------------------------------------
# WebSocket Firebase auth — token required in the first config frame
# ---------------------------------------------------------------------------
# The WS handshake can't carry an Authorization header, so the first frame must
# be a config carrying a valid id_token (conftest maps "fake-id-token" and
# "tok-user-a"/"tok-user-b" to uids via the fake verify_id_token). A missing/
# invalid token — or a stored session owned by another user — is closed 4401
# before any provider work.

def _expect_4401(ws) -> None:
    """The server sends an auth_error notice, then closes the WS with 4401."""
    first = json.loads(ws.receive_text())
    assert first["type"] == "auth_error", first
    with pytest.raises(WebSocketDisconnect) as excinfo:
        ws.receive_text()
    assert excinfo.value.code == 4401


class TestWebSocketAuth:
    def test_missing_id_token_rejected_4401(self, fake_ws):
        sid = str(uuid.uuid4())
        with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config"}))  # no id_token
            _expect_4401(ws)

    def test_invalid_id_token_rejected_4401(self, fake_ws):
        sid = str(uuid.uuid4())
        with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config", "id_token": "bogus"}))
            _expect_4401(ws)

    def test_audio_before_auth_rejected_4401(self, fake_ws):
        """Binary audio before authenticating is refused — no transcript work
        happens for an unauthenticated client."""
        sid = str(uuid.uuid4())
        with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
            ws.send_bytes(b"\x00" * 50)
            _expect_4401(ws)

    def test_non_config_first_frame_rejected_4401(self, fake_ws):
        sid = str(uuid.uuid4())
        with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "stop"}))  # not a config
            _expect_4401(ws)

    def test_valid_id_token_authenticates_and_flows(self, fake_ws):
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"

    def test_cannot_open_ws_on_another_users_session(self, fake_ws):
        """A stored session owned by user-a cannot be opened by another user —
        the live audio pipeline is never attached to a foreign session."""
        created = fake_ws.post(
            "/session", json={"turns": [], "metadata": {}},
            headers={"X-Test-Uid": "user-a"},
        )
        sid = created.json()["id"]
        # "fake-id-token" → uid "test-user" ≠ owner "user-a" → 4401.
        with fake_ws.websocket_connect(f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config", "id_token": "fake-id-token"}))
            _expect_4401(ws)
        # The real owner (tok-user-a → "user-a") opens it fine.
        with open_ws(fake_ws, f"/ws/session/{sid}", token="tok-user-a") as ws:
            ws.send_bytes(b"\x00" * 50)
            assert json.loads(ws.receive_text())["type"] == "suggestion"
