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
import logging
import threading
import time
import types
import uuid
from unittest.mock import MagicMock

import numpy as np
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


def recv_skipping_transcripts(ws):
    """Receive the next non-transcript event.

    Every finalized utterance now emits an immediate ``transcript`` event
    ahead of its (optional) suggestion; tests asserting on the coaching flow
    skip those. Transcript events themselves are covered by dedicated tests.
    """
    while True:
        msg = json.loads(ws.receive_text())
        if msg.get("type") != "transcript":
            return msg

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


class SequentialSegmentTranscriber:
    """Double that yields ONE queued segment per ``stream()`` call, in order.

    Unlike :class:`RecordingSegmentTranscriber` (which returns every queued
    segment on the first call — one frame carrying several finalized
    utterances), this models one utterance finalizing per audio frame, each
    with distinct text — so a test can tell which suggestion belongs to
    which frame (the suggestion carries ``utterance_text``).
    """

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self._segments = list(segments)

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        if not self._segments:
            return []
        return [self._segments.pop(0)]

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
    for attr in (
        "transcriber_factory", "tts_client", "diarizer_factory",
        # Track 3-server injection points (fake clock, fake voiceprint store).
        "monotonic_clock", "recordings_store",
    ):
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
            resp = recv_skipping_transcripts(ws)

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
            resp = recv_skipping_transcripts(ws)
            assert resp["empathy_slider"] == 10

    def test_multiple_chunks_produce_multiple_suggestions(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/245b20ae-77dc-588d-85c4-c199bbadbaeb") as ws:
            for i in range(3):
                ws.send_bytes(bytes([i]) * 50)
                resp = recv_skipping_transcripts(ws)
                assert resp["type"] == "suggestion"
                assert resp["session_id"] == "245b20ae-77dc-588d-85c4-c199bbadbaeb"

    def test_llm_called_with_empathy_prompt(self, fake_ws):
        mock_llm = app.state.llm_client
        mock_llm.complete.reset_mock()

        with open_ws(fake_ws, "/ws/session/0c7b8bd6-c8dd-5dc3-9210-c0af33cddc7b") as ws:
            ws.send_bytes(b"\x00" * 50)
            recv_skipping_transcripts(ws)

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
                resp = recv_skipping_transcripts(ws)
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
                resp = recv_skipping_transcripts(ws)
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
            resp = recv_skipping_transcripts(ws)
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
            resp = recv_skipping_transcripts(ws)
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
            resp = recv_skipping_transcripts(ws)
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
        with code 1000.

        Every finalized utterance now emits its ``transcript`` event
        immediately, ahead of the (async, LLM+TTS-backed) ``suggestion`` for
        it — this test is explicitly about event ordering, so it asserts the
        full real sequence rather than skipping transcripts. Reading the
        "Live one." suggestion before sending "stop" (as before) keeps the
        worker idle when the two final segments are enqueued, so neither is
        dropped by the latest-wins supersede policy (see enqueue_segments).
        """
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
                live_transcript = json.loads(ws.receive_text())
                first = json.loads(ws.receive_text())
                ws.send_text(json.dumps({"type": "stop"}))
                final_transcript_1 = json.loads(ws.receive_text())
                final_transcript_2 = json.loads(ws.receive_text())
                second = json.loads(ws.receive_text())
                third = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    ws.receive_text()
        finally:
            _clear_overrides()

        assert live_transcript["type"] == "transcript"
        assert live_transcript["text"] == "Live one."
        assert first["type"] == "suggestion"
        assert first["utterance_text"] == "Live one."
        assert [final_transcript_1["type"], final_transcript_2["type"]] == [
            "transcript", "transcript",
        ]
        assert [final_transcript_1["text"], final_transcript_2["text"]] == [
            "Final one.", "Final two.",
        ]
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
                transcript = json.loads(ws.receive_text())
                suggestion = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert transcript["type"] == "transcript"
        assert transcript["text"] == "Only final."
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
                resp = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        # All three frames reached the transcriber BEFORE the LLM completed.
        assert frames_while_llm_blocked == 3
        assert resp["type"] == "suggestion"
        assert resp["utterance_text"] == "Blocks the LLM."

    def test_suggestion_events_preserve_segment_order(self):
        """A slow first suggestion must not let later (fast) ones overtake it.

        All three segments finalize on the SAME ``stream()`` call, so they are
        enqueued back-to-back inside one ``enqueue_segments`` invocation. The
        worker picks up "First." (and starts its slow 200ms LLM call) before
        "Third." is enqueued, so "Second." is still only QUEUED — not yet
        started — when "Third." lands, and the latest-wins policy (see
        enqueue_segments) supersedes it. This is the new, intended behavior,
        not a bug: what matters is that the ones which DO arrive are never
        reordered — "Third." never overtakes the still-generating "First.".
        """
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
                # 3 transcript events (one per segment) + 2 suggestion events
                # ("Second." is superseded before the worker starts it).
                events = [json.loads(ws.receive_text()) for _ in range(5)]
        finally:
            _clear_overrides()

        transcripts = [e for e in events if e["type"] == "transcript"]
        suggestions = [e for e in events if e["type"] == "suggestion"]
        assert [t["text"] for t in transcripts] == ["First.", "Second.", "Third."]
        assert [s["utterance_text"] for s in suggestions] == ["First.", "Third."]


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
                assert recv_skipping_transcripts(ws)["type"] == "suggestion"
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
                transcript_raw = ws.receive_text()
                raw = ws.receive_text()
                resp = json.loads(raw)
                # Session survives the failure — control channel still works.
                ws.send_text(json.dumps({"type": "config"}))
                ack = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert json.loads(transcript_raw)["type"] == "transcript"
        assert resp["type"] == "suggestion_error"
        assert resp["utterance_text"] == FAKE_TRANSCRIPT
        assert resp["reason"] == "RuntimeError"
        assert "SECRET" not in raw  # exception message never hits the wire
        assert "SECRET" not in transcript_raw
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
                resp = recv_skipping_transcripts(ws)
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
                resp = recv_skipping_transcripts(ws)
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
                suggestion = recv_skipping_transcripts(ws)
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

    def test_segments_after_reconnect_are_on_the_session_timeline(self, monkeypatch):
        """Review 2026-08-24: Deepgram stamps `start` relative to ITS
        connection, so a replacement connection's clock restarts at 0 while
        the session (ring buffer, turn_local ranges, the client's transcript)
        keeps counting. A segment the replacement reports at [0, 1] after
        1.1 s of audio had already flowed must surface as [1.1, 2.1]."""
        monkeypatch.setattr(
            audio_pipeline, "TRANSCRIBER_RECONNECT_BACKOFFS_S", (0.0, 0.0, 0.0)
        )

        class DiesAfter:
            """Accepts ``n`` frames silently, then dies like DyingTranscriber."""

            def __init__(self, n: int) -> None:
                self.n = n

            async def connect(self) -> None:
                pass

            async def stream(self, audio_bytes: bytes):
                if self.n > 0:
                    self.n -= 1
                    return []
                raise TranscriberUnavailable("mid-stream death")

            async def close(self) -> None:
                pass

        factory_calls: list[int] = []
        healthy = RecordingSegmentTranscriber(
            [TranscriptSegment("After the blip.", 0.0, 1.0, speaker=0)],
        )

        def factory():
            factory_calls.append(1)
            return DiesAfter(10) if len(factory_calls) == 1 else healthy

        client = _inject(DiesAfter(10))
        app.state.transcriber_factory = factory
        try:
            with open_ws(client, "/ws/session/a56b5f93-1eab-5de5-bd6a-934d454ca97d") as ws:
                for _ in range(10):
                    ws.send_bytes(FRAME_100MS)   # 1.0 s flows to the first connection
                ws.send_bytes(FRAME_100MS)       # 1.1 s: hits the dead socket → reconnect
                assert json.loads(ws.receive_text()) == {"type": "transcription_restored"}
                ws.send_bytes(FRAME_100MS)       # first frame the replacement sees
                transcript = json.loads(ws.receive_text())
                suggestion = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        assert transcript["type"] == "transcript"
        assert transcript["text"] == "After the blip."
        assert transcript["start_time"] == pytest.approx(1.1)
        assert transcript["end_time"] == pytest.approx(2.1)
        assert suggestion["utterance_text"] == "After the blip."


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
                # The transcript event fires immediately on finalize (before
                # the LLM is even called), well before the hang — read it now
                # so it isn't mistaken for session_complete below.
                transcript = json.loads(ws.receive_text())
                ws.send_text(json.dumps({"type": "stop"}))
                done = json.loads(ws.receive_text())
                llm.release.set()  # unblock the worker thread for teardown
        finally:
            _clear_overrides()

        assert transcript["type"] == "transcript"
        assert transcript["text"] == "Never finishes."
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
                # transcript, then suggestion, then session_complete.
                transcript = json.loads(ws.receive_text())
                suggestion = json.loads(ws.receive_text())
                done = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert transcript["type"] == "transcript"
        assert transcript["text"] == "Quick."
        assert suggestion["type"] == "suggestion"
        assert suggestion["utterance_text"] == "Quick."
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
        """The transcript event is sent unconditionally, before the budget
        check — the limit check (see enqueue_segments) only gates the
        suggestion. So even over-budget frames still emit a transcript;
        the difference is no suggestion (and, once, a limit_reached)."""
        monkeypatch.setattr(audio_pipeline, "MAX_UTTERANCES", 2)
        with open_ws(fake_ws, "/ws/session/206ef3fb-6502-54d8-b7c1-555711c5f449") as ws:
            # Send/receive in lockstep: limit_reached goes out from the receive
            # loop while suggestions come from the worker, so firing all frames
            # at once could interleave the two streams.
            for i in range(2):  # FakeTranscriber yields one utterance per frame
                ws.send_bytes(bytes([i]) * 50)
                assert json.loads(ws.receive_text())["type"] == "transcript"
                assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_bytes(b"\x02" * 50)  # over budget → transcript, then notified once
            assert json.loads(ws.receive_text())["type"] == "transcript"
            assert json.loads(ws.receive_text())["type"] == "limit_reached"
            ws.send_bytes(b"\x03" * 50)  # still over budget → transcript, then silence
            assert json.loads(ws.receive_text())["type"] == "transcript"
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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

    def test_role_is_clamped_to_100_chars(self, fake_ws):
        long_role = "R" * 300
        with open_ws(fake_ws, "/ws/session/7356ce8d-30af-5360-a2d8-f2e8bb114856") as ws:
            ws.send_text(json.dumps({"type": "config", "role": long_role}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

        system = app.state.llm_client.complete.call_args.kwargs["system"]
        assert "R" * 100 in system
        assert "R" * 101 not in system

    def test_non_string_role_is_ignored(self, fake_ws):
        with open_ws(fake_ws, "/ws/session/b0692156-a25f-5471-b63b-794314ae2f9c") as ws:
            ws.send_text(json.dumps({"type": "config", "role": ["not", "a", "str"]}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"

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
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"


# ---------------------------------------------------------------------------
# NEW: transcript event precedes the suggestion for every finalized utterance
# ---------------------------------------------------------------------------

class TestTranscriptEvents:
    def test_audio_frame_yields_transcript_before_suggestion(self, fake_ws):
        """Every finalized utterance now emits an immediate TranscriptEvent —
        session_id/speaker/text/timing — strictly BEFORE its suggestion."""
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_bytes(b"\x00" * 50)
            transcript = json.loads(ws.receive_text())
            suggestion = json.loads(ws.receive_text())

        assert transcript["type"] == "transcript"
        assert transcript["session_id"] == sid
        assert transcript["text"] == FAKE_TRANSCRIPT
        assert transcript["speaker"] in ("Speaker A", "Speaker B")
        assert transcript["start_time"] == 0.0
        assert transcript["end_time"] == 0.0  # FakeTranscriber returns a bare str

        assert suggestion["type"] == "suggestion"
        assert suggestion["utterance_text"] == FAKE_TRANSCRIPT


# ---------------------------------------------------------------------------
# NEW: interjection gating — importance/speak/audio_b64 (session interject_level)
# ---------------------------------------------------------------------------

class TestInterjectGating:
    """SuggestionEvent now carries the LLM-scored ``importance`` (0-100) and
    ``speak`` (importance >= the session's interject_level). TTS is only
    synthesized when ``speak`` is True — FakeTTS always returns audio when
    invoked, so a ``None`` audio_b64 here proves TTS was genuinely skipped,
    not just that the fake happened to return nothing."""

    def test_below_threshold_suppresses_voice_but_keeps_suggestion(self, fake_ws):
        app.state.llm_client.complete.return_value = json.dumps({
            "suggestions": ["Quiet aside."],
            "importance": 20,
        })
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config", "interject_level": 50}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)

        assert resp["type"] == "suggestion"  # still delivered, just not voiced
        assert resp["importance"] == 20
        assert resp["speak"] is False
        assert resp["audio_b64"] is None

    def test_at_or_above_threshold_voices_and_synthesizes(self, fake_ws):
        app.state.llm_client.complete.return_value = json.dumps({
            "suggestions": ["Urgent nudge."],
            "importance": 80,
        })
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config", "interject_level": 50}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)

        assert resp["type"] == "suggestion"
        assert resp["importance"] == 80
        assert resp["speak"] is True
        assert resp["audio_b64"] is not None

    def test_missing_importance_fails_open_to_100_and_speaks(self, fake_ws):
        """No 'importance' key in the LLM JSON → fail open to 100 (always
        voice), preserving pre-slider behaviour for an older/non-conforming
        model response — even with a high interject_level configured."""
        app.state.llm_client.complete.return_value = json.dumps({
            "suggestions": ["No importance key at all."],
        })
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_text(json.dumps({"type": "config", "interject_level": 90}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)

        assert resp["importance"] == 100
        assert resp["speak"] is True
        assert resp["audio_b64"] is not None


# ---------------------------------------------------------------------------
# NEW: interject_level config parsing (_apply_config), like other config tests
# ---------------------------------------------------------------------------

class TestInterjectLevelConfigParsing:
    @pytest.mark.anyio
    async def test_interject_level_applied_within_range(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="interject-ok")
        await _apply_config(ctx, {"interject_level": 65})
        assert ctx.interject_level == 65

    @pytest.mark.anyio
    async def test_interject_level_out_of_range_is_ignored(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="interject-oob")
        await _apply_config(ctx, {"interject_level": 150})
        assert ctx.interject_level == 0  # default unchanged

        await _apply_config(ctx, {"interject_level": -1})
        assert ctx.interject_level == 0

    @pytest.mark.anyio
    async def test_interject_level_wrong_type_is_ignored(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="interject-badtype")
        await _apply_config(ctx, {"interject_level": "50"})
        assert ctx.interject_level == 0


# ---------------------------------------------------------------------------
# NEW: latest-wins queue — a pending (not-yet-started) turn is superseded by
# a newer one; the dropped turn still gets its transcript + is remembered.
# ---------------------------------------------------------------------------

class TestLatestWinsQueue:
    def test_pending_turn_is_superseded_by_newer_one(self):
        """While turn one is being generated (LLM blocked), turns two and
        three both arrive. Turn two is still only QUEUED — never started —
        when turn three lands, so enqueue_segments's latest-wins policy drops
        it. All three still get their transcript event; only turn one (in
        flight) and turn three (the newest) get suggestions."""
        llm = BlockingLLM(MOCK_LLM_JSON)
        t = SequentialSegmentTranscriber([
            TranscriptSegment("Turn one.", 0.0, 1.0, speaker=0),
            TranscriptSegment("Turn two.", 1.0, 2.0, speaker=1),
            TranscriptSegment("Turn three.", 2.0, 3.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client = llm
        try:
            with open_ws(client, "/ws/session/e3f0a6ac-6f0a-4a53-9a1a-9c5b6a4e2b8a") as ws:
                ws.send_bytes(b"\x00" * 50)  # turn one — worker picks it up
                assert llm.started.wait(timeout=5)  # worker is inside the LLM
                ws.send_bytes(b"\x01" * 50)  # turn two — queues, not started
                ws.send_bytes(b"\x02" * 50)  # turn three — supersedes turn two
                transcripts = [json.loads(ws.receive_text()) for _ in range(3)]
                llm.release.set()  # unblock turn one's suggestion
                suggestions = [recv_skipping_transcripts(ws) for _ in range(2)]
        finally:
            _clear_overrides()

        assert [tr["text"] for tr in transcripts] == [
            "Turn one.", "Turn two.", "Turn three.",
        ]
        assert [s["utterance_text"] for s in suggestions] == [
            "Turn one.", "Turn three.",
        ]


# ---------------------------------------------------------------------------
# NEW: side-aware coaching — the coach knows who the user is
# ---------------------------------------------------------------------------
# self_speaker (the coached user's diarized label) turns their OWN turns into a
# single delivery "nudge" (kind="nudge") instead of the 3-suggestion
# "response". None (unset) = legacy behaviour: every turn is a "response". A
# nudge LLM payload differs from MOCK_LLM_JSON — the contract is exactly
# {"nudge": str, "importance": int} (no tone_score: nothing reads it, and dead
# output is wasted tokens on every self turn of a real-time whisper).

NUDGE_LLM_JSON = json.dumps({
    "nudge": "ease up",
    "importance": 70,
})

# A self turn that needs no correction: the model returns an empty (here
# whitespace-only, to prove it is .strip()ed) nudge → the coach stays silent.
EMPTY_NUDGE_LLM_JSON = json.dumps({
    "nudge": "   ",
    "importance": 0,
})


class TestSelfSpeakerConfigParsing:
    """self_speaker parses like the other validated config fields: a valid
    "Speaker X" string is stored, malformed shapes are ignored, JSON null
    resets, and an absent key leaves the current value untouched."""

    @pytest.mark.anyio
    async def test_valid_self_speaker_stored(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="self-ok")
        await _apply_config(ctx, {"self_speaker": "Speaker A"})
        assert ctx.self_speaker == "Speaker A"
        # Two-letter generated labels (past Z) are valid too.
        await _apply_config(ctx, {"self_speaker": "Speaker AB"})
        assert ctx.self_speaker == "Speaker AB"

    @pytest.mark.anyio
    async def test_invalid_self_speaker_shapes_ignored(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="self-bad")
        ctx.self_speaker = "Speaker A"  # a previously-set value must survive
        for bad in ("bob", 42, "Speaker", "speaker a", "Speaker 1", ["Speaker A"]):
            await _apply_config(ctx, {"self_speaker": bad})
            assert ctx.self_speaker == "Speaker A"

    @pytest.mark.anyio
    async def test_null_resets_self_speaker(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="self-null")
        ctx.self_speaker = "Speaker B"
        await _apply_config(ctx, {"self_speaker": None})
        assert ctx.self_speaker is None

    @pytest.mark.anyio
    async def test_absent_key_leaves_self_speaker_unchanged(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="self-absent")
        ctx.self_speaker = "Speaker A"
        await _apply_config(ctx, {"empathy_slider": 30})  # unrelated field
        assert ctx.self_speaker == "Speaker A"


class TestSideAwareCoaching:
    """End-to-end: with self_speaker set, the coached user's own turns get a
    'nudge'; the other person's turns are unchanged 'response's.

    speaker=0 maps to "Speaker A" and speaker=1 to "Speaker B" via the
    SpeakerLabelAssigner, so a SequentialSegmentTranscriber can emit one SELF
    and one OTHER turn deterministically."""

    def test_other_turn_unchanged_with_self_speaker_set(self):
        # self is Speaker B; a Speaker A (speaker=0) turn is the OTHER person →
        # today's exact 3-suggestion response, kind="response".
        t = SequentialSegmentTranscriber([
            TranscriptSegment("You never listen.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)  # mock LLM returns MOCK_LLM_JSON (3 suggestions)
        try:
            with open_ws(client, "/ws/session/5f0a1b2c-0000-4000-8000-000000000101") as ws:
                ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker B"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                ws.send_bytes(b"\x00" * 50)
                resp = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion"
        assert resp["kind"] == "response"
        assert len(resp["suggestions"]) == 3

    def test_self_turn_nudge_voiced_above_threshold(self):
        # self is Speaker A; a Speaker A (speaker=0) turn is a SELF turn →
        # kind="nudge", one nudge string. importance 70 >= interject 50 → voiced.
        t = SequentialSegmentTranscriber([
            TranscriptSegment("I'm sorry, maybe I'm wrong.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client.complete.return_value = NUDGE_LLM_JSON
        try:
            with open_ws(client, "/ws/session/5f0a1b2c-0000-4000-8000-000000000102") as ws:
                ws.send_text(json.dumps({
                    "type": "config", "self_speaker": "Speaker A",
                    "interject_level": 50,
                }))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                ws.send_bytes(b"\x00" * 50)
                resp = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion"
        assert resp["kind"] == "nudge"
        assert resp["suggestions"] == ["ease up"]
        assert resp["importance"] == 70
        assert resp["speak"] is True
        assert resp["audio_b64"] is not None  # voiced → TTS synthesized

    def test_self_turn_nudge_not_voiced_below_threshold(self):
        # Same nudge (importance 70) but interject_level 90 → not voiced, no TTS,
        # yet the nudge event is still delivered (client may render it dimmed).
        t = SequentialSegmentTranscriber([
            TranscriptSegment("I'm sorry, maybe I'm wrong.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client.complete.return_value = NUDGE_LLM_JSON
        try:
            with open_ws(client, "/ws/session/5f0a1b2c-0000-4000-8000-000000000103") as ws:
                ws.send_text(json.dumps({
                    "type": "config", "self_speaker": "Speaker A",
                    "interject_level": 90,
                }))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                ws.send_bytes(b"\x00" * 50)
                resp = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion"
        assert resp["kind"] == "nudge"
        assert resp["speak"] is False
        assert resp["audio_b64"] is None

    def test_empty_nudge_sends_no_suggestion_event(self):
        # A self turn that needs no correction → empty nudge → NOTHING sent
        # beyond the transcript. Mirror TestSessionCaps: prove absence by
        # showing the next event on the wire is the config_ack.
        t = SequentialSegmentTranscriber([
            TranscriptSegment("Sounds good, thanks.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client.complete.return_value = EMPTY_NUDGE_LLM_JSON
        try:
            with open_ws(client, "/ws/session/5f0a1b2c-0000-4000-8000-000000000104") as ws:
                ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker A"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                ws.send_bytes(b"\x00" * 50)
                transcript = json.loads(ws.receive_text())
                assert transcript["type"] == "transcript"
                assert transcript["text"] == "Sounds good, thanks."
                # No suggestion event for the silent nudge — the next event is
                # the ack for this config frame.
                ws.send_text(json.dumps({"type": "config"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
        finally:
            _clear_overrides()

    def test_legacy_no_self_speaker_all_response_kind(self, fake_ws):
        # self_speaker unset → every turn is a "response" (kind present + correct
        # for old clients that never send self_speaker).
        sid = str(uuid.uuid4())
        with open_ws(fake_ws, f"/ws/session/{sid}") as ws:
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)
        assert resp["type"] == "suggestion"
        assert resp["kind"] == "response"

    def test_self_speaker_change_does_not_retype_queued_turn(self):
        """Snapshot semantics: a turn enqueued while self_speaker was None is
        typed OTHER even if self_speaker is set before the worker finishes it.

        Proof exploits that MOCK_LLM_JSON has 'suggestions' but no 'nudge':
        typed OTHER, turn one yields a kind='response' suggestion; a (buggy)
        SELF retype would read an empty nudge from the same payload and send
        NOTHING. So seeing the response proves the enqueue-time snapshot held.
        """
        llm = BlockingLLM(MOCK_LLM_JSON)
        t = SequentialSegmentTranscriber([
            TranscriptSegment("First turn.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client = llm
        try:
            with open_ws(client, "/ws/session/5f0a1b2c-0000-4000-8000-000000000105") as ws:
                ws.send_bytes(b"\x00" * 50)  # turn one — self_speaker None → OTHER
                assert llm.started.wait(timeout=5)  # worker is inside the LLM
                transcript = json.loads(ws.receive_text())  # immediate transcript
                assert transcript["type"] == "transcript"
                # Flip self_speaker AFTER the turn was snapshotted at enqueue.
                ws.send_text(json.dumps({
                    "type": "config", "self_speaker": "Speaker A",
                }))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                llm.release.set()
                resp = recv_skipping_transcripts(ws)
        finally:
            _clear_overrides()

        assert resp["type"] == "suggestion"
        assert resp["kind"] == "response"  # OTHER — the queued turn was not retyped
        assert len(resp["suggestions"]) == 3


class TestSelfFeedbackPrompt:
    """The nudge prompt follows the empathy dial (owner forbade always-soften)
    and preserves the byte-identical-when-None voice-profile property."""

    def test_direction_follows_empathy_dial(self):
        from main import self_feedback_prompt

        assertive = self_feedback_prompt(10, "Husband").lower()
        assert "assertive" in assertive or "firm" in assertive
        assert "never tell" in assertive  # explicitly forbids soften at low dial

        empathetic = self_feedback_prompt(90, "Husband").lower()
        assert "warm" in empathetic or "validat" in empathetic

    def test_voice_profile_none_is_byte_identical(self):
        from main import self_feedback_prompt

        assert self_feedback_prompt(50, "Wife", None) == self_feedback_prompt(50, "Wife")

    def test_voice_profile_block_appended(self):
        from main import self_feedback_prompt

        profile = {"pairs": [], "style_notes": "short, dry"}
        out = self_feedback_prompt(50, "Wife", profile)
        assert "Style notes: short, dry" in out


class TestGenerateNudgeUnit:
    """Unit coverage for _generate_nudge's fail-open / empty-nudge contract."""

    @pytest.mark.anyio
    async def test_empty_nudge_returns_zero_importance(self):
        from audio_pipeline import _generate_nudge
        from models.audio import Utterance

        llm = MagicMock()
        llm.complete.return_value = EMPTY_NUDGE_LLM_JSON  # whitespace nudge
        u = Utterance(
            session_id="s", speaker="Speaker A", text="hi",
            start_time=0.0, end_time=1.0,
        )
        nudge, importance = await _generate_nudge(llm, u, 50, "Husband")
        assert nudge == ""
        assert importance == 0

    @pytest.mark.anyio
    async def test_missing_importance_fails_open_only_with_nudge(self):
        from audio_pipeline import _generate_nudge
        from models.audio import Utterance

        llm = MagicMock()
        llm.complete.return_value = json.dumps({"nudge": "be firmer"})  # no importance
        u = Utterance(
            session_id="s", speaker="Speaker A", text="hi",
            start_time=0.0, end_time=1.0,
        )
        nudge, importance = await _generate_nudge(llm, u, 10, "Husband")
        assert nudge == "be firmer"
        assert importance == 100  # fail open — a real nudge is worth voicing

    @pytest.mark.anyio
    async def test_unparseable_output_raises_suggestion_unavailable(self):
        from audio_pipeline import SuggestionUnavailable, _generate_nudge
        from models.audio import Utterance

        llm = MagicMock()
        llm.complete.return_value = "not json at all"
        u = Utterance(
            session_id="s", speaker="Speaker A", text="hi",
            start_time=0.0, end_time=1.0,
        )
        with pytest.raises(SuggestionUnavailable) as excinfo:
            await _generate_nudge(llm, u, 50, "Husband")
        assert excinfo.value.reason == "llm_parse_error"


# ---------------------------------------------------------------------------
# Track 3-server: latency instrumentation, local-first (turn_local) path,
# PCM ring buffer + enrichment, streaming cloud suggestions
# ---------------------------------------------------------------------------
# The phone (Track 3-mobile) becomes the orchestrator: it still streams PCM,
# and ALSO sends one ``turn_local`` per turn it finalized itself. For such a
# session the server must not duplicate the phone's work (no transcript echo,
# no Deepgram duplicate, no server TTS) and enriches asynchronously (cloud
# suggestion, audio tone, identity, watch relay). Legacy clients never send
# turn_local and get exactly the behaviour every test above pins.

LOCAL_SID = "5f0a1b2c-0000-4000-8000-0000000003a1"
FRAME_100MS = b"\x01\x00" * 1600  # 1600 int16 samples = 100 ms at 16 kHz = 3200 bytes


def _turn_local(sid: str = LOCAL_SID, **overrides) -> dict:
    """A well-formed turn_local payload; overrides replace/add fields."""
    payload = {
        "type": "turn_local",
        "session_id": sid,
        "speaker": "Speaker A",
        "text": "You never listen to me.",
        "start_time": 0.0,
        "end_time": 1.0,
        "transcript_source": "on-device",
    }
    payload.update(overrides)
    return payload


def recv_until(ws, predicate, limit: int = 12) -> tuple[dict, list[dict]]:
    """Receive events until one satisfies ``predicate``; return it and every
    event seen. Enrichment events and the cloud suggestion come from
    independent tasks, so their relative order on the wire is not fixed."""
    seen: list[dict] = []
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        seen.append(msg)
        if predicate(msg):
            return msg, seen
    raise AssertionError(f"no event matched within {limit} events: {seen}")


class SteppingClock:
    """Fake monotonic clock: every call advances by ``step`` seconds."""

    def __init__(self, step: float = 0.5) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class StreamingLLM:
    """LLM double with a REAL ``stream_complete`` (defined on the class, so
    ``_supports_streaming`` sees it) that yields the response in chunks."""

    def __init__(self, response: str, chunk: int = 7) -> None:
        self._response = response
        self._chunk = chunk
        self.complete_calls: list[str] = []
        self.stream_calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.complete_calls.append(user)
        return self._response

    def stream_complete(self, system: str, user: str):
        self.stream_calls.append(user)
        for i in range(0, len(self._response), self._chunk):
            yield self._response[i:i + self._chunk]


class FakeToneId:
    """Stand-in for server/tone_id.py at the module-attribute seam."""

    MIN_TURN_SECONDS = 1.0
    MAX_TURN_SECONDS = 30.0

    class ToneUnavailable(RuntimeError):
        pass

    def __init__(self, result=None, surface=True, error=None) -> None:
        self._result = result or {
            "label": "angry", "scores": {"neutral": 0.1, "angry": 0.8, "happy": 0.05, "sad": 0.05},
            "confidence": 0.8, "model": "fake",
        }
        self._surface = surface
        self._error = error
        self.calls: list[tuple[int, int]] = []

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def surface_allowed(self) -> bool:
        return self._surface

    def classify_pcm(self, pcm, sr):
        self.calls.append((int(pcm.size), sr))
        if self._error is not None:
            raise self._error
        return dict(self._result)


class FakeSpeakerId:
    """Stand-in for server/speaker_id.py's Foundation B surface: the slice
    always embeds to [1, 0], so cosine against each enrolled print is that
    print's first component — the test picks the prints to decide who wins.
    Returns the documented identify_speakers_multi report shape."""

    MATCH_THRESHOLD = 0.5
    MIN_MATCH_SECONDS = 1.0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def identify_speakers_multi(self, pcm, sr, turns, voiceprints, *, threshold=None,
                                people=None):
        self.calls.append({
            "samples": int(pcm.size), "sr": sr, "turns": turns,
            "people": sorted(voiceprints),
        })
        emb = np.array([1.0, 0.0], dtype=np.float32)
        speaker = turns[0]["speaker"]
        scores = {pid: round(float(np.dot(emb, vec)), 4) for pid, vec in voiceprints.items()}
        best = max(scores, key=scores.get)
        matched = best if scores[best] >= self.MATCH_THRESHOLD else None
        meta = (people or {}).get(matched or "", {})
        return {
            "matched": {speaker: matched} if matched else {},
            "speakers": {speaker: {
                "scores": scores,
                "matched_person_id": matched,
                "is_self": bool(meta.get("is_self")) if matched else False,
                "display_name": meta.get("display_name") if matched else None,
            }},
        }


SELF_DOC = {"person_id": "self", "display_name": "You", "is_self": True, "embedding": [1.0, 0.0]}
ALEX_DOC = {"person_id": "alex", "display_name": "Alex", "is_self": False, "embedding": [0.0, 1.0]}


class FakeVoiceprintStore:
    """Per-person voiceprint docs, as recordings_store.list_voiceprints
    returns them (person views: person_id / display_name / is_self)."""

    def __init__(self, docs=None, error=None) -> None:
        self._docs = list(docs or [])
        self._error = error
        self.reads: list[str] = []

    async def list_voiceprints(self, uid: str):
        self.reads.append(uid)
        if self._error is not None:
            raise self._error
        return list(self._docs)


class FakeRelay:
    """Track 1's relay surface: push_turn_local(uid, event, *, tone_flag=None)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def push_turn_local(self, uid: str, event, *, tone_flag=None) -> None:
        self.calls.append({"uid": uid, "event": event, "tone_flag": tone_flag})


@pytest.fixture
def local_first_env(monkeypatch):
    """Isolate the local-first tests from whatever optional modules this
    checkout has: no real tone/speaker models, no relay, no slice grace wait
    (every test streams its audio BEFORE the turn_local, so the ring buffer
    is already caught up). Tests install fakes on top as needed."""
    monkeypatch.setattr(audio_pipeline, "tone_id", None)
    monkeypatch.setattr(audio_pipeline, "speaker_id", None)
    monkeypatch.setattr(audio_pipeline, "watch_relay", None)
    monkeypatch.setattr(audio_pipeline, "SLICE_GRACE_S", 0.0)
    yield
    _clear_overrides()


# --- Latency instrumentation ----------------------------------------------

class TestLatencyInstrumentation:
    def test_stage_ms_exact_arithmetic(self):
        from audio_pipeline import UtteranceTiming

        t = UtteranceTiming(
            frame_received=10.0, segment_finalized=10.1, enqueued=10.15,
            llm_start=10.2, llm_first_partial=10.6, llm_end=11.2,
            tts_start=11.25, tts_end=11.75, sent=11.8,
        )
        assert t.stage_ms() == {
            "seg_to_enqueue": 50.0,
            "queue_wait": 50.0,
            "llm": 1000.0,
            "llm_first_partial": 400.0,
            "tts": 500.0,
            "total": 1800.0,
        }

    def test_unreached_stages_are_absent_not_zero(self):
        from audio_pipeline import UtteranceTiming

        # A local-first turn: no TTS, no partial — those stages must be
        # MISSING, never reported as 0 ms.
        t = UtteranceTiming(
            frame_received=1.0, segment_finalized=1.0, enqueued=1.0,
            llm_start=1.0, llm_end=2.0, sent=2.0,
        )
        stages = t.stage_ms()
        assert "tts" not in stages and "llm_first_partial" not in stages
        assert stages["llm"] == 1000.0

    def test_recorder_with_fake_clock_logs_and_summarizes(self, caplog):
        from audio_pipeline import LatencyRecorder

        clock = SteppingClock(step=0.1)
        rec = LatencyRecorder(clock=clock, window=3)
        with caplog.at_level(logging.INFO, logger="audio_pipeline"):
            for _ in range(5):
                timing = rec.start(frame_received=rec.now(), segment_finalized=rec.now())
                timing.enqueued = rec.now()
                timing.llm_start = rec.now()
                timing.llm_end = rec.now()
                timing.sent = rec.now()
                stages = rec.record(timing, "sess-1")
        # Every step is exactly one clock tick (100 ms) — fake-clock exactness.
        assert stages == {
            "seg_to_enqueue": 100.0, "queue_wait": 100.0, "llm": 100.0,
            "total": 500.0,
        }
        # window=3 keeps only the last 3 samples per stage.
        summary = rec.summary()
        assert summary["llm"] == {"p50": 100.0, "p95": 100.0, "n": 3}
        assert "tts" not in summary  # never stamped → omitted
        lines = [r.getMessage() for r in caplog.records if "latency session=" in r.getMessage()]
        assert len(lines) == 5
        assert lines[-1].startswith("latency session=sess-1 seg_to_enqueue=100.0ms ")
        assert "llm=100.0ms" in lines[-1]
        assert "tts=-" in lines[-1]  # unreached stage rendered as "-"
        assert "queue_depth=0" in lines[-1]

    def test_nearest_rank_percentiles(self):
        from audio_pipeline import _nearest_rank

        ordered = [10.0, 20.0, 30.0, 40.0, 100.0]
        assert _nearest_rank(ordered, 50) == 30.0
        assert _nearest_rank(ordered, 95) == 100.0
        assert _nearest_rank([7.0], 95) == 7.0

    def test_report_latency_config_adds_summary_to_session_complete(self, fake_ws, caplog):
        """A legacy-protocol client can opt in with config report_latency;
        the fake clock makes every stage a known, positive number and the
        per-utterance INFO line is emitted through the real hot path."""
        app.state.monotonic_clock = SteppingClock(step=0.25)
        try:
            with caplog.at_level(logging.INFO, logger="audio_pipeline"):
                with open_ws(fake_ws, f"/ws/session/{LOCAL_SID}") as ws:
                    ws.send_text(json.dumps({"type": "config", "report_latency": True}))
                    assert json.loads(ws.receive_text())["type"] == "config_ack"
                    ws.send_bytes(b"\x00" * 50)
                    assert recv_skipping_transcripts(ws)["type"] == "suggestion"
                    ws.send_text(json.dumps({"type": "stop"}))
                    done = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert done["type"] == "session_complete"
        summary = done["latency_summary"]
        for stage in ("seg_to_enqueue", "queue_wait", "llm", "tts", "total"):
            assert summary[stage]["n"] == 1
            assert summary[stage]["p50"] == summary[stage]["p95"] > 0
        assert any("latency session=" in r.getMessage() for r in caplog.records)

    def test_legacy_stop_payload_has_no_summary(self, fake_ws):
        """Without report_latency (and without turn_local) the completion
        payload stays exactly the pre-existing bare dict."""
        with open_ws(fake_ws, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_bytes(b"\x00" * 50)
            assert recv_skipping_transcripts(ws)["type"] == "suggestion"
            ws.send_text(json.dumps({"type": "stop"}))
            assert json.loads(ws.receive_text()) == {"type": "session_complete"}


# --- turn_local → cloud suggestion ------------------------------------------

class TestTurnLocal:
    def test_turn_local_yields_cloud_suggestion_without_transcript_echo(
        self, local_first_env,
    ):
        client = _inject(StoppableTranscriber())
        app.state.llm_client.complete.reset_mock()
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local(
                text_tone={"warmth": 20, "defensiveness": 80, "label": "defensive"},
                prosody={"rms_dbfs": -18.0, "pitch_hz": 210.0, "speech_rate": None},
            )))
            resp = json.loads(ws.receive_text())

        # The very first event is the suggestion — no transcript echo.
        assert resp["type"] == "suggestion"
        assert resp["suggestion_source"] == "cloud"
        assert resp["partial"] is False
        assert resp["utterance_text"] == "You never listen to me."
        assert resp["speaker"] == "Speaker A"
        assert len(resp["suggestions"]) == 3
        # The phone's tone context reached the LLM prompt, as hints.
        user = app.state.llm_client.complete.call_args.kwargs["user"]
        assert user.startswith('Transcript turn: "You never listen to me."')
        assert "On-device signals for this turn" in user
        assert "defensiveness 80/100" in user and 'label "defensive"' in user
        assert "median pitch 210.0 Hz" in user and "loudness -18.0 dBFS" in user
        assert "speech rate" not in user  # null measurement never rendered

    def test_is_self_true_routes_to_nudge_even_without_self_speaker(self, local_first_env):
        client = _inject(StoppableTranscriber())
        app.state.llm_client.complete.return_value = NUDGE_LLM_JSON
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local(is_self=True)))
            resp = json.loads(ws.receive_text())
        assert resp["type"] == "suggestion"
        assert resp["kind"] == "nudge"
        assert resp["suggestions"] == ["ease up"]
        # The nudge prompt gets the tone context too when present.
        assert "Transcript turn" in app.state.llm_client.complete.call_args.kwargs["user"]

    def test_is_self_false_wins_over_matching_self_speaker(self, local_first_env):
        """The phone's voiceprint verdict beats the label compare: speaker
        label matches self_speaker, but is_self=false → OTHER → response."""
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker A"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_text(json.dumps(_turn_local(speaker="Speaker A", is_self=False)))
            resp = json.loads(ws.receive_text())
        assert resp["kind"] == "response"
        assert len(resp["suggestions"]) == 3

    def test_is_self_null_falls_back_to_label_compare(self, local_first_env):
        client = _inject(StoppableTranscriber())
        app.state.llm_client.complete.return_value = NUDGE_LLM_JSON
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps({"type": "config", "self_speaker": "Speaker A"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_text(json.dumps(_turn_local(speaker="Speaker A")))  # is_self absent
            resp = json.loads(ws.receive_text())
        assert resp["kind"] == "nudge"

    def test_invalid_turn_local_is_rejected_without_leaking_values(self, local_first_env):
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            bad = _turn_local(text="SECRET WORDS", transcript_source="magic")
            del bad["start_time"]
            ws.send_text(json.dumps(bad))
            resp = json.loads(ws.receive_text())
            # Session survives: control channel still works.
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
        assert resp["error"].startswith("invalid turn_local: ")
        assert "start_time" in resp["error"] and "transcript_source" in resp["error"]
        assert "SECRET" not in resp["error"] and "magic" not in resp["error"]

    def test_session_id_mismatch_and_inverted_times_rejected(self, local_first_env):
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local(session_id="someone-else")))
            assert json.loads(ws.receive_text())["error"] == "turn_local session_id mismatch"
            ws.send_text(json.dumps(_turn_local(start_time=2.0, end_time=1.0)))
            assert "end_time" in json.loads(ws.receive_text())["error"]

    def test_turn_local_counts_against_utterance_budget(self, local_first_env, monkeypatch):
        monkeypatch.setattr(audio_pipeline, "MAX_UTTERANCES", 1)
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local()))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_text(json.dumps(_turn_local(start_time=2.0, end_time=3.0)))
            assert json.loads(ws.receive_text())["type"] == "limit_reached"


# --- Deepgram overlap suppression -------------------------------------------

class TestDeepgramOverlapSuppression:
    def test_covered_by_local_range_uses_midpoint_with_pad(self):
        from audio_pipeline import _covered_by_local_range

        ranges = [(1.0, 3.0)]
        assert _covered_by_local_range(ranges, 1.1, 2.9)        # inside
        assert _covered_by_local_range(ranges, 0.9, 3.1)        # slightly wider — pad
        assert _covered_by_local_range(ranges, 2.8, 3.6)        # midpoint 3.2 ≤ 3.25
        assert not _covered_by_local_range(ranges, 2.6, 4.0)    # midpoint 3.3 > pad
        assert not _covered_by_local_range(ranges, 3.5, 4.5)    # the next span
        assert not _covered_by_local_range([], 1.0, 2.0)

    def test_local_ranges_are_bounded(self):
        from audio_pipeline import LOCAL_RANGES_MAX, _remember_local_range

        ranges: list = []
        for i in range(LOCAL_RANGES_MAX + 10):
            _remember_local_range(ranges, float(i), float(i) + 0.5)
        assert len(ranges) == LOCAL_RANGES_MAX
        assert ranges[-1] == (float(LOCAL_RANGES_MAX + 9), float(LOCAL_RANGES_MAX + 9) + 0.5)

    def test_no_deepgram_segments_at_all_once_local_first(self, local_first_env):
        """Superseded the old "covered dropped / uncovered passes" contract:
        production e2e (2026-08-24) showed Deepgram finalizing on pauses
        BEFORE the phone's turn_local arrived, so midpoint suppression let
        every turn be coached twice. Once local_first, audio is buffered for
        enrichment only — the transcriber never sees it, so neither the
        duplicate NOR the "phone missed this" span produces a transcript."""
        t = SequentialSegmentTranscriber([
            TranscriptSegment("Duplicate of the phone's turn.", 1.0, 3.0, speaker=0),
            TranscriptSegment("The phone missed this.", 5.0, 6.0, speaker=1),
        ])
        client = _inject(t)
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local(
                text="You never listen.", start_time=0.9, end_time=3.1,
            )))
            first = json.loads(ws.receive_text())
            ws.send_bytes(FRAME_100MS)  # would have been the covered segment
            ws.send_bytes(FRAME_100MS)  # would have been the uncovered span
            ws.send_text(json.dumps({"type": "stop"}))
            tail = []
            while True:
                ev = json.loads(ws.receive_text())
                tail.append(ev)
                if ev["type"] == "session_complete":
                    break

        assert first["type"] == "suggestion"
        assert first["utterance_text"] == "You never listen."
        assert [e["type"] for e in tail if e["type"] == "transcript"] == []
        assert all(e.get("utterance_text") != "The phone missed this." for e in tail)


# --- Server TTS ownership in a local-first session --------------------------

class TestLocalFirstTTS:
    def test_audio_stops_reaching_transcriber_once_local_first(self, local_first_env):
        """Production e2e (2026-08-24) showed Deepgram and the phone's on-device
        STT both coaching every turn: Deepgram finalizes on pauses BEFORE the
        phone's turn_local lands, so overlap suppression can't catch it. Once
        a session is local_first, audio frames are ring-buffered for
        enrichment but no longer forwarded to the transcriber at all."""
        class CountingTranscriber(StoppableTranscriber):
            def __init__(self):
                super().__init__()
                self.stream_calls = 0

            async def stream(self, audio_bytes):
                self.stream_calls += 1
                return await super().stream(audio_bytes)

        transcriber = CountingTranscriber()
        client = _inject(transcriber)
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_bytes(b"\x00\x01" * 800)          # legacy frame: forwarded
            ws.send_text(json.dumps(_turn_local()))    # session becomes local_first
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_bytes(b"\x00\x01" * 800)          # local_first frame: buffered only
            ws.send_bytes(b"\x00\x01" * 800)
        assert transcriber.stream_calls == 1

    def test_server_tts_skipped_once_local_first(self, local_first_env):
        """FakeTTS always returns audio when called — a None audio_b64 proves
        synthesize() was genuinely skipped. `speak` stays True: the phone
        voices it."""
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local()))
            resp = json.loads(ws.receive_text())
        assert resp["type"] == "suggestion"
        assert resp["speak"] is True
        assert resp["audio_b64"] is None

    def test_config_tts_server_keeps_server_voice(self, local_first_env):
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps({"type": "config", "tts": "server"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
            ws.send_text(json.dumps(_turn_local()))
            resp = json.loads(ws.receive_text())
        assert resp["audio_b64"] is not None

    def test_queued_deepgram_turn_loses_server_tts_after_first_turn_local(
        self, local_first_env,
    ):
        """TTS ownership is decided at the moment of synthesis: a Deepgram
        turn already IN FLIGHT (LLM running) when the first turn_local lands
        is voiced by the phone, not the server, because by the time its
        suggestion is ready the session is local-first. (A merely QUEUED
        Deepgram turn is superseded by the turn_local job — latest-wins.)"""
        llm = BlockingLLM(MOCK_LLM_JSON)
        t = SequentialSegmentTranscriber([
            TranscriptSegment("Queued before local-first.", 0.0, 1.0, speaker=0),
        ])
        client = _inject(t)
        app.state.llm_client = llm
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_bytes(FRAME_100MS)
            assert llm.started.wait(timeout=5)
            assert json.loads(ws.receive_text())["type"] == "transcript"
            ws.send_text(json.dumps(_turn_local(start_time=2.0, end_time=3.0)))
            llm.release.set()
            first = json.loads(ws.receive_text())
            second = json.loads(ws.receive_text())
        assert [first["utterance_text"], second["utterance_text"]] == [
            "Queued before local-first.", "You never listen to me.",
        ]
        assert first["audio_b64"] is None and second["audio_b64"] is None

    @pytest.mark.anyio
    async def test_tts_config_parsing(self):
        from audio_pipeline import SessionContext, _apply_config

        ctx = SessionContext(session_id="tts-cfg")
        await _apply_config(ctx, {"tts": "server"})
        assert ctx.tts_mode == "server"
        await _apply_config(ctx, {"tts": "elevenlabs"})  # unknown → ignored
        assert ctx.tts_mode == "server"
        await _apply_config(ctx, {"tts": None})
        assert ctx.tts_mode is None
        await _apply_config(ctx, {"tts": "on-device"})
        assert ctx.tts_mode == "on-device"
        await _apply_config(ctx, {"report_latency": "yes"})  # wrong type → ignored
        assert ctx.report_latency is False
        await _apply_config(ctx, {"report_latency": True})
        assert ctx.report_latency is True


# --- PCM ring buffer ---------------------------------------------------------

class TestPcmRingBuffer:
    def test_slice_by_session_time_is_exact(self):
        from audio_pipeline import PcmRingBuffer

        buf = PcmRingBuffer(seconds=2.0, sample_rate=1000)  # 1 kHz keeps the math legible
        # Ten 100 ms frames whose samples encode the frame index.
        for i in range(10):
            buf.append(np.full(100, i, dtype="<i2").tobytes())
        assert buf.seconds_received == 1.0
        samples = np.frombuffer(buf.slice(0.3, 0.5), dtype="<i2")
        assert samples.size == 200
        assert set(samples[:100].tolist()) == {3} and set(samples[100:].tolist()) == {4}
        # End clamped to what has been received; start before origin clamped to 0.
        assert np.frombuffer(buf.slice(0.95, 5.0), dtype="<i2").size == 50
        assert buf.slice(0.5, 0.5) == b"" and buf.slice(0.6, 0.4) == b""
        assert buf.slice(3.0, 4.0) == b""  # not yet received

    def test_old_audio_is_trimmed_but_addressing_stays_session_relative(self):
        from audio_pipeline import PcmRingBuffer

        buf = PcmRingBuffer(seconds=1.0, sample_rate=1000)  # capacity 2000 bytes
        for i in range(40):  # 4 s of audio into a 1 s buffer
            buf.append(np.full(100, i, dtype="<i2").tobytes())
        assert buf.seconds_received == 4.0
        # Something recent is still addressable by its ORIGINAL session time.
        recent = np.frombuffer(buf.slice(3.8, 3.9), dtype="<i2")
        assert recent.size == 100 and set(recent.tolist()) == {38}
        # The oldest audio is gone: an all-old window yields nothing, a window
        # straddling the trim point yields only the retained tail.
        assert buf.slice(0.0, 0.5) == b""
        assert len(buf._buf) <= int(1.1 * 2000) + 200  # bounded (hysteresis block)

    def test_pcm16_to_float32(self):
        from audio_pipeline import _pcm16_to_float32

        raw = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes() + b"\x01"  # stray odd byte
        out = _pcm16_to_float32(raw)
        assert out.dtype == np.float32 and out.size == 4
        assert out[0] == 0.0 and out[1] == 0.5 and out[2] == -1.0


# --- Enrichment on the streamed PCM ----------------------------------------

def _stream_one_second(ws) -> None:
    for _ in range(10):
        ws.send_bytes(FRAME_100MS)


class TestTurnLocalEnrichment:
    def test_tone_identity_and_relay_emitted_from_recovered_slice(
        self, local_first_env, monkeypatch,
    ):
        tone = FakeToneId(surface=True)
        spk = FakeSpeakerId()
        # Two enrolled people: the owner's print matches the slice (cosine 1.0),
        # the partner's doesn't (0.0) — so the verdict is "self".
        store = FakeVoiceprintStore(docs=[SELF_DOC, ALEX_DOC])
        relay = FakeRelay()
        monkeypatch.setattr(audio_pipeline, "tone_id", tone)
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        monkeypatch.setattr(audio_pipeline, "watch_relay", relay)
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = store
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            # The phone says Speaker A is NOT self; the server's voiceprint
            # disagrees and corrects it.
            ws.send_text(json.dumps(_turn_local(start_time=0.0, end_time=1.0, is_self=False)))
            types_seen: dict[str, dict] = {}
            for _ in range(3):
                msg = json.loads(ws.receive_text())
                types_seen[msg["type"]] = msg
            ws.send_text(json.dumps({"type": "stop"}))
            done = json.loads(ws.receive_text())

        assert set(types_seen) == {"suggestion", "tone_flag", "speaker_identity"}
        flag = types_seen["tone_flag"]
        assert flag["source"] == "audio" and flag["label"] == "angry"
        assert flag["scores"]["angry"] == 0.8 and flag["confidence"] == 0.8
        assert flag["speaker"] == "Speaker A" and flag["end_time"] == 1.0
        # The slice handed to the model is the 1 s the phone reported, at 16 kHz.
        assert tone.calls == [(16000, 16000)]
        ident = types_seen["speaker_identity"]
        assert ident["is_self"] is True and ident["score"] == 1.0
        assert ident["person_id"] == "self" and ident["display_name"] == "You"
        assert store.reads == ["test-user"]
        # Embedded as ONE turn by the phone's label, against every enrolled print.
        assert spk.calls == [{
            "samples": 16000, "sr": 16000,
            "turns": [{"speaker": "Speaker A", "start_time": 0.0, "end_time": 1.0}],
            "people": ["alex", "self"],
        }]
        # The relay got the most informed view: the server-CORRECTED identity
        # (phone said is_self=False) and the audio tone flag.
        assert len(relay.calls) == 1
        call = relay.calls[0]
        assert call["uid"] == "test-user"
        assert call["event"].text == "You never listen to me."
        assert call["event"].is_self is True
        assert call["event"].speaker_person_id == "self"
        assert call["event"].speaker_match_score == 1.0
        assert call["tone_flag"].label == "angry" and call["tone_flag"].source == "audio"
        assert "latency_summary" in done  # local-first → summary automatically

    def test_dark_tone_is_logged_not_surfaced(self, local_first_env, monkeypatch, caplog):
        tone = FakeToneId(surface=False)
        relay = FakeRelay()
        monkeypatch.setattr(audio_pipeline, "tone_id", tone)
        monkeypatch.setattr(audio_pipeline, "watch_relay", relay)
        client = _inject(StoppableTranscriber())
        with caplog.at_level(logging.INFO, logger="audio_pipeline"):
            with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
                _stream_one_second(ws)
                ws.send_text(json.dumps(_turn_local()))
                assert json.loads(ws.receive_text())["type"] == "suggestion"
                ws.send_text(json.dumps({"type": "stop"}))  # drains enrichment first
                done = json.loads(ws.receive_text())
        assert done["type"] == "session_complete"  # no tone_flag ever hit the wire
        assert tone.calls  # …but it WAS computed
        assert any("audio tone (dark)" in r.getMessage() for r in caplog.records)
        # …and it did not leak to the watch through the relay either.
        assert relay.calls and relay.calls[0]["tone_flag"] is None

    def test_escalation_delta_tracked_per_speaker_across_turns(self, local_first_env, monkeypatch):
        """The round-2 seam: a dimensional backend result carries ``arousal``
        and the module exposes ``EscalationTracker`` / ``annotate_escalation``
        (the REAL tone_id ones, on a fake classifier). The session keeps one
        tracker, so the same speaker's second turn is judged against their
        first: unscored → escalating, with the delta on the wire."""
        import tone_id

        class EscalatingToneId(FakeToneId):
            EscalationTracker = tone_id.EscalationTracker
            annotate_escalation = staticmethod(tone_id.annotate_escalation)

            def __init__(self):
                super().__init__(surface=True)
                self.arousals = [0.40, 0.55]

            def classify_pcm(self, pcm, sr):
                self.calls.append((int(pcm.size), sr))
                a = self.arousals.pop(0)
                return {"label": tone_id.UNSCORED_LABEL, "confidence": 0.0, "arousal": a, "kind": "dimensional",
                        "backend": "odyssey_dim", "model": "fake",
                        "scores": {"arousal": a, "dominance": 0.5, "valence": 0.5}}

        tone = EscalatingToneId()
        monkeypatch.setattr(audio_pipeline, "tone_id", tone)
        client = _inject(StoppableTranscriber())
        flags = []
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            ws.send_text(json.dumps(_turn_local(start_time=0.0, end_time=1.0)))
            for _ in range(2):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "tone_flag":
                    flags.append(msg)
            _stream_one_second(ws)
            ws.send_text(json.dumps(_turn_local(start_time=1.0, end_time=2.0)))
            for _ in range(2):
                msg = json.loads(ws.receive_text())
                if msg["type"] == "tone_flag":
                    flags.append(msg)
            ws.send_text(json.dumps({"type": "stop"}))
            json.loads(ws.receive_text())
        assert [f["label"] for f in flags] == [tone_id.UNSCORED_LABEL, tone_id.ESCALATION_LABEL]
        assert flags[0]["scores"]["arousal"] == 0.40 and "arousal_delta" not in flags[0]["scores"]
        assert flags[1]["scores"]["arousal_delta"] == pytest.approx(0.15)
        assert flags[1]["confidence"] == 1.0  # 0.15 ≥ 2× the pinned 0.03
        assert tone.calls == [(16000, 16000), (16000, 16000)]

    def test_too_little_audio_skips_models_cleanly(self, local_first_env, monkeypatch):
        tone = FakeToneId()
        spk = FakeSpeakerId()
        monkeypatch.setattr(audio_pipeline, "tone_id", tone)
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = FakeVoiceprintStore(docs=[SELF_DOC])
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_bytes(FRAME_100MS)  # 100 ms — below both 1 s floors
            ws.send_text(json.dumps(_turn_local(start_time=0.0, end_time=0.1)))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_text(json.dumps({"type": "stop"}))
            assert json.loads(ws.receive_text())["type"] == "session_complete"
        assert tone.calls == [] and spk.calls == []

    def test_partner_match_names_the_person(self, local_first_env, monkeypatch):
        """A slice matching a PARTNER's print names them (is_self False) even
        though the phone claimed is_self=True; the relay is handed the
        corrected event and no tone flag (tone_id absent here)."""
        spk = FakeSpeakerId()
        relay = FakeRelay()
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        monkeypatch.setattr(audio_pipeline, "watch_relay", relay)
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = FakeVoiceprintStore(docs=[
            dict(SELF_DOC, embedding=[0.2, 0.0]),   # 0.2 < threshold
            dict(ALEX_DOC, embedding=[0.9, 0.0]),   # 0.9 → Alex
        ])
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            ws.send_text(json.dumps(_turn_local(speaker="Speaker B", is_self=True)))
            ident, _ = recv_until(ws, lambda m: m["type"] == "speaker_identity")
            ws.send_text(json.dumps({"type": "stop"}))
            json.loads(ws.receive_text())
        assert ident == {
            "type": "speaker_identity", "session_id": LOCAL_SID,
            "speaker": "Speaker B", "person_id": "alex", "display_name": "Alex",
            "is_self": False, "score": 0.9,
        }
        assert relay.calls[0]["event"].is_self is False
        assert relay.calls[0]["event"].speaker_person_id == "alex"
        assert relay.calls[0]["tone_flag"] is None

    def test_no_match_is_an_honest_unknown(self, local_first_env, monkeypatch):
        spk = FakeSpeakerId()
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = FakeVoiceprintStore(docs=[
            dict(SELF_DOC, embedding=[0.3, 0.0]),
        ])
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            ws.send_text(json.dumps(_turn_local()))
            ident, _ = recv_until(ws, lambda m: m["type"] == "speaker_identity")
        assert ident["person_id"] is None and ident["display_name"] is None
        assert ident["is_self"] is False
        assert ident["score"] == 0.3  # the best near-miss, for inspectability

    def test_enrichment_failures_never_break_the_session(
        self, local_first_env, monkeypatch, caplog,
    ):
        """Tone model raises, voiceprint store raises, relay raises — the
        cloud suggestion still arrives, the other steps still run, the
        control channel still works, and stop completes."""
        tone = FakeToneId(error=RuntimeError("model exploded"))
        spk = FakeSpeakerId()
        store = FakeVoiceprintStore(error=OSError("gcs down"))

        class ExplodingRelay:
            def push_turn_local(self, uid, event, *, tone_flag=None):
                raise RuntimeError("watch relay down")

        monkeypatch.setattr(audio_pipeline, "tone_id", tone)
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        monkeypatch.setattr(audio_pipeline, "watch_relay", ExplodingRelay())
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = store
        with caplog.at_level(logging.WARNING, logger="audio_pipeline"):
            with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
                _stream_one_second(ws)
                ws.send_text(json.dumps(_turn_local()))
                assert json.loads(ws.receive_text())["type"] == "suggestion"
                ws.send_text(json.dumps({"type": "config"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"
                ws.send_text(json.dumps({"type": "stop"}))
                done = json.loads(ws.receive_text())
        assert done["type"] == "session_complete"
        assert tone.calls and store.reads  # every step was attempted
        failed = {
            r.getMessage().split(" enrichment failed")[0].split()[-1]
            for r in caplog.records if "enrichment failed" in r.getMessage()
        }
        assert failed == {"tone", "identity", "relay"}

    def test_enrichment_skipped_without_optional_modules(self, local_first_env):
        # tone_id / speaker_id / relay all None (the fixture) → only the
        # suggestion, and the session is otherwise indistinguishable.
        client = _inject(StoppableTranscriber())
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            ws.send_text(json.dumps(_turn_local()))
            assert json.loads(ws.receive_text())["type"] == "suggestion"
            ws.send_text(json.dumps({"type": "stop"}))
            assert json.loads(ws.receive_text())["type"] == "session_complete"

    # -- review 2026-08-24: enrichment is bounded per session -----------------

    def test_inflight_enrichment_is_capped(self, local_first_env, monkeypatch, caplog):
        """turn_local frames arriving faster than enrichment finishes must
        NOT pile up unbounded model passes + store reads: beyond
        MAX_ENRICHMENT_INFLIGHT in-flight tasks, further turns are simply not
        enriched (the cloud suggestion path is unaffected — it has its own
        budget). Here the identity model blocks until released, so every
        task started stays in flight while a burst of turns lands."""
        monkeypatch.setattr(audio_pipeline, "MAX_ENRICHMENT_INFLIGHT", 2)
        release = threading.Event()

        class BlockingSpeakerId(FakeSpeakerId):
            def identify_speakers_multi(self, *args, **kwargs):
                assert release.wait(timeout=10), "test never released the model"
                return super().identify_speakers_multi(*args, **kwargs)

        spk = BlockingSpeakerId()
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        client = _inject(StoppableTranscriber())
        app.state.recordings_store = FakeVoiceprintStore(docs=[SELF_DOC])
        burst = 6
        with caplog.at_level(logging.WARNING, logger="audio_pipeline"):
            with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
                _stream_one_second(ws)
                for i in range(burst):
                    ws.send_text(json.dumps(_turn_local(
                        start_time=0.0, end_time=1.0, text=f"turn {i}",
                    )))
                # Every turn still gets its cloud suggestion (or is superseded
                # by latest-wins) — wait for the LAST one so we know the
                # receive loop has processed the whole burst before releasing.
                recv_until(
                    ws, lambda m: m["type"] == "suggestion"
                    and m["utterance_text"] == f"turn {burst - 1}", limit=40,
                )
                release.set()
                ws.send_text(json.dumps({"type": "stop"}))
                recv_until(ws, lambda m: m["type"] == "session_complete", limit=40)
        # Only the capped number of model passes ever ran.
        assert len(spk.calls) == 2
        assert "enrichment tasks already in flight" in caplog.text

    def test_voiceprints_are_read_once_per_session(self, local_first_env, monkeypatch):
        """The account's voiceprint documents are a store read (GCS list +
        downloads); they are cached per session, not fetched on every turn."""
        spk = FakeSpeakerId()
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        client = _inject(StoppableTranscriber())
        store = FakeVoiceprintStore(docs=[SELF_DOC])
        app.state.recordings_store = store
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            for i in range(3):
                ws.send_text(json.dumps(_turn_local(text=f"turn {i}")))
                recv_until(ws, lambda m: m["type"] == "speaker_identity")
            ws.send_text(json.dumps({"type": "stop"}))
            recv_until(ws, lambda m: m["type"] == "session_complete", limit=20)
        assert len(spk.calls) == 3          # every turn WAS identified…
        assert store.reads == ["test-user"]  # …from ONE read of the prints

    def test_voiceprint_cache_refreshes_after_ttl(self, local_first_env, monkeypatch):
        """An enrollment made mid-conversation is still picked up: the cache
        expires after VOICEPRINT_CACHE_TTL_S."""
        monkeypatch.setattr(audio_pipeline, "VOICEPRINT_CACHE_TTL_S", 0.0)
        spk = FakeSpeakerId()
        monkeypatch.setattr(audio_pipeline, "speaker_id", spk)
        client = _inject(StoppableTranscriber())
        store = FakeVoiceprintStore(docs=[SELF_DOC])
        app.state.recordings_store = store
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            _stream_one_second(ws)
            for i in range(2):
                ws.send_text(json.dumps(_turn_local(text=f"turn {i}")))
                recv_until(ws, lambda m: m["type"] == "speaker_identity")
            ws.send_text(json.dumps({"type": "stop"}))
            recv_until(ws, lambda m: m["type"] == "session_complete", limit=20)
        assert store.reads == ["test-user", "test-user"]


# --- Streaming cloud LLM: partial preview then final -------------------------

class TestStreamingCloudSuggestion:
    def test_first_suggestion_extraction(self):
        from audio_pipeline import _first_suggestion_in

        assert _first_suggestion_in('{"suggestions": ["I hear') is None  # unterminated
        assert _first_suggestion_in('{"suggestions": ["I hear you."') == "I hear you."
        assert _first_suggestion_in('```json\n{"suggestions": ["Say \\"no\\"."') == 'Say "no".'
        assert _first_suggestion_in('{"suggestions": [""') is None  # empty string is no preview
        assert _first_suggestion_in('{"tone_score": {}') is None

    def test_partial_then_final_for_local_first_client(self, local_first_env):
        llm = StreamingLLM(MOCK_LLM_JSON, chunk=5)
        client = _inject(StoppableTranscriber())
        app.state.llm_client = llm
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local()))
            partial = json.loads(ws.receive_text())
            final = json.loads(ws.receive_text())

        assert partial["type"] == "suggestion" and partial["partial"] is True
        assert partial["suggestions"] == ["I hear what you're saying."]
        assert partial["speak"] is False and partial["audio_b64"] is None
        assert partial["suggestion_source"] == "cloud"
        assert partial["utterance_text"] == final["utterance_text"]
        assert final["partial"] is False
        assert len(final["suggestions"]) == 3
        assert llm.stream_calls and not llm.complete_calls  # streamed, not completed
        assert "You never listen to me." in llm.stream_calls[0]

    def test_legacy_client_never_sees_a_partial(self, local_first_env):
        """Same streaming-capable LLM, but no turn_local → the plain
        complete() path and a single final event (an old client would render
        a partial as a second suggestion)."""
        llm = StreamingLLM(MOCK_LLM_JSON)
        client = _inject(FakeTranscriber())
        app.state.llm_client = llm
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_bytes(b"\x00" * 50)
            resp = recv_skipping_transcripts(ws)
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"  # nothing else queued
        assert resp["type"] == "suggestion" and resp["partial"] is False
        assert llm.complete_calls and not llm.stream_calls

    def test_stream_without_a_complete_suggestion_still_yields_final(self, local_first_env):
        llm = StreamingLLM(json.dumps({"suggestions": [], "importance": 10}))
        client = _inject(StoppableTranscriber())
        app.state.llm_client = llm
        with open_ws(client, f"/ws/session/{LOCAL_SID}") as ws:
            ws.send_text(json.dumps(_turn_local()))
            resp = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "config"}))
            assert json.loads(ws.receive_text())["type"] == "config_ack"
        assert resp["partial"] is False and resp["suggestions"] == []

    @pytest.mark.anyio
    async def test_generate_suggestions_prompt_byte_identical_without_tone(self):
        from audio_pipeline import _generate_suggestions
        from models.audio import Utterance

        llm = MagicMock()
        llm.complete.return_value = MOCK_LLM_JSON
        u = Utterance(session_id="s", speaker="Speaker A", text="hi", start_time=0.0, end_time=1.0)
        await _generate_suggestions(llm, u, 50, "Husband")
        assert llm.complete.call_args.kwargs["user"] == 'Transcript turn: "hi"'
        await _generate_suggestions(llm, u, 50, "Husband", None, {"text_tone": {"sarcasm": 90}})
        assert llm.complete.call_args.kwargs["user"] == (
            'Transcript turn: "hi"\n\nOn-device signals for this turn (measured by '
            "the phone; treat as hints, not facts):\n- text tone: sarcasm 90/100"
        )
