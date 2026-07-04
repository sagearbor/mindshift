"""Tests for the M2 real-time audio pipeline (WebSocket, transcription, diarization, TTS).

The pipeline depends on external speech providers (Deepgram for transcription,
a TTS service for earpiece audio). Those are credential-gated and report
themselves *unavailable* when not configured — the pipeline never fabricates
transcripts or audio. To exercise the pipeline logic without live providers,
these tests inject the test doubles defined below via ``app.state``.
"""

import json
import threading
import time
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from audio_pipeline import TranscriberUnavailable, TranscriptSegment
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
        """Audio chunks must never yield a fabricated suggestion when unavailable.

        The unavailable notice is sent ONCE (on entering the state); binary
        frames afterwards are ignored silently — no suggestion, no re-send
        flood. The next reply on the wire is the config ack.
        """
        with unavailable_ws.websocket_connect("/ws/session/unavail-2") as ws:
            assert json.loads(ws.receive_text())["type"] == "transcription_unavailable"
            ws.send_bytes(b"\x00" * 50)
            ws.send_bytes(b"\x00" * 50)
            ws.send_text(json.dumps({"type": "config"}))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "config_ack"  # nothing sent for the frames

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
            with client.websocket_connect("/ws/session/stop-1") as ws:
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
            with client.websocket_connect("/ws/session/stop-2") as ws:
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
        with unavailable_ws.websocket_connect("/ws/session/stop-3") as ws:
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
            with client.websocket_connect("/ws/session/worker-1") as ws:
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
            with client.websocket_connect("/ws/session/worker-2") as ws:
                ws.send_bytes(b"\x00" * 50)
                events = [json.loads(ws.receive_text()) for _ in range(3)]
        finally:
            _clear_overrides()

        assert [e["utterance_text"] for e in events] == ["First.", "Second.", "Third."]


# ---------------------------------------------------------------------------
# Unavailable notice is sent once, not per frame (F7)
# ---------------------------------------------------------------------------

class TestUnavailableNoticeOnce:
    def test_midstream_failure_notice_sent_once(self):
        """After a mid-stream failure the client is told once; further binary
        frames are ignored silently (no per-frame re-send flood)."""
        client = _inject(DyingTranscriber())
        try:
            with client.websocket_connect("/ws/session/once-1") as ws:
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
            with client.websocket_connect("/ws/session/tts-own-1") as ws:
                ws.send_bytes(b"\x00" * 50)
                assert json.loads(ws.receive_text())["type"] == "suggestion"
        finally:
            _clear_overrides()

        assert tts.closed is False  # shared instance must survive the session
