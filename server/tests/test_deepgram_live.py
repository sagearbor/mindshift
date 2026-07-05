"""Tests for the real Deepgram live-streaming integration.

These run WITHOUT a DEEPGRAM_API_KEY: a local fake Deepgram WebSocket server
(``FakeDeepgramServer``) speaks just enough of Deepgram's live protocol to
exercise ``DeepgramTranscriber`` end-to-end — auth header, binary audio in,
``Results``/``UtteranceEnd`` JSON out, ``KeepAlive``/``CloseStream`` control
messages. TTS is exercised against ``httpx.MockTransport``. The one test that
touches the real Deepgram API is skipped unless a key is present.
"""

import asyncio
import base64
import json
import os
import socket
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
from starlette.testclient import TestClient
from websockets.asyncio.server import serve

import audio_pipeline
from audio_pipeline import (
    DeepgramTranscriber,
    TranscriberUnavailable,
    TranscriptSegment,
    TTSClient,
    _normalize_segments,
)
from main import app


# ---------------------------------------------------------------------------
# Fake Deepgram live server
# ---------------------------------------------------------------------------

def _results_payload(
    text: str,
    start: float,
    duration: float,
    speaker: int | None = None,
    is_final: bool = True,
    speech_final: bool = True,
    confidence: float = 0.97,
) -> dict:
    """Build a Deepgram ``Results`` message shaped like the real API's."""
    words = []
    t = start
    for w in text.split():
        word = {
            "word": w.lower().strip(".,"),
            "start": t,
            "end": t + 0.2,
            "confidence": confidence,
            "punctuated_word": w,
        }
        if speaker is not None:
            word["speaker"] = speaker
        words.append(word)
        t += 0.25
    return {
        "type": "Results",
        "channel_index": [0, 1],
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {
            "alternatives": [
                {"transcript": text, "confidence": confidence, "words": words}
            ]
        },
    }


class FakeDeepgramServer:
    """Local WebSocket server standing in for wss://api.deepgram.com/v1/listen.

    Records auth headers, binary audio frames, and JSON control messages.
    Replies with one queued JSON payload per binary audio frame received.
    """

    def __init__(self) -> None:
        self.auth_headers: list[str | None] = []
        self.audio_frames: list[bytes] = []
        self.control_messages: list[dict] = []
        self.responses: list[dict] = []  # popped one per binary frame
        # Sent (all) upon CloseStream, before Metadata — models Deepgram
        # flushing its remaining Results when the client ends the stream.
        self.close_stream_responses: list[dict] = []
        self.close_after_frames: int | None = None
        self._server = None
        self.url: str = ""

    async def __aenter__(self) -> "FakeDeepgramServer":
        self._server = await serve(self._handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, connection) -> None:
        self.auth_headers.append(connection.request.headers.get("Authorization"))
        async for message in connection:
            if isinstance(message, bytes):
                self.audio_frames.append(message)
                # Queued response goes out BEFORE any simulated death, so a
                # final Results followed by a dropped connection is expressible.
                if self.responses:
                    await connection.send(json.dumps(self.responses.pop(0)))
                if (
                    self.close_after_frames is not None
                    and len(self.audio_frames) >= self.close_after_frames
                ):
                    # Simulate Deepgram dropping the connection mid-session.
                    await connection.close(code=1011, reason="fake mid-session death")
                    return
            else:
                payload = json.loads(message)
                self.control_messages.append(payload)
                if payload.get("type") == "CloseStream":
                    # Deepgram flushes remaining Results, then Metadata, then
                    # closes the socket.
                    for pending in self.close_stream_responses:
                        await connection.send(json.dumps(pending))
                    self.close_stream_responses = []
                    await connection.send(json.dumps({"type": "Metadata"}))
                    return


async def _stream_until_segments(
    transcriber: DeepgramTranscriber, attempts: int = 100,
) -> list[TranscriptSegment]:
    """Send audio frames until finalized segments come back (or give up)."""
    for _ in range(attempts):
        segments = await transcriber.stream(b"\x00\x00" * 800)  # 100ms of PCM
        if segments:
            return segments
        await asyncio.sleep(0.01)
    return []


async def _wait_for(predicate, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# DeepgramTranscriber against the fake server
# ---------------------------------------------------------------------------

class TestDeepgramTranscriberLive:
    @pytest.mark.anyio
    async def test_stream_returns_finalized_segment_with_real_metadata(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.responses.append(
                _results_payload("Hello there.", start=1.5, duration=1.0, speaker=1)
            )
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                segments = await _stream_until_segments(t)
            finally:
                await t.close()

        assert server.auth_headers == ["Token test-key"]
        assert len(segments) == 1
        seg = segments[0]
        assert seg.text == "Hello there."
        assert seg.start_time == pytest.approx(1.5)
        assert seg.end_time == pytest.approx(2.5)
        assert seg.speaker == 1
        assert 0.0 <= seg.confidence <= 1.0

    @pytest.mark.anyio
    async def test_is_final_segments_accumulate_until_speech_final(self, monkeypatch):
        """Two is_final results join into ONE utterance at the speech_final."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.responses.append(_results_payload(
                "I just feel like", start=0.5, duration=1.0,
                speaker=0, speech_final=False,
            ))
            server.responses.append(_results_payload(
                "you never listen to me.", start=1.5, duration=1.5,
                speaker=0, speech_final=True,
            ))
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                segments = await _stream_until_segments(t)
            finally:
                await t.close()

        assert len(segments) == 1
        seg = segments[0]
        assert seg.text == "I just feel like you never listen to me."
        assert seg.start_time == pytest.approx(0.5)
        assert seg.end_time == pytest.approx(3.0)
        assert seg.speaker == 0

    @pytest.mark.anyio
    async def test_utterance_end_flushes_pending_finals(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.responses.append(_results_payload(
                "Trailing thought", start=2.0, duration=0.8,
                speaker=None, speech_final=False,
            ))
            server.responses.append(
                {"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 2.8}
            )
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                segments = await _stream_until_segments(t)
            finally:
                await t.close()

        assert len(segments) == 1
        assert segments[0].text == "Trailing thought"
        assert segments[0].speaker is None  # no diarization data — not invented

    @pytest.mark.anyio
    async def test_interim_results_are_not_emitted(self, monkeypatch):
        """Interim (is_final=false) text must never surface as an utterance."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.responses.append(_results_payload(
                "unstable interim guess", start=0.0, duration=0.5,
                is_final=False, speech_final=False,
            ))
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                for _ in range(10):
                    assert await t.stream(b"\x00\x00" * 800) == []
                    await asyncio.sleep(0.01)
            finally:
                await t.close()

    @pytest.mark.anyio
    async def test_keepalive_sent_while_idle(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            t = DeepgramTranscriber(url=server.url, keepalive_interval=0.05)
            await t.connect()
            try:
                got_keepalive = await _wait_for(lambda: any(
                    m.get("type") == "KeepAlive" for m in server.control_messages
                ))
            finally:
                await t.close()
            assert got_keepalive

    @pytest.mark.anyio
    async def test_mid_session_socket_death_raises_unavailable(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.close_after_frames = 1
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                with pytest.raises(TranscriberUnavailable):
                    for _ in range(100):
                        await t.stream(b"\x00\x00" * 800)
                        await asyncio.sleep(0.01)
            finally:
                await t.close()

    @pytest.mark.anyio
    async def test_close_sends_closestream_and_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            await t.close()
            await t.close()  # double-close must not raise
            got_closestream = await _wait_for(lambda: any(
                m.get("type") == "CloseStream" for m in server.control_messages
            ))
            assert got_closestream
        assert not t.is_connected

    @pytest.mark.anyio
    async def test_finish_delivers_flushed_final_segments(self, monkeypatch):
        """F2: finish() sends Finalize+CloseStream, awaits Deepgram's flush,
        and returns the remaining segments — the session's LAST utterance is
        delivered instead of dropped."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.close_stream_responses.append(_results_payload(
                "The very last utterance.", start=4.0, duration=1.2, speaker=0,
            ))
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            assert await t.stream(b"\x00\x00" * 800) == []
            segments = await t.finish()

            assert [s.text for s in segments] == ["The very last utterance."]
            assert segments[0].speaker == 0
            types = [m.get("type") for m in server.control_messages]
            assert types.index("Finalize") < types.index("CloseStream")
            # Idempotent, and close() afterwards is a safe no-op.
            assert await t.finish() == []
            await t.close()
        assert not t.is_connected

    @pytest.mark.anyio
    async def test_finish_without_connect_is_safe(self):
        t = DeepgramTranscriber()
        assert await t.finish() == []

    @pytest.mark.anyio
    async def test_finalized_segments_survive_connection_death(self, monkeypatch):
        """F3: a segment finalized before the socket died is still DELIVERED;
        the recorded failure raises only once the queue is empty."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        async with FakeDeepgramServer() as server:
            server.responses.append(_results_payload(
                "Last words before the drop.", start=0.0, duration=1.0, speaker=0,
            ))
            server.close_after_frames = 1
            t = DeepgramTranscriber(url=server.url)
            await t.connect()
            try:
                # Server replies with a final Results, then drops (1011).
                delivered = await t.stream(b"\x00\x00" * 800)
                assert await _wait_for(lambda: not t.is_connected)
                if not delivered:  # segment queued but not yet drained
                    delivered = await t.stream(b"\x00\x00" * 800)
                assert [s.text for s in delivered] == ["Last words before the drop."]
                # Queue now empty → the failure surfaces honestly.
                with pytest.raises(TranscriberUnavailable):
                    await t.stream(b"\x00\x00" * 800)
            finally:
                await t.close()

    @pytest.mark.anyio
    async def test_connect_failure_raises_unavailable(self, monkeypatch):
        """Nothing listening on the port → honest TranscriberUnavailable."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        # Grab a free port, then close it so the connection is refused.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        t = DeepgramTranscriber(url=f"ws://127.0.0.1:{port}")
        with pytest.raises(TranscriberUnavailable):
            await t.connect()
        await t.close()  # close after failed connect must not raise


# ---------------------------------------------------------------------------
# Pipeline consumption of TranscriptSegment (speaker mapping, normalization)
# ---------------------------------------------------------------------------

MOCK_LLM_JSON = json.dumps({
    "suggestions": ["One.", "Two.", "Three."],
    "tone_score": {
        "warmth": 60, "defensiveness": 30, "sarcasm": 10,
        "constructiveness": 55, "overall": 65,
    },
})


class SegmentTranscriber:
    """Test double whose stream() returns TranscriptSegment lists (the real
    DeepgramTranscriber's contract), draining a preloaded queue."""

    def __init__(self, segments: list[TranscriptSegment]) -> None:
        self._segments = list(segments)

    async def connect(self) -> None:
        pass

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        segments, self._segments = self._segments, []
        return segments

    async def close(self) -> None:
        pass


def _clear_overrides() -> None:
    for attr in ("transcriber_factory", "tts_client", "diarizer_factory"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)


class TestPipelineSegmentConsumption:
    def test_deepgram_speaker_ints_map_to_labels_and_bypass_heuristic(self):
        """speaker=1 → 'Speaker B', speaker=0 → 'Speaker A' — NOT alternation
        (alternation would label consecutive utterances A then B regardless)."""
        _clear_overrides()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = mock_llm
        app.state.transcriber_factory = lambda: SegmentTranscriber([
            TranscriptSegment("First utterance.", 0.2, 1.4, speaker=1),
            TranscriptSegment("Second utterance.", 1.9, 3.0, speaker=1),
            TranscriptSegment("Third utterance.", 3.5, 4.2, speaker=0),
        ])
        try:
            with TestClient(app).websocket_connect("/ws/session/b3583132-3ac3-5f01-ad83-bb1590e1b5d4") as ws:
                ws.send_bytes(b"\x00" * 50)
                events = [json.loads(ws.receive_text()) for _ in range(3)]
        finally:
            _clear_overrides()

        assert [e["type"] for e in events] == ["suggestion"] * 3
        assert [e["speaker"] for e in events] == ["Speaker B", "Speaker B", "Speaker A"]
        assert [e["utterance_text"] for e in events] == [
            "First utterance.", "Second utterance.", "Third utterance.",
        ]

    def test_segment_without_speaker_falls_back_to_diarizer(self):
        _clear_overrides()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = mock_llm
        app.state.transcriber_factory = lambda: SegmentTranscriber([
            TranscriptSegment("No speaker data here.", 0.0, 1.0, speaker=None),
        ])
        try:
            with TestClient(app).websocket_connect("/ws/session/26e90aed-f581-5810-a375-d810a82e8ffa") as ws:
                ws.send_bytes(b"\x00" * 50)
                event = json.loads(ws.receive_text())
        finally:
            _clear_overrides()

        assert event["speaker"] == "Speaker A"  # first turn of the heuristic

    def test_normalize_segments_contract(self):
        assert _normalize_segments(None) == []
        assert _normalize_segments("") == []

        legacy = _normalize_segments("plain string transcript")
        assert len(legacy) == 1
        assert legacy[0].text == "plain string transcript"
        assert legacy[0].speaker is None
        # No timing data from a legacy str — zeros, not a fabricated duration.
        assert (legacy[0].start_time, legacy[0].end_time) == (0.0, 0.0)

        seg = TranscriptSegment("real", 1.0, 2.0, speaker=1)
        assert _normalize_segments([seg]) == [seg]


# ---------------------------------------------------------------------------
# Deepgram Aura TTS (httpx.MockTransport — no network)
# ---------------------------------------------------------------------------

class TestDeepgramAuraTTS:
    @pytest.mark.anyio
    async def test_synthesize_success_returns_base64_audio(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("Authorization")
            captured["model"] = request.url.params.get("model")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=b"fake-mp3-bytes")

        tts = TTSClient(transport=httpx.MockTransport(handler))
        out = await tts.synthesize("Hello world")

        assert out == base64.b64encode(b"fake-mp3-bytes").decode("ascii")
        assert captured["auth"] == "Token test-key"
        assert captured["model"] == "aura-2-thalia-en"
        assert captured["body"] == {"text": "Hello world"}

    @pytest.mark.anyio
    async def test_synthesize_http_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"err_code": "INVALID_AUTH"})

        tts = TTSClient(transport=httpx.MockTransport(handler))
        assert await tts.synthesize("Hello world") is None

    @pytest.mark.anyio
    async def test_synthesize_network_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        tts = TTSClient(transport=httpx.MockTransport(handler))
        assert await tts.synthesize("Hello world") is None

    @pytest.mark.anyio
    async def test_synthesize_reuses_one_http_client(self, monkeypatch):
        """F9: one lazily created AsyncClient per TTSClient instance — no
        per-call TCP+TLS handshake churn. aclose() is idempotent."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, content=b"mp3")

        tts = TTSClient(transport=httpx.MockTransport(handler))
        assert await tts.synthesize("one") is not None
        first_client = tts._client
        assert first_client is not None
        assert await tts.synthesize("two") is not None
        assert tts._client is first_client  # reused, not rebuilt
        assert len(requests) == 2

        await tts.aclose()
        await tts.aclose()  # idempotent — double-close must not raise
        assert tts._client is None


def test_requirements_pin_websockets_14():
    """F1: connect(additional_headers=...) needs websockets>=14 — 13.x only
    accepts extra_headers, so every session would falsely report unavailable."""
    import pathlib
    import re

    import websockets as ws_lib

    assert int(ws_lib.__version__.split(".")[0]) >= 14

    root = pathlib.Path(__file__).resolve().parents[2]
    for req in (root / "requirements.txt", root / "server" / "requirements.txt"):
        line = next(
            ln for ln in req.read_text().splitlines()
            if ln.strip().startswith("websockets")
        )
        m = re.match(r"websockets>=(\d+)", line.strip())
        assert m is not None and int(m.group(1)) >= 14, f"{req}: {line!r}"


def test_tls_context_only_for_wss():
    """wss:// (real Deepgram) gets a verifying TLS context anchored on certifi
    so an empty system CA store can't break the connection; plain ws:// (the
    local fake server used in these tests) gets no TLS context."""
    import ssl

    from audio_pipeline import _tls_context_for

    assert _tls_context_for("ws://127.0.0.1:1234") is None

    ctx = _tls_context_for("wss://api.deepgram.com/v1/listen")
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    # at least one CA is loaded (empty store would defeat the purpose)
    assert ctx.cert_store_stats().get("x509_ca", 0) > 0


# ---------------------------------------------------------------------------
# Pipeline reconnect against the fake Deepgram server (P1-1): kill it,
# bring it back on the same port, and the live session recovers.
# ---------------------------------------------------------------------------

class ThreadedFakeDeepgramServer:
    """A :class:`FakeDeepgramServer` handler on a private event loop in a
    daemon thread.

    The pipeline-level reconnect test drives the app through the sync
    ``TestClient`` (whose loop lives in its own portal thread), so the fake
    Deepgram server cannot share a test-owned loop — and the test must be able
    to KILL the server and resurrect a fresh one on the SAME port while the
    WebSocket session stays live.
    """

    def __init__(self, inner: FakeDeepgramServer, port: int = 0) -> None:
        self.inner = inner  # records frames/control messages, holds responses
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._serve()), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(timeout=5), "fake Deepgram server did not start"

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = await serve(self.inner._handler, "127.0.0.1", self.port)
        self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        try:
            await self._stop.wait()
        finally:
            # Force-close the listener AND any live connection immediately —
            # exactly what a dying Deepgram looks like to the transcriber, and
            # (crucially) frees the port promptly so the test can rebind a fresh
            # server on it. A graceful drain could otherwise wait on the still-
            # open transcriber socket and stall shutdown past the join timeout
            # on a loaded CI runner (the source of an intermittent failure).
            server.close(close_connections=True)
            await server.wait_closed()

    def stop(self) -> None:
        assert self._loop is not None and self._stop is not None
        self._loop.call_soon_threadsafe(self._stop.set)
        # Generous timeout: shutdown is normally sub-second, but shared CI
        # runners are slow and this must not flake (the port must be free before
        # the replacement server binds it).
        self._thread.join(timeout=20)
        assert not self._thread.is_alive(), "fake Deepgram server did not stop"


class TestPipelineReconnectLive:
    def test_kill_and_resurrect_deepgram_midsession(self, monkeypatch):
        """P1-1 end-to-end: the fake Deepgram dies mid-session; a fresh one
        comes back on the same port; the pipeline reconnects, announces
        transcription_restored, and transcription flows again."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        # Short backoffs — the replacement server is already listening when
        # the first reconnect attempt fires.
        monkeypatch.setattr(
            audio_pipeline, "TRANSCRIBER_RECONNECT_BACKOFFS_S", (0.05, 0.1, 0.2)
        )

        server_a = ThreadedFakeDeepgramServer(FakeDeepgramServer())
        server_a.start()
        port = server_a.port

        _clear_overrides()
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MOCK_LLM_JSON
        app.state.llm_client = mock_llm
        app.state.transcriber_factory = lambda: DeepgramTranscriber(
            url=f"ws://127.0.0.1:{port}"
        )
        server_b = None
        frame = b"\x00\x00" * 800  # 100ms of PCM
        try:
            with TestClient(app).websocket_connect("/ws/session/4dc3f6f5-a32b-5ec8-89dc-c6a5ce746d22") as ws:
                # Prove the first connection is alive before killing it.
                ws.send_text(json.dumps({"type": "config"}))
                assert json.loads(ws.receive_text())["type"] == "config_ack"

                # KILL Deepgram mid-session…
                server_a.stop()
                # …and bring a fresh one back on the SAME port, primed with a
                # Results payload for the post-reconnect audio.
                revived = FakeDeepgramServer()
                revived.responses.append(_results_payload(
                    "Back from the dead.", start=0.0, duration=1.0, speaker=0,
                ))
                server_b = ThreadedFakeDeepgramServer(revived, port=port)
                server_b.start()

                # Keep streaming. The dead socket surfaces within a frame or
                # two (a TCP send can succeed into a closed peer once), then
                # the pipeline reconnects to the revived server.
                for _ in range(5):
                    ws.send_bytes(frame)
                    time.sleep(0.05)
                assert json.loads(ws.receive_text()) == {
                    "type": "transcription_restored"
                }

                # Post-restore audio reaches the revived server, whose queued
                # Results flows through to a real suggestion.
                for _ in range(10):
                    ws.send_bytes(frame)
                    time.sleep(0.02)
                suggestion = json.loads(ws.receive_text())
                assert suggestion["type"] == "suggestion"
                assert suggestion["utterance_text"] == "Back from the dead."
                assert revived.audio_frames  # audio really flowed to server B
        finally:
            _clear_overrides()
            if server_b is not None:
                server_b.stop()


# ---------------------------------------------------------------------------
# Live smoke test — only runs when a real key is configured
# ---------------------------------------------------------------------------

@pytest.mark.anyio
@pytest.mark.skipif(
    not os.getenv("DEEPGRAM_API_KEY"),
    reason="DEEPGRAM_API_KEY not set — live Deepgram smoke test skipped",
)
async def test_live_deepgram_smoke():
    """Stream one second of silence to the real Deepgram API — asserts the
    connection lifecycle works, without expecting any transcript from silence."""
    t = DeepgramTranscriber()
    await t.connect()
    try:
        silence = b"\x00\x00" * 1600  # 100ms of int16 silence @ 16 kHz
        for _ in range(10):
            segments = await t.stream(silence)
            assert isinstance(segments, list)
            await asyncio.sleep(0.05)
    finally:
        await t.close()
