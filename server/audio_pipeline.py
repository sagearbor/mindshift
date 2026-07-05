"""M2 real-time audio pipeline — WebSocket endpoint with credential-gated
transcription, diarization, and TTS.

Design note (honesty over mock data)
------------------------------------
The speech providers below are credential-gated. When their API keys are not
configured they report themselves *unavailable* and the pipeline says so
explicitly over the WebSocket — it never fabricates transcripts or audio that
could be mistaken for real output. With a ``DEEPGRAM_API_KEY`` present, the
pipeline streams raw PCM to Deepgram's live WebSocket API for transcription
(+ per-word diarization) and uses Deepgram Aura for TTS. The full
transcribe → diarize → suggest → speak flow is exercised in tests by injecting
test doubles via ``app.state`` (see ``tests/test_audio_pipeline.py``) and a
local fake Deepgram server (see ``tests/test_deepgram_live.py``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import ssl
import time
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlencode

import httpx
import websockets
from fastapi import WebSocket, WebSocketDisconnect

from llm_client import LLMClient
from models.audio import DiarizationConfig, SuggestionEvent, Utterance

logger = logging.getLogger(__name__)


def _tls_context_for(url: str) -> ssl.SSLContext | None:
    """Return a TLS context for ``wss://`` URLs, or ``None`` for plain ``ws://``.

    Deepgram is reached over ``wss://``, so certificate verification must
    succeed. Some Python installs (notably python.org builds on macOS) ship
    with an empty default CA store, which makes verification fail even though
    the key and network are fine. We anchor trust on the ``certifi`` bundle
    when it's importable so the app doesn't depend on each machine having its
    system CA store wired into Python; we fall back to the stdlib default
    otherwise. Plain ``ws://`` (the local fake server used in tests) needs no
    TLS, so we return ``None``.
    """
    if not url.lower().startswith("wss://"):
        return None
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # certifi absent or unreadable — use the stdlib default
        return ssl.create_default_context()


# Incoming client audio contract: raw PCM, int16 little-endian, mono, 16 kHz.
# The mobile client sends ~50-100ms binary WS frames of exactly this format,
# so the Deepgram connection is parameterised to match.
DEEPGRAM_SAMPLE_RATE = 16000
DEEPGRAM_LIVE_URL = "wss://api.deepgram.com/v1/listen"
DEEPGRAM_LIVE_PARAMS: dict[str, str] = {
    "model": "nova-3",
    "encoding": "linear16",
    "sample_rate": str(DEEPGRAM_SAMPLE_RATE),
    "channels": "1",
    "interim_results": "true",
    "smart_format": "true",
    "diarize": "true",
    "utterance_end_ms": "1200",
    "vad_events": "false",
}
# Deepgram kills idle live connections after ~10s (NET-0001); a KeepAlive
# every few seconds while no audio is flowing prevents that.
DEEPGRAM_KEEPALIVE_INTERVAL_S = 4.0
# On graceful finish(), how long to wait for Deepgram to flush its remaining
# Results/Metadata and close the socket before giving up.
DEEPGRAM_FINISH_TIMEOUT_S = 5.0

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_AURA_MODEL = "aura-2-thalia-en"

# ---------------------------------------------------------------------------
# Production guardrails (env-configurable; read once at import time, so a
# changed env var needs a process restart — tests monkeypatch the module
# attributes directly instead)
# ---------------------------------------------------------------------------

# P2-1: hard cap on concurrent live sessions. Each session holds a Deepgram
# socket, an LLM worker, and a TTS client — unbounded acceptance would let a
# burst of clients exhaust the process. Beyond the cap, new WebSockets are
# closed with 1013 ("try again later") instead of degrading everyone.
MAX_WS_SESSIONS = int(os.getenv("MAX_WS_SESSIONS", "100"))
_session_slots = asyncio.Semaphore(MAX_WS_SESSIONS)

# P2-1: per-session utterance budget. After this many utterances the client
# gets one {"type": "limit_reached"} and no further suggestions are generated
# (each one is an LLM + TTS spend). Generous: 500 utterances is hours of talk.
MAX_UTTERANCES = int(os.getenv("MAX_UTTERANCES", "500"))

# P2-3: the mobile client's contract is ~3200-byte PCM frames (~100ms). A
# frame vastly beyond that is a broken or hostile client — reject it honestly
# instead of forwarding it to the transcription backend.
MAX_AUDIO_FRAME_BYTES = int(os.getenv("MAX_AUDIO_FRAME_BYTES", str(64 * 1024)))

# P2-3: the configured role is interpolated into the LLM system prompt, so an
# unbounded value is both a token-cost and a prompt-injection surface.
MAX_ROLE_CHARS = 100

# P1-1: backoff schedule for mid-session transcriber reconnects. One attempt
# per entry; sleeps the entry's seconds before that attempt.
TRANSCRIBER_RECONNECT_BACKOFFS_S: tuple[float, ...] = (1.0, 2.0, 4.0)

# P1-8: bound on draining pending suggestions during a graceful stop. A hung
# LLM/TTS call must not hold the client's "stop" hostage for minutes.
STOP_DRAIN_TIMEOUT_S = 30.0

# P1-9: bounds for the per-session in-memory utterance buffer. When the
# buffer exceeds MAX, only the most recent KEEP entries are retained.
UTTERANCE_BUFFER_MAX = 1000
UTTERANCE_BUFFER_KEEP = 500


def _redact(text: str) -> str:
    """PII-safe stand-in for user speech in log lines (P1-4).

    This is a therapy product: transcript text must never sit in server logs.
    Length + a short digest keeps log lines correlatable (same utterance →
    same digest) without storing what was said.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"<utterance len={len(text)} sha256={digest}>"


class TranscriberUnavailable(RuntimeError):
    """Raised when a transcription backend is not configured/available.

    The pipeline catches this and reports ``transcription_unavailable`` to the
    client rather than inventing a transcript.
    """


class SuggestionUnavailable(RuntimeError):
    """Raised when no suggestion could be produced for an utterance (P0-3).

    Carries a short machine-readable ``reason`` — no transcript text, no
    provider error details — that is safe to forward to the client in a
    ``suggestion_error`` event. The pipeline reports the failure honestly
    rather than fabricating a suggestion line.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Transcript segments — the unit of finalized transcription output
# ---------------------------------------------------------------------------

@dataclass
class TranscriptSegment:
    """A finalized utterance segment with real timing/speaker metadata.

    ``speaker`` is Deepgram's per-word diarization speaker index (majority
    vote across the utterance's words), or ``None`` when the backend provided
    no diarization — the pipeline then falls back to the turn-alternation
    heuristic rather than inventing a speaker.
    """

    text: str
    start_time: float
    end_time: float
    speaker: int | None = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Deepgram transcriber (credential-gated, live WebSocket streaming)
# ---------------------------------------------------------------------------

class DeepgramTranscriber:
    """Real-time transcription via Deepgram's live streaming WebSocket API.

    Requires ``DEEPGRAM_API_KEY``; without it ``connect()`` raises
    :class:`TranscriberUnavailable` so the project's true state is never
    hidden behind fabricated transcripts. Connection or auth failures raise
    the same, with an honest reason.

    Speaks the raw Deepgram wire protocol (no SDK — the protocol is the
    stable surface): binary frames carry PCM audio out; JSON ``Results``
    messages come back. A background receive task accumulates ``is_final``
    result segments and emits one :class:`TranscriptSegment` per completed
    utterance (on ``speech_final`` or ``UtteranceEnd``) into a queue that
    :meth:`stream` drains. A background keepalive task sends
    ``{"type": "KeepAlive"}`` while no audio is flowing.
    """

    def __init__(
        self,
        url: str | None = None,
        keepalive_interval: float = DEEPGRAM_KEEPALIVE_INTERVAL_S,
    ) -> None:
        # URL injectable for tests (point at a local fake Deepgram server).
        self._base_url = url or os.getenv("DEEPGRAM_URL", "").strip() or DEEPGRAM_LIVE_URL
        self._keepalive_interval = keepalive_interval
        self._connected = False
        self._ws: websockets.ClientConnection | None = None
        self._segments: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._pending: list[dict] = []
        self._failure: str | None = None
        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._last_send = 0.0
        # Set once a graceful shutdown (finish/close) has been requested: the
        # receive loop then treats the socket closing as expected, not a failure.
        self._closing = False

    async def connect(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise TranscriberUnavailable(
                "DEEPGRAM_API_KEY not set — real-time transcription is disabled"
            )
        url = f"{self._base_url}?{urlencode(DEEPGRAM_LIVE_PARAMS)}"
        try:
            self._ws = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Token {api_key}"},
                open_timeout=10,
                ssl=_tls_context_for(url),
            )
        except Exception as exc:  # DNS failure, refused, 401/4xx handshake, timeout
            raise TranscriberUnavailable(
                f"Could not connect to Deepgram live transcription: {exc}"
            ) from exc
        self._connected = True
        self._failure = None
        self._last_send = time.monotonic()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        """Send *audio_bytes* to Deepgram; return any finalized segments so far.

        Returns an empty list while no utterance has been finalized yet.
        Raises :class:`TranscriberUnavailable` if the Deepgram socket has died
        — the pipeline reports that honestly instead of dropping audio silently.
        Segments that were already finalized before a failure are still
        delivered: the queue is drained first, and the recorded failure is only
        raised on a subsequent call once the queue is empty. Real transcripts
        must never be discarded just because the connection died afterwards.
        """
        if not self._connected or self._ws is None or self._failure:
            segments = self._drain_segments()
            if segments:
                return segments
            self._connected = False
            raise TranscriberUnavailable(self._failure or "Transcriber not connected")
        try:
            await self._ws.send(audio_bytes)
            self._last_send = time.monotonic()
        except Exception as exc:
            self._connected = False
            if not self._failure:
                self._failure = f"Deepgram connection lost: {exc}"
            segments = self._drain_segments()
            if segments:
                return segments
            raise TranscriberUnavailable(self._failure) from exc
        # Yield once so the receive task can process frames already on the wire.
        await asyncio.sleep(0)
        return self._drain_segments()

    def _drain_segments(self) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        while True:
            try:
                segments.append(self._segments.get_nowait())
            except asyncio.QueueEmpty:
                return segments

    async def finish(self) -> list[TranscriptSegment]:
        """Gracefully end the stream and return every remaining finalized segment.

        Sends ``Finalize`` (force-final any buffered interim) then
        ``CloseStream``, awaits the receive task's natural completion (Deepgram
        flushes its remaining ``Results`` + ``Metadata`` and closes the socket)
        with a hard timeout, then drains the queue. This is how the *last*
        utterance of a session gets delivered instead of dropped. Idempotent;
        after ``finish()``, :meth:`close` is a safe no-op. Never raises.
        """
        self._closing = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "Finalize"}))
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            if self._receive_task is not None:
                try:
                    await asyncio.wait_for(
                        self._receive_task, timeout=DEEPGRAM_FINISH_TIMEOUT_S
                    )
                except Exception:
                    # Timeout (wait_for cancels the task) or socket error — the
                    # drain below still returns whatever was finalized in time.
                    pass
                self._receive_task = None
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        self._connected = False
        return self._drain_segments()

    async def close(self) -> None:
        """Gracefully end the stream. Idempotent — never raises on double-close."""
        self._connected = False
        self._closing = True
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._keepalive_task
            self._keepalive_task = None
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            if self._receive_task is not None:
                self._receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._receive_task
                self._receive_task = None
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

    # -- background tasks ---------------------------------------------------

    async def _receive_loop(self) -> None:
        """Parse Deepgram messages; enqueue finalized utterance segments."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg_type = msg.get("type")
                if msg_type == "Results":
                    self._handle_results(msg)
                elif msg_type == "UtteranceEnd":
                    self._flush_pending()
        except Exception as exc:
            if self._closing:
                # Socket closed after we requested CloseStream — expected.
                return
            self._failure = f"Deepgram connection lost: {exc}"
            self._connected = False
        else:
            # Server closed the socket. After a requested CloseStream that is
            # the normal end of stream; otherwise (e.g. NET-0001 idle timeout)
            # it is a failure to report.
            if self._connected and not self._closing:
                self._failure = "Deepgram closed the connection"
                self._connected = False

    def _handle_results(self, msg: dict) -> None:
        try:
            alt = msg["channel"]["alternatives"][0]
        except (KeyError, IndexError, TypeError):
            return
        transcript = (alt.get("transcript") or "").strip()
        # Only trust finals: interim text (and interim speaker labels) are
        # unstable and must never surface as a completed utterance.
        if msg.get("is_final") and transcript:
            start = float(msg.get("start", 0.0))
            self._pending.append({
                "text": transcript,
                "start": start,
                "end": start + float(msg.get("duration", 0.0)),
                "confidence": float(alt.get("confidence", 1.0)),
                "speakers": [
                    w["speaker"] for w in alt.get("words", [])
                    if isinstance(w.get("speaker"), int)
                ],
            })
        if msg.get("speech_final"):
            self._flush_pending()

    def _flush_pending(self) -> None:
        """Assemble accumulated final segments into one completed utterance."""
        if not self._pending:
            return
        parts, self._pending = self._pending, []
        speakers = [s for p in parts for s in p["speakers"]]
        self._segments.put_nowait(TranscriptSegment(
            text=" ".join(p["text"] for p in parts),
            start_time=min(p["start"] for p in parts),
            end_time=max(p["end"] for p in parts),
            speaker=Counter(speakers).most_common(1)[0][0] if speakers else None,
            confidence=min(1.0, max(
                0.0, sum(p["confidence"] for p in parts) / len(parts)
            )),
        ))

    async def _keepalive_loop(self) -> None:
        """Send KeepAlive while idle so Deepgram doesn't drop the connection."""
        try:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                if self._ws is None:
                    return
                if time.monotonic() - self._last_send >= self._keepalive_interval:
                    await self._ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            # Socket died — the receive loop / next stream() reports it.
            return


# ---------------------------------------------------------------------------
# Speaker diarization (alternation heuristic)
# ---------------------------------------------------------------------------

class SpeakerDiarizer:
    """Assigns speaker labels by alternating across configured labels.

    This is an explicit placeholder heuristic, not acoustic diarization: it
    rotates through ``config.labels`` on each utterance. Real speaker
    separation (e.g. from Deepgram diarization or an embedding model) will
    replace this once transcription is wired to a live backend.
    """

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self.config = config or DiarizationConfig()
        self._turn_counter = 0

    def assign_speaker(self) -> str:
        label = self.config.labels[self._turn_counter % len(self.config.labels)]
        self._turn_counter += 1
        return label

    def reset(self) -> None:
        self._turn_counter = 0


def _generated_speaker_label(index: int) -> str:
    """Spreadsheet-style label for a diarized speaker index: 0→A … 25→Z, 26→AA."""
    letters = ""
    i = index
    while i >= 0:
        letters = chr(ord("A") + i % 26) + letters
        i = i // 26 - 1
    return f"Speaker {letters}"


class SpeakerLabelAssigner:
    """Maps a segment's diarization data to a stable per-session speaker label.

    Policy:

    * Deepgram speaker int → positional label from the diarizer's configured
      labels; indexes beyond the configured list get generated labels
      ("Speaker C", "Speaker D", …). Never modulo — distinct diarized speakers
      must never be merged into one label.
    * ``speaker is None`` after the session has seen a diarized speaker →
      attribute to the MOST RECENT diarized label. Continuation assumption: an
      un-diarized fragment (Deepgram omitted word-level speakers) most likely
      belongs to whoever was just talking — better than restarting an unrelated
      alternation sequence mid-conversation.
    * ``speaker is None`` and the session has NEVER seen a diarized speaker
      (legacy transcribers carry no speaker data at all) → the
      :class:`SpeakerDiarizer` alternation heuristic, exactly as before.
    """

    def __init__(self, diarizer) -> None:
        self._diarizer = diarizer
        self._last_diarized_label: str | None = None

    def label_for(self, speaker: int | None) -> str:
        if speaker is None:
            if self._last_diarized_label is not None:
                return self._last_diarized_label
            return self._diarizer.assign_speaker()
        labels = getattr(
            getattr(self._diarizer, "config", None), "labels", None,
        ) or DiarizationConfig().labels
        if speaker < len(labels):
            label = labels[speaker]
        else:
            label = _generated_speaker_label(speaker)
        self._last_diarized_label = label
        return label


# ---------------------------------------------------------------------------
# Text-to-speech (credential-gated)
# ---------------------------------------------------------------------------

class TTSClient:
    """Text-to-speech for earpiece output via Deepgram Aura.

    Requires ``DEEPGRAM_API_KEY``. When unconfigured — or when the request
    fails for any reason — ``synthesize`` returns ``None`` (no audio) rather
    than fabricating placeholder bytes; the suggestion still flows as
    on-screen text. Other TTS provider keys (``TTS_API_KEY``,
    ``ELEVENLABS_API_KEY``) are recognised but not implemented, and also
    honestly yield ``None``.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        speak_url: str | None = None,
    ) -> None:
        # Transport injectable for tests (httpx.MockTransport).
        self._transport = transport
        self._speak_url = speak_url or DEEPGRAM_SPEAK_URL
        # Lazily created once and reused across calls — a fresh AsyncClient per
        # synthesize() would redo the TCP+TLS handshake on every suggestion.
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=15.0)
        return self._client

    async def aclose(self) -> None:
        """Release the pooled HTTP connection. Idempotent — safe to call twice."""
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    async def synthesize(self, text: str) -> str | None:
        """Return base64-encoded audio (mp3) for *text*, or ``None`` if TTS is unavailable."""
        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            if os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"):
                logger.info(
                    "Non-Deepgram TTS key detected but only Deepgram Aura is "
                    "implemented — returning no audio"
                )
            else:
                logger.info("TTS unavailable (no DEEPGRAM_API_KEY) — returning no audio")
            return None
        try:
            resp = await self._get_client().post(
                self._speak_url,
                params={"model": DEEPGRAM_AURA_MODEL},
                headers={"Authorization": f"Token {api_key}"},
                json={"text": text},
            )
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode("ascii")
        except httpx.HTTPError as exc:
            logger.warning("Deepgram TTS request failed — returning no audio: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Session context (in-memory, per-connection)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    session_id: str
    empathy_slider: int = 50
    role: str = "Husband"
    utterances: list[Utterance] = field(default_factory=list)


def _remember_utterance(ctx: SessionContext, utterance: Utterance) -> None:
    """Append to the session's in-memory utterance buffer, bounded (P1-9).

    Nothing reads this buffer yet; it is kept (rather than removed) as the
    natural attachment point for a future in-session summary/context feature,
    but capped so an hour-long session cannot grow process memory without
    bound. Deliberately NOT persisted anywhere: whether live-session
    transcripts may be stored server-side at all is a flagged human/product
    decision, and this module must not pre-empt it.
    """
    ctx.utterances.append(utterance)
    if len(ctx.utterances) > UTTERANCE_BUFFER_MAX:
        # Keep only the most recent entries; older ones are dropped.
        del ctx.utterances[:-UTTERANCE_BUFFER_KEEP]


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def _finish_transcriber(transcriber) -> list[TranscriptSegment]:
    """Flush a transcriber that supports graceful ``finish()``.

    Legacy/test transcribers without ``finish()`` buffer nothing between
    ``stream()`` calls, so there is nothing to flush for them.
    """
    finish = getattr(transcriber, "finish", None)
    if finish is None:
        return []
    return await finish()


async def audio_ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Handle a single audio-streaming WebSocket connection.

    Protocol
    --------
    Client → Server (binary):  raw audio chunks
    Client → Server (text):    JSON control messages, e.g.
        {"type": "config", "empathy_slider": 75, "role": "Husband"}
        {"type": "stop"} — graceful end-of-session: the server flushes the
        transcriber, delivers every remaining ``SuggestionEvent`` (bounded by
        ``STOP_DRAIN_TIMEOUT_S``), then sends {"type": "session_complete"}
        and closes the socket with code 1000.
    Server → Client (text):    JSON events (the mobile client tolerates
    unknown types, so these are additive):
        {"type": "suggestion", ...}            — SuggestionEvent per utterance
        {"type": "suggestion_error",
         "utterance_text": ..., "reason": ...} — an utterance produced NO
                                                 suggestion (LLM/TTS failure);
                                                 reported, never fabricated
        {"type": "transcription_unavailable", "reason": ...}
        {"type": "transcription_restored"}     — a mid-session transcriber
                                                 drop was recovered (P1-1)
        {"type": "limit_reached"}              — per-session utterance budget
                                                 exhausted; no more suggestions
        {"type": "session_complete"}           — plus "pending_dropped": n when
                                                 the stop drain timed out
        {"error": ...}                         — malformed client input

    New WebSockets beyond ``MAX_WS_SESSIONS`` concurrent sessions are closed
    immediately with code 1013 ("try again later") — an honest rejection
    instead of letting every session degrade (P2-1).
    """
    await websocket.accept()

    if _session_slots.locked():
        # At capacity. Suppress send failures — the client may vanish mid-close.
        logger.info(
            "Rejecting session %s: %d concurrent sessions already active",
            session_id, MAX_WS_SESSIONS,
        )
        with contextlib.suppress(Exception):
            await websocket.close(
                code=1013, reason="server at session capacity — try again later"
            )
        return

    # NOTE: locked()+acquire() is not atomic; under a connection race an extra
    # session may briefly WAIT here for a slot instead of being rejected.
    # That's benign — the cap on concurrently *running* sessions still holds.
    await _session_slots.acquire()
    try:
        await _run_session(websocket, session_id)
    finally:
        _session_slots.release()


async def _run_session(websocket: WebSocket, session_id: str) -> None:
    """Body of one accepted, slot-holding audio session (see endpoint above)."""
    # Per-connection state
    ctx = SessionContext(session_id=session_id)

    # Resolve providers from app.state (tests inject doubles here), falling
    # back to the real, credential-gated implementations.
    state = websocket.app.state
    transcriber_factory = getattr(state, "transcriber_factory", None) or DeepgramTranscriber
    diarizer_factory = getattr(state, "diarizer_factory", None) or SpeakerDiarizer
    injected_tts = getattr(state, "tts_client", None)
    tts = injected_tts or TTSClient()
    owns_tts = injected_tts is None  # only close a client we created ourselves
    llm_client: LLMClient = state.llm_client

    transcriber = transcriber_factory()
    diarizer = diarizer_factory()
    labeler = SpeakerLabelAssigner(diarizer)

    # The receive loop (acks/errors) and the suggestion worker both send on
    # this socket — serialize so frames never interleave mid-send.
    send_lock = asyncio.Lock()

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(payload))

    # Suggestion generation (LLM via thread + TTS HTTP, up to ~15s) runs on a
    # single background worker so it never stalls the audio receive loop —
    # audio keeps flowing to Deepgram while a suggestion is being generated.
    # One worker (not a pool) keeps SuggestionEvents in utterance order.
    # Queue items carry the (empathy_slider, role) snapshot at enqueue time.
    suggestion_queue: asyncio.Queue[tuple[TranscriptSegment, int, str]] = asyncio.Queue()
    # enqueued-vs-finished counters let the stop handler report honestly how
    # many suggestions were dropped when draining times out (P1-8) — qsize()
    # alone would miss the item the worker is currently processing.
    queue_stats = {"enqueued": 0, "finished": 0}

    async def process_segment(
        segment: TranscriptSegment, empathy_slider: int, role: str,
    ) -> None:
        speaker = labeler.label_for(segment.speaker)

        utterance = Utterance(
            session_id=session_id,
            speaker=speaker,
            text=segment.text,
            start_time=segment.start_time,
            end_time=segment.end_time,
            confidence=segment.confidence,
        )
        _remember_utterance(ctx, utterance)

        # Generate suggestion via LLM
        suggestion_texts = await _generate_suggestions(
            llm_client, utterance, empathy_slider, role,
        )

        # TTS for first suggestion
        tts_audio = (
            await tts.synthesize(suggestion_texts[0])
            if suggestion_texts else None
        )

        event = SuggestionEvent(
            session_id=session_id,
            utterance_text=segment.text,
            speaker=speaker,
            suggestions=suggestion_texts,
            empathy_slider=empathy_slider,
            audio_b64=tts_audio,
        )
        async with send_lock:
            await websocket.send_text(event.model_dump_json())

    async def suggestion_worker() -> None:
        while True:
            segment, empathy_slider, role = await suggestion_queue.get()
            try:
                await process_segment(segment, empathy_slider, role)
            except Exception as exc:
                # P0-2: a failed suggestion (LLM/TTS error, missing key, rate
                # limit) must never be silently swallowed — the client is told
                # WHICH utterance produced nothing and why. Only the exception
                # class name goes over the wire: provider error messages can
                # carry key fragments or internals. task_done() must still run
                # or queue.join() deadlocks.
                logger.warning(
                    "Suggestion processing failed for session %s (%s)",
                    session_id, _redact(segment.text), exc_info=True,
                )
                reason = (
                    exc.reason if isinstance(exc, SuggestionUnavailable)
                    else type(exc).__name__
                )
                # Suppress send failures — the socket may already be gone.
                with contextlib.suppress(Exception):
                    await send_json({
                        "type": "suggestion_error",
                        "utterance_text": segment.text,
                        "reason": reason,
                    })
            finally:
                queue_stats["finished"] += 1
                suggestion_queue.task_done()

    limit_notified = False

    async def enqueue_segments(result) -> None:
        nonlocal limit_notified
        for segment in _normalize_segments(result):
            if queue_stats["enqueued"] >= MAX_UTTERANCES:
                # P2-1: utterance budget exhausted — every suggestion is an
                # LLM + TTS spend. Say so ONCE, then drop further segments
                # (transcription itself keeps running; only suggestion
                # generation stops).
                if not limit_notified:
                    limit_notified = True
                    logger.info(
                        "Session %s reached MAX_UTTERANCES=%d — no further "
                        "suggestions will be generated", session_id, MAX_UTTERANCES,
                    )
                    with contextlib.suppress(Exception):
                        await send_json({"type": "limit_reached"})
                return
            queue_stats["enqueued"] += 1
            suggestion_queue.put_nowait((segment, ctx.empathy_slider, ctx.role))

    worker_task = asyncio.create_task(suggestion_worker())
    # P1-7: this try must start IMMEDIATELY after the worker task is created.
    # The initial connect + notify below can raise (e.g. the client
    # disconnects before we get a word in); without the try/finally around
    # them, each such occurrence would leak one forever-pending worker task.
    try:
        # Connect transcription; if unavailable, tell the client plainly
        # instead of fabricating transcripts. The client is told ONCE on
        # entering the unavailable state; further binary frames are then
        # ignored silently (the phone streams ~10 frames/sec — re-sending per
        # frame is a flood).
        transcription_available = True
        # P1-1: only a transcriber that WAS live is worth reconnecting. An
        # initial connect failure is a config problem (no API key, missing
        # package, bad URL) that retrying cannot fix.
        transcriber_connected_once = False
        try:
            await transcriber.connect()
            transcriber_connected_once = True
        except TranscriberUnavailable as exc:
            transcription_available = False
            with contextlib.suppress(Exception):  # client may already be gone
                await send_json(
                    {"type": "transcription_unavailable", "reason": str(exc)}
                )
            logger.info(
                "Transcription unavailable for session %s: %s", session_id, exc
            )

        async def reconnect_transcriber():
            """P1-1: try to bring transcription back after a mid-session drop.

            Deepgram dropping a live socket (idle timeout, network blip,
            upstream restart) must not silently kill transcription for the
            rest of a possibly hour-long session. Tries one fresh transcriber
            per backoff entry; returns the connected replacement, or ``None``
            when every attempt failed. Runs inline in the receive loop, so
            client frames buffer in the socket while retrying — the audio sent
            during the outage is lost either way (the backend was down), and
            the client hears about it honestly if retries exhaust.
            """
            attempts = len(TRANSCRIBER_RECONNECT_BACKOFFS_S)
            for attempt, delay in enumerate(TRANSCRIBER_RECONNECT_BACKOFFS_S, 1):
                await asyncio.sleep(delay)
                candidate = transcriber_factory()
                try:
                    await candidate.connect()
                except TranscriberUnavailable as exc:
                    logger.info(
                        "Transcriber reconnect %d/%d failed for session %s: %s",
                        attempt, attempts, session_id, exc,
                    )
                    continue
                logger.info(
                    "Transcriber reconnected for session %s (attempt %d/%d)",
                    session_id, attempt, attempts,
                )
                return candidate
            return None

        while True:
            message = await websocket.receive()

            # --- Disconnect ---
            if message.get("type") == "websocket.disconnect":
                break

            # --- Binary audio chunk ---
            if "bytes" in message and message["bytes"] is not None:
                audio_bytes: bytes = message["bytes"]
                if len(audio_bytes) == 0 or not transcription_available:
                    continue
                if len(audio_bytes) > MAX_AUDIO_FRAME_BYTES:
                    # P2-3: contract frames are ~3200 bytes — reject the
                    # anomaly honestly rather than forwarding it upstream.
                    await send_json({
                        "error": (
                            f"audio frame too large ({len(audio_bytes)} bytes; "
                            f"max {MAX_AUDIO_FRAME_BYTES})"
                        )
                    })
                    continue

                try:
                    result = await transcriber.stream(audio_bytes)
                except TranscriberUnavailable as exc:
                    if transcriber_connected_once:
                        # A previously-live backend dropped mid-session —
                        # attempt recovery before declaring it dead (P1-1).
                        with contextlib.suppress(Exception):
                            await transcriber.close()
                        replacement = await reconnect_transcriber()
                        if replacement is not None:
                            transcriber = replacement
                            await send_json({"type": "transcription_restored"})
                            # The frame that hit the dead socket is gone — its
                            # audio cannot be recovered, only acknowledged.
                            continue
                    transcription_available = False
                    await send_json(
                        {"type": "transcription_unavailable", "reason": str(exc)}
                    )
                    continue
                await enqueue_segments(result)

            # --- Text control message ---
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_json({"error": "invalid JSON"})
                    continue

                msg_type = payload.get("type")
                if msg_type == "config":
                    if "empathy_slider" in payload:
                        val = payload["empathy_slider"]
                        if isinstance(val, int) and 0 <= val <= 100:
                            ctx.empathy_slider = val
                    if "role" in payload:
                        role_val = payload["role"]
                        # P2-3: role reaches the LLM system prompt — ignore
                        # wrong-typed values, clamp the length (cost +
                        # injection surface).
                        if isinstance(role_val, str):
                            ctx.role = role_val[:MAX_ROLE_CHARS]
                    await send_json({"type": "config_ack"})
                elif msg_type == "stop":
                    # Graceful stop: flush the transcriber so the FINAL
                    # utterance is delivered, wait (bounded — P1-8) for the
                    # pending SuggestionEvents to go out, then confirm
                    # completion and close server-side.
                    await enqueue_segments(await _finish_transcriber(transcriber))
                    completion: dict = {"type": "session_complete"}
                    try:
                        await asyncio.wait_for(
                            suggestion_queue.join(), timeout=STOP_DRAIN_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        # A hung LLM/TTS call must not stall the client's stop
                        # for minutes — close out, honestly reporting how many
                        # suggestions never made it.
                        pending = queue_stats["enqueued"] - queue_stats["finished"]
                        completion["pending_dropped"] = pending
                        logger.warning(
                            "Graceful stop drain timed out after %.0fs with %d "
                            "pending suggestion(s) for session %s",
                            STOP_DRAIN_TIMEOUT_S, pending, session_id,
                        )
                    await send_json(completion)
                    await websocket.close(code=1000)
                    break
                else:
                    await send_json({"error": f"unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    finally:
        # Cleanup must never raise, whatever state the connection died in.
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task
        try:
            # Abrupt disconnect (no "stop"): still finish() so Deepgram closes
            # cleanly, but the client is gone — drained segments are discarded.
            # After a graceful stop this is an idempotent no-op.
            discarded = await _finish_transcriber(transcriber)
            if discarded:
                logger.debug(
                    "Discarding %d transcript segment(s) drained after "
                    "disconnect from session %s", len(discarded), session_id,
                )
            await transcriber.close()
        except Exception:
            logger.debug(
                "Transcriber cleanup failed for session %s", session_id,
                exc_info=True,
            )
        if owns_tts and hasattr(tts, "aclose"):  # hasattr: tolerate doubles
            with contextlib.suppress(Exception):
                await tts.aclose()


# ---------------------------------------------------------------------------
# stream() result normalization
# ---------------------------------------------------------------------------

def _normalize_segments(
    result: list[TranscriptSegment] | str | None,
) -> list[TranscriptSegment]:
    """Normalize a transcriber's ``stream()`` result to a segment list.

    The real :class:`DeepgramTranscriber` returns ``list[TranscriptSegment]``
    (with genuine timings/speaker data). Legacy/test transcribers may return a
    plain ``str`` (one utterance, no timing — kept at 0.0/0.0 rather than
    fabricating a duration) or ``None`` (nothing finalized yet).
    """
    if result is None:
        return []
    if isinstance(result, str):
        if not result.strip():
            return []
        return [TranscriptSegment(text=result, start_time=0.0, end_time=0.0)]
    return list(result)


# ---------------------------------------------------------------------------
# LLM suggestion helper
# ---------------------------------------------------------------------------

async def _generate_suggestions(
    llm: LLMClient,
    utterance: Utterance,
    empathy_slider: int,
    role: str,
) -> list[str]:
    """Call LLMClient.complete() and parse suggestions from the response.

    Raises :class:`SuggestionUnavailable` ("llm_parse_error") when the LLM
    output cannot be parsed (P0-3). It used to fabricate an
    "I hear you — <utterance>" line here — a fake suggestion that would then
    be TTS-spoken as if the coach really produced it. Honest failure instead:
    the suggestion worker turns the exception into a ``suggestion_error``
    event so the client knows this utterance yielded nothing.
    """
    from main import empathy_system_prompt, parse_llm_json

    system = empathy_system_prompt(empathy_slider, role)
    user_content = f'Transcript turn: "{utterance.text}"'

    raw = await asyncio.to_thread(llm.complete, system=system, user=user_content)

    try:
        data = parse_llm_json(raw)
        suggestions = data.get("suggestions", [])
    # AttributeError/TypeError: valid JSON of the wrong shape (list, string…).
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as exc:
        # P1-4: log length/hash, never the transcript text itself.
        logger.warning(
            "LLM returned unparseable response for %s", _redact(utterance.text)
        )
        raise SuggestionUnavailable("llm_parse_error") from exc
    if not isinstance(suggestions, list):
        logger.warning(
            "LLM returned non-list suggestions for %s", _redact(utterance.text)
        )
        raise SuggestionUnavailable("llm_parse_error")
    return suggestions
