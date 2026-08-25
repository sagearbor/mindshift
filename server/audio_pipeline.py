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
import hmac
import inspect
import json
import logging
import math
import os
import re
import ssl
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlencode

import httpx
import numpy as np
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

import calls
import llm_client
from llm_client import LLMClient
from models.audio import (
    DiarizationConfig,
    SpeakerIdentityEvent,
    SuggestionEvent,
    ToneFlagEvent,
    TranscriptEvent,
    TurnLocalEvent,
    Utterance,
)

# Optional enrichment modules for local-first (phone-orchestrated) sessions —
# see the "turn_local" section of audio_ws_endpoint's docstring. Each is
# guarded so this module imports (and every legacy code path runs) on a
# checkout or deployment that lacks it: audio tone (Foundation C,
# server/tone_id.py — the model deps themselves are a further optional
# install), voiceprint identity (server/speaker_id.py, torch optional) and
# the watch relay (Track 1, server/watch/relay.py). Tests swap these module
# attributes for fakes; production leaves them alone.
try:
    import tone_id
except ImportError:  # pragma: no cover — depends on which foundations landed
    tone_id = None
try:
    import speaker_id
except ImportError:  # pragma: no cover
    speaker_id = None
try:
    from watch import relay as watch_relay
except ImportError:  # pragma: no cover
    watch_relay = None

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

# Voice-profile ids from the config message only feed a DB lookup, but bound
# their length as defence in depth against absurd input.
MAX_ID_CHARS = 200

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

# Track 3-server: latency instrumentation. The last N per-stage timings are
# kept per session (a deque each) so the stop handler can report p50/p95
# without the memory growing with an hour-long session. 200 utterances is
# well past a whole conversation's worth of coaching turns.
LATENCY_WINDOW = 200

# Track 3-server: the per-session PCM ring buffer that lets a turn the PHONE
# finalized (turn_local, reported by session-relative time) be recovered as
# audio for server-side enrichment (tone / identity). 90 s at 16 kHz int16 is
# ~2.9 MB per session — bounded, and comfortably longer than the phone's
# on-device pipeline lag plus the longest plausible turn.
PCM_RING_SECONDS = float(os.getenv("PCM_RING_SECONDS", "90"))

# Track 3-server: a Deepgram segment whose midpoint falls inside a
# locally-handled turn's [start, end] (padded by this much on both sides) is
# a duplicate of what the phone already showed and is dropped. The pad
# absorbs the small disagreement between Deepgram's endpointing and the
# phone's VAD about exactly where a turn begins/ends.
LOCAL_RANGE_PAD_S = 0.25
# Bound on remembered locally-handled ranges (oldest dropped). Deepgram never
# finalizes minutes late, so 500 turns of history is far more than enough.
LOCAL_RANGES_MAX = 500

# Track 3-server: on a graceful stop, how long to wait for in-flight
# enrichment tasks (tone / identity / relay for the last turn) before
# sending session_complete without them. Short — enrichment is best-effort.
ENRICHMENT_DRAIN_TIMEOUT_S = 5.0

# Review 2026-08-24: enrichment is spawned per turn_local BEFORE (and
# independent of) the utterance budget, and each task is a voiceprint read
# plus one or two CPU model passes. Unbounded, a client sending turn_local
# frames as fast as the socket carries them could pile up thousands of
# tasks (and store reads) per session. Beyond this many IN-FLIGHT tasks,
# further turns are simply not enriched (the cloud suggestion is unaffected;
# it has its own budget + latest-wins). Real phones report one turn every
# few seconds and a task finishes in well under that, so 4 is never reached
# in honest use.
MAX_ENRICHMENT_INFLIGHT = int(os.getenv("MAX_ENRICHMENT_INFLIGHT", "4"))

# Review 2026-08-24: the account's enrolled voiceprint documents are read
# ONCE per session (then refreshed at most this often) instead of on every
# turn — a list_blobs + N downloads against GCS per turn was pure cost, and
# an enrollment made mid-conversation is still picked up within a minute.
VOICEPRINT_CACHE_TTL_S = float(os.getenv("VOICEPRINT_CACHE_TTL_S", "60"))

# Cloud-suggestion latency knobs (perf/cloud-suggestion-latency). Each is
# env-overridable so a regression can be rolled back with a config change,
# and so scripts/bench_suggestions.py can A/B the legacy behaviour in-process.
#
# Output budget: a suggestion turn is 3 sentences + one integer (~80 output
# tokens with the live prompt; ~130 with the legacy tone_score contract), a
# nudge is ≤6 words + one integer. The caps bound the worst case (a model
# that starts rambling) without ever cutting a normal answer short.
SUGGESTION_MAX_TOKENS = int(os.getenv("MINDSHIFT_SUGGESTION_MAX_TOKENS", "200"))
NUDGE_MAX_TOKENS = int(os.getenv("MINDSHIFT_NUDGE_MAX_TOKENS", "60"))
# The lean live output contract (no tone_score, suggestions first, bounded
# length) — see main.empathy_system_prompt(live=True). "0" restores the REST
# contract for the WS path (the pre-perf behaviour).
LIVE_PROMPT = os.getenv("MINDSHIFT_LIVE_PROMPT", "1") != "0"
# When the model's answer is not parseable JSON, ask it once (tiny prompt,
# temperature 0) to return the same content as valid JSON before giving up
# with llm_parse_error. "0" restores fail-on-first-parse-error.
PARSE_REPAIR = os.getenv("MINDSHIFT_PARSE_REPAIR", "1") != "0"
REPAIR_MAX_TOKENS = 200
# How many suggestion jobs may be in the LLM at once for a LOCAL-FIRST
# session (a legacy client always gets exactly one worker — its server TTS
# audio must play in strict order). Final events stay in utterance order
# regardless (see the ordering chain in _run_session); what concurrency
# buys is that a turn arriving while the previous one is still generating
# starts its own LLM call immediately instead of waiting (queue_wait).
LOCAL_FIRST_CONCURRENCY = max(1, int(os.getenv("MINDSHIFT_LOCAL_FIRST_CONCURRENCY", "2")))

# P2-7: server-generated session/relationship ids are UUIDs. Path params that
# reach routing and (for session_id) the export ``Content-Disposition``
# filename header must be validated — a CR/LF/quote in a free-form id could
# malform that header. Lenient across UUID versions: we only assert the
# canonical 8-4-4-4-12 hex shape. Shared with main.py's REST ``Path(pattern=)``.
UUID_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UUID_RE = re.compile(UUID_PATTERN)


def _is_valid_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


# A live WebSocket session id is client-chosen — the mobile app sends
# "live-<timestamp>". It is only used as a short log/session key, so it needs a
# safe, bounded shape rather than a full UUID (requiring a UUID here wrongly
# rejected the real app). Conservative charset + length caps injection/abuse.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _is_valid_session_id(value: str) -> bool:
    return bool(_SESSION_ID_RE.match(value))


# Side-aware coaching: the coached user identifies themself by their diarized
# label ("Speaker A"/"Speaker B", or a generated "Speaker AA" past Z — see
# _generated_speaker_label). We only accept exactly that shape from config so a
# malformed value can never mis-type turns; JSON null resets it (see
# _apply_config). Mirrors the "Speaker <letters>" labels the labeler emits.
_SELF_SPEAKER_RE = re.compile(r"^Speaker [A-Z]{1,2}$")


# P0-1: WebSockets bypass CORS / the same-origin policy — any web page in any
# browser can open ws://.../ws/session/x, stream audio, and burn the owner's
# Deepgram + Anthropic credits. Native mobile apps send NO Origin header;
# browsers always do. Policy: allow a missing Origin (native clients); for a
# PRESENT Origin, require it to be in this allowlist. Default empty = reject
# every browser origin. Read once at import time (tests monkeypatch this
# module attribute directly, like the other guardrails above). The concrete
# allowlist is a flagged human/ops decision — set MINDSHIFT_ALLOWED_ORIGINS.
ALLOWED_ORIGINS: frozenset[str] = frozenset(
    o.strip()
    for o in os.getenv("MINDSHIFT_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
)


def _origin_allowed(origin: str | None, host: str | None = None) -> bool:
    """Whether a WS connection carrying this Origin header may proceed (P0-1).

    Allowed when:
      * there is no Origin (some native clients omit it), OR
      * the Origin is explicitly in :data:`ALLOWED_ORIGINS` (the web app), OR
      * the Origin is *same-origin* with the server (``host`` matches). React
        Native's WebSocket defaults its Origin to the target server's URL, so
        the mobile app arrives same-origin — which is not a cross-site request
        and is safe. The real gate is the id-token auth in the config frame.

    Any other present Origin is a cross-site browser connection and is rejected
    before ``accept()``.
    """
    if origin is None:
        return True
    if origin in ALLOWED_ORIGINS:
        return True
    # Same-origin: strip the scheme and compare the host[:port] to the request's
    # own Host header. Same-origin can't be a cross-site attacker.
    if host is not None and origin.split("://", 1)[-1] == host:
        return True
    return False


# Per-process random salt: makes the redaction digest a keyed HMAC so short or
# guessable utterances ("no", a name, "I want a divorce") can't be confirmed
# from logs by hashing candidates. Correlation holds within one process
# lifetime; confirmability across the dictionary does not. Regenerated each boot.
_REDACT_SALT = os.urandom(32)


def _redact(text: str) -> str:
    """PII-safe stand-in for user speech in log lines (P1-4).

    This is a therapy product: transcript text must never sit in server logs.
    A coarse length bucket + a salted HMAC digest keeps log lines correlatable
    (same utterance → same digest, this process) without storing what was said
    and without being a dictionary-attack oracle on short phrases.
    """
    digest = hmac.new(_REDACT_SALT, text.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    # Bucket the length so a 3-char utterance isn't advertised as exactly len=3.
    bucket = "0" if not text else f"~{max(8, (len(text) + 7) // 8 * 8)}"
    return f"<utterance len={bucket} hmac={digest}>"


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
# Latency instrumentation (Track 3-server)
# ---------------------------------------------------------------------------
#
# "Measure everything" is the precondition for the phone-orchestrated
# (local-first) path: we need a baseline of where the seconds go in today's
# cloud path — Deepgram endpointing, queue wait, LLM, TTS, send — before we
# can claim the on-device path beats it. Stamps are ``time.monotonic()``
# (never wall-clock: NTP steps would corrupt a stage), taken through the
# existing hot path at the same places for both paths so the numbers are
# comparable. One structured INFO line per utterance; the session keeps the
# last LATENCY_WINDOW samples per stage for the p50/p95 stop summary.

@dataclass
class UtteranceTiming:
    """Monotonic stamps for one utterance's trip through the hot path.

    Every stamp is ``None`` until that point is reached, so a stage whose
    endpoints were never both hit (no TTS because the phone speaks; a nudge
    that stayed silent) is simply absent from the report — never a fake 0.
    """

    frame_received: float | None = None    # the WS frame that finalized the turn arrived
    segment_finalized: float | None = None  # transcriber returned the segment / turn_local parsed
    enqueued: float | None = None           # handed to the suggestion worker
    llm_start: float | None = None
    llm_first_partial: float | None = None  # first suggestion string complete (streaming)
    llm_end: float | None = None
    tts_start: float | None = None
    tts_end: float | None = None
    sent: float | None = None               # SuggestionEvent on the wire (or decided: nothing to send)
    queue_depth: int = 0                    # items ahead of this one at enqueue time
    # Hedged streaming (perf/llm-hedging): None = the LLM stage was not a
    # hedge-capable streaming call (legacy complete() path, or a test double);
    # otherwise whether a second request was fired, and whether it was the
    # one whose answer was used. Logged and counted, never a duration.
    hedged: bool | None = None
    hedge_won: bool = False

    # (stage name, start stamp, end stamp) — in the order they are logged.
    _STAGES = (
        ("seg_to_enqueue", "segment_finalized", "enqueued"),
        ("queue_wait", "enqueued", "llm_start"),
        ("llm", "llm_start", "llm_end"),
        ("llm_first_partial", "llm_start", "llm_first_partial"),
        ("tts", "tts_start", "tts_end"),
        ("total", "frame_received", "sent"),
    )

    def stage_ms(self) -> dict[str, float]:
        """Per-stage durations in milliseconds, only for stages fully stamped."""
        out: dict[str, float] = {}
        for name, a, b in self._STAGES:
            start, end = getattr(self, a), getattr(self, b)
            if start is not None and end is not None:
                out[name] = round((end - start) * 1000.0, 1)
        return out


class LatencyRecorder:
    """Per-session collector of :class:`UtteranceTiming` results.

    ``clock`` is injectable (tests pass a fake monotonic clock via
    ``app.state.monotonic_clock``) so the stage arithmetic is testable
    exactly, not just "some positive number".
    """

    STAGES = tuple(name for name, _a, _b in UtteranceTiming._STAGES)

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window: int = LATENCY_WINDOW,
    ) -> None:
        self.clock = clock
        self._samples: dict[str, deque[float]] = {
            stage: deque(maxlen=window) for stage in self.STAGES
        }
        self.count = 0
        # Hedged-streaming counters over the WHOLE session (not windowed —
        # they are the cost side of the tail fix and must add up): n =
        # hedge-capable LLM streaming calls, hedged / hedge_won as on
        # UtteranceTiming, slow_llm = turns abandoned at the first-token
        # deadline (those never reach record(); see record_abandoned()).
        self.hedge = {"n": 0, "hedged": 0, "hedge_won": 0, "slow_llm": 0}

    def now(self) -> float:
        return self.clock()

    def record_abandoned(self, reason: str, session_id: str) -> None:
        """Count a turn whose LLM stage was abandoned (no durations to fold —
        the stage never completed). Only ``slow_llm`` is counted today."""
        if reason == "slow_llm":
            self.hedge["slow_llm"] += 1
            logger.info("latency session=%s abandoned=slow_llm", session_id)

    def start(
        self,
        frame_received: float | None = None,
        segment_finalized: float | None = None,
    ) -> UtteranceTiming:
        return UtteranceTiming(
            frame_received=frame_received, segment_finalized=segment_finalized,
        )

    def record(self, timing: UtteranceTiming, session_id: str) -> dict[str, float]:
        """Fold one finished utterance into the window and log its line.

        The log line is the per-utterance product of this instrumentation:
        one INFO record, fixed field order, ``-`` for a stage that didn't
        happen, so Cloud Run log queries can grep/aggregate it. No transcript
        text (P1-4) — only durations and the session id.
        """
        stages = timing.stage_ms()
        for name, value in stages.items():
            self._samples[name].append(value)
        self.count += 1
        if timing.hedged is not None:
            self.hedge["n"] += 1
            self.hedge["hedged"] += int(timing.hedged)
            self.hedge["hedge_won"] += int(timing.hedge_won)
        logger.info(
            "latency session=%s seg_to_enqueue=%s queue_wait=%s llm=%s "
            "llm_first_partial=%s tts=%s total=%s queue_depth=%d "
            "hedged=%s hedge_won=%s",
            session_id,
            *(
                f"{stages[name]:.1f}ms" if name in stages else "-"
                for name in self.STAGES
            ),
            timing.queue_depth,
            "-" if timing.hedged is None else int(timing.hedged),
            "-" if timing.hedged is None else int(timing.hedge_won),
        )
        return stages

    def summary(self) -> dict[str, dict[str, float | int]]:
        """``{stage: {"p50": ms, "p95": ms, "n": count}}`` over the window,
        omitting stages with no samples. Nearest-rank percentiles — exact for
        small n (no interpolation inventing a value nobody measured).

        Plus one non-stage entry, ``"hedge"`` (``{"n", "hedged",
        "hedge_won", "slow_llm"}`` — whole-session counts, see ``self.hedge``)
        whenever at least one hedge-capable LLM call happened or a turn was
        abandoned as ``slow_llm``; absent otherwise, so a legacy session's
        summary is exactly what it was."""
        out: dict[str, dict[str, float | int]] = {}
        for stage, samples in self._samples.items():
            if not samples:
                continue
            ordered = sorted(samples)
            out[stage] = {
                "p50": _nearest_rank(ordered, 50),
                "p95": _nearest_rank(ordered, 95),
                "n": len(ordered),
            }
        if self.hedge["n"] or self.hedge["slow_llm"]:
            out["hedge"] = dict(self.hedge)
        return out


def _nearest_rank(ordered: list[float], percentile: int) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


# ---------------------------------------------------------------------------
# PCM ring buffer (Track 3-server)
# ---------------------------------------------------------------------------

class PcmRingBuffer:
    """The last ~``seconds`` of the client's raw PCM, addressable by
    session-relative time, so a phone-finalized turn can be recovered as
    audio for server-side enrichment.

    Timeline contract — t = 0 is the FIRST audio frame of the session and
    time advances by AUDIO RECEIVED (``bytes / (2 * sample_rate)``), not by
    wall clock. Three consequences the phone must honour:

    * ``turn_local.start_time/end_time`` must be stamped on the phone's own
      capture timeline (seconds of audio captured since its first streamed
      frame). That timeline and this one advance in lockstep, so there is no
      drift as long as every captured frame is also streamed.
    * A frame the phone captured but never sent (network drop) shifts every
      later server-side time EARLIER by that frame's duration. The pad on
      overlap suppression (LOCAL_RANGE_PAD_S) absorbs a frame or two; a
      sustained loss makes the recovered slice wrong, which is why every
      consumer of a slice is best-effort and a clearly-wrong slice degrades
      to "no enrichment", never to a confident wrong label.
    * Deepgram's ``start``/``duration`` are on the same audio timeline (its
      clock is the bytes we forwarded), which is what makes the overlap check
      between Deepgram segments and turn_local ranges meaningful at all.

    Frames are the endpoint's contract: int16 little-endian mono 16 kHz.
    Stored as raw bytes (no numpy per frame — the receive loop is hot);
    converted to float32 only when a slice is actually consumed.
    """

    def __init__(
        self,
        seconds: float = PCM_RING_SECONDS,
        sample_rate: int = DEEPGRAM_SAMPLE_RATE,
    ) -> None:
        self.sample_rate = sample_rate
        self._capacity_bytes = int(seconds * sample_rate) * 2
        self._buf = bytearray()
        # Bytes trimmed off the front so far: the session-timeline offset of
        # ``_buf[0]``. Always even so a slice can never start mid-sample.
        self._dropped_bytes = 0
        self.total_bytes = 0

    @property
    def seconds_received(self) -> float:
        """Session-relative time of the END of the audio received so far."""
        return self.total_bytes / (2.0 * self.sample_rate)

    def append(self, frame: bytes) -> None:
        self._buf += frame
        self.total_bytes += len(frame)
        excess = len(self._buf) - self._capacity_bytes
        # Trim in blocks (≥10% of capacity) rather than per frame: a bytearray
        # front-delete is a memmove of the whole buffer, and doing ~2.9 MB of
        # that ten times a second for every session is needless churn. The
        # buffer therefore holds between 100% and 110% of ``seconds``.
        if excess > self._capacity_bytes // 10:
            excess -= excess % 2
            del self._buf[:excess]
            self._dropped_bytes += excess

    def slice(self, start_s: float, end_s: float) -> bytes:
        """Raw PCM16 bytes for ``[start_s, end_s)`` on the session timeline,
        clamped to what is still held. Empty when nothing usable remains
        (window entirely trimmed, inverted, or not yet received)."""
        b0 = int(start_s * self.sample_rate) * 2
        b1 = int(end_s * self.sample_rate) * 2
        b0 = max(b0, self._dropped_bytes)
        b1 = min(b1, self.total_bytes)
        if b1 <= b0:
            return b""
        lo = b0 - self._dropped_bytes
        return bytes(self._buf[lo:lo + (b1 - b0)])


def _pcm16_to_float32(raw: bytes) -> np.ndarray:
    """int16 LE bytes → float32 in [-1, 1) — the shape tone_id/speaker_id take."""
    usable = len(raw) - (len(raw) % 2)
    return np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Suggestion job (what the worker queue carries)
# ---------------------------------------------------------------------------

@dataclass
class SuggestionJob:
    """One queued coaching job plus its enqueue-time snapshot.

    empathy/interject/role/self_speaker are snapshotted at ENQUEUE time so a
    mid-flight config change never retypes an already-queued turn (the
    pre-existing tuple contract, now named). ``is_self`` is the phone's
    verdict from turn_local — when it is not None it WINS over the fragile
    ``self_speaker`` label comparison; None means "decide the legacy way".
    ``tone_context`` is the phone's text-tone/prosody for the prompt.

    ``prev_done`` / ``done`` form the per-session ORDERING CHAIN that keeps
    final events in utterance order when more than one job is in the LLM
    at once (local-first concurrency): a job sends its final event only
    after the previous job's ``done`` future resolved, and resolves its own
    ``done`` when it finishes (delivered, errored, or dropped as superseded
    — a dropped job's ``done`` resolves as soon as its predecessor's does,
    so the chain never stalls on a turn nobody is generating).
    """

    utterance: Utterance
    empathy_slider: int
    interject_level: int
    role: str
    self_speaker: str | None
    timing: UtteranceTiming
    is_self: bool | None = None
    tone_context: dict | None = None
    prev_done: "asyncio.Future[None] | None" = field(default=None, repr=False, compare=False)
    done: "asyncio.Future[None] | None" = field(default=None, repr=False, compare=False)

    def mark_done(self) -> None:
        """Resolve ``done`` (idempotent; a no-op for an unchained job)."""
        if self.done is not None and not self.done.done():
            self.done.set_result(None)

    async def wait_turn(self) -> None:
        """Block until every earlier job in the chain has finished — the
        gate in front of a FINAL event (suggestion, nudge, or error)."""
        if self.prev_done is not None and not self.prev_done.done():
            await asyncio.shield(self.prev_done)

    def release_when_predecessor_done(self) -> None:
        """For a job dropped before it started (latest-wins): keep the chain
        intact by resolving ``done`` exactly when the predecessor's resolves."""
        prev = self.prev_done
        if prev is None or prev.done():
            self.mark_done()
            return
        prev.add_done_callback(lambda _fut: self.mark_done())


# ---------------------------------------------------------------------------
# turn_local helpers (Track 3-server) — pure, unit-testable
# ---------------------------------------------------------------------------

def _remember_local_range(ranges: list[tuple[float, float]], start: float, end: float) -> None:
    ranges.append((start, end))
    if len(ranges) > LOCAL_RANGES_MAX:
        del ranges[:-LOCAL_RANGES_MAX]


def _covered_by_local_range(
    ranges: list[tuple[float, float]], start: float, end: float,
    pad: float = LOCAL_RANGE_PAD_S,
) -> bool:
    """Whether a Deepgram segment duplicates a turn the phone already
    handled: its midpoint lies inside some remembered ``[start-pad, end+pad]``.

    Midpoint (not any-overlap) on purpose: Deepgram and the phone's VAD split
    speech at slightly different points, and a Deepgram segment that merely
    brushes the edge of a local turn is far more likely to be the NEXT (or
    previous) un-covered span than a duplicate — dropping it would lose
    speech the phone never reported. The pad handles the edge disagreement
    for genuine duplicates.
    """
    mid = (start + end) / 2.0
    return any(lo - pad <= mid <= hi + pad for lo, hi in ranges)


def _tone_context_from_event(event: TurnLocalEvent) -> dict | None:
    """The phone's measurements, as the dict the prompt renderer takes.

    ``{"text_tone": {...}, "prosody": {...}}`` with every ``None`` field
    dropped (a phone that couldn't estimate pitch sends null and the prompt
    must not say "pitch: None"); ``None`` when there is nothing at all so the
    prompt stays byte-identical to the legacy one.
    """
    return _tone_context_from_parts(
        event.text_tone.model_dump() if event.text_tone is not None else None,
        event.prosody.model_dump() if event.prosody is not None else None,
    )


def _tone_context_from_parts(text_tone: dict | None, prosody: dict | None) -> dict | None:
    """Same as :func:`_tone_context_from_event` from already-dumped dicts
    (a call peer's turn arrives as the merged-transcript row)."""
    out: dict = {}
    for key, values in (("text_tone", text_tone), ("prosody", prosody)):
        if not isinstance(values, dict):
            continue
        kept = {k: v for k, v in values.items() if v is not None}
        if kept:
            out[key] = kept
    return out or None


# Human-readable labels/units for the prompt; anything not listed renders as
# "key: value" so a new field (or a free-form ``label``) still reaches the
# model without a code change.
_TONE_FIELD_FORMAT: dict[str, str] = {
    "warmth": "warmth {}/100",
    "defensiveness": "defensiveness {}/100",
    "sarcasm": "sarcasm {}/100",
    "sadness": "sadness {}/100",
    "frustration": "frustration {}/100",
    "label": 'label "{}"',
    "rms_dbfs": "loudness {} dBFS",
    "pitch_hz": "median pitch {} Hz",
    "speech_rate": "speech rate {} syl/s",
}


def _render_tone_context(tone_context: dict | None) -> str:
    """Render the phone's tone/prosody as a short prompt block, or ``""``.

    Framed as HINTS, explicitly, because these are best-effort on-device
    estimates (see TurnProsody's honesty rule) and the model must weigh the
    words first. Deterministic order so prompts are reproducible in tests.
    """
    if not tone_context:
        return ""
    lines: list[str] = []
    for section, title in (("text_tone", "text tone"), ("prosody", "prosody")):
        values = tone_context.get(section)
        if not values:
            continue
        parts = [
            _TONE_FIELD_FORMAT.get(k, k + " {}").format(v)
            for k, v in sorted(values.items())
        ]
        lines.append(f"- {title}: " + ", ".join(parts))
    if not lines:
        return ""
    return (
        "On-device signals for this turn (measured by the phone; treat as "
        "hints, not facts):\n" + "\n".join(lines)
    )


def _turn_prompt(
    utterance: Utterance,
    tone_context: dict | None = None,
    speaker_name: str | None = None,
) -> str:
    """The user-turn content for both coaching prompts. Byte-identical to the
    pre-Track-3 prompt when there is no tone context and no name.
    ``speaker_name`` (mid-call naming — see ``apply_speaker_label``) tells
    the coach WHO said it ("Mom"), so the suggestion is for that person."""
    if speaker_name:
        content = f'Transcript turn from {speaker_name}: "{utterance.text}"'
    else:
        content = f'Transcript turn: "{utterance.text}"'
    block = _render_tone_context(tone_context)
    if block:
        content += "\n\n" + block
    return content


def _validation_summary(exc: ValidationError) -> str:
    """Field locations + error types only — never the offending values,
    which for turn_local include transcript text (P1-4 applies to error
    frames too: they end up in client logs and crash reports)."""
    return "; ".join(
        ".".join(str(p) for p in err.get("loc", ())) + ": " + str(err.get("type"))
        for err in exc.errors()
    ) or "invalid"


# ---------------------------------------------------------------------------
# Session context (in-memory, per-connection)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    session_id: str
    empathy_slider: int = 50
    # Interjection threshold 0-100: a suggestion is only VOICED when the
    # LLM-scored importance of the moment clears this bar. 0 (default) voices
    # every turn — the pre-slider behaviour. Orthogonal to empathy_slider,
    # which sets the STYLE of suggestions, not when to deliver them.
    interject_level: int = 0
    role: str = "Husband"
    utterances: list[Utterance] = field(default_factory=list)
    # Verified Firebase uid, set by the WS auth handshake before any audio is
    # processed. None only during the pre-auth window; a session that reaches
    # provider setup always has it.
    uid: str | None = None
    # Net-new voice-profile context (all optional, all backward-compatible):
    # the config message may carry the relationship + coached-speaker ids so the
    # WS coach can load a voice profile once, at config time. With none set,
    # voice_profile stays None and behaviour is byte-identical to today.
    relationship_id: str | None = None
    from_participant_id: str | None = None
    voice_profile: dict | None = None
    voice_profile_loaded: bool = False
    # Side-aware coaching ("the coach knows who you are"): the diarized label
    # ("Speaker A"/"Speaker B") of the coached user, set via config. None = the
    # legacy behaviour: every turn is coached as an OTHER turn (suggest what to
    # say back), exactly as before this field existed. When set, a turn whose
    # speaker matches is a SELF turn and gets a single delivery nudge instead.
    # Snapshotted at ENQUEUE time (like empathy/interject/role) so a mid-flight
    # config change never retypes an already-queued turn.
    self_speaker: str | None = None
    # Mid-call naming: raw diarized label → {person_id, display_name,
    # is_self} the user asserted via a `speaker_label` frame. Applied to the
    # RUNNING session only (the coach's prompts name the person; `is_self`
    # sets self_speaker); persistence rides the phone's POST /sessions/live
    # `speaker_labels` at session end — this server keeps no transcript.
    speaker_labels: dict[str, dict] = field(default_factory=dict)
    # Track 3-server — local-first (phone-orchestrated) sessions. Latched
    # True by the FIRST turn_local frame and never reset: a client that has
    # proven it segments/transcribes/speaks on-device keeps that role for
    # the session. Effects: server TTS off (unless tts_mode == "server"),
    # progressive `partial` suggestions on, latency_summary in
    # session_complete. A client that never sends turn_local sees exactly
    # the pre-Track-3 behaviour.
    local_first: bool = False
    # Time ranges (session-relative seconds) of turns the phone finalized;
    # Deepgram segments landing inside one are duplicates and are dropped.
    local_ranges: list[tuple[float, float]] = field(default_factory=list)
    # Config `tts`: "server" | "on-device" | None (unset). Only consulted
    # when local_first — a legacy client never sends it and always gets
    # server TTS. When local_first and unset, the phone speaks (expo-speech),
    # so the server must NOT also synthesize (double voice in the earpiece).
    tts_mode: str | None = None
    # Audio-tone escalation state (tone_id.EscalationTracker): each speaker's
    # running arousal baseline for THIS session, so a turn is judged against
    # that speaker's own earlier turns (the round-2 finding: absolute tone
    # reads voice identity; only the per-speaker delta reads escalation).
    # Created lazily by _enrich_tone; None until audio tone runs once.
    tone_tracker: object | None = None
    # Config `report_latency`: opt a legacy-protocol client into the
    # latency_summary on session_complete (local-first clients get it
    # automatically). Off by default so the old payload stays byte-identical.
    report_latency: bool = False
    latency: LatencyRecorder = field(default_factory=LatencyRecorder)
    pcm: PcmRingBuffer = field(default_factory=PcmRingBuffer)
    # Per-session cache of the uid's enrolled voiceprint documents for the
    # identity enrichment (see VOICEPRINT_CACHE_TTL_S). ``None`` = never
    # loaded; the stamp is time.monotonic() of the last successful load.
    voiceprints: list[dict] | None = None
    voiceprints_loaded_at: float | None = None
    # Review 2026-08-24: Deepgram stamps segments relative to the start of
    # ITS connection's audio stream. After a mid-session reconnect (P1-1)
    # the replacement connection's clock restarts at 0 while the session
    # timeline (ring buffer, turn_local ranges, the client's transcript)
    # keeps counting — so every segment from a replacement transcriber is
    # shifted by the session time at which that transcriber took over.
    transcriber_offset_s: float = 0.0
    # In-app calls (server/calls.py): the call this session is bound to via
    # a `call_join` frame, this participant's slot label ("Speaker A"/"B" —
    # every own turn is relabelled to it and is_self, structurally) and the
    # peer's. None = a solo session, byte-identical to before calls existed.
    call: "calls.Call | None" = None
    call_label: str | None = None
    call_peer_label: str | None = None
    # "participant" (coached) or "therapist" (observer: transcribed and
    # merged, never coached; receives the participants' coaching read-only).
    call_role: str | None = None


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


async def _close_ws_unauthorized(websocket: WebSocket, reason: str) -> None:
    """Tell the client why (best-effort) and close the WS with code 4401.

    4401 is the WebSocket analogue of HTTP 401: the token was missing,
    malformed, invalid/expired, or named a session the caller does not own.
    """
    with contextlib.suppress(Exception):
        await websocket.send_text(
            json.dumps({"type": "auth_error", "reason": reason})
        )
    with contextlib.suppress(Exception):
        await websocket.close(code=4401, reason=reason)


async def _apply_config(ctx: SessionContext, payload: dict) -> None:
    """Apply a config frame's non-auth fields to the session context.

    Shared by the initial auth handshake and later in-session config updates so
    empathy/role/voice-profile handling lives in exactly one place. The voice
    profile is loaded once, uid-scoped, the first time both ids are known.
    """
    if "empathy_slider" in payload:
        val = payload["empathy_slider"]
        if isinstance(val, int) and 0 <= val <= 100:
            ctx.empathy_slider = val
    if "interject_level" in payload:
        val = payload["interject_level"]
        if isinstance(val, int) and 0 <= val <= 100:
            ctx.interject_level = val
    if "role" in payload:
        role_val = payload["role"]
        # P2-3: role reaches the LLM system prompt — ignore wrong-typed values,
        # clamp the length (cost + injection surface).
        if isinstance(role_val, str):
            ctx.role = role_val[:MAX_ROLE_CHARS]
    # Optional voice-profile context. These only feed a DB lookup (not the
    # prompt directly), but clamp length anyway as defence in depth.
    rel_val = payload.get("relationship_id")
    if isinstance(rel_val, str):
        ctx.relationship_id = rel_val[:MAX_ID_CHARS]
    part_val = payload.get("from_participant_id")
    if isinstance(part_val, str):
        ctx.from_participant_id = part_val[:MAX_ID_CHARS]
    # Side-aware coaching: which diarized label is the coached user. Applied on
    # the initial config AND live updates. A string matching the "Speaker X"
    # shape sets it; JSON null (Python None) resets to the legacy every-turn-
    # OTHER behaviour; anything else (wrong type, "bob", bare "Speaker") is
    # ignored, like the other validated fields above. The `in payload` guard
    # distinguishes an explicit null (reset) from an absent key (leave as-is).
    if "self_speaker" in payload:
        self_val = payload["self_speaker"]
        if isinstance(self_val, str) and _SELF_SPEAKER_RE.match(self_val):
            ctx.self_speaker = self_val
        elif self_val is None:
            ctx.self_speaker = None
    # Track 3-server: who voices suggestions in a local-first session. Same
    # validated-or-ignored / null-resets shape as self_speaker. Has no effect
    # until the session is local_first (see SessionContext.tts_mode).
    if "tts" in payload:
        tts_val = payload["tts"]
        if tts_val in ("server", "on-device"):
            ctx.tts_mode = tts_val
        elif tts_val is None:
            ctx.tts_mode = None
    if "report_latency" in payload and isinstance(payload["report_latency"], bool):
        ctx.report_latency = payload["report_latency"]
    # Load the profile ONCE, the first time both ids are known — not per
    # utterance. uid-scoped, so a session can never load another user's stored
    # voice. A lookup failure degrades to no profile, never breaks the session.
    if (
        not ctx.voice_profile_loaded
        and ctx.relationship_id
        and ctx.from_participant_id
    ):
        ctx.voice_profile_loaded = True
        from main import _resolve_voice_profile
        try:
            ctx.voice_profile = await _resolve_voice_profile(
                ctx.relationship_id, ctx.from_participant_id, ctx.uid,
            )
        except Exception:
            logger.warning(
                "Voice profile lookup failed for session %s",
                ctx.session_id, exc_info=True,
            )


# Mid-call naming (`speaker_label` frames). Same slug rule as
# speaker_id.PERSON_ID_PATTERN / DISPLAY_NAME_MAX, kept as literals here so
# the pipeline never needs the (optional) voice deps to validate a name.
_PERSON_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_DISPLAY_NAME_MAX = 60
_SPEAKER_LABEL_MAX = 64


def apply_speaker_label(ctx: SessionContext, payload: dict) -> dict | None:
    """Validate + apply a client ``speaker_label`` frame to the running
    session. Returns the ack dict to send, or ``None`` when the frame is
    malformed (the caller answers ``{"error": …}``).

    ``speaker`` is the raw label being named; ``display_name`` the name;
    ``person_id`` an optional enrolled-person slug (null = name only);
    ``is_self`` (bool) marks the coached user — and, when the label is a
    diarizer-shaped "Speaker X", also sets ``self_speaker`` so side-aware
    coaching switches to that voice without a separate config frame.
    Naming a label as someone ELSE clears a stale self claim on it."""
    speaker = payload.get("speaker")
    name = payload.get("display_name")
    if not isinstance(speaker, str) or not speaker.strip() or len(speaker) > _SPEAKER_LABEL_MAX:
        return None
    if not isinstance(name, str) or not name.strip() or len(name) > _DISPLAY_NAME_MAX:
        return None
    person_id = payload.get("person_id")
    if person_id is not None and not (isinstance(person_id, str) and _PERSON_ID_RE.match(person_id)):
        return None
    is_self = payload.get("is_self", False)
    if not isinstance(is_self, bool):
        return None
    speaker = speaker.strip()
    entry = {"person_id": person_id, "display_name": name.strip(), "is_self": is_self}
    # One label per person: a person re-bound to another label frees the old one.
    if person_id is not None:
        for other, existing in list(ctx.speaker_labels.items()):
            if other != speaker and existing.get("person_id") == person_id:
                del ctx.speaker_labels[other]
    ctx.speaker_labels[speaker] = entry
    if is_self:
        if _SELF_SPEAKER_RE.match(speaker):
            ctx.self_speaker = speaker
    elif ctx.self_speaker == speaker:
        ctx.self_speaker = None
    return {"type": "speaker_label_ack", "speaker": speaker, **entry}


def display_speaker(ctx: SessionContext, speaker: str) -> str:
    """What the coach's prompt calls a raw label: its mid-call name, else
    the label itself."""
    entry = ctx.speaker_labels.get(speaker)
    name = entry.get("display_name") if isinstance(entry, dict) else None
    return name if isinstance(name, str) and name else speaker


async def _session_owner_ok(session_id: str, uid: str) -> bool:
    """Whether ``uid`` may open a live WS on ``session_id``.

    A WS session_id may be an ad-hoc/ephemeral id with no ``sessions`` row —
    that is allowed. But if a row DOES exist it must belong to the caller, so a
    user cannot attach the live audio pipeline (transcripts, suggestions) to
    another user's stored session. A DB error fails closed (denied).
    """
    from main import get_db
    try:
        db = await get_db()
        try:
            cursor = await db.execute(
                "SELECT user_id FROM sessions WHERE id = ?", (session_id,),
            )
            row = await cursor.fetchone()
        finally:
            await db.close()
    except Exception:
        logger.warning(
            "Session ownership check failed for %s", session_id, exc_info=True,
        )
        return False
    if row is None:
        return True
    return row["user_id"] == uid


async def _authenticate(
    websocket: WebSocket, ctx: SessionContext, send_json,
) -> bool:
    """Gate the session on a verified Firebase token in the first config frame.

    The WebSocket handshake cannot carry an ``Authorization`` header, so the
    client sends its ``id_token`` in the opening ``config`` message. This
    consumes that first frame: it must be a JSON ``config`` carrying a valid
    ``id_token`` whose session (if it maps to a stored row) the caller owns. On
    success the verified uid is stored on ``ctx``, the frame's other config
    fields are applied, a ``config_ack`` is sent, and True is returned. On any
    failure the socket is closed 4401 and False is returned — no audio, no
    transcript, and no provider work ever happen for an unauthenticated client.
    """
    try:
        message = await websocket.receive()
    except Exception:
        return False
    if message.get("type") == "websocket.disconnect":
        return False
    text = message.get("text")
    if text is None:
        # Binary audio before authenticating — reject before any provider work.
        await _close_ws_unauthorized(websocket, "authentication required")
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await _close_ws_unauthorized(websocket, "authentication required")
        return False
    if not isinstance(payload, dict) or payload.get("type") != "config":
        await _close_ws_unauthorized(websocket, "authentication required")
        return False
    token = payload.get("id_token")
    if not isinstance(token, str) or not token.strip():
        await _close_ws_unauthorized(websocket, "missing id_token")
        return False
    try:
        from auth import verify_id_token
        # verify_id_token is a blocking SDK call (cert fetch) — off the loop.
        uid = await asyncio.to_thread(verify_id_token, token.strip())
    except Exception:
        await _close_ws_unauthorized(websocket, "invalid id_token")
        return False
    if not await _session_owner_ok(ctx.session_id, uid):
        await _close_ws_unauthorized(websocket, "session not owned by user")
        return False
    ctx.uid = uid
    await _apply_config(ctx, payload)
    await send_json({"type": "config_ack"})
    return True


async def audio_ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Handle a single audio-streaming WebSocket connection.

    Protocol
    --------
    Client → Server (binary):  raw audio chunks
    Client → Server (text):    JSON control messages, e.g.
        {"type": "config", "empathy_slider": 75, "role": "Husband",
         "id_token": "<firebase id token>"}
        The FIRST frame must be such a config carrying a valid ``id_token``
        (the WS handshake cannot send an Authorization header): the server
        verifies it, stores the uid, and only then does any provider work.
        A missing/invalid token — or a session owned by another user — is
        closed with code 4401 before a single byte of audio is processed.
        Later config frames reuse the established uid (no token needed).
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

    Track 3-server — local-first (phone-orchestrated) clients. A phone that
    captures, segments, identifies the speaker, transcribes and coaches
    ON-DEVICE still streams the same raw PCM frames as above, and adds:
    Client → Server (text):
        {"type": "turn_local", ...}            — a ``TurnLocalEvent`` per turn the
                                                 phone finalized itself. The FIRST
                                                 one latches the session local-first
                                                 (see ``SessionContext.local_first``):
                                                 no transcript is echoed back, a
                                                 CLOUD suggestion is generated from
                                                 the phone's transcript (+ its tone
                                                 measurements), server TTS is off
                                                 unless config ``tts: "server"``,
                                                 and Deepgram segments overlapping
                                                 a reported turn are dropped.
        config keys ``tts`` ("server" | "on-device" | null) and
        ``report_latency`` (bool) — see ``_apply_config``.
        {"type": "speaker_label", "speaker": "Speaker B",
         "display_name": "Mom", "person_id": "mom" | null,
         "is_self": false}                    — mid-call naming: the coach's
                                                 prompts use the name from the
                                                 next turn on; ``is_self`` sets
                                                 the coached voice. Answered
                                                 with ``speaker_label_ack`` (or
                                                 ``{"error": "invalid
                                                 speaker_label"}``). See
                                                 ``apply_speaker_label``.
    Server → Client (text):
        {"type": "suggestion", "suggestion_source": "cloud", "partial": bool}
                                               — the cloud suggestion for a
                                                 turn_local; when the LLM streams,
                                                 a ``partial: true`` preview with the
                                                 first suggestion precedes the final
        {"type": "tone_flag", "source": "audio", ...}
                                               — ``ToneFlagEvent`` from the streamed
                                                 audio (only if tone_id surfaces)
        {"type": "speaker_identity", ...}      — ``SpeakerIdentityEvent``: the
                                                 server's voiceprint verdict on the
                                                 turn's speaker
        {"type": "session_complete",
         "latency_summary": {stage: {p50, p95, n}}}
                                               — per-stage ms percentiles for
                                                 local-first / report_latency clients

    In-app calls (2026-08-25, server/calls.py) — MindShift IS the call, so
    every side can be coached. Audio is peer-to-peer (WebRTC, full mesh);
    this socket carries the signaling and the merged transcript. Members
    are two coached "participant"s (host = Speaker A, second = Speaker B)
    and at most one "therapist" observer (Speaker C: transcribed and
    merged, never coached, sees the participants' coaching read-only):
    Client → Server (text):
        {"type": "call_join", "call_id", "join_code"?, "display_name"?,
         "role"?: "participant" | "therapist"} — bind this session to the call.
                                                 A non-member with the join code
                                                 joins here too. Answered with
                                                 `call_state` (or {"error":
                                                 "call_join: …"}).
        {"type": "rtc_signal", "call_id", "to"?: uid,
         "payload": {sdp | candidate | …}}      — relayed verbatim to the addressed
                                                 member's socket as {"type":
                                                 "rtc_signal", "call_id", "from":
                                                 uid, "payload"}. `to` may be
                                                 omitted only in a two-member call.
        turn_local                             — as above; in a call the turn is
                                                 relabelled to this member's slot
                                                 label, is_self, appended to the
                                                 shared transcript and pushed to
                                                 every other member (below). A
                                                 therapist's turn is never coached.
        speaker_label                          — call-wide: naming another member's
                                                 slot label persists to the call
                                                 record and this member's episode.
    Server → Client (text):
        {"type": "call_state", "call_id", "status", "self_role", "self_label",
         "peer_label", "therapist_label", "participants": [{uid, slot, label,
         role, display_name, is_self, connected}], "ice_servers": [...], ...}
                                               — on every bind/leave/name change
        {"type": "transcript", "speaker": <sender's label>, "display_name",
         "role", "text", "start_time", "end_time", "call_id", "participant_uid",
         "is_self": false, "seq", "local_start_time", "local_end_time",
         "text_tone", "prosody"}               — a turn another member's phone
                                                 finalized; a participant is
                                                 coached on it (a `suggestion`
                                                 follows), a therapist just sees it
        suggestion / tone_flag / speaker_identity + "for_uid"
                                               — THERAPIST sockets only: a read-only
                                                 copy of each participant's coaching
                                                 event, tagged with that participant
        {"type": "call_ended", "call_id", "reason", "ended_by", "episode_id",
         "episodes": {uid: episode_id}}        — the call is over; `episode_id` is
                                                 THIS participant's stored episode
                                                 (null for the therapist, who gets
                                                 `episodes` and a share of each)
        session_complete also carries "call": {call_id, status, episode_id}.

    New WebSockets beyond ``MAX_WS_SESSIONS`` concurrent sessions are closed
    immediately with code 1013 ("try again later") — an honest rejection
    instead of letting every session degrade (P2-1).

    Two checks run BEFORE ``accept()`` so a rejected connection never reaches
    Deepgram/Anthropic (no credit spend, no session slot held):
      * P0-1 Origin allowlist — cross-site browser connections closed 4403.
      * P2-7 session_id shape — a non-UUID id is closed 4403.
    """
    # P0-1: reject cross-site browser WS before accepting. Closing in the
    # "connecting" state rejects the handshake cleanly (the client sees 4403).
    origin = websocket.headers.get("origin")
    host = websocket.headers.get("host")
    if not _origin_allowed(origin, host):
        logger.info("Rejecting WS session: Origin %r not allowlisted", origin)
        await websocket.close(code=4403)
        return

    # P2-7: session_id is a short client-chosen identifier (the app uses
    # "live-<timestamp>"). Require a safe, bounded shape — reject anything with
    # unsafe characters or an abusive length — but do NOT require a UUID, which
    # wrongly rejected the real mobile client.
    if not _is_valid_session_id(session_id):
        logger.info("Rejecting WS session: session_id %r has an unsafe shape",
                    session_id)
        await websocket.close(code=4403)
        return

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

    # The receive loop (acks/errors) and the suggestion worker both send on
    # this socket — serialize so frames never interleave mid-send.
    send_lock = asyncio.Lock()

    async def send_json(payload: dict) -> None:
        async with send_lock:
            await websocket.send_text(json.dumps(payload))

    # AUTH GATE — before ANY provider work, credit spend, or transcript. The
    # first client frame must be a config carrying a valid Firebase id_token
    # for a session this user owns. On failure the socket is already closed
    # 4401; just return (no worker task has been created yet, so nothing leaks).
    if not await _authenticate(websocket, ctx, send_json):
        return

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
    # Track 3-server: the enrichment path reads the user's voiceprint through
    # the recordings store when one is configured (None → identity skipped).
    recordings_store = getattr(state, "recordings_store", None)
    # Injectable monotonic clock so latency tests can assert exact stage
    # arithmetic; production uses time.monotonic.
    ctx.latency = LatencyRecorder(
        clock=getattr(state, "monotonic_clock", None) or time.monotonic,
    )

    # Suggestion generation (LLM via thread + TTS HTTP, up to ~15s) runs on
    # background workers so it never stalls the audio receive loop — audio
    # keeps flowing to Deepgram while a suggestion is being generated.
    # A legacy client gets exactly ONE worker (its server-TTS audio plays in
    # arrival order); a local-first client unlocks LOCAL_FIRST_CONCURRENCY
    # workers once it has proven itself with a turn_local, and the ordering
    # chain on SuggestionJob (prev_done/done) keeps FINAL events in utterance
    # order even then — a later turn's answer never overtakes an earlier one.
    # Queue items are SuggestionJobs carrying the (empathy, interject, role,
    # self_speaker) snapshot at enqueue time — self_speaker is snapshotted too
    # so a mid-flight config change never retypes a queued turn (SELF vs OTHER).
    suggestion_queue: "asyncio.Queue[SuggestionJob]" = asyncio.Queue()
    concurrency_unlocked = asyncio.Event()
    chain_tail: list[SuggestionJob | None] = [None]  # the newest chained job
    # enqueued-vs-finished counters let the stop handler report honestly how
    # many suggestions were dropped when draining times out (P1-8) — qsize()
    # alone would miss the item the worker is currently processing.
    # "superseded" counts stale turns dropped by the latest-wins policy below;
    # "suppressed" counts Deepgram segments dropped as duplicates of turns the
    # phone already handled (Track 3-server).
    queue_stats = {
        "enqueued": 0, "finished": 0, "superseded": 0, "suppressed": 0,
        # turn_local frames whose enrichment was skipped because
        # MAX_ENRICHMENT_INFLIGHT tasks were already running.
        "enrichment_skipped": 0,
    }

    # Track 3-server: best-effort background enrichment (audio tone / voice
    # identity / watch relay) spawned per turn_local. Tracked so a graceful
    # stop can drain them (bounded) and cleanup can cancel them; never awaited
    # inline, so a slow model can't stall the receive loop.
    enrichment_tasks: set[asyncio.Task] = set()

    async def send_event(event) -> None:
        async with send_lock:
            await websocket.send_text(event.model_dump_json())
        # In-app call: the observing therapist sees this participant's
        # coaching read-only (tagged for_uid by the call). Best-effort.
        if ctx.call is not None and ctx.call_role == calls.ROLE_PARTICIPANT:
            with contextlib.suppress(Exception):
                await ctx.call.fan_out(ctx.uid, event.model_dump())

    async def process_segment(job: SuggestionJob) -> None:
        utterance, timing = job.utterance, job.timing
        # Mid-call naming: the LLM is told the turn came from "Mom"; the wire
        # keeps the raw label (events are built from the original utterance,
        # so the phone's per-label bookkeeping and self_speaker stay exact).
        named = display_speaker(ctx, utterance.speaker)
        speaker_name = named if named != utterance.speaker else None

        def server_owns_tts() -> bool:
            # Who voices this suggestion. Decided at the moment we would
            # SYNTHESIZE (not at enqueue, not at the start of processing) on
            # purpose: the first turn_local can arrive while a Deepgram-
            # transcribed job is mid-LLM, and by the time that job is voiced
            # the phone is already speaking its own suggestions — a server
            # voice on top would be two voices in one earpiece. A legacy
            # client is never local_first, so it always gets server TTS
            # exactly as before.
            return (not ctx.local_first) or ctx.tts_mode == "server"

        # Progressive `partial` previews only for clients that have proven
        # they understand the field (see SuggestionEvent.partial).
        progressive = ctx.local_first

        # Side-aware coaching: is this the coached user's OWN turn? The phone's
        # voiceprint verdict (job.is_self, from turn_local) wins when present;
        # otherwise the legacy label comparison — self_speaker is the enqueue-
        # time snapshot, so a mid-flight config change never retypes this
        # already-queued turn. Neither known → every turn is an OTHER turn,
        # i.e. the original behaviour.
        if job.is_self is not None:
            self_turn = job.is_self
        else:
            self_turn = (
                job.self_speaker is not None
                and utterance.speaker == job.self_speaker
            )

        # Hedge bookkeeping for this turn's LLM call (filled by the streaming
        # helper when the client's stream is a hedge-capable one).
        hedge_stats: dict = {}

        def note_hedge() -> None:
            if "hedged" in hedge_stats:
                timing.hedged = bool(hedge_stats["hedged"])
                timing.hedge_won = bool(hedge_stats.get("hedge_won"))

        if self_turn:
            # Coach their DELIVERY (one whispered nudge) rather than suggesting
            # what to say to the other person. A local-first client's nudge
            # goes through the same hedged streaming call as a suggestion
            # (there is nothing to preview, but the first-token tail is the
            # same tail); a legacy client keeps the plain complete() call.
            timing.llm_start = ctx.latency.now()
            nudge, importance = await _generate_nudge(
                llm_client, utterance, job.empathy_slider, job.role,
                ctx.voice_profile, job.tone_context, speaker_name=speaker_name,
                stream=progressive, stats=hedge_stats,
            )
            timing.llm_end = ctx.latency.now()
            note_hedge()
            if not nudge:
                # "Only speak when something should change." The transcript
                # event already went out at enqueue; a self turn that needs no
                # correction sends NOTHING further — no suggestion event, no TTS.
                timing.sent = ctx.latency.now()
                ctx.latency.record(timing, session_id)
                return
            # Same interjection gate as below: voice (and synthesize TTS) only
            # when the nudge's urgency clears the session's threshold.
            speak = importance >= job.interject_level
            tts_audio = None
            if speak and server_owns_tts():
                timing.tts_start = ctx.latency.now()
                tts_audio = await tts.synthesize(nudge)
                timing.tts_end = ctx.latency.now()
            await job.wait_turn()  # final events go out in utterance order
            await send_event(SuggestionEvent(
                session_id=session_id,
                utterance_text=utterance.text,
                speaker=utterance.speaker,
                suggestions=[nudge],
                empathy_slider=job.empathy_slider,
                audio_b64=tts_audio,
                importance=importance,
                speak=speak,
                kind="nudge",
            ))
            timing.sent = ctx.latency.now()
            ctx.latency.record(timing, session_id)
            return

        # OTHER turn (including the legacy self_speaker=None case): the
        # original behaviour. Generate suggestion via LLM. ctx.voice_profile
        # was loaded once at config time (None when unset → the exact legacy
        # prompt; likewise a None tone_context).
        async def on_first_suggestion(text: str) -> None:
            # Streaming preview: the first suggestion string is complete while
            # the model is still writing the rest. Never voiced, placeholder
            # importance; the final event below supersedes it. Best-effort — a
            # failed preview send must never sink the final suggestion.
            timing.llm_first_partial = ctx.latency.now()
            with contextlib.suppress(Exception):
                await send_event(SuggestionEvent(
                    session_id=session_id,
                    utterance_text=utterance.text,
                    speaker=utterance.speaker,
                    suggestions=[text],
                    empathy_slider=job.empathy_slider,
                    audio_b64=None,
                    speak=False,
                    partial=True,
                ))

        timing.llm_start = ctx.latency.now()
        suggestion_texts, importance = await _generate_suggestions(
            llm_client, utterance, job.empathy_slider, job.role,
            ctx.voice_profile, job.tone_context,
            on_first_suggestion=on_first_suggestion if progressive else None,
            speaker_name=speaker_name, stats=hedge_stats,
        )
        timing.llm_end = ctx.latency.now()
        note_hedge()

        # Interjection gate: the coach only VOICES a suggestion when the
        # LLM-scored importance of the moment clears the session's threshold.
        # The event is still sent (client may render it dimmed) — but no TTS
        # is synthesized for it, so the earpiece stays quiet. `speak` stays
        # True for a local-first client even without server TTS: it means
        # "worth voicing", and the phone voices it itself.
        speak = importance >= job.interject_level

        # TTS for first suggestion (only when it will actually be voiced, and
        # only when this server is the voice).
        tts_audio = None
        if speak and suggestion_texts and server_owns_tts():
            timing.tts_start = ctx.latency.now()
            tts_audio = await tts.synthesize(suggestion_texts[0])
            timing.tts_end = ctx.latency.now()

        # Ordering chain: with more than one job in flight (local-first
        # concurrency) this turn's FINAL event waits for the previous turn's
        # to be on the wire. The partial preview above is deliberately NOT
        # gated — it is a best-effort glimpse, keyed by utterance_text, and
        # gating it would give back the time-to-first-partial we bought.
        await job.wait_turn()
        await send_event(SuggestionEvent(
            session_id=session_id,
            utterance_text=utterance.text,
            speaker=utterance.speaker,
            suggestions=suggestion_texts,
            empathy_slider=job.empathy_slider,
            audio_b64=tts_audio,
            importance=importance,
            speak=speak,
        ))
        timing.sent = ctx.latency.now()
        ctx.latency.record(timing, session_id)

    async def suggestion_worker(index: int = 0) -> None:
        if index > 0:
            # Extra workers only serve a session that has proven local-first
            # (see handle_turn_local); until then this is exactly the
            # single-worker pipeline a legacy client has always had.
            await concurrency_unlocked.wait()
        while True:
            job = await suggestion_queue.get()
            try:
                await process_segment(job)
            except Exception as exc:
                # P0-2: a failed suggestion (LLM/TTS error, missing key, rate
                # limit) must never be silently swallowed — the client is told
                # WHICH utterance produced nothing and why. Only the exception
                # class name goes over the wire: provider error messages can
                # carry key fragments or internals. task_done() must still run
                # or queue.join() deadlocks.
                logger.warning(
                    "Suggestion processing failed for session %s (%s)",
                    session_id, _redact(job.utterance.text), exc_info=True,
                )
                reason = (
                    exc.reason if isinstance(exc, SuggestionUnavailable)
                    else type(exc).__name__
                )
                # A turn abandoned at the LLM first-token deadline (hedged
                # streaming) is counted in the session's latency report —
                # it is the tail this instrumentation exists to see.
                ctx.latency.record_abandoned(reason, session_id)
                # Suppress send failures — the socket may already be gone.
                with contextlib.suppress(Exception):
                    await job.wait_turn()  # errors keep utterance order too
                    await send_json({
                        "type": "suggestion_error",
                        "utterance_text": job.utterance.text,
                        "reason": reason,
                    })
            finally:
                job.mark_done()  # release the next turn in the chain
                queue_stats["finished"] += 1
                suggestion_queue.task_done()

    limit_notified = False

    async def enqueue_job(
        utterance: Utterance,
        timing: UtteranceTiming,
        *,
        is_self: bool | None = None,
        tone_context: dict | None = None,
    ) -> None:
        """Hand one turn to the suggestion worker: utterance budget (P2-1),
        latest-wins supersede, then put. Shared by the Deepgram path and the
        turn_local path so the two can never drift apart on policy."""
        nonlocal limit_notified
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

        # Latest-wins: a suggestion takes seconds (LLM + TTS). If newer
        # speech has arrived while one is still cooking, coaching the
        # backlog is stale by definition — the user hears advice about
        # something said a minute ago, arriving after they stopped
        # talking. Drop pending (not-yet-started) turns so the coach
        # always reacts to the most recent thing said; the dropped turns
        # remain in the transcript and the utterance buffer above.
        while not suggestion_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                dropped = suggestion_queue.get_nowait()
                suggestion_queue.task_done()
                queue_stats["finished"] += 1
                queue_stats["superseded"] += 1
                # Keep the ordering chain intact without the dropped turn:
                # its slot resolves the moment its predecessor's does.
                dropped.release_when_predecessor_done()

        queue_stats["enqueued"] += 1
        timing.enqueued = ctx.latency.now()
        timing.queue_depth = suggestion_queue.qsize()
        previous = chain_tail[0]
        job = SuggestionJob(
            utterance=utterance,
            empathy_slider=ctx.empathy_slider,
            interject_level=ctx.interject_level,
            role=ctx.role,
            self_speaker=ctx.self_speaker,
            timing=timing,
            is_self=is_self,
            tone_context=tone_context,
            prev_done=previous.done if previous is not None else None,
            done=asyncio.get_running_loop().create_future(),
        )
        chain_tail[0] = job
        suggestion_queue.put_nowait(job)

    async def enqueue_segments(result, frame_received: float | None = None) -> None:
        segment_finalized = ctx.latency.now()
        for raw_segment in _normalize_segments(result):
            # Re-base onto the session timeline (see transcriber_offset_s).
            # 0.0 for the original connection, so the legacy path is exact.
            segment = raw_segment
            if ctx.transcriber_offset_s:
                segment = TranscriptSegment(
                    text=raw_segment.text,
                    start_time=raw_segment.start_time + ctx.transcriber_offset_s,
                    end_time=raw_segment.end_time + ctx.transcriber_offset_s,
                    speaker=raw_segment.speaker,
                    confidence=raw_segment.confidence,
                )
            if ctx.local_ranges and _covered_by_local_range(
                ctx.local_ranges, segment.start_time, segment.end_time,
            ):
                # Track 3-server: the phone already showed (and coached) this
                # span from its own transcript — a second transcript line and
                # a second suggestion for the same words would be the exact
                # duplication local-first exists to avoid. Deepgram keeps
                # running only as the fallback for spans the phone did NOT
                # report (a turn its VAD missed), which pass straight through.
                queue_stats["suppressed"] += 1
                logger.debug(
                    "Suppressed Deepgram segment [%.2f, %.2f] for session %s — "
                    "covered by a phone-handled turn",
                    segment.start_time, segment.end_time, session_id,
                )
                continue
            # Label + remember + surface the transcript line IMMEDIATELY —
            # the words should never wait on (or be dropped with) a
            # suggestion. This also keeps the utterance buffer complete even
            # when the latest-wins policy below skips coaching a turn.
            speaker = labeler.label_for(segment.speaker)
            is_self: bool | None = None
            if ctx.call is not None:
                # In a call the server-STT fallback (a participant whose phone
                # has no on-device STT) hears only THIS participant, so every
                # segment is structurally their own turn.
                speaker = ctx.call_label or speaker
                is_self = True
            utterance = Utterance(
                session_id=session_id,
                speaker=speaker,
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                confidence=segment.confidence,
            )
            _remember_utterance(ctx, utterance)
            with contextlib.suppress(Exception):
                await send_json(TranscriptEvent(
                    session_id=session_id,
                    speaker=speaker,
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                ).model_dump())

            if ctx.call_role != calls.ROLE_THERAPIST:
                await enqueue_job(utterance, ctx.latency.start(
                    frame_received=frame_received, segment_finalized=segment_finalized,
                ), is_self=is_self)
            if ctx.call is not None:
                with contextlib.suppress(calls.CallError):
                    await ctx.call.push_turn(ctx.uid, {
                        "text": segment.text, "start_time": segment.start_time,
                        "end_time": segment.end_time, "transcript_source": "cloud",
                    })

    async def handle_turn_local(event: TurnLocalEvent, received: float) -> None:
        """A turn the PHONE finalized (Track 3-server).

        Emit nothing redundant: the phone already rendered the transcript
        and spoke its own fast suggestion. What the server adds, all async:
        a richer CLOUD suggestion from the phone's transcript + tone context
        (the same worker/queue as the Deepgram path, so ordering, budget and
        latest-wins hold across both), and best-effort enrichment on the
        streamed audio (tone, identity, watch relay) in a background task.
        """
        if not ctx.local_first:
            ctx.local_first = True
            # Unlock the extra suggestion worker(s): the phone voices its
            # own suggestions and reads partial previews, so overlapping
            # LLM calls (finals still in order) are pure latency win here.
            concurrency_unlocked.set()
            logger.info(
                "Session %s is local-first: the phone orchestrates capture/STT/"
                "speech; server TTS %s; suggestion concurrency %d",
                session_id,
                "on (config tts=server)" if ctx.tts_mode == "server" else "off",
                LOCAL_FIRST_CONCURRENCY,
            )
        _remember_local_range(ctx.local_ranges, event.start_time, event.end_time)
        utterance = Utterance(
            session_id=session_id,
            speaker=event.speaker,
            text=event.text,
            start_time=event.start_time,
            end_time=event.end_time,
        )
        _remember_utterance(ctx, utterance)

        # Enrichment BEFORE the suggestion budget check on purpose: it is not
        # an LLM/TTS spend, and identity/tone on a turn are worth having even
        # once coaching has stopped. Fire-and-forget; see enrichment_tasks.
        # Bounded by MAX_ENRICHMENT_INFLIGHT: a flood of turn_local frames
        # must not become a flood of model passes + store reads.
        if len(enrichment_tasks) >= MAX_ENRICHMENT_INFLIGHT:
            queue_stats["enrichment_skipped"] += 1
            if queue_stats["enrichment_skipped"] == 1:
                logger.warning(
                    "Session %s: %d enrichment tasks already in flight — "
                    "skipping enrichment for further turns until they drain",
                    session_id, len(enrichment_tasks),
                )
        else:
            task = asyncio.create_task(
                _enrich_turn_local(ctx, event, send_json, recordings_store)
            )
            enrichment_tasks.add(task)
            task.add_done_callback(enrichment_tasks.discard)

        if not event.text.strip():
            return  # nothing to coach; the range is still remembered above
        await enqueue_job(
            utterance,
            ctx.latency.start(frame_received=received, segment_finalized=received),
            is_self=event.is_self,
            tone_context=_tone_context_from_event(event),
        )

    # In-app calls: what this session exposes to a call it is bound to (see
    # calls.CallEndpoint). Remote turns arrive here from the PEER's socket
    # handler (same process, same loop) and are rendered + coached exactly as
    # an OTHER turn — the peer's phone is this session's transcriber for
    # that voice.
    class _CallSessionEndpoint(calls.CallEndpoint):
        def __init__(self) -> None:
            self.uid = ctx.uid or ""
            self.session_id = session_id

        async def send_json(self, payload: dict) -> None:
            await send_json(payload)

        async def on_remote_turn(self, turn: dict, *, display_name: str) -> None:
            received = ctx.latency.now()
            call = ctx.call
            await send_json({
                "type": "transcript",
                "session_id": session_id,
                "speaker": turn["speaker"],
                "display_name": display_name,
                "role": turn.get("role"),
                "text": turn["text"],
                "start_time": turn["start_time"],
                "end_time": turn["end_time"],
                "call_id": call.call_id if call is not None else None,
                "participant_uid": turn.get("participant_uid"),
                "is_self": False,
                "seq": turn.get("seq"),
                "local_start_time": turn.get("local_start_time"),
                "local_end_time": turn.get("local_end_time"),
                # The sender's on-device measurements, so an observer can
                # run the scoreboard over the whole conversation.
                "text_tone": turn.get("text_tone"),
                "prosody": turn.get("prosody"),
            })
            utterance = Utterance(
                session_id=session_id,
                speaker=turn["speaker"],
                text=turn["text"],
                start_time=float(turn["start_time"]),
                end_time=float(turn["end_time"]),
            )
            _remember_utterance(ctx, utterance)
            if not utterance.text.strip():
                return
            if ctx.call_role != calls.ROLE_PARTICIPANT:
                return  # the therapist observes; nobody coaches her
            await enqueue_job(
                utterance,
                ctx.latency.start(frame_received=received, segment_finalized=received),
                is_self=False,
                tone_context=_tone_context_from_parts(turn.get("text_tone"), turn.get("prosody")),
            )

        def set_peer_name(self, label: str, display_name: str) -> None:
            # The coach's prompts say "from Mom"; a person id the user
            # attached by a speaker_label frame is kept.
            existing = ctx.speaker_labels.get(label) or {}
            ctx.speaker_labels[label] = {
                "person_id": existing.get("person_id"),
                "display_name": display_name,
                "is_self": False,
            }

        def detach(self) -> None:
            ctx.call = None

    call_endpoint = _CallSessionEndpoint()

    async def handle_call_join(payload: dict) -> None:
        call_id = payload.get("call_id")
        if not isinstance(call_id, str) or not _is_valid_uuid(call_id):
            await send_json({"error": "call_join: invalid call_id"})
            return
        call = calls.registry.get(call_id)
        if call is None:
            await send_json({"error": "call_join: no such call"})
            return
        display_name = calls.clean_display_name(payload.get("display_name"))
        try:
            if ctx.uid not in call.participants:
                email = await calls.resolve_email(ctx.uid)
                calls.registry.join(
                    call, ctx.uid, join_code=payload.get("join_code"),
                    email=email, display_name=display_name, role=payload.get("role"),
                )
            if ctx.call is not None and ctx.call is not call:
                await ctx.call.leave(ctx.uid, call_endpoint)
            participant = await call.bind(
                ctx.uid, call_endpoint, store=recordings_store, display_name=display_name,
            )
        except calls.CallError as exc:
            await send_json({"error": f"call_join: {exc.detail}"})
            return
        ctx.call = call
        ctx.call_label = participant.label
        ctx.call_role = participant.role
        ctx.call_peer_label = call.state_for(ctx.uid)["peer_label"]
        # Structural attribution: in a call this member IS its slot label.
        ctx.self_speaker = participant.label
        logger.info(
            "Session %s bound to call %s as %s (%s, %s)",
            session_id, call.call_id, participant.slot, participant.label, participant.role,
        )

    async def handle_rtc_signal(payload: dict, raw_len: int) -> None:
        call = ctx.call
        if call is None or payload.get("call_id") != call.call_id:
            await send_json({"error": "rtc_signal: not in that call"})
            return
        if raw_len > calls.RTC_PAYLOAD_MAX_BYTES:
            await send_json({"error": "rtc_signal: payload too large"})
            return
        signal = payload.get("payload")
        if not isinstance(signal, dict) or not signal:
            await send_json({"error": "rtc_signal: payload must be a non-empty object"})
            return
        to_uid = payload.get("to")
        if to_uid is not None and not isinstance(to_uid, str):
            await send_json({"error": "rtc_signal: invalid to"})
            return
        try:
            await call.relay_signal(ctx.uid, signal, to_uid=to_uid)
        except calls.CallError as exc:
            await send_json({"error": f"rtc_signal: {exc.detail}"})

    async def apply_call_speaker_label(ack: dict) -> None:
        """Call-wide naming: a name for another member's label is the
        viewer's naming of that member; a real name on the OWN label is a
        self-declared name the others' screens show. Either way this
        member stays its slot label for coaching."""
        call = ctx.call
        if call is None:
            return
        name = ack["display_name"]
        target = call.by_label(ack["speaker"])
        if target is not None and target.uid != ctx.uid:
            await call.set_viewer_name(ctx.uid, target.uid, name)
        elif ack["speaker"] == ctx.call_label and name.strip().lower() not in ("you", "me"):
            await call.set_declared_name(ctx.uid, name)
        ctx.self_speaker = ctx.call_label

    worker_tasks = [
        asyncio.create_task(suggestion_worker(i))
        for i in range(LOCAL_FIRST_CONCURRENCY)
    ]

    async def cancel_workers() -> None:
        for task in worker_tasks:
            task.cancel()
        for task in worker_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # P1-7: this try must start IMMEDIATELY after the worker tasks are created.
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
            client frames buffer in the socket while retrying and are delivered
            to the replacement transcriber after the swap — only the single
            frame that hit the dead socket is lost. The client hears about it
            honestly if retries exhaust.
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
            # Latency: the moment this frame arrived — the "frame received"
            # stamp for any utterance it finalizes (or, for a turn_local, the
            # moment the phone's report landed).
            frame_received = ctx.latency.now()

            # --- Disconnect ---
            if message.get("type") == "websocket.disconnect":
                break

            # --- Binary audio chunk ---
            if "bytes" in message and message["bytes"] is not None:
                audio_bytes: bytes = message["bytes"]
                if len(audio_bytes) == 0:
                    continue
                # Track 3-server: keep the audio (contract-sized frames only)
                # whether or not cloud transcription is up — a local-first
                # phone with no Deepgram still wants tone/identity enrichment
                # on the slices it reports. Before the availability check so
                # the wire behaviour below stays exactly as it was.
                if len(audio_bytes) <= MAX_AUDIO_FRAME_BYTES:
                    ctx.pcm.append(audio_bytes)
                # Local-first sessions transcribe on the phone: once the first
                # turn_local has arrived, STOP feeding Deepgram. Measured on
                # production (scripts/live_e2e.py, 2026-08-24): with both
                # running, Deepgram finalizes on pauses BEFORE the phone's
                # turn_local lands, so 37 spans of a 13-turn scene were
                # transcribed and coached twice — the midpoint-overlap
                # suppression below only catches segments that arrive AFTER
                # the phone's turn, and a real phone's STT lags more than the
                # e2e client. Audio is still ring-buffered above for tone /
                # identity enrichment; the Deepgram socket stays open on its
                # keepalive so a client that stops sending turn_local (its STT
                # died) doesn't need a reconnect — see the turn_local docstring.
                if ctx.local_first:
                    continue
                if not transcription_available:
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
                            # The replacement's clock starts at 0 with the
                            # NEXT frame; the frame that hit the dead socket
                            # was already ring-buffered, so "now" on the
                            # session timeline is exactly the audio received.
                            ctx.transcriber_offset_s = ctx.pcm.seconds_received
                            # The client may have disconnected during the multi-
                            # second reconnect; a send on a dead socket can raise
                            # a non-WebSocketDisconnect error — suppress so it
                            # doesn't escape the handler (cleanup still runs).
                            with contextlib.suppress(Exception):
                                await send_json({"type": "transcription_restored"})
                            # The frame that hit the dead socket is gone — its
                            # audio cannot be recovered, only acknowledged.
                            continue
                    transcription_available = False
                    with contextlib.suppress(Exception):
                        await send_json(
                            {"type": "transcription_unavailable", "reason": str(exc)}
                        )
                    continue
                await enqueue_segments(result, frame_received=frame_received)

            # --- Text control message ---
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_json({"error": "invalid JSON"})
                    continue

                msg_type = payload.get("type")
                if msg_type == "config":
                    # Re-config (empathy / role / voice context). The id_token
                    # is verified only on the FIRST config — the auth handshake
                    # above — so later frames reuse the established uid and need
                    # not (and do not) re-present a token.
                    await _apply_config(ctx, payload)
                    await send_json({"type": "config_ack"})
                elif msg_type == "turn_local":
                    # Track 3-server: a phone-finalized turn. Validated with
                    # the shared model so a malformed report is rejected at
                    # the door ({"error": ...}, like invalid JSON) rather than
                    # half-applied; the summary names fields, never values.
                    try:
                        event = TurnLocalEvent.model_validate(payload)
                    except ValidationError as exc:
                        await send_json({
                            "error": f"invalid turn_local: {_validation_summary(exc)}"
                        })
                        continue
                    if event.session_id != session_id:
                        await send_json({"error": "turn_local session_id mismatch"})
                        continue
                    if event.end_time < event.start_time:
                        await send_json({"error": "turn_local end_time before start_time"})
                        continue
                    if ctx.call is not None:
                        # In a call the phone only ever hears its owner: the
                        # turn is this member's, whatever label its
                        # diarizer picked, and is_self by construction.
                        event = event.model_copy(
                            update={"speaker": ctx.call_label, "is_self": True},
                        )
                    if ctx.call is not None and ctx.call_role == calls.ROLE_THERAPIST:
                        # The observer's own words: transcribed and merged
                        # for everyone, never coached or enriched.
                        ctx.local_first = True
                        _remember_local_range(ctx.local_ranges, event.start_time, event.end_time)
                        _remember_utterance(ctx, Utterance(
                            session_id=session_id, speaker=event.speaker, text=event.text,
                            start_time=event.start_time, end_time=event.end_time,
                        ))
                    else:
                        await handle_turn_local(event, frame_received)
                    if ctx.call is not None:
                        try:
                            await ctx.call.push_turn(ctx.uid, event)
                        except calls.CallError as exc:
                            await send_json({"error": f"turn_local: {exc.detail}"})
                elif msg_type == "speaker_label":
                    # Mid-call naming ("Speaker B is Mom"): applied to this
                    # running session; the phone persists it at session end.
                    ack = apply_speaker_label(ctx, payload)
                    if ack is None:
                        await send_json({"error": "invalid speaker_label"})
                        continue
                    logger.info(
                        "Session %s: speaker %r named (person=%s, self=%s)",
                        session_id, ack["speaker"], ack["person_id"], ack["is_self"],
                    )
                    await send_json(ack)
                    await apply_call_speaker_label(ack)
                elif msg_type == "call_join":
                    await handle_call_join(payload)
                elif msg_type == "rtc_signal":
                    await handle_rtc_signal(payload, len(message["text"]))
                elif msg_type == "stop":
                    # Graceful stop: flush the transcriber so the FINAL
                    # utterance is delivered, wait (bounded — P1-8) for the
                    # pending SuggestionEvents to go out, then confirm
                    # completion and close server-side.
                    await enqueue_segments(
                        await _finish_transcriber(transcriber),
                        frame_received=frame_received,
                    )
                    completion: dict = {"type": "session_complete"}
                    # In-app call: hang up this side. When this was the last
                    # socket the call ends here (episodes persisted) and the
                    # `call_ended` frame precedes session_complete.
                    call = ctx.call
                    if call is not None:
                        with contextlib.suppress(Exception):
                            await call.leave(ctx.uid, call_endpoint)
                        me = call.participant(ctx.uid)
                        completion["call"] = {
                            "call_id": call.call_id,
                            "status": call.status,
                            "episode_id": me.episode_id if me else None,
                        }
                    try:
                        await asyncio.wait_for(
                            suggestion_queue.join(), timeout=STOP_DRAIN_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        # A hung LLM/TTS/send must not stall the client's stop for
                        # minutes. Count pending FIRST — an in-flight suggestion
                        # stuck in the LLM was never delivered, so it is genuinely
                        # dropped (the worker's finally would mark it "finished"
                        # on cancel, which would dishonestly under-report it).
                        pending = queue_stats["enqueued"] - queue_stats["finished"]
                        # THEN cancel before the completion send: the worker may
                        # be holding send_lock (a large TTS frame to a non-reading
                        # client), which would otherwise deadlock the send on the
                        # same lock — and it can't deliver a late suggestion after.
                        await cancel_workers()
                        completion["pending_dropped"] = pending
                        logger.warning(
                            "Graceful stop drain timed out after %.0fs with %d "
                            "pending suggestion(s) for session %s",
                            STOP_DRAIN_TIMEOUT_S, pending, session_id,
                        )
                    # Track 3-server: let in-flight enrichment (tone/identity/
                    # relay for the last turn_local) land before completion —
                    # bounded, because it is best-effort by definition.
                    if enrichment_tasks:
                        with contextlib.suppress(Exception):
                            await asyncio.wait(
                                set(enrichment_tasks),
                                timeout=ENRICHMENT_DRAIN_TIMEOUT_S,
                            )
                    # The per-stage p50/p95 report — only for clients that
                    # asked (report_latency) or proved they speak the new
                    # protocol (local_first); the legacy payload stays exact.
                    if ctx.local_first or ctx.report_latency:
                        completion["latency_summary"] = ctx.latency.summary()
                    # Bound the final send + close too — a connected-but-not-reading
                    # client must not hang the stop indefinitely.
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(send_json(completion), timeout=5.0)
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(websocket.close(code=1000), timeout=5.0)
                    break
                else:
                    await send_json({"error": f"unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    finally:
        # Cleanup must never raise, whatever state the connection died in.
        # In-app call: an abrupt drop leaves the call (the peer's call_state
        # shows us disconnected and it keeps coaching solo; if we were the
        # last one, the call ends and the episodes are persisted).
        if ctx.call is not None:
            with contextlib.suppress(Exception):
                await ctx.call.leave(ctx.uid, call_endpoint)
        await cancel_workers()
        # Enrichment tasks may still be inside a model call (to_thread) — the
        # thread finishes on its own; the task just stops mattering.
        for task in list(enrichment_tasks):
            task.cancel()
        for task in list(enrichment_tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
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
# turn_local enrichment (Track 3-server) — best-effort, never sinks a session
# ---------------------------------------------------------------------------
#
# Mirrors the stance of main._match_enrolled_speaker: "a cross-check must
# never sink the analysis". Every step here is independent, wrapped, and
# logged on failure; the session (and the cloud suggestion, which runs on
# the worker, not here) is unaffected by anything that goes wrong below.

# How long an enrichment task will wait for the ring buffer to catch up to
# the turn's end_time. The phone reports a turn only after its own STT
# finished, so the audio frames for the turn's tail have normally arrived
# already; this covers a network hiccup that reorders the report ahead of
# the last frames. Tests set it to 0.
SLICE_GRACE_S = 1.0


async def _await_audio_through(ctx: SessionContext, end_time: float) -> None:
    deadline = time.monotonic() + SLICE_GRACE_S
    while ctx.pcm.seconds_received < end_time and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


async def _enrich_turn_local(
    ctx: SessionContext, event: TurnLocalEvent, send_json, store,
) -> None:
    """Server-side enrichment of a phone-finalized turn: (a) audio tone,
    (b) voiceprint identity, (c) watch relay. Each is best-effort and
    isolated — one failing is logged and the others still run."""
    await _await_audio_through(ctx, event.end_time)
    pcm_bytes = ctx.pcm.slice(event.start_time, event.end_time)

    async def guarded(name: str, coro):
        try:
            return await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "turn_local %s enrichment failed for session %s",
                name, ctx.session_id, exc_info=True,
            )
            return None

    # Ordered on purpose: the relay (Track 1) escalates the WATCH from the
    # tone flag and only for the user's own turns, so it gets the audio tone
    # (if computed) and the server-corrected identity (if any) — the most
    # informed view of the turn, not the phone's first guess.
    tone_flag = await guarded("tone", _enrich_tone(ctx, event, pcm_bytes, send_json))
    # In a call the speaker is known structurally (participant = speaker), so
    # a voiceprint verdict could only contradict the truth — skipped.
    identity = None
    if ctx.call is None:
        identity = await guarded(
            "identity", _enrich_identity(ctx, event, pcm_bytes, send_json, store),
        )
    relayed = event
    if identity is not None:
        relayed = event.model_copy(update={
            "is_self": identity.is_self,
            "speaker_person_id": identity.person_id,
            "speaker_match_score": identity.score,
        })
    await guarded("relay", _relay_turn_local(ctx, relayed, tone_flag))


async def _enrich_tone(
    ctx: SessionContext, event: TurnLocalEvent, pcm_bytes: bytes, send_json,
) -> ToneFlagEvent | None:
    """(a) Classify the turn's AUDIO tone (tone_id) and surface it as a
    ToneFlagEvent(source="audio") — but only when tone_id.surface_allowed();
    otherwise it ships DARK: computed and logged, never shown (the owner's
    rule for a signal that measured weak, see server/tone_id.py). Returns
    the flag ONLY when it was surfaced, for the watch relay — a dark flag
    must not reach the user through the watch's haptics either, so dark
    mode returns None exactly as "skipped" does."""
    if tone_id is None or not tone_id.is_enabled():
        return None
    # is_available() imports torch/speechbrain on first use (seconds) — off
    # the loop, like every model-touching call.
    if not await asyncio.to_thread(tone_id.is_available):
        return None
    sr = ctx.pcm.sample_rate
    pcm = _pcm16_to_float32(pcm_bytes)
    min_seconds = float(getattr(tone_id, "MIN_TURN_SECONDS", 1.0))
    if pcm.size < int(min_seconds * sr):
        logger.debug(
            "Skipping audio tone for session %s: %.2fs of audio recovered "
            "for [%.2f, %.2f] (< %.1fs)",
            ctx.session_id, pcm.size / sr, event.start_time, event.end_time,
            min_seconds,
        )
        return None
    max_seconds = float(getattr(tone_id, "MAX_TURN_SECONDS", 30.0))
    pcm = pcm[: int(max_seconds * sr)]
    unavailable = getattr(tone_id, "ToneUnavailable", ())
    try:
        result = await asyncio.to_thread(tone_id.classify_pcm, pcm, sr)
    except unavailable as exc:
        # Flag off / model missing — an expected skip, not a failure.
        logger.debug("Audio tone unavailable for session %s: %s", ctx.session_id, exc)
        return None
    # Per-speaker escalation: this turn's arousal against THIS speaker's own
    # earlier turns in the session (pure numpy, microseconds — inline). The
    # tracker lives on the session so baselines accumulate across turns; a
    # speaker's first turn is honestly "unscored", never compared with
    # someone else's voice.
    scores = {str(k): float(v) for k, v in (result.get("scores") or {}).items()}
    tracker_cls = getattr(tone_id, "EscalationTracker", None)
    annotate = getattr(tone_id, "annotate_escalation", None)
    escalation = None
    if tracker_cls is not None and callable(annotate) and isinstance(result.get("arousal"), (int, float)):
        if ctx.tone_tracker is None:
            ctx.tone_tracker = tracker_cls()
        result = annotate(result, event.speaker, ctx.tone_tracker)
        escalation = result.get("escalation") or {}
        scores["arousal"] = float(result["arousal"])
        if escalation.get("delta") is not None:
            scores["arousal_delta"] = float(escalation["delta"])
    flag = ToneFlagEvent(
        session_id=ctx.session_id,
        speaker=event.speaker,
        start_time=event.start_time,
        end_time=event.end_time,
        source="audio",
        scores=scores,
        label=str(result["label"]),
        confidence=max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
    )
    if tone_id.surface_allowed():
        await send_json(flag.model_dump())
        if ctx.call is not None:
            with contextlib.suppress(Exception):
                await ctx.call.fan_out(ctx.uid, flag.model_dump())
        return flag
    # Dark mode: this log line IS the feature's output — nothing reaches the
    # client, and nothing reaches the watch (see the return contract above).
    logger.info(
        "audio tone (dark) backend=%s session=%s speaker=%s label=%s confidence=%.2f "
        "arousal=%s delta=%s history=%s seconds=%.1f phone_label=%s",
        result.get("backend"), ctx.session_id, event.speaker, flag.label, flag.confidence,
        result.get("arousal"),
        None if escalation is None else escalation.get("delta"),
        None if escalation is None else escalation.get("history"),
        pcm.size / sr,
        event.text_tone.label if event.text_tone else None,
    )
    return None


def _identify_turn_person(
    pcm: np.ndarray, sr: int, speaker: str, docs: list[dict],
) -> dict | None:
    """Blocking: who is this slice, among the uid's enrolled people?

    Runs Foundation B's :func:`speaker_id.identify_speakers_multi` over the
    recovered slice as ONE turn by ``speaker`` against every enrolled
    voiceprint (the owner is ``"self"``; partners are named people), so the
    verdict is a person id + display name, not just self/other. Returns
    that speaker's report entry — ``{matched_person_id, is_self,
    display_name, scores, ...}`` — or ``None`` when the slice was too short
    to embed (the report omits it honestly rather than scoring noise).
    """
    voiceprints = {
        d["person_id"]: np.asarray(d["embedding"], dtype=np.float32)
        for d in docs
        if isinstance(d.get("embedding"), list) and d.get("person_id")
    }
    if not voiceprints:
        return None
    people = {
        d["person_id"]: {
            "display_name": d.get("display_name"), "is_self": bool(d.get("is_self")),
        }
        for d in docs if d.get("person_id") in voiceprints
    }
    turns = [{"speaker": speaker, "start_time": 0.0, "end_time": pcm.size / sr}]
    report = speaker_id.identify_speakers_multi(
        pcm, sr, turns, voiceprints, people=people,
    )
    return (report.get("speakers") or {}).get(speaker)


async def _session_voiceprints(ctx: SessionContext, store) -> list[dict]:
    """The uid's enrolled voiceprint documents, read from the store at most
    once per VOICEPRINT_CACHE_TTL_S per session (see the constant). A read
    failure propagates (the caller's guard logs it) and leaves any earlier
    cached copy in place for the next turn."""
    now = time.monotonic()
    if (
        ctx.voiceprints is not None
        and ctx.voiceprints_loaded_at is not None
        and now - ctx.voiceprints_loaded_at < VOICEPRINT_CACHE_TTL_S
    ):
        return ctx.voiceprints
    docs = list(await store.list_voiceprints(ctx.uid) or [])
    ctx.voiceprints = docs
    ctx.voiceprints_loaded_at = now
    return docs


async def _enrich_identity(
    ctx: SessionContext, event: TurnLocalEvent, pcm_bytes: bytes, send_json, store,
) -> SpeakerIdentityEvent | None:
    """(b) Confirm or correct the phone's speaker verdict against the user's
    server-side voiceprints (every enrolled person, Foundation B); emit a
    SpeakerIdentityEvent either way (the client reconciles) and return it
    for the relay. Skipped cleanly (None) without deps, store, enrollment,
    or enough audio."""
    if speaker_id is None or store is None or not ctx.uid:
        return None
    if not await asyncio.to_thread(speaker_id.is_available):
        return None
    docs = await _session_voiceprints(ctx, store)
    if not docs:
        return None
    sr = ctx.pcm.sample_rate
    pcm = _pcm16_to_float32(pcm_bytes)
    min_seconds = float(getattr(speaker_id, "MIN_MATCH_SECONDS", 1.0))
    if pcm.size < int(min_seconds * sr):
        return None
    entry = await asyncio.to_thread(
        _identify_turn_person, pcm, sr, event.speaker, docs,
    )
    if entry is None:
        return None
    person_id = entry.get("matched_person_id")
    is_self = bool(entry.get("is_self"))
    scores = entry.get("scores") or {}
    # The score that justified the verdict; for "unknown", the best near-miss
    # so a client (or a log reader) can see how close it came.
    score = float(scores.get(person_id, max(scores.values(), default=0.0)))
    if event.is_self is not None and event.is_self != is_self:
        logger.info(
            "Correcting phone speaker verdict for session %s: phone is_self=%s, "
            "server is_self=%s person=%s (score %.3f)",
            ctx.session_id, event.is_self, is_self, person_id, score,
        )
    identity = SpeakerIdentityEvent(
        session_id=ctx.session_id,
        speaker=event.speaker,
        person_id=person_id,
        display_name=entry.get("display_name") if person_id else None,
        is_self=is_self,
        score=round(score, 4),
    )
    await send_json(identity.model_dump())
    if ctx.call is not None:
        with contextlib.suppress(Exception):
            await ctx.call.fan_out(ctx.uid, identity.model_dump())
    return identity


async def _relay_turn_local(
    ctx: SessionContext, event: TurnLocalEvent, tone_flag: ToneFlagEvent | None,
) -> None:
    """(c) Hand the turn to the watch relay (Track 1's
    ``watch.relay.push_turn_local(uid, event, *, tone_flag=None)``) so a
    paired watch escalates on the phone's turns too. The relay itself keeps
    the self-turns-only rule and its own confidence gate on the tone flag;
    this just delivers the most informed view. Sync or async relay accepted."""
    if watch_relay is None or not ctx.uid:
        return
    push = getattr(watch_relay, "push_turn_local", None)
    if push is None:
        return
    result = push(ctx.uid, event, tone_flag=tone_flag)
    if inspect.isawaitable(result):
        await result


# ---------------------------------------------------------------------------
# Streaming LLM helpers (Track 3-server)
# ---------------------------------------------------------------------------

def _supports_streaming(llm) -> bool:
    """Whether ``llm`` offers ``stream_complete()``. Checked on the TYPE so
    test doubles built from MagicMock (which auto-create any attribute on the
    instance) keep the plain ``complete()`` path and its exact call shape."""
    return callable(getattr(type(llm), "stream_complete", None))


# The first complete string inside `"suggestions": [ ... ]` of a (possibly
# fenced, possibly still-streaming) JSON response. Only matches once the
# closing quote has arrived, so a preview is never a truncated sentence.
_FIRST_SUGGESTION_RE = re.compile(r'"suggestions"\s*:\s*\[\s*("(?:[^"\\]|\\.)*")')


def _first_suggestion_in(buffer: str) -> str | None:
    match = _FIRST_SUGGESTION_RE.search(buffer)
    if not match:
        return None
    try:
        text = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return text if isinstance(text, str) and text.strip() else None


_REPAIR_SYSTEM = (
    "You repair malformed JSON. The user message is a model response that "
    "was supposed to be ONLY a JSON object with these keys: {keys}. Return "
    "that same content as one valid, complete JSON object with exactly those "
    "keys — no prose, no code fences, nothing else."
)


async def _parse_or_repair(
    llm, raw: str, *, keys: str, what: str, utterance_text: str,
) -> dict:
    """Parse the model's JSON, tolerantly; on failure ask the model ONCE to
    repair its own answer (tiny prompt, temperature 0, bounded tokens);
    raise :class:`SuggestionUnavailable` ("llm_parse_error") only when the
    repaired answer is unparseable too.

    A repaired result is a real model answer for this turn — it is never
    surfaced as an error. What the repair call cannot fix (a provider error
    on the repair call itself) propagates as its own exception, exactly as a
    provider error on the primary call does.
    """
    from main import parse_llm_json

    def _parse(text: str) -> dict:
        data = parse_llm_json(text)
        if not isinstance(data, dict):
            # Valid JSON of the wrong shape (list, string, number…).
            raise TypeError(f"expected a JSON object, got {type(data).__name__}")
        return data

    try:
        return _parse(raw)
    # ValueError covers json.JSONDecodeError and the empty-content case.
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        first_error = exc

    if not PARSE_REPAIR or not raw or not raw.strip():
        # P1-4: log a redacted marker, never the transcript text itself.
        logger.warning(
            "LLM returned unparseable %s for %s", what, _redact(utterance_text),
        )
        raise SuggestionUnavailable("llm_parse_error") from first_error

    logger.info(
        "LLM %s for %s was not valid JSON (%d chars) — asking for a repair",
        what, _redact(utterance_text), len(raw),
    )
    repaired = await asyncio.to_thread(
        llm.complete,
        system=_REPAIR_SYSTEM.format(keys=keys),
        user=raw,
        temperature=0.0,
        max_tokens=REPAIR_MAX_TOKENS,
    )
    try:
        return _parse(repaired)
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        logger.warning(
            "LLM %s for %s still unparseable after repair", what,
            _redact(utterance_text),
        )
        raise SuggestionUnavailable("llm_parse_error") from exc


async def _stream_with_first_suggestion(
    llm, system: str, user: str, on_first_suggestion=None, *,
    max_tokens: int = 512, stats: dict | None = None,
) -> str:
    """Consume ``llm.stream_complete`` in a thread; fire ``on_first_suggestion``
    (a coroutine function, optional) exactly once, as soon as the first
    suggestion string is complete; return the full response text.

    The preview coroutine is scheduled onto the event loop from the worker
    thread and awaited (failures suppressed) before returning, so the
    partial event is always on the wire BEFORE the caller sends the final
    one — the client never sees a preview after the real thing.

    Hedged streaming (perf/llm-hedging): when the client's stream is an
    :class:`~llm_client.HedgedStream` its ``hedged`` / ``hedge_won`` /
    ``first_token_ms`` are copied into ``stats`` (when given) after the
    stream is drained; a stream that produced no first token by the
    deadline raises :class:`SuggestionUnavailable` (``slow_llm``) so the
    worker reports the turn honestly and moves on instead of holding the
    queue for the SDK timeout. ``stats`` stays empty for a plain generator
    (a test double, a non-Anthropic provider).
    """
    loop = asyncio.get_running_loop()
    previews: list = []

    def run() -> str:
        parts: list[str] = []
        notified = on_first_suggestion is None
        stream = llm.stream_complete(
            system=system, user=user, max_tokens=max_tokens,
        )
        try:
            for delta in stream:
                parts.append(delta)
                if not notified:
                    first = _first_suggestion_in("".join(parts))
                    if first is not None:
                        notified = True
                        previews.append(asyncio.run_coroutine_threadsafe(
                            on_first_suggestion(first), loop,
                        ))
        # Resolved through the module at catch time (not imported by name)
        # so a reloaded llm_client — the test suite does that — still matches.
        except llm_client.LLMFirstTokenTimeout as exc:
            raise SuggestionUnavailable("slow_llm") from exc
        finally:
            if stats is not None and hasattr(stream, "hedged"):
                stats["hedged"] = bool(getattr(stream, "hedged", False))
                stats["hedge_won"] = bool(getattr(stream, "hedge_won", False))
                stats["first_token_ms"] = getattr(stream, "first_token_ms", None)
        return "".join(parts)

    raw = await asyncio.to_thread(run)
    for fut in previews:
        with contextlib.suppress(Exception):
            await asyncio.wrap_future(fut)
    return raw


# ---------------------------------------------------------------------------
# LLM suggestion helper
# ---------------------------------------------------------------------------

async def _generate_suggestions(
    llm: LLMClient,
    utterance: Utterance,
    empathy_slider: int,
    role: str,
    voice_profile: dict | None = None,
    tone_context: dict | None = None,
    *,
    on_first_suggestion=None,
    speaker_name: str | None = None,
    stats: dict | None = None,
) -> tuple[list[str], int]:
    """Call LLMClient.complete(); parse suggestions + moment importance.

    Returns ``(suggestions, importance)`` where importance is the model's
    0-100 rating of how much this moment warranted a coaching interjection.
    A missing/invalid importance fails OPEN to 100 (always voice) so an older
    or non-conforming model response degrades to pre-slider behaviour rather
    than silencing the coach.

    Raises :class:`SuggestionUnavailable` ("llm_parse_error") when the LLM
    output cannot be parsed (P0-3). It used to fabricate an
    "I hear you — <utterance>" line here — a fake suggestion that would then
    be TTS-spoken as if the coach really produced it. Honest failure instead:
    the suggestion worker turns the exception into a ``suggestion_error``
    event so the client knows this utterance yielded nothing.

    ``tone_context`` (Track 3-server) is the phone's text-tone/prosody dict
    (see :func:`_tone_context_from_event`); it is rendered into the user
    turn as hints. ``on_first_suggestion`` — when given AND the client
    supports ``stream_complete`` — is awaited once with the first complete
    suggestion string while the rest is still streaming. Both None → the
    prompt and the ``complete()`` call are byte-identical to before.
    """
    from main import empathy_system_prompt

    system = empathy_system_prompt(
        empathy_slider, role, voice_profile, live=LIVE_PROMPT,
    )
    user_content = _turn_prompt(utterance, tone_context, speaker_name)

    if on_first_suggestion is not None and _supports_streaming(llm):
        raw = await _stream_with_first_suggestion(
            llm, system, user_content, on_first_suggestion,
            max_tokens=SUGGESTION_MAX_TOKENS, stats=stats,
        )
    else:
        raw = await asyncio.to_thread(
            llm.complete, system=system, user=user_content,
            max_tokens=SUGGESTION_MAX_TOKENS,
        )

    data = await _parse_or_repair(
        llm, raw, keys='"suggestions" (list of strings), "importance" (integer)',
        what="response", utterance_text=utterance.text,
    )
    suggestions = data.get("suggestions", [])
    if not isinstance(suggestions, list):
        logger.warning(
            "LLM returned non-list suggestions for %s", _redact(utterance.text)
        )
        raise SuggestionUnavailable("llm_parse_error")

    importance_raw = data.get("importance")
    if isinstance(importance_raw, bool) or not isinstance(importance_raw, (int, float)):
        # Fail open — voice it, like before the slider. Logged because a model
        # that stops emitting importance silently turns the interject slider
        # into a no-op; this line is how we'd notice from Cloud Run logs.
        logger.info(
            "LLM omitted/invalid importance for %s — failing open to 100",
            _redact(utterance.text),
        )
        importance = 100
    else:
        importance = max(0, min(100, int(importance_raw)))
    return suggestions, importance


async def _generate_nudge(
    llm: LLMClient,
    utterance: Utterance,
    empathy_slider: int,
    role: str,
    voice_profile: dict | None = None,
    tone_context: dict | None = None,
    speaker_name: str | None = None,
    *,
    stream: bool = False,
    stats: dict | None = None,
) -> tuple[str, int]:
    """Call the LLM for a SELF turn; parse the single delivery nudge + urgency.

    Mirrors :func:`_generate_suggestions` but for the coached user's OWN
    speech: :func:`~main.self_feedback_prompt` asks for ONE tiny course-
    correction, or an empty string when the delivery is already fine ("only
    speak when something should change"). Returns ``(nudge, importance)``:

      * an empty/whitespace nudge → ``("", 0)`` — nothing to say, and nothing
        to voice, so importance is forced to 0 regardless of what the model
        emitted alongside the empty nudge.
      * a non-empty nudge with a missing/invalid importance fails OPEN to 100
        (a nudge worth emitting is worth voicing) — the same spirit as
        _generate_suggestions, but only when there actually IS a nudge.

    Raises :class:`SuggestionUnavailable` ("llm_parse_error") when the LLM
    output cannot be parsed (P0-3), with PII-safe logging via :func:`_redact` —
    it never fabricates a nudge.
    """
    from main import self_feedback_prompt

    system = self_feedback_prompt(empathy_slider, role, voice_profile)
    # tone_context renders the phone's measurements as hints (Track 3-server);
    # None keeps the prompt byte-identical. No streaming PREVIEW for a nudge
    # (one short phrase, nothing to preview) — but with ``stream=True`` and
    # a streaming-capable client the call still goes through the hedged
    # stream, so a nudge gets the same first-token tail protection as a
    # suggestion. ``stream=False`` (legacy clients) is the exact old call.
    user_content = _turn_prompt(utterance, tone_context, speaker_name)

    if stream and _supports_streaming(llm):
        raw = await _stream_with_first_suggestion(
            llm, system, user_content, None,
            max_tokens=NUDGE_MAX_TOKENS, stats=stats,
        )
    else:
        raw = await asyncio.to_thread(
            llm.complete, system=system, user=user_content, max_tokens=NUDGE_MAX_TOKENS,
        )

    data = await _parse_or_repair(
        llm, raw, keys='"nudge" (string), "importance" (integer)',
        what="nudge", utterance_text=utterance.text,
    )
    nudge = data.get("nudge", "")
    if not isinstance(nudge, str):
        logger.warning(
            "LLM returned non-string nudge for %s", _redact(utterance.text)
        )
        raise SuggestionUnavailable("llm_parse_error")

    nudge = nudge.strip()
    if not nudge:
        # Delivery is fine — say nothing. importance 0 keeps callers from ever
        # voicing an absent nudge.
        return "", 0

    importance_raw = data.get("importance")
    if isinstance(importance_raw, bool) or not isinstance(importance_raw, (int, float)):
        # Fail open — a nudge the model bothered to emit is worth voicing (see
        # _generate_suggestions for the same rationale/log).
        logger.info(
            "LLM omitted/invalid importance for nudge %s — failing open to 100",
            _redact(utterance.text),
        )
        importance = 100
    else:
        importance = max(0, min(100, int(importance_raw)))
    return nudge, importance
