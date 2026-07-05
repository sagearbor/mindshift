"""Free local speech-to-text via faster-whisper (an OPTIONAL dependency).

Honesty notes
-------------
Whisper is **not** a streaming model: it transcribes complete audio windows,
not a rolling frame-by-frame stream. This transcriber therefore buffers the
incoming raw PCM (int16 little-endian, 16 kHz, mono — the pipeline's wire
contract) and only runs the model when an utterance boundary is likely:

* the buffered audio ends in a stretch of trailing quiet (a simple RMS
  energy gate over the last ~1.2 s, matching the Deepgram path's
  ``utterance_end_ms=1200``), i.e. the speaker probably finished; or
* the buffer hits a hard safety cap (~10 s), so a monologue with no pauses
  still produces output instead of growing unboundedly.

Energy is used ONLY to detect those boundaries and to trim *pure* silence
while nothing has been said yet (so offsets stay accurate and a silent room
is never stored or transcribed). It is never used to decide whether audio is
"worth keeping": once anything above the pure-silence floor arrives, every
sample is retained until it is transcribed — quiet speakers are Whisper's
call to judge (``vad_filter=True``), not this gate's.

Streaming design (mirrors ``DeepgramTranscriber``): ``stream()`` only
enqueues audio and drains already-finalized segments — it never waits for the
model, so the WebSocket receive loop stays responsive (review finding F6). A
single background worker task owns the rolling buffer, detects MID-STREAM
boundaries, runs the blocking ``model.transcribe`` in a thread
(``asyncio.to_thread``) and pushes finished
:class:`~audio_pipeline.TranscriptSegment` objects onto a results queue.

``finish()`` is deliberately NOT routed through that worker. It is called from
the pipeline's ``stop`` handler — off the hot receive loop — so it stops the
worker and transcribes the session's FINAL utterance DIRECTLY
(``await asyncio.to_thread`` in the caller), returning the segments. Routing
the final flush through the worker and awaiting it with
``asyncio.wait_for(task, timeout)`` was the original correctness bug: a real
CPU decode of a few seconds of audio can take tens of seconds (the ``base``
int8 model measured ~13x slower than real time on a laptop), so the wait timed
out, CANCELLED the in-flight ``to_thread`` transcription and DISCARDED its
already-finalized segment. For continuous speech with no >1.2 s internal pause
that final flush is the ONLY flush, so the *entire* transcript came back empty.
A transcription is finite work (O(audio length)) and a running ``to_thread``
cannot actually be interrupted anyway, so ``finish()`` lets it complete and
delivers the words instead of throwing them away; ``close()`` cancels the
worker and abandons any residual audio.

A transient per-utterance decode failure drops only that utterance (logged,
offsets preserved) — it does NOT kill the session. Only ``connect()``-time
failures (package missing, model cannot load) raise
:class:`~audio_pipeline.TranscriberUnavailable`, and the pipeline then
reports transcription unavailable instead of fabricating anything.

The loaded ``WhisperModel`` is cached at module level keyed by
(model_size, device, compute_type): concurrent sessions share ONE model
instead of loading a copy each (faster-whisper models are safe for
concurrent ``transcribe`` calls). ``close()`` drops only the per-connection
reference, never the shared cache entry.

``faster-whisper`` (and its native deps ctranslate2/onnxruntime) is kept OUT
of the base requirements on purpose — see ``requirements-whisper.txt``. It is
imported lazily inside the model loader; when it is not installed,
``connect()`` raises :class:`~audio_pipeline.TranscriberUnavailable`. No
transcript is ever fabricated.

Whisper performs no speaker diarization, so every segment carries
``speaker=None`` — the pipeline's existing alternation heuristic then labels
turns, which is the honest representation of what the model knows.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading

import numpy as np

from audio_pipeline import TranscriberUnavailable, TranscriptSegment

logger = logging.getLogger(__name__)

# Incoming client audio contract (same as the Deepgram path): raw PCM,
# int16 little-endian, mono, 16 kHz, ~100 ms (3200-byte) frames.
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # int16

# Utterance-boundary tuning.
#
# SILENCE_RMS_THRESHOLD is the PURE-SILENCE floor, in int16 RMS units:
# 100 ≈ -50 dBFS — above electrical/ADC noise but safely below even very
# quiet speech (soft speakers commonly measure only ~200-300 RMS; normal
# speech measures thousands). Audio above this floor is always retained and
# transcribed; audio below it only marks boundaries / gets trimmed while
# nothing has been said. Keeping this floor LOW is deliberate: a false
# "that's speech" costs one Whisper pass over near-silence (its own VAD then
# discards it), while a false "that's silence" would silently drop words.
SILENCE_RMS_THRESHOLD = 100.0
# How much trailing quiet counts as "the speaker finished". Aligned with the
# Deepgram path's utterance_end_ms=1200 so brief mid-sentence pauses
# (~0.3-0.8 s) do not split one sentence into two utterances.
UTTERANCE_END_SILENCE_S = 1.2
# Cap on buffered audio before a forced flush (bounds latency and memory for
# pause-free monologues).
MAX_BUFFER_S = 10.0
# finish(): how long to wait for the worker to STOP after being asked to (so
# finish() can safely take over the buffer). This bounds ONLY the handoff — a
# mid-stream flush that happens to be in flight when finish() is called — never
# the final transcription itself (that runs directly in finish() and is allowed
# to complete). Generous because an in-flight flush may still be decoding up to
# MAX_BUFFER_S of audio on a slow CPU; finite so a wedged worker can't hang
# shutdown forever. On timeout the worker is cancelled and the final utterance
# is transcribed directly regardless, so no words depend on this bound.
FINISH_TIMEOUT_S = 30.0

# WHISPER_MODEL size trade-offs (approximate, English, CPU int8):
#   tiny   — fastest, least accurate; fine for smoke tests / weak hardware
#   base   — DEFAULT: good accuracy/speed balance, near-realtime on laptops
#   small  — noticeably better accuracy, ~2-3x slower than base
#   medium — best of the practical CPU sizes, but slow without a GPU
DEFAULT_MODEL_SIZE = "base"
# int8 quantization: big speed/memory win on CPU with a small,
# well-documented accuracy cost; "auto" picks GPU when present.
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "int8"

# Module-level model cache: one WhisperModel per (size, device, compute_type)
# shared by every connection, loaded once under a lock (a threading.Lock —
# loads happen inside asyncio.to_thread worker threads). close() never evicts
# entries: other live sessions may be using the model.
_MODEL_CACHE: dict[tuple[str, str, str], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Sentinel enqueued by finish(): tells the worker to flush what remains and exit.
_FINISH = object()


def _rms(pcm: bytes) -> float:
    """RMS energy of raw int16-LE PCM, in int16 units (0..32767-ish)."""
    # Frames are int16 so an even byte count is guaranteed by the wire
    # contract; truncate defensively so a malformed frame can't crash us.
    pcm = pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)]
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    return float(np.sqrt(np.mean(np.square(samples))))


def _confidence(avg_logprob: float | None) -> float:
    """Map Whisper's ``avg_logprob`` to a 0-1 score, honestly documented.

    ``avg_logprob`` is the mean per-token log-probability (≤ 0);
    ``exp(avg_logprob)`` is therefore the geometric-mean token probability —
    a monotone, interpretable 0-1 transform, NOT a calibrated confidence.
    ``None`` (segment carried no score) maps to the dataclass default 1.0,
    matching how other backends omit per-segment confidence.
    """
    if avg_logprob is None:
        return 1.0
    return max(0.0, min(1.0, math.exp(avg_logprob)))


class WhisperTranscriber:
    """Local Whisper transcription satisfying the pipeline's transcriber
    interface: ``connect()`` / ``stream()`` / ``finish()`` / ``close()``
    returning ``list[TranscriptSegment]``.

    Zero-arg-callable (all constructor args are optional), so the class
    itself can be installed as ``app.state.transcriber_factory``.

    ``model`` is injectable purely for tests: pass a double exposing
    ``transcribe(audio, **kwargs) -> (segments, info)`` to exercise the
    buffering/flush logic without the heavy real model. An injected model
    bypasses the module-level cache entirely. Production code leaves it
    ``None`` and the real (shared, cached) model is resolved in ``connect()``.
    """

    def __init__(
        self,
        model=None,
        model_size: str | None = None,
        silence_rms_threshold: float = SILENCE_RMS_THRESHOLD,
        utterance_end_silence_s: float = UTTERANCE_END_SILENCE_S,
        max_buffer_s: float = MAX_BUFFER_S,
    ) -> None:
        self._model = model
        self._model_size = (
            model_size or os.getenv("WHISPER_MODEL", "").strip() or DEFAULT_MODEL_SIZE
        )
        self._silence_rms = silence_rms_threshold
        self._utterance_end_silence_s = utterance_end_silence_s
        self._max_buffer_s = max_buffer_s
        # Rolling PCM buffer + the absolute sample index (since session start)
        # of the buffer's first sample — segment times are offset by this so
        # they are REAL session-relative timings, not per-buffer ones.
        # Owned EXCLUSIVELY by the worker task once connect() starts it.
        self._buffer = bytearray()
        self._buffer_start_sample = 0
        self._heard_speech = False
        self._connected = False
        self._failure: str | None = None
        # stream() → worker: raw PCM chunks (or the _FINISH sentinel).
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        # worker → stream()/finish(): finalized segments, in utterance order.
        self._results: asyncio.Queue[TranscriptSegment] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    # -- lifecycle ------------------------------------------------------------

    async def connect(self) -> None:
        """Resolve the (shared, cached) Whisper model and start the worker.

        Model load (and a possible first-time download) is slow — it runs in
        a worker thread so the event loop stays live, and the result is
        cached at module level so concurrent sessions load it ONCE.

        Raises :class:`TranscriberUnavailable` when faster-whisper is not
        installed or the model cannot be loaded — the pipeline then reports
        transcription unavailable instead of fabricating anything. These
        connect()-time failures are the ONLY unrecoverable ones; per-utterance
        decode errors later never kill the session.
        """
        if self._model is None:
            try:
                self._model = await asyncio.to_thread(self._load_model)
            except TranscriberUnavailable:
                raise
            except Exception as exc:
                raise TranscriberUnavailable(
                    f"Could not load Whisper model {self._model_size!r}: {exc}"
                ) from exc
        self._failure = None
        self._buffer = bytearray()
        self._buffer_start_sample = 0
        self._heard_speech = False
        self._audio_queue = asyncio.Queue()
        self._results = asyncio.Queue()
        self._connected = True
        self._worker_task = asyncio.create_task(self._worker_loop())

    def _load_model(self):
        """Return the shared cached model, importing/loading lazily (blocking).

        Runs inside ``asyncio.to_thread``; the threading lock closes the race
        where two sessions connect at once and would both load a model.
        Overridable in tests; NEVER import faster_whisper at module top —
        the base install must keep working without it.
        """
        key = (self._model_size, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
        with _MODEL_CACHE_LOCK:
            model = _MODEL_CACHE.get(key)
            if model is not None:
                return model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriberUnavailable(
                    "faster-whisper not installed — pip install faster-whisper "
                    "(or pip install -r requirements-whisper.txt), "
                    "or use STT_PROVIDER=deepgram"
                ) from exc
            model = WhisperModel(
                self._model_size,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            _MODEL_CACHE[key] = model
            return model

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def close(self) -> None:
        """Cancel the worker and drop per-connection state. Idempotent,
        never raises.

        Deliberately does NOT touch the module-level model cache — the model
        is shared with other live sessions; only this connection's reference
        and buffers are released.
        """
        self._connected = False
        task, self._worker_task = self._worker_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._model = None
        self._buffer.clear()
        self._heard_speech = False

    # -- streaming ------------------------------------------------------------

    async def stream(self, audio_bytes: bytes) -> list[TranscriptSegment]:
        """Enqueue *audio_bytes* for the worker; return finalized segments.

        Never waits for a transcription: the model runs on the background
        worker, so this call stays fast and the pipeline's WebSocket receive
        loop is never blocked mid-utterance. Returns ``[]`` while nothing has
        been finalized yet.

        Raises :class:`TranscriberUnavailable` only when the transcriber is
        genuinely unusable (not connected / worker died) — never for a
        transient decode failure, which drops just that utterance. Segments
        finalized before a failure are still delivered first.
        """
        if self._failure:
            segments = self._drain_results()
            if segments:
                return segments
            self._connected = False
            raise TranscriberUnavailable(self._failure)
        if not self._connected or self._model is None:
            raise TranscriberUnavailable("Whisper transcriber is not connected")
        self._audio_queue.put_nowait(bytes(audio_bytes))
        # Yield once so the worker can pick up already-queued audio.
        await asyncio.sleep(0)
        return self._drain_results()

    async def finish(self) -> list[TranscriptSegment]:
        """Flush the remaining audio and return all outstanding segments.

        Mirrors ``DeepgramTranscriber.finish()``: this is how the LAST
        utterance of a session gets delivered instead of dropped. The worker is
        told to stop (so this call can own the buffer exclusively), then the
        final utterance is transcribed DIRECTLY here — not routed through the
        worker under a cancel-on-timeout wait, which used to discard a decode
        that ran longer than the timeout. Called from the pipeline's ``stop``
        handler (off the hot receive loop), so a direct blocking transcribe is
        fine here.

        Idempotent; after ``finish()``, :meth:`close` is a safe no-op. Never
        raises (the pipeline calls it during cleanup); on failure the honest
        result is simply no further segments.
        """
        self._connected = False
        task, self._worker_task = self._worker_task, None
        if task is not None and not task.done():
            # Ask the worker to stop. It drains any earlier-queued audio first
            # (FIFO), so when it sees _FINISH the buffer holds exactly the
            # post-last-boundary residual, which we transcribe below. The only
            # thing we wait on is a mid-stream flush already in flight; bound it
            # so a wedged worker can't hang shutdown (on timeout wait_for
            # cancels the worker — safe, since the final utterance is
            # transcribed directly regardless).
            self._audio_queue.put_nowait(_FINISH)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=FINISH_TIMEOUT_S)
        # The worker has stopped touching the buffer: deliver the mid-stream
        # segments it already finalized, then the final utterance (transcribed
        # directly, allowed to run to completion so its words aren't dropped).
        segments = self._drain_results()
        segments += await self._flush()
        self._buffer.clear()
        self._heard_speech = False
        return segments

    def _drain_results(self) -> list[TranscriptSegment]:
        segments: list[TranscriptSegment] = []
        while True:
            try:
                segments.append(self._results.get_nowait())
            except asyncio.QueueEmpty:
                return segments

    # -- background worker ----------------------------------------------------

    async def _worker_loop(self) -> None:
        """Single owner of the rolling buffer: ingest → boundary → transcribe.

        Exits when it sees the ``finish()`` sentinel (leaving the residual
        buffer for ``finish()`` to transcribe directly) or when cancelled by
        ``close()``. Any OTHER exception is a genuine malfunction: it is
        recorded so the next ``stream()`` reports the transcriber unavailable
        instead of silently swallowing audio.
        """
        try:
            while True:
                chunk = await self._audio_queue.get()
                try:
                    if chunk is _FINISH:
                        # Stop here; finish() owns the residual buffer now and
                        # transcribes it directly (the final flush is NOT run in
                        # the worker, so it can't be cancelled by finish()'s
                        # bounded wait).
                        return
                    await self._ingest(chunk)
                finally:
                    self._audio_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Whisper worker failed")
            self._failure = f"Whisper worker failed: {exc}"
            self._connected = False

    async def _ingest(self, chunk: bytes) -> None:
        """Append *chunk* to the rolling buffer and flush on a boundary.

        ALL audio is retained until transcribed (up to the max-window cap);
        energy only decides WHEN to flush and lets leading pure silence be
        trimmed while nothing has been said yet. Segments from a mid-stream
        flush are pushed onto the results queue for ``stream()`` to drain.
        """
        self._buffer.extend(chunk)
        if not self._heard_speech and _rms(chunk) >= self._silence_rms:
            self._heard_speech = True

        if self._buffer_seconds() >= self._max_buffer_s:
            self._emit(await self._flush())  # safety cap — pause-free monologue
        elif self._heard_speech and self._ends_in_silence():
            self._emit(await self._flush())  # speaker went quiet — utterance end
        elif not self._heard_speech:
            # Nothing above the pure-silence floor so far: keep only the
            # trailing window so a silent room is neither stored forever nor
            # ever transcribed, while a just-starting word keeps its lead-in.
            self._trim_silent_buffer()

    # -- internals ------------------------------------------------------------

    def _buffer_seconds(self) -> float:
        return len(self._buffer) / BYTES_PER_SAMPLE / SAMPLE_RATE

    def _window_bytes(self) -> int:
        return int(self._utterance_end_silence_s * SAMPLE_RATE) * BYTES_PER_SAMPLE

    def _ends_in_silence(self) -> bool:
        """True when the buffer extends past the trailing-silence window and
        that whole window is below the RMS floor."""
        window = self._window_bytes()
        if len(self._buffer) <= window:
            return False
        return _rms(bytes(self._buffer[-window:])) < self._silence_rms

    def _trim_silent_buffer(self) -> None:
        """Drop leading all-silence audio, keeping the trailing window (so a
        just-starting word keeps its lead-in) and advancing the absolute
        sample offset to stay time-accurate. Only ever called while NOTHING
        above the pure-silence floor is buffered — real audio is never trimmed."""
        excess = len(self._buffer) - self._window_bytes()
        if excess <= 0:
            return
        excess -= excess % BYTES_PER_SAMPLE
        del self._buffer[:excess]
        self._buffer_start_sample += excess // BYTES_PER_SAMPLE

    def _emit(self, segments: list[TranscriptSegment]) -> None:
        """Push a mid-stream flush's segments onto the results queue for
        ``stream()`` to drain (finish() returns its segments directly)."""
        for seg in segments:
            self._results.put_nowait(seg)

    async def _flush(self) -> list[TranscriptSegment]:
        """Transcribe the buffered utterance and RETURN its segments.

        The buffer and offsets are reset FIRST, so a decode failure cannot
        corrupt session timing: the broken utterance is dropped (logged
        honestly) and the session continues — a transient error must not kill
        transcription for the rest of the call (the pipeline treats
        ``TranscriberUnavailable`` from ``stream()`` as terminal).

        The blocking model call runs in a worker thread and is allowed to run
        to completion: it is finite work (O(audio length)) and a running
        ``to_thread`` cannot be interrupted, so bounding it with a
        cancel-on-timeout wait would only leak the thread AND drop a segment
        that was about to be delivered — the original bug. Callers that must
        stay responsive (``stream()``) never await a flush; only the worker and
        ``finish()`` do, both off the receive loop.

        Skips the model entirely when the buffer holds nothing above the
        pure-silence floor (e.g. ``finish()`` right after a flush): there is
        nothing to transcribe, so no inference is wasted and nothing can be
        fabricated from silence.
        """
        pcm = bytes(self._buffer)
        pcm = pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)]
        heard = self._heard_speech
        offset_s = self._buffer_start_sample / SAMPLE_RATE
        self._buffer_start_sample += len(pcm) // BYTES_PER_SAMPLE
        self._buffer.clear()
        self._heard_speech = False
        if not pcm or not heard:
            return []

        # int16 PCM → float32 normalized to [-1, 1], as Whisper expects.
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        try:
            raw_segments = await asyncio.to_thread(self._transcribe, audio)
        except Exception:
            logger.warning(
                "Whisper transcription failed — dropping this %.1fs utterance "
                "and continuing the session",
                len(pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE,
                exc_info=True,
            )
            return []

        segments: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append(TranscriptSegment(
                text=text,
                # Whisper times are buffer-relative; offset to session time.
                start_time=float(seg.start) + offset_s,
                end_time=float(seg.end) + offset_s,
                speaker=None,  # Whisper has no diarization — never invent one
                confidence=_confidence(getattr(seg, "avg_logprob", None)),
            ))
        return segments

    def _transcribe(self, audio: np.ndarray) -> list:
        """Blocking model call — runs in a worker thread via ``_flush``."""
        segments, _info = self._model.transcribe(
            audio, language="en", vad_filter=True,
        )
        # faster-whisper returns a LAZY generator: materialize it here, still
        # inside the worker thread, so no decoding happens on the event loop.
        return list(segments)
