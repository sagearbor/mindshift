"""Tests for the free local WhisperTranscriber (server/whisper_transcriber.py).

faster-whisper is an OPTIONAL dependency: everything here must pass WITHOUT
it installed. The buffering / silence-boundary / offset logic is exercised
with an injected fake model; the real model only runs in the opt-in smoke
test (requires faster-whisper importable AND RUN_WHISPER_SMOKE=1).

The transcriber runs transcription on a background worker task (mirroring
DeepgramTranscriber): ``stream()`` only enqueues audio and drains finished
segments. Tests therefore wait for the worker to settle (``_audio_queue.join``)
before asserting on what the model saw.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
import os
import sys
import threading
import types
from dataclasses import dataclass

import numpy as np
import pytest

import whisper_transcriber
from audio_pipeline import TranscriberUnavailable, TranscriptSegment
from whisper_transcriber import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    SILENCE_RMS_THRESHOLD,
    UTTERANCE_END_SILENCE_S,
    WhisperTranscriber,
    _confidence,
    _rms,
)

FRAME_BYTES = 3200  # ~100 ms of int16 @ 16 kHz — the pipeline's wire framing
SPEECH_AMPLITUDE = 8000  # int16 sine amplitude — RMS ~5657, far above the floor
# A genuinely soft speaker: RMS ≈ 310/√2 ≈ 219 — BELOW the old 300 gate that
# silently dropped quiet speech (F1), above the pure-silence floor.
QUIET_AMPLITUDE = 310
# Enough trailing silence to cross the utterance-end window (1.2 s).
UTTERANCE_GAP_S = UTTERANCE_END_SILENCE_S


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    """Keep the module-level model cache isolated between tests."""
    whisper_transcriber._MODEL_CACHE.clear()
    yield
    whisper_transcriber._MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# PCM helpers — synthesized audio, honestly labelled (a sine is not speech;
# it only stands in for "loud" energy for the boundary detector).
# ---------------------------------------------------------------------------

def pcm(seconds: float, amplitude: int = 0) -> bytes:
    """Raw int16-LE 16 kHz mono PCM: silence (amplitude=0) or a 440 Hz sine."""
    n = int(seconds * SAMPLE_RATE)
    if amplitude == 0:
        return b"\x00\x00" * n
    t = np.arange(n)
    wave = (amplitude * np.sin(2 * np.pi * 440 * t / SAMPLE_RATE)).astype("<i2")
    return wave.tobytes()


def frames(data: bytes, size: int = FRAME_BYTES):
    for i in range(0, len(data), size):
        yield data[i:i + size]


async def stream_all(t: WhisperTranscriber, data: bytes) -> list[TranscriptSegment]:
    """Stream *data* frame by frame, then wait for the background worker to
    process everything enqueued so far and drain the finished segments."""
    out: list[TranscriptSegment] = []
    for frame in frames(data):
        out.extend(await t.stream(frame))
    await t._audio_queue.join()  # worker has consumed every queued frame
    out.extend(t._drain_results())
    return out


# ---------------------------------------------------------------------------
# Fake model — a double for faster_whisper.WhisperModel, so the buffering
# logic is testable without the heavy dependency.
# ---------------------------------------------------------------------------

@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = -0.2


class FakeWhisperModel:
    """Records every transcribe() call; returns canned segments each time,
    mimicking faster-whisper's (lazy segment iterator, info) return shape."""

    def __init__(self, segments: list[FakeSegment] | None = None) -> None:
        self.segments = segments if segments is not None else [
            FakeSegment(0.5, 1.5, " hello there"),
        ]
        self.calls: list[tuple[np.ndarray, dict]] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter(list(self.segments)), {"language": "en"}


async def connected(model=None, **kwargs) -> WhisperTranscriber:
    t = WhisperTranscriber(model=model or FakeWhisperModel(), **kwargs)
    await t.connect()
    return t


# ---------------------------------------------------------------------------
# Gate: without faster-whisper, connect() is honestly unavailable
# ---------------------------------------------------------------------------

class TestAvailabilityGate:
    @pytest.mark.anyio
    async def test_connect_without_faster_whisper_raises_unavailable(self, monkeypatch):
        # None in sys.modules forces `from faster_whisper import ...` to raise
        # ImportError — simulating the package being absent even on machines
        # that have it installed.
        monkeypatch.setitem(sys.modules, "faster_whisper", None)
        t = WhisperTranscriber()
        with pytest.raises(TranscriberUnavailable) as excinfo:
            await t.connect()
        assert "faster-whisper" in str(excinfo.value)
        assert "STT_PROVIDER=deepgram" in str(excinfo.value)

    @pytest.mark.anyio
    async def test_stream_before_connect_raises_unavailable(self):
        t = WhisperTranscriber(model=FakeWhisperModel())
        with pytest.raises(TranscriberUnavailable):
            await t.stream(pcm(0.1, SPEECH_AMPLITUDE))

    @pytest.mark.anyio
    async def test_model_load_failure_raises_unavailable(self):
        class Exploding(WhisperTranscriber):
            def _load_model(self):
                raise RuntimeError("download failed")

        t = Exploding()
        with pytest.raises(TranscriberUnavailable) as excinfo:
            await t.connect()
        assert "download failed" in str(excinfo.value)


# ---------------------------------------------------------------------------
# F3 — module-level model cache: one shared model across connections
# ---------------------------------------------------------------------------

def _fake_faster_whisper(loads: list):
    """A stand-in faster_whisper module whose WhisperModel counts loads."""
    module = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, model_size, device=None, compute_type=None):
            loads.append((model_size, device, compute_type))

        def transcribe(self, audio, **kwargs):
            return iter([FakeSegment(0.0, 1.0, " cached hello")]), {}

    module.WhisperModel = WhisperModel
    return module


class TestModelCache:
    @pytest.mark.anyio
    async def test_same_config_connections_share_one_model(self, monkeypatch):
        loads: list = []
        monkeypatch.setitem(sys.modules, "faster_whisper", _fake_faster_whisper(loads))
        t1 = WhisperTranscriber(model_size="base")
        t2 = WhisperTranscriber(model_size="base")
        await t1.connect()
        await t2.connect()
        try:
            assert t1._model is t2._model  # ONE model object shared
            assert len(loads) == 1  # the loader ran exactly once
        finally:
            await t1.close()
            await t2.close()

    @pytest.mark.anyio
    async def test_close_does_not_evict_shared_model(self, monkeypatch):
        loads: list = []
        monkeypatch.setitem(sys.modules, "faster_whisper", _fake_faster_whisper(loads))
        t1 = WhisperTranscriber(model_size="base")
        await t1.connect()
        shared = t1._model
        await t1.close()  # session 1 ends...
        assert t1._model is None  # ...its own reference is dropped
        # ...but the cache still holds the model for other/new sessions:
        t2 = WhisperTranscriber(model_size="base")
        await t2.connect()
        try:
            assert t2._model is shared
            assert len(loads) == 1  # no reload after the close
        finally:
            await t2.close()

    @pytest.mark.anyio
    async def test_injected_fake_model_bypasses_cache(self):
        t = await connected(FakeWhisperModel())
        assert whisper_transcriber._MODEL_CACHE == {}
        await t.close()


# ---------------------------------------------------------------------------
# Buffering / silence-boundary flush / absolute time offsets (fake model)
# ---------------------------------------------------------------------------

class TestBufferingAndFlush:
    @pytest.mark.anyio
    async def test_no_flush_while_speech_is_ongoing(self):
        model = FakeWhisperModel()
        t = await connected(model)
        assert await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE)) == []
        assert model.calls == []  # model never invoked without a boundary

    @pytest.mark.anyio
    async def test_trailing_silence_flushes_one_utterance(self):
        model = FakeWhisperModel([FakeSegment(0.5, 1.5, " hello there")])
        t = await connected(model)
        out = await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE))
        out += await stream_all(t, pcm(UTTERANCE_GAP_S, 0))  # trailing-silence boundary
        assert len(out) == 1
        seg = out[0]
        assert seg.text == "hello there"  # whitespace stripped
        assert seg.speaker is None  # Whisper has no diarization — never invented
        # First buffer starts at session sample 0 → model-relative times kept.
        assert seg.start_time == pytest.approx(0.5)
        assert seg.end_time == pytest.approx(1.5)
        assert len(model.calls) == 1
        audio, kwargs = model.calls[0]
        # int16 → float32 normalized to [-1, 1]
        assert audio.dtype == np.float32
        assert float(np.max(np.abs(audio))) <= SPEECH_AMPLITUDE / 32768.0 + 1e-6
        # 1.0 s speech + the 1.2 s utterance-end window buffered at flush time
        assert audio.shape[0] == int((1.0 + UTTERANCE_GAP_S) * SAMPLE_RATE)
        assert kwargs.get("vad_filter") is True
        assert kwargs.get("language") == "en"

    @pytest.mark.anyio
    async def test_quiet_speech_is_buffered_and_transcribed(self):
        """F1 regression: a soft speaker (RMS ~220, below the OLD 300 gate)
        must still reach the model — energy decides WHEN to flush, never
        whether audio is worth keeping."""
        assert QUIET_AMPLITUDE / math.sqrt(2) > SILENCE_RMS_THRESHOLD
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, " quiet words")])
        t = await connected(model)
        out = await stream_all(
            t, pcm(1.0, QUIET_AMPLITUDE) + pcm(UTTERANCE_GAP_S, 0),
        )
        assert [s.text for s in out] == ["quiet words"]
        assert len(model.calls) == 1
        audio, _ = model.calls[0]
        # The model received the quiet speech itself, not a trimmed remnant.
        assert audio.size >= int(1.0 * SAMPLE_RATE)
        assert float(np.max(np.abs(audio))) > 0.0

    @pytest.mark.anyio
    async def test_second_utterance_gets_absolute_session_offset(self):
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, " again")])
        t = await connected(model)
        gap = UTTERANCE_GAP_S
        # Utterance 1: 1.0 s speech + 1.2 s silence → flush at 2.2 s.
        first = await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(gap, 0))
        # Utterance 2 immediately after: its buffer starts at 2.2 s absolute.
        second = await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(gap, 0))
        assert [s.start_time for s in first] == [pytest.approx(0.0)]
        assert [s.start_time for s in second] == [pytest.approx(1.0 + gap)]
        assert [s.end_time for s in second] == [pytest.approx(2.0 + gap)]

    @pytest.mark.anyio
    async def test_max_buffer_cap_flushes_pause_free_speech(self):
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, " nonstop")])
        t = await connected(model, max_buffer_s=1.0)
        out = await stream_all(t, pcm(2.0, SPEECH_AMPLITUDE))  # never silent
        assert len(model.calls) == 2  # cap hit at 1.0 s, twice
        assert [s.start_time for s in out] == [pytest.approx(0.0), pytest.approx(1.0)]

    @pytest.mark.anyio
    async def test_pure_silence_never_reaches_the_model(self):
        model = FakeWhisperModel()
        t = await connected(model)
        assert await stream_all(t, pcm(5.0, 0)) == []
        assert model.calls == []  # nothing to transcribe — nothing fabricated
        # ...and the silent buffer stays trimmed instead of growing forever.
        assert len(t._buffer) <= t._window_bytes() + FRAME_BYTES

    @pytest.mark.anyio
    async def test_leading_silence_keeps_offsets_accurate(self):
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, " late start")])
        t = await connected(model)
        out = await stream_all(t, pcm(2.0, 0))  # 2 s of nothing (trimmed)
        out += await stream_all(
            t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(UTTERANCE_GAP_S, 0),
        )
        assert len(out) == 1
        # The flushed buffer began 1.2 s (the kept trailing window) before the
        # speech: absolute offset = 2.0 - 1.2 = 0.8 s.
        assert out[0].start_time == pytest.approx(2.0 - UTTERANCE_END_SILENCE_S)

    @pytest.mark.anyio
    async def test_short_midsentence_pause_does_not_split_utterance(self):
        """F7 regression: a 0.5 s breath inside one sentence must NOT flush;
        only a gap past the 1.2 s utterance-end window may."""
        model = FakeWhisperModel([FakeSegment(0.0, 2.5, " one whole sentence")])
        t = await connected(model)
        out = await stream_all(
            t,
            pcm(1.0, SPEECH_AMPLITUDE) + pcm(0.5, 0) + pcm(1.0, SPEECH_AMPLITUDE),
        )
        assert out == []
        assert model.calls == []  # the pause did not split the utterance
        # A >1.2 s trailing gap IS an utterance end: ONE flush of the whole thing.
        out = await stream_all(t, pcm(1.3, 0))
        assert [s.text for s in out] == ["one whole sentence"]
        assert len(model.calls) == 1
        audio, _ = model.calls[0]
        # speech + internal gap + speech + the 1.2 s window, all in one buffer
        assert audio.shape[0] == int((1.0 + 0.5 + 1.0 + 1.2) * SAMPLE_RATE)

    @pytest.mark.anyio
    async def test_empty_text_segments_are_dropped(self):
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, "   ")])
        t = await connected(model)
        out = await stream_all(
            t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(UTTERANCE_GAP_S, 0),
        )
        assert out == []
        assert len(model.calls) == 1  # model ran; its empty output stayed empty

    @pytest.mark.anyio
    async def test_transient_decode_error_drops_utterance_not_session(self):
        """F2 regression: one failing transcribe() drops ONE utterance and the
        session keeps going — it must not become terminally unavailable."""
        class FlakyModel:
            def __init__(self):
                self.calls = 0

            def transcribe(self, audio, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("decoder blew up")
                return iter([FakeSegment(0.0, 1.0, " recovered")]), {}

        model = FlakyModel()
        t = await connected(model)
        gap = UTTERANCE_GAP_S
        # First utterance: transcribe raises → honestly dropped, no exception.
        out = await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(gap, 0))
        assert out == []
        assert t.is_connected  # NOT marked terminally unavailable
        # Second utterance: works, with UNCORRUPTED absolute offsets — the
        # dropped buffer (1.0 s speech + 1.2 s silence) still advanced time.
        out = await stream_all(t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(gap, 0))
        assert [s.text for s in out] == ["recovered"]
        assert out[0].start_time >= 1.0 + gap
        assert model.calls == 2


# ---------------------------------------------------------------------------
# F6 — stream() never blocks on a transcription in progress
# ---------------------------------------------------------------------------

class TestNonBlockingStream:
    @pytest.mark.anyio
    async def test_stream_returns_promptly_while_model_is_busy(self):
        release = threading.Event()
        started = threading.Event()

        class BlockingModel(FakeWhisperModel):
            def transcribe(self, audio, **kwargs):
                started.set()
                # Simulate a slow decode: hold the worker thread until the
                # test releases it. stream() must not be affected.
                assert release.wait(timeout=10), "test never released the model"
                return super().transcribe(audio, **kwargs)

        model = BlockingModel([FakeSegment(0.0, 0.4, " chunk")])
        # 0.5 s cap → the first 5 frames trigger a (blocked) flush.
        t = await connected(model, max_buffer_s=0.5)
        try:
            for frame in frames(pcm(0.5, SPEECH_AMPLITUDE)):
                await t.stream(frame)
            # The worker is now stuck inside transcribe()...
            await asyncio.to_thread(started.wait, 5)
            # ...yet further stream() calls return promptly (they only enqueue).
            for frame in frames(pcm(0.5, SPEECH_AMPLITUDE)):
                out = await asyncio.wait_for(t.stream(frame), timeout=1.0)
                assert out == []  # nothing finalized while the model is busy
        finally:
            release.set()
        # Once released the worker catches up: both flushes surface, in order,
        # with correct ABSOLUTE session offsets (0.0 s and 0.5 s).
        await t._audio_queue.join()
        out = t._drain_results() + await t.finish()
        assert [s.text for s in out] == ["chunk", "chunk"]
        assert [s.start_time for s in out] == [pytest.approx(0.0), pytest.approx(0.5)]
        assert len(model.calls) == 2
        # No queued audio was lost while the model was busy.
        total = sum(a.shape[0] for a, _ in model.calls)
        assert total == int(1.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# finish() — the last utterance is drained, not dropped
# ---------------------------------------------------------------------------

class TestFinish:
    @pytest.mark.anyio
    async def test_finish_flushes_remaining_buffer(self):
        model = FakeWhisperModel([FakeSegment(0.0, 0.5, " last words")])
        t = await connected(model)
        # Speech with NO trailing silence — stream() alone would never flush.
        assert await stream_all(t, pcm(0.5, SPEECH_AMPLITUDE)) == []
        out = await t.finish()
        assert [s.text for s in out] == ["last words"]
        assert len(model.calls) == 1

    @pytest.mark.anyio
    async def test_finish_twice_is_safe_and_empty(self):
        model = FakeWhisperModel([FakeSegment(0.0, 0.5, " once")])
        t = await connected(model)
        await stream_all(t, pcm(0.5, SPEECH_AMPLITUDE))
        assert len(await t.finish()) == 1
        assert await t.finish() == []  # buffer already drained — nothing invented

    @pytest.mark.anyio
    async def test_finish_skips_inference_on_pure_silence_residual(self):
        """F8 regression: speak → pause (flush) → silence → finish() must not
        waste a model pass on audio that is nothing but trimmed silence."""
        model = FakeWhisperModel([FakeSegment(0.0, 1.0, " spoken")])
        t = await connected(model)
        out = await stream_all(
            t, pcm(1.0, SPEECH_AMPLITUDE) + pcm(UTTERANCE_GAP_S, 0),
        )
        assert [s.text for s in out] == ["spoken"]
        assert len(model.calls) == 1  # the pause flushed the utterance
        assert await stream_all(t, pcm(2.0, 0)) == []  # residual: pure silence
        assert await t.finish() == []
        assert len(model.calls) == 1  # ZERO extra transcriptions on silence

    @pytest.mark.anyio
    async def test_finish_never_raises_even_when_model_fails(self):
        class FailingModel:
            def transcribe(self, audio, **kwargs):
                raise RuntimeError("boom")

        t = await connected(FailingModel())
        await stream_all(t, pcm(0.5, SPEECH_AMPLITUDE))
        assert await t.finish() == []  # honest empty result, no exception

    @pytest.mark.anyio
    async def test_close_is_idempotent_and_stream_after_close_unavailable(self):
        t = await connected(FakeWhisperModel())
        await t.close()
        await t.close()  # double-close must not raise
        assert t.is_connected is False
        with pytest.raises(TranscriberUnavailable):
            await t.stream(pcm(0.1, SPEECH_AMPLITUDE))


# ---------------------------------------------------------------------------
# Unit helpers — confidence mapping and RMS
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_confidence_is_exp_of_avg_logprob_clamped(self):
        assert _confidence(-0.5) == pytest.approx(math.exp(-0.5))
        assert _confidence(0.0) == 1.0
        assert _confidence(-50.0) == pytest.approx(0.0, abs=1e-6)
        assert _confidence(None) == 1.0  # dataclass default when unscored

    def test_rms_silence_vs_speech(self):
        assert _rms(pcm(0.1, 0)) == 0.0
        assert _rms(pcm(0.1, SPEECH_AMPLITUDE)) == pytest.approx(
            SPEECH_AMPLITUDE / math.sqrt(2), rel=0.05,
        )
        assert _rms(b"") == 0.0
        assert _rms(b"\x01") == 0.0  # odd byte truncated, not crashed

    def test_env_selects_model_size(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL", "small")
        assert WhisperTranscriber()._model_size == "small"
        monkeypatch.delenv("WHISPER_MODEL")
        assert WhisperTranscriber()._model_size == "base"

    def test_bytes_per_sample_matches_int16(self):
        assert BYTES_PER_SAMPLE == 2

    def test_utterance_end_window_matches_deepgram(self):
        # Deepgram runs with utterance_end_ms=1200; the local path must not
        # split sentences faster than the paid path does.
        assert UTTERANCE_END_SILENCE_S == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# Provider selection — STT_PROVIDER wiring in the app lifespan
# ---------------------------------------------------------------------------

def _clear_factory(app) -> None:
    if hasattr(app.state, "transcriber_factory"):
        delattr(app.state, "transcriber_factory")


class TestProviderSelection:
    def test_whisper_provider_installs_factory(self, monkeypatch):
        import main

        monkeypatch.setenv("STT_PROVIDER", "whisper")
        _clear_factory(main.app)
        try:
            main._configure_stt(main.app)
            assert main.app.state.transcriber_factory is WhisperTranscriber
        finally:
            _clear_factory(main.app)

    def test_default_leaves_deepgram_fallback(self, monkeypatch):
        import main

        monkeypatch.delenv("STT_PROVIDER", raising=False)
        _clear_factory(main.app)
        try:
            main._configure_stt(main.app)
            # Unset factory → the pipeline falls back to DeepgramTranscriber.
            assert not hasattr(main.app.state, "transcriber_factory")
        finally:
            _clear_factory(main.app)

    def test_unknown_provider_warns_and_defaults_to_deepgram(self, monkeypatch, caplog):
        import main

        monkeypatch.setenv("STT_PROVIDER", "banana")
        _clear_factory(main.app)
        try:
            with caplog.at_level(logging.WARNING, logger="main"):
                main._configure_stt(main.app)
            assert not hasattr(main.app.state, "transcriber_factory")
            assert any("STT_PROVIDER" in rec.message for rec in caplog.records)
        finally:
            _clear_factory(main.app)

    def test_lifespan_installs_whisper_factory(self, monkeypatch):
        """End-to-end: starting the app with STT_PROVIDER=whisper installs the
        factory via the real lifespan (no transcriber connection is made)."""
        from starlette.testclient import TestClient

        import main

        monkeypatch.setenv("STT_PROVIDER", "whisper")
        _clear_factory(main.app)
        try:
            with TestClient(main.app):
                assert main.app.state.transcriber_factory is WhisperTranscriber
        finally:
            _clear_factory(main.app)


# ---------------------------------------------------------------------------
# Real end-to-end smoke test — opt-in only, honest about what it asserts
# ---------------------------------------------------------------------------

HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None


@pytest.mark.skipif(
    not (HAS_FASTER_WHISPER and os.getenv("RUN_WHISPER_SMOKE") == "1"),
    reason="real-model smoke test: needs faster-whisper installed and RUN_WHISPER_SMOKE=1",
)
@pytest.mark.anyio
async def test_real_whisper_smoke():
    """Load the real model and run synthesized PCM through the full
    connect/stream/finish/close cycle. A sine tone is not speech, so no
    recognized words are asserted — only that the real path executes and
    returns honestly-shaped output."""
    t = WhisperTranscriber(model_size=os.getenv("WHISPER_MODEL", "tiny"))
    await t.connect()
    try:
        out = await stream_all(t, pcm(1.0, 6000) + pcm(1.0, 0))
        out += await t.finish()
        for seg in out:
            assert isinstance(seg, TranscriptSegment)
            assert seg.speaker is None
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.end_time >= seg.start_time >= 0.0
    finally:
        await t.close()
