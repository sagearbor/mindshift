# Ported from gauge@2157433 server/tests/test_post_episode.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# SCOPE NOTE (Task B10, RESOLVED by Task B11): gauge's test_post_episode.py
# also covers `server/main.py`'s `create_app`/`_build_transcriber`/
# `_build_llm` (the POST /episodes/{id}/analyze endpoint + Settings-driven
# backend selection) and `server/ws_ingest.py`'s `_on_episode_end`
# fire-and-forget wrapper. B10's target files were ONLY post_session.py and
# diarize.py, so `TestAnalyzeEndpoint`, `TestBuildTranscriberAndLLM`, and
# `TestOnEpisodeEndFireAndForget` (below) were DEFERRED to B11 rather than
# ported alongside the rest of this file — see task-B10-report.md's
# disposition table for the original accounting of all 20 gauge tests in
# this file. Task B11 landed the three things they needed
# (`watch/routers/live_sessions.py`'s `POST /live-sessions/{id}/analyze`,
# `watch/services.py`'s `build_transcriber`/`build_llm`, and
# `watch/routers/ws.py`'s `_on_live_session_end`) and appends the 8 tests
# here, at the end of this file, with per-test adaptation notes where the
# behavior isn't a straight rename.
"""Tests for the post-session analysis pipeline (server/watch/post_session.py).

Covers the honest-degradation contract end to end:
* happy path — transcription succeeds, word-analysis finds absolutist
  language and appends a VectorEvent, the LLM produces a summary, and the
  updated live session is persisted with status "analyzed".
* transcriber unavailable — status becomes "transcription_unavailable",
  summary stays None, and NO transcript-derived events are added (nothing
  fabricated from a transcript that doesn't exist).
* LLM failure — the transcript itself is real, so status stays "analyzed",
  but summary is honestly None rather than a fabricated string.

FakeTranscriber/FakeLLM are test doubles for the `TranscriptionService`
protocol and `LLMClient.complete`'s shape — no real STT/LLM ever runs in
this file.
"""

from __future__ import annotations

import asyncio
import base64
import time

import pytest

from audio_pipeline import TranscriberUnavailable, TranscriptSegment
from watch.models import LiveSession, Participant
from watch.post_session import (
    NullTranscriptionService,
    WhisperTranscriptionService,
    analyze_live_session,
)
from watch.store import MemoryLiveSessionStore


class FakeTranscriber:
    """Test double for the TranscriptionService protocol.

    ``delay`` (seconds) is honored via a blocking ``time.sleep`` — kept from
    gauge's original for parity even though the WS end-to-end timing test it
    served (TestOnEpisodeEndFireAndForget) is deferred to B11.
    """

    def __init__(self, segments=None, error: Exception | None = None, delay: float = 0.0):
        self._segments = segments if segments is not None else []
        self._error = error
        self._delay = delay

    def transcribe(self, pcm: bytes, sample_rate: int) -> list[TranscriptSegment]:
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(self._segments)


class FakeLLM:
    """Test double matching LLMClient.complete's shape."""

    def __init__(self, summary: str | None = "canned summary", error: Exception | None = None):
        self._summary = summary
        self._error = error

    def complete(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        if self._error is not None:
            raise self._error
        return self._summary


class TestNullAndWhisperServices:
    """Coverage added in review round 1: with the 8 gauge tests deferred to
    B11 (TestBuildTranscriberAndLLM etc.), NullTranscriptionService and
    WhisperTranscriptionService had zero importers and zero direct test
    coverage in this port — gauge at least isinstance-exercised them via
    Settings-driven selection. These exercise each class's own honest-
    degradation behavior directly, independent of any B11 wiring."""

    def test_null_transcription_service_reports_unavailable(self):
        service = NullTranscriptionService()
        with pytest.raises(TranscriberUnavailable):
            service.transcribe(b"\x00\x00" * 1600, 16000)

    def test_whisper_transcription_service_rejects_wrong_sample_rate(self):
        # No model load needed: WhisperTranscriptionService.transcribe()
        # asserts the pipeline's fixed 16kHz wire contract BEFORE it ever
        # touches WhisperTranscriber/connect() — see its own "Interface
        # honesty" comment. This is the one WhisperTranscriptionService code
        # path that is honestly exercisable without faster-whisper installed
        # (a happy-path transcribe() would need a real/injected model, which
        # is out of B10's scope here — this test only proves the wrong-rate
        # guard fires, not the transcription itself).
        service = WhisperTranscriptionService()
        with pytest.raises(AssertionError, match="16000"):
            service.transcribe(b"\x00\x00" * 800, 8000)


# Two segments per the brief: "self" says something absolutist, "other" replies.
# Neither carries real diarization (speaker=None, matching Whisper's honest
# no-diarization output) — turns are alternation-labeled self/other/self/...
TWO_SEGMENTS = [
    TranscriptSegment(text="you always do this", start_time=0.0, end_time=2.0),
    TranscriptSegment(text="I was just trying to help", start_time=2.5, end_time=4.5),
]


def _live_session(**overrides) -> LiveSession:
    fields = dict(
        id="ls1",
        owner_account="acct1",
        started_at="2026-08-01T00:00:00Z",
        ended_at="2026-08-01T00:05:00Z",
        status="captured",
        participants=[Participant(id="p1", role="self", speaker_label="You")],
        vector_events=[],
        nudge_events=[],
        pcm_b64=base64.b64encode(b"\x00\x00" * 1600).decode("ascii"),
    )
    fields.update(overrides)
    return LiveSession(**fields)


async def _store_with(ls: LiveSession) -> MemoryLiveSessionStore:
    store = MemoryLiveSessionStore()
    await store.put_live_session(ls)
    return store


class TestHappyPath:
    def test_analyzed_status_summary_and_transcript_events(self):
        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM(summary="They discussed household chores.")

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "analyzed"
            assert result.summary == "They discussed household chores."
            absolutist = [e for e in result.vector_events if e.vector == "aggressive_tone"]
            assert absolutist, "expected an aggressive_tone event for absolutist language"
            assert "always" in absolutist[0].detail
            assert absolutist[0].detail.startswith("post:")
            assert absolutist[0].level >= 1

            persisted = await store.get_live_session("ls1")
            assert persisted.status == "analyzed"
            assert persisted.summary == "They discussed household chores."
            assert len(persisted.vector_events) == len(result.vector_events)

        asyncio.run(run())

    def test_transcript_events_appended_not_replacing_prosody_events(self):
        from watch.models import VectorEvent

        async def run():
            prior = VectorEvent(vector="yelling", level=2, t=1.0, value=10.0)
            store = await _store_with(_live_session(vector_events=[prior]))
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM()

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert prior in result.vector_events
            assert any(e.vector == "aggressive_tone" for e in result.vector_events)

        asyncio.run(run())


class TestIdempotentReanalysis:
    """A repeated re-analysis (e.g. after installing real STT) must be safe
    to run more than once. Transcript-derived events are REPLACED, never
    accumulated, on each analyze_live_session call."""

    def test_analyze_twice_does_not_duplicate_transcript_events(self):
        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM()

            first = await analyze_live_session("ls1", store, transcriber, llm)
            second = await analyze_live_session("ls1", store, transcriber, llm)

            aggressive = [e for e in second.vector_events if e.vector == "aggressive_tone"]
            assert len(aggressive) == 1, "re-analysis must not duplicate the transcript event"
            assert len(second.vector_events) == len(first.vector_events)

        asyncio.run(run())

    def test_analyze_twice_keeps_prosody_events_untouched(self):
        from watch.models import VectorEvent

        async def run():
            prior = VectorEvent(vector="yelling", level=2, t=1.0, value=10.0)
            store = await _store_with(_live_session(vector_events=[prior]))
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM()

            await analyze_live_session("ls1", store, transcriber, llm)
            second = await analyze_live_session("ls1", store, transcriber, llm)

            assert second.vector_events.count(prior) == 1

        asyncio.run(run())


class TestTranscriberUnavailable:
    def test_transcription_unavailable_degrades_honestly(self):
        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(error=TranscriberUnavailable("no whisper installed"))
            llm = FakeLLM()

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "transcription_unavailable"
            assert result.summary is None
            assert result.vector_events == []

            persisted = await store.get_live_session("ls1")
            assert persisted.status == "transcription_unavailable"
            assert persisted.summary is None
            assert persisted.vector_events == []

        asyncio.run(run())

    def test_prosody_events_kept_on_transcriber_unavailable(self):
        from watch.models import VectorEvent

        async def run():
            prior = VectorEvent(vector="hr_spike", level=1, t=5.0, value=90.0)
            store = await _store_with(_live_session(vector_events=[prior]))
            transcriber = FakeTranscriber(error=TranscriberUnavailable("no whisper installed"))

            result = await analyze_live_session("ls1", store, transcriber, FakeLLM())

            assert result.status == "transcription_unavailable"
            assert result.vector_events == [prior]

        asyncio.run(run())


class TestLLMFailure:
    def test_llm_failure_keeps_analyzed_with_no_summary(self):
        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM(error=RuntimeError("provider outage"))

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "analyzed"
            assert result.summary is None
            # Word analysis is local — it must still have run despite the LLM failure.
            assert any(e.vector == "aggressive_tone" for e in result.vector_events)

        asyncio.run(run())

    def test_missing_llm_keeps_analyzed_with_no_summary(self):
        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)

            result = await analyze_live_session("ls1", store, transcriber, None)

            assert result.status == "analyzed"
            assert result.summary is None

        asyncio.run(run())


class TestEmptyAudio:
    def test_empty_pcm_degrades_honestly(self):
        async def run():
            store = await _store_with(_live_session(pcm_b64=""))
            transcriber = FakeTranscriber(segments=[])
            llm = FakeLLM()

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "analyzed"
            assert result.summary is None
            assert result.vector_events == []

        asyncio.run(run())

    def test_audio_dropped_due_to_size_still_degrades_honestly(self):
        # Final-review Finding 1c (gauge): a long live session whose pcm_b64
        # was dropped by server/watch/store.py's live_session_to_doc
        # (MAX_FIRESTORE_PCM_B64) round-trips through a real Firestore-shaped
        # doc as pcm_b64=="" — this simulates exactly that doc shape (not
        # just "nobody ever recorded audio") and exercises the same path a
        # future re-analysis endpoint (B11) would take: re-analysis with no
        # live buffer, honestly degrading rather than crashing or
        # fabricating a transcript.
        from watch.store import live_session_to_doc

        async def run():
            big_session = _live_session(pcm_b64=base64.b64encode(b"\x00\x00" * 500_000).decode("ascii"))
            doc = live_session_to_doc(big_session)
            assert doc["pcm_b64"] == ""  # confirms this test actually exercises the drop

            store = await _store_with(LiveSession(**doc))
            transcriber = FakeTranscriber(segments=[])
            llm = FakeLLM()

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "analyzed"
            assert result.summary is None
            assert result.vector_events == []

        asyncio.run(run())


class TestDirectPcmHandoff:
    """Final-review Finding 1a (gauge): the live "end" path (B11's WS
    handler) hands analyze_live_session the in-memory PCM bytes directly
    (via the ``pcm=`` parameter) instead of relying on
    ``live_session.pcm_b64`` surviving a store round-trip. These prove that
    path works even when the store's copy of the live session has NO audio
    at all — the scenario Firestore's 1MiB doc limit forces for a long live
    session (server/watch/store.py's live_session_to_doc)."""

    def test_analysis_uses_directly_passed_pcm_even_when_store_doc_has_no_audio(self):
        class SpyTranscriber:
            """Records exactly the pcm bytes it was called with, so the test
            can prove analyze_live_session used the pcm= argument rather
            than re-deriving (empty) audio from the store's
            live_session.pcm_b64."""

            def __init__(self):
                self.received_pcm: bytes | None = None

            def transcribe(self, pcm: bytes, sample_rate: int) -> list[TranscriptSegment]:
                self.received_pcm = pcm
                return list(TWO_SEGMENTS)

        async def run():
            # The store's live session has pcm_b64="" — exactly what a real
            # Firestore doc looks like after live_session_to_doc dropped
            # oversized audio (or any other reason the store copy lacks it).
            store = await _store_with(_live_session(pcm_b64=""))
            transcriber = SpyTranscriber()
            llm = FakeLLM(summary="Summarized from directly-handed-off audio.")

            # The actual audio bytes never touched the store — they came
            # straight from the caller's in-memory buffer (B11's WS handler
            # pcm_buffer), simulated here as an arbitrary non-empty payload.
            live_pcm = b"\x00\x01" * 16000

            result = await analyze_live_session("ls1", store, transcriber, llm, pcm=live_pcm)

            # The transcriber received the DIRECTLY-PASSED pcm, not b"" (what
            # it would have gotten from the store's empty pcm_b64).
            assert transcriber.received_pcm == live_pcm

            # And transcription/word-analysis/summary all ran normally on it.
            assert result.status == "analyzed"
            assert result.summary == "Summarized from directly-handed-off audio."
            assert any(e.vector == "aggressive_tone" for e in result.vector_events)

            persisted = await store.get_live_session("ls1")
            assert persisted.status == "analyzed"
            assert persisted.summary == "Summarized from directly-handed-off audio."

        asyncio.run(run())

    def test_omitting_pcm_falls_back_to_store_copy(self):
        # Sanity check for the other branch: a future re-analysis endpoint
        # (B11, no live buffer to hand off) must still work exactly as
        # before by reading live_session.pcm_b64 from the store.
        async def run():
            store = await _store_with(_live_session())  # has real pcm_b64
            transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
            llm = FakeLLM(summary="From the store's own audio.")

            result = await analyze_live_session("ls1", store, transcriber, llm)

            assert result.status == "analyzed"
            assert result.summary == "From the store's own audio."

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Task B11: the 8 tests deferred here by B10 (see SCOPE NOTE at top of file).
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from watch.config import Settings  # noqa: E402
from watch.services import build_llm, build_transcriber  # noqa: E402
from watch.testing import create_watch_test_app  # noqa: E402


class TestAnalyzeEndpoint:
    """Exercises `watch/routers/live_sessions.py`'s
    `POST /live-sessions/{id}/analyze` (gauge's `POST /episodes/{id}/analyze`,
    ported verbatim in behavior — only the path/model renamed)."""

    def test_endpoint_returns_updated_live_session(self):
        store = MemoryLiveSessionStore()
        asyncio.run(store.put_live_session(_live_session()))
        transcriber = FakeTranscriber(segments=TWO_SEGMENTS)
        llm = FakeLLM(summary="Summary text")
        app = create_watch_test_app(
            store=store, transcriber=transcriber, llm=llm, allow_legacy=True
        )
        client = TestClient(app)

        # /analyze is auth-gated (watch/auth.py) same as every other REST
        # route; _live_session()'s owner_account="acct1" authenticates via
        # the legacy ?account= path (no verifier configured in this test app).
        resp = client.post("/live-sessions/ls1/analyze", params={"account": "acct1"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "analyzed"
        assert body["summary"] == "Summary text"
        assert "pcm_b64" not in body

    def test_endpoint_404_for_missing_live_session(self):
        store = MemoryLiveSessionStore()
        app = create_watch_test_app(
            store=store, transcriber=FakeTranscriber(), llm=FakeLLM(), allow_legacy=True
        )
        client = TestClient(app)

        # Legacy ?account= authenticates the caller as *some* account; the
        # 404 is decided on the live session lookup, before any ownership check.
        resp = client.post("/live-sessions/does-not-exist/analyze", params={"account": "acct1"})

        assert resp.status_code == 404


class TestBuildTranscriberAndLLM:
    """Fast unit tests for `watch/services.py`'s Settings -> backend
    selection logic, used by both the /analyze endpoint and the WS "end"
    wiring. These never touch a real model/network — WhisperTranscriptionService
    /LLMClient construction itself is cheap; only calling .transcribe()/
    .complete() would be slow or need a key, and neither is invoked here.

    ADAPTED (Task B11): gauge's `_build_transcriber(settings)` ports
    verbatim (`GAUGE_STT` -> `MINDSHIFT_WATCH_STT`, already
    `watch/config.py`'s `Settings.stt`). gauge's `_build_llm(settings)` read
    `Settings.model` (backed by `GAUGE_MODEL`) — the plan's global
    constraints DROP `GAUGE_MODEL` entirely ("use this repo's existing LLM
    config via llm_client"), and `watch/config.py`'s `Settings` deliberately
    has no `model` field (see `watch/services.py`'s own ADAPTED note). So
    `build_llm()` here takes no `Settings` argument at all — it reads
    `MINDSHIFT_MODEL` directly (the SAME env var `server/main.py`'s
    `lifespan()` already builds its top-level LLM client from), and the two
    tests below set that env var instead of constructing a `Settings(...)`.
    Behavior (construct-time success/failure -> client-or-None) is otherwise
    identical to gauge's originals.
    """

    def test_whisper_stt_selects_whisper_service(self, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_WATCH_STT", "whisper")
        assert isinstance(build_transcriber(Settings()), WhisperTranscriptionService)

    def test_none_stt_degrades_to_null_service(self, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_WATCH_STT", "none")
        assert isinstance(build_transcriber(Settings()), NullTranscriptionService)

    def test_unrecognized_stt_degrades_to_null_service(self, monkeypatch):
        monkeypatch.setenv("MINDSHIFT_WATCH_STT", "some-future-provider")
        assert isinstance(build_transcriber(Settings()), NullTranscriptionService)

    def test_supported_model_without_key_still_builds_a_client(self, monkeypatch):
        # The anthropic SDK defers the "missing API key" error to request
        # time rather than construction time, so build_llm can't catch it
        # here — that's exactly why analyze_live_session's own LLM-failure
        # handling (TestLLMFailure above) is what actually delivers the
        # honest summary=None degradation for a missing key.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("MINDSHIFT_MODEL", "claude-haiku-4-5-20251001")
        assert build_llm() is not None

    def test_unsupported_model_provider_returns_none_not_a_crash(self, monkeypatch):
        # An unrecognized model prefix DOES fail at construction time
        # (LLMClient._detect_provider raises ValueError) — this is the case
        # build_llm's try/except actually guards against.
        monkeypatch.setenv("MINDSHIFT_MODEL", "llama-3-70b")
        assert build_llm() is None


class TestOnLiveSessionEndFireAndForget:
    """The WS 'end' handler wraps analyze_live_session in
    `_on_live_session_end` (`watch/routers/ws.py`, gauge's `_on_episode_end`)
    and runs it via asyncio.create_task — this must never crash the WS
    handler, and a genuinely unexpected failure must be LOGGED, not silently
    vanish."""

    def test_unexpected_failure_is_logged_not_swallowed_silently(self, caplog):
        from watch.routers.ws import _on_live_session_end

        async def run():
            store = await _store_with(_live_session())
            transcriber = FakeTranscriber(error=RuntimeError("transcriber blew up"))
            live_session = await store.get_live_session("ls1")
            pcm = base64.b64decode(live_session.pcm_b64)
            with caplog.at_level("ERROR"):
                await _on_live_session_end(live_session, store, transcriber, FakeLLM(), pcm)

        asyncio.run(run())

        assert any(
            "Post-session analysis failed" in rec.message and "ls1" in rec.message
            for rec in caplog.records
        )
