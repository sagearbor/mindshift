# Ported from gauge@2157433 server/post_episode.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Post-session analysis pipeline: transcription + word/dynamics analysis +
LLM summary, run once a live session ends.

Honest-degradation contract (non-negotiable):

* Transcriber unavailable (no STT configured/installed, or a genuine
  transcription failure) -> ``status="transcription_unavailable"``. Whatever
  prosody-era data the live session already collected (vector_events,
  series) is left untouched, ``summary`` stays ``None``, and NO
  transcript-derived :class:`~watch.models.VectorEvent` is ever added —
  nothing is fabricated from a transcript that doesn't exist.
* Missing or failing LLM -> ``status`` stays ``"analyzed"`` (the transcript
  itself is real and locally-computed word analysis still ran), ``summary``
  is simply ``None`` rather than a fabricated string.
* A failure here must never corrupt or delete the stored live session: the
  live session is read once, updated in memory, and written back only once
  with a fully-decided status/summary/events — never partially.

Word analysis, not full dynamics
---------------------------------
The full dynamics statistics (interruptions, coupling, de-escalation
leadership, ...) require genuine multi-speaker overlap timing or per-turn
heat scores from a separate LLM scoring pass — neither of which the default
STT path (local Whisper, which does NO diarization; every segment carries
``speaker=None``) can honestly provide. Rather than fabricate speaker
overlap or heat scores this module does not have, transcript turns are
alternation-labeled self/other (the same disclosed heuristic
``SpeakerDiarizer``/``SpeakerLabelAssigner`` already use elsewhere in
``server/audio_pipeline.py`` for exactly this no-diarization case), and only
``word_metrics`` — pure per-speaker text counts that need nothing but the
labeled turns — is run against them. Absolutist-language counts (a
well-known escalation marker: "you always...", "you never...") are surfaced
as an ``aggressive_tone`` :class:`~watch.models.VectorEvent` with the
specifics in ``detail``; the wire contract's 5 v1 ``VectorName`` literals
are frozen (Kotlin mirrors them), so this is never a new vector.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import Counter
from typing import Protocol

import numpy as np

import word_metrics
from audio_pipeline import TranscriberUnavailable, TranscriptSegment
from watch.diarize import DiarizationService
from watch.models import SELF_PARTICIPANT_ID, LiveSession, Participant, VectorEvent
from watch.store import LiveSessionStore
from watch.vectors import VectorEngine
from whisper_transcriber import WhisperTranscriber

logger = logging.getLogger(__name__)

# The pipeline's one fixed wire contract: raw PCM, int16 little-endian, mono,
# 16 kHz (see audio_pipeline.py / server/watch/routers/ws.py, Task B11).
SAMPLE_RATE = 16000
# Frame size fed to the streaming transcribers, matching the live pipeline's
# ~100 ms wire framing — post-session batch transcription runs the buffered
# recording through the exact same boundary-detection logic, just all at once.
_FRAME_BYTES = 3200

# Absolutist language: claims that something is total/universal ("always",
# "never", ...). A well-documented escalation/blame marker in relationship
# language research. Counting these feeds the aggressive_tone vector's
# `detail` field — never a new VectorName literal.
ABSOLUTIST_WORDS = frozenset({
    "always", "never", "every", "everyone", "everybody", "everything",
    "nobody", "none", "nothing", "constantly", "completely", "totally",
})

# Every transcript-derived VectorEvent carries this detail prefix so
# re-analysis (a future POST /live-sessions/{id}/analyze, Task B11, can be
# called more than once — e.g. after installing STT for an
# already-captured live session) is idempotent by REPLACEMENT rather than
# accumulation: analyze_live_session strips any event whose detail starts
# with this prefix before appending fresh ones, so a double analyze never
# double-counts. Prosody-era events (yelling, hr_spike, ...) never carry
# this prefix and are never touched by it.
TRANSCRIPT_EVENT_DETAIL_PREFIX = "post:"

# Same idempotent-replacement contract as TRANSCRIPT_EVENT_DETAIL_PREFIX,
# but for the diarization-derived interrupting/airtime events: analyze_live_
# session strips any event whose detail starts with this prefix before
# appending fresh ones, so re-analysis never double-counts.
DIARIZATION_EVENT_DETAIL_PREFIX = "diar:"

SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing a transcribed conversation for the person who "
    "recorded it, so they can reflect on how it went. Be concise (2-4 "
    "sentences), neutral, and specific to what was actually said. Do not "
    "give relationship advice or diagnose anyone."
)


class LiveSessionNotFound(LookupError):
    """Raised when analyze_live_session is asked to analyze an unknown
    live session id."""


class TranscriptionService(Protocol):
    """What analyze_live_session needs from a speech-to-text backend."""

    def transcribe(self, pcm: bytes, sample_rate: int) -> list[TranscriptSegment]:
        """Return finalized transcript segments for the whole recording.

        Raises :class:`TranscriberUnavailable` when no transcription backend
        is configured/available — never fabricates a transcript.
        """
        ...


class WhisperTranscriptionService:
    """Batch adapter over the streaming, session-oriented ``WhisperTranscriber``.

    ``WhisperTranscriber`` is designed for a live session (connect / stream /
    finish, run on a background worker so the WS receive loop stays
    responsive). Post-session analysis instead has the WHOLE recording
    already buffered, so this feeds it through that exact same
    boundary-detection logic in one shot — a fresh, short-lived transcriber
    per call — and collects every segment it produces (mid-stream flushes
    plus the final ``finish()`` flush).

    Delegates to THIS repo's ``whisper_transcriber.WhisperTranscriber``,
    which shares a process-wide cached model (``load_shared_model``) with
    the prerecorded-upload path — gauge's copy had no such cache.
    """

    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = model_size

    def transcribe(self, pcm: bytes, sample_rate: int) -> list[TranscriptSegment]:
        # Interface honesty (final-review MINOR): the pipeline's one fixed
        # wire contract (see SAMPLE_RATE above) is baked into every downstream
        # call this makes (WhisperTranscriber, _FRAME_BYTES framing, ...) —
        # silently accepting a different sample_rate would mistranscribe
        # rather than fail loudly, so assert instead.
        assert sample_rate == SAMPLE_RATE, (
            f"WhisperTranscriptionService only supports {SAMPLE_RATE}Hz PCM, got {sample_rate}"
        )
        return asyncio.run(self._transcribe_async(pcm))

    async def _transcribe_async(self, pcm: bytes) -> list[TranscriptSegment]:
        transcriber = WhisperTranscriber(model_size=self._model_size)
        await transcriber.connect()  # raises TranscriberUnavailable honestly
        try:
            segments: list[TranscriptSegment] = []
            for i in range(0, len(pcm), _FRAME_BYTES):
                segments.extend(await transcriber.stream(pcm[i:i + _FRAME_BYTES]))
            segments.extend(await transcriber.finish())
            return segments
        finally:
            await transcriber.close()


class NullTranscriptionService:
    """Honest stand-in for MINDSHIFT_WATCH_STT="none" (or any unrecognized
    value): always reports transcription as unavailable rather than
    guessing at a backend that was never configured."""

    def transcribe(self, pcm: bytes, sample_rate: int) -> list[TranscriptSegment]:
        raise TranscriberUnavailable("MINDSHIFT_WATCH_STT=none — transcription is disabled")


# ---------------------------------------------------------------------------
# Transcript -> labeled turns -> word analysis
# ---------------------------------------------------------------------------

def _label_turns(segments: list[TranscriptSegment]) -> list[dict]:
    """Alternation-label transcript segments self/other/self/... .

    Whisper performs no diarization (every segment carries ``speaker=None``
    — see whisper_transcriber.py's module docstring), so there is no genuine
    per-segment speaker identity to read. Alternating labels is the same
    disclosed heuristic ``server/audio_pipeline.py``'s ``SpeakerDiarizer``
    already uses elsewhere for exactly this no-diarization case — not a new
    fabrication. Empty-text segments are dropped (nothing to analyze), but
    the alternation index is taken over ALL segments so a stray empty one
    doesn't shift who's "self" vs "other".
    """
    turns: list[dict] = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        speaker = "self" if i % 2 == 0 else "other"
        turns.append({"speaker": speaker, "text": text, "end": seg.end_time})
    return turns


def _format_transcript(turns: list[dict]) -> str:
    """Plain dialogue text for the LLM summary prompt."""
    lines = []
    for turn in turns:
        role = "You" if turn["speaker"] == "self" else "The other person"
        lines.append(f"{role}: {turn['text']}")
    return "\n".join(lines)


def _absolutist_events(turns: list[dict]) -> list[VectorEvent]:
    """VectorEvent(s) for absolutist language found in the "self" speaker's
    turns — this vector is about coaching the recording device's own owner,
    matching aggressive_tone's default channel ("A" / self) elsewhere in the
    app. Returns ``[]`` when nothing was found (never a zero-value event)."""
    self_tokens: list[str] = []
    last_end = 0.0
    for turn in turns:
        last_end = max(last_end, turn.get("end") or 0.0)
        if turn["speaker"] == "self":
            self_tokens.extend(word_metrics.tokenize(turn["text"]))

    counts = Counter(t for t in self_tokens if t in ABSOLUTIST_WORDS)
    total = sum(counts.values())
    if total == 0:
        return []

    level = 1 if total <= 2 else 2 if total <= 5 else 3
    examples = ", ".join(f"'{word}' x{count}" for word, count in counts.most_common(3))
    detail = f"{TRANSCRIPT_EVENT_DETAIL_PREFIX} absolutist language: {examples}"
    return [VectorEvent(vector="aggressive_tone", level=level, t=last_end, value=float(total), detail=detail)]


# ---------------------------------------------------------------------------
# Diarization -> attributed interrupting/airtime events
# ---------------------------------------------------------------------------

def _ensure_diarized_participants(session: LiveSession, turns: list[tuple[str, float, float]]) -> None:
    """Ensure a :class:`Participant` exists for every distinct diarized
    speaker label seen in ``turns``, mutating ``session.participants``.

    ``"self"`` maps to the existing :data:`SELF_PARTICIPANT_ID` participant
    (created if absent); ``"other-N"`` maps to an anonymous
    ``Participant(id=f"other-{n}", role="other", speaker_label=f"Speaker {chr(64+n)}")``
    per spec §6's anonymous-by-default rule (no ``display_name``).
    Idempotent — re-running (re-analysis) never creates a duplicate id.
    """
    existing_ids = {p.id for p in session.participants}
    new_participants: list[Participant] = []

    for label, _start, _end in turns:
        if label in existing_ids:
            continue
        if label == "self":
            new_participants.append(Participant(id=SELF_PARTICIPANT_ID, role="self", speaker_label="You"))
            existing_ids.add(SELF_PARTICIPANT_ID)
            continue
        try:
            n = int(label.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue  # not an "other-N" label — nothing honest to construct
        new_participants.append(
            Participant(id=label, role="other", speaker_label=f"Speaker {chr(64 + n)}")
        )
        existing_ids.add(label)

    if new_participants:
        session.participants = [*session.participants, *new_participants]


async def _diarization_events(
    session: LiveSession, pcm: bytes, store: LiveSessionStore, diarizer: DiarizationService
) -> list[VectorEvent]:
    """Diarize, feed VectorEngine.push_diarization, and return attributed
    events. Returns [] (never raises) when: no diarizer, no PCM, no stored
    SpeakerProfile for the owner, or no "self" turn was found.

    As a side effect (``session`` is mutated in place), ensures a
    Participant exists for every distinct diarized label — see
    :func:`_ensure_diarized_participants`.
    """
    if not pcm:
        return []

    profile = await store.get_speaker_profile(session.owner_account)
    if profile is None:
        # No honest basis to say which diarized cluster is the wearer.
        return []

    # v2 note: SpeakerProfile.embedding is the BLENDED voiceprint across all
    # of the account's samples (server/speaker_id.py's new_profile/
    # blend_samples) — same field name/shape gauge's v1 profile carried, so
    # this read needs no v1->v2 translation.
    self_print = np.asarray(profile.embedding, dtype=np.float32)

    try:
        # diarize() may do real (blocking) embedding work; keep it off the
        # event loop like the transcribe() call below.
        turns = await asyncio.to_thread(diarizer.diarize, pcm, SAMPLE_RATE, self_print)
    except Exception:  # noqa: BLE001 — a broken diarizer must never break analysis
        logger.exception("Diarization failed for live session %s", session.id)
        return []

    if not turns:
        return []

    _ensure_diarized_participants(session, turns)

    # A fresh engine per call: interrupting/airtime are computed over
    # exactly this diarization pass's turns, not accumulated across
    # re-analyses (idempotent replacement, mirroring the transcript path).
    engine = VectorEngine(baseline=None, sample_rate=SAMPLE_RATE)
    raw_events = engine.push_diarization(turns)

    # Attribution rule (plan resolution #3): the event measures the
    # WEARER's behavior, so participant_id is always SELF_PARTICIPANT_ID;
    # the other party involved is already named in `detail`.
    return [
        e.model_copy(update={
            "participant_id": SELF_PARTICIPANT_ID,
            "detail": f"{DIARIZATION_EVENT_DETAIL_PREFIX} {e.detail}",
        })
        for e in raw_events
    ]


# ---------------------------------------------------------------------------
# The pipeline entrypoint
# ---------------------------------------------------------------------------

async def analyze_live_session(
    live_session_id: str,
    store: LiveSessionStore,
    transcriber: TranscriptionService,
    llm,
    pcm: bytes | None = None,
    diarizer: DiarizationService | None = None,
) -> LiveSession:
    """Transcribe, run word analysis, summarize, and persist the live session.

    ``llm`` may be ``None`` (no key/provider configured) — summarization is
    then honestly skipped rather than attempted. Any object exposing
    ``complete(system: str, user: str) -> str`` (e.g. ``server/llm_client.py``'s
    ``LLMClient``) works.

    ``pcm`` (final-review Finding 1a, gauge): when the caller already has the
    raw audio in memory (a future WS "end" handler, Task B11, would, straight
    out of its own pcm buffer), pass it here so this function never depends
    on ``live_session.pcm_b64`` having survived a store round-trip. That
    matters because ``server/watch/store.py``'s ``live_session_to_doc`` may
    have persisted ``pcm_b64=""`` for an oversized live session (Firestore's
    1MiB doc limit — see ``MAX_FIRESTORE_PCM_B64``): reading audio back out
    of the store in that case would silently lose analysis for exactly the
    longer live sessions that need it most. Only re-analysis with no live
    buffer at hand (a future ``POST /live-sessions/{id}/analyze``, Task B11)
    omits ``pcm`` and falls back to whatever ``live_session.pcm_b64`` the
    store has (honestly empty if it was dropped for size, or if it never had
    audio).

    ``diarizer``: a :class:`~watch.diarize.DiarizationService`, or ``None``
    to skip diarization entirely. Run BEFORE transcription — it needs only
    PCM + a stored voiceprint, not a working STT backend — so its events
    survive a ``transcription_unavailable`` early return below.
    """
    session = await store.get_live_session(live_session_id)
    if session is None:
        raise LiveSessionNotFound(live_session_id)

    if pcm is None:
        pcm = base64.b64decode(session.pcm_b64) if session.pcm_b64 else b""

    # Idempotent re-analysis: replace any previously-added diarization
    # events rather than accumulating duplicates. Prosody-era and
    # transcript-derived events never carry this prefix, so they're
    # untouched by this pass.
    session.vector_events = [
        e for e in session.vector_events
        if not e.detail.startswith(DIARIZATION_EVENT_DETAIL_PREFIX)
    ]
    if diarizer is not None:
        diarization_events = await _diarization_events(session, pcm, store, diarizer)
        session.vector_events = [*session.vector_events, *diarization_events]

    try:
        # transcribe() is a plain (possibly blocking) call per the
        # TranscriptionService protocol; run it off the event loop so a real
        # transcription never blocks other work on this fire-and-forget task.
        segments = await asyncio.to_thread(transcriber.transcribe, pcm, SAMPLE_RATE)
    except TranscriberUnavailable as exc:
        logger.info("Transcription unavailable for live session %s: %s", live_session_id, exc)
        session.status = "transcription_unavailable"
        session.summary = None
        await store.put_live_session(session)
        return session

    turns = _label_turns(segments)

    # Idempotent re-analysis: replace any previously-added transcript-derived
    # events rather than accumulating duplicates (a double re-analyze, or a
    # future WS "end" followed by a manual re-analyze, must never double-
    # count). Prosody-era events never carry this prefix, so they're never
    # touched.
    session.vector_events = [
        e for e in session.vector_events
        if not e.detail.startswith(TRANSCRIPT_EVENT_DETAIL_PREFIX)
    ]

    transcript_text = ""
    if turns:
        new_events = _absolutist_events(turns)
        session.vector_events = [*session.vector_events, *new_events]
        transcript_text = _format_transcript(turns)

    summary: str | None = None
    if llm is not None and transcript_text:
        try:
            summary = await asyncio.to_thread(llm.complete, SUMMARY_SYSTEM_PROMPT, transcript_text)
        except Exception:  # noqa: BLE001 — a down/misconfigured LLM must never
            # crash analysis; the transcript itself is still real and kept.
            logger.exception("LLM summary failed for live session %s — leaving summary unset", live_session_id)
            summary = None

    session.status = "analyzed"
    session.summary = summary
    await store.put_live_session(session)
    return session
