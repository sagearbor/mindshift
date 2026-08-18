"""Voice-enrollment router — "This is me" + the enrolled voiceprint's lifecycle.

Six endpoints under ``/voice`` (included from main.py with one line):

* ``GET  /voice/profile``      — status: is the feature available, is the user
                                 enrolled, and enrollment metadata incl. the v2
                                 per-sample provenance list (never an embedding —
                                 the raw signature never leaves the server).
* ``POST /voice/enroll``       — "This is me": embed one diarized speaker from a
                                 stored recording and store it as an individual
                                 sample (the blend is recomputed over all samples),
                                 AND relabel that same recording's stored analysis
                                 so it counts as identified in Growth immediately
                                 (see ``_label_enrolled_and_persist``).
* ``POST /voice/enroll-direct``— guided "Train my voice": embed ONE uploaded
                                 clip of prompted phrases (single voice by
                                 client promise — no diarization, no stored
                                 recording) into a sample noted
                                 "guided enrollment".
* ``POST /voice/catch-up``     — bulk re-match every already-stored recording
                                 that predates enrollment (or predates any
                                 "This is me" tap) against the enrolled
                                 voiceprint — cheap (decode + embed, no
                                 re-transcription) so it can process several
                                 recordings in one call.
* ``DELETE /voice/samples/{id}`` — remove ONE enrollment sample and recompute the
                                 blend; deleting the last sample leaves the same
                                 state as "forget my voice".
* ``DELETE /voice/voiceprint`` — "Forget my voice": REALLY delete the biometric
                                 signature (idempotent — reports whether one was
                                 removed).

Samples are INDEPENDENT of recordings: deleting a recording never touches the
samples enrolled from it (the client renders "source recording deleted" when a
sample's recording no longer resolves).

Kept OUT of main.py deliberately: the concurrent label-ladder work edits main's
analysis prompt section, so this feature owns its own file and touches main.py
only through the one include_router line.

Honesty / availability (house rule): when torch/speechbrain are not installed the
enroll endpoint returns an honest **503 "voice enrollment not available on this
server"** rather than pretending; when recording storage is disabled it returns
503; a missing/foreign recording or speaker is a 404/422 that never leaks another
user's data. The verified Firebase ``uid`` is the only trusted identity.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

import episodes
import recordings_store
import speaker_id
from audio_ingest import AudioDecodeError, decode_to_pcm, decode_to_pcm_16k
from audio_pipeline import UUID_PATTERN
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

_VOICE_UNAVAILABLE = "voice enrollment not available on this server"
_STORAGE_DISABLED = "recording storage is not enabled"

# Same default as main.EPISODE_GAP_SECONDS (kept independently readable here
# rather than importing main at module load time — see _rate_limit's note on
# the main<->voice circular import; both read the SAME env var so a deployed
# override stays in sync across both modules).
_EPISODE_GAP_SECONDS = float(os.getenv("EPISODE_GAP_SECONDS", "60"))

# Up to this many not-yet-identified candidates are actually decoded+matched
# per /voice/catch-up call — each candidate is a GCS download + ffmpeg decode
# + speechbrain embedding, the single most expensive per-item cost in the API.
# Any candidates beyond the cap are reported via CatchUpResponse.remaining
# rather than silently ignored.
_CATCHUP_BATCH_LIMIT = 25

# A dedicated, much TIGHTER per-IP budget than the generic per-route limiter
# (main._rate_limit / RATE_LIMIT_PER_MINUTE, default 60/min shared by every
# other route) — catch-up can decode+embed up to _CATCHUP_BATCH_LIMIT
# recordings in a single call, so the generic budget is far too loose here.
# A small, self-contained fixed-window counter (same algorithm as
# main._RateLimiter) rather than a lazy cross-module construction — this
# router already avoids importing main at module load time.
_CATCHUP_RATE_LIMIT_PER_MINUTE = int(
    os.getenv("VOICE_CATCHUP_RATE_LIMIT_PER_MINUTE", "5")
)


class _CatchUpRateLimiter:
    """Fixed-window per-key request counter, scoped to /voice/catch-up only."""

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window_s = window_s
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        async with self._lock:
            now = time.monotonic()
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0  # window elapsed — reset
            count += 1
            self._hits[key] = (start, count)
            return count <= self.limit

    def reset(self) -> None:
        """Drop all counters (used by tests to isolate windows)."""
        self._hits.clear()


_catchup_rate_limiter = _CatchUpRateLimiter(_CATCHUP_RATE_LIMIT_PER_MINUTE)


async def _catchup_rate_limit(request: Request) -> None:
    """A much tighter per-IP budget than ``_rate_limit`` — see the comment
    above ``_CATCHUP_RATE_LIMIT_PER_MINUTE``. Honors the same
    ``RATE_LIMIT_ENABLED`` escape hatch main.py's limiter does (read lazily —
    main isn't imported at module load — so tests can disable rate limiting
    globally the same way they already do for the rest of the API)."""
    import main  # lazy — see _rate_limit's note on the circular import

    if not main.RATE_LIMIT_ENABLED:
        return
    client = request.client
    key = client.host if client else "unknown"
    if not await _catchup_rate_limiter.allow(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — too many requests; please slow down.",
        )

# The provenance note stored on (and shown for) a guided-enrollment sample —
# these samples have no source recording, so the note IS their provenance.
GUIDED_NOTE = "guided enrollment"

# Direct-enroll uploads are a few short prompted phrases: a 30 s 16 kHz mono
# wav is ~1 MB, so 5 MB is generous headroom while keeping the endpoint far
# under the general 25 MB analyze-upload cap (this path never needs a video).
MAX_DIRECT_ENROLL_BYTES = 5 * 1024 * 1024


async def _rate_limit(request: Request) -> None:
    """Reuse main's per-IP rate limiter. Imported lazily at request time:
    main.py imports this router at module load, so a top-level import here
    would be circular (and main's limiter is defined after the include)."""
    import main

    await main._rate_limit(request)


def _get_store(request: Request) -> "recordings_store.RecordingsStore | None":
    """The app's recordings store (set in main's lifespan), or None when storage
    is disabled. Read off ``app.state`` so this router never imports main."""
    return getattr(request.app.state, "recordings_store", None)


def _require_store(request: Request) -> "recordings_store.RecordingsStore":
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail=_STORAGE_DISABLED)
    return store


# The label ladder's top rung, written directly (see main.py's
# LABEL_SOURCE_ENROLLED / ENROLLED_DISPLAY_LABEL) — duplicated as plain string
# literals here rather than importing main at module load time (main imports
# THIS router; a top-level `import main` here would be circular). The shape
# matches main.SpeakerLabelOut exactly: display_label + label_source, nothing
# else — confirmed by reading that model, which has only those two fields.
_ENROLLED_LABEL_SOURCE = "enrolled"
_ENROLLED_DISPLAY_LABEL = "You"
_GENERIC_LABEL_SOURCE = "generic"


async def _label_enrolled_and_persist(
    store: "recordings_store.RecordingsStore",
    uid: str,
    recording_id: str,
    rec: dict,
    speaker: str,
    now_iso: str,
) -> bool:
    """Merge an "enrolled" display label for ``speaker`` into ``rec``'s stored
    analysis and persist it via ``store.overwrite_analysis`` — the "relabel one
    recording" logic shared by ``enroll_voice`` (Part A: relabel the recording
    that was tapped) and ``catch_up_voice`` (Part B: relabel a recording matched
    in bulk). Deliberately NOT ``manual_speaker_labels`` — that overlay always
    carries ``label_source="manual"`` and is the human-correction rung, not this
    one (see main.py's label-ladder docstring).

    At most ONE speaker may carry "enrolled" (``main._growth_point`` requires
    EXACTLY one — two reads as "no confident me" and drops the recording out of
    Growth entirely). Any OTHER speaker currently holding "enrolled" — a stale
    auto-match from the original analysis, or an earlier "This is me" tap being
    corrected — is demoted to a plain generic label first, so a correction
    tap (SpeakerEnrollment offers "This is me" on every speaker, filtered on
    nothing) can never leave two speakers both "enrolled".

    Also keeps ``analysis["speaker_identity"].matched_speaker`` in agreement
    (``episodes_from_analysis`` PREFERS it over ``speaker_labels`` when
    present — a stale identity would keep showing a stale/wrong "You" in the
    day-timeline even after this correctly relabels ``speaker_labels``), and
    recomputes ``analysis["episodes"]`` when present so its ``participants``
    reflect the new label immediately rather than waiting for a reanalysis.

    Best-effort by design (same "swallow and log" house style as
    ``main._identify_enrolled_speakers``): returns ``False`` — never raises —
    when ``rec`` has no analysis to update, or when the persist itself fails
    (a storage hiccup here must never sink the caller, which already did the
    part that matters most: writing the voiceprint). Returns ``True`` only when
    the recording was actually updated.
    """
    analysis = rec.get("analysis")
    if not isinstance(analysis, dict):
        return False  # nothing analyzed yet — nothing to relabel
    try:
        updated = dict(analysis)
        labels = dict(updated.get("speaker_labels") or {})

        # Demote any OTHER "enrolled" speaker before writing the new one.
        for other, entry in labels.items():
            if (
                other != speaker
                and isinstance(entry, dict)
                and entry.get("label_source") == _ENROLLED_LABEL_SOURCE
            ):
                labels[other] = {
                    "display_label": other, "label_source": _GENERIC_LABEL_SOURCE,
                }

        labels[speaker] = {
            "display_label": _ENROLLED_DISPLAY_LABEL,
            "label_source": _ENROLLED_LABEL_SOURCE,
        }
        updated["speaker_labels"] = labels

        # Keep speaker_identity in agreement — episodes_from_analysis prefers
        # its matched_speaker over speaker_labels when both are present.
        existing_identity = updated.get("speaker_identity")
        identity = dict(existing_identity) if isinstance(existing_identity, dict) else {}
        identity["matched_speaker"] = speaker
        updated["speaker_identity"] = identity

        # Recompute stored episodes (participants) so the day timeline agrees
        # with the new label immediately — only when this analysis carries
        # episodes at all (older/degraded analyses may not).
        if isinstance(updated.get("episodes"), list):
            updated["episodes"] = episodes.segment_episodes(
                rec.get("turns") or [],
                per_turn=updated.get("per_turn"),
                speaker_labels=labels,
                speaker_identity=identity,
                title=updated.get("title"),
                gap_seconds=_EPISODE_GAP_SECONDS,
            )

        result = await store.overwrite_analysis(
            uid, recording_id,
            turns=rec.get("turns") or [],
            analysis=updated,
            reanalyzed_at=now_iso,
        )
        return result is not None
    except Exception:  # noqa: BLE001 — best-effort, must never sink the caller
        logger.warning(
            "Failed to persist enrolled label uid=%s recording=%s speaker=%s",
            uid, recording_id, speaker, exc_info=True,
        )
        return False


class EnrollRequest(BaseModel):
    recording_id: str = Field(pattern=UUID_PATTERN)
    # The diarized speaker label the user tapped as "me" (e.g. "Speaker A").
    speaker: str = Field(min_length=1, max_length=60)


class EnrollResponse(BaseModel):
    enrolled: bool
    speaker: str
    # How many enrollments the stored print now averages (>=1). More refines it.
    enroll_count: int
    dim: int
    updated_at: str
    # Plain-language statement of WHAT was stored — biometric transparency.
    stored: str = (
        "a numeric voice signature (192 numbers), not your audio"
    )


class VoiceSampleOut(BaseModel):
    """Provenance of ONE enrollment sample — metadata only, NEVER the embedding
    (the raw signature never leaves the server). ``recording_id`` is null for
    the migrated pre-v2 legacy blend (``note`` says what it is); the client
    resolves the id against the recordings list and states honestly when the
    source recording has since been deleted."""
    id: str
    recording_id: str | None = None
    speaker: str | None = None
    at: str | None = None
    note: str | None = None


class VoiceProfileResponse(BaseModel):
    # Whether the server can do voice ID at all (deps installed). The client
    # hides the "This is me" affordance when False.
    available: bool
    # Whether recording storage (where the print lives) is enabled server-side.
    storage_enabled: bool
    enrolled: bool
    enroll_count: int
    updated_at: str | None = None
    model: str | None = None
    dim: int | None = None
    # v2 — the per-sample provenance list (empty when unenrolled). A stored v1
    # profile is served through the same view: one legacy-blend sample.
    samples: list[VoiceSampleOut] = []


class DeleteSampleResponse(BaseModel):
    deleted: bool
    # Whether any profile remains: deleting the last sample leaves the SAME
    # state as "forget my voice" (enrolled False, count 0, nothing stored).
    enrolled: bool
    enroll_count: int


class ForgetResponse(BaseModel):
    deleted: bool


@router.get("/profile", response_model=VoiceProfileResponse)
async def get_voice_profile(
    request: Request,
    uid: str = Depends(get_current_uid),
) -> VoiceProfileResponse:
    """Report voice-ID availability + this user's enrollment status.

    Never 503s on absent deps/storage — it is the very check the client uses to
    decide whether to OFFER enrollment, so it must always answer. No embedding
    vector is ever returned — the samples carry provenance metadata only. A v1
    profile is served through the v2 view (one legacy-blend sample) WITHOUT
    rewriting the stored doc: reads stay side-effect free."""
    available = speaker_id.is_available()
    store = _get_store(request)
    if store is None:
        return VoiceProfileResponse(
            available=available, storage_enabled=False,
            enrolled=False, enroll_count=0,
        )
    profile = speaker_id.as_v2(await store.read_voiceprint(uid))
    if profile is None:
        return VoiceProfileResponse(
            available=available, storage_enabled=True,
            enrolled=False, enroll_count=0,
        )
    return VoiceProfileResponse(
        available=available,
        storage_enabled=True,
        enrolled=True,
        enroll_count=int(profile.get("enroll_count", 0) or 0),
        updated_at=profile.get("updated_at"),
        model=profile.get("model"),
        dim=profile.get("dim"),
        samples=[
            VoiceSampleOut(
                id=str(s.get("id")),
                recording_id=s.get("recording_id"),
                speaker=s.get("speaker"),
                at=s.get("at"),
                note=s.get("note"),
            )
            for s in profile.get("samples", [])
            if isinstance(s, dict) and s.get("id")
        ],
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_voice(
    body: EnrollRequest,
    request: Request,
    uid: str = Depends(get_current_uid),
) -> EnrollResponse:
    """"This is me" — enroll a diarized speaker from a stored recording.

    Pulls the recording's stored ``audio.m4a`` derivative + turns, decodes to PCM,
    pools the chosen speaker's segments, embeds them (ECAPA-TDNN, CPU), and folds
    the result into the user's voiceprint (a running mean across enrollments).

    Honest failures: deps absent → 503; storage disabled → 503; recording missing
    or not this user's → 404; speaker not in the recording → 422; too little of
    that speaker's speech to enroll trustworthily → 422."""
    if not speaker_id.is_available():
        raise HTTPException(status_code=503, detail=_VOICE_UNAVAILABLE)
    store = _require_store(request)

    rec = await store.get_recording(uid, body.recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    turns = rec.get("turns") or []
    speakers = {t.get("speaker") for t in turns}
    if body.speaker not in speakers:
        raise HTTPException(
            status_code=422,
            detail=f"speaker {body.speaker!r} is not in this recording",
        )

    audio = await store.get_audio_bytes(uid, body.recording_id)
    if audio is None:
        raise HTTPException(
            status_code=404, detail="Recording audio is not available to enroll",
        )

    # Decode the stored derivative back to PCM, then pool + embed the speaker —
    # both blocking, so off the event loop. embed_speaker returns None when there
    # is too little pooled speech to trust (an honest 422, never a weak print).
    try:
        pcm, sr = await asyncio.to_thread(decode_to_pcm, audio, "audio.m4a")
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"could not decode the stored audio: {exc}",
        )
    try:
        embedding = await asyncio.to_thread(
            speaker_id.embed_speaker, pcm, sr, turns, body.speaker,
            min_seconds=speaker_id.MIN_ENROLL_SECONDS,
        )
    except speaker_id.SpeakerIdUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if embedding is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "not enough of that speaker's voice in this recording to enroll "
                f"(need at least {speaker_id.MIN_ENROLL_SECONDS:.0f}s of their speech)"
            ),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    existing = await store.read_voiceprint(uid)
    profile = speaker_id.new_profile(
        embedding, existing,
        recording_id=body.recording_id, speaker=body.speaker, now_iso=now_iso,
    )
    await store.write_voiceprint(uid, profile)

    # Relabel the VERY recording this was tapped on so it shows up in Growth
    # immediately — the fix for "enrolled, but Growth still says no data yet"
    # (the enrolled voiceprint alone never touched this recording's own stored
    # labels before). Best-effort: never fails the enrollment response, which
    # already carries the part that matters most (the voiceprint write above).
    await _label_enrolled_and_persist(
        store, uid, body.recording_id, rec, body.speaker, now_iso,
    )

    logger.info(
        "Voice enrolled uid=%s recording=%s speaker=%s count=%d",
        uid, body.recording_id, body.speaker, profile["enroll_count"],
    )
    return EnrollResponse(
        enrolled=True,
        speaker=body.speaker,
        enroll_count=profile["enroll_count"],
        dim=profile["dim"],
        updated_at=profile["updated_at"],
    )


class DirectEnrollResponse(BaseModel):
    enrolled: bool
    # How many samples the stored print now blends (>=1). More refines it.
    enroll_count: int
    dim: int
    updated_at: str
    # Plain-language statement of WHAT was stored — biometric transparency.
    stored: str = (
        "a numeric voice signature (192 numbers), not your audio"
    )


@router.post("/enroll-direct", response_model=DirectEnrollResponse)
async def enroll_voice_direct(
    request: Request,
    file: UploadFile = File(...),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> DirectEnrollResponse:
    """Guided enrollment ("Train my voice") — enroll from an uploaded clip.

    The client records a few prompted phrases in-app and uploads ONE short
    audio file that it PROMISES contains only the enrolling user's voice, so
    no diarization runs: the whole clip is embedded (capped like the pooled
    path) and appended as a v2 sample with note "guided enrollment". Nothing
    about the clip is persisted — only the numeric signature.

    Honest failures: deps absent → 503; storage disabled → 503; upload over
    the cap → 413; undecodable → 422; less than MIN_ENROLL_SECONDS of ACTUAL
    speech (a long silent clip does not count) → 422."""
    if not speaker_id.is_available():
        raise HTTPException(status_code=503, detail=_VOICE_UNAVAILABLE)
    store = _require_store(request)

    data = await file.read()
    if len(data) > MAX_DIRECT_ENROLL_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "audio exceeds the "
                f"{MAX_DIRECT_ENROLL_BYTES // (1024 * 1024)}MB guided-enrollment "
                "limit — record the phrases, not a long session"
            ),
        )
    if not data:
        raise HTTPException(status_code=422, detail="no audio was uploaded")

    # Decode to 16 kHz PCM (blocking → off the event loop). decode_to_pcm_16k
    # re-decodes through ffmpeg when the container's native rate differs, so
    # the embedder never sees a mis-rated clip.
    try:
        pcm, sr = await asyncio.to_thread(
            decode_to_pcm_16k, data, file.filename or "clip.wav",
        )
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"could not decode the audio: {exc}",
        )

    # Enough ACTUAL speech? Clip length is not the measure — a long silent
    # upload is honestly rejected, never embedded into a garbage voiceprint.
    voiced = speaker_id.speech_seconds(pcm, sr)
    if voiced < speaker_id.MIN_ENROLL_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                "not enough speech in the clip to enroll trustworthily "
                f"(need at least {speaker_id.MIN_ENROLL_SECONDS:.0f}s of speech; "
                f"heard {voiced:.1f}s)"
            ),
        )

    # Embed the whole clip (the client promises a single voice), capped the
    # same way the pooled path is so one embed call stays bounded.
    max_samples = int(speaker_id.MAX_POOL_SECONDS * sr)
    clip = pcm[:max_samples]
    try:
        embedding = await asyncio.to_thread(speaker_id.embed_pcm, clip, sr)
    except speaker_id.SpeakerIdUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    now_iso = datetime.now(timezone.utc).isoformat()
    existing = await store.read_voiceprint(uid)
    profile = speaker_id.new_profile(
        embedding, existing,
        recording_id=None, speaker=None, now_iso=now_iso, note=GUIDED_NOTE,
    )
    await store.write_voiceprint(uid, profile)
    logger.info(
        "Voice enrolled (guided) uid=%s speech=%.1fs count=%d",
        uid, voiced, profile["enroll_count"],
    )
    return DirectEnrollResponse(
        enrolled=True,
        enroll_count=profile["enroll_count"],
        dim=profile["dim"],
        updated_at=profile["updated_at"],
    )


class CatchUpResponse(BaseModel):
    # How many not-yet-identified analyzed recordings were actually attempted
    # (decode + match) THIS call. Recordings already identified, never
    # analyzed, or beyond the per-call batch cap are NOT wasted work — they
    # don't count here — except a no-audio recording, which IS an
    # attempted-but-failed candidate.
    checked: int
    # Of those, how many are now EFFECTIVELY identified (confirmed by
    # re-computing the read-time label, not just "the write succeeded") —
    # this is exactly the count Growth will show for this call.
    newly_identified: int
    # Not-yet-identified candidates that exist but were NOT attempted this
    # call because the batch cap (_CATCHUP_BATCH_LIMIT) was reached — a
    # future call (the client can offer "keep going") would pick these up.
    remaining: int


@router.post("/catch-up", response_model=CatchUpResponse)
async def catch_up_voice(
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_catchup_rate_limit),
) -> CatchUpResponse:
    """Bulk re-match already-stored recordings against the caller's ENROLLED
    voiceprint — the "Catch up my past recordings" affordance on the empty
    Growth screen, for recordings that predate enrollment entirely (the guided
    "Train my voice" flow only ever writes the account-level voiceprint — it
    never touches a single recording) or predate any "This is me" tap.

    Deliberately NOT a full reanalyze: no re-transcription, just an audio
    decode + embedding match against the ALREADY-computed turns (reusing
    ``main._identify_enrolled_speakers`` — the exact function the initial
    analysis pipeline and POST …/reanalyze use for this rung), so this endpoint
    can cheaply process several stored recordings in one call where a full
    reanalyze per recording would be needlessly expensive. Capped at
    ``_CATCHUP_BATCH_LIMIT`` decode+match attempts per call (most-recent
    candidates first — ``list_recordings`` is already newest-first) and rate
    limited far more tightly than the rest of the API (``_catchup_rate_limit``)
    — this is the single most expensive endpoint in the API.

    Honest gates: deps absent / storage disabled → 503 (same as ``enroll_voice``);
    no enrolled voiceprint yet → ``{"checked": 0, "newly_identified": 0,
    "remaining": 0}``, never a 422 — "nothing to catch up against yet" is a
    normal state the client already renders (the empty-state copy), not an
    error. A candidate whose matched speaker was already given a MANUAL label
    by the user is skipped entirely (no persist, no count) — a human's
    explicit correction is never silently overwritten by an automatic match,
    even though the manual overlay would already hide the effect at read time.

    Per-recording best-effort (house rule, same as ``_identify_enrolled_speakers``
    itself: "enrollment matching must NEVER sink an analysis"): one recording's
    decode/match failure is logged and skipped, never aborting the batch."""
    if not speaker_id.is_available():
        raise HTTPException(status_code=503, detail=_VOICE_UNAVAILABLE)
    store = _require_store(request)

    voiceprint = await store.read_voiceprint(uid)
    if not voiceprint:
        return CatchUpResponse(checked=0, newly_identified=0, remaining=0)

    import main  # lazy — see _rate_limit's note on the circular import

    metas = await store.list_recordings(uid)
    analyzed_ids = [m["id"] for m in metas if m.get("has_analysis")]

    checked = 0
    newly_identified = 0
    remaining = 0
    for recording_id in analyzed_ids:
        try:
            rec = await store.get_recording(uid, recording_id)
        except Exception:  # noqa: BLE001 — one bad fetch must not sink the batch
            logger.warning(
                "Catch-up: failed to fetch uid=%s recording=%s",
                uid, recording_id, exc_info=True,
            )
            continue
        if rec is None:
            continue
        analysis = rec.get("analysis")
        if not isinstance(analysis, dict):
            continue  # never analyzed — not a candidate

        manual = rec.get("manual_speaker_labels") or {}

        # Same effective-label computation GET /growth uses — skip a recording
        # that already has a confident "me" (no wasted re-matching).
        effective = main._effective_speaker_labels(
            analysis.get("speaker_labels"), manual, main._recording_speaker_ids(rec),
        )
        already_identified = any(
            entry.get("label_source") == main.LABEL_SOURCE_ENROLLED
            for entry in effective.values()
        )
        if already_identified:
            continue

        # A genuine remaining candidate. Once the per-call batch cap is hit,
        # stop doing expensive work but keep counting how many are left.
        if checked >= _CATCHUP_BATCH_LIMIT:
            remaining += 1
            continue

        checked += 1
        try:
            audio = await store.get_audio_bytes(uid, recording_id)
            if not audio:
                raise AudioDecodeError("no stored audio for this recording")
            pcm, sr = await asyncio.to_thread(decode_to_pcm, audio, "audio.m4a")
            turns = [main.AnalyzeTurn(**t) for t in (rec.get("turns") or [])]
            report = await main._identify_enrolled_speakers(uid, pcm, sr, turns)
        except Exception:  # noqa: BLE001 — one bad recording must not sink the batch
            logger.warning(
                "Catch-up: match failed uid=%s recording=%s",
                uid, recording_id, exc_info=True,
            )
            continue

        matched = report.get("matched_speaker") if report else None
        if not matched:
            continue  # honest no-match — never guess

        # A human already named this speaker — never silently overwritten by
        # an automatic match, even though the manual overlay already hides
        # the effect right now (clearing that manual label later would
        # otherwise wrongly reveal "You").
        manual_name = manual.get(matched)
        if isinstance(manual_name, str) and manual_name.strip():
            continue

        now_iso = datetime.now(timezone.utc).isoformat()
        if not await _label_enrolled_and_persist(
            store, uid, recording_id, rec, matched, now_iso,
        ):
            continue

        # Confirm the recording is now EFFECTIVELY identified — mirrors
        # exactly what GET /growth computes — rather than trusting
        # persist-success alone (a manual label makes the write invisible;
        # after the guard above that can't happen for `matched` itself, but
        # this keeps the count honest against any future overlay subtlety).
        after_labels = dict(analysis.get("speaker_labels") or {})
        after_labels[matched] = {
            "display_label": _ENROLLED_DISPLAY_LABEL,
            "label_source": _ENROLLED_LABEL_SOURCE,
        }
        after_effective = main._effective_speaker_labels(
            after_labels, manual, main._recording_speaker_ids(rec),
        )
        if after_effective.get(matched, {}).get("label_source") == main.LABEL_SOURCE_ENROLLED:
            newly_identified += 1

    logger.info(
        "Voice catch-up uid=%s checked=%d newly_identified=%d remaining=%d",
        uid, checked, newly_identified, remaining,
    )
    return CatchUpResponse(
        checked=checked, newly_identified=newly_identified, remaining=remaining,
    )


@router.delete("/samples/{sample_id}", response_model=DeleteSampleResponse)
async def delete_voice_sample(
    sample_id: str,
    request: Request,
    uid: str = Depends(get_current_uid),
) -> DeleteSampleResponse:
    """Remove ONE enrollment sample and recompute the blended voiceprint.

    404 when the user has no profile or the sample id isn't in it (uid-scoped:
    another user's sample ids never resolve here). Deleting the LAST sample
    deletes the whole stored profile — exactly the "forget my voice" state, never
    a hollow doc. A v1 profile is migrated on this write, so its legacy blend
    sample is deletable whole. Storage disabled → 503."""
    store = _require_store(request)
    profile = await store.read_voiceprint(uid)
    if profile is None:
        raise HTTPException(status_code=404, detail="No voice profile to edit")
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        remaining = speaker_id.remove_sample(profile, sample_id, now_iso=now_iso)
    except KeyError:
        raise HTTPException(
            status_code=404, detail="That sample is not in your voice profile",
        )
    if remaining is None:
        await store.delete_voiceprint(uid)
        logger.info(
            "Voice sample deleted (last) uid=%s sample=%s — profile removed",
            uid, sample_id,
        )
        return DeleteSampleResponse(deleted=True, enrolled=False, enroll_count=0)
    await store.write_voiceprint(uid, remaining)
    logger.info(
        "Voice sample deleted uid=%s sample=%s remaining=%d",
        uid, sample_id, remaining["enroll_count"],
    )
    return DeleteSampleResponse(
        deleted=True, enrolled=True, enroll_count=remaining["enroll_count"],
    )


@router.delete("/voiceprint", response_model=ForgetResponse)
async def forget_voice(
    request: Request,
    uid: str = Depends(get_current_uid),
) -> ForgetResponse:
    """"Forget my voice" — delete the stored biometric signature for real.

    Idempotent: ``deleted`` is True when a print existed and was removed, False
    when there was nothing stored. Storage disabled → 503 (there is nothing this
    server could have stored to delete, reported honestly)."""
    store = _require_store(request)
    deleted = await store.delete_voiceprint(uid)
    logger.info("Voice forget uid=%s deleted=%s", uid, deleted)
    return ForgetResponse(deleted=deleted)
