"""Voice-enrollment router — "This is me" + the enrolled voiceprint's lifecycle.

Five endpoints under ``/voice`` (included from main.py with one line):

* ``GET  /voice/profile``      — status: is the feature available, is the user
                                 enrolled, and enrollment metadata incl. the v2
                                 per-sample provenance list (never an embedding —
                                 the raw signature never leaves the server).
* ``POST /voice/enroll``       — "This is me": embed one diarized speaker from a
                                 stored recording and store it as an individual
                                 sample (the blend is recomputed over all samples).
* ``POST /voice/enroll-direct``— guided "Train my voice": embed ONE uploaded
                                 clip of prompted phrases (single voice by
                                 client promise — no diarization, no stored
                                 recording) into a sample noted
                                 "guided enrollment".
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
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

import recordings_store
import speaker_id
from audio_ingest import AudioDecodeError, decode_to_pcm, decode_to_pcm_16k
from audio_pipeline import UUID_PATTERN
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice"])

_VOICE_UNAVAILABLE = "voice enrollment not available on this server"
_STORAGE_DISABLED = "recording storage is not enabled"

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
