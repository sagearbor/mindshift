"""Voice-enrollment router — "This is me" / "This is Alex" + the enrolled
voiceprints' lifecycle.

An account holds N named PEOPLE (multi-person voiceprints, Foundation B): the
owner's own voice is the reserved person ``"self"`` (displayed "You") and any
partner the user names ("alex" → "Alex") is another person with its own v2
profile. Every endpoint below that took no person before defaults to
``"self"``, so the original "This is me" contract is unchanged.

Eight endpoints under ``/voice`` (included from main.py with one line):

* ``GET  /voice/profile``      — status: is the feature available, is the user
                                 enrolled, and enrollment metadata incl. the v2
                                 per-sample provenance list (never an embedding —
                                 the raw signature never leaves the server).
                                 ``?person_id=`` reads another enrolled person.
* ``GET  /voice/people``       — every enrolled person (self first) with the
                                 same per-person metadata. ``?include_embeddings
                                 =true`` is the ONE deliberate exception to
                                 "the signature never leaves the server": it
                                 adds each person's blended voiceprint so the
                                 caller's OWN phone can match speakers on-device
                                 (see ``list_voice_people``).
* ``POST /voice/enroll``       — "This is me" (default) or "This is <name>"
                                 (``person_id`` + ``display_name``): embed one
                                 diarized speaker from a stored recording and
                                 store it as an individual sample (the blend is
                                 recomputed over all samples), AND relabel that
                                 same recording's stored analysis so it counts
                                 as identified in Growth immediately
                                 (see ``_label_enrolled_and_persist``).
* ``POST /voice/enroll-direct``— guided "Train my voice": embed ONE uploaded
                                 clip of prompted phrases (single voice by
                                 client promise — no diarization, no stored
                                 recording) into a sample noted
                                 "guided enrollment". Same optional person
                                 form fields as ``/voice/enroll``.
* ``DELETE /voice/people/{id}``— forget ONE named person's voiceprint for real
                                 (idempotent; ``self`` here is the same as
                                 ``DELETE /voice/voiceprint``).
* ``PATCH /voice/people/{id}`` — rename an enrolled partner (``display_name``).
* ``POST /voice/people/{id}/enroll-from-recording``
                               — "Remember this voice": learn a person's voice
                                 from ONE diarized speaker of a stored
                                 recording (creates the person when new +
                                 named). Refuses honestly when the speaker has
                                 too little speech, when the recording kept no
                                 audio (live sessions), or when the voice is
                                 clearly someone ELSE already enrolled — see
                                 ``enroll_person_from_recording``.
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

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Path, Query, Request, UploadFile,
)
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


async def _resolve_person(
    store: "recordings_store.RecordingsStore",
    uid: str,
    person_id: str,
    display_name: str | None,
) -> tuple[dict | None, str, str | None]:
    """Who is being enrolled: ``(existing_profile, person_id, display_name)``.

    The owner ("self") never needs a name — it is always "You". A PARTNER
    needs a display name from SOMEWHERE: the request (a first enrollment, or
    a rename), else the stored profile (re-enrolling "alex" without repeating
    the name). A brand-new partner with no name at all is a 422 — we never
    store a person we'd have to label with an invented placeholder."""
    existing = await store.read_voiceprint(uid, person_id)
    if person_id == speaker_id.SELF_PERSON_ID:
        return existing, person_id, speaker_id.SELF_DISPLAY_NAME
    name = (display_name or "").strip() or None
    if name is None and existing is not None:
        name = existing.get("display_name")
    if not name:
        raise HTTPException(
            status_code=422,
            detail=f"display_name is required to enroll a new person {person_id!r}",
        )
    return existing, person_id, name


def _profile_response(
    profile: dict, *, available: bool, storage_enabled: bool = True,
    include_embedding: bool = False,
) -> "VoiceProfileResponse":
    """The per-person status view of one stored profile (person view) — the
    metadata GET /voice/profile and GET /voice/people share. The embedding is
    attached ONLY on ``include_embedding`` (the on-device opt-in of
    ``GET /voice/people?include_embeddings=true``): the CURRENT blend
    (``speaker_id.current_blend`` — one centroid per recording, re-blended
    from the stored samples at read time), L2-normalized — which is exactly
    the vector ``main._identify_enrolled_speakers`` hands the matcher
    (cosine normalizes anyway, so the phone and the server score a turn
    identically). A stored profile without a usable vector (never expected
    — it can't have been enrolled) is served without one rather than with an
    invented zero vector. ``settings`` (distinct recordings pooled) is
    always served: it gates the phone's contrast match exactly as it gates
    the server's."""
    embedding: list[float] | None = None
    if include_embedding:
        blend = speaker_id.current_blend(profile)
        if blend is not None:
            embedding = [float(x) for x in speaker_id.l2_normalize(blend).tolist()]
    return VoiceProfileResponse(
        available=available,
        storage_enabled=storage_enabled,
        enrolled=True,
        person_id=profile.get("person_id") or speaker_id.SELF_PERSON_ID,
        display_name=profile.get("display_name"),
        is_self=bool(profile.get("is_self", True)),
        enroll_count=int(profile.get("enroll_count", 0) or 0),
        settings=speaker_id.profile_settings(profile),
        updated_at=profile.get("updated_at"),
        model=profile.get("model"),
        dim=profile.get("dim"),
        embedding=embedding,
        samples=[
            VoiceSampleOut(
                id=str(s.get("id")),
                recording_id=s.get("recording_id"),
                speaker=s.get("speaker"),
                at=s.get("at"),
                note=s.get("note"),
                seconds=s.get("seconds") if isinstance(s.get("seconds"), (int, float)) else None,
            )
            for s in profile.get("samples", [])
            if isinstance(s, dict) and s.get("id")
        ],
    )


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
    *,
    person_id: str = speaker_id.SELF_PERSON_ID,
    display_label: str = _ENROLLED_DISPLAY_LABEL,
) -> dict | None:
    """Merge an "enrolled" display label for ``speaker`` into ``rec``'s stored
    analysis and persist it via ``store.overwrite_analysis`` — the "relabel one
    recording" logic shared by ``enroll_voice`` (Part A: relabel the recording
    that was tapped) and ``catch_up_voice`` (Part B: relabel a recording matched
    in bulk). Deliberately NOT ``manual_speaker_labels`` — that overlay always
    carries ``label_source="manual"`` and is the human-correction rung, not this
    one (see main.py's label-ladder docstring).

    ``person_id``/``display_label`` say WHO the speaker is: the owner (default —
    "self"/"You") or an enrolled partner ("alex"/"Alex"). A PERSON may be at
    most ONE speaker per recording (a person is one voice; and for the owner
    specifically, ``main._growth_point`` requires EXACTLY one "You" — two reads
    as "no confident me" and drops the recording out of Growth entirely). Any
    OTHER speaker currently holding this same person's enrolled label — a
    stale auto-match from the original analysis, or an earlier "This is me"
    tap being corrected — is demoted to a plain generic label first, so a
    correction tap (SpeakerEnrollment offers "This is me" on every speaker,
    filtered on nothing) can never leave two speakers both "You". Other
    people's enrolled labels on other speakers are left alone: "You" + "Alex"
    in one recording is exactly the multi-person outcome.

    Also keeps ``analysis["speaker_identity"]`` in agreement — its ``matched``
    map (and the legacy ``matched_speaker`` for the owner), which
    ``episodes_from_analysis`` PREFERS over ``speaker_labels`` when present (a
    stale identity would keep showing a stale/wrong "You" in the day-timeline
    even after this correctly relabels ``speaker_labels``) — and recomputes
    ``analysis["episodes"]`` when present so its ``participants`` reflect the
    new label immediately rather than waiting for a reanalysis.

    Best-effort by design (same "swallow and log" house style as
    ``main._identify_enrolled_speakers``): returns ``False`` — never raises —
    when ``rec`` has no analysis to update, or when the persist itself fails
    (a storage hiccup here must never sink the caller, which already did the
    part that matters most: writing the voiceprint). Returns the UPDATED
    analysis dict only when the recording was actually persisted (so a caller
    labeling several people in one recording can build each on the last),
    else ``None``.
    """
    analysis = rec.get("analysis")
    if not isinstance(analysis, dict):
        return None  # nothing analyzed yet — nothing to relabel
    try:
        updated = dict(analysis)
        labels = dict(updated.get("speaker_labels") or {})

        # Demote any OTHER speaker carrying THIS person's enrolled label
        # before writing the new one.
        for other, entry in labels.items():
            if (
                other != speaker
                and isinstance(entry, dict)
                and entry.get("label_source") == _ENROLLED_LABEL_SOURCE
                and entry.get("display_label") == display_label
            ):
                labels[other] = {
                    "display_label": other, "label_source": _GENERIC_LABEL_SOURCE,
                }

        labels[speaker] = {
            "display_label": display_label,
            "label_source": _ENROLLED_LABEL_SOURCE,
        }
        updated["speaker_labels"] = labels

        # Keep speaker_identity in agreement — episodes_from_analysis prefers
        # it over speaker_labels when both are present. The multi-person
        # ``matched``/``people`` maps are the source of truth; the legacy
        # ``matched_speaker`` mirrors the owner's match for older readers.
        existing_identity = updated.get("speaker_identity")
        identity = dict(existing_identity) if isinstance(existing_identity, dict) else {}
        matched = dict(identity.get("matched") or {})
        if not matched and isinstance(identity.get("matched_speaker"), str):
            matched[identity["matched_speaker"]] = speaker_id.SELF_PERSON_ID
        matched = {sp: pid for sp, pid in matched.items() if pid != person_id and sp != speaker}
        matched[speaker] = person_id
        people = dict(identity.get("people") or {})
        people[person_id] = {
            "display_name": display_label,
            "is_self": person_id == speaker_id.SELF_PERSON_ID,
        }
        identity["matched"] = matched
        identity["people"] = people
        identity["matched_speaker"] = next(
            (sp for sp, pid in matched.items() if pid == speaker_id.SELF_PERSON_ID), None,
        )
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
        return updated if result is not None else None
    except Exception:  # noqa: BLE001 — best-effort, must never sink the caller
        logger.warning(
            "Failed to persist enrolled label uid=%s recording=%s speaker=%s",
            uid, recording_id, speaker, exc_info=True,
        )
        return None


class EnrollRequest(BaseModel):
    recording_id: str = Field(pattern=UUID_PATTERN)
    # The diarized speaker label the user tapped as "me" / "Alex" (e.g. "Speaker A").
    speaker: str = Field(min_length=1, max_length=60)
    # WHO this voice is. Default: the account owner ("self" → "You"). Any other
    # id is a partner the user is naming; it doubles as a storage path segment,
    # hence the strict slug pattern (validated here, never taken raw).
    person_id: str = Field(
        default=speaker_id.SELF_PERSON_ID, pattern=speaker_id.PERSON_ID_PATTERN,
    )
    # The partner's display label ("Alex"). Required for a NEW partner (422
    # otherwise); optional when re-enrolling an existing one (keeps the stored
    # name unless given — giving it renames). Ignored for self (always "You").
    display_name: str | None = Field(
        default=None, min_length=1, max_length=speaker_id.DISPLAY_NAME_MAX,
    )


class EnrollResponse(BaseModel):
    enrolled: bool
    speaker: str
    person_id: str = speaker_id.SELF_PERSON_ID
    display_name: str | None = None
    is_self: bool = True
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
    # Seconds of pooled speech the sample was embedded from, when the
    # enrollment path measured it (enroll-from-recording); null otherwise.
    seconds: float | None = None


class VoiceProfileResponse(BaseModel):
    # Whether the server can do voice ID at all (deps installed). The client
    # hides the "This is me" affordance when False.
    available: bool
    # Whether recording storage (where the print lives) is enabled server-side.
    storage_enabled: bool
    enrolled: bool
    # WHOSE profile this is (multi-person voiceprints): the owner is "self" /
    # "You" / is_self; a partner carries the name the user gave. Defaults
    # describe the owner so the pre-existing GET /voice/profile shape (always
    # the owner) is unchanged for older clients.
    person_id: str = speaker_id.SELF_PERSON_ID
    display_name: str | None = speaker_id.SELF_DISPLAY_NAME
    is_self: bool = True
    enroll_count: int
    # How many DISTINCT recordings the print pools (speaker_id.profile_settings;
    # 0 when unenrolled, >= 1 otherwise — a legacy v1 blend counts as one).
    # Not the same as enroll_count: three "This is me" taps on one clip are
    # three samples but ONE setting. Gates the cross-recording contrast match
    # (speaker_id.CROSS_MATCH_MIN_SETTINGS) on the server and on the phone
    # alike, so the phone's verdict can never be more permissive than the
    # server's for the same print.
    settings: int = 0
    updated_at: str | None = None
    model: str | None = None
    dim: int | None = None
    # v2 — the per-sample provenance list (empty when unenrolled). A stored v1
    # profile is served through the same view: one legacy-blend sample.
    samples: list[VoiceSampleOut] = []
    # The blended, L2-normalized voiceprint (``dim`` floats) — present ONLY
    # for GET /voice/people?include_embeddings=true, the on-device speaker-ID
    # opt-in. ``exclude_if`` drops the key entirely (not ``null``) otherwise,
    # so every pre-existing response is byte-identical to before and the
    # default remains "the raw signature never leaves the server".
    embedding: list[float] | None = Field(default=None, exclude_if=lambda v: v is None)


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
    person_id: str = Query(
        default=speaker_id.SELF_PERSON_ID, pattern=speaker_id.PERSON_ID_PATTERN,
    ),
) -> VoiceProfileResponse:
    """Report voice-ID availability + one person's enrollment status (the
    account owner by default; ``?person_id=alex`` for an enrolled partner).

    Never 503s on absent deps/storage — it is the very check the client uses to
    decide whether to OFFER enrollment, so it must always answer. No embedding
    vector is ever returned — the samples carry provenance metadata only. A v1
    / legacy-layout profile is served through the person view (one
    legacy-blend sample) WITHOUT rewriting the stored doc: reads stay
    side-effect free."""
    available = speaker_id.is_available()
    store = _get_store(request)
    if store is None:
        return VoiceProfileResponse(
            available=available, storage_enabled=False,
            enrolled=False, enroll_count=0, person_id=person_id,
            display_name=None if person_id != speaker_id.SELF_PERSON_ID else speaker_id.SELF_DISPLAY_NAME,
            is_self=person_id == speaker_id.SELF_PERSON_ID,
        )
    profile = speaker_id.as_person(await store.read_voiceprint(uid, person_id), person_id=person_id)
    if profile is None:
        return VoiceProfileResponse(
            available=available, storage_enabled=True,
            enrolled=False, enroll_count=0, person_id=person_id,
            display_name=None if person_id != speaker_id.SELF_PERSON_ID else speaker_id.SELF_DISPLAY_NAME,
            is_self=person_id == speaker_id.SELF_PERSON_ID,
        )
    return _profile_response(profile, available=available)


class VoicePeopleResponse(BaseModel):
    available: bool
    storage_enabled: bool
    # Every enrolled person, the owner ("self") first — each in the same
    # per-person shape GET /voice/profile serves. Empty when nobody is
    # enrolled or storage is disabled (never a 503: like /profile, this is
    # what the client consults to decide what to offer).
    people: list[VoiceProfileResponse] = []


@router.get("/people", response_model=VoicePeopleResponse)
async def list_voice_people(
    request: Request,
    uid: str = Depends(get_current_uid),
    include_embeddings: bool = Query(
        default=False,
        description=(
            "Also return each person's blended, L2-normalized voiceprint "
            "(`embedding`, `dim` floats) so the caller's own device can match "
            "speakers locally with the ECAPA model from GET /models/ecapa.onnx. "
            "Off by default: the signature never leaves the server unless asked."
        ),
    ),
) -> VoicePeopleResponse:
    """Every person this account has enrolled a voice for (the owner first,
    then partners by name). Same honesty rules as GET /voice/profile: never a
    503, a legacy single-document owner print is served as "self" without
    being rewritten, and — by default — never an embedding.

    ``include_embeddings=true`` is the deliberate exception, for the phone's
    on-device speaker-ID (apps/mobile/src/live/speakerId.ts): the realtime
    loop can only tell "you" from "Mom" locally if it holds the same
    voiceprints the server matches with. Scope is structural, not a filter:
    ``store.list_voiceprints(uid)`` is keyed by the VERIFIED uid, so a caller
    can only ever receive the prints their own account enrolled (a partner's
    voiceprint is data the account owner enrolled, on their own device, of a
    voice they recorded — never another account's biometric). Each returned
    person carries ``embedding`` + ``dim`` + ``model`` (the pinned ECAPA
    revision) so the client can refuse a print from a different model
    rather than match across embedding spaces."""
    available = speaker_id.is_available()
    store = _get_store(request)
    if store is None:
        return VoicePeopleResponse(available=available, storage_enabled=False)
    profiles = await store.list_voiceprints(uid)
    return VoicePeopleResponse(
        available=available,
        storage_enabled=True,
        people=[
            _profile_response(
                speaker_id.as_person(p), available=available,
                include_embedding=include_embeddings,
            )
            for p in profiles
            if isinstance(p, dict)
        ],
    )


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_voice(
    body: EnrollRequest,
    request: Request,
    uid: str = Depends(get_current_uid),
    # Review 2026-08-24: a GCS download + ffmpeg decode + ECAPA embed per
    # call, and the ONE /voice route that was not behind the per-IP limiter.
    _rl: None = Depends(_rate_limit),
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
    existing, person_id, display_name = await _resolve_person(
        store, uid, body.person_id, body.display_name,
    )
    profile = speaker_id.new_profile(
        embedding, existing,
        recording_id=body.recording_id, speaker=body.speaker, now_iso=now_iso,
        person_id=person_id, display_name=display_name,
    )
    await store.write_voiceprint(uid, profile)

    # Relabel the VERY recording this was tapped on so it shows up in Growth
    # immediately — the fix for "enrolled, but Growth still says no data yet"
    # (the enrolled voiceprint alone never touched this recording's own stored
    # labels before). Best-effort: never fails the enrollment response, which
    # already carries the part that matters most (the voiceprint write above).
    await _label_enrolled_and_persist(
        store, uid, body.recording_id, rec, body.speaker, now_iso,
        person_id=person_id, display_label=profile["display_name"],
    )

    logger.info(
        "Voice enrolled uid=%s recording=%s speaker=%s person=%s count=%d",
        uid, body.recording_id, body.speaker, person_id, profile["enroll_count"],
    )
    return EnrollResponse(
        enrolled=True,
        speaker=body.speaker,
        person_id=person_id,
        display_name=profile["display_name"],
        is_self=profile["is_self"],
        enroll_count=profile["enroll_count"],
        dim=profile["dim"],
        updated_at=profile["updated_at"],
    )


class DirectEnrollResponse(BaseModel):
    enrolled: bool
    person_id: str = speaker_id.SELF_PERSON_ID
    display_name: str | None = None
    is_self: bool = True
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
    # Multipart form fields (this endpoint is a file upload, so the person
    # goes in the form, not a JSON body). Same defaults/validation as
    # EnrollRequest: owner ("self") unless a partner slug + name is given.
    person_id: str = Form(
        default=speaker_id.SELF_PERSON_ID, pattern=speaker_id.PERSON_ID_PATTERN,
    ),
    display_name: str | None = Form(
        default=None, min_length=1, max_length=speaker_id.DISPLAY_NAME_MAX,
    ),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> DirectEnrollResponse:
    """Guided enrollment ("Train my voice" / "Train Alex's voice") — enroll
    from an uploaded clip.

    The client records a few prompted phrases in-app and uploads ONE short
    audio file that it PROMISES contains only the enrolling person's voice, so
    no diarization runs: the whole clip is embedded (capped like the pooled
    path) and appended as a v2 sample with note "guided enrollment". Nothing
    about the clip is persisted — only the numeric signature.

    Honest failures: deps absent → 503; storage disabled → 503; upload over
    the cap → 413; undecodable → 422; less than MIN_ENROLL_SECONDS of ACTUAL
    speech (a long silent clip does not count) → 422; a new partner with no
    display_name → 422."""
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
    existing, resolved_pid, resolved_name = await _resolve_person(
        store, uid, person_id, display_name,
    )
    profile = speaker_id.new_profile(
        embedding, existing,
        recording_id=None, speaker=None, now_iso=now_iso, note=GUIDED_NOTE,
        person_id=resolved_pid, display_name=resolved_name,
    )
    await store.write_voiceprint(uid, profile)
    logger.info(
        "Voice enrolled (guided) uid=%s person=%s speech=%.1fs count=%d",
        uid, resolved_pid, voiced, profile["enroll_count"],
    )
    return DirectEnrollResponse(
        enrolled=True,
        person_id=resolved_pid,
        display_name=profile["display_name"],
        is_self=profile["is_self"],
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

    # Anyone enrolled at all? (The owner OR a named partner — a partner-only
    # account still benefits: its recordings get "Alex" labels. Growth's
    # newly_identified count below is still about the owner specifically.)
    if not await store.list_voiceprints(uid):
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
            main._recording_manual_people(rec),
        )
        already_identified = any(
            entry.get("label_source") == main.LABEL_SOURCE_ENROLLED
            or main._is_me_label(entry)
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
            # FAST PATH — the stored analysis already carries every speaker's
            # pooled ECAPA embedding (written by a previous audio pass): re-score
            # those against the CURRENT prints in memory. No download, no
            # decode, no re-embed — the whole batch runs in milliseconds, which
            # is what lets a print that just grew a second setting re-check
            # every past recording without the phone waiting on a cold
            # instance. Otherwise (older analyses) do the audio pass once and
            # persist the embeddings it computed so the NEXT catch-up is fast.
            turn_speakers = {
                t.get("speaker") for t in (rec.get("turns") or [])
                if isinstance(t, dict) and t.get("speaker")
            }
            stored_embs = speaker_id.stored_speaker_embeddings(
                analysis.get("speaker_identity")
            )
            if turn_speakers and turn_speakers <= set(stored_embs):
                report = await main._identify_enrolled_speakers_from_embeddings(
                    uid, {sp: stored_embs[sp] for sp in turn_speakers},
                )
            else:
                audio = await store.get_audio_bytes(uid, recording_id)
                if not audio:
                    raise AudioDecodeError("no stored audio for this recording")
                pcm, sr = await asyncio.to_thread(decode_to_pcm, audio, "audio.m4a")
                turns = [main.AnalyzeTurn(**t) for t in (rec.get("turns") or [])]
                report = await main._identify_enrolled_speakers(uid, pcm, sr, turns)
                if isinstance(report, dict) and speaker_id.stored_speaker_embeddings(report):
                    # Persist the embeddings (and fresh scores) even on a
                    # no-match, so this recording never needs audio again.
                    # Labels are NOT touched here — _label_enrolled_and_persist
                    # below is the only writer of speaker_labels/matched.
                    existing = analysis.get("speaker_identity")
                    identity = dict(existing) if isinstance(existing, dict) else {}
                    identity["speakers"] = report["speakers"]
                    identity["model"] = report.get("model")
                    identity["match_threshold"] = report.get("match_threshold")
                    analysis = {**analysis, "speaker_identity": identity}
                    await store.update_analysis(uid, recording_id, analysis)
                    rec = {**rec, "analysis": analysis}
        except Exception:  # noqa: BLE001 — one bad recording must not sink the batch
            logger.warning(
                "Catch-up: match failed uid=%s recording=%s",
                uid, recording_id, exc_info=True,
            )
            continue

        # Every enrolled person the match found in this recording — the owner
        # ("You") and any named partner ("Alex"). The report's own reader
        # (speaker_id.enrolled_display_labels) decides who gets a label, so
        # this loop can never invent one.
        enrolled_labels = speaker_id.enrolled_display_labels(report) if report else {}
        if not enrolled_labels:
            continue  # honest no-match — never guess
        matched_people = (report.get("matched") if isinstance(report, dict) else None) or {}

        for matched, display_label in enrolled_labels.items():
            # A human already named this speaker — never silently overwritten
            # by an automatic match, even though the manual overlay already
            # hides the effect right now (clearing that manual label later
            # would otherwise wrongly reveal "You").
            manual_name = manual.get(matched)
            if isinstance(manual_name, str) and manual_name.strip():
                continue

            person_id = matched_people.get(matched) or speaker_id.SELF_PERSON_ID
            now_iso = datetime.now(timezone.utc).isoformat()
            persisted = await _label_enrolled_and_persist(
                store, uid, recording_id, rec, matched, now_iso,
                person_id=person_id, display_label=display_label,
            )
            if persisted is None:
                continue
            # Later people in this same recording must build on the analysis
            # the previous persist just wrote, not the pre-loop snapshot.
            rec = {**rec, "analysis": persisted}

            if display_label != _ENROLLED_DISPLAY_LABEL:
                continue  # a partner label — real, but not "me" for Growth

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
    person_id: str = Query(
        default=speaker_id.SELF_PERSON_ID, pattern=speaker_id.PERSON_ID_PATTERN,
    ),
) -> DeleteSampleResponse:
    """Remove ONE enrollment sample and recompute the blended voiceprint —
    the owner's by default, ``?person_id=alex`` for a partner's.

    404 when that person has no profile or the sample id isn't in it
    (uid-scoped: another user's sample ids never resolve here). Deleting the
    LAST sample deletes that person's whole stored profile — exactly the
    "forget my voice" state, never a hollow doc. A v1 profile is migrated on
    this write, so its legacy blend sample is deletable whole. Storage
    disabled → 503."""
    store = _require_store(request)
    profile = await store.read_voiceprint(uid, person_id)
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
        await store.delete_voiceprint(uid, person_id)
        logger.info(
            "Voice sample deleted (last) uid=%s person=%s sample=%s — profile removed",
            uid, person_id, sample_id,
        )
        return DeleteSampleResponse(deleted=True, enrolled=False, enroll_count=0)
    await store.write_voiceprint(uid, speaker_id.as_person(remaining, person_id=person_id))
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
    """"Forget my voice" — delete the OWNER's stored biometric signature for
    real (partners are forgotten one at a time via DELETE /voice/people/{id}).

    Idempotent: ``deleted`` is True when a print existed and was removed, False
    when there was nothing stored. Storage disabled → 503 (there is nothing this
    server could have stored to delete, reported honestly)."""
    store = _require_store(request)
    deleted = await store.delete_voiceprint(uid)
    logger.info("Voice forget uid=%s deleted=%s", uid, deleted)
    return ForgetResponse(deleted=deleted)


class ForgetPersonResponse(BaseModel):
    deleted: bool
    person_id: str


@router.delete("/people/{person_id}", response_model=ForgetPersonResponse)
async def forget_voice_person(
    request: Request,
    person_id: str = Path(pattern=speaker_id.PERSON_ID_PATTERN),
    uid: str = Depends(get_current_uid),
) -> ForgetPersonResponse:
    """Forget ONE enrolled person's voiceprint for real — a named partner, or
    the owner via ``self`` (identical to DELETE /voice/voiceprint).

    Same contract as "forget my voice": idempotent (``deleted`` reports whether
    a print existed), REAL deletion (the biometric signature is gone, not
    tombstoned), uid-scoped (another account's people never resolve here),
    storage disabled → 503. The person id is validated as a path segment
    (422 on anything outside PERSON_ID_PATTERN) so it never reaches storage
    raw."""
    store = _require_store(request)
    deleted = await store.delete_voiceprint(uid, person_id)
    logger.info("Voice forget person uid=%s person=%s deleted=%s", uid, person_id, deleted)
    return ForgetPersonResponse(deleted=deleted, person_id=person_id)


# ---------------------------------------------------------------------------
# People labeling — rename a person, learn a voice from a recording's speaker
# ---------------------------------------------------------------------------

class RenamePersonRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=speaker_id.DISPLAY_NAME_MAX)


@router.patch("/people/{person_id}", response_model=VoiceProfileResponse)
async def rename_voice_person(
    body: RenamePersonRequest,
    request: Request,
    person_id: str = Path(pattern=speaker_id.PERSON_ID_PATTERN),
    uid: str = Depends(get_current_uid),
) -> VoiceProfileResponse:
    """Rename an enrolled partner ("alex" → "Alexander"). The owner is always
    "You" and cannot be renamed (422). 404 when that person has no profile
    (uid-scoped: another account's people never resolve here). The new name
    applies to every FUTURE label the enrolled rung writes; labels already
    stored on recordings keep the name they were written with (a stored
    analysis is never rewritten by a rename — the People screen says so)."""
    store = _require_store(request)
    if person_id == speaker_id.SELF_PERSON_ID:
        raise HTTPException(
            status_code=422, detail="the account owner is always shown as \"You\"",
        )
    existing = await store.read_voiceprint(uid, person_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="No such enrolled person")
    name = body.display_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="display_name must not be blank")
    renamed = speaker_id.as_person(existing, person_id=person_id, display_name=name)
    renamed = {**renamed, "updated_at": datetime.now(timezone.utc).isoformat()}
    await store.write_voiceprint(uid, renamed)
    logger.info("Voice person renamed uid=%s person=%s", uid, person_id)
    return _profile_response(renamed, available=speaker_id.is_available())


class EnrollFromRecordingRequest(BaseModel):
    recording_id: str = Field(pattern=UUID_PATTERN)
    # The diarized speaker label the user tapped ("Speaker B").
    speaker_label: str = Field(min_length=1, max_length=60)
    # Required when ``person_id`` is new (422 otherwise); optional for an
    # existing person (given → renames, like POST /voice/enroll).
    display_name: str | None = Field(
        default=None, min_length=1, max_length=speaker_id.DISPLAY_NAME_MAX,
    )


class EnrollFromRecordingResponse(BaseModel):
    enrolled: bool
    person_id: str
    display_name: str | None = None
    is_self: bool = False
    created: bool
    enroll_count: int
    # Seconds of that speaker's pooled speech the new sample was embedded from.
    seconds: float
    dim: int
    updated_at: str
    # The recording's effective speaker labels after the relabel (empty when
    # the recording has no stored analysis to relabel) — so the client can
    # render the "enrolled" badge without a refetch.
    speaker_labels: dict[str, dict] = {}
    stored: str = "a numeric voice signature (192 numbers), not the audio"


# The refusal reasons the client shows inline. Each detail starts with a
# stable bracketed tag so the UI can pick its copy without parsing prose.
REASON_NO_AUDIO = "no-audio"
REASON_TOO_LITTLE_SPEECH = "too-little-speech"
REASON_SOUNDS_LIKE = "sounds-like-someone-else"


def _reason(tag: str, text: str) -> str:
    return f"[{tag}] {text}"


@router.post(
    "/people/{person_id}/enroll-from-recording",
    response_model=EnrollFromRecordingResponse,
)
async def enroll_person_from_recording(
    body: EnrollFromRecordingRequest,
    request: Request,
    person_id: str = Path(pattern=speaker_id.PERSON_ID_PATTERN),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
) -> EnrollFromRecordingResponse:
    """"Remember this voice" — learn ``person_id``'s voice from ONE diarized
    speaker of a stored recording, creating the person when it is new (a
    ``display_name`` is then required).

    Pools that speaker's turns from the stored ``audio.m4a`` (decoded to
    16 kHz), embeds them once, and appends the result as a NEW SAMPLE on the
    person's v2 profile (provenance: recording id + speaker label + seconds
    of pooled speech), reblending over all samples. Then relabels THIS
    recording's stored analysis so the speaker shows the person's name on the
    enrolled rung immediately (same relabel ``POST /voice/enroll`` does).

    Honest refusals — every one a 422 whose detail starts with a bracketed
    reason tag the client keys its copy on:

    * ``[no-audio]`` — a live session keeps no audio on the server
      (``media_type: none``), or the derivative is missing: nothing to learn
      from; record a 20-second sample on the People screen instead.
    * ``[too-little-speech]`` — fewer than ``MIN_ENROLL_SECONDS`` (3 s) of
      pooled speech under that speaker's turns. The MATCHING floor is 1 s,
      but a print every future match depends on wants more than a matched
      turn does (same rule as POST /voice/enroll).
    * ``[sounds-like-someone-else]`` — the pooled voice clears
      ``MATCH_THRESHOLD`` against a DIFFERENT enrolled person and is not
      closer to this one by ``ENROLL_CONFLICT_MARGIN`` (see
      ``speaker_id.enrollment_conflict``). The user most likely tapped the
      wrong row; appending it would poison this print.

    Other gates as for ``/voice/enroll``: deps absent / storage disabled →
    503; recording missing or foreign → 404; speaker not in the recording →
    422; a brand-new person with no name → 422."""
    if not speaker_id.is_available():
        raise HTTPException(status_code=503, detail=_VOICE_UNAVAILABLE)
    store = _require_store(request)

    rec = await store.get_recording(uid, body.recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    turns = rec.get("turns") or []
    speakers = {t.get("speaker") for t in turns}
    if body.speaker_label not in speakers:
        raise HTTPException(
            status_code=422,
            detail=f"speaker {body.speaker_label!r} is not in this recording",
        )
    # Who is being enrolled — resolved BEFORE the expensive decode+embed so a
    # nameless new person is a cheap 422.
    existing, resolved_pid, resolved_name = await _resolve_person(
        store, uid, person_id, body.display_name,
    )
    if rec.get("media_type") == "none":
        raise HTTPException(
            status_code=422,
            detail=_reason(
                REASON_NO_AUDIO,
                "this live session kept no audio on the server, so there is no "
                "voice to learn from — record a 20-second sample from the People "
                "screen instead",
            ),
        )
    audio = await store.get_audio_bytes(uid, body.recording_id)
    if not audio:
        raise HTTPException(
            status_code=422,
            detail=_reason(
                REASON_NO_AUDIO,
                "this recording's audio is no longer stored, so there is no voice "
                "to learn from",
            ),
        )

    try:
        pcm, sr = await asyncio.to_thread(decode_to_pcm_16k, audio, "audio.m4a")
    except AudioDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"could not decode the stored audio: {exc}",
        )
    pooled = speaker_id.pool_speaker_pcm(pcm, sr, turns, body.speaker_label)
    seconds = pooled.size / float(sr) if sr > 0 else 0.0
    if seconds < speaker_id.MIN_ENROLL_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=_reason(
                REASON_TOO_LITTLE_SPEECH,
                f"only {seconds:.1f}s of that speaker's voice is in this recording "
                f"— at least {speaker_id.MIN_ENROLL_SECONDS:.0f}s is needed to "
                "learn a voice trustworthily",
            ),
        )
    try:
        embedding = await asyncio.to_thread(speaker_id.embed_pcm, pooled, sr)
    except speaker_id.SpeakerIdUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Would this voice be mistaken for someone ELSE already enrolled? Refuse
    # rather than poison the print (see speaker_id.enrollment_conflict).
    profiles = await store.list_voiceprints(uid)
    conflict = speaker_id.enrollment_conflict(embedding, profiles, resolved_pid)
    if conflict is not None:
        own = conflict.get("own_score")
        own_text = (
            f" and only {own:.2f} to {resolved_name}" if isinstance(own, float) else ""
        )
        raise HTTPException(
            status_code=422,
            detail=_reason(
                REASON_SOUNDS_LIKE,
                f"that voice sounds like {conflict['display_name']} (similarity "
                f"{conflict['score']:.2f}{own_text}; the match floor is "
                f"{speaker_id.MATCH_THRESHOLD:.2f}). Not saved, so "
                f"{resolved_name}'s voiceprint stays clean — if this really is "
                f"{resolved_name}, label the speaker as {conflict['display_name']} "
                "first or remove their conflicting sample under People",
            ),
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    profile = speaker_id.new_profile(
        embedding, existing,
        recording_id=body.recording_id, speaker=body.speaker_label, now_iso=now_iso,
        person_id=resolved_pid, display_name=resolved_name, seconds=seconds,
    )
    await store.write_voiceprint(uid, profile)

    persisted = await _label_enrolled_and_persist(
        store, uid, body.recording_id, rec, body.speaker_label, now_iso,
        person_id=resolved_pid, display_label=profile["display_name"],
    )
    labels = persisted.get("speaker_labels") if isinstance(persisted, dict) else None

    logger.info(
        "Voice learned from recording uid=%s recording=%s speaker=%s person=%s "
        "seconds=%.1f count=%d created=%s",
        uid, body.recording_id, body.speaker_label, resolved_pid, seconds,
        profile["enroll_count"], existing is None,
    )
    return EnrollFromRecordingResponse(
        enrolled=True,
        person_id=resolved_pid,
        display_name=profile["display_name"],
        is_self=profile["is_self"],
        created=existing is None,
        enroll_count=profile["enroll_count"],
        seconds=round(seconds, 1),
        dim=profile["dim"],
        updated_at=profile["updated_at"],
        speaker_labels=labels if isinstance(labels, dict) else {},
    )
