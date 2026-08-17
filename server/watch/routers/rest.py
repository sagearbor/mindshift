# Ported from gauge@2157433 server/rest_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5):
# * Episode -> LiveSession per the locked "episode" rename map: HTTP
#   `/episodes*` -> `/live-sessions*`, `EpisodeStore` -> `LiveSessionStore`,
#   `store.list_episodes`/`get_episode`/`put_episode`/`delete_episode`/
#   `claim_legacy_episode` -> their `live_session` equivalents. Wire FIELD
#   names are UNCHANGED (PeriodStats.episodes, ClaimLegacyResponse.
#   episodes_moved, LegacyClaim.episodes_moved_total) — the watch client
#   parses them (B1 precedent) and none of them are TYPE references.
# * `make_rest_router`'s closure-factory signature is
#   `(store, auth_dep, strict_auth_dep, embedder=None)` — both auth
#   dependencies are now required positional params (gauge's `full_auth`
#   defaulted to `auth` when omitted; this repo's `testing.py` always builds
#   both explicitly via `watch.auth.require_full_auth`, so there is no
#   "caller didn't care" case left to default around).
# * `server.engine.speaker_id` (gauge's vendored v1-style embedder seam) ->
#   THIS repo's flat, v2 `speaker_id` module (server/speaker_id.py) — see
#   `_maybe_enroll_voice`'s docstring below for the concrete v1->v2 call-shape
#   adaptation (per-sample profiles, blend recompute over ALL samples).
"""REST API: live sessions, labeling/consent, sharing, settings, enrollment.

This is the phone app's read/write surface — everything that isn't the live
WS ingest (``server/watch/routers/ws.py``, Task B11) or a future analyze
endpoint.

Every handler sits behind ``Depends(auth_dep)`` (``watch/auth.py``'s
``make_auth_dependency``), which resolves either a verified bearer token or
the legacy ``?account=`` query param into a ``Principal``; the handler's
first line pulls ``account = principal.account_id`` and the
ownership/sharing logic below is otherwise unchanged from the original
``?account=``-only design.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, StrictBool

import speaker_id
from watch.aggregates import member_standing
from watch.auth import AuthDep, Principal
from watch.models import (
    LEGACY_ACCOUNT_ID,
    SELF_PARTICIPANT_ID,
    ConsentRecord,
    EnrollmentBaseline,
    LegacyClaim,
    LiveSession,
    MemberStanding,
    SpeakerProfile,
    VectorSubscription,
)
from watch.pairing_store import PairingStore
from watch.store import LiveSessionStore
from watch.vectors import SILENCE_FLOOR_DBFS, estimate_f0, rms_dbfs

logger = logging.getLogger(__name__)

# Mirrors a future groups router's DEFAULT_PERIOD_DAYS/bounds — /me/standing
# and /groups/{id}/standing must agree on the window so the dashboard's
# personal and pair views are never comparing apples to oranges by default.
DEFAULT_PERIOD_DAYS = 7

# (float32 mono PCM, sample_rate) -> L2-normed voiceprint vector.
Embedder = Callable[[np.ndarray, int], np.ndarray]

# Enrollment audio contract: mono PCM16 at this sample rate, WAV or raw.
ENROLL_SAMPLE_RATE = 16000
ENROLL_MIN_SECONDS = 3.0
# Standard (no extra chunks) 44-byte canonical PCM WAV header size.
WAV_HEADER_SIZE = 44

# Task H1: /enroll takes a raw WAV/PCM baseline clip read via
# `request.body()` (uncapped before this task — an authenticated caller
# could still push an arbitrarily large body). A 16 kHz mono PCM16 clip of
# ENROLL_MIN_SECONDS*10 (30s) is under 1 MB, so 5 MB is generous headroom
# for a calibration clip — mirrors server/routers/voice.py's
# MAX_DIRECT_ENROLL_BYTES reasoning/pattern (413 over-cap).
MAX_ENROLL_BYTES = 5 * 1024 * 1024

# The provenance note stored on a watch-enrolled sample: the raw-PCM /enroll
# endpoint has no stored source recording to reference at all (unlike the
# phone's diarization-based /voice/enroll — server/routers/voice.py), so
# recording_id/speaker are honestly None (v2's "no source recording" shape,
# not a fabricated id) and this note is the sample's only provenance.
WATCH_ENROLL_NOTE = "watch enrollment"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LabelRequest(BaseModel):
    participant_id: str
    display_name: str
    # StrictBool: pydantic's default lax `bool` coerces truthy non-booleans
    # ("yes", 1, "true", ...) to True, which would let those bypass the
    # "attested must be exactly true" gate below. StrictBool only accepts an
    # actual JSON boolean, so anything else fails validation (422) before the
    # handler even runs.
    attested: StrictBool


class ShareRequest(BaseModel):
    with_account: str


class AccountLookupResponse(BaseModel):
    account_id: str


class MeResponse(Principal):
    """GET /me's actual response shape (Task P3-6): every ``Principal`` field
    (account_id, email, legacy — unchanged wire contract, see
    server/tests/watch/test_auth_routes.py's test_me_reports_legacy_flag) plus
    ``has_paired_watch``, the one extra fact the mobile Settings screen needs
    to show "Set up your watch" as live state instead of guessing. Backed by
    ``PairingStore.has_device_tokens_for_account`` — see ``me()`` below for
    the honest-degradation default when no pairing store is wired at all."""
    has_paired_watch: bool = False


class VoiceEnrollmentStatus(BaseModel):
    available: bool                 # can this server compute embeddings at all?
    enrolled: bool
    enroll_count: int = 0
    dim: int | None = None
    model: str | None = None
    updated_at: str | None = None


class ClaimLegacyResponse(BaseModel):
    status: Literal["claimed", "nothing_to_claim"]
    episodes_moved: int = 0
    baseline_copied: bool = False
    subscriptions_copied: bool = False
    speaker_profile_copied: bool = False
    previously_claimed_at: str | None = None   # this uid's prior claim, if any


def make_claim_mutator(account: str, now_iso: str) -> Callable[["LegacyClaim | None"], LegacyClaim]:
    """Reserve/refresh the claim marker atomically. 409 on another uid's
    marker happens HERE, against the seam's fresh read — the router-level
    pre-check is a fast path only, never the authoritative one (same
    pattern as a future groups router's make_join_mutator)."""
    def mutate(claim: LegacyClaim | None) -> LegacyClaim:
        if claim is not None and claim.account_id != account:
            raise HTTPException(
                status_code=409,
                detail="watch history was already claimed by a different account",
            )
        if claim is None:
            return LegacyClaim(account_id=account, first_claimed_at=now_iso, last_claimed_at=now_iso)
        return claim.model_copy(update={"last_claimed_at": now_iso})
    return mutate


async def _get_owned_or_404(store: LiveSessionStore, live_session_id: str) -> LiveSession:
    ls = await store.get_live_session(live_session_id)
    if ls is None:
        raise HTTPException(status_code=404, detail="episode not found")
    return ls


def _require_owner(ls: LiveSession, account: str) -> None:
    if ls.owner_account != account:
        raise HTTPException(status_code=403, detail="only the episode owner may do this")


def _rms_and_f0(pcm_bytes: bytes) -> tuple[float, float]:
    """Mean RMS dBFS + median F0 over 1s windows, via vectors.py's DSP.

    Windows at/under the silence floor (``watch.vectors.SILENCE_FLOOR_DBFS``
    — the same threshold ``VectorEngine`` uses to decide "not speech") or
    with no periodicity (no F0) are excluded from their respective aggregate
    rather than dragging it toward silence/zero.
    """
    window_bytes = ENROLL_SAMPLE_RATE * 2  # 1s of 16-bit mono samples
    rms_values: list[float] = []
    f0_values: list[float] = []
    for offset in range(0, len(pcm_bytes) - window_bytes + 1, window_bytes):
        window = np.frombuffer(pcm_bytes[offset:offset + window_bytes], dtype=np.int16)
        db = rms_dbfs(window)
        if db > SILENCE_FLOOR_DBFS:
            rms_values.append(db)
        f0 = estimate_f0(window, ENROLL_SAMPLE_RATE)
        if f0 is not None:
            f0_values.append(f0)

    if not rms_values:
        # Every window was at/under the silence floor -- e.g. a muted mic or
        # a clip recorded from across the room. Never store a baseline from
        # this: an rms_db of -inf (or any bogus floor value) would poison
        # every subsequent yelling comparison for this account forever (see
        # VectorEngine.push_pcm, which measures "over baseline" against
        # exactly this value).
        raise HTTPException(
            status_code=422,
            detail="clip is silent — record in a normal speaking voice",
        )

    mean_rms = float(np.mean(rms_values))
    median_f0 = float(np.median(f0_values)) if f0_values else 0.0
    return mean_rms, median_f0


def _resolve_embedder(embedder: Embedder | None) -> Embedder | None:
    """The injected embedder wins; otherwise fall back to this repo's real
    speaker_id.embed_pcm when its heavy deps (torch/speechbrain) are actually
    importable on this server. Never raises — honest degradation to ``None``
    (unavailable) is the whole point of this seam."""
    if embedder is not None:
        return embedder
    if speaker_id.is_available():
        return speaker_id.embed_pcm
    return None


async def _maybe_enroll_voice(
    store: LiveSessionStore, account: str, pcm_bytes: bytes, embedder: Embedder | None,
) -> None:
    """Fold this clip into the account's voiceprint when embedding is
    available. NEVER fails the enrollment: an unavailable or throwing
    embedder logs and returns — the loudness baseline is real and already
    stored, and GET /enroll/voice reports the voiceprint's true state.

    v2 adaptation: this repo's ``speaker_id.new_profile`` (server/speaker_id.py)
    stores each enrollment as an individual, independently-deletable
    ``samples`` entry and recomputes the blend over ALL samples — gauge's v1
    engine stored only a running-mean blend. The raw-PCM watch clip has no
    stored source recording to reference (unlike the phone's diarization-based
    ``/voice/enroll`` — see ``server/routers/voice.py``), so ``recording_id``/
    ``speaker`` are honestly ``None`` (v2's real "no source recording" shape,
    matching the phone's own guided ``/voice/enroll-direct`` path) rather than
    a fabricated recording id; ``WATCH_ENROLL_NOTE`` carries the provenance
    instead.
    """
    if embedder is None:
        return
    try:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        embedding = await asyncio.to_thread(embedder, audio, ENROLL_SAMPLE_RATE)
        existing = await store.get_speaker_profile(account)
        existing_dict = existing.model_dump() if existing is not None else None
        profile_dict = speaker_id.new_profile(
            embedding,
            existing_dict,
            recording_id=None,
            speaker=None,
            now_iso=_now_iso(),
            note=WATCH_ENROLL_NOTE,
        )
        profile = SpeakerProfile(account_id=account, **profile_dict)
        await store.put_speaker_profile(profile)
    except Exception:  # noqa: BLE001 — voiceprint side effect must never fail enrollment
        logger.warning("voice enrollment side effect failed for account %r", account, exc_info=True)


def make_rest_router(
    store: LiveSessionStore,
    auth_dep: AuthDep,
    strict_auth_dep: AuthDep,
    embedder: Embedder | None = None,
    pairing_store: PairingStore | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/me", response_model=MeResponse)
    async def me(principal: Principal = Depends(auth_dep)) -> MeResponse:
        # pairing_store is optional (unlike the pairing router's own
        # required dependency — see watch/testing.py's module docstring on
        # why /me stays unconditionally mounted): a caller that never wired
        # one gets the honest default `has_paired_watch=False` rather than a
        # 500, matching this repo's honest-degradation doctrine (a missing
        # signal reads as "unknown/no", never a fabricated guess either way).
        has_paired_watch = False
        if pairing_store is not None:
            has_paired_watch = await pairing_store.has_device_tokens_for_account(
                principal.account_id
            )
        return MeResponse(**principal.model_dump(), has_paired_watch=has_paired_watch)

    @router.get("/me/standing", response_model=MemberStanding)
    async def my_standing(
        period_days: int = Query(DEFAULT_PERIOD_DAYS, ge=1, le=90),
        principal: Principal = Depends(auth_dep),
    ) -> MemberStanding:
        # Server-track item 13c: the personal counterpart to a future
        # /groups/{id}/standing route (watch/aggregates.py's member_standing
        # core, same owned-live-sessions-only filter -- never shared_with: a
        # shared live session is someone ELSE's behavior), just for the
        # caller's own account. No group, no membership check, no consent
        # gate — this is self-data, gated only by auth.
        account = principal.account_id
        live_sessions = [
            ls for ls in await store.list_live_sessions(account)
            if ls.owner_account == account
        ]
        acct = await store.get_account(account)
        display_name = acct.display_name if acct is not None else None
        now = datetime.now(timezone.utc)
        return member_standing(account, live_sessions, now, period_days, display_name)

    @router.post("/me/claim-legacy", response_model=ClaimLegacyResponse)
    async def claim_legacy(principal: Principal = Depends(strict_auth_dep)) -> ClaimLegacyResponse:
        # Wave B Task 2 (D2): the one-shot default->uid history re-key.
        # strict_auth_dep (not auth_dep): a legacy `?account=` caller must
        # never be able to claim — it could claim INTO "default" or any
        # impersonated account id since that path carries no verified
        # identity at all.
        account = principal.account_id
        # Fast-path 409 + previously_claimed_at snapshot (authoritative check
        # is inside make_claim_mutator, against the atomic seam's fresh read).
        prior = await store.get_legacy_claim()
        if prior is not None and prior.account_id != account:
            raise HTTPException(
                status_code=409, detail="watch history was already claimed by a different account")
        previously = prior.last_claimed_at if prior is not None else None

        claimable = [
            ls for ls in await store.list_live_sessions(LEGACY_ACCOUNT_ID)
            if ls.owner_account == LEGACY_ACCOUNT_ID
        ]
        legacy_baseline = await store.get_baseline(LEGACY_ACCOUNT_ID)
        legacy_profile = await store.get_speaker_profile(LEGACY_ACCOUNT_ID)
        legacy_has_subs = await store.has_subscriptions(LEGACY_ACCOUNT_ID)
        baseline_wanted = legacy_baseline is not None and await store.get_baseline(account) is None
        profile_wanted = legacy_profile is not None and await store.get_speaker_profile(account) is None
        subs_wanted = legacy_has_subs and not await store.has_subscriptions(account)

        if not claimable and not (baseline_wanted or profile_wanted or subs_wanted):
            # Honest none-found. Crucially, a first-time empty claim writes NO
            # marker — an account that never had legacy history must not lock
            # the legacy account away from whoever actually owns that history.
            return ClaimLegacyResponse(status="nothing_to_claim", previously_claimed_at=previously)

        # Phase 1 — atomically reserve/refresh the marker (409 inside the
        # mutator closes the pre-check race; nothing else has been written yet).
        await store.update_legacy_claim_atomically(make_claim_mutator(account, _now_iso()))

        # Final-review Important finding: `claimable` above was captured
        # BEFORE Phase 1 -- for a same-uid concurrent double-submission
        # (two overlapping requests from the same signed-in caller, e.g. a
        # UI double-tap or a client retry-on-timeout that actually landed),
        # both requests could capture the SAME pre-sweep snapshot and then
        # each move+count every live session in it, double-bumping
        # `episodes_moved_total`.
        #
        # Fix, in two parts (the second part is a necessary extension of the
        # literal "re-read after Phase 1" prescription -- see below for why
        # the reorder ALONE was verified insufficient):
        #
        # 1. Re-read the claimable list HERE, after Phase 1 has reserved the
        #    marker, instead of trusting the pre-Phase-1 snapshot above
        #    (which exists only to decide the early "nothing to claim"
        #    return, and stays there unmodified).
        # 2. Each live session in Phase 2 below is claimed via the
        #    `store.claim_legacy_live_session` primitive (server/watch/store.py),
        #    which performs "is this live session still legacy-owned? if so,
        #    move it AND bump the total" as ONE lock/transaction-guarded
        #    step, instead of two separate calls (a `put_live_session`
        #    followed by a separate counter-bump). Empirically verified (a
        #    two-real-thread race harness matching this file's other
        #    concurrency tests, run repeatedly) that reordering the
        #    `claimable` re-read ALONE closes the race only for a
        #    single-claimable-live-session sweep -- for 2+ live sessions it
        #    reliably still double-counted, because the SECOND request's
        #    re-read can land after the FIRST request's Phase 1 but before
        #    the first request has gotten around to actually moving every
        #    live session in its own (correct) list, so the second request
        #    still sees some of them as claimable and redundantly counts
        #    them. Item 2 closes that gap: whichever request's
        #    `claim_legacy_live_session` call for a given live session
        #    actually observes it as still legacy-owned wins that live
        #    session's count; the loser's call for the SAME live session
        #    (now already moved) is a no-op `False` return, never counted.
        #    See test_claim_legacy.py's
        #    test_concurrent_same_uid_claims_count_each_live_session_once
        #    for the two-thread reproduction this closes.
        claimable = [
            ls for ls in await store.list_live_sessions(LEGACY_ACCOUNT_ID)
            if ls.owner_account == LEGACY_ACCOUNT_ID
        ]

        # Phase 2 — idempotent re-key. Crash-safe by design: a partial sweep
        # leaves the marker owned by this uid, and the SAME uid may re-claim
        # to finish the job. Live sessions are a true MOVE (owner_account =
        # uid); account docs are copy-if-absent, default's copies stay for
        # the watch.
        #
        # Fix-round 1 finding 1 (superseded in spirit, kept accurate here):
        # `episodes_moved_total` was originally bumped in a separate call
        # right after each live session's own `put_live_session`, bounding a
        # mid-sweep crash's residual undercount to at most one live session
        # (see
        # `test_crash_between_live_session_move_and_total_bump_undercounts_by_at_most_one`,
        # updated for the new call shape below, not deleted -- the SAME
        # bounded-undercount behavior still holds for MemoryLiveSessionStore,
        # since `claim_legacy_live_session`'s move-then-bump there is
        # lock-guarded against OTHER CALLERS but not itself rolled back by an
        # in-process exception between its two internal writes). What this
        # round's `claim_legacy_live_session` primitive ADDS on top, for
        # FirestoreLiveSessionStore specifically, is true transactional
        # atomicity across both writes (a crash mid-transaction there commits
        # NEITHER, not "moved but uncounted") — a strictly stronger guarantee
        # than before, not a weaker one, obtained as a side effect of the
        # concurrency fix this round actually required.
        moved = 0
        for ls in claimable:
            if await store.claim_legacy_live_session(ls.id, account):
                moved += 1
        if baseline_wanted:
            await store.put_baseline(legacy_baseline.model_copy(update={"account_id": account}))
        if profile_wanted:
            await store.put_speaker_profile(legacy_profile.model_copy(update={"account_id": account}))
        if subs_wanted:
            await store.put_subscriptions(account, await store.get_subscriptions(LEGACY_ACCOUNT_ID))

        return ClaimLegacyResponse(
            status="claimed",
            episodes_moved=moved,
            baseline_copied=baseline_wanted,
            subscriptions_copied=subs_wanted,
            speaker_profile_copied=profile_wanted,
            previously_claimed_at=previously,
        )

    @router.get("/accounts/lookup", response_model=AccountLookupResponse)
    async def lookup_account(
        email: str, principal: Principal = Depends(strict_auth_dep)
    ) -> AccountLookupResponse:
        # Exact match, NOT case-folded: watch.auth.ensure_account persists
        # principal.email verbatim from the token claim (no .lower()/.strip())
        # at account-creation time, so folding case here would be
        # inconsistent with how emails are actually stored -- see
        # store.get_account_by_email's own "exact email match" contract.
        account = await store.get_account_by_email(email)
        if account is None:
            raise HTTPException(status_code=404, detail="no account for that email")
        return AccountLookupResponse(account_id=account.id)

    @router.get("/live-sessions", response_model=list[LiveSession])
    async def list_live_sessions(principal: Principal = Depends(auth_dep)) -> list[LiveSession]:
        account = principal.account_id
        return await store.list_live_sessions(account)

    @router.get("/live-sessions/{live_session_id}", response_model=LiveSession)
    async def get_live_session(
        live_session_id: str, principal: Principal = Depends(auth_dep)
    ) -> LiveSession:
        account = principal.account_id
        ls = await _get_owned_or_404(store, live_session_id)
        if ls.owner_account != account and account not in ls.shared_with:
            raise HTTPException(status_code=403, detail="not authorized for this episode")
        return ls

    @router.delete("/live-sessions/{live_session_id}", status_code=204)
    async def delete_live_session(
        live_session_id: str, principal: Principal = Depends(strict_auth_dep)
    ) -> None:
        # Wave B Task 8 (D4): hard delete, owner-only, full-auth-only.
        # strict_auth_dep (not auth_dep) — the destructive surface must never
        # be reachable by an unauthenticated legacy `?account=` caller,
        # unlike every read/write route above. Deletion removes the live
        # session from every view (owner's own list, and any shared_with
        # viewer's list/detail) since both read purely from the live store —
        # no separate cleanup needed, no tombstone left behind.
        ls = await _get_owned_or_404(store, live_session_id)
        _require_owner(ls, principal.account_id)       # shared-with viewers get the honest 403
        await store.delete_live_session(live_session_id)

    @router.post("/live-sessions/{live_session_id}/labels", response_model=LiveSession)
    async def label_participant(
        live_session_id: str, body: LabelRequest, principal: Principal = Depends(auth_dep)
    ) -> LiveSession:
        account = principal.account_id
        if body.attested is not True:
            raise HTTPException(status_code=422, detail="attested must be true")

        ls = await _get_owned_or_404(store, live_session_id)
        _require_owner(ls, account)

        participant = next((p for p in ls.participants if p.id == body.participant_id), None)
        if participant is None:
            raise HTTPException(status_code=404, detail="participant not found")

        participant.display_name = body.display_name
        ls.consents.append(ConsentRecord(
            id=uuid.uuid4().hex,
            participant_id=body.participant_id,
            kind="labeling",
            attested_by=account,
            confirmed=False,
            ts=_now_iso(),
        ))
        await store.put_live_session(ls)
        return ls

    @router.post("/live-sessions/{live_session_id}/share", response_model=LiveSession)
    async def share_live_session(
        live_session_id: str, body: ShareRequest, principal: Principal = Depends(auth_dep)
    ) -> LiveSession:
        account = principal.account_id
        ls = await _get_owned_or_404(store, live_session_id)
        _require_owner(ls, account)

        if body.with_account not in ls.shared_with:
            ls.shared_with.append(body.with_account)

        ls.consents.append(ConsentRecord(
            id=uuid.uuid4().hex,
            participant_id=SELF_PARTICIPANT_ID,
            kind="sharing",
            attested_by=account,
            confirmed=False,
            ts=_now_iso(),
        ))
        await store.put_live_session(ls)
        return ls

    @router.get("/settings/vectors", response_model=list[VectorSubscription])
    async def get_vector_settings(principal: Principal = Depends(auth_dep)) -> list[VectorSubscription]:
        account = principal.account_id
        return await store.get_subscriptions(account)

    @router.put("/settings/vectors", response_model=list[VectorSubscription])
    async def put_vector_settings(
        subs: list[VectorSubscription], principal: Principal = Depends(auth_dep)
    ) -> list[VectorSubscription]:
        account = principal.account_id
        await store.put_subscriptions(account, subs)
        return await store.get_subscriptions(account)

    @router.post("/enroll", response_model=EnrollmentBaseline)
    async def enroll(request: Request, principal: Principal = Depends(auth_dep)) -> EnrollmentBaseline:
        account = principal.account_id
        raw = await request.body()
        if len(raw) > MAX_ENROLL_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    "enrollment clip exceeds the "
                    f"{MAX_ENROLL_BYTES // (1024 * 1024)}MB limit — record a short "
                    "calibration clip, not a long session"
                ),
            )
        pcm_bytes = raw[WAV_HEADER_SIZE:] if raw[:4] == b"RIFF" else raw

        duration = (len(pcm_bytes) // 2) / ENROLL_SAMPLE_RATE
        if duration < ENROLL_MIN_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"clip too short ({duration:.1f}s); need at least {ENROLL_MIN_SECONDS:.0f}s",
            )

        mean_rms, median_f0 = _rms_and_f0(pcm_bytes)
        baseline = EnrollmentBaseline(
            account_id=account, rms_db=mean_rms, f0_median=median_f0, updated_at=_now_iso(),
        )
        await store.put_baseline(baseline)
        await _maybe_enroll_voice(store, account, pcm_bytes, _resolve_embedder(embedder))
        return baseline

    @router.get("/enroll/voice", response_model=VoiceEnrollmentStatus)
    async def voice_status(principal: Principal = Depends(auth_dep)) -> VoiceEnrollmentStatus:
        account = principal.account_id
        available = _resolve_embedder(embedder) is not None
        profile = await store.get_speaker_profile(account)
        if profile is None:
            return VoiceEnrollmentStatus(available=available, enrolled=False)
        return VoiceEnrollmentStatus(
            available=available,
            enrolled=True,
            enroll_count=profile.enroll_count,
            dim=profile.dim,
            model=profile.model,
            updated_at=profile.updated_at,
        )

    return router
