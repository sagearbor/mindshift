"""Live-session router — Track 2 ("phone later analysis").

Three endpoints, included from main.py with one line:

* ``POST /sessions/live``          — Track 3-mobile POSTs a FINISHED live
                                     coaching session here at session end
                                     (the phone's ``TurnLocalEvent`` turns +
                                     optional tone flags / identity verdicts).
                                     Stored into the SAME recordings store an
                                     upload uses, so YourDay / Growth / Replay
                                     show it like any recording. Idempotent on
                                     ``session_id`` (a re-POST rewrites).
                                     201 ``{episode_id, …}`` immediately; the
                                     LLM batch analysis (heats, report cards)
                                     and the "what you could have said"
                                     reflection run AFTERWARDS as a tracked
                                     background task — the 201 never waits on
                                     an LLM.
* ``POST /episodes/{id}/reflect``  — on-demand "what you could have said" for
                                     the user's OWN turns of one episode (a
                                     live session OR an upload whose enrolled
                                     voice is known). Cached on the episode's
                                     analysis.json keyed by a transcript hash:
                                     repeat views never re-bill; ``force=true``
                                     re-runs.
* ``GET  /sessions``               — the therapist dashboard's list: every
                                     stored recording the caller owns ("You")
                                     or that a patient SHARED with them (the
                                     existing read-only grant), projected to
                                     the dashboard's session shape with the
                                     tone/identity summary and reflections.

Why its own router (not main.py): main's analyze/recordings region is being
edited by the concurrent tracks (audio_pipeline for realtime, the enrolled-
voiceprint label ladder); this feature touches main.py through the one
include line plus two additive fields on the recording read. ``main`` is
imported LAZILY inside handlers (main imports this module at load) — the
same circular-import discipline routers/voice.py follows.

Honesty (house rules): storage disabled → 503; a foreign/missing episode →
404 (never confirmed); a shared episode → 403 on writes (the recipient can
see it, so pretending it's absent would be dishonest); an episode with no
turn the user can be identified in → 422 for reflect (there is no "you" to
coach). The batch pass failing never un-stores a session — the analysis
stays honestly "lite" with the error recorded.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field, field_validator, model_validator

import live_sessions
import recordings_store
from audio_pipeline import UUID_PATTERN
from auth import get_current_uid
from models.audio import SpeakerIdentityEvent, ToneFlagEvent, TurnLocalEvent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["live-sessions"])

_STORAGE_DISABLED = "recording storage is not enabled"
_SHARED_READ_ONLY = (
    "this episode was shared with you as read-only — only its owner can "
    "reflect on it"
)

# Same env-driven gap as main.EPISODE_GAP_SECONDS, read independently (no
# main import at module load — see the module docstring).
_EPISODE_GAP_SECONDS = float(os.getenv("EPISODE_GAP_SECONDS", "60"))

# Bounds mirror main.ANALYZE_MAX_TURNS / ANALYZE_MAX_TRANSCRIPT_CHARS so a
# live session can never exceed what the batch analysis could carry. Kept as
# literals here for the same no-import-at-load reason; the values are
# asserted equal to main's in the test suite.
LIVE_MAX_TURNS = 400
LIVE_MAX_TRANSCRIPT_CHARS = 60_000
LIVE_SESSION_ID_MAX = 128

# uuid5 namespace for deriving a recording id from (uid, session_id). A
# DERIVED id is what makes ingest idempotent without a lookup table: the
# same session from the same user always lands on the same objects, and
# the id still matches audio_pipeline.UUID_PATTERN so every existing
# ``/recordings/{id}`` route accepts it.
_LIVE_NAMESPACE = uuid.UUID("6f1c2f2e-3b3a-4c6e-9d8e-7a5b2c1d0e9f")

LiveMode = Literal["earpiece", "speaker", "therapist"]

# Background post-ingest tasks (batch analysis + reflection), strong-ref'd
# like main._JOB_TASKS so the event loop never garbage-collects one mid-run.
# Tests await these directly to observe the async path deterministically.
BACKGROUND_TASKS: set[asyncio.Task] = set()

# One lock per (uid, episode) so two concurrent reflect requests for the
# same episode spend ONE LLM call: the second waits, then finds the cache.
# Review 2026-08-24: entries are reference-counted and dropped once no
# request holds or waits on them — the map used to grow by one Lock per
# (uid, episode) ever reflected, for the life of the process.
_REFLECT_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}
_REFLECT_LOCK_USERS: dict[tuple[str, str], int] = {}


@contextlib.asynccontextmanager
async def _reflect_lock(key: tuple[str, str]):
    _REFLECT_LOCK_USERS[key] = _REFLECT_LOCK_USERS.get(key, 0) + 1
    lock = _REFLECT_LOCKS.setdefault(key, asyncio.Lock())
    try:
        async with lock:
            yield
    finally:
        remaining = _REFLECT_LOCK_USERS[key] - 1
        if remaining <= 0:
            _REFLECT_LOCK_USERS.pop(key, None)
            _REFLECT_LOCKS.pop(key, None)
        else:
            _REFLECT_LOCK_USERS[key] = remaining


def live_recording_id(uid: str, session_id: str) -> str:
    """The deterministic recording id for a user's live session."""
    return str(uuid.uuid5(_LIVE_NAMESPACE, f"mindshift:live:{uid}:{session_id}"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LiveSessionIn(BaseModel):
    """The seam contract with Track 3-mobile (see the module docstring)."""
    session_id: str = Field(min_length=1, max_length=LIVE_SESSION_ID_MAX)
    started_at: str
    ended_at: str
    mode: LiveMode
    turns: list[TurnLocalEvent] = Field(min_length=1, max_length=LIVE_MAX_TURNS)
    tone_flags: list[ToneFlagEvent] = Field(default_factory=list)
    speaker_identities: list[SpeakerIdentityEvent] = Field(default_factory=list)
    # Optional user-facing title; absent → "Live session · <mode>".
    title: Optional[str] = Field(default=None, max_length=120)
    # Free-text context forwarded to the batch analysis + reflection prompts.
    context: str = Field(default="", max_length=500)
    # Escape hatches for a client that wants the record but not the spend.
    analyze: bool = True
    reflect: bool = True

    @field_validator("started_at", "ended_at")
    @classmethod
    def _iso(cls, value: str) -> str:
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError("must be an ISO-8601 timestamp")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "LiveSessionIn":
        # Every turn must belong to THIS session — a mixed payload is a
        # client bug we refuse at the door rather than store as one episode.
        for turn in self.turns:
            if turn.session_id != self.session_id:
                raise ValueError(
                    f"turn session_id {turn.session_id!r} does not match "
                    f"session_id {self.session_id!r}"
                )
        total = sum(len(t.text) for t in self.turns)
        if total > LIVE_MAX_TRANSCRIPT_CHARS:
            raise ValueError(
                f"transcript too large: {total} characters exceeds the "
                f"{LIVE_MAX_TRANSCRIPT_CHARS} limit"
            )
        return self


class LiveSessionOut(BaseModel):
    episode_id: str
    # The same id — every /recordings/{id} route (detail, share, delete,
    # speaker labels) works on it. Carried under both names so the seam's
    # ``episode_id`` and the existing client's ``recording_id`` both read.
    recording_id: str
    session_id: str
    created: bool  # False when a re-POST rewrote an existing episode
    turn_count: int
    self_speaker: Optional[str]
    analysis_status: str
    analysis_scheduled: bool
    reflect_scheduled: bool


class ReflectionOut(BaseModel):
    turn_index: int
    could_have_said: str
    why: str
    tone_read: str


class ReflectOut(BaseModel):
    episode_id: str
    self_speaker: str
    could_have_said: list[ReflectionOut]
    cached: bool
    reflected_at: Optional[str]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_store(request: Request) -> "recordings_store.RecordingsStore | None":
    return getattr(request.app.state, "recordings_store", None)


def _require_store(request: Request) -> "recordings_store.RecordingsStore":
    store = _get_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail=_STORAGE_DISABLED)
    return store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schedule(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return task


async def _rate_limit(request: Request) -> None:
    """main's per-IP limiter, imported lazily (see routers/voice.py)."""
    import main

    await main._rate_limit(request)


async def _complete_json(system: str, user: str, max_tokens: int) -> dict:
    """One LLM completion parsed as a JSON object, with /analyze's one
    corrective retry. Raises ``ValueError`` when both attempts fail to yield
    an object — the caller decides whether that is a 502 (on-demand) or a
    recorded failure (background)."""
    import main

    llm = main.get_llm_client()

    async def attempt(prompt: str) -> dict:
        # to_thread: llm.complete is a blocking SDK call — keep it off the
        # event loop (house pattern, see /respond).
        raw = await asyncio.to_thread(
            llm.complete, system=system, user=prompt, max_tokens=max_tokens,
        )
        parsed = main.parse_llm_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned invalid JSON")
        return parsed

    try:
        return await attempt(user)
    except (ValueError, IndexError, KeyError, TypeError):
        logger.info("reflection LLM retry after parse failure")
        try:
            return await attempt(user + "\n\n" + live_sessions.REFLECT_RETRY_SUFFIX)
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ValueError("LLM returned invalid JSON") from exc


def _effective_labels_of(rec: dict) -> dict[str, dict]:
    """The recording's speaker labels as the detail endpoint serves them:
    the stored ladder with the human's manual (name / person) overlay."""
    import main

    analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
    return main._effective_speaker_labels(
        analysis.get("speaker_labels"),
        rec.get("manual_speaker_labels") or {},
        main._recording_speaker_ids(rec),
        main._recording_manual_people(rec),
    )


def _self_speaker_of(rec: dict) -> str | None:
    """The speaker that is the viewing user in a stored recording — the same
    "me" rule /growth uses (``main._me_speaker`` over the EFFECTIVE labels,
    manual overlay applied), so an upload the user enrolled from can be
    reflected on too, and so a human correction ("that's me" on another
    speaker, or renaming the machine's "You") is honored here as well.

    Review 2026-08-24: the live block's ``self_speaker`` (the phone's
    verdict) used to be returned FIRST, before any manual label was
    consulted — so after the user corrected a wrong on-device identity,
    reflect still coached the wrong person's turns. It is now only the
    fallback for a live session whose self speaker the user never touched
    (a lite analysis always labels that speaker as the enrolled "You", so
    in practice the effective ladder already resolves it)."""
    import main

    analysis = rec.get("analysis")
    if not isinstance(analysis, dict):
        return None
    me = main._me_speaker(_effective_labels_of(rec))
    if me is not None:
        return me
    live = analysis.get("live")
    if isinstance(live, dict) and isinstance(live.get("self_speaker"), str):
        manual = rec.get("manual_speaker_labels") or {}
        if live["self_speaker"] not in manual:
            return live["self_speaker"]
    return None


async def _run_reflection(
    store: "recordings_store.RecordingsStore",
    uid: str,
    recording_id: str,
    *,
    context: str | None = None,
) -> tuple[list[dict], bool]:
    """Compute (or serve the cached) reflection for one episode and persist
    it on analysis.json. Returns ``(reflections, cached)``. Raises
    ``ValueError`` (LLM unusable / no self speaker) and ``LookupError``
    (recording vanished) for the caller to translate."""
    rec = await store.get_recording(uid, recording_id)
    if rec is None:
        raise LookupError(recording_id)
    turns = rec.get("turns") or []
    cached = live_sessions.cached_reflection(rec.get("analysis"), turns)
    if cached is not None:
        return cached, True
    self_label = _self_speaker_of(rec)
    if self_label is None:
        raise ValueError("no turn in this episode is identified as yours")
    analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
    live = analysis.get("live") if isinstance(analysis.get("live"), dict) else {}
    # EFFECTIVE labels (manual overlay applied) so the prompt names people
    # the way the user does — and never tags a speaker the user has
    # relabeled with the machine's stale "You" next to the real (YOU).
    user, self_indexes = live_sessions.build_reflect_prompt(
        turns, self_label, _effective_labels_of(rec),
        mode=live.get("mode"), context=context,
    )
    if not self_indexes:
        raise ValueError("no turn in this episode is identified as yours")
    data = await _complete_json(
        live_sessions.REFLECT_SYSTEM_PROMPT, user,
        live_sessions.reflect_max_tokens(len(self_indexes)),
    )
    reflections = live_sessions.parse_reflections(data, self_indexes)
    # Persist onto the CURRENT analysis (re-read so a concurrent batch pass
    # landing between our read and write isn't clobbered).
    fresh = await store.get_recording(uid, recording_id)
    if fresh is None:
        raise LookupError(recording_id)
    current = fresh.get("analysis") if isinstance(fresh.get("analysis"), dict) else {}
    current_live = dict(current.get("live") or {})
    current_live.setdefault("self_speaker", self_label)
    current_live["could_have_said"] = reflections
    current_live["reflection"] = {
        "reflected_at": _now_iso(),
        "turns_hash": live_sessions.turns_hash(turns),
        "self_turns": len(self_indexes),
    }
    await store.update_analysis(uid, recording_id, {**current, "live": current_live})
    return reflections, False


async def _post_ingest(
    store: "recordings_store.RecordingsStore",
    uid: str,
    recording_id: str,
    *,
    context: str,
    analyze: bool,
    reflect: bool,
) -> None:
    """The async tail of ingest: the batch LLM analysis (heats, report
    cards — what Growth scores and YourDay colors), then the reflection.
    Sequential on purpose (two LLM calls, one at a time per session). Every
    failure is RECORDED on the analysis (``analysis_status: failed`` +
    reason) rather than raised — a background task has no caller to
    surface to, and the stored session must stay readable."""
    import main

    rec = await store.get_recording(uid, recording_id)
    if rec is None:
        return
    turns = rec.get("turns") or []
    analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else None
    if analysis is None:
        return
    live = analysis.get("live") if isinstance(analysis.get("live"), dict) else {}
    self_label = live.get("self_speaker")

    if analyze and len(turns) >= main.ANALYZE_MIN_TURNS:
        try:
            analyze_turns = [
                main.AnalyzeTurn(
                    speaker=str(t.get("speaker")), text=str(t.get("text") or ""),
                    start_time=t.get("start_time"), end_time=t.get("end_time"),
                )
                for t in turns
            ]
            # The live identity report is in the matcher's multi shape, so
            # the batch pass's own label ladder labels "You" + matched
            # partners itself; merge_full_analysis re-overlays them anyway.
            full = await main._run_analysis(
                analyze_turns, context,
                speaker_identity=analysis.get("speaker_identity"),
            )
            merged = live_sessions.merge_full_analysis(
                analysis, full.model_dump(), turns, gap_seconds=_EPISODE_GAP_SECONDS,
            )
            await store.update_analysis(uid, recording_id, merged)
            analysis = merged
        except Exception as exc:  # noqa: BLE001 — recorded, never raised
            detail = getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Live-session batch analysis failed for uid=%s rid=%s: %s",
                uid, recording_id, detail,
            )
            failed_live = {
                **live,
                "analysis_status": live_sessions.ANALYSIS_FAILED,
                "analysis_error": str(detail)[:200],
            }
            analysis = {**analysis, "live": failed_live}
            await store.update_analysis(uid, recording_id, analysis)

    if reflect and isinstance(self_label, str):
        try:
            await _run_reflection(store, uid, recording_id, context=context or None)
        except (ValueError, LookupError) as exc:
            logger.warning(
                "Live-session reflection failed for uid=%s rid=%s: %s",
                uid, recording_id, exc,
            )
        except Exception:  # noqa: BLE001 — a background task must never crash the loop
            logger.warning(
                "Live-session reflection crashed for uid=%s rid=%s",
                uid, recording_id, exc_info=True,
            )


# ---------------------------------------------------------------------------
# POST /sessions/live
# ---------------------------------------------------------------------------

@router.post("/sessions/live", response_model=LiveSessionOut, status_code=201)
async def ingest_live_session(
    body: LiveSessionIn,
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    store = _require_store(request)
    recording_id = live_recording_id(uid, body.session_id)
    turn_events = [t.model_dump() for t in body.turns]
    turns = live_sessions.storage_turns(turn_events)
    tone_flags = [f.model_dump() for f in body.tone_flags]
    identities = [s.model_dump() for s in body.speaker_identities]
    title = (body.title or "").strip() or f"Live session · {body.mode}"

    # Foundation B: the phone's speaker_person_id values are the account's
    # voiceprint person ids — resolve their display names from the enrolled
    # documents so "with Mom" works even when the phone sent no identity
    # events. Best-effort: a store read failure just means raw labels.
    known_people: list[dict] = []
    list_people = getattr(store, "list_voiceprints", None)
    if callable(list_people):
        try:
            known_people = list(await list_people(uid) or [])
        except Exception:  # noqa: BLE001 — names are a nicety, never a blocker
            logger.warning("Voiceprint listing failed for uid=%s", uid, exc_info=True)

    analysis = live_sessions.lite_analysis(
        session_id=body.session_id,
        mode=body.mode,
        started_at=body.started_at,
        ended_at=body.ended_at,
        turns=turns,
        tone_flags=tone_flags,
        speaker_identities=identities,
        title=title,
        gap_seconds=_EPISODE_GAP_SECONDS,
        known_people=known_people,
    )
    existed = await store.get_recording(uid, recording_id)
    # Idempotent re-POST: keep what the previous ingest of the SAME words
    # already paid for (batch analysis, reflection) — see carry_over_previous.
    analysis = live_sessions.carry_over_previous(
        analysis, (existed or {}).get("analysis"), turns,
        gap_seconds=_EPISODE_GAP_SECONDS,
    )
    now = _now_iso()
    meta = {
        "id": recording_id,
        # created_at is WHEN THE CONVERSATION HAPPENED (the phone's
        # started_at), not when it reached us — YourDay buckets by it.
        "created_at": body.started_at,
        "ingested_at": now,
        "filename": "live-session",
        "title": title,
        # No audio on the server: "none" so the client never asks for media.
        "media_type": "none",
        "duration_seconds": live_sessions.duration_seconds(
            turns, body.started_at, body.ended_at,
        ),
        "size_bytes": 0,
        "stored_variants": [],
        "storage_note": "live session — no audio kept",
        "original_bytes": 0,
        "original_filename": None,
        "original_content_type": None,
        "source": {"type": "live", "url": None, "original_filename": None},
        "mode": body.mode,
        "session_id": body.session_id,
        "ended_at": body.ended_at,
    }
    try:
        await store.save_live_session(
            uid, recording_id, meta=meta, turns=turns, analysis=analysis,
        )
    except Exception as exc:  # noqa: BLE001 — honest 503, never a bare 500
        logger.warning("Live-session persistence failed for uid=%s: %s", uid, exc)
        raise HTTPException(
            status_code=503, detail=f"storage failed: {type(exc).__name__}",
        )

    import main

    self_label = analysis["live"]["self_speaker"]
    # Nothing is scheduled that the carried-over analysis already holds.
    analysis_scheduled = bool(
        body.analyze
        and len(turns) >= main.ANALYZE_MIN_TURNS
        and analysis["live"]["analysis_status"] != live_sessions.ANALYSIS_FULL
    )
    reflect_scheduled = bool(
        body.reflect
        and self_label is not None
        and analysis["live"].get("could_have_said") is None
    )
    if analysis_scheduled or reflect_scheduled:
        _schedule(_post_ingest(
            store, uid, recording_id,
            context=body.context, analyze=analysis_scheduled,
            reflect=reflect_scheduled,
        ))
    return LiveSessionOut(
        episode_id=recording_id,
        recording_id=recording_id,
        session_id=body.session_id,
        created=existed is None,
        turn_count=len(turns),
        self_speaker=self_label,
        analysis_status=analysis["live"]["analysis_status"],
        analysis_scheduled=analysis_scheduled,
        reflect_scheduled=reflect_scheduled,
    )


# ---------------------------------------------------------------------------
# POST /episodes/{id}/reflect
# ---------------------------------------------------------------------------

@router.post("/episodes/{episode_id}/reflect", response_model=ReflectOut)
async def reflect_episode(
    request: Request,
    episode_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    force: bool = False,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """"What you could have said" for the caller's OWN turns in one episode.

    Served from the cache stored on the episode when the transcript is
    unchanged (``cached: true`` — no LLM spend); ``force=true`` discards it
    and re-runs. Owner-only: a shared episode is 403 (the recipient can see
    it; reflecting is the owner's spend), a foreign/missing one 404."""
    store = _require_store(request)
    rec = await store.get_recording(uid, episode_id)
    if rec is None:
        if await store.find_share(uid, episode_id) is not None:
            raise HTTPException(status_code=403, detail=_SHARED_READ_ONLY)
        raise HTTPException(status_code=404, detail="Episode not found")
    self_label = _self_speaker_of(rec)
    if self_label is None:
        raise HTTPException(
            status_code=422,
            detail="no turn in this episode is identified as yours",
        )
    if force:
        analysis = rec.get("analysis") if isinstance(rec.get("analysis"), dict) else {}
        live = dict(analysis.get("live") or {})
        live["could_have_said"] = None
        live["reflection"] = None
        await store.update_analysis(uid, episode_id, {**analysis, "live": live})

    async with _reflect_lock((uid, episode_id)):
        try:
            reflections, cached = await _run_reflection(store, uid, episode_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="Episode not found")
        except ValueError as exc:
            # "no self" was checked above, so a ValueError here is the LLM
            # not returning anything parseable — an honest 502.
            raise HTTPException(status_code=502, detail=str(exc))
    fresh = await store.get_recording(uid, episode_id)
    reflection = (
        ((fresh or {}).get("analysis") or {}).get("live") or {}
    ).get("reflection") or {}
    return ReflectOut(
        episode_id=episode_id,
        self_speaker=self_label,
        could_have_said=[ReflectionOut(**r) for r in reflections],
        cached=cached,
        reflected_at=reflection.get("reflected_at"),
    )


# ---------------------------------------------------------------------------
# GET /sessions — therapist dashboard list
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_dashboard_sessions(
    request: Request,
    uid: str = Depends(get_current_uid),
):
    """Every analyzed episode the caller can see, in the dashboard's
    session shape: their own (patient label "You") and those SHARED with
    them by other accounts (patient label = the owner's email — the therapist
    → patient navigation is the existing read-only share grant). Newest
    first. Unanalyzed recordings are skipped (nothing to show)."""
    store = _require_store(request)
    metas = await store.list_recordings(uid)
    own_ids = [m["id"] for m in metas if m.get("has_analysis")]
    own = await asyncio.gather(*(store.get_recording(uid, rid) for rid in own_ids))
    sessions: list[dict] = [
        live_sessions.dashboard_session(rec, patient="You", shared=False)
        for rec in own if rec is not None
    ]
    for grant_meta in await store.list_shared_with(uid):
        if not grant_meta.get("has_analysis", True):
            continue
        grant = await store.find_share(uid, grant_meta["id"])
        if grant is None:
            continue
        rec = await store.get_recording(grant["owner_uid"], grant_meta["id"])
        if rec is None or not isinstance(rec.get("analysis"), dict):
            continue
        sessions.append(live_sessions.dashboard_session(
            rec, patient=grant.get("owner_email") or "Shared", shared=True,
        ))
    sessions.sort(key=lambda s: s.get("date") or "", reverse=True)
    return {"sessions": sessions}
