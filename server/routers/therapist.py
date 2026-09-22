"""Therapist-link router — the two-sided "patient + their therapist" setup.

Patient side (the account whose sessions get shared):

* ``GET    /therapist/link``   — the current link (``{linked: false}`` when none).
* ``PUT    /therapist/link``   — name a therapist by account email. Resolves
                                the email to a uid (404 when no account, 400
                                for yourself), writes a PENDING link with
                                ``auto_share`` on, and returns the patient view.
                                Re-PUT with another email replaces the link.
* ``PATCH  /therapist/link``   — flip ``auto_share``.
* ``DELETE /therapist/link``   — unlink (204). Existing per-episode grants
                                are NOT revoked here (they are the patient's
                                to manage per episode in Replay).

Therapist side:

* ``GET  /therapist/patients``                    — every patient that named
                                                    this account (pending +
                                                    accepted), oldest first.
                                                    Each row carries the
                                                    patient's consent scopes
                                                    and ``last_seen_at`` (the
                                                    dashboard's unread mark).
* ``POST /therapist/patients/{uid}/accept``       — acknowledge; the patient
                                                    moves into the patient list.
* ``POST /therapist/patients/{uid}/decline``      — remove the link (204).
                                                    Nothing further is auto-
                                                    shared; grants already
                                                    made stay until the
                                                    patient revokes them.
* ``POST /therapist/patients/{uid}/seen``         — "I have read up to now"
                                                    for THIS therapist. Marks
                                                    the link's
                                                    ``therapist_last_seen_at``
                                                    so a patient with a newer
                                                    session shows as unread.

Consent (server/consent.py), patient side:

* ``POST /therapist/consent`` — grant / revoke one scope (``episodes``: the
  therapist may see stored sessions; ``live``: she may observe a call as it
  happens). ``PUT /therapist/link`` grants ``episodes`` on the way in — the
  client shows the disclosure above the email field — so linking is never a
  silent hand-over; revoking ``episodes`` stops auto-share without unlinking
  or touching episodes already shared. ``GET /therapist/link`` returns the
  scopes AND the exact disclosure wording, so no client hard-codes it.

Export:

* ``GET /therapist/export/{episode_id}?format=text|pdf`` — one session as a
  document for a patient file (transcript, tone, escalation markers, "what
  could have been said", and the CALLER's own private notes). Visibility is
  the notes rule: own it or hold a share grant, else 404.

Notes (private to whoever writes them — a therapist's notes on a patient's
session are never visible to the patient, and vice versa):

* ``GET /therapist/notes/{episode_id}`` / ``PUT`` / ``DELETE`` — the caller
  must be able to SEE the episode (own it, or hold a share grant); anything
  else is a 404 (never confirming a foreign id).

Why a router: the sharing endpoints in main.py are untouched — this file
consumes ``store.add_share`` through server/therapist_links.py and adds
nothing to the grant model. ``main`` is imported lazily inside handlers (the
same circular-import discipline routers/sessions.py and routers/voice.py
follow) so the email resolvers tests monkeypatch on ``main`` are honoured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, Field

import consent as consent_mod
import live_sessions
import therapist_export
import therapist_links
from audio_pipeline import UUID_PATTERN
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/therapist", tags=["therapist"])

_STORAGE_DISABLED = "recording storage is not enabled"
_UID_MAX = 128


class LinkRequest(BaseModel):
    # The therapist's MindShift account email; the server resolves it. Loose
    # bounds only — existence is decided by the Firebase lookup.
    email: str = Field(min_length=3, max_length=320)


class LinkPatch(BaseModel):
    auto_share: bool


class ConsentIn(BaseModel):
    # "episodes" (the therapist may see stored sessions) or "live" (she may
    # observe a call as it happens). Validated against consent.SCOPES.
    scope: str = Field(min_length=1, max_length=32)
    granted: bool = True


class SeenOut(BaseModel):
    patient_uid: str
    last_seen_at: str


class NoteIn(BaseModel):
    text: str = Field(max_length=therapist_links.NOTE_MAX_CHARS)


class NoteOut(BaseModel):
    episode_id: str
    text: str
    updated_at: Optional[str]


def _require_store(request: Request):
    store = getattr(request.app.state, "recordings_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail=_STORAGE_DISABLED)
    return store


async def _rate_limit(request: Request) -> None:
    import main

    await main._rate_limit(request)


def _clean_email(raw: str) -> str:
    email = raw.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(status_code=422, detail="enter a valid email address")
    return email


# ---------------------------------------------------------------------------
# Patient side
# ---------------------------------------------------------------------------

@router.get("/link")
async def get_link(request: Request, uid: str = Depends(get_current_uid)):
    store = _require_store(request)
    return therapist_links.patient_view(await store.read_therapist_link(uid))


@router.put("/link")
async def set_link(
    body: LinkRequest,
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    import main

    store = _require_store(request)
    email = _clean_email(body.email)
    therapist_uid = await asyncio.to_thread(main.resolve_uid_by_email, email)
    if therapist_uid is None:
        raise HTTPException(status_code=404, detail="no MindShift account with that email")
    if therapist_uid == uid:
        raise HTTPException(status_code=400, detail="you can't be your own therapist")
    patient_email = await asyncio.to_thread(main.resolve_email_by_uid, uid)
    existing = await store.read_therapist_link(uid)
    if existing and existing.get("therapist_uid") == therapist_uid:
        # Same therapist again: keep status/accepted_at, refresh the email.
        link = {**existing, "therapist_email": email, "patient_email": patient_email}
    else:
        link = therapist_links.new_link(
            patient_uid=uid, patient_email=patient_email,
            therapist_uid=therapist_uid, therapist_email=email,
        )
    # Naming a therapist IS agreeing that they see your sessions — the client
    # shows consent.DISCLOSURES["episodes"] above the field. Recorded here so
    # a patient file can name the wording and the date (idempotent: a re-PUT
    # keeps the ORIGINAL "consented since"). "live" stays a separate tap.
    link = consent_mod.grant(link, granted_by=uid, scope=consent_mod.SCOPE_EPISODES)
    await store.write_therapist_link(uid, link)
    return therapist_links.patient_view(link)


@router.post("/consent")
async def set_consent(
    body: ConsentIn,
    request: Request,
    uid: str = Depends(get_current_uid),
):
    """Grant or revoke ONE consent scope on the caller's therapist link.

    Revoking ``episodes`` stops auto-share at the next ingest and leaves
    everything already shared alone (un-sharing a specific episode is
    Replay's job — the patient can see and revoke each grant there).
    Revoking ``live`` means a therapist joining a call the patient is on
    needs the in-call approval instead of being let straight in."""
    store = _require_store(request)
    try:
        scope = consent_mod.clean_scope(body.scope)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    link = await store.read_therapist_link(uid)
    if link is None:
        raise HTTPException(status_code=404, detail="no therapist linked")
    link = (
        consent_mod.grant(link, granted_by=uid, scope=scope)
        if body.granted else consent_mod.revoke(link, scope)
    )
    await store.write_therapist_link(uid, link)
    return therapist_links.patient_view(link)


@router.patch("/link")
async def patch_link(
    body: LinkPatch,
    request: Request,
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    link = await store.read_therapist_link(uid)
    if link is None:
        raise HTTPException(status_code=404, detail="no therapist linked")
    link = {**link, "auto_share": body.auto_share}
    await store.write_therapist_link(uid, link)
    return therapist_links.patient_view(link)


@router.delete("/link", status_code=204)
async def delete_link(request: Request, uid: str = Depends(get_current_uid)):
    store = _require_store(request)
    await store.delete_therapist_link(uid)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Therapist side
# ---------------------------------------------------------------------------

@router.get("/patients")
async def list_patients(request: Request, uid: str = Depends(get_current_uid)):
    store = _require_store(request)
    links = await store.list_therapist_patients(uid)
    return {
        "patients": [
            therapist_links.therapist_view(link)
            for link in links
            if link.get("therapist_uid") == uid and link.get("patient_uid")
        ],
    }


async def _patient_link_for(store, therapist_uid: str, patient_uid: str) -> dict:
    link = await store.read_therapist_link(patient_uid)
    if link is None or link.get("therapist_uid") != therapist_uid:
        # A foreign/absent link reads as 404 — never confirm another
        # account's therapist choice.
        raise HTTPException(status_code=404, detail="no such patient link")
    return link


@router.post("/patients/{patient_uid}/accept")
async def accept_patient(
    request: Request,
    patient_uid: Annotated[str, Path(min_length=1, max_length=_UID_MAX)],
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    link = await _patient_link_for(store, uid, patient_uid)
    if link.get("status") != therapist_links.STATUS_ACCEPTED:
        link = {
            **link,
            "status": therapist_links.STATUS_ACCEPTED,
            "accepted_at": therapist_links.now_iso(),
        }
        await store.write_therapist_link(patient_uid, link)
    return therapist_links.therapist_view(link)


@router.post("/patients/{patient_uid}/seen", response_model=SeenOut)
async def mark_patient_seen(
    request: Request,
    patient_uid: Annotated[str, Path(min_length=1, max_length=_UID_MAX)],
    uid: str = Depends(get_current_uid),
):
    """"I have read this patient up to now." Stamps the link with THIS
    therapist's read mark, so a dashboard on another device agrees about
    which patients have something new. A patient whose newest session is
    later than the mark shows as unread; a patient the therapist never
    opened has no mark and reads as unread the moment they share anything.

    Stored on the link (which is per patient AND per therapist — a patient
    names one therapist), so there is no second document to keep in step."""
    store = _require_store(request)
    link = await _patient_link_for(store, uid, patient_uid)
    seen_at = therapist_links.now_iso()
    await store.write_therapist_link(
        patient_uid, {**link, "therapist_last_seen_at": seen_at},
    )
    return SeenOut(patient_uid=patient_uid, last_seen_at=seen_at)


@router.post("/patients/{patient_uid}/decline", status_code=204)
async def decline_patient(
    request: Request,
    patient_uid: Annotated[str, Path(min_length=1, max_length=_UID_MAX)],
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    await _patient_link_for(store, uid, patient_uid)
    await store.delete_therapist_link(patient_uid)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

async def _require_visible(store, uid: str, episode_id: str) -> None:
    if await store.recording_exists(uid, episode_id):
        return
    if await store.find_share(uid, episode_id) is not None:
        return
    raise HTTPException(status_code=404, detail="Episode not found")


@router.get("/notes/{episode_id}", response_model=NoteOut)
async def get_note(
    request: Request,
    episode_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    await _require_visible(store, uid, episode_id)
    note = await store.read_therapist_note(uid, episode_id) or {}
    return NoteOut(
        episode_id=episode_id,
        text=str(note.get("text") or ""),
        updated_at=note.get("updated_at"),
    )


@router.put("/notes/{episode_id}", response_model=NoteOut)
async def put_note(
    body: NoteIn,
    request: Request,
    episode_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    await _require_visible(store, uid, episode_id)
    text = body.text.strip()
    if not text:
        await store.delete_therapist_note(uid, episode_id)
        return NoteOut(episode_id=episode_id, text="", updated_at=None)
    note = {"text": text, "updated_at": therapist_links.now_iso()}
    await store.write_therapist_note(uid, episode_id, note)
    return NoteOut(episode_id=episode_id, text=text, updated_at=note["updated_at"])


@router.delete("/notes/{episode_id}", status_code=204)
async def delete_note(
    request: Request,
    episode_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    store = _require_store(request)
    await _require_visible(store, uid, episode_id)
    await store.delete_therapist_note(uid, episode_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Export — one session as a document for a patient file
# ---------------------------------------------------------------------------

@router.get("/export/{episode_id}")
async def export_episode(
    request: Request,
    episode_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    format: str = Query(default="text", pattern="^(text|pdf)$"),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """The session the caller is looking at, as text or PDF (server/
    therapist_export.py): transcript, tone, escalation markers, "what could
    have been said", the sharing/consent record, and the CALLER's own private
    notes.

    Visibility is the notes rule — own the episode or hold a share grant,
    anything else is a 404 (never confirming a foreign id). The notes
    included are the caller's own by construction (notes are keyed by
    viewer), so one therapist's export can never carry another's."""
    import main

    store = _require_store(request)
    owner_uid = uid
    patient = "You"
    if not await store.recording_exists(uid, episode_id):
        grant = await store.find_share(uid, episode_id)
        if grant is None:
            raise HTTPException(status_code=404, detail="Episode not found")
        owner_uid = grant.get("owner_uid") or ""
        patient = grant.get("owner_email") or "Shared with you"
    rec = await store.get_recording(owner_uid, episode_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    session = live_sessions.dashboard_session(
        rec, patient=patient, shared=owner_uid != uid,
    )
    note = await store.read_therapist_note(uid, episode_id) or {}
    exported_by = await asyncio.to_thread(main.resolve_email_by_uid, uid) or uid
    kwargs = {
        "note": str(note.get("text") or ""),
        "patient": patient,
        "exported_by": exported_by,
        "exported_at": therapist_links.now_iso(),
    }
    if format == "pdf":
        return Response(
            content=therapist_export.build_pdf(session, **kwargs),
            media_type="application/pdf",
            headers={
                "Content-Disposition":
                    f"attachment; filename=mindshift_session_{episode_id}.pdf",
            },
        )
    return Response(
        content=therapist_export.build_text(session, **kwargs),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition":
                f"attachment; filename=mindshift_session_{episode_id}.txt",
        },
    )
