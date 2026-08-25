"""In-app calls — the REST half (server/calls.py holds the model).

* ``POST /calls``             — create a call: the caller is the host
                                (a coached participant, slot A, ``"Speaker
                                A"``). Optional ``invitee_email`` (resolved
                                to an account — 404 when none, 400 for
                                yourself) or ``invitee_uid``; without
                                either, anyone with the join code may join.
                                ``max_participants`` 2 or 3 (default 3: two
                                participants + one therapist). Returns
                                ``call_id``, the short ``join_code``, a
                                ``join_url`` for a text-message invite, and
                                the ICE servers.
* ``POST /calls/join``        — join by code alone (the invite link only
                                carries the code). ``role`` = participant
                                (default; coached, slot B) or therapist
                                (observer, slot C, never coached).
* ``POST /calls/{id}/join``   — join a known call: the named invitee needs
                                no code, anyone else must send ``join_code``.
                                At most two participants and one therapist
                                (409 when the seat is taken or the call is
                                full); idempotent for a member.
* ``GET  /calls/{id}``        — the call as the caller sees it (participants
                                with ``is_self``/``connected``, the labels,
                                ICE servers, and — once ended — the caller's
                                own ``episode_id``). Participants and the
                                invitee only; anyone else gets 404.
* ``POST /calls/{id}/end``    — hang up: persists one episode per participant
                                and tells the other socket ``call_ended``.
                                Idempotent.

Signaling (``rtc_signal``), binding a socket (``call_join``), the merged
transcript and per-participant coaching all happen over the EXISTING
``/ws/session/{id}`` WebSocket — one auth path, see audio_pipeline.py.

``main`` is imported lazily inside handlers (the same circular-import
discipline the other routers follow) so the email resolvers tests
monkeypatch on ``main`` are honoured.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from pydantic import BaseModel, Field

import calls
from audio_pipeline import UUID_PATTERN
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calls", tags=["calls"])

_UID_MAX = 128


CallRole = Literal["participant", "therapist"]


class CallCreateIn(BaseModel):
    # The other participant, by MindShift account email (resolved server-side)
    # or uid. Both optional: an open call admits anyone holding the code.
    invitee_email: Optional[str] = Field(default=None, min_length=3, max_length=320)
    invitee_uid: Optional[str] = Field(default=None, min_length=1, max_length=_UID_MAX)
    # How the host wants to be shown on the others' screens ("Sage").
    display_name: Optional[str] = Field(default=None, max_length=calls.DISPLAY_NAME_MAX)
    ttl_minutes: Optional[int] = Field(default=None, ge=1, le=calls.CALL_TTL_MAX_MINUTES)
    # Total members: 2 (a plain two-way call) or 3 (two coached participants
    # + one observing therapist). Default 3.
    max_participants: Optional[int] = Field(default=None, ge=2, le=3)


class CallJoinIn(BaseModel):
    join_code: Optional[str] = Field(default=None, max_length=16)
    display_name: Optional[str] = Field(default=None, max_length=calls.DISPLAY_NAME_MAX)
    # "participant" (coached; at most two per call, the host is one) or
    # "therapist" (observer; at most one). Default participant.
    role: Optional[CallRole] = None


class CallJoinByCodeIn(BaseModel):
    join_code: str = Field(min_length=1, max_length=16)
    display_name: Optional[str] = Field(default=None, max_length=calls.DISPLAY_NAME_MAX)
    role: Optional[CallRole] = None


class IceServerOut(BaseModel):
    urls: list[str]
    username: Optional[str] = None
    credential: Optional[str] = None


class CallParticipantOut(BaseModel):
    uid: str
    slot: str
    label: str
    role: CallRole
    display_name: str
    is_self: bool
    connected: bool
    joined_at: str


class CallInviteeOut(BaseModel):
    uid: Optional[str] = None
    email: Optional[str] = None


class CallOut(BaseModel):
    call_id: str
    status: str
    host_uid: str
    max_participants: int
    self_uid: str
    self_role: Optional[CallRole]
    self_label: Optional[str]
    # The OTHER coached participant's label (fixed by slot even before they
    # join); the therapist's label is always "Speaker C".
    peer_label: Optional[str]
    therapist_label: str
    therapist_uid: Optional[str]
    participants: list[CallParticipantOut]
    invitee: Optional[CallInviteeOut]
    ice_servers: list[IceServerOut]
    join_code: str
    join_url: str
    invitee_uid: Optional[str]
    invitee_email: Optional[str]
    created_at: str
    expires_at: str
    started_at: Optional[str]
    ended_at: Optional[str]
    end_reason: Optional[str]
    turn_count: int
    # The caller's OWN episode, once the call ended and storage persisted it,
    # and the therapist emails it was auto-shared with (the linked therapist).
    episode_id: Optional[str]
    shared_with: list[str] = []


def _store(request: Request):
    return getattr(request.app.state, "recordings_store", None)


async def _rate_limit(request: Request) -> None:
    import main

    await main._rate_limit(request)


def _raise(exc: calls.CallError) -> None:
    raise HTTPException(status_code=exc.status, detail=exc.detail)


def _clean_email(raw: str) -> str:
    email = raw.strip().lower()
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise HTTPException(status_code=422, detail="enter a valid email address")
    return email


def _visible(call_id: str, uid: str) -> calls.Call:
    call = calls.registry.get(call_id)
    if call is None or not call.can_see(uid):
        # A foreign call reads as absent — never confirm another account's call.
        raise HTTPException(status_code=404, detail="no such call")
    return call


@router.post("", response_model=CallOut, status_code=201)
async def create_call(
    body: CallCreateIn,
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    import main

    invitee_uid = body.invitee_uid
    invitee_email = _clean_email(body.invitee_email) if body.invitee_email else None
    if invitee_email and not invitee_uid:
        invitee_uid = await asyncio.to_thread(main.resolve_uid_by_email, invitee_email)
        if invitee_uid is None:
            raise HTTPException(status_code=404, detail="no MindShift account with that email")
    if invitee_uid == uid:
        raise HTTPException(status_code=400, detail="you can't call yourself")
    host_email = await calls.resolve_email(uid, main.resolve_email_by_uid)
    try:
        call = calls.registry.create(
            uid,
            host_email=host_email,
            display_name=calls.clean_display_name(body.display_name),
            invitee_uid=invitee_uid,
            invitee_email=invitee_email,
            ttl_minutes=body.ttl_minutes,
            max_participants=body.max_participants,
        )
    except calls.CallError as exc:
        _raise(exc)
    return call.rest_view(uid)


@router.post("/join", response_model=CallOut)
async def join_by_code(
    body: CallJoinByCodeIn,
    request: Request,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    import main

    code = calls.normalize_join_code(body.join_code)
    call = calls.registry.by_code(code) if code else None
    if call is None:
        raise HTTPException(status_code=404, detail="no call with that join code")
    email = await calls.resolve_email(uid, main.resolve_email_by_uid)
    try:
        calls.registry.join(
            call, uid, join_code=code, email=email,
            display_name=calls.clean_display_name(body.display_name), role=body.role,
        )
    except calls.CallError as exc:
        _raise(exc)
    await call.broadcast_state()
    return call.rest_view(uid)


@router.get("/{call_id}", response_model=CallOut)
async def get_call(
    call_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    return _visible(call_id, uid).rest_view(uid)


@router.post("/{call_id}/join", response_model=CallOut)
async def join_call(
    body: CallJoinIn,
    request: Request,
    call_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    import main

    call = calls.registry.get(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="no such call")
    email = await calls.resolve_email(uid, main.resolve_email_by_uid)
    try:
        calls.registry.join(
            call, uid, join_code=body.join_code, email=email,
            display_name=calls.clean_display_name(body.display_name), role=body.role,
        )
    except calls.CallError as exc:
        _raise(exc)
    # The host's phone (if already on the socket) learns the peer joined.
    await call.broadcast_state()
    return call.rest_view(uid)


@router.post("/{call_id}/end", response_model=CallOut)
async def end_call(
    request: Request,
    call_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    call = _visible(call_id, uid)
    me = call.participant(uid)
    if me is None or me.is_therapist:
        # The invitee who never joined, and the observing therapist (a
        # member, not a coached participant), may not hang up for everyone.
        raise HTTPException(status_code=403, detail="only a participant can end the call")
    await call.end(reason="ended", ended_by=uid, store=_store(request))
    return call.rest_view(uid)
