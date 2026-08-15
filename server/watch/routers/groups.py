# Ported from gauge@2157433 server/groups_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B6): Episode -> LiveSession per the locked "episode" rename
# map: `store.list_episodes` -> `store.list_live_sessions`, `EpisodeStore` ->
# `LiveSessionStore`. `make_groups_router`'s single `auth` param is now the
# FULL-auth dependency (`watch/auth.py`'s `require_full_auth`-wrapped one,
# built once in `watch/testing.py`/the real app assembly and passed in here)
# -- this mirrors Gauge's own I2/I3 controller ruling verbatim (every route
# in this router already required a non-legacy principal there too), just
# made explicit in the signature instead of being a `server.auth` module-level
# default. `PAIR_MAX_MEMBERS` now comes from `watch.store` (Task B2) instead
# of `server.store`.
#
# CARRY-OVER NOTE (B2 review): `server/tests/watch/test_store.py` carried
# inlined, behaviorally-equivalent copies of `make_invite_mutator`/
# `make_join_mutator`/`make_leave_mutator` (with a `TODO(Task B6)` directive)
# because this router didn't exist yet when those store-atomicity tests were
# written. This task re-points those tests to import the REAL mutators below
# and deletes the copies -- see that file's diff.
"""Groups API: create pair/team, invite, join, list, leave.

Mutual-visibility consent has exactly one source of truth — a
``ConsentRecord(kind="mutual_visibility")`` on the group document — never a
member boolean (see ``watch/models.py``'s ``GroupMember`` docstring). Every
handler sits behind ``Depends(auth)`` (the FULL-auth dependency — see this
module's own adaptation note above), the same pattern as ``rest.py``'s
strict-auth-only routes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from watch.aggregates import group_standing, member_standing
from watch.auth import AuthDep, Principal
from watch.models import ConsentRecord, Group, GroupInvite, GroupKind, GroupMember, GroupStanding
from watch.store import PAIR_MAX_MEMBERS, LiveSessionStore

INVITE_CODE_CHARS = 8
DEFAULT_PERIOD_DAYS = 7


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreateGroupRequest(BaseModel):
    kind: GroupKind = "pair"
    name: str = ""


class InviteRequest(BaseModel):
    email: str | None = None


class JoinRequest(BaseModel):
    code: str


def _mutual_visibility_consent(account_id: str) -> ConsentRecord:
    """Membership + this record is the ONLY gate for mutual visibility —
    ``confirmed=True`` because the actor is consenting on their own behalf,
    from their own device, by the act of creating/joining the group."""
    return ConsentRecord(
        id=uuid.uuid4().hex,
        participant_id=account_id,
        kind="mutual_visibility",
        attested_by=account_id,
        confirmed=True,
        ts=_now_iso(),
    )


async def _get_group_or_404(store: LiveSessionStore, group_id: str) -> Group:
    group = await store.get_group(group_id)
    if group is None:
        raise HTTPException(status_code=404, detail="group not found")
    return group


def _require_member(group: Group, account_id: str) -> None:
    if not any(m.account_id == account_id for m in group.members):
        raise HTTPException(status_code=403, detail="only members may do this")


def make_invite_mutator(account: str, email: str | None) -> Callable[[Group | None], Group]:
    """Builds the invite-mint mutator run inside `store.update_group_atomically`.

    The PAIR_MAX_MEMBERS check runs against the atomic seam's freshly
    (consistently) read `group`, not a value read earlier in the request —
    this closes the TOCTOU window where two concurrent invites (or an
    invite racing a join) could otherwise both pass a stale "there's still
    room" check before either write lands. Raising here aborts the whole
    operation with nothing persisted; the router lets the HTTPException
    propagate unchanged.
    """
    def mutate(group: Group | None) -> Group:
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        _require_member(group, account)
        if group.kind == "pair" and len(group.members) >= PAIR_MAX_MEMBERS:
            raise HTTPException(status_code=409, detail="this pair is already full")
        group.invites.append(GroupInvite(
            code=uuid.uuid4().hex[:INVITE_CODE_CHARS],
            email=email,
            invited_by=account,
            created_at=_now_iso(),
        ))
        return group
    return mutate


def make_join_mutator(account: str, code: str) -> Callable[[Group | None], Group]:
    """Builds the join mutator run inside `store.update_group_atomically`.

    All three invariants — single-use invite code, "already a member", and
    PAIR_MAX_MEMBERS — are re-checked here against the atomic seam's fresh
    read, so two concurrent joins (same code, or two different codes on the
    same pair) can never both pass: whichever runs second sees the first's
    already-committed write and is correctly rejected instead of racing it.
    """
    def mutate(group: Group | None) -> Group:
        if group is None:
            raise HTTPException(status_code=404, detail="invite not found")
        invite_rec = next((inv for inv in group.invites if inv.code == code), None)
        if invite_rec is None:
            raise HTTPException(status_code=404, detail="invite not found")
        if invite_rec.accepted_by is not None:
            raise HTTPException(status_code=409, detail="invite already accepted")
        if any(m.account_id == account for m in group.members):
            raise HTTPException(status_code=409, detail="already a member of this group")
        if group.kind == "pair" and len(group.members) >= PAIR_MAX_MEMBERS:
            raise HTTPException(status_code=409, detail="this pair is already full")

        now = _now_iso()
        group.members.append(GroupMember(account_id=account, joined_at=now))
        group.consents.append(_mutual_visibility_consent(account))
        invite_rec.accepted_by = account
        invite_rec.accepted_at = now
        return group
    return mutate


def make_leave_mutator(account: str) -> Callable[[Group | None], Group]:
    """Builds the leave mutator run inside `store.update_group_atomically`
    (M1 fold-in: leave now goes through the SAME atomic seam as every other
    group mutation — invite mint and join — instead of a plain read then
    ``store.put_group``, which could silently drop a concurrent invite/join
    write it raced against."""
    def mutate(group: Group | None) -> Group:
        if group is None:
            raise HTTPException(status_code=404, detail="group not found")
        if not any(m.account_id == account for m in group.members):
            raise HTTPException(status_code=404, detail="not a member of this group")

        # Consent records are retained as an audit trail — visibility is
        # revoked immediately because the standing endpoint gates on
        # CURRENT membership AND consent, not on consent history alone.
        group.members = [m for m in group.members if m.account_id != account]
        return group
    return mutate


def make_groups_router(store: LiveSessionStore, auth: AuthDep) -> APIRouter:
    router = APIRouter()

    @router.post("/groups", response_model=Group)
    async def create_group(
        body: CreateGroupRequest, principal: Principal = Depends(auth)
    ) -> Group:
        account = principal.account_id
        now = _now_iso()
        group = Group(
            id=uuid.uuid4().hex,
            kind=body.kind,
            name=body.name,
            created_by=account,
            created_at=now,
            members=[GroupMember(account_id=account, joined_at=now)],
            consents=[_mutual_visibility_consent(account)],
        )
        await store.put_group(group)
        return group

    @router.post("/groups/{group_id}/invite", response_model=Group)
    async def invite(
        group_id: str, body: InviteRequest, principal: Principal = Depends(auth)
    ) -> Group:
        # The membership/PAIR_MAX checks live INSIDE the mutator, which runs
        # atomically inside store.update_group_atomically — see
        # make_invite_mutator's docstring for why this can't be a plain
        # read-check-write here in the router.
        return await store.update_group_atomically(
            group_id, make_invite_mutator(principal.account_id, body.email)
        )

    @router.post("/groups/join", response_model=Group)
    async def join(body: JoinRequest, principal: Principal = Depends(auth)) -> Group:
        # This lookup only resolves WHICH group the code belongs to; it is
        # not the authoritative check. The authoritative single-use/
        # already-member/PAIR_MAX checks re-run inside make_join_mutator
        # against the atomic seam's fresh read, so a stale result here can
        # never cause an invariant violation — see make_join_mutator's
        # docstring.
        lookup = await store.get_group_by_invite_code(body.code)
        if lookup is None:
            raise HTTPException(status_code=404, detail="invite not found")
        return await store.update_group_atomically(
            lookup.id, make_join_mutator(principal.account_id, body.code)
        )

    @router.get("/groups", response_model=list[Group])
    async def list_groups(principal: Principal = Depends(auth)) -> list[Group]:
        return await store.list_groups(principal.account_id)

    @router.delete("/groups/{group_id}/me", response_model=Group)
    async def leave(group_id: str, principal: Principal = Depends(auth)) -> Group:
        # Runs inside store.update_group_atomically — see
        # make_leave_mutator's docstring for why this can't be a plain
        # read-check-write here in the router.
        return await store.update_group_atomically(
            group_id, make_leave_mutator(principal.account_id)
        )

    @router.get("/groups/{group_id}/standing", response_model=GroupStanding)
    async def group_standing_endpoint(
        group_id: str,
        period_days: int = Query(DEFAULT_PERIOD_DAYS, ge=1, le=90),
        principal: Principal = Depends(auth),
    ) -> GroupStanding:
        group = await _get_group_or_404(store, group_id)
        _require_member(group, principal.account_id)

        consenting = {
            c.participant_id
            for c in group.consents
            if c.kind == "mutual_visibility" and c.confirmed
        }
        if len(group.members) < 2 or not all(m.account_id in consenting for m in group.members):
            raise HTTPException(
                status_code=409,
                detail="mutual visibility requires every member's consent",
            )

        now = datetime.now(timezone.utc)
        standings = []
        for m in group.members:
            live_sessions = [
                ls for ls in await store.list_live_sessions(m.account_id)
                if ls.owner_account == m.account_id
            ]
            account = await store.get_account(m.account_id)
            display_name = account.display_name if account is not None else None
            standings.append(
                member_standing(m.account_id, live_sessions, now, period_days, display_name)
            )

        return group_standing(group_id, standings, now, period_days)

    return router
