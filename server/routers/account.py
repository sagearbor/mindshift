"""Account router — ``DELETE /me``, self-serve account deletion.

Google Play requires any app that lets a user create an account to let the
same user delete it, in-app, without emailing a human. This is that endpoint:
one call that erases every tier of storage this server keeps for the caller's
uid and then deletes the Firebase Auth user itself.

WHAT IT DELETES is documented — exhaustively, tier by tier, including the
shared-data rule and the short list of things that genuinely cannot be reached
— in ``server/account_deletion.py``'s module docstring. That file is the single
source of truth; the privacy policy, the public /delete-account page and the
Play data-safety answer pack all restate the same rules in user-facing words.

Kept OUT of main.py for the same reason routers/voice.py is: main.py is already
5.5k lines, and this feature touches enough stores that it deserves its own
edit surface. It reaches the recordings store off ``app.state`` and imports
``main`` lazily (the circular-import dance every router here does).

Three guards, all deliberate:

* **The uid is the only identity.** ``get_fresh_uid`` returns a uid verified
  from the token's own signed claims. Nothing in the path, query or body names
  an account, so a therapist can never delete a patient (nor a patient a
  therapist) — the endpoint has no way to be pointed at another account, not
  merely a check that says no.
* **A fresh token.** Not ``get_current_uid`` but ``get_fresh_uid`` (see
  server/auth.py): a Firebase ID token stays valid for ~an hour after it is
  minted, and an irreversible delete should not be reachable with one that has
  been sitting around. The client force-refreshes right before calling.
* **Type-to-confirm.** The body must be exactly ``{"confirm": "DELETE"}``;
  anything else is a 422 from the model itself. This is not decoration — a
  bodyless DELETE is the single easiest request to fire by accident, and this
  one cannot be undone.

Plus a per-IP budget far tighter than the generic one (see
``_DELETE_RATE_LIMIT_PER_MINUTE``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import account_deletion
import auth
from auth import get_fresh_uid

logger = logging.getLogger(__name__)

router = APIRouter(tags=["account"])

# Deleting an account is a once-in-an-account's-lifetime action, so the budget
# is deliberately tiny — several attempts a minute is already a client bug or
# an attack, never a real person. Same fixed-window algorithm as
# main._RateLimiter and routers/voice.py's catch-up limiter, self-contained for
# the same reason (this router doesn't import main at module load).
_DELETE_RATE_LIMIT_PER_MINUTE = int(
    os.getenv("ACCOUNT_DELETE_RATE_LIMIT_PER_MINUTE", "3")
)


class _DeleteRateLimiter:
    """Fixed-window per-key request counter, scoped to DELETE /me only."""

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


_delete_rate_limiter = _DeleteRateLimiter(_DELETE_RATE_LIMIT_PER_MINUTE)


async def _delete_rate_limit(request: Request) -> None:
    """A much tighter per-IP budget than main's generic limiter. Honors the
    same ``RATE_LIMIT_ENABLED`` escape hatch (read lazily — main isn't imported
    at module load) so the test suite can disable rate limiting globally the
    way it already does everywhere else."""
    import main  # lazy — see the module docstring on the circular import

    if not main.RATE_LIMIT_ENABLED:
        return
    client = request.client
    key = client.host if client else "unknown"
    if not await _delete_rate_limiter.allow(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — too many requests; please slow down.",
        )


class DeleteAccountRequest(BaseModel):
    """The type-to-confirm guard, as a model so FastAPI answers 422 for a
    missing body, a wrong value and a malformed one alike — one rejection
    path, no hand-rolled check to drift."""

    confirm: Literal["DELETE"] = Field(
        description=(
            'Must be exactly "DELETE". The client asks the user to type it; '
            "the server refuses anything else."
        ),
    )


class DeleteAccountResponse(BaseModel):
    """What was actually erased, so the client can show it rather than assert
    a generic "done"."""

    deleted: bool
    """True when the whole walk succeeded, Firebase Auth user included."""

    firebase_user_deleted: bool
    """Whether a Firebase Auth user existed and was removed. False on the
    idempotent second call — the data walk still ran and still found nothing."""

    counts: dict[str, int]
    """Per-category counts; every key in ``account_deletion.COUNT_KEYS`` is
    always present, so a zero reads as "none of these", never as "unknown"."""


@router.delete("/me", response_model=DeleteAccountResponse)
async def delete_me(
    body: DeleteAccountRequest,
    request: Request,
    uid: str = Depends(get_fresh_uid),
    _rl: None = Depends(_delete_rate_limit),
) -> DeleteAccountResponse:
    """Permanently delete the caller's account and everything under it.

    Acts on the verified token's uid and nothing else. Runs every storage tier
    (``account_deletion.delete_account_data``), and only if every one of them
    succeeded deletes the Firebase Auth user — last, on purpose: a failure
    part-way through leaves an account that can still sign in and retry,
    instead of data orphaned behind an unusable login.

    Idempotent: calling it again on an already-deleted account walks the same
    tiers, finds nothing, reports zeros and ``firebase_user_deleted: false``.

    A tier that failed makes this a **500** whose detail carries both the
    failed tiers and the counts that DID succeed, so the client can say what
    happened and the user can retry — never a fabricated success."""
    import main  # lazy — see the module docstring

    started = time.monotonic()
    store = getattr(request.app.state, "recordings_store", None)
    watch_deps = getattr(request.app.state, "watch_deps", None)

    db = await main.get_db()
    try:
        summary = await account_deletion.delete_account_data(
            uid,
            recordings_store=store,
            watch_store=getattr(watch_deps, "store", None),
            pairing_store=getattr(watch_deps, "pairing_store", None),
            telemetry_store=getattr(watch_deps, "telemetry_store", None),
            blobs=getattr(watch_deps, "blobs", None),
            db=db,
        )
    finally:
        await db.close()

    if not summary.ok:
        logger.error(
            "Account deletion INCOMPLETE uid=%s failed_tiers=%s deleted=%s "
            "elapsed=%.2fs — Firebase user left in place so the user can retry",
            uid, summary.errors, summary.counts, time.monotonic() - started,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Some of your data could not be deleted, so your account "
                    "was left in place. Nothing was half-deleted silently — "
                    "please try again."
                ),
                "failed": summary.errors,
                "counts": summary.counts,
            },
        )

    try:
        summary.firebase_user_deleted = await asyncio.to_thread(
            auth.delete_firebase_user, uid,
        )
    except Exception:  # noqa: BLE001 — reported honestly, never swallowed
        logger.exception("Firebase Auth user delete failed uid=%s", uid)
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Your data was deleted, but your sign-in account could not "
                    "be removed. Please try again — signing in will find "
                    "nothing left."
                ),
                "failed": ["firebase_auth"],
                "counts": summary.counts,
            },
        )

    logger.info(
        "Account deleted uid=%s firebase_user_deleted=%s counts=%s "
        "total_items=%d elapsed=%.2fs",
        uid, summary.firebase_user_deleted, summary.counts, summary.total,
        time.monotonic() - started,
    )
    return DeleteAccountResponse(
        deleted=True,
        firebase_user_deleted=summary.firebase_user_deleted,
        counts=summary.counts,
    )
