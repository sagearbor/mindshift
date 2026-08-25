"""Firebase authentication — ID-token verification for REST + WebSocket.

Every data route and the audio WebSocket require a verified Firebase user. The
verified ``uid`` is the ONLY trusted identity: it comes straight from the
signed token's claims, never from a request body or query param. This is a
therapy-adjacent product, so the rule is absolute — no cross-user data may be
read or written on an unverified (or another user's) identity.

Verification uses the Firebase Admin SDK initialized with Application Default
Credentials (ADC) plus an explicit project id. On Cloud Run inside the
``arborfam-hub`` project ADC resolves with no key file and no secret; verifying
ID tokens needs only the project id and Google's public signing keys.

``firebase_admin`` is imported lazily inside the functions that use it, so this
module (and the whole test suite) import cleanly even where the package is
absent or no credentials exist. Tests override :func:`get_current_uid` and/or
monkeypatch :func:`verify_id_token`; they never touch real Firebase.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# The Firebase/GCP project that mints the ID tokens. On Cloud Run this is the
# same project the service runs in, so ADC needs no key file. Overridable via
# env for a different deployment without a code change.
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "arborfam-hub")

# Guard so initialization is attempted at most once per process.
_init_attempted = False


def init_firebase() -> None:
    """Initialize the Firebase Admin SDK once, with ADC + the project id.

    Idempotent and best-effort: a missing package or an unresolved credential
    is logged, not raised, so server startup never dies for an auth-config
    reason (and a keyless CI import stays clean). Verification still fails
    *closed* — an unusable SDK makes :func:`verify_id_token` raise, which
    callers turn into a 401, never an open door.
    """
    global _init_attempted
    if _init_attempted:
        return
    _init_attempted = True
    try:
        import firebase_admin

        if not firebase_admin._apps:
            firebase_admin.initialize_app(
                options={"projectId": FIREBASE_PROJECT_ID},
            )
        logger.info(
            "Firebase Admin initialized (project=%s)", FIREBASE_PROJECT_ID,
        )
    except Exception:  # noqa: BLE001 — startup must not die on auth config
        logger.warning(
            "Firebase Admin init deferred/failed — token verification will "
            "reject until this is resolved",
            exc_info=True,
        )


def verify_id_token_claims(token: str) -> dict:
    """Verify a Firebase ID token and return its full decoded claims.

    The single place the Admin SDK is asked to verify. :func:`verify_id_token`
    is the thin "just the uid" wrapper over this; :func:`get_fresh_uid` needs
    the ``iat``/``auth_time`` claims as well, and must never re-verify with a
    second, independently-drifting code path.
    """
    from firebase_admin import auth as fb_auth

    init_firebase()
    return fb_auth.verify_id_token(token)


def verify_id_token(token: str) -> str:
    """Verify a Firebase ID token and return its ``uid``.

    Raises on anything wrong with the token (bad signature, expired, wrong
    audience/issuer, or an unusable SDK). The returned uid is taken only from
    the verified claims.
    """
    return verify_id_token_claims(token)["uid"]


def delete_firebase_user(uid: str) -> bool:
    """Permanently delete the Firebase Auth user ``uid``.

    Returns ``True`` when a user existed and was deleted, ``False`` when there
    was no such user (the idempotent re-run of an account deletion that already
    got this far). Every OTHER failure — an unusable SDK, a transient Admin
    error — propagates, exactly like :func:`resolve_uid_by_email`'s contract:
    only a genuine "no such account" is allowed to read as a clean no-op, so a
    broken delete is never reported as a successful one.

    Called LAST by ``DELETE /me`` (routers/account.py) so a failure anywhere
    earlier leaves the account signed-in-able and the deletion retryable,
    rather than orphaning data behind an account nobody can authenticate as.
    """
    from firebase_admin import auth as fb_auth

    init_firebase()
    try:
        fb_auth.delete_user(uid)
    except fb_auth.UserNotFoundError:
        return False
    return True


def resolve_uid_by_email(email: str) -> str | None:
    """Resolve a MindShift account's Firebase uid from its email, or ``None`` when
    no account has that email.

    Used by the account-to-account recording share flow (POST
    /recordings/{id}/shares): the owner names a recipient by email, which only the
    server can turn into the trusted uid the grant is stored against. Firebase
    Admin's ``get_user_by_email`` is the authoritative lookup. A genuinely absent
    account (``UserNotFoundError``) returns ``None`` — the ONE honest "no such
    account" signal the endpoint needs; every OTHER failure (unusable SDK, transient
    error) is re-raised so the caller reports a 5xx rather than a false "not found"
    that would let a real email be denied. The blocking SDK call is the caller's to
    offload (``asyncio.to_thread``); this stays a thin sync wrapper like
    :func:`verify_id_token`."""
    from firebase_admin import auth as fb_auth

    init_firebase()
    try:
        user = fb_auth.get_user_by_email(email)
    except fb_auth.UserNotFoundError:
        return None
    return user.uid


def resolve_email_by_uid(uid: str) -> str | None:
    """Resolve an account's email from its Firebase uid, or ``None`` when unknown.

    Used when recording a share grant so the RECIPIENT's list can show who a
    recording came from ("from linda@…") — the owner's email, looked up once at
    share time from the trusted owner uid. A missing user/email returns ``None``
    (the recipient simply sees no from-address rather than a fabricated one);
    unexpected SDK failures propagate."""
    from firebase_admin import auth as fb_auth

    init_firebase()
    try:
        user = fb_auth.get_user(uid)
    except fb_auth.UserNotFoundError:
        return None
    return getattr(user, "email", None)


async def get_current_uid(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: the verified Firebase uid from ``Authorization``.

    Expects ``Authorization: Bearer <idToken>``. Rejects with 401 on a missing
    or malformed header and on an invalid/expired token.
    """
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme != "Bearer" or not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return verify_id_token(token)
    except HTTPException:
        raise
    except Exception:
        # Never leak provider internals (they can carry key ids / request urls).
        raise HTTPException(status_code=401, detail="invalid or expired token")


# How recently an ID token must have been MINTED for the irreversible account
# deletion (DELETE /me) to accept it. Firebase ID tokens live ~1 h and refresh
# silently, so a token in flight or scraped off a device stays usable for the
# rest of that hour; requiring a freshly issued one means the caller must still
# hold the refresh credential *right now* and is deliberately re-asking for a
# token seconds before the destructive call (the client does exactly that —
# ``getFreshToken(true)`` → Firebase ``getIdToken(true)``).
#
# Honest about what this is and is not: it proves CURRENT possession of the
# refresh credential, not that a human just re-entered a password. It closes
# the replayed-stale-token window; it does not replace a re-authentication
# prompt, and the type-to-confirm step in the UI is the other half of the
# guard. Ten minutes is generous enough to survive a slow phone, a paused
# confirm dialog and clock skew without ever prompting a real user twice.
FRESH_TOKEN_MAX_AGE_SECONDS = int(
    os.getenv("MINDSHIFT_FRESH_TOKEN_MAX_AGE_SECONDS", "600")
)


def token_age_seconds(claims: dict, *, now: float | None = None) -> float | None:
    """How long ago the ID token in ``claims`` was minted, in seconds.

    Reads ``iat`` (issued-at), falling back to ``auth_time`` (last sign-in)
    when a token carries no ``iat``. ``None`` means neither claim was present
    — the caller MUST treat that as "not fresh" (fail closed), never as fresh.
    A negative age (the issuer's clock ahead of ours) is clamped to 0 rather
    than reported as stale, so ordinary skew can't lock a user out.
    """
    stamp = claims.get("iat", claims.get("auth_time"))
    try:
        issued = float(stamp)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    current = time.time() if now is None else now
    return max(0.0, current - issued)


async def get_fresh_uid(authorization: str = Header(default="")) -> str:
    """FastAPI dependency: a verified uid from a *freshly issued* ID token.

    Same contract as :func:`get_current_uid` plus the freshness gate described
    on :data:`FRESH_TOKEN_MAX_AGE_SECONDS`. A valid but stale token is a 401
    with a distinct, actionable detail so the client knows to force-refresh (or
    ask the user to sign in again) rather than showing a generic auth error.

    Used only by irreversible account-scoped actions — today, ``DELETE /me``.
    """
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme != "Bearer" or not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        claims = verify_id_token_claims(token)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    age = token_age_seconds(claims)
    if age is None or age > FRESH_TOKEN_MAX_AGE_SECONDS:
        raise HTTPException(
            status_code=401,
            detail=(
                "a freshly issued sign-in token is required for this action — "
                "sign in again and retry"
            ),
        )
    return claims["uid"]
