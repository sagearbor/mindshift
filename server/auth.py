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


def verify_id_token(token: str) -> str:
    """Verify a Firebase ID token and return its ``uid``.

    Raises on anything wrong with the token (bad signature, expired, wrong
    audience/issuer, or an unusable SDK). The returned uid is taken only from
    the verified claims.
    """
    from firebase_admin import auth as fb_auth

    init_firebase()
    decoded = fb_auth.verify_id_token(token)
    return decoded["uid"]


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
