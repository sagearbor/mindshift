# Ported from gauge@2157433 server/auth.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Principal resolution: bearer-token verification or the legacy ``?account=``
query param, as one decision ladder shared by REST and WS entry points.

ADAPTED (Task B3): Gauge's ``FirebaseTokenVerifier`` managed its own
``firebase_admin`` app, scoped to ``GAUGE_FIREBASE_PROJECT`` (dropped from
this repo's env map — see docs/plans/2026-08-15-phase1-one-repo-one-engine.md's
global constraints). Here it instead reuses THIS repo's existing top-level
``server/auth.py`` — imported below as ``mindshift_auth`` — calling
``mindshift_auth.init_firebase()`` (idempotent, best-effort, never raises) to
ensure the shared default Firebase Admin app exists, then
``firebase_admin.auth.verify_id_token(token)`` against that default app. The
``{"sub": ..., "email": ...}`` return shape and "any verify failure ->
InvalidToken" exception mapping are unchanged from Gauge. Because
``FIREBASE_PROJECT_ID`` always has a default (see ``mindshift_auth``'s
module docstring), Firebase is treated as unconditionally available here —
there is no "unconfigured" state (unlike Gauge's ``get_verifier``, which
returned ``None`` when ``GAUGE_FIREBASE_PROJECT`` was unset). An actually
broken Firebase Admin setup (bad credentials, package missing, etc.) still
fails safe: every ``FirebaseTokenVerifier.verify()`` call raises
``InvalidToken`` for it, and ``ChainedTokenVerifier`` (see ``get_full_verifier``
below) falls through to ``DeviceTokenVerifier`` — an already-paired device
can keep authenticating even while Firebase itself is unusable.

``firebase_admin`` (via ``mindshift_auth``) is imported LAZILY, function-local,
at ``FirebaseTokenVerifier.verify()`` call time only — never at module import
time — so its absence can never break server startup, and a base install
without the package can still build the app and serve legacy ``?account=``
traffic. ``TokenVerifier`` is a one-method ``Protocol`` so this module is
fully testable with a hand-written fake (see ``server/tests/watch/test_auth.py``'s
``StubVerifier``) and never needs the real SDK or network calls in tests.

The decision ladder in ``resolve_principal`` is the security contract: a
bearer token that fails verification is ALWAYS a hard 401, never a silent
fallback to the legacy account param. Falling back would let anyone bypass
verification by sending a garbage token plus ``?account=<victim>``. Legacy
applies only when there is no bearer token at all.

FIX ROUND 3 ADDENDUM (cross-lane contract ruling, gauge-watch T9 review):
``resolve_principal`` distinguishes two ways ``verifier.verify()`` can fail,
and they map to DIFFERENT status codes:

- ``InvalidToken`` -- the token itself is bad (unknown, expired, malformed,
  wrong signature). This is a verdict ON THE CREDENTIAL -> **401**.
- ``VerifierUnavailable`` -- the verifier couldn't even reach a verdict
  because its backing store/service is transiently down (network timeout,
  Firestore outage, quota). This says nothing about the token -> **503**.

This split is a LOCKED cross-repo contract with the watch client: it treats
401 as "credentials genuinely revoked" and clears its stored device token,
forcing the wearer through re-pairing (which needs a second device) --
appropriate for a real bad token, but wrong for a transient backend hiccup,
which would silently sign a legitimate watch out. The watch already treats
any non-401 failure as retryable transport, so 503 fits its existing
contract with zero client-side changes. See ``DeviceTokenVerifier.verify()``
for the concrete case this was introduced for (a Firestore-backed
``PairingStore`` exception during device-token lookup).
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel

import auth as mindshift_auth
from watch.pairing_store import hash_secret

if TYPE_CHECKING:
    from watch.models import Account
    from watch.pairing_store import PairingStore
    from watch.store import LiveSessionStore


class InvalidToken(Exception):
    """A bearer token could not be verified. NEVER downgrades to legacy auth."""


class VerifierUnavailable(Exception):
    """The verifier could not reach a verdict on the token because its
    backing store/service is transiently unreachable -- NOT a statement
    about the token itself. Deliberately NOT a subclass of ``InvalidToken``:
    ``resolve_principal`` and ``ChainedTokenVerifier`` must treat the two
    differently (see module docstring's FIX ROUND 3 ADDENDUM) --
    ``ChainedTokenVerifier.verify()`` only catches ``InvalidToken`` to try
    the next verifier in the chain, so a ``VerifierUnavailable`` here
    propagates immediately rather than being masked as "this verifier
    rejected it, try the next one."
    """


class Principal(BaseModel):
    account_id: str
    email: str | None = None
    legacy: bool = False


class TokenVerifier(Protocol):
    def verify(self, token: str) -> dict:
        """Return {"sub": str, "email": str | None}. Raise InvalidToken otherwise."""
        ...


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from a case-insensitive ``"Bearer <token>"`` header.

    Returns ``None`` for a missing header, a non-Bearer scheme, or a blank
    remainder (e.g. ``"Bearer    "``).
    """
    if header is None:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def resolve_principal(
    auth_header: str | None,
    account_param: str | None,
    verifier: TokenVerifier | None,
    allow_legacy: bool,
) -> Principal:
    """The security-contract decision ladder — order matters, see module docstring."""
    token = parse_bearer(auth_header)

    if token is not None:
        if verifier is None:
            raise HTTPException(
                status_code=401, detail="token authentication is not configured on this server"
            )
        try:
            claims = verifier.verify(token)
        except VerifierUnavailable:
            # FIX ROUND 3 ADDENDUM: transient backend failure, not a bad
            # token -- see module docstring. Must NOT be 401 (the watch
            # client would clear its stored device token on a 401).
            raise HTTPException(
                status_code=503, detail="authentication service temporarily unavailable"
            )
        except InvalidToken:
            raise HTTPException(status_code=401, detail="invalid or expired token")
        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="token has no subject")
        return Principal(account_id=sub, email=claims.get("email"), legacy=False)

    if allow_legacy and account_param:
        return Principal(account_id=account_param, email=None, legacy=True)
    if allow_legacy:
        raise HTTPException(status_code=401, detail="missing account")
    raise HTTPException(status_code=401, detail="authentication required")


def resolve_ws_principal(
    token: str | None,
    account_param: str | None,
    verifier: TokenVerifier | None,
    allow_legacy: bool,
) -> Principal:
    """Same ladder as ``resolve_principal``, with the token arriving as a
    ``?token=`` query param instead of an ``Authorization`` header (a Wear OS
    OkHttp WebSocket *can* set headers, but the query form keeps the watch's
    URL builder trivial and matches the existing ``?account=`` convention).
    """
    auth_header = f"Bearer {token}" if token else None
    return resolve_principal(auth_header, account_param, verifier, allow_legacy)


class FirebaseTokenVerifier:
    """Verifies Google/Firebase ID tokens via THIS repo's shared Firebase
    Admin app (``server/auth.py``'s ``init_firebase()`` + default app) —
    see module docstring's ADAPTED note for why this differs from Gauge's
    own per-instance app management.

    ``firebase_admin`` is imported LAZILY (the same pattern as
    ``FirestoreLiveSessionStore._get_db`` in ``server/watch/store.py``) so a
    base install without the package — and the whole test suite, which never
    touches real Firebase — can still construct this class. Only
    ``.verify()`` imports the SDK.
    """

    def verify(self, token: str) -> dict:
        try:
            mindshift_auth.init_firebase()
            from firebase_admin import auth as fb_auth

            decoded = fb_auth.verify_id_token(token)
        except Exception as exc:  # noqa: BLE001 — any SDK/verify failure is an invalid token
            raise InvalidToken(str(exc)) from exc
        sub = decoded.get("uid") or decoded.get("sub")
        if not sub:
            raise InvalidToken("token has no subject")
        return {"sub": sub, "email": decoded.get("email")}


class DeviceTokenVerifier:
    """Verifies opaque device-pairing tokens minted by ``POST /me/pair/claim``
    (server/watch/routers/pairing.py — Task B8) — Wave C's watch sign-in
    companion (see docs/superpowers/plans/2026-08-04-gauge-wave-c-couples-wrist.md's
    Open Question 1). Full-auth-grade identity, exactly like
    ``FirebaseTokenVerifier``: a successful ``.verify()`` here produces a
    ``Principal`` with ``legacy=False`` downstream — ``resolve_principal``
    treats any non-``InvalidToken`` ``.verify()`` result as fully verified,
    with no distinction between "real Firebase ID token" and "real device
    token" once it gets past this Protocol boundary.

    The presented token is hashed (never compared/looked-up in plaintext —
    see ``watch/pairing_store.py``'s hash-at-rest contract) before the store
    lookup.
    """

    def __init__(self, store: "PairingStore") -> None:
        self._store = store

    def verify(self, token: str) -> dict:
        # FIX ROUND 1 (review Important finding): mirrors FirebaseTokenVerifier's
        # own defensive `except Exception` around its SDK call -- a store
        # failure must never propagate as an unhandled 500, and because
        # get_full_verifier tries Firebase first, an uncaught exception here
        # would hit EVERY bearer token Firebase rejects, not just genuine
        # device tokens.
        #
        # FIX ROUND 3 ADDENDUM (locked cross-repo contract with gauge-watch's
        # T9 review): a store EXCEPTION (transient infra failure -- network
        # timeout, Firestore outage, quota) is NOT the same thing as the
        # store cleanly reporting "no such token." The former says nothing
        # about the token's validity and must map to VerifierUnavailable ->
        # HTTPException(503) via resolve_principal, never InvalidToken ->
        # 401 -- the watch client clears its stored device token on 401,
        # treating it as "credentials genuinely revoked," which would
        # silently sign a legitimate watch out over a transient backend
        # hiccup. A CLEAN miss (record is None, no exception) is still a
        # genuine credential rejection -> InvalidToken -> 401, unchanged.
        try:
            record = self._store.get_device_token_by_hash(hash_secret(token))
        except Exception as exc:  # noqa: BLE001 — any store failure is transient-unavailable, never a 500 or a false 401
            raise VerifierUnavailable(str(exc)) from exc
        if record is None:
            raise InvalidToken("device token not recognized")
        return {"sub": record.account_id, "email": None}


class ChainedTokenVerifier:
    """Tries each verifier in order; the first to accept the token wins.

    Composes server-lane additions (e.g. ``DeviceTokenVerifier``) alongside
    ``FirebaseTokenVerifier`` WITHOUT changing ``resolve_principal``'s ladder
    at all — from ``resolve_principal``'s perspective this is still just one
    ``TokenVerifier``, so a verified Firebase ID token behaves EXACTLY as it
    did before this class existed: ``FirebaseTokenVerifier.verify()``
    succeeds on the first try (it's tried first — see ``get_full_verifier``)
    and this class never even looks at the remaining verifiers in the chain.
    A bearer token every verifier rejects re-raises the LAST verifier's
    ``InvalidToken`` (order is otherwise immaterial here: Firebase ID tokens
    are JWTs, device tokens are opaque random strings with no dots, so there
    is no realistic shape collision between the two).

    FIX ROUND 3 ADDENDUM: only ``InvalidToken`` is caught to advance to the
    next verifier -- a ``VerifierUnavailable`` (see its own docstring)
    propagates out of ``verify()`` immediately, uncaught, rather than being
    swallowed and treated as "this verifier said no, try the next one."
    """

    def __init__(self, verifiers: list[TokenVerifier]) -> None:
        if not verifiers:
            raise ValueError("ChainedTokenVerifier needs at least one verifier")
        self._verifiers = verifiers

    def verify(self, token: str) -> dict:
        error: InvalidToken = InvalidToken("no verifiers configured")
        for v in self._verifiers:
            try:
                return v.verify(token)
            except InvalidToken as exc:
                error = exc
        raise error


def get_full_verifier(pairing_store: "PairingStore") -> TokenVerifier:
    """Composes the real default verifier chain: ``FirebaseTokenVerifier``
    (see module docstring's ADAPTED note — unconditionally available, since
    ``FIREBASE_PROJECT_ID`` always has a default) tried FIRST, then
    ``DeviceTokenVerifier`` for Wave C's device-pairing flow.

    ADAPTED (Task B3): Gauge's ``get_full_verifier`` took a ``settings`` arg
    and returned a bare ``DeviceTokenVerifier`` (no ``ChainedTokenVerifier``
    wrapper) when ``GAUGE_FIREBASE_PROJECT`` was unset. That "unconfigured"
    state doesn't exist here — see the module docstring — so this always
    returns a two-element ``ChainedTokenVerifier``, Firebase first. A device
    token still authenticates even if Firebase itself is misconfigured,
    because ``FirebaseTokenVerifier.verify()`` fails with ``InvalidToken``
    (never ``VerifierUnavailable``) for any SDK-level problem, which
    ``ChainedTokenVerifier`` treats as "try the next verifier," not as a
    hard stop.
    """
    firebase = FirebaseTokenVerifier()
    device = DeviceTokenVerifier(pairing_store)
    return ChainedTokenVerifier([firebase, device])


AuthDep = Callable[..., Awaitable[Principal]]


async def ensure_account(store: "LiveSessionStore", principal: Principal) -> "Account":
    """Just-in-time provisioning: first verified sight of a uid writes an
    accounts doc; later sights refresh email/updated_at. Legacy principals
    NEVER write an accounts row — "default" is a transition artifact, not a
    real account."""
    # Function-local (not top-of-file) import of a concrete model: avoids
    # loading watch.models until an authenticated request actually needs
    # it, while still letting TYPE_CHECKING give real type-checking above.
    from watch.models import Account

    now = datetime.now(timezone.utc).isoformat()
    existing = await store.get_account(principal.account_id)
    if existing is None:
        account = Account(
            id=principal.account_id,
            provider="google",
            email=principal.email,
            created_at=now,
            updated_at=now,
        )
    else:
        account = existing.model_copy(
            update={"email": principal.email, "updated_at": now}
        )
    await store.put_account(account)
    return account


def make_auth_dependency(
    verifier: TokenVerifier | None, allow_legacy: bool, store: "LiveSessionStore"
) -> AuthDep:
    async def dependency(request: Request, account: str | None = Query(None)) -> Principal:
        principal = resolve_principal(
            request.headers.get("authorization"), account, verifier, allow_legacy
        )
        if not principal.legacy:
            await ensure_account(store, principal)
        return principal

    return dependency


def require_full_auth(auth: AuthDep) -> AuthDep:
    """Wrap an auth dependency so a legacy (unauthenticated ``?account=``)
    principal is a hard 401 instead of being let through.

    Controller ruling (server-track final review, I2/I3, ported verbatim
    from Gauge): with ``MINDSHIFT_ALLOW_LEGACY_ACCOUNT=true`` the plain
    ``auth`` dependency lets ANY caller impersonate ANY account by sending
    ``?account=<victim>`` and no token at all. That is fine for the surfaces
    the shipped watch and the phone's legacy-era account-override bridge
    depend on (live sessions, settings, enroll, /me, WS ingest, telemetry) —
    but it must NOT reach captures, groups, or account lookup-by-email,
    where an unauthenticated caller could read/join/leave someone else's
    data or resolve their email to an account id. Endpoints that need this
    stronger guarantee depend on the wrapped dependency instead of the raw
    one.
    """
    async def dependency(principal: Principal = Depends(auth)) -> Principal:
        if principal.legacy:
            raise HTTPException(status_code=401, detail="this endpoint requires sign-in")
        return principal

    return dependency
