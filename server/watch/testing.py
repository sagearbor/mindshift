# Ported from gauge@2157433 server/main.py's create_app; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Test-only app assembly — replaces Gauge's ``create_app()`` in ported tests.

``create_watch_test_app`` builds a bare FastAPI app and mounts ONLY the watch
routers, wired to in-memory stores by default. It is deliberately
**keyword-only** (no positional args at all) and takes NO environment
variables — gauge's ``create_app`` read ``GAUGE_ALLOW_LEGACY_ACCOUNT`` etc.
from ``Settings()``; every equivalent knob here (``allow_legacy``, the
store/verifier seams) is an explicit kwarg instead, so a test's auth
posture is visible at the call site rather than hidden in a monkeypatched
env var.

Task B5 mounted the REST router (``watch/routers/rest.py``: ``/me``,
``/me/standing``, ``/me/claim-legacy``, ``/accounts/lookup``,
``/live-sessions*``, ``/settings/vectors``, ``/enroll*``). Task B6 added the
groups router (``watch/routers/groups.py``: ``/groups*``) — mounted
unconditionally, same rationale as REST: ``MemoryLiveSessionStore`` is a
sensible default and every route requires ``strict_auth_dep`` regardless of
any other kwarg's availability. Task B7 adds the captures router
(``watch/routers/captures.py``: ``/captures*``) — also mounted
unconditionally: ``blobs`` defaults to ``None``, which is a legitimate,
already-tested runtime state for that router (every upload/download/delete
route honestly 503s when no blob store is configured — see
``test_captures_api.py``'s ``test_upload_without_blob_store_is_503_...``),
so there is no "absent dependency" case to gate the mount on. Task B8 (this
task) adds the pairing router (``watch/routers/pairing.py``: ``/me/pair/*``)
— mounted CONDITIONALLY on ``pairing_store is not None``, unlike REST/
groups/captures: a throwaway default ``MemoryPairingStore()`` minted
silently here would let a test that never asked for device-pairing at all
still see ``/me/pair/*`` "work", masking a missing dependency instead of
surfacing it (the "gated on its own dependency being meaningfully available"
branch the docstring below already called out before this task existed).
Every other kwarg below (``telemetry_store``, ``transcriber``, ``llm``,
``diarizer``) is still accepted but UNUSED — reserved for B9 telemetry, B10
post-session analysis, and B11 WS ingest to extend this same function
signature without breaking already-ported callers.

Task B8 also puts ``full_verifier`` to its first use (previously accepted
but unused): it builds a SECOND, independent auth dependency —
``strict_auth_dep = require_full_auth(make_auth_dependency(full_verifier or
verifier, ...))`` — feeding the full-auth-only routes (groups, captures, and
now pairing's ``/me/pair/claim``) a potentially STRONGER verifier chain than
the plain ``auth_dep`` built from ``verifier`` alone, without touching any
already-ported caller: every existing test omits ``full_verifier``, so it
falls back to ``verifier`` and reproduces the exact pre-B8 behavior
(``strict_auth_dep`` built from the same verifier as ``auth_dep``). This is
what lets a test wire the REAL ``watch.auth.get_full_verifier(pairing_store)``
chain (Firebase-then-DeviceToken) onto just the full-auth surface — see
``server/tests/watch/test_pairing_api.py``'s
``test_end_to_end_device_token_authenticates_as_full_auth_via_the_real_verifier_chain``.

**Extension pattern for B9-B11**: each later task adds its router's mount
here, gated on its own dependency being meaningfully available (as B8 did
for ``pairing_store`` above), or "mount unconditionally with a default
store" for a router that (like REST, groups, and captures) always has
something sensible to run against in tests. Mount each new router with its
own ``app.include_router(...)`` call, immediately after the existing mounts
— never replace or wrap an earlier mount. The auth dependencies
(``auth_dep``/``strict_auth_dep``) are built ONCE, here, and reused by every
router task's mount, so every route shares one identity resolution ladder
(see ``watch/auth.py``'s module docstring) no matter which task's router it
lives in.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from watch.auth import TokenVerifier, make_auth_dependency, require_full_auth
from watch.routers.captures import make_captures_router
from watch.routers.groups import make_groups_router
from watch.routers.pairing import make_pairing_router
from watch.routers.rest import make_rest_router
from watch.store import LiveSessionStore, MemoryLiveSessionStore


def create_watch_test_app(
    *,
    store: LiveSessionStore | None = None,
    pairing_store: Any = None,
    telemetry_store: Any = None,
    blobs: Any = None,
    verifier: TokenVerifier | None = None,
    full_verifier: TokenVerifier | None = None,
    transcriber: Any = None,
    llm: Any = None,
    diarizer: Any = None,
    embedder: Any = None,
    allow_legacy: bool = False,
) -> FastAPI:
    """Assemble a throwaway FastAPI app for ported-test use.

    ``store`` defaults to a fresh ``MemoryLiveSessionStore()`` when omitted —
    most tests build their own store first (to seed fixtures, then assert
    against it after making requests) and pass it in explicitly instead.

    ``verifier`` is the ``TokenVerifier`` the plain ``auth_dep`` is built
    from: a legacy ``?account=`` principal is accepted by ``auth_dep``
    whenever ``allow_legacy`` is True. ``strict_auth_dep`` (the
    ``require_full_auth``-wrapped dependency every full-auth-only route
    uses — groups, captures, and pairing's ``/me/pair/claim``) is built from
    ``full_verifier`` when given, falling back to ``verifier`` otherwise —
    see this module's docstring's Task B8 paragraph for why (lets a test
    wire the REAL ``get_full_verifier`` chain onto just the full-auth
    surface). A legacy principal is always hard-rejected by
    ``strict_auth_dep`` regardless of ``allow_legacy`` or which verifier fed
    it (that's the whole point of ``require_full_auth`` — see its own
    docstring).
    """
    app = FastAPI()

    resolved_store: LiveSessionStore = store if store is not None else MemoryLiveSessionStore()

    auth_dep = make_auth_dependency(verifier, allow_legacy, resolved_store)
    # Task B8: strict_auth_dep now sources from `full_verifier` when given,
    # instead of always reusing `auth_dep` verbatim — see this function's
    # own docstring and the module docstring's Task B8 paragraph. Every
    # pre-B8 caller omits `full_verifier`, so `strict_verifier` resolves to
    # `verifier` and this is byte-for-byte the old `require_full_auth(auth_dep)`.
    strict_verifier = full_verifier if full_verifier is not None else verifier
    strict_auth_dep = require_full_auth(
        make_auth_dependency(strict_verifier, allow_legacy, resolved_store)
    )

    # Task B5: REST router — always mounted (it has no dependency that would
    # ever be "absent" in a test; MemoryLiveSessionStore is a sensible
    # default). Later router tasks (B6-B11) each add one more
    # `app.include_router(...)` call here, gated on their own kwarg per this
    # module's docstring, without touching this line.
    app.include_router(make_rest_router(resolved_store, auth_dep, strict_auth_dep, embedder=embedder))

    # Task B6: groups router — always mounted (like REST above, it has no
    # dependency that would ever be "absent" in a test). Every route in it
    # requires strict_auth_dep (full-auth only, no legacy `?account=` caller)
    # per watch/auth.py's require_full_auth — see watch/routers/groups.py's
    # module docstring for the I2/I3 controller ruling this preserves.
    app.include_router(make_groups_router(resolved_store, strict_auth_dep))

    # Task B7: captures router — always mounted (like REST and groups
    # above). `blobs` defaults to None, which is a legitimate runtime state
    # the router itself handles (503 on upload/download/delete) rather than
    # something this assembly function needs to gate the mount on — see the
    # module docstring's B7 paragraph. Every route requires strict_auth_dep,
    # same I2/I3 posture as groups.
    app.include_router(make_captures_router(resolved_store, blobs, strict_auth_dep))

    # Task B8: pairing router — CONDITIONALLY mounted, unlike REST/groups/
    # captures above (see module docstring's Task B8 paragraph for why a
    # throwaway default pairing_store would mask a missing dependency
    # instead of surfacing it). Uses strict_auth_dep for /me/pair/claim
    # (the only route in this router that takes any auth at all — /me/pair/
    # start and /me/pair/status are deliberately unauthenticated, see
    # watch/routers/pairing.py's module docstring).
    if pairing_store is not None:
        app.include_router(make_pairing_router(pairing_store, strict_auth_dep))

    return app
