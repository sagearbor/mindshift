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

Task B5 (this task) mounts only the REST router (``watch/routers/rest.py``:
``/me``, ``/me/standing``, ``/me/claim-legacy``, ``/accounts/lookup``,
``/live-sessions*``, ``/settings/vectors``, ``/enroll*``). Every other kwarg
below (``pairing_store``, ``telemetry_store``, ``blobs``, ``full_verifier``,
``transcriber``, ``llm``, ``diarizer``) is accepted now but UNUSED — they are
reserved so later router tasks (B6 groups, B7 captures, B8 pairing, B9
telemetry, B10 post-session analysis, B11 WS ingest) can extend this same
function signature without breaking already-ported callers.

**Extension pattern for B6-B11**: each later task adds its router's mount
here, gated on its own dependency being meaningfully available — e.g.
``if pairing_store is not None: app.include_router(make_pairing_router(...))``
for a router whose whole PURPOSE is that dependency, or "mount
unconditionally with a default store" for a router that (like REST here)
always has something sensible to run against in tests. Mount each new
router with its own ``app.include_router(...)`` call, immediately after the
existing mounts — never replace or wrap an earlier mount. The auth
dependencies (``auth_dep``/``strict_auth_dep``) are built ONCE, here, and
reused by every router task's mount, so every route shares one identity
resolution ladder (see ``watch/auth.py``'s module docstring) no matter which
task's router it lives in.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from watch.auth import TokenVerifier, make_auth_dependency, require_full_auth
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

    ``verifier`` is the single ``TokenVerifier`` both the plain and
    full-auth dependencies are built from (``strict_auth_dep =
    require_full_auth(auth_dep)`` — see ``watch/auth.py``): a legacy
    ``?account=`` principal is accepted by ``auth_dep`` whenever
    ``allow_legacy`` is True, and always hard-rejected by ``strict_auth_dep``
    regardless of ``allow_legacy`` (that's the whole point of
    ``require_full_auth`` — see its own docstring). ``full_verifier`` is
    reserved for a future device-pairing router task (B8's
    ``ChainedTokenVerifier`` composition — see ``watch/auth.py``'s
    ``get_full_verifier``) and is currently unused.
    """
    app = FastAPI()

    resolved_store: LiveSessionStore = store if store is not None else MemoryLiveSessionStore()

    auth_dep = make_auth_dependency(verifier, allow_legacy, resolved_store)
    strict_auth_dep = require_full_auth(auth_dep)

    # Task B5: REST router — always mounted (it has no dependency that would
    # ever be "absent" in a test; MemoryLiveSessionStore is a sensible
    # default). Later router tasks (B6-B11) each add one more
    # `app.include_router(...)` call here, gated on their own kwarg per this
    # module's docstring, without touching this line.
    app.include_router(make_rest_router(resolved_store, auth_dep, strict_auth_dep, embedder=embedder))

    return app
