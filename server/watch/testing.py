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
Task B9 puts ``telemetry_store`` to its first use: the telemetry router
(``watch/routers/telemetry.py``: ``POST``/``GET /telemetry``) is mounted
UNCONDITIONALLY — like REST/groups/captures, not gated like pairing_store —
because a throwaway default ``MemoryTelemetryStore()`` carries no security
posture to mask (unlike a throwaway pairing store) and is a perfectly
sensible thing for a test to run against. Both of its routes are
deliberately unauthenticated (see that router's own module docstring), so
this mount takes no auth dependency at all.

Task B11 puts ``transcriber``/``llm``/``diarizer`` to their first use: the WS
ingest router (``watch/routers/ws.py``: ``WEBSOCKET /ws/live-session/{id}``)
and the live-session re-analyze router (``watch/routers/live_sessions.py``:
``POST /live-sessions/{id}/analyze``) are BOTH mounted unconditionally, same
"always has something sensible to run against in tests" rationale as REST/
groups/captures/telemetry above — ``transcriber``/``llm``/``diarizer``
default to ``None``, a legitimate runtime state both routers already handle
honestly (no live-analysis spawn without a transcriber wired for the WS
router's ``stt`` gate below; a 500-free `None.transcribe` is never reached
because no ported test calls ``/analyze`` without injecting fakes first).
The new ``stt`` kwarg (default ``"none"``) is the explicit, env-var-free
knob this function's docstring's Task B8 paragraph already established the
pattern for — see ``watch/routers/ws.py``'s own ADAPTED note for why gauge's
single ``settings: Settings | None`` param became these explicit kwargs
instead of a `Settings()` built here from the process environment.

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

**Extension pattern for B10-B11**: each later task adds its router's mount
here, gated on its own dependency being meaningfully available (as B8 did
for ``pairing_store`` above), or "mount unconditionally with a default
store" for a router that (like REST, groups, captures, and now telemetry)
always has something sensible to run against in tests. Mount each new router
with its
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
from watch.routers.live_sessions import make_live_sessions_router
from watch.routers.pairing import make_pairing_router
from watch.routers.rest import make_rest_router
from watch.routers.telemetry import make_telemetry_router
from watch.routers.ws import make_ws_router
from watch.store import LiveSessionStore, MemoryLiveSessionStore
from watch.telemetry_store import MemoryTelemetryStore


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
    # Task B11: gates the WS "end" handler's fire-and-forget analysis spawn
    # (watch/routers/ws.py's `if stt != "none":`) — mirrors watch/config.py's
    # Settings.stt naming/values ("whisper" | "none" | anything-else-is-null),
    # but defaults to "none" here (NOT Settings' own "whisper" default):
    # this is a TEST assembly, and most callers never inject a transcriber —
    # spawning fire-and-forget analysis by default would either silently
    # invoke a real, slow local Whisper model (no transcriber given) or just
    # be dead weight. Same rationale as conftest.py's
    # `MINDSHIFT_WATCH_STT=none` setdefault. Tests exercising the real spawn
    # path pass `stt="whisper"` explicitly alongside a FakeTranscriber/FakeLLM.
    stt: str = "none",
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
    # Task P3-6: `pairing_store` is passed straight through (not gated like
    # the pairing router's own mount below) — GET /me handles a None
    # pairing_store itself (honest `has_paired_watch=False` default, see
    # rest.py's `me()`), so there is no "absent dependency" case to hide
    # here; a test that wires pairing_store only to exercise /me/pair/* (no
    # /me assertions) is unaffected either way.
    app.include_router(
        make_rest_router(
            resolved_store, auth_dep, strict_auth_dep, embedder=embedder, pairing_store=pairing_store,
        )
    )

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
    # `diarizer=` (journal A/B): threaded through so tests can hand the journal
    # self-filtering hook a fake DiarizationService; None (the default) leaves
    # journal captures honestly unfiltered ("diarization_unavailable").
    app.include_router(make_captures_router(resolved_store, blobs, strict_auth_dep, diarizer=diarizer))

    # Task B8: pairing router — CONDITIONALLY mounted, unlike REST/groups/
    # captures above (see module docstring's Task B8 paragraph for why a
    # throwaway default pairing_store would mask a missing dependency
    # instead of surfacing it). Uses strict_auth_dep for /me/pair/claim
    # (the only route in this router that takes any auth at all — /me/pair/
    # start and /me/pair/status are deliberately unauthenticated, see
    # watch/routers/pairing.py's module docstring).
    if pairing_store is not None:
        app.include_router(make_pairing_router(pairing_store, strict_auth_dep))

    # Task B9: telemetry router — mounted unconditionally with a default
    # MemoryTelemetryStore(), same "always has something sensible to run
    # against in tests" rationale as REST/groups/captures above (NOT gated
    # like pairing_store: a throwaway in-memory telemetry store carries no
    # security posture to mask, unlike a throwaway pairing store — see this
    # module's docstring). Mirrors gauge's own `create_app`, which resolves
    # `resolved_telemetry = telemetry if telemetry is not None else
    # get_telemetry_store()` unconditionally. Takes no auth dependency at
    # all — both routes are deliberately unauthenticated, see
    # watch/routers/telemetry.py's module docstring.
    resolved_telemetry = telemetry_store if telemetry_store is not None else MemoryTelemetryStore()
    app.include_router(make_telemetry_router(resolved_telemetry))

    # Task B11: WS ingest router — mounted unconditionally (see module
    # docstring's Task B11 paragraph). WS auth is resolved from the SAME
    # `verifier`/`allow_legacy` values `auth_dep` above was built from, not a
    # separately-constructed `Settings()` — keeps WS auth posture identical
    # to every REST route in this same test app.
    app.include_router(
        make_ws_router(
            resolved_store, transcriber, llm,
            verifier=verifier, allow_legacy=allow_legacy, stt=stt, diarizer=diarizer,
        )
    )

    # Task B11: live-session re-analyze router — POST
    # /live-sessions/{id}/analyze. Uses `auth_dep` (not `strict_auth_dep`):
    # matches every other live-session read/write route in rest.py above,
    # all of which stay on the plain auth dependency per the I2/I3
    # controller ruling (only captures/groups/account-lookup require full
    # auth) — see watch/auth.py's require_full_auth docstring.
    app.include_router(
        make_live_sessions_router(resolved_store, auth_dep, transcriber, llm, diarizer=diarizer)
    )

    return app
