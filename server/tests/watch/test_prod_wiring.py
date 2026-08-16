# NEW test (not ported from gauge) — see task-B12-brief.md.
"""Production wiring: proves the watch domain is actually mounted on the
SAME FastAPI app MindShift's own main.py serves (`main.app`), not just on
`watch.testing.create_watch_test_app`'s throwaway assembly.

Both tests import the real `main` module with NO env vars set beyond
whatever conftest.py already arranges — keyless dev/CI must still be able
to build the full, both-domains app (honest-degradation doctrine: missing
creds/deps degrade individual endpoints, never block app startup).
"""

import logging

import main


def _iter_leaf_routes(routes, _seen=None):
    """Yield every leaf route object (one that carries its own `.path`)
    reachable from `routes`, on ANY FastAPI/Starlette layout.

    - Old FastAPI (<0.110, confirmed locally: fastapi 0.133.0 /
      starlette 0.52.1 as of this writing): `app.include_router()` just
      appends the included router's routes directly onto `app.routes` —
      every entry already has `.path`.
    - Newer FastAPI (>=0.110, confirmed against CI's pin: fastapi 0.141.1 /
      starlette 1.6.0): `app.include_router()` instead appends a
      `fastapi.routing._IncludedRouter` wrapper with `.path = None` and NO
      `.routes` attribute at all — the real routes live on
      `wrapper.original_router.routes` (verified via
      `vars(wrapper)` against the CI-pinned version). A plain `.routes`
      fallback (e.g. for `Mount`/`APIRouter` objects that DO expose
      `.routes` directly) is also handled for robustness.

    `_seen` guards against revisiting the same router object twice (e.g. if
    both `.routes` and `.original_router.routes` happened to resolve to the
    same underlying list on some version).
    """
    if _seen is None:
        _seen = set()
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield route

        sub_route_lists = []
        sub_routes = getattr(route, "routes", None)
        if sub_routes:
            sub_route_lists.append(sub_routes)
        original_router = getattr(route, "original_router", None)
        if original_router is not None:
            orig_routes = getattr(original_router, "routes", None)
            if orig_routes:
                sub_route_lists.append(orig_routes)

        for sub in sub_route_lists:
            key = id(sub)
            if key in _seen:
                continue
            _seen.add(key)
            yield from _iter_leaf_routes(sub, _seen)


def _collect_route_paths(routes):
    """Yield the `.path` of every leaf route reachable from `routes`."""
    for route in _iter_leaf_routes(routes):
        yield route.path


def test_prod_store_warnings_fire_when_env_unset(caplog):
    """I2 (final whole-branch review 2026-08-15): building the watch routers
    with MINDSHIFT_FIRESTORE_PROJECT / MINDSHIFT_CAPTURE_BUCKET unset (as in
    CI) must log a loud warning for each in-memory/None fallback, so a
    misconfigured prod deploy is caught rather than silently losing data.
    main.py already calls build_watch_routers() once at import time, so this
    calls it again directly under caplog to observe the warnings.
    """
    from watch.app import build_watch_routers

    with caplog.at_level(logging.WARNING, logger="watch.app"):
        build_watch_routers()

    messages = "\n".join(r.message for r in caplog.records)
    assert "in-memory stores" in messages
    assert "MINDSHIFT_FIRESTORE_PROJECT" in messages
    assert "capture blob store" in messages
    assert "MINDSHIFT_CAPTURE_BUCKET" in messages


def test_watch_routes_mounted_on_main_app():
    paths = set(_collect_route_paths(main.app.routes))
    assert "/me/pair/claim" in paths and "/telemetry" in paths and "/health" in paths
    assert "/live-sessions" in paths and "/analyze" in paths  # both domains coexist


def test_no_route_collisions():
    from collections import Counter

    dupes = {
        p: c
        for p, c in Counter(
            (r.path, ",".join(sorted(getattr(r, "methods", []) or ["WS"])))
            for r in _iter_leaf_routes(main.app.routes)
        ).items()
        if c > 1
    }
    assert not dupes, f"colliding routes: {dupes}"
