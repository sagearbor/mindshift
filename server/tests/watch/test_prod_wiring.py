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


def _collect_route_paths(routes):
    """Yield every route path reachable from `routes`.

    Newer FastAPI/Starlette (>=0.110) can list mounted sub-routers as
    `_IncludedRouter` entries that have no `.path` of their own but expose
    the underlying routes via `.routes`. Recurse into those so paths from
    included routers (e.g. the watch domain) are still genuinely asserted
    on both the old flat-route layout and the newer wrapped layout.
    """
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            yield path
        sub_routes = getattr(route, "routes", None)
        if sub_routes:
            yield from _collect_route_paths(sub_routes)


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

    def _leaf_routes(routes):
        """Yield only routes that carry their own path (skip wrapper
        entries like `_IncludedRouter` that have no path of their own —
        recurse into their `.routes` instead so the real leaf routes are
        still checked for collisions)."""
        for route in routes:
            path = getattr(route, "path", None)
            if path is not None:
                yield route
            sub_routes = getattr(route, "routes", None)
            if sub_routes:
                yield from _leaf_routes(sub_routes)

    dupes = {
        p: c
        for p, c in Counter(
            (r.path, ",".join(sorted(getattr(r, "methods", []) or ["WS"])))
            for r in _leaf_routes(main.app.routes)
        ).items()
        if c > 1
    }
    assert not dupes, f"colliding routes: {dupes}"
