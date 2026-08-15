# NEW test (not ported from gauge) — see task-B12-brief.md.
"""Production wiring: proves the watch domain is actually mounted on the
SAME FastAPI app MindShift's own main.py serves (`main.app`), not just on
`watch.testing.create_watch_test_app`'s throwaway assembly.

Both tests import the real `main` module with NO env vars set beyond
whatever conftest.py already arranges — keyless dev/CI must still be able
to build the full, both-domains app (honest-degradation doctrine: missing
creds/deps degrade individual endpoints, never block app startup).
"""

import main


def test_watch_routes_mounted_on_main_app():
    paths = {r.path for r in main.app.routes}
    assert "/me/pair/claim" in paths and "/telemetry" in paths and "/health" in paths
    assert "/live-sessions" in paths and "/analyze" in paths  # both domains coexist


def test_no_route_collisions():
    from collections import Counter

    dupes = {
        p: c
        for p, c in Counter(
            (r.path, ",".join(sorted(getattr(r, "methods", []) or ["WS"])))
            for r in main.app.routes
        ).items()
        if c > 1
    }
    assert not dupes, f"colliding routes: {dupes}"
