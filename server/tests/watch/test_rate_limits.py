# NEW test (Task H1, not ported from gauge) — see task-H1-brief.md.
"""Rate-limit + body-cap hardening for the ported unauthenticated watch
surface (Phase 2 finding: these routes now sit on the internet-facing app
without main.py's per-IP rate limiter — see docs/plans/2026-08-15-phase2-
one-backend-in-production.md's Task H1).

Covers three things, all via REAL dependency injection (a fake rate-limit
dependency that trips after N calls, mounted exactly the way
``build_watch_routers()`` will mount the real one) rather than mocking the
thing under test:

1. ``POST``/``GET /telemetry`` and ``POST /me/pair/start`` /
   ``GET /me/pair/status`` 429 once the injected limiter trips.
2. The factories' default (no ``rate_limit_dep`` passed, exactly
   ``create_watch_test_app``'s call shape) is a no-op — unauthenticated
   access to all four routes is unchanged, never 429s.
3. ``POST /enroll`` 413s over a 5 MB body cap (real oversized body, not a
   mock), and a body AT the cap is never rejected for size.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from watch.auth import make_auth_dependency, require_full_auth
from watch.pairing_store import MemoryPairingStore
from watch.routers.pairing import make_pairing_router
from watch.routers.rest import MAX_ENROLL_BYTES
from watch.store import MemoryLiveSessionStore
from watch.routers.telemetry import make_telemetry_router
from watch.telemetry_store import MemoryTelemetryStore
from watch.testing import create_watch_test_app


def _tripping_limiter(allow: int):
    """A fake rate-limit dependency: allows `allow` calls, 429s every call
    after that. Real dependency injection (mirrors main.py's `_rate_limit`
    dependency shape: `async def(request: Request) -> None`), not a mock of
    the router under test."""
    state = {"count": 0}

    async def _dep(request: Request) -> None:
        state["count"] += 1
        if state["count"] > allow:
            raise HTTPException(status_code=429, detail="rate limit exceeded — test double")

    return _dep


# --------------------------------------------------------------------- telemetry --

def _telemetry_client(rate_limit_dep=None):
    store = MemoryTelemetryStore()
    app = FastAPI()
    kwargs = {} if rate_limit_dep is None else {"rate_limit_dep": rate_limit_dep}
    app.include_router(make_telemetry_router(store, **kwargs))
    return TestClient(app)


def _telemetry_post_body():
    return {"device": "watch-abc", "app_version": "0.1.1", "events": []}


def test_telemetry_post_429s_once_injected_limiter_trips():
    client = _telemetry_client(rate_limit_dep=_tripping_limiter(1))
    assert client.post("/telemetry", json=_telemetry_post_body()).status_code == 200
    resp = client.post("/telemetry", json=_telemetry_post_body())
    assert resp.status_code == 429


def test_telemetry_get_429s_once_injected_limiter_trips():
    client = _telemetry_client(rate_limit_dep=_tripping_limiter(1))
    assert client.get("/telemetry").status_code == 200
    assert client.get("/telemetry").status_code == 429


def test_telemetry_default_is_a_noop_and_stays_unauthenticated():
    # No rate_limit_dep passed — exactly create_watch_test_app's call shape.
    client = _telemetry_client()
    for _ in range(5):
        assert client.post("/telemetry", json=_telemetry_post_body()).status_code == 200
        assert client.get("/telemetry").status_code == 200


# --------------------------------------------------------------------- pairing --

def _pairing_client(rate_limit_dep=None):
    store = MemoryLiveSessionStore()
    pstore = MemoryPairingStore()
    auth_dep = make_auth_dependency(None, False, store)
    strict_auth_dep = require_full_auth(auth_dep)
    app = FastAPI()
    kwargs = {} if rate_limit_dep is None else {"rate_limit_dep": rate_limit_dep}
    app.include_router(make_pairing_router(pstore, strict_auth_dep, **kwargs))
    return TestClient(app)


def test_pair_start_429s_once_injected_limiter_trips():
    client = _pairing_client(rate_limit_dep=_tripping_limiter(1))
    assert client.post("/me/pair/start").status_code == 200
    assert client.post("/me/pair/start").status_code == 429


def test_pair_status_429s_once_injected_limiter_trips():
    client = _pairing_client(rate_limit_dep=_tripping_limiter(1))
    ok = client.get("/me/pair/status", params={"pairing_id": "nope"})
    assert ok.status_code == 200 and ok.json()["status"] == "expired"
    assert client.get("/me/pair/status", params={"pairing_id": "nope"}).status_code == 429


def test_pairing_default_is_a_noop_and_stays_unauthenticated():
    # No rate_limit_dep passed — exactly create_watch_test_app's call shape.
    client = _pairing_client()
    for _ in range(5):
        assert client.post("/me/pair/start").status_code == 200
        assert client.get("/me/pair/status", params={"pairing_id": "nope"}).status_code == 200


# --------------------------------------------------------------------- enroll cap --

def _rest_client():
    store = MemoryLiveSessionStore()
    return store, TestClient(create_watch_test_app(store=store, allow_legacy=True))


def test_enroll_over_cap_is_413():
    _, client = _rest_client()
    oversized = b"\x00" * (MAX_ENROLL_BYTES + 1)
    resp = client.post("/enroll", params={"account": "alice"}, content=oversized)
    assert resp.status_code == 413


def test_enroll_at_cap_is_not_rejected_for_size():
    # Exactly MAX_ENROLL_BYTES of silence still goes through normal handling
    # (422 for silence content, never 413 for size) — the cap is inclusive.
    _, client = _rest_client()
    body = b"\x00" * MAX_ENROLL_BYTES
    resp = client.post("/enroll", params={"account": "alice"}, content=body)
    assert resp.status_code != 413
