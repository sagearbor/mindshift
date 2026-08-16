# Ported from gauge@2157433 server/tests/test_auth_routes.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# FIX (B11 review round 1, IMPORTANT finding 1): the three WS tests below
# used to run against a hand-rolled `/ws/test` mirror in `_build_app` that
# re-implemented `resolve_ws_principal` -> `close(1008)` -> `accept()` by
# hand -- a copy of the sequence, not the real one. That made sense at Task
# B3 (no real WS router existed yet -- see the ADAPTED note below, kept for
# history), but now that `watch/routers/ws.py`'s real
# `/ws/live-session/{id}` route exists (Task B11), testing a hand-rolled
# copy risks the copy silently drifting from ws.py's actual pre-accept
# ladder while still reporting green. These three tests now drive the REAL
# route via `watch.testing.create_watch_test_app`, and the mirror route (and
# its now-unused `WebSocket`/`Query`/`HTTPException`/`resolve_ws_principal`
# imports) is deleted. `/me` and `/protected` below are UNCHANGED -- they
# were never the finding's concern, and B3's original rationale for the
# hand-rolled non-WS routes still holds (make_auth_dependency/
# require_full_auth's own generic FastAPI-integration contract, independent
# of any one real router).
"""Integration tests for watch.auth's FastAPI wiring (make_auth_dependency,
require_full_auth, resolve_ws_principal) against a minimal throwaway app
for the HTTP routes, and the REAL `/ws/live-session/{id}` route (Task B11)
for the WS ones.

ADAPTED (Task B3): Gauge's original test_auth_routes.py exercised this wiring
through its real `/episodes` and `/me` routes (server/main.py's create_app).
Those routers are later tasks here (B5-B11's rest.py/ws.py) and don't exist
yet -- Task B3 only ports the auth layer they will depend on. This file
builds the smallest possible FastAPI app inline instead -- a legacy-or-bearer
route and a full-auth-only route -- wired to the exact same `watch.auth`
dependencies the real routers use (`make_auth_dependency`,
`require_full_auth`). That proves the FastAPI integration end-to-end
(header/query parsing, JIT account provisioning via `ensure_account`, the
401/503 status codes) without depending on any router task landing first.
Gauge's own ownership-check tests (`test_bearer_cannot_read_another_accounts_episode`,
the `/episodes/{id}/analyze` tests) are 403/ownership logic that belongs to
the real routers, not to this module's contract -- they are NOT ported here
and were B5/B11's job to cover once those routers existed.
"""
import asyncio
import json

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from watch.auth import Principal, make_auth_dependency, require_full_auth
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app
from server.tests.watch.test_auth import StubVerifier
from server.tests.watch.test_vectors import pcm

AUTH = {"Authorization": "Bearer good-token"}


def _build_app(store, verifier, allow_legacy):
    app = FastAPI()
    auth = make_auth_dependency(verifier, allow_legacy, store)
    full_auth = require_full_auth(auth)

    @app.get("/me")
    async def me(principal: Principal = Depends(auth)):
        return principal.model_dump()

    @app.get("/protected")
    async def protected(principal: Principal = Depends(full_auth)):
        return {"account_id": principal.account_id}

    return app


def _client(allow_legacy=True, verifier=StubVerifier):
    store = MemoryLiveSessionStore()
    v = verifier() if callable(verifier) else verifier
    return store, TestClient(_build_app(store, v, allow_legacy))


def _ws_client(allow_legacy=True, verifier=StubVerifier):
    # FIX (B11 review round 1, finding 1): the real app assembly, mounting
    # the REAL `/ws/live-session/{id}` route (watch/routers/ws.py) -- not
    # the hand-rolled `/ws/test` mirror `_client`/`_build_app` used to serve.
    store = MemoryLiveSessionStore()
    v = verifier() if callable(verifier) else verifier
    return store, TestClient(create_watch_test_app(store=store, verifier=v, allow_legacy=allow_legacy))


def test_legacy_account_param_still_works():
    # THE LIVE-WATCH GUARD: the shipped app sends ?account=default and no header.
    _, client = _client()
    resp = client.get("/me", params={"account": "default"})
    assert resp.status_code == 200
    assert resp.json() == {"account_id": "default", "email": None, "legacy": True}


def test_bearer_token_identifies_account():
    _, client = _client()
    resp = client.get("/me", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["account_id"] == "uid-123"


def test_bearer_token_provisions_account_row():
    import asyncio

    store, client = _client()
    assert client.get("/me", headers=AUTH).status_code == 200
    acct = asyncio.run(store.get_account("uid-123"))
    assert acct is not None and acct.email == "a@example.com" and acct.created_at


def test_legacy_principal_never_provisions_an_account_row():
    import asyncio

    store, client = _client()
    assert client.get("/me", params={"account": "default"}).status_code == 200
    assert asyncio.run(store.get_account("default")) is None


def test_me_reports_legacy_flag():
    _, client = _client()
    assert client.get("/me", params={"account": "default"}).json() == {
        "account_id": "default", "email": None, "legacy": True}
    assert client.get("/me", headers=AUTH).json() == {
        "account_id": "uid-123", "email": "a@example.com", "legacy": False}


def test_bad_token_is_401_even_with_account_param():
    _, client = _client()
    resp = client.get("/me", params={"account": "default"},
                       headers={"Authorization": "Bearer forged"})
    assert resp.status_code == 401


def test_legacy_rejected_when_flag_off_but_bearer_works():
    _, client = _client(allow_legacy=False)
    assert client.get("/me", params={"account": "default"}).status_code == 401
    assert client.get("/me", headers=AUTH).status_code == 200


def test_full_auth_rejects_legacy_principal():
    _, client = _client()
    resp = client.get("/protected", params={"account": "default"})
    assert resp.status_code == 401


def test_full_auth_accepts_bearer_principal():
    _, client = _client()
    resp = client.get("/protected", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"account_id": "uid-123"}


def test_ws_legacy_account_still_connects():
    # THE LIVE-WATCH GUARD: the shipped app sends ?account=default and no
    # header. Matches gauge's original test_ws_legacy_account_still_connects
    # assertion shape (drive a real session through the real WS route, then
    # check the persisted owner_account) — not a hand-rolled echo frame.
    store, client = _ws_client()
    with client.websocket_connect("/ws/live-session/w1?account=default") as ws:
        ws.send_bytes(pcm(0.02))
        ws.send_text(json.dumps({"type": "end"}))
        while json.loads(ws.receive_text())["type"] != "live_session_saved":
            pass
    assert asyncio.run(store.get_live_session("w1")).owner_account == "default"


def test_ws_token_query_param_owns_live_session_as_uid():
    # Matches gauge's original test_ws_token_query_param_owns_episode_as_uid.
    store, client = _ws_client()
    with client.websocket_connect("/ws/live-session/w2?token=good-token") as ws:
        ws.send_bytes(pcm(0.02))
        ws.send_text(json.dumps({"type": "end"}))
        while json.loads(ws.receive_text())["type"] != "live_session_saved":
            pass
    assert asyncio.run(store.get_live_session("w2")).owner_account == "uid-123"


def test_ws_bad_token_is_closed_not_accepted():
    _, client = _ws_client()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/live-session/w3?token=forged") as ws:
            ws.receive_text()
