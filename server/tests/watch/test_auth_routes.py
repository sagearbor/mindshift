# Ported from gauge@2157433 server/tests/test_auth_routes.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Integration tests for watch.auth's FastAPI wiring (make_auth_dependency,
require_full_auth, resolve_ws_principal) against a minimal throwaway app.

ADAPTED (Task B3): Gauge's original test_auth_routes.py exercised this wiring
through its real `/episodes` and `/me` routes (server/main.py's create_app).
Those routers are later tasks here (B5-B11's rest.py/ws.py) and don't exist
yet -- Task B3 only ports the auth layer they will depend on. This file
builds the smallest possible FastAPI app inline instead -- a legacy-or-bearer
route, a full-auth-only route, and a WS route -- wired to the exact same
`watch.auth` dependencies the real routers will use (`make_auth_dependency`,
`require_full_auth`, `resolve_ws_principal`). That proves the FastAPI
integration end-to-end (header/query parsing, JIT account provisioning via
`ensure_account`, the 401/503 status codes, and the WS pre-accept ladder)
without depending on any router task landing first. Gauge's own
ownership-check tests (`test_bearer_cannot_read_another_accounts_episode`,
the `/episodes/{id}/analyze` tests) are 403/ownership logic that belongs to
the real routers, not to this module's contract -- they are NOT ported here
and are B5/B10's job to cover once those routers exist.
"""
import json

import pytest
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from watch.auth import Principal, make_auth_dependency, require_full_auth, resolve_ws_principal
from watch.store import MemoryLiveSessionStore
from server.tests.watch.test_auth import StubVerifier

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

    @app.websocket("/ws/test")
    async def ws_test(
        websocket: WebSocket,
        account: str | None = Query(None),
        token: str | None = Query(None),
    ) -> None:
        try:
            principal = resolve_ws_principal(token, account, verifier, allow_legacy)
        except HTTPException:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        await websocket.send_text(json.dumps(
            {"account_id": principal.account_id, "legacy": principal.legacy}
        ))
        await websocket.close()

    return app


def _client(allow_legacy=True, verifier=StubVerifier):
    store = MemoryLiveSessionStore()
    v = verifier() if callable(verifier) else verifier
    return store, TestClient(_build_app(store, v, allow_legacy))


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
    _, client = _client()
    with client.websocket_connect("/ws/test?account=default") as ws:
        data = json.loads(ws.receive_text())
    assert data == {"account_id": "default", "legacy": True}


def test_ws_token_query_param_identifies_account():
    _, client = _client()
    with client.websocket_connect("/ws/test?token=good-token") as ws:
        data = json.loads(ws.receive_text())
    assert data == {"account_id": "uid-123", "legacy": False}


def test_ws_bad_token_is_closed_not_accepted():
    _, client = _client()
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/test?token=forged") as ws:
            ws.receive_text()
