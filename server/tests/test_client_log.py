"""POST /client-log — phones self-report client-side failures (upload errors
that never reach the server) so they're diagnosable from Cloud Run logs
instead of user screenshots. Auth-gated, size-capped, never fails the app."""

import pytest
from httpx import ASGITransport, AsyncClient

import main
from auth import get_current_uid
from main import app, init_db


@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.anyio
async def test_client_log_accepts_and_returns_204(client, caplog):
    with caplog.at_level("WARNING"):
        resp = await client.post(
            "/client-log",
            json={"kind": "upload-failure", "detail": {"status": 0, "causeName": "TypeError"}},
        )
    assert resp.status_code == 204
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "upload-failure" in joined
    assert "TypeError" in joined


@pytest.mark.anyio
async def test_client_log_truncates_huge_payloads(client, caplog):
    with caplog.at_level("WARNING"):
        resp = await client.post(
            "/client-log",
            json={"kind": "upload-failure", "detail": {"blob": "x" * 50_000}},
        )
    assert resp.status_code == 204
    for r in caplog.records:
        assert len(r.getMessage()) < 5_000


@pytest.mark.anyio
async def test_client_log_requires_auth(client, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post("/client-log", json={"kind": "x", "detail": {}})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_client_log_rejects_log_injection_in_kind(client):
    resp = await client.post(
        "/client-log",
        json={"kind": "bad\nFAKE-LOG-LINE", "detail": {}},
    )
    assert resp.status_code == 422
