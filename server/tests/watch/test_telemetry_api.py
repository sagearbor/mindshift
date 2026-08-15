# Ported from gauge@2157433 server/tests/test_telemetry_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B9): server.main.create_app(store=None, transcriber=None,
# llm=None, telemetry=ts) -> watch.testing.create_watch_test_app(telemetry_store=ts)
# (keyword-only assembly, no env vars -- see its own docstring); server.
# telemetry_store.MemoryTelemetryStore -> watch.telemetry_store.
# MemoryTelemetryStore; server.telemetry_api -> watch.routers.telemetry. Both
# routes are deliberately unauthenticated in gauge and stay that way here --
# no auth dep is threaded through `_client`, matching the brief.
import asyncio

from fastapi.testclient import TestClient

from watch.telemetry_store import MemoryTelemetryStore
from watch.testing import create_watch_test_app


def _client():
    ts = MemoryTelemetryStore()
    app = create_watch_test_app(telemetry_store=ts)
    return ts, TestClient(app)


def _post_body(n=1, device="watch-abc", level="crash"):
    return {"device": device, "app_version": "0.1.1",
            "events": [{"level": level, "tag": "SentinelService",
                        "message": f"boom {i}", "stack": "java.lang.SecurityException: ...",
                        "ts": "2026-08-02T01:00:00Z"} for i in range(n)]}


def test_post_stores_events_with_server_fields():
    ts, client = _client()
    resp = client.post("/telemetry", json=_post_body(2))
    assert resp.status_code == 200
    assert resp.json() == {"stored": 2, "dropped": 0}
    events = asyncio.run(ts.list_events(None, None, 10))
    assert len(events) == 2
    assert all(e.device == "watch-abc" and e.id and e.received_at for e in events)


def test_post_batch_over_cap_drops_tail():
    ts, client = _client()
    resp = client.post("/telemetry", json=_post_body(105))
    assert resp.status_code == 200
    assert resp.json() == {"stored": 100, "dropped": 5}


def test_post_malformed_is_422():
    _, client = _client()
    assert client.post("/telemetry", json={"nope": 1}).status_code == 422


def test_get_filters_by_device_and_limit():
    ts, client = _client()
    client.post("/telemetry", json=_post_body(3, device="d1"))
    client.post("/telemetry", json=_post_body(2, device="d2"))
    resp = client.get("/telemetry", params={"device": "d1", "limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert all(e["device"] == "d1" for e in body)
    assert "message" in body[0] and "received_at" in body[0]


def test_get_all_devices_when_no_filter():
    ts, client = _client()
    client.post("/telemetry", json=_post_body(1, device="d1"))
    client.post("/telemetry", json=_post_body(1, device="d2"))
    assert len(client.get("/telemetry").json()) == 2
