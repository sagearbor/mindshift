# Emotion dev tester bridge (2026-11-07): phone -> watch push, watch -> phone control queue.
import json

import pytest
from fastapi.testclient import TestClient

from watch import emotion_bridge, relay
from watch.emotion_bridge import CtlQueue, parse_ctl
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app


@pytest.fixture(autouse=True)
def _clean():
    relay._registry.clear()
    emotion_bridge.reset_for_tests()
    yield
    relay._registry.clear()
    emotion_bridge.reset_for_tests()


def _client():
    return TestClient(create_watch_test_app(store=MemoryLiveSessionStore(), allow_legacy=True))


EVENT = {"v": 1, "id": "w1", "t": 1700000000000, "label": "anger", "prob": 0.71, "nudge": True, "paused": False, "suggestion": None}


def test_event_reaches_the_live_watch_socket_and_reports_delivered():
    client = _client()
    with client.websocket_connect("/ws/live-session/companion-1?account=alice") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        assert json.loads(ws.receive_text()) == {"type": "companion_ack"}
        r = client.post("/emotion/event?account=alice", json=EVENT)
        assert r.status_code == 200 and r.json() == {"delivered": True}
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "emotion"
        assert frame["label"] == "anger" and frame["nudge"] is True and frame["prob"] == 0.71


def test_event_without_a_live_watch_is_honestly_not_delivered():
    r = _client().post("/emotion/event?account=alice", json=EVENT)
    assert r.status_code == 200 and r.json() == {"delivered": False}


def test_event_for_another_account_never_reaches_this_watch():
    client = _client()
    with client.websocket_connect("/ws/live-session/companion-1?account=alice") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        ws.receive_text()
        assert client.post("/emotion/event?account=bob", json=EVENT).json() == {"delivered": False}


def test_event_validation_rejects_garbage():
    client = _client()
    assert client.post("/emotion/event?account=alice", json={**EVENT, "prob": 7}).status_code == 422
    assert client.post("/emotion/event?account=alice", json={"label": "anger"}).status_code == 422


def test_watch_controls_are_queued_for_the_phone_in_order_and_drained_by_seq():
    client = _client()
    with client.websocket_connect("/ws/live-session/companion-2?account=alice") as ws:
        ws.send_text(json.dumps({"type": "companion"}))
        ws.receive_text()
        ws.send_text(json.dumps({"type": "emotion_ctl", "action": "pause"}))
        ws.send_text(json.dumps({"type": "emotion_ctl", "action": "flag", "flag": "bad", "nudge_id": "w1"}))
        ws.send_text(json.dumps({"type": "emotion_ctl", "action": "flag", "flag": "ignore"}))
        # a barrier: the error reply proves the socket processed the frames above
        ws.send_text(json.dumps({"type": "bogus"}))
        assert json.loads(ws.receive_text()) == {"type": "error", "detail": "unknown_type"}

    items = client.get("/emotion/ctl?account=alice&after=0").json()["items"]
    assert [i["action"] for i in items] == ["pause", "flag", "flag"]
    assert items[1]["flag"] == "bad" and items[1]["nudge_id"] == "w1"
    assert [i["seq"] for i in items] == [1, 2, 3]
    later = client.get("/emotion/ctl?account=alice&after=2").json()["items"]
    assert [i["seq"] for i in later] == [3]
    # another account sees nothing
    assert client.get("/emotion/ctl?account=bob&after=0").json()["items"] == []


def test_malformed_control_frame_gets_the_standard_error_and_queues_nothing():
    client = _client()
    with client.websocket_connect("/ws/live-session/companion-3?account=carol") as ws:
        ws.send_text(json.dumps({"type": "emotion_ctl", "action": "flag", "flag": "banana"}))
        assert json.loads(ws.receive_text()) == {"type": "error", "detail": "malformed_json"}
    assert client.get("/emotion/ctl?account=carol&after=0").json()["items"] == []


def test_parse_ctl_contract():
    assert parse_ctl({"action": "resume"}) == {"action": "resume"}
    assert parse_ctl({"action": "flag", "flag": "good"}) == {"action": "flag", "flag": "good"}
    assert parse_ctl({"action": "flag"}) is None
    assert parse_ctl({"action": "reboot"}) is None


def test_queue_is_bounded_and_entries_expire():
    now = [1000.0]
    q = CtlQueue(clock=lambda: now[0])
    for _ in range(emotion_bridge.MAX_QUEUE + 20):
        q.push("a", {"action": "pause"})
    assert len(q.since("a", 0)) == emotion_bridge.MAX_QUEUE
    now[0] += emotion_bridge.QUEUE_TTL_S + 1
    assert q.since("a", 0) == []


def test_unauthenticated_caller_is_refused():
    client = TestClient(create_watch_test_app(store=MemoryLiveSessionStore(), allow_legacy=False))
    assert client.post("/emotion/event", json=EVENT).status_code in (401, 403)
    assert client.get("/emotion/ctl").status_code in (401, 403)


WIN = {"t": 1700000000000, "p": [0.7, 0.1, 0.05, 0.05, 0.05, 0.05], "ms": 41, "b": 80, "n": True}


def test_on_watch_windows_reach_the_phone_queue_numbers_only():
    client = _client()
    r = client.post("/emotion/watch-windows?account=alice", json={"v": 1, "rows": [WIN, {**WIN, "t": 1700000003000, "n": False, "why": "neutral"}]})
    assert r.status_code == 200 and r.json() == {"accepted": 2}
    items = client.get("/emotion/ctl?account=alice&after=0").json()["items"]
    assert len(items) == 1 and items[0]["action"] == "windows"
    assert [row["t"] for row in items[0]["rows"]] == [1700000000000, 1700000003000]
    assert set(items[0]["rows"][0]) == {"t", "p", "ms", "b", "n", "why"}  # no audio field exists
    # another account sees nothing
    assert client.get("/emotion/ctl?account=bob&after=0").json()["items"] == []


def test_on_watch_windows_reject_non_probability_rows_and_oversize_batches():
    client = _client()
    bad = {**WIN, "p": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]}
    assert client.post("/emotion/watch-windows?account=alice", json={"rows": [bad]}).json() == {"accepted": 0}
    assert client.post("/emotion/watch-windows?account=alice", json={"rows": [{**WIN, "p": [1.0]}]}).status_code == 422
    too_many = [WIN] * (emotion_bridge.MAX_WATCH_ROWS + 1)
    assert client.post("/emotion/watch-windows?account=alice", json={"rows": too_many}).status_code == 422
    assert client.get("/emotion/ctl?account=alice&after=0").json()["items"] == []


def test_windows_cannot_be_injected_over_the_websocket_ctl_path():
    assert emotion_bridge.push_ctl("alice", {"action": "windows", "rows": [WIN]}) is False
    assert parse_ctl({"action": "windows"}) == {"action": "windows"}  # parse alone is shape-only; push_ctl blocks it


def test_watch_windows_requires_auth():
    client = TestClient(create_watch_test_app(store=MemoryLiveSessionStore(), allow_legacy=False))
    assert client.post("/emotion/watch-windows", json={"rows": [WIN]}).status_code in (401, 403)
