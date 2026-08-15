# Ported from gauge@2157433 server/tests/test_telemetry_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import asyncio
from watch.models import TelemetryEvent
from watch.telemetry_store import MemoryTelemetryStore, clamp_event, MAX_MESSAGE_CHARS


def _ev(id, device="dev1", ts="2026-08-02T01:00:00+00:00", received="2026-08-02T01:00:01+00:00",
        level="error", message="boom"):
    return TelemetryEvent(id=id, device=device, app_version="0.1.1", level=level,
                           tag="test", message=message, stack=None, ts=ts, received_at=received)


def test_add_and_list_newest_first():
    s = MemoryTelemetryStore()
    asyncio.run(s.add_events([_ev("a", received="2026-08-02T01:00:01+00:00"),
                              _ev("b", received="2026-08-02T02:00:00+00:00")]))
    out = asyncio.run(s.list_events(device=None, since=None, limit=10))
    assert [e.id for e in out] == ["b", "a"]


def test_filters_by_device_and_since_and_limit():
    s = MemoryTelemetryStore()
    asyncio.run(s.add_events([
        _ev("a", device="d1", received="2026-08-02T01:00:00+00:00"),
        _ev("b", device="d2", received="2026-08-02T02:00:00+00:00"),
        _ev("c", device="d1", received="2026-08-02T03:00:00+00:00"),
        _ev("d", device="d1", received="2026-08-02T04:00:00+00:00"),
    ]))
    out = asyncio.run(s.list_events(device="d1", since="2026-08-02T02:30:00+00:00", limit=1))
    assert [e.id for e in out] == ["d"]


def test_clamp_truncates_long_message_and_stack():
    e = _ev("a", message="x" * (MAX_MESSAGE_CHARS + 500))
    c = clamp_event(e)
    assert len(c.message) <= MAX_MESSAGE_CHARS
    assert c.message.endswith("…[truncated]")


def test_memory_store_copies_out():
    s = MemoryTelemetryStore()
    asyncio.run(s.add_events([_ev("a")]))
    out = asyncio.run(s.list_events(None, None, 10))
    out[0].message = "mutated"
    again = asyncio.run(s.list_events(None, None, 10))
    assert again[0].message == "boom"
