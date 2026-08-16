# Ported from gauge@2157433 server/tests/test_me_standing.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5, deferred from B4 -- this test is REST-router-bound, not
# aggregates-bound): Episode -> LiveSession; server.main.create_app ->
# watch.testing.create_watch_test_app; GAUGE_ALLOW_LEGACY_ACCOUNT env var ->
# the explicit allow_legacy=True kwarg. `current`/`prior`'s `episodes` field
# name is UNCHANGED per the locked rename map (PeriodStats.episodes was
# already ported as-is in Task B1/B4).
"""GET /me/standing — personal, non-group-scoped standing (server-track item 13c).

Same live-session fetch + ownership filter pattern as a future groups router's
per-member loop (server/watch/aggregates.py's member_standing, Task B4), just
for the caller's own account: no group, no membership, no consent gate — this
is self-data. Mirrors a future test_group_standing.py's conventions.
"""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.models import Account, LiveSession, NudgeEvent, Participant, VectorEvent
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

A = {"account": "alice"}


def _client():
    store = MemoryLiveSessionStore()
    return store, TestClient(create_watch_test_app(
        store=store, verifier=StubVerifier(), allow_legacy=True,
    ))


def _ls(id, owner, days_ago, level=0, shared=()):
    started = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return LiveSession(id=id, owner_account=owner, started_at=started.isoformat(), ended_at=None,
                        status="analyzed",
                        participants=[Participant(id="self", role="self", speaker_label="You")],
                        vector_events=([VectorEvent(vector="yelling", level=level, t=0.0, value=1.0,
                                                    participant_id="self")] if level else []),
                        nudge_events=[NudgeEvent(channel="A", level=1, t=0.0, vectors=["yelling"])],
                        shared_with=list(shared))


def test_standing_reports_caller_against_their_own_prior_period():
    store, client = _client()
    import asyncio
    for ls in (_ls("a-cur", "alice", 2, level=1), _ls("a-pri", "alice", 9, level=3)):
        asyncio.run(store.put_live_session(ls))

    body = client.get("/me/standing", params=A).json()
    assert body["account_id"] == "alice"
    assert body["current"]["calm"] == 75.0 and body["prior"]["calm"] == 25.0
    assert body["delta_vs_self"] == 50.0 and body["improving"] is True


def test_standing_requires_auth():
    _, client = _client()
    assert client.get("/me/standing").status_code == 401      # no header, no ?account=


def test_period_days_is_honored_and_bounded():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("a-old", "alice", 20, level=1)))
    assert client.get("/me/standing", params={**A, "period_days": 7}
                      ).json()["current"]["episodes"] == 0
    assert client.get("/me/standing", params={**A, "period_days": 30}
                      ).json()["current"]["episodes"] == 1
    assert client.get("/me/standing", params={**A, "period_days": 0}).status_code == 422
    assert client.get("/me/standing", params={**A, "period_days": 91}).status_code == 422


def test_standing_excludes_live_sessions_merely_shared_with_the_caller():
    # A shared live session is someone ELSE's behavior; it must never enter
    # the caller's own standing.
    store, client = _client()
    import asyncio
    asyncio.run(store.put_live_session(_ls("bobs", "bob", 1, level=3, shared=("alice",))))
    asyncio.run(store.put_live_session(_ls("a-cur", "alice", 1, level=0)))
    body = client.get("/me/standing", params=A).json()
    assert body["current"]["episodes"] == 1 and body["current"]["calm"] == 100.0


def test_standing_empty_period_reports_null_not_zero():
    _, client = _client()
    body = client.get("/me/standing", params=A).json()
    assert body["current"]["episodes"] == 0 and body["current"]["calm"] is None
    assert body["delta_vs_self"] is None and body["improving"] is None


def test_display_name_is_null_for_accounts_with_no_row():
    _, client = _client()
    body = client.get("/me/standing", params=A).json()
    assert body["display_name"] is None   # legacy principal, never faked


def test_display_name_reflects_the_account_row_when_present():
    store, client = _client()
    import asyncio
    asyncio.run(store.put_account(Account(id="alice", email="alice@example.com",
                                          display_name="Alice", created_at="2026-07-01T00:00:00Z",
                                          updated_at="2026-07-01T00:00:00Z")))
    body = client.get("/me/standing", params=A).json()
    assert body["display_name"] == "Alice"


def test_response_is_watch_sized_and_flat():
    _, client = _client()
    resp = client.get("/me/standing", params=A)
    assert len(resp.content) < 1024
    body = resp.json()
    assert set(body) == {"account_id", "display_name", "current", "prior",
                         "delta_vs_self", "improving"}
    assert set(body["current"]) == {"episodes", "calm", "nudges", "escalations"}
