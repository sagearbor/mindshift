# Ported from gauge@2157433 server/tests/test_group_standing.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B6, deferred from B4): Episode -> LiveSession (store methods,
# model, field names, HTTP path /episodes* -> /live-sessions*);
# server.main.create_app -> watch.testing.create_watch_test_app (keyword-only
# assembly); server.store.MemoryEpisodeStore -> watch.store.
# MemoryLiveSessionStore; GAUGE_ALLOW_LEGACY_ACCOUNT env var -> the explicit
# allow_legacy=True kwarg (testing.py takes no env vars at all). Wire field
# names (e.g. PeriodStats.episodes) are UNCHANGED per the locked rename map --
# only the "episode" TYPE name and its HTTP paths/store methods move.
import asyncio
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.models import LiveSession, NudgeEvent, Participant, VectorEvent
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

# I2/I3 controller ruling: /groups/{id}/standing is part of the groups
# router, which requires a NON-LEGACY (real, verified) principal on every
# route -- see watch/auth.py's require_full_auth -- so these tests
# authenticate via fake bearer tokens instead of the legacy `?account=` query
# param. Subs are kept as the plain names ("alice"/"bob"/"carol") so existing
# account-id assertions below don't need to change.
TOKENS = {
    "alice-token": {"sub": "alice", "email": "alice@example.com"},
    "bob-token": {"sub": "bob", "email": "bob@example.com"},
    "carol-token": {"sub": "carol", "email": "carol@example.com"},
}
A = {"Authorization": "Bearer alice-token"}
B = {"Authorization": "Bearer bob-token"}
C = {"Authorization": "Bearer carol-token"}


def _client():
    store = MemoryLiveSessionStore()
    return store, TestClient(create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), allow_legacy=True,
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


def _pair(client):
    g = client.post("/groups", headers=A, json={"kind": "pair", "name": "Us"}).json()
    code = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][0]["code"]
    client.post("/groups/join", headers=B, json={"code": code})
    return g["id"]


def test_standing_reports_each_member_against_their_own_prior_period():
    store, client = _client()
    gid = _pair(client)
    for ls in (_ls("a-cur", "alice", 2, level=1), _ls("a-pri", "alice", 9, level=3),
               _ls("b-cur", "bob", 1, level=2), _ls("b-pri", "bob", 10, level=3)):
        asyncio.run(store.put_live_session(ls))

    body = client.get(f"/groups/{gid}/standing", headers=A).json()
    assert body["group_id"] == gid and body["period_days"] == 7
    assert [m["account_id"] for m in body["members"]] == ["alice", "bob"]
    alice, bob = body["members"]
    assert alice["current"]["calm"] == 75.0 and alice["prior"]["calm"] == 25.0
    assert alice["delta_vs_self"] == 50.0 and alice["improving"] is True
    assert bob["current"]["calm"] == 50.0 and bob["improving"] is True
    assert body["both_improving"] is True
    assert body["ahead"] == "alice"


def test_standing_is_readable_by_either_member():
    store, client = _client()
    gid = _pair(client)
    assert client.get(f"/groups/{gid}/standing", headers=A).status_code == 200
    assert client.get(f"/groups/{gid}/standing", headers=B).status_code == 200


def test_standing_requires_membership():
    _, client = _client()
    gid = _pair(client)
    assert client.get(f"/groups/{gid}/standing", headers=C).status_code == 403


def test_standing_unknown_group_404():
    _, client = _client()
    assert client.get("/groups/nope/standing", headers=A).status_code == 404


def test_standing_409_until_the_partner_joins():
    _, client = _client()
    g = client.post("/groups", headers=A, json={"kind": "pair"}).json()
    resp = client.get(f"/groups/{g['id']}/standing", headers=A)
    assert resp.status_code == 409
    assert "consent" in resp.json()["detail"]


def test_standing_409_after_the_partner_leaves():
    _, client = _client()
    gid = _pair(client)
    client.delete(f"/groups/{gid}/me", headers=B)
    assert client.get(f"/groups/{gid}/standing", headers=A).status_code == 409


def test_standing_excludes_episodes_merely_shared_with_a_member():
    # A shared live session is someone ELSE's behavior; it must never enter
    # the recipient's own standing.
    store, client = _client()
    gid = _pair(client)
    asyncio.run(store.put_live_session(_ls("carols", "carol", 1, level=3, shared=("bob",))))
    asyncio.run(store.put_live_session(_ls("b-cur", "bob", 1, level=0)))
    bob = next(m for m in client.get(f"/groups/{gid}/standing", headers=B).json()["members"]
               if m["account_id"] == "bob")
    assert bob["current"]["episodes"] == 1 and bob["current"]["calm"] == 100.0


def test_standing_empty_period_reports_null_not_zero():
    _, client = _client()
    gid = _pair(client)
    body = client.get(f"/groups/{gid}/standing", headers=A).json()
    for m in body["members"]:
        assert m["current"]["episodes"] == 0 and m["current"]["calm"] is None
        assert m["delta_vs_self"] is None and m["improving"] is None
    assert body["both_improving"] is False and body["ahead"] is None


def test_period_days_is_honored_and_bounded():
    store, client = _client()
    gid = _pair(client)
    asyncio.run(store.put_live_session(_ls("a-old", "alice", 20, level=1)))
    assert client.get(f"/groups/{gid}/standing", headers=A, params={"period_days": 7}
                      ).json()["members"][0]["current"]["episodes"] == 0
    assert client.get(f"/groups/{gid}/standing", headers=A, params={"period_days": 30}
                      ).json()["members"][0]["current"]["episodes"] == 1
    assert client.get(f"/groups/{gid}/standing", headers=A, params={"period_days": 0}).status_code == 422
    assert client.get(f"/groups/{gid}/standing", headers=A, params={"period_days": 91}).status_code == 422


def test_display_name_is_null_for_accounts_with_no_row():
    _, client = _client()
    gid = _pair(client)
    body = client.get(f"/groups/{gid}/standing", headers=A).json()
    assert all(m["display_name"] is None for m in body["members"])   # never faked


def test_response_is_watch_sized_and_flat():
    # The tag-along glance polls this on a wrist: keep it small and stable.
    _, client = _client()
    gid = _pair(client)
    resp = client.get(f"/groups/{gid}/standing", headers=A)
    assert len(resp.content) < 2048
    body = resp.json()
    assert set(body) == {"group_id", "period_days", "period_start", "period_end",
                         "members", "both_improving", "ahead"}
    assert set(body["members"][0]) == {"account_id", "display_name", "current", "prior",
                                       "delta_vs_self", "improving"}
    assert set(body["members"][0]["current"]) == {"episodes", "calm", "nudges", "escalations"}


def test_deleted_episode_no_longer_appears_in_the_partners_standing():
    # T8 review minor 3: deletion honesty. group_standing_endpoint sources
    # each member's live sessions fresh from store.list_live_sessions on
    # every call (see watch/routers/groups.py) rather than from any
    # cached/derived aggregate, so DELETE /live-sessions/{id} -- which hard-
    # deletes from that same live store (rest.py's delete_live_session) --
    # should already make a deleted live session vanish from a partner's
    # standing view with no separate cleanup step. Previously only
    # established by code trace; this pins it as an observable, end-to-end
    # behavior via the actual REST surface a client would use.
    store, client = _client()
    gid = _pair(client)
    asyncio.run(store.put_live_session(_ls("a-cur", "alice", 1, level=3)))

    # Bob (the partner) sees alice's live session in the shared standing view.
    before = client.get(f"/groups/{gid}/standing", headers=B).json()
    alice_before = next(m for m in before["members"] if m["account_id"] == "alice")
    assert alice_before["current"]["episodes"] == 1
    assert alice_before["current"]["calm"] == 25.0

    # Alice deletes her own live session.
    resp = client.delete("/live-sessions/a-cur", headers=A)
    assert resp.status_code == 204

    # It's gone from Bob's view of alice's standing -- back to the honest
    # "empty period" shape (null calm, not a stale/zeroed leftover value).
    after = client.get(f"/groups/{gid}/standing", headers=B).json()
    alice_after = next(m for m in after["members"] if m["account_id"] == "alice")
    assert alice_after["current"]["episodes"] == 0
    assert alice_after["current"]["calm"] is None

    # And, symmetrically, from alice's own /me/standing (same underlying
    # live list_live_sessions source, no separate code path to miss).
    mine = client.get("/me/standing", headers=A).json()
    assert mine["current"]["episodes"] == 0


def test_standing_rejects_legacy_account_param_even_though_flag_is_on():
    """I2/I3 pinning test: /groups/{id}/standing is part of the groups
    router and must reject an unauthenticated `?account=` legacy principal
    just like every other route in that router."""
    _, client = _client()
    gid = _pair(client)
    resp = client.get(f"/groups/{gid}/standing", params={"account": "alice"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "this endpoint requires sign-in"
