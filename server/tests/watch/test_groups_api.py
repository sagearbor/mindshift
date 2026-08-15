# Ported from gauge@2157433 server/tests/test_groups_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B6): server.main.create_app -> watch.testing.create_watch_test_app
# (keyword-only assembly); server.store.MemoryEpisodeStore -> watch.store.
# MemoryLiveSessionStore; GAUGE_ALLOW_LEGACY_ACCOUNT env var -> the explicit
# allow_legacy=True kwarg (testing.py takes no env vars at all -- see its own
# docstring, same B5 precedent as test_rest_api.py/test_claim_legacy.py). The
# source's `test_groups_require_auth` sanity-checked GAUGE_ALLOW_LEGACY_ACCOUNT
# via os.environ; that check has no equivalent here (no env var to read) and is
# dropped -- the 401 assertion itself is kept unchanged.
from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

# I2/I3 controller ruling: the groups router requires a NON-LEGACY (real,
# verified) principal on every route -- see watch/auth.py's require_full_auth
# -- so these tests authenticate via fake bearer tokens instead of the legacy
# `?account=` query param. Subs are kept as the plain names
# ("alice"/"bob"/"carol") so existing account-id assertions below don't need
# to change.
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


def _create_pair(client, headers=A, name="Us"):
    return client.post("/groups", headers=headers, json={"kind": "pair", "name": name}).json()


def test_create_pair_makes_creator_a_consenting_member():
    _, client = _client()
    g = _create_pair(client)
    assert g["kind"] == "pair" and g["name"] == "Us" and g["created_by"] == "alice"
    assert [m["account_id"] for m in g["members"]] == ["alice"]
    consent = g["consents"][0]
    assert consent["kind"] == "mutual_visibility"
    assert consent["participant_id"] == "alice" and consent["confirmed"] is True


def test_invite_then_join_forms_the_pair():
    _, client = _client()
    g = _create_pair(client)
    invited = client.post(f"/groups/{g['id']}/invite", headers=A, json={"email": "bob@example.com"}).json()
    code = invited["invites"][0]["code"]
    assert len(code) == 8 and invited["invites"][0]["email"] == "bob@example.com"

    joined = client.post("/groups/join", headers=B, json={"code": code})
    assert joined.status_code == 200
    body = joined.json()
    assert sorted(m["account_id"] for m in body["members"]) == ["alice", "bob"]
    assert {c["participant_id"] for c in body["consents"]} == {"alice", "bob"}
    assert body["invites"][0]["accepted_by"] == "bob" and body["invites"][0]["accepted_at"]


def test_unknown_invite_code_404():
    _, client = _client()
    assert client.post("/groups/join", headers=B, json={"code": "deadbeef"}).status_code == 404


def test_invite_cannot_be_reused():
    _, client = _client()
    g = _create_pair(client)
    code = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][0]["code"]
    assert client.post("/groups/join", headers=B, json={"code": code}).status_code == 200
    assert client.post("/groups/join", headers=C, json={"code": code}).status_code == 409


def test_pair_cannot_take_a_third_member():
    _, client = _client()
    g = _create_pair(client)
    code1 = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][0]["code"]
    client.post("/groups/join", headers=B, json={"code": code1})
    assert client.post(f"/groups/{g['id']}/invite", headers=A, json={}).status_code == 409


def test_team_takes_a_third_member():
    _, client = _client()
    g = client.post("/groups", headers=A, json={"kind": "team", "name": "Standup"}).json()
    for who in (B, C):
        code = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][-1]["code"]
        assert client.post("/groups/join", headers=who, json={"code": code}).status_code == 200
    final = client.get("/groups", headers=A).json()[0]
    assert len(final["members"]) == 3


def test_joining_twice_is_409():
    _, client = _client()
    g = _create_pair(client)
    code = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][0]["code"]
    client.post("/groups/join", headers=B, json={"code": code})
    assert client.post("/groups/join", headers=B, json={"code": code}).status_code == 409


def test_non_member_cannot_invite():
    _, client = _client()
    g = _create_pair(client)
    assert client.post(f"/groups/{g['id']}/invite", headers=C, json={}).status_code == 403


def test_list_groups_is_membership_scoped():
    _, client = _client()
    _create_pair(client)
    assert len(client.get("/groups", headers=A).json()) == 1
    assert client.get("/groups", headers=B).json() == []


def test_leave_removes_membership_but_keeps_the_consent_audit_trail():
    _, client = _client()
    g = _create_pair(client)
    code = client.post(f"/groups/{g['id']}/invite", headers=A, json={}).json()["invites"][0]["code"]
    client.post("/groups/join", headers=B, json={"code": code})

    left = client.delete(f"/groups/{g['id']}/me", headers=B)
    assert left.status_code == 200
    body = left.json()
    assert [m["account_id"] for m in body["members"]] == ["alice"]
    assert {c["participant_id"] for c in body["consents"]} == {"alice", "bob"}   # history retained
    assert client.get("/groups", headers=B).json() == []


def test_leaving_a_group_you_are_not_in_is_404():
    _, client = _client()
    g = _create_pair(client)
    assert client.delete(f"/groups/{g['id']}/me", headers=C).status_code == 404


def test_groups_require_auth():
    _, client = _client()
    assert client.get("/groups").status_code == 401      # no header, no ?account=


def test_groups_reject_legacy_account_param_even_though_flag_is_on():
    """I2/I3 pinning test: before the fix, GAUGE_ALLOW_LEGACY_ACCOUNT=true let
    ANY caller reach the whole groups router by sending an unauthenticated
    `?account=<anyone>` -- no token, no proof of identity. The groups router
    now requires a full (non-legacy) principal on every route, so the legacy
    query param alone must be a 401, not a 200."""
    _, client = _client()
    resp = client.get("/groups", params={"account": "alice"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "this endpoint requires sign-in"
