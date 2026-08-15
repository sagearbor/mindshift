# Ported from gauge@2157433 server/tests/test_accounts_lookup.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5): server.main.create_app -> watch.testing.create_watch_test_app;
# EpisodeStore -> LiveSessionStore (Firestore/Memory).
"""GET /accounts/lookup?email= -- account-id lookup by email for the web
dashboard's share-by-email UX (server plan addendum item 2, controller-
sanctioned; reuses the already-planned `store.get_account_by_email`).

Auth-required, same `Depends(auth)` pattern as a future groups router. Accounts
only exist once a *verified* (non-legacy) principal has been seen at least
once -- `watch.auth.ensure_account` never provisions a row for a legacy
`?account=` principal -- so these tests provision accounts via a bearer-token
request (`GET /me`) the same way `test_auth_routes.py` does, rather than the
`?account=` shorthand a legacy-only test would use.
"""

import inspect

from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.store import FirestoreLiveSessionStore, MemoryLiveSessionStore
from watch.testing import create_watch_test_app

TOKENS = {
    "alice-token": {"sub": "uid-alice", "email": "alice@example.com"},
    # Deliberately mixed-case, to probe how the route treats casing -- see
    # test_lookup_is_case_sensitive_matching_how_emails_are_stored below.
    "bob-token": {"sub": "uid-bob", "email": "Bob@Example.com"},
}
ALICE = {"Authorization": "Bearer alice-token"}
BOB = {"Authorization": "Bearer bob-token"}


def _client():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(store=store, verifier=StubVerifier(TOKENS)))
    return store, client


def _provision(client, headers):
    # Any authed request runs auth.ensure_account's just-in-time
    # provisioning -- /me is the cheapest one.
    assert client.get("/me", headers=headers).status_code == 200


def test_lookup_by_email_200_on_match():
    _, client = _client()
    _provision(client, ALICE)
    resp = client.get("/accounts/lookup", params={"email": "alice@example.com"}, headers=BOB)
    assert resp.status_code == 200
    assert resp.json() == {"account_id": "uid-alice"}


def test_lookup_by_email_404_on_miss():
    _, client = _client()
    _provision(client, ALICE)
    resp = client.get("/accounts/lookup", params={"email": "nobody@example.com"}, headers=BOB)
    assert resp.status_code == 404
    assert resp.json()["detail"] == "no account for that email"


def test_lookup_requires_auth():
    _, client = _client()
    _provision(client, ALICE)
    # No Authorization header and (per this app's default) no ?account=.
    resp = client.get("/accounts/lookup", params={"email": "alice@example.com"})
    assert resp.status_code == 401


def test_lookup_rejects_legacy_account_param_even_when_legacy_is_allowed():
    """I2/I3 pinning test: before the fix, an allow-legacy server let ANY
    caller resolve someone's email to their account id by sending an
    unauthenticated `?account=<anyone>` -- no token, no proof of identity.
    This route now requires a full (non-legacy) principal (see
    watch/auth.py's require_full_auth), so the legacy query param alone
    must be a 401, not a 200 -- even when the server is built with
    allow_legacy=True (unlike this file's other tests, which build without it
    to prove the strict route needs no legacy support at all)."""
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), allow_legacy=True,
    ))
    _provision(client, ALICE)
    resp = client.get(
        "/accounts/lookup", params={"email": "alice@example.com", "account": "bob"}
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "this endpoint requires sign-in"


def test_lookup_is_case_sensitive_matching_how_emails_are_stored():
    """auth.ensure_account persists the token claim's email VERBATIM
    (`email=principal.email`, no `.lower()`/`.strip()`) at account-creation
    time -- so a route that case-folded its query would search for
    something that could never equal what's actually on the Account row.
    The route therefore does the same exact match store.get_account_by_email
    already contracts for (see test_store.py's test_account_lookup_by_email),
    and this test documents that verified behavior rather than assuming
    case-insensitivity that the write path doesn't actually provide."""
    _, client = _client()
    _provision(client, BOB)  # stored as "Bob@Example.com" -- see TOKENS above

    exact = client.get("/accounts/lookup", params={"email": "Bob@Example.com"}, headers=ALICE)
    assert exact.status_code == 200
    assert exact.json() == {"account_id": "uid-bob"}

    lowered = client.get("/accounts/lookup", params={"email": "bob@example.com"}, headers=ALICE)
    assert lowered.status_code == 404


def test_route_reuses_the_shared_get_account_by_email_seam():
    """Firestore-vs-memory parity: the route calls store.get_account_by_email
    rather than re-implementing its own account scan, so whichever
    LiveSessionStore backend get_store() resolves to (Memory locally,
    Firestore in prod) behaves identically here. Both implementations already
    carry the identical `(email: str) -> Account | None` contract in
    server/watch/store.py; asserting their signatures match is the honest
    check available without a live Firestore (no test in this suite exercises
    FirestoreLiveSessionStore directly, since it requires real GCP
    credentials -- see server/watch/store.py's lazy `_get_db` import)."""
    mem_sig = inspect.signature(MemoryLiveSessionStore.get_account_by_email)
    fs_sig = inspect.signature(FirestoreLiveSessionStore.get_account_by_email)
    assert mem_sig == fs_sig
    assert (
        MemoryLiveSessionStore.get_account_by_email.__doc__
        == FirestoreLiveSessionStore.get_account_by_email.__doc__
    )
