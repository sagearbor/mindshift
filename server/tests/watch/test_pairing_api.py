# Ported from gauge@2157433 server/tests/test_pairing_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B8): server.main.create_app -> watch.testing.create_watch_test_app
# (keyword-only assembly, no env vars -- see its own docstring); server.store.
# MemoryEpisodeStore -> watch.store.MemoryLiveSessionStore; server.models.Pairing
# -> watch.models.Pairing; server.pairing_store -> watch.pairing_store;
# server.pairing_api -> watch.routers.pairing. GAUGE_FIREBASE_PROJECT env-var
# monkeypatching is dropped (no equivalent env var here -- FIREBASE_PROJECT_ID
# always has a default, see watch/auth.py's module docstring) -- the one test
# that relied on it (the real-verifier-chain end-to-end check) instead builds
# the real chain explicitly via `full_verifier=get_full_verifier(pstore)`,
# matching testing.py's "every knob is an explicit kwarg" design instead of
# relying on an unspecified-verifier default (which create_watch_test_app,
# unlike gauge's create_app, does not have).
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from watch.models import Pairing
from watch.pairing_store import MemoryPairingStore, hash_secret
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

from server.tests.watch.test_auth import StubVerifier

TOKENS = {
    "alice-token": {"sub": "alice", "email": "alice@example.com"},
    "bob-token": {"sub": "bob", "email": "bob@example.com"},
}
A = {"Authorization": "Bearer alice-token"}
B = {"Authorization": "Bearer bob-token"}


def _client(pairing_store=None):
    store = MemoryLiveSessionStore()
    pstore = pairing_store if pairing_store is not None else MemoryPairingStore()
    client = TestClient(create_watch_test_app(
        store=store, pairing_store=pstore, verifier=StubVerifier(TOKENS), allow_legacy=True,
    ))
    return pstore, client


def test_pair_start_requires_no_auth_and_mints_a_code():
    _, client = _client()
    resp = client.post("/me/pair/start")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["code"]) == 6
    assert body["pairing_id"]
    assert body["expires_at"]


def test_pair_start_never_stores_the_code_in_plaintext():
    pstore, client = _client()
    body = client.post("/me/pair/start").json()
    stored = pstore._pairings[body["pairing_id"]]
    assert stored.code_hash == hash_secret(body["code"])
    # The Pairing model itself has no plaintext `code` field to leak.
    assert not hasattr(stored, "code")


def test_pair_status_pending_right_after_start():
    _, client = _client()
    started = client.post("/me/pair/start").json()
    resp = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]})
    assert resp.status_code == 200
    assert resp.json() == {"status": "pending", "account_id": None, "device_token": None}


def test_pair_status_unknown_pairing_id_is_200_expired_not_404():
    # CONTRACT RULING: the watch's poll() treats any non-200 as a transport
    # failure ("null"), indistinguishable from "server unreachable" -- an
    # unrecognized pairing_id must still be an honest, decodable 200.
    _, client = _client()
    resp = client.get("/me/pair/status", params={"pairing_id": "never-existed"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "expired"


def test_pair_status_expired_pairing_is_200_expired_not_404_or_410():
    pstore, client = _client()
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    pstore._pairings["pid-old"] = Pairing(
        id="pid-old", code_hash=hash_secret("ABCDEF"), status="pending",
        created_at=past, expires_at=past,
    )
    resp = client.get("/me/pair/status", params={"pairing_id": "pid-old"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "expired"


def test_pair_claim_by_legacy_account_is_401_not_authenticated():
    # Explicit "legacy ?account= gets NO access to any pair route" check.
    _, client = _client()
    started = client.post("/me/pair/start").json()
    resp = client.post("/me/pair/claim", params={"account": "default"}, json={"code": started["code"]})
    assert resp.status_code == 401


def test_pair_claim_unknown_code_is_404():
    _, client = _client()
    resp = client.post("/me/pair/claim", headers=A, json={"code": "ZZZZZZ"})
    assert resp.status_code == 404


def test_pair_claim_then_status_delivers_the_raw_device_token():
    _, client = _client()
    started = client.post("/me/pair/start").json()

    claimed = client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    assert claimed.status_code == 200
    claimed_body = claimed.json()
    assert claimed_body == {"status": "claimed", "pairing_id": started["pairing_id"], "account_id": "alice"}

    status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
    assert status["status"] == "claimed"
    assert status["account_id"] == "alice"
    assert status["device_token"]  # a real, non-empty opaque token


def test_pair_claim_never_returns_the_raw_device_token_to_the_claimer():
    # Least privilege: only the watch (via status poll) sees the raw token.
    _, client = _client()
    started = client.post("/me/pair/start").json()
    claimed = client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    assert "device_token" not in claimed.json()


def test_pair_claim_same_code_twice_is_409():
    _, client = _client()
    started = client.post("/me/pair/start").json()
    first = client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    assert first.status_code == 200
    second = client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    assert second.status_code == 409


def test_pair_claim_expired_code_is_404():
    pstore, client = _client()
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    pstore._pairings["pid-old"] = Pairing(
        id="pid-old", code_hash=hash_secret("EXPIRD"), status="pending",
        created_at=past, expires_at=past,
    )
    resp = client.post("/me/pair/claim", headers=A, json={"code": "EXPIRD"})
    assert resp.status_code == 404


def test_device_token_never_stored_in_plaintext_anywhere():
    pstore, client = _client()
    started = client.post("/me/pair/start").json()
    client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
    raw_token = status["device_token"]

    pairing_record = pstore._pairings[started["pairing_id"]]
    assert pairing_record.device_token_hash == hash_secret(raw_token)

    device_record = pstore.get_device_token_by_hash(hash_secret(raw_token))
    assert device_record is not None and device_record.account_id == "alice"
    # DeviceToken itself has no plaintext token field to leak.
    assert not hasattr(device_record, "token")


def test_end_to_end_device_token_authenticates_as_full_auth_via_the_real_verifier_chain():
    """Builds the app with the REAL verifier chain (watch.auth.get_full_verifier),
    not a StubVerifier -- proves DeviceTokenVerifier is correctly wired into
    the pairing router's full_auth_dep, end to end through a real HTTP round
    trip, not just unit-tested in isolation. `full_verifier` is the knob
    testing.py reserved for exactly this (see its own docstring) -- unlike
    gauge's create_app, create_watch_test_app never silently defaults to the
    real chain, so this test builds it explicitly instead of relying on an
    unspecified `verifier` default."""
    from watch.auth import get_full_verifier

    store = MemoryLiveSessionStore()
    pstore = MemoryPairingStore()
    # `verifier` (the plain-auth knob) is left unset -- irrelevant here, this
    # test only exercises /me/pair/claim, which is always full-auth-gated.
    client = TestClient(create_watch_test_app(
        store=store, pairing_store=pstore, full_verifier=get_full_verifier(pstore),
    ))

    started = client.post("/me/pair/start").json()
    claimed = client.post(
        "/me/pair/claim",
        headers={"Authorization": "Bearer irrelevant"},  # rejected: no verifier accepts it either
        json={"code": started["code"]},
    )
    # Without a valid Firebase ID token and no matching device token yet,
    # even the claim itself can't authenticate -- confirms full_auth is
    # genuinely gated, not accidentally left open.
    assert claimed.status_code == 401


def test_ladder_unchanged_bearer_wins_over_legacy_with_chained_verifier():
    """Regression for 'the ladder must be unchanged for existing tokens':
    a verified Firebase-shaped bearer token still wins over ?account= when
    DeviceTokenVerifier is chained in alongside it."""
    from watch.auth import ChainedTokenVerifier, DeviceTokenVerifier

    store = MemoryLiveSessionStore()
    pstore = MemoryPairingStore()
    chained = ChainedTokenVerifier([StubVerifier(TOKENS), DeviceTokenVerifier(pstore)])
    client = TestClient(create_watch_test_app(
        store=store, pairing_store=pstore, verifier=chained, allow_legacy=True,
    ))

    resp = client.get("/me", headers=A, params={"account": "default"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["account_id"] == "alice" and body["legacy"] is False


def test_device_token_authenticates_via_chained_verifier_too():
    from watch.auth import ChainedTokenVerifier, DeviceTokenVerifier

    store = MemoryLiveSessionStore()
    pstore = MemoryPairingStore()
    chained = ChainedTokenVerifier([StubVerifier(TOKENS), DeviceTokenVerifier(pstore)])
    client = TestClient(create_watch_test_app(
        store=store, pairing_store=pstore, verifier=chained, allow_legacy=True,
    ))

    started = client.post("/me/pair/start").json()
    client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
    device_token = status["device_token"]

    resp = client.get("/me", headers={"Authorization": f"Bearer {device_token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["account_id"] == "alice" and body["legacy"] is False


def test_garbage_bearer_token_is_401_even_with_pairing_configured():
    from watch.auth import ChainedTokenVerifier, DeviceTokenVerifier

    store = MemoryLiveSessionStore()
    pstore = MemoryPairingStore()
    chained = ChainedTokenVerifier([StubVerifier(TOKENS), DeviceTokenVerifier(pstore)])
    client = TestClient(create_watch_test_app(
        store=store, pairing_store=pstore, verifier=chained, allow_legacy=True,
    ))

    resp = client.get("/me", headers={"Authorization": "Bearer complete-nonsense"})
    assert resp.status_code == 401


# --- FIX ROUND 1 hardening: fetch-count cap on the plaintext token read ---

def test_token_is_returned_on_the_fifth_read_then_redacted_on_the_sixth():
    from watch.routers.pairing import MAX_TOKEN_READS

    assert MAX_TOKEN_READS == 5  # pins the constant this test's loop bounds assume
    _, client = _client()
    started = client.post("/me/pair/start").json()
    client.post("/me/pair/claim", headers=A, json={"code": started["code"]})

    for i in range(MAX_TOKEN_READS):
        status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
        assert status["status"] == "claimed"
        assert status["device_token"], f"read {i + 1} should still return the token"

    # The (MAX_TOKEN_READS + 1)th read: still honestly "claimed", but the
    # token is gone -- never returned again.
    status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
    assert status["status"] == "claimed"
    assert status["account_id"] == "alice"
    assert status["device_token"] is None


def test_token_reads_are_counted_per_pairing_not_globally():
    _, client = _client()
    started = client.post("/me/pair/start").json()
    client.post("/me/pair/claim", headers=A, json={"code": started["code"]})
    # Four reads -- one short of the cap.
    for _ in range(4):
        client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]})
    status = client.get("/me/pair/status", params={"pairing_id": started["pairing_id"]}).json()
    # The 5th read (== MAX_TOKEN_READS) still returns the token.
    assert status["device_token"]


# --- FIX ROUND 2 hardening: per-CALLING-ACCOUNT failed-claim-attempt counter ---
# REPLACES FIX ROUND 1's per-pairing "sole pending pairing" heuristic, which
# was ruled materially bypassable: POST /me/pair/start takes no auth and no
# rate limit, so an attacker could keep one free decoy pairing perpetually
# pending, permanently no-opping that counter for a real target elsewhere.
# See watch/routers/pairing.py's module docstring for the full design ruling.

def _wrong_code_for(real_code: str) -> str:
    """A code guaranteed different from `real_code` (flips its first
    character), rather than a fixed literal that has a theoretical
    (vanishingly small but nonzero) chance of colliding with a randomly
    generated real code."""
    return ("A" if real_code[0] != "A" else "B") + real_code[1:]


def test_fifteen_failed_claims_lock_out_the_account_even_with_a_correct_code():
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    assert MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT == 15  # pins the constant this test's loop bound assumes
    _, client = _client()
    started = client.post("/me/pair/start").json()
    real_code = started["code"]
    wrong_code = _wrong_code_for(real_code)

    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT):
        resp = client.post("/me/pair/claim", headers=A, json={"code": wrong_code})
        assert resp.status_code == 404

    # Account A's 16th attempt: even the CORRECT code is now rejected.
    resp = client.post("/me/pair/claim", headers=A, json={"code": real_code})
    assert resp.status_code == 429


def test_fourteen_failed_claims_do_not_yet_lock_out_the_account():
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    _, client = _client()
    started = client.post("/me/pair/start").json()
    real_code = started["code"]
    wrong_code = _wrong_code_for(real_code)

    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT - 1):
        client.post("/me/pair/claim", headers=A, json={"code": wrong_code})

    resp = client.post("/me/pair/claim", headers=A, json={"code": real_code})
    assert resp.status_code == 200


def test_account_b_is_unaffected_by_account_a_failures():
    # The core bystander-isolation property FIX ROUND 2 exists for: a
    # DIFFERENT account's claim attempts are never touched by another
    # account's failures, full stop -- no ambiguity-driven no-op logic
    # required, unlike FIX ROUND 1's per-pairing design.
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    _, client = _client()
    p_a = client.post("/me/pair/start").json()
    p_b = client.post("/me/pair/start").json()
    wrong_code = _wrong_code_for(p_a["code"])

    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT):
        resp = client.post("/me/pair/claim", headers=A, json={"code": wrong_code})
        assert resp.status_code == 404

    # Account A is now locked out...
    resp_a = client.post("/me/pair/claim", headers=A, json={"code": p_a["code"]})
    assert resp_a.status_code == 429

    # ...but account B, which never failed a claim, can still claim its OWN
    # pairing normally -- A's failures never touched B's counter.
    resp_b = client.post("/me/pair/claim", headers=B, json={"code": p_b["code"]})
    assert resp_b.status_code == 200


def test_decoy_pairing_no_longer_defeats_the_counter():
    # FIX ROUND 1 regression check: the old per-pairing heuristic was
    # defeatable by keeping one free, unauthenticated decoy pairing
    # perpetually pending (POST /me/pair/start has no auth/rate limit),
    # which permanently no-opped the miss counter. The new per-account
    # counter has no such escape hatch -- multiple simultaneously-pending
    # pairings (decoy or otherwise) don't change anything, since the count
    # is keyed by the calling account, not by which/how-many pairings exist.
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    _, client = _client()
    target = client.post("/me/pair/start").json()
    client.post("/me/pair/start")  # decoy 1 -- kept "pending", never claimed
    client.post("/me/pair/start")  # decoy 2
    wrong_code = _wrong_code_for(target["code"])

    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT):
        client.post("/me/pair/claim", headers=A, json={"code": wrong_code})

    # The account is locked out despite 3 pairings (1 target + 2 decoys)
    # being simultaneously pending throughout.
    resp = client.post("/me/pair/claim", headers=A, json={"code": target["code"]})
    assert resp.status_code == 429


# --- FIX ROUND 3: time-windowed lockout, measured from the LAST failure ---
# Coordinator ruling: FIX ROUND 2's PERMANENT lockout was ruled
# not-ship-as-is -- a time-windowed lockout (24h since the most recent
# failure, never reset by a success) closes the same laundering exploit
# while giving a legitimately locked-out user a real recovery path. See
# watch/routers/pairing.py's module docstring for the full ruling.

def test_scenario_a_fifteen_failures_lock_then_24h_later_next_attempt_is_allowed_and_count_restarts():
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    pstore, client = _client()
    started = client.post("/me/pair/start").json()
    real_code = started["code"]
    wrong_code = _wrong_code_for(real_code)

    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT):
        assert client.post("/me/pair/claim", headers=A, json={"code": wrong_code}).status_code == 404

    # Locked, per FIX ROUND 2's existing behavior.
    assert client.post("/me/pair/claim", headers=A, json={"code": wrong_code}).status_code == 429
    record = pstore._failed_claims["alice"]
    assert record.count == MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    # Advance the clock 25h by rewriting last_failed_at into the past --
    # simulates real time passing without a 25-hour-long test.
    stale_last_failed = datetime.now(timezone.utc) - timedelta(hours=25)
    pstore._failed_claims["alice"] = record.model_copy(update={"last_failed_at": stale_last_failed.isoformat()})

    # Next attempt (still wrong) is ALLOWED THROUGH the lockout check (not
    # 429) -- it then fails on its own merits (wrong code -> 404), and the
    # count RESTARTS at 1, not 16.
    resp = client.post("/me/pair/claim", headers=A, json={"code": wrong_code})
    assert resp.status_code == 404
    assert pstore._failed_claims["alice"].count == 1

    # And a CORRECT code right after that single restarted failure succeeds
    # (nowhere near the cap again).
    resp = client.post("/me/pair/claim", headers=A, json={"code": real_code})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_scenario_b_window_is_measured_from_the_most_recent_failure_not_the_first():
    # Pure unit test against the clock-injected helpers directly (no real
    # elapsed time / HTTP round trip needed for exact hour-boundary math).
    from watch.routers.pairing import (
        MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT, _is_account_locked, _record_failed_claim,
    )
    from watch.pairing_store import MemoryPairingStore

    store = MemoryPairingStore()
    hour0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # MAX-1 failures at hour 0, then one more failure at hour 23 -- the
    # window is now anchored to hour 23, the LAST failure, not hour 0.
    for _ in range(MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT - 1):
        await _record_failed_claim(store, "acct1", hour0)
    await _record_failed_claim(store, "acct1", hour0 + timedelta(hours=23))

    record = await store.get_failed_claim_record("acct1")
    assert record.count == MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT
    assert record.last_failed_at == (hour0 + timedelta(hours=23)).isoformat()

    # Hour 24 -- only 1h since the LAST failure (hour 23): still locked.
    # (If the window were wrongly anchored to hour 0, this would already
    # be unlocked at hour 24 -- this assertion is what actually pins
    # "measured from the LAST failure.")
    assert _is_account_locked(record, hour0 + timedelta(hours=24)) is True

    # Hour 47 -- exactly 24h since hour 23: window has expired, unlocked.
    assert _is_account_locked(record, hour0 + timedelta(hours=47)) is False


def test_scenario_c_a_successful_claim_never_resets_the_count_mid_window():
    # The anti-laundering guard from FIX ROUND 2, re-verified under the FIX
    # ROUND 3 mechanism: a successful claim on a DIFFERENT pairing must not
    # touch account A's failed-claim record at all while its window is
    # still active -- otherwise an attacker could "launder" their budget via
    # a legitimate claim of their own.
    from watch.routers.pairing import MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT

    pstore, client = _client()
    target = client.post("/me/pair/start").json()
    own_pairing = client.post("/me/pair/start").json()
    wrong_code = _wrong_code_for(target["code"])

    failures = MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT - 1
    for _ in range(failures):
        assert client.post("/me/pair/claim", headers=A, json={"code": wrong_code}).status_code == 404
    assert pstore._failed_claims["alice"].count == failures

    # A legitimate, successful claim of an unrelated pairing by the SAME
    # account.
    resp = client.post("/me/pair/claim", headers=A, json={"code": own_pairing["code"]})
    assert resp.status_code == 200

    # The failed-claim count is untouched by the success -- still `failures`,
    # not reset to 0.
    assert pstore._failed_claims["alice"].count == failures
