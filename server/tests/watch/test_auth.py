# Ported from gauge@2157433 server/tests/test_auth.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import pytest
from fastapi import HTTPException

from watch.auth import (
    InvalidToken, Principal, parse_bearer, resolve_principal, resolve_ws_principal,
)


class StubVerifier:
    """Accepts exactly the tokens in `tokens`; everything else is invalid."""

    def __init__(self, tokens=None):
        self.tokens = tokens or {"good-token": {"sub": "uid-123", "email": "a@example.com"}}

    def verify(self, token: str) -> dict:
        if token not in self.tokens:
            raise InvalidToken("bad token")
        return self.tokens[token]


def test_parse_bearer():
    assert parse_bearer("Bearer abc.def") == "abc.def"
    assert parse_bearer("bearer abc.def") == "abc.def"
    assert parse_bearer("Basic abc") is None
    assert parse_bearer("Bearer    ") is None
    assert parse_bearer(None) is None


def test_verified_token_wins_over_account_param():
    p = resolve_principal("Bearer good-token", "default", StubVerifier(), allow_legacy=True)
    assert p == Principal(account_id="uid-123", email="a@example.com", legacy=False)


def test_bad_token_is_401_and_never_falls_back_to_legacy():
    with pytest.raises(HTTPException) as exc:
        resolve_principal("Bearer nope", "default", StubVerifier(), allow_legacy=True)
    assert exc.value.status_code == 401


def test_token_without_subject_is_401():
    v = StubVerifier({"subless": {"sub": "", "email": "a@example.com"}})
    with pytest.raises(HTTPException) as exc:
        resolve_principal("Bearer subless", "default", v, allow_legacy=True)
    assert exc.value.status_code == 401


def test_bearer_with_no_verifier_configured_is_401():
    with pytest.raises(HTTPException) as exc:
        resolve_principal("Bearer good-token", "default", None, allow_legacy=True)
    assert exc.value.status_code == 401


def test_legacy_account_param_accepted_while_flag_on():
    p = resolve_principal(None, "default", StubVerifier(), allow_legacy=True)
    assert p == Principal(account_id="default", email=None, legacy=True)


def test_legacy_account_param_rejected_when_flag_off():
    with pytest.raises(HTTPException) as exc:
        resolve_principal(None, "default", StubVerifier(), allow_legacy=False)
    assert exc.value.status_code == 401


def test_no_credentials_at_all_is_401():
    with pytest.raises(HTTPException) as exc:
        resolve_principal(None, None, StubVerifier(), allow_legacy=True)
    assert exc.value.status_code == 401


def test_ws_principal_prefers_token_query_param():
    p = resolve_ws_principal("good-token", "default", StubVerifier(), allow_legacy=True)
    assert p.account_id == "uid-123" and p.legacy is False


def test_ws_principal_falls_back_to_legacy_account():
    p = resolve_ws_principal(None, "default", StubVerifier(), allow_legacy=True)
    assert p.account_id == "default" and p.legacy is True


def test_ws_principal_bad_token_is_401():
    with pytest.raises(HTTPException) as exc:
        resolve_ws_principal("nope", "default", StubVerifier(), allow_legacy=True)
    assert exc.value.status_code == 401


def test_settings_allow_legacy_defaults_true(monkeypatch):
    monkeypatch.delenv("MINDSHIFT_ALLOW_LEGACY_ACCOUNT", raising=False)
    from watch.config import Settings
    assert Settings().allow_legacy_account is True


def test_settings_allow_legacy_off(monkeypatch):
    from watch.config import Settings
    for value in ("false", "FALSE", "0", "no", "off"):
        monkeypatch.setenv("MINDSHIFT_ALLOW_LEGACY_ACCOUNT", value)
        assert Settings().allow_legacy_account is False
    monkeypatch.setenv("MINDSHIFT_ALLOW_LEGACY_ACCOUNT", "true")
    assert Settings().allow_legacy_account is True


def test_firebase_verifier_constructs_without_importing_sdk():
    # Lazy-import contract: constructing must never touch firebase_admin, so a
    # base install (no firebase-admin) can still build the app and serve legacy
    # traffic. Only .verify() may import.
    #
    # ADAPTED (Task B3): unlike Gauge, this FirebaseTokenVerifier has no
    # per-instance project id -- it reuses THIS repo's shared
    # server/auth.py's init_firebase()/FIREBASE_PROJECT_ID (see auth.py's
    # module docstring), so construction is a bare no-arg no-op.
    import sys
    from watch.auth import FirebaseTokenVerifier
    before = "firebase_admin" in sys.modules
    FirebaseTokenVerifier()
    assert ("firebase_admin" in sys.modules) == before


# --- Wave C: DeviceTokenVerifier / ChainedTokenVerifier / get_full_verifier ---
# (docs/superpowers/plans/2026-08-04-gauge-wave-c-couples-wrist.md, Open
# Question 1 — the server-lane companion to the watch's short-code pairing
# flow). FakePairingStore below implements only the two synchronous methods
# DeviceTokenVerifier actually calls (watch/pairing_store.py's
# PairingStore.get_device_token_by_hash contract) — no need to pull in the
# real MemoryPairingStore just to unit-test the verifier in isolation.

class FakePairingStore:
    def __init__(self, tokens: dict[str, str] | None = None):
        # token_hash -> account_id
        self.tokens = tokens or {}

    def get_device_token_by_hash(self, token_hash: str):
        account_id = self.tokens.get(token_hash)
        if account_id is None:
            return None
        from watch.models import DeviceToken
        return DeviceToken(token_hash=token_hash, account_id=account_id, created_at="now", pairing_id="pid")


def test_device_token_verifier_accepts_a_known_token():
    from watch.auth import DeviceTokenVerifier
    from watch.pairing_store import hash_secret

    store = FakePairingStore({hash_secret("real-device-token"): "acct1"})
    v = DeviceTokenVerifier(store)
    claims = v.verify("real-device-token")
    assert claims == {"sub": "acct1", "email": None}


def test_device_token_verifier_rejects_an_unknown_token():
    from watch.auth import DeviceTokenVerifier

    v = DeviceTokenVerifier(FakePairingStore())
    with pytest.raises(InvalidToken):
        v.verify("never-issued")


class RaisingPairingStore:
    """Simulates a Firestore-backed store hitting a transient infra
    exception (network timeout, service unavailable, quota) -- FIX ROUND 1
    Important finding regression test."""

    def get_device_token_by_hash(self, token_hash: str):
        raise RuntimeError("simulated Firestore outage")


def test_device_token_verifier_store_exception_becomes_verifier_unavailable_not_a_crash():
    # FIX ROUND 1 (review Important finding): DeviceTokenVerifier.verify()
    # must mirror FirebaseTokenVerifier's defensive `except Exception`
    # around its backing call -- a store exception must never propagate as
    # an unhandled 500.
    #
    # FIX ROUND 3 ADDENDUM (superseding this test's original assertion): a
    # store EXCEPTION is transient-infra-unavailable, not a verdict on the
    # token -- it must become VerifierUnavailable, NOT InvalidToken, so
    # resolve_principal maps it to 503, not 401 (see auth.py's module
    # docstring for the locked cross-repo watch contract this protects).
    from watch.auth import DeviceTokenVerifier, VerifierUnavailable

    v = DeviceTokenVerifier(RaisingPairingStore())
    with pytest.raises(VerifierUnavailable):
        v.verify("any-token")


def test_device_token_store_exception_is_503_not_401_through_resolve_principal():
    # End-to-end through the same path a real request takes: a bearer token
    # whose backing store call raises must resolve to a 503 HTTPException
    # (transient-unavailable), not an unhandled exception and NOT a 401 --
    # FIX ROUND 3 ADDENDUM: a 401 here would make the watch client clear its
    # stored device token over what is actually just a backend hiccup.
    from watch.auth import DeviceTokenVerifier

    v = DeviceTokenVerifier(RaisingPairingStore())
    with pytest.raises(HTTPException) as exc:
        resolve_principal("Bearer any-token", None, v, allow_legacy=True)
    assert exc.value.status_code == 503


def test_device_token_genuinely_unknown_token_is_still_401_through_resolve_principal():
    # FIX ROUND 3 ADDENDUM regression guard: the 401-vs-503 split must not
    # blur the case that SHOULD still be 401 -- a store that cleanly
    # answers "no such token" (no exception at all) is a genuine credential
    # rejection, unchanged from before this round.
    from watch.auth import DeviceTokenVerifier

    v = DeviceTokenVerifier(FakePairingStore())
    with pytest.raises(HTTPException) as exc:
        resolve_principal("Bearer never-issued", None, v, allow_legacy=True)
    assert exc.value.status_code == 401


def test_chained_token_verifier_tries_in_order_first_success_wins():
    from watch.auth import ChainedTokenVerifier

    chained = ChainedTokenVerifier([StubVerifier({"a": {"sub": "uid-a"}}), StubVerifier({"b": {"sub": "uid-b"}})])
    assert chained.verify("a") == {"sub": "uid-a"}
    assert chained.verify("b") == {"sub": "uid-b"}


def test_chained_token_verifier_raises_when_every_verifier_rejects():
    from watch.auth import ChainedTokenVerifier

    chained = ChainedTokenVerifier([StubVerifier({"a": {"sub": "uid-a"}}), StubVerifier({"b": {"sub": "uid-b"}})])
    with pytest.raises(InvalidToken):
        chained.verify("neither")


def test_chained_token_verifier_propagates_verifier_unavailable_uncaught():
    # FIX ROUND 3 ADDENDUM: a VerifierUnavailable from one verifier in the
    # chain must NOT be swallowed and treated as "try the next verifier" the
    # way InvalidToken is -- it says nothing about whether the token itself
    # is good, so masking it as a rejection would be a false verdict, not
    # just a missed opportunity to fall through.
    from watch.auth import ChainedTokenVerifier, VerifierUnavailable

    class UnavailableVerifier:
        def verify(self, token: str) -> dict:
            raise VerifierUnavailable("simulated outage")

    chained = ChainedTokenVerifier([UnavailableVerifier(), StubVerifier({"a": {"sub": "uid-a"}})])
    with pytest.raises(VerifierUnavailable):
        chained.verify("a")


def test_chained_token_verifier_rejects_empty_list():
    from watch.auth import ChainedTokenVerifier

    with pytest.raises(ValueError):
        ChainedTokenVerifier([])


def test_chained_verifier_does_not_change_a_verified_firebase_token_behavior():
    # "The ladder must be unchanged for existing tokens": a token the FIRST
    # verifier accepts must resolve identically whether or not a second
    # verifier is chained in behind it.
    from watch.auth import ChainedTokenVerifier

    firebase_shaped = StubVerifier()  # accepts "good-token" -> uid-123
    solo = resolve_principal("Bearer good-token", "default", firebase_shaped, allow_legacy=True)
    chained_verifier = ChainedTokenVerifier([firebase_shaped, DeviceTokenVerifierStub()])
    chained_result = resolve_principal("Bearer good-token", "default", chained_verifier, allow_legacy=True)
    assert solo == chained_result


class DeviceTokenVerifierStub:
    """A verifier that always rejects -- stands in for DeviceTokenVerifier
    in the chain-ordering test above without needing a real pairing store."""
    def verify(self, token: str) -> dict:
        raise InvalidToken("stub always rejects")


def test_get_full_verifier_always_chains_firebase_first():
    # ADAPTED (Task B3): Gauge gated Firebase inclusion on GAUGE_FIREBASE_PROJECT
    # being set (dropped per the rename map -- this repo reuses
    # server/auth.py's FIREBASE_PROJECT_ID, which always has a default, so
    # Firebase is unconditionally in the chain, tried first; DeviceTokenVerifier
    # is always chained in behind it, exactly as in Gauge).
    from watch.auth import ChainedTokenVerifier, DeviceTokenVerifier, FirebaseTokenVerifier, get_full_verifier

    chained = get_full_verifier(FakePairingStore())
    assert isinstance(chained, ChainedTokenVerifier)
    assert isinstance(chained._verifiers[0], FirebaseTokenVerifier)
    assert isinstance(chained._verifiers[1], DeviceTokenVerifier)
