# Ported from gauge@2157433 server/tests/test_pairing_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import sys

import pytest

from watch.models import DeviceToken, Pairing
from watch.pairing_store import MemoryPairingStore, hash_secret


def _pairing(**overrides) -> Pairing:
    defaults = dict(
        id="pid-1",
        code_hash=hash_secret("K7QP2M"),
        status="pending",
        created_at="2026-08-04T10:00:00+00:00",
        expires_at="2026-08-04T10:10:00+00:00",
    )
    defaults.update(overrides)
    return Pairing(**defaults)


def test_hash_secret_is_deterministic_and_looks_like_sha256_hex():
    h1 = hash_secret("K7QP2M")
    h2 = hash_secret("K7QP2M")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_hash_secret_different_inputs_differ():
    assert hash_secret("K7QP2M") != hash_secret("K7QP2N")


def test_hash_secret_never_contains_the_raw_input():
    # Not a rigorous crypto property in general, but for this specific short
    # alphanumeric code it's a cheap sanity check that we're not doing
    # something silly like base64-encoding instead of hashing.
    assert "K7QP2M" not in hash_secret("K7QP2M")


@pytest.mark.anyio
async def test_create_and_get_pairing_roundtrips():
    store = MemoryPairingStore()
    await store.create_pairing(_pairing())
    fetched = await store.get_pairing("pid-1")
    assert fetched is not None
    assert fetched.id == "pid-1"
    assert fetched.status == "pending"


@pytest.mark.anyio
async def test_get_pairing_unknown_id_returns_none():
    store = MemoryPairingStore()
    assert await store.get_pairing("nope") is None


@pytest.mark.anyio
async def test_get_pairing_by_code_hash_finds_it():
    store = MemoryPairingStore()
    await store.create_pairing(_pairing())
    found = await store.get_pairing_by_code_hash(hash_secret("K7QP2M"))
    assert found is not None and found.id == "pid-1"


@pytest.mark.anyio
async def test_get_pairing_by_code_hash_unknown_hash_returns_none():
    store = MemoryPairingStore()
    await store.create_pairing(_pairing())
    assert await store.get_pairing_by_code_hash(hash_secret("wrong-code")) is None


@pytest.mark.anyio
async def test_update_pairing_atomically_persists_the_mutator_result():
    store = MemoryPairingStore()
    await store.create_pairing(_pairing())

    def claim(p):
        p.status = "claimed"
        p.claimed_account_id = "acct1"
        return p

    updated = await store.update_pairing_atomically("pid-1", claim)
    assert updated.status == "claimed" and updated.claimed_account_id == "acct1"
    refetched = await store.get_pairing("pid-1")
    assert refetched.status == "claimed"


@pytest.mark.anyio
async def test_update_pairing_atomically_persists_nothing_if_mutator_raises():
    store = MemoryPairingStore()
    await store.create_pairing(_pairing())

    def boom(p):
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await store.update_pairing_atomically("pid-1", boom)

    unchanged = await store.get_pairing("pid-1")
    assert unchanged.status == "pending"


@pytest.mark.anyio
async def test_update_pairing_atomically_mutator_sees_none_for_unknown_id():
    store = MemoryPairingStore()
    seen = {}

    def record(p):
        seen["value"] = p
        return _pairing(id="brand-new")

    await store.update_pairing_atomically("does-not-exist", record)
    assert seen["value"] is None


@pytest.mark.anyio
async def test_get_failed_claim_record_is_none_for_an_account_never_seen():
    store = MemoryPairingStore()
    assert await store.get_failed_claim_record("never-tried") is None


@pytest.mark.anyio
async def test_set_and_get_failed_claim_record_roundtrips():
    # FIX ROUND 3: raw store-level roundtrip -- the reset-or-increment
    # POLICY (whether a new failure restarts the count or extends it) lives
    # in watch/routers/pairing.py's _record_failed_claim (Task B8), not here.
    # This store method is a plain full-replace write.
    store = MemoryPairingStore()
    await store.set_failed_claim_record("acct1", count=3, last_failed_at="2026-08-04T10:00:00+00:00")

    record = await store.get_failed_claim_record("acct1")
    assert record is not None
    assert record.account_id == "acct1"
    assert record.count == 3
    assert record.last_failed_at == "2026-08-04T10:00:00+00:00"


@pytest.mark.anyio
async def test_set_failed_claim_record_overwrites_the_prior_record():
    store = MemoryPairingStore()
    await store.set_failed_claim_record("acct1", count=3, last_failed_at="2026-08-04T10:00:00+00:00")
    await store.set_failed_claim_record("acct1", count=1, last_failed_at="2026-08-05T10:00:00+00:00")

    record = await store.get_failed_claim_record("acct1")
    assert record.count == 1
    assert record.last_failed_at == "2026-08-05T10:00:00+00:00"


@pytest.mark.anyio
async def test_failed_claim_records_are_tracked_independently_per_account():
    store = MemoryPairingStore()
    await store.set_failed_claim_record("acct-a", count=2, last_failed_at="2026-08-04T10:00:00+00:00")
    await store.set_failed_claim_record("acct-b", count=1, last_failed_at="2026-08-04T10:00:00+00:00")

    a = await store.get_failed_claim_record("acct-a")
    b = await store.get_failed_claim_record("acct-b")
    assert a.count == 2
    assert b.count == 1


def test_put_and_get_device_token_by_hash_roundtrips():
    store = MemoryPairingStore()
    token = DeviceToken(
        token_hash=hash_secret("raw-device-token"),
        account_id="acct1",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    )
    store.put_device_token(token)
    fetched = store.get_device_token_by_hash(hash_secret("raw-device-token"))
    assert fetched is not None
    assert fetched.account_id == "acct1"
    assert fetched.pairing_id == "pid-1"


def test_get_device_token_by_hash_unknown_returns_none():
    store = MemoryPairingStore()
    assert store.get_device_token_by_hash(hash_secret("never-issued")) is None


def test_device_token_store_methods_are_synchronous():
    # Load-bearing contract: watch/auth.py's (Task B3) DeviceTokenVerifier.verify()
    # is called from the synchronous TokenVerifier.verify() Protocol method
    # (resolve_principal, its only caller, is itself synchronous) — these
    # two methods must NOT be coroutine functions, unlike every other
    # PairingStore method.
    import inspect

    assert not inspect.iscoroutinefunction(MemoryPairingStore.put_device_token)
    assert not inspect.iscoroutinefunction(MemoryPairingStore.get_device_token_by_hash)


def test_firestore_pairing_store_constructs_without_importing_sdk():
    # Lazy-import contract, same as FirestoreLiveSessionStore/FirestoreTelemetryStore:
    # constructing must never touch google.cloud.firestore, so a base install
    # (no google-cloud-firestore) can still build the app.
    from watch.pairing_store import FirestorePairingStore

    before = "google.cloud.firestore" in sys.modules
    store = FirestorePairingStore("some-project")
    assert store.project == "some-project"
    assert ("google.cloud.firestore" in sys.modules) == before


def test_get_pairing_store_defaults_to_memory(monkeypatch):
    from watch.pairing_store import MemoryPairingStore as MPS
    from watch.pairing_store import get_pairing_store

    monkeypatch.delenv("MINDSHIFT_FIRESTORE_PROJECT", raising=False)
    assert isinstance(get_pairing_store(), MPS)


def test_get_pairing_store_uses_firestore_when_project_set(monkeypatch):
    from watch.pairing_store import FirestorePairingStore, get_pairing_store

    monkeypatch.setenv("MINDSHIFT_FIRESTORE_PROJECT", "arborfam-hub")
    store = get_pairing_store()
    assert isinstance(store, FirestorePairingStore) and store.project == "arborfam-hub"


# ------------------------------------------------ has_device_tokens_for_account --

@pytest.mark.anyio
async def test_has_device_tokens_for_account_false_when_none_issued():
    store = MemoryPairingStore()
    assert await store.has_device_tokens_for_account("acct1") is False


@pytest.mark.anyio
async def test_has_device_tokens_for_account_true_after_a_token_is_put():
    store = MemoryPairingStore()
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("raw-device-token"),
        account_id="acct1",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    ))
    assert await store.has_device_tokens_for_account("acct1") is True


@pytest.mark.anyio
async def test_has_device_tokens_for_account_is_scoped_per_account():
    store = MemoryPairingStore()
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("raw-device-token"),
        account_id="acct-a",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    ))
    assert await store.has_device_tokens_for_account("acct-a") is True
    assert await store.has_device_tokens_for_account("acct-b") is False


# --------------------------------------------- delete_device_tokens_for_account --

@pytest.mark.anyio
async def test_delete_device_tokens_for_account_returns_zero_when_none_issued():
    store = MemoryPairingStore()
    assert await store.delete_device_tokens_for_account("acct1") == 0


@pytest.mark.anyio
async def test_delete_device_tokens_for_account_deletes_the_token_and_returns_count():
    store = MemoryPairingStore()
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("raw-device-token"),
        account_id="acct1",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    ))
    deleted = await store.delete_device_tokens_for_account("acct1")
    assert deleted == 1
    assert await store.has_device_tokens_for_account("acct1") is False
    assert store.get_device_token_by_hash(hash_secret("raw-device-token")) is None


@pytest.mark.anyio
async def test_delete_device_tokens_for_account_deletes_all_tokens_for_that_account():
    store = MemoryPairingStore()
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("token-1"),
        account_id="acct1",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    ))
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("token-2"),
        account_id="acct1",
        created_at="2026-08-05T10:00:00+00:00",
        pairing_id="pid-2",
    ))
    deleted = await store.delete_device_tokens_for_account("acct1")
    assert deleted == 2
    assert await store.has_device_tokens_for_account("acct1") is False


@pytest.mark.anyio
async def test_delete_device_tokens_for_account_is_scoped_per_account():
    store = MemoryPairingStore()
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("token-a"),
        account_id="acct-a",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-1",
    ))
    store.put_device_token(DeviceToken(
        token_hash=hash_secret("token-b"),
        account_id="acct-b",
        created_at="2026-08-04T10:00:00+00:00",
        pairing_id="pid-2",
    ))
    deleted = await store.delete_device_tokens_for_account("acct-a")
    assert deleted == 1
    assert await store.has_device_tokens_for_account("acct-a") is False
    assert await store.has_device_tokens_for_account("acct-b") is True
