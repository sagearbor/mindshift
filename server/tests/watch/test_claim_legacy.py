# Ported from gauge@2157433 server/tests/test_claim_legacy.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B5): Episode -> LiveSession (store methods, model, field
# names in this file's own helpers); server.main.create_app ->
# watch.testing.create_watch_test_app; GAUGE_ALLOW_LEGACY_ACCOUNT env var ->
# the explicit allow_legacy=True kwarg. Response field names (`episodes_moved`
# on ClaimLegacyResponse, `episodes_moved_total` on the stored LegacyClaim)
# stay UNCHANGED per the locked rename map -- only the "episode" TYPE and its
# store methods move to "live_session".
"""POST /me/claim-legacy — one-shot default->uid history re-key (Wave B Task 2).

Marker-first ordering (server/watch/store.py's update_legacy_claim_atomically,
Task B2/B4): the claim marker is reserved/refreshed atomically BEFORE any
re-key mutation, so a crash mid-sweep is always resumable by the SAME uid
re-claiming (idempotent — already-moved live sessions are naturally skipped
since owner_account no longer equals LEGACY_ACCOUNT_ID). A DIFFERENT uid
racing in is a 409 with no partial mutation on the losing side. Live sessions
are re-keyed (owner_account default->uid); baseline/subscriptions/profile
are copy-if-absent — the legacy docs are left in place so the shipped
watch (still writing to "default") keeps working.
"""
import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from server.tests.watch.test_auth import StubVerifier
from watch.models import EnrollmentBaseline, LiveSession, Participant, SpeakerProfile, VectorSubscription
from watch.store import MemoryLiveSessionStore
from watch.testing import create_watch_test_app

TOKENS = {
    "tok-a": {"sub": "uid-a", "email": "a@example.com"},
    "tok-b": {"sub": "uid-b", "email": "b@example.com"},
}
AUTH_A = {"Authorization": "Bearer tok-a"}
AUTH_B = {"Authorization": "Bearer tok-b"}


def _ls(id, owner, started="2026-08-01T00:00:00Z", shared=()):
    return LiveSession(id=id, owner_account=owner, started_at=started, ended_at=None,
                        status="captured",
                        participants=[Participant(id="self", role="self", speaker_label="You")],
                        vector_events=[], nudge_events=[], shared_with=list(shared))


def _client():
    store = MemoryLiveSessionStore()
    client = TestClient(create_watch_test_app(
        store=store, verifier=StubVerifier(TOKENS), allow_legacy=True,
    ))
    return store, client


def test_claim_moves_default_live_sessions_to_uid():
    store, client = _client()
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    asyncio.run(store.put_live_session(_ls("e2", "default")))
    asyncio.run(store.put_live_session(_ls("e3", "someone-else")))
    resp = client.post("/me/claim-legacy", headers=AUTH_A)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "claimed"
    assert body["episodes_moved"] == 2
    assert body["previously_claimed_at"] is None
    mine = client.get("/live-sessions", headers=AUTH_A).json()
    assert sorted(e["id"] for e in mine) == ["e1", "e2"]
    assert all(e["owner_account"] == "uid-a" for e in mine)
    # the legacy watch now legitimately sees nothing under "default"
    assert client.get("/live-sessions", params={"account": "default"}).json() == []


def test_claim_requires_full_auth():
    # A legacy ?account= principal must never be able to claim (it could
    # claim INTO the string "default" or any impersonated account id).
    store, client = _client()
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    resp = client.post("/me/claim-legacy", params={"account": "uid-a"})
    assert resp.status_code == 401
    assert asyncio.run(store.get_live_session("e1")).owner_account == "default"


def test_reclaim_same_uid_sweeps_new_live_sessions():
    store, client = _client()
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    assert client.post("/me/claim-legacy", headers=AUTH_A).json()["episodes_moved"] == 1
    # the watch (still legacy, X1) writes another live session after the claim
    asyncio.run(store.put_live_session(_ls("e2", "default", started="2026-08-02T00:00:00Z")))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["status"] == "claimed"
    assert body["episodes_moved"] == 1
    assert body["previously_claimed_at"] is not None
    claim = asyncio.run(store.get_legacy_claim())
    assert claim.account_id == "uid-a" and claim.episodes_moved_total == 2


def test_reclaim_same_uid_is_idempotent_at_the_store_seam():
    # Controller fold-in 2: the same-uid-twice test at the store seam
    # (idempotent re-claim by the same uid), not just via the router.
    store, _ = _client()
    from watch.routers.rest import make_claim_mutator
    now1 = "2026-08-01T00:00:00Z"
    now2 = "2026-08-01T00:05:00Z"
    first = asyncio.run(store.update_legacy_claim_atomically(make_claim_mutator("uid-a", now1)))
    assert first.account_id == "uid-a"
    assert first.first_claimed_at == now1 and first.last_claimed_at == now1
    second = asyncio.run(store.update_legacy_claim_atomically(make_claim_mutator("uid-a", now2)))
    assert second.account_id == "uid-a"
    # first_claimed_at is preserved across the re-claim; only last_claimed_at moves.
    assert second.first_claimed_at == now1 and second.last_claimed_at == now2


def test_claim_by_second_uid_is_409():
    store, client = _client()
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    assert client.post("/me/claim-legacy", headers=AUTH_A).status_code == 200
    asyncio.run(store.put_live_session(_ls("e2", "default")))
    resp = client.post("/me/claim-legacy", headers=AUTH_B)
    assert resp.status_code == 409
    assert "different account" in resp.json()["detail"]
    assert asyncio.run(store.get_live_session("e2")).owner_account == "default"  # untouched


def test_nothing_to_claim_is_honest_and_writes_no_marker():
    store, client = _client()
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body == {"status": "nothing_to_claim", "episodes_moved": 0,
                    "baseline_copied": False, "subscriptions_copied": False,
                    "speaker_profile_copied": False, "previously_claimed_at": None}
    # no marker: uid-b (the real watch owner, say) can still claim later
    assert asyncio.run(store.get_legacy_claim()) is None
    asyncio.run(store.put_live_session(_ls("e1", "default")))
    assert client.post("/me/claim-legacy", headers=AUTH_B).status_code == 200


def test_claim_copies_baseline_only_when_uid_has_none_and_keeps_defaults():
    store, client = _client()
    legacy = EnrollmentBaseline(account_id="default", rms_db=-31.5, f0_median=148.0, updated_at="t")
    asyncio.run(store.put_baseline(legacy))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["status"] == "claimed" and body["baseline_copied"] is True
    copied = asyncio.run(store.get_baseline("uid-a"))
    assert copied.rms_db == -31.5 and copied.account_id == "uid-a"
    # D2: the watch still reads the default baseline — it MUST survive
    assert asyncio.run(store.get_baseline("default")) is not None
    # and an existing uid baseline is never clobbered on re-claim
    mine = EnrollmentBaseline(account_id="uid-a", rms_db=-25.0, f0_median=200.0, updated_at="t2")
    asyncio.run(store.put_baseline(mine))
    asyncio.run(store.put_live_session(_ls("e9", "default")))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["baseline_copied"] is False
    assert asyncio.run(store.get_baseline("uid-a")).rms_db == -25.0


def _profile(account_id, dim=4):
    return SpeakerProfile(account_id=account_id, version=2, embedding=[0.1] * dim, dim=dim,
                          enroll_count=1, model="test-model", created_at="t", updated_at="t")


def test_claim_copies_speaker_profile_only_when_uid_has_none_and_keeps_defaults():
    # Review fix-round 1, finding 2: the symmetric no-clobber test for
    # SpeakerProfile — the wife's own voiceprint must be provably safe from
    # a claim by a different account that shares the legacy "default" login.
    store, client = _client()
    legacy = _profile("default")
    asyncio.run(store.put_speaker_profile(legacy))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["status"] == "claimed" and body["speaker_profile_copied"] is True
    copied = asyncio.run(store.get_speaker_profile("uid-a"))
    assert copied.account_id == "uid-a" and copied.embedding == legacy.embedding
    # D2: the watch still reads the default profile — it MUST survive
    assert asyncio.run(store.get_speaker_profile("default")) is not None
    # and an existing uid profile (the wife's own enrolled voiceprint) is
    # NEVER clobbered by a re-claim, even when there's new history to sweep
    mine = _profile("uid-a", dim=8)
    asyncio.run(store.put_speaker_profile(mine))
    asyncio.run(store.put_live_session(_ls("e9", "default")))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["speaker_profile_copied"] is False
    untouched = asyncio.run(store.get_speaker_profile("uid-a"))
    assert untouched.dim == 8 and untouched.embedding == mine.embedding


def test_crash_between_live_session_move_and_total_bump_undercounts_by_at_most_one(monkeypatch):
    # Review fix-round 1, finding 1, UPDATED for the overnight server-
    # hardening round's `store.claim_legacy_live_session` primitive (see
    # watch/routers/rest.py's claim_legacy Phase 2 comment): the move-then-bump
    # for ONE live session is now a single store call, internally doing
    # "write the live session, then write the claim total" back to back inside
    # one lock-held critical section (server/watch/store.py's MemoryLiveSessionStore.
    # claim_legacy_live_session). This test simulates a crash landing exactly
    # between those two internal writes by monkeypatching
    # `store._read_claim_locked` (the seam `claim_legacy_live_session` calls
    # AFTER its live-session-dict write but BEFORE its claim-dict write) to
    # raise on a specific call: the live session's move survives (never lost —
    # live sessions are the thing that actually matters), but that one live
    # session's count is permanently lost. Same bounded, honestly-documented
    # residual as before this round — this round's fix closed the CONCURRENCY
    # race (two same-uid requests double-counting), not this single-process,
    # mid-write crash scenario, which remains an accepted trade for
    # MemoryLiveSessionStore (FirestoreLiveSessionStore's version of this SAME
    # scenario is now actually atomic — see claim_legacy_live_session's
    # Firestore docstring — a strict improvement obtained as a side effect,
    # not tested here since this suite never touches real Firestore).
    store, client = _client()
    # Sorted newest-first by list_live_sessions, so processing order is
    # first, second, third exactly as named.
    asyncio.run(store.put_live_session(_ls("e-first", "default", started="2026-08-03T00:00:00Z")))
    asyncio.run(store.put_live_session(_ls("e-second", "default", started="2026-08-02T00:00:00Z")))
    asyncio.run(store.put_live_session(_ls("e-third", "default", started="2026-08-01T00:00:00Z")))

    orig = store._read_claim_locked
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        # call 1 = Phase-1 marker reservation's own read (must succeed so
        # the marker is set); call 2 = e-first's claim_legacy_live_session
        # internal read (must succeed -- e-first fully moves+counts); call 3
        # = e-second's claim_legacy_live_session internal read -- simulate the
        # crash HERE: e-second's live-session-dict write has already happened
        # (it happens before this read inside claim_legacy_live_session), but
        # its claim-total write has not.
        if calls["n"] == 3:
            raise RuntimeError("simulated crash between move and total-bump")
        return orig()

    monkeypatch.setattr(store, "_read_claim_locked", flaky)

    with pytest.raises(RuntimeError):
        client.post("/me/claim-legacy", headers=AUTH_A)

    # e-first and e-second both moved (their live-session-dict writes committed
    # before the crash); e-third was never reached (the loop halted).
    assert asyncio.run(store.get_live_session("e-first")).owner_account == "uid-a"
    assert asyncio.run(store.get_live_session("e-second")).owner_account == "uid-a"
    assert asyncio.run(store.get_live_session("e-third")).owner_account == "default"
    # Only e-first's move was ever counted -- e-second's count is lost.
    claim = asyncio.run(store.get_legacy_claim())
    assert claim.episodes_moved_total == 1

    # Retry (same uid, marker already reserved): sweeps only e-third (the
    # only one still LEGACY_ACCOUNT_ID-owned) and counts it correctly.
    monkeypatch.setattr(store, "_read_claim_locked", orig)
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["status"] == "claimed" and body["episodes_moved"] == 1
    assert asyncio.run(store.get_live_session("e-third")).owner_account == "uid-a"

    # Permanent undercount of exactly 1: three live sessions were actually
    # moved (e-first, e-second, e-third), but the audited total only ever
    # reaches 2 -- e-second's count can never be recovered. Bounded, never
    # zero, never fabricated, never more than the true number moved.
    final_claim = asyncio.run(store.get_legacy_claim())
    assert final_claim.episodes_moved_total == 2


def test_claim_copies_saved_subscriptions_not_fabricated_defaults():
    store, client = _client()
    asyncio.run(store.put_subscriptions("default", [
        VectorSubscription(vector="yelling", sensitivity=0.5, haptics=False, channel="A")]))
    body = client.post("/me/claim-legacy", headers=AUTH_A).json()
    assert body["subscriptions_copied"] is True
    mine = asyncio.run(store.get_subscriptions("uid-a"))
    assert [s.vector for s in mine] == ["yelling"] and mine[0].sensitivity == 0.5


def test_claim_drops_self_share_artifact():
    store, client = _client()
    asyncio.run(store.put_live_session(_ls("e1", "default", shared=["uid-a", "uid-x"])))
    client.post("/me/claim-legacy", headers=AUTH_A)
    ls = asyncio.run(store.get_live_session("e1"))
    assert ls.owner_account == "uid-a" and ls.shared_with == ["uid-x"]


# --- Overnight server-hardening batch, item 1 (final-review Important):
# same-uid concurrent double-submission can't double-bump episodes_moved_total ---
#
# Two real OS threads (same pattern as test_store.py's
# test_concurrent_join_same_code_never_double_accepts / _SlowClaimStore),
# each running its OWN TestClient (its own independent asyncio event loop)
# against the SAME shared store, so a `time.sleep` widening a critical
# section genuinely stalls only that thread's progress -- the other
# thread's independent loop keeps running, guaranteeing real wall-clock
# overlap instead of relying on cooperative-yield luck within one loop.
#
# This reproduces the exact bug the fix (watch/routers/rest.py's claim_legacy
# Phase 2 comment; server/watch/store.py's claim_legacy_live_session) closes:
# without it, two same-uid requests racing in (e.g. a UI double-tap, or a
# client retry-on-timeout that actually landed) could each capture the same
# pre-sweep live-session snapshot and both move+count every live session in
# it, double-bumping the audited total even though every live session is
# only ever actually re-owned once.

class _SlowClaimStore(MemoryLiveSessionStore):
    """Widens update_legacy_claim_atomically's (and claim_legacy_live_session's)
    locked read with a real time.sleep, forcing genuine thread overlap --
    mirrors test_store.py's _SlowClaimStore exactly."""
    def _read_claim_locked(self):
        value = super()._read_claim_locked()
        time.sleep(0.05)
        return value


def _post_claim_legacy_in_thread(app, headers, results, index):
    def target():
        # A fresh TestClient per thread -- its own independent event loop,
        # not shared with the other thread's -- is what makes the other
        # thread's `time.sleep` inside the store an effective (not merely
        # cosmetic) concurrency-forcing tool; see this section's docstring.
        client = TestClient(app)
        results[index] = client.post("/me/claim-legacy", headers=AUTH_A)
    return threading.Thread(target=target)


def test_concurrent_same_uid_claims_count_each_live_session_once():
    store = _SlowClaimStore()
    # 3 legacy live sessions: big enough to exercise a multi-live-session
    # sweep (the scenario a naive "just re-read the list once after Phase 1"
    # fix was verified, empirically, to still get wrong -- see
    # watch/routers/rest.py's comment), small enough to keep the test fast.
    for i, ls_id in enumerate(["e0", "e1", "e2"]):
        asyncio.run(store.put_live_session(_ls(ls_id, "default", started=f"2026-08-0{i+1}T00:00:00Z")))
    app = create_watch_test_app(store=store, verifier=StubVerifier(TOKENS), allow_legacy=True)

    results = [None, None]
    t1 = _post_claim_legacy_in_thread(app, AUTH_A, results, 0)
    t2 = _post_claim_legacy_in_thread(app, AUTH_A, results, 1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both requests succeed (same uid re-claiming is always idempotent, per
    # test_reclaim_same_uid_is_idempotent_at_the_store_seam) -- neither is a
    # 409, since a same-uid marker refresh never conflicts with itself.
    for r in results:
        assert r.status_code == 200
        assert r.json()["status"] == "claimed"

    # Every live session ends up owned by uid-a exactly once...
    for ls_id in ["e0", "e1", "e2"]:
        assert asyncio.run(store.get_live_session(ls_id)).owner_account == "uid-a"

    # ...and the audited total reflects exactly 3 real moves -- NOT 6 (which
    # is what double-counting would produce if both racing requests each
    # counted all 3 live sessions). Whichever request actually performed a
    # given live session's move counts it; the other's attempt on that SAME
    # live session is a no-op (0 contribution), split however the real thread
    # scheduling happened to land between the two responses.
    claim = asyncio.run(store.get_legacy_claim())
    assert claim.episodes_moved_total == 3
    assert claim.account_id == "uid-a"
    assert results[0].json()["episodes_moved"] + results[1].json()["episodes_moved"] == 3
