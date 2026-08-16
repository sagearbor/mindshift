# Ported from gauge@2157433 server/tests/test_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# Task B6 CLOSE-OUT: the four `test_concurrent_*` tests below (group-
# atomicity races) originally imported their mutators from
# `server.groups_api` (`make_join_mutator`, `make_invite_mutator`,
# `make_leave_mutator`). At B2 time that router didn't exist yet, so the
# three mutator builders were inlined here as a behaviorally equivalent local
# port (see git history for that version). Now that `watch/routers/groups.py`
# has landed, these tests import the REAL mutators from it directly —
# confirmed byte-for-byte identical control flow/messages to the copies they
# replace (same invariant checks, same HTTPException status codes/details,
# same field mutations), so no test assertion below needed to change.
import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException

from watch.routers.groups import make_invite_mutator, make_join_mutator, make_leave_mutator
from watch.store import MAX_FIRESTORE_PCM_B64, MemoryLiveSessionStore, live_session_to_doc
from watch.models import (
    LiveSession, Participant, ConsentRecord, VectorEvent, LegacyClaim,
    Group, GroupInvite, GroupMember,
)


def _ls(id, owner, started, shared=()):
    return LiveSession(id=id, owner_account=owner, started_at=started, ended_at=None, status="captured",
                        participants=[Participant(id="p", role="self", speaker_label="You")],
                        vector_events=[], nudge_events=[], shared_with=list(shared))


def test_list_owner_and_shared():
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_live_session(_ls("e1", "alice", "2026-07-30T00:00:00Z")))
    asyncio.run(s.put_live_session(_ls("e2", "bob", "2026-07-31T00:00:00Z", shared=("alice",))))
    asyncio.run(s.put_live_session(_ls("e3", "carol", "2026-07-29T00:00:00Z")))
    got = asyncio.run(s.list_live_sessions("alice"))
    assert [e.id for e in got] == ["e2", "e1"]


def test_default_subscriptions():
    s = MemoryLiveSessionStore()
    subs = asyncio.run(s.get_subscriptions("alice"))
    assert {x.vector for x in subs} == {"yelling", "aggressive_tone", "interrupting", "airtime", "hr_spike"}
    assert next(x for x in subs if x.vector == "hr_spike").channel == "B"


def test_live_session_to_doc_carries_pcm_b64_but_wire_still_excludes_it():
    # pcm_b64 must survive real (Firestore-shaped) persistence so Task 8 can
    # read raw audio back out of storage, but it must never appear on the
    # wire (model_dump_json) — those are two different serialization paths.
    ls = _ls("e1", "alice", "2026-07-30T00:00:00Z")
    ls.pcm_b64 = "c29tZS1hdWRpby1ieXRlcw=="

    doc = live_session_to_doc(ls)
    assert doc["pcm_b64"] == "c29tZS1hdWRpby1ieXRlcw=="

    # And the doc round-trips cleanly back through the model.
    restored = LiveSession(**doc)
    assert restored.pcm_b64 == ls.pcm_b64


def test_live_session_to_doc_drops_oversized_pcm_b64_but_keeps_everything_else(caplog):
    # Final-review Finding 1b: a doc whose pcm_b64 alone would risk blowing
    # Firestore's 1MiB per-document limit must still persist — just without
    # the raw audio — rather than the whole write raising InvalidArgument and
    # losing the live session entirely.
    ls = _ls("big-ep", "alice", "2026-07-30T00:00:00Z")
    ls.pcm_b64 = "A" * (MAX_FIRESTORE_PCM_B64 + 1)
    ls.vector_events = [VectorEvent(vector="yelling", level=2, t=1.0, value=10.0)]
    ls.series = {"rms_db": [-20.0, -18.0]}

    with caplog.at_level("WARNING"):
        doc = live_session_to_doc(ls)

    assert doc["pcm_b64"] == ""
    # Everything else survives untouched.
    assert doc["id"] == "big-ep"
    assert doc["owner_account"] == "alice"
    assert doc["vector_events"] == [
        {"vector": "yelling", "level": 2, "t": 1.0, "value": 10.0, "detail": "", "participant_id": None}
    ]
    assert doc["series"] == {"rms_db": [-20.0, -18.0]}

    # And it round-trips cleanly back through the model with empty audio.
    restored = LiveSession(**doc)
    assert restored.pcm_b64 == ""
    assert restored.vector_events == ls.vector_events

    assert any(
        "big-ep" in rec.message and str(MAX_FIRESTORE_PCM_B64 + 1) in rec.message
        for rec in caplog.records
    )


def test_live_session_to_doc_keeps_pcm_b64_at_or_under_the_cap():
    # Boundary case: exactly at the cap is still "small enough" — included.
    ls = _ls("small-ep", "alice", "2026-07-30T00:00:00Z")
    ls.pcm_b64 = "A" * MAX_FIRESTORE_PCM_B64

    doc = live_session_to_doc(ls)

    assert doc["pcm_b64"] == ls.pcm_b64

    assert "pcm_b64" not in json.loads(ls.model_dump_json())


def test_get_missing_live_session_returns_none():
    s = MemoryLiveSessionStore()
    result = asyncio.run(s.get_live_session("nonexistent"))
    assert result is None


def test_roundtrip_isolation():
    """Test that put/get isolate the stored state from mutations on returned/passed objects."""
    s = MemoryLiveSessionStore()

    # (a) Put a live session, get it back, assert full equality
    orig_ls = _ls("e1", "alice", "2026-07-30T00:00:00Z")
    asyncio.run(s.put_live_session(orig_ls))
    retrieved = asyncio.run(s.get_live_session("e1"))
    assert retrieved == orig_ls
    assert retrieved.id == "e1"
    assert retrieved.owner_account == "alice"

    # (b) Mutate the returned object (append a VectorEvent) and assert fresh get_live_session doesn't see it
    event = VectorEvent(vector="yelling", level=2, t=1.5, value=0.8)
    retrieved.vector_events.append(event)
    assert len(retrieved.vector_events) == 1

    # Fresh get should not see the mutation
    fresh_get = asyncio.run(s.get_live_session("e1"))
    assert len(fresh_get.vector_events) == 0
    assert fresh_get != retrieved

    # (c) Mutate the originally-passed-in object after put and assert store is unaffected
    orig_ls.participants[0].display_name = "Modified"
    stored_via_get = asyncio.run(s.get_live_session("e1"))
    assert stored_via_get.participants[0].display_name is None
    assert stored_via_get != orig_ls


def test_account_roundtrip():
    from watch.models import Account
    s = MemoryLiveSessionStore()
    a = Account(id="uid-1", email="a@example.com", display_name="A",
                created_at="2026-08-02T00:00:00+00:00", updated_at="2026-08-02T00:00:00+00:00")
    asyncio.run(s.put_account(a))
    got = asyncio.run(s.get_account("uid-1"))
    assert got == a
    assert asyncio.run(s.get_account("nope")) is None


def test_account_lookup_by_email():
    from watch.models import Account
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_account(Account(id="uid-1", email="a@example.com",
                                       created_at="t", updated_at="t")))
    asyncio.run(s.put_account(Account(id="uid-2", email="b@example.com",
                                       created_at="t", updated_at="t")))
    assert asyncio.run(s.get_account_by_email("b@example.com")).id == "uid-2"
    assert asyncio.run(s.get_account_by_email("nobody@example.com")) is None
    assert asyncio.run(s.get_account_by_email("")) is None


def test_account_store_copies_out():
    from watch.models import Account
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_account(Account(id="uid-1", created_at="t", updated_at="t")))
    got = asyncio.run(s.get_account("uid-1"))
    got.display_name = "mutated"
    assert asyncio.run(s.get_account("uid-1")).display_name is None


def test_group_roundtrip_and_doc_denormalizes_members():
    from watch.models import Group, GroupMember
    from watch.store import group_to_doc
    g = Group(id="g1", kind="pair", name="Us", created_by="uid-1", created_at="t",
              members=[GroupMember(account_id="uid-1", joined_at="t"),
                       GroupMember(account_id="uid-2", joined_at="t")])
    doc = group_to_doc(g)
    assert doc["member_account_ids"] == ["uid-1", "uid-2"]
    assert Group(**doc) == g            # the extra key round-trips harmlessly

    s = MemoryLiveSessionStore()
    asyncio.run(s.put_group(g))
    assert asyncio.run(s.get_group("g1")) == g
    assert asyncio.run(s.get_group("nope")) is None


def test_list_groups_by_membership_only():
    from watch.models import Group, GroupMember
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_group(Group(id="g1", kind="pair", created_by="uid-1", created_at="2026-08-01",
                                   members=[GroupMember(account_id="uid-1", joined_at="t")])))
    asyncio.run(s.put_group(Group(id="g2", kind="team", created_by="uid-2", created_at="2026-08-02",
                                   members=[GroupMember(account_id="uid-2", joined_at="t"),
                                            GroupMember(account_id="uid-1", joined_at="t")])))
    asyncio.run(s.put_group(Group(id="g3", kind="pair", created_by="uid-3", created_at="2026-08-03",
                                   members=[GroupMember(account_id="uid-3", joined_at="t")])))
    assert [g.id for g in asyncio.run(s.list_groups("uid-1"))] == ["g1", "g2"]
    assert asyncio.run(s.list_groups("nobody")) == []


def test_group_lookup_by_invite_code():
    from watch.models import Group, GroupInvite
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_group(Group(id="g1", kind="pair", created_by="uid-1", created_at="t",
                                   invites=[GroupInvite(code="abc123", invited_by="uid-1", created_at="t")])))
    assert asyncio.run(s.get_group_by_invite_code("abc123")).id == "g1"
    assert asyncio.run(s.get_group_by_invite_code("nope")) is None


def test_mutual_visibility_is_a_valid_consent_kind():
    rec = ConsentRecord(id="c1", participant_id="uid-1", kind="mutual_visibility",
                         attested_by="uid-1", confirmed=True, ts="t")
    assert rec.kind == "mutual_visibility" and rec.confirmed is True


def test_group_store_copies_out():
    from watch.models import Group
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_group(Group(id="g1", kind="pair", created_by="uid-1", created_at="t")))
    got = asyncio.run(s.get_group("g1"))
    got.name = "mutated"
    assert asyncio.run(s.get_group("g1")).name == ""


# --- update_group_atomically: the TOCTOU-race fix (Task 11 review, fix round 1) ---
#
# These pin the atomicity seam that watch/routers/groups.py's (Task B6)
# invite-mint and join handlers run their invariant checks inside.
# `_SlowMemoryStore` widens the read half of update_group_atomically's
# critical section with a real `time.sleep` so two genuine OS threads (each
# running their own asyncio loop against the SAME shared store instance,
# mirroring two concurrent requests hitting one process) reliably overlap in
# wall-clock time. Without `MemoryLiveSessionStore._groups_lock` actually
# serializing the two callers, both threads would read the same
# pre-mutation state during the sleep and both would pass their invariant
# checks -- reproducing exactly the bug the fix closes. With the lock, the
# second thread's read cannot start until the first thread's write has fully
# committed, so it correctly sees the already-mutated state and rejects.

class _SlowMemoryStore(MemoryLiveSessionStore):
    def _read_group_locked(self, group_id):
        result = super()._read_group_locked(group_id)
        time.sleep(0.05)
        return result


def _threaded_update(store, group_id, mutator, results, index):
    def target():
        try:
            group = asyncio.run(store.update_group_atomically(group_id, mutator))
            results[index] = ("ok", group)
        except HTTPException as exc:
            results[index] = ("error", exc)
    return threading.Thread(target=target)


def test_concurrent_join_same_code_never_double_accepts():
    store = _SlowMemoryStore()
    asyncio.run(store.put_group(Group(
        id="g1", kind="pair", created_by="alice", created_at="t",
        members=[GroupMember(account_id="alice", joined_at="t")],
        invites=[GroupInvite(code="samecode", invited_by="alice", created_at="t")],
    )))

    results = [None, None]
    t1 = _threaded_update(store, "g1", make_join_mutator("bob", "samecode"), results, 0)
    t2 = _threaded_update(store, "g1", make_join_mutator("carol", "samecode"), results, 1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    outcomes = sorted(r[0] for r in results)
    assert outcomes == ["error", "ok"]      # exactly one winner, one loser -- never both
    error = next(r[1] for r in results if r[0] == "error")
    assert error.status_code == 409 and error.detail == "invite already accepted"

    final = asyncio.run(store.get_group("g1"))
    assert len(final.members) == 2          # never 3 -- the code was consumed exactly once
    winner = next(m.account_id for m in final.members if m.account_id != "alice")
    assert winner in {"bob", "carol"}
    # Exactly one mutual_visibility consent was appended -- the winner's,
    # not both (no phantom consent for the joiner who was rejected).
    assert [c.participant_id for c in final.consents] == [winner]


def test_concurrent_join_two_codes_enforces_pair_max():
    store = _SlowMemoryStore()
    asyncio.run(store.put_group(Group(
        id="g1", kind="pair", created_by="alice", created_at="t",
        members=[GroupMember(account_id="alice", joined_at="t")],
        invites=[GroupInvite(code="code-b", invited_by="alice", created_at="t"),
                 GroupInvite(code="code-c", invited_by="alice", created_at="t")],
    )))

    results = [None, None]
    t1 = _threaded_update(store, "g1", make_join_mutator("bob", "code-b"), results, 0)
    t2 = _threaded_update(store, "g1", make_join_mutator("carol", "code-c"), results, 1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    outcomes = sorted(r[0] for r in results)
    assert outcomes == ["error", "ok"]      # only one of the two codes wins the last slot
    error = next(r[1] for r in results if r[0] == "error")
    assert error.status_code == 409 and error.detail == "this pair is already full"

    final = asyncio.run(store.get_group("g1"))
    assert len(final.members) == 2          # never 3 -- PAIR_MAX_MEMBERS held under contention


def test_concurrent_invite_mint_races_a_join_without_corrupting_the_pair():
    """invite-mint and join go through the SAME `update_group_atomically` seam
    (same lock/transaction), so a mint racing a join on the very last slot can
    never observe a torn state: the mint's PAIR_MAX check always runs against
    a read that is either strictly before or strictly after the join's commit,
    never a half-applied one. Whichever order wins, membership must land at
    exactly 2 (never 3) and the invite list must exactly reflect the mint's
    own reported outcome."""
    store = _SlowMemoryStore()
    asyncio.run(store.put_group(Group(
        id="g1", kind="pair", created_by="alice", created_at="t",
        members=[GroupMember(account_id="alice", joined_at="t")],
        invites=[GroupInvite(code="existing", invited_by="alice", created_at="t")],
    )))

    results = [None, None]
    t1 = _threaded_update(store, "g1", make_join_mutator("bob", "existing"), results, 0)
    t2 = _threaded_update(store, "g1", make_invite_mutator("alice", None), results, 1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    join_outcome, mint_outcome = results
    assert join_outcome[0] == "ok"   # nothing about invite-mint can block the pre-existing join
    final = asyncio.run(store.get_group("g1"))
    assert len(final.members) == 2   # alice + bob -- never 3, whichever ran first

    if mint_outcome[0] == "ok":
        # The mint's atomic read landed before the join's commit (still 1
        # member at that instant) -- exactly one new invite, no corruption.
        assert len(final.invites) == 2
    else:
        # The mint's atomic read landed after the join's commit (already 2
        # members) -- correctly rejected, no phantom invite minted.
        assert mint_outcome[1].status_code == 409
        assert mint_outcome[1].detail == "this pair is already full"
        assert len(final.invites) == 1


def test_concurrent_leave_races_a_join_without_losing_either_write():
    """M1 fold-in: leave() now goes through update_group_atomically (same
    seam as invite/join) instead of a plain read-then-``put_group``. Before
    the fix, leave's write was built from a read taken before the race and
    could silently clobber a concurrent join's write (or vice versa) --
    whichever committed last would win with a STALE members list, losing the
    other thread's change entirely. With both going through the same lock,
    one write always happens-before the other's read, so both changes land."""
    store = _SlowMemoryStore()
    asyncio.run(store.put_group(Group(
        id="g1", kind="team", created_by="alice", created_at="t",
        members=[GroupMember(account_id="alice", joined_at="t"),
                 GroupMember(account_id="bob", joined_at="t")],
        invites=[GroupInvite(code="for-carol", invited_by="alice", created_at="t")],
    )))

    results = [None, None]
    t1 = _threaded_update(store, "g1", make_leave_mutator("bob"), results, 0)
    t2 = _threaded_update(store, "g1", make_join_mutator("carol", "for-carol"), results, 1)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert [r[0] for r in results] == ["ok", "ok"]  # neither op fails; they aren't in conflict

    final = asyncio.run(store.get_group("g1"))
    # Both changes must have landed: bob is gone AND carol is in -- if leave
    # had used a stale pre-race read, one of these would be silently lost.
    assert sorted(m.account_id for m in final.members) == ["alice", "carol"]


def test_capture_roundtrip_and_newest_first():
    from watch.models import Capture
    s = MemoryLiveSessionStore()
    for cid, when in (("c1", "2026-08-01T00:00:00Z"), ("c2", "2026-08-03T00:00:00Z"),
                       ("c3", "2026-08-02T00:00:00Z")):
        asyncio.run(s.put_capture(Capture(id=cid, account_id="alice", captured_at=when,
                                           received_at=when, duration_s=180.0)))
    asyncio.run(s.put_capture(Capture(id="other", account_id="bob", captured_at="2026-08-09T00:00:00Z",
                                       received_at="t", duration_s=10.0)))
    assert [c.id for c in asyncio.run(s.list_captures("alice"))] == ["c2", "c3", "c1"]
    assert asyncio.run(s.get_capture("c1")).account_id == "alice"
    assert asyncio.run(s.get_capture("nope")) is None
    assert asyncio.run(s.list_captures("nobody")) == []


def test_capture_store_copies_out():
    from watch.models import Capture
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_capture(Capture(id="c1", account_id="alice", captured_at="t",
                                       received_at="t", duration_s=1.0)))
    got = asyncio.run(s.get_capture("c1"))
    got.labels = {"mutated": True}
    assert asyncio.run(s.get_capture("c1")).labels == {}


def test_delete_capture_removes_and_is_idempotent():
    from watch.models import Capture
    s = MemoryLiveSessionStore()
    asyncio.run(s.put_capture(Capture(id="c1", account_id="alice", captured_at="t",
                                       received_at="t", duration_s=1.0)))
    asyncio.run(s.delete_capture("c1"))
    assert asyncio.run(s.get_capture("c1")) is None
    asyncio.run(s.delete_capture("c1"))               # absent -> no-op, no raise
    asyncio.run(s.delete_capture("never-existed"))


def test_capture_is_a_valid_consent_kind():
    rec = ConsentRecord(id="c1", participant_id="self", kind="capture",
                         attested_by="alice", confirmed=False, ts="t")
    assert rec.kind == "capture"


def test_legacy_claim_roundtrip_and_absent_is_none():
    store = MemoryLiveSessionStore()
    assert asyncio.run(store.get_legacy_claim()) is None
    claim = LegacyClaim(account_id="uid-1", first_claimed_at="t1", last_claimed_at="t1")
    out = asyncio.run(store.update_legacy_claim_atomically(lambda c: claim))
    assert out.account_id == "uid-1"
    got = asyncio.run(store.get_legacy_claim())
    assert got == claim
    got.episodes_moved_total = 99          # copies out — mutating the returned
    assert asyncio.run(store.get_legacy_claim()).episodes_moved_total == 0  # object never leaks in


def test_legacy_claim_mutator_raise_persists_nothing():
    store = MemoryLiveSessionStore()

    def boom(current):
        raise HTTPException(status_code=409, detail="claimed by someone else")

    with pytest.raises(HTTPException):
        asyncio.run(store.update_legacy_claim_atomically(boom))
    assert asyncio.run(store.get_legacy_claim()) is None


def test_concurrent_legacy_claims_serialize_one_wins():
    # Same two-thread harness as the group-concurrency tests: a store whose
    # locked read is slowed so both threads WOULD interleave without the lock.
    class _SlowClaimStore(MemoryLiveSessionStore):
        def _read_claim_locked(self):
            value = super()._read_claim_locked()
            time.sleep(0.05)
            return value

    store = _SlowClaimStore()
    results = [None, None]

    def make_mutator(uid):
        def mutate(current):
            if current is not None and current.account_id != uid:
                raise HTTPException(status_code=409, detail="already claimed")
            return LegacyClaim(account_id=uid, first_claimed_at="t", last_claimed_at="t")
        return mutate

    def run(index, uid):
        def target():
            try:
                asyncio.run(store.update_legacy_claim_atomically(make_mutator(uid)))
                results[index] = "ok"
            except HTTPException:
                results[index] = "409"
        return threading.Thread(target=target)

    threads = [run(0, "uid-a"), run(1, "uid-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["409", "ok"]
    assert asyncio.run(store.get_legacy_claim()).account_id in {"uid-a", "uid-b"}


def test_has_subscriptions_only_true_after_save():
    store = MemoryLiveSessionStore()
    assert not asyncio.run(store.has_subscriptions("a"))
    # A read fabricates defaults but must NOT count as a save (D7 purity fix).
    subs = asyncio.run(store.get_subscriptions("a"))
    assert len(subs) == 5
    assert not asyncio.run(store.has_subscriptions("a"))
    asyncio.run(store.put_subscriptions("a", subs))
    assert asyncio.run(store.has_subscriptions("a"))


def test_delete_live_session_removes_and_is_idempotent():
    store = MemoryLiveSessionStore()
    asyncio.run(store.put_live_session(_ls("e1", "a", "2026-08-01T00:00:00Z")))
    asyncio.run(store.delete_live_session("e1"))
    assert asyncio.run(store.get_live_session("e1")) is None
    asyncio.run(store.delete_live_session("e1"))          # absent -> no-op, no raise
    asyncio.run(store.delete_live_session("never-existed"))
