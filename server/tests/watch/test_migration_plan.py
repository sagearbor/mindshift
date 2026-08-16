"""Tests for scripts/migrate_episodes_to_live_sessions.py's pure decision
logic (Task D1, docs/plans/2026-08-15-phase2-one-backend-in-production.md).

Firestore is NEVER touched here — plan_migration/plan_verify take plain
dicts/sets and return a plan; the script wires that pure logic to real
Firestore I/O (untested by design — see the task brief's "NEVER run against
real Firestore in tests"). Exercised with plain dicts, the "memory pattern"
used elsewhere in this repo (e.g. MemoryLiveSessionStore) for I/O-free tests.
"""
from migrate_episodes_to_live_sessions import plan_migration, plan_verify


# ---- plan_migration -------------------------------------------------------

def test_plan_migration_all_new_when_target_empty():
    source_docs = {"a": {"x": 1}, "b": {"x": 2}}
    plan = plan_migration(source_docs, target_ids=set())
    assert plan.to_copy == {"a": {"x": 1}, "b": {"x": 2}}
    assert plan.to_skip == []


def test_plan_migration_skips_ids_already_in_target():
    source_docs = {"a": {"x": 1}, "b": {"x": 2}, "c": {"x": 3}}
    plan = plan_migration(source_docs, target_ids={"b"})
    assert plan.to_copy == {"a": {"x": 1}, "c": {"x": 3}}
    assert plan.to_skip == ["b"]


def test_plan_migration_never_overwrites_skip_never_carries_doc_content():
    """Idempotency contract: a skipped id's source doc content must never
    surface in to_copy — the target's existing doc is left untouched,
    whatever it currently contains."""
    source_docs = {"a": {"stale": True}}
    plan = plan_migration(source_docs, target_ids={"a"})
    assert plan.to_copy == {}
    assert plan.to_skip == ["a"]


def test_plan_migration_all_skipped_when_all_present():
    source_docs = {"a": {}, "b": {}}
    plan = plan_migration(source_docs, target_ids={"a", "b", "z"})
    assert plan.to_copy == {}
    assert sorted(plan.to_skip) == ["a", "b"]


def test_plan_migration_empty_source():
    plan = plan_migration({}, target_ids={"a"})
    assert plan.to_copy == {}
    assert plan.to_skip == []


def test_plan_migration_preserves_source_encounter_order():
    source_docs = {"z": {}, "a": {}, "m": {}}
    plan = plan_migration(source_docs, target_ids=set())
    assert list(plan.to_copy.keys()) == ["z", "a", "m"]


# ---- plan_verify ------------------------------------------------------

def test_plan_verify_all_present_reports_no_missing():
    result = plan_verify(source_ids=["a", "b"], target_ids={"a", "b"})
    assert result.missing_from_target == []
    assert result.target_only == []


def test_plan_verify_reports_missing_source_ids():
    result = plan_verify(source_ids=["a", "b", "c"], target_ids={"a"})
    assert result.missing_from_target == ["b", "c"]


def test_plan_verify_target_only_ids_are_reported_not_treated_as_error():
    """target-only ids (docs the unified service already wrote) are info,
    never an error condition — the caller must not fail verify for these."""
    result = plan_verify(source_ids=["a"], target_ids={"a", "extra-from-app"})
    assert result.missing_from_target == []
    assert result.target_only == ["extra-from-app"]


def test_plan_verify_empty_source_is_ok():
    result = plan_verify(source_ids=[], target_ids={"a"})
    assert result.missing_from_target == []
    assert result.target_only == ["a"]


def test_plan_verify_preserves_source_order_for_missing():
    result = plan_verify(source_ids=["c", "a", "b"], target_ids=set())
    assert result.missing_from_target == ["c", "a", "b"]
