#!/usr/bin/env python3
"""One-time Firestore data migration: `episodes` -> `live_sessions`.

Phase 1 (docs/plans/2026-08-15-phase1-one-repo-one-engine.md) renamed the
Firestore collection `episodes` -> `live_sessions` in code. Production
Firestore (project arborfam-hub, written by the still-live gauge-api) still
has real documents sitting in the OLD `episodes` collection. This script
copies them into `live_sessions`, PRESERVING document ids, before the
unified service goes live and starts reading/writing `live_sessions`.

Dataset is tiny (personal testing — dozens of docs, not thousands), so this
intentionally does no pagination cleverness: `.stream()` reads the whole
collection into memory once per run.

IDEMPOTENT: a source id that already exists in `live_sessions` is SKIPPED,
never overwritten — by the time this runs at deploy time (Task D3) the
unified service may already have written new `live_sessions` docs, and this
script must never clobber those.

Default mode is DRY-RUN: prints the per-doc plan and summary counts, writes
nothing. Pass --execute to actually copy. Pass --verify to compare the two
collections after a migration (or independently) without writing anything.

Auth: Application Default Credentials, exactly like server/watch/store.py's
FirestoreLiveSessionStore (lazy `google.cloud.firestore` import; run
`gcloud auth application-default login` if this fails locally).

Usage:
  python3 scripts/migrate_episodes_to_live_sessions.py               # dry-run
  python3 scripts/migrate_episodes_to_live_sessions.py --execute      # copy
  python3 scripts/migrate_episodes_to_live_sessions.py --verify       # compare
  python3 scripts/migrate_episodes_to_live_sessions.py --project other-project
"""
import argparse
import sys
from dataclasses import dataclass
from typing import Iterable

SOURCE_COLLECTION = "episodes"
TARGET_COLLECTION = "live_sessions"
DEFAULT_PROJECT = "arborfam-hub"


@dataclass(frozen=True)
class MigrationPlan:
    """The decision output of plan_migration.

    ``to_copy`` maps id -> doc for every source doc that should be written
    to the target collection. ``to_skip`` lists ids that already exist in
    the target and must be left untouched.
    """
    to_copy: dict[str, dict]
    to_skip: list[str]


def plan_migration(source_docs: dict[str, dict], target_ids: Iterable[str]) -> MigrationPlan:
    """Pure decision logic: given every source doc (id -> doc) and the ids
    already present in the target collection, decide which source docs to
    copy and which to skip. Does no I/O — the script wires this to
    Firestore reads/writes. A skipped id's doc content is never inspected
    or carried into to_copy, so the target's existing doc (whatever it
    currently contains) is guaranteed left alone.
    """
    target_id_set = set(target_ids)
    to_copy: dict[str, dict] = {}
    to_skip: list[str] = []
    for doc_id, doc in source_docs.items():
        if doc_id in target_id_set:
            to_skip.append(doc_id)
        else:
            to_copy[doc_id] = doc
    return MigrationPlan(to_copy=to_copy, to_skip=to_skip)


@dataclass(frozen=True)
class VerifyResult:
    """The decision output of plan_verify.

    ``missing_from_target`` is every source id NOT found in the target —
    the actual problem verify checks for, in source order. ``target_only``
    is every target id NOT found in source — reported as info (e.g. docs
    the unified service already wrote), never treated as an error.
    """
    missing_from_target: list[str]
    target_only: list[str]


def plan_verify(source_ids: Iterable[str], target_ids: Iterable[str]) -> VerifyResult:
    """Pure decision logic for --verify: every source id must be present in
    the target; target ids with no matching source id are informational
    only. Does no I/O."""
    source_id_list = list(source_ids)
    target_id_set = set(target_ids)
    missing = [i for i in source_id_list if i not in target_id_set]
    target_only = sorted(target_id_set - set(source_id_list))
    return VerifyResult(missing_from_target=missing, target_only=target_only)


def _get_db(project: str):
    """Lazily import and initialize the Firestore client, exactly like
    FirestoreLiveSessionStore._get_db (server/watch/store.py). Fails loudly
    and clearly if the SDK isn't installed or ADC isn't set up — no silent
    partial degradation."""
    try:
        from google.cloud import firestore
    except ImportError:
        sys.exit(
            "ERROR: google-cloud-firestore is not installed.\n"
            "Run: pip install google-cloud-firestore"
        )
    try:
        return firestore.Client(project=project)
    except Exception as e:
        sys.exit(
            f"ERROR: could not initialize a Firestore client for project '{project}'.\n"
            "Check Application Default Credentials, e.g.:\n"
            "  gcloud auth application-default login\n"
            f"Underlying error: {e}"
        )


def _stream_docs(db, collection: str) -> dict[str, dict]:
    return {doc.id: doc.to_dict() for doc in db.collection(collection).stream()}


def _stream_ids(db, collection: str) -> set[str]:
    return {doc.id for doc in db.collection(collection).stream()}


def _print_plan(plan: MigrationPlan, source_count: int, *, executed: bool) -> None:
    verb = "copied" if executed else "would-copy"
    for doc_id in plan.to_copy:
        print(f"  {verb:<10} {doc_id}")
    for doc_id in plan.to_skip:
        print(f"  {'skipped' if executed else 'would-skip':<10} {doc_id}  (already in {TARGET_COLLECTION})")
    mode = "EXECUTE" if executed else "DRY RUN"
    print(
        f"\n{mode}: {len(plan.to_copy)} to copy, {len(plan.to_skip)} to skip "
        f"(of {source_count} source docs in {SOURCE_COLLECTION})."
    )
    if not executed:
        print("Nothing written. Pass --execute to perform the copy.")


def run_dry_run(db) -> MigrationPlan:
    source_docs = _stream_docs(db, SOURCE_COLLECTION)
    target_ids = _stream_ids(db, TARGET_COLLECTION)
    plan = plan_migration(source_docs, target_ids)
    _print_plan(plan, len(source_docs), executed=False)
    return plan


def run_execute(db) -> MigrationPlan:
    source_docs = _stream_docs(db, SOURCE_COLLECTION)
    target_ids = _stream_ids(db, TARGET_COLLECTION)
    plan = plan_migration(source_docs, target_ids)
    for doc_id, doc in plan.to_copy.items():
        db.collection(TARGET_COLLECTION).document(doc_id).set(doc)
    _print_plan(plan, len(source_docs), executed=True)
    return plan


def run_verify(db) -> VerifyResult:
    source_ids = _stream_ids(db, SOURCE_COLLECTION)
    target_ids = _stream_ids(db, TARGET_COLLECTION)
    result = plan_verify(source_ids, target_ids)
    if result.missing_from_target:
        print(f"MISSING from {TARGET_COLLECTION} ({len(result.missing_from_target)}):")
        for doc_id in result.missing_from_target:
            print(f"  {doc_id}")
    else:
        print(f"OK: all {len(source_ids)} {SOURCE_COLLECTION} ids are present in {TARGET_COLLECTION}.")
    if result.target_only:
        print(
            f"\nINFO: {len(result.target_only)} id(s) exist only in {TARGET_COLLECTION} "
            f"(not in {SOURCE_COLLECTION} — e.g. already written by the unified service; "
            "not an error):"
        )
        for doc_id in result.target_only:
            print(f"  {doc_id}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Copy Firestore docs from '{SOURCE_COLLECTION}' to "
                    f"'{TARGET_COLLECTION}', preserving ids (idempotent; dry-run by default).",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT,
        help=f"GCP project (default: {DEFAULT_PROJECT})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true",
        help="Perform the copy. Without this flag, the script only prints "
             "the plan and writes nothing.",
    )
    mode.add_argument(
        "--verify", action="store_true",
        help=f"Compare {SOURCE_COLLECTION} and {TARGET_COLLECTION}: report any "
             "source id missing from the target. Writes nothing.",
    )
    args = parser.parse_args(argv)

    db = _get_db(args.project)

    if args.verify:
        result = run_verify(db)
        return 1 if result.missing_from_target else 0
    if args.execute:
        run_execute(db)
        return 0
    run_dry_run(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
