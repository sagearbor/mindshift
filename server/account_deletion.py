"""Account deletion — everything ``DELETE /me`` erases, and the rules it follows.

``routers/account.py`` owns the endpoint (auth, the type-to-confirm guard, the
rate limit); this module owns the actual erasure, one function per storage tier
plus :func:`delete_account_data` to run them in order. Split the same way
``therapist_links.py`` is split from ``routers/therapist.py``: the endpoint
stays readable, and the tier walk is testable against fakes without a request.

WHAT IS DELETED, for the authenticated uid and no one else
----------------------------------------------------------
Recordings bucket (``recordings_store.RecordingsStore``):

* ``recordings/{uid}/…``      every episode — stored audio, 360p video,
                              transcript turns, analysis, reflections,
                              titles, manual speaker labels
* every share grant ON those episodes, both sides (owner meta + the
  recipient's ``shared/{recipient}/…`` reverse index)
* ``shared/{uid}/…``          grants OTHER people issued to this user, revoked
                              on the owner's side too so no owner is left
                              showing a share to an account that no longer
                              exists
* ``voiceprints/{uid}/…``     every enrolled person's biometric signature —
                              self and every named partner, plus the legacy
                              single-document layout
* ``uploads/{uid}/…``         in-progress chunked uploads (manifest + parts)
* ``jobs/{uid}/…``            stored analysis-job state
* ``therapist_links/{uid}/…`` this user's link AS A PATIENT, plus the linked
                              therapist's reverse-index row
* ``therapist_patients/{uid}/…`` this user's patients AS A THERAPIST, plus each
                              of those patients' own link document (a link to a
                              deleted therapist is not a link)
* ``therapist_notes/{uid}/…`` every private note this user wrote as a viewer
* ``therapist_notes/{other}/{episode_id}.json`` — notes OTHER people wrote about
                              THIS user's episodes (see the shared-data rule)

Watch/Firestore tier (``watch.store`` / ``watch.pairing_store`` /
``watch.telemetry_store`` / ``watch.blobs``):

* live sessions owned by the uid; the uid removed from any live session
  someone else owns and shared with them
* the enrollment baseline, vector subscriptions, account document and
  watch-domain speaker profile
* captures — the stored audio blob first, then the metadata document
* group memberships (and the group itself once it has no members left)
* watch pairings: device tokens, claimed pairing records, the failed-claim
  counter
* diagnostics ("Send diagnostics") reports whose payload names this uid

Relational tier (SQLite, ``main.DB_PATH``):

* ``sessions`` rows owned by the uid (the text-tool transcripts)
* ``relationships`` owned by the uid, with their ``participants`` and
  ``voice_profiles`` rows

Finally, and ONLY once every one of the above reported success, the Firebase
Auth user itself (``auth.delete_firebase_user``). Deliberately last: if any
tier fails, the account still exists and the user can sign in and retry rather
than being locked out of data that outlived them.

THE SHARED-DATA RULE (decided here; stated identically in the privacy policy,
apps/mobile/public/delete-account/index.html and docs/play/play-answers-
mindshift.yaml)
----------------------------------------------------------------------------
A session you shared with a therapist is still YOURS — sharing grants a view,
it never transfers ownership, and the whole product's consent story rests on
that. So deleting your account deletes the session itself, and with it every
grant that let anyone see it. Private notes a viewer wrote ABOUT ONE OF YOUR
SESSIONS go too: a note is an annotation on your content, and keeping a
therapist's notes about a session that no longer exists would leave a record of
your conversation behind under someone else's account — exactly the outcome the
user asked us to prevent. Notes that viewer wrote about anyone ELSE's sessions,
and everything else in their account, are untouched.

WHAT IS NOT DELETED, and why (also stated in all three documents)
-----------------------------------------------------------------
* Videos the app saved to the user's own phone photo library — that copy is
  theirs, on their device, and the server cannot reach it.
* Anything another person exported, screenshotted or wrote down themselves.
* Operational server logs (timestamps, status codes, request ids and the uid),
  which rotate out on the platform's own schedule and never contain
  transcripts, audio or analysis.
* Watch-sent telemetry, which identifies a DEVICE and carries no account id at
  all — there is nothing in it to attribute to the deleted person.
* An in-flight live call the user is currently on: calls live only in process
  memory (``server/calls.py``) and vanish when the call ends or the process
  restarts. Nothing about them is persisted except the episode each
  participant's own account stores, which IS deleted above.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Every counter reported by DELETE /me, in the order the tiers run. Declared as
# a constant so the response always carries the FULL set of keys (a tier that
# was disabled reports 0, not a missing field — the caller can render "0
# recordings" honestly instead of guessing what the absence meant).
COUNT_KEYS: tuple[str, ...] = (
    "recordings",
    "shares_you_granted",
    "shares_granted_to_you",
    "notes_others_wrote_about_your_sessions",
    "notes_you_wrote",
    "voiceprints",
    "unfinished_uploads",
    "analysis_jobs",
    "therapist_links",
    "live_sessions",
    "watch_captures",
    "watch_pairings",
    "groups_left",
    "diagnostic_reports",
    "text_sessions",
    "relationships",
)


@dataclass
class DeletionSummary:
    """What one ``DELETE /me`` actually removed.

    ``counts`` always carries every key in :data:`COUNT_KEYS`. ``errors`` is
    the honest record of tiers that failed: a non-empty list means the account
    was NOT finished off (the Firebase user is left in place) and the caller
    should retry, never that deletion silently half-happened.
    """

    counts: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in COUNT_KEYS}
    )
    errors: list[str] = field(default_factory=list)
    firebase_user_deleted: bool = False

    def add(self, key: str, n: int) -> None:
        self.counts[key] = self.counts.get(key, 0) + int(n)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def total(self) -> int:
        return sum(self.counts.values())


async def _run_tier(summary: DeletionSummary, name: str, coro) -> None:
    """Await one tier's deletion, recording a failure instead of raising.

    Every tier is attempted even when an earlier one failed: a user whose GCS
    delete errors should still have their Firestore documents removed, and the
    recorded error is what stops the Firebase user from being deleted over the
    top of the leftovers. The exception text is logged in full and summarized
    (type + tier) to the caller — provider internals never reach the wire.
    """
    try:
        await coro
    except Exception as exc:  # noqa: BLE001 — one failed tier must not hide the rest
        logger.exception("Account deletion tier %s failed", name)
        summary.errors.append(f"{name}: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Tier 1 — the recordings bucket
# ---------------------------------------------------------------------------

async def delete_recordings_tier(store, uid: str, summary: DeletionSummary) -> None:
    """Erase everything ``uid`` owns in the recordings bucket, plus the two
    cross-account artifacts their episodes created: grants issued to other
    accounts and notes other accounts wrote about those episodes.

    Order matters. The share recipients are read BEFORE the episodes are
    deleted — meta.json is the only record of who could see each one, and it
    goes away with the episode."""
    # Who could see what, read while the metadata still exists.
    recipients_by_recording = await store.list_recording_share_recipients(uid)

    # Notes other people kept about THOSE episodes (the shared-data rule).
    notes_removed = 0
    for recording_id, recipients in recipients_by_recording.items():
        for recipient_uid in recipients:
            if recipient_uid == uid:
                continue
            if await store.delete_therapist_note(recipient_uid, recording_id):
                notes_removed += 1
    summary.add("notes_others_wrote_about_your_sessions", notes_removed)
    summary.add(
        "shares_you_granted", sum(len(r) for r in recipients_by_recording.values()),
    )

    # The episodes themselves (delete_recording tears down each recipient's
    # reverse-index grant on the way past — the same path a single delete uses).
    summary.add("recordings", await store.delete_all_recordings(uid))

    # Grants OTHER people issued to this user: revoke on the owner's side so no
    # surviving owner is left listing a share to an account that is gone, then
    # the user's own reverse index disappears with it.
    received = await store.list_received_share_grants(uid)
    for grant in received:
        owner_uid = grant.get("owner_uid")
        recording_id = grant.get("recording_id")
        if not owner_uid or not recording_id:
            continue
        await store.remove_share(owner_uid, recording_id, uid)
    summary.add("shares_granted_to_you", len(received))

    summary.add("voiceprints", await store.delete_all_voiceprints(uid))
    summary.add("unfinished_uploads", await store.delete_all_uploads(uid))
    summary.add("analysis_jobs", await store.delete_all_jobs(uid))
    summary.add("notes_you_wrote", await store.delete_all_therapist_notes(uid))

    # Therapist links, BOTH directions.
    links = 0
    if await store.delete_therapist_link(uid):       # this user as the patient
        links += 1
    for link in await store.list_therapist_patients(uid):  # ...as the therapist
        patient_uid = link.get("patient_uid")
        if not patient_uid:
            continue
        await store.delete_therapist_link(patient_uid)
        await store.delete_therapist_patient_entry(uid, patient_uid)
        links += 1
    summary.add("therapist_links", links)


# ---------------------------------------------------------------------------
# Tier 2 — the watch/Firestore documents
# ---------------------------------------------------------------------------

def purge_group_member(uid: str):
    """Mutator for ``store.update_group_atomically``: scrub every trace of
    ``uid`` from a group.

    Membership, the mutual-visibility consent record, and any invite this user
    minted or accepted all carry the account id, so all three go — this is an
    account deletion, not the ordinary "leave" (which deliberately KEEPS the
    consent record as an audit trail, see ``watch/routers/groups.py``'s
    ``make_leave_mutator``). ``created_by`` is blanked rather than left
    pointing at an account that no longer exists; nothing reads it as an
    identity, only as provenance."""
    def mutate(group):
        if group is None:
            raise KeyError("group vanished")
        group.members = [m for m in group.members if m.account_id != uid]
        group.consents = [
            c for c in group.consents
            if c.participant_id != uid and c.attested_by != uid
        ]
        group.invites = [
            i for i in group.invites
            if i.invited_by != uid and i.accepted_by != uid
        ]
        if group.created_by == uid:
            group.created_by = ""
        return group
    return mutate


async def delete_watch_tier(
    uid: str, summary: DeletionSummary, *, store, pairing_store, telemetry_store,
    blobs,
) -> None:
    """Erase the uid's watch-domain documents and capture audio.

    Each store is optional: a deployment (or a test) that wired none of them
    simply has nothing here to delete, which is reported as zeros rather than a
    failure."""
    if store is not None:
        # Live sessions: delete the ones this account OWNS; for one someone
        # else owns and shared with them, drop just the sharing pointer — the
        # owner's session is not this user's to delete.
        deleted_sessions = 0
        for session in await store.list_live_sessions(uid):
            if session.owner_account == uid:
                await store.delete_live_session(session.id)
                deleted_sessions += 1
            elif uid in session.shared_with:
                session.shared_with = [a for a in session.shared_with if a != uid]
                await store.put_live_session(session)
        summary.add("live_sessions", deleted_sessions)

        # Captures: blob first, then the metadata doc — never orphan audio
        # behind a deleted pointer (the same order DELETE /captures/{id} uses).
        captures = await store.list_captures(uid)
        for capture in captures:
            if capture.audio_uri is not None and blobs is not None:
                from watch.routers.captures import capture_key

                await blobs.delete(capture_key(uid, capture.id))
            await store.delete_capture(capture.id)
        summary.add("watch_captures", len(captures))

        # Groups: scrub the member out; delete the group once it is empty.
        groups = await store.list_groups(uid)
        for group in groups:
            try:
                remaining = await store.update_group_atomically(
                    group.id, purge_group_member(uid),
                )
            except KeyError:
                continue  # deleted underneath us — nothing left to scrub
            if not remaining.members:
                await store.delete_group(remaining.id)
        summary.add("groups_left", len(groups))

        await store.delete_baseline(uid)
        await store.delete_subscriptions(uid)
        await store.delete_speaker_profile(uid)
        await store.delete_account(uid)

    if pairing_store is not None:
        pairings = await pairing_store.delete_device_tokens_for_account(uid)
        pairings += await pairing_store.delete_pairings_for_account(uid)
        await pairing_store.delete_failed_claim_record(uid)
        summary.add("watch_pairings", pairings)

    if telemetry_store is not None:
        summary.add(
            "diagnostic_reports",
            await telemetry_store.delete_events_for_account(uid),
        )


# ---------------------------------------------------------------------------
# Tier 3 — the relational (SQLite) rows
# ---------------------------------------------------------------------------

async def delete_sqlite_tier(db, uid: str, summary: DeletionSummary) -> None:
    """Delete the uid's ``sessions`` and ``relationships`` rows (and everything
    that hangs off a relationship) from the aiosqlite database.

    ``sessions`` and ``relationships`` are the two ownership roots — see
    ``main.init_db``'s migration note; ``participants`` and ``voice_profiles``
    inherit ownership through ``relationship_id``, so they are removed by the
    relationship ids rather than by a user column they do not have. A session
    filed under one of those relationships is deleted too even if its own
    ``user_id`` is NULL (rows written before the auth migration)."""
    cursor = await db.execute(
        "SELECT id FROM relationships WHERE user_id = ?", (uid,),
    )
    relationship_ids = [row[0] for row in await cursor.fetchall()]
    await cursor.close()

    if relationship_ids:
        marks = ",".join("?" for _ in relationship_ids)
        await db.execute(
            f"DELETE FROM voice_profiles WHERE relationship_id IN ({marks})",
            relationship_ids,
        )
        await db.execute(
            f"DELETE FROM participants WHERE relationship_id IN ({marks})",
            relationship_ids,
        )
        result = await db.execute(
            f"DELETE FROM sessions WHERE user_id = ? OR relationship_id IN ({marks})",
            (uid, *relationship_ids),
        )
    else:
        result = await db.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
    summary.add("text_sessions", result.rowcount if result.rowcount > 0 else 0)

    await db.execute("DELETE FROM relationships WHERE user_id = ?", (uid,))
    summary.add("relationships", len(relationship_ids))
    await db.commit()


# ---------------------------------------------------------------------------
# The whole walk
# ---------------------------------------------------------------------------

async def delete_account_data(
    uid: str,
    *,
    recordings_store=None,
    watch_store=None,
    pairing_store=None,
    telemetry_store=None,
    blobs=None,
    db=None,
) -> DeletionSummary:
    """Erase every tier for ``uid`` and return what was removed.

    Does NOT touch Firebase Auth — the caller does that, last, and only when
    :attr:`DeletionSummary.ok`. Every argument is optional: a tier whose store
    was never configured contributes zeros, exactly like the endpoints that
    already answer honestly when recording storage is disabled.

    Idempotent by construction — every underlying delete is "remove if
    present" — so a re-run against an already-deleted account walks the same
    tiers and reports all zeros."""
    summary = DeletionSummary()

    if recordings_store is not None:
        await _run_tier(
            summary, "recordings",
            delete_recordings_tier(recordings_store, uid, summary),
        )

    await _run_tier(
        summary, "watch",
        delete_watch_tier(
            uid, summary, store=watch_store, pairing_store=pairing_store,
            telemetry_store=telemetry_store, blobs=blobs,
        ),
    )

    if db is not None:
        await _run_tier(summary, "sessions_db", delete_sqlite_tier(db, uid, summary))

    return summary
