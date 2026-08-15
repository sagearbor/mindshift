# Ported from gauge@2157433 server/store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# INHERITED WART (kept for parity, not fixed here): store mutators
# (update_group_atomically's mutator, update_legacy_claim_atomically's
# mutator, claim_legacy_live_session) raise fastapi.HTTPException directly
# from the storage layer instead of a store-native exception type. This is
# Gauge's original design (the mutator's HTTPException propagates unchanged
# to the router) and is preserved verbatim — parity beats refactor here.
import logging
import os
import threading
from typing import Callable, Protocol

from fastapi import HTTPException

from watch.models import (
    Account, LiveSession, EnrollmentBaseline, VectorSubscription, VectorName, SpeakerProfile,
    Group, Capture, LegacyClaim, LEGACY_ACCOUNT_ID,
)

logger = logging.getLogger(__name__)

# Final-review Finding 1b: Firestore's hard per-document limit is 1,048,576
# bytes (1 MiB) — see https://cloud.google.com/firestore/quotas. A whole
# LiveSession doc (vector_events, series, consents, ...) has to fit under
# that ALONGSIDE pcm_b64, so this caps pcm_b64 well below the ceiling rather
# than at it, leaving headroom for the rest of the doc plus Firestore's own
# field overhead. At the wire contract's 32,000 bytes/s of raw PCM16 (base64
# blows that up ~4/3x, ~42,667 b64 chars/s), 900,000 bytes is roughly 21s of
# audio — live sessions longer than that persist everything EXCEPT raw audio
# (events, series, and summary are computed from the live session / a direct
# in-memory PCM handoff and are unaffected — see server/watch/routers/ws.py's
# (Task B11) _spawn_live_session_analysis and server/watch/post_session.py's
# (Task B10) analyze_live_session(pcm=...) parameter).
MAX_FIRESTORE_PCM_B64 = 900_000

# The 5 v1 default vector subscriptions, returned by both store
# implementations' get_subscriptions() when an account has none saved yet.
#
# Final-review Finding 2b: "interrupting" and "airtime" are listed here (and
# subscribable) but CANNOT actually fire in v1 — VectorEngine.push_diarization
# is the only thing that ever emits them, and nothing in this codebase calls
# it: there's no WS diarization frame in the wire protocol and no diarization
# source wired into server/watch/routers/ws.py (Task B11). They're included
# now so the subscription list/UI is already shaped for Plan 2, when a
# diarization pipeline lands and starts calling push_diarization for real.
# See also the comment on push_diarization in server/watch/vectors.py (Task B4).
DEFAULT_VECTOR_NAMES: list[VectorName] = [
    "yelling", "aggressive_tone", "interrupting", "airtime", "hr_spike",
]


def live_session_to_doc(ls: LiveSession) -> dict:
    """Serialize a LiveSession for persistence, including ``pcm_b64``.

    ``LiveSession.pcm_b64`` is ``Field(exclude=True)`` so it never reaches
    the wire via ``model_dump()``/``model_dump_json()`` — but real
    persistence backends still need the raw audio (Task 8's post-session
    analysis reads it back out of storage). ``exclude=True`` only affects
    dumping, not validation, so a doc built this way round-trips cleanly
    through ``LiveSession(**doc)``. This is the one place that re-attaches
    it.

    Final-review Finding 1b: pcm_b64 is only included when it fits well
    under Firestore's 1MiB document limit (``MAX_FIRESTORE_PCM_B64``);
    otherwise it's dropped to ``""`` and logged once, so a long live
    session still persists its metadata/events/series/summary instead of
    the whole write raising ``InvalidArgument`` and losing the live session
    entirely.
    """
    data = ls.model_dump()
    pcm_b64 = ls.pcm_b64
    if len(pcm_b64) > MAX_FIRESTORE_PCM_B64:
        logger.warning(
            "LiveSession %s pcm_b64 is %d bytes, over MAX_FIRESTORE_PCM_B64 (%d) — "
            "dropping raw audio from the persisted doc so the rest of the "
            "live session (events/series/summary) still saves. See README's "
            "Limitations section.",
            ls.id, len(pcm_b64), MAX_FIRESTORE_PCM_B64,
        )
        pcm_b64 = ""
    data["pcm_b64"] = pcm_b64
    return data


PAIR_MAX_MEMBERS = 2


def group_to_doc(g: Group) -> dict:
    """Serialize a Group for persistence, adding a denormalized
    `member_account_ids` list. Firestore cannot `array_contains` on a
    property of objects inside an array, so membership lookup needs a flat
    string array alongside `members`. Extra keys are ignored by
    `Group(**doc)` (pydantic v2's default), so the doc round-trips."""
    data = g.model_dump()
    data["member_account_ids"] = [m.account_id for m in g.members]
    return data


class LiveSessionStore(Protocol):
    """Protocol for live-session storage backend."""

    async def put_live_session(self, ls: LiveSession) -> None:
        """Store or update a live session."""
        ...

    async def get_live_session(self, id: str) -> LiveSession | None:
        """Retrieve a live session by id, or None if not found."""
        ...

    async def list_live_sessions(self, account: str) -> list[LiveSession]:
        """List live sessions owned by account or shared with account, newest first by started_at."""
        ...

    async def delete_live_session(self, id: str) -> None:
        """Remove a live session. Idempotent — deleting an absent id is a
        no-op (the API layer 404s first; the store never throws for absence)."""
        ...

    async def put_baseline(self, b: EnrollmentBaseline) -> None:
        """Store or update an enrollment baseline."""
        ...

    async def get_baseline(self, account_id: str) -> EnrollmentBaseline | None:
        """Retrieve an enrollment baseline by account_id, or None if not found."""
        ...

    async def put_subscriptions(self, account_id: str, subs: list[VectorSubscription]) -> None:
        """Store vector subscriptions for an account."""
        ...

    async def get_subscriptions(self, account_id: str) -> list[VectorSubscription]:
        """Get subscriptions for an account. Empty → return 5 v1 defaults."""
        ...

    async def put_account(self, a: Account) -> None:
        """Store or update an account."""
        ...

    async def get_account(self, account_id: str) -> Account | None:
        """Retrieve an account by id, or None if not found."""
        ...

    async def get_account_by_email(self, email: str) -> Account | None:
        """Retrieve an account by exact email match, or None (also None for a falsy email)."""
        ...

    async def put_speaker_profile(self, p: SpeakerProfile) -> None:
        """Store or update a speaker (voice enrollment) profile."""
        ...

    async def get_speaker_profile(self, account_id: str) -> SpeakerProfile | None:
        """Retrieve a speaker profile by account_id, or None if not found."""
        ...

    async def put_group(self, g: Group) -> None:
        """Store or update a group."""
        ...

    async def get_group(self, group_id: str) -> Group | None:
        """Retrieve a group by id, or None if not found."""
        ...

    async def list_groups(self, account_id: str) -> list[Group]:
        """List groups the account is a current member of, oldest-first by created_at."""
        ...

    async def get_group_by_invite_code(self, code: str) -> Group | None:
        """Retrieve a group by an (unaccepted or accepted) invite code, or None if not found."""
        ...

    async def update_group_atomically(
        self, group_id: str, mutator: Callable[["Group | None"], Group]
    ) -> Group:
        """Read-check-write a group with no other writer able to interleave.

        ``mutator`` receives the freshly (consistently) read ``Group``, or
        ``None`` if the id doesn't exist, and returns the ``Group`` to
        persist — or raises to abort the whole operation with NOTHING
        persisted (the exception propagates to the caller unchanged). This
        is the seam that enforces invariants like single-use invite codes
        and ``PAIR_MAX_MEMBERS`` atomically: the caller's invariant checks
        run *inside* the atomic section (as part of ``mutator``), against a
        read that cannot be stale by the time the write happens.

        ``FirestoreLiveSessionStore`` backs this with
        ``@firestore.transactional``; ``MemoryLiveSessionStore`` backs it
        with a ``threading.Lock`` around the identical read-mutate-write
        sequence — both implementations run the exact same caller-supplied
        invariant checks.
        """
        ...

    async def put_capture(self, c: Capture) -> None:
        """Store or update a capture's metadata."""
        ...

    async def get_capture(self, capture_id: str) -> Capture | None:
        """Retrieve a capture by id, or None if not found."""
        ...

    async def list_captures(self, account_id: str) -> list[Capture]:
        """List captures owned by account_id, newest-first by captured_at."""
        ...

    async def delete_capture(self, capture_id: str) -> None:
        """Remove a capture's metadata doc. Same idempotent-on-absent contract."""
        ...

    async def get_legacy_claim(self) -> "LegacyClaim | None":
        """The singleton claim marker, or None if the legacy account is unclaimed."""
        ...

    async def update_legacy_claim_atomically(
        self, mutator: Callable[["LegacyClaim | None"], LegacyClaim]
    ) -> LegacyClaim:
        """Read-check-write the claim marker with no other writer able to
        interleave — the same seam contract as update_group_atomically: the
        mutator sees a fresh consistent read (None when absent), returns the
        LegacyClaim to persist, or raises to abort with NOTHING persisted."""
        ...

    async def claim_legacy_live_session(self, live_session_id: str, account: str) -> bool:
        """Atomically re-key ONE legacy-owned live session to `account` AND
        bump the audited `LegacyClaim.episodes_moved_total` by 1, as a
        single indivisible step -- see server/watch/routers/rest.py's (Task B5)
        claim_legacy Phase 2 comment for why "check ownership, move, then separately
        bump the total" (two calls, however ordered) leaves a same-uid
        concurrent-double-submission race open. Returns True if this call
        performed the move+bump (the live session was still
        LEGACY_ACCOUNT_ID-owned the moment this executed), False if a
        racing same-uid claim had already claimed it first -- the caller
        MUST NOT count a False return toward `moved`, and MUST NOT treat
        False as an error (it's the expected outcome for whichever of two
        racing requests loses a given live session).

        Requires the marker to already be reserved for `account` (Phase 1
        must have run first) -- raises 409 if the marker is somehow owned by
        a DIFFERENT account when this executes (should be unreachable in
        practice, same invariant `make_claim_mutator`'s Phase 1 reservation
        already establishes before Phase 2 ever runs)."""
        ...

    async def has_subscriptions(self, account_id: str) -> bool:
        """True only when a SAVED subscriptions doc exists for the account —
        never True merely because get_subscriptions would fabricate defaults."""
        ...


class MemoryLiveSessionStore:
    """In-memory implementation of LiveSessionStore for testing and default runtime."""

    def __init__(self):
        self._live_sessions: dict[str, LiveSession] = {}
        self._baselines: dict[str, EnrollmentBaseline] = {}
        self._subscriptions: dict[str, list[VectorSubscription]] = {}
        self._accounts: dict[str, Account] = {}
        self._profiles: dict[str, SpeakerProfile] = {}
        self._groups: dict[str, Group] = {}
        self._groups_lock = threading.Lock()
        self._captures: dict[str, Capture] = {}
        self._legacy_claim: LegacyClaim | None = None
        self._claim_lock = threading.Lock()

    async def put_live_session(self, ls: LiveSession) -> None:
        """Store or update a live session."""
        self._live_sessions[ls.id] = ls.model_copy(deep=True)

    async def get_live_session(self, id: str) -> LiveSession | None:
        """Retrieve a live session by id, or None if not found."""
        ls = self._live_sessions.get(id)
        return ls.model_copy(deep=True) if ls else None

    async def delete_live_session(self, id: str) -> None:
        """Remove a live session. Idempotent — deleting an absent id is a no-op."""
        self._live_sessions.pop(id, None)

    async def list_live_sessions(self, account: str) -> list[LiveSession]:
        """List live sessions owned by account or shared with account, newest first by started_at."""
        live_sessions = [
            ls.model_copy(deep=True) for ls in self._live_sessions.values()
            if ls.owner_account == account or account in ls.shared_with
        ]
        # Sort by started_at descending (newest first)
        live_sessions.sort(key=lambda e: e.started_at, reverse=True)
        return live_sessions

    async def put_baseline(self, b: EnrollmentBaseline) -> None:
        """Store or update an enrollment baseline."""
        self._baselines[b.account_id] = b.model_copy(deep=True)

    async def get_baseline(self, account_id: str) -> EnrollmentBaseline | None:
        """Retrieve an enrollment baseline by account_id, or None if not found."""
        b = self._baselines.get(account_id)
        return b.model_copy(deep=True) if b else None

    async def put_subscriptions(self, account_id: str, subs: list[VectorSubscription]) -> None:
        """Store vector subscriptions for an account."""
        self._subscriptions[account_id] = [s.model_copy(deep=True) for s in subs]

    async def get_subscriptions(self, account_id: str) -> list[VectorSubscription]:
        """Get subscriptions for an account. Empty → return 5 v1 defaults.

        Pure read (D7 fix): fabricating defaults must NOT write them into
        ``self._subscriptions`` — that would make an unsaved GET
        indistinguishable from a real save and poison ``has_subscriptions``
        after any phone GET. This matches FirestoreLiveSessionStore, which
        never had the bug (its "doc doesn't exist" branch never calls
        ``.set()``).
        """
        if account_id not in self._subscriptions:
            # Build 5 v1 defaults with proper channels, WITHOUT persisting.
            return [VectorSubscription(vector=v) for v in DEFAULT_VECTOR_NAMES]
        return [s.model_copy(deep=True) for s in self._subscriptions[account_id]]

    async def put_account(self, a: Account) -> None:
        """Store or update an account."""
        self._accounts[a.id] = a.model_copy(deep=True)

    async def get_account(self, account_id: str) -> Account | None:
        """Retrieve an account by id, or None if not found."""
        a = self._accounts.get(account_id)
        return a.model_copy(deep=True) if a else None

    async def get_account_by_email(self, email: str) -> Account | None:
        """Retrieve an account by exact email match, or None (also None for a falsy email)."""
        if not email:
            return None
        for a in self._accounts.values():
            if a.email == email:
                return a.model_copy(deep=True)
        return None

    async def put_speaker_profile(self, p: SpeakerProfile) -> None:
        """Store or update a speaker (voice enrollment) profile."""
        self._profiles[p.account_id] = p.model_copy(deep=True)

    async def get_speaker_profile(self, account_id: str) -> SpeakerProfile | None:
        """Retrieve a speaker profile by account_id, or None if not found."""
        p = self._profiles.get(account_id)
        return p.model_copy(deep=True) if p else None

    async def put_group(self, g: Group) -> None:
        """Store or update a group."""
        self._groups[g.id] = g.model_copy(deep=True)

    async def get_group(self, group_id: str) -> Group | None:
        """Retrieve a group by id, or None if not found."""
        g = self._groups.get(group_id)
        return g.model_copy(deep=True) if g else None

    async def list_groups(self, account_id: str) -> list[Group]:
        """List groups the account is a current member of, oldest-first by created_at."""
        groups = [
            g.model_copy(deep=True) for g in self._groups.values()
            if any(m.account_id == account_id for m in g.members)
        ]
        groups.sort(key=lambda g: g.created_at)
        return groups

    async def get_group_by_invite_code(self, code: str) -> Group | None:
        """Retrieve a group by an (unaccepted or accepted) invite code, or None if not found."""
        for g in self._groups.values():
            if any(inv.code == code for inv in g.invites):
                return g.model_copy(deep=True)
        return None

    def _read_group_locked(self, group_id: str) -> Group | None:
        """The read half of update_group_atomically, split out so tests can
        override it to widen the critical section (see test_store.py's
        concurrency tests) — the override proves the lock actually serializes
        callers rather than merely happening to not race in practice."""
        g = self._groups.get(group_id)
        return g.model_copy(deep=True) if g else None

    async def update_group_atomically(
        self, group_id: str, mutator: Callable[["Group | None"], Group]
    ) -> Group:
        """See LiveSessionStore.update_group_atomically. threading.Lock (not
        asyncio.Lock) because the concurrency this guards against is real
        OS-thread concurrency (e.g. a test driving two threads each running
        their own event loop against one shared store instance), matching
        the real-world hazard (two Cloud Run instances) this seam exists
        to rule out."""
        with self._groups_lock:
            current = self._read_group_locked(group_id)
            new_group = mutator(current)
            self._groups[new_group.id] = new_group.model_copy(deep=True)
            return new_group.model_copy(deep=True)

    async def put_capture(self, c: Capture) -> None:
        """Store or update a capture's metadata."""
        self._captures[c.id] = c.model_copy(deep=True)

    async def get_capture(self, capture_id: str) -> Capture | None:
        """Retrieve a capture by id, or None if not found."""
        c = self._captures.get(capture_id)
        return c.model_copy(deep=True) if c else None

    async def list_captures(self, account_id: str) -> list[Capture]:
        """List captures owned by account_id, newest-first by captured_at."""
        captures = [
            c.model_copy(deep=True) for c in self._captures.values()
            if c.account_id == account_id
        ]
        captures.sort(key=lambda c: c.captured_at, reverse=True)
        return captures

    async def delete_capture(self, capture_id: str) -> None:
        """Remove a capture's metadata doc. Idempotent — deleting an absent id is a no-op."""
        self._captures.pop(capture_id, None)

    async def get_legacy_claim(self) -> LegacyClaim | None:
        """The singleton claim marker, or None if the legacy account is unclaimed."""
        c = self._legacy_claim
        return c.model_copy(deep=True) if c else None

    def _read_claim_locked(self) -> LegacyClaim | None:
        """The read half of update_legacy_claim_atomically, split out so
        tests can override it to widen the critical section (see
        test_store.py's concurrency tests) — the override proves the lock
        actually serializes callers rather than merely happening to not
        race in practice. Mirrors _read_group_locked."""
        c = self._legacy_claim
        return c.model_copy(deep=True) if c else None

    async def update_legacy_claim_atomically(
        self, mutator: Callable[["LegacyClaim | None"], LegacyClaim]
    ) -> LegacyClaim:
        """See LiveSessionStore.update_legacy_claim_atomically.
        threading.Lock (not asyncio.Lock) for the same
        real-OS-thread-concurrency reason as update_group_atomically."""
        with self._claim_lock:
            current = self._read_claim_locked()
            new_claim = mutator(current)
            self._legacy_claim = new_claim.model_copy(deep=True)
            return new_claim.model_copy(deep=True)

    async def claim_legacy_live_session(self, live_session_id: str, account: str) -> bool:
        """See LiveSessionStore.claim_legacy_live_session. Reuses
        `_claim_lock` -- the SAME lock update_legacy_claim_atomically holds
        -- so this live session's ownership check, its move, AND its
        counter-bump execute as one indivisible step no other
        claim-workflow write (Phase 1 for another request, or another live
        session's claim_legacy_live_session call, same or different
        account) can land in the middle of."""
        with self._claim_lock:
            ls = self._live_sessions.get(live_session_id)
            if ls is None or ls.owner_account != LEGACY_ACCOUNT_ID:
                return False
            moved = ls.model_copy(deep=True)
            moved.owner_account = account
            moved.shared_with = [a for a in moved.shared_with if a != account]
            self._live_sessions[live_session_id] = moved

            current = self._read_claim_locked()
            if current is None or current.account_id != account:
                raise HTTPException(status_code=409, detail="claim marker changed underneath this claim")
            self._legacy_claim = current.model_copy(
                update={"episodes_moved_total": current.episodes_moved_total + 1}
            )
            return True

    async def has_subscriptions(self, account_id: str) -> bool:
        """True only when a SAVED subscriptions doc exists for the account —
        never True merely because get_subscriptions would fabricate defaults."""
        return account_id in self._subscriptions


class FirestoreLiveSessionStore:
    """Firestore-backed implementation of LiveSessionStore. Lazily imports google-cloud-firestore."""

    def __init__(self, project: str):
        self.project = project
        self._db = None

    def _get_db(self):
        """Lazily import and initialize Firestore client."""
        if self._db is None:
            from google.cloud import firestore
            self._db = firestore.Client(project=self.project)
        return self._db

    async def put_live_session(self, ls: LiveSession) -> None:
        """Store or update a live session."""
        db = self._get_db()
        db.collection("live_sessions").document(ls.id).set(live_session_to_doc(ls))

    async def get_live_session(self, id: str) -> LiveSession | None:
        """Retrieve a live session by id, or None if not found."""
        db = self._get_db()
        doc = db.collection("live_sessions").document(id).get()
        if doc.exists:
            return LiveSession(**doc.to_dict())
        return None

    async def delete_live_session(self, id: str) -> None:
        """Remove a live session. Firestore document deletes are already
        idempotent — deleting an absent id is a no-op, no existence check needed."""
        db = self._get_db()
        db.collection("live_sessions").document(id).delete()

    async def list_live_sessions(self, account: str) -> list[LiveSession]:
        """List live sessions owned by account or shared with account, newest first by started_at."""
        db = self._get_db()
        # Get owned live sessions
        owned_docs = db.collection("live_sessions").where("owner_account", "==", account).stream()
        live_sessions = [LiveSession(**doc.to_dict()) for doc in owned_docs]

        # Get shared live sessions
        # NB: this client version's operator token is underscore-style
        # ("array_contains"); the hyphen form raises ValueError at query time.
        shared_docs = db.collection("live_sessions").where("shared_with", "array_contains", account).stream()
        live_sessions.extend([LiveSession(**doc.to_dict()) for doc in shared_docs])

        # Remove duplicates (shouldn't happen but be safe)
        seen = set()
        unique = []
        for ls in live_sessions:
            if ls.id not in seen:
                unique.append(ls)
                seen.add(ls.id)

        # Sort by started_at descending (newest first)
        unique.sort(key=lambda e: e.started_at, reverse=True)
        return unique

    async def put_baseline(self, b: EnrollmentBaseline) -> None:
        """Store or update an enrollment baseline."""
        db = self._get_db()
        db.collection("baselines").document(b.account_id).set(b.model_dump())

    async def get_baseline(self, account_id: str) -> EnrollmentBaseline | None:
        """Retrieve an enrollment baseline by account_id, or None if not found."""
        db = self._get_db()
        doc = db.collection("baselines").document(account_id).get()
        if doc.exists:
            return EnrollmentBaseline(**doc.to_dict())
        return None

    async def put_subscriptions(self, account_id: str, subs: list[VectorSubscription]) -> None:
        """Store vector subscriptions for an account."""
        db = self._get_db()
        db.collection("subscriptions").document(account_id).set({
            "subscriptions": [s.model_dump() for s in subs]
        })

    async def get_subscriptions(self, account_id: str) -> list[VectorSubscription]:
        """Get subscriptions for an account. Empty → return 5 v1 defaults."""
        db = self._get_db()
        doc = db.collection("subscriptions").document(account_id).get()
        if doc.exists:
            data = doc.to_dict()
            return [VectorSubscription(**s) for s in data.get("subscriptions", [])]

        # Return defaults
        return [VectorSubscription(vector=v) for v in DEFAULT_VECTOR_NAMES]

    async def has_subscriptions(self, account_id: str) -> bool:
        """True only when a SAVED subscriptions doc exists for the account —
        never True merely because get_subscriptions would fabricate defaults."""
        db = self._get_db()
        doc = db.collection("subscriptions").document(account_id).get()
        return doc.exists

    async def put_account(self, a: Account) -> None:
        """Store or update an account."""
        db = self._get_db()
        db.collection("accounts").document(a.id).set(a.model_dump())

    async def get_account(self, account_id: str) -> Account | None:
        """Retrieve an account by id, or None if not found."""
        db = self._get_db()
        doc = db.collection("accounts").document(account_id).get()
        if doc.exists:
            return Account(**doc.to_dict())
        return None

    async def get_account_by_email(self, email: str) -> Account | None:
        """Retrieve an account by exact email match, or None (also None for a falsy email)."""
        if not email:
            return None
        db = self._get_db()
        docs = db.collection("accounts").where("email", "==", email).stream()
        for doc in docs:
            return Account(**doc.to_dict())
        return None

    async def put_speaker_profile(self, p: SpeakerProfile) -> None:
        """Store or update a speaker (voice enrollment) profile."""
        db = self._get_db()
        db.collection("speaker_profiles").document(p.account_id).set(p.model_dump())

    async def get_speaker_profile(self, account_id: str) -> SpeakerProfile | None:
        """Retrieve a speaker profile by account_id, or None if not found."""
        db = self._get_db()
        doc = db.collection("speaker_profiles").document(account_id).get()
        if doc.exists:
            return SpeakerProfile(**doc.to_dict())
        return None

    async def put_group(self, g: Group) -> None:
        """Store or update a group."""
        db = self._get_db()
        db.collection("groups").document(g.id).set(group_to_doc(g))

    async def get_group(self, group_id: str) -> Group | None:
        """Retrieve a group by id, or None if not found."""
        db = self._get_db()
        doc = db.collection("groups").document(group_id).get()
        if doc.exists:
            return Group(**doc.to_dict())
        return None

    async def list_groups(self, account_id: str) -> list[Group]:
        """List groups the account is a current member of, oldest-first by created_at."""
        db = self._get_db()
        # NB: this client version's operator token is underscore-style
        # ("array_contains"); the hyphen form raises ValueError at query time.
        docs = db.collection("groups").where("member_account_ids", "array_contains", account_id).stream()
        groups = [Group(**doc.to_dict()) for doc in docs]
        groups.sort(key=lambda g: g.created_at)
        return groups

    async def get_group_by_invite_code(self, code: str) -> Group | None:
        """Retrieve a group by an (unaccepted or accepted) invite code, or None if not found.

        Streams the collection and filters in Python (invite codes are few
        and this avoids a second index)."""
        db = self._get_db()
        for doc in db.collection("groups").stream():
            g = Group(**doc.to_dict())
            if any(inv.code == code for inv in g.invites):
                return g
        return None

    async def update_group_atomically(
        self, group_id: str, mutator: Callable[["Group | None"], Group]
    ) -> Group:
        """See LiveSessionStore.update_group_atomically. `@firestore.transactional`
        gives the read-check-write ACID semantics: the transaction retries
        the whole read+mutator+write if the doc changed underneath it, and
        commits nothing at all if `mutator` raises (Firestore never sees a
        `.set()` call in that case)."""
        from google.cloud import firestore

        db = self._get_db()
        ref = db.collection("groups").document(group_id)

        @firestore.transactional
        def _run(transaction):
            snapshot = ref.get(transaction=transaction)
            current = Group(**snapshot.to_dict()) if snapshot.exists else None
            new_group = mutator(current)
            transaction.set(ref, group_to_doc(new_group))
            return new_group

        return _run(db.transaction())

    async def put_capture(self, c: Capture) -> None:
        """Store or update a capture's metadata."""
        db = self._get_db()
        db.collection("captures").document(c.id).set(c.model_dump())

    async def get_capture(self, capture_id: str) -> Capture | None:
        """Retrieve a capture by id, or None if not found."""
        db = self._get_db()
        doc = db.collection("captures").document(capture_id).get()
        if doc.exists:
            return Capture(**doc.to_dict())
        return None

    async def list_captures(self, account_id: str) -> list[Capture]:
        """List captures owned by account_id, newest-first by captured_at."""
        db = self._get_db()
        docs = db.collection("captures").where("account_id", "==", account_id).stream()
        captures = [Capture(**doc.to_dict()) for doc in docs]
        captures.sort(key=lambda c: c.captured_at, reverse=True)
        return captures

    async def delete_capture(self, capture_id: str) -> None:
        """Remove a capture's metadata doc. Same idempotent-on-absent contract
        as delete_live_session — Firestore deletes never raise for an absent doc."""
        db = self._get_db()
        db.collection("captures").document(capture_id).delete()

    async def get_legacy_claim(self) -> LegacyClaim | None:
        """The singleton claim marker, or None if the legacy account is unclaimed."""
        db = self._get_db()
        doc = db.collection("legacy_claims").document(LEGACY_ACCOUNT_ID).get()
        if doc.exists:
            return LegacyClaim(**doc.to_dict())
        return None

    async def update_legacy_claim_atomically(
        self, mutator: Callable[["LegacyClaim | None"], LegacyClaim]
    ) -> LegacyClaim:
        """See LiveSessionStore.update_legacy_claim_atomically. Structural
        copy of update_group_atomically: `@firestore.transactional` gives
        the read-check-write ACID semantics, and a fixed singleton doc id
        (LEGACY_ACCOUNT_ID) since exactly one LegacyClaim ever exists."""
        from google.cloud import firestore

        db = self._get_db()
        ref = db.collection("legacy_claims").document(LEGACY_ACCOUNT_ID)

        @firestore.transactional
        def _run(transaction):
            snapshot = ref.get(transaction=transaction)
            current = LegacyClaim(**snapshot.to_dict()) if snapshot.exists else None
            new_claim = mutator(current)
            transaction.set(ref, new_claim.model_dump())
            return new_claim

        return _run(db.transaction())

    async def claim_legacy_live_session(self, live_session_id: str, account: str) -> bool:
        """See LiveSessionStore.claim_legacy_live_session. A single
        Firestore transaction spanning BOTH the live session doc and the
        singleton legacy_claims doc -- all reads happen before any write,
        matching every other transactional method here
        (update_group_atomically, update_legacy_claim_atomically), so this
        live session's ownership check, its move, and its counter-bump
        commit (or abort) together."""
        from google.cloud import firestore

        db = self._get_db()
        ls_ref = db.collection("live_sessions").document(live_session_id)
        claim_ref = db.collection("legacy_claims").document(LEGACY_ACCOUNT_ID)

        @firestore.transactional
        def _run(transaction):
            ls_snapshot = ls_ref.get(transaction=transaction)
            if not ls_snapshot.exists:
                return False
            ls = LiveSession(**ls_snapshot.to_dict())
            if ls.owner_account != LEGACY_ACCOUNT_ID:
                return False

            claim_snapshot = claim_ref.get(transaction=transaction)
            current = LegacyClaim(**claim_snapshot.to_dict()) if claim_snapshot.exists else None
            if current is None or current.account_id != account:
                raise HTTPException(status_code=409, detail="claim marker changed underneath this claim")

            ls.owner_account = account
            ls.shared_with = [a for a in ls.shared_with if a != account]
            transaction.set(ls_ref, live_session_to_doc(ls))
            transaction.set(claim_ref, current.model_copy(
                update={"episodes_moved_total": current.episodes_moved_total + 1}
            ).model_dump())
            return True

        return _run(db.transaction())


def get_store() -> LiveSessionStore:
    """Factory function to get the appropriate LiveSessionStore implementation.

    Reads MINDSHIFT_FIRESTORE_PROJECT env var. If set, uses
    FirestoreLiveSessionStore. Otherwise defaults to MemoryLiveSessionStore.
    """
    project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if project:
        return FirestoreLiveSessionStore(project)
    return MemoryLiveSessionStore()
