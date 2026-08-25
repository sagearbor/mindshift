# Ported from gauge@2157433 server/pairing_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Device-pairing storage: the ephemeral short-code handshake (``Pairing``)
and the long-lived credential it mints (``DeviceToken``) — see
docs/superpowers/plans/2026-08-04-gauge-wave-c-couples-wrist.md's Open
Question 1 and server/watch/routers/pairing.py's (Task B8) module docstring
for the full flow this is the storage half of.

Hash-at-rest contract: the human-typeable pairing CODE is never stored
except as a SHA-256 hash (``Pairing.code_hash``); the long-lived DEVICE
TOKEN is never stored except as a SHA-256 hash (``DeviceToken.token_hash``).
The one deliberate, narrow exception is ``Pairing.device_token``: the raw
token must be handed to the watch at least once (it polls
``GET /me/pair/status``), so it is held in plaintext ONLY on the ephemeral
``Pairing`` record, ONLY until ``expires_at`` (the same short TTL as the
whole pairing handshake, ~10 minutes per server/watch/routers/pairing.py's
(Task B8) PAIRING_TTL_MINUTES) — server/watch/routers/pairing.py's status
handler refuses to return it once that TTL has passed, even if the record
itself still lingers in storage (no background cleanup job is implemented;
this module never actively deletes expired records).

INHERITED WART (kept for parity, not fixed here): update_pairing_atomically's
mutator (like store.py's update_group_atomically/
update_legacy_claim_atomically mutators) raises fastapi.HTTPException
directly from the storage layer -- Gauge's original design, preserved
verbatim.
"""

import asyncio
import hashlib
import os
import threading
from typing import Callable, Protocol

from watch.models import DeviceToken, FailedClaimRecord, Pairing


def hash_secret(raw: str) -> str:
    """SHA-256 hex digest — the one hashing primitive both the pairing code
    and the device token use for at-rest storage / lookup-by-hash.

    Not a password hash (no salt/stretching): both secrets are high-entropy,
    server-generated, single-purpose random values (never a user-chosen,
    low-entropy password), so a fast cryptographic hash is the right tool —
    matches the threat model of e.g. an API-key hash, not a login password.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PairingStore(Protocol):
    """Protocol for device-pairing storage."""

    async def create_pairing(self, p: Pairing) -> None:
        """Store a brand-new pairing record."""
        ...

    async def get_pairing(self, pairing_id: str) -> Pairing | None:
        """Retrieve a pairing by its id, or None if not found."""
        ...

    async def get_pairing_by_code_hash(self, code_hash: str) -> Pairing | None:
        """Retrieve a pairing by its code's hash, or None if not found."""
        ...

    async def update_pairing_atomically(
        self, pairing_id: str, mutator: Callable[["Pairing | None"], Pairing]
    ) -> Pairing:
        """Read-check-write a pairing with no other writer able to
        interleave — same seam contract as
        LiveSessionStore.update_group_atomically (server/watch/store.py):
        ``mutator`` sees a fresh consistent read (None if absent), returns
        the Pairing to persist, or raises to abort the whole operation with
        NOTHING persisted. This is what makes single-use code claiming safe
        against two concurrent claims of the same code racing each other.
        """
        ...

    def put_device_token(self, t: DeviceToken) -> None:
        """Store a new device token record.

        SYNCHRONOUS (not async) — unlike every other method on this
        Protocol. server/watch/auth.py's (Task B3) DeviceTokenVerifier.verify()
        is called from the synchronous TokenVerifier.verify() Protocol
        method (resolve_principal, its only caller, is itself a plain
        synchronous function — see server/watch/auth.py's (Task B3) module
        docstring), so it cannot await anything. Both implementations'
        underlying I/O (a dict write
        / a Firestore client call) is already effectively synchronous even
        where OTHER store methods here wrap the identical kind of call in
        ``async def`` purely for consistency with FastAPI's async handlers
        — see FirestoreLiveSessionStore in server/watch/store.py, whose
        "async" methods make the same blocking calls under the hood.
        """
        ...

    def get_device_token_by_hash(self, token_hash: str) -> DeviceToken | None:
        """Retrieve a device token by its hash, or None if not found.
        SYNCHRONOUS — see put_device_token's docstring for why."""
        ...

    async def get_failed_claim_record(self, account_id: str) -> FailedClaimRecord | None:
        """Raw current failed-``POST /me/pair/claim`` record for
        ``account_id`` (None if it has never failed a claim). Pure read, no
        time-math — whether this record currently means "locked out" is a
        window-expiry policy decision made by server/watch/routers/pairing.py
        (Task B8) against an injected clock (see ``FailedClaimRecord``'s
        docstring and ``routers.pairing._is_account_locked`` (Task B8)),
        not by this store.

        FIX ROUND 2: keyed by the ALREADY-AUTHENTICATED calling account
        (``principal.account_id``, always known since ``POST /me/pair/claim``
        requires ``full_auth``), REPLACING FIX ROUND 1's
        ``list_pending_pairings``-based per-pairing attribution heuristic —
        see server/watch/routers/pairing.py's (Task B8) module docstring for
        why that was defeatable (a free, unauthenticated decoy ``POST /me/pair/start``
        pairing made it permanently no-op). Keying by account instead of an
        ambiguous target pairing sidesteps the SHA-256 attribution problem
        entirely and is immune to decoy pairings.
        """
        ...

    async def set_failed_claim_record(self, account_id: str, count: int, last_failed_at: str) -> None:
        """Persist (create or fully replace) the failed-claim record for
        ``account_id``. A plain write, not a compare-and-swap: the caller
        (server/watch/routers/pairing.py's (Task B8) ``_record_failed_claim``) does its own
        read-then-decide first. This is a deliberate, reviewer-accepted
        narrow check-then-act race (FIX ROUND 3: two concurrent failures at
        the exact threshold could both read the same pre-increment count) —
        bounded in impact (worst case, the lockout triggers one attempt
        later than exactly ``MAX_FAILED_CLAIM_ATTEMPTS_PER_ACCOUNT``), so no
        atomic-mutator seam (like ``update_pairing_atomically``) was added
        for this method.
        """
        ...

    async def has_device_tokens_for_account(self, account_id: str) -> bool:
        """Whether ANY device token is currently bound to ``account_id`` —
        the ``has_paired_watch`` signal ``GET /me`` (server/watch/routers/rest.py,
        Task P3-6) exposes to the signed-in phone/web caller so "Set up your
        watch" can show live paired state instead of guessing. True as soon
        as a single watch has completed ``POST /me/pair/claim`` into this
        account (``pairing.py``'s ``pair_claim`` stamps
        ``DeviceToken.account_id = principal.account_id`` — the same account
        id a ``Principal``/``/me`` caller is keyed by), and stays True even
        if that specific token later goes stale — there is no revocation/
        expiry flow for device tokens yet, so this is honestly "has this
        account EVER completed a pairing claim", not "is a watch currently
        online"."""
        ...

    async def delete_device_tokens_for_account(self, account_id: str) -> int:
        """Delete every ``DeviceToken`` bound to ``account_id`` — the
        unpair/disconnect primitive behind ``DELETE /me/watch-pairing``
        (server/watch/routers/rest.py). Returns the number of tokens
        actually deleted (0 is a valid, non-error result — idempotent, same
        house style as ``forgetVoice``'s ``{"deleted": bool}`` and
        ``deleteVoiceSample``'s remaining-count response).

        This ONLY revokes the watch's ability to authenticate as the
        account (``DeviceToken`` is pure auth-linkage — see its docstring:
        ``{token_hash, account_id, created_at, pairing_id}``, no reference
        to any real data). Recordings, growth, live sessions, the speaker
        profile, everything else keyed by ``account_id`` is completely
        untouched; a fresh ``POST /me/pair/claim`` immediately sees all the
        same cloud data again."""
        ...

    # -- account deletion (DELETE /me, server/account_deletion.py) ----------

    async def delete_pairings_for_account(self, account_id: str) -> int:
        """Delete every ``Pairing`` record CLAIMED by ``account_id``. Returns
        how many were deleted (0 is a valid result).

        Pairings are short-lived handshake records, but a claimed one keeps
        ``claimed_account_id`` — and, inside its own TTL, the raw device token
        — so an account deletion must take them with it rather than wait for
        expiry. Unclaimed pairings carry no account id at all and are left
        alone: they belong to nobody yet."""
        ...

    async def delete_failed_claim_record(self, account_id: str) -> bool:
        """Remove the account's brute-force circuit-breaker counter. True when
        one existed."""
        ...


class MemoryPairingStore:
    """In-memory implementation of PairingStore for testing and default runtime."""

    def __init__(self):
        self._pairings: dict[str, Pairing] = {}
        self._device_tokens: dict[str, DeviceToken] = {}
        self._lock = threading.Lock()
        self._failed_claims: dict[str, FailedClaimRecord] = {}
        self._failed_claims_lock = threading.Lock()

    async def create_pairing(self, p: Pairing) -> None:
        self._pairings[p.id] = p.model_copy(deep=True)

    async def get_pairing(self, pairing_id: str) -> Pairing | None:
        p = self._pairings.get(pairing_id)
        return p.model_copy(deep=True) if p else None

    async def get_pairing_by_code_hash(self, code_hash: str) -> Pairing | None:
        for p in self._pairings.values():
            if p.code_hash == code_hash:
                return p.model_copy(deep=True)
        return None

    def _read_pairing_locked(self, pairing_id: str) -> Pairing | None:
        """The read half of update_pairing_atomically, split out so tests
        could override it to widen the critical section — mirrors
        MemoryLiveSessionStore._read_group_locked's exact purpose."""
        p = self._pairings.get(pairing_id)
        return p.model_copy(deep=True) if p else None

    async def update_pairing_atomically(
        self, pairing_id: str, mutator: Callable[["Pairing | None"], Pairing]
    ) -> Pairing:
        """See PairingStore.update_pairing_atomically. threading.Lock (not
        asyncio.Lock) for the same real-OS-thread-concurrency reason as
        MemoryLiveSessionStore.update_group_atomically."""
        with self._lock:
            current = self._read_pairing_locked(pairing_id)
            new_pairing = mutator(current)
            self._pairings[new_pairing.id] = new_pairing.model_copy(deep=True)
            return new_pairing.model_copy(deep=True)

    def put_device_token(self, t: DeviceToken) -> None:
        self._device_tokens[t.token_hash] = t.model_copy(deep=True)

    def get_device_token_by_hash(self, token_hash: str) -> DeviceToken | None:
        t = self._device_tokens.get(token_hash)
        return t.model_copy(deep=True) if t else None

    async def get_failed_claim_record(self, account_id: str) -> FailedClaimRecord | None:
        r = self._failed_claims.get(account_id)
        return r.model_copy(deep=True) if r else None

    async def set_failed_claim_record(self, account_id: str, count: int, last_failed_at: str) -> None:
        with self._failed_claims_lock:
            self._failed_claims[account_id] = FailedClaimRecord(
                account_id=account_id, count=count, last_failed_at=last_failed_at,
            )

    async def has_device_tokens_for_account(self, account_id: str) -> bool:
        return any(t.account_id == account_id for t in self._device_tokens.values())

    async def delete_device_tokens_for_account(self, account_id: str) -> int:
        # No lock, matching put_device_token/get_device_token_by_hash's own
        # unguarded dict access above — device token storage was never given
        # the same threading.Lock protection as _pairings in this store.
        matching = [h for h, t in self._device_tokens.items() if t.account_id == account_id]
        for h in matching:
            del self._device_tokens[h]
        return len(matching)

    # -- account deletion --------------------------------------------------

    async def delete_pairings_for_account(self, account_id: str) -> int:
        with self._lock:
            matching = [
                pid for pid, p in self._pairings.items()
                if p.claimed_account_id == account_id
            ]
            for pid in matching:
                del self._pairings[pid]
            return len(matching)

    async def delete_failed_claim_record(self, account_id: str) -> bool:
        with self._failed_claims_lock:
            return self._failed_claims.pop(account_id, None) is not None


class FirestorePairingStore:
    """Firestore-backed implementation of PairingStore. Lazily imports
    google-cloud-firestore, matching FirestoreLiveSessionStore/
    FirestoreTelemetryStore's exact pattern."""

    def __init__(self, project: str):
        self.project = project
        self._db = None

    def _get_db(self):
        if self._db is None:
            from google.cloud import firestore
            self._db = firestore.Client(project=self.project)
        return self._db

    async def create_pairing(self, p: Pairing) -> None:
        await asyncio.to_thread(self._create_pairing_sync, p)

    def _create_pairing_sync(self, p: Pairing) -> None:
        db = self._get_db()
        db.collection("pairings").document(p.id).set(p.model_dump())

    async def get_pairing(self, pairing_id: str) -> Pairing | None:
        return await asyncio.to_thread(self._get_pairing_sync, pairing_id)

    def _get_pairing_sync(self, pairing_id: str) -> Pairing | None:
        db = self._get_db()
        doc = db.collection("pairings").document(pairing_id).get()
        if doc.exists:
            return Pairing(**doc.to_dict())
        return None

    async def get_pairing_by_code_hash(self, code_hash: str) -> Pairing | None:
        """Streams the collection and filters in Python — pairings are few
        and short-lived, matching get_group_by_invite_code's same tradeoff
        (avoids needing a composite index for a low-cardinality lookup)."""
        return await asyncio.to_thread(self._get_pairing_by_code_hash_sync, code_hash)

    def _get_pairing_by_code_hash_sync(self, code_hash: str) -> Pairing | None:
        db = self._get_db()
        for doc in db.collection("pairings").stream():
            p = Pairing(**doc.to_dict())
            if p.code_hash == code_hash:
                return p
        return None

    async def update_pairing_atomically(
        self, pairing_id: str, mutator: Callable[["Pairing | None"], Pairing]
    ) -> Pairing:
        """See PairingStore.update_pairing_atomically.
        ``@firestore.transactional`` gives the read-check-write ACID
        semantics: the transaction retries the whole read+mutator+write if
        the doc changed underneath it, and commits nothing at all if
        ``mutator`` raises (Firestore never sees a ``.set()`` call in that
        case) — structural copy of FirestoreLiveSessionStore.update_group_atomically.
        The entire transactional call runs in a worker thread via
        asyncio.to_thread, so a mutator-raised exception (including the
        HTTPException abort path) propagates unchanged to the awaiting
        caller."""
        return await asyncio.to_thread(self._update_pairing_atomically_sync, pairing_id, mutator)

    def _update_pairing_atomically_sync(
        self, pairing_id: str, mutator: Callable[["Pairing | None"], Pairing]
    ) -> Pairing:
        from google.cloud import firestore

        db = self._get_db()
        ref = db.collection("pairings").document(pairing_id)

        @firestore.transactional
        def _run(transaction):
            snapshot = ref.get(transaction=transaction)
            current = Pairing(**snapshot.to_dict()) if snapshot.exists else None
            new_pairing = mutator(current)
            transaction.set(ref, new_pairing.model_dump())
            return new_pairing

        return _run(db.transaction())

    def put_device_token(self, t: DeviceToken) -> None:
        # SYNCHRONOUS by Protocol contract (see PairingStore.put_device_token's
        # docstring) — auth.py's DeviceTokenVerifier calls this from a plain
        # sync function that cannot await, so this stays as a direct
        # blocking SDK call and is NOT wrapped in asyncio.to_thread.
        db = self._get_db()
        db.collection("device_tokens").document(t.token_hash).set(t.model_dump())

    def get_device_token_by_hash(self, token_hash: str) -> DeviceToken | None:
        # SYNCHRONOUS by Protocol contract — see put_device_token above and
        # server/watch/auth.py's DeviceTokenVerifier.verify(), which calls
        # this synchronously. Left as a direct blocking SDK call, matching
        # the ONLY-wrap-async-def-methods scope of this task.
        db = self._get_db()
        doc = db.collection("device_tokens").document(token_hash).get()
        if doc.exists:
            return DeviceToken(**doc.to_dict())
        return None

    async def get_failed_claim_record(self, account_id: str) -> FailedClaimRecord | None:
        return await asyncio.to_thread(self._get_failed_claim_record_sync, account_id)

    def _get_failed_claim_record_sync(self, account_id: str) -> FailedClaimRecord | None:
        db = self._get_db()
        doc = db.collection("failed_claim_attempts").document(account_id).get()
        if doc.exists:
            return FailedClaimRecord(**doc.to_dict())
        return None

    async def set_failed_claim_record(self, account_id: str, count: int, last_failed_at: str) -> None:
        """FIX ROUND 3: full replace (not ``firestore.Increment``) — the
        caller now sometimes needs to RESET the count to 1 (window expired),
        not just increment it, so the atomic-increment field transform no
        longer fits; see ``PairingStore.set_failed_claim_record``'s
        docstring for why a plain (non-transactional) write is acceptable
        here."""
        await asyncio.to_thread(
            self._set_failed_claim_record_sync, account_id, count, last_failed_at
        )

    def _set_failed_claim_record_sync(self, account_id: str, count: int, last_failed_at: str) -> None:
        db = self._get_db()
        db.collection("failed_claim_attempts").document(account_id).set(
            FailedClaimRecord(account_id=account_id, count=count, last_failed_at=last_failed_at).model_dump()
        )

    async def has_device_tokens_for_account(self, account_id: str) -> bool:
        return await asyncio.to_thread(self._has_device_tokens_for_account_sync, account_id)

    def _has_device_tokens_for_account_sync(self, account_id: str) -> bool:
        # A bounded existence query (limit(1)), not a full-collection stream
        # like get_pairing_by_code_hash's — device_tokens is a from-scratch
        # collection with no low-cardinality guarantee, and this call sits on
        # GET /me's hot path (every authenticated phone/web request), unlike
        # the pairing lookups above which run once per claim attempt.
        db = self._get_db()
        docs = db.collection("device_tokens").where("account_id", "==", account_id).limit(1).stream()
        return next(iter(docs), None) is not None

    async def delete_device_tokens_for_account(self, account_id: str) -> int:
        return await asyncio.to_thread(self._delete_device_tokens_for_account_sync, account_id)

    def _delete_device_tokens_for_account_sync(self, account_id: str) -> int:
        # Unlike _has_device_tokens_for_account_sync's limit(1) existence
        # check, this needs every matching doc (there is no bulk
        # delete-by-query in the Firestore client) -- an unpair is a rare,
        # explicit user action (unlike the GET /me hot path above), so an
        # unbounded stream over this account's own tokens is the right
        # tradeoff, matching get_pairing_by_code_hash's same reasoning.
        db = self._get_db()
        docs = db.collection("device_tokens").where("account_id", "==", account_id).stream()
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1
        return count

    # -- account deletion --------------------------------------------------

    async def delete_pairings_for_account(self, account_id: str) -> int:
        return await asyncio.to_thread(self._delete_pairings_for_account_sync, account_id)

    def _delete_pairings_for_account_sync(self, account_id: str) -> int:
        # Same reasoning as _delete_device_tokens_for_account_sync: an equality
        # query scoped to THIS account's own claimed pairings, streamed and
        # deleted one by one (no bulk delete-by-query exists), on a rare and
        # explicit user action rather than a hot path.
        db = self._get_db()
        docs = (
            db.collection("pairings")
            .where("claimed_account_id", "==", account_id)
            .stream()
        )
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1
        return count

    async def delete_failed_claim_record(self, account_id: str) -> bool:
        return await asyncio.to_thread(self._delete_failed_claim_record_sync, account_id)

    def _delete_failed_claim_record_sync(self, account_id: str) -> bool:
        db = self._get_db()
        ref = db.collection("failed_claim_attempts").document(account_id)
        if not ref.get().exists:
            return False
        ref.delete()
        return True


def get_pairing_store() -> PairingStore:
    """Factory function to get the appropriate PairingStore implementation.

    Reads MINDSHIFT_FIRESTORE_PROJECT env var — same convention as
    get_store()/get_telemetry_store(). If set, uses FirestorePairingStore.
    Otherwise defaults to MemoryPairingStore.
    """
    project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if project:
        return FirestorePairingStore(project)
    return MemoryPairingStore()
