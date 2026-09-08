# Ported from gauge@2157433 server/telemetry_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Device→backend telemetry storage.

Companion to ``store.py`` for Phase 3's crash/log reporting channel: watch
devices POST batches of ``TelemetryEvent``s so an agent can ``curl`` device
crashes/logs directly instead of relying on the user screen-capturing
logcat (there is no adb workflow for this device — see project CLAUDE.md).
"""

import asyncio
import os
from typing import Protocol

from watch.models import TelemetryEvent

# Caps applied at the store boundary (clamp_event), not at the API layer, so
# every store implementation enforces them uniformly regardless of caller.
MAX_MESSAGE_CHARS = 4000
MAX_STACK_CHARS = 20000

_TRUNCATED_SUFFIX = "…[truncated]"


def _clamp_str(s: str, max_chars: int) -> str:
    """Truncate ``s`` to at most ``max_chars``, appending a marker when cut."""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - len(_TRUNCATED_SUFFIX)] + _TRUNCATED_SUFFIX


def clamp_event(e: TelemetryEvent) -> TelemetryEvent:
    """Return a copy of ``e`` with ``message``/``stack`` truncated to the caps.

    This is the persistence boundary helper: both store implementations call
    it from ``add_events`` so a misbehaving device (e.g. dumping a huge
    stack trace) can never blow past Firestore/document-size-friendly caps.
    """
    clamped = e.model_copy(deep=True)
    clamped.message = _clamp_str(clamped.message, MAX_MESSAGE_CHARS)
    if clamped.stack is not None:
        clamped.stack = _clamp_str(clamped.stack, MAX_STACK_CHARS)
    if clamped.data is not None:
        clamped.data = firestore_safe(clamped.data)
    return clamped


def firestore_safe(value):
    """Rewrite ``value`` so Firestore accepts it: an array may not directly
    contain another array ("Property data contains an invalid nested
    entity" — the 500 the phone's ``device_diarization`` event hit on
    2026-08-30 with ``segments: [[start, end, label], ...]``). A list nested
    directly in a list becomes an object: a ``[number, number, label]``
    triple becomes ``{"start", "end", "label"}`` (the shape a segment reader
    wants), anything else ``{"items": [...]}``. Dicts and scalars pass
    through; the rewrite recurses so deeper nests are covered too."""
    if isinstance(value, dict):
        return {str(k): firestore_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)):
                if (
                    len(item) == 3
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in item[:2])
                    and isinstance(item[2], (str, int))
                ):
                    out.append({"start": item[0], "end": item[1], "label": item[2]})
                else:
                    out.append({"items": firestore_safe(list(item))})
            else:
                out.append(firestore_safe(item))
        return out
    return value


def _event_account(e: TelemetryEvent) -> str | None:
    """The account id a diagnostics event names in its payload, or ``None``.

    The phone's ``DiagnosticsPayload`` carries ``uid`` at the top level of
    ``data``; watch events have no ``data`` at all. Pure — the memory store
    and (conceptually) the Firestore ``data.uid`` query share this one
    definition of "whose event is this"."""
    data = e.data
    if not isinstance(data, dict):
        return None
    uid = data.get("uid")
    return uid if isinstance(uid, str) and uid else None


def _sort_key(e: TelemetryEvent) -> tuple[str, str]:
    """Newest-first ordering: by ``received_at``, then ``ts`` as a tiebreak."""
    return (e.received_at, e.ts)


class TelemetryStore(Protocol):
    """Protocol for telemetry event storage backend."""

    async def add_events(self, events: list[TelemetryEvent]) -> None:
        """Store a batch of telemetry events (message/stack clamped to caps)."""
        ...

    async def list_events(self, device: str | None, since: str | None, limit: int) -> list[TelemetryEvent]:
        """List events, newest-first by received_at then ts.

        ``device`` filters to a single device id when given. ``since``
        filters to events with ``received_at >= since`` when given.
        Result is capped to ``limit`` entries.
        """
        ...

    async def delete_events_for_account(self, account_id: str) -> int:
        """Delete every diagnostics event this account's own phone sent —
        those whose structured payload carries ``data.uid == account_id``
        (apps/mobile/src/diagnostics/diagnostics.ts's ``DiagnosticsPayload``).
        Returns how many were deleted.

        Deliberately keyed on the payload uid and NOT on ``device``: the
        device id a phone sends is ``phone:<platform>:<uid>``, which embeds
        the uid as a SUFFIX no store can query by, and watch-sent events carry
        a hardware device id with no account in it at all. Events with no
        ``data.uid`` (every watch event) are not this account's to delete —
        they identify a device, never a person, and are left alone."""
        ...


class MemoryTelemetryStore:
    """In-memory implementation of TelemetryStore for testing and default runtime."""

    def __init__(self):
        self._events: list[TelemetryEvent] = []

    async def add_events(self, events: list[TelemetryEvent]) -> None:
        """Store a batch of telemetry events (message/stack clamped to caps)."""
        for e in events:
            self._events.append(clamp_event(e))

    async def list_events(self, device: str | None, since: str | None, limit: int) -> list[TelemetryEvent]:
        """List events, newest-first by received_at then ts."""
        events = self._events
        if device is not None:
            events = [e for e in events if e.device == device]
        if since is not None:
            events = [e for e in events if e.received_at >= since]
        events = sorted(events, key=_sort_key, reverse=True)
        return [e.model_copy(deep=True) for e in events[:limit]]

    async def delete_events_for_account(self, account_id: str) -> int:
        """See TelemetryStore.delete_events_for_account."""
        keep = [e for e in self._events if _event_account(e) != account_id]
        removed = len(self._events) - len(keep)
        self._events = keep
        return removed


class FirestoreTelemetryStore:
    """Firestore-backed implementation of TelemetryStore. Lazily imports google-cloud-firestore."""

    def __init__(self, project: str):
        self.project = project
        self._db = None

    def _get_db(self):
        """Lazily import and initialize Firestore client."""
        if self._db is None:
            from google.cloud import firestore
            self._db = firestore.Client(project=self.project)
        return self._db

    async def add_events(self, events: list[TelemetryEvent]) -> None:
        """Store a batch of telemetry events (message/stack clamped to caps)."""
        await asyncio.to_thread(self._add_events_sync, events)

    def _add_events_sync(self, events: list[TelemetryEvent]) -> None:
        db = self._get_db()
        for e in events:
            clamped = clamp_event(e)
            db.collection("telemetry").document(clamped.id).set(clamped.model_dump())

    async def list_events(self, device: str | None, since: str | None, limit: int) -> list[TelemetryEvent]:
        """List events, newest-first by received_at then ts.

        Only ``device`` is pushed down to Firestore's query (``.where``);
        ``since`` filtering and the newest-first sort/limit happen in
        Python — house style (see FirestoreLiveSessionStore.list_live_sessions
        in store.py), which avoids needing a composite index.
        """
        return await asyncio.to_thread(self._list_events_sync, device, since, limit)

    def _list_events_sync(self, device: str | None, since: str | None, limit: int) -> list[TelemetryEvent]:
        db = self._get_db()
        query = db.collection("telemetry")
        if device is not None:
            query = query.where("device", "==", device)
        events = [TelemetryEvent(**doc.to_dict()) for doc in query.stream()]
        if since is not None:
            events = [e for e in events if e.received_at >= since]
        events.sort(key=_sort_key, reverse=True)
        return events[:limit]

    async def delete_events_for_account(self, account_id: str) -> int:
        """See TelemetryStore.delete_events_for_account."""
        return await asyncio.to_thread(self._delete_events_for_account_sync, account_id)

    def _delete_events_for_account_sync(self, account_id: str) -> int:
        # An equality query on the nested payload field (Firestore's automatic
        # single-field indexes cover nested paths, so this needs no composite
        # index), then a delete per matching doc — same shape as
        # FirestorePairingStore._delete_device_tokens_for_account_sync, and
        # scoped to this account's own events rather than a collection scan.
        db = self._get_db()
        docs = db.collection("telemetry").where("data.uid", "==", account_id).stream()
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1
        return count


def get_telemetry_store() -> TelemetryStore:
    """Factory function to get the appropriate TelemetryStore implementation.

    Reads MINDSHIFT_FIRESTORE_PROJECT env var. If set, uses
    FirestoreTelemetryStore. Otherwise defaults to MemoryTelemetryStore.
    """
    project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if project:
        return FirestoreTelemetryStore(project)
    return MemoryTelemetryStore()
