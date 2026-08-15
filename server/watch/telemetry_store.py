# Ported from gauge@2157433 server/telemetry_store.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Device→backend telemetry storage.

Companion to ``store.py`` for Phase 3's crash/log reporting channel: watch
devices POST batches of ``TelemetryEvent``s so an agent can ``curl`` device
crashes/logs directly instead of relying on the user screen-capturing
logcat (there is no adb workflow for this device — see project CLAUDE.md).
"""

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
    return clamped


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
        db = self._get_db()
        query = db.collection("telemetry")
        if device is not None:
            query = query.where("device", "==", device)
        events = [TelemetryEvent(**doc.to_dict()) for doc in query.stream()]
        if since is not None:
            events = [e for e in events if e.received_at >= since]
        events.sort(key=_sort_key, reverse=True)
        return events[:limit]


def get_telemetry_store() -> TelemetryStore:
    """Factory function to get the appropriate TelemetryStore implementation.

    Reads MINDSHIFT_FIRESTORE_PROJECT env var. If set, uses
    FirestoreTelemetryStore. Otherwise defaults to MemoryTelemetryStore.
    """
    project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
    if project:
        return FirestoreTelemetryStore(project)
    return MemoryTelemetryStore()
