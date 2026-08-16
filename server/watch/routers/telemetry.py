# Ported from gauge@2157433 server/telemetry_api.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
#
# ADAPTED (Task B9): server.models.TelemetryEvent -> watch.models.
# TelemetryEvent; server.telemetry_store.TelemetryStore -> watch.
# telemetry_store.TelemetryStore. `make_telemetry_router`'s signature
# (`store: TelemetryStore) -> APIRouter`) and both route bodies are otherwise
# byte-for-byte -- this is the no-adb device debug channel, not a "clean up
# while you're here" port.
"""REST API: device→backend telemetry (crash/log reporting channel).

Companion to ``rest.py``, mounted separately because its only client is the
watch's own diagnostics path (see ``watch/telemetry_store.py``'s module
docstring) rather than the phone app's account-scoped surface.

SECURITY NOTE (inherited from gauge, unchanged here): both ``POST`` and
``GET /telemetry`` are DELIBERATELY unauthenticated -- this is the on-device
debugging channel that lets an agent ``curl`` a watch's crash/log history
directly, since there is no adb workflow for this device (see project
CLAUDE.md). That means ``GET /telemetry`` is an unauthenticated read of
whatever devices have POSTed (device ids, log messages, stack traces) to
anyone who can reach this server. This is an INHERITED risk carried over
verbatim from gauge, not a new decision made in this port. Token-gating
``GET /telemetry`` is a queued OWNER decision for a later phase, not
something this task changes -- behavior here is unchanged from gauge@2157433.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from typing import Awaitable, Callable

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from watch.models import TelemetryEvent
from watch.telemetry_store import TelemetryStore

# Batches larger than this are truncated (first N kept, rest reported as
# dropped) so one runaway device can't blow past Firestore write limits or
# dominate the store in a single request.
MAX_BATCH_EVENTS = 100

RateLimitDep = Callable[[Request], Awaitable[None]]


async def _noop_rate_limit(request: Request) -> None:
    """Default ``rate_limit_dep``: no limiting at all. Keeps
    ``create_watch_test_app`` and every existing ported test unaffected —
    only ``build_watch_routers()`` (server/watch/app.py) supplies the real
    per-IP limiter (Task H1)."""
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelemetryEventIn(BaseModel):
    level: str
    tag: str
    message: str
    stack: str | None = None
    ts: str


class TelemetryPost(BaseModel):
    device: str
    app_version: str
    events: list[TelemetryEventIn]


def make_telemetry_router(
    store: TelemetryStore, rate_limit_dep: RateLimitDep = _noop_rate_limit,
) -> APIRouter:
    """``rate_limit_dep`` (Task H1): both routes stay deliberately
    unauthenticated (owner decision — see module docstring) but now sit
    behind main.py's per-IP rate limiter in production, threaded in by
    ``build_watch_routers()``. Defaults to a no-op so
    ``create_watch_test_app`` and every existing ported test are unaffected.
    """
    router = APIRouter()

    @router.post("/telemetry")
    async def post_telemetry(
        body: TelemetryPost, _rl: None = Depends(rate_limit_dep),
    ) -> dict[str, int]:
        accepted, dropped_events = body.events[:MAX_BATCH_EVENTS], body.events[MAX_BATCH_EVENTS:]
        received_at = _now_iso()
        events = [
            TelemetryEvent(
                id=uuid.uuid4().hex,
                device=body.device,
                app_version=body.app_version,
                level=e.level,
                tag=e.tag,
                message=e.message,
                stack=e.stack,
                ts=e.ts,
                received_at=received_at,
            )
            for e in accepted
        ]
        await store.add_events(events)
        return {"stored": len(events), "dropped": len(dropped_events)}

    @router.get("/telemetry", response_model=list[TelemetryEvent])
    async def get_telemetry(
        device: str | None = Query(None),
        since: str | None = Query(None),
        limit: int = Query(200, ge=1, le=1000),
        _rl: None = Depends(rate_limit_dep),
    ) -> list[TelemetryEvent]:
        """List telemetry events, newest-first.

        ``since`` is INCLUSIVE (``received_at >= since``): cursor-polling
        with ``since=<last received_at>`` re-delivers that boundary event.
        """
        return await store.list_events(device, since, limit)

    return router
