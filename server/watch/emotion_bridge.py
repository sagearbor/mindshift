# Emotion dev tester (2026-11-07) — see docs/plans/2026-11-07-emotion-dev-mode.md.
"""Phone <-> watch bridge for the owner-only Emotion dev tester.

The phone runs two on-device emotion models and decides on a buzz; the watch
only shows the latest emotion and offers pause/flag buttons. The transport is
the websocket the watch's companion mode already holds open, so nothing new
needs to run on the watch's network side:

* phone -> watch: ``POST /emotion/event`` (phone bearer auth) ->
  :func:`push_emotion` -> an ``{"type": "emotion", ...}`` frame down the
  account's live watch socket (``watch/relay.py``'s registry).
* watch -> phone: the watch sends ``{"type": "emotion_ctl", ...}`` up the same
  socket; ws.py hands it to :func:`push_ctl`; the phone drains the queue with
  ``GET /emotion/ctl?after=<seq>``.

Privacy: this carries a class label, a probability and a pause/flag verb. No
audio, no transcript, nothing persisted — the queue is in memory, bounded, and
expires. Like the rest of the relay it is per-process (see relay.py): a
multi-instance deployment would need sticky routing, which the shipped relay
already requires.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from watch.relay import live_session_for, _schedule_coro

EMOTION_LABELS = ("anger", "disgust", "fear", "happiness", "neutral", "sadness")
FLAGS = ("bad", "good", "ignore")
ACTIONS = ("pause", "resume", "flag")

MAX_QUEUE = 100
QUEUE_TTL_S = 15 * 60


class EmotionEvent(BaseModel):
    """The phone's state for the wrist (mirrors apps/mobile/src/emotion/watchLink.ts)."""

    v: int = 1
    id: str = Field(default="", max_length=64)
    t: float
    label: str = Field(max_length=16)
    prob: float = Field(ge=0.0, le=1.0)
    nudge: bool = False
    paused: bool = False
    suggestion: str | None = Field(default=None, max_length=80)


class CtlQueue:
    """Bounded, expiring, per-account queue of watch controls with a monotonic seq."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}
        self._items: dict[str, deque[dict[str, Any]]] = {}

    def push(self, account: str, item: dict[str, Any]) -> int:
        with self._lock:
            seq = self._seq.get(account, 0) + 1
            self._seq[account] = seq
            q = self._items.setdefault(account, deque(maxlen=MAX_QUEUE))
            q.append({**item, "seq": seq, "t": self._clock()})
            return seq

    def since(self, account: str, after: int) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            q = self._items.get(account)
            if not q:
                return []
            while q and now - q[0]["t"] > QUEUE_TTL_S:
                q.popleft()
            return [dict(i) for i in q if i["seq"] > after]

    def clear(self) -> None:
        with self._lock:
            self._seq.clear()
            self._items.clear()


_queue = CtlQueue()


def reset_for_tests() -> None:
    _queue.clear()


def push_emotion(uid: str, event: EmotionEvent) -> bool:
    """Send one state to ``uid``'s live watch socket. False when no watch is live."""
    session = live_session_for(uid)
    if session is None or session.send_raw is None:
        return False
    frame = {"type": "emotion", **event.model_dump()}
    _schedule_coro(session, session.send_raw(frame))
    return True


def parse_ctl(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a watch ``emotion_ctl`` frame; None for anything malformed."""
    action = payload.get("action")
    if action not in ACTIONS:
        return None
    item: dict[str, Any] = {"action": action}
    if action == "flag":
        flag = payload.get("flag")
        if flag not in FLAGS:
            return None
        item["flag"] = flag
        nid = payload.get("nudge_id")
        if isinstance(nid, str) and len(nid) <= 64:
            item["nudge_id"] = nid
    return item


def push_ctl(account: str, payload: dict[str, Any]) -> bool:
    item = parse_ctl(payload)
    if item is None:
        return False
    _queue.push(account, item)
    return True


def make_emotion_router(auth_dep) -> APIRouter:
    router = APIRouter()

    @router.post("/emotion/event")
    async def post_event(event: EmotionEvent, principal=Depends(auth_dep)) -> dict[str, bool]:
        return {"delivered": push_emotion(principal.account_id, event)}

    @router.get("/emotion/ctl")
    async def get_ctl(after: int = Query(0, ge=0), principal=Depends(auth_dep)) -> dict[str, Any]:
        items = _queue.since(principal.account_id, after)
        return {"items": items}

    return router
