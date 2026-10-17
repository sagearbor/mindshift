"""The library in a LIVE coaching session (WebSocket + POST /respond).

A session selects up to :data:`MAX_SELECTED` library items
(``library_item_ids``). Every coached turn asks :class:`LiveLibrary` for the
block to put in the prompt. The rule that matters: the library must never
stall or break coaching. Each turn waits at most :data:`TURN_TIMEOUT_S` for
the lookup; past that (or on any error) the turn is coached WITHOUT the
library and the miss is logged (ids/counts only, never titles or text).

Budgets
-------
``FULL_BUDGET_TOKENS`` (6,000): the selected items are sent whole when they
fit. In full mode the block is byte-identical turn after turn, goes first in
the system prompt and carries a prompt-cache marker, so after the first turn
it is a cache read: it costs little time-to-first-token. 6k tokens is about
4,500 words (a few pages of notes, a short brief). Bigger would mean a slower
FIRST (uncached) turn and a bigger cache write, and the 5-minute cache can
lapse in a quiet stretch of conversation.

``RETRIEVED_BUDGET_TOKENS`` (1,500): once the selection is known to be too
big for full mode, each turn gets the nearest excerpts within this smaller
budget. Retrieved excerpts change every turn, ride in the (uncached) user
message, and are re-processed on every call, so they are kept small.

The lookup is not cancelled on timeout: a slow first fetch (cold blob read,
embedding call) keeps running in the background and serves a later turn.
That is also how the session warms up: the fetch starts when the selection
is set, before the first turn.

Table FACTS
-----------
When the selection holds spreadsheets, every turn also asks
:func:`library.build_library_facts` for exact values computed from them
(library/tables.py — a deterministic matcher, no model call). It runs
concurrently with the block lookup and shares the SAME per-turn deadline
(one :data:`TURN_TIMEOUT_S` in total, not one each). A lookup that misses the
deadline keeps running and, if it produced facts, is served to the next turn
as long as its question is still among the latest two turns. A failure is
logged and ignored. Once the warm-up learns the selection has no table, no
further lookups are started. The facts change per turn, so they ride with
the turn (``LibraryContext.facts``), never inside the cached full block.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time

from library import build_library_context, build_library_facts
from library.models import LibraryContext, LibraryFacts

logger = logging.getLogger("library.live")

MAX_SELECTED = 20
MAX_ID_CHARS = 64
FULL_BUDGET_TOKENS = int(os.getenv("MINDSHIFT_LIBRARY_FULL_BUDGET", "6000"))
RETRIEVED_BUDGET_TOKENS = int(os.getenv("MINDSHIFT_LIBRARY_RETRIEVED_BUDGET", "1500"))
# Per-turn wait. Retrieval is an embedding call plus a vector query; ~300 ms
# is what a live turn can spare before the coach's own LLM call.
TURN_TIMEOUT_S = float(os.getenv("MINDSHIFT_LIBRARY_TIMEOUT_MS", "300")) / 1000.0
# Config-time ownership check (one store read per id). Slower than a turn is
# fine here; it only delays the config_ack.
VALIDATE_TIMEOUT_S = 2.0
# How long a full-mode block is reused before it is refreshed, so an edit to
# a selected item reaches a running session within a minute.
FULL_CACHE_TTL_S = 60.0
# How many recent turns feed the retrieval query.
RECENT_TURNS = 6


def validate_selection(value: object) -> list[str]:
    """Deduplicated non-blank ids (None → []). Raises ValueError, with a
    message fit to show the user, for a non-list, non-string entries or more
    than :data:`MAX_SELECTED` ids. Never truncates."""
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("library_item_ids must be a list of item ids")
    ids = list(dict.fromkeys(v.strip()[:MAX_ID_CHARS] for v in value if v.strip()))
    if len(ids) > MAX_SELECTED:
        raise ValueError(
            f"too many library items selected: {len(ids)}; the limit is {MAX_SELECTED}"
        )
    return ids


async def resolve_owned(uid: str, ids: list[str]) -> tuple[list[str], list[str]]:
    """``(accepted, ignored)``: ids that are ``uid``'s own items, and the rest
    (another user's, deleted, or made up — not distinguished, no existence
    oracle). If the store cannot be asked in time, every id is kept as
    accepted: the per-turn lookup is uid-scoped anyway, so a foreign id can
    never contribute text, and a store blip does not drop the selection."""
    if not ids:
        return [], []
    import library

    async def check() -> tuple[list[str], list[str]]:
        svc = library.get_service()
        accepted: list[str] = []
        ignored: list[str] = []
        for item_id in ids:
            rec = await svc.get_item(uid, item_id)
            (accepted if rec is not None else ignored).append(item_id)
        return accepted, ignored

    try:
        return await asyncio.wait_for(check(), VALIDATE_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — never fail the config frame
        logger.warning(
            "library selection check failed (%d ids), keeping them unverified: %s",
            len(ids), type(exc).__name__,
        )
        return list(ids), []


class LiveLibrary:
    """One session's selected items and its per-turn lookup state."""

    def __init__(self, uid: str, item_ids: list[str], *, warm: bool = True) -> None:
        self.uid = uid
        self.item_ids = list(item_ids)
        self._full: LibraryContext | None = None
        self._full_at = 0.0
        self._needs_retrieval = False
        self._inflight: asyncio.Task | None = None
        self._unavailable_logged = False
        # Table facts: None = unknown yet, False = no table selected (stop).
        self._has_tables: bool | None = None
        self._facts_warm: asyncio.Task | None = None
        self._facts_late: tuple[asyncio.Task, str] | None = None
        if self.item_ids and warm:
            self._start("")  # warm up before the first turn
            self._facts_warm = self._spawn_facts("")

    def _budget(self) -> int:
        return RETRIEVED_BUDGET_TOKENS if self._needs_retrieval else FULL_BUDGET_TOKENS

    def _start(self, recent_text: str) -> asyncio.Task | None:
        try:
            self._inflight = asyncio.ensure_future(
                build_library_context(self.uid, self.item_ids, recent_text, self._budget())
            )
        except RuntimeError:  # no running loop (sync construction in a test)
            self._inflight = None
        return self._inflight

    # -- table facts -----------------------------------------------------------

    def _spawn_facts(self, recent_text: str) -> asyncio.Task | None:
        try:
            return asyncio.ensure_future(
                build_library_facts(self.uid, self.item_ids, recent_text)
            )
        except RuntimeError:  # no running loop (sync construction in a test)
            return None

    def _facts_result(self, task: asyncio.Task) -> LibraryFacts | None:
        """A finished facts task's result; None (logged) on failure."""
        if task.cancelled():
            return None
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "library facts lookup failed (%d items); coaching without it: %s",
                len(self.item_ids), type(exc).__name__,
            )
            return None
        res = task.result()
        if res.has_tables is not None:
            self._has_tables = res.has_tables
        return res

    def _start_facts(self, recent_text: str) -> asyncio.Task | None:
        warm = self._facts_warm
        if warm is not None and warm.done():
            self._facts_result(warm)
            self._facts_warm = None
        if self._has_tables is False or not recent_text.strip():
            return None
        return self._spawn_facts(recent_text)

    async def _facts_for_turn(
        self, task: asyncio.Task | None, recent_text: str, deadline: float,
    ) -> str:
        late, self._facts_late = self._facts_late, None
        text = ""
        if task is not None:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                res = self._facts_result(task)
                text = res.text if res is not None else ""
            else:
                logger.warning(
                    "library facts lookup timed out (%d items); coaching without it",
                    len(self.item_ids),
                )
                self._facts_late = (task, _question_key(recent_text))
        if not text and late is not None:
            late_task, key = late
            if late_task.done() and key and key in _latest_turns(recent_text):
                res = self._facts_result(late_task)
                text = res.text if res is not None else ""
            elif not late_task.done() and self._facts_late is None:
                self._facts_late = late  # still running: keep it for the next turn
        return text

    # -- per turn ------------------------------------------------------------------

    async def for_turn(self, recent_text: str) -> LibraryContext | None:
        """The block for this turn, or ``None`` (nothing selected, nothing
        fits, lookup slow or failing). Waits at most :data:`TURN_TIMEOUT_S`
        in total, table facts included."""
        if not self.item_ids:
            return None
        deadline = asyncio.get_running_loop().time() + TURN_TIMEOUT_S
        facts_task = None
        try:
            facts_task = self._start_facts(recent_text)
        except Exception as exc:  # noqa: BLE001 — facts are optional
            logger.warning("library facts lookup could not start: %s", type(exc).__name__)
        ctx = await self._context_for_turn(recent_text, deadline)
        try:
            facts = await self._facts_for_turn(facts_task, recent_text, deadline)
        except Exception as exc:  # noqa: BLE001
            logger.warning("library facts lookup failed: %s", type(exc).__name__)
            facts = ""
        return _with_facts(ctx, facts)

    async def _context_for_turn(self, recent_text: str, deadline: float) -> LibraryContext | None:
        if self._full is not None and time.monotonic() - self._full_at < FULL_CACHE_TTL_S:
            return self._full
        task = self._inflight if self._inflight is not None else self._start(recent_text)
        if task is None:
            return None
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if not done:
            logger.warning(
                "library lookup timed out after %d ms (%d items); coaching without it",
                int(TURN_TIMEOUT_S * 1000), len(self.item_ids),
            )
            return self._full  # a stale full block beats none; usually None
        self._inflight = None
        try:
            ctx = task.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "library lookup failed (%d items); coaching without it: %s",
                len(self.item_ids), type(exc).__name__,
            )
            return self._full
        if ctx.mode == "full":
            if ctx.text:
                self._full, self._full_at = ctx, time.monotonic()
            return ctx if ctx.text else None
        self._needs_retrieval = True
        self._full = None
        if not ctx.text:
            if not self._unavailable_logged:
                self._unavailable_logged = True
                logger.warning(
                    "library %s for %d items: %s", ctx.mode, len(self.item_ids), ctx.reason,
                )
            return None
        return ctx

    def close(self) -> None:
        for task in (
            self._inflight, self._facts_warm,
            self._facts_late[0] if self._facts_late else None,
        ):
            if task is not None and not task.done():
                task.cancel()
        self._inflight = None
        self._facts_warm = None
        self._facts_late = None


def _latest_turns(text: str, n: int = 2) -> list[str]:
    return [t.strip() for t in (text or "").split("\n") if t.strip()][-n:]


def _question_key(text: str) -> str:
    latest = _latest_turns(text, 1)
    return latest[0] if latest else ""


def _with_facts(ctx: LibraryContext | None, facts: str) -> LibraryContext | None:
    """Attach this turn's facts without touching a cached full block: full
    mode keeps its byte-stable ``text`` and carries ``facts`` separately;
    retrieved excerpts (already per-turn) get the facts in front; with no
    library text at all the facts alone ride with the turn."""
    if not facts:
        return ctx
    if ctx is None or not ctx.text:
        return LibraryContext(text=facts, mode="retrieved")
    if ctx.mode == "full":
        return dataclasses.replace(ctx, facts=facts)
    return dataclasses.replace(ctx, text=facts + ctx.text)


def recent_text(texts: list[str]) -> str:
    """The retrieval query: the last :data:`RECENT_TURNS` turns' text."""
    return "\n".join(t for t in texts[-RECENT_TURNS:] if t and t.strip())
