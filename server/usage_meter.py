"""Per-uid usage accounting and soft daily quotas — the cost guardrails.

Why this exists: every coached turn is an Anthropic call, every live minute is
a Deepgram minute, every phone that has never held the model pulls 80 MB out of
GCS. Before this module nothing in the server bounded what ONE account could
spend in a day, and nothing told the owner where the bill came from. If the
product reaches real therapists, that is an unbounded liability on somebody's
personal credit card.

Three pieces, deliberately small:

1. **Counters** — per uid, per UTC day, a flat ``{key: number}`` dict. LLM
   tokens are keyed BY CALL SITE (``llm.live_suggestion.input_tokens``) so the
   bill can be attributed to a feature, not just to a user. Everything else is
   a plain unit counter (``stt.seconds``, ``live.minutes``, ``model.downloads``,
   ``calls.started``). Recording is an in-memory dict add under a lock —
   nanoseconds, safe to call from the per-utterance hot path — and a background
   flusher persists it to the recordings store every
   :data:`FLUSH_INTERVAL_S`. Accounting must NEVER fail a request: every store
   interaction is best-effort and logged, never raised.

2. **Quotas** — generous, env-tunable daily caps that DEGRADE rather than
   break. Over the cloud-coaching cap the live session stops calling the LLM
   but keeps transcribing, keeps relaying the on-device fast loop, and sends
   ONE ``quota_notice`` frame saying what stopped and when it resets. Nothing
   is ever dropped silently.

3. **Attribution** — the LLM call sites are spread across main.py, the routers
   and audio_pipeline, and they all share ONE process-wide
   :class:`~llm_client.LLMClient`. Threading a uid through every signature
   would be a huge diff, so the uid + call site ride a :class:`ContextVar`
   (:func:`attribute` / :func:`usage_scope`) that ``LLMClient._record_usage``
   reads. ``asyncio.to_thread`` and ``asyncio.create_task`` both COPY the
   current context, so an LLM call offloaded to a worker thread (every call in
   this codebase) or a spawned analysis job still lands on the right uid.

Accuracy contract (stated honestly rather than implied):

* Counters are **best-effort**. Up to :data:`FLUSH_INTERVAL_S` of usage is lost
  if the process dies, and usage recorded while the store is unconfigured lives
  only in memory. They are a spend signal, not an invoice.
* Quotas are enforced from THIS process's view of the day: its own counters
  plus a snapshot of other instances' shards refreshed every
  :data:`SEED_TTL_S`. With N Cloud Run instances a determined user can overrun
  a cap by roughly one refresh window's spend before every instance sees it.
  That is a deliberate trade: a synchronous read-modify-write per utterance
  would put GCS in the coaching hot path.

Storage layout (one blob per process per uid per day — see
``recordings_store.write_usage_shard``)::

    usage/{YYYY-MM-DD}/{uid}/{instance_id}.json

Each process owns its own shard exclusively, so there is no read-modify-write
race and no lost update; the total for a uid/day is the SUM of its shards, and
the owner rollup for a day is the sum per uid under ``usage/{day}/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Counter vocabulary
# ---------------------------------------------------------------------------

# The Anthropic usage fields we bill on, plus the hedge surcharge (#163: a
# hedged stream pays for the loser's identical prompt too).
LLM_FIELDS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "hedge_extra_input_tokens",
)

# Call sites. Not an enum — an unknown site is recorded, not rejected (losing a
# counter is worse than an unfamiliar key) — but every site the code uses is
# named here so scripts/usage_report.py and the cost doc can enumerate them.
SITE_LIVE_SUGGESTION = "live_suggestion"   # per-utterance coaching (the hot one)
SITE_LIVE_NUDGE = "live_nudge"             # ≤6-word local-first nudge
SITE_LIVE_REPAIR = "live_repair"           # JSON repair retry
SITE_BATCH_ANALYSIS = "batch_analysis"     # /analyze, /analyze/upload, /analyze/link
SITE_COUNTERFACTUAL = "counterfactual"     # /analyze/counterfactual
SITE_REFLECTION = "reflection"             # /episodes/{id}/reflect + live post-ingest
SITE_RESPOND = "respond"                   # /respond
SITE_SCORE = "score"                       # /score
SITE_EXPORT = "export"                     # /session/{id}/export insights
SITE_WATCH_SUMMARY = "watch_summary"       # watch post-session summary
SITE_UNKNOWN = "unattributed"

# Non-LLM units.
KEY_STT_SECONDS = "stt.seconds"        # audio seconds sent to the cloud STT
KEY_LIVE_MINUTES = "live.minutes"      # wall-clock minutes of live WS session
KEY_MODEL_DOWNLOADS = "model.downloads"  # GET /models/ecapa.onnx bodies served
KEY_MODEL_BYTES = "model.bytes"        # bytes those downloads moved (egress)
KEY_CALLS_STARTED = "calls.started"    # in-app calls created

# Usage that could not be attributed to a signed-in uid (a startup probe, a
# background job whose scope was lost). Kept as a real bucket so the owner
# report shows it rather than the tokens vanishing.
UNATTRIBUTED_UID = "_unattributed"


def llm_key(site: str, field: str) -> str:
    """The counter key for one LLM field at one call site."""
    return f"llm.{site}.{field}"


def split_llm_key(key: str) -> "tuple[str, str] | None":
    """``llm.<site>.<field>`` → ``(site, field)``, or ``None`` for other keys."""
    if not key.startswith("llm."):
        return None
    _, _, rest = key.partition("llm.")
    site, _, field = rest.rpartition(".")
    if not site or not field:
        return None
    return site, field


# ---------------------------------------------------------------------------
# Quotas — generous defaults, env-tunable, 0 disables the cap entirely
# ---------------------------------------------------------------------------

def _cap(name: str, default: int) -> int:
    try:
        value = int((os.getenv(name) or "").strip() or default)
    except ValueError:
        logger.warning("%s is not an integer — using default %d", name, default)
        return default
    return max(0, value)


# ~1.5M tokens/day. A measured 30-minute coached call is ~40k tokens
# (docs/plans/2026-08-25-cost-model.md), so this is a full working day of
# back-to-back sessions before anything degrades.
DAILY_LLM_TOKEN_CAP = _cap("MINDSHIFT_DAILY_LLM_TOKENS", 1_500_000)
# 6 hours of cloud speech-to-text per account per day.
DAILY_STT_SECONDS_CAP = _cap("MINDSHIFT_DAILY_STT_SECONDS", 6 * 3600)
# 8 hours of live socket per account per day.
DAILY_LIVE_MINUTES_CAP = _cap("MINDSHIFT_DAILY_LIVE_MINUTES", 8 * 60)
# The ONNX model is ~80 MB and the phone caches it for a day; more than a
# handful of full downloads is a broken client or a scraper.
DAILY_MODEL_DOWNLOAD_CAP = _cap("MINDSHIFT_DAILY_MODEL_DOWNLOADS", 25)
# In-app calls created per day (each can carry three coached participants).
DAILY_CALLS_CAP = _cap("MINDSHIFT_DAILY_CALLS", 50)

# limit name -> (cap, the counter keys summed against it, human unit)
LIMITS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "llm_tokens": (
        "DAILY_LLM_TOKEN_CAP",
        ("input_tokens", "output_tokens", "cache_read_input_tokens",
         "cache_creation_input_tokens", "hedge_extra_input_tokens"),
        "tokens",
    ),
    "stt_seconds": ("DAILY_STT_SECONDS_CAP", (KEY_STT_SECONDS,), "seconds"),
    "live_minutes": ("DAILY_LIVE_MINUTES_CAP", (KEY_LIVE_MINUTES,), "minutes"),
    "model_downloads": (
        "DAILY_MODEL_DOWNLOAD_CAP", (KEY_MODEL_DOWNLOADS,), "downloads",
    ),
    "calls": ("DAILY_CALLS_CAP", (KEY_CALLS_STARTED,), "calls"),
}

# What each feature spends, and therefore which caps gate it. A feature is
# blocked when ANY of its limits is exhausted.
FEATURE_LIMITS: dict[str, tuple[str, ...]] = {
    # The live loop's cloud half. Stopping it leaves transcription and the
    # phone's on-device coaching untouched — that is the whole point.
    "cloud_suggestions": ("llm_tokens", "live_minutes"),
    # Cloud speech-to-text for a live session (the phone's local STT is free).
    "cloud_transcription": ("stt_seconds", "live_minutes"),
    "batch_analysis": ("llm_tokens",),
    "model_download": ("model_downloads",),
    "call_create": ("calls",),
}

# What keeps working when a feature stops — quoted verbatim to the client so
# the notice is specific instead of "quota exceeded".
FEATURE_FALLBACK: dict[str, tuple[str, ...]] = {
    "cloud_suggestions": ("transcript", "on_device_coaching", "session_recording"),
    "cloud_transcription": ("on_device_transcription", "on_device_coaching"),
    "batch_analysis": ("recording_playback", "existing_analyses"),
    "model_download": ("cached_on_device_model",),
    "call_create": ("existing_calls",),
}


@dataclass(frozen=True)
class Exceeded:
    """One exhausted daily limit, with everything the client needs to explain
    itself to a human: which budget, how much of it was used, when it resets,
    what stopped, and what still works."""

    feature: str
    limit: str
    used: float
    cap: float
    unit: str
    resets_at: str

    @property
    def stopped(self) -> tuple[str, ...]:
        return (self.feature,)

    @property
    def still_working(self) -> tuple[str, ...]:
        return FEATURE_FALLBACK.get(self.feature, ())

    @property
    def message(self) -> str:
        working = ", ".join(self.still_working) or "nothing else on this path"
        return (
            f"Daily {self.limit} budget reached "
            f"({self.used:g}/{self.cap:g} {self.unit}). "
            f"{self.feature} is paused until {self.resets_at}; "
            f"still working: {working}."
        )

    def notice(self) -> dict:
        """The ``quota_notice`` frame / 429 body. One shape everywhere."""
        return {
            "type": "quota_notice",
            "feature": self.feature,
            "limit": self.limit,
            "used": round(self.used, 3),
            "cap": self.cap,
            "unit": self.unit,
            "resets_at": self.resets_at,
            "stopped": list(self.stopped),
            "still_working": list(self.still_working),
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Time helpers — the day boundary is UTC, everywhere, no exceptions
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def day_key(when: "datetime | None" = None) -> str:
    return (when or utc_now()).strftime("%Y-%m-%d")


def next_reset_iso(when: "datetime | None" = None) -> str:
    """Start of the next UTC day — when the counters a user is bumping against
    roll over. Told to the client verbatim so "try later" has a time on it."""
    now = when or utc_now()
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ")


def days_since(since: str) -> list[str]:
    """Inclusive list of UTC day keys from ``since`` (YYYY-MM-DD) to today.

    Raises :class:`ValueError` on a malformed date so callers can answer 422
    rather than silently scanning nothing.
    """
    start = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    today = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
    if start > today:
        return []
    out: list[str] = []
    cursor = start
    while cursor <= today:
        out.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
        if len(out) > MAX_ROLLUP_DAYS:
            break
    return out


# Bound on how far back one /admin/usage call may scan — each day is a
# list_blobs + N downloads, so an unbounded `since` is a self-inflicted bill.
MAX_ROLLUP_DAYS = int(os.getenv("MINDSHIFT_USAGE_MAX_DAYS", "92"))

# How often pending counters are written to the store, and how stale this
# process's view of OTHER instances' counters may get before a quota check
# refreshes it.
FLUSH_INTERVAL_S = float(os.getenv("MINDSHIFT_USAGE_FLUSH_S", "30"))
SEED_TTL_S = float(os.getenv("MINDSHIFT_USAGE_SEED_TTL_S", "60"))

# This process's shard id. Each process owns one blob per uid/day, so shard
# writes never race and no update is ever lost.
INSTANCE_ID = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# The meter
# ---------------------------------------------------------------------------

_scope: ContextVar["tuple[str, str] | None"] = ContextVar(
    "mindshift_usage_scope", default=None,
)


class UsageMeter:
    """In-memory per-(uid, day) counters plus the persisted view of the same.

    ``_own`` is what THIS process has counted since it started; ``_seed`` is
    what other processes' shards held when last read. ``totals`` is the sum —
    the number a quota check compares against. ``_pending`` marks which
    (uid, day) pairs changed since the last flush so the flusher writes only
    those shards.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._own: dict[tuple[str, str], dict[str, float]] = {}
        self._seed: dict[tuple[str, str], dict[str, float]] = {}
        self._seeded_at: dict[tuple[str, str], float] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._store = None
        self._flusher: "asyncio.Task | None" = None

    # -- wiring ------------------------------------------------------------
    def bind(self, store) -> None:
        """Attach the persistence backend (``None`` = memory only)."""
        self._store = store

    @property
    def store(self):
        return self._store

    # -- recording ---------------------------------------------------------
    def add(self, uid: "str | None", counters: Mapping[str, float]) -> None:
        """Add counters for ``uid`` today. Hot path: a dict add under a lock,
        no I/O, never raises."""
        if not counters:
            return
        key = ((uid or UNATTRIBUTED_UID), day_key())
        with self._lock:
            bucket = self._own.setdefault(key, {})
            for name, value in counters.items():
                if not isinstance(value, (int, float)) or value != value:  # NaN
                    continue
                if value <= 0:
                    continue
                bucket[name] = bucket.get(name, 0) + value
            self._dirty.add(key)

    def add_llm(self, uid: "str | None", site: str, usage: Mapping[str, float]) -> None:
        """Record one LLM response's usage fields against ``site``."""
        counters = {
            llm_key(site, field): usage[field]
            for field in LLM_FIELDS
            if isinstance(usage.get(field), (int, float)) and usage.get(field, 0) > 0
        }
        counters.setdefault(llm_key(site, "calls"), 1)
        self.add(uid, counters)

    # -- reading -----------------------------------------------------------
    def totals(self, uid: str, day: "str | None" = None) -> dict[str, float]:
        """This process's best view of ``uid``'s counters for ``day``."""
        key = (uid, day or day_key())
        with self._lock:
            out = dict(self._seed.get(key, {}))
            for name, value in self._own.get(key, {}).items():
                out[name] = out.get(name, 0) + value
        return out

    @staticmethod
    def _used_from(totals: Mapping[str, float], limit: str) -> float:
        _, keys, _ = LIMITS[limit]
        if limit == "llm_tokens":
            # Sum the billed token fields across EVERY call site.
            total = 0.0
            for name, value in totals.items():
                parts = split_llm_key(name)
                if parts and parts[1] in keys:
                    total += value
            return total
        return float(sum(totals.get(k, 0) for k in keys))

    def limit_used(self, uid: str, limit: str, day: "str | None" = None) -> float:
        """How much of one named limit ``uid`` has spent today."""
        return self._used_from(self.totals(uid, day), limit)

    # -- quotas ------------------------------------------------------------
    def check(self, uid: "str | None", feature: str) -> "Exceeded | None":
        """The first exhausted limit gating ``feature``, or ``None``.

        Unknown features and the unattributed bucket are never blocked — a
        quota bug must not become an outage.
        """
        if not uid or uid == UNATTRIBUTED_UID:
            return None
        limits = FEATURE_LIMITS.get(feature, ())
        if not limits:
            return None
        # ONE snapshot for every limit this feature is gated on: the STT check
        # runs per audio frame (~10/s per session), so a per-limit copy of the
        # counters under the lock would be the wrong shape at scale.
        snapshot = self.totals(uid)
        for limit in limits:
            cap_attr, _, unit = LIMITS[limit]
            cap = float(globals().get(cap_attr) or 0)
            if cap <= 0:  # 0 / unset = no cap
                continue
            used = self._used_from(snapshot, limit)
            if used >= cap:
                return Exceeded(
                    feature=feature, limit=limit, used=used, cap=cap,
                    unit=unit, resets_at=next_reset_iso(),
                )
        return None

    # -- persistence (best-effort, always off the hot path) -----------------
    async def prime(self, uid: str) -> None:
        """Refresh this process's view of what OTHER instances recorded for
        ``uid`` today, at most once per :data:`SEED_TTL_S`. Called when a
        session opens and from the flusher; never on the per-utterance path."""
        store = self._store
        if store is None or not uid or uid == UNATTRIBUTED_UID:
            return
        key = (uid, day_key())
        loop_now = utc_now().timestamp()
        with self._lock:
            last = self._seeded_at.get(key, 0.0)
            if loop_now - last < SEED_TTL_S:
                return
            self._seeded_at[key] = loop_now
        try:
            other = await store.read_usage_totals(
                uid, key[1], exclude_instance=INSTANCE_ID,
            )
        except Exception:  # noqa: BLE001 — accounting never fails a request
            logger.debug("Usage seed read failed for uid=%s", uid, exc_info=True)
            return
        with self._lock:
            self._seed[key] = dict(other or {})

    async def flush(self) -> int:
        """Write every changed shard. Returns how many shards were written."""
        store = self._store
        if store is None:
            with self._lock:
                self._dirty.clear()
            return 0
        with self._lock:
            dirty = sorted(self._dirty)
            self._dirty.clear()
            snapshot = {key: dict(self._own.get(key, {})) for key in dirty}
        written = 0
        for (uid, day), counters in snapshot.items():
            if not counters:
                continue
            try:
                await store.write_usage_shard(uid, day, INSTANCE_ID, counters)
                written += 1
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Usage flush failed for uid=%s day=%s — retrying next tick",
                    uid, day, exc_info=True,
                )
                with self._lock:
                    self._dirty.add((uid, day))
        self._prune_old_days()
        return written

    def _prune_old_days(self) -> None:
        """Drop fully-flushed counters for days that are over.

        A Cloud Run instance can live for days and see thousands of accounts;
        without this, ``_own``/``_seed`` would grow by one entry per uid per
        day forever. Only entries that are NOT today and NOT dirty are
        dropped, so nothing is discarded before it has been persisted.
        """
        today = day_key()
        with self._lock:
            stale = [
                key for key in list(self._own)
                if key[1] != today and key not in self._dirty
            ]
            for key in stale:
                self._own.pop(key, None)
            for key in [k for k in list(self._seed) if k[1] != today]:
                self._seed.pop(key, None)
                self._seeded_at.pop(key, None)

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_S)
            with contextlib.suppress(asyncio.CancelledError):
                await self.flush()
                # Keep the quota view of long-lived sessions fresh.
                with self._lock:
                    active = {uid for uid, day in self._own if day == day_key()}
                for uid in active:
                    await self.prime(uid)

    def start(self) -> None:
        """Start the background flusher (idempotent; needs a running loop)."""
        if self._flusher is not None and not self._flusher.done():
            return
        try:
            self._flusher = asyncio.get_running_loop().create_task(self._flush_loop())
        except RuntimeError:  # no loop (unit tests) — flush() still works
            self._flusher = None

    async def stop(self) -> None:
        """Cancel the flusher and write whatever is pending (shutdown path)."""
        task, self._flusher = self._flusher, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        with contextlib.suppress(Exception):
            await self.flush()

    # -- tests -------------------------------------------------------------
    def reset(self) -> None:
        with self._lock:
            self._own.clear()
            self._seed.clear()
            self._seeded_at.clear()
            self._dirty.clear()


_meter = UsageMeter()


def meter() -> UsageMeter:
    return _meter


# Module-level conveniences — the call sites read better as verbs.
def bind_store(store) -> None:
    _meter.bind(store)


def record(uid: "str | None", **counters: float) -> None:
    """``record(uid, **{'stt.seconds': 12.5})`` — best-effort, non-blocking."""
    _meter.add(uid, counters)


def record_counters(uid: "str | None", counters: Mapping[str, float]) -> None:
    _meter.add(uid, counters)


def check(uid: "str | None", feature: str) -> "Exceeded | None":
    return _meter.check(uid, feature)


def totals(uid: str, day: "str | None" = None) -> dict[str, float]:
    return _meter.totals(uid, day)


async def prime(uid: str) -> None:
    await _meter.prime(uid)


# ---------------------------------------------------------------------------
# Attribution — who is spending, on what
# ---------------------------------------------------------------------------

class attribute:
    """Context manager binding LLM spend inside it to ``(uid, site)``.

    Both a sync ``with`` and an ``async with`` (the async form exists so a
    FastAPI dependency can hold it open across an awaited handler). The value
    rides a :class:`ContextVar`, which ``asyncio.to_thread`` and
    ``asyncio.create_task`` copy — so an offloaded blocking SDK call and a
    spawned analysis job both stay attributed.
    """

    __slots__ = ("_scope", "_token")

    def __init__(self, uid: "str | None", site: str) -> None:
        self._scope = ((uid or UNATTRIBUTED_UID), site)
        self._token = None

    def __enter__(self) -> "attribute":
        self._token = _scope.set(self._scope)
        return self

    def __exit__(self, *exc) -> None:
        if self._token is not None:
            with contextlib.suppress(ValueError):
                _scope.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "attribute":
        return self.__enter__()

    async def __aexit__(self, *exc) -> None:
        self.__exit__(*exc)


def current_scope() -> "tuple[str, str] | None":
    return _scope.get()


def note_llm_usage(usage: Mapping[str, float]) -> None:
    """Record one Anthropic response's usage against the active scope.

    Called by :meth:`llm_client.LLMClient._record_usage`. Deliberately
    swallows everything: an accounting bug must never break a coaching turn.
    """
    try:
        uid, site = _scope.get() or (UNATTRIBUTED_UID, SITE_UNKNOWN)
        _meter.add_llm(uid, site, usage)
    except Exception:  # noqa: BLE001
        logger.debug("note_llm_usage failed", exc_info=True)


def note_hedge_extra(input_tokens: int) -> None:
    """The hedged-stream surcharge (#163): the loser's identical prompt is
    billed too. Recorded against the same scope as the winner."""
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        return
    try:
        uid, site = _scope.get() or (UNATTRIBUTED_UID, SITE_UNKNOWN)
        _meter.add(uid, {llm_key(site, "hedge_extra_input_tokens"): input_tokens})
    except Exception:  # noqa: BLE001
        logger.debug("note_hedge_extra failed", exc_info=True)


# ---------------------------------------------------------------------------
# FastAPI glue
# ---------------------------------------------------------------------------

def quota_error(exceeded: Exceeded):
    """A 429 carrying the same ``quota_notice`` body the live socket sends.

    REST callers that spend LLM credit fail CLOSED (there is no half of
    ``/analyze`` worth returning without the model), but they fail loudly and
    with a reset time — never a silent empty result. The live socket takes the
    other branch: it degrades and keeps going. See the module docstring.
    """
    from fastapi import HTTPException

    return HTTPException(
        status_code=429,
        detail=exceeded.notice(),
        headers={"Retry-After": str(_seconds_until_reset())},
    )


def _seconds_until_reset() -> int:
    now = utc_now()
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    return max(1, int((tomorrow - now).total_seconds()))


def usage_scope(site: str, feature: "str | None" = None):
    """FastAPI dependency: attribute this request's LLM spend, and (when
    ``feature`` is given) refuse it with a 429 once that feature's daily budget
    is spent.

    Declared as an ASYNC generator on purpose: FastAPI solves async
    dependencies in the request's own task, so the ContextVar set here is
    visible to the endpoint body (a sync dependency would run in a threadpool
    with a copied context and the binding would be lost).
    """
    from auth import get_current_uid
    from fastapi import Depends

    async def _scope_dep(uid: str = Depends(get_current_uid)):
        if feature:
            exceeded = _meter.check(uid, feature)
            if exceeded is not None:
                logger.info(
                    "Quota %s exhausted for uid=%s — refusing %s",
                    exceeded.limit, uid, site,
                )
                raise quota_error(exceeded)
        with attribute(uid, site):
            yield

    return _scope_dep


# ---------------------------------------------------------------------------
# Owner rollups
# ---------------------------------------------------------------------------

def merge(into: dict[str, float], other: Mapping[str, float]) -> dict[str, float]:
    for name, value in (other or {}).items():
        if isinstance(value, (int, float)):
            into[name] = into.get(name, 0) + value
    return into


def summarize(counters: Mapping[str, float]) -> dict:
    """Group a flat counter dict into the shape the owner report renders:
    per-site LLM tokens plus the flat unit counters."""
    by_site: dict[str, dict[str, float]] = {}
    other: dict[str, float] = {}
    for name, value in counters.items():
        parts = split_llm_key(name)
        if parts is None:
            other[name] = value
            continue
        site, field = parts
        by_site.setdefault(site, {})[field] = value
    totals_in = sum(
        v.get("input_tokens", 0) + v.get("cache_read_input_tokens", 0)
        + v.get("cache_creation_input_tokens", 0)
        + v.get("hedge_extra_input_tokens", 0)
        for v in by_site.values()
    )
    totals_out = sum(v.get("output_tokens", 0) for v in by_site.values())
    return {
        "llm": by_site,
        "llm_input_tokens": totals_in,
        "llm_output_tokens": totals_out,
        "llm_calls": sum(v.get("calls", 0) for v in by_site.values()),
        "stt_seconds": other.get(KEY_STT_SECONDS, 0),
        "live_minutes": other.get(KEY_LIVE_MINUTES, 0),
        "model_downloads": other.get(KEY_MODEL_DOWNLOADS, 0),
        "model_bytes": other.get(KEY_MODEL_BYTES, 0),
        "calls_started": other.get(KEY_CALLS_STARTED, 0),
    }


async def rollup(store, days: Iterable[str]) -> dict[str, dict]:
    """Per-uid counters summed over ``days``, read from the store, with THIS
    process's unflushed counters folded in so a fresh spend is never invisible
    to the owner just because the flusher hasn't ticked yet."""
    out: dict[str, dict[str, float]] = {}
    day_list = list(days)
    if store is not None:
        for day in day_list:
            try:
                per_uid = await store.list_usage(day)
            except Exception:  # noqa: BLE001
                logger.warning("Usage rollup failed for day=%s", day, exc_info=True)
                continue
            for uid, counters in (per_uid or {}).items():
                merge(out.setdefault(uid, {}), counters)
    with _meter._lock:  # noqa: SLF001 — same module, documented internal
        own = {k: dict(v) for k, v in _meter._own.items()}  # noqa: SLF001
    for (uid, day), counters in own.items():
        if day not in day_list:
            continue
        if store is not None:
            # Already-flushed counters are in the shard we just read, so folding
            # the process copy in would double-count. Take the max per key: the
            # shard is a prefix of this process's totals for the day.
            bucket = out.setdefault(uid, {})
            for name, value in counters.items():
                bucket[name] = max(bucket.get(name, 0), value)
        else:
            merge(out.setdefault(uid, {}), counters)
    return {uid: summarize(counters) for uid, counters in out.items()}


# ---------------------------------------------------------------------------
# Admin allowlist
# ---------------------------------------------------------------------------

def admin_uids() -> frozenset[str]:
    """The uids allowed to read /admin/usage. Empty (the default) means the
    endpoint is CLOSED — no ambient owner, no accidental exposure of every
    account's usage on a fresh deployment."""
    raw = os.getenv("MINDSHIFT_ADMIN_UIDS", "")
    return frozenset(u.strip() for u in raw.split(",") if u.strip())


def is_admin(uid: str) -> bool:
    return bool(uid) and uid in admin_uids()
