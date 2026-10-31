"""Plans: free / pro levels, the limits table, redeem codes (phase 1).

Design mirrors the owner's oralhistory_timeline subscriptions (docs/SUBSCRIPTIONS.md):

* Server-owned fields per user: ``plan`` (level), ``plan_expires_at``,
  ``plan_source`` (``revenuecat`` | ``admin`` | ``promo``). Clients never write
  them. Stored in Firestore ``ms_plans/{uid}`` (a MindShift-specific collection
  name: the GCP project is shared with other apps that own ``users``).
* The level in force is the stored one ONLY while unexpired (or no expiry);
  otherwise ``free``. Nothing is ever deleted when a user is over a limit: the
  limit blocks *creating new things*, and the 429 body is typed
  (``type: "plan_limit"`` / ``quota_notice``) so the app shows an upgrade sheet.
* ADDING A LEVEL is a one-line change: append to :data:`LEVELS`, add a row to
  :data:`LIMITS`, add a label to :data:`LABELS`.

Where limits come from (``limit_for``): env ``MINDSHIFT_PLAN_<LEVEL>_<KEY>``
if set, else the table. The ``free`` row's daily caps are ``None`` = "whatever
usage_meter's module constants say right now" so today's numbers (and the old
``MINDSHIFT_DAILY_*`` env vars) keep working unchanged for existing users.

Phase 2 (RevenueCat webhook) will write the same three fields with
``plan_source="revenuecat"``; see docs/plans/2026-10-31-paid-plans.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Levels and the limits table  (tune HERE)
# ---------------------------------------------------------------------------

LEVELS: tuple[str, ...] = ("free", "pro")  # ordered low -> high
LABELS: dict[str, str] = {"free": "Free", "pro": "Pro"}

# Daily caps (names match usage_meter.LIMITS); 0 = no cap. ``None`` in the free
# row = defer to usage_meter's constants (today's numbers).
DAILY_KEYS = ("llm_tokens", "stt_seconds", "live_minutes", "model_downloads", "calls")
LIBRARY_KEYS = ("library_items", "library_chars")
FLAG_KEYS = ("room_mode", "heads_up_cues")
LIMIT_KEYS = DAILY_KEYS + LIBRARY_KEYS + FLAG_KEYS

LIMITS: dict[str, dict[str, Any]] = {
    "free": {
        "llm_tokens": None, "stt_seconds": None, "live_minutes": None,
        "model_downloads": None, "calls": None,
        # A trial-sized library; existing larger libraries are never trimmed.
        "library_items": 5, "library_chars": 200_000,
        # Both stay on for free so nothing regresses; flip here to make them
        # Pro perks (enforcement is the caller's job, see /plan "flags").
        "room_mode": True, "heads_up_cues": True,
    },
    "pro": {
        "llm_tokens": 4_000_000, "stt_seconds": 12 * 3600, "live_minutes": 16 * 60,
        "model_downloads": 50, "calls": 150,
        "library_items": 200, "library_chars": 20_000_000,
        "room_mode": True, "heads_up_cues": True,
    },
}

PRICE_TEXT = None  # phase 2: the app shows the store's localized price instead


def _base_cap(key: str) -> float:
    import usage_meter

    attr = usage_meter.LIMITS[key][0]
    return float(getattr(usage_meter, attr) or 0)


def limit_for(level: str, key: str):
    """The limit ``key`` for ``level``: env override, else table, else (free
    daily caps) usage_meter's current constant. Flags return bool."""
    if level not in LIMITS:
        level = "free"
    raw = os.getenv(f"MINDSHIFT_PLAN_{level.upper()}_{key.upper()}")
    if raw is not None and raw.strip() != "":
        if key in FLAG_KEYS:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        try:
            return max(0, int(raw.strip()))
        except ValueError:
            logger.warning("bad int in MINDSHIFT_PLAN_%s_%s", level.upper(), key.upper())
    val = LIMITS[level].get(key)
    if val is None and key in DAILY_KEYS:
        return _base_cap(key)
    return val


def upgrade_level(level: str) -> Optional[str]:
    """The next level up, or None at the top."""
    try:
        i = LEVELS.index(level)
    except ValueError:
        return LEVELS[1] if len(LEVELS) > 1 else None
    return LEVELS[i + 1] if i + 1 < len(LEVELS) else None


def level_rank(level: str) -> int:
    return LEVELS.index(level) if level in LEVELS else 0


# ---------------------------------------------------------------------------
# Effective plan
# ---------------------------------------------------------------------------

def _aware(dt):
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Effective:
    level: str = "free"
    expires_at: Optional[datetime] = None
    source: Optional[str] = None


def effective(doc: Optional[dict], now: Optional[datetime] = None) -> Effective:
    """The level in force for a stored plan doc (expiry honoured, unknown
    levels and missing docs are ``free``)."""
    now = now or datetime.now(timezone.utc)
    if not doc:
        return Effective()
    level = doc.get("plan")
    if level not in LIMITS or level == "free":
        return Effective()
    exp = _aware(doc.get("plan_expires_at"))
    if exp is not None and exp <= now:
        return Effective()
    return Effective(level=level, expires_at=exp, source=doc.get("plan_source"))


# ---------------------------------------------------------------------------
# Redeem codes
# ---------------------------------------------------------------------------

# No 0/O, 1/I/L, U (avoids look-alikes and accidental words).
CODE_CHARSET = "ABCDEFGHJKMNPQRSTVWXYZ23456789"
CODE_LEN = 12  # ~58 bits


def generate_code() -> str:
    raw = "".join(secrets.choice(CODE_CHARSET) for _ in range(CODE_LEN))
    return "-".join(raw[i:i + 4] for i in range(0, CODE_LEN, 4))


def normalize_code(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


@dataclass
class Decision:
    kind: str  # granted | already_redeemed | covered | invalid
    code_updates: dict = field(default_factory=dict)
    user_updates: dict = field(default_factory=dict)


def decide_redeem(code: Optional[dict], already: bool, user: Optional[dict],
                  now: datetime) -> Decision:
    """Pure redemption rules (both stores run this inside their atomic section).

    * unknown / revoked / expired / used-up / malformed -> ``invalid`` (one
      generic outcome, so a probe learns nothing about which it was);
    * same user + code twice -> ``already_redeemed`` (idempotent, no use spent);
    * user already holds >= this level with no expiry, or via a paying
      subscription -> ``covered`` (no use spent, nothing shortened);
    * otherwise grant, EXTENDING any current same-level grant by ``days``.
    """
    if not code or code.get("revoked"):
        return Decision("invalid")
    level = code.get("plan")
    days = code.get("days")
    if level not in LIMITS or level == "free" or not isinstance(days, int) or days <= 0:
        return Decision("invalid")
    exp = _aware(code.get("expires_at"))
    if exp is not None and exp <= now:
        return Decision("invalid")
    if already:
        return Decision("already_redeemed")
    max_uses = int(code.get("max_uses") or 0)
    uses = int(code.get("uses") or 0)
    if max_uses and uses >= max_uses:
        return Decision("invalid")

    cur = effective(user, now)
    if level_rank(cur.level) > level_rank(level) or (
        cur.level == level and (cur.expires_at is None or cur.source == "revenuecat")
    ):
        return Decision("covered")
    start = cur.expires_at if (cur.level == level and cur.expires_at) else now
    return Decision(
        "granted",
        code_updates={"uses": uses + 1},
        user_updates={
            "plan": level,
            "plan_expires_at": start + timedelta(days=days),
            "plan_source": "promo",
            "plan_updated_at": now,
        },
    )


@dataclass
class RedeemResult:
    kind: str
    plan: str = "free"
    expires_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

USERS = "ms_plans"
CODES = "ms_plan_codes"


class MemoryPlanStore:
    persistent = False

    def __init__(self) -> None:
        self._users: dict[str, dict] = {}
        self._codes: dict[str, dict] = {}
        self._redemptions: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    async def get(self, uid):
        with self._lock:
            return dict(self._users[uid]) if uid in self._users else None

    async def set_fields(self, uid, fields):
        with self._lock:
            self._users.setdefault(uid, {}).update(fields)

    async def create_code(self, doc):
        with self._lock:
            if doc["code"] in self._codes:
                raise ValueError("code exists")
            self._codes[doc["code"]] = dict(doc)

    async def get_code(self, code):
        with self._lock:
            return dict(self._codes[code]) if code in self._codes else None

    async def list_codes(self):
        with self._lock:
            return [dict(v) for v in self._codes.values()]

    async def revoke_code(self, code):
        with self._lock:
            if code not in self._codes:
                return False
            self._codes[code]["revoked"] = True
            return True

    async def redeem(self, uid, code, now):
        with self._lock:
            c = self._codes.get(code)
            d = decide_redeem(c, (uid, code) in self._redemptions, self._users.get(uid), now)
            if d.kind == "granted":
                c.update(d.code_updates)
                self._users.setdefault(uid, {}).update(d.user_updates)
                self._redemptions.add((uid, code))
            eff = effective(self._users.get(uid), now)
        return RedeemResult(d.kind, eff.level, eff.expires_at)


class FirestorePlanStore:
    persistent = True

    def __init__(self, project: str) -> None:
        self.project = project
        self._db = None

    def _get_db(self):
        if self._db is None:
            from google.cloud import firestore

            self._db = firestore.Client(project=self.project)
        return self._db

    def _get(self, uid):
        d = self._get_db().collection(USERS).document(uid).get()
        return d.to_dict() if d.exists else None

    def _set_fields(self, uid, fields):
        self._get_db().collection(USERS).document(uid).set(fields, merge=True)

    def _create_code(self, doc):
        from google.api_core.exceptions import AlreadyExists

        try:
            self._get_db().collection(CODES).document(doc["code"]).create(doc)
        except AlreadyExists as e:
            raise ValueError("code exists") from e

    def _get_code(self, code):
        d = self._get_db().collection(CODES).document(code).get()
        return d.to_dict() if d.exists else None

    def _list_codes(self):
        return [d.to_dict() for d in self._get_db().collection(CODES).stream()]

    def _revoke(self, code):
        ref = self._get_db().collection(CODES).document(code)
        if not ref.get().exists:
            return False
        ref.update({"revoked": True})
        return True

    def _redeem(self, uid, code, now):
        from google.cloud import firestore

        db = self._get_db()
        code_ref = db.collection(CODES).document(code)
        user_ref = db.collection(USERS).document(uid)
        red_ref = user_ref.collection("redemptions").document(code)

        @firestore.transactional
        def run(tx):
            c = code_ref.get(transaction=tx)
            r = red_ref.get(transaction=tx)
            u = user_ref.get(transaction=tx)
            udoc = u.to_dict() if u.exists else None
            d = decide_redeem(c.to_dict() if c.exists else None, r.exists, udoc, now)
            if d.kind == "granted":
                tx.update(code_ref, d.code_updates)
                tx.set(user_ref, d.user_updates, merge=True)
                tx.set(red_ref, {"redeemed_at": now, "plan": d.user_updates["plan"]})
                udoc = {**(udoc or {}), **d.user_updates}
            return d.kind, udoc

        kind, udoc = run(db.transaction())
        eff = effective(udoc, now)
        return RedeemResult(kind, eff.level, eff.expires_at)

    async def get(self, uid):
        return await asyncio.to_thread(self._get, uid)

    async def set_fields(self, uid, fields):
        await asyncio.to_thread(self._set_fields, uid, fields)

    async def create_code(self, doc):
        await asyncio.to_thread(self._create_code, doc)

    async def get_code(self, code):
        return await asyncio.to_thread(self._get_code, code)

    async def list_codes(self):
        return await asyncio.to_thread(self._list_codes)

    async def revoke_code(self, code):
        return await asyncio.to_thread(self._revoke, code)

    async def redeem(self, uid, code, now):
        return await asyncio.to_thread(self._redeem, uid, code, now)


_store = None
_store_set = False


def set_store(store) -> None:
    """Tests (and scripts) inject a store; ``None`` = plans unavailable."""
    global _store, _store_set
    _store, _store_set = store, store is not None


def get_store():
    """The configured store, or None when plans are unavailable (no Firestore
    project configured) -- never a silent in-memory stand-in in production."""
    global _store
    if _store is None and not _store_set:
        project = os.environ.get("MINDSHIFT_FIRESTORE_PROJECT")
        if project:
            _store = FirestorePlanStore(project)
    return _store


# ---------------------------------------------------------------------------
# Per-process cache so usage_meter.check() (sync, hot path) can see the level
# ---------------------------------------------------------------------------

PLAN_TTL_S = float(os.getenv("MINDSHIFT_PLAN_TTL_S", "60"))
_cache: dict[str, tuple[Optional[dict], float]] = {}
_cache_lock = threading.Lock()


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def invalidate(uid: str) -> None:
    with _cache_lock:
        _cache.pop(uid, None)


async def prime(uid: str, force: bool = False) -> None:
    """Refresh the cached plan doc for ``uid`` (no-op within PLAN_TTL_S).
    Never raises: a store failure leaves the previous view (default free)."""
    store = get_store()
    if store is None or not uid:
        return
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(uid)
        if hit and not force and now - hit[1] < PLAN_TTL_S:
            return
    try:
        doc = await store.get(uid)
    except Exception:  # noqa: BLE001
        logger.warning("plan read failed uid=%s", uid, exc_info=True)
        return
    with _cache_lock:
        _cache[uid] = (doc, now)


def effective_cached(uid: Optional[str]) -> Effective:
    """Sync view for hot paths. Expiry is evaluated NOW against the cached doc."""
    if not uid:
        return Effective()
    with _cache_lock:
        hit = _cache.get(uid)
    return effective(hit[0] if hit else None)


def cap_for(uid: Optional[str], key: str) -> float:
    return float(limit_for(effective_cached(uid).level, key) or 0)


# ---------------------------------------------------------------------------
# Typed plan-limit error (same family as usage_meter's quota_notice)
# ---------------------------------------------------------------------------

def plan_limit_detail(level: str, key: str, used: float, cap: float, message: str) -> dict:
    up = upgrade_level(level)
    return {
        "type": "plan_limit",
        "reason": "plan-limit",
        "limit": key,
        "used": used,
        "cap": cap,
        "plan": level,
        "upgrade_plan": up,
        "message": message,
    }


async def check_library_create(uid: str, svc, incoming_chars: int = 0) -> None:
    """Gate for creating a NEW library item. Raises HTTP 429 (``plan_limit``)
    when the user's plan is at its item-count or total-size cap. Never touches
    existing items: a user over the cap (e.g. after a plan expires) keeps
    everything and simply cannot add more. Called once from routers/library.py."""
    from fastapi import HTTPException

    if get_store() is None:
        # Plans unavailable (no Firestore configured): fail open rather than
        # enforce a plan-only cap nobody can lift with a code.
        return
    await prime(uid)
    level = effective_cached(uid).level
    max_items = int(limit_for(level, "library_items") or 0)
    max_chars = int(limit_for(level, "library_chars") or 0)
    if not (max_items or max_chars):
        return
    items = await svc.store.list_items(uid)
    count = len(items)
    total = sum(int(getattr(i, "chars", 0) or 0) for i in items)
    nxt = upgrade_level(level)
    hint = f" Upgrade to {LABELS[nxt]} for more." if nxt else ""
    if max_items and count >= max_items:
        raise HTTPException(429, plan_limit_detail(
            level, "library_items", count, max_items,
            f"Your {LABELS[level]} plan holds {max_items} library items. "
            f"Your existing items are safe.{hint}"))
    if max_chars and total + max(0, incoming_chars) > max_chars:
        raise HTTPException(429, plan_limit_detail(
            level, "library_chars", total, max_chars,
            f"That would go over your {LABELS[level]} library size "
            f"({max_chars:,} characters). Your existing items are safe.{hint}"))


# ---------------------------------------------------------------------------
# Redeem failure limiter (per uid, in-process; the per-IP limiter is main's)
# ---------------------------------------------------------------------------

REDEEM_MAX_FAILURES = int(os.getenv("MINDSHIFT_REDEEM_MAX_FAILURES", "8"))
REDEEM_WINDOW_S = 3600.0
_failures: dict[str, list[float]] = {}


def reset_redeem_failures() -> None:
    _failures.clear()


def redeem_blocked(uid: str) -> bool:
    now = time.monotonic()
    recent = [t for t in _failures.get(uid, []) if now - t < REDEEM_WINDOW_S]
    _failures[uid] = recent
    return len(recent) >= REDEEM_MAX_FAILURES


def note_redeem_failure(uid: str) -> None:
    _failures.setdefault(uid, []).append(time.monotonic())
