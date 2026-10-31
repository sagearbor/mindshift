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

LEVELS: tuple[str, ...] = ("free", "basic", "plus", "pro", "max")  # ordered low -> high
LABELS: dict[str, str] = {"free": "Free", "basic": "Basic", "plus": "Plus", "pro": "Pro", "max": "Max"}

# Daily caps (names match usage_meter.LIMITS); 0 = no cap. ``None`` in the free
# row = defer to usage_meter's constants.
DAILY_KEYS = ("llm_tokens", "stt_seconds", "live_minutes", "model_downloads", "calls")
LIBRARY_KEYS = ("library_items", "library_chars")
FLAG_KEYS = ("room_mode", "heads_up_cues")

# ---- The unit is LISTENING HOURS (tune HERE) --------------------------------
# What costs us money, and so what the allowance meters:
#   * cloud speech-to-text seconds (stt.seconds): ~$0.45 per hour. A session
#     whose STT runs ON THE PHONE (local-first) sends no audio and burns none.
#   * AI coaching tokens (llm.*): ~33k tokens per listening hour, ~$0.06/hour.
#     Charged on every session, on-device STT or not.
# Live wall-clock minutes are NOT metered against the allowance (on-device
# listening is free for us); only a generous daily abuse backstop remains.
# Worst case per listening hour = $0.51 (0.45 + 33k tokens at $1.80/M).
TOKENS_PER_HOUR = 40_000   # cap per hour = ~33k measured/hour plus ~20% headroom
BACKSTOP_FRACTION = 0.25   # daily STT/AI backstop = 25% of the monthly allowance

# level -> (hours, "lifetime" for free / "month" for paid, price USD/mo)
ALLOWANCE_HOURS = {"free": 1, "basic": 4, "plus": 8, "pro": 20, "max": 80}
PRICE_USD = {"free": 0, "basic": 5, "plus": 10, "pro": 25, "max": 100}
# level -> (library items, library chars)
LIBRARY = {
    "free": (5, 200_000), "basic": (10, 500_000), "plus": (25, 1_000_000),
    "pro": (200, 8_000_000), "max": (1000, 40_000_000),
}
CALLS_MODELS = {"basic": (20, 50), "plus": (30, 75), "pro": (50, 150), "max": (100, 300)}

WINDOW_KEYS = (
    "stt_lifetime_seconds", "stt_monthly_seconds",
    "llm_lifetime_tokens", "llm_monthly_tokens", "live_lifetime_minutes",
)
LIMIT_KEYS = DAILY_KEYS + WINDOW_KEYS + LIBRARY_KEYS + FLAG_KEYS

# window name -> (table key, "lifetime"|"month", counter key)
WINDOW_RULES = {
    "stt_lifetime": ("stt_lifetime_seconds", "lifetime", "stt_seconds"),
    "stt_monthly": ("stt_monthly_seconds", "month", "stt_seconds"),
    "llm_lifetime": ("llm_lifetime_tokens", "lifetime", "llm_tokens"),
    "llm_monthly": ("llm_monthly_tokens", "month", "llm_tokens"),
    "live_lifetime": ("live_lifetime_minutes", "lifetime", "live_minutes"),
}


def _row(level: str) -> dict[str, Any]:
    hours = ALLOWANCE_HOURS[level]
    secs, toks = hours * 3600, hours * TOKENS_PER_HOUR
    items, chars = LIBRARY[level]
    paid = level != "free"
    models, calls = CALLS_MODELS.get(level, (None, None))
    return {
        # free daily caps defer to usage_meter (MINDSHIFT_DAILY_* env)
        "llm_tokens": int(toks * BACKSTOP_FRACTION) if paid else None,
        "stt_seconds": int(secs * BACKSTOP_FRACTION) if paid else None,
        "live_minutes": 12 * 60 if paid else None,
        "model_downloads": models, "calls": calls,
        "stt_lifetime_seconds": 0 if paid else secs,
        "stt_monthly_seconds": secs if paid else 0,
        "llm_lifetime_tokens": 0 if paid else toks,
        "llm_monthly_tokens": toks if paid else 0,
        "live_lifetime_minutes": 0,  # not metered: on-device listening is free
        "library_items": items, "library_chars": chars,
        # Both stay on everywhere so nothing regresses; flip here to make them
        # perks (enforcement is the caller's job, see /plan "flags").
        "room_mode": True, "heads_up_cues": True,
    }


LIMITS: dict[str, dict[str, Any]] = {lv: _row(lv) for lv in LEVELS}

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

    kind, updates = grant_days(user, level, days, now)
    if kind != "granted":
        return Decision("covered")
    return Decision("granted", code_updates={"uses": uses + 1}, user_updates=updates)


def grant_days(user: Optional[dict], level: str, days: int, now: datetime):
    """Pure: how ``days`` of ``level`` land on ``user``.

    -> ("granted", fields)  extends a same-level grant (never shortens);
       ("covered", {})      already has this level (or higher) with no expiry,
                            or pays through a store subscription (a promo never
                            overwrites the store's expiry).
    """
    cur = effective(user, now)
    if level_rank(cur.level) > level_rank(level) or (
        cur.level == level and (cur.expires_at is None or cur.source == "revenuecat")
    ):
        return "covered", {}
    start = cur.expires_at if (cur.level == level and cur.expires_at) else now
    return "granted", {
        "plan": level,
        "plan_expires_at": start + timedelta(days=days),
        "plan_source": "promo",
        "plan_updated_at": now,
    }


@dataclass
class RedeemResult:
    kind: str
    plan: str = "free"
    expires_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Usage windows (lifetime / month) kept on the plan doc
# ---------------------------------------------------------------------------

def month_key(now: datetime) -> str:
    return now.strftime("%Y-%m")


def apply_usage(doc: Optional[dict], key: str, amount: float, now: datetime) -> dict:
    """Pure: the ``lifetime`` / ``monthly`` fields after adding ``amount`` of
    ``key``. Lifetime never resets; monthly resets when the month changes.
    Plan grants / redeems never touch these fields."""
    doc = doc or {}
    life = dict(doc.get("lifetime") or {})
    life[key] = float(life.get(key, 0)) + amount
    mon = dict(doc.get("monthly") or {})
    if mon.get("period") != month_key(now):
        mon = {"period": month_key(now)}
    mon[key] = float(mon.get(key, 0)) + amount
    return {"lifetime": life, "monthly": mon}


def window_used(doc: Optional[dict], key: str, window: str, now: datetime) -> float:
    doc = doc or {}
    if window == "lifetime":
        return float((doc.get("lifetime") or {}).get(key, 0))
    mon = doc.get("monthly") or {}
    return float(mon.get(key, 0)) if mon.get("period") == month_key(now) else 0.0


# ---------------------------------------------------------------------------
# Referrals: personal share code, +1 free listening hour for BOTH sides
# ---------------------------------------------------------------------------

REFERRAL_BONUS_SECONDS = 1 * 3600       # each side's reward
REFERRAL_MAX_BONUS_SECONDS = 1 * 3600   # total bonus any one account can hold
REFERRAL_MIN_LIVE_MINUTES = 10          # the new user must really use the app first
REFERRAL_CODE_LEN = 8                   # admin codes are 12: length tells them apart
REFERRALS = "ms_referrals"              # {referee_uid} -> status doc, no PII
REFERRAL_CODES = "ms_referral_codes"    # {CODE} -> {uid}


def add_bonus(doc: Optional[dict]) -> dict:
    """Pure: the plan-doc fields after one referral bonus (capped)."""
    have = float((doc or {}).get("stt_bonus_seconds") or 0)
    return {"stt_bonus_seconds": min(REFERRAL_MAX_BONUS_SECONDS, have + REFERRAL_BONUS_SECONDS)}


def generate_referral_code() -> str:
    raw = "".join(secrets.choice(CODE_CHARSET) for _ in range(REFERRAL_CODE_LEN))
    return f"{raw[:4]}-{raw[4:]}"


def is_referral_code(normalized: str) -> bool:
    return len(normalized) == REFERRAL_CODE_LEN


@dataclass
class ReferralRedeem:
    kind: str  # granted | covered | invalid | own_code | already_referred
    referee_updates: dict = field(default_factory=dict)
    referral_doc: Optional[dict] = None


def decide_referral_redeem(referrer_uid: Optional[str], referee_uid: str,
                           referee: Optional[dict], existing: Optional[dict],
                           now: datetime) -> ReferralRedeem:
    """New user B enters A's code. B gets +REFERRAL_BONUS_SECONDS of free
    speech-to-text now; A's reward waits (decide_referral_reward). One referral
    per account, ever."""
    if not referrer_uid:
        return ReferralRedeem("invalid")
    if referrer_uid == referee_uid:
        return ReferralRedeem("own_code")
    if existing is not None or (referee or {}).get("referred_by"):
        return ReferralRedeem("already_referred")
    return ReferralRedeem(
        "granted",
        {**add_bonus(referee), "referred_by": referrer_uid},
        {"referrer": referrer_uid, "created_at": now, "status": "pending",
         "bonus_seconds": REFERRAL_BONUS_SECONDS},
    )


@dataclass
class ReferralReward:
    kind: str  # rewarded | capped | not_ready | none
    referral_updates: dict = field(default_factory=dict)
    referrer_updates: dict = field(default_factory=dict)


def decide_referral_reward(referral: Optional[dict], referee: Optional[dict],
                           referrer: Optional[dict], now: datetime) -> ReferralReward:
    """Credit the referrer once the referee has >= REFERRAL_MIN_LIVE_MINUTES
    of real live listening. Exactly once per referee; the referrer's total
    bonus is capped at REFERRAL_MAX_BONUS_SECONDS."""
    if not referral or referral.get("status") != "pending":
        return ReferralReward("none")
    if window_used(referee, "live_minutes", "lifetime", now) < REFERRAL_MIN_LIVE_MINUTES:
        return ReferralReward("not_ready")
    have = float((referrer or {}).get("stt_bonus_seconds") or 0)
    if have >= REFERRAL_MAX_BONUS_SECONDS:
        return ReferralReward("capped", {"status": "capped", "closed_at": now})
    return ReferralReward(
        "rewarded",
        {"status": "rewarded", "closed_at": now},
        {**add_bonus(referrer),
         "referral_rewards": int((referrer or {}).get("referral_rewards") or 0) + 1},
    )


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
        self._referrals: dict[str, dict] = {}
        self._ref_codes: dict[str, str] = {}
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


    # -- usage windows, referrals, deletion ---------------------------------

    async def add_usage(self, uid, key, amount, now):
        with self._lock:
            u = self._users.setdefault(uid, {})
            u.update(apply_usage(u, key, amount, now))

    async def delete_user(self, uid):
        with self._lock:
            n = 1 if self._users.pop(uid, None) is not None else 0
            reds = {r for r in self._redemptions if r[0] == uid}
            self._redemptions -= reds
            n += len(reds)
            if self._referrals.pop(uid, None) is not None:
                n += 1
            for code, owner in [c for c in self._ref_codes.items() if c[1] == uid]:
                del self._ref_codes[code]
                n += 1
            return n

    async def ensure_referral_code(self, uid):
        with self._lock:
            u = self._users.setdefault(uid, {})
            if u.get("referral_code"):
                return u["referral_code"]
            while True:
                c = normalize_code(generate_referral_code())
                if c not in self._ref_codes:
                    break
            self._ref_codes[c] = uid
            u["referral_code"] = c
            return c

    async def redeem_referral(self, referee, code, now):
        with self._lock:
            d = decide_referral_redeem(
                self._ref_codes.get(code), referee, self._users.get(referee),
                self._referrals.get(referee), now)
            if d.kind in ("granted", "covered"):
                self._users.setdefault(referee, {}).update(d.referee_updates)
                self._referrals[referee] = d.referral_doc
            eff = effective(self._users.get(referee), now)
        return RedeemResult(d.kind, eff.level, eff.expires_at)

    async def reward_referral(self, referee, now):
        with self._lock:
            ref = self._referrals.get(referee)
            r = decide_referral_reward(
                ref, self._users.get(referee),
                self._users.get((ref or {}).get("referrer")), now)
            if r.kind in ("rewarded", "capped"):
                ref.update(r.referral_updates)
                if r.referrer_updates:
                    self._users.setdefault(ref["referrer"], {}).update(r.referrer_updates)
        return r.kind

    def referral_doc(self, referee):
        return self._referrals.get(referee)


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

    def _add_usage(self, uid, key, amount, now):
        from google.cloud import firestore

        db = self._get_db()
        ref = db.collection(USERS).document(uid)

        @firestore.transactional
        def run(tx):
            snap = ref.get(transaction=tx)
            tx.set(ref, apply_usage(snap.to_dict() if snap.exists else None, key, amount, now),
                   merge=True)

        run(db.transaction())

    def _delete_user(self, uid):
        db = self._get_db()
        ref = db.collection(USERS).document(uid)
        n = 0
        for d in ref.collection("redemptions").stream():
            d.reference.delete()
            n += 1
        snap = ref.get()
        code = (snap.to_dict() or {}).get("referral_code") if snap.exists else None
        if code:
            db.collection(REFERRAL_CODES).document(code).delete()
            n += 1
        rr = db.collection(REFERRALS).document(uid)
        if rr.get().exists:
            rr.delete()
            n += 1
        if snap.exists:
            ref.delete()
            n += 1
        return n

    def _ensure_referral_code(self, uid):
        from google.api_core.exceptions import AlreadyExists

        db = self._get_db()
        ref = db.collection(USERS).document(uid)
        snap = ref.get()
        have = (snap.to_dict() or {}).get("referral_code") if snap.exists else None
        if have:
            return have
        for _ in range(10):
            c = normalize_code(generate_referral_code())
            try:
                db.collection(REFERRAL_CODES).document(c).create({"uid": uid})
            except AlreadyExists:
                continue
            ref.set({"referral_code": c}, merge=True)
            return c
        raise RuntimeError("could not allocate a referral code")

    def _redeem_referral(self, referee, code, now):
        from google.cloud import firestore

        db = self._get_db()
        code_ref = db.collection(REFERRAL_CODES).document(code)
        user_ref = db.collection(USERS).document(referee)
        ref_ref = db.collection(REFERRALS).document(referee)

        @firestore.transactional
        def run(tx):
            c = code_ref.get(transaction=tx)
            u = user_ref.get(transaction=tx)
            r = ref_ref.get(transaction=tx)
            udoc = u.to_dict() if u.exists else None
            d = decide_referral_redeem(
                (c.to_dict() or {}).get("uid") if c.exists else None, referee, udoc,
                r.to_dict() if r.exists else None, now)
            if d.kind in ("granted", "covered"):
                tx.set(user_ref, d.referee_updates, merge=True)
                tx.set(ref_ref, d.referral_doc)
                udoc = {**(udoc or {}), **d.referee_updates}
            return d.kind, udoc

        kind, udoc = run(db.transaction())
        eff = effective(udoc, now)
        return RedeemResult(kind, eff.level, eff.expires_at)

    def _reward_referral(self, referee, now):
        from google.cloud import firestore

        db = self._get_db()
        ref_ref = db.collection(REFERRALS).document(referee)
        pre = ref_ref.get()
        if not pre.exists or (pre.to_dict() or {}).get("status") != "pending":
            return "none"
        referrer = pre.to_dict()["referrer"]
        user_ref = db.collection(USERS).document(referee)
        a_ref = db.collection(USERS).document(referrer)

        @firestore.transactional
        def run(tx):
            r = ref_ref.get(transaction=tx)
            u = user_ref.get(transaction=tx)
            a = a_ref.get(transaction=tx)
            d = decide_referral_reward(
                r.to_dict() if r.exists else None, u.to_dict() if u.exists else None,
                a.to_dict() if a.exists else None, now)
            if d.kind in ("rewarded", "capped"):
                tx.update(ref_ref, d.referral_updates)
                if d.referrer_updates:
                    tx.set(a_ref, d.referrer_updates, merge=True)
            return d.kind

        return run(db.transaction())

    async def add_usage(self, uid, key, amount, now):
        await asyncio.to_thread(self._add_usage, uid, key, amount, now)

    async def delete_user(self, uid):
        return await asyncio.to_thread(self._delete_user, uid)

    async def ensure_referral_code(self, uid):
        return await asyncio.to_thread(self._ensure_referral_code, uid)

    async def redeem_referral(self, referee, code, now):
        return await asyncio.to_thread(self._redeem_referral, referee, code, now)

    async def reward_referral(self, referee, now):
        return await asyncio.to_thread(self._reward_referral, referee, now)

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


# Usage that feeds the lifetime/monthly windows and the referral reward. Recorded
# in memory (hot path) and persisted by usage_meter's flusher via flush_usage().
_TRACKED = {"stt.seconds": "stt_seconds", "live.minutes": "live_minutes"}
_pending: dict[str, dict[str, float]] = {}


_LLM_TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens", "hedge_extra_input_tokens",
)


def _tracked_key(name: str):
    key = _TRACKED.get(name)
    if key:
        return key
    if name.startswith("llm.") and name.rpartition(".")[2] in _LLM_TOKEN_FIELDS:
        return "llm_tokens"
    return None


def note_usage(uid: str, counters) -> None:
    """Called from usage_meter.add (hot path: dict add under a lock)."""
    for name, value in counters.items():
        key = _tracked_key(name)
        if key and isinstance(value, (int, float)) and value > 0:
            with _cache_lock:
                p = _pending.setdefault(uid, {})
                p[key] = p.get(key, 0.0) + value


def drop_pending(uid: str) -> None:
    with _cache_lock:
        _pending.pop(uid, None)


def clear_pending() -> None:
    with _cache_lock:
        _pending.clear()


def window_used_now(uid: str, key: str, window: str) -> float:
    """Persisted (as of the last prime) + not-yet-persisted usage."""
    now = datetime.now(timezone.utc)
    with _cache_lock:
        hit = _cache.get(uid)
        pend = _pending.get(uid, {}).get(key, 0.0)
    return window_used(hit[0] if hit else None, key, window, now) + pend


async def flush_usage() -> None:
    """Persist pending window usage and credit referral rewards. Never raises."""
    store = get_store()
    if store is None:
        return
    with _cache_lock:
        batch = {u: dict(p) for u, p in _pending.items() if p}
        _pending.clear()
    now = datetime.now(timezone.utc)
    for uid, keys in batch.items():
        for key, amount in keys.items():
            try:
                await store.add_usage(uid, key, amount, now)
            except Exception:  # noqa: BLE001
                logger.warning("plan usage flush failed uid=%s", uid, exc_info=True)
                with _cache_lock:
                    p = _pending.setdefault(uid, {})
                    p[key] = p.get(key, 0.0) + amount
                continue
            with _cache_lock:  # keep the cached doc current without a re-read
                hit = _cache.get(uid)
                if hit:
                    doc = hit[0] if hit[0] is not None else {}
                    doc.update(apply_usage(doc, key, amount, now))
                    _cache[uid] = (doc, hit[1])
        if "live_minutes" in keys:
            await _maybe_reward_referrer(store, uid, now)


async def _maybe_reward_referrer(store, uid: str, now: datetime) -> None:
    try:
        kind = await store.reward_referral(uid, now)
        if kind == "rewarded":
            logger.info("referral reward credited (referee uid=%s)", uid)
            with _cache_lock:
                _cache.clear()  # the referrer's cached plan is now stale
    except Exception:  # noqa: BLE001
        logger.warning("referral reward failed uid=%s", uid, exc_info=True)


def bonus_seconds(uid: str) -> float:
    with _cache_lock:
        hit = _cache.get(uid)
    return min(REFERRAL_MAX_BONUS_SECONDS,
               float(((hit[0] if hit else None) or {}).get("stt_bonus_seconds") or 0))


def window_cap(uid: str, level: str, name: str) -> float:
    """Cap for one window rule. A free user's referral bonus adds listening
    hours to BOTH the STT and the AI-token lifetime caps (they stay in step)."""
    table_key = WINDOW_RULES[name][0]
    cap = float(limit_for(level, table_key) or 0)
    if cap > 0 and name == "stt_lifetime":
        cap += bonus_seconds(uid)
    elif cap > 0 and name == "llm_lifetime":
        cap += bonus_seconds(uid) / 3600 * TOKENS_PER_HOUR
    return cap


def window_hours(name: str, cap: float) -> float:
    """A window cap expressed in listening hours."""
    counter = WINDOW_RULES[name][2]
    if counter == "stt_seconds":
        return cap / 3600
    if counter == "llm_tokens":
        return cap / TOKENS_PER_HOUR
    return cap / 60


def _hrs(h: float) -> str:
    h = round(h, 1)
    n = int(h) if h == int(h) else h
    return f"{n} hour" if n == 1 else f"{n} hours"


def window_message(name: str, cap: float) -> str:
    """The friendly sentence for an exhausted window (shown verbatim in the app)."""
    kind = WINDOW_RULES[name][1]
    hrs = _hrs(window_hours(name, cap))
    if kind == "lifetime":
        return (f"You're out of free time: you've used your {hrs} of free listening. "
                "Redeem a code or upgrade to keep coaching. Your recordings and "
                "notes stay safe.")
    return (f"You've used this month's {hrs} of listening. It refreshes at the "
            "start of next month; you can also upgrade for more.")


def listening_left(uid: str, level: str) -> Optional[dict]:
    """The BINDING listening allowance for ``uid`` right now, in seconds: the
    smallest of every window / daily cap that applies, with AI tokens converted
    at TOKENS_PER_HOUR. None when nothing is capped. Used by GET /plan so the
    app can show 'time left'."""
    best = None

    def consider(remaining_s: float, period: str, name: str, cap_s: float):
        nonlocal best
        if best is None or remaining_s < best["remaining_seconds"]:
            best = {"remaining_seconds": max(0, int(remaining_s)), "period": period,
                    "limit": name, "allowance_seconds": int(cap_s)}

    for name, (_, kind, counter) in WINDOW_RULES.items():
        if counter == "live_minutes":
            continue
        cap = window_cap(uid, level, name)
        if cap <= 0:
            continue
        used = window_used_now(uid, counter, kind)
        scale = 1.0 if counter == "stt_seconds" else 3600.0 / TOKENS_PER_HOUR
        consider((cap - used) * scale, kind, name, cap * scale)
    try:
        import usage_meter

        snap = usage_meter.meter().totals(uid)
        for lim, scale in (("stt_seconds", 1.0), ("llm_tokens", 3600.0 / TOKENS_PER_HOUR)):
            cap = float(limit_for(level, lim) or 0)
            if cap > 0:
                used = usage_meter.meter()._used_from(snap, lim)  # noqa: SLF001
                consider((cap - used) * scale, "day", lim, cap * scale)
    except Exception:  # noqa: BLE001 - the allowance still renders without it
        logger.debug("daily listening view failed", exc_info=True)
    return best


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


def tier_catalog() -> list[dict]:
    """The paid ladder for the app's plan cards (numbers come from the tables
    above, so the UI never hard-codes them)."""
    return [
        {"level": lv, "label": LABELS[lv], "price_usd": PRICE_USD[lv],
         "hours_per_month": ALLOWANCE_HOURS[lv], "library_items": LIBRARY[lv][0],
         "popular": lv == "plus", "business": lv == "max"}
        for lv in LEVELS if lv != "free"
    ]
