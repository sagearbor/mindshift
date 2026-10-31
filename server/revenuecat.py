"""RevenueCat webhook rules: one event -> the ms_plans field changes it implies.

Mirrors oralhistory_timeline functions/src/revenuecat.ts (``mapEventToChanges``,
``eventPlanLevel``, ``isSupersededProductEvent``) for MindShift's ladder
(basic < plus < pro < max, products ``mindshift_<level>_monthly``). Everything
here is pure so the router and the tests share one source of truth.

Rules in short:
* app_user_id == our Firebase uid (the app calls ``Purchases.logIn(uid)``,
  never for guests); ``$RCAnonymousID`` ids are ignored.
* Grant types set level + expiry; CANCELLATION keeps the level until expiry;
  BILLING_ISSUE keeps it until the grace period ends; EXPIRATION -> free;
  TRANSFER downgrades the losing uid; TEST and everything else is ignored.
* An event older than ``plan_last_event_ms`` is skipped (RevenueCat can retry or
  deliver out of order); a negative event for a product the user has moved off
  is skipped while the stored, different product is still in force.
* A promo/admin grant that is stronger (higher level, or same level lasting
  longer) is never clobbered: only store metadata is recorded.
* When a store subscription ends and the user banked ``credit_days`` (with
  ``credit_plan``), those days start then.
"""

from __future__ import annotations

import hmac
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import plans

PRODUCT_RE = re.compile(r"^mindshift_(basic|plus|pro|max)_", re.I)
UID_RE = re.compile(r"^[A-Za-z0-9_-]{6,128}$")

GRANT_TYPES = frozenset({
    "INITIAL_PURCHASE", "RENEWAL", "UNCANCELLATION", "NON_RENEWING_PURCHASE",
    "SUBSCRIPTION_EXTENDED", "TEMPORARY_ENTITLEMENT_GRANT", "REFUND_REVERSED",
})
STRONG_SOURCES = ("promo", "admin")


def verify_auth(header: Optional[str], secret: str) -> bool:
    """Constant-time compare; accepts the value verbatim or as ``Bearer <v>``."""
    if not secret or not header:
        return False
    h = header.strip()
    want = secret.encode()
    ok = False
    for cand in (h, re.sub(r"^Bearer\s+", "", h, flags=re.I)):
        ok |= hmac.compare_digest(cand.encode(), want)
    return ok


def _valid_uid(x: Any) -> bool:
    return isinstance(x, str) and not x.startswith("$RCAnonymousID") and bool(UID_RE.match(x))


def resolve_uid(e: dict) -> Optional[str]:
    for x in [e.get("app_user_id"), e.get("original_app_user_id"), *(e.get("aliases") or [])]:
        if _valid_uid(x):
            return x
    return None


def product_level(pid: Any) -> Optional[str]:
    m = PRODUCT_RE.match(str(pid or ""))
    return m.group(1).lower() if m else None


def event_level(e: dict) -> Optional[str]:
    """Highest paid level named by the event's entitlements or product id."""
    found = []
    ents = list(e.get("entitlement_ids") or []) + ([e["entitlement_id"]] if e.get("entitlement_id") else [])
    for x in ents:
        x = str(x).lower()
        if x in plans.LEVELS and x != "free":
            found.append(x)
    pl = product_level(e.get("product_id"))
    if pl:
        found.append(pl)
    return max(found, key=plans.level_rank) if found else None


def _ms(v: Any) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(int(v) / 1000, tz=timezone.utc) if v is not None else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def targets(e: dict) -> list[tuple[str, str]]:
    """[(uid, mode)] this event touches; mode is ``normal`` or ``lost``."""
    if e.get("type") == "TRANSFER":
        return [(u, "lost") for u in (e.get("transferred_from") or []) if _valid_uid(u)]
    uid = resolve_uid(e)
    return [(uid, "normal")] if uid else []


def _base_pid(pid: Any) -> str:
    return str(pid or "").split(":")[0]


def _meta(e: dict, now: datetime) -> dict:
    out = {
        "plan_last_event_ms": int(e.get("event_timestamp_ms") or 0),
        "plan_last_event_type": e.get("type"),
        "plan_updated_at": now,
    }
    if e.get("product_id"):
        out["store_product_id"] = e["product_id"]
    if e.get("store"):
        out["store"] = str(e["store"]).lower()
    if e.get("environment"):
        out["store_environment"] = str(e["environment"]).lower()
    return out


def _ended(doc: Optional[dict], now: datetime) -> dict:
    """Fields once a store subscription is over: free, or banked credit days."""
    days = int((doc or {}).get("credit_days") or 0)
    lvl = (doc or {}).get("credit_plan")
    if days > 0 and lvl in plans.LIMITS and lvl != "free":
        return {"plan": lvl, "plan_expires_at": now + timedelta(days=days),
                "plan_source": "promo", "credit_days": 0}
    return {"plan": "free", "plan_expires_at": now, "plan_source": "revenuecat"}


def _strong(cur: plans.Effective) -> bool:
    return cur.level != "free" and cur.source in STRONG_SOURCES


def plan_changes(e: dict, mode: str, doc: Optional[dict], now: datetime) -> Optional[dict]:
    """The fields to merge into ``ms_plans/{uid}``, or None to skip."""
    typ = e.get("type")
    ev_ms = int(e.get("event_timestamp_ms") or 0)
    last = int((doc or {}).get("plan_last_event_ms") or 0)
    if ev_ms and last and ev_ms < last:
        return None  # stale / out of order
    cur = plans.effective(doc, now)
    meta = _meta(e, now)

    if mode == "lost":
        if _strong(cur):
            return meta
        return {**meta, **_ended(doc, now), "plan_status": "transferred", "plan_will_renew": False}

    level = event_level(e)
    if level is None:
        return None
    exp = _ms(e.get("expiration_at_ms"))

    if typ in GRANT_TYPES:
        want = {"plan": level, "plan_expires_at": exp, "plan_source": "revenuecat",
                "plan_status": "active", "plan_will_renew": typ != "NON_RENEWING_PURCHASE",
                "plan_billing_issue": False}
    elif typ == "CANCELLATION":
        want = {"plan": level, "plan_expires_at": exp, "plan_source": "revenuecat",
                "plan_status": "cancelled", "plan_will_renew": False}
    elif typ == "BILLING_ISSUE":
        want = {"plan": level, "plan_source": "revenuecat", "plan_status": "billing_issue",
                "plan_billing_issue": True,
                "plan_expires_at": _ms(e.get("grace_period_expiration_at_ms")) or exp}
    elif typ == "SUBSCRIPTION_PAUSED":
        return {**meta, "plan_status": "paused", "plan_will_renew": False}
    elif typ == "EXPIRATION":
        want = {**_ended(doc, now), "plan_status": "expired",
                "plan_will_renew": False, "plan_billing_issue": False}
    else:
        return None

    if typ in ("EXPIRATION", "CANCELLATION", "BILLING_ISSUE"):
        incoming, current = _base_pid(e.get("product_id")), _base_pid((doc or {}).get("store_product_id"))
        if incoming and current and incoming != current and cur.level != "free":
            return None  # the user already moved to another product

    if _strong(cur):
        if typ == "EXPIRATION":
            return meta  # a promo/admin grant outlives the store subscription
        stronger = plans.level_rank(cur.level) > plans.level_rank(want["plan"]) or (
            cur.level == want["plan"]
            and (cur.expires_at is None
                 or (want.get("plan_expires_at") and cur.expires_at >= want["plan_expires_at"])))
        if stronger:
            return meta
    return {**meta, **want}
