"""GET /plan and POST /plan/redeem (server/plans.py has the rules)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

import plans
import usage_meter
from auth import Identity, get_current_identity

logger = logging.getLogger(__name__)
router = APIRouter(tags=["plan"])


async def _rate_limit(request: Request) -> None:
    import main  # lazy: main includes this router

    await main._rate_limit(request)


class RedeemBody(BaseModel):
    code: str = Field(min_length=1, max_length=64)


async def _library_usage(uid: str) -> dict:
    try:
        import library

        items = await library.get_service().store.list_items(uid)
        return {"items": len(items), "chars": sum(int(i.chars or 0) for i in items)}
    except Exception:  # noqa: BLE001 - the plan card must render without it
        return {"items": None, "chars": None}


async def plan_summary(uid: str, is_guest: bool) -> dict:
    await plans.prime(uid, force=True)
    eff = plans.effective_cached(uid)
    level = eff.level
    meter = usage_meter.meter()
    daily = {}
    for key in plans.DAILY_KEYS:
        cap = plans.limit_for(level, key)
        daily[key] = {"cap": cap, "used": round(meter.limit_used(uid, key), 3)}
    lib = await _library_usage(uid)
    nxt = plans.upgrade_level(level)
    return {
        "plan": level,
        "label": plans.LABELS[level],
        "expires_at": eff.expires_at.isoformat() if eff.expires_at else None,
        "source": eff.source,
        "is_guest": is_guest,
        "limits": {k: plans.limit_for(level, k) for k in plans.LIMIT_KEYS},
        "daily": daily,
        "library": lib,
        "flags": {k: bool(plans.limit_for(level, k)) for k in plans.FLAG_KEYS},
        "upgrade_plan": nxt,
        "redeem_available": plans.get_store() is not None and not is_guest,
        # Phase 2 turns this on (RevenueCat); until then there is no purchase.
        "purchase": {"available": False, "message": "Paid plans are coming soon."},
    }


@router.get("/plan")
async def get_plan(identity: Identity = Depends(get_current_identity)) -> dict:
    return await plan_summary(identity.uid, identity.is_guest)


_BAD = {
    "type": "redeem_error",
    "reason": "invalid_code",
    "message": "That code isn't valid or has already been used up.",
}


@router.post("/plan/redeem")
async def redeem(
    body: RedeemBody,
    identity: Identity = Depends(get_current_identity),
    _rl: None = Depends(_rate_limit),
) -> dict:
    uid = identity.uid
    if identity.is_guest:
        raise HTTPException(403, {
            "type": "redeem_error", "reason": "sign_in_required",
            "message": "Create an account (or sign in) to redeem a code.",
        })
    store = plans.get_store()
    if store is None:
        raise HTTPException(503, {
            "type": "redeem_error", "reason": "unavailable",
            "message": "Plans aren't available right now. Please try again later.",
        })
    if plans.redeem_blocked(uid):
        raise HTTPException(429, {
            "type": "redeem_error", "reason": "too_many_attempts",
            "message": "Too many attempts. Please wait a while and try again.",
        }, headers={"Retry-After": "3600"})
    code = plans.normalize_code(body.code)
    now = datetime.now(timezone.utc)
    try:
        result = await store.redeem(uid, code, now)
    except Exception:  # noqa: BLE001
        logger.exception("redeem failed uid=%s", uid)
        raise HTTPException(503, {
            "type": "redeem_error", "reason": "unavailable",
            "message": "Couldn't redeem right now. Please try again.",
        })
    if result.kind == "invalid":
        plans.note_redeem_failure(uid)
        raise HTTPException(400, _BAD)
    plans.invalidate(uid)
    logger.info("plan redeem uid=%s outcome=%s", uid, result.kind)
    return {"status": result.kind, "plan": await plan_summary(uid, False)}
