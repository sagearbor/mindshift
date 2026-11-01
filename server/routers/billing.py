"""POST /billing/revenuecat: RevenueCat webhook -> ms_plans (rules: server/revenuecat.py).

Auth is the Authorization header, compared in constant time with
``MINDSHIFT_REVENUECAT_WEBHOOK_SECRET``. Unset secret => 503 (never open).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException

import plans
import revenuecat

logger = logging.getLogger(__name__)
router = APIRouter(tags=["billing"])


def webhook_secret() -> str:
    return (os.environ.get("MINDSHIFT_REVENUECAT_WEBHOOK_SECRET") or "").strip()


@router.post("/billing/revenuecat")
async def revenuecat_webhook(payload: dict, authorization: str = Header(default="")) -> dict:
    secret = webhook_secret()
    if not secret:
        raise HTTPException(503, "Webhook auth not configured")
    if not revenuecat.verify_auth(authorization, secret):
        raise HTTPException(401, "Unauthorized")
    event = payload.get("event")
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise HTTPException(400, "No event")
    targets = revenuecat.targets(event)
    if not targets:
        return {"status": "ignored"}  # TEST, anonymous ids, unrelated events
    store = plans.get_store()
    if store is None:
        raise HTTPException(503, "Plans unavailable")  # RevenueCat retries
    now = datetime.now(timezone.utc)
    applied = 0
    try:
        for uid, mode in targets:
            if await store.apply_atomic(
                uid, lambda doc, m=mode: revenuecat.plan_changes(event, m, doc, now)
            ):
                applied += 1
                plans.invalidate(uid)
    except Exception:  # noqa: BLE001 - 5xx makes RevenueCat retry with backoff
        logger.exception("revenuecat webhook failed type=%s", event.get("type"))
        raise HTTPException(500, "Error")
    logger.info("revenuecat event type=%s id=%s applied=%d", event.get("type"), event.get("id"), applied)
    return {"status": "ok" if applied else "ignored"}
