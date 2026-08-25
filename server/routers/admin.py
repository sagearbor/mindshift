"""Owner-only usage visibility — where the Anthropic / Deepgram / GCP bill
comes from.

* ``GET /admin/usage?since=YYYY-MM-DD`` — per-uid rollups of everything
  ``server/usage_meter.py`` counts (LLM tokens by call site, cloud STT
  seconds, live minutes, ONNX model downloads, calls started), summed over the
  UTC days from ``since`` to today. ``since`` defaults to today; the window is
  bounded by ``usage_meter.MAX_ROLLUP_DAYS`` because each day is a bucket
  prefix scan and an unbounded query would itself be a cost bug.

Access: the verified Firebase uid must appear in ``MINDSHIFT_ADMIN_UIDS`` (a
comma-separated allowlist). **Unset means closed** — a fresh deployment must
not expose every account's usage to whoever signs in first. A signed-in
non-admin gets 404, not 403: the endpoint's existence is not something a
regular user needs to learn.

Read-only. There is no admin write surface here on purpose — quotas are tuned
with env vars and a redeploy, which leaves an audit trail; an HTTP switch that
raises a spend cap would not.

``scripts/usage_report.py`` renders this payload as a table with a projected
monthly cost (prices + sources: docs/plans/2026-08-25-cost-model.md).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request

import usage_meter
from auth import get_current_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


async def _rate_limit(request: Request) -> None:
    """Reuse main's per-IP limiter, imported lazily at request time (main
    includes this router at module load — see routers/models.py)."""
    import main

    await main._rate_limit(request)


def require_admin(uid: str = Depends(get_current_uid)) -> str:
    """The allowlist gate. 404 rather than 403 — see the module docstring."""
    if not usage_meter.is_admin(uid):
        logger.info("Non-admin uid attempted /admin/usage")
        raise HTTPException(status_code=404, detail="Not Found")
    return uid


@router.get(
    "/usage",
    summary="Per-uid usage rollups since a UTC date (owner allowlist only)",
)
async def get_usage(
    request: Request,
    since: str | None = Query(
        default=None,
        description="First UTC day to include, YYYY-MM-DD. Defaults to today.",
    ),
    uid: str = Depends(require_admin),
    _rl: None = Depends(_rate_limit),
) -> dict:
    day = since or usage_meter.day_key()
    try:
        days = usage_meter.days_since(day)
    except ValueError:
        raise HTTPException(
            status_code=422, detail="since must be a UTC date, YYYY-MM-DD",
        )
    if not days:
        raise HTTPException(status_code=422, detail="since is in the future")

    import main

    store = main.get_recordings_store()
    per_uid = await usage_meter.rollup(store, days)
    # Sorted by the loudest spender first — the reason anyone opens this.
    ordered = sorted(
        per_uid.items(),
        key=lambda kv: -(kv[1].get("llm_input_tokens", 0)
                         + kv[1].get("llm_output_tokens", 0)),
    )
    return {
        "since": days[0],
        "until": days[-1],
        "days": len(days),
        "persistent": store is not None,
        "instance": usage_meter.INSTANCE_ID,
        "caps": {
            name: getattr(usage_meter, cap_attr)
            for name, (cap_attr, _, _) in usage_meter.LIMITS.items()
        },
        "users": [{"uid": u, **counters} for u, counters in ordered],
    }
