"""RevenueCat webhook: auth, every event type, ordering, promo safety, credit days."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

import plans
import revenuecat
from main import app

pytestmark = pytest.mark.anyio

FIX = json.loads((Path(__file__).parent / "fixtures" / "revenuecat_events.json").read_text())
T = 1_800_000_000_000  # event time in the fixtures
EXP = datetime.fromtimestamp((T + 30 * 86400 * 1000) / 1000, tz=timezone.utc)
SECRET = "s3cret-value"
H = {"Authorization": SECRET}
UID = "uid-alpha1"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    store = plans.MemoryPlanStore()
    plans.set_store(store)
    plans.clear_cache()
    monkeypatch.setenv("MINDSHIFT_REVENUECAT_WEBHOOK_SECRET", SECRET)
    yield store
    plans.set_store(None)
    plans.clear_cache()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def fx(name, **over):
    body = copy.deepcopy(FIX[name])
    body["event"].update(over)
    return body


async def send(client, body, headers=H):
    return await client.post("/billing/revenuecat", json=body, headers=headers)


# -- auth ---------------------------------------------------------------------

async def test_unset_secret_is_503_never_open(client, monkeypatch):
    monkeypatch.delenv("MINDSHIFT_REVENUECAT_WEBHOOK_SECRET")
    assert (await send(client, fx("INITIAL_PURCHASE"), headers={})).status_code == 503
    assert (await send(client, fx("INITIAL_PURCHASE"))).status_code == 503


async def test_bad_or_missing_auth_is_401(client, env):
    assert (await send(client, fx("INITIAL_PURCHASE"), headers={})).status_code == 401
    assert (await send(client, fx("INITIAL_PURCHASE"), headers={"Authorization": "nope"})).status_code == 401
    assert await env.get(UID) is None


def test_verify_auth_forms():
    assert revenuecat.verify_auth(f"Bearer {SECRET}", SECRET)
    assert revenuecat.verify_auth(SECRET, SECRET)
    assert not revenuecat.verify_auth("", SECRET) and not revenuecat.verify_auth(SECRET, "")


async def test_no_event_is_400(client):
    assert (await send(client, {"api_version": "1.0"})).status_code == 400


# -- grants -------------------------------------------------------------------

@pytest.mark.parametrize("typ", ["INITIAL_PURCHASE", "RENEWAL", "UNCANCELLATION",
                                 "NON_RENEWING_PURCHASE", "SUBSCRIPTION_EXTENDED",
                                 "TEMPORARY_ENTITLEMENT_GRANT", "REFUND_REVERSED"])
async def test_grant_events_set_level_and_expiry(client, env, typ):
    r = await send(client, fx(typ))
    assert r.status_code == 200 and r.json()["status"] == "ok"
    doc = await env.get(UID)
    assert doc["plan"] == "pro" and doc["plan_source"] == "revenuecat"
    assert doc["plan_expires_at"] == EXP and doc["plan_status"] == "active"
    assert doc["plan_will_renew"] is (typ != "NON_RENEWING_PURCHASE")


async def test_level_comes_from_product_and_entitlement_highest_wins(client, env):
    await send(client, fx("INITIAL_PURCHASE", product_id="mindshift_basic_monthly",
                          entitlement_ids=["basic"]))
    assert (await env.get(UID))["plan"] == "basic"
    # product says max though only the basic entitlement is listed: still max
    await send(client, fx("RENEWAL", product_id="mindshift_max_monthly:monthly",
                          entitlement_ids=["basic"], event_timestamp_ms=T + 1))
    assert (await env.get(UID))["plan"] == "max"


async def test_level_in_force_only_while_unexpired(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    doc = await env.get(UID)
    assert plans.effective(doc, datetime.fromtimestamp(T / 1000 + 86400, tz=timezone.utc)).level == "pro"
    assert plans.effective(doc, datetime.fromtimestamp(T / 1000 + 40 * 86400, tz=timezone.utc)).level == "free"


async def test_cancellation_keeps_until_expiry(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    await send(client, fx("CANCELLATION", event_timestamp_ms=T + 10))
    doc = await env.get(UID)
    assert doc["plan"] == "pro" and doc["plan_status"] == "cancelled"
    assert doc["plan_will_renew"] is False and doc["plan_expires_at"] == EXP


async def test_refund_cancellation_ends_now(client, env):
    await send(client, fx("CANCELLATION", expiration_at_ms=1_700_000_000_000))  # in the past
    assert plans.effective(await env.get(UID)).level == "free"


async def test_billing_issue_grace_period(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    await send(client, fx("BILLING_ISSUE", event_timestamp_ms=T + 5))
    doc = await env.get(UID)
    assert doc["plan_status"] == "billing_issue" and doc["plan_billing_issue"] is True
    assert doc["plan_expires_at"].timestamp() * 1000 == T + 35 * 86400 * 1000


async def test_paused_only_flags(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    await send(client, fx("SUBSCRIPTION_PAUSED", event_timestamp_ms=T + 5))
    doc = await env.get(UID)
    assert doc["plan"] == "pro" and doc["plan_status"] == "paused" and not doc["plan_will_renew"]


async def test_expiration_is_free(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    await send(client, fx("EXPIRATION", event_timestamp_ms=T + 20))
    doc = await env.get(UID)
    assert doc["plan"] == "free" and doc["plan_status"] == "expired"
    assert plans.effective(doc).level == "free"


async def test_transfer_downgrades_losing_uid_only(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    r = await send(client, FIX["TRANSFER"])
    assert r.status_code == 200
    doc = await env.get(UID)
    assert doc["plan"] == "free" and doc["plan_status"] == "transferred"
    assert await env.get("uid-beta22") is None


# -- ignored ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["TEST", "PRODUCT_CHANGE", "ANONYMOUS"])
async def test_ignored_events_answer_200_and_change_nothing(client, env, name):
    r = await send(client, FIX[name])
    assert r.status_code == 200 and r.json()["status"] == "ignored"
    assert await env.get(UID) is None


async def test_unrelated_product_ignored(client, env):
    r = await send(client, fx("INITIAL_PURCHASE", product_id="other_app_pro", entitlement_ids=["premium"]))
    assert r.status_code == 200 and await env.get(UID) is None


# -- ordering / idempotency -----------------------------------------------------

async def test_older_event_skipped_newer_applied(client, env):
    await send(client, fx("RENEWAL", event_timestamp_ms=T + 100))
    r = await send(client, fx("EXPIRATION", event_timestamp_ms=T + 50))
    assert r.json()["status"] == "ignored"
    assert (await env.get(UID))["plan"] == "pro"


async def test_redelivery_is_idempotent(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    first = {k: v for k, v in (await env.get(UID)).items() if k != "plan_updated_at"}
    await send(client, fx("INITIAL_PURCHASE"))
    second = {k: v for k, v in (await env.get(UID)).items() if k != "plan_updated_at"}
    assert first == second


async def test_expiration_of_old_product_after_upgrade_is_skipped(client, env):
    await send(client, fx("INITIAL_PURCHASE", product_id="mindshift_basic_monthly",
                          entitlement_ids=["basic"], event_timestamp_ms=T - 10))
    await send(client, fx("INITIAL_PURCHASE", event_timestamp_ms=T + 5))  # moved to pro
    await send(client, fx("EXPIRATION", product_id="mindshift_basic_monthly",
                          entitlement_ids=["basic"], event_timestamp_ms=T + 6))
    assert (await env.get(UID))["plan"] == "pro"


# -- promo safety and banked credit ---------------------------------------------

async def test_stronger_promo_is_not_clobbered(client, env):
    await env.set_fields(UID, {"plan": "max", "plan_source": "promo",
                               "plan_expires_at": datetime(2040, 1, 1, tzinfo=timezone.utc)})
    await send(client, fx("INITIAL_PURCHASE"))
    doc = await env.get(UID)
    assert doc["plan"] == "max" and doc["plan_source"] == "promo"
    assert doc["store_product_id"] == "mindshift_pro_monthly"
    await send(client, fx("EXPIRATION", event_timestamp_ms=T + 9))
    assert plans.effective(await env.get(UID)).level == "max"


async def test_banked_credit_days_start_when_store_sub_ends(client, env):
    await send(client, fx("INITIAL_PURCHASE"))
    await env.set_fields(UID, {"credit_days": 7, "credit_plan": "plus"})
    now = datetime.now(timezone.utc)
    await send(client, fx("EXPIRATION", event_timestamp_ms=T + 9))
    doc = await env.get(UID)
    assert doc["plan"] == "plus" and doc["plan_source"] == "promo" and doc["credit_days"] == 0
    assert 6.9 < (doc["plan_expires_at"] - now).total_seconds() / 86400 < 7.1


# -- /plan reports purchase availability -----------------------------------------

def test_plan_purchase_available_follows_config(monkeypatch):
    from routers import plan as plan_router

    assert plan_router._purchase_info(False) == {"available": True}
    assert plan_router._purchase_info(True)["available"] is False
    monkeypatch.delenv("MINDSHIFT_REVENUECAT_WEBHOOK_SECRET")
    assert plan_router._purchase_info(False)["available"] is False
