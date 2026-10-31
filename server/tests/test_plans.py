"""Plans, effective level, redeem codes, usage_meter wiring, library gate."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import plans
import usage_meter
from main import app

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
A = {"X-Test-Uid": "user-a"}
B = {"X-Test-Uid": "user-b"}


@pytest.fixture(autouse=True)
def plan_env():
    store = plans.MemoryPlanStore()
    plans.set_store(store)
    plans.clear_cache()
    usage_meter.meter().reset()
    yield store
    plans.set_store(None)
    plans.clear_cache()


@pytest.fixture
async def client():
    import main

    main._rate_limiter.reset()
    plans.reset_redeem_failures()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


def _code(**kw):
    d = {"code": "ABCD2345EFGH", "plan": "pro", "days": 30, "max_uses": 2, "uses": 0,
         "expires_at": None, "revoked": False, "note": "t", "created_by": "test"}
    d.update(kw)
    return d


# -- table / effective level -------------------------------------------------

def test_free_defaults_equal_todays_usage_meter_caps():
    assert plans.limit_for("free", "llm_tokens") == usage_meter.DAILY_LLM_TOKEN_CAP
    assert plans.limit_for("free", "stt_seconds") == usage_meter.DAILY_STT_SECONDS_CAP
    assert plans.limit_for("free", "calls") == usage_meter.DAILY_CALLS_CAP


def test_free_follows_usage_meter_monkeypatch(monkeypatch):
    monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 123)
    assert plans.limit_for("free", "llm_tokens") == 123


def test_pro_is_larger_than_free_and_env_overrides(monkeypatch):
    assert plans.limit_for("pro", "library_items") > plans.limit_for("free", "library_items")
    monkeypatch.setenv("MINDSHIFT_PLAN_PRO_LIBRARY_ITEMS", "7")
    assert plans.limit_for("pro", "library_items") == 7


def test_every_level_defines_every_limit_key():
    for level in plans.LEVELS:
        for key in plans.LIMIT_KEYS:
            assert plans.limit_for(level, key) is not None


def test_effective_only_while_unexpired():
    live = {"plan": "pro", "plan_expires_at": NOW + timedelta(days=1), "plan_source": "promo"}
    assert plans.effective(live, NOW).level == "pro"
    dead = {"plan": "pro", "plan_expires_at": NOW - timedelta(seconds=1), "plan_source": "promo"}
    assert plans.effective(dead, NOW).level == "free"
    assert plans.effective({"plan": "pro", "plan_expires_at": None}, NOW).level == "pro"
    assert plans.effective({"plan": "platinum"}, NOW).level == "free"
    assert plans.effective(None, NOW).level == "free"


# -- redeem decision ---------------------------------------------------------

def test_grant_from_free():
    d = plans.decide_redeem(_code(), False, None, NOW)
    assert d.kind == "granted"
    assert d.user_updates["plan"] == "pro"
    assert d.user_updates["plan_source"] == "promo"
    assert d.user_updates["plan_expires_at"] == NOW + timedelta(days=30)
    assert d.code_updates == {"uses": 1}


def test_grant_extends_same_level_never_shortens():
    user = {"plan": "pro", "plan_source": "promo", "plan_expires_at": NOW + timedelta(days=10)}
    d = plans.decide_redeem(_code(days=5), False, user, NOW)
    assert d.user_updates["plan_expires_at"] == NOW + timedelta(days=15)


def test_unlimited_or_paid_user_is_covered_and_code_not_consumed():
    user = {"plan": "pro", "plan_source": "admin", "plan_expires_at": None}
    d = plans.decide_redeem(_code(), False, user, NOW)
    assert d.kind == "covered" and not d.code_updates
    paid = {"plan": "pro", "plan_source": "revenuecat", "plan_expires_at": NOW + timedelta(days=3)}
    assert plans.decide_redeem(_code(), False, paid, NOW).kind == "covered"


def test_idempotent_per_user_and_code():
    assert plans.decide_redeem(_code(), True, None, NOW).kind == "already_redeemed"


@pytest.mark.parametrize("over", [
    {"revoked": True}, {"uses": 2}, {"expires_at": NOW - timedelta(days=1)},
    {"plan": "platinum"}, {"days": 0},
])
def test_bad_codes_are_invalid(over):
    assert plans.decide_redeem(_code(**over), False, None, NOW).kind == "invalid"
    assert plans.decide_redeem(None, False, None, NOW).kind == "invalid"


def test_normalize_code():
    assert plans.normalize_code(" abcd-2345 efgh ") == "ABCD2345EFGH"


def test_generated_codes_use_unambiguous_charset():
    for _ in range(50):
        c = plans.generate_code()
        raw = plans.normalize_code(c)
        assert len(raw) == 12
        assert not set(raw) & set("01OILU")


# -- store: atomic use counting ---------------------------------------------

async def test_store_respects_max_uses(plan_env):
    await plan_env.create_code(_code(max_uses=1))
    r1 = await plan_env.redeem("u1", "ABCD2345EFGH", NOW)
    r2 = await plan_env.redeem("u2", "ABCD2345EFGH", NOW)
    r1b = await plan_env.redeem("u1", "ABCD2345EFGH", NOW)
    assert (r1.kind, r2.kind, r1b.kind) == ("granted", "invalid", "already_redeemed")


# -- HTTP ---------------------------------------------------------------------

async def test_plan_endpoint_defaults_to_free(client):
    r = await client.get("/plan", headers=A)
    assert r.status_code == 200
    body = r.json()
    assert body["plan"] == "free" and body["expires_at"] is None
    assert body["limits"]["library_items"] == plans.limit_for("free", "library_items")
    assert body["purchase"]["available"] is False


async def test_redeem_flow(client, plan_env):
    await plan_env.create_code(_code())
    r = await client.post("/plan/redeem", json={"code": "abcd-2345-efgh"}, headers=A)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "granted" and r.json()["plan"]["plan"] == "pro"
    again = await client.post("/plan/redeem", json={"code": "ABCD2345EFGH"}, headers=A)
    assert again.json()["status"] == "already_redeemed"
    assert (await plan_env.get_code("ABCD2345EFGH"))["uses"] == 1
    me = await client.get("/plan", headers=A)
    assert me.json()["plan"] == "pro" and me.json()["source"] == "promo"
    other = await client.get("/plan", headers=B)
    assert other.json()["plan"] == "free"


async def test_redeem_bad_code_generic_error(client):
    r = await client.post("/plan/redeem", json={"code": "NOPE-NOPE-NOPE"}, headers=A)
    assert r.status_code == 400
    assert r.json()["detail"]["reason"] == "invalid_code"


async def test_redeem_guest_refused(client, plan_env):
    await plan_env.create_code(_code())
    r = await client.post("/plan/redeem", json={"code": "ABCD2345EFGH"},
                          headers={**A, "X-Test-Guest": "1"})
    assert r.status_code == 403


async def test_redeem_failures_are_rate_limited(client):
    codes = []
    for _ in range(plans.REDEEM_MAX_FAILURES + 2):
        r = await client.post("/plan/redeem", json={"code": "ZZZZ-ZZZZ-ZZZZ"}, headers=A)
        codes.append(r.status_code)
    assert codes[0] == 400 and codes[-1] == 429


async def test_redeem_unavailable_without_store(client):
    plans.set_store(None)
    r = await client.post("/plan/redeem", json={"code": "ABCD2345EFGH"}, headers=A)
    assert r.status_code == 503


# -- usage_meter wiring -------------------------------------------------------

async def test_pro_user_gets_pro_caps_and_notice_carries_plan(plan_env, monkeypatch):
    monkeypatch.setattr(usage_meter, "DAILY_LLM_TOKEN_CAP", 1000)
    monkeypatch.setenv("MINDSHIFT_PLAN_PRO_LLM_TOKENS", "5000")
    m = usage_meter.meter()
    m.add("u-pro", {usage_meter.llm_key("respond", "input_tokens"): 2000})
    m.add("u-free", {usage_meter.llm_key("respond", "input_tokens"): 2000})
    await plan_env.set_fields("u-pro", {"plan": "pro", "plan_source": "admin", "plan_expires_at": None})
    await plans.prime("u-pro", force=True)
    await plans.prime("u-free", force=True)
    assert m.check("u-pro", "batch_analysis") is None
    ex = m.check("u-free", "batch_analysis")
    assert ex is not None
    n = ex.notice()
    assert n["plan"] == "free" and n["upgrade_plan"] == "pro"


# -- library gate -------------------------------------------------------------

class _Svc:
    class store:  # noqa: N801
        items: list = []

        @classmethod
        async def list_items(cls, uid):
            return cls.items


class _Item:
    def __init__(self, chars):
        self.chars = chars


async def test_library_gate_blocks_new_never_deletes(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setenv("MINDSHIFT_PLAN_FREE_LIBRARY_ITEMS", "2")
    _Svc.store.items = [_Item(10), _Item(10)]
    with pytest.raises(HTTPException) as ei:
        await plans.check_library_create("u-free", _Svc)
    d = ei.value.detail
    assert ei.value.status_code == 429
    assert d["type"] == "plan_limit" and d["limit"] == "library_items"
    assert d["upgrade_plan"] == "pro"
    _Svc.store.items = [_Item(10)]
    await plans.check_library_create("u-free", _Svc)  # under the cap: fine


async def test_library_gate_size(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setenv("MINDSHIFT_PLAN_FREE_LIBRARY_CHARS", "100")
    _Svc.store.items = [_Item(90)]
    with pytest.raises(HTTPException):
        await plans.check_library_create("u-free", _Svc, incoming_chars=20)
    await plans.check_library_create("u-free", _Svc, incoming_chars=5)
