# ruff: noqa: F811, F401
"""Lifetime STT, 7-day code via mint script, account deletion, referrals."""

import argparse
from datetime import datetime, timedelta, timezone

import pytest

import plans
import usage_meter
from test_plans import A, B, NOW, _code, client, plan_env  # noqa: F401 (fixtures)

pytestmark = pytest.mark.anyio


# -- lifetime speech-to-text (free = 2 h total) -------------------------------

async def test_free_stt_is_a_lifetime_total_with_friendly_message(plan_env):
    m = usage_meter.meter()
    assert plans.limit_for("free", "stt_lifetime_seconds") == 3600
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 3599})
    assert m.check("u-life", "cloud_transcription") is None
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 5})
    ex = m.check("u-life", "cloud_transcription")
    assert ex is not None and ex.limit == "stt_lifetime"
    assert "out of free time" in ex.notice()["message"]
    assert "1 hour of free listening" in ex.notice()["message"]
    assert m.check("u-life", "batch_analysis") is None  # AI tokens are a separate window


async def test_lifetime_survives_flush_restart_and_redeem(plan_env, monkeypatch):
    m = usage_meter.meter()
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 3700})
    await m.flush()
    assert plan_env._users["u-life"]["lifetime"]["stt_seconds"] == 3700
    m.reset()
    plans.clear_cache()
    await plans.prime("u-life", force=True)  # "restart": only the store remembers
    assert m.check("u-life", "cloud_transcription").limit == "stt_lifetime"
    monkeypatch.setattr(usage_meter, "day_key", lambda when=None: "2099-01-01")
    assert m.check("u-life", "cloud_transcription").limit == "stt_lifetime"
    await plan_env.create_code(_code(plan="basic"))
    await plan_env.redeem("u-life", "ABCD2345EFGH", NOW)
    assert plan_env._users["u-life"]["lifetime"]["stt_seconds"] == 3700
    await plans.prime("u-life", force=True)
    assert m.check("u-life", "cloud_transcription") is None  # basic: monthly 8 h


def test_monthly_window_resets_lifetime_does_not():
    oct31 = datetime(2026, 10, 31, tzinfo=timezone.utc)
    doc = plans.apply_usage(None, "stt_seconds", 100, oct31)
    assert plans.window_used(doc, "stt_seconds", "month", oct31) == 100
    assert plans.window_used(doc, "stt_seconds", "month", NOW) == 0
    assert plans.window_used(doc, "stt_seconds", "lifetime", NOW) == 100


# -- 7-day code via the mint script ------------------------------------------

async def test_mint_code_script_seven_day_pro(plan_env, monkeypatch, capsys):
    import mint_code

    monkeypatch.setattr(mint_code, "_store", lambda project: plan_env)
    await mint_code.cmd_mint(argparse.Namespace(
        project=None, plan="pro", days=7, uses=5, valid_days=0, count=1, note="sales friend"))
    shown = capsys.readouterr().out.split()[0]
    assert plans.normalize_code(shown) in plan_env._codes
    r = await plan_env.redeem("u-seven", plans.normalize_code(shown), NOW)
    assert r.kind == "granted" and r.expires_at == NOW + timedelta(days=7)
    monkeypatch.setattr("sys.argv", ["mint_code", "mint", "--plan", "pro", "--days", "0"])
    with pytest.raises(SystemExit):
        mint_code.main()


# -- account deletion ---------------------------------------------------------

async def test_delete_account_removes_plan_records(plan_env):
    import account_deletion

    await plan_env.create_code(_code())
    await plan_env.redeem("u-del", "ABCD2345EFGH", NOW)
    code = await plan_env.ensure_referral_code("u-del")
    summary = await account_deletion.delete_account_data("u-del")
    assert summary.counts["plan_records"] >= 3
    assert await plan_env.get("u-del") is None
    assert code not in plan_env._ref_codes
    again = await account_deletion.delete_account_data("u-del")
    assert again.counts["plan_records"] == 0


# -- referrals: +1 free listening hour each, never Pro days ------------------

HOUR = 3600


async def _live(uid, minutes):
    usage_meter.meter().add(uid, {usage_meter.KEY_LIVE_MINUTES: minutes})
    await plans.flush_usage()


async def test_referral_bonus_both_sides_after_real_use(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert await plan_env.ensure_referral_code("ref-a") == code
    r = await plan_env.redeem_referral("new-b", code, NOW)
    assert r.kind == "granted" and r.plan == "free"  # no paid days
    assert (await plan_env.get("new-b"))["stt_bonus_seconds"] == HOUR
    assert "stt_bonus_seconds" not in (await plan_env.get("ref-a"))  # not yet
    await _live("new-b", 4)
    assert "stt_bonus_seconds" not in (await plan_env.get("ref-a"))
    await _live("new-b", 6)  # 10 live minutes
    a = await plan_env.get("ref-a")
    assert a["stt_bonus_seconds"] == HOUR and a["referral_rewards"] == 1
    assert plans.effective(a).level == "free"
    await _live("new-b", 30)  # exactly one reward per new account
    assert (await plan_env.get("ref-a"))["stt_bonus_seconds"] == HOUR
    assert plan_env.referral_doc("new-b")["status"] == "rewarded"
    assert "email" not in str(plan_env.referral_doc("new-b"))


async def test_bonus_doubles_free_hours_for_stt_and_ai(plan_env):
    await plan_env.set_fields("u-bonus", {"stt_bonus_seconds": HOUR})
    await plans.prime("u-bonus", force=True)
    m = usage_meter.meter()
    m.add("u-bonus", {usage_meter.KEY_STT_SECONDS: 1.5 * HOUR})
    assert m.check("u-bonus", "cloud_transcription") is None
    m.add("u-bonus", {usage_meter.KEY_STT_SECONDS: 0.6 * HOUR})
    ex = m.check("u-bonus", "cloud_transcription")
    assert ex.limit == "stt_lifetime" and ex.cap == 2 * HOUR
    assert plans.window_cap("u-bonus", "free", "llm_lifetime") == 2 * plans.TOKENS_PER_HOUR


async def test_referral_rules(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert (await plan_env.redeem_referral("ref-a", code, NOW)).kind == "own_code"
    assert (await plan_env.redeem_referral("x", "ZZZZ2345", NOW)).kind == "invalid"
    assert (await plan_env.redeem_referral("b", code, NOW)).kind == "granted"
    other = await plan_env.ensure_referral_code("ref-c")
    assert (await plan_env.redeem_referral("b", other, NOW)).kind == "already_referred"


async def test_referral_bonus_capped_at_1_hour_per_account(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    for i in range(3):
        await plan_env.redeem_referral(f"n{i}", code, NOW)
        await _live(f"n{i}", 11)
    assert (await plan_env.get("ref-a"))["stt_bonus_seconds"] == HOUR  # max 2 h total
    assert plan_env.referral_doc("n2")["status"] == "capped"


def test_store_subscriber_is_covered_by_a_code_and_not_overwritten():
    user = {"plan": "pro", "plan_source": "revenuecat",
            "plan_expires_at": NOW + timedelta(days=20)}
    assert plans.grant_days(user, "pro", 7, NOW) == ("covered", {})


async def test_referral_over_http(client, plan_env):  # noqa: F811
    mine = (await client.get("/plan", headers=A)).json()["referral"]
    assert len(mine["code"]) == 9 and mine["max_bonus_hours"] == 1
    r = await client.post("/plan/redeem", json={"code": mine["code"].lower()}, headers=B)
    assert r.status_code == 200 and r.json()["plan"]["plan"] == "free"
    assert r.json()["plan"]["windows"]["stt_lifetime"]["cap"] == 2 * HOUR
    again = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=B)
    assert again.status_code == 400 and again.json()["detail"]["reason"] == "already_referred"
    own = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=A)
    assert own.json()["detail"]["reason"] == "own_code"


# -- tiers: free / basic / pro / max ---------------------------------------------

def _tok(n):
    return {usage_meter.llm_key("respond", "input_tokens"): n}


def test_levels_and_listening_hour_numbers():
    assert plans.LEVELS == ("free", "basic", "plus", "pro", "max")
    for lv, hours in (("free", 1), ("basic", 4), ("plus", 8), ("pro", 20), ("max", 80)):
        secs = plans.limit_for(lv, "stt_lifetime_seconds" if lv == "free" else "stt_monthly_seconds")
        toks = plans.limit_for(lv, "llm_lifetime_tokens" if lv == "free" else "llm_monthly_tokens")
        assert secs == hours * 3600 and toks == hours * plans.TOKENS_PER_HOUR
    assert [plans.limit_for(lv, "library_items") for lv in plans.LEVELS] == [5, 10, 25, 200, 1000]
    # daily backstop = 25% of the monthly allowance
    assert plans.limit_for("pro", "stt_seconds") == 20 * 3600 // 4
    assert plans.limit_for("pro", "llm_tokens") == 20 * plans.TOKENS_PER_HOUR // 4
    assert plans.upgrade_level("max") is None and plans.upgrade_level("basic") == "plus"
    assert plans.limit_for("free", "live_lifetime_minutes") == 0  # wall-clock not metered


async def test_free_llm_lifetime_is_40k_tokens_with_friendly_message(plan_env):
    await plan_env.set_fields("u-llm", {"lifetime": {"llm_tokens": 40_000.0}})
    await plans.prime("u-llm", force=True)
    ex = usage_meter.meter().check("u-llm", "batch_analysis")
    assert ex.limit == "llm_lifetime" and "out of free time" in ex.notice()["message"]


def test_free_daily_llm_default_is_150k():
    assert usage_meter.DAILY_LLM_TOKEN_CAP == 150_000


async def test_on_device_stt_session_burns_ai_but_not_stt(plan_env):
    """A phone-only (local-first) session sends no audio => no stt.seconds; it
    still spends AI tokens, and live minutes are not part of the allowance."""
    m = usage_meter.meter()
    m.add("u-dev", {usage_meter.KEY_LIVE_MINUTES: 300, **_tok(10_000)})
    assert m.check("u-dev", "cloud_transcription") is None  # live minutes unmetered
    left = plans.listening_left("u-dev", "free")
    assert left["limit"] == "llm_lifetime"  # AI is what binds
    assert left["remaining_seconds"] == int(30_000 / plans.TOKENS_PER_HOUR * 3600)


async def test_listening_left_picks_the_binding_limit(plan_env):
    m = usage_meter.meter()
    m.add("u-bind", {usage_meter.KEY_STT_SECONDS: 3000})
    plans_left = plans.listening_left("u-bind", "free")
    assert plans_left["limit"] == "stt_lifetime"
    assert plans_left["remaining_seconds"] == 600 and plans_left["period"] == "lifetime"


async def test_listening_endpoint_for_paid_is_monthly(client, plan_env):
    await plan_env.set_fields("user-a", {"plan": "pro", "plan_source": "admin", "plan_expires_at": None})
    body = (await client.get("/plan", headers=A)).json()
    assert body["plan"] == "pro"
    lis = body["listening"]
    assert lis["period"] in ("month", "day") and lis["allowance_seconds"] > 0
    assert body["windows"]["stt_monthly"]["cap"] == 20 * 3600


async def test_pro_monthly_llm_cap_message(plan_env):
    await plan_env.set_fields("u-pro2", {"plan": "pro", "plan_source": "admin", "plan_expires_at": None})
    mon = plans.month_key(datetime.now(timezone.utc))
    await plan_env.set_fields("u-pro2", {"monthly": {"period": mon, "llm_tokens": 800_000.0}})
    await plans.prime("u-pro2", force=True)
    ex = usage_meter.meter().check("u-pro2", "batch_analysis")
    assert ex.limit == "llm_monthly" and "refreshes at the start of next month" in ex.notice()["message"]
    assert ex.upgrade_plan == "max"


async def test_mint_codes_can_grant_any_level(plan_env):
    for lv in ("basic", "plus", "pro", "max"):
        await plan_env.create_code(_code(code=f"CODE{lv.upper()}XX", plan=lv))
        r = await plan_env.redeem(f"u-{lv}", f"CODE{lv.upper()}XX", NOW)
        assert r.kind == "granted" and r.plan == lv


def test_tier_catalog_for_the_plan_cards():
    cat = plans.tier_catalog()
    assert [t["level"] for t in cat] == ["basic", "plus", "pro", "max"]
    assert [t["price_usd"] for t in cat] == [5, 10, 25, 100]
    assert [t["hours_per_month"] for t in cat] == [4, 8, 20, 80]
    assert [t["popular"] for t in cat] == [False, True, False, False]
    assert [t["business"] for t in cat] == [False, False, False, True]
