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
    assert plans.limit_for("free", "stt_lifetime_seconds") == plans.FREE_STT_LIFETIME_SECONDS == 7200
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 7199})
    assert m.check("u-life", "cloud_transcription") is None
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 5})
    ex = m.check("u-life", "cloud_transcription")
    assert ex is not None and ex.limit == "stt_lifetime"
    assert "used your 2 free hours" in ex.notice()["message"]
    assert m.check("u-life", "batch_analysis") is None  # other metrics untouched


async def test_lifetime_survives_flush_restart_and_redeem(plan_env, monkeypatch):
    m = usage_meter.meter()
    m.add("u-life", {usage_meter.KEY_STT_SECONDS: 7300})
    await m.flush()
    assert plan_env._users["u-life"]["lifetime"]["stt_seconds"] == 7300
    m.reset()
    plans.clear_cache()
    await plans.prime("u-life", force=True)  # "restart": only the store remembers
    assert m.check("u-life", "cloud_transcription").limit == "stt_lifetime"
    monkeypatch.setattr(usage_meter, "day_key", lambda when=None: "2099-01-01")
    assert m.check("u-life", "cloud_transcription").limit == "stt_lifetime"
    await plan_env.create_code(_code())
    await plan_env.redeem("u-life", "ABCD2345EFGH", NOW)
    assert plan_env._users["u-life"]["lifetime"]["stt_seconds"] == 7300
    await plans.prime("u-life", force=True)
    assert m.check("u-life", "cloud_transcription") is None  # pro: monthly 60 h


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


# -- referrals: bonus free STT hours, never Pro days --------------------------

HOUR = 3600


async def _live(uid, minutes):
    usage_meter.meter().add(uid, {usage_meter.KEY_LIVE_MINUTES: minutes})
    await plans.flush_usage()


async def test_referral_bonus_both_sides_after_real_use(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert await plan_env.ensure_referral_code("ref-a") == code
    r = await plan_env.redeem_referral("new-b", code, NOW)
    assert r.kind == "granted" and r.plan == "free"  # no Pro days
    assert (await plan_env.get("new-b"))["stt_bonus_seconds"] == 2 * HOUR
    assert "stt_bonus_seconds" not in (await plan_env.get("ref-a"))  # not yet
    await _live("new-b", 4)
    assert "stt_bonus_seconds" not in (await plan_env.get("ref-a"))
    await _live("new-b", 6)  # 10 live minutes
    a = await plan_env.get("ref-a")
    assert a["stt_bonus_seconds"] == 2 * HOUR and a["referral_rewards"] == 1
    assert plans.effective(a).level == "free"
    await _live("new-b", 30)  # exactly one reward per new account
    assert (await plan_env.get("ref-a"))["stt_bonus_seconds"] == 2 * HOUR
    assert plan_env.referral_doc("new-b")["status"] == "rewarded"
    assert "email" not in str(plan_env.referral_doc("new-b"))


async def test_bonus_raises_the_free_stt_cap_from_2_to_4_hours(plan_env):
    await plan_env.set_fields("u-bonus", {"stt_bonus_seconds": 2 * HOUR})
    await plans.prime("u-bonus", force=True)
    m = usage_meter.meter()
    m.add("u-bonus", {usage_meter.KEY_STT_SECONDS: 3 * HOUR})
    assert m.check("u-bonus", "cloud_transcription") is None
    m.add("u-bonus", {usage_meter.KEY_STT_SECONDS: 1 * HOUR})
    ex = m.check("u-bonus", "cloud_transcription")
    assert ex.limit == "stt_lifetime" and "4 free hours" in ex.notice()["message"]


async def test_referral_rules(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert (await plan_env.redeem_referral("ref-a", code, NOW)).kind == "own_code"
    assert (await plan_env.redeem_referral("x", "ZZZZ2345", NOW)).kind == "invalid"
    assert (await plan_env.redeem_referral("b", code, NOW)).kind == "granted"
    other = await plan_env.ensure_referral_code("ref-c")
    assert (await plan_env.redeem_referral("b", other, NOW)).kind == "already_referred"


async def test_referral_bonus_capped_at_10_hours_per_account(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    n = plans.REFERRAL_MAX_BONUS_SECONDS // plans.REFERRAL_BONUS_SECONDS + 2
    for i in range(n):
        await plan_env.redeem_referral(f"n{i}", code, NOW)
        await _live(f"n{i}", 11)
    assert (await plan_env.get("ref-a"))["stt_bonus_seconds"] == 10 * HOUR
    assert plan_env.referral_doc(f"n{n - 1}")["status"] == "capped"


def test_store_subscriber_is_covered_by_a_code_and_not_overwritten():
    user = {"plan": "pro", "plan_source": "revenuecat",
            "plan_expires_at": NOW + timedelta(days=20)}
    assert plans.grant_days(user, "pro", 7, NOW) == ("covered", {})


async def test_referral_over_http(client, plan_env):  # noqa: F811
    mine = (await client.get("/plan", headers=A)).json()["referral"]
    assert len(mine["code"]) == 9 and mine["max_bonus_hours"] == 10
    r = await client.post("/plan/redeem", json={"code": mine["code"].lower()}, headers=B)
    assert r.status_code == 200 and r.json()["plan"]["plan"] == "free"
    assert r.json()["plan"]["windows"]["stt_lifetime"]["cap"] == 4 * HOUR
    again = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=B)
    assert again.status_code == 400 and again.json()["detail"]["reason"] == "already_referred"
    own = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=A)
    assert own.json()["detail"]["reason"] == "own_code"


# -- free / pro cost caps -------------------------------------------------------

def _tok(n):
    return {usage_meter.llm_key("respond", "input_tokens"): n}


async def test_free_llm_lifetime_cap_with_friendly_message(plan_env):
    m = usage_meter.meter()
    assert plans.limit_for("free", "llm_lifetime_tokens") == plans.FREE_LLM_LIFETIME_TOKENS == 1_000_000
    m.add("u-llm", _tok(plans.FREE_LLM_LIFETIME_TOKENS - 1))
    # the daily cap (150k) bites first for a single-day burst
    assert m.check("u-llm", "batch_analysis").limit == "llm_tokens"
    # spread across days: only the lifetime window remains
    m.reset()
    plans.clear_cache()
    await plan_env.set_fields("u-llm", {"lifetime": {"llm_tokens": 1_000_000.0}})
    await plans.prime("u-llm", force=True)
    ex = m.check("u-llm", "batch_analysis")
    assert ex.limit == "llm_lifetime" and "free allowance" in ex.notice()["message"]


def test_free_daily_llm_default_is_150k():
    assert usage_meter.DAILY_LLM_TOKEN_CAP == 150_000
    assert plans.limit_for("free", "llm_tokens") == 150_000


async def test_free_live_minutes_lifetime_cap(plan_env):
    await plan_env.set_fields("u-live", {"lifetime": {"live_minutes": 240.0}})
    await plans.prime("u-live", force=True)
    ex = usage_meter.meter().check("u-live", "cloud_transcription")
    assert ex.limit == "live_lifetime" and "4 hours of live listening" in ex.notice()["message"]


def test_pro_numbers():
    assert plans.limit_for("pro", "stt_monthly_seconds") == 40 * HOUR
    assert plans.limit_for("pro", "llm_monthly_tokens") == 2_000_000
    assert plans.limit_for("pro", "llm_tokens") == 300_000
    assert plans.limit_for("pro", "live_minutes") == 720
    assert plans.limit_for("pro", "library_items") == 200


async def test_pro_monthly_llm_cap_message(plan_env):
    await plan_env.set_fields("u-pro2", {"plan": "pro", "plan_source": "admin", "plan_expires_at": None})
    mon = plans.month_key(datetime.now(timezone.utc))
    await plan_env.set_fields("u-pro2", {"monthly": {"period": mon, "llm_tokens": 2_000_000.0}})
    await plans.prime("u-pro2", force=True)
    ex = usage_meter.meter().check("u-pro2", "batch_analysis")
    assert ex.limit == "llm_monthly" and "refreshes at the start of next month" in ex.notice()["message"]
