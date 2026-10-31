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


# -- referrals ------------------------------------------------------------------

async def _live(uid, minutes):
    usage_meter.meter().add(uid, {usage_meter.KEY_LIVE_MINUTES: minutes})
    await plans.flush_usage()


async def test_referral_double_sided_after_real_use(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert await plan_env.ensure_referral_code("ref-a") == code
    r = await plan_env.redeem_referral("new-b", code, NOW)
    assert r.kind == "granted" and r.expires_at == NOW + timedelta(days=7)
    assert plans.effective(await plan_env.get("ref-a")).level == "free"
    await _live("new-b", 4)
    assert plans.effective(await plan_env.get("ref-a")).level == "free"
    await _live("new-b", 6)
    a = await plan_env.get("ref-a")
    assert plans.effective(a).level == "pro" and a["referral_rewards"] == 1
    await _live("new-b", 30)  # exactly one reward per new account
    assert (await plan_env.get("ref-a"))["referral_rewards"] == 1
    assert plan_env.referral_doc("new-b")["status"] == "rewarded"
    assert "email" not in str(plan_env.referral_doc("new-b"))


async def test_referral_rules(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    assert (await plan_env.redeem_referral("ref-a", code, NOW)).kind == "own_code"
    assert (await plan_env.redeem_referral("x", "ZZZZ2345", NOW)).kind == "invalid"
    assert (await plan_env.redeem_referral("b", code, NOW)).kind == "granted"
    other = await plan_env.ensure_referral_code("ref-c")
    assert (await plan_env.redeem_referral("b", other, NOW)).kind == "already_referred"


async def test_referral_reward_cap_per_referrer(plan_env):
    code = await plan_env.ensure_referral_code("ref-a")
    n = plans.REFERRAL_MAX_REWARDS + 2
    for i in range(n):
        await plan_env.redeem_referral(f"n{i}", code, NOW)
        await _live(f"n{i}", 11)
    a = await plan_env.get("ref-a")
    assert a["referral_rewards"] == plans.REFERRAL_MAX_REWARDS
    assert plan_env.referral_doc(f"n{n - 1}")["status"] == "capped"


def test_paid_store_subscriber_banks_days():
    user = {"plan": "pro", "plan_source": "revenuecat",
            "plan_expires_at": NOW + timedelta(days=20), "credit_days": 3}
    kind, upd = plans.grant_days(user, "pro", 7, NOW)
    assert kind == "banked" and upd == {"credit_days": 10}


async def test_referral_over_http(client, plan_env):  # noqa: F811
    mine = (await client.get("/plan", headers=A)).json()["referral"]
    assert len(mine["code"]) == 9 and mine["max_rewards"] == plans.REFERRAL_MAX_REWARDS
    r = await client.post("/plan/redeem", json={"code": mine["code"].lower()}, headers=B)
    assert r.status_code == 200 and r.json()["plan"]["plan"] == "pro"
    again = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=B)
    assert again.status_code == 400 and again.json()["detail"]["reason"] == "already_referred"
    own = await client.post("/plan/redeem", json={"code": mine["code"]}, headers=A)
    assert own.json()["detail"]["reason"] == "own_code"
