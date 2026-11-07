"""scripts/recreplay/nudge_quality.py: the U_2/h yardstick (pure functions)."""

from __future__ import annotations

import pytest

from recreplay import nudge_quality as nq


def L(item, helpful):
    return {"item": item, "helpful": helpful}


def test_silence_is_zero_and_break_even_is_two_thirds():
    assert nq.u_per_h([], 2.0) == 0.0
    # 2 helpful : 1 unhelpful at lambda 2 -> exactly break-even
    assert nq.u_per_h([True, True, False], 1.0) == 0.0
    assert nq.u_per_h([True, True, True, False], 1.0) == pytest.approx(1.0)
    assert nq.u_per_h([True, False], 0.5) == pytest.approx(-2.0)


def test_summarize_counts_and_rates():
    s = nq.summarize([L("a", True), L("a", False), L("b", True)], hours=0.5)
    assert (s["n_spoken"], s["helpful"], s["unhelpful"]) == (3, 2, 1)
    assert s["helpful_share"] == pytest.approx(2 / 3)
    assert s["spoken_per_h"] == pytest.approx(6.0)
    assert s["U_per_h"] == pytest.approx(0.0)
    assert nq.summarize([], 1.0)["helpful_share"] is None


def test_bootstrap_paired_delta_is_positive_when_one_arm_is_clearly_better():
    hours = {f"s{i}": 0.1 for i in range(10)}
    good = [L(f"s{i}", True) for i in range(10)]
    bad = [L(f"s{i}", False) for i in range(10)]
    r = nq.bootstrap(good, bad, hours, B=500)
    assert r["delta_ci95"][0] > 0
    assert r["p_delta_le_0"] == 0.0
    same = nq.bootstrap(good, good, hours, B=500)
    assert same["delta_ci95"] == (0.0, 0.0)


def test_bootstrap_includes_sessions_where_an_arm_was_silent():
    hours = {"quiet": 1.0, "talk": 1.0}
    r = nq.bootstrap([L("talk", True)], None, hours, B=400)
    lo, hi = r["a_ci95"]
    assert lo == 0.0 and hi == pytest.approx(1.0)      # resampling can draw "quiet" twice


def test_combine_judges_and_kappa():
    g = {"A": {"1": True, "2": True, "3": False}, "B": {"1": True, "2": False, "3": False}}
    assert nq.combine_judges(g, "both") == {"1": True, "2": False, "3": False}
    assert nq.combine_judges(g, "any") == {"1": True, "2": True, "3": False}
    assert nq.combine_judges(g, "B")["2"] is False
    assert nq.cohen_kappa([True, False, True, False], [True, False, True, False]) == pytest.approx(1.0)
    assert nq.cohen_kappa([True, True], [True, True]) is None


def test_wilson_interval_brackets_the_share():
    lo, hi = nq.wilson(8, 12)
    assert lo < 8 / 12 < hi
    assert nq.wilson(0, 0) is None
