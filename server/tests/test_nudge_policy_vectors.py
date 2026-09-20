"""Golden-vector contract test for server/nudge_policy.py.

The cases live in server/tests/fixtures/policy_vectors/nudge_policy.json (see
the README next to it) and are deliberately language-neutral: the same file
is meant to drive the watch's Kotlin NudgeStateMachine and the phone's future
TypeScript port, so this test is the reference consumer — if the JSON and
this driver disagree about a field's meaning, the JSON's "_schema" wins and
this file is what gets fixed.

Beyond replaying each case, this test also checks the FIXTURE's own internal
consistency for the watch-applicable cases (that `db_over_baseline` maps to
the stated `level` under the watch's +6/+10/+14 thresholds), so a case can't
silently mean two different things to the two runtimes.
"""

import json
from pathlib import Path

import pytest

import nudge_policy
from nudge_policy import DEFAULT_CHANNELS, NudgePolicy
from watch.models import VectorEvent, VectorSubscription
from watch.vectors import YELLING_LEVELS

VECTORS_PATH = Path(__file__).parent / "fixtures" / "policy_vectors" / "nudge_policy.json"


def _load_cases() -> list[dict]:
    with VECTORS_PATH.open() as f:
        doc = json.load(f)
    assert doc["_schema"]["version"] == 2
    return doc["cases"]


CASES = _load_cases()


def _build_policy(config: dict) -> NudgePolicy:
    subs = [VectorSubscription(**s) for s in config["subscriptions"]]
    # `hold_s` is REQUIRED by schema v2 (the loudness ladder's hold-N
    # hysteresis) — never defaulted here, so a case can't silently be replayed
    # under a different gate than the one it was written for.
    return NudgePolicy(
        subs,
        cooldown_s=config["cooldown_s"],
        channels=config["channels"],
        hold_s=config["hold_s"],
    )


def _watch_level_for(db_over_baseline: float) -> int:
    """The Kotlin NudgeStateMachine.levelFor thresholds, taken from the SAME
    constants the streaming VectorEngine uses so this test can't drift from
    either mirror."""
    for threshold, level in YELLING_LEVELS:
        if db_over_baseline >= threshold:
            return level
    return 0


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_replays_identically(case):
    policy = _build_policy(case["config"])
    assert len(case["inputs"]) == len(case["expected"]), case["name"]

    for step_idx, (step, want) in enumerate(zip(case["inputs"], case["expected"])):
        t = step["t"]
        events = [
            VectorEvent(vector=e["vector"], level=e["level"], t=t, value=e.get("db_over_baseline", 0.0))
            for e in step["events"]
        ]
        got = policy.on_events(events, t=t)

        got_nudges = [{"channel": n.channel, "level": n.level, "vectors": list(n.vectors)} for n in got]
        assert got_nudges == want["nudges"], f"{case['name']} step {step_idx} (t={t}) nudges"
        # Every emitted nudge is stamped with the policy clock of the call that
        # produced it — the wire consumer relies on that for ordering.
        assert all(n.t == t for n in got), f"{case['name']} step {step_idx} nudge timestamps"
        assert policy.current() == want["levels"], f"{case['name']} step {step_idx} (t={t}) levels"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_is_well_formed(case):
    """Structural guarantees other-language consumers will lean on."""
    config = case["config"]
    assert set(case["applies_to"]) <= {"server", "watch", "phone"}
    assert "server" in case["applies_to"], "every case must run on the reference implementation"
    assert config["channels"], "at least one channel"
    assert len(set(config["channels"])) == len(config["channels"]), "channels are unique"
    assert isinstance(config["hold_s"], float), "schema v2: every case declares hold_s explicitly"
    assert config["hold_s"] >= 0.0

    last_t = float("-inf")
    for step in case["inputs"]:
        assert isinstance(step["t"], float), "timestamps are seconds as floats"
        assert step["t"] > last_t, "policy clock is monotonic within a case"
        last_t = step["t"]
        for e in step["events"]:
            assert 0 <= e["level"] <= 3

    for want in case["expected"]:
        assert set(want["levels"]) == set(config["channels"]), "levels cover exactly the configured channels"
        for n in want["nudges"]:
            assert n["channel"] in config["channels"]
            assert n["vectors"] == sorted(set(n["vectors"])), "vectors are sorted + de-duplicated"


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if "watch" in c["applies_to"]],
    ids=[c["name"] for c in CASES if "watch" in c["applies_to"]],
)
def test_watch_applicable_cases_match_kotlin_machine_shape(case):
    """A case tagged for the watch must be runnable by the single-channel
    Kotlin NudgeStateMachine: one lane, no sensitivity scaling, at most one
    yelling observation per step, and a `db_over_baseline` whose threshold
    level agrees with the `level` the server side is fed."""
    config = case["config"]
    assert config["channels"] == ["A"]
    for sub in config["subscriptions"]:
        assert sub["sensitivity"] == 1.0 and sub["haptics"] is True and sub["channel"] == "A"

    for step in case["inputs"]:
        assert len(step["events"]) <= 1
        for e in step["events"]:
            assert e["vector"] == "yelling"
            assert "db_over_baseline" in e, "watch-applicable events carry the raw loudness"
            assert _watch_level_for(e["db_over_baseline"]) == e["level"]

    # And at least one Kotlin-shaped case must exercise a de-escalation, or
    # the watch would never see its cooldown rule under test from this file.
    assert any("watch" in c["applies_to"] and any(n["vectors"] == [] for w in c["expected"] for n in w["nudges"]) for c in CASES)


def test_coverage_of_required_scenarios():
    """The plan asked for specific scenarios; name-check that they exist so a
    future fixture edit can't quietly drop one."""
    names = {c["name"] for c in CASES}
    assert len(names) == len(CASES), "case names are unique"
    assert len(CASES) >= 8
    required = {
        "below_threshold_no_nudge",
        "single_nudge_then_sustain_is_silent",
        "cooldown_is_strictly_greater_than",
        "sustained_observation_refreshes_clock",
        "stepwise_deescalation_3_to_0",
        "full_decay_then_fresh_escalation",
        # hold-3s (2026-09-20): the three cases the ladder's hysteresis is
        # pinned by on all three runtimes.
        "hold_two_loud_windows_then_quiet_never_buzzes",
        "hold_three_loud_windows_escalates_on_the_third",
        "hold_of_one_is_the_pre_hysteresis_ladder",
    }
    assert required <= names


def test_the_shipped_hold_is_actually_exercised():
    """A fixture where every case declared the legacy `hold_s: 1.0` would be
    green and prove nothing about what ships. At least one case must run at
    the runtime default, and at least one at 0.0 — the two ends of the knob."""
    holds = {c["config"]["hold_s"] for c in CASES}
    assert nudge_policy.HEAT_HOLD_S_DEFAULT in holds, (
        f"no case runs at the shipped hold of {nudge_policy.HEAT_HOLD_S_DEFAULT}s"
    )
    assert 0.0 in holds and 1.0 in holds, "both no-gate values must stay pinned"


def test_hold_env_override(monkeypatch):
    """MINDSHIFT_HEAT_HOLD_S is the server's knob, read once per policy."""
    monkeypatch.setenv(nudge_policy.HEAT_HOLD_ENV, "5")
    assert nudge_policy.heat_hold_s() == 5.0
    assert NudgePolicy([VectorSubscription(vector="yelling")]).hold.hold_s == 5.0
    # Garbage and negatives fall back to the default rather than disabling the
    # gate silently — a typo must never quietly restore the 97/hour ladder.
    for bad in ("", "  ", "three", "-1"):
        monkeypatch.setenv(nudge_policy.HEAT_HOLD_ENV, bad)
        assert nudge_policy.heat_hold_s() == nudge_policy.HEAT_HOLD_S_DEFAULT
    monkeypatch.delenv(nudge_policy.HEAT_HOLD_ENV)
    assert nudge_policy.heat_hold_s() == nudge_policy.HEAT_HOLD_S_DEFAULT


def test_hold_counts_seconds_not_calls():
    """A turn-driven caller (watch/relay.py) observes once per TURN, so an
    observation carries how much audio it covers. One 4 s loud turn is four
    windows of hold; a sub-second one is still worth a whole window, which is
    what keeps hold_s <= 1 byte-identical to the pre-hold ladder everywhere."""
    hold = nudge_policy.LoudnessHold(3.0)
    assert hold.observe(True, 4.0) is True
    hold = nudge_policy.LoudnessHold(3.0)
    assert hold.observe(True, 0.2) is False   # worth one window, not 0.2
    assert hold.observe(True, 0.2) is False
    assert hold.observe(True, 0.2) is True
    # peek is pure: asking twice can't advance the run (the phone's instant
    # haptic tier asks before the same turn's policy tick).
    hold = nudge_policy.LoudnessHold(3.0)
    assert hold.peek(True, 5.0) is True
    assert hold.peek(True, 5.0) is True
    assert hold.run_s == 0.0


def test_watch_module_reexports_canonical_policy():
    """server/watch/nudge_policy.py is a thin alias, not a fork — the watch WS
    ingest and the phone path must be running the same class object."""
    import watch.nudge_policy as legacy

    assert legacy.NudgePolicy is nudge_policy.NudgePolicy
    assert legacy.DEFAULT_CHANNELS == DEFAULT_CHANNELS == ("A", "B")


def test_default_channels_unchanged_for_watch_callers():
    """The shipped watch constructs NudgePolicy(subs) with no channel arg and
    reads both lanes from current() — that surface must not move."""
    p = NudgePolicy([VectorSubscription(vector="yelling"), VectorSubscription(vector="hr_spike")])
    assert p.current() == {"A": 0, "B": 0}
    assert p.channels == ("A", "B")


def test_empty_channels_rejected():
    with pytest.raises(ValueError):
        NudgePolicy([VectorSubscription(vector="yelling")], channels=())
