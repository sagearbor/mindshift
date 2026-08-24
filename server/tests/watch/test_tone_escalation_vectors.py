"""Golden-vector contract test for the phone->watch tone-escalation relay
(server/watch/relay.py). Cases live in
server/tests/fixtures/policy_vectors/tone_escalation.json — see the README
next to it and the JSON's own `_schema`. This driver is the reference
consumer: replays every case through the REAL `turn_local_to_vector_events`
+ `NudgePolicy`, and separately checks the fixture's internal consistency
so a future edit can't quietly change what a rung means.
"""

import json
from pathlib import Path

import pytest

from models.audio import ToneFlagEvent, TurnLocalEvent
from nudge_policy import NudgePolicy
from watch import relay
from watch.models import VectorSubscription

VECTORS_PATH = Path(__file__).parents[1] / "fixtures" / "policy_vectors" / "tone_escalation.json"


def _load_cases() -> list[dict]:
    with VECTORS_PATH.open() as f:
        doc = json.load(f)
    assert doc["_schema"]["version"] == 1
    return doc["cases"]


CASES = _load_cases()


def _turn_from(step: dict) -> tuple[TurnLocalEvent, ToneFlagEvent | None]:
    turn = step["turn"]
    prosody = None if turn.get("rms_dbfs", None) is None else {"rms_dbfs": turn["rms_dbfs"]}
    event = TurnLocalEvent(
        session_id="vec", speaker="Speaker A", is_self=turn["is_self"], text="…",
        start_time=step["t"], end_time=step["t"] + 1.0, transcript_source="on-device",
        prosody=prosody, text_tone=turn.get("text_tone"),
    )
    flag = turn.get("tone_flag")
    tone_flag = None
    if flag is not None:
        tone_flag = ToneFlagEvent(
            session_id="vec", speaker="Speaker A", start_time=step["t"], end_time=step["t"] + 1.0,
            source="audio", scores=flag["scores"], label="hostile", confidence=flag["confidence"],
        )
    return event, tone_flag


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_replays_identically(case):
    config = case["config"]
    subs = [VectorSubscription(**s) for s in config["subscriptions"]]
    policy = NudgePolicy(subs, cooldown_s=config["cooldown_s"], channels=config["channels"])
    assert len(case["inputs"]) == len(case["expected"]), case["name"]

    for idx, (step, want) in enumerate(zip(case["inputs"], case["expected"])):
        event, tone_flag = _turn_from(step)
        events = relay.turn_local_to_vector_events(
            event, t=step["t"], baseline_rms_db=config["baseline_rms_db"], tone_flag=tone_flag,
        )
        got_events = [{"vector": e.vector, "level": e.level} for e in events]
        assert got_events == want["events"], f"{case['name']} step {idx} events"
        assert all(e.t == step["t"] for e in events), f"{case['name']} step {idx} events run on the caller's clock"

        # The relay's rule: an empty conversion does NOT tick the policy.
        nudges = policy.on_events(events, t=step["t"]) if events else []
        got_nudges = [{"channel": n.channel, "level": n.level, "vectors": list(n.vectors)} for n in nudges]
        assert got_nudges == want["nudges"], f"{case['name']} step {idx} nudges"
        assert policy.current() == want["levels"], f"{case['name']} step {idx} levels"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_is_well_formed(case):
    config = case["config"]
    assert case["applies_to"] == ["server"], "no other runtime consumes this file yet"
    assert config["channels"], "at least one channel"
    last_t = float("-inf")
    for step in case["inputs"]:
        assert isinstance(step["t"], float), "timestamps are seconds as floats"
        assert step["t"] > last_t, "clock is monotonic within a case"
        last_t = step["t"]
        turn = step["turn"]
        assert turn["is_self"] in (True, False, None)
        tone = turn.get("text_tone")
        if tone is not None:
            for key in ("frustration", "defensiveness"):
                v = tone.get(key)
                assert v is None or 0 <= v <= 100
        flag = turn.get("tone_flag")
        if flag is not None:
            assert 0.0 <= flag["confidence"] <= 1.0
    for want in case["expected"]:
        assert set(want["levels"]) == set(config["channels"])
        seen = [e["vector"] for e in want["events"]]
        assert seen == sorted(seen, key=["yelling", "aggressive_tone"].index), "yelling before aggressive_tone"
        for n in want["nudges"]:
            assert n["vectors"] == sorted(set(n["vectors"]))


def test_schema_tone_levels_match_the_constants():
    """The JSON's prose description of TONE_LEVELS is what a future port
    reads — pin it to the actual constants so they can't drift."""
    with VECTORS_PATH.open() as f:
        schema = json.load(f)["_schema"]
    text = schema["tone_levels"]
    for threshold, level in relay.TONE_LEVELS:
        assert f">={threshold:.0f} -> {level}" in text, (threshold, level, text)
    assert f"({relay.TONE_FLAG_MIN_CONFIDENCE})" in schema["inputs"]["turn"]["tone_flag"]


def test_coverage_of_required_scenarios():
    names = {c["name"] for c in CASES}
    assert len(names) == len(CASES), "case names are unique"
    required = {
        "calm_volume_hostile_tone_escalates",
        "other_speaker_hostile_turn_is_ignored",
        "unknown_speaker_is_ignored",
        "tone_threshold_rungs",
        "db_and_tone_combine_as_max",
        "tone_flag_counts_only_when_confident",
        "missing_measurements_are_not_zero",
    }
    assert required <= names
