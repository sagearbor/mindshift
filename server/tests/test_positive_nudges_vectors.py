"""Golden-vector driver for the positive nudge detectors D / E / R / K.

Replays ``server/tests/fixtures/policy_vectors/positive_nudges.json`` against
:mod:`server.positive_nudges`; ``apps/mobile/__tests__/positiveNudges.test.ts``
replays the same file against the TypeScript mirror. The cases that say a code
must NOT fire matter as much as the ones that say it must.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nudge_vocabulary import POSITIVE_CAP_S
from positive_nudges import (
    CALM_STREAK_MIN_S,
    DEESCALATION_WINDOW_TURNS,
    LISTEN_MIN_TURN_S,
    REPAIR_PHRASES,
    REPAIR_SOFTEN_DROP,
    PositiveTurn,
    coalesce_turns,
    detect_positive_nudges,
    has_repair_language,
    normalize_for_repair,
    self_heat_level,
)

FIXTURE = Path(__file__).parent / "fixtures" / "policy_vectors" / "positive_nudges.json"
DOC = json.loads(FIXTURE.read_text())
CASES = DOC["cases"]


def _turn(spec: dict) -> PositiveTurn:
    return PositiveTurn(
        index=spec["index"],
        start=spec["start"],
        end=spec["end"],
        speaker=spec["speaker"],
        is_self=spec["is_self"],
        text=spec["text"],
        loud_level=spec["loud_level"],
        tone_heat=spec["tone_heat"],
        tone_negativity=spec["tone_negativity"],
        cut_in=spec["cut_in"],
    )


def test_schema_version_constants_and_lexicon():
    assert DOC["_schema"]["version"] == 1
    c = DOC["constants"]
    assert c["listen_min_turn_s"] == LISTEN_MIN_TURN_S
    assert c["deescalation_window_turns"] == DEESCALATION_WINDOW_TURNS
    assert c["repair_soften_drop"] == REPAIR_SOFTEN_DROP
    assert c["calm_streak_min_s"] == CALM_STREAK_MIN_S
    assert c["positive_cap_s"] == POSITIVE_CAP_S
    assert tuple(DOC["repair_phrases"]) == REPAIR_PHRASES


def test_coverage_of_required_scenarios():
    names = {c["name"] for c in CASES}
    for required in (
        "de_escalation_after_a_spike",
        "de_escalation_must_land_within_two_of_your_turns",
        "listened_to_a_long_uninterrupted_turn",
        "cutting_in_forfeits_the_listened_badge",
        "repair_softens_the_other_person",
        "a_repair_lands_when_they_go_from_hurt_to_relieved",
        "aggression_falling_while_they_stay_hurt_is_not_a_repair",
        "repair_language_without_softening_is_not_repair",
        "softening_without_repair_language_is_not_repair",
        "positives_are_capped_at_one_per_two_minutes",
        "coalescing_restores_the_long_turn_the_vad_cut_up",
        "two_different_people_are_never_one_turn",
        "calm_streak_is_the_longest_quiet_run",
    ):
        assert required in names, required
    fired = {n["code"] for c in CASES for n in c["expected"]["nudges"]}
    assert fired == {"D", "E", "R"}, "K must never be a live event"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_replays_identically(case):
    rows = [_turn(t) for t in case["turns"]]
    got = detect_positive_nudges(
        coalesce_turns(rows) if case.get("coalesce") else rows,
        case["alert_nudge_times"],
        case["session_end_s"],
        **({"cap_s": case["cap_s"]} if "cap_s" in case else {}),
    )
    assert [
        {"code": n.code, "t": n.t, "turn_index": n.turn_index, "delivered": n.delivered}
        for n in got.nudges
    ] == case["expected"]["nudges"]
    assert got.calm.longest_s == pytest.approx(case["expected"]["calm"]["longest_s"])
    assert got.calm.badge is case["expected"]["calm"]["badge"]
    # Every emitted nudge carries a human line — the report prints it verbatim.
    assert all(n.detail for n in got.nudges)


def test_repair_language_survives_stt_punctuation():
    assert normalize_for_repair("I'm sorry.") == "im sorry"
    assert has_repair_language("I'm sorry.")
    assert has_repair_language("im  SORRY!!")
    assert has_repair_language("You're right — that was on me")


def test_repair_language_does_not_fire_on_its_absence():
    assert not has_repair_language("")
    # The lexicon is first-person on purpose: sarcasm and blame are not repairs.
    assert not has_repair_language("sorry not sorry")
    assert not has_repair_language("you are wrong")
    assert not has_repair_language("whatever")


def test_self_heat_level_never_guesses_zero():
    blank = PositiveTurn(0, 0.0, 1.0, "You", True, "", None, None, None, False)
    assert self_heat_level(blank) is None
    assert self_heat_level(PositiveTurn(0, 0.0, 1.0, "You", True, "", 1, 88, 88, False)) == 3
    assert self_heat_level(PositiveTurn(0, 0.0, 1.0, "You", True, "", 3, 10, 10, False)) == 3
    assert self_heat_level(PositiveTurn(0, 0.0, 1.0, "You", True, "", 0, None, None, False)) == 0
    assert self_heat_level(PositiveTurn(0, 0.0, 1.0, "You", True, "", None, 40, 40, False)) == 0
