"""Golden-vector contract test for server/pleasantness.py.

The cases live in server/tests/fixtures/policy_vectors/pleasantness.json
(see the README next to it) and are replayed identically by the phone's
apps/mobile/src/live/pleasantness.ts (__tests__/livePleasantness.test.ts),
so the scoreboard a couple watched live and the one the therapist sees
post-session are the same numbers. If the JSON and this driver disagree
about a field's meaning, the JSON's ``_schema`` wins.
"""

import json
from pathlib import Path

import pytest

import pleasantness

VECTORS_PATH = Path(__file__).parent / "fixtures" / "policy_vectors" / "pleasantness.json"


def _load() -> dict:
    with VECTORS_PATH.open() as f:
        return json.load(f)


DOC = _load()
CASES = DOC["cases"]


def test_fixture_matches_module_constants():
    assert DOC["_schema"]["version"] == 1
    c = DOC["constants"]
    assert c["weights"] == pleasantness.WEIGHTS
    assert c["neutral_calm_prior"] == pleasantness.NEUTRAL_CALM_PRIOR
    assert c["loud_db_free"] == pleasantness.LOUD_DB_FREE
    assert c["loud_penalty_per_db"] == pleasantness.LOUD_PENALTY_PER_DB
    assert c["loud_penalty_max"] == pleasantness.LOUD_PENALTY_MAX
    assert c["fast_rate_wps"] == pleasantness.FAST_RATE_WPS
    assert c["fast_penalty"] == pleasantness.FAST_PENALTY
    assert c["contempt_respect_cap"] == pleasantness.CONTEMPT_RESPECT_CAP
    assert set(c["contempt_labels"]) == set(pleasantness.CONTEMPT_LABELS)
    assert c["balance_window"] == pleasantness.BALANCE_WINDOW
    assert c["current_window"] == pleasantness.CURRENT_WINDOW
    assert c["series_length"] == pleasantness.SERIES_LENGTH
    assert c["lead_min_margin"] == pleasantness.LEAD_MIN_MARGIN
    assert abs(sum(pleasantness.WEIGHTS.values()) - 1.0) < 1e-9


def test_coverage_of_required_scenarios():
    names = [c["name"] for c in CASES]
    assert len(set(names)) == len(names)
    for required in (
        "warm_then_defensive_two_people",
        "no_tone_no_prosody_is_unscored",
        "single_speaker_engagement_unmeasured",
        "contempt_label_caps_respect",
        "even_when_margin_below_three",
        "loud_penalty_capped_and_series_window_ten",
        "fast_speech_penalty_with_neutral_prior",
        "clamps_and_missing_keys",
    ):
        assert required in names


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_replays_identically(case):
    tracker = pleasantness.PleasantnessTracker()
    assert len(case["turns"]) == len(case["expected"]["per_turn"])
    for i, (turn, want) in enumerate(zip(case["turns"], case["expected"]["per_turn"])):
        got = tracker.observe(turn["speaker"], turn["text_tone"], turn["prosody"])
        assert got["dims"] == want["dims"], f"{case['name']} turn {i} dims"
        assert got["score"] == want["score"], f"{case['name']} turn {i} score"
    board = tracker.board()
    assert [
        {"speaker": p["speaker"], "current": p["current"], "series": p["series"]}
        for p in board["people"]
    ] == case["expected"]["people"], case["name"]
    assert board["lead"] == case["expected"]["lead"], case["name"]

    # The batch scorer is the same arithmetic in one call.
    batch = pleasantness.score_session(case["turns"])
    assert [t["score"] for t in batch["per_turn"]] == [t["score"] for t in case["expected"]["per_turn"]]
    assert batch["lead"] == case["expected"]["lead"]


def test_round_half_up_is_not_bankers():
    assert pleasantness.round_half_up(0.5) == 1
    assert pleasantness.round_half_up(2.5) == 3
    assert round(2.5) == 2  # the trap this helper exists to avoid


def test_score_session_keeps_index_alignment_for_malformed_turns():
    out = pleasantness.score_session([
        {"speaker": "A", "text_tone": {"warmth": 50}},
        {"text": "no speaker"},
        {"speaker": "B", "text_tone": {"warmth": 50}},
    ])
    # B's turn: warmth 50 + a balanced window (engagement 100) → 63.
    assert [t["score"] for t in out["per_turn"]] == [50, None, 63]
    assert [p["speaker"] for p in out["people"]] == ["A", "B"]


def test_engagement_needs_two_voices():
    assert pleasantness.engagement_from_window(["A"], "A") is None
    assert pleasantness.engagement_from_window(["A", "A"], "A") is None
    assert pleasantness.engagement_from_window(["A", "B"], "B") == 100
