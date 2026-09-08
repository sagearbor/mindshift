"""Golden-vector contract test for server/natural_turn.py.

The cases live in server/tests/fixtures/policy_vectors/natural_turn.json (see
its "_schema") and are deliberately language-neutral: the SAME file also
drives apps/mobile/__tests__/liveNaturalTurnVectors.test.ts against the TS
port (apps/mobile/src/live/naturalTurn.ts). If the JSON and this driver
disagree about a field's meaning, the JSON's "_schema" wins and this file is
what gets fixed — see server/natural_turn.py's module docstring for how the
two ports relate (classify_utterance/label_turns/merge_primaries are meant to
be bit-identical; sentences_from_words/collapse_short_pauses are server-only
additions with no TS counterpart).
"""

import json
from pathlib import Path

import pytest

import natural_turn
from natural_turn import (
    MAX_PAUSE_SECONDS,
    classify_utterance,
    collapse_short_pauses,
    label_turns,
    live_turn_kind,
    merge_primaries,
    natural_turns,
    sentences_from_words,
    words_of,
)

VECTORS_PATH = Path(__file__).parent / "fixtures" / "policy_vectors" / "natural_turn.json"


def _load() -> dict:
    with VECTORS_PATH.open() as f:
        doc = json.load(f)
    assert doc["_schema"]["version"] == 1
    return doc


DOC = _load()
CLASSIFICATION_CASES = DOC["classification_cases"]
CONTAINMENT_CASES = DOC["containment_cases"]


# ---------------------------------------------------------------------------
# classify_utterance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CLASSIFICATION_CASES, ids=[c["name"] for c in CLASSIFICATION_CASES])
def test_classify_utterance_replays_identically(case):
    got = classify_utterance(case["text"], case["duration_s"])
    assert got == case["expected"], case["name"]


def test_classification_fixture_covers_every_decision_branch():
    """Structural guarantee: the fixture actually exercises all four outcomes
    (a regression could silently drop a branch's only case)."""
    outcomes = {c["expected"] for c in CLASSIFICATION_CASES}
    assert outcomes == {"backchannel", "secondary", "other", None}
    assert len(CLASSIFICATION_CASES) >= 30
    names = [c["name"] for c in CLASSIFICATION_CASES]
    assert len(names) == len(set(names)), "case names are unique"


def test_live_turn_kind_matches_classification():
    for case in CLASSIFICATION_CASES:
        want = "backchannel" if case["expected"] == "backchannel" else "primary"
        assert live_turn_kind(case["text"], case["duration_s"]) == want, case["name"]


# ---------------------------------------------------------------------------
# label_turns + merge_primaries (containment/merge scenarios)
# ---------------------------------------------------------------------------


def _utterance_key(u):
    return (u["speaker"], u["start"], u["end"])


@pytest.mark.parametrize("case", CONTAINMENT_CASES, ids=[c["name"] for c in CONTAINMENT_CASES])
def test_label_turns_replays_identically(case):
    utterances = [
        {"speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"]}
        for u in case["utterances"]
    ]
    labeled = label_turns(utterances)
    assert len(labeled) == len(case["expected_labels"]), case["name"]
    for got, want in zip(labeled, case["expected_labels"]):
        assert got["speaker"] == want["speaker"], case["name"]
        assert got["start"] == want["start"], case["name"]
        assert got["end"] == want["end"], case["name"]
        assert got["kind"] == want["kind"], case["name"]
        assert got["interjects"] == want["interjects"], case["name"]


@pytest.mark.parametrize("case", CONTAINMENT_CASES, ids=[c["name"] for c in CONTAINMENT_CASES])
def test_merge_primaries_replays_identically(case):
    utterances = [
        {"speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"]}
        for u in case["utterances"]
    ]
    merged = merge_primaries(label_turns(utterances))
    assert len(merged) == len(case["expected_merged"]), case["name"]
    for got, want in zip(merged, case["expected_merged"]):
        assert got["speaker"] == want["speaker"], case["name"]
        assert got["start"] == want["start"], case["name"]
        assert got["end"] == want["end"], case["name"]
        assert got["text"] == want["text"], case["name"]
        assert got["parts"] == want["parts"], case["name"]
        assert [a["kind"] for a in got["attached"]] == want["attached_kinds"], case["name"]


def test_containment_fixture_has_at_least_three_scenarios():
    assert len(CONTAINMENT_CASES) >= 3
    names = [c["name"] for c in CONTAINMENT_CASES]
    assert len(names) == len(set(names))


def test_natural_turns_reproduces_pause_merge_scenario_without_containment():
    """naturalTurns() (the one-call convenience) on a scenario with no
    overlap must agree with calling label_turns + merge_primaries directly."""
    case = next(c for c in CONTAINMENT_CASES if c["name"] == "pause_merge_within_threshold_new_turn_after")
    utterances = [
        {"speaker": u["speaker"], "start": u["start"], "end": u["end"], "text": u["text"]}
        for u in case["utterances"]
    ]
    direct = merge_primaries(label_turns(utterances))
    via_merge = merge_primaries(label_turns(utterances), max_pause=MAX_PAUSE_SECONDS)
    assert len(direct) == len(via_merge) == len(case["expected_merged"])


# ---------------------------------------------------------------------------
# sentences_from_words / collapse_short_pauses — the server-only pre-stage
# that fixes the recall regression (no TS counterpart, no shared fixture).
# ---------------------------------------------------------------------------


def _word(word, start, end, confidence=None):
    w = {"word": word, "start": start, "end": end}
    if confidence is not None:
        w["confidence"] = confidence
    return w


def test_sentences_from_words_splits_at_terminal_punctuation():
    words = [
        _word("Hi,", 0.0, 0.3),
        _word("how", 0.3, 0.5),
        _word("are", 0.5, 0.7),
        _word("you?", 0.7, 1.0),
        _word("Good", 1.5, 1.8),
        _word("thanks.", 1.8, 2.1),
    ]
    out = sentences_from_words(words, "spk-a")
    assert [u["text"] for u in out] == ["Hi, how are you?", "Good thanks."]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0
    assert out[1]["start"] == 1.5 and out[1]["end"] == 2.1


def test_sentences_from_words_flushes_trailing_content_without_terminal_punc():
    words = [_word("so", 0.0, 0.2), _word("anyway", 0.2, 0.6)]
    out = sentences_from_words(words, "spk-a")
    assert [u["text"] for u in out] == ["so anyway"]


def test_sentences_from_words_drops_low_confidence_tokens_when_present():
    words = [
        _word("hello", 0.0, 0.3, confidence=0.95),
        _word("garble", 0.3, 0.5, confidence=0.2),  # below default 0.6 threshold
        _word("there.", 0.5, 0.9, confidence=0.9),
    ]
    out = sentences_from_words(words, "spk-a")
    assert [u["text"] for u in out] == ["hello there."]


def test_sentences_from_words_keeps_all_tokens_when_no_confidence_present():
    words = [_word("hello", 0.0, 0.3), _word("there.", 0.3, 0.7)]
    out = sentences_from_words(words, "spk-a")
    assert [u["text"] for u in out] == ["hello there."]


def test_sentences_from_words_confidence_threshold_is_configurable():
    words = [
        _word("hello", 0.0, 0.3, confidence=0.55),
        _word("there.", 0.3, 0.7, confidence=0.9),
    ]
    # Default threshold (0.6) drops the first word.
    assert [u["text"] for u in sentences_from_words(words, "spk-a")] == ["there."]
    # A lower threshold keeps it.
    assert [u["text"] for u in sentences_from_words(words, "spk-a", confidence_threshold=0.5)] == ["hello there."]


def test_sentences_from_words_supports_start_time_end_time_keys():
    words = [
        {"word": "hello", "start_time": 0.0, "end_time": 0.3},
        {"word": "there.", "start_time": 0.3, "end_time": 0.7},
    ]
    out = sentences_from_words(words, "spk-a")
    assert [u["text"] for u in out] == ["hello there."]


def test_sentences_from_words_prefers_punctuated_word():
    words = [
        {"word": "im", "punctuated_word": "I'm", "start": 0.0, "end": 0.3},
        {"word": "good", "punctuated_word": "good.", "start": 0.3, "end": 0.6},
    ]
    out = sentences_from_words(words, "spk-a")
    assert out[0]["text"] == "I'm good."


def test_sentences_from_words_empty_input():
    assert sentences_from_words([], "spk-a") == []


def test_collapse_short_pauses_merges_under_max_pause_and_splits_at_or_above():
    utterances = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "Hi,"},
        {"speaker": "A", "start": 1.4, "end": 2.0, "text": "how are you?"},  # gap 0.4 < 1.5: merge
        {"speaker": "A", "start": 4.0, "end": 4.5, "text": "Anyway."},  # gap 2.0 >= 1.5: new group
    ]
    out = collapse_short_pauses(utterances, max_pause=1.5)
    assert [u["text"] for u in out] == ["Hi, how are you?", "Anyway."]
    assert out[0]["start"] == 0.0 and out[0]["end"] == 2.0
    assert out[1]["start"] == 4.0 and out[1]["end"] == 4.5


def test_collapse_short_pauses_boundary_is_strict_not_inclusive():
    """Upstream: a gap >= max_pause starts a NEW group (merge only on <)."""
    utterances = [
        {"speaker": "A", "start": 0.0, "end": 1.0, "text": "one"},
        {"speaker": "A", "start": 2.5, "end": 3.0, "text": "two"},  # gap exactly 1.5
    ]
    out = collapse_short_pauses(utterances, max_pause=1.5)
    assert len(out) == 2, "an exact-tie gap must NOT merge"


def test_natural_turns_end_to_end_word_level_fixes_fragmented_speaker():
    """The whole point of the server port: raw word-timed ASR that fragments
    one speaker's continuous speech into many short utterances (a plausible
    Deepgram-utterance-boundary artifact) must still resolve to ONE merged
    primary turn once sentence-stitching + short-pause collapse runs before
    containment — the stage the recall regression was traced to."""
    words_by_speaker = {
        "A": [
            _word("So", 0.0, 0.2),
            _word("I", 0.4, 0.5),
            _word("went", 0.5, 0.7),
            _word("to", 0.7, 0.8),
            _word("the", 0.8, 0.9),
            _word("store.", 1.6, 2.0),  # 0.7s pause mid-sentence, < 1.5s -> collapse
            _word("Then", 2.3, 2.5),
            _word("I", 2.5, 2.6),
            _word("came", 2.6, 2.8),
            _word("home.", 2.8, 3.2),
        ],
        "B": [_word("mhm", 1.0, 1.3)],
    }
    turns = natural_turns(words_by_speaker)
    a_turns = [t for t in turns if t["speaker"] == "A"]
    assert len(a_turns) == 1, "the whole A narrative collapses into one primary turn"
    assert a_turns[0]["text"] == "So I went to the store. Then I came home."
    b_backchannels = [a for t in turns for a in t["attached"] if a["speaker"] == "B"]
    assert len(b_backchannels) == 1
    assert b_backchannels[0]["kind"] == "backchannel"


def test_words_of_matches_ts_regex_semantics():
    assert words_of("Yeah, exactly!") == ["yeah", "exactly"]
    assert words_of("I'm... okay?") == ["i'm", "okay"]
    assert words_of("...") == []
