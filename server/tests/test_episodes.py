"""Unit tests for episodes.py — pure episode segmentation (Companion P1).

No app, no LLM, no I/O: segment_episodes/episodes_from_analysis are pure
functions over the transcript + stored-analysis shapes, so every case here is
hand-computed. Endpoint coverage (episodes in the upload response and the
detail read's backfill) lives in test_recordings.py alongside the storage
fixtures.
"""

import episodes


def _turn(speaker, text, start, end):
    return {
        "speaker": speaker, "text": text,
        "start_time": start, "end_time": end,
    }


# A short two-speaker conversation with no gap anywhere near the threshold.
SHORT_TURNS = [
    _turn("Speaker A", "Hey, can we talk about the schedule?", 0.0, 2.0),
    _turn("Speaker B", "Sure, what about it.", 2.5, 4.0),
    _turn("Speaker A", "You never stick to what we agree.", 4.5, 6.0),
    _turn("Speaker B", "That is not fair and you know it.", 6.5, 8.0),
]

# Three conversations separated by >60s silences: turns 0-1, 2-3, 4.
GAPPY_TURNS = [
    _turn("Speaker A", "Morning. Coffee's ready.", 0.0, 2.0),
    _turn("Speaker B", "Thanks. Running late though.", 3.0, 5.0),
    # 95s gap (5.0 → 100.0)
    _turn("Speaker A", "Did you call the plumber?", 100.0, 102.0),
    _turn("Speaker B", "Not yet, I will after lunch.", 103.0, 105.0),
    # 295s gap (105.0 → 400.0)
    _turn("Speaker A", "Okay, dinner's on the table.", 400.0, 402.0),
]


def _heats(values):
    """per_turn stubs carrying just the heat (the only field episodes reads)."""
    return [{"heat": h} for h in values]


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def test_short_recording_is_one_episode():
    eps = episodes.segment_episodes(SHORT_TURNS)
    assert len(eps) == 1
    ep = eps[0]
    assert ep["index"] == 0
    assert ep["first_turn_index"] == 0
    assert ep["last_turn_index"] == 3
    assert ep["turn_count"] == 4
    assert ep["start_time"] == 0.0
    assert ep["end_time"] == 8.0
    assert ep["duration_seconds"] == 8.0


def test_gaps_split_into_episodes_with_correct_spans():
    eps = episodes.segment_episodes(GAPPY_TURNS)
    assert [(e["first_turn_index"], e["last_turn_index"]) for e in eps] == [
        (0, 1), (2, 3), (4, 4),
    ]
    assert [e["index"] for e in eps] == [0, 1, 2]
    assert [(e["start_time"], e["end_time"]) for e in eps] == [
        (0.0, 5.0), (100.0, 105.0), (400.0, 402.0),
    ]
    assert [e["turn_count"] for e in eps] == [2, 2, 1]
    assert eps[1]["duration_seconds"] == 5.0


def test_gap_exactly_at_threshold_does_not_split():
    # Boundary rule is STRICTLY greater than gap_seconds.
    turns = [
        _turn("A", "one", 0.0, 2.0),
        _turn("B", "two", 62.0, 64.0),  # gap of exactly 60.0
    ]
    assert len(episodes.segment_episodes(turns, gap_seconds=60.0)) == 1
    # A hair over splits.
    turns[1]["start_time"] = 62.1
    assert len(episodes.segment_episodes(turns, gap_seconds=60.0)) == 2


def test_gap_seconds_is_parameterized():
    # The 95s gap in GAPPY_TURNS: a 100s threshold merges episodes 1+2.
    eps = episodes.segment_episodes(GAPPY_TURNS, gap_seconds=100.0)
    assert [(e["first_turn_index"], e["last_turn_index"]) for e in eps] == [
        (0, 3), (4, 4),
    ]


def test_missing_timestamps_read_as_contiguous_single_episode():
    # A text-analyze transcript carries no timing: no gap can be detected, so
    # it is ONE episode with honest null timing — never invented boundaries.
    turns = [
        {"speaker": "A", "text": "hello", "start_time": None, "end_time": None},
        {"speaker": "B", "text": "hi", "start_time": None, "end_time": None},
    ]
    eps = episodes.segment_episodes(turns)
    assert len(eps) == 1
    assert eps[0]["start_time"] is None
    assert eps[0]["end_time"] is None
    assert eps[0]["duration_seconds"] is None


def test_out_of_order_end_times_do_not_manufacture_gaps():
    # An overlapping turn whose end rewinds the clock must not create a
    # phantom gap: the tracker keeps the FURTHEST spoken moment.
    turns = [
        _turn("A", "long monologue", 0.0, 90.0),
        _turn("B", "quick interjection", 10.0, 12.0),  # ends before A does
        _turn("A", "continues", 95.0, 97.0),  # 5s after A's end, 83s after B's
    ]
    assert len(episodes.segment_episodes(turns)) == 1


def test_empty_turns_yield_no_episodes():
    assert episodes.segment_episodes([]) == []


# ---------------------------------------------------------------------------
# Heat math
# ---------------------------------------------------------------------------

def test_heat_mean_and_peak_per_episode():
    eps = episodes.segment_episodes(
        GAPPY_TURNS, per_turn=_heats([10, 20, 30, 50, 40]),
    )
    assert [(e["mean_heat"], e["peak_heat"]) for e in eps] == [
        (15.0, 20), (40.0, 50), (40.0, 40),
    ]


def test_misaligned_per_turn_yields_null_heat():
    # A per_turn list that doesn't align with the transcript cannot be trusted
    # against it — heats are honest nulls, never zeros.
    eps = episodes.segment_episodes(GAPPY_TURNS, per_turn=_heats([10, 20]))
    assert all(e["mean_heat"] is None and e["peak_heat"] is None for e in eps)


def test_non_numeric_heats_are_skipped_not_invented():
    eps = episodes.segment_episodes(
        SHORT_TURNS,
        per_turn=[{"heat": 10}, {"heat": None}, {}, {"heat": 30}],
    )
    assert eps[0]["mean_heat"] == 20.0
    assert eps[0]["peak_heat"] == 30


# ---------------------------------------------------------------------------
# Participants + display labels
# ---------------------------------------------------------------------------

def test_participants_use_display_labels_and_enrolled_you():
    eps = episodes.segment_episodes(
        SHORT_TURNS,
        speaker_labels={
            "Speaker A": {"display_label": "Jordan", "label_source": "name"},
            "Speaker B": {"display_label": "Speaker B", "label_source": "generic"},
        },
        speaker_identity={"matched_speaker": "Speaker B"},
    )
    ep = eps[0]
    # Canonical ids in first-appearance order…
    assert ep["speakers"] == ["Speaker A", "Speaker B"]
    # …and display labels: name rung for A, enrolled "You" beats generic for B.
    assert ep["participants"] == ["Jordan", "You"]


def test_participants_fall_back_to_raw_ids():
    eps = episodes.segment_episodes(SHORT_TURNS)
    assert eps[0]["participants"] == ["Speaker A", "Speaker B"]


def test_participants_are_scoped_per_episode():
    turns = [
        _turn("A", "solo start", 0.0, 2.0),
        # >60s gap — B only ever appears in the second episode.
        _turn("B", "later arrival", 100.0, 102.0),
        _turn("A", "reply", 103.0, 104.0),
    ]
    eps = episodes.segment_episodes(turns)
    assert [e["participants"] for e in eps] == [["A"], ["B", "A"]]


def test_malformed_speaker_identity_is_ignored():
    eps = episodes.segment_episodes(
        SHORT_TURNS, speaker_identity={"matched_speaker": "  "},
    )
    assert eps[0]["participants"] == ["Speaker A", "Speaker B"]


# ---------------------------------------------------------------------------
# Summaries — derived only, never fabricated
# ---------------------------------------------------------------------------

def test_single_episode_reuses_existing_title_as_summary():
    eps = episodes.segment_episodes(SHORT_TURNS, title="Schedule friction")
    assert eps[0]["summary"] == "Schedule friction"
    assert eps[0]["summary_source"] == "title"


def test_multi_episode_summaries_are_verbatim_opening_excerpts():
    # The whole-recording title cannot describe one of several episodes, so
    # each episode quotes its own opening turn instead.
    eps = episodes.segment_episodes(GAPPY_TURNS, title="A whole day")
    assert [e["summary"] for e in eps] == [
        "Morning. Coffee's ready.",
        "Did you call the plumber?",
        "Okay, dinner's on the table.",
    ]
    assert all(e["summary_source"] == "excerpt" for e in eps)


def test_excerpt_is_truncated_with_ellipsis():
    long_text = "word " * 60  # ~300 chars
    eps = episodes.segment_episodes([_turn("A", long_text, 0.0, 1.0)])
    assert eps[0]["summary"].endswith("…")
    assert len(eps[0]["summary"]) <= 100


def test_blank_text_yields_null_summary():
    eps = episodes.segment_episodes(
        [{"speaker": "A", "text": "   ", "start_time": 0.0, "end_time": 1.0}],
    )
    assert eps[0]["summary"] is None
    assert eps[0]["summary_source"] is None


# ---------------------------------------------------------------------------
# episodes_from_analysis — the stored-recording backfill
# ---------------------------------------------------------------------------

def test_from_analysis_derives_from_stored_shapes():
    analysis = {
        "per_turn": _heats([10, 20, 30, 50, 40]),
        "speaker_labels": {
            "Speaker A": {"display_label": "Sam", "label_source": "name"},
        },
        "speaker_identity": {"matched_speaker": "Speaker B"},
        "title": "ignored for multi-episode",
    }
    eps = episodes.episodes_from_analysis(GAPPY_TURNS, analysis)
    assert len(eps) == 3
    assert eps[0]["participants"] == ["Sam", "You"]
    assert eps[0]["mean_heat"] == 15.0
    assert eps[2]["peak_heat"] == 40


def test_from_analysis_without_analysis_is_none():
    # A stored-but-unanalyzed recording has no heats/labels/title to derive
    # from — the honest answer is None, not an invented episode list.
    assert episodes.episodes_from_analysis(SHORT_TURNS, None) is None


def test_from_analysis_tolerates_minimal_old_analysis():
    # A very old analysis.json (no per_turn/speaker_labels/title keys) still
    # segments — with null heats and raw-id participants.
    eps = episodes.episodes_from_analysis(SHORT_TURNS, {"narrative": "x"})
    assert len(eps) == 1
    assert eps[0]["mean_heat"] is None
    assert eps[0]["participants"] == ["Speaker A", "Speaker B"]
