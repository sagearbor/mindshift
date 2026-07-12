"""Tests for POST /analyze (conversation-dynamics) and the pure statistics in
dynamics.py.

The endpoint tests mock the LLM via ``main.get_llm_client`` (MagicMock) exactly
like the existing REST suite — the LLM only ever supplies per-turn
{heat, markers, trigger_phrase} + requests + narrative; every statistic is
Python's, so a fixed mock makes the whole pipeline deterministic.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import dynamics
import main
from auth import get_current_uid
from main import app, init_db


# ---------------------------------------------------------------------------
# Client fixture — resets the shared rate limiter so cross-test call volume on
# the shared client IP can never trip the limiter (mirrors test_endpoints).
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    await init_db()
    main._rate_limiter.reset()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# The hand-built 2-speaker fixture (verified by hand; see per-value asserts).
# 12 turns, Alice/Bob alternating (6 each), with timestamps so interruptions
# are computable and one Alice + one Bob interruption occur.
# ---------------------------------------------------------------------------

FIXTURE_TURNS = [
    # (speaker, text-length via marker below, heat, start, end)
    {"speaker": "Alice", "text": "a" * 20, "start_time": 0.0, "end_time": 2.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 2.5, "end_time": 4.0},
    {"speaker": "Alice", "text": "a" * 20, "start_time": 4.5, "end_time": 6.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 5.5, "end_time": 7.0},
    {"speaker": "Alice", "text": "a" * 20, "start_time": 6.5, "end_time": 9.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 9.5, "end_time": 11.0},
    {"speaker": "Alice", "text": "a" * 20, "start_time": 11.5, "end_time": 13.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 13.5, "end_time": 15.0},
    {"speaker": "Alice", "text": "a" * 20, "start_time": 15.5, "end_time": 17.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 17.5, "end_time": 19.0},
    {"speaker": "Alice", "text": "a" * 20, "start_time": 19.5, "end_time": 21.0},
    {"speaker": "Bob", "text": "b" * 10, "start_time": 21.5, "end_time": 23.0},
]

# Per-turn LLM output aligned 1:1 with FIXTURE_TURNS. Note idx0 carries an
# UNKNOWN marker ("not_a_marker") that must be dropped.
FIXTURE_PER_TURN = [
    {"heat": 10, "markers": ["not_a_marker"], "trigger_phrase": None},
    {"heat": 15, "markers": [], "trigger_phrase": None},
    {"heat": 40, "markers": ["criticism"], "trigger_phrase": "you never listen"},
    {"heat": 30, "markers": ["defensiveness"], "trigger_phrase": None},
    {"heat": 55, "markers": ["contempt"], "trigger_phrase": "that's not fair"},
    {"heat": 20, "markers": [], "trigger_phrase": None},
    {"heat": 25, "markers": ["repair_attempt"], "trigger_phrase": None},
    {"heat": 10, "markers": ["validation"], "trigger_phrase": None},
    {"heat": 15, "markers": [], "trigger_phrase": None},
    {"heat": 12, "markers": [], "trigger_phrase": None},
    {"heat": 18, "markers": [], "trigger_phrase": None},
    {"heat": 14, "markers": [], "trigger_phrase": None},
]

def _report_cards(*speakers, **overrides) -> dict:
    """A well-formed report_cards block: one card per speaker. Individual cards
    can be replaced via ``overrides={speaker: card}`` to exercise clamping /
    truncation / omission."""
    cards = {
        sp: {
            "score": 70,
            "headline": f"{sp} stayed engaged",
            "did_well": "Kept trying to reconnect under pressure.",
            "work_on": "Pause and breathe before answering criticism.",
        }
        for sp in speakers
    }
    cards.update(overrides)
    return cards


FIXTURE_LLM_JSON = json.dumps({
    "per_turn": FIXTURE_PER_TURN,
    "requests": [
        {"speaker": "Bob", "request": "clean the kitchen", "outcome": "granted"},
        {"speaker": "Alice", "request": "more time", "outcome": "someday"},
        {"speaker": "", "request": "x", "outcome": "granted"},  # malformed → dropped
    ],
    "narrative": "You both clearly care and keep trying to reconnect. The "
    "friction shows up as a criticism/defensiveness loop that Alice's repair "
    "attempt eventually breaks.",
    "report_cards": _report_cards("Alice", "Bob"),
})


def _mock_llm(payload: str) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = payload
    return m


def _mock_per_turn(n: int, heats=None) -> list[dict]:
    heats = heats if heats is not None else [30] * n
    return [
        {"heat": heats[i], "markers": [], "trigger_phrase": None}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Happy path — full shape + every Python statistic on the hand-built fixture
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_happy_path_full_shape_and_stats(client):
    with patch("main.get_llm_client", return_value=_mock_llm(FIXTURE_LLM_JSON)):
        resp = await client.post("/analyze", json={
            "turns": FIXTURE_TURNS,
            "context": "Argument about chores",
        })

    assert resp.status_code == 200
    data = resp.json()

    # --- per_turn: length, alignment, spike flag, dropped-unknown marker ---
    assert len(data["per_turn"]) == 12
    pt = data["per_turn"]
    assert pt[0]["index"] == 0 and pt[0]["speaker"] == "Alice"
    assert pt[0]["markers"] == []            # unknown "not_a_marker" dropped
    assert pt[2]["markers"] == ["criticism"]
    assert pt[2]["is_spike"] is True         # Alice 10 → 40 (>= +20)
    assert pt[0]["is_spike"] is False        # speaker's first turn never spikes
    assert all(t["is_spike"] is False for i, t in enumerate(pt) if i != 2)
    assert pt[2]["trigger_phrase"] == "you never listen"

    # --- per_speaker: exact talk_share (char-based), heat stats, repairs ---
    ps = data["per_speaker"]
    assert set(ps) == {"Alice", "Bob"}
    assert ps["Alice"]["talk_share"] == 0.6667   # 120 / 180 chars
    assert ps["Bob"]["talk_share"] == 0.3333
    assert ps["Alice"]["turns"] == 6 and ps["Bob"]["turns"] == 6
    assert ps["Alice"]["avg_heat"] == 27.17
    assert ps["Alice"]["peak_heat"] == 55
    assert ps["Alice"]["peak_turn_index"] == 4
    assert ps["Alice"]["heat_variance"] == 245.14
    assert ps["Bob"]["avg_heat"] == 16.83
    assert ps["Bob"]["peak_turn_index"] == 3
    # interruptions: timestamps present → one each (Bob@3, Alice@4)
    assert ps["Alice"]["interruptions"] == 1
    assert ps["Bob"]["interruptions"] == 1
    # horsemen counts
    assert ps["Alice"]["horsemen"] == {
        "criticism": 1, "contempt": 1, "defensiveness": 0, "stonewalling": 0,
    }
    assert ps["Bob"]["horsemen"]["defensiveness"] == 1
    # repairs: Alice attempted 1 and it was accepted (Bob then cooled >= 10)
    assert ps["Alice"]["repair_attempts"] == 1
    assert ps["Alice"]["repairs_accepted"] == 1
    assert ps["Bob"]["repair_attempts"] == 0

    # --- dynamics ---
    dyn = data["dynamics"]
    # coupling non-null with >= 6 turns each
    assert dyn["coupling"]["strength"] is not None
    assert -1.0 <= dyn["coupling"]["strength"] <= 1.0
    assert dyn["coupling"]["description"]
    # de-escalation: Alice cooled first, Bob followed 100%
    assert dyn["deescalation"]["who_first"] == "Alice"
    assert dyn["deescalation"]["follow_rate"] == 1.0
    # triggers sorted by heat_delta desc
    assert dyn["triggers"][0]["phrase"] == "you never listen"
    assert dyn["triggers"][0]["heat_delta"] == 15
    # requests: valid + normalized-unknown kept, malformed dropped
    outcomes = {r["request"]: r["outcome"] for r in dyn["requests"]}
    assert outcomes == {"clean the kitchen": "granted", "more time": "unclear"}

    assert isinstance(data["narrative"], str) and data["narrative"]


# ---------------------------------------------------------------------------
# Honest-failure paths (no fabrication)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_misaligned_per_turn_length_is_502(client):
    """per_turn shorter than the input turns → 502, never padded."""
    bad = json.dumps({
        "per_turn": _mock_per_turn(3),  # fixture has 12 turns
        "requests": [],
        "narrative": "x",
    })
    with patch("main.get_llm_client", return_value=_mock_llm(bad)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502
    assert "misaligned" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_analyze_unparseable_llm_is_502(client):
    with patch("main.get_llm_client", return_value=_mock_llm("not json at all")):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_analyze_non_dict_llm_json_is_502(client):
    """Valid JSON that isn't an object ("[]") must be an honest 502, not an
    AttributeError-driven 500 (review MINOR-2)."""
    with patch("main.get_llm_client", return_value=_mock_llm("[]")):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_analyze_non_numeric_heat_is_502(client):
    pt = _mock_per_turn(12)
    pt[5]["heat"] = "very hot"
    bad = json.dumps({"per_turn": pt, "requests": [], "narrative": "x"})
    with patch("main.get_llm_client", return_value=_mock_llm(bad)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502
    assert "heat" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_analyze_empty_narrative_is_502(client):
    bad = json.dumps({
        "per_turn": _mock_per_turn(12), "requests": [], "narrative": "  ",
    })
    with patch("main.get_llm_client", return_value=_mock_llm(bad)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_analyze_heat_clamped_to_range(client):
    """Out-of-range heats are clamped to 0-100, never rejected."""
    pt = _mock_per_turn(12, heats=[150, -5, 40, 30, 55, 20, 25, 10, 15, 12, 18, 14])
    payload = json.dumps({
        "per_turn": pt, "requests": [], "narrative": "ok",
        "report_cards": _report_cards("Alice", "Bob"),
    })
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 200
    heats = [t["heat"] for t in resp.json()["per_turn"]]
    assert heats[0] == 100 and heats[1] == 0


# ---------------------------------------------------------------------------
# Timestamps absent → interruptions None (never fabricated)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_no_timestamps_interruptions_none(client):
    turns = [{"speaker": t["speaker"], "text": t["text"]} for t in FIXTURE_TURNS]
    with patch("main.get_llm_client", return_value=_mock_llm(FIXTURE_LLM_JSON)):
        resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 200
    ps = resp.json()["per_speaker"]
    assert ps["Alice"]["interruptions"] is None
    assert ps["Bob"]["interruptions"] is None


# ---------------------------------------------------------------------------
# Speaker-count validation (422) + 3-speaker top-2 coupling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_zero_speakers_impossible_but_eleven_rejected(client):
    # Monologues (1 speaker) are now ACCEPTED — see
    # test_analyze_single_speaker_monologue_succeeds. The upper bound stands.
    turns = [{"speaker": f"S{i}", "text": "hello"} for i in range(11)]
    resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_analyze_eleven_speakers_rejected_422(client):
    turns = [{"speaker": f"S{i}", "text": "hello"} for i in range(11)]
    resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_analyze_too_few_turns_rejected_422(client):
    turns = [
        {"speaker": "Alice", "text": "hi"},
        {"speaker": "Bob", "text": "yo"},
        {"speaker": "Alice", "text": "ok"},
    ]  # 3 turns < min 4
    resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_analyze_three_speakers_top_two_coupling(client):
    # A and B get 6 turns each (enough for coupling); C gets 2. Coupling must
    # be measured on the top-2 pair and say so.
    turns = []
    heats = []
    a_heats = [10, 40, 55, 25, 15, 18]
    b_heats = [15, 30, 20, 10, 12, 14]
    for i in range(6):
        turns.append({"speaker": "A", "text": "a" * 12})
        heats.append(a_heats[i])
        turns.append({"speaker": "B", "text": "b" * 12})
        heats.append(b_heats[i])
    turns.append({"speaker": "C", "text": "c" * 12})
    heats.append(20)
    turns.append({"speaker": "C", "text": "c" * 12})
    heats.append(22)

    payload = json.dumps({
        "per_turn": _mock_per_turn(len(turns), heats=heats),
        "requests": [],
        "narrative": "A working analysis of the group dynamic.",
        "report_cards": _report_cards("A", "B", "C"),
    })
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["per_speaker"]) == {"A", "B", "C"}
    coupling = data["dynamics"]["coupling"]
    # top-2 (A, B) each have >= 6 turns → measurable, and description says so.
    assert coupling["strength"] is not None
    assert "most active" in coupling["description"].lower()
    assert "A" in coupling["description"] and "B" in coupling["description"]


# ---------------------------------------------------------------------------
# Transcript size cap (413)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_transcript_too_large_is_413(client):
    # 40 turns × 2000 chars = 80k > 60k cap. Two speakers, valid otherwise.
    turns = [
        {"speaker": "Alice" if i % 2 == 0 else "Bob", "text": "x" * 2000}
        for i in range(40)
    ]
    resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Auth required (mirror existing pattern: drop the override → real dependency)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_requires_auth_401(client, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# §2 — per-person report cards (present for every speaker, clamped, truncated)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_analyze_report_cards_present_clamped_truncated(client):
    """Every speaker gets a card; an out-of-range score clamps and over-long
    strings truncate to their caps (never a rejection)."""
    payload = json.dumps({
        "per_turn": FIXTURE_PER_TURN,
        "requests": [],
        "narrative": "You both keep showing up for each other.",
        "report_cards": _report_cards(
            "Alice", "Bob",
            Alice={
                "score": 150,                       # clamps to 100
                "headline": "H" * 200,              # truncates to 80
                "did_well": "d" * 300,              # truncates to 200
                "work_on": "w" * 300,               # truncates to 200
            },
        ),
    })
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 200
    cards = resp.json()["report_cards"]
    assert set(cards) == {"Alice", "Bob"}
    assert cards["Alice"]["score"] == 100
    assert len(cards["Alice"]["headline"]) == main.REPORT_CARD_HEADLINE_MAX
    assert len(cards["Alice"]["did_well"]) == main.REPORT_CARD_TEXT_MAX
    assert len(cards["Alice"]["work_on"]) == main.REPORT_CARD_TEXT_MAX
    # A well-formed card passes through intact.
    assert cards["Bob"]["score"] == 70
    assert cards["Bob"]["headline"] == "Bob stayed engaged"


@pytest.mark.anyio
async def test_analyze_missing_speaker_report_card_is_502(client):
    """A speaker with no report card is an honest 502 misalignment, never a
    fabricated card."""
    payload = json.dumps({
        "per_turn": FIXTURE_PER_TURN,
        "requests": [],
        "narrative": "A working narrative.",
        "report_cards": _report_cards("Alice"),  # Bob omitted
    })
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze", json={"turns": FIXTURE_TURNS})
    assert resp.status_code == 502
    detail = resp.json()["detail"].lower()
    assert "report card" in detail and "bob" in detail


# ---------------------------------------------------------------------------
# §3 — POST /analyze/counterfactual
# ---------------------------------------------------------------------------

def _counterfactual_json(
    simulated_heat,
    rewritten="Let's find a split of the chores that feels fair to us both.",
    rationale="States the need directly without blame, inviting a partnership.",
) -> str:
    return json.dumps({
        "rewritten_text": rewritten,
        "rationale": rationale,
        "simulated_heat": simulated_heat,
    })


@pytest.mark.anyio
async def test_counterfactual_happy_path(client):
    """Length/order/speaker mapping correct, heats clamped, disclaimer exact."""
    pivot_index = 4  # Alice's turn; 12 - 4 = 8 subsequent turns (incl. pivot)
    # 8 values; first is out-of-range to prove clamping.
    sim = [150, 10, 14, 8, 11, 9, 13, 7]
    with patch("main.get_llm_client", return_value=_mock_llm(_counterfactual_json(sim))):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS,
            "pivot_index": pivot_index,
            "context": "Argument about chores",
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["pivot_index"] == 4
    assert data["rewritten_text"].startswith("Let's find")
    assert len(data["rationale"]) <= main.COUNTERFACTUAL_RATIONALE_MAX
    assert data["disclaimer"] == main.COUNTERFACTUAL_DISCLAIMER

    spt = data["simulated_per_turn"]
    assert len(spt) == 8
    # index runs pivot..last, in order
    assert [t["index"] for t in spt] == [4, 5, 6, 7, 8, 9, 10, 11]
    # speaker sequence mirrors the real transcript (Alice/Bob alternating)
    assert [t["speaker"] for t in spt] == [
        "Alice", "Bob", "Alice", "Bob", "Alice", "Bob", "Alice", "Bob",
    ]
    # heats clamped; the 150 became 100
    assert [t["heat"] for t in spt] == [100, 10, 14, 8, 11, 9, 13, 7]


@pytest.mark.anyio
async def test_counterfactual_rationale_truncated(client):
    sim = [12, 10, 14, 8, 11, 9, 13, 7]
    long_rationale = "r" * 400
    payload = _counterfactual_json(sim, rationale=long_rationale)
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS, "pivot_index": 4,
        })
    assert resp.status_code == 200
    assert len(resp.json()["rationale"]) == main.COUNTERFACTUAL_RATIONALE_MAX


@pytest.mark.anyio
async def test_counterfactual_pivot_out_of_range_is_422(client):
    with patch("main.get_llm_client", return_value=_mock_llm(_counterfactual_json([0]))):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS, "pivot_index": 12,  # len == 12 → index 12 invalid
        })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_counterfactual_negative_pivot_is_422(client):
    resp = await client.post("/analyze/counterfactual", json={
        "turns": FIXTURE_TURNS, "pivot_index": -1,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_counterfactual_misaligned_simulated_heat_is_502(client):
    # pivot 4 needs 8 values; supply 3 → misaligned.
    with patch("main.get_llm_client", return_value=_mock_llm(_counterfactual_json([1, 2, 3]))):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS, "pivot_index": 4,
        })
    assert resp.status_code == 502
    assert "misaligned" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_counterfactual_non_dict_llm_json_is_502(client):
    with patch("main.get_llm_client", return_value=_mock_llm("[]")):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS, "pivot_index": 4,
        })
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_counterfactual_non_numeric_simulated_heat_is_502(client):
    sim = [12, "boiling", 14, 8, 11, 9, 13, 7]
    with patch("main.get_llm_client", return_value=_mock_llm(_counterfactual_json(sim))):
        resp = await client.post("/analyze/counterfactual", json={
            "turns": FIXTURE_TURNS, "pivot_index": 4,
        })
    assert resp.status_code == 502
    assert "heat" in resp.json()["detail"].lower()


@pytest.mark.anyio
async def test_counterfactual_requires_auth_401(client, monkeypatch):
    monkeypatch.delitem(app.dependency_overrides, get_current_uid)
    resp = await client.post("/analyze/counterfactual", json={
        "turns": FIXTURE_TURNS, "pivot_index": 4,
    })
    assert resp.status_code == 401


# ===========================================================================
# Unit tests for dynamics.py pure functions
# ===========================================================================

class TestTalkShare:
    def test_char_based_shares_sum_to_one(self):
        shares = dynamics.talk_share(["A", "B", "A"], [30, 10, 10])
        assert shares == {"A": 0.8, "B": 0.2}

    def test_all_empty_no_division_error(self):
        assert dynamics.talk_share(["A", "B"], [0, 0]) == {"A": 0.0, "B": 0.0}


class TestInterruptions:
    def test_none_when_any_timestamp_missing(self):
        assert dynamics.count_interruptions(
            ["A", "B"], [0.0, None], [1.0, 2.0],
        ) is None

    def test_counts_overlap_by_interrupter(self):
        # B starts (1.5) before A's turn ends (2.0) → B interrupts A.
        counts = dynamics.count_interruptions(
            ["A", "B", "A"], [0.0, 1.5, 3.0], [2.0, 2.5, 4.0],
        )
        assert counts == {"A": 0, "B": 1}

    def test_same_speaker_back_to_back_not_an_interruption(self):
        counts = dynamics.count_interruptions(
            ["A", "A"], [0.0, 0.5], [1.0, 2.0],
        )
        assert counts == {"A": 0}


class TestSpikeFlags:
    def test_first_per_speaker_never_spikes(self):
        flags = dynamics.spike_flags(["A", "B"], [90, 95])
        assert flags == [False, False]

    def test_detects_and_ignores_below_threshold(self):
        # A: 10 → 30 (+20 spike) → 45 (+15 no) ; B first never
        flags = dynamics.spike_flags(["A", "B", "A", "A"], [10, 5, 30, 45])
        assert flags == [False, False, True, False]


class TestSpeakerHeatStats:
    def test_single_turn_zero_variance(self):
        stats = dynamics.speaker_heat_stats(["A"], [40])
        assert stats["A"] == {
            "turns": 1, "avg_heat": 40.0, "peak_heat": 40,
            "peak_turn_index": 0, "heat_variance": 0.0,
        }


class TestRepairs:
    def test_accepted_when_other_party_cools(self):
        # A repairs at idx0; B's next turn drops >= 10 vs B's previous.
        speakers = ["B", "A", "B"]
        heats = [40, 20, 25]
        markers = [[], ["repair_attempt"], []]
        attempts, accepted = dynamics.count_repairs(speakers, heats, markers)
        assert attempts == {"A": 1, "B": 0}
        assert accepted == {"A": 1, "B": 0}  # B 40 → 25 (>= 10 drop)

    def test_not_accepted_when_other_party_stays_hot(self):
        speakers = ["B", "A", "B"]
        heats = [40, 20, 38]
        markers = [[], ["repair_attempt"], []]
        _, accepted = dynamics.count_repairs(speakers, heats, markers)
        assert accepted == {"A": 0, "B": 0}


class TestCoupling:
    def test_short_conversation_none(self):
        # Only 3 turns each → below COUPLING_MIN_TURNS.
        speakers = ["A", "B", "A", "B", "A", "B"]
        heats = [10, 20, 30, 40, 50, 60]
        result = dynamics.compute_coupling(speakers, heats)
        assert result["strength"] is None and result["leader"] is None
        assert "not enough data" in result["description"].lower()

    def test_more_than_two_speakers_names_top_pair(self):
        speakers = ["A", "B"] * 6 + ["C", "C"]
        heats = [10, 15, 40, 30, 55, 20, 25, 10, 15, 12, 18, 14, 20, 22]
        result = dynamics.compute_coupling(speakers, heats)
        assert "most active" in result["description"].lower()
        assert "A" in result["description"] and "B" in result["description"]


class TestDeescalation:
    def test_single_event_who_first(self):
        # A drops 50 → 30 (>= 15) at idx2; no follow within 2 turns.
        speakers = ["A", "B", "A", "B"]
        heats = [50, 20, 30, 20]
        result = dynamics.compute_deescalation(speakers, heats)
        assert result["who_first"] == "A"
        # B at idx3 is 20 vs B previous 20 → no >=10 drop → follow_rate 0.0
        assert result["follow_rate"] == 0.0

    def test_no_events_returns_none(self):
        speakers = ["A", "B", "A", "B"]
        heats = [10, 12, 14, 16]
        result = dynamics.compute_deescalation(speakers, heats)
        assert result["who_first"] is None
        assert result["follow_rate"] is None
        assert "no clear de-escalation" in result["description"].lower()


class TestTriggers:
    def test_ordering_and_delta(self):
        speakers = ["A", "B", "A", "B"]
        heats = [40, 30, 55, 20]
        phrases = ["mild", None, "harsh", None]
        # idx0 trigger: B first turn → base=heats[0]=40, delta 30-40=-10
        # idx2 trigger: B next=idx3 heat20, B prev(idx1)=30, delta 20-30=-10
        triggers = dynamics.extract_triggers(speakers, heats, phrases)
        assert len(triggers) == 2
        # Equal deltas → tie broken by turn_index ascending.
        assert triggers[0]["turn_index"] == 0
        assert triggers[0]["heat_delta"] == -10


class TestClampHeatHelper:
    def test_clamps_and_rejects(self):
        assert main._clamp_heat(150) == 100
        assert main._clamp_heat(-5) == 0
        assert main._clamp_heat(42.6) == 43
        assert main._clamp_heat("hot") is None
        assert main._clamp_heat(True) is None


@pytest.mark.anyio
async def test_analyze_single_speaker_monologue_succeeds(client):
    """One distinct speaker is IN scope (a real recording of one person doing
    two voices got diarized as a single speaker and bounced at validation).
    Pair dynamics come back as honest nulls with plain descriptions."""
    turns = [
        {"speaker": "Me", "text": f"turn number {i} of my solo rant"}
        for i in range(6)
    ]
    pt = _mock_per_turn(6, heats=[10, 25, 45, 60, 30, 15])
    payload = json.dumps({
        "per_turn": pt,
        "requests": [],
        "narrative": "A solo reflection that rose and settled.",
        "report_cards": _report_cards("Me"),
    })
    with patch("main.get_llm_client", return_value=_mock_llm(payload)):
        resp = await client.post("/analyze", json={"turns": turns})
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["per_speaker"]) == {"Me"}
    assert data["dynamics"]["coupling"]["strength"] is None
    assert "one speaker" in data["dynamics"]["coupling"]["description"].lower()
    assert data["dynamics"]["deescalation"]["follow_rate"] is None
    assert data["report_cards"]["Me"]["score"] == 70
