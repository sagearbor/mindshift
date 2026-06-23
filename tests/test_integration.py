"""
End-to-end integration tests for MindShift API.

Drives the FastAPI app through httpx TestClient, exercising /respond, /score,
and /session endpoints against the sample conversation fixtures.
All LLM calls are mocked — no API key needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from _mock_data import (
    MOCK_ASSERTIVE_JSON,
    MOCK_FULL_EMPATHY_JSON,
    MOCK_RESPOND_JSON,
    MOCK_SCORE_JSON,
    TONE_SCORE_KEYS,
)


# ---------------------------------------------------------------------------
# Slider-aware mock: return different suggestions per empathy range
# ---------------------------------------------------------------------------

def _slider_side_effect(mock_jsons: dict[str, str]):
    """Return a side_effect that picks a mock response based on system prompt."""
    def _side_effect(*, system, user, **kwargs):
        lower = system.lower()
        if "assertive" in lower and "fully" not in lower:
            return mock_jsons["assertive"]
        elif "fully empathetic" in lower:
            return mock_jsons["full_empathy"]
        else:
            return mock_jsons["balanced"]
    return _side_effect


SLIDER_MOCKS = {
    "assertive": MOCK_ASSERTIVE_JSON,
    "balanced": MOCK_RESPOND_JSON,
    "full_empathy": MOCK_FULL_EMPATHY_JSON,
}


# ---------------------------------------------------------------------------
# 1. POST /respond for each fixture conversation turn
# ---------------------------------------------------------------------------

class TestRespondFixtures:
    """POST every turn from all fixture conversations to /respond."""

    @pytest.mark.anyio
    async def test_all_conversations_all_turns(self, client, sample_transcripts):
        conversations = sample_transcripts["conversations"]
        assert len(conversations) == 10

        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_RESPOND_JSON
            mock_get.return_value = mock_client

            for conv in conversations:
                role = conv["roles"]["speaker_a"]["role"]
                for turn in conv["turns"]:
                    resp = await client.post("/respond", json={
                        "transcript_turn": turn["text"],
                        "role": role,
                        "empathy_slider": 50,
                    })
                    assert resp.status_code == 200, (
                        f"{conv['id']} turn failed: {resp.text}"
                    )
                    data = resp.json()

                    # suggestions: list with at least 1 item
                    assert isinstance(data["suggestions"], list)
                    assert len(data["suggestions"]) >= 1

                    # tone_score: dict with all 5 keys
                    assert isinstance(data["tone_score"], dict)
                    assert set(data["tone_score"].keys()) == TONE_SCORE_KEYS


# ---------------------------------------------------------------------------
# 2. Slider range: same turn at 0, 50, 100 should produce different prompts
# ---------------------------------------------------------------------------

class TestSliderRange:
    """Send the same turn at slider=0, 50, 100 and assert suggestions differ."""

    @pytest.mark.anyio
    async def test_slider_produces_different_suggestions(self, client):
        turn_text = "You never listen to me!"

        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.side_effect = _slider_side_effect(SLIDER_MOCKS)
            mock_get.return_value = mock_client

            results = {}
            for slider_val in (0, 50, 100):
                resp = await client.post("/respond", json={
                    "transcript_turn": turn_text,
                    "role": "Husband",
                    "empathy_slider": slider_val,
                })
                assert resp.status_code == 200
                results[slider_val] = resp.json()["suggestions"]

        # Assertive (0) vs balanced (50) vs full empathy (100) should differ
        assert results[0] != results[50], "slider=0 and slider=50 should differ"
        assert results[50] != results[100], "slider=50 and slider=100 should differ"
        assert results[0] != results[100], "slider=0 and slider=100 should differ"

    @pytest.mark.anyio
    async def test_slider_system_prompt_varies(self, client):
        """Verify the system prompt changes with the slider value."""
        turn_text = "I feel ignored."

        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_RESPOND_JSON
            mock_get.return_value = mock_client

            prompts = {}
            for slider_val in (0, 50, 100):
                await client.post("/respond", json={
                    "transcript_turn": turn_text,
                    "role": "Husband",
                    "empathy_slider": slider_val,
                })
                call_kwargs = mock_client.complete.call_args
                prompts[slider_val] = call_kwargs.kwargs["system"]

        assert "assertive" in prompts[0].lower()
        assert "balanced" in prompts[50].lower()
        assert "fully empathetic" in prompts[100].lower()


# ---------------------------------------------------------------------------
# 3. POST /score on sample utterances
# ---------------------------------------------------------------------------

SCORE_UTTERANCES = [
    "I really appreciate you sharing that with me.",
    "That's not what I said. You always twist my words.",
    "The appointment is at 3 PM on Thursday.",
]


class TestScoreEndpoint:
    """Test /score returns all 5 tone dimensions in 0-100 range."""

    @pytest.mark.anyio
    @pytest.mark.parametrize("utterance", SCORE_UTTERANCES)
    async def test_score_returns_all_dimensions(self, client, utterance):
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_SCORE_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/score", json={"text": utterance})

        assert resp.status_code == 200
        data = resp.json()

        for dim in TONE_SCORE_KEYS:
            assert dim in data, f"Missing dimension: {dim}"
            assert isinstance(data[dim], int)
            assert 0 <= data[dim] <= 100, f"{dim} out of range: {data[dim]}"


# ---------------------------------------------------------------------------
# 4. Session lifecycle: create -> add turns -> retrieve -> assert count
# ---------------------------------------------------------------------------

class TestSessionLifecycle:
    """Create a session, verify it persists, and check turn count."""

    @pytest.mark.anyio
    async def test_create_and_retrieve_session(self, client, sample_transcripts):
        conv = sample_transcripts["conversations"][0]
        turns = [
            {"speaker": t["speaker"], "text": t["text"]}
            for t in conv["turns"]
        ]
        metadata = {"scenario": conv["scenario"], "id": conv["id"]}

        # Create session
        create_resp = await client.post("/session", json={
            "turns": turns,
            "metadata": metadata,
        })
        assert create_resp.status_code == 201
        session = create_resp.json()
        assert "id" in session
        assert "created_at" in session
        assert len(session["turns"]) == len(conv["turns"])

        # Retrieve and verify
        get_resp = await client.get(f"/session/{session['id']}")
        assert get_resp.status_code == 200
        fetched = get_resp.json()
        assert fetched["id"] == session["id"]
        assert len(fetched["turns"]) == len(conv["turns"])
        assert fetched["metadata"] == metadata

    @pytest.mark.anyio
    async def test_session_turn_count_matches_all_conversations(
        self, client, sample_transcripts
    ):
        """Create a session for each conversation and verify turn counts."""
        for conv in sample_transcripts["conversations"]:
            turns = [
                {"speaker": t["speaker"], "text": t["text"]}
                for t in conv["turns"]
            ]

            create_resp = await client.post("/session", json={
                "turns": turns,
                "metadata": {"id": conv["id"]},
            })
            assert create_resp.status_code == 201
            session_id = create_resp.json()["id"]

            get_resp = await client.get(f"/session/{session_id}")
            assert get_resp.status_code == 200
            assert len(get_resp.json()["turns"]) == len(conv["turns"]), (
                f"{conv['id']}: turn count mismatch"
            )

    @pytest.mark.anyio
    async def test_session_not_found(self, client):
        resp = await client.get("/session/does-not-exist")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_incremental_session_building(self, client):
        """Simulate building a session turn by turn, then saving."""
        accumulated_turns = []
        turn_texts = [
            ("Alex", "You said you'd help."),
            ("Jordan", "I was going to get to it."),
            ("Alex", "That's what you always say."),
        ]

        for speaker, text in turn_texts:
            accumulated_turns.append({"speaker": speaker, "text": text})

        create_resp = await client.post("/session", json={
            "turns": accumulated_turns,
            "metadata": {"type": "incremental_test"},
        })
        assert create_resp.status_code == 201
        session = create_resp.json()
        assert len(session["turns"]) == len(turn_texts)

        get_resp = await client.get(f"/session/{session['id']}")
        assert get_resp.status_code == 200
        assert len(get_resp.json()["turns"]) == len(turn_texts)
