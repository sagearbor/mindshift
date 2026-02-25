"""
Frontend contract validation tests.

Verifies that the FastAPI backend response shapes match what the
React Native frontend (SessionScreen, SuggestionCard, client.ts) expects.
Catches schema drift between backend and frontend early.
"""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import (
    MOCK_RESPOND_JSON,
    MOCK_SCORE_JSON,
    TONE_SCORE_KEYS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. /respond response matches frontend RespondResponse + Suggestion shape
# ---------------------------------------------------------------------------

class TestRespondContract:
    """The /respond endpoint must return data matching client.ts types."""

    @pytest.mark.anyio
    async def test_respond_returns_suggestions_list(self, client):
        """Frontend expects { suggestions: Suggestion[] } from /respond."""
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_RESPOND_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/respond", json={
                "transcript_turn": "You never listen to me.",
                "role": "Husband",
                "empathy_slider": 50,
            })

        data = resp.json()
        assert "suggestions" in data, "Response must contain 'suggestions'"
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) >= 1

    @pytest.mark.anyio
    async def test_respond_returns_tone_score(self, client):
        """Frontend displays tone info — tone_score must have all 5 keys."""
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_RESPOND_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/respond", json={
                "transcript_turn": "I feel ignored.",
                "role": "Wife",
                "empathy_slider": 75,
            })

        data = resp.json()
        assert "tone_score" in data
        assert set(data["tone_score"].keys()) == TONE_SCORE_KEYS

    @pytest.mark.anyio
    async def test_suggestion_card_prop_shape(self, client):
        """
        SuggestionCard expects { text: string, tone: string }.
        Backend /respond returns suggestions as plain strings.

        This documents the current contract: the frontend store
        (sessionStore.ts) maps backend strings into Suggestion objects.
        The backend's raw suggestion items must be strings.
        """
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_RESPOND_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/respond", json={
                "transcript_turn": "Test",
                "role": "Husband",
                "empathy_slider": 50,
            })

        suggestions = resp.json()["suggestions"]
        for s in suggestions:
            # Backend returns plain strings; frontend maps them to {text, tone}
            assert isinstance(s, str), (
                f"Backend suggestion should be a string, got {type(s)}"
            )


# ---------------------------------------------------------------------------
# 2. EXPO_PUBLIC_API_URL env var handling
# ---------------------------------------------------------------------------

class TestApiUrlConfig:
    """Validate that the frontend API client configuration is correct."""

    def test_client_ts_uses_expo_public_api_url(self):
        """client.ts must reference EXPO_PUBLIC_API_URL for the base URL."""
        client_path = PROJECT_ROOT / "apps" / "mobile" / "src" / "api" / "client.ts"
        content = client_path.read_text()

        assert "EXPO_PUBLIC_API_URL" in content, (
            "client.ts must use EXPO_PUBLIC_API_URL env var"
        )

    def test_client_ts_has_localhost_fallback(self):
        """Default fallback should be localhost:8000."""
        client_path = PROJECT_ROOT / "apps" / "mobile" / "src" / "api" / "client.ts"
        content = client_path.read_text()

        assert "localhost:8000" in content, (
            "client.ts should fall back to localhost:8000"
        )

    def test_client_ts_posts_to_respond(self):
        """client.ts must POST to /respond endpoint."""
        client_path = PROJECT_ROOT / "apps" / "mobile" / "src" / "api" / "client.ts"
        content = client_path.read_text()

        assert "/respond" in content, "client.ts must call /respond endpoint"
        assert "POST" in content, "client.ts must use POST method"


# ---------------------------------------------------------------------------
# 3. SuggestionCard prop interface validation
# ---------------------------------------------------------------------------

class TestSuggestionCardContract:
    """Verify SuggestionCard.tsx defines the expected Suggestion interface."""

    def test_suggestion_interface_has_text_and_tone(self):
        """Suggestion type must have 'text' and 'tone' string fields."""
        card_path = (
            PROJECT_ROOT / "apps" / "mobile" / "src" / "components" / "SuggestionCard.tsx"
        )
        content = card_path.read_text()

        # Check the exported interface shape
        assert "text: string" in content, (
            "Suggestion interface must have 'text: string'"
        )
        assert "tone: string" in content, (
            "Suggestion interface must have 'tone: string'"
        )

    def test_suggestion_card_renders_text_and_tone(self):
        """SuggestionCard must use both text and tone props."""
        card_path = (
            PROJECT_ROOT / "apps" / "mobile" / "src" / "components" / "SuggestionCard.tsx"
        )
        content = card_path.read_text()

        assert "{ text, tone }" in content or "{text, tone}" in content, (
            "SuggestionCard must destructure text and tone props"
        )

    def test_suggestion_card_has_test_id(self):
        """SuggestionCard must have testID for testing."""
        card_path = (
            PROJECT_ROOT / "apps" / "mobile" / "src" / "components" / "SuggestionCard.tsx"
        )
        content = card_path.read_text()

        assert 'testID="suggestion-card"' in content, (
            "SuggestionCard must have testID for testing"
        )


# ---------------------------------------------------------------------------
# 4. Session endpoint contract
# ---------------------------------------------------------------------------

class TestSessionContract:
    """Session endpoints must return shapes the frontend can consume."""

    @pytest.mark.anyio
    async def test_session_create_returns_required_fields(self, client):
        """POST /session must return id, created_at, turns, metadata."""
        resp = await client.post("/session", json={
            "turns": [{"speaker": "Alex", "text": "Hello"}],
            "metadata": {},
        })
        assert resp.status_code == 201
        data = resp.json()

        required_fields = {"id", "created_at", "turns", "metadata"}
        assert required_fields.issubset(set(data.keys())), (
            f"Missing fields: {required_fields - set(data.keys())}"
        )

    @pytest.mark.anyio
    async def test_session_get_returns_same_shape(self, client):
        """GET /session/{id} must return the same shape as POST /session."""
        create_resp = await client.post("/session", json={
            "turns": [{"speaker": "Jordan", "text": "Hi there"}],
            "metadata": {"test": True},
        })
        session_id = create_resp.json()["id"]

        get_resp = await client.get(f"/session/{session_id}")
        assert get_resp.status_code == 200

        create_keys = set(create_resp.json().keys())
        get_keys = set(get_resp.json().keys())
        assert create_keys == get_keys, (
            f"Shape mismatch: POST has {create_keys}, GET has {get_keys}"
        )


# ---------------------------------------------------------------------------
# 5. Score endpoint contract
# ---------------------------------------------------------------------------

class TestScoreContract:
    """POST /score must return all tone dimensions the frontend needs."""

    @pytest.mark.anyio
    async def test_score_returns_all_dimensions(self, client):
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_SCORE_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/score", json={
                "text": "I appreciate you."
            })

        data = resp.json()
        for dim in TONE_SCORE_KEYS:
            assert dim in data, f"Score response missing '{dim}'"
            assert isinstance(data[dim], int), f"'{dim}' must be an int"

    @pytest.mark.anyio
    async def test_score_dimensions_match_prd(self, client):
        """Score dimensions must include the 5 PRD-defined dimensions."""
        with patch("main.get_llm_client") as mock_get:
            mock_client = MagicMock()
            mock_client.complete.return_value = MOCK_SCORE_JSON
            mock_get.return_value = mock_client

            resp = await client.post("/score", json={"text": "Test utterance"})

        data = resp.json()
        # PRD defines: warmth, defensiveness, sarcasm, constructiveness, overall
        prd_dimensions = {"warmth", "defensiveness", "sarcasm", "constructiveness", "overall"}
        assert prd_dimensions.issubset(set(data.keys())), (
            f"Missing PRD dimensions: {prd_dimensions - set(data.keys())}"
        )
