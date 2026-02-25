import os
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app  # noqa: E402 — must set env before import


MOCK_RESPOND_JSON = json.dumps({
    "suggestions": [
        "I hear what you're saying.",
        "That sounds really frustrating.",
        "Can you tell me more about how that made you feel?",
    ],
    "tone_score": {
        "warmth": 60,
        "defensiveness": 30,
        "sarcasm": 10,
        "constructiveness": 55,
        "overall": 65,
    },
})

MOCK_SCORE_JSON = json.dumps({
    "warmth": 70,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 80,
    "overall": 75,
})


@pytest.fixture
def mock_llm():
    """Patch the LLMClient so no real API calls are made."""
    mock_client = MagicMock()
    with patch("main.get_llm_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_respond(mock_llm):
    mock_llm.complete.return_value = MOCK_RESPOND_JSON
    return mock_llm


@pytest.fixture
def mock_score(mock_llm):
    mock_llm.complete.return_value = MOCK_SCORE_JSON
    return mock_llm


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
