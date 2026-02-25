import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# Add server/ to sys.path so we can import main
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app, init_db  # noqa: E402


# ---------------------------------------------------------------------------
# Mock LLM helpers (reused from server/conftest.py pattern)
# ---------------------------------------------------------------------------

def _make_anthropic_response(content: str):
    """Build a mock Anthropic message response."""
    block = MagicMock()
    block.text = content
    msg = MagicMock()
    msg.content = [block]
    return msg


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

MOCK_ASSERTIVE_JSON = json.dumps({
    "suggestions": [
        "Set a clear boundary here.",
        "Be direct about your needs.",
        "State your position firmly.",
    ],
    "tone_score": {
        "warmth": 20,
        "defensiveness": 60,
        "sarcasm": 15,
        "constructiveness": 40,
        "overall": 35,
    },
})

MOCK_FULL_EMPATHY_JSON = json.dumps({
    "suggestions": [
        "That must be so hard for you.",
        "Your feelings are completely valid.",
        "I'm here for you no matter what.",
    ],
    "tone_score": {
        "warmth": 90,
        "defensiveness": 5,
        "sarcasm": 2,
        "constructiveness": 70,
        "overall": 85,
    },
})

MOCK_SCORE_JSON = json.dumps({
    "warmth": 70,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 80,
    "overall": 75,
})

TONE_SCORE_KEYS = {"warmth", "defensiveness", "sarcasm", "constructiveness", "overall"}


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def sample_transcripts():
    """Load the sample_transcripts.json fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_transcripts.json"
    with open(fixture_path) as f:
        return json.load(f)
