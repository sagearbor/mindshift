import asyncio
import os
import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Header
from httpx import ASGITransport, AsyncClient

# Use a temp database for each test session
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["MINDSHIFT_DB_PATH"] = _tmp.name
_tmp.close()

from main import app, init_db  # noqa: E402 — must set env before import
from auth import get_current_uid  # noqa: E402


# ---------------------------------------------------------------------------
# Auth test harness — keyless, never touches real Firebase
# ---------------------------------------------------------------------------
# Fake Firebase tokens → uids. The default token/uid keep the REST + WS tests
# green once they present it; the two extra users drive cross-user isolation.
FAKE_TOKENS = {
    "fake-id-token": "test-user",
    "tok-user-a": "user-a",
    "tok-user-b": "user-b",
}
DEFAULT_TEST_UID = "test-user"


def _test_uid_override(x_test_uid: str = Header(default=DEFAULT_TEST_UID)) -> str:
    """Stand-in for :func:`auth.get_current_uid`: returns the ``X-Test-Uid``
    header (or the default), so the REST suite runs authenticated WITHOUT real
    Firebase. Cross-user tests send a different ``X-Test-Uid`` to act as another
    user; the 401/valid-token tests remove this override to exercise the real
    dependency against the fake verifier below.
    """
    return x_test_uid


# Installed once on the shared app: every TestClient/AsyncClient built from
# ``main.app`` (this conftest, tests/conftest, or any test module) inherits it.
app.dependency_overrides[get_current_uid] = _test_uid_override


@pytest.fixture(autouse=True)
def _server_test_auth(monkeypatch):
    """Per-test auth harness for the server suite.

    * Ensures the DB schema exists — the WS auth handshake checks session
      ownership in the ``sessions`` table, so it must be present even for the
      WS tests that never build the ``client`` fixture.
    * Replaces ``auth.verify_id_token`` with a keyless fake used by the WS
      handshake and by the REST tests that drop the dependency override to hit
      the real :func:`auth.get_current_uid`.
    """
    asyncio.run(init_db())

    import auth

    def _verify(token: str) -> str:
        try:
            return FAKE_TOKENS[token]
        except KeyError:
            raise ValueError("invalid test token")

    monkeypatch.setattr(auth, "verify_id_token", _verify)


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
