import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, empathy_system_prompt, init_db, parse_llm_json


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_anthropic_response(content: str):
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

MOCK_SCORE_JSON = json.dumps({
    "warmth": 70,
    "defensiveness": 20,
    "sarcasm": 5,
    "constructiveness": 80,
    "overall": 75,
})


@pytest.fixture
async def client():
    await init_db()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_root(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert resp.json()["message"] == "MindShift API"


# ---------------------------------------------------------------------------
# POST /respond
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_respond_success(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            MOCK_RESPOND_JSON
        )
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "You never listen to me!",
            "role": "Husband",
            "empathy_slider": 75,
            "context": "Argument about chores",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["suggestions"]) == 3
    assert "tone_score" in data
    assert data["tone_score"]["warmth"] == 60


@pytest.mark.anyio
async def test_respond_slider_assertive(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            MOCK_RESPOND_JSON
        )
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "That's not fair.",
            "role": "Wife",
            "empathy_slider": 10,
        })

    assert resp.status_code == 200
    # Verify the system prompt used assertive language
    call_kwargs = mock_client.messages.create.call_args
    assert "assertive" in call_kwargs.kwargs["system"].lower()


@pytest.mark.anyio
async def test_respond_slider_full_empathy(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            MOCK_RESPOND_JSON
        )
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "I feel invisible.",
            "role": "Husband",
            "empathy_slider": 95,
        })

    assert resp.status_code == 200
    call_kwargs = mock_client.messages.create.call_args
    assert "fully empathetic" in call_kwargs.kwargs["system"].lower()


@pytest.mark.anyio
async def test_respond_invalid_slider(client):
    resp = await client.post("/respond", json={
        "transcript_turn": "Hello",
        "role": "Husband",
        "empathy_slider": 150,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_respond_missing_field(client):
    resp = await client.post("/respond", json={
        "role": "Husband",
        "empathy_slider": 50,
    })
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_respond_bad_llm_json(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "This is not JSON at all"
        )
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "Test",
            "role": "Husband",
            "empathy_slider": 50,
        })

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /score
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_score_success(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            MOCK_SCORE_JSON
        )
        mock_get.return_value = mock_client

        resp = await client.post("/score", json={
            "text": "I appreciate you taking the time to explain that.",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["warmth"] == 70
    assert data["defensiveness"] == 20
    assert data["sarcasm"] == 5
    assert data["constructiveness"] == 80
    assert data["overall"] == 75


@pytest.mark.anyio
async def test_score_bad_llm_json(client):
    with patch("main.get_anthropic_client") as mock_get:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_anthropic_response(
            "not json!"
        )
        mock_get.return_value = mock_client

        resp = await client.post("/score", json={"text": "Hello"})

    assert resp.status_code == 502


@pytest.mark.anyio
async def test_score_missing_text(client):
    resp = await client.post("/score", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /session + GET /session/{id}
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_create_and_get_session(client):
    turns = [
        {"speaker": "Wife", "text": "You forgot again.", "score": 40},
        {"speaker": "Husband", "text": "I'm sorry, I'll set a reminder.", "score": 75},
    ]
    metadata = {"therapist": "Dr. Smith", "date": "2026-02-25"}

    create_resp = await client.post("/session", json={
        "turns": turns,
        "metadata": metadata,
    })
    assert create_resp.status_code == 201
    session = create_resp.json()
    assert "id" in session
    assert session["turns"] == turns
    assert session["metadata"] == metadata

    get_resp = await client.get(f"/session/{session['id']}")
    assert get_resp.status_code == 200
    fetched = get_resp.json()
    assert fetched["id"] == session["id"]
    assert fetched["turns"] == turns
    assert fetched["metadata"] == metadata
    assert fetched["created_at"] == session["created_at"]


@pytest.mark.anyio
async def test_get_session_not_found(client):
    resp = await client.get("/session/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_create_session_empty_turns(client):
    resp = await client.post("/session", json={
        "turns": [],
        "metadata": {},
    })
    assert resp.status_code == 201
    assert resp.json()["turns"] == []


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestEmpathySystemPrompt:
    def test_assertive_range(self):
        prompt = empathy_system_prompt(0, "Husband")
        assert "assertive" in prompt.lower()
        prompt = empathy_system_prompt(20, "Wife")
        assert "assertive" in prompt.lower()

    def test_balanced_range(self):
        prompt = empathy_system_prompt(21, "Husband")
        assert "balanced" in prompt.lower()
        prompt = empathy_system_prompt(50, "Wife")
        assert "balanced" in prompt.lower()

    def test_empathetic_range(self):
        prompt = empathy_system_prompt(51, "Husband")
        assert "empathetic" in prompt.lower()
        prompt = empathy_system_prompt(80, "Wife")
        assert "empathetic" in prompt.lower()

    def test_full_empathy_range(self):
        prompt = empathy_system_prompt(81, "Husband")
        assert "fully empathetic" in prompt.lower()
        prompt = empathy_system_prompt(100, "Wife")
        assert "fully empathetic" in prompt.lower()

    def test_role_included(self):
        prompt = empathy_system_prompt(50, "Therapist")
        assert "Therapist" in prompt


class TestParseLlmJson:
    def test_plain_json(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        result = parse_llm_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_fenced_no_lang(self):
        result = parse_llm_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("not json")
