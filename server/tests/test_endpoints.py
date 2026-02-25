import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, empathy_system_prompt, init_db, parse_llm_json, _resolve_session_token


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

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
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_RESPOND_JSON
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
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_RESPOND_JSON
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "That's not fair.",
            "role": "Wife",
            "empathy_slider": 10,
        })

    assert resp.status_code == 200
    call_kwargs = mock_client.complete.call_args
    assert "assertive" in call_kwargs.kwargs["system"].lower()


@pytest.mark.anyio
async def test_respond_slider_full_empathy(client):
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_RESPOND_JSON
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "I feel invisible.",
            "role": "Husband",
            "empathy_slider": 95,
        })

    assert resp.status_code == 200
    call_kwargs = mock_client.complete.call_args
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
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = "This is not JSON at all"
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
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_SCORE_JSON
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
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = "not json!"
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


# ---------------------------------------------------------------------------
# POST /session/{id}/turns — multi-turn sessions
# ---------------------------------------------------------------------------

SAMPLE_TURNS = [
    {"speaker": "Wife", "text": "You forgot again.",
     "score": {"warmth": 20, "defensiveness": 60, "sarcasm": 30, "constructiveness": 25, "overall": 35}},
    {"speaker": "Husband", "text": "I'm sorry, I'll set a reminder.",
     "score": {"warmth": 70, "defensiveness": 10, "sarcasm": 5, "constructiveness": 80, "overall": 75}},
]


@pytest.mark.anyio
async def test_add_turn_and_retrieve(client):
    """Create session, add a turn, verify it's appended."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {"therapist": "Dr. Lee"},
    })
    assert create_resp.status_code == 201
    sid = create_resp.json()["id"]

    new_turn = {
        "speaker": "Wife",
        "text": "Thank you for hearing me.",
        "score": {"warmth": 80, "defensiveness": 5, "sarcasm": 0, "constructiveness": 70, "overall": 80},
    }
    turn_resp = await client.post(f"/session/{sid}/turns", json=new_turn)
    assert turn_resp.status_code == 201
    turn_data = turn_resp.json()
    assert turn_data["session_id"] == sid
    assert turn_data["turn_index"] == 2
    assert turn_data["turn"]["speaker"] == "Wife"

    # Verify session now has 3 turns
    get_resp = await client.get(f"/session/{sid}")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["turns"]) == 3
    assert get_resp.json()["turns"][2]["text"] == "Thank you for hearing me."


@pytest.mark.anyio
async def test_add_turn_session_not_found(client):
    resp = await client.post("/session/nonexistent-id/turns", json={
        "speaker": "Wife",
        "text": "Hello",
    })
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_add_turn_no_score(client):
    """Turn without score should still be accepted (score is optional)."""
    create_resp = await client.post("/session", json={"turns": [], "metadata": {}})
    sid = create_resp.json()["id"]

    turn_resp = await client.post(f"/session/{sid}/turns", json={
        "speaker": "Husband",
        "text": "Let's talk later.",
    })
    assert turn_resp.status_code == 201
    assert turn_resp.json()["turn"]["score"] is None


# ---------------------------------------------------------------------------
# GET /session/{id}/export — text format
# ---------------------------------------------------------------------------

MOCK_INSIGHTS = "The session showed a pattern of defensiveness from the wife with constructive repair attempts from the husband."


@pytest.mark.anyio
async def test_export_text_structure(client):
    """Text export should contain metadata, transcript, stats, and insights."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {"therapist": "Dr. Smith", "date": "2026-02-25"},
    })
    sid = create_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_INSIGHTS
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"

    body = resp.text
    # Metadata section
    assert "SESSION METADATA" in body
    assert sid in body
    assert "Dr. Smith" in body

    # Transcript section
    assert "TRANSCRIPT" in body
    assert "You forgot again." in body
    assert "I'm sorry" in body
    assert "warmth=" in body

    # Aggregate stats
    assert "AGGREGATE STATISTICS" in body
    assert "warmth" in body.lower()

    # Insights
    assert "SESSION INSIGHTS" in body
    assert "defensiveness" in body

    # LLM was called once for insights
    mock_client.complete.assert_called_once()


@pytest.mark.anyio
async def test_export_text_format_param(client):
    """Explicit ?format=text should work identically."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {},
    })
    sid = create_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_INSIGHTS
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export?format=text")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


@pytest.mark.anyio
async def test_export_not_found(client):
    resp = await client.get("/session/nonexistent-id/export")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /session/{id}/export?format=pdf
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_export_pdf_returns_bytes(client):
    """PDF export should return application/pdf with valid PDF bytes."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {"therapist": "Dr. Lee"},
    })
    sid = create_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_INSIGHTS
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export?format=pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    # PDF files start with %PDF
    assert resp.content[:5] == b"%PDF-"
    # LLM was called for insights
    mock_client.complete.assert_called_once()


# ---------------------------------------------------------------------------
# POST /auth/session
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_create_auth_session(client):
    resp = await client.post("/auth/session", json={
        "therapist_id": "dr-smith",
        "patient_id": "patient-001",
        "role_pair": "Husband/Wife",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "session_token" in data
    assert data["therapist_id"] == "dr-smith"
    assert data["patient_id"] == "patient-001"
    assert data["role_pair"] == "Husband/Wife"
    assert "created_at" in data


@pytest.mark.anyio
async def test_create_auth_session_default_role_pair(client):
    resp = await client.post("/auth/session", json={
        "therapist_id": "dr-lee",
        "patient_id": "patient-002",
    })
    assert resp.status_code == 201
    assert resp.json()["role_pair"] == "Husband/Wife"


# ---------------------------------------------------------------------------
# X-Session-Token on /session
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_create_session_with_auth_token(client):
    """Session created with X-Session-Token should be linked to therapist/patient."""
    # Create auth session first
    auth_resp = await client.post("/auth/session", json={
        "therapist_id": "dr-smith",
        "patient_id": "patient-001",
    })
    token = auth_resp.json()["session_token"]

    # Create a data session using the auth token
    session_resp = await client.post(
        "/session",
        json={"turns": [{"speaker": "Wife", "text": "Hello"}], "metadata": {}},
        headers={"X-Session-Token": token},
    )
    assert session_resp.status_code == 201

    # Verify it shows up under the therapist's patient sessions
    list_resp = await client.get("/therapist/dr-smith/patient/patient-001/sessions")
    assert list_resp.status_code == 200
    sessions = list_resp.json()
    assert len(sessions) >= 1
    assert sessions[0]["id"] == session_resp.json()["id"]


@pytest.mark.anyio
async def test_create_session_without_token(client):
    """Session without X-Session-Token still works (no auth association)."""
    resp = await client.post("/session", json={
        "turns": [],
        "metadata": {},
    })
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# GET /therapist/{id}/patients
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_patients(client):
    # Create auth session + linked data sessions
    auth_resp = await client.post("/auth/session", json={
        "therapist_id": "dr-jones",
        "patient_id": "p-alpha",
    })
    token_a = auth_resp.json()["session_token"]

    auth_resp2 = await client.post("/auth/session", json={
        "therapist_id": "dr-jones",
        "patient_id": "p-beta",
    })
    token_b = auth_resp2.json()["session_token"]

    # Create sessions for each patient
    await client.post("/session", json={"turns": [], "metadata": {}},
                      headers={"X-Session-Token": token_a})
    await client.post("/session", json={"turns": [], "metadata": {}},
                      headers={"X-Session-Token": token_a})
    await client.post("/session", json={"turns": [], "metadata": {}},
                      headers={"X-Session-Token": token_b})

    resp = await client.get("/therapist/dr-jones/patients")
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) == 2

    by_id = {p["patient_id"]: p["session_count"] for p in patients}
    assert by_id["p-alpha"] == 2
    assert by_id["p-beta"] == 1


@pytest.mark.anyio
async def test_list_patients_empty(client):
    resp = await client.get("/therapist/nonexistent-therapist/patients")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /therapist/{id}/patient/{pid}/sessions
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_patient_sessions(client):
    auth_resp = await client.post("/auth/session", json={
        "therapist_id": "dr-patel",
        "patient_id": "p-gamma",
    })
    token = auth_resp.json()["session_token"]

    turns = [{"speaker": "Husband", "text": "I need to talk."}]
    await client.post("/session", json={"turns": turns, "metadata": {"note": "first"}},
                      headers={"X-Session-Token": token})

    resp = await client.get("/therapist/dr-patel/patient/p-gamma/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    assert sessions[0]["turn_count"] == 1
    assert sessions[0]["metadata"]["note"] == "first"


@pytest.mark.anyio
async def test_list_patient_sessions_empty(client):
    resp = await client.get("/therapist/dr-nobody/patient/p-nobody/sessions")
    assert resp.status_code == 200
    assert resp.json() == []
