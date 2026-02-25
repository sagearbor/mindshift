import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from main import app, empathy_system_prompt, init_db, parse_llm_json


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
# Relationship graph: couple topology
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_couple_relationship_lifecycle(client):
    """Create couple relationship, list edges (2 directed edges), start session on edge."""
    resp = await client.post("/relationships", json={
        "type": "couple",
        "name": "Smith Marriage",
        "participants": [
            {"id": "alex", "role": "husband", "display_name": "Alex Smith"},
            {"id": "jordan", "role": "wife", "display_name": "Jordan Smith"},
        ],
    })
    assert resp.status_code == 201
    rel = resp.json()
    rel_id = rel["id"]
    assert rel["type"] == "couple"
    assert rel["name"] == "Smith Marriage"
    assert len(rel["participants"]) == 2

    # GET relationship
    get_resp = await client.get(f"/relationships/{rel_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Smith Marriage"

    # List edges — couple has 2 directed edges
    edges_resp = await client.get(f"/relationships/{rel_id}/edges")
    assert edges_resp.status_code == 200
    edges = edges_resp.json()
    assert len(edges) == 2
    contexts = {e["context"] for e in edges}
    assert contexts == {"partner_to_partner"}

    # Start session on alex → jordan edge
    session_resp = await client.post(f"/relationships/{rel_id}/sessions", json={
        "from_participant_id": "alex",
        "to_participant_id": "jordan",
        "empathy_slider": 65,
    })
    assert session_resp.status_code == 201
    session = session_resp.json()
    assert session["relationship_id"] == rel_id
    assert session["from_participant_id"] == "alex"
    assert session["to_participant_id"] == "jordan"
    assert session["edge_context"] == "partner_to_partner"
    assert session["empathy_slider"] == 65

    # List sessions for relationship
    list_resp = await client.get(f"/relationships/{rel_id}/sessions")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1


# ---------------------------------------------------------------------------
# Relationship graph: org topology
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_org_relationship_edges(client):
    """Create org (director + 2 managers + 4 reports), verify correct edges."""
    resp = await client.post("/relationships", json={
        "type": "org",
        "name": "Engineering Dept",
        "participants": [
            {"id": "dir", "role": "director", "display_name": "Director D", "parent_id": None},
            {"id": "mgr1", "role": "manager", "display_name": "Manager M1", "parent_id": "dir"},
            {"id": "mgr2", "role": "manager", "display_name": "Manager M2", "parent_id": "dir"},
            {"id": "eng1", "role": "engineer", "display_name": "Engineer E1", "parent_id": "mgr1"},
            {"id": "eng2", "role": "engineer", "display_name": "Engineer E2", "parent_id": "mgr1"},
            {"id": "eng3", "role": "engineer", "display_name": "Engineer E3", "parent_id": "mgr2"},
            {"id": "eng4", "role": "engineer", "display_name": "Engineer E4", "parent_id": "mgr2"},
        ],
    })
    assert resp.status_code == 201
    rel_id = resp.json()["id"]

    edges_resp = await client.get(f"/relationships/{rel_id}/edges")
    assert edges_resp.status_code == 200
    edges = edges_resp.json()

    # Expected edges:
    # dir <-> mgr1, dir <-> mgr2 (4 directed manager_to_report/upward)
    # mgr1 <-> eng1, mgr1 <-> eng2 (4 directed)
    # mgr2 <-> eng3, mgr2 <-> eng4 (4 directed)
    # Peers under dir: mgr1 <-> mgr2 (2 directed)
    # Peers under mgr1: eng1 <-> eng2 (2 directed)
    # Peers under mgr2: eng3 <-> eng4 (2 directed)
    # Total = 4 + 4 + 4 + 2 + 2 + 2 = 18

    assert len(edges) == 18

    contexts = {e["context"] for e in edges}
    assert "manager_to_report" in contexts
    assert "upward" in contexts
    assert "peer" in contexts


# ---------------------------------------------------------------------------
# Relationship graph: coach_team topology
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_coach_team_relationship(client):
    """Create coach/team (1 coach + 5 players), start session coach→player."""
    participants = [
        {"id": "coach1", "role": "coach", "display_name": "Coach C"},
    ]
    for i in range(1, 6):
        participants.append(
            {"id": f"player{i}", "role": "player", "display_name": f"Player P{i}"}
        )

    resp = await client.post("/relationships", json={
        "type": "coach_team",
        "name": "Basketball Team",
        "participants": participants,
    })
    assert resp.status_code == 201
    rel_id = resp.json()["id"]

    edges_resp = await client.get(f"/relationships/{rel_id}/edges")
    assert edges_resp.status_code == 200
    edges = edges_resp.json()
    # 1 coach × 5 players × 2 directions = 10
    assert len(edges) == 10

    coach_to_player = [e for e in edges if e["context"] == "coach_to_player"]
    player_to_coach = [e for e in edges if e["context"] == "player_to_coach"]
    assert len(coach_to_player) == 5
    assert len(player_to_coach) == 5

    # Start session coach → player1
    session_resp = await client.post(f"/relationships/{rel_id}/sessions", json={
        "from_participant_id": "coach1",
        "to_participant_id": "player1",
        "empathy_slider": 50,
    })
    assert session_resp.status_code == 201
    assert session_resp.json()["edge_context"] == "coach_to_player"


# ---------------------------------------------------------------------------
# Relationship graph: participant sessions
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_participant_sessions(client):
    """Sessions involving a specific participant should be queryable."""
    resp = await client.post("/relationships", json={
        "type": "couple",
        "name": "Test Couple",
        "participants": [
            {"id": "p1", "role": "husband", "display_name": "Person 1"},
            {"id": "p2", "role": "wife", "display_name": "Person 2"},
        ],
    })
    rel_id = resp.json()["id"]

    # Create two sessions
    await client.post(f"/relationships/{rel_id}/sessions", json={
        "from_participant_id": "p1",
        "to_participant_id": "p2",
        "empathy_slider": 50,
    })
    await client.post(f"/relationships/{rel_id}/sessions", json={
        "from_participant_id": "p2",
        "to_participant_id": "p1",
        "empathy_slider": 75,
    })

    # p1 involved in both (as from or to)
    resp = await client.get(f"/relationships/{rel_id}/participant/p1/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2

    # p2 also involved in both
    resp = await client.get(f"/relationships/{rel_id}/participant/p2/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# /respond with relationship context enriches prompt
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_respond_with_relationship_context(client):
    """POST /respond with relationship_id should enrich the LLM prompt."""
    # Create a relationship first
    rel_resp = await client.post("/relationships", json={
        "type": "org",
        "name": "Team Alpha",
        "participants": [
            {"id": "mgr", "role": "manager", "display_name": "Manager M", "parent_id": None},
            {"id": "rpt", "role": "report", "display_name": "Report R", "parent_id": "mgr"},
        ],
    })
    rel_id = rel_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_RESPOND_JSON
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "I need more time on this project.",
            "role": "Manager",
            "empathy_slider": 50,
            "relationship_id": rel_id,
            "from_participant_id": "mgr",
            "to_participant_id": "rpt",
        })

    assert resp.status_code == 200
    # Verify that the LLM was called with relationship context in the user prompt
    call_kwargs = mock_client.complete.call_args.kwargs
    assert "Relationship context" in call_kwargs["user"]
    assert "Team Alpha" in call_kwargs["user"]
    assert "Manager M" in call_kwargs["user"]
    assert "Report R" in call_kwargs["user"]


# ---------------------------------------------------------------------------
# Relationship not found
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_get_relationship_not_found(client):
    resp = await client.get("/relationships/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_relationship_session_invalid_participant(client):
    """Creating session with participant not in relationship returns 400."""
    resp = await client.post("/relationships", json={
        "type": "couple",
        "name": "Test",
        "participants": [
            {"id": "a", "role": "husband", "display_name": "A"},
            {"id": "b", "role": "wife", "display_name": "B"},
        ],
    })
    rel_id = resp.json()["id"]

    resp = await client.post(f"/relationships/{rel_id}/sessions", json={
        "from_participant_id": "a",
        "to_participant_id": "UNKNOWN",
        "empathy_slider": 50,
    })
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Parent-child topology
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_parent_child_edges(client):
    """Parent-child relationship should generate parent_to_child and child_to_parent edges."""
    resp = await client.post("/relationships", json={
        "type": "parent_child",
        "name": "Family",
        "participants": [
            {"id": "parent1", "role": "parent", "display_name": "Mom", "parent_id": None},
            {"id": "child1", "role": "child", "display_name": "Kid 1", "parent_id": "parent1"},
            {"id": "child2", "role": "child", "display_name": "Kid 2", "parent_id": "parent1"},
        ],
    })
    assert resp.status_code == 201
    rel_id = resp.json()["id"]

    edges_resp = await client.get(f"/relationships/{rel_id}/edges")
    edges = edges_resp.json()
    # 1 parent × 2 children × 2 directions = 4
    assert len(edges) == 4
    contexts = {e["context"] for e in edges}
    assert "parent_to_child" in contexts
    assert "child_to_parent" in contexts
