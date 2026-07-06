import asyncio
import json
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import main
from auth import get_current_uid
from main import app, empathy_system_prompt, get_db, init_db, parse_llm_json


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
    # P1-5: isolate each test's rate-limit window so cross-test call volume on
    # the shared client IP can never trip the limiter. The default 60/min limit
    # is left in place; one dedicated test lowers it to prove the 429 path.
    main._rate_limiter.reset()
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
# GET /healthz (P1-2)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"] is True
    assert isinstance(data["llm_key_present"], bool)
    assert isinstance(data["stt_provider"], str) and data["stt_provider"]


# ---------------------------------------------------------------------------
# X-Request-ID middleware (P1-3)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_request_id_echoed(client):
    resp = await client.get("/", headers={"X-Request-ID": "req-abc-123"})
    assert resp.headers["X-Request-ID"] == "req-abc-123"


@pytest.mark.anyio
async def test_request_id_generated_when_missing(client):
    resp = await client.get("/")
    # Generated IDs are uuid4 — parseable as a UUID.
    uuid.UUID(resp.headers["X-Request-ID"])


# ---------------------------------------------------------------------------
# init_db creates the relationship-session index (P1-6)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_init_db_creates_relationship_index(client):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_sessions_rel'"
        )
        row = await cursor.fetchone()
    finally:
        await db.close()
    assert row is not None


# ---------------------------------------------------------------------------
# Lifespan shutdown closes the LLM client (P1-8)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_lifespan_closes_llm_client():
    with patch("main.LLMClient") as mock_cls:
        async with app.router.lifespan_context(app):
            mock_cls.return_value.close.assert_not_called()
        mock_cls.return_value.close.assert_called_once()


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


async def _post_respond_with_llm_payload(client, payload: str):
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = payload
        mock_get.return_value = mock_client

        return await client.post("/respond", json={
            "transcript_turn": "Test",
            "role": "Husband",
            "empathy_slider": 50,
        })


@pytest.mark.anyio
async def test_respond_empty_suggestions_is_502(client):
    """P2-4: an empty suggestions list is an LLM failure, not a valid answer."""
    payload = json.dumps({"suggestions": [], "tone_score": {"warmth": 50}})
    resp = await _post_respond_with_llm_payload(client, payload)
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_respond_missing_suggestions_is_502(client):
    payload = json.dumps({"tone_score": {"warmth": 50}})
    resp = await _post_respond_with_llm_payload(client, payload)
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_respond_non_integer_tone_is_502(client):
    """P2-4: non-integer tone values must be a 502, not a 500 validation error."""
    payload = json.dumps({
        "suggestions": ["a", "b", "c"],
        "tone_score": {"warmth": "very warm"},
    })
    resp = await _post_respond_with_llm_payload(client, payload)
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_respond_empty_tone_score_is_502(client):
    """Review-fix: an empty tone_score dict must 502 (all() over {} is vacuously
    True), never yield a response the client will KeyError on."""
    payload = json.dumps({"suggestions": ["a", "b", "c"], "tone_score": {}})
    resp = await _post_respond_with_llm_payload(client, payload)
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_respond_accepts_whole_number_float_tone(client):
    """Review-fix: LLMs sometimes emit 82.0 for integer scores; a whole-number
    float must be accepted and coerced, not 502'd."""
    scores = {d: 50.0 for d in
              ("warmth", "defensiveness", "sarcasm", "constructiveness", "overall")}
    payload = json.dumps({"suggestions": ["a", "b", "c"], "tone_score": scores})
    resp = await _post_respond_with_llm_payload(client, payload)
    assert resp.status_code == 200
    assert all(isinstance(v, int) for v in resp.json()["tone_score"].values())


@pytest.mark.anyio
async def test_respond_none_llm_content_is_502(client):
    """Review-fix: a provider returning None content (content filter/refusal)
    must 502, not raise a raw 500 from None.strip()."""
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = None
        mock_get.return_value = mock_client
        resp = await client.post("/respond", json={
            "transcript_turn": "Test", "role": "Husband", "empathy_slider": 50,
        })
    assert resp.status_code == 502


@pytest.mark.anyio
async def test_score_accepts_whole_number_float(client):
    """Review-fix: whole-number float scores are coerced to int, not 502'd."""
    scores = {d: 42.0 for d in
              ("warmth", "defensiveness", "sarcasm", "constructiveness", "overall")}
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps(scores)
        mock_get.return_value = mock_client
        resp = await client.post("/score", json={"text": "hello"})
    assert resp.status_code == 200
    assert resp.json()["warmth"] == 42


@pytest.mark.anyio
async def test_respond_llm_runs_off_event_loop(client):
    """P0-1: the blocking LLM call must not run on the event-loop thread."""
    loop_thread = threading.current_thread()
    seen: dict = {}

    def record_thread(**kwargs):
        seen["thread"] = threading.current_thread()
        return MOCK_RESPOND_JSON

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.side_effect = record_thread
        mock_get.return_value = mock_client

        resp = await client.post("/respond", json={
            "transcript_turn": "Test",
            "role": "Husband",
            "empathy_slider": 50,
        })

    assert resp.status_code == 200
    assert seen["thread"] is not loop_thread


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


@pytest.mark.anyio
async def test_score_missing_dimension_is_502(client):
    """P2-4: a missing dimension must be a 502, never a fabricated 0."""
    incomplete = json.dumps({
        "warmth": 70, "defensiveness": 20, "sarcasm": 5, "constructiveness": 80,
        # "overall" missing
    })
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = incomplete
        mock_get.return_value = mock_client

        resp = await client.post("/score", json={"text": "Hello"})

    assert resp.status_code == 502
    assert "overall" in resp.json()["detail"]


@pytest.mark.anyio
async def test_score_non_integer_dimension_is_502(client):
    invalid = json.dumps({
        "warmth": "high", "defensiveness": 20, "sarcasm": 5,
        "constructiveness": 80, "overall": 75,
    })
    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = invalid
        mock_get.return_value = mock_client

        resp = await client.post("/score", json={"text": "Hello"})

    assert resp.status_code == 502
    assert "warmth" in resp.json()["detail"]


@pytest.mark.anyio
async def test_score_llm_runs_off_event_loop(client):
    """P0-1: the blocking LLM call must not run on the event-loop thread."""
    loop_thread = threading.current_thread()
    seen: dict = {}

    def record_thread(**kwargs):
        seen["thread"] = threading.current_thread()
        return MOCK_SCORE_JSON

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.side_effect = record_thread
        mock_get.return_value = mock_client

        resp = await client.post("/score", json={"text": "Hello"})

    assert resp.status_code == 200
    assert seen["thread"] is not loop_thread


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
    # Re-pinned for P2-7: the id must now be a well-formed UUID to reach the
    # handler (a malformed id is a 422 — see test_get_session_invalid_uuid_422).
    resp = await client.get(f"/session/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_session_invalid_uuid_422(client):
    """P2-7: a non-UUID session_id is rejected up front with 422."""
    resp = await client.get("/session/not-a-uuid")
    assert resp.status_code == 422


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
    # Re-pinned for P2-7: valid-UUID-but-absent → 404.
    resp = await client.post(f"/session/{uuid.uuid4()}/turns", json={
        "speaker": "Wife",
        "text": "Hello",
    })
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_add_turn_invalid_uuid_422(client):
    """P2-7: a non-UUID session_id is rejected with 422."""
    resp = await client.post("/session/not-a-uuid/turns", json={
        "speaker": "Wife",
        "text": "Hello",
    })
    assert resp.status_code == 422


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


@pytest.mark.anyio
async def test_add_turn_concurrent_no_lost_updates(client):
    """P0-4: N concurrent adds must persist all N turns (no lost updates)."""
    create_resp = await client.post("/session", json={"turns": [], "metadata": {}})
    sid = create_resp.json()["id"]

    n = 10
    responses = await asyncio.gather(*[
        client.post(f"/session/{sid}/turns", json={
            "speaker": "Speaker",
            "text": f"concurrent-turn-{i}",
        })
        for i in range(n)
    ])
    assert all(r.status_code == 201 for r in responses)

    get_resp = await client.get(f"/session/{sid}")
    turns = get_resp.json()["turns"]
    assert len(turns) == n
    assert {t["text"] for t in turns} == {f"concurrent-turn-{i}" for i in range(n)}


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
    # Re-pinned for P2-7: valid-UUID-but-absent → 404.
    resp = await client.get(f"/session/{uuid.uuid4()}/export")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_export_invalid_uuid_422(client):
    """P2-7: a non-UUID session_id is rejected with 422 (also closes the
    Content-Disposition filename header-injection surface)."""
    resp = await client.get("/session/not-a-uuid/export")
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_export_insights_failure_still_delivers_transcript(client):
    """P2-5: an LLM failure degrades insights honestly, keeps the transcript."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {"therapist": "Dr. Smith"},
    })
    sid = create_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.side_effect = RuntimeError("API down")
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export")

    assert resp.status_code == 200
    body = resp.text
    assert "Insights unavailable" in body
    # The raw provider error must NOT leak into the user-facing document
    # (it can carry request URLs/IDs); it belongs in the logs only.
    assert "API down" not in body
    # Transcript still fully delivered
    assert "You forgot again." in body
    assert "AGGREGATE STATISTICS" in body


@pytest.mark.anyio
async def test_export_corrupt_session_data_is_explicit_500(client):
    """P2-5: corrupt stored JSON yields an explicit 500, not a raw traceback."""
    # Re-pinned for P2-7: sid must be a valid UUID to reach the handler (the
    # old "corrupt-<uuid>" prefix would now be a 422, not the 500 under test).
    sid = str(uuid.uuid4())
    db = await get_db()
    try:
        # Stamp the owning uid so the (now uid-scoped) export reaches the
        # corrupt-JSON branch instead of 404-ing on ownership. "test-user" is
        # the default uid supplied by the conftest dependency override.
        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (sid, "2026-01-01T00:00:00+00:00", "{not valid json", "{}", "test-user"),
        )
        await db.commit()
    finally:
        await db.close()

    with patch("main.get_llm_client") as mock_get:
        mock_get.return_value = MagicMock()
        resp = await client.get(f"/session/{sid}/export")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "corrupt session data"


@pytest.mark.anyio
async def test_export_llm_runs_off_event_loop(client):
    """P0-1: the insights LLM call must not run on the event-loop thread."""
    create_resp = await client.post("/session", json={
        "turns": SAMPLE_TURNS,
        "metadata": {},
    })
    sid = create_resp.json()["id"]

    loop_thread = threading.current_thread()
    seen: dict = {}

    def record_thread(**kwargs):
        seen["thread"] = threading.current_thread()
        return MOCK_INSIGHTS

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.side_effect = record_thread
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export")

    assert resp.status_code == 200
    assert seen["thread"] is not loop_thread


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


@pytest.mark.anyio
async def test_export_pdf_escapes_markup(client):
    """P1-3: transcript text with reportlab mini-HTML metacharacters (`<`, `&`,
    a stray `</font>`) must not break Paragraph parsing (500) or inject styling
    — the export still yields valid PDF bytes."""
    hostile_turns = [
        {"speaker": "<b>Wife</b>", "text": "You said <font color='red'>this</font> & that </b>!",
         "score": {"warmth": 20, "defensiveness": 60, "sarcasm": 30,
                   "constructiveness": 25, "overall": 35}},
        {"speaker": "Husband & Co", "text": "1 < 2 && 3 > 2, right? <not-a-tag>",
         "score": None},
    ]
    create_resp = await client.post("/session", json={
        "turns": hostile_turns,
        "metadata": {"note": "<script>alert(1)</script>"},
    })
    sid = create_resp.json()["id"]

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = "Insights with <b>markup</b> & ampersand."
        mock_get.return_value = mock_client

        resp = await client.get(f"/session/{sid}/export?format=pdf")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:5] == b"%PDF-"


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
    # Re-pinned for P2-7: valid-UUID-but-absent → 404.
    resp = await client.get(f"/relationships/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_relationship_invalid_uuid_422(client):
    """P2-7: a non-UUID relationship_id is rejected with 422."""
    resp = await client.get("/relationships/not-a-uuid")
    assert resp.status_code == 422


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


# ---------------------------------------------------------------------------
# P1-5: per-IP rate limiting on cost-bearing endpoints
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_rate_limit_triggers_429(client, monkeypatch):
    """With a tiny limit, requests beyond the budget get an honest 429. The
    monkeypatched limit auto-restores, so the generous default is untouched
    for every other test."""
    monkeypatch.setattr(main._rate_limiter, "limit", 2)
    main._rate_limiter.reset()

    with patch("main.get_llm_client") as mock_get:
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_SCORE_JSON
        mock_get.return_value = mock_client

        r1 = await client.post("/score", json={"text": "one"})
        r2 = await client.post("/score", json={"text": "two"})
        r3 = await client.post("/score", json={"text": "three"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert "rate limit" in r3.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Voice profile ("speaks in your actual voice")
# ---------------------------------------------------------------------------

SAMPLE_PAIRS = [
    {
        "suggestion": "I understand you're frustrated, and I want to work through this.",
        "rephrase": "Okay — I get it, let's just figure it out.",
    },
    {
        "suggestion": "I feel hurt when plans change last minute.",
        "rephrase": "Yeah, springing that on me kinda sucks, ngl.",
    },
]


async def _make_relationship(client, pid: str = "alex") -> str:
    """Create a couple relationship containing participant ``pid``; return its id."""
    resp = await client.post("/relationships", json={
        "type": "couple",
        "name": "Test Marriage",
        "participants": [
            {"id": pid, "role": "husband", "display_name": "Alex"},
            {"id": "jordan", "role": "wife", "display_name": "Jordan"},
        ],
    })
    assert resp.status_code == 201
    return resp.json()["id"]


class TestVoiceProfileRendering:
    """The prompt-injection contract, in isolation (no DB / HTTP)."""

    def test_none_renders_blank(self):
        assert main._render_voice_profile(None) == ""

    def test_empty_profile_renders_blank(self):
        assert main._render_voice_profile({"pairs": [], "style_notes": None}) == ""

    def test_prompt_byte_identical_without_profile(self):
        """The load-bearing safety property: no profile → today's exact string.
        None AND a stored-but-empty profile must both leave the prompt unchanged
        across every slider band and role."""
        for slider in (0, 20, 21, 50, 51, 80, 81, 100):
            for role in ("Husband", "Wife", "Therapist"):
                base = empathy_system_prompt(slider, role)
                assert empathy_system_prompt(slider, role, None) == base
                assert empathy_system_prompt(
                    slider, role, {"pairs": [], "style_notes": None},
                ) == base

    def test_pairs_injected_after_output_contract(self):
        base = empathy_system_prompt(50, "Husband")
        prompt = empathy_system_prompt(
            50, "Husband", {"pairs": SAMPLE_PAIRS, "style_notes": "short, dry"},
        )
        # The base (with its JSON output contract) stays first and intact.
        assert prompt.startswith(base)
        assert len(prompt) > len(base)
        # The few-shot pairs and style notes are rendered verbatim.
        assert "Okay — I get it, let's just figure it out." in prompt
        assert "springing that on me kinda sucks" in prompt
        assert "Style notes: short, dry" in prompt
        assert "Match that voice" in prompt

    def test_render_caps_pairs_at_max(self):
        many = [
            {"suggestion": f"s{i}", "rephrase": f"r{i}"}
            for i in range(main.MAX_PAIRS + 3)
        ]
        rendered = main._render_voice_profile({"pairs": many, "style_notes": None})
        assert rendered.count("Generic:") == main.MAX_PAIRS


@pytest.mark.anyio
async def test_voice_profile_get_empty_when_unset(client):
    rel_id = await _make_relationship(client)
    resp = await client.get(
        f"/relationships/{rel_id}/participants/alex/voice-profile"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"pairs": [], "style_notes": None, "updated_at": None}


@pytest.mark.anyio
async def test_voice_profile_put_then_get_roundtrip(client):
    rel_id = await _make_relationship(client)
    put = await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": SAMPLE_PAIRS, "style_notes": "short, dry, no exclamation marks"},
    )
    assert put.status_code == 200
    assert put.json()["updated_at"] is not None

    got = await client.get(
        f"/relationships/{rel_id}/participants/alex/voice-profile"
    )
    assert got.status_code == 200
    body = got.json()
    assert body["pairs"] == SAMPLE_PAIRS
    assert body["style_notes"] == "short, dry, no exclamation marks"
    assert body["updated_at"] is not None


@pytest.mark.anyio
async def test_voice_profile_put_is_full_replace(client):
    rel_id = await _make_relationship(client)
    await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": SAMPLE_PAIRS, "style_notes": "first"},
    )
    replacement = [{"suggestion": "Sorry about that.", "rephrase": "my bad"}]
    await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": replacement, "style_notes": None},
    )
    got = await client.get(
        f"/relationships/{rel_id}/participants/alex/voice-profile"
    )
    assert got.json()["pairs"] == replacement
    assert got.json()["style_notes"] is None


@pytest.mark.anyio
async def test_voice_profile_put_caps_pairs_and_truncates(client):
    rel_id = await _make_relationship(client)
    many = [
        {"suggestion": f"suggestion {i}", "rephrase": f"rephrase {i}"}
        for i in range(main.MAX_PAIRS + 4)
    ]
    long_field = "x" * (main.MAX_PAIR_CHARS + 50)
    many.append({"suggestion": long_field, "rephrase": long_field})
    put = await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": many, "style_notes": "y" * (main.MAX_STYLE_NOTES_CHARS + 50)},
    )
    assert put.status_code == 200
    body = put.json()
    # Kept only the most recent MAX_PAIRS.
    assert len(body["pairs"]) == main.MAX_PAIRS
    assert body["pairs"][-1]["suggestion"] == long_field[:main.MAX_PAIR_CHARS]
    assert len(body["style_notes"]) == main.MAX_STYLE_NOTES_CHARS


@pytest.mark.anyio
async def test_voice_profile_rejects_empty_pair_fields(client):
    rel_id = await _make_relationship(client)
    resp = await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": [{"suggestion": "", "rephrase": "hi"}]},
    )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_voice_profile_unknown_participant_404(client):
    rel_id = await _make_relationship(client)
    get = await client.get(
        f"/relationships/{rel_id}/participants/nobody/voice-profile"
    )
    assert get.status_code == 404
    put = await client.put(
        f"/relationships/{rel_id}/participants/nobody/voice-profile",
        json={"pairs": SAMPLE_PAIRS},
    )
    assert put.status_code == 404


@pytest.mark.anyio
async def test_voice_profile_unknown_relationship_404(client):
    missing = str(uuid.uuid4())
    resp = await client.get(
        f"/relationships/{missing}/participants/alex/voice-profile"
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_respond_injects_voice_profile(client):
    """/respond looks up the profile from relationship + from_participant_id and
    threads it into the system prompt."""
    rel_id = await _make_relationship(client)
    await client.put(
        f"/relationships/{rel_id}/participants/alex/voice-profile",
        json={"pairs": SAMPLE_PAIRS, "style_notes": "short, dry"},
    )

    mock_client = MagicMock()
    mock_client.complete.return_value = MOCK_RESPOND_JSON
    with patch("main.get_llm_client", return_value=mock_client):
        resp = await client.post("/respond", json={
            "transcript_turn": "Whatever, do what you want.",
            "role": "Husband",
            "empathy_slider": 50,
            "relationship_id": rel_id,
            "from_participant_id": "alex",
        })
    assert resp.status_code == 200
    system = mock_client.complete.call_args.kwargs["system"]
    assert "Okay — I get it, let's just figure it out." in system
    assert "Style notes: short, dry" in system


@pytest.mark.anyio
async def test_respond_without_profile_prompt_unchanged(client):
    """A /respond with no relationship context sends today's exact prompt."""
    mock_client = MagicMock()
    mock_client.complete.return_value = MOCK_RESPOND_JSON
    with patch("main.get_llm_client", return_value=mock_client):
        resp = await client.post("/respond", json={
            "transcript_turn": "Whatever, do what you want.",
            "role": "Husband",
            "empathy_slider": 50,
        })
    assert resp.status_code == 200
    system = mock_client.complete.call_args.kwargs["system"]
    assert system == empathy_system_prompt(50, "Husband")


# ---------------------------------------------------------------------------
# Firebase auth — token required + strict per-user data scoping
# ---------------------------------------------------------------------------
# The conftest installs a dependency override for get_current_uid (reads the
# X-Test-Uid header, default "test-user") and a keyless fake verify_id_token
# (FAKE_TOKENS: "fake-id-token"→"test-user", "tok-user-a"→"user-a", ...). These
# tests either drive different users via X-Test-Uid, or drop the override to
# exercise the REAL dependency against the fake verifier — never real Firebase.


async def _make_couple(client, uid: str, pid: str = "alex") -> str:
    """Create a couple relationship owned by ``uid``; return its id."""
    resp = await client.post(
        "/relationships",
        json={
            "type": "couple",
            "name": "Owned Marriage",
            "participants": [
                {"id": pid, "role": "husband", "display_name": "Alex"},
                {"id": "jordan", "role": "wife", "display_name": "Jordan"},
            ],
        },
        headers={"X-Test-Uid": uid},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


class TestAuthTokenRequired:
    """With the real dependency in force, a valid Bearer token is required."""

    @pytest.mark.anyio
    async def test_missing_token_is_401(self, client, monkeypatch):
        monkeypatch.delitem(app.dependency_overrides, get_current_uid)
        resp = await client.post("/session", json={"turns": [], "metadata": {}})
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_malformed_header_is_401(self, client, monkeypatch):
        monkeypatch.delitem(app.dependency_overrides, get_current_uid)
        resp = await client.post(
            "/session", json={"turns": [], "metadata": {}},
            headers={"Authorization": "Token abc"},  # not "Bearer ..."
        )
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_invalid_token_is_401(self, client, monkeypatch):
        monkeypatch.delitem(app.dependency_overrides, get_current_uid)
        resp = await client.post(
            "/session", json={"turns": [], "metadata": {}},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401

    @pytest.mark.anyio
    async def test_valid_token_accepted_and_scopes_to_uid(self, client, monkeypatch):
        """A verified token creates a session owned by that token's uid, which
        the same token can then read back."""
        monkeypatch.delitem(app.dependency_overrides, get_current_uid)
        auth = {"Authorization": "Bearer fake-id-token"}  # → uid "test-user"
        created = await client.post(
            "/session", json={"turns": [], "metadata": {}}, headers=auth,
        )
        assert created.status_code == 201
        sid = created.json()["id"]
        got = await client.get(f"/session/{sid}", headers=auth)
        assert got.status_code == 200


class TestCrossUserIsolation:
    """The critical security property: one user can never see or mutate
    another user's data. Foreign rows return 404 (not 403) — existence is
    never confirmed."""

    @pytest.mark.anyio
    async def test_cannot_read_or_mutate_another_users_session(self, client):
        created = await client.post(
            "/session", json={"turns": [], "metadata": {}},
            headers={"X-Test-Uid": "user-a"},
        )
        sid = created.json()["id"]
        # Owner sees it.
        assert (await client.get(
            f"/session/{sid}", headers={"X-Test-Uid": "user-a"},
        )).status_code == 200
        # Another user cannot read, append to, or export it.
        other = {"X-Test-Uid": "user-b"}
        assert (await client.get(f"/session/{sid}", headers=other)).status_code == 404
        assert (await client.post(
            f"/session/{sid}/turns", json={"speaker": "x", "text": "y"},
            headers=other,
        )).status_code == 404
        assert (await client.get(
            f"/session/{sid}/export", headers=other,
        )).status_code == 404

    @pytest.mark.anyio
    async def test_cannot_reach_another_users_relationship(self, client):
        rel_id = await _make_couple(client, "user-a")
        other = {"X-Test-Uid": "user-b"}
        # Every relationship-scoped read is 404 for a non-owner.
        for path in (
            f"/relationships/{rel_id}",
            f"/relationships/{rel_id}/edges",
            f"/relationships/{rel_id}/sessions",
            f"/relationships/{rel_id}/participant/alex/sessions",
            f"/relationships/{rel_id}/participants/alex/voice-profile",
        ):
            assert (await client.get(path, headers=other)).status_code == 404, path
        # ...and every write is 404 too (no orphan rows under another owner).
        assert (await client.put(
            f"/relationships/{rel_id}/participants/alex/voice-profile",
            json={"pairs": SAMPLE_PAIRS}, headers=other,
        )).status_code == 404
        assert (await client.post(
            f"/relationships/{rel_id}/sessions",
            json={"from_participant_id": "alex", "to_participant_id": "jordan",
                  "empathy_slider": 50},
            headers=other,
        )).status_code == 404
        # The owner is unaffected.
        assert (await client.get(
            f"/relationships/{rel_id}", headers={"X-Test-Uid": "user-a"},
        )).status_code == 200

    @pytest.mark.anyio
    async def test_voice_profile_is_not_leaked_across_users(self, client):
        """A voice profile stored by one user must never surface in another
        user's coaching prompt (it is scoped through the owning relationship)."""
        rel_id = await _make_couple(client, "user-a")
        await client.put(
            f"/relationships/{rel_id}/participants/alex/voice-profile",
            json={"pairs": SAMPLE_PAIRS, "style_notes": "short, dry"},
            headers={"X-Test-Uid": "user-a"},
        )
        # user-b references user-a's ids in /respond — must NOT pick up the
        # profile (prompt stays the default) and must NOT error.
        mock_client = MagicMock()
        mock_client.complete.return_value = MOCK_RESPOND_JSON
        with patch("main.get_llm_client", return_value=mock_client):
            resp = await client.post(
                "/respond",
                json={
                    "transcript_turn": "Whatever.",
                    "role": "Husband",
                    "empathy_slider": 50,
                    "relationship_id": rel_id,
                    "from_participant_id": "alex",
                },
                headers={"X-Test-Uid": "user-b"},
            )
        assert resp.status_code == 200
        system = mock_client.complete.call_args.kwargs["system"]
        assert system == empathy_system_prompt(50, "Husband")
        assert "short, dry" not in system

    @pytest.mark.anyio
    async def test_relationship_sessions_list_scoped_by_user(self, client):
        """Two users' relationships and sessions stay fully partitioned."""
        rel_a = await _make_couple(client, "user-a")
        await client.post(
            f"/relationships/{rel_a}/sessions",
            json={"from_participant_id": "alex", "to_participant_id": "jordan",
                  "empathy_slider": 50},
            headers={"X-Test-Uid": "user-a"},
        )
        # user-a sees their one session; user-b sees a 404 on the foreign rel.
        a_list = await client.get(
            f"/relationships/{rel_a}/sessions", headers={"X-Test-Uid": "user-a"},
        )
        assert a_list.status_code == 200 and len(a_list.json()) == 1
        b_list = await client.get(
            f"/relationships/{rel_a}/sessions", headers={"X-Test-Uid": "user-b"},
        )
        assert b_list.status_code == 404
