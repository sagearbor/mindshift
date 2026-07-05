import asyncio
import io
import logging
import os
import json
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# Load a repo-root `.env` (if present) BEFORE any configuration is read below.
# python-dotenv's default is override=False, so real environment variables
# always win over .env values. Defensive try/except: a missing python-dotenv
# must never break the server — .env support simply degrades to a no-op.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover — python-dotenv is in requirements
    pass
else:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import aiosqlite
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from audio_pipeline import audio_ws_endpoint
from llm_client import LLMClient
from models.relationship import (
    EdgeOut,
    Participant,
    RelationshipCreate,
    RelationshipOut,
    RelationshipSessionCreate,
    RelationshipSessionOut,
    RelationshipType,
)

logger = logging.getLogger(__name__)

# Default to an absolute repo-root path so every launch directory uses the
# same database; MINDSHIFT_DB_PATH still overrides.
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "mindshift.db"
DB_PATH = os.getenv("MINDSHIFT_DB_PATH") or str(_DEFAULT_DB_PATH)
MINDSHIFT_MODEL = os.getenv("MINDSHIFT_MODEL", "claude-3-haiku-20240307")

# API-key env var per provider — used for health/startup reporting only.
_PROVIDER_KEY_ENVS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def _detected_provider() -> str:
    """Best-effort provider name for the configured model ("unknown" if none)."""
    try:
        return LLMClient._detect_provider(MINDSHIFT_MODEL)
    except ValueError:
        return "unknown"


def _llm_key_present() -> bool:
    env_var = _PROVIDER_KEY_ENVS.get(_detected_provider())
    return bool(env_var and os.environ.get(env_var))


def _stt_provider() -> str:
    return (os.getenv("STT_PROVIDER") or "deepgram").strip().lower() or "deepgram"


# ---------------------------------------------------------------------------
# Logging + request correlation
# ---------------------------------------------------------------------------

_request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    """Stamp every record with the current request's ID ("-" outside HTTP)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


def _configure_logging() -> None:
    """Install a timestamped root handler — only if none exists yet.

    pytest and uvicorn install their own handlers; when they have, this is a
    no-op so their configuration survives untouched.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    ))
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        logger.warning("Invalid LOG_LEVEL=%r — defaulting to INFO", level)
        level = "INFO"
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RespondRequest(BaseModel):
    transcript_turn: str
    role: str
    empathy_slider: int = Field(ge=0, le=100)
    context: str = ""
    relationship_id: Optional[str] = None
    from_participant_id: Optional[str] = None
    to_participant_id: Optional[str] = None


class RespondResponse(BaseModel):
    suggestions: list[str]
    tone_score: dict[str, int]


class ScoreRequest(BaseModel):
    text: str
    relationship_id: Optional[str] = None
    from_participant_id: Optional[str] = None
    to_participant_id: Optional[str] = None


class ScoreResponse(BaseModel):
    warmth: int
    defensiveness: int
    sarcasm: int
    constructiveness: int
    overall: int


class SessionCreate(BaseModel):
    turns: list[dict]
    metadata: dict = {}


class SessionOut(BaseModel):
    id: str
    created_at: str
    turns: list[dict]
    metadata: dict


class ExportFormat(str, Enum):
    text = "text"
    pdf = "pdf"


class SessionTurn(BaseModel):
    speaker: str
    text: str
    score: dict | None = None


class TurnResponse(BaseModel):
    session_id: str
    turn_index: int
    turn: dict



# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    # Production tuning: WAL lets readers and a writer coexist; busy_timeout
    # waits out short write locks instead of failing immediately.
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA busy_timeout=5000")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute("PRAGMA synchronous=NORMAL")
    return db


async def init_db() -> None:
    db = await get_db()
    try:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                turns TEXT NOT NULL,
                metadata TEXT NOT NULL,
                relationship_id TEXT,
                from_participant_id TEXT,
                to_participant_id TEXT,
                edge_context TEXT,
                empathy_slider INTEGER
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                id TEXT NOT NULL,
                relationship_id TEXT NOT NULL,
                role TEXT NOT NULL,
                display_name TEXT NOT NULL,
                parent_id TEXT,
                PRIMARY KEY (id, relationship_id),
                FOREIGN KEY (relationship_id) REFERENCES relationships(id)
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_rel "
            "ON sessions(relationship_id)"
        )
        await db.commit()
    finally:
        await db.close()


def _configure_stt(app: FastAPI) -> None:
    """Install the transcriber factory selected by the ``STT_PROVIDER`` env var.

    * ``"deepgram"`` (the default, also used when unset) leaves
      ``app.state.transcriber_factory`` unset — the audio pipeline then falls
      back to :class:`~audio_pipeline.DeepgramTranscriber`, exactly as before.
    * ``"whisper"`` installs the free local
      :class:`~whisper_transcriber.WhisperTranscriber`. The heavy
      ``faster-whisper`` package is optional (``requirements-whisper.txt``)
      and is only imported when the first session connects; without it the
      pipeline honestly reports transcription as unavailable.
    * Anything else logs a warning and keeps the deepgram default.
    """
    provider = (os.getenv("STT_PROVIDER") or "deepgram").strip().lower() or "deepgram"
    if provider == "whisper":
        # WhisperTranscriber is zero-arg-callable, satisfying the pipeline's
        # injection contract (transcriber_factory()).
        from whisper_transcriber import WhisperTranscriber

        app.state.transcriber_factory = WhisperTranscriber
    elif provider != "deepgram":
        logger.warning(
            "Unknown STT_PROVIDER=%r — defaulting to deepgram", provider,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    await init_db()
    app.state.llm_client = LLMClient(model=MINDSHIFT_MODEL)
    _configure_stt(app)
    logger.info(
        "MindShift API started — model=%s provider=%s llm_key_present=%s "
        "stt_provider=%s db_path=%s",
        MINDSHIFT_MODEL,
        _detected_provider(),
        "yes" if _llm_key_present() else "no",
        _stt_provider(),
        Path(DB_PATH).resolve(),
    )
    yield
    app.state.llm_client.close()
    logger.info("MindShift API shut down — LLM client closed")


app = FastAPI(title="MindShift API", lifespan=lifespan)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Accept or mint an X-Request-ID and expose it to logs + the response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = _request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        _request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# WebSocket — real-time audio pipeline (M2)
# ---------------------------------------------------------------------------

@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    await audio_ws_endpoint(websocket, session_id)


# ---------------------------------------------------------------------------
# Empathy slider → prompt mapping
# ---------------------------------------------------------------------------

def empathy_system_prompt(slider: int, role: str) -> str:
    if slider <= 20:
        stance = (
            "You are an assertive communication coach. "
            "Help the user push back firmly, set clear boundaries, "
            "and challenge assumptions. Be direct and confident."
        )
    elif slider <= 50:
        stance = (
            "You are a balanced communication coach. "
            "Acknowledge the other person's feelings briefly, then redirect "
            "toward constructive solutions. Be fair but practical."
        )
    elif slider <= 80:
        stance = (
            "You are an empathetic communication coach. "
            "Validate the other person's emotions, reflect what they said, "
            "and minimize judgment. Prioritize understanding."
        )
    else:
        stance = (
            "You are a fully empathetic communication coach. "
            "Offer pure validation and emotional support. Do not redirect "
            "or challenge — only affirm and show deep understanding."
        )

    return (
        f"{stance}\n\n"
        f"The user's role in this conversation is: {role}.\n"
        "Provide exactly 3 short suggested responses the user could say next. "
        "Return ONLY a JSON object with key \"suggestions\" (a list of strings) "
        "and \"tone_score\" (an object with integer keys: warmth, defensiveness, "
        "sarcasm, constructiveness, overall — each 0-100, scoring the transcript turn)."
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    return app.state.llm_client


def parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "MindShift API"}


@app.get("/healthz")
async def healthz():
    """Liveness/readiness probe — checks the DB, never calls external APIs."""
    db_ok = False
    try:
        db = await get_db()
        try:
            await db.execute("SELECT 1")
            db_ok = True
        finally:
            await db.close()
    except Exception:  # noqa: BLE001 — a health probe must report, not crash
        logger.exception("Health check: database unreachable")

    payload = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "llm_key_present": _llm_key_present(),
        "stt_provider": _stt_provider(),
    }
    return JSONResponse(payload, status_code=200 if db_ok else 503)


async def _resolve_relationship_context(
    relationship_id: str | None,
    from_id: str | None,
    to_id: str | None,
) -> str | None:
    """Build a relationship context string for LLM prompt enrichment."""
    if not relationship_id or not from_id or not to_id:
        return None
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT type, name FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        rel_row = await cursor.fetchone()
        if rel_row is None:
            return None

        cursor = await db.execute(
            "SELECT role, display_name FROM participants WHERE id = ? AND relationship_id = ?",
            (from_id, relationship_id),
        )
        from_row = await cursor.fetchone()

        cursor = await db.execute(
            "SELECT role, display_name FROM participants WHERE id = ? AND relationship_id = ?",
            (to_id, relationship_id),
        )
        to_row = await cursor.fetchone()
    finally:
        await db.close()

    if not from_row or not to_row:
        return None

    rel_type = rel_row["type"]
    rel_name = rel_row["name"]
    from_role = from_row["role"]
    from_name = from_row["display_name"]
    to_role = to_row["role"]
    to_name = to_row["display_name"]

    return (
        f"Relationship: {rel_name} ({rel_type}). "
        f"You are coaching {from_name} (role: {from_role}) "
        f"speaking to {to_name} (role: {to_role})."
    )


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    system = empathy_system_prompt(req.empathy_slider, req.role)

    rel_context = await _resolve_relationship_context(
        req.relationship_id, req.from_participant_id, req.to_participant_id,
    )

    user_content = f"Transcript turn: \"{req.transcript_turn}\""
    if req.context:
        user_content += f"\n\nConversation context: {req.context}"
    if rel_context:
        user_content += f"\n\nRelationship context: {rel_context}"

    llm = get_llm_client()
    # to_thread: llm.complete is a blocking SDK call — never run it on the
    # event loop, or one slow request stalls every other request/WebSocket.
    raw = await asyncio.to_thread(llm.complete, system=system, user=user_content)
    try:
        data = parse_llm_json(raw)
    except (json.JSONDecodeError, IndexError, KeyError):
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")

    suggestions = data.get("suggestions")
    tone_score = data.get("tone_score")
    if (
        not isinstance(suggestions, list)
        or not suggestions
        or not all(isinstance(s, str) for s in suggestions)
    ):
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid suggestions (expected non-empty list of strings)",
        )
    if not isinstance(tone_score, dict) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in tone_score.values()
    ):
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid tone_score (expected integer scores)",
        )
    return RespondResponse(suggestions=suggestions, tone_score=tone_score)


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    system = (
        "You are a tone analysis engine. Analyze the following text and return "
        "ONLY a JSON object with integer scores 0-100 for these dimensions: "
        "warmth, defensiveness, sarcasm, constructiveness, overall. "
        "Higher means more of that quality."
    )

    rel_context = await _resolve_relationship_context(
        req.relationship_id, req.from_participant_id, req.to_participant_id,
    )

    user_content = req.text
    if rel_context:
        user_content += f"\n\nRelationship context: {rel_context}"

    llm = get_llm_client()
    # to_thread: keep the blocking SDK call off the event loop (see /respond).
    raw = await asyncio.to_thread(
        llm.complete, system=system, user=user_content, max_tokens=256,
    )
    try:
        data = parse_llm_json(raw)
    except (json.JSONDecodeError, IndexError, KeyError):
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")

    # Honest failure: a missing dimension must be a 502, never a fabricated 0.
    dimensions = ("warmth", "defensiveness", "sarcasm", "constructiveness", "overall")
    invalid = [
        d for d in dimensions
        if not isinstance(data.get(d), int) or isinstance(data.get(d), bool)
    ]
    if invalid:
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM response missing or invalid score dimensions: "
                + ", ".join(invalid)
            ),
        )

    return ScoreResponse(
        warmth=data["warmth"],
        defensiveness=data["defensiveness"],
        sarcasm=data["sarcasm"],
        constructiveness=data["constructiveness"],
        overall=data["overall"],
    )


@app.post("/session", response_model=SessionOut, status_code=201)
async def create_session(req: SessionCreate):
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata) "
            "VALUES (?, ?, ?, ?)",
            (session_id, created_at, json.dumps(req.turns), json.dumps(req.metadata)),
        )
        await db.commit()
    finally:
        await db.close()

    return SessionOut(
        id=session_id,
        created_at=created_at,
        turns=req.turns,
        metadata=req.metadata,
    )


@app.get("/session/{session_id}", response_model=SessionOut)
async def get_session(session_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionOut(
        id=row["id"],
        created_at=row["created_at"],
        turns=json.loads(row["turns"]),
        metadata=json.loads(row["metadata"]),
    )


@app.post("/session/{session_id}/turns", response_model=TurnResponse, status_code=201)
async def add_turn(session_id: str, turn: SessionTurn):
    turn_dict = turn.model_dump()

    db = await get_db()
    try:
        # Atomic append via SQLite JSON1 — a single UPDATE cannot lose turns
        # to a concurrent read-modify-write race (last-writer-wins data loss).
        cursor = await db.execute(
            "UPDATE sessions SET turns = json_insert(turns, '$[#]', json(?)) "
            "WHERE id = ?",
            (json.dumps(turn_dict), session_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        await db.commit()

        cursor = await db.execute(
            "SELECT turns FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        turns = json.loads(row["turns"])
    finally:
        await db.close()

    return TurnResponse(
        session_id=session_id,
        turn_index=len(turns) - 1,
        turn=turn_dict,
    )


# ---------------------------------------------------------------------------
# Session export helpers
# ---------------------------------------------------------------------------

def _compute_aggregate_stats(turns: list[dict]) -> dict[str, float]:
    """Compute average tone scores across all turns that have score dicts."""
    dimensions = ["warmth", "defensiveness", "sarcasm", "constructiveness", "overall"]
    totals = {d: 0.0 for d in dimensions}
    count = 0
    for t in turns:
        score = t.get("score")
        if isinstance(score, dict):
            count += 1
            for d in dimensions:
                totals[d] += score.get(d, 0)
    if count == 0:
        return {d: 0.0 for d in dimensions}
    return {d: round(totals[d] / count, 1) for d in dimensions}


def _build_text_export(session: dict, insights: str) -> str:
    """Build structured text export for a session."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MINDSHIFT SESSION EXPORT")
    lines.append("=" * 60)
    lines.append("")

    # Metadata
    lines.append("SESSION METADATA")
    lines.append("-" * 40)
    lines.append(f"  Session ID : {session['id']}")
    lines.append(f"  Created    : {session['created_at']}")
    for k, v in session.get("metadata", {}).items():
        lines.append(f"  {k.title():11s}: {v}")
    lines.append("")

    # Transcript
    turns = session.get("turns", [])
    lines.append("TRANSCRIPT")
    lines.append("-" * 40)
    for i, t in enumerate(turns, 1):
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "")
        lines.append(f"  Turn {i} [{speaker}]: {text}")
        score = t.get("score")
        if isinstance(score, dict):
            parts = [f"{k}={v}" for k, v in score.items()]
            lines.append(f"    Tone: {', '.join(parts)}")
    lines.append("")

    # Aggregate stats
    stats = _compute_aggregate_stats(turns)
    lines.append("AGGREGATE STATISTICS")
    lines.append("-" * 40)
    for k, v in stats.items():
        lines.append(f"  Avg {k:20s}: {v}")
    lines.append("")

    # Insights
    lines.append("SESSION INSIGHTS")
    lines.append("-" * 40)
    lines.append(f"  {insights}")
    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def _build_pdf_export(session: dict, insights: str) -> bytes:
    """Generate a PDF report for a session using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story: list = []

    # Title
    story.append(Paragraph("MindShift Session Export", styles["Title"]))
    story.append(Spacer(1, 12))

    # Metadata
    story.append(Paragraph("Session Metadata", styles["Heading2"]))
    meta_data = [
        ["Session ID", session["id"]],
        ["Created", session["created_at"]],
    ]
    for k, v in session.get("metadata", {}).items():
        meta_data.append([k.title(), str(v)])
    meta_table = Table(meta_data, colWidths=[1.5 * inch, 4.5 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Transcript
    turns = session.get("turns", [])
    story.append(Paragraph("Transcript", styles["Heading2"]))
    for i, t in enumerate(turns, 1):
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "")
        story.append(Paragraph(f"<b>Turn {i} [{speaker}]:</b> {text}", styles["Normal"]))
        score = t.get("score")
        if isinstance(score, dict):
            parts = [f"{k}={v}" for k, v in score.items()]
            story.append(Paragraph(f"<i>Tone: {', '.join(parts)}</i>", styles["Normal"]))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 12))

    # Aggregate stats
    stats = _compute_aggregate_stats(turns)
    story.append(Paragraph("Aggregate Statistics", styles["Heading2"]))
    stat_data = [["Dimension", "Average"]] + [[k.title(), str(v)] for k, v in stats.items()]
    stat_table = Table(stat_data, colWidths=[2 * inch, 1.5 * inch])
    stat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#cccccc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 12))

    # Insights
    story.append(Paragraph("Session Insights", styles["Heading2"]))
    story.append(Paragraph(insights, styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


@app.get("/session/{session_id}/export")
async def export_session(
    session_id: str,
    format: ExportFormat = Query(default=ExportFormat.text),
):
    # Fetch session
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        session = {
            "id": row["id"],
            "created_at": row["created_at"],
            "turns": json.loads(row["turns"]),
            "metadata": json.loads(row["metadata"]),
        }
    except (json.JSONDecodeError, TypeError):
        logger.exception("Corrupt stored JSON for session %s", session_id)
        raise HTTPException(status_code=500, detail="corrupt session data")

    # Generate AI insights — an LLM failure must not sink the transcript.
    llm = get_llm_client()
    turns_summary = "\n".join(
        f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in session["turns"]
    )
    insights_prompt = (
        "You are a therapist assistant. Summarize the following session in one short paragraph. "
        "Highlight communication patterns, emotional dynamics, and areas for improvement.\n\n"
        f"{turns_summary}"
    )
    try:
        # to_thread: keep the blocking SDK call off the event loop.
        insights = await asyncio.to_thread(
            llm.complete,
            system="You are a clinical communication analyst.",
            user=insights_prompt,
            max_tokens=300,
        )
    except Exception as exc:  # noqa: BLE001 — degrade honestly, keep transcript
        logger.warning(
            "Insights generation failed for session %s: %s", session_id, exc,
        )
        insights = f"Insights unavailable: {exc}"

    if format == ExportFormat.pdf:
        pdf_bytes = _build_pdf_export(session, insights)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=session_{session_id}.pdf"},
        )

    text_export = _build_text_export(session, insights)
    return Response(
        content=text_export,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=session_{session_id}.txt"},
    )


# ---------------------------------------------------------------------------
# Relationship graph endpoints
# ---------------------------------------------------------------------------

def _generate_edges(rel_type: str, participants: list[dict]) -> list[dict]:
    """Generate valid communication edges based on relationship type."""
    edges = []
    by_id = {p["id"]: p for p in participants}

    if rel_type == "couple":
        # Bidirectional between both partners
        if len(participants) >= 2:
            a, b = participants[0], participants[1]
            edges.append({
                "from_participant_id": a["id"],
                "from_display_name": a["display_name"],
                "to_participant_id": b["id"],
                "to_display_name": b["display_name"],
                "context": "partner_to_partner",
            })
            edges.append({
                "from_participant_id": b["id"],
                "from_display_name": b["display_name"],
                "to_participant_id": a["id"],
                "to_display_name": a["display_name"],
                "context": "partner_to_partner",
            })

    elif rel_type == "parent_child":
        parents = [p for p in participants if p.get("parent_id") is None]
        children = [p for p in participants if p.get("parent_id") is not None]
        for parent in parents:
            for child in children:
                edges.append({
                    "from_participant_id": parent["id"],
                    "from_display_name": parent["display_name"],
                    "to_participant_id": child["id"],
                    "to_display_name": child["display_name"],
                    "context": "parent_to_child",
                })
                edges.append({
                    "from_participant_id": child["id"],
                    "from_display_name": child["display_name"],
                    "to_participant_id": parent["id"],
                    "to_display_name": parent["display_name"],
                    "context": "child_to_parent",
                })

    elif rel_type == "coach_team":
        coaches = [p for p in participants if p["role"] == "coach"]
        players = [p for p in participants if p["role"] != "coach"]
        for coach in coaches:
            for player in players:
                edges.append({
                    "from_participant_id": coach["id"],
                    "from_display_name": coach["display_name"],
                    "to_participant_id": player["id"],
                    "to_display_name": player["display_name"],
                    "context": "coach_to_player",
                })
                edges.append({
                    "from_participant_id": player["id"],
                    "from_display_name": player["display_name"],
                    "to_participant_id": coach["id"],
                    "to_display_name": coach["display_name"],
                    "context": "player_to_coach",
                })

    elif rel_type == "org":
        # Edges between managers and their direct reports (via parent_id)
        for p in participants:
            if p.get("parent_id") and p["parent_id"] in by_id:
                manager = by_id[p["parent_id"]]
                edges.append({
                    "from_participant_id": manager["id"],
                    "from_display_name": manager["display_name"],
                    "to_participant_id": p["id"],
                    "to_display_name": p["display_name"],
                    "context": "manager_to_report",
                })
                edges.append({
                    "from_participant_id": p["id"],
                    "from_display_name": p["display_name"],
                    "to_participant_id": manager["id"],
                    "to_display_name": manager["display_name"],
                    "context": "upward",
                })
        # Peer edges: participants sharing the same parent_id
        from itertools import combinations
        children_by_parent: dict[str, list[dict]] = {}
        for p in participants:
            pid = p.get("parent_id")
            if pid:
                children_by_parent.setdefault(pid, []).append(p)
        for siblings in children_by_parent.values():
            for a, b in combinations(siblings, 2):
                edges.append({
                    "from_participant_id": a["id"],
                    "from_display_name": a["display_name"],
                    "to_participant_id": b["id"],
                    "to_display_name": b["display_name"],
                    "context": "peer",
                })
                edges.append({
                    "from_participant_id": b["id"],
                    "from_display_name": b["display_name"],
                    "to_participant_id": a["id"],
                    "to_display_name": a["display_name"],
                    "context": "peer",
                })

    else:  # custom — all-to-all
        from itertools import permutations
        for a, b in permutations(participants, 2):
            edges.append({
                "from_participant_id": a["id"],
                "from_display_name": a["display_name"],
                "to_participant_id": b["id"],
                "to_display_name": b["display_name"],
                "context": "custom",
            })

    return edges


@app.post("/relationships", response_model=RelationshipOut, status_code=201)
async def create_relationship(req: RelationshipCreate):
    rel_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO relationships (id, type, name, created_at) VALUES (?, ?, ?, ?)",
            (rel_id, req.type.value, req.name, created_at),
        )
        for p in req.participants:
            await db.execute(
                "INSERT INTO participants (id, relationship_id, role, display_name, parent_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (p.id, rel_id, p.role, p.display_name, p.parent_id),
            )
        await db.commit()
    finally:
        await db.close()

    return RelationshipOut(
        id=rel_id,
        type=req.type,
        name=req.name,
        participants=req.participants,
        created_at=created_at,
    )


@app.get("/relationships/{relationship_id}", response_model=RelationshipOut)
async def get_relationship(relationship_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, type, name, created_at FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        rel_row = await cursor.fetchone()
        if rel_row is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, role, display_name, parent_id FROM participants WHERE relationship_id = ?",
            (relationship_id,),
        )
        p_rows = await cursor.fetchall()
    finally:
        await db.close()

    return RelationshipOut(
        id=rel_row["id"],
        type=RelationshipType(rel_row["type"]),
        name=rel_row["name"],
        participants=[
            Participant(
                id=r["id"], role=r["role"],
                display_name=r["display_name"], parent_id=r["parent_id"],
            )
            for r in p_rows
        ],
        created_at=rel_row["created_at"],
    )


@app.get("/relationships/{relationship_id}/edges", response_model=list[EdgeOut])
async def list_edges(relationship_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, type FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        rel_row = await cursor.fetchone()
        if rel_row is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, role, display_name, parent_id FROM participants WHERE relationship_id = ?",
            (relationship_id,),
        )
        p_rows = await cursor.fetchall()
    finally:
        await db.close()

    participants = [
        {"id": r["id"], "role": r["role"], "display_name": r["display_name"], "parent_id": r["parent_id"]}
        for r in p_rows
    ]
    edges = _generate_edges(rel_row["type"], participants)
    return [EdgeOut(**e) for e in edges]


@app.post(
    "/relationships/{relationship_id}/sessions",
    response_model=RelationshipSessionOut,
    status_code=201,
)
async def create_relationship_session(relationship_id: str, req: RelationshipSessionCreate):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, type FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        rel_row = await cursor.fetchone()
        if rel_row is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        # Verify participants exist in this relationship
        cursor = await db.execute(
            "SELECT id, role, display_name, parent_id FROM participants WHERE relationship_id = ?",
            (relationship_id,),
        )
        p_rows = await cursor.fetchall()
        participant_ids = {r["id"] for r in p_rows}
        if req.from_participant_id not in participant_ids:
            raise HTTPException(status_code=400, detail="from_participant_id not in relationship")
        if req.to_participant_id not in participant_ids:
            raise HTTPException(status_code=400, detail="to_participant_id not in relationship")

        # Determine edge context
        participants = [
            {"id": r["id"], "role": r["role"], "display_name": r["display_name"], "parent_id": r["parent_id"]}
            for r in p_rows
        ]
        edges = _generate_edges(rel_row["type"], participants)
        edge_context = "custom"
        for e in edges:
            if e["from_participant_id"] == req.from_participant_id and e["to_participant_id"] == req.to_participant_id:
                edge_context = e["context"]
                break

        session_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata, relationship_id, "
            "from_participant_id, to_participant_id, edge_context, empathy_slider) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, created_at, json.dumps([]), json.dumps(req.metadata),
             relationship_id, req.from_participant_id, req.to_participant_id,
             edge_context, req.empathy_slider),
        )
        await db.commit()
    finally:
        await db.close()

    return RelationshipSessionOut(
        id=session_id,
        relationship_id=relationship_id,
        from_participant_id=req.from_participant_id,
        to_participant_id=req.to_participant_id,
        edge_context=edge_context,
        empathy_slider=req.empathy_slider,
        created_at=created_at,
        turns=[],
        metadata=req.metadata,
    )


@app.get(
    "/relationships/{relationship_id}/sessions",
    response_model=list[RelationshipSessionOut],
)
async def list_relationship_sessions(relationship_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata, relationship_id, "
            "from_participant_id, to_participant_id, edge_context, empathy_slider "
            "FROM sessions WHERE relationship_id = ? ORDER BY created_at DESC",
            (relationship_id,),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        RelationshipSessionOut(
            id=row["id"],
            relationship_id=row["relationship_id"],
            from_participant_id=row["from_participant_id"],
            to_participant_id=row["to_participant_id"],
            edge_context=row["edge_context"],
            empathy_slider=row["empathy_slider"] or 50,
            created_at=row["created_at"],
            turns=json.loads(row["turns"]),
            metadata=json.loads(row["metadata"]),
        )
        for row in rows
    ]


@app.get(
    "/relationships/{relationship_id}/participant/{participant_id}/sessions",
    response_model=list[RelationshipSessionOut],
)
async def list_participant_sessions(relationship_id: str, participant_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM relationships WHERE id = ?",
            (relationship_id,),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata, relationship_id, "
            "from_participant_id, to_participant_id, edge_context, empathy_slider "
            "FROM sessions WHERE relationship_id = ? "
            "AND (from_participant_id = ? OR to_participant_id = ?) "
            "ORDER BY created_at DESC",
            (relationship_id, participant_id, participant_id),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        RelationshipSessionOut(
            id=row["id"],
            relationship_id=row["relationship_id"],
            from_participant_id=row["from_participant_id"],
            to_participant_id=row["to_participant_id"],
            edge_context=row["edge_context"],
            empathy_slider=row["empathy_slider"] or 50,
            created_at=row["created_at"],
            turns=json.loads(row["turns"]),
            metadata=json.loads(row["metadata"]),
        )
        for row in rows
    ]
