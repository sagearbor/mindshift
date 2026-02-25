import hashlib
import io
import os
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import aiosqlite
from fastapi import FastAPI, Header, HTTPException, Query, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel, Field

from audio_pipeline import audio_ws_endpoint
from llm_client import LLMClient

DB_PATH = os.getenv("MINDSHIFT_DB_PATH", "mindshift.db")
MINDSHIFT_MODEL = os.getenv("MINDSHIFT_MODEL", "claude-3-haiku-20240307")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RespondRequest(BaseModel):
    transcript_turn: str
    role: str
    empathy_slider: int = Field(ge=0, le=100)
    context: str = ""


class RespondResponse(BaseModel):
    suggestions: list[str]
    tone_score: dict[str, int]


class ScoreRequest(BaseModel):
    text: str


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


class AuthSessionCreate(BaseModel):
    therapist_id: str
    patient_id: str
    role_pair: str = "Husband/Wife"


class AuthSessionOut(BaseModel):
    session_token: str
    therapist_id: str
    patient_id: str
    role_pair: str
    created_at: str


class PatientSummary(BaseModel):
    patient_id: str
    session_count: int


class PatientSessionOut(BaseModel):
    id: str
    created_at: str
    turn_count: int
    metadata: dict


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
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
                therapist_id TEXT,
                patient_id TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                session_token TEXT PRIMARY KEY,
                therapist_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                role_pair TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        await db.commit()
    finally:
        await db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.llm_client = LLMClient(model=MINDSHIFT_MODEL)
    yield


app = FastAPI(title="MindShift API", lifespan=lifespan)


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


async def _resolve_session_token(token: str | None) -> dict | None:
    """Look up an auth session by token. Returns dict or None."""
    if not token:
        return None
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT therapist_id, patient_id, role_pair FROM auth_sessions WHERE session_token = ?",
            (token,),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()
    if row is None:
        return None
    return {"therapist_id": row["therapist_id"], "patient_id": row["patient_id"], "role_pair": row["role_pair"]}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    system = empathy_system_prompt(req.empathy_slider, req.role)
    user_content = f"Transcript turn: \"{req.transcript_turn}\""
    if req.context:
        user_content += f"\n\nConversation context: {req.context}"

    llm = get_llm_client()
    raw = llm.complete(system=system, user=user_content)
    try:
        data = parse_llm_json(raw)
    except (json.JSONDecodeError, IndexError, KeyError):
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")

    suggestions = data.get("suggestions", [])
    tone_score = data.get("tone_score", {})
    return RespondResponse(suggestions=suggestions, tone_score=tone_score)


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    system = (
        "You are a tone analysis engine. Analyze the following text and return "
        "ONLY a JSON object with integer scores 0-100 for these dimensions: "
        "warmth, defensiveness, sarcasm, constructiveness, overall. "
        "Higher means more of that quality."
    )

    llm = get_llm_client()
    raw = llm.complete(system=system, user=req.text, max_tokens=256)
    try:
        data = parse_llm_json(raw)
    except (json.JSONDecodeError, IndexError, KeyError):
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")

    return ScoreResponse(
        warmth=data.get("warmth", 0),
        defensiveness=data.get("defensiveness", 0),
        sarcasm=data.get("sarcasm", 0),
        constructiveness=data.get("constructiveness", 0),
        overall=data.get("overall", 0),
    )


@app.post("/session", response_model=SessionOut, status_code=201)
async def create_session(
    req: SessionCreate,
    x_session_token: Optional[str] = Header(None),
):
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    auth = await _resolve_session_token(x_session_token)
    therapist_id = auth["therapist_id"] if auth else None
    patient_id = auth["patient_id"] if auth else None

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata, therapist_id, patient_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, created_at, json.dumps(req.turns), json.dumps(req.metadata),
             therapist_id, patient_id),
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
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT turns FROM sessions WHERE id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")

        turns = json.loads(row["turns"])
        turn_dict = turn.model_dump()
        turns.append(turn_dict)

        await db.execute(
            "UPDATE sessions SET turns = ? WHERE id = ?",
            (json.dumps(turns), session_id),
        )
        await db.commit()
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

    session = {
        "id": row["id"],
        "created_at": row["created_at"],
        "turns": json.loads(row["turns"]),
        "metadata": json.loads(row["metadata"]),
    }

    # Generate AI insights
    llm = get_llm_client()
    turns_summary = "\n".join(
        f"{t.get('speaker', '?')}: {t.get('text', '')}" for t in session["turns"]
    )
    insights_prompt = (
        "You are a therapist assistant. Summarize the following session in one short paragraph. "
        "Highlight communication patterns, emotional dynamics, and areas for improvement.\n\n"
        f"{turns_summary}"
    )
    insights = llm.complete(
        system="You are a clinical communication analyst.",
        user=insights_prompt,
        max_tokens=300,
    )

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
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/session", response_model=AuthSessionOut, status_code=201)
async def create_auth_session(req: AuthSessionCreate):
    token = hashlib.sha256(
        f"{req.therapist_id}:{req.patient_id}:{uuid.uuid4()}".encode()
    ).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO auth_sessions (session_token, therapist_id, patient_id, role_pair, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (token, req.therapist_id, req.patient_id, req.role_pair, created_at),
        )
        await db.commit()
    finally:
        await db.close()

    return AuthSessionOut(
        session_token=token,
        therapist_id=req.therapist_id,
        patient_id=req.patient_id,
        role_pair=req.role_pair,
        created_at=created_at,
    )


@app.get("/therapist/{therapist_id}/patients", response_model=list[PatientSummary])
async def list_patients(therapist_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT patient_id, COUNT(*) as session_count FROM sessions "
            "WHERE therapist_id = ? GROUP BY patient_id",
            (therapist_id,),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        PatientSummary(patient_id=row["patient_id"], session_count=row["session_count"])
        for row in rows
    ]


@app.get(
    "/therapist/{therapist_id}/patient/{patient_id}/sessions",
    response_model=list[PatientSessionOut],
)
async def list_patient_sessions(therapist_id: str, patient_id: str):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata FROM sessions "
            "WHERE therapist_id = ? AND patient_id = ? ORDER BY created_at DESC",
            (therapist_id, patient_id),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    return [
        PatientSessionOut(
            id=row["id"],
            created_at=row["created_at"],
            turn_count=len(json.loads(row["turns"])),
            metadata=json.loads(row["metadata"]),
        )
        for row in rows
    ]
