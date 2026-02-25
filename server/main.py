import os
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DB_PATH = os.getenv("MINDSHIFT_DB_PATH", "mindshift.db")
ANTHROPIC_MODEL = "claude-3-haiku-20240307"


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
                metadata TEXT NOT NULL
            )
            """
        )
        await db.commit()
    finally:
        await db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="MindShift API", lifespan=lifespan)


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

def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


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


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    system = empathy_system_prompt(req.empathy_slider, req.role)
    user_content = f"Transcript turn: \"{req.transcript_turn}\""
    if req.context:
        user_content += f"\n\nConversation context: {req.context}"

    client = get_anthropic_client()
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = message.content[0].text
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

    client = get_anthropic_client()
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": req.text}],
    )

    raw = message.content[0].text
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
async def create_session(req: SessionCreate):
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata) VALUES (?, ?, ?, ?)",
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
