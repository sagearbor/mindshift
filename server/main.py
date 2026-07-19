import asyncio
import base64
import hashlib
import hmac
import io
import logging
import os
import json
import time
import uuid
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path as FilePath
from typing import Annotated, Awaitable, Callable, Optional

# Load a repo-root `.env` (if present) BEFORE any configuration is read below.
# python-dotenv's default is override=False, so real environment variables
# always win over .env values. Defensive try/except: a missing python-dotenv
# must never break the server — .env support simply degrades to a no-op.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover — python-dotenv is in requirements
    pass
else:
    load_dotenv(FilePath(__file__).resolve().parent.parent / ".env")

import aiosqlite
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
    WebSocket,
)
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

import dynamics
import episodes
import link_fetch
import prosody
import recordings_store
import speaker_id
import word_metrics as word_metrics_mod
from audio_ingest import (
    AudioDecodeError,
    NoSpeechFound,
    TranscriptionUnavailable,
    build_derivatives,
    decode_to_pcm,
    transcribe_prerecorded,
)
from audio_pipeline import UUID_PATTERN, audio_ws_endpoint
from auth import get_current_uid, init_firebase
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
_DEFAULT_DB_PATH = FilePath(__file__).resolve().parent.parent / "mindshift.db"
DB_PATH = os.getenv("MINDSHIFT_DB_PATH") or str(_DEFAULT_DB_PATH)
MINDSHIFT_MODEL = os.getenv("MINDSHIFT_MODEL", "claude-3-haiku-20240307")

# P1-5: conservative per-IP rate limiting on the cost-bearing REST endpoints
# (/respond, /score, /session/{id}/export each spend LLM tokens). Both values
# are env-configurable and DELIBERATELY generous — the exact numbers are a
# flagged human decision, so they are tunable rather than hardcoded. Set
# RATE_LIMIT_ENABLED=0 to turn the limiter off entirely.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))  # HUMAN-TUNABLE
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").strip().lower() not in (
    "0", "false", "no", "off", "",
)

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


# --- Voice profile ("speaks in your actual voice") -------------------------
# Caps are enforced server-side by design: this is few-shot prompt material,
# not fine-tuning. Bounded storage keeps the injected prompt small and
# predictable, which is a latency property (it feeds a real-time coach), not
# just cleanliness.
MAX_PAIRS = 5
MAX_PAIR_CHARS = 200
MAX_STYLE_NOTES_CHARS = 300


class VoicePair(BaseModel):
    # min_length keeps empty examples out of the prompt; the upper bound is a
    # truncation on write (MAX_PAIR_CHARS), not a rejection, per the spec.
    suggestion: str = Field(min_length=1)
    rephrase: str = Field(min_length=1)


class VoiceProfileIn(BaseModel):
    pairs: list[VoicePair] = []
    style_notes: Optional[str] = None


class VoiceProfileOut(BaseModel):
    pairs: list[VoicePair]
    style_notes: Optional[str] = None
    updated_at: Optional[str] = None


# --- POST /analyze — post-hoc conversation-dynamics ("the impartial third
# chair"). One batch LLM pass scores every turn; Python computes the statistics
# in dynamics.py. Framing is therapy-adjacent: the DYNAMIC, never a winner. ----

# The exact marker vocabulary (Gottman Four Horsemen + repair/validation). Any
# marker the LLM emits outside this set is dropped — never fabricated, never
# renamed. HORSEMEN (a subset) drives the per-speaker horsemen counts.
ANALYZE_MARKER_VOCAB = frozenset(
    (*dynamics.HORSEMEN, "repair_attempt", "validation"),
)
ANALYZE_REQUEST_OUTCOMES = frozenset(("granted", "denied", "deferred", "unclear"))

# §2 speaker display-label ladder — the ordered set of provenance rungs, highest
# precedence FIRST. enrolled (a matched enrolled voiceprint — the viewer's own
# voice, from the voice-enrollment pipeline's speaker_identity) beats name (§2a,
# high-confidence transcript evidence) beats voice (§2b, relative pitch) beats
# generic (§2c, the raw speaker id). The resolver reads speaker_identity
# defensively: the rung is dormant until the enrollment pipeline (PR #56) merges
# and passes it, and any absent/malformed shape simply skips the rung.
LABEL_SOURCE_ENROLLED = "enrolled"
LABEL_SOURCE_NAME = "name"
LABEL_SOURCE_VOICE = "voice"
LABEL_SOURCE_GENERIC = "generic"

# The display label for the speaker matched to the authenticated user's own
# enrolled voiceprint (speaker_identity.matched_speaker). Second person on
# purpose: the recording belongs to the viewing uid, so the matched voice IS the
# viewer — more honest and more useful than any inferred third-person name.
ENROLLED_DISPLAY_LABEL = "You"

# Only a name the LLM inferred with EXACTLY this confidence (from direct
# transcript evidence — §2a) is applied; medium/low are treated as "no name" and
# fall through to the voice/generic rungs. Never guess.
_NAME_CONFIDENCE_APPLIED = "high"

# Display names are capped to the same length as a speaker label (AnalyzeTurn.
# speaker) so a pathological LLM name can't bloat the stored analysis.json.
SPEAKER_NAME_MAX = 60

# A 2..10-speaker conversation of 4..400 turns. The per-turn upper bound and the
# total-transcript char cap are independent belts: 400 * 2000 chars is far more
# than any single LLM pass should carry, so the total cap (a 413) bites first on
# a pathological payload.
ANALYZE_MIN_TURNS = 4
ANALYZE_MAX_TURNS = 400
ANALYZE_MIN_SPEAKERS = 1
ANALYZE_MAX_SPEAKERS = 10
ANALYZE_MAX_TRANSCRIPT_CHARS = 60_000

# Per-person report-card field caps (§2). Enforced server-side as truncation on
# write, never rejection — a slightly-too-long LLM string is trimmed, not a 502.
REPORT_CARD_HEADLINE_MAX = 80
REPORT_CARD_TEXT_MAX = 200

# Counterfactual (§3): rationale cap + the FIXED server-owned disclaimer. The
# disclaimer is never LLM-authored — it is a constant so the "this is an
# estimate, not a measurement" framing can never be softened or dropped.
COUNTERFACTUAL_RATIONALE_MAX = 200
COUNTERFACTUAL_DISCLAIMER = (
    "Simulation — an estimate of how the conversation might have unfolded, not "
    "a measurement."
)


class AnalyzeTurn(BaseModel):
    speaker: str = Field(min_length=1, max_length=60)
    text: str = Field(min_length=1, max_length=2000)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AnalyzeRequest(BaseModel):
    turns: list[AnalyzeTurn] = Field(
        min_length=ANALYZE_MIN_TURNS, max_length=ANALYZE_MAX_TURNS,
    )
    context: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _validate_speaker_count(self) -> "AnalyzeRequest":
        # 1..10 DISTINCT speakers. Monologues are IN scope (min was 2 until a
        # real recording of one person performing two voices got diarized as a
        # single speaker and bounced — the merged-diarization case is common,
        # and a solo heat timeline / report card is still honest, useful
        # output; pair dynamics are suppressed with plain descriptions).
        distinct = {t.speaker for t in self.turns}
        if not (ANALYZE_MIN_SPEAKERS <= len(distinct) <= ANALYZE_MAX_SPEAKERS):
            raise ValueError(
                f"conversation must have between {ANALYZE_MIN_SPEAKERS} and "
                f"{ANALYZE_MAX_SPEAKERS} distinct speakers, got {len(distinct)}"
            )
        return self


class VoiceOut(BaseModel):
    # Compact per-turn delivery labels, RELATIVE to this recording's own
    # distribution (see prosody.py). Only present on /analyze/upload turns when
    # prosody succeeded; absent/null on /analyze and on decode-degraded uploads.
    # pitch_label is None for an unvoiced turn (silence/noise) — never invented.
    energy_label: str
    pitch_label: Optional[str]
    rate_label: str


class PerTurnOut(BaseModel):
    index: int
    speaker: str
    heat: int
    markers: list[str]
    is_spike: bool
    trigger_phrase: Optional[str]
    # None on /analyze (no audio) and on uploads where decode failed. Adding it
    # here as an Optional default keeps the /analyze response byte-compatible
    # (voice is simply null), while /analyze/upload fills it when prosody ran.
    voice: Optional[VoiceOut] = None


class HorsemenOut(BaseModel):
    criticism: int
    contempt: int
    defensiveness: int
    stonewalling: int


class PerSpeakerOut(BaseModel):
    turns: int
    talk_share: float
    avg_heat: float
    peak_heat: int
    peak_turn_index: int
    heat_variance: float
    interruptions: Optional[int]
    horsemen: HorsemenOut
    repair_attempts: int
    repairs_accepted: int


class CouplingOut(BaseModel):
    strength: Optional[float]
    leader: Optional[str]
    description: str


class DeescalationOut(BaseModel):
    who_first: Optional[str]
    follow_rate: Optional[float]
    description: str


class TriggerOut(BaseModel):
    phrase: str
    speaker: str
    turn_index: int
    heat_delta: int


class RequestOut(BaseModel):
    speaker: str
    request: str
    outcome: str


class DynamicsOut(BaseModel):
    coupling: CouplingOut
    deescalation: DeescalationOut
    triggers: list[TriggerOut]
    requests: list[RequestOut]


class ReportCardOut(BaseModel):
    # §2 — an overt, comparable per-person report card. score is composure /
    # constructiveness on the ABSOLUTE scale (higher = better conduct), so it is
    # comparable across people AND across sessions, exactly like heat.
    score: int
    headline: str
    did_well: str
    work_on: str


class SpeakerLabelOut(BaseModel):
    # §2 — a human-friendlier display name for one speaker, resolved by the
    # precedence ladder in :func:`_resolve_speaker_labels`. Keyed (in the parent
    # map) by the SAME canonical speaker id as per_speaker/report_cards, which
    # keep keying on the id — this map is purely presentational.
    display_label: str
    # Which rung produced display_label: "enrolled" > "name" > "voice" >
    # "generic" (see LABEL_SOURCE_*). Lets the client — and the future
    # voice-enrollment work — reason about provenance without re-deriving it.
    label_source: str


class AnalyzeResponse(BaseModel):
    per_turn: list[PerTurnOut]
    per_speaker: dict[str, PerSpeakerOut]
    dynamics: DynamicsOut
    narrative: str
    # One card per speaker in the request — validated present in the endpoint
    # (a missing speaker is a 502 misalignment, never a fabricated card).
    report_cards: dict[str, ReportCardOut]
    # §2 — per-speaker display labels (name → deeper/higher voice → generic id),
    # keyed by the canonical speaker id. Additive + backward-compatible: an old
    # stored analysis.json omits this map and the client falls back to the raw
    # id. Empty only on the degenerate path where no speakers resolved.
    speaker_labels: dict[str, SpeakerLabelOut] = Field(default_factory=dict)
    # §1 — an LLM-suggested short conversation title, present ONLY when the caller
    # asked for one (an upload with no user-provided title). None otherwise; the
    # text /analyze path never requests it, so its response stays unchanged.
    title: Optional[str] = None
    # Transparent word-level metrics — per-speaker pronoun profile (I/you/we
    # rates) + emotion-word density (anger/fear/sadness/joy/trust), counted
    # LOCALLY from the turns with no LLM (see word_metrics.py). Additive +
    # backward-compatible: an old stored analysis.json omits it and the detail
    # endpoint backfills from the stored turns on read. None only when there are
    # no usable turns to count.
    word_metrics: Optional[dict] = None


# --- POST /analyze/upload — analyze a RECORDING (audio, or a video whose audio
# track we extract). Process-and-discard: nothing is persisted. The response is
# a superset of AnalyzeResponse: it ADDS the transcribed turns (the client never
# had a transcript — the recording was raw audio) and an optional top-level note
# when prosody had to be skipped. Per-turn voice labels ride on PerTurnOut.voice.


class TranscribedTurn(BaseModel):
    """One diarized turn recovered from the recording, echoed back so the
    client can render the transcript it never had."""
    speaker: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AnalyzeUploadResponse(AnalyzeResponse):
    turns: list[TranscribedTurn]
    # Set to "unavailable: <reason>" when transcription succeeded but the audio
    # could not be decoded for prosody — an HONEST degrade (voice is null on
    # every turn) rather than a hard failure or fabricated labels. None when
    # prosody ran normally.
    voice_analysis: Optional[str] = None
    # Consent-gated persistence outcome (defaults keep the /analyze response
    # byte-compatible for callers that ignore them). ``stored`` is True only
    # when the recording was actually written; ``storage_note`` states plainly
    # why it was NOT ("consent not given" / "storage not enabled" / "storage
    # failed: <class>"). A storage failure never fails the analysis.
    stored: bool = False
    recording_id: Optional[str] = None
    storage_note: Optional[str] = None
    # Enrollment-based speaker identity ("You" — top rung of the label ladder).
    # None when the user hasn't enrolled, the voice deps aren't installed, or the
    # audio couldn't be decoded — the feature is fully optional and never blocks
    # analysis. When present it is speaker_id.identify_speakers()'s report:
    # {matched_speaker, match_threshold, model, speakers:{<label>:{score,is_you}}}.
    # The label-ladder consumer reads ``matched_speaker`` as the highest-precedence
    # source (display_label="You", label_source="enrolled"); per-speaker cosine
    # scores ride along for debugging.
    speaker_identity: Optional[dict] = None
    # Companion P1 — conversation episodes: the transcript split on silence gaps
    # (> EPISODE_GAP_SECONDS with no turns), each with timing, participants,
    # heat stats, and a derived one-line summary (see episodes.py — pure
    # derivation, no extra LLM call). Additive: a short recording is exactly one
    # episode, and callers that ignore the field see the response they always
    # did. None only when segmentation was skipped (e.g. an empty transcript).
    episodes: Optional[list[dict]] = None


# Upload caps (a 413 when exceeded). File-size is a cheap first gate; the
# decoded-duration cap bounds LLM + prosody work on a legitimately-typed but
# very long recording. Both are deliberate product limits, tunable via env.
#
# MAX_UPLOAD_BYTES gates the DIRECT /analyze/upload path only. It cannot exceed
# ~25MB in practice: Cloud Run's HTTP/1 request limit (~32MB) hard-rejects a
# larger body before FastAPI ever sees it, so a bigger direct upload could never
# return an honest 413 anyway. Phone videos are routinely 50-300MB — those go
# through the CHUNKED upload endpoints below (/uploads/*), which stream 8MB
# parts through this same server and reassemble them server-side.
MAX_UPLOAD_BYTES = int(os.getenv("ANALYZE_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))
MAX_UPLOAD_DURATION_S = float(os.getenv("ANALYZE_UPLOAD_MAX_SECONDS", str(40 * 60)))

# Companion P1 — silence gap (seconds, no diarized turns) that splits a long
# recording into separate conversation EPISODES on the "Your Day" timeline.
# Tunable per deployment; the segmentation itself is pure (episodes.py).
EPISODE_GAP_SECONDS = float(os.getenv("EPISODE_GAP_SECONDS", "60"))

# Chunked-upload caps. The 200MB ceiling bounds total reassembled bytes (a 413
# at /uploads/start); the 8MB chunk size keeps every PUT far under Cloud Run's
# ~32MB request limit. With 8MB chunks a 200MB upload is at most 25 parts, which
# fits GCS compose's 32-source limit (see recordings_store.assemble_upload).
# CHUNK_SLACK_BYTES tolerates a slightly-over chunk (e.g. a client that rounds
# up) without rejecting an otherwise-valid part.
MAX_CHUNKED_UPLOAD_BYTES = int(
    os.getenv("CHUNKED_UPLOAD_MAX_BYTES", str(200 * 1024 * 1024))
)
UPLOAD_CHUNK_BYTES = int(os.getenv("CHUNKED_UPLOAD_CHUNK_BYTES", str(8 * 1024 * 1024)))
CHUNK_SLACK_BYTES = 4096

# Async-job staleness + TTL. A non-terminal job whose state has not advanced in
# JOB_STALL_SECONDS is reported as "stalled" (computed on read — an in-process
# task orphaned by an instance restart never gets to write "failed", so without
# this the client would spin forever). Terminal (done/failed) states older than
# JOB_TTL_SECONDS are lazily deleted on read — cheap cleanup, no cron needed.
JOB_STALL_SECONDS = float(os.getenv("ANALYZE_JOB_STALL_SECONDS", "120"))
JOB_TTL_SECONDS = float(os.getenv("ANALYZE_JOB_TTL_SECONDS", str(24 * 60 * 60)))
# While a job sits in ONE long blocking stage (downloading a 100MB+ video, or
# transcoding an HEVC clip to 360p — either can exceed JOB_STALL_SECONDS), a
# background heartbeat refreshes updated_at this often so the poll's "stalled"
# heuristic doesn't false-positive on work that is genuinely still running. Well
# under JOB_STALL_SECONDS so several beats are missed before a job reads stalled.
JOB_HEARTBEAT_SECONDS = float(os.getenv("ANALYZE_JOB_HEARTBEAT_SECONDS", "15"))


# --- Chunked upload session (POST /uploads/start → PUT chunks → complete) -----
# The session lets a phone video (50-300MB) reach analysis despite Cloud Run's
# ~32MB per-request limit: the client streams 8MB parts, the server reassembles
# them, then runs the EXACT same pipeline as the direct /analyze/upload path.
# Session state (manifest + parts) lives in GCS, so the chunked path REQUIRES a
# recordings bucket; without one every /uploads endpoint returns an honest 503.


# User-facing recording title. Optional on submit (falls back to the filename);
# settable/renamable via PATCH /recordings/{id}. Bounded so a pathological title
# can't bloat meta.json.
RECORDING_TITLE_MAX = 200


class UploadStartRequest(BaseModel):
    filename: Optional[str] = None
    content_type: Optional[str] = None
    total_bytes: int = Field(gt=0)
    context: str = Field(default="", max_length=500)
    # Consent-gated persistence, exactly as the direct upload's form fields — but
    # here they are real JSON booleans carried in the manifest and honored at
    # complete(). Default store=True mirrors the direct endpoint.
    consent: bool = False
    store: bool = True
    # Optional user-chosen display title; when absent the recording falls back to
    # its filename (see save_recording). Additive — old clients omit it.
    title: Optional[str] = Field(default=None, max_length=RECORDING_TITLE_MAX)


class UploadStartResponse(BaseModel):
    upload_id: str
    chunk_bytes: int
    expected_chunks: int


class AnalyzeLinkRequest(BaseModel):
    # A user-pasted share URL (Drive share links are rewritten server-side). The
    # server downloads it itself — see link_fetch for the SSRF/size/HTML guards.
    url: str = Field(min_length=1, max_length=2000)
    context: str = Field(default="", max_length=500)
    consent: bool = False
    store: bool = True
    # Optional user-chosen display title; falls back to the fetched filename.
    title: Optional[str] = Field(default=None, max_length=RECORDING_TITLE_MAX)


class RecordingTitleRequest(BaseModel):
    # Rename an existing recording. Stripped + non-empty enforced at the endpoint
    # (a whitespace-only title is a 422, not a silent no-op).
    title: str = Field(min_length=1, max_length=RECORDING_TITLE_MAX)


class RecordingSourceRequest(BaseModel):
    # A user-pasted share URL to attach as an existing recording's HD source for
    # replay. Only RESOLVED (not downloaded) — see PATCH /recordings/{id}/source.
    url: str = Field(min_length=1, max_length=2000)


# --- Submit-and-poll analysis jobs (POST /analyze/link/jobs,
# /uploads/{id}/complete/jobs → 202 {job_id}; GET /analyze/jobs/{job_id}) -----
# A link download or a chunked-upload completion is a MULTI-MINUTE synchronous
# request today: Android backgrounding / socket loss kills the response the
# server has already finished producing, so the user sees "Something went wrong"
# on work that actually succeeded. These endpoints run that exact pipeline as an
# in-process background task and record staged progress in GCS
# (jobs/{uid}/{job_id}/state.json), which the client polls — decoupling the
# result from a single fragile long-lived connection.


class JobCreatedResponse(BaseModel):
    # 202 body: the id to poll GET /analyze/jobs/{job_id} with.
    job_id: str


class JobStateResponse(BaseModel):
    job_id: str
    # queued|downloading|transcribing|analyzing|storing|done|failed, plus the
    # COMPUTED "stalled" (a non-terminal job whose state stopped advancing — see
    # the GET endpoint). Kept a plain str so a future stage needs no client change.
    status: str
    created_at: str
    updated_at: str
    stage_started_at: Optional[str] = None
    # A human string for the current stage (e.g. "38 MB downloaded"), never a
    # fabricated percentage — honest about what the server actually knows.
    progress_note: Optional[str] = None
    # Known once the recording has been decoded/transcribed; lets the client
    # render a rough ETA. None until then.
    duration_seconds: Optional[float] = None
    # During the "downloading" stage: bytes fetched so far and the total (from the
    # source's Content-Length), so the client can render a real "fetching video
    # (NN/116 MB)" progress bar — the download size is NOT the amount transcribed,
    # so it stays a separate field from duration_seconds (which paces the
    # transcription ETA). Both None outside/ before the download stage; bytes_total
    # stays None when the source omits Content-Length. Additive + backward-compatible.
    bytes_downloaded: Optional[int] = None
    bytes_total: Optional[int] = None
    # Honest failure detail — the SAME message the synchronous path would 4xx/5xx
    # with. None unless status is "failed".
    error: Optional[str] = None
    # The full AnalyzeUploadResponse, included ONLY when status is "done".
    result: Optional[AnalyzeUploadResponse] = None


# --- POST /analyze/counterfactual — the "what if they'd said it differently"
# simulation. One LLM call rewrites ONE pivot turn constructively, then
# estimates how the REST of the conversation would likely have unfolded on the
# same absolute heat rubric. Every number is explicitly a simulation, never a
# measurement (see COUNTERFACTUAL_DISCLAIMER). ------------------------------


class CounterfactualRequest(BaseModel):
    turns: list[AnalyzeTurn] = Field(
        min_length=ANALYZE_MIN_TURNS, max_length=ANALYZE_MAX_TURNS,
    )
    pivot_index: int
    context: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _validate_pivot_index(self) -> "CounterfactualRequest":
        # A pivot outside the transcript is a request-shape error → 422 (exactly
        # like the turn-count bounds), never a 500 on an out-of-range index.
        if not (0 <= self.pivot_index < len(self.turns)):
            raise ValueError(
                f"pivot_index {self.pivot_index} out of range for a "
                f"{len(self.turns)}-turn conversation"
            )
        return self


class CounterfactualPerTurnOut(BaseModel):
    index: int
    speaker: str
    heat: int


class CounterfactualResponse(BaseModel):
    pivot_index: int
    rewritten_text: str
    rationale: str
    simulated_per_turn: list[CounterfactualPerTurnOut]
    disclaimer: str


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
                empathy_slider INTEGER,
                user_id TEXT
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                user_id TEXT
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
        # Optional, capped per-(relationship, participant) voice profile. Pairs
        # are a JSON blob (matching the sessions.turns/metadata convention) so
        # there is no join and no per-pair row management. CREATE TABLE IF NOT
        # EXISTS is an additive migration: existing databases keep working and
        # simply gain an empty table.
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_profiles (
                relationship_id TEXT NOT NULL,
                participant_id  TEXT NOT NULL,
                pairs           TEXT NOT NULL DEFAULT '[]',
                style_notes     TEXT,
                updated_at      TEXT NOT NULL,
                PRIMARY KEY (relationship_id, participant_id),
                FOREIGN KEY (relationship_id) REFERENCES relationships(id)
            )
            """
        )
        # --- Auth: additive user-ownership migration (tolerates existing DBs) ---
        # `sessions` and `relationships` are the two ownership roots (a Firebase
        # uid); participants/voice_profiles inherit ownership through their
        # relationship FK. Fresh installs already have the column from the
        # CREATE TABLE above; existing databases gain it here. Guarded against
        # the "duplicate column" error so init_db() stays safe to run every boot
        # (there is no migration framework). No anon-row backfill: prod SQLite is
        # ephemeral, so rows predating auth are simply NULL and match no user.
        await _add_column_if_missing(db, "sessions", "user_id", "TEXT")
        await _add_column_if_missing(db, "relationships", "user_id", "TEXT")
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_user "
            "ON relationships(user_id)"
        )
        await db.commit()
    finally:
        await db.close()


async def _add_column_if_missing(
    db: aiosqlite.Connection, table: str, column: str, coltype: str,
) -> None:
    """Add ``column`` to ``table`` only when absent (idempotent ALTER).

    SQLite has no ``ADD COLUMN IF NOT EXISTS``; running the ALTER twice raises
    "duplicate column name". Checking PRAGMA table_info first keeps init_db()
    safe to run on every boot against a database that already has the column.
    ``table``/``column`` are internal constants, never user input.
    """
    cursor = await db.execute(f"PRAGMA table_info({table})")
    existing = {row["name"] for row in await cursor.fetchall()}
    if column not in existing:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


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
    # Initialize Firebase Admin for ID-token verification (ADC on Cloud Run).
    # Best-effort: init never raises, and verification fails closed (401).
    init_firebase()
    app.state.llm_client = LLMClient(model=MINDSHIFT_MODEL)
    _configure_stt(app)
    # Recording persistence (opt-in). None when MINDSHIFT_RECORDINGS_BUCKET is
    # unset → the recordings endpoints report an honest 503 and /analyze/upload
    # keeps its process-and-discard behaviour.
    app.state.recordings_store = recordings_store.create_store()
    logger.info(
        "MindShift API started — model=%s provider=%s llm_key_present=%s "
        "stt_provider=%s db_path=%s",
        MINDSHIFT_MODEL,
        _detected_provider(),
        "yes" if _llm_key_present() else "no",
        _stt_provider(),
        FilePath(DB_PATH).resolve(),
    )
    yield
    app.state.llm_client.close()
    logger.info("MindShift API shut down — LLM client closed")


app = FastAPI(title="MindShift API", lifespan=lifespan)

# HTTP CORS for the browser app. Native apps never preflight, so this gap only
# bit when the web app made authenticated fetch() calls: the OPTIONS preflight
# hit the router (405) and the browser refused to send the real request. Same
# allowlist the WebSocket origin gate uses (MINDSHIFT_ALLOWED_ORIGINS), plus
# localhost fallbacks for local dev. Bearer auth, no cookies → credentials off.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

_cors_origins = [
    o.strip()
    for o in os.getenv("MINDSHIFT_ALLOWED_ORIGINS", "").split(",")
    if o.strip()
] or ["http://localhost:8081", "http://localhost:19006"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Voice enrollment ("This is me" + "Forget my voice"). Its own file so the
# monolith's edit surface stays tiny; the enroll/match path is torch-gated and
# degrades to honest 503s when the optional deps aren't installed.
from routers import voice as _voice_router  # noqa: E402

app.include_router(_voice_router.router)


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

def _render_voice_profile(profile: dict | None) -> str:
    """Render the trailing few-shot "voice" block, or "" when there is nothing
    to say. Empty output is the load-bearing property: it lets
    ``empathy_system_prompt`` stay byte-identical to today whenever no profile
    is set. Pairs are rendered in stored order (oldest→newest, recency last)
    and hard-capped at ``MAX_PAIRS`` as belt-and-suspenders over the write-time
    cap.
    """
    if not profile:
        return ""
    pairs = profile.get("pairs") or []
    if not isinstance(pairs, list):
        pairs = []
    pairs = pairs[:MAX_PAIRS]
    style_notes = profile.get("style_notes")
    if not pairs and not style_notes:
        return ""

    lines = [
        "The user rephrases coaching suggestions into their own voice. Match "
        "that voice",
        "in every suggestion you produce, while keeping the coaching intent and "
        "the exact",
        "JSON output format described above.",
    ]
    if pairs:
        lines += [
            "",
            "Examples of how this user rephrases a generic suggestion into how "
            "they'd",
            "actually say it:",
        ]
        for pair in pairs:
            suggestion = pair.get("suggestion", "")
            rephrase = pair.get("rephrase", "")
            lines.append(f'- Generic: "{suggestion}"')
            lines.append(f'  They\'d say: "{rephrase}"')
    if style_notes:
        lines += ["", f"Style notes: {style_notes}"]
    return "\n".join(lines)


def empathy_system_prompt(
    slider: int, role: str, voice_profile: dict | None = None,
) -> str:
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

    prompt = (
        f"{stance}\n\n"
        f"The user's role in this conversation is: {role}.\n"
        "Provide exactly 3 short suggested responses the user could say next. "
        "Return ONLY a JSON object with key \"suggestions\" (a list of strings), "
        "\"tone_score\" (an object with integer keys: warmth, defensiveness, "
        "sarcasm, constructiveness, overall — each 0-100, scoring the transcript "
        "turn), and \"importance\" (an integer 0-100: how much the user needs a "
        "coaching interjection at THIS moment — high for emotionally charged, "
        "high-stakes, or pivotal turns; low for small talk, filler, or logistics)."
    )
    # Append the voice-profile few-shot block AFTER the output contract so the
    # required JSON format stays stated last and authoritative. When there is
    # no profile (None) or it renders empty, the prompt is byte-identical to
    # before — the working coach cannot regress.
    if voice_profile is not None:
        block = _render_voice_profile(voice_profile)
        if block:
            prompt += "\n\n" + block
    return prompt


def self_feedback_prompt(
    slider: int, role: str, voice_profile: dict | None = None,
) -> str:
    """System prompt for coaching the user on THEIR OWN just-spoken turn.

    Where :func:`empathy_system_prompt` suggests what to say to the OTHER
    person, this is a real-time delivery coach whispering in the user's ear:
    the user themself just spoke, and the model returns ONE tiny, instantly
    absorbable course-correction on HOW they came across.

    The correction DIRECTION follows the empathy dial — a hard product
    requirement (the owner explicitly forbade an always-soften coach). At low
    empathy the user is trying to be MORE assertive, so hedging / over-
    apologising is what needs fixing; at high empathy harshness is. Same four
    stance buckets as :func:`empathy_system_prompt`, so the two prompts move in
    lockstep with the slider.
    """
    if slider <= 20:
        stance = (
            "The user is working to come across MORE assertive and direct. "
            "When they hedge, over-apologize, soften too much, or back down, "
            "nudge them toward firmness and standing their ground. Never tell "
            "them to soften."
        )
    elif slider <= 50:
        stance = (
            "The user is working toward balanced, clear delivery. Nudge them "
            "firmer when they over-hedge or over-apologize, and warmer when "
            "they turn harsh or blaming — whichever keeps them fair and direct."
        )
    elif slider <= 80:
        stance = (
            "The user is working to come across warmer and less combative. "
            "When they sound harsh, blaming, dismissive, or defensive, nudge "
            "them toward warmth, validation, and softening their tone."
        )
    else:
        stance = (
            "The user is working to be fully warm and validating. The moment "
            "any harshness, sarcasm, or defensiveness creeps into their "
            "delivery, nudge them toward gentleness and validation."
        )

    prompt = (
        "You are a real-time delivery coach whispering in the user's ear. The "
        "user THEMSELF just spoke; coach HOW they came across, not what to say "
        f"back. {stance}\n\n"
        f"The user's role in this conversation is: {role}.\n"
        "Produce ONE nudge: an imperative course-correction of at most 6 "
        "words, instantly absorbable mid-conversation (e.g. \"ease up\", "
        "\"that sounded blaming — soften\", \"good — hold that tone\", \"be "
        "firmer, don't back down\", \"stop apologizing\"). Only speak when "
        "something should change: if the delivery does NOT need adjusting, "
        "return an empty string for the nudge.\n"
        "Return ONLY a JSON object with key \"nudge\" (the string above, or "
        "\"\" when nothing should change) and \"importance\" (an integer "
        "0-100: how urgently the user needs THIS nudge right now — 0 when the "
        "nudge is empty). No other keys — the pipeline reads only these two, "
        "so anything else is wasted latency on a real-time whisper."
    )
    # Append the voice-profile few-shot block AFTER the output contract, exactly
    # as empathy_system_prompt does — same helper, same byte-identical-when-None
    # property (no profile → the prompt is unchanged).
    if voice_profile is not None:
        block = _render_voice_profile(voice_profile)
        if block:
            prompt += "\n\n" + block
    return prompt


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def get_llm_client() -> LLMClient:
    return app.state.llm_client


class _LLMResponseError(Exception):
    """A retryable LLM parse/shape failure carrying the honest 502 detail.

    Raised inside the analyze/counterfactual completion helpers so the caller can
    retry once and, on a second failure, surface the SAME specific detail the
    non-retried path used to (e.g. "misaligned analysis" vs "invalid JSON")."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences.

    Raises ValueError when the provider returned no text at all (e.g. an
    OpenAI content-filter/refusal yields ``message.content is None``) so callers
    surface an honest 502 rather than a raw 500 from ``None.strip()``.
    """
    if not text:
        raise ValueError("LLM returned no content")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


# ---------------------------------------------------------------------------
# Rate limiting (P1-5) — per-IP, in-process, dependency-free
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Fixed-window per-key request counter for cost-exhaustion protection.

    In-process only (one worker) — honest about its scope: it protects a
    single process, not a cluster, but needs no external store and cannot
    itself fail a request for infrastructure reasons. A production multi-worker
    deployment would move this to a shared store; that is a flagged decision.
    """

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window_s = window_s
        # key -> (window_start_monotonic, count_in_window)
        self._hits: dict[str, tuple[float, int]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """Record a hit for *key*; return False once the window limit is passed."""
        async with self._lock:
            now = time.monotonic()
            start, count = self._hits.get(key, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0  # window elapsed — reset
            count += 1
            self._hits[key] = (start, count)
            return count <= self.limit

    def reset(self) -> None:
        """Drop all counters (used by tests to isolate windows)."""
        self._hits.clear()


_rate_limiter = _RateLimiter(RATE_LIMIT_PER_MINUTE)


async def _rate_limit(request: Request) -> None:
    """FastAPI dependency: 429 once a client IP exceeds the per-minute budget."""
    if not RATE_LIMIT_ENABLED:
        return
    client = request.client
    key = client.host if client else "unknown"
    if not await _rate_limiter.allow(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded — too many requests; please slow down.",
        )


# ---------------------------------------------------------------------------
# Recording persistence — store accessor + short-lived media tokens
# ---------------------------------------------------------------------------

def get_recordings_store() -> "recordings_store.RecordingsStore | None":
    """The process-wide recordings store, or ``None`` when storage is disabled.

    Read from ``app.state`` (set in lifespan). ``getattr`` with a None default
    means the test suite — which never runs lifespan — is "storage disabled" by
    default; a test enables storage by injecting a fake into app.state.
    """
    return getattr(app.state, "recordings_store", None)


# Per-process HMAC secret for media-stream tokens. Minted fresh each boot (like
# _REDACT_SALT), so a restart invalidates outstanding links — acceptable for
# 15-minute URLs and cheaper than a shared secret store. Tokens let a media
# element (which cannot send an Authorization header) fetch a private object.
_MEDIA_TOKEN_SECRET = os.urandom(32)
MEDIA_TOKEN_TTL_SECONDS = 900  # 15 minutes


def _b64url(raw: bytes) -> str:
    """URL-safe base64 without padding (safe in a query string)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    """Inverse of :func:`_b64url`, re-adding the stripped padding."""
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _make_media_token(uid: str, recording_id: str, expiry_ts: int) -> str:
    """Mint an opaque token binding (uid, recording_id, expiry) under the
    per-process secret. The uid+expiry travel in the token (the media endpoint
    has no Authorization header to recover them from); the signature covers all
    three so neither can be tampered with, and a token for one recording can
    never address another. Encoded as ``b64(uid).b64(expiry).b64(sig)``."""
    msg = f"{uid}:{recording_id}:{expiry_ts}".encode("utf-8")
    sig = hmac.new(_MEDIA_TOKEN_SECRET, msg, hashlib.sha256).digest()
    return ".".join((
        _b64url(uid.encode("utf-8")),
        _b64url(str(expiry_ts).encode("ascii")),
        _b64url(sig),
    ))


def _verify_media_token(token: str, recording_id: str) -> str | None:
    """Return the token's uid when valid (good signature, unexpired) for THIS
    recording_id, else ``None``. Constant-time signature comparison."""
    try:
        uid_part, exp_part, sig_part = token.split(".")
        uid = _b64url_decode(uid_part).decode("utf-8")
        expiry_ts = int(_b64url_decode(exp_part).decode("ascii"))
        sig = _b64url_decode(sig_part)
    except Exception:  # noqa: BLE001 — any malformed token is simply invalid
        return None
    if expiry_ts < int(time.time()):
        return None  # expired
    msg = f"{uid}:{recording_id}:{expiry_ts}".encode("utf-8")
    expected = hmac.new(_MEDIA_TOKEN_SECRET, msg, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    return uid


def _request_base_url(request: Request) -> str:
    """Absolute scheme://host for building media URLs, honoring Cloud Run's
    X-Forwarded-Proto / X-Forwarded-Host (the app sits behind Google's proxy,
    so request.url.scheme/netloc would otherwise read the internal http/host).
    Takes the first value of any comma-list the proxy may send."""
    proto = (
        request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        or request.url.scheme
    )
    host = (
        request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        or request.headers.get("host", "").strip()
        or request.url.netloc
    )
    return f"{proto}://{host}"


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
            # Probe a real table so an uninitialized/misconfigured DB fails the
            # check — aiosqlite auto-creates the file and `SELECT 1` touches no
            # schema, so it would falsely pass on a fresh/empty database.
            await db.execute("SELECT 1 FROM sessions LIMIT 1")
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
    uid: str,
) -> str | None:
    """Build a relationship context string for LLM prompt enrichment.

    Scoped to ``uid``: a relationship owned by another user resolves to None
    (no context), exactly like a missing one — never leaking another user's
    relationship/participant data into the prompt.
    """
    if not relationship_id or not from_id or not to_id:
        return None
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT type, name FROM relationships WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
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


async def _resolve_voice_profile(
    relationship_id: str | None,
    participant_id: str | None,
    uid: str,
) -> dict | None:
    """Load the voice profile for ``(relationship_id, participant_id)``.

    Returns ``None`` — meaning "no profile, render nothing" — for an incomplete
    key, a missing row, or a stored-but-empty profile. That ``None`` is what
    keeps the prompt byte-identical to today when no profile applies. One
    indexed ``SELECT``; mirrors the guard style of
    ``_resolve_relationship_context``.

    Scoped to ``uid`` via a join on the owning relationship: a voice profile
    whose relationship belongs to another user resolves to None, so one user
    can never load another user's stored voice into their prompt.
    """
    if not relationship_id or not participant_id:
        return None
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT vp.pairs, vp.style_notes, vp.updated_at FROM voice_profiles vp "
            "JOIN relationships r ON r.id = vp.relationship_id "
            "WHERE vp.relationship_id = ? AND vp.participant_id = ? "
            "AND r.user_id = ?",
            (relationship_id, participant_id, uid),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if row is None:
        return None
    try:
        pairs = json.loads(row["pairs"])
    except (ValueError, TypeError):
        pairs = []
    if not isinstance(pairs, list):
        pairs = []
    style_notes = row["style_notes"]
    if not pairs and not style_notes:
        return None
    return {
        "pairs": pairs,
        "style_notes": style_notes,
        "updated_at": row["updated_at"],
    }


TONE_DIMENSIONS = ("warmth", "defensiveness", "sarcasm", "constructiveness", "overall")


def _coerce_score(value: object) -> int | None:
    """Return an int score, accepting whole-number floats the LLM may emit
    (e.g. ``82.0``); return None for anything that isn't a whole number.
    Rejects bool (``True``/``False`` are ints in Python but never a score)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


@app.post("/respond", response_model=RespondResponse)
async def respond(
    req: RespondRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    # The voice profile is keyed on the speaker being coached — the
    # from_participant_id. Absent (or no profile) → None → today's exact prompt.
    # Both lookups are uid-scoped so relationship context never crosses users.
    voice_profile = await _resolve_voice_profile(
        req.relationship_id, req.from_participant_id, uid,
    )
    system = empathy_system_prompt(req.empathy_slider, req.role, voice_profile)

    rel_context = await _resolve_relationship_context(
        req.relationship_id, req.from_participant_id, req.to_participant_id, uid,
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
    except (ValueError, IndexError, KeyError, TypeError):
        # ValueError covers json.JSONDecodeError and the empty-content case;
        # TypeError guards other non-text provider payloads. Honest 502.
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
    # Honest failure: require the five named dimensions as whole-number scores;
    # an empty or wrong-keyed tone_score must 502, never yield a response the
    # client will KeyError on. (all() over an empty dict is vacuously True.)
    coerced: dict[str, int] = {}
    if not isinstance(tone_score, dict):
        tone_score = {}
    missing = []
    for dim in TONE_DIMENSIONS:
        s = _coerce_score(tone_score.get(dim))
        if s is None:
            missing.append(dim)
        else:
            coerced[dim] = s
    if missing:
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid tone_score, missing/invalid: " + ", ".join(missing),
        )
    return RespondResponse(suggestions=suggestions, tone_score=coerced)


@app.post("/score", response_model=ScoreResponse)
async def score(
    req: ScoreRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    system = (
        "You are a tone analysis engine. Analyze the following text and return "
        "ONLY a JSON object with integer scores 0-100 for these dimensions: "
        "warmth, defensiveness, sarcasm, constructiveness, overall. "
        "Higher means more of that quality."
    )

    rel_context = await _resolve_relationship_context(
        req.relationship_id, req.from_participant_id, req.to_participant_id, uid,
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
    except (ValueError, IndexError, KeyError, TypeError):
        # ValueError covers json.JSONDecodeError and the empty-content case;
        # TypeError guards other non-text provider payloads. Honest 502.
        raise HTTPException(status_code=502, detail="LLM returned invalid JSON")

    # Honest failure: a missing dimension must be a 502, never a fabricated 0.
    scores: dict[str, int] = {}
    invalid = []
    for d in TONE_DIMENSIONS:
        s = _coerce_score(data.get(d))
        if s is None:
            invalid.append(d)
        else:
            scores[d] = s
    if invalid:
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM response missing or invalid score dimensions: "
                + ", ".join(invalid)
            ),
        )

    return ScoreResponse(**scores)


# ---------------------------------------------------------------------------
# POST /analyze — conversation-dynamics analysis ("the impartial third chair")
# ---------------------------------------------------------------------------

# The absolute, cross-session heat rubric. Heat is scored against these FIXED
# anchors — NOT normalized to a single conversation's own baseline — so a score
# means the same thing across turns, speakers, AND separate sessions (which is
# what lets longitudinal averages be built later). Embedded VERBATIM in both
# /analyze and /analyze/counterfactual so the two share one thermometer.
# Appended to the user content on the ONE retry after an LLM JSON parse/shape
# failure (see _run_analysis / the counterfactual endpoint). Terse on purpose —
# the model already has the full output contract in its system prompt.
_LLM_JSON_RETRY_SUFFIX = (
    "Your previous reply was not valid JSON. Reply with ONLY the JSON object."
)


HEAT_ANCHOR_RUBRIC = (
    "heat is an integer 0-100 on an ABSOLUTE thermometer of emotional "
    "escalation/hostility. Score each turn against these FIXED anchors — "
    "explicitly NOT relative to this conversation's own baseline:\n"
    "- 0-15: calm, neutral, or warm (e.g. \"Could you hand me that?\" / \"I get "
    "why you'd feel that way\").\n"
    "- 25-40: tension present — clipped, defensive, or mildly sarcastic.\n"
    "- 50-65: clearly heated — blame, raised stakes, \"you always/never\".\n"
    "- 75-90: hostile — contempt, name-calling, shouting-register language.\n"
    "- 95+: abusive or explosive.\n"
    "A conversation between calm people should score low throughout; do not "
    "inflate differences to fill the scale."
)


ANALYZE_SYSTEM_PROMPT = (
    "You are an impartial couples therapist observing a conversation from the "
    "third chair. You read the DYNAMIC between people — never who is right or "
    "wrong — and you never pick a winner.\n\n"
    "You will receive a transcript in which every turn is numbered and tagged "
    "with its speaker, like `0. [Alice] ...`. Analyze EVERY turn.\n\n"
    f"HEAT SCALE — {HEAT_ANCHOR_RUBRIC}\n\n"
    "For each turn, produce:\n"
    "- heat: scored on the absolute anchored HEAT SCALE above.\n"
    "- markers: a list drawn ONLY from this exact vocabulary — criticism, "
    "contempt, defensiveness, stonewalling, repair_attempt, validation. Label "
    "a marker only when it is clearly present; use [] when none apply.\n"
    "- trigger_phrase: the short phrase within THIS turn most likely to have "
    "provoked the other party, or null if none.\n\n"
    "Then, across the whole conversation, produce:\n"
    "- speaker_names: for EACH speaker label, infer that speaker's real name "
    "ONLY from DIRECT evidence in the transcript — being addressed by name "
    "(\"Joe, listen to me\"), signing off (\"— love, Mia\"), or referring to "
    "themselves by name. Each entry is keyed by the EXACT speaker label and is "
    "{name, confidence}, where confidence is exactly one of high, medium, low. "
    "Use \"high\" ONLY when the transcript names that speaker unmistakably. "
    "NEVER guess a name from speaking style, topic, tone, gender, or role — when "
    "there is no direct evidence, return an empty name with confidence low.\n"
    "- requests: the concrete asks each speaker made and how each landed. Each "
    "item is {speaker, request, outcome}, where outcome is exactly one of "
    "granted, denied, deferred, unclear.\n"
    "- narrative: ONE paragraph describing the DYNAMIC between these people. "
    "Lead with their strengths FIRST, then name the friction. Describe the "
    "pattern, never \"X is the problem\". At most 1200 characters.\n"
    "- report_cards: a per-person report card for EVERY speaker, keyed by the "
    "EXACT speaker label used in the transcript. Each card is "
    "{score, headline, did_well, work_on}: score is an integer 0-100 for that "
    "speaker's composure and constructiveness on the SAME absolute scale "
    "(higher = better conduct in this conversation — comparable across people "
    "and across sessions); headline is a <=80-char summary of how they showed "
    "up; did_well (<=200 chars) names a concrete strength; work_on (<=200 "
    "chars) is ONE concrete, actionable thing to change (e.g. \"count to three "
    "before responding to criticism\"), never generic advice. Be honest and "
    "direct, kind but not mushy — do not soften real feedback away.\n\n"
    "Return ONLY a JSON object of exactly this shape, with per_turn holding "
    "one entry per input turn in the SAME order and length, and report_cards "
    "holding one card per distinct speaker:\n"
    '{"per_turn": [{"heat": 0, "markers": [], "trigger_phrase": null}], '
    '"speaker_names": {"Alice": {"name": "", "confidence": "low"}}, '
    '"requests": [{"speaker": "", "request": "", "outcome": "unclear"}], '
    '"narrative": "", '
    '"report_cards": {"Alice": {"score": 0, "headline": "", "did_well": "", '
    '"work_on": ""}}}'
)


# Appended to the analyze system prompt ONLY when the caller asked for a title
# (§1 — an upload with no user-provided title). It requests one extra top-level
# field; the text /analyze path never adds it, so no title is ever fabricated for
# a conversation the user already named or for a non-recording analysis.
ANALYZE_TITLE_PROMPT_ADDENDUM = (
    "\n\nAlso add a top-level \"title\" field: a short, specific title for THIS "
    "conversation, 3-6 words, no surrounding quotes and no trailing punctuation "
    "(e.g. \"Argument about the cat\", \"Planning the weekend trip\"). Base it "
    "only on what was actually discussed — never invent details."
)


# Appended to ANALYZE_SYSTEM_PROMPT ONLY when the transcript carries per-turn
# voice annotations (an /analyze/upload recording with successful prosody). It
# is never added for text /analyze, so that prompt stays byte-identical. It
# tells the model to read DELIVERY alongside words — the whole point of adding
# audio: a shouted line is hotter than its words; a cold, flat, quiet line with
# hostile words is still high heat and often contempt.
ANALYZE_VOICE_PROMPT_ADDENDUM = (
    "\n\nSome turns include a bracketed voice annotation, e.g. "
    "`[voice: loud, fast, pitch varied]`, describing HOW that turn was "
    "delivered (vocal energy, speech rate, pitch movement). These are relative "
    "delivery cues for THIS recording, not absolute measurements. Weigh "
    "delivery ALONGSIDE the words when scoring heat and markers: an aggressive, "
    "raised-voice ('loud') delivery RAISES heat even when the words are mild; a "
    "cold, flat, quiet delivery paired with hostile or dismissive words is "
    "still HIGH heat and often contempt. Let tone corroborate or intensify what "
    "the words imply — never ignore the words themselves."
)


COUNTERFACTUAL_SYSTEM_PROMPT = (
    "You are an experienced couples therapist running a 'what if they'd said it "
    "differently' simulation. You are given a full transcript in which every "
    "turn is numbered and tagged with its speaker, like `0. [Alice] ...`, plus "
    "the index of one PIVOT turn.\n\n"
    "Do BOTH of the following:\n"
    "1. rewritten_text: rewrite ONLY the pivot turn, spoken by the SAME speaker, "
    "expressing the SAME underlying need constructively — a balanced, "
    "empathetic register that keeps their intent and does NOT capitulate for "
    "them. Return just the rewritten words of that one turn.\n"
    "2. simulated_heat: assuming the pivot turn HAD BEEN your rewrite, estimate "
    "the likely heat of each SUBSEQUENT turn's speaker, in the SAME speaker "
    "order as the real transcript, one value per turn from the pivot index "
    "through the last turn (INCLUDING the pivot turn's own rewritten heat, "
    "first). Score every value on this absolute rubric — "
    f"{HEAT_ANCHOR_RUBRIC}\n\n"
    "Also give a rationale (<=200 chars) for WHY this phrasing helps.\n\n"
    "Return ONLY a JSON object of exactly this shape:\n"
    '{"rewritten_text": "", "rationale": "", "simulated_heat": [0]}\n'
    "simulated_heat MUST contain exactly one integer per turn from the pivot "
    "index through the last turn (inclusive) — no more, no fewer."
)


def _clamp_heat(value: object) -> int | None:
    """Coerce an LLM heat to an int in 0-100, or ``None`` if it is not a
    number. Accepts whole-or-fractional floats (LLMs emit both); rejects bool.
    ``None`` is an honest contract violation the caller turns into a 502 —
    never a fabricated score."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0, min(100, int(round(value))))
    return None


def _clean_markers(value: object) -> list[str]:
    """Keep only markers in the exact vocabulary, de-duplicated in first-seen
    order. Anything else (unknown label, non-string, non-list) is dropped —
    the house rule is drop, never invent."""
    if not isinstance(value, list):
        return []
    seen: list[str] = []
    for m in value:
        if isinstance(m, str) and m in ANALYZE_MARKER_VOCAB and m not in seen:
            seen.append(m)
    return seen


def _clean_trigger_phrase(value: object) -> str | None:
    """A non-empty string trigger phrase, else ``None``."""
    if isinstance(value, str) and value.strip():
        return value
    return None


def _clean_requests(value: object) -> list[dict[str, str]]:
    """Normalize the LLM's requests[]: keep well-formed {speaker, request}
    items, coerce an unknown/missing outcome to "unclear". Malformed entries
    are skipped rather than fabricated into the output."""
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker")
        request = item.get("request")
        if not isinstance(speaker, str) or not speaker.strip():
            continue
        if not isinstance(request, str) or not request.strip():
            continue
        outcome = item.get("outcome")
        if outcome not in ANALYZE_REQUEST_OUTCOMES:
            outcome = "unclear"
        out.append(
            {"speaker": speaker, "request": request, "outcome": outcome}
        )
    return out


def _clean_report_card(value: object) -> dict | None:
    """Validate/normalize one speaker's report card, or ``None`` when unusable.

    ``score`` is clamped to 0-100; the three text fields are truncated to their
    caps (§2 — truncate on write, never reject). ``None`` means the LLM omitted
    or malformed this speaker's card entirely — the caller turns that missing
    speaker into an honest 502, never a fabricated card.
    """
    if not isinstance(value, dict):
        return None
    score = _clamp_heat(value.get("score"))  # score shares heat's 0-100 clamp
    if score is None:
        return None
    headline = value.get("headline")
    did_well = value.get("did_well")
    work_on = value.get("work_on")
    if not all(isinstance(s, str) for s in (headline, did_well, work_on)):
        return None
    return {
        "score": score,
        "headline": headline[:REPORT_CARD_HEADLINE_MAX],
        "did_well": did_well[:REPORT_CARD_TEXT_MAX],
        "work_on": work_on[:REPORT_CARD_TEXT_MAX],
    }


def _clean_title(value: object) -> str | None:
    """A non-empty, trimmed, length-capped LLM title, or ``None`` (§1).

    ``None`` (missing/blank/non-string) means the LLM omitted a usable title —
    the caller then falls back to the recording's filename, never a fabricated
    name."""
    if isinstance(value, str) and value.strip():
        return value.strip()[:RECORDING_TITLE_MAX]
    return None


def _clean_speaker_names(value: object) -> dict[str, str]:
    """Map speaker id -> confidently-inferred real name (§2a).

    Keeps ONLY entries whose confidence is exactly ``high`` and whose name is a
    non-empty string (trimmed, capped). Everything else — wrong shape, missing/
    medium/low confidence, empty name — is dropped so that speaker falls through
    to the voice/generic rungs. The house rule holds: drop, never invent."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for speaker, info in value.items():
        if not isinstance(speaker, str) or not isinstance(info, dict):
            continue
        if info.get("confidence") != _NAME_CONFIDENCE_APPLIED:
            continue
        name = info.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if name:
            out[speaker] = name[:SPEAKER_NAME_MAX]
    return out


def _enrolled_speaker(speaker_identity: object) -> str | None:
    """The speaker id matched to the user's enrolled voiceprint, or ``None``.

    Defensive by design: the voice-enrollment pipeline (PR #56) stores
    ``speaker_identity`` as ``{matched_speaker, match_threshold, model,
    speakers: {label: {score, is_you}}}`` and deliberately does NOT set display
    labels itself — this ladder owns presentation. Anything other than a dict
    carrying a non-empty string ``matched_speaker`` (absent field, ``None``, a
    malformed shape) means "no match" and the enrolled rung is skipped."""
    if not isinstance(speaker_identity, dict):
        return None
    matched = speaker_identity.get("matched_speaker")
    if isinstance(matched, str) and matched.strip():
        return matched
    return None


def _resolve_speaker_labels(
    speakers: list[str],
    llm_names: dict[str, str],
    voice_labels: list[dict] | None,
    speaker_identity: dict | None = None,
) -> dict[str, SpeakerLabelOut]:
    """Per-speaker display label via the §2 precedence ladder.

    enrolled > name (§2a) > voice (§2b) > generic (§2c). ``llm_names`` is the
    already-cleaned high-confidence name map; ``voice_labels`` is the per-turn
    prosody labels (``None`` for text /analyze or a decode-degraded upload);
    ``speaker_identity`` is the voice-enrollment pipeline's match object (PR
    #56), read defensively — absent/malformed simply skips the enrolled rung.
    Distinct speakers keep first-seen order for stable rendering.
    """
    distinct = list(dict.fromkeys(speakers))
    labels: dict[str, SpeakerLabelOut] = {
        sp: SpeakerLabelOut(display_label=sp, label_source=LABEL_SOURCE_GENERIC)
        for sp in distinct
    }

    # §2b voice — ONLY when NO speaker earned a confident name (a relative pair
    # label is meaningless mixed with a real name) and prosody is available for
    # exactly two speakers whose median pitches differ meaningfully.
    if not llm_names and voice_labels is not None and len(distinct) == 2:
        pitch = prosody.speaker_median_pitch(speakers, voice_labels)
        voice_pair = prosody.pitch_voice_labels(pitch)
        if voice_pair:
            for sp, lbl in voice_pair.items():
                labels[sp] = SpeakerLabelOut(
                    display_label=lbl, label_source=LABEL_SOURCE_VOICE,
                )

    # §2a name — OVERRIDES voice/generic. Voice only ran when there were no
    # names, so a name never collides with a voice label.
    for sp, name in llm_names.items():
        if sp in labels:  # ignore a name for a speaker not in the transcript
            labels[sp] = SpeakerLabelOut(
                display_label=name, label_source=LABEL_SOURCE_NAME,
            )

    # Top rung — enrolled: a voiceprint match to the viewing user's own enrolled
    # voice wins over EVERYTHING (name/voice/generic). Applied last, and only to
    # a speaker actually in this transcript. Other speakers keep their rungs —
    # "You" + "Higher voice" stays coherent (higher than you).
    enrolled = _enrolled_speaker(speaker_identity)
    if enrolled is not None and enrolled in labels:
        labels[enrolled] = SpeakerLabelOut(
            display_label=ENROLLED_DISPLAY_LABEL,
            label_source=LABEL_SOURCE_ENROLLED,
        )
    return labels


async def _run_analysis(
    turns: list[AnalyzeTurn],
    context: str,
    voice_labels: list[dict] | None = None,
    request_title: bool = False,
    speaker_identity: dict | None = None,
) -> AnalyzeResponse:
    """Shared /analyze pipeline body — one implementation for both endpoints.

    ``voice_labels`` (one dict per turn, from :func:`prosody.label_turns`) is
    the ONLY difference between the text and recording paths: when present, each
    numbered turn gains a bracketed ``[voice: …]`` delivery cue, the system
    prompt gains the voice addendum, and each PerTurnOut carries its labels.
    When ``None`` (text /analyze) the prompt and output are byte-identical to
    before — the working text analyzer cannot regress.

    ``speaker_identity`` is the voice-enrollment pipeline's voiceprint-match
    object (PR #56 — ``{matched_speaker, ...}``), forwarded to the display-label
    ladder so a matched speaker is labeled "You" (source "enrolled"). ``None``
    (every current caller, until the enrollment branch merges and passes it)
    skips that rung — the ladder resolves exactly as before.
    """
    # Total-transcript cap (a 413) — the per-turn length bounds are validation
    # (422); this guards the aggregate size a single LLM pass must carry.
    total_chars = sum(len(t.text) for t in turns)
    if total_chars > ANALYZE_MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"transcript too large: {total_chars} characters exceeds the "
                f"{ANALYZE_MAX_TRANSCRIPT_CHARS} limit"
            ),
        )

    speakers = [t.speaker for t in turns]
    char_counts = [len(t.text) for t in turns]
    starts = [t.start_time for t in turns]
    ends = [t.end_time for t in turns]
    distinct_speakers = len(set(speakers))

    # Number every turn so per_turn alignment is explicit for the model. With
    # voice labels, each line also carries a bracketed delivery cue between the
    # speaker tag and the text: `7. [Bob] [voice: loud, fast, pitch varied] …`.
    numbered_lines = []
    for i, t in enumerate(turns):
        prefix = f"{i}. [{t.speaker}]"
        if voice_labels is not None:
            prefix += f" [voice: {prosody.annotate(voice_labels[i])}]"
        numbered_lines.append(f"{prefix} {t.text}")
    numbered = "\n".join(numbered_lines)
    user_content = (
        f"Conversation ({len(turns)} turns, {distinct_speakers} speakers):\n"
        f"{numbered}"
    )
    if context:
        user_content += f"\n\nContext: {context}"

    # Build the system prompt conditionally: the voice addendum is added ONLY
    # when delivery cues are present, so text /analyze stays byte-identical.
    system_prompt = ANALYZE_SYSTEM_PROMPT
    if voice_labels is not None:
        system_prompt += ANALYZE_VOICE_PROMPT_ADDENDUM
    # §1 — ask for a title only when the caller wants one (an upload with no
    # user-provided title); the text /analyze path leaves this off.
    if request_title:
        system_prompt += ANALYZE_TITLE_PROMPT_ADDENDUM

    # Output budget scales with turn count. Measured reality (production 502
    # caught by the v1.6.0 ship e2e): each per-turn JSON object costs ~60-90
    # output tokens, and report cards + requests + narrative need ~1200 on
    # top — the old 800 + 16/turn budget truncated an 18-turn analysis
    # mid-JSON, which parse-fails into an honest 502. 90/turn + 1200 fits
    # ~77 turns inside the 8192 cap; beyond that the cap wins and a very long
    # transcript may still truncate (chunked analysis is the eventual fix).
    max_tokens = min(8192, 1200 + 90 * len(turns))

    llm = get_llm_client()

    async def _complete_analysis_json(attempt_user: str) -> dict:
        # to_thread: llm.complete is a blocking SDK call — keep it off the event
        # loop (see /respond). Raises _LLMResponseError for any non-JSON /
        # wrong-shape response (not an object, or a per_turn list misaligned with
        # the transcript) so the caller can retry once before surfacing a 502.
        raw = await asyncio.to_thread(
            llm.complete,
            system=system_prompt,
            user=attempt_user,
            max_tokens=max_tokens,
        )
        try:
            parsed = parse_llm_json(raw)
        except (ValueError, IndexError, KeyError, TypeError):
            raise _LLMResponseError("LLM returned invalid JSON")
        # Valid JSON that isn't an object ("[]", "null", a bare number) would
        # AttributeError on .get() downstream — treat it as a parse failure.
        if not isinstance(parsed, dict):
            raise _LLMResponseError("LLM returned invalid JSON")
        # No padding, no truncation: a misaligned per_turn length means the
        # scores cannot be trusted against the transcript at all.
        per_turn_field = parsed.get("per_turn")
        if not isinstance(per_turn_field, list) or len(per_turn_field) != len(turns):
            raise _LLMResponseError("LLM returned misaligned analysis")
        return parsed

    # ~10% of production batch-analysis calls come back non-JSON (or truncated
    # mid-object → invalid JSON). Retry ONCE with a terse corrective suffix
    # before surfacing the honest 502 — this recovers most of those without
    # failing the whole request (or, for an async job, the whole job). The retry
    # re-raises the SAME specific detail so diagnostics stay honest.
    try:
        data = await _complete_analysis_json(user_content)
    except _LLMResponseError:
        logger.info("analysis LLM retry after parse failure")
        try:
            data = await _complete_analysis_json(
                user_content + "\n\n" + _LLM_JSON_RETRY_SUFFIX
            )
        except _LLMResponseError as exc:
            raise HTTPException(status_code=502, detail=exc.detail)

    # Guaranteed a list of the correct length by _complete_analysis_json above.
    llm_per_turn = data["per_turn"]

    # Extract + clean the per-turn LLM fields into parallel arrays.
    heats: list[int] = []
    markers: list[list[str]] = []
    trigger_phrases: list[str | None] = []
    bad_heat_indices: list[int] = []
    for i, entry in enumerate(llm_per_turn):
        entry = entry if isinstance(entry, dict) else {}
        heat = _clamp_heat(entry.get("heat"))
        if heat is None:
            bad_heat_indices.append(i)
            heat = 0  # placeholder; request is about to 502 anyway
        heats.append(heat)
        markers.append(_clean_markers(entry.get("markers")))
        trigger_phrases.append(_clean_trigger_phrase(entry.get("trigger_phrase")))

    if bad_heat_indices:
        raise HTTPException(
            status_code=502,
            detail="LLM returned non-numeric heat at turns: "
            + ", ".join(str(i) for i in bad_heat_indices),
        )

    narrative = data.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        raise HTTPException(
            status_code=502, detail="LLM returned an empty narrative",
        )

    # §1/§2 — additive, presentation-only fields. Both degrade to a safe default
    # (generic labels / no title) if the LLM omits or malforms them, so a missing
    # field is never a 502: the core analysis above is unaffected.
    speaker_labels = _resolve_speaker_labels(
        speakers,
        _clean_speaker_names(data.get("speaker_names")),
        voice_labels,
        speaker_identity=speaker_identity,
    )
    llm_title = _clean_title(data.get("title")) if request_title else None

    # --- Python owns every statistic below (pure functions in dynamics.py) ---
    spikes = dynamics.spike_flags(speakers, heats)
    shares = dynamics.talk_share(speakers, char_counts)
    interruptions = dynamics.count_interruptions(speakers, starts, ends)
    heat_stats = dynamics.speaker_heat_stats(speakers, heats)
    attempts, accepted = dynamics.count_repairs(speakers, heats, markers)
    horsemen = dynamics.count_horsemen(speakers, markers)
    coupling = dynamics.compute_coupling(speakers, heats)
    deescalation = dynamics.compute_deescalation(speakers, heats)
    triggers = dynamics.extract_triggers(speakers, heats, trigger_phrases)

    per_turn = [
        PerTurnOut(
            index=i,
            speaker=speakers[i],
            heat=heats[i],
            markers=markers[i],
            is_spike=spikes[i],
            trigger_phrase=trigger_phrases[i],
            voice=(
                None if voice_labels is None
                else VoiceOut(
                    energy_label=voice_labels[i]["energy_label"],
                    pitch_label=voice_labels[i]["pitch_label"],
                    rate_label=voice_labels[i]["rate_label"],
                )
            ),
        )
        for i in range(len(turns))
    ]

    per_speaker = {
        sp: PerSpeakerOut(
            turns=stats["turns"],
            talk_share=shares[sp],
            avg_heat=stats["avg_heat"],
            peak_heat=stats["peak_heat"],
            peak_turn_index=stats["peak_turn_index"],
            heat_variance=stats["heat_variance"],
            interruptions=(None if interruptions is None else interruptions[sp]),
            horsemen=HorsemenOut(**horsemen[sp]),
            repair_attempts=attempts[sp],
            repairs_accepted=accepted[sp],
        )
        for sp, stats in heat_stats.items()
    }

    # §2 — one report card per speaker. Every request speaker MUST be present;
    # a missing (or malformed) card is an honest 502 misalignment, never a
    # fabricated card. per_speaker's keys are exactly the distinct speakers.
    llm_cards = data.get("report_cards")
    llm_cards = llm_cards if isinstance(llm_cards, dict) else {}
    report_cards: dict[str, ReportCardOut] = {}
    missing_cards: list[str] = []
    for sp in per_speaker:
        cleaned = _clean_report_card(llm_cards.get(sp))
        if cleaned is None:
            missing_cards.append(sp)
        else:
            report_cards[sp] = ReportCardOut(**cleaned)
    if missing_cards:
        raise HTTPException(
            status_code=502,
            detail="LLM returned misaligned report cards; missing/invalid for: "
            + ", ".join(missing_cards),
        )

    # Transparent, LLM-free word metrics (pure — word_metrics.py). Keyed by the
    # same canonical speaker ids as per_speaker; computed straight from the turn
    # text so it is identical on the text /analyze and recording paths.
    metrics = word_metrics_mod.compute_word_metrics(
        [{"speaker": t.speaker, "text": t.text} for t in turns]
    )

    return AnalyzeResponse(
        per_turn=per_turn,
        per_speaker=per_speaker,
        dynamics=DynamicsOut(
            coupling=CouplingOut(**coupling),
            deescalation=DeescalationOut(**deescalation),
            triggers=[TriggerOut(**t) for t in triggers],
            requests=[RequestOut(**r) for r in _clean_requests(data.get("requests"))],
        ),
        narrative=narrative,
        report_cards=report_cards,
        speaker_labels=speaker_labels,
        title=llm_title,
        word_metrics=metrics,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    req: AnalyzeRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    # Text path: no audio, so no voice labels — the shared helper produces the
    # exact same prompt and response it always did.
    return await _run_analysis(req.turns, req.context, voice_labels=None)


# A progress hook the async-job runner injects so the shared pipeline can
# report stage transitions (status, human progress_note, duration_seconds once
# known) without knowing anything about jobs. None on the synchronous paths, so
# they are byte-for-byte unchanged.
JobProgressFn = Callable[[str, Optional[str], Optional[float]], Awaitable[None]]


async def _emit_progress(
    progress: "JobProgressFn | None",
    status: str,
    note: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Call the job progress hook if one is wired; a no-op otherwise."""
    if progress is not None:
        await progress(status, note, duration_seconds)


async def _identify_enrolled_speakers(
    uid: str,
    pcm,
    sr: int | None,
    turns: "list[AnalyzeTurn]",
) -> dict | None:
    """Match each diarized speaker against the user's enrolled voiceprint.

    The auto-label half of voice enrollment. Fully OPTIONAL and best-effort — it
    returns ``None`` (skip, no label) whenever any precondition is missing:
    the voice deps aren't installed, prosody decode failed (``pcm`` is None),
    storage is disabled, or the user hasn't enrolled. Any unexpected failure is
    logged and swallowed — enrollment matching must NEVER sink an analysis.

    On success returns :func:`speaker_id.identify_speakers`'s report; the top rung
    of the label ladder reads ``matched_speaker`` as "You" (label_source
    "enrolled"), and the per-speaker cosine scores are retained for debugging."""
    if pcm is None or sr is None or not speaker_id.is_available():
        return None
    store_backend = get_recordings_store()
    if store_backend is None:
        return None
    try:
        profile = await store_backend.read_voiceprint(uid)
    except Exception:  # noqa: BLE001 — a read failure must not sink analysis
        logger.warning("Voiceprint read failed for uid=%s", uid, exc_info=True)
        return None
    if not profile or not isinstance(profile.get("embedding"), list):
        return None
    import numpy as np

    voiceprint = np.asarray(profile["embedding"], dtype=np.float32)
    try:
        return await asyncio.to_thread(
            speaker_id.identify_speakers,
            pcm, sr, [t.model_dump() for t in turns], voiceprint,
        )
    except Exception:  # noqa: BLE001 — matching is optional; degrade to no label
        logger.warning("Speaker identification failed for uid=%s", uid, exc_info=True)
        return None


async def _analyze_recording_bytes(
    uid: str,
    *,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    context: str,
    consent: bool,
    store: bool,
    source: dict,
    title: str | None = None,
    progress: "JobProgressFn | None" = None,
) -> AnalyzeUploadResponse:
    """Analyze one recording's raw bytes and optionally persist the result.

    This is the WHOLE /analyze/upload pipeline minus the transport-level read +
    size gate: it is shared VERBATIM by the direct ``/analyze/upload`` endpoint
    (which reads a multipart body), the chunked ``/uploads/{id}/complete``
    endpoint (which reassembles ``data`` from GCS parts), and ``/analyze/link``
    (which downloads ``data`` from a URL), so every path runs the exact same
    transcription/prosody/_run_analysis/storage code. ``consent`` and ``store``
    are booleans (each caller parses its own form field / manifest / JSON).
    ``source`` is provenance stored verbatim in the recording's meta.json — a
    future replay feature can stream the user's own hosted copy instead of our
    derivative (``{"type": "upload"|"link", "url": str|None,
    "original_filename": str|None}``).

    ``progress`` is an optional hook the async-job runner injects to record staged
    progress (transcribing → analyzing → storing) as it goes; it is None on the
    synchronous paths, which then run exactly as before.

    Honest failures throughout: transcription unconfigured → 503, undecodable or
    speechless → 422, over the duration cap → 413. A storage failure never sinks
    the analysis — the response returns with stored=false and a note carrying the
    failure's class name.
    """
    # 1) Transcribe the recording (transcribe_prerecorded downmixes to 16 kHz mono
    #    for reliable diarization, then sends that to Deepgram). to_thread: it is a
    #    blocking HTTP call. NOTE: the note is deliberately NOT a byte size — len(data)
    #    is the DOWNLOAD size (a 116MB video), not the amount transcribed, and
    #    surfacing it here read as "116 MB to transcribe" on the client (Bug 4).
    await _emit_progress(progress, "transcribing", "transcribing audio")
    try:
        raw_turns = await asyncio.to_thread(
            transcribe_prerecorded, data, content_type,
        )
    except TranscriptionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except NoSpeechFound as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 2) The recovered conversation must satisfy the same shape rules as text
    #    /analyze (2-10 speakers, 4-400 turns, per-turn length). Reuse the
    #    AnalyzeRequest validators; a violation is an honest 422.
    try:
        analyze_req = AnalyzeRequest(turns=raw_turns, context=context)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "transcribed conversation is out of bounds for analysis "
                f"({exc.error_count()} issue(s)): "
                + "; ".join(e["msg"] for e in exc.errors()[:3])
            ),
        )
    turns = analyze_req.turns

    # 3) Decode to PCM for prosody. If decoding fails we DEGRADE HONESTLY:
    #    transcription already succeeded, so we still analyze — just without
    #    voice labels — and flag it in voice_analysis rather than 422-ing or
    #    inventing prosody. The duration cap (a 413) bounds LLM + prosody COST on
    #    a legitimately-typed but very long recording; it is NOT an upload-size
    #    limit (chunked upload already bounds bytes at 200MB).
    voice_labels: list[dict] | None = None
    voice_note: str | None = None
    decoded_duration: float | None = None
    decoded_pcm = None
    decoded_sr: int | None = None
    try:
        pcm, sr = await asyncio.to_thread(
            decode_to_pcm, data, filename or "",
        )
        decoded_pcm, decoded_sr = pcm, sr
        duration = pcm.shape[0] / sr if sr else 0.0
        decoded_duration = duration
        if duration > MAX_UPLOAD_DURATION_S:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"recording too long: {duration:.0f}s exceeds the "
                    f"{MAX_UPLOAD_DURATION_S:.0f}s limit"
                ),
            )
        features = [
            prosody.turn_features(
                pcm, sr, t.start_time or 0.0, t.end_time or 0.0,
            )
            for t in turns
        ]
        voice_labels = prosody.label_turns(
            features, [t.model_dump() for t in turns],
        )
    except AudioDecodeError as exc:
        voice_labels = None
        voice_note = f"unavailable: {exc}"

    # 4) Run the shared analysis with the voice labels (or None on degrade). The
    #    decoded duration (when we have it) rides along so the client can start
    #    computing an ETA the moment the analysis stage begins.
    await _emit_progress(progress, "analyzing", None, decoded_duration)
    # §1 — when the user did NOT name this recording, ask the SAME analysis call
    # for a short title and use it (save_recording falls back to the filename if
    # the LLM omits one). When the user DID provide a title we neither request nor
    # override it.
    core = await _run_analysis(
        turns, context, voice_labels=voice_labels, request_title=title is None,
    )
    effective_title = title or core.title

    transcribed = [
        TranscribedTurn(
            speaker=t.speaker,
            text=t.text,
            start_time=t.start_time,
            end_time=t.end_time,
        )
        for t in turns
    ]
    response = AnalyzeUploadResponse(
        **core.model_dump(),
        turns=transcribed,
        voice_analysis=voice_note,
    )
    # Surface the EFFECTIVE display title (user-provided, else LLM-suggested, else
    # None) so the analyze-complete UI can show it without a second round-trip.
    response.title = effective_title

    # Enrollment-based identity ("You"). Best-effort: skipped cleanly when the
    # voice deps aren't installed, storage/voiceprint is absent, or decode failed
    # (decoded_pcm is None). Never fails the analysis and never forces a label.
    response.speaker_identity = await _identify_enrolled_speakers(
        uid, decoded_pcm, decoded_sr, turns,
    )

    # Companion P1 — episode segmentation (pure derivation over data this
    # response already carries; no extra LLM call). Computed AFTER
    # speaker_identity so an enrolled match labels its participant "You", and
    # BEFORE persistence so analysis.json stores the episodes verbatim.
    response.episodes = episodes.segment_episodes(
        [t.model_dump() for t in transcribed],
        per_turn=[p.model_dump() for p in response.per_turn],
        speaker_labels={
            k: v.model_dump() for k, v in response.speaker_labels.items()
        },
        speaker_identity=response.speaker_identity,
        title=effective_title,
        gap_seconds=EPISODE_GAP_SECONDS,
    )

    # 5) Consent-gated persistence. Store ONLY when the user consented, did not
    #    opt this recording out, and storage is enabled — and NEVER let a
    #    storage failure sink the analysis (the response is already complete).
    store_backend = get_recordings_store()
    if not consent:
        response.storage_note = "consent not given"
    elif not store:
        response.storage_note = "storage not requested"
    elif store_backend is None:
        response.storage_note = "storage not enabled"
    else:
        await _emit_progress(progress, "storing", None, decoded_duration)
        # We persist compressed DERIVATIVES, never the original bytes (a cost
        # decision): always an AAC audio.m4a, plus a 360p video_360p.mp4 when the
        # input carried video. Build them off the event loop. A failed AUDIO
        # derivative means there is nothing useful to store → honest stored=false;
        # a failed VIDEO derivative degrades to audio-only with a note (replay of
        # the audio still works).
        # Whether this recording SHOULD carry video, from its content-type /
        # filename. build_derivatives receives the ORIGINAL downloaded bytes (the
        # full video) — the 16 kHz mono downmix is built independently inside
        # transcribe_prerecorded and never reaches here. expect_video makes the
        # "missing video with no note" path impossible: a video input that yields
        # no clip ALWAYS comes back with an honest video_note.
        expect_video = _looks_like_video(content_type, filename)
        try:
            derivatives = await asyncio.to_thread(
                build_derivatives, data, expect_video=expect_video,
            )
        except Exception as exc:  # noqa: BLE001 — persistence must not fail analysis
            logger.warning("Derivative transcode failed for uid=%s: %s", uid, exc)
            response.storage_note = f"storage failed: {type(exc).__name__}"
        else:
            # Mark stored=True BEFORE dumping so the persisted analysis blob
            # describes its own stored state truthfully; the except arm below
            # resets it on a save failure. A video-derivative note rides along in
            # storage_note (the recording IS stored — just audio-only).
            response.stored = True
            if derivatives.video_note:
                response.storage_note = derivatives.video_note
            # analysis.json holds the full analysis payload (turns + voice
            # included); turns.json is the transcript alone for the lighter
            # detail read. recording_id inside the blob stays null — the id is
            # the blob's own storage key, returned by GET /recordings/{id}.
            analysis_json = response.model_dump()
            turns_json = [t.model_dump() for t in transcribed]
            # Prefer the decoded duration; fall back to the transcript's last end.
            duration_seconds = decoded_duration
            if duration_seconds is None:
                duration_seconds = max(
                    (t.end_time or 0.0 for t in turns), default=0.0
                ) or None
            try:
                recording_id = await store_backend.save_recording(
                    uid,
                    audio_m4a=derivatives.audio_m4a,
                    video_360p=derivatives.video_360p,
                    original_filename=filename,
                    original_content_type=content_type,
                    original_bytes=len(data),
                    duration_seconds=duration_seconds,
                    turns=turns_json,
                    analysis=analysis_json,
                    source=source,
                    title=effective_title,
                    storage_note=response.storage_note,
                )
                response.recording_id = recording_id
            except Exception as exc:  # noqa: BLE001 — persistence must not fail analysis
                logger.warning("Recording persistence failed for uid=%s: %s", uid, exc)
                response.stored = False
                response.recording_id = None
                # Preserve a video note if there was one; otherwise report the save failure.
                response.storage_note = f"storage failed: {type(exc).__name__}"

    return response


# Container extensions that carry video (used only as a fallback when the
# content-type is generic, e.g. a Google-Photos download served as
# application/octet-stream). Not exhaustive — the probe inside build_derivatives
# is the real detector; this only decides whether an ABSENT video deserves a note.
_VIDEO_EXTS = frozenset({
    ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".hevc", ".3gp", ".mpg",
    ".mpeg", ".mts", ".m2ts",
})


def _looks_like_video(content_type: str | None, filename: str | None) -> bool:
    """True when the input SHOULD carry video, by content-type or filename.

    Drives ``build_derivatives(expect_video=...)`` so a video recording that
    yields no 360p clip is always accompanied by an honest storage note, never a
    silent audio-only drop."""
    if (content_type or "").strip().lower().startswith("video/"):
        return True
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in _VIDEO_EXTS


def _parse_bool_form(value: str) -> bool:
    """Parse a multipart form flag ("true"/"false") into a bool, matching the
    original direct-upload semantics (only an exact case-insensitive "true" is
    True)."""
    return value.strip().lower() == "true"


@app.post("/analyze/upload", response_model=AnalyzeUploadResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    context: str = Form(default="", max_length=500),
    consent: str = Form(default="false"),
    store: str = Form(default="true"),
    title: str = Form(default="", max_length=RECORDING_TITLE_MAX),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Analyze a RECORDING in ONE request: an audio file, or a video whose audio
    track we extract. By default process-and-discard; with explicit consent the
    original bytes + turns + analysis are persisted so the user can replay/delete
    later.

    This DIRECT path is capped at ~25MB (MAX_UPLOAD_BYTES) — Cloud Run's ~32MB
    request limit makes a larger single body impossible anyway. For phone videos
    (routinely 50-300MB) use the CHUNKED endpoints (/uploads/start → PUT chunks →
    /uploads/{id}/complete), which stream 8MB parts and reassemble server-side.
    Both paths converge on :func:`_analyze_recording_bytes`.
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file too large: {len(data)} bytes exceeds the "
                f"{MAX_UPLOAD_BYTES}-byte direct-upload limit — use chunked "
                "upload (POST /uploads/start) for files above "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB"
            ),
        )
    if not data:
        raise HTTPException(status_code=422, detail="empty file")

    return await _analyze_recording_bytes(
        uid,
        data=data,
        filename=file.filename,
        content_type=file.content_type,
        context=context,
        consent=_parse_bool_form(consent),
        store=_parse_bool_form(store),
        source={
            "type": "upload",
            "url": None,
            "original_filename": file.filename,
        },
        title=title or None,
    )


@app.post("/analyze/link", response_model=AnalyzeUploadResponse)
async def analyze_link(
    req: AnalyzeLinkRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Analyze a recording the user links to instead of uploading.

    The server downloads the bytes itself (Drive share links are rewritten to
    their direct-download form; the body is streamed with a hard 200MB cap and a
    10-minute timeout), guarding against SSRF (private/internal addresses are
    rejected) and against a share/preview HTML page being mistaken for a file.
    The downloaded bytes then run the EXACT same analyze+derivative-store
    pipeline as an upload. The ORIGINAL user-provided URL (pre-Drive-rewrite —
    the durable share link) is kept in the recording's source metadata."""
    try:
        # fetch_link is blocking (httpx.Client) — keep it off the event loop.
        data, filename, content_type = await asyncio.to_thread(
            link_fetch.fetch_link, req.url,
        )
    except link_fetch.LinkError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    if not data:
        raise HTTPException(status_code=422, detail="linked file is empty")

    return await _analyze_recording_bytes(
        uid,
        data=data,
        filename=filename,
        content_type=content_type,
        context=req.context,
        consent=req.consent,
        store=req.store,
        source={
            "type": "link",
            # The ORIGINAL pasted URL, not the Drive-rewritten one — that is the
            # durable share link a future replay feature would re-fetch.
            "url": req.url,
            "original_filename": filename,
        },
        title=req.title,
    )


@app.post("/analyze/counterfactual", response_model=CounterfactualResponse)
async def analyze_counterfactual(
    req: CounterfactualRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    turns = req.turns
    pivot_index = req.pivot_index  # already validated in-range (422 otherwise)

    # Same 60k aggregate cap as /analyze (a 413) — one LLM pass, one belt.
    total_chars = sum(len(t.text) for t in turns)
    if total_chars > ANALYZE_MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"transcript too large: {total_chars} characters exceeds the "
                f"{ANALYZE_MAX_TRANSCRIPT_CHARS} limit"
            ),
        )

    speakers = [t.speaker for t in turns]
    pivot = turns[pivot_index]
    numbered = "\n".join(
        f"{i}. [{t.speaker}] {t.text}" for i, t in enumerate(turns)
    )
    user_content = (
        f"Conversation ({len(turns)} turns):\n{numbered}\n\n"
        f"PIVOT turn to rewrite: index {pivot_index}, spoken by "
        f"[{pivot.speaker}]: {pivot.text}"
    )
    if req.context:
        user_content += f"\n\nContext: {req.context}"

    # One heat per turn from the pivot through the end (pivot included).
    expected = len(turns) - pivot_index
    # Output budget scales with the simulated tail plus headroom for the
    # rewrite + rationale, capped so a huge transcript can't request an absurd
    # generation (mirrors /analyze).
    max_tokens = min(8192, 800 + 16 * expected)

    llm = get_llm_client()

    async def _complete_counterfactual_json(attempt_user: str) -> dict:
        # to_thread: keep the blocking SDK call off the event loop (see /respond).
        # Raises _LLMResponseError for a non-JSON / non-object reply so the caller
        # can retry once — same corrective-retry policy as /analyze.
        raw = await asyncio.to_thread(
            llm.complete,
            system=COUNTERFACTUAL_SYSTEM_PROMPT,
            user=attempt_user,
            max_tokens=max_tokens,
        )
        try:
            parsed = parse_llm_json(raw)
        except (ValueError, IndexError, KeyError, TypeError):
            raise _LLMResponseError("LLM returned invalid JSON")
        if not isinstance(parsed, dict):
            raise _LLMResponseError("LLM returned invalid JSON")
        return parsed

    try:
        data = await _complete_counterfactual_json(user_content)
    except _LLMResponseError:
        logger.info("analysis LLM retry after parse failure")
        try:
            data = await _complete_counterfactual_json(
                user_content + "\n\n" + _LLM_JSON_RETRY_SUFFIX
            )
        except _LLMResponseError as exc:
            raise HTTPException(status_code=502, detail=exc.detail)

    rewritten = data.get("rewritten_text")
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise HTTPException(
            status_code=502, detail="LLM returned an empty rewrite",
        )

    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise HTTPException(
            status_code=502, detail="LLM returned an empty rationale",
        )
    rationale = rationale[:COUNTERFACTUAL_RATIONALE_MAX]

    sim = data.get("simulated_heat")
    # Honest failure: a wrong-length simulation cannot be aligned to the tail of
    # the transcript at all — no padding, no truncation.
    if not isinstance(sim, list) or len(sim) != expected:
        raise HTTPException(
            status_code=502, detail="LLM returned misaligned simulation",
        )

    sim_heats: list[int] = []
    bad_heat_indices: list[int] = []
    for offset, value in enumerate(sim):
        heat = _clamp_heat(value)
        if heat is None:
            bad_heat_indices.append(pivot_index + offset)
            heat = 0  # placeholder; request is about to 502 anyway
        sim_heats.append(heat)
    if bad_heat_indices:
        raise HTTPException(
            status_code=502,
            detail="LLM returned non-numeric simulated heat at turns: "
            + ", ".join(str(i) for i in bad_heat_indices),
        )

    simulated_per_turn = [
        CounterfactualPerTurnOut(
            index=pivot_index + offset,
            speaker=speakers[pivot_index + offset],
            heat=sim_heats[offset],
        )
        for offset in range(expected)
    ]

    return CounterfactualResponse(
        pivot_index=pivot_index,
        rewritten_text=rewritten,
        rationale=rationale,
        simulated_per_turn=simulated_per_turn,
        disclaimer=COUNTERFACTUAL_DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# Recordings — list / detail / delete / media (persisted /analyze/upload runs)
# ---------------------------------------------------------------------------

_STORAGE_DISABLED_DETAIL = "recording storage is not enabled"


def _require_store() -> "recordings_store.RecordingsStore":
    """Return the store or raise an honest 503 when storage is disabled."""
    store_backend = get_recordings_store()
    if store_backend is None:
        raise HTTPException(status_code=503, detail=_STORAGE_DISABLED_DETAIL)
    return store_backend


@app.get("/recordings")
async def list_recordings(uid: str = Depends(get_current_uid)):
    """List the caller's stored recordings, newest first. 503 when storage is
    disabled. Scoped to ``uid`` — another user's recordings are never listed."""
    store_backend = _require_store()
    metas = await store_backend.list_recordings(uid)
    return {
        "recordings": [
            {
                "id": m["id"],
                "created_at": m["created_at"],
                "filename": m["filename"],
                # Display name; older recordings written before titles fall back
                # to the filename so the client always has one to render.
                "title": m.get("title") or m["filename"],
                "media_type": m["media_type"],
                "duration_seconds": m.get("duration_seconds"),
                "has_analysis": m.get("has_analysis", False),
                # Honest reason a derivative is absent (e.g. video transcode timed
                # out → audio-only). Surfaced in the LIST so a video link that
                # landed as audio is explained here, not silently dropped.
                "storage_note": m.get("storage_note"),
                # List carries only the source TYPE (upload/link); the full
                # source object (incl. the durable url) is on the detail read.
                "source_type": (m.get("source") or {}).get("type"),
            }
            for m in metas
        ]
    }


@app.get("/recordings/{recording_id}")
async def get_recording(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    """One recording's transcript + full analysis. 404 when it does not exist
    for THIS user (a foreign recording reads as 404, never confirming it)."""
    store_backend = _require_store()
    rec = await store_backend.get_recording(uid, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    # Companion P1 — episodes for the day timeline. New analyses store them in
    # analysis.json; older recordings are segmented on the fly from the stored
    # turns + analysis (episodes.py is pure, so the backfill costs microseconds
    # and no recording needs a migration). None when there is no analysis.
    analysis = rec.get("analysis")
    stored_episodes = (
        analysis.get("episodes") if isinstance(analysis, dict) else None
    )
    if stored_episodes is None:
        stored_episodes = episodes.episodes_from_analysis(
            rec.get("turns", []), analysis, gap_seconds=EPISODE_GAP_SECONDS,
        )
    # Transparent word metrics — same read-path backfill as episodes. New analyses
    # store them in analysis.json; older recordings are recomputed on the fly from
    # the stored turns (word_metrics.py is pure — no LLM, microseconds).
    stored_word_metrics = (
        analysis.get("word_metrics") if isinstance(analysis, dict) else None
    )
    if stored_word_metrics is None:
        stored_word_metrics = word_metrics_mod.compute_word_metrics(
            rec.get("turns", [])
        )
    return {
        "id": rec["id"],
        "created_at": rec["created_at"],
        "filename": rec["filename"],
        # Display name; falls back to the filename for pre-title recordings.
        "title": rec.get("title") or rec["filename"],
        "media_type": rec["media_type"],
        "duration_seconds": rec.get("duration_seconds"),
        # Honest reason a derivative is absent (e.g. video transcode timed out).
        "storage_note": rec.get("storage_note"),
        # When the recording was last re-analyzed (POST …/reanalyze). None for a
        # recording that has only ever had its original analysis.
        "reanalyzed_at": rec.get("reanalyzed_at"),
        "turns": rec.get("turns", []),
        "analysis": analysis,
        "episodes": stored_episodes,
        "word_metrics": stored_word_metrics,
        # Provenance verbatim (type/url/original_filename) so a future replay
        # feature can stream the user's own hosted copy. Metadata only.
        "source": rec.get("source"),
    }


@app.patch("/recordings/{recording_id}")
async def update_recording_title(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    req: RecordingTitleRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Rename a recording (set its user-facing ``title``).

    uid-scoped exactly like PATCH …/source: a missing or foreign recording reads
    as 404 (never confirming another user's recording exists). The title is
    stripped and must be non-empty after stripping (a whitespace-only title is a
    422 via the request model + the guard below), bounded at
    ``RECORDING_TITLE_MAX``. Returns the updated meta (200)."""
    store_backend = _require_store()
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty")
    updated = await store_backend.update_title(uid, recording_id, title)
    if updated is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    return {
        "id": updated["id"],
        "title": updated.get("title") or updated.get("filename"),
    }


@app.delete("/recordings/{recording_id}", status_code=204)
async def delete_recording(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    """Delete a recording and all its objects. 404 when nothing was deleted
    (missing, or owned by another user). 204 on success."""
    store_backend = _require_store()
    deleted = await store_backend.delete_recording(uid, recording_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Recording not found")
    return Response(status_code=204)


@app.get("/recordings/{recording_id}/media_url")
async def get_recording_media_url(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    request: Request,
    uid: str = Depends(get_current_uid),
):
    """Mint a short-lived, absolute media URL a player can hit WITHOUT an
    Authorization header (media elements cannot send one). The token binds the
    caller's uid + this recording + an expiry under the per-process secret; a
    process restart invalidates outstanding links (acceptable for 15-min URLs).
    404 when the recording does not exist for this user."""
    store_backend = _require_store()
    if not await store_backend.recording_exists(uid, recording_id):
        raise HTTPException(status_code=404, detail="Recording not found")
    expiry_ts = int(time.time()) + MEDIA_TOKEN_TTL_SECONDS
    token = _make_media_token(uid, recording_id, expiry_ts)
    base = _request_base_url(request)
    return {
        "url": f"{base}/recordings/{recording_id}/media?tk={token}",
        "expires_in": MEDIA_TOKEN_TTL_SECONDS,
    }


@app.get("/recordings/{recording_id}/source_url")
async def get_recording_source_url(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Resolve the CURRENT direct media URL for a LINK-sourced recording so the
    client can stream the user's own HD original straight from its CDN, falling
    back to our stored derivative on any failure.

    404 (``no remote source for this recording``) when the recording is an upload
    or a link recording whose durable url was not kept — the client then uses the
    derivative path unchanged. Otherwise the stored share URL is re-resolved
    server-side under the FULL SSRF guard: a Google Photos share page is
    re-parsed to its current media URL, a Drive share link is rewritten, a direct
    URL passes through. We do NOT proxy the bytes — only hand back the URL, which
    ``may expire`` (the client refetches, or falls back, on failure).

    A resolution failure — revoked link, changed Photos page format, dead host —
    is surfaced honestly (the resolver's 422/413, or a 502 for an unexpected
    upstream error) so the client falls back to the stored derivative rather than
    seeing a broken player."""
    store_backend = _require_store()
    rec = await store_backend.get_recording(uid, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    source = rec.get("source") or {}
    url = source.get("url")
    if source.get("type") != "link" or not url:
        raise HTTPException(
            status_code=404, detail="no remote source for this recording",
        )
    try:
        # resolve_media_url is blocking (httpx.Client for the Photos re-parse) —
        # keep it off the event loop, matching fetch_link on /analyze/link.
        direct_url, content_type = await asyncio.to_thread(
            link_fetch.resolve_media_url, url,
        )
    except link_fetch.LinkError as exc:
        # Honest, expected resolution failure (blocked host, revoked/unparseable
        # link) — surface the resolver's status so the client falls back.
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 — resolver hit an unexpected upstream
        logger.warning(
            "source_url resolution failed for %s: %s", recording_id, exc,
        )
        raise HTTPException(
            status_code=502,
            detail="couldn't reach the linked source — falling back to the "
            "stored copy",
        )
    return {
        "url": direct_url,
        "content_type": content_type,
        "expires_hint": "may expire; refetch on failure",
    }


@app.patch("/recordings/{recording_id}/source")
async def update_recording_source(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    req: RecordingSourceRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Attach (or replace) a recording's HD source link after the fact.

    The flow: the user records in-app (we store a compressed derivative), the
    original later backs up to their own cloud (Google Photos/Drive), then they
    paste that durable share link onto the existing recording so replay can
    stream their HD original instead of our derivative.

    We validate the link by RESOLVING it (``link_fetch.resolve_media_url`` in a
    worker thread) — an unusable link (not media / a multi-item Photos album /
    an SSRF-blocked or dead host) surfaces its honest LinkError as the resolver's
    status (422/413) with the existing user-facing detail. We deliberately do
    NOT download or re-analyze: the analysis came from the uploaded copy, and
    the link is playback-only provenance. NOTE: we cannot cheaply verify the
    link points at the SAME video (no content check without downloading), so we
    trust the owner here — an acceptable trade for a user attaching their own
    backup, but stated plainly rather than pretended away.

    404 when the recording does not exist for this user (a foreign recording
    reads as 404, never confirming it). On success meta.json's source becomes
    ``{"type": "link", "url": <original pasted url>, "original_filename":
    <unchanged>}`` and the updated source object is returned (200). Replacing is
    just another PATCH — the new source overwrites the old."""
    store_backend = _require_store()
    rec = await store_backend.get_recording(uid, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    try:
        # resolve_media_url is blocking (httpx.Client for the Photos re-parse) —
        # keep it off the event loop, matching /analyze/link and source_url. We
        # discard the resolved URL: this call is validation only (the client
        # re-resolves the durable link at replay time via GET .../source_url).
        await asyncio.to_thread(link_fetch.resolve_media_url, req.url)
    except link_fetch.LinkError as exc:
        # Honest, expected failure (not media / blocked / dead) — surface the
        # resolver's status + user-facing detail; meta.json is left UNCHANGED.
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    # Preserve the recording's existing original_filename; only the type+url
    # change. The stored url is the ORIGINAL pasted share link (durable), which
    # source_url re-resolves to a current direct media URL at replay time.
    existing_source = rec.get("source") or {}
    new_source = {
        "type": "link",
        "url": req.url,
        "original_filename": existing_source.get("original_filename")
        or rec.get("original_filename"),
    }
    updated = await store_backend.update_source(uid, recording_id, new_source)
    if updated is None:
        # Raced with a delete between the read above and the write — honest 404.
        raise HTTPException(status_code=404, detail="Recording not found")
    return updated


# NOTE: deliberately NOT rate-limited (no _rate_limit dependency). A media
# element seeking through audio/video fires many small Range requests in quick
# succession; the per-IP limiter would 429 normal playback into uselessness.
# Access is instead gated by the unforgeable, short-lived, uid+recording-bound
# token — and it is a pure GCS read, not an LLM-cost endpoint. Also NO auth
# dependency: a <video>/<audio> src cannot carry an Authorization header, so the
# token in the query string is the sole credential.
@app.get("/recordings/{recording_id}/media")
async def get_recording_media(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    request: Request,
    tk: str = Query(default=""),
):
    uid = _verify_media_token(tk, recording_id)
    if uid is None:
        raise HTTPException(status_code=403, detail="invalid or expired token")
    store_backend = _require_store()
    result = await store_backend.open_media_stream(
        uid, recording_id, request.headers.get("range"),
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    iterator, status, headers = result
    # Content-Type is already in `headers` (read from the stored blob metadata),
    # so it is not passed again as media_type — avoids a duplicated header.
    return StreamingResponse(iterator, status_code=status, headers=headers)


# ---------------------------------------------------------------------------
# Chunked upload sessions — start / PUT chunk / complete / abort
# ---------------------------------------------------------------------------
# Phone videos are routinely 50-300MB; Cloud Run's ~32MB per-request limit hard-
# rejects such a body before FastAPI sees it, so the direct /analyze/upload path
# can never carry one. These endpoints stream the file in 8MB parts, stash them
# in GCS, reassemble server-side, then run the SAME analysis pipeline. Session
# state (manifest + parts) lives in GCS, so the chunked path REQUIRES a bucket:
# without one, /uploads/start returns an honest 503.

_LARGE_UPLOAD_DISABLED_DETAIL = (
    "large uploads need storage enabled — set MINDSHIFT_RECORDINGS_BUCKET (the "
    "direct /analyze/upload path remains available for files up to "
    f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB)"
)


def _require_store_for_uploads() -> "recordings_store.RecordingsStore":
    """Return the store, or raise the chunked-upload-specific 503 when storage
    is disabled. Distinct message from the recordings 503 so the client knows the
    *large upload* feature (not replay) is what needs the bucket."""
    store_backend = get_recordings_store()
    if store_backend is None:
        raise HTTPException(status_code=503, detail=_LARGE_UPLOAD_DISABLED_DETAIL)
    return store_backend


@app.post("/uploads/start", response_model=UploadStartResponse)
async def start_upload(
    req: UploadStartRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Open a chunked-upload session for a large recording.

    Validates the declared total size (<=200MB, else a plain-message 413),
    writes a manifest to GCS, and tells the client the fixed 8MB chunk size and
    how many chunks to expect. 503 when storage is disabled (the manifest + parts
    have nowhere to live)."""
    store_backend = _require_store_for_uploads()
    if req.total_bytes > MAX_CHUNKED_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"upload too large: {req.total_bytes} bytes exceeds the "
                f"{MAX_CHUNKED_UPLOAD_BYTES // (1024 * 1024)}MB limit"
            ),
        )

    upload_id = str(uuid.uuid4())
    # ceil division — a final short chunk still counts. total_bytes > 0 (model),
    # so expected_chunks is always >= 1.
    expected_chunks = (req.total_bytes + UPLOAD_CHUNK_BYTES - 1) // UPLOAD_CHUNK_BYTES
    manifest = {
        "filename": req.filename,
        "content_type": req.content_type,
        "total_bytes": req.total_bytes,
        "chunk_bytes": UPLOAD_CHUNK_BYTES,
        "expected_chunks": expected_chunks,
        "context": req.context,
        "consent": req.consent,
        "store": req.store,
        # Optional user title carried to complete()/complete-job; None → filename.
        "title": req.title,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await store_backend.write_upload_manifest(uid, upload_id, manifest)
    return UploadStartResponse(
        upload_id=upload_id,
        chunk_bytes=UPLOAD_CHUNK_BYTES,
        expected_chunks=expected_chunks,
    )


# NOTE: deliberately NOT rate-limited (no _rate_limit dependency), mirroring the
# /media precedent. A single 200MB upload fires up to 25 sequential PUTs in quick
# succession; the per-IP minute budget would 429 a normal large upload into
# uselessness. This is a pure GCS write (no LLM cost) and is still fully gated:
# auth (uid from the verified token) + an existing uid-scoped manifest + the
# per-chunk size cap. The cost-bearing work (transcription/LLM) happens only at
# /complete, which IS rate-limited.
@app.put("/uploads/{upload_id}/chunks/{index}")
async def put_upload_chunk(
    upload_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    index: int,
    request: Request,
    uid: str = Depends(get_current_uid),
):
    """Store one part of an in-progress upload. Idempotent: re-PUTting the same
    index overwrites. 404 for an unknown (or foreign) upload_id, 409 for an index
    outside the manifest's expected range, 413 for an oversize chunk."""
    if index < 0:
        raise HTTPException(status_code=422, detail="chunk index must be >= 0")
    store_backend = _require_store_for_uploads()
    manifest = await store_backend.read_upload_manifest(uid, upload_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="unknown upload")
    expected_chunks = manifest["expected_chunks"]
    if index >= expected_chunks:
        raise HTTPException(
            status_code=409,
            detail=(
                f"chunk index {index} out of range for a {expected_chunks}-chunk "
                "upload"
            ),
        )

    limit = manifest.get("chunk_bytes", UPLOAD_CHUNK_BYTES) + CHUNK_SLACK_BYTES
    # Cheap pre-read gate on the declared Content-Length so a huge body is
    # rejected before it is buffered. The post-read len() check below is the
    # authoritative one (a chunked transfer may omit Content-Length).
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > limit:
                raise HTTPException(
                    status_code=413,
                    detail=f"chunk too large: exceeds {limit}-byte limit",
                )
        except ValueError:
            pass  # malformed header — fall through to the authoritative check

    body = await request.body()
    if not body:
        raise HTTPException(status_code=422, detail="empty chunk")
    if len(body) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"chunk too large: {len(body)} bytes exceeds {limit}-byte limit",
        )

    await store_backend.write_upload_part(uid, upload_id, index, body)
    return {"index": index, "received_bytes": len(body)}


@app.post("/uploads/{upload_id}/complete", response_model=AnalyzeUploadResponse)
async def complete_upload(
    upload_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Reassemble a completed upload and run the FULL analysis pipeline.

    Verifies every part is present (400 listing the missing indexes) and the
    reassembled size matches the manifest (400), assembles the original bytes
    (GCS compose, <=32 parts), then hands them to the shared
    :func:`_analyze_recording_bytes` honoring the manifest's consent/store/
    context. The parts + manifest are cleaned up best-effort afterward. Returns
    the same AnalyzeUploadResponse as the direct path. Long-running: transcribing
    a 200MB video can take minutes (Deepgram pre-recorded timeout is 600s)."""
    store_backend = _require_store_for_uploads()
    manifest = await _validate_upload_complete(store_backend, uid, upload_id)
    expected_chunks = manifest["expected_chunks"]

    data = await store_backend.assemble_upload(uid, upload_id, expected_chunks)
    if not data:
        raise HTTPException(status_code=400, detail="assembled upload is empty")

    try:
        response = await _analyze_recording_bytes(
            uid,
            data=data,
            filename=manifest.get("filename"),
            content_type=manifest.get("content_type"),
            context=manifest.get("context", "") or "",
            consent=bool(manifest.get("consent", False)),
            store=bool(manifest.get("store", True)),
            source={
                "type": "upload",
                "url": None,
                "original_filename": manifest.get("filename"),
            },
            title=manifest.get("title"),
        )
    finally:
        # Best-effort cleanup: the reassembled bytes are analyzed (and, on
        # consent, already persisted under recordings/), so the temporary
        # parts/manifest are dead weight regardless of success. A cleanup failure
        # must never mask the analysis result or its error.
        try:
            await store_backend.cleanup_upload(uid, upload_id)
        except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
            logger.warning(
                "Upload cleanup failed for uid=%s upload_id=%s: %s",
                uid, upload_id, exc,
            )

    return response


@app.delete("/uploads/{upload_id}", status_code=204)
async def abort_upload(
    upload_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Abort an in-progress upload and delete its parts + manifest. Idempotent:
    always 204, even for an unknown/foreign upload_id (cleanup is uid-scoped, so
    it simply deletes nothing and never confirms another user's upload)."""
    store_backend = _require_store_for_uploads()
    await store_backend.cleanup_upload(uid, upload_id)
    return Response(status_code=204)


async def _validate_upload_complete(
    store_backend: "recordings_store.RecordingsStore",
    uid: str,
    upload_id: str,
) -> dict:
    """Read a chunked upload's manifest and verify it is ready to assemble.

    Returns the manifest, or raises the SAME 404/400s as :func:`complete_upload`
    (unknown upload, missing chunk indexes, size mismatch). Does NOT assemble the
    bytes — the caller does that when it is ready to run analysis. Shared by the
    synchronous complete endpoint and the async job endpoint so both validate
    identically before doing any expensive work."""
    manifest = await store_backend.read_upload_manifest(uid, upload_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="unknown upload")

    expected_chunks = manifest["expected_chunks"]
    total_bytes = manifest["total_bytes"]
    part_sizes = await store_backend.get_upload_part_sizes(uid, upload_id)
    missing = [i for i in range(expected_chunks) if i not in part_sizes]
    if missing:
        # Honest, actionable: name exactly which chunks to (re-)PUT before retry.
        shown = ", ".join(str(i) for i in missing[:50])
        more = "" if len(missing) <= 50 else f" (+{len(missing) - 50} more)"
        raise HTTPException(
            status_code=400,
            detail=f"upload incomplete — missing chunk index(es): {shown}{more}",
        )

    assembled_bytes = sum(part_sizes.values())
    if assembled_bytes != total_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"assembled size mismatch: parts total {assembled_bytes} bytes "
                f"but manifest declared {total_bytes}"
            ),
        )
    return manifest


# ---------------------------------------------------------------------------
# Async analysis jobs — submit-and-poll with staged progress (see the models
# above). A link download or chunked-upload completion is a multi-minute
# synchronous request whose response an Android phone routinely loses to
# backgrounding/socket-death; these run the SAME pipeline as a background task
# and record staged progress the client polls.
# ---------------------------------------------------------------------------

_JOBS_DISABLED_DETAIL = (
    "async analysis jobs need storage enabled — set MINDSHIFT_RECORDINGS_BUCKET "
    "(the synchronous /analyze/link and /uploads/{id}/complete endpoints remain "
    "available)"
)

# Background job tasks are held here so the event loop keeps a strong reference
# (asyncio only weak-refs tasks) until they finish. HONEST MVP CAVEAT: a job is
# an in-process asyncio task, so an instance restart orphans it — the task never
# writes "failed", and the client's poll then sees a stale non-terminal state
# that GET /analyze/jobs/{id} reports as "stalled" (never an eternal spinner).
# Acceptable at min-instances 1; a durable queue is the production upgrade.
_JOB_TASKS: set[asyncio.Task] = set()


def _new_job_state() -> dict:
    """A fresh queued job-state document (all timestamps = now)."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "stage_started_at": now,
        "progress_note": None,
        "duration_seconds": None,
        # Populated during the download stage (see analyze_link_job); additive.
        "bytes_downloaded": None,
        "bytes_total": None,
        "error": None,
        "result": None,
    }


def _spawn_job(coro) -> "asyncio.Task":
    """Run a job coroutine as a tracked background task (strong-ref held)."""
    task = asyncio.create_task(coro)
    _JOB_TASKS.add(task)
    task.add_done_callback(_JOB_TASKS.discard)
    return task


def _parse_iso(value) -> "datetime | None":
    """Parse an ISO-8601 timestamp to an aware UTC datetime, or None if unusable
    (a naive value is assumed UTC — every timestamp we write carries a tz)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@asynccontextmanager
async def _job_heartbeat(
    store_backend: "recordings_store.RecordingsStore",
    uid: str,
    job_id: str,
    state: dict,
):
    """Keep a running job's ``updated_at`` fresh while it sits in one long blocking
    stage, so GET /analyze/jobs' >120s "stalled" heuristic doesn't false-positive
    on a legitimately-slow download or transcode.

    Every :data:`JOB_HEARTBEAT_SECONDS` it re-writes the CURRENT ``state`` (which
    the owning task's set_stage mutates in place, and which the download progress
    hook mutates with bytes_downloaded/bytes_total) with a bumped ``updated_at`` —
    no fabricated progress, just an honest "still working" signal. A write failure
    is logged, never raised, and the beat loop is cancelled on exit."""
    async def _beat() -> None:
        while True:
            await asyncio.sleep(JOB_HEARTBEAT_SECONDS)
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                await store_backend.write_job_state(uid, job_id, dict(state))
            except Exception:  # noqa: BLE001 — heartbeat is best-effort
                logger.debug(
                    "job heartbeat write failed for %s", job_id, exc_info=True,
                )

    task = asyncio.create_task(_beat())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _run_analysis_job(
    uid: str,
    job_id: str,
    store_backend: "recordings_store.RecordingsStore",
    state: dict,
    *,
    prepare: "Callable[[JobProgressFn], Awaitable[tuple]]",
    context: str,
    consent: bool,
    store: bool,
    title: str | None = None,
    cleanup: "Callable[[], Awaitable[None]] | None" = None,
    on_success: "Callable[[AnalyzeUploadResponse, JobProgressFn], Awaitable[None]] | None" = None,
) -> None:
    """Run one analysis job to a terminal state, writing staged progress.

    ``prepare`` produces ``(data, filename, content_type, source)`` — the link
    download, the upload reassembly, or (re-analysis) the stored audio bytes —
    reporting its own "downloading" stage via the passed hook. The shared
    :func:`_analyze_recording_bytes` then runs with that same hook (transcribing →
    analyzing → storing). ``on_success`` is an optional hook run with the finished
    response + the stage hook BEFORE the done-state is written — re-analysis uses
    it to overwrite the existing recording in place (the pipeline itself runs with
    ``store=False``, so nothing new is persisted); it may raise an HTTPException to
    fail the job honestly, and any mutation it makes to the response is reflected in
    the stored ``result``. On success the full response is written under ``result``
    with status "done"; on any failure the state is written with status "failed"
    and the SAME honest detail the synchronous path would 4xx/5xx with. ``cleanup``
    (upload parts) always runs.
    """
    async def set_stage(
        status: str,
        note: str | None = None,
        duration: float | None = None,
    ) -> None:
        # The owning task is the sole writer, so full-document overwrite is safe.
        state["status"] = status
        now = datetime.now(timezone.utc).isoformat()
        state["stage_started_at"] = now
        state["updated_at"] = now
        if note is not None:
            state["progress_note"] = note
        if duration is not None:
            state["duration_seconds"] = duration
        await store_backend.write_job_state(uid, job_id, dict(state))

    try:
        try:
            # The heartbeat keeps updated_at fresh through prepare()'s download and
            # _analyze_recording_bytes' transcode/store, so neither long blocking
            # stage reads as "stalled" while it is genuinely still running.
            async with _job_heartbeat(store_backend, uid, job_id, state):
                data, filename, content_type, source = await prepare(set_stage)
                response = await _analyze_recording_bytes(
                    uid,
                    data=data,
                    filename=filename,
                    content_type=content_type,
                    context=context,
                    consent=consent,
                    store=store,
                    source=source,
                    title=title,
                    progress=set_stage,
                )
                if on_success is not None:
                    await on_success(response, set_stage)
        except HTTPException as exc:
            await _write_job_failed(store_backend, uid, job_id, state, str(exc.detail))
            return
        except Exception as exc:  # noqa: BLE001 — a background task must not crash silently
            logger.exception("Analysis job %s failed for uid=%s", job_id, uid)
            await _write_job_failed(
                store_backend, uid, job_id, state,
                f"analysis failed: {type(exc).__name__}",
            )
            return

        now = datetime.now(timezone.utc).isoformat()
        state["status"] = "done"
        state["updated_at"] = now
        state["stage_started_at"] = now
        state["progress_note"] = None
        state["error"] = None
        state["result"] = response.model_dump()
        try:
            await store_backend.write_job_state(uid, job_id, dict(state))
        except Exception:  # noqa: BLE001 — nothing left to salvage the result into
            logger.exception(
                "Failed to persist done-state for job %s uid=%s", job_id, uid,
            )
    finally:
        if cleanup is not None:
            try:
                await cleanup()
            except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
                logger.warning(
                    "Job cleanup failed for uid=%s job_id=%s: %s",
                    uid, job_id, exc,
                )


async def _write_job_failed(
    store_backend: "recordings_store.RecordingsStore",
    uid: str,
    job_id: str,
    state: dict,
    detail: str,
) -> None:
    """Write a job's failed terminal state with an honest error detail."""
    now = datetime.now(timezone.utc).isoformat()
    state["status"] = "failed"
    state["updated_at"] = now
    state["stage_started_at"] = now
    state["error"] = detail
    state["result"] = None
    try:
        await store_backend.write_job_state(uid, job_id, dict(state))
    except Exception:  # noqa: BLE001 — the poll will report "stalled" if this fails
        logger.exception(
            "Failed to persist failed-state for job %s uid=%s", job_id, uid,
        )


@app.post("/analyze/link/jobs", response_model=JobCreatedResponse, status_code=202)
async def analyze_link_job(
    req: AnalyzeLinkRequest,
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Submit a link analysis as a background job → 202 {job_id}. Poll
    GET /analyze/jobs/{job_id} for staged progress and the final result. The job
    runs the EXACT pipeline as the synchronous /analyze/link (SSRF/size/HTML
    guards included), just decoupled from a single long-lived connection. 503
    when storage is disabled (jobs have nowhere to live — the synchronous
    endpoint still works)."""
    store_backend = get_recordings_store()
    if store_backend is None:
        raise HTTPException(status_code=503, detail=_JOBS_DISABLED_DETAIL)

    job_id = str(uuid.uuid4())
    state = _new_job_state()
    await store_backend.write_job_state(uid, job_id, state)

    url = req.url

    async def _prepare(set_stage: "JobProgressFn") -> tuple:
        await set_stage("downloading", "fetching video", None)

        def _on_progress(done: int, total: "int | None") -> None:
            # Runs in the fetch worker thread; a plain int assignment into the
            # shared state dict is GIL-atomic. The heartbeat (event loop) reads and
            # writes these out — so the client sees live download progress without
            # this thread touching GCS or the loop.
            state["bytes_downloaded"] = done
            state["bytes_total"] = total

        try:
            # fetch_link is blocking (httpx.Client) — keep it off the event loop.
            data, filename, content_type = await asyncio.to_thread(
                link_fetch.fetch_link, url, progress_cb=_on_progress,
            )
        except link_fetch.LinkError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        if not data:
            raise HTTPException(status_code=422, detail="linked file is empty")
        # Honest label: this size is the fetched VIDEO, not what gets transcribed
        # (we downmix to a small audio stream before Deepgram) — Bug-4 fix so the
        # client no longer renders the download size as "N MB to transcribe".
        mb = len(data) // (1024 * 1024)
        state["bytes_downloaded"] = len(data)
        state["bytes_total"] = state.get("bytes_total") or len(data)
        await set_stage("downloading", f"fetched video ({mb} MB)", None)
        return data, filename, content_type, {
            # The ORIGINAL pasted URL (pre-Drive-rewrite), matching /analyze/link.
            "type": "link", "url": url, "original_filename": filename,
        }

    _spawn_job(_run_analysis_job(
        uid, job_id, store_backend, state,
        prepare=_prepare, context=req.context, consent=req.consent,
        store=req.store, title=req.title,
    ))
    return JobCreatedResponse(job_id=job_id)


@app.post(
    "/uploads/{upload_id}/complete/jobs",
    response_model=JobCreatedResponse,
    status_code=202,
)
async def complete_upload_job(
    upload_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Submit a chunked-upload completion as a background job → 202 {job_id}.

    Validates every part is present FIRST (the same 404/400 as the synchronous
    /complete), so a malformed upload fails fast and synchronously; only a
    ready-to-assemble upload spawns the job. The job reassembles + runs the same
    pipeline, then cleans up the parts. Poll GET /analyze/jobs/{job_id}."""
    store_backend = _require_store_for_uploads()
    manifest = await _validate_upload_complete(store_backend, uid, upload_id)
    expected_chunks = manifest["expected_chunks"]

    job_id = str(uuid.uuid4())
    state = _new_job_state()
    await store_backend.write_job_state(uid, job_id, state)

    async def _prepare(set_stage: "JobProgressFn") -> tuple:
        await set_stage("downloading", "reassembling upload", None)
        data = await store_backend.assemble_upload(uid, upload_id, expected_chunks)
        if not data:
            raise HTTPException(status_code=400, detail="assembled upload is empty")
        return data, manifest.get("filename"), manifest.get("content_type"), {
            "type": "upload", "url": None,
            "original_filename": manifest.get("filename"),
        }

    async def _cleanup() -> None:
        await store_backend.cleanup_upload(uid, upload_id)

    _spawn_job(_run_analysis_job(
        uid, job_id, store_backend, state,
        prepare=_prepare,
        context=manifest.get("context", "") or "",
        consent=bool(manifest.get("consent", False)),
        store=bool(manifest.get("store", True)),
        title=manifest.get("title"),
        cleanup=_cleanup,
    ))
    return JobCreatedResponse(job_id=job_id)


@app.post(
    "/recordings/{recording_id}/reanalyze",
    response_model=JobCreatedResponse,
    status_code=202,
)
async def reanalyze_recording(
    recording_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    """Re-run the CURRENT full analysis pipeline over a stored recording, as a
    submit-and-poll background job → 202 {job_id}. Poll GET /analyze/jobs/{job_id}
    for staged progress and the final result.

    "Re-analyze" means re-running from the stored AUDIO derivative (audio.m4a),
    NOT merely re-scoring the old transcript: transcription + diarization +
    prosody + voice-enrollment matching + episodes + word metrics ALL re-run, so a
    recording benefits from every pipeline improvement made since it was first
    analyzed. The result OVERWRITES analysis.json + turns.json in place and stamps
    meta.reanalyzed_at; the recording's id, title, source, and stored derivatives
    are preserved (recordings_store.overwrite_analysis).

    503 when storage is disabled (a job has nowhere to live); uid-scoped 404 for
    an unknown/foreign recording (never confirming another user's); 422 when the
    recording has no stored audio to re-analyze."""
    store_backend = get_recordings_store()
    if store_backend is None:
        raise HTTPException(status_code=503, detail=_JOBS_DISABLED_DETAIL)

    # Existence (404) and stored-audio (422) are decided synchronously — a job is
    # spawned only for a recording we can actually re-analyze. The audio is read
    # here (to make the 422 honest) and handed straight to the job.
    rec = await store_backend.get_recording(uid, recording_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Recording not found")
    audio_bytes = await store_backend.get_audio_bytes(uid, recording_id)
    if not audio_bytes:
        raise HTTPException(
            status_code=422,
            detail="recording has no stored audio to re-analyze",
        )

    # Preserve the recording's own metadata across the re-run (title is passed to
    # the pipeline so no NEW title is requested; source is stamped back verbatim).
    preserved_title = rec.get("title")
    preserved_source = rec.get("source")

    job_id = str(uuid.uuid4())
    state = _new_job_state()
    await store_backend.write_job_state(uid, job_id, state)

    async def _prepare(set_stage: "JobProgressFn") -> tuple:
        # Audio already in hand — hand it straight to the shared pipeline. The
        # filename/content-type describe the stored AAC derivative so decode +
        # transcription treat it correctly.
        await set_stage("transcribing", "re-analyzing recording", None)
        return audio_bytes, "audio.m4a", "audio/mp4", preserved_source

    async def _persist(
        response: AnalyzeUploadResponse, set_stage: "JobProgressFn",
    ) -> None:
        # The pipeline ran with store=False (no NEW recording was created); persist
        # the fresh analysis over the EXISTING one, preserving everything else.
        await set_stage("storing", "saving re-analysis", None)
        response.stored = True
        response.recording_id = recording_id
        response.storage_note = None
        updated = await store_backend.overwrite_analysis(
            uid, recording_id,
            turns=[t.model_dump() for t in response.turns],
            analysis=response.model_dump(),
            reanalyzed_at=datetime.now(timezone.utc).isoformat(),
        )
        if updated is None:
            # Deleted between our existence check and this write — fail honestly.
            raise HTTPException(status_code=404, detail="Recording not found")

    _spawn_job(_run_analysis_job(
        uid, job_id, store_backend, state,
        prepare=_prepare,
        context="",
        consent=True,
        store=False,
        title=preserved_title,
        on_success=_persist,
    ))
    return JobCreatedResponse(job_id=job_id)


@app.get("/analyze/jobs/{job_id}", response_model=JobStateResponse)
async def get_analyze_job(
    job_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    """Poll an analysis job's staged progress (and its result once done).

    Deliberately NOT rate-limited (no ``_rate_limit`` dependency): the client
    polls this ~every 3s for the life of a multi-minute job, so the per-IP minute
    budget would 429 a normal poll into uselessness. It is a cheap uid-scoped GCS
    read with no LLM cost and is still fully auth-gated — the same reasoning as
    the /media precedent.

    uid-scoped 404 for an unknown/foreign job. Two computed-on-read behaviours:
    a non-terminal job whose state stopped advancing for >120s is reported as
    "stalled" (an orphaned in-process task never writes "failed"), and a terminal
    state older than 24h is lazily deleted (→ 404) as cheap TTL cleanup."""
    store_backend = get_recordings_store()
    if store_backend is None:
        raise HTTPException(status_code=503, detail=_JOBS_DISABLED_DETAIL)

    state = await store_backend.read_job_state(uid, job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown job")

    now = datetime.now(timezone.utc)
    status = state.get("status", "queued")
    updated_at = _parse_iso(state.get("updated_at"))
    age = (now - updated_at).total_seconds() if updated_at is not None else 0.0

    # Lazy TTL cleanup — a terminal state older than 24h is dead weight.
    if status in ("done", "failed") and age > JOB_TTL_SECONDS:
        await store_backend.delete_job(uid, job_id)
        raise HTTPException(status_code=404, detail="unknown job")

    # Staleness honesty — a non-terminal job that stopped advancing is stalled.
    progress_note = state.get("progress_note")
    if status not in ("done", "failed") and age > JOB_STALL_SECONDS:
        status = "stalled"
        progress_note = (
            "the analysis appears to have stalled — it may have been "
            "interrupted; try again"
        )

    return JobStateResponse(
        job_id=job_id,
        status=status,
        created_at=state.get("created_at", ""),
        updated_at=state.get("updated_at", ""),
        stage_started_at=state.get("stage_started_at"),
        progress_note=progress_note,
        duration_seconds=state.get("duration_seconds"),
        bytes_downloaded=state.get("bytes_downloaded"),
        bytes_total=state.get("bytes_total"),
        error=state.get("error"),
        # Result is only carried once done — excluded on every non-terminal poll.
        result=state.get("result") if status == "done" else None,
    )


@app.post("/session", response_model=SessionOut, status_code=201)
async def create_session(
    req: SessionCreate, uid: str = Depends(get_current_uid),
):
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        # user_id is stamped from the verified token, never from the request.
        await db.execute(
            "INSERT INTO sessions (id, created_at, turns, metadata, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, created_at, json.dumps(req.turns),
             json.dumps(req.metadata), uid),
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
async def get_session(
    session_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        # Scoped to the caller: another user's session reads as 404 (not 403),
        # so its existence is never confirmed.
        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata FROM sessions "
            "WHERE id = ? AND user_id = ?",
            (session_id, uid),
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
async def add_turn(
    session_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    turn: SessionTurn,
    uid: str = Depends(get_current_uid),
):
    turn_dict = turn.model_dump()

    db = await get_db()
    try:
        # Atomic append via SQLite JSON1 — a single UPDATE cannot lose turns to a
        # concurrent read-modify-write race (last-writer-wins data loss). RETURNING
        # the post-update array length in the SAME statement gives THIS append's
        # own index; a separate post-commit SELECT could read a concurrent
        # append's state and report the wrong turn_index. The user_id predicate
        # makes a foreign (or missing) session a no-op → 404 below.
        cursor = await db.execute(
            "UPDATE sessions SET turns = json_insert(turns, '$[#]', json(?)) "
            "WHERE id = ? AND user_id = ? RETURNING json_array_length(turns)",
            (json.dumps(turn_dict), session_id, uid),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found")
        new_len = row[0]
        await db.commit()
    finally:
        await db.close()

    return TurnResponse(
        session_id=session_id,
        turn_index=new_len - 1,
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
    from xml.sax.saxutils import escape

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
        # P1-3: every interpolated field is XML-escaped before it enters the
        # reportlab Paragraph mini-HTML markup — a stray "<", "&" or "</font>"
        # in transcript speech must not break parsing (500) or inject styling.
        speaker = escape(str(t.get("speaker", "Unknown")))
        text = escape(str(t.get("text", "")))
        story.append(Paragraph(f"<b>Turn {i} [{speaker}]:</b> {text}", styles["Normal"]))
        score = t.get("score")
        if isinstance(score, dict):
            parts = [f"{escape(str(k))}={escape(str(v))}" for k, v in score.items()]
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
    story.append(Paragraph(escape(insights), styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


@app.get("/session/{session_id}/export")
async def export_session(
    session_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    format: ExportFormat = Query(default=ExportFormat.text),
    uid: str = Depends(get_current_uid),
    _rl: None = Depends(_rate_limit),
):
    # Fetch session — scoped to the caller (foreign/missing → 404 below).
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata FROM sessions "
            "WHERE id = ? AND user_id = ?",
            (session_id, uid),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        turns = json.loads(row["turns"])
        metadata = json.loads(row["metadata"])
        # Valid JSON of the wrong shape (e.g. a manual DB edit leaving turns as a
        # string or object) would slip past a decode-only guard and later raise a
        # raw AttributeError; require the expected container shapes here.
        if not isinstance(turns, list) or not isinstance(metadata, dict):
            raise TypeError("unexpected stored shape")
        session = {
            "id": row["id"],
            "created_at": row["created_at"],
            "turns": turns,
            "metadata": metadata,
        }
    except (json.JSONDecodeError, TypeError, ValueError):
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
        # Log the detail; keep it OUT of the user-facing document — provider
        # error strings can carry request URLs/IDs and other internals.
        logger.warning(
            "Insights generation failed for session %s: %s", session_id, exc,
        )
        insights = "Insights unavailable (generation failed; see server logs)."

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
async def create_relationship(
    req: RelationshipCreate, uid: str = Depends(get_current_uid),
):
    rel_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        # user_id from the verified token; participants inherit this ownership.
        await db.execute(
            "INSERT INTO relationships (id, type, name, created_at, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (rel_id, req.type.value, req.name, created_at, uid),
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
async def get_relationship(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        # Foreign/missing relationship → 404 (never confirm another's existence).
        cursor = await db.execute(
            "SELECT id, type, name, created_at FROM relationships "
            "WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
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
async def list_edges(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, type FROM relationships WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
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


async def _require_participant(
    db: aiosqlite.Connection, relationship_id: str, participant_id: str, uid: str,
) -> None:
    """404 if the relationship or the participant-in-relationship is unknown.

    Honest failure: editing a voice profile for a participant that does not
    exist must not silently create an orphaned row. Scoped to ``uid`` — another
    user's relationship reads as "not found" (404, not 403), so its existence
    is never confirmed.
    """
    cursor = await db.execute(
        "SELECT 1 FROM relationships WHERE id = ? AND user_id = ?",
        (relationship_id, uid),
    )
    if await cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Relationship not found")
    cursor = await db.execute(
        "SELECT 1 FROM participants WHERE id = ? AND relationship_id = ?",
        (participant_id, relationship_id),
    )
    if await cursor.fetchone() is None:
        raise HTTPException(status_code=404, detail="Participant not found")


@app.get(
    "/relationships/{relationship_id}/participants/{participant_id}/voice-profile",
    response_model=VoiceProfileOut,
)
async def get_voice_profile(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    participant_id: str,
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        await _require_participant(db, relationship_id, participant_id, uid)
        cursor = await db.execute(
            "SELECT pairs, style_notes, updated_at FROM voice_profiles "
            "WHERE relationship_id = ? AND participant_id = ?",
            (relationship_id, participant_id),
        )
        row = await cursor.fetchone()
    finally:
        await db.close()

    # Never 404 for "not yet set" — an unset profile is an empty profile.
    if row is None:
        return VoiceProfileOut(pairs=[], style_notes=None, updated_at=None)
    try:
        pairs = json.loads(row["pairs"])
    except (ValueError, TypeError):
        pairs = []
    if not isinstance(pairs, list):
        pairs = []
    return VoiceProfileOut(
        pairs=pairs,
        style_notes=row["style_notes"],
        updated_at=row["updated_at"],
    )


@app.put(
    "/relationships/{relationship_id}/participants/{participant_id}/voice-profile",
    response_model=VoiceProfileOut,
)
async def put_voice_profile(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    participant_id: str,
    req: VoiceProfileIn,
    uid: str = Depends(get_current_uid),
):
    # Full replace (simplest editable surface). Apply the §2 caps: keep the
    # most recent MAX_PAIRS, truncate each field, then upsert.
    kept = req.pairs[-MAX_PAIRS:]
    pairs = [
        {
            "suggestion": p.suggestion[:MAX_PAIR_CHARS],
            "rephrase": p.rephrase[:MAX_PAIR_CHARS],
        }
        for p in kept
    ]
    style_notes = (
        req.style_notes[:MAX_STYLE_NOTES_CHARS] if req.style_notes else None
    )
    updated_at = datetime.now(timezone.utc).isoformat()

    db = await get_db()
    try:
        await _require_participant(db, relationship_id, participant_id, uid)
        await db.execute(
            "INSERT INTO voice_profiles "
            "(relationship_id, participant_id, pairs, style_notes, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(relationship_id, participant_id) DO UPDATE SET "
            "pairs = excluded.pairs, style_notes = excluded.style_notes, "
            "updated_at = excluded.updated_at",
            (relationship_id, participant_id, json.dumps(pairs), style_notes,
             updated_at),
        )
        await db.commit()
    finally:
        await db.close()

    return VoiceProfileOut(
        pairs=pairs, style_notes=style_notes, updated_at=updated_at,
    )


@app.post(
    "/relationships/{relationship_id}/sessions",
    response_model=RelationshipSessionOut,
    status_code=201,
)
async def create_relationship_session(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    req: RelationshipSessionCreate,
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, type FROM relationships WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
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
            "from_participant_id, to_participant_id, edge_context, empathy_slider, "
            "user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, created_at, json.dumps([]), json.dumps(req.metadata),
             relationship_id, req.from_participant_id, req.to_participant_id,
             edge_context, req.empathy_slider, uid),
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
async def list_relationship_sessions(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM relationships WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata, relationship_id, "
            "from_participant_id, to_participant_id, edge_context, empathy_slider "
            "FROM sessions WHERE relationship_id = ? AND user_id = ? "
            "ORDER BY created_at DESC",
            (relationship_id, uid),
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
async def list_participant_sessions(
    relationship_id: Annotated[str, Path(pattern=UUID_PATTERN)],
    # P2-7: participant_id is CLIENT-supplied (Participant.id is free-form in the
    # create request — e.g. "alex", "p1"), NOT a server-generated UUID, so it is
    # intentionally left unvalidated here. It only ever reaches parameterized SQL
    # and routing — never a response header — so it carries no injection surface.
    participant_id: str,
    uid: str = Depends(get_current_uid),
):
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id FROM relationships WHERE id = ? AND user_id = ?",
            (relationship_id, uid),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Relationship not found")

        cursor = await db.execute(
            "SELECT id, created_at, turns, metadata, relationship_id, "
            "from_participant_id, to_participant_id, edge_context, empathy_slider "
            "FROM sessions WHERE relationship_id = ? AND user_id = ? "
            "AND (from_participant_id = ? OR to_participant_id = ?) "
            "ORDER BY created_at DESC",
            (relationship_id, uid, participant_id, participant_id),
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
