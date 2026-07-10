"""Pydantic models for the M2 real-time audio pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AudioChunk(BaseModel):
    """A single chunk of streaming audio data sent over WebSocket."""
    session_id: str
    sequence: int = Field(ge=0, description="Monotonically increasing chunk index")
    audio_b64: str = Field(description="Base64-encoded audio bytes")
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)


class Utterance(BaseModel):
    """A completed spoken utterance after transcription + diarization."""
    session_id: str
    speaker: str = Field(description="Speaker label, e.g. 'Speaker A'")
    text: str
    start_time: float = Field(ge=0, description="Utterance start in seconds")
    end_time: float = Field(ge=0, description="Utterance end in seconds")
    confidence: float = Field(ge=0, le=1, default=1.0)


class TranscriptEvent(BaseModel):
    """A finalized transcript line, sent immediately on utterance end.

    Decoupled from SuggestionEvent so the transcript renders in real time
    (a suggestion takes seconds of LLM+TTS; the words themselves should not
    wait on that) and so a turn can appear in the transcript even when the
    coach chooses not to interject on it.
    """
    type: str = Field(default="transcript", description="Event type discriminator")
    session_id: str
    speaker: str
    text: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)


class SuggestionEvent(BaseModel):
    """A coaching suggestion sent back over WebSocket."""
    type: str = Field(default="suggestion", description="Event type discriminator")
    # Which coaching mode produced this: "response" (suggest what to say to the
    # OTHER person — the original behaviour) or "nudge" (a single delivery
    # course-correction for the user's OWN just-spoken turn, side-aware
    # coaching). A plain str with a default keeps the wire back-compatible:
    # older clients that never read this field still parse the event, and every
    # legacy event is a "response".
    kind: str = Field(default="response", description='"response" | "nudge"')
    session_id: str
    utterance_text: str
    speaker: str
    suggestions: list[str]
    empathy_slider: int = Field(ge=0, le=100)
    audio_b64: str | None = Field(default=None, description="TTS audio for earpiece, base64")
    # How much this moment warranted a coaching interjection (LLM-scored).
    # 100 (the fail-open default) preserves pre-importance behaviour.
    importance: int = Field(default=100, ge=0, le=100)
    # Whether the client should voice this suggestion (importance cleared the
    # session's interject threshold). False → show silently/dimmed at most.
    speak: bool = Field(default=True)


class DiarizationConfig(BaseModel):
    """Configuration for speaker diarization."""
    num_speakers: int = Field(default=2, ge=1, le=10)
    silence_threshold_ms: int = Field(default=500, ge=100, le=5000,
                                       description="Silence gap (ms) to switch speakers")
    labels: list[str] = Field(default_factory=lambda: ["Speaker A", "Speaker B"])
