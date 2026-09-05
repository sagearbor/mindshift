"""Pydantic models for the M2 real-time audio pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_serializer

# Where a piece of the realtime pipeline ran. "on-device" is the phone doing
# the work itself (Foundation A's hybrid on-device/cloud path); "cloud" is
# this server. Literal (not a free str) on the CLIENT->server TurnLocalEvent
# because a new client is the only thing that ever sends it and we want a typo
# rejected at the door; the server->client SuggestionEvent keeps a plain str
# for the same reason `kind` does — an older client must never choke on it.
TranscriptSource = Literal["on-device", "cloud"]
SuggestionSource = Literal["on-device", "cloud"]
TtsSource = Literal["on-device", "server"]

# Bounds on the CLIENT-supplied strings of a TurnLocalEvent. Unlike a Deepgram
# segment (bounded by how fast a human can talk), a turn_local's text is
# whatever bytes the socket delivers — uvicorn's default WS frame cap is
# 16 MiB — and it is (a) kept in the per-session utterance buffer (500 of
# them), (b) rendered into the cloud LLM prompt (token spend), (c) stored on
# the episode by POST /sessions/live. A hostile or buggy client must not be
# able to turn one frame into megabytes of memory or prompt. 8 000 chars is
# ~10 minutes of uninterrupted speech — far past any real turn, and well
# inside the 60 000-char per-session cap the ingest endpoint enforces.
TURN_TEXT_MAX = 8_000
TURN_SPEAKER_MAX = 64
TURN_LABEL_MAX = 64
TURN_PERSON_ID_MAX = 200


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


class ObserverTagged(BaseModel):
    """Mixin for the coaching events an in-app call's THERAPIST socket
    receives as a read-only copy (server/calls.py ``fan_out``): ``for_uid``
    names the participant the event was for. It is serialized ONLY when
    set — on your own events the key is absent, so every pre-call wire
    payload stays byte-identical."""

    for_uid: str | None = Field(
        default=None,
        description="In-app calls: the participant this event was coaching (observer copies only)",
    )

    @model_serializer(mode="wrap")
    def _drop_absent_for_uid(self, handler, info):
        data = handler(self)
        if isinstance(data, dict) and data.get("for_uid") is None:
            data.pop("for_uid", None)
        return data


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


class SuggestionEvent(ObserverTagged):
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
    # Which runtime produced the suggestion text: "cloud" (this server's LLM —
    # the only thing that existed before Foundation A) or "on-device" (the
    # phone's local model, echoed back through the server so the transcript
    # has one source of truth). Plain str + default, same back-compat reasoning
    # as `kind`: every legacy event is "cloud", and an old client that never
    # reads this field still parses the event.
    suggestion_source: str = Field(default="cloud", description='"cloud" | "on-device"')
    # True for a PROGRESSIVE preview of a cloud suggestion (Track 3-server):
    # the streaming LLM has completed the FIRST suggestion string, and it is
    # sent ahead of the full event so the phone can show something ~1s
    # earlier. A partial is never voiced (speak=False, audio_b64=None) and
    # its importance is a placeholder; the final event for the same
    # utterance_text (partial=False) supersedes it. Only sent to clients
    # that have proven they speak the new protocol (a session that sent a
    # turn_local) — an older client would render it as a second suggestion.
    # Default False keeps every legacy event byte-compatible.
    partial: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Foundation A additions — hybrid on-device/cloud realtime protocol.
#
# Everything below is ADDITIVE. Nothing in audio_pipeline.py emits or consumes
# these yet (that's a later track); they exist so the phone client and the
# server agree on the wire shape before either side is wired up. Each
# server->client event keeps the same `type` string-discriminator convention
# as TranscriptEvent/SuggestionEvent so the client's single `switch (msg.type)`
# keeps working and unknown types stay ignorable on old clients.
# ---------------------------------------------------------------------------

class TurnProsody(BaseModel):
    """Acoustic measurements the phone took on its own copy of a turn.

    Every field is optional because on-device measurement is best-effort:
    a phone that couldn't estimate pitch (unvoiced/too short — see
    server/prosody.py's honesty rule) sends ``pitch_hz: null`` rather than a
    guess, and the server must never treat a missing value as 0.
    """
    # allow_inf_nan=False on every measurement: an ``Infinity`` here would ride
    # straight into the watch relay's dB arithmetic and out as a non-JSON
    # ``Infinity`` literal on the watch socket.
    rms_dbfs: float | None = Field(default=None, allow_inf_nan=False, description="Loudness of the turn, dB relative to full scale")
    pitch_hz: float | None = Field(default=None, ge=0, allow_inf_nan=False, description="Median F0 in Hz; null when unvoiced")
    speech_rate: float | None = Field(default=None, ge=0, allow_inf_nan=False, description="Syllables (or words) per second")


class TurnTextTone(BaseModel):
    """Text-derived tone scores (0–100 each) the phone computed locally.

    All optional for the same best-effort reason as TurnProsody; ``label`` is
    a free string ("defensive", "warm", ...) so an on-device model can name
    a tone the fixed score set doesn't cover without a protocol bump.
    """
    warmth: int | None = Field(default=None, ge=0, le=100)
    defensiveness: int | None = Field(default=None, ge=0, le=100)
    sarcasm: int | None = Field(default=None, ge=0, le=100)
    sadness: int | None = Field(default=None, ge=0, le=100)
    frustration: int | None = Field(default=None, ge=0, le=100)
    label: str | None = Field(
        default=None, max_length=TURN_LABEL_MAX,
        description="Free-text tone label from the on-device classifier",
    )


class TurnLocalEvent(BaseModel):
    """Client→server: a turn the PHONE finalized itself (on-device STT and/or
    on-device coaching), reported so the server's transcript, episode record
    and coaching context stay complete even when the cloud didn't hear the
    audio. Distinct from AudioChunk on purpose — this carries words and
    measurements, never PCM.

    ``speaker`` is the label the phone assigned; ``speaker_person_id`` /
    ``speaker_match_score`` / ``is_self`` are the phone's voiceprint verdict
    (null when it has no enrolled voiceprint to match against — there is no
    "self" without one, mirroring server/watch/diarize.py's rule).
    """
    type: str = Field(default="turn_local", description="Event type discriminator")
    session_id: str = Field(max_length=128)
    speaker: str = Field(max_length=TURN_SPEAKER_MAX, description="Speaker label as the phone assigned it, e.g. 'Speaker A'")
    speaker_person_id: str | None = Field(default=None, max_length=TURN_PERSON_ID_MAX, description="Matched person/profile id, if any")
    speaker_match_score: float | None = Field(default=None, allow_inf_nan=False, description="Voiceprint similarity that produced the match")
    # WHY the phone called this voice that person: "absolute" (cosine >= the
    # 0.65 bar) or "contrast" (>= 0.40, a multi-recording print, and this
    # voice beat every other voice in the session for that person by the
    # margin — speaker_id.CROSS_MATCH_*). Null when no voiceprint matched (a
    # mid-call binding carries a person without a basis). Recorded so a
    # contrast "You" is never mistaken for a 0.65 one when reading the record.
    speaker_match_basis: Literal["absolute", "contrast"] | None = Field(
        default=None, description="How the voiceprint match was reached: absolute cosine or in-session contrast",
    )
    is_self: bool | None = Field(default=None, description="True/False when the phone could decide; null when it couldn't")
    text: str = Field(max_length=TURN_TEXT_MAX)
    # allow_inf_nan=False: these address the server's PCM ring buffer by
    # session time (int(start * sample_rate)); an inf/NaN would only surface as
    # an OverflowError/ValueError deep inside the enrichment task. Reject at
    # the door like every other malformed field.
    start_time: float = Field(ge=0, allow_inf_nan=False, description="Turn start in seconds")
    end_time: float = Field(ge=0, allow_inf_nan=False, description="Turn end in seconds")
    transcript_source: TranscriptSource
    prosody: TurnProsody | None = None
    text_tone: TurnTextTone | None = None
    # Coaching the phone already produced for this turn, if it did it locally.
    # A null `suggestion` with a non-null `suggestion_source` is meaningless,
    # but not rejected here — the pipeline decides what to do with partial
    # reports when it's wired up.
    suggestion: str | None = Field(default=None, max_length=TURN_TEXT_MAX)
    suggestion_source: SuggestionSource | None = None
    tts_source: TtsSource | None = None


class ToneFlagEvent(ObserverTagged):
    """Server→client: the server noticed a tone worth surfacing on a turn.

    ``source`` says which analysis raised it — "text" (LLM/lexical tone over
    the words) or "audio" (prosody over the signal); the client may render
    them differently. ``scores`` is an open dict (e.g. {"frustration": 78})
    rather than a fixed model so new dimensions don't need a protocol bump —
    the same reasoning as TurnTextTone.label.
    """
    type: str = Field(default="tone_flag", description="Event type discriminator")
    session_id: str
    speaker: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    source: Literal["text", "audio"]
    scores: dict[str, float] = Field(default_factory=dict)
    label: str
    confidence: float = Field(ge=0, le=1)


class SpeakerIdentityEvent(ObserverTagged):
    """Server→client: the server's (possibly revised) identity for a speaker
    label — who "Speaker A" turned out to be once a voiceprint matched.

    ``person_id``/``display_name`` are null when the speaker is unknown;
    ``is_self`` is a definite bool here (unlike TurnLocalEvent, this is the
    server's verdict, and "unknown" is expressed as is_self=False with a null
    person_id). ``score`` is the match similarity that justified it.
    """
    type: str = Field(default="speaker_identity", description="Event type discriminator")
    session_id: str
    speaker: str
    person_id: str | None = None
    display_name: str | None = None
    is_self: bool
    score: float


class DiarizationConfig(BaseModel):
    """Configuration for speaker diarization."""
    num_speakers: int = Field(default=2, ge=1, le=10)
    silence_threshold_ms: int = Field(default=500, ge=100, le=5000,
                                       description="Silence gap (ms) to switch speakers")
    labels: list[str] = Field(default_factory=lambda: ["Speaker A", "Speaker B"])
