"""M2 real-time audio pipeline — WebSocket endpoint with credential-gated
transcription, diarization, and TTS.

Design note (honesty over mock data)
------------------------------------
The speech providers below are credential-gated. When their API keys are not
configured they report themselves *unavailable* and the pipeline says so
explicitly over the WebSocket — it never fabricates transcripts or audio that
could be mistaken for real output. The full transcribe → diarize → suggest →
speak flow is exercised in tests by injecting test doubles via ``app.state``
(see ``tests/test_audio_pipeline.py``); the live provider integrations remain
to be implemented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from llm_client import LLMClient
from models.audio import DiarizationConfig, SuggestionEvent, Utterance

logger = logging.getLogger(__name__)


class TranscriberUnavailable(RuntimeError):
    """Raised when a transcription backend is not configured/available.

    The pipeline catches this and reports ``transcription_unavailable`` to the
    client rather than inventing a transcript.
    """


# ---------------------------------------------------------------------------
# Deepgram transcriber (credential-gated)
# ---------------------------------------------------------------------------

class DeepgramTranscriber:
    """Real-time transcription via Deepgram.

    Requires ``DEEPGRAM_API_KEY``. The live streaming integration is not yet
    implemented; ``connect()`` reports precisely why it is unavailable so the
    project's true state is never hidden behind fabricated transcripts.
    """

    def __init__(self) -> None:
        self._connected = False

    async def connect(self) -> None:
        api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
        if not api_key:
            raise TranscriberUnavailable(
                "DEEPGRAM_API_KEY not set — real-time transcription is disabled"
            )
        # A key is present but the live Deepgram streaming bridge is not built
        # yet. Be honest rather than returning placeholder text.
        raise TranscriberUnavailable(
            "Deepgram live transcription is not yet implemented "
            "(API key detected, streaming integration pending)"
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def stream(self, audio_bytes: bytes) -> str | None:
        if not self._connected:
            raise TranscriberUnavailable("Transcriber not connected")
        raise TranscriberUnavailable("Deepgram live transcription is not yet implemented")

    async def close(self) -> None:
        self._connected = False


# ---------------------------------------------------------------------------
# Speaker diarization (alternation heuristic)
# ---------------------------------------------------------------------------

class SpeakerDiarizer:
    """Assigns speaker labels by alternating across configured labels.

    This is an explicit placeholder heuristic, not acoustic diarization: it
    rotates through ``config.labels`` on each utterance. Real speaker
    separation (e.g. from Deepgram diarization or an embedding model) will
    replace this once transcription is wired to a live backend.
    """

    def __init__(self, config: DiarizationConfig | None = None) -> None:
        self.config = config or DiarizationConfig()
        self._turn_counter = 0

    def assign_speaker(self) -> str:
        label = self.config.labels[self._turn_counter % len(self.config.labels)]
        self._turn_counter += 1
        return label

    def reset(self) -> None:
        self._turn_counter = 0


# ---------------------------------------------------------------------------
# Text-to-speech (credential-gated)
# ---------------------------------------------------------------------------

class TTSClient:
    """Text-to-speech for earpiece output.

    Requires a TTS provider key (``TTS_API_KEY`` or ``ELEVENLABS_API_KEY``).
    When unconfigured, ``synthesize`` returns ``None`` (no audio) rather than
    fabricating placeholder bytes — the suggestion still flows as on-screen text.
    """

    async def synthesize(self, text: str) -> str | None:
        """Return base64-encoded audio for *text*, or ``None`` if TTS is unavailable."""
        api_key = os.getenv("TTS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.info("TTS unavailable (no TTS key) — returning no audio")
            return None
        # Live TTS synthesis is not yet implemented; do not fabricate audio.
        logger.info("TTS key detected but synthesis integration is not yet implemented")
        return None


# ---------------------------------------------------------------------------
# Session context (in-memory, per-connection)
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    session_id: str
    empathy_slider: int = 50
    role: str = "Husband"
    utterances: list[Utterance] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

async def audio_ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Handle a single audio-streaming WebSocket connection.

    Protocol
    --------
    Client → Server (binary):  raw audio chunks
    Client → Server (text):    JSON control messages, e.g.
        {"type": "config", "empathy_slider": 75, "role": "Husband"}
    Server → Client (text):    JSON ``SuggestionEvent`` on each utterance
    """
    await websocket.accept()

    # Per-connection state
    ctx = SessionContext(session_id=session_id)

    # Resolve providers from app.state (tests inject doubles here), falling
    # back to the real, credential-gated implementations.
    state = websocket.app.state
    transcriber_factory = getattr(state, "transcriber_factory", None) or DeepgramTranscriber
    diarizer_factory = getattr(state, "diarizer_factory", None) or SpeakerDiarizer
    tts = getattr(state, "tts_client", None) or TTSClient()
    llm_client: LLMClient = state.llm_client

    transcriber = transcriber_factory()
    diarizer = diarizer_factory()

    # Connect transcription; if unavailable, tell the client plainly instead of
    # fabricating transcripts.
    transcription_available = True
    unavailable_reason = ""
    try:
        await transcriber.connect()
    except TranscriberUnavailable as exc:
        transcription_available = False
        unavailable_reason = str(exc)
        await websocket.send_text(json.dumps(
            {"type": "transcription_unavailable", "reason": unavailable_reason}
        ))
        logger.info("Transcription unavailable for session %s: %s", session_id, unavailable_reason)

    try:
        while True:
            message = await websocket.receive()

            # --- Disconnect ---
            if message.get("type") == "websocket.disconnect":
                break

            # --- Binary audio chunk ---
            if "bytes" in message and message["bytes"] is not None:
                audio_bytes: bytes = message["bytes"]
                if len(audio_bytes) == 0:
                    continue

                if not transcription_available:
                    await websocket.send_text(json.dumps(
                        {"type": "transcription_unavailable", "reason": unavailable_reason}
                    ))
                    continue

                try:
                    transcript = await transcriber.stream(audio_bytes)
                except TranscriberUnavailable as exc:
                    transcription_available = False
                    unavailable_reason = str(exc)
                    await websocket.send_text(json.dumps(
                        {"type": "transcription_unavailable", "reason": unavailable_reason}
                    ))
                    continue
                if transcript is None:
                    continue

                # Diarize
                speaker = diarizer.assign_speaker()
                utterance = Utterance(
                    session_id=session_id,
                    speaker=speaker,
                    text=transcript,
                    start_time=0.0,
                    end_time=1.0,
                )
                ctx.utterances.append(utterance)

                # Generate suggestion via LLM
                suggestion_texts = await _generate_suggestions(
                    llm_client, utterance, ctx.empathy_slider, ctx.role,
                )

                # TTS for first suggestion
                tts_audio = await tts.synthesize(suggestion_texts[0]) if suggestion_texts else None

                event = SuggestionEvent(
                    session_id=session_id,
                    utterance_text=transcript,
                    speaker=speaker,
                    suggestions=suggestion_texts,
                    empathy_slider=ctx.empathy_slider,
                    audio_b64=tts_audio,
                )
                await websocket.send_text(event.model_dump_json())

            # --- Text control message ---
            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                    continue

                msg_type = payload.get("type")
                if msg_type == "config":
                    if "empathy_slider" in payload:
                        val = payload["empathy_slider"]
                        if isinstance(val, int) and 0 <= val <= 100:
                            ctx.empathy_slider = val
                    if "role" in payload:
                        ctx.role = str(payload["role"])
                    await websocket.send_text(json.dumps({"type": "config_ack"}))
                else:
                    await websocket.send_text(json.dumps({"error": f"unknown type: {msg_type}"}))

    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    finally:
        await transcriber.close()


# ---------------------------------------------------------------------------
# LLM suggestion helper
# ---------------------------------------------------------------------------

async def _generate_suggestions(
    llm: LLMClient,
    utterance: Utterance,
    empathy_slider: int,
    role: str,
) -> list[str]:
    """Call LLMClient.complete() and parse suggestions from the response."""
    from main import empathy_system_prompt, parse_llm_json

    system = empathy_system_prompt(empathy_slider, role)
    user_content = f'Transcript turn: "{utterance.text}"'

    raw = await asyncio.to_thread(llm.complete, system=system, user=user_content)

    try:
        data = parse_llm_json(raw)
        return data.get("suggestions", [])
    except (json.JSONDecodeError, KeyError):
        logger.warning("LLM returned unparseable response for utterance: %s", utterance.text)
        return [f"I hear you — {utterance.text}"]
