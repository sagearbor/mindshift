"""M2 real-time audio pipeline — WebSocket endpoint with Deepgram, diarization, and TTS stubs."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from llm_client import LLMClient
from models.audio import DiarizationConfig, SuggestionEvent, Utterance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deepgram transcriber stub
# ---------------------------------------------------------------------------

class DeepgramTranscriber:
    """Stub for Deepgram streaming transcription.

    In production this would hold a Deepgram SDK client and stream audio
    via their WebSocket API.  For now it accumulates chunks and returns
    mock transcriptions.
    """

    def __init__(self) -> None:
        self._connected = False
        self._chunks: list[bytes] = []

    async def connect(self) -> None:
        self._connected = True
        logger.info("DeepgramTranscriber connected (stub)")

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def stream(self, audio_bytes: bytes) -> str | None:
        """Feed audio bytes; returns a transcription when an utterance is complete.

        Stub logic: every chunk produces a mock transcription so tests can
        exercise the full pipeline.  A real implementation would buffer and
        return ``None`` until Deepgram signals an utterance boundary.
        """
        if not self._connected:
            raise RuntimeError("Transcriber not connected")
        self._chunks.append(audio_bytes)
        return f"Mock transcription for chunk {len(self._chunks)}"

    async def close(self) -> None:
        self._connected = False
        self._chunks.clear()
        logger.info("DeepgramTranscriber closed (stub)")


# ---------------------------------------------------------------------------
# Speaker diarization stub
# ---------------------------------------------------------------------------

class SpeakerDiarizer:
    """Assigns speaker labels based on simple alternation / silence heuristic.

    Stub: alternates between configured labels on each utterance.
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
# TTS stub
# ---------------------------------------------------------------------------

class TTSClient:
    """Stub for text-to-speech synthesis (earpiece output).

    In production this would call a TTS API (e.g. ElevenLabs, Google TTS)
    and return real audio bytes.  The stub returns a base64-encoded
    placeholder.
    """

    async def synthesize(self, text: str) -> str:
        """Return base64-encoded mock audio for *text*."""
        mock_audio = f"[TTS audio: {text}]".encode()
        return base64.b64encode(mock_audio).decode()


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
    transcriber = DeepgramTranscriber()
    diarizer = SpeakerDiarizer()
    tts = TTSClient()

    # Resolve LLM client from app state
    llm_client: LLMClient = websocket.app.state.llm_client

    await transcriber.connect()

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

                transcript = await transcriber.stream(audio_bytes)
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
