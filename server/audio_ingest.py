"""Prerecorded-audio ingestion for POST /analyze/upload (process-and-discard).

Two responsibilities, both honest about failure (house rule: report unavailable,
never fabricate):

* :func:`decode_to_pcm` — turn an uploaded audio/video file into mono float32
  PCM + sample rate, for local prosody analysis. WAV is parsed with the stdlib
  ``wave`` module + numpy (no external binary needed). Everything else
  (mp3/m4a/mp4/mov/webm/…) is decoded through the ``imageio-ffmpeg`` static
  ffmpeg binary; ``-vn`` drops any video track so a video file yields just its
  audio. If the binary is unavailable or the decode fails we raise
  :class:`AudioDecodeError` — we NEVER invent audio.

* :func:`transcribe_prerecorded` — send the ORIGINAL file bytes to Deepgram's
  pre-recorded API (Deepgram accepts the container directly) and map its
  diarized utterances into the same turn shape the rest of the pipeline uses.
  A missing key is :class:`TranscriptionUnavailable` (the endpoint reports 503,
  never a mock transcript); an empty result is :class:`NoSpeechFound` (422).

Nothing here is persisted: the caller reads bytes, analyses, and discards.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import wave

import httpx
import numpy as np

# Reuse the EXACT diarization-index → label mapping the live pipeline uses, so a
# recording labelled "Speaker A/B/…" is indistinguishable from a live session.
from audio_pipeline import _generated_speaker_label

logger = logging.getLogger(__name__)

# ffmpeg decode target: mono 16 kHz s16le. 16 kHz matches the live pipeline's
# contract and is plenty for prosody (F0 search tops out at 400 Hz).
FFMPEG_TARGET_SR = 16000
# Bounded so a pathological/huge upload can never hang a worker on decode.
FFMPEG_TIMEOUT_S = 120

DEEPGRAM_PRERECORDED_URL = "https://api.deepgram.com/v1/listen"
# Pre-recorded params: diarize for per-speaker turns, utterances for
# sentence-level segmentation (what we map to turns), smart_format + punctuate
# for readable transcript text.
DEEPGRAM_PRERECORDED_PARAMS: dict[str, str] = {
    "model": "nova-3",
    "diarize": "true",
    "utterances": "true",
    "smart_format": "true",
    "punctuate": "true",
}
# Pre-recorded transcription of a long file can take a while server-side; be
# generous so a legitimately long recording is not cut off mid-decode.
DEEPGRAM_PRERECORDED_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Typed errors — each maps to one honest HTTP status at the endpoint
# ---------------------------------------------------------------------------

class AudioDecodeError(RuntimeError):
    """The uploaded file could not be decoded to PCM (endpoint → 422).

    Raised when the ffmpeg binary is unavailable or the decode fails. The
    pipeline degrades honestly (prosody unavailable) or rejects — it never
    fabricates audio.
    """


class TranscriptionUnavailable(RuntimeError):
    """Transcription backend is not configured/available (endpoint → 503).

    Chiefly a missing ``DEEPGRAM_API_KEY``. The house rule is to report the
    feature unavailable, never to return a mock transcript.
    """


class NoSpeechFound(RuntimeError):
    """Transcription returned no usable speech (endpoint → 422)."""


# ---------------------------------------------------------------------------
# Decode to PCM
# ---------------------------------------------------------------------------

def _looks_like_wav(data: bytes, filename: str) -> bool:
    """RIFF/WAVE magic bytes, or a ``.wav`` name. Content wins; the name is a
    fallback for streams that omit it."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return True
    return filename.lower().endswith(".wav")


def _decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a PCM WAV with the stdlib ``wave`` module — no ffmpeg needed.

    Handles 8/16/32-bit integer PCM (16-bit is the common case and the minimum
    the spec requires). Multi-channel audio is down-mixed to mono by averaging.
    Returns mono float32 in roughly [-1, 1] and the native sample rate.
    """
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        # WAV 8-bit PCM is UNSIGNED (0..255, centred on 128).
        arr = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        arr = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        # Trailing partial frame (corrupt file) would break the reshape — trim.
        usable = (arr.size // n_channels) * n_channels
        arr = arr[:usable].reshape(-1, n_channels).mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), sr


def _decode_via_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Decode any container via the imageio-ffmpeg static binary.

    ``-vn`` drops the video stream, so a video upload yields just its audio
    track. Output is forced to mono 16 kHz signed-16-bit little-endian raw PCM
    on stdout, which we read straight into numpy. Any failure — no binary, a
    non-zero exit, empty output, or a timeout — is an honest
    :class:`AudioDecodeError`; we never synthesize audio.
    """
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 — report honestly, never fabricate
        raise AudioDecodeError(
            f"could not decode this file: ffmpeg is unavailable ({exc})"
        ) from exc

    cmd = [
        exe, "-nostdin", "-loglevel", "error",
        "-i", "pipe:0",
        "-vn",                       # drop any video track
        "-ac", "1",                  # mono
        "-ar", str(FFMPEG_TARGET_SR),
        "-f", "s16le", "pipe:1",     # raw 16-bit LE PCM to stdout
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(
            "could not decode this file: decoding timed out"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — subprocess/OS failure → honest 422
        raise AudioDecodeError(f"could not decode this file: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-200:]
        raise AudioDecodeError(
            "could not decode this file"
            + (f": {detail}" if detail else "")
        )

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm, dtype=np.float32), FFMPEG_TARGET_SR


def decode_to_pcm(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode uploaded audio/video bytes to (mono float32 PCM, sample_rate).

    WAV is parsed in-process with the stdlib; anything else goes through
    ffmpeg. A WAV whose header parses but body does not (e.g. a compressed
    payload inside a WAV container) falls back to ffmpeg rather than failing
    outright. Honest :class:`AudioDecodeError` on any unrecoverable failure.
    """
    if _looks_like_wav(data, filename):
        try:
            return _decode_wav(data)
        except Exception as exc:  # noqa: BLE001 — try ffmpeg before giving up
            logger.info(
                "stdlib WAV parse failed (%s); falling back to ffmpeg", exc,
            )
    return _decode_via_ffmpeg(data)


# ---------------------------------------------------------------------------
# Deepgram pre-recorded transcription
# ---------------------------------------------------------------------------

def transcribe_prerecorded(
    data: bytes, content_type: str | None,
) -> list[dict]:
    """Transcribe a recording via Deepgram's pre-recorded API.

    Sends the ORIGINAL file bytes (Deepgram decodes the container itself — we do
    NOT send our resampled PCM). Returns one dict per diarized utterance:
    ``{speaker, text, start_time, end_time}``, with ``speaker`` mapped through
    the SAME :func:`_generated_speaker_label` the live pipeline uses.

    * Missing ``DEEPGRAM_API_KEY`` → :class:`TranscriptionUnavailable` (503).
    * Network/HTTP failure → :class:`TranscriptionUnavailable` (503) with the
      real reason (a provider outage is "unavailable", not "no speech").
    * Empty / speechless result → :class:`NoSpeechFound` (422).
    """
    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise TranscriptionUnavailable("transcription not configured")

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": content_type or "application/octet-stream",
    }
    try:
        resp = httpx.post(
            DEEPGRAM_PRERECORDED_URL,
            params=DEEPGRAM_PRERECORDED_PARAMS,
            headers=headers,
            content=data,
            timeout=DEEPGRAM_PRERECORDED_TIMEOUT_S,
        )
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPError as exc:
        raise TranscriptionUnavailable(
            f"transcription request failed: {exc}"
        ) from exc
    except ValueError as exc:  # non-JSON body
        raise TranscriptionUnavailable(
            "transcription returned an unreadable response"
        ) from exc

    utterances = (
        payload.get("results", {}).get("utterances")
        if isinstance(payload, dict) else None
    )
    if not isinstance(utterances, list) or not utterances:
        raise NoSpeechFound("no speech found in this recording")

    turns: list[dict] = []
    for utt in utterances:
        if not isinstance(utt, dict):
            continue
        text = utt.get("transcript")
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            speaker_idx = int(utt.get("speaker", 0))
        except (TypeError, ValueError):
            speaker_idx = 0
        turns.append({
            "speaker": _generated_speaker_label(max(0, speaker_idx)),
            "text": text.strip(),
            "start_time": float(utt.get("start", 0.0) or 0.0),
            "end_time": float(utt.get("end", 0.0) or 0.0),
        })

    if not turns:
        raise NoSpeechFound("no speech found in this recording")
    return turns
