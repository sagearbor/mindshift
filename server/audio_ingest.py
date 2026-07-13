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
import tempfile
import wave
from dataclasses import dataclass

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
# Sample rate we downmix to before diarizing (see _audio_for_transcription). A
# real two-speaker recording proved nova-3's diarizer COLLAPSES both voices into
# one speaker on 48 kHz input (the phone-video default) but splits them correctly
# at 16 kHz — the rate the model is trained for. 16 kHz is Deepgram's native ASR
# rate and matches the live pipeline + our stored audio derivative.
TRANSCRIBE_TARGET_SR = 16000
# Pre-recorded transcription of a long file can take a while server-side; be
# generous so a legitimately long recording is not cut off mid-decode. Chunked
# uploads reach 200MB (a long phone video), whose upload+transcription can run
# several minutes — 600s keeps httpx from timing out before Deepgram responds.
DEEPGRAM_PRERECORDED_TIMEOUT_S = 600.0


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


class TranscodeError(RuntimeError):
    """A storage derivative could not be produced by ffmpeg.

    Raised by :func:`build_derivatives` when the ALWAYS-required audio.m4a
    extraction fails (ffmpeg missing, bad input, timeout). A FAILED optional
    video derivative does NOT raise — it degrades to audio-only with a note (see
    :class:`Derivatives`), so replay of the audio still works.
    """


# ---------------------------------------------------------------------------
# Storage derivatives — we persist compressed AAC audio (and, for a video
# input, a small 360p H.264 clip), NEVER the original bytes. A phone video is
# 50-300MB; the audio derivative is ~0.5MB/min and the 360p clip a few percent
# of the original, so replay stays cheap. All ffmpeg work uses temp FILES (not
# stdin/stdout pipes): an MP4 muxer needs a seekable output, and a phone MP4
# often carries its moov atom at the end, needing a seekable INPUT — pipes
# satisfy neither reliably.
# ---------------------------------------------------------------------------

# 10 minutes per transcode — generous for a long recording, bounded so a
# pathological input can never wedge a worker thread. A video transcode that
# hits it degrades to audio-only (honest note); an audio transcode that hits it
# is a TranscodeError (nothing useful to store). Raised from 5min after a real
# 48s HEVC 10-bit HDR 1080p phone video (20 Mbps) blew past 300s on Cloud Run's
# limited vCPU — software HEVC decode dominates the transcode — and stored
# audio-only. The async job now heartbeats through the storing stage, so a long
# transcode no longer trips the client's "stalled" heuristic (see main.py).
DERIVATIVE_TIMEOUT_S = 600

# Mono 16 kHz AAC in an MP4/m4a container. 48 kbps is ample for speech.
_AUDIO_M4A_ARGS = [
    "-vn", "-ac", "1", "-ar", "16000",
    "-c:a", "aac", "-b:a", "48k",
    "-movflags", "+faststart",
]
# 360p H.264 + AAC. scale=-2:360 keeps the aspect ratio (width rounded to an
# even number, required by libx264). CRF 28 / veryfast keeps it small and fast.
_VIDEO_360P_ARGS = [
    "-vf", "scale=-2:360",
    "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
    "-c:a", "aac", "-b:a", "48k",
    "-movflags", "+faststart",
]


@dataclass
class Derivatives:
    """Result of :func:`build_derivatives`.

    ``audio_m4a`` is always present (its failure raises TranscodeError instead).
    ``video_360p`` is bytes only when the input carried a video stream AND the
    360p transcode succeeded; when a video was present but the transcode failed
    or timed out, ``video_360p`` is None and ``video_note`` explains why (the
    caller surfaces it as a storage_note — audio replay still works).
    """
    audio_m4a: bytes
    video_360p: bytes | None
    has_video: bool
    video_note: str | None


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # noqa: BLE001 — report honestly, never fabricate
        raise TranscodeError(f"ffmpeg unavailable: {exc}") from exc


def _probe_has_video(exe: str, in_path: str) -> bool:
    """True when the input has a real video stream (not just cover art).

    imageio-ffmpeg ships only ffmpeg (no ffprobe), so we parse ``ffmpeg -i``'s
    stderr. An audio file's embedded album art appears as a ``Video:`` line
    tagged ``(attached pic)`` — excluded so a cover-art MP3 is not mistaken for
    a video. A probe failure returns False (treat as audio-only, honestly).
    """
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-i", in_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
    except Exception:  # noqa: BLE001 — probe failure → assume audio-only
        return False
    stderr = proc.stderr.decode("utf-8", "replace")
    for line in stderr.splitlines():
        if "Video:" in line and "attached pic" not in line.lower():
            return True
    return False


def _transcode(exe: str, in_path: str, out_args: list[str], suffix: str,
               timeout: int) -> bytes:
    """Run one ffmpeg transcode from ``in_path`` to a temp file; return bytes.

    Raises :class:`TranscodeError` on a non-zero exit, empty output, or timeout
    (timeout is normalized to TranscodeError so callers handle one type).
    """
    fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        cmd = [exe, "-nostdin", "-loglevel", "error", "-y", "-i", in_path,
               *out_args, out_path]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscodeError("transcode timed out") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip()[-200:]
            raise TranscodeError(detail or "ffmpeg transcode failed")
        with open(out_path, "rb") as f:
            data = f.read()
        if not data:
            raise TranscodeError("transcode produced no output")
        return data
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def build_derivatives(
    data: bytes, *, timeout: int = DERIVATIVE_TIMEOUT_S,
) -> Derivatives:
    """Build the compressed storage derivatives for a recording's raw bytes.

    Always produces a mono AAC ``audio.m4a``; when the input has a video stream,
    additionally produces a 360p ``video_360p.mp4`` (degrading to audio-only
    with a note on transcode failure/timeout). The input is written to ONE temp
    file that the probe + both transcodes share. This is blocking (subprocess +
    file I/O) — the caller runs it in a worker thread.
    """
    exe = _ffmpeg_exe()
    fd, in_path = tempfile.mkstemp(suffix=".src")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        has_video = _probe_has_video(exe, in_path)
        # Audio is mandatory — its failure is a TranscodeError (nothing to store).
        audio = _transcode(exe, in_path, _AUDIO_M4A_ARGS, ".m4a", timeout)
        video: bytes | None = None
        note: str | None = None
        if has_video:
            try:
                video = _transcode(
                    exe, in_path, _VIDEO_360P_ARGS, ".mp4", timeout,
                )
            except TranscodeError as exc:
                video = None
                note = f"video replay unavailable: {exc}"
        return Derivatives(
            audio_m4a=audio, video_360p=video, has_video=has_video,
            video_note=note,
        )
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass


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


def _decode_via_ffmpeg(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
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

    # INPUT VIA TEMP FILE, NOT stdin: phone/Photos MP4s put the moov index at
    # the END of the file, and ffmpeg cannot seek a pipe — piping such a file
    # fails with "partial file / Invalid data found when processing input"
    # (bit a real Google-Photos video in production: analysis succeeded but
    # every voice label was lost). A seekable temp file decodes them fine;
    # the derivatives path already worked this way for the same reason.
    with tempfile.NamedTemporaryFile(suffix=_suffix_for(filename), delete=False) as tf:
        tf.write(data)
        in_path = tf.name
    cmd = [
        exe, "-nostdin", "-loglevel", "error",
        "-i", in_path,
        "-vn",                       # drop any video track
        "-ac", "1",                  # mono
        "-ar", str(FFMPEG_TARGET_SR),
        "-f", "s16le", "pipe:1",     # raw 16-bit LE PCM to stdout
    ]
    try:
        try:
            proc = subprocess.run(
                cmd,
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
    finally:
        try:
            os.unlink(in_path)
        except OSError:
            pass

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-200:]
        raise AudioDecodeError(
            "could not decode this file"
            + (f": {detail}" if detail else "")
        )

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm, dtype=np.float32), FFMPEG_TARGET_SR



def _suffix_for(filename: str) -> str:
    """A safe file suffix for the temp decode input — the extension helps
    ffmpeg pick a demuxer; anything unrecognizable becomes ".bin"."""
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext and len(ext) <= 8 and ext[1:].isalnum() else ".bin"

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
    return _decode_via_ffmpeg(data, filename)


# ---------------------------------------------------------------------------
# Deepgram pre-recorded transcription
# ---------------------------------------------------------------------------

def _pcm_to_wav16(pcm: np.ndarray, sr: int) -> bytes:
    """Wrap mono float32 PCM in [-1, 1] as a 16-bit little-endian WAV in memory."""
    ints = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())
    return buf.getvalue()


def _audio_for_transcription(data: bytes) -> bytes | None:
    """Downmix a recording's bytes to a 16 kHz mono WAV for Deepgram, or ``None``.

    Why not just send the original container: nova-3's diarizer merged two clearly
    distinct speakers into one on a real 48 kHz phone video, yet split them
    correctly once the SAME audio was fed at 16 kHz (:data:`TRANSCRIBE_TARGET_SR`).
    Downmixing here also shrinks a 100MB+ video to a ~1MB upload, so Deepgram no
    longer has to demux an exotic HEVC container and the request is far faster.
    Reuses :func:`_decode_via_ffmpeg`, which already outputs mono 16 kHz.

    Returns ``None`` when the downmix is unavailable (no ffmpeg / undecodable), so
    the caller falls back to sending the raw container — Deepgram's own decoder is
    the honest backstop; we never fabricate audio.
    """
    try:
        pcm, sr = _decode_via_ffmpeg(data, "")
    except AudioDecodeError as exc:
        logger.info("transcribe downmix unavailable (%s); sending raw container", exc)
        return None
    return _pcm_to_wav16(pcm, sr)


def transcribe_prerecorded(
    data: bytes, content_type: str | None,
) -> list[dict]:
    """Transcribe a recording via Deepgram's pre-recorded API.

    Sends a 16 kHz mono WAV downmix of the recording (see
    :func:`_audio_for_transcription` for WHY — nova-3's diarizer collapses
    speakers on 48 kHz input), falling back to the ORIGINAL container bytes if the
    downmix is unavailable (Deepgram then decodes the container itself). Returns
    one dict per diarized utterance: ``{speaker, text, start_time, end_time}``,
    with ``speaker`` mapped through the SAME :func:`_generated_speaker_label` the
    live pipeline uses.

    * Missing ``DEEPGRAM_API_KEY`` → :class:`TranscriptionUnavailable` (503).
    * Network/HTTP failure → :class:`TranscriptionUnavailable` (503) with the
      real reason (a provider outage is "unavailable", not "no speech").
    * Empty / speechless result → :class:`NoSpeechFound` (422).
    """
    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise TranscriptionUnavailable("transcription not configured")

    # Prefer a 16 kHz mono downmix (reliable diarization + tiny upload); fall back
    # to the raw container when ffmpeg can't produce it.
    wav = _audio_for_transcription(data)
    if wav is not None:
        send_data: bytes = wav
        send_content_type = "audio/wav"
    else:
        send_data = data
        send_content_type = content_type or "application/octet-stream"

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": send_content_type,
    }
    try:
        resp = httpx.post(
            DEEPGRAM_PRERECORDED_URL,
            params=DEEPGRAM_PRERECORDED_PARAMS,
            headers=headers,
            content=send_data,
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
