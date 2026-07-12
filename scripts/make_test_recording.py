#!/usr/bin/env python3
"""Synthesize a physically-grounded two-person argument fixture for the
prerecorded-audio prosody tests.

Why physical modulation? Deepgram Aura is a neutral TTS voice — it cannot *act*
shouting or contempt. So we create the emotional ground truth OURSELVES, in
numpy, by modulating each synthesized turn's signal according to a scripted
emotion plan:

  * SHOUTING  — gain x4 soft-clipped through tanh (~+12 dB, agitated) AND
                tempo x1.15 (faster; the naive resample also raises pitch,
                which suits a raised, agitated voice).
  * COLD/CONTEMPT — gain x0.5 (~-6 dB, withdrawn) AND tempo x0.85 (slower,
                flat, deliberate).
  * AGITATED/SCARED — tempo x1.2 (fast, pitch rises — desired here).
  * CALM      — untouched.

This makes the fixture a KNOWN-answer test: the modulation dimensions we control
(energy, rate) become the ground truth the prosody pipeline must recover. We do
NOT claim to control pitch labels via the naive resample, so the metadata only
asserts energy/rate on the turns where the modulation forces them.

Outputs (under ``tmp/``, both process-and-discard test fixtures):
  * ``test_recording.wav``       — one concatenated 16 kHz mono WAV (0.4 s gaps)
  * ``test_recording_meta.json`` — per-turn {speaker, text, scripted_emotion,
                                    start_time, end_time, expected:{...labels}}

Requires ``DEEPGRAM_API_KEY`` (read from the repo-root ``.env`` at runtime, like
the rest of the app). Run directly to (re)generate and print the turn list:

    python scripts/make_test_recording.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Load the repo-root .env at runtime so DEEPGRAM_API_KEY is available exactly as
# the server reads it (we never read the .env file's contents ourselves).
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:  # pragma: no cover — python-dotenv is a dependency
    pass

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
TARGET_SR = 16000
GAP_SECONDS = 0.4

# Two distinct Aura-2 voices: a female voice for Speaker A, a male voice for B.
VOICE_A = "aura-2-thalia-en"   # female
VOICE_B = "aura-2-apollo-en"   # male

# The scripted arc: calm open → escalation with a shouted spike → cold contempt
# → repair attempt → calm close. Word counts are kept similar (~7-8) so that the
# tempo modulation — not sentence length — dominates the relative speech-rate
# label. `expected` holds ONLY the label dimensions the modulation physically
# forces; empty means "don't assert" (calm/agitated turns are context).
SCRIPT: list[dict] = [
    {"speaker": "A", "text": "Hey, can we talk about the weekend plan?",
     "emotion": "calm", "expected": {}},
    {"speaker": "B", "text": "Sure, I have a few minutes right now.",
     "emotion": "calm", "expected": {}},
    {"speaker": "A", "text": "You always cancel and never tell me why!",
     "emotion": "shouting", "expected": {"energy_label": "loud", "rate_label": "fast"}},
    {"speaker": "B", "text": "Please stop yelling, you are scaring me now.",
     "emotion": "agitated", "expected": {}},
    {"speaker": "A", "text": "Fine. Do whatever you want. I am done.",
     "emotion": "cold", "expected": {"energy_label": "quiet", "rate_label": "slow"}},
    {"speaker": "B", "text": "I hear you. Let us slow down together.",
     "emotion": "calm", "expected": {}},
    {"speaker": "A", "text": "Okay. I am sorry I raised my voice.",
     "emotion": "calm", "expected": {}},
    {"speaker": "B", "text": "Thank you. That really means a lot to me.",
     "emotion": "calm", "expected": {}},
]


def _speak(text: str, voice: str, api_key: str) -> np.ndarray:
    """Synthesize one turn via Deepgram Aura → mono float32 PCM at TARGET_SR."""
    resp = httpx.post(
        DEEPGRAM_SPEAK_URL,
        params={
            "model": voice,
            "encoding": "linear16",
            "container": "wav",
            "sample_rate": str(TARGET_SR),
        },
        headers={"Authorization": f"Token {api_key}"},
        json={"text": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    with wave.open(io.BytesIO(resp.content), "rb") as wf:
        assert wf.getsampwidth() == 2, "expected linear16"
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if wf.getnchannels() > 1:
        pcm = pcm.reshape(-1, wf.getnchannels()).mean(axis=1)
    if sr != TARGET_SR:  # defensive — we requested TARGET_SR
        pcm = _time_stretch(pcm, sr / TARGET_SR)
    return pcm.astype(np.float32)


def _time_stretch(sig: np.ndarray, tempo: float) -> np.ndarray:
    """Resample by linear interpolation. tempo>1 → fewer samples → faster (and,
    with this naive method, higher-pitched — intentional for agitation)."""
    n = sig.shape[0]
    new_len = max(1, int(round(n / tempo)))
    idx = np.linspace(0, n - 1, new_len)
    return np.interp(idx, np.arange(n), sig).astype(np.float32)


def _modulate(pcm: np.ndarray, emotion: str) -> np.ndarray:
    """Apply the scripted emotion's physical signal modulation (numpy only)."""
    if emotion == "shouting":
        return _time_stretch(np.tanh(pcm * 4.0).astype(np.float32), 1.15)
    if emotion == "cold":
        return _time_stretch((pcm * 0.5).astype(np.float32), 0.85)
    if emotion == "agitated":
        return _time_stretch(pcm, 1.2)
    return pcm  # calm — untouched


def generate(out_dir: Path | None = None) -> tuple[Path, Path]:
    """Synthesize + modulate the scripted argument into one WAV plus metadata.

    Returns (wav_path, meta_path). Raises RuntimeError if DEEPGRAM_API_KEY is
    absent — synthesis needs it (the live test is gated on the same key).
    """
    api_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set — cannot synthesize fixture")

    out_dir = out_dir or (_REPO_ROOT / "tmp")
    out_dir.mkdir(parents=True, exist_ok=True)

    gap = np.zeros(int(GAP_SECONDS * TARGET_SR), dtype=np.float32)
    segments: list[np.ndarray] = []
    meta_turns: list[dict] = []
    cursor = 0.0
    for turn in SCRIPT:
        voice = VOICE_A if turn["speaker"] == "A" else VOICE_B
        pcm = _modulate(_speak(turn["text"], voice, api_key), turn["emotion"])
        start = cursor
        end = start + pcm.shape[0] / TARGET_SR
        meta_turns.append({
            "speaker": f"Speaker {turn['speaker']}",
            "text": turn["text"],
            "scripted_emotion": turn["emotion"],
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "expected": turn["expected"],
        })
        segments.append(pcm)
        segments.append(gap)
        cursor = end + GAP_SECONDS

    final = np.concatenate(segments).astype(np.float32)
    pcm16 = (np.clip(final, -1.0, 1.0) * 32767).astype("<i2")

    wav_path = out_dir / "test_recording.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm16.tobytes())

    meta_path = out_dir / "test_recording_meta.json"
    meta = {"sample_rate": TARGET_SR, "turns": meta_turns}
    meta_path.write_text(json.dumps(meta, indent=2))
    return wav_path, meta_path


def main() -> int:
    try:
        wav_path, meta_path = generate()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text())
    print(f"wrote {wav_path}")
    print(f"wrote {meta_path}")
    print(json.dumps(meta["turns"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
