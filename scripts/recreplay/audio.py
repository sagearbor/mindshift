"""Audio in: find the recording in a drop folder and normalise it with
ffmpeg to the wire format the phone streams (16 kHz mono PCM16 WAV)."""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

AUDIO_EXTS = (".m4a", ".mp3", ".wav", ".ogg", ".webm", ".aac", ".flac", ".mp4", ".3gp", ".amr", ".opus")
SAMPLE_RATE = 16000
# Homebrew on Apple silicon / Intel: a fresh shell on the owner's Mac often
# lacks /opt/homebrew/bin on PATH (auto-memory: user-devices-and-machine).
_EXTRA_BIN = ("/opt/homebrew/bin", "/usr/local/bin")


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for d in _EXTRA_BIN:
        p = os.path.join(d, "ffmpeg")
        if os.access(p, os.X_OK):
            return p
    return None


def find_audio(folder: Path, name: str) -> Path:
    folder = Path(folder)
    for ext in AUDIO_EXTS:
        for cand in (folder / f"{name}{ext}", folder / f"{name}{ext.upper()}"):
            if cand.exists():
                return cand
    # any single audio file in the folder (voice-memo apps pick their own names)
    found = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_EXTS] if folder.is_dir() else []
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"no audio for {name!r} in {folder} (looked for {name}.{{{','.join(e[1:] for e in AUDIO_EXTS)}}})"
    )


def normalize(src: Path, dst: Path, *, timeout_s: float = 300.0) -> dict:
    """ffmpeg -> 16 kHz mono s16 WAV. Raises RuntimeError with ffmpeg's own
    message when it fails (never a silent empty file)."""
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg is not installed (brew install ffmpeg)")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    proc = subprocess.run(
        [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", "-f", "wav", str(tmp)],
        capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 44:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg could not decode {src.name}: {proc.stderr.strip()[-400:]}")
    tmp.replace(dst)
    pcm = read_wav16(dst)
    return {"path": str(dst), "sample_rate": SAMPLE_RATE, "channels": 1,
            "duration_s": round(len(pcm) / SAMPLE_RATE, 3), "source": str(src)}


def read_wav16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz mono int16, got {w.getframerate()} Hz/{w.getnchannels()} ch/{w.getsampwidth() * 8}-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
