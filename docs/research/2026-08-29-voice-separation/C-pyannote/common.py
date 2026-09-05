"""Shared helpers for approach C (pyannote). Runs under tmp/venv-pyannote.
No absolute paths: everything is resolved relative to this file."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = HERE.parents[3]
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

# --- shared scorer (stdlib only) --------------------------------------------
_spec = importlib.util.spec_from_file_location("score", RESEARCH / "score.py")
score = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score)


def fixture_audio(name: str) -> str:
    """Path to a 16 kHz mono WAV for the fixture. maggiano3 is decoded (by
    ffmpeg, see README) into this directory and never copied elsewhere."""
    fx = score.load_fixture(name)
    if name == "maggiano3":
        p = HERE / "maggiano3_16k.wav"
        if not p.exists():
            raise FileNotFoundError("decode the private clip first (README)")
        return str(p)
    return fx["audio_path"]


def load_wav(path: str) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def load_env_token() -> str | None:
    """HF_TOKEN from the environment or the repo .env (never printed)."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def load_pipeline(name: str = "pyannote/speaker-diarization-3.1"):
    """Load a pyannote pipeline on CPU (Cloud Run has no GPU; MPS is
    deliberately NOT used so timings transfer)."""
    import torch
    _orig = torch.load

    def _patched(*a, **k):
        k["weights_only"] = False
        return _orig(*a, **k)

    torch.load = _patched
    from pyannote.audio import Pipeline
    p = Pipeline.from_pretrained(name, use_auth_token=load_env_token())
    torch.load = _orig
    if p is None:
        raise RuntimeError(f"{name}: gated / licence not accepted for this token")
    return p


def annotation_to_pred(ann) -> list[list]:
    return [[round(float(seg.start), 4), round(float(seg.end), 4), str(lab)]
            for seg, _, lab in ann.itertracks(yield_label=True)]


def write_pred(fixture: str, variant: str, pred: list) -> Path:
    p = HERE / "preds" / f"pred_{fixture}__{variant}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(pred))
    return p


def scored(fixture: str, pred: list, **extra) -> dict:
    out = score.score_fixture(fixture, [tuple(x) for x in pred])
    out.update(extra)
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        pass

    @property
    def s(self) -> float:
        return time.perf_counter() - self.t0
