"""Experiment 2, step 1 — frame-level overlapped-speech probability from
pyannote's segmentation-3.0 (powerset) model. Runs under tmp/venv-pyannote
(HF_TOKEN from .env; the model is gated).

The model is run directly (10 s chunks, 2.5 s step, CPU) — not through the
diarization pipeline — and its powerset posteriors are turned into
    p_overlap(t) = sum of the posteriors of every class with >= 2 speakers
    p_speech(t)  = 1 - posterior of the empty class
averaged over the chunks covering each 10 ms frame. Written to
cache/overlap_<fixture>.npz (t, p_overlap, p_speech); the maggiano3 one is
derived from the private clip and stays local (cache/ is gitignored).

Usage: python exp2_overlap_pyannote.py [fixture ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "2026-08-29-voice-separation"
ROOT = HERE.parents[2]
sys.path.insert(0, str(RESEARCH))
import score  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
CHUNK_S, STEP_S, GRID = 10.0, 2.5, 0.01
DEFAULT = ["maggiano3", "scene_family3", "scene_meeting4", "family_real", "poker6", "scene_couple"]


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def audio_path(name: str) -> str:
    if name == "maggiano3":
        return str(HERE / "maggiano3_16k.wav")
    return score.load_fixture(name)["audio_path"]


def load16k(path: str) -> np.ndarray:
    import torch
    import torchaudio
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        x = torchaudio.functional.resample(torch.from_numpy(x.astype(np.float32)), sr, 16000).numpy()
    return np.ascontiguousarray(x, dtype=np.float32)


def main(names: list[str]) -> None:
    import torch
    torch.set_num_threads(4)
    _orig = torch.load                      # pyannote checkpoints predate weights_only=True (C-pyannote/common.py)
    torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
    from pyannote.audio import Model
    from pyannote.audio.utils.powerset import Powerset
    t0 = time.perf_counter()
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=hf_token())
    model.eval()
    spec = model.specifications
    ps = Powerset(len(spec.classes), spec.powerset_max_classes)
    card = ps.mapping.sum(dim=1).numpy()          # speakers per powerset class
    rf = model.receptive_field                     # frame SlidingWindow
    print(f"model loaded in {time.perf_counter() - t0:.1f}s; classes {list(spec.classes)}, powerset "
          f"cardinalities {card.tolist()}, frame step {rf.step * 1000:.2f} ms, duration {rf.duration * 1000:.1f} ms")
    summary = {}
    for name in names:
        x = load16k(audio_path(name))
        dur = x.size / 16000
        n = int(dur / GRID) + 1
        acc_ov, acc_sp, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
        chunk, step = int(CHUNK_S * 16000), int(STEP_S * 16000)
        starts = list(range(0, max(1, x.size - chunk + 1), step))
        if not starts or starts[-1] + chunk < x.size:
            starts.append(max(0, x.size - chunk))
        t1 = time.perf_counter()
        with torch.no_grad():
            for s in starts:
                seg = x[s:s + chunk]
                real = seg.size
                if real < chunk:
                    seg = np.pad(seg, (0, chunk - real))
                logp = model(torch.from_numpy(seg)[None, None, :])[0].numpy()   # (frames, classes)
                p = np.exp(logp)
                p_ov = p[:, card >= 2].sum(axis=1)
                p_sp = 1.0 - p[:, card == 0]
                for f in range(p.shape[0]):
                    tc = s / 16000 + rf.start + f * rf.step + rf.duration / 2
                    if tc < s / 16000 or tc > (s + real) / 16000:
                        continue
                    a, b = int(round((tc - rf.step / 2) / GRID)), int(round((tc + rf.step / 2) / GRID))
                    a, b = max(0, a), min(n, max(a + 1, b))
                    acc_ov[a:b] += p_ov[f]; acc_sp[a:b] += p_sp[f]; cnt[a:b] += 1
        dt = time.perf_counter() - t1
        ok = cnt > 0
        p_overlap = np.where(ok, acc_ov / np.maximum(cnt, 1), 0.0)
        p_speech = np.where(ok, acc_sp / np.maximum(cnt, 1), 0.0)
        t = np.arange(n) * GRID
        np.savez(CACHE / f"overlap_{name}.npz", t=t, p_overlap=p_overlap, p_speech=p_speech)
        flagged = float((p_overlap > 0.5).sum() * GRID)
        summary[name] = {"duration_s": round(dur, 2), "flagged_s_p>0.5": round(flagged, 2),
                         "flagged_s_p>0.3": round(float((p_overlap > 0.3).sum() * GRID), 2),
                         "speech_s_p>0.5": round(float((p_speech > 0.5).sum() * GRID), 2),
                         "chunks": len(starts), "t_infer_s": round(dt, 2)}
        print(name, json.dumps(summary[name]), flush=True)
    (CACHE / "overlap_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT)
