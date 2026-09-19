"""Experiment 3, step 1 — DeepFilterNet3 enhancement (CPU). Runs under
tmp/venv-dfn (Python 3.11: DeepFilterLib 0.5.6 ships no cp312 wheel and this
Mac has no Rust toolchain, so `pip install deepfilternet` failed in
tmp/venv-voice; a python@3.11 venv with the prebuilt arm64 wheel + torch 2.5
was used instead).

For each cache/<name>_src.wav (16 kHz, production-decoded):
  16 kHz -> 48 kHz (torchaudio sinc resample) -> df.enhance (DeepFilterNet3,
  default full attenuation; also a 12 dB attenuation-limited variant)
  -> 16 kHz, trimmed/padded to the source's sample count, then the
  residual lag is measured by cross-correlation and, if non-zero, removed —
  so window/transcript timings stay valid.
Writes cache/<name>_dfn.wav, cache/<name>_dfn12.wav and cache/dfn_timing.json
(wall + CPU seconds per audio second, measured on this Mac with 4 threads).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from scipy.signal import correlate

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
THREADS = 4
NAMES = [a for a in sys.argv[1:] if not a.startswith("--")] or ["maggiano3", "poker6"]
VARIANTS = {"dfn": None, "dfn12": 12.0}      # atten_lim_db


def read(path: Path) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(path)
    if x.dtype != np.float32:
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def lag_of(x: np.ndarray, y: np.ndarray, max_lag: int = 4800) -> int:
    """Lag (samples) that best aligns y to x, searched within +/- max_lag."""
    c = correlate(y, x, mode="full", method="fft")
    mid = len(x) - 1
    w = c[mid - max_lag: mid + max_lag + 1]
    return int(np.argmax(w) - max_lag)


def frame_p10_rms(x: np.ndarray, sr: int) -> float:
    n = sr * 30 // 1000
    m = x.size // n
    rms = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1))
    return float(np.percentile(rms, 10))


def main() -> None:
    torch.set_num_threads(THREADS)
    from df import enhance, init_df
    t0 = time.perf_counter()
    model, df_state, _ = init_df()
    sr_df = df_state.sr()
    timing = {"model_load_s": round(time.perf_counter() - t0, 2), "threads": THREADS, "df_sr": sr_df, "files": {}}
    for name in NAMES:
        x, sr = read(CACHE / f"{name}_src.wav")
        for tag, atten in VARIANTS.items():
            wall0, cpu0 = time.perf_counter(), time.process_time()
            x48 = torchaudio.functional.resample(torch.from_numpy(x), sr, sr_df)
            with torch.no_grad():
                y48 = enhance(model, df_state, x48[None, :], atten_lim_db=atten)
            y = torchaudio.functional.resample(y48, sr_df, sr)[0].numpy().astype(np.float32)
            wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
            if y.size < x.size:
                y = np.pad(y, (0, x.size - y.size))
            y = y[: x.size]
            lag = lag_of(x, y)
            if lag > 0:
                y = np.concatenate([y[lag:], np.zeros(lag, np.float32)])
            elif lag < 0:
                y = np.concatenate([np.zeros(-lag, np.float32), y[:lag]])
            lag_after = lag_of(x, y)
            wavfile.write(CACHE / f"{name}_{tag}.wav", sr, y)
            removed = x - y
            info = {
                "audio_s": round(x.size / sr, 2), "wall_s": round(wall, 2), "cpu_s": round(cpu, 2),
                "wall_per_audio_s": round(wall / (x.size / sr), 4), "cpu_per_audio_s": round(cpu / (x.size / sr), 4),
                "lag_samples_before_fix": lag, "lag_samples_after_fix": lag_after,
                "rms_in": round(float(np.sqrt((x ** 2).mean())), 5), "rms_out": round(float(np.sqrt((y ** 2).mean())), 5),
                "rms_removed": round(float(np.sqrt((removed ** 2).mean())), 5),
                "noise_floor_p10_rms_in": round(frame_p10_rms(x, sr), 5),
                "noise_floor_p10_rms_out": round(frame_p10_rms(y, sr), 5),
            }
            timing["files"][f"{name}_{tag}"] = info
            print(name, tag, json.dumps(info), flush=True)
    (CACHE / "dfn_timing.json").write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
