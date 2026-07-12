"""Per-turn prosody (pure numpy) for POST /analyze/upload.

Given the recording's PCM + each turn's start/end timestamps, we measure three
delivery signals and turn them into compact categorical labels:

* energy   — RMS amplitude over the turn window  → quiet / normal / loud
* pitch    — median F0 + F0 variability (frame-wise autocorrelation)
             → low / mid / high pitch, and flat / varied
* speech_rate — words / duration                 → slow / normal / fast

Two deliberate honesty rules:

1. Pitch is ``None`` when a turn is mostly unvoiced (silence, noise, non-speech):
   fewer than ``MIN_VOICED_FRACTION`` of frames show enough periodicity. We
   never invent an F0 for something that isn't voiced.

2. Labels are RELATIVE to *this recording's own* distribution (tertiles/median
   over its turns), not to any absolute dB/Hz scale. That is the correct choice
   here: microphone gain, distance, and device vary per recording, so "loud"
   can only mean "loud for this speaker on this recording". The LLM is told the
   labels are relative delivery cues, not absolute measurements.

Everything is a plain function over numpy arrays / dicts — trivially unit
testable with synthetic signals, no framework coupling.
"""

from __future__ import annotations

import numpy as np

# Frame-wise F0 estimation params. 40 ms frames / 10 ms hop is a standard
# pitch-analysis window; the 60–400 Hz search spans low male speech through
# high female/child speech.
FRAME_MS = 40.0
HOP_MS = 10.0
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0
# A frame is "voiced" when its normalized autocorrelation peak in the F0 lag
# range clears this. A clean periodic tone sits near 1.0; broadband noise and
# silence sit well below.
VOICED_AUTOCORR_THRESHOLD = 0.35
# Below this fraction of voiced frames we report pitch as None (honest null).
MIN_VOICED_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Single-turn feature extraction
# ---------------------------------------------------------------------------

def _turn_slice(pcm: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """The PCM samples spanning [start, end) seconds, clipped to the array."""
    if sr <= 0 or end <= start:
        return np.empty(0, dtype=np.float32)
    a = max(0, int(round(start * sr)))
    b = min(pcm.shape[0], int(round(end * sr)))
    if b <= a:
        return np.empty(0, dtype=np.float32)
    return pcm[a:b]


def rms_energy(samples: np.ndarray) -> float:
    """Root-mean-square amplitude (0.0 for an empty window)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _frame_f0(frame: np.ndarray, sr: int) -> float | None:
    """Estimate one frame's F0 via autocorrelation, or ``None`` if unvoiced.

    The frame is mean-removed, autocorrelated, and normalized by its zero-lag
    energy. The strongest peak within the 60–400 Hz lag window decides voicing:
    a normalized peak below ``VOICED_AUTOCORR_THRESHOLD`` means "not periodic
    enough" → unvoiced → ``None``.
    """
    frame = frame.astype(np.float64)
    frame = frame - frame.mean()
    energy = float(np.dot(frame, frame))
    if energy <= 0.0:
        return None
    corr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
    min_lag = int(sr / F0_MAX_HZ)
    max_lag = int(sr / F0_MIN_HZ)
    max_lag = min(max_lag, corr.size - 1)
    if max_lag <= min_lag:
        return None
    window = corr[min_lag:max_lag + 1]
    peak_offset = int(np.argmax(window))
    peak_lag = min_lag + peak_offset
    if peak_lag <= 0:
        return None
    if corr[peak_lag] / energy < VOICED_AUTOCORR_THRESHOLD:
        return None
    return sr / peak_lag


def estimate_pitch(
    samples: np.ndarray, sr: int,
) -> tuple[float | None, float | None, float]:
    """(median_f0, f0_std, voiced_fraction) over a turn's samples.

    ``median_f0`` and ``f0_std`` are ``None`` when fewer than
    ``MIN_VOICED_FRACTION`` of frames are voiced — an honest null for
    silence/noise rather than a fabricated pitch.
    """
    if sr <= 0 or samples.size == 0:
        return None, None, 0.0
    frame_len = max(1, int(sr * FRAME_MS / 1000.0))
    hop = max(1, int(sr * HOP_MS / 1000.0))
    if samples.size < frame_len:
        return None, None, 0.0

    f0s: list[float] = []
    n_frames = 0
    for start in range(0, samples.size - frame_len + 1, hop):
        n_frames += 1
        f0 = _frame_f0(samples[start:start + frame_len], sr)
        if f0 is not None:
            f0s.append(f0)

    if n_frames == 0:
        return None, None, 0.0
    voiced_fraction = len(f0s) / n_frames
    if voiced_fraction < MIN_VOICED_FRACTION or not f0s:
        return None, None, voiced_fraction
    arr = np.asarray(f0s, dtype=np.float64)
    return float(np.median(arr)), float(np.std(arr)), voiced_fraction


def turn_features(pcm: np.ndarray, sr: int, start: float, end: float) -> dict:
    """Raw prosody numbers for one turn window (no labels yet).

    Keys: ``rms``, ``f0_median`` (None if unvoiced), ``f0_std`` (None if
    unvoiced), ``voiced_fraction``.
    """
    samples = _turn_slice(pcm, sr, start, end)
    f0_median, f0_std, voiced_fraction = estimate_pitch(samples, sr)
    return {
        "rms": rms_energy(samples),
        "f0_median": f0_median,
        "f0_std": f0_std,
        "voiced_fraction": voiced_fraction,
    }


# ---------------------------------------------------------------------------
# Relative labeling across the recording's own turns
# ---------------------------------------------------------------------------


# The exact label vocabularies the mobile client types against
# (apps/mobile/src/api/client.ts `Voice`). Locked by test — do not drift.
ENERGY_LABELS = ("quiet", "normal", "loud")
PITCH_LABELS = ("low", "mid", "high")
RATE_LABELS = ("slow", "normal", "fast")

def _tertile_label(
    value: float, values: list[float], labels: tuple[str, str, str],
) -> str:
    """Label ``value`` low/mid/high by its position in ``values``' tertiles.

    Uses the 33rd/66th percentiles of the recording's own distribution — so
    labels are always relative to this recording. Degenerate distributions (all
    equal) collapse to the middle label.
    """
    arr = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(arr, [100.0 / 3.0, 200.0 / 3.0])
    if lo == hi:
        # Degenerate distribution (all values equal, or a single turn): nothing
        # stands out, so everything is baseline — the middle label. Without
        # this, `value <= lo` labels every turn of an evenly-delivered
        # recording "quiet"/"slow", which the UI then renders as noteworthy.
        return labels[1]
    if value <= lo:
        return labels[0]
    if value <= hi:
        return labels[1]
    return labels[2]


def label_turns(
    features: list[dict], transcript_turns: list[dict],
) -> list[dict]:
    """Compact per-turn voice labels + the raw numbers behind them.

    ``features[i]`` is a :func:`turn_features` dict; ``transcript_turns[i]``
    supplies the text (for speech rate) and timestamps (for duration). Each
    output dict carries:

    * ``energy_label``  — quiet / normal / loud   (relative to this recording)
    * ``pitch_label``   — low / mid / high pitch, or ``None`` when unvoiced
    * ``pitch_var_label`` — flat / varied, or ``None`` when unvoiced
    * ``rate_label``    — slow / normal / fast
    * raw ``rms``, ``f0_median``, ``f0_std``, ``speech_rate``

    Energy/pitch/rate are labelled by tertiles over the recording's own turns
    (see module docstring for why relative is correct here).
    """
    n = len(features)
    rms_values = [f["rms"] for f in features]

    # Speech rate = words / duration; duration from the turn's own timestamps.
    rates: list[float] = []
    for f, t in zip(features, transcript_turns):
        start = t.get("start_time") or 0.0
        end = t.get("end_time") or 0.0
        duration = end - start
        words = len((t.get("text") or "").split())
        rates.append(words / duration if duration > 0 else 0.0)

    # Pitch level + variability are only meaningful over VOICED turns; unvoiced
    # turns get a None label rather than being forced onto the scale.
    voiced_idx = [i for i in range(n) if features[i]["f0_median"] is not None]
    voiced_f0 = [features[i]["f0_median"] for i in voiced_idx]
    voiced_std = [features[i]["f0_std"] for i in voiced_idx]
    std_median = float(np.median(voiced_std)) if voiced_std else 0.0

    out: list[dict] = []
    for i in range(n):
        f = features[i]
        energy_label = _tertile_label(
            f["rms"], rms_values, ENERGY_LABELS,
        )
        rate_label = _tertile_label(
            rates[i], rates, RATE_LABELS,
        )
        if f["f0_median"] is None:
            pitch_label = None
            pitch_var_label = None
        else:
            pitch_label = _tertile_label(
                f["f0_median"], voiced_f0,
                PITCH_LABELS,
            )
            # Binary flat/varied around the recording's median F0 spread.
            pitch_var_label = "varied" if f["f0_std"] > std_median else "flat"
        out.append({
            "energy_label": energy_label,
            "pitch_label": pitch_label,
            "pitch_var_label": pitch_var_label,
            "rate_label": rate_label,
            "rms": round(f["rms"], 6),
            "f0_median": (
                None if f["f0_median"] is None else round(f["f0_median"], 2)
            ),
            "f0_std": None if f["f0_std"] is None else round(f["f0_std"], 2),
            "speech_rate": round(rates[i], 3),
        })
    return out


def annotate(labels: dict) -> str:
    """One-line delivery annotation for the LLM prompt, e.g. ``loud, fast,
    pitch varied``. Built purely from the labels — no second model call."""
    parts = [labels["energy_label"], labels["rate_label"]]
    if labels["pitch_var_label"] is not None:
        parts.append(f"pitch {labels['pitch_var_label']}")
    else:
        parts.append("pitch unclear")
    return ", ".join(parts)
