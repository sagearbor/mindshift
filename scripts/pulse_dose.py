"""How OFTEN does the wrist buzz? — the dose question, measured on real audio.

Every haptic test in this repo checks that the RIGHT cue plays. None counted
how often, and on 2026-09-10 the owner reported the watch "buzzing all over the
place" during a real session. This script answers it by porting
`apps/watch/shared/.../SentinelDetector.kt` and `.../PulseEngine.kt` EXACTLY —
same seeding rule, same silence floor, same "loud windows never feed the
baseline" rule, same 1 s window — and counting the pulses over a recording.

Porting rather than approximating matters: the server's own baseline
(`watch/vectors.py` running_median) is a DIFFERENT algorithm with a different
cold-start behaviour, and using it here would produce numbers the watch never
actually experiences.

Usage:  python scripts/pulse_dose.py [wav ...]      (default: the real fixtures)
Prints the per-window dbOverBaseline envelope (the input to
apps/watch/.../PulseDoseTest.kt) and the buzzes-per-minute dose.
"""
import os
import sys
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "server/tests/fixtures/audio")

# --- mirrored constants (apps/watch/shared/.../SentinelDetector.kt) ---
SILENCE_FLOOR_DBFS = -45.0
BASELINE_WINDOW_COUNT = 30
SEED_WINDOWS = 3
WINDOW_SECONDS = 1.0          # SentinelController MAX_SLICE_BYTES 32000 / 2 / 16000
SESSION_PULSE_THRESHOLD_DB = 6.0   # pulseThresholdDbFor(SESSION) -> STANDARD's trigger bar


def rms_dbfs(window: np.ndarray) -> float:
    """Dsp.rmsDbfs: 20*log10(rms/32768) on the int16 scale."""
    rms = float(np.sqrt(np.mean(np.square(window.astype(np.float64)))))
    return -120.0 if rms <= 0 else 20.0 * np.log10(rms / 32768.0)


def observe_all(windows):
    """SentinelDetector.observe(), window by window -> list of dbOverBaseline.

    Returns the same number the device hands PulseEngine, including the 0.0 it
    uses for an unvoiced window (that is the detector's own convention, and it
    is why silence can never pulse).
    """
    voiced_history: list[float] = []
    baseline = None
    out = []
    for w in windows:
        dbfs = rms_dbfs(w)
        if dbfs <= SILENCE_FLOOR_DBFS:
            out.append(0.0)
            continue
        if len(voiced_history) < SEED_WINDOWS:
            if not voiced_history:
                baseline = dbfs
            voiced_history.append(dbfs)
            if len(voiced_history) >= SEED_WINDOWS:
                baseline = float(np.median(voiced_history))
            out.append(dbfs - (baseline if baseline is not None else dbfs))
            continue
        over = dbfs - (baseline if baseline is not None else dbfs)
        if over < SESSION_PULSE_THRESHOLD_DB:
            # Only non-triggering windows feed the baseline, so sustained
            # loudness can never drag the bar up and silence itself.
            voiced_history.append(dbfs)
            del voiced_history[:-BASELINE_WINDOW_COUNT]
            baseline = float(np.median(voiced_history))
        out.append(over)
    return out


def pulses(envelope, interval_ms, window_ms=int(WINDOW_SECONDS * 1000)):
    """PulseEngine.onWindow(): one pulse per window over threshold, subject to a
    minimum spacing. The spacing is measured here (not assumed irrelevant) so
    the script shows for itself that a 250/500/1000 ms preference cannot bind
    against a 1000 ms window."""
    if interval_ms is None:
        return 0
    last, now, n = None, 0, 0
    for over in envelope:
        if over < SESSION_PULSE_THRESHOLD_DB:
            # PulseEngine.onWindow resets its spacing clock on any below-threshold
            # window, so the clamp only ever applies WITHIN one continuous loud
            # streak. Mirrored here rather than approximated.
            last = None
        elif last is None or now - last >= interval_ms:
            n += 1
            last = now
        now += window_ms
    return n


def windows_of(path):
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), "<i2")
    x = raw.reshape(-1, ch).mean(axis=1) if ch > 1 else raw
    n = int(sr * WINDOW_SECONDS)
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)]


def main(paths):
    for path in paths:
        if not os.path.exists(path):
            print(f"{os.path.basename(path)}: MISSING (not committed?) — skipped")
            continue
        env = observe_all(windows_of(path))
        secs = len(env) * WINDOW_SECONDS
        name = os.path.basename(path)
        print(f"\n{name}  ({secs:.0f}s, {len(env)} windows)")
        print("  dbOverBaseline: " + ", ".join(f"{v:.2f}" for v in env))
        for iv in (250, 500, 1000, None):
            n = pulses(env, iv)
            label = "off" if iv is None else f"{iv}ms"
            print(f"  pulse pref {label:>6}: {n:>3} buzzes  =  {n * 60 / secs:5.1f} per minute")


if __name__ == "__main__":
    args = sys.argv[1:] or [
        os.path.join(FIXTURES, f"test_recording_{n}.wav") for n in ("family_real", "poker6_real")
    ]
    main(args)
