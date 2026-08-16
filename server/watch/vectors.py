# Ported from gauge@2157433 server/vectors.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
"""Streaming vector engine — turns live PCM/HR/diarization into behavior events.

This is a fresh, pure-numpy streaming DSP engine (separate from any one-shot
per-turn analysis over a completed recording). It is fed 1-second PCM16
windows, heart-rate samples, and diarization turns as a live session
unfolds, and emits ``VectorEvent``s.

Non-negotiable bias guard (from the design spec): every rule here measures
the wearer against their OWN enrollment baseline (``EnrollmentBaseline``), or
against a live-session running median when no baseline is enrolled yet —
never an absolute/universal threshold. The engine reports physics (dB over
baseline, seconds of overlap, share of airtime), never a judgment. Sensitivity
scaling (``VectorSubscription.sensitivity``) is applied by the caller's nudge
policy, not here.

Mirror contract: this is the server-side counterpart of the on-watch
NudgeStateMachine at
apps/watch/shared/src/commonMain/kotlin/app/gauge/shared/NudgeStateMachine.kt
— thresholds/semantics must stay identical between the two.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from watch.models import EnrollmentBaseline, VectorEvent

# --- yelling ----------------------------------------------------------------
# Below this window RMS, we're looking at silence/noise floor, not speech —
# never report loudness vectors for it.
SILENCE_FLOOR_DBFS = -45.0
# (dB over baseline, level), checked highest-first.
YELLING_LEVELS: tuple[tuple[float, int], ...] = ((14.0, 3), (10.0, 2), (6.0, 1))

# --- aggressive_tone ----------------------------------------------------------
AGGRESSIVE_TONE_F0_RATIO = 1.3
F0_MIN_HZ = 50.0
F0_MAX_HZ = 400.0
# Normalized autocorrelation peak below this means "not periodic enough" ->
# unvoiced -> no F0, matching the honesty rule in engine/prosody.py.
VOICED_AUTOCORR_THRESHOLD = 0.35

# --- hr_spike -----------------------------------------------------------------
HR_RESTING_DEFAULT = 65.0
HR_RESTING_SAMPLE_COUNT = 5
HR_SPIKE_LEVELS: tuple[tuple[float, int], ...] = ((35.0, 3), (25.0, 2), (15.0, 1))

# --- interrupting / airtime -----------------------------------------------------
INTERRUPT_MIN_LEAD_S = 0.5
AIRTIME_WINDOW_S = 120.0
AIRTIME_LEVELS: tuple[tuple[float, int], ...] = ((0.9, 3), (0.75, 2), (0.6, 1))

# Bounded history length (in windows) for the live-session running-median
# fallback used when there's no enrollment baseline yet.
RUNNING_STAT_WINDOW = 60


def rms_dbfs(samples: np.ndarray) -> float:
    """``20*log10(rms/32768)`` for int16-scaled PCM; ``-inf`` for pure silence.

    Public (not module-private) so other callers — e.g. ``rest_api.py``'s
    ``/enroll`` handler — can reuse this exact DSP instead of duplicating it.
    """
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms / 32768.0)


def estimate_f0(samples: np.ndarray, sr: int) -> float | None:
    """Single-shot autocorrelation F0 estimate (Hz) over the whole window.

    Returns ``None`` when the window isn't periodic enough to be voiced — we
    never invent a pitch for silence/noise/unvoiced speech.
    """
    if samples.size < 2:
        return None
    x = samples.astype(np.float64)
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy <= 0.0:
        return None
    corr = np.correlate(x, x, mode="full")[x.size - 1:]
    min_lag = max(1, int(sr / F0_MAX_HZ))
    max_lag = min(int(sr / F0_MIN_HZ), corr.size - 1)
    if max_lag <= min_lag:
        return None
    window = corr[min_lag:max_lag + 1]
    peak_lag = min_lag + int(np.argmax(window))
    if peak_lag <= 0:
        return None
    if corr[peak_lag] / energy < VOICED_AUTOCORR_THRESHOLD:
        return None
    return sr / peak_lag


def _level_for(value: float, thresholds: tuple[tuple[float, int], ...]) -> int:
    """Highest level whose threshold ``value`` clears; 0 if none (thresholds
    must be given highest-first)."""
    for threshold, level in thresholds:
        if value >= threshold:
            return level
    return 0


def _running_median(history: deque[float]) -> float | None:
    if not history:
        return None
    return float(np.median(np.asarray(history, dtype=np.float64)))


class VectorEngine:
    """Streaming, baseline-relative behavior detector for one live session.

    Feed it 1-second PCM16 windows (``push_pcm``), heart-rate samples
    (``push_hr``), and diarization turns (``push_diarization``); it returns
    the ``VectorEvent``s each push produces. ``self.t`` is the running
    live-session clock, advanced by each pushed PCM window's duration (1.0 s
    for the standard 16 kHz/1 s window).
    """

    def __init__(self, baseline: EnrollmentBaseline | None, sample_rate: int = 16000):
        self.baseline = baseline
        self.sample_rate = sample_rate
        self.t: float = 0.0

        self._rms_db_history: deque[float] = deque(maxlen=RUNNING_STAT_WINDOW)
        self._f0_history: deque[float] = deque(maxlen=RUNNING_STAT_WINDOW)
        self._aggressive_streak = 0

        self._hr_history: list[float] = []

        self._turns: list[tuple[str, float, float]] = []

    # ------------------------------------------------------------------ pcm --
    def push_pcm(self, pcm: bytes, speaker: str = "self") -> list[VectorEvent]:
        window_t = self.t
        samples = np.frombuffer(pcm, dtype=np.int16)
        duration = (samples.size / self.sample_rate) if self.sample_rate else 1.0
        self.t += duration

        events: list[VectorEvent] = []
        if speaker != "self":
            # Only the wearer's own channel drives self-behavior vectors.
            return events

        dbfs = rms_dbfs(samples)
        if dbfs <= SILENCE_FLOOR_DBFS:
            self._aggressive_streak = 0
            return events

        baseline_db = self.baseline.rms_db if self.baseline else _running_median(self._rms_db_history)
        self._rms_db_history.append(dbfs)
        over_db = (dbfs - baseline_db) if baseline_db is not None else 0.0

        yelling_level = _level_for(over_db, YELLING_LEVELS)
        if yelling_level:
            events.append(VectorEvent(
                vector="yelling", level=yelling_level, t=window_t, value=over_db,
                detail=f"{over_db:.1f} dB over baseline",
            ))

        f0 = estimate_f0(samples, self.sample_rate)
        baseline_f0 = None
        if f0 is not None:
            baseline_f0 = self.baseline.f0_median if self.baseline else _running_median(self._f0_history)
            self._f0_history.append(f0)

        loud_enough = over_db >= YELLING_LEVELS[-1][0]  # clears the level-1 loudness bar
        high_pitch = f0 is not None and bool(baseline_f0) and f0 >= AGGRESSIVE_TONE_F0_RATIO * baseline_f0

        if loud_enough and high_pitch:
            self._aggressive_streak += 1
        else:
            self._aggressive_streak = 0

        if self._aggressive_streak >= 2:
            level = min(3, self._aggressive_streak - 1)
            ratio = f0 / baseline_f0
            events.append(VectorEvent(
                vector="aggressive_tone", level=level, t=window_t, value=ratio,
                detail=f"F0 {ratio:.2f}x baseline, sustained {self._aggressive_streak} windows",
            ))

        return events

    # ------------------------------------------------------------------- hr --
    def push_hr(self, bpm: float, t: float) -> list[VectorEvent]:
        self._hr_history.append(bpm)
        if len(self._hr_history) >= HR_RESTING_SAMPLE_COUNT:
            resting = min(self._hr_history[:HR_RESTING_SAMPLE_COUNT])
        else:
            resting = HR_RESTING_DEFAULT

        over = bpm - resting
        level = _level_for(over, HR_SPIKE_LEVELS)
        if not level:
            return []
        return [VectorEvent(
            vector="hr_spike", level=level, t=t, value=over,
            detail=f"{bpm:.0f} bpm, {over:.0f} over resting {resting:.0f}",
        )]

    # ----------------------------------------------------------- diarization --
    # Final-review Finding 2c: this is the ONLY source of "interrupting" and
    # "airtime" VectorEvents, and v1 has NO production caller for it — there's
    # no WS diarization frame in the wire protocol (see server/watch/routers/ws.py,
    # Task B11) and no diarization pipeline wired up anywhere. It's implemented
    # and tested now (server/tests/watch/test_vectors.py) so Plan 2 only has to
    # wire a diarization source in and call this; it activates neither vector
    # in v1. See also DEFAULT_VECTOR_NAMES's comment in server/watch/store.py.
    def push_diarization(self, turns: list[tuple[str, float, float]]) -> list[VectorEvent]:
        events: list[VectorEvent] = []
        context = self._turns + turns  # history + this batch, for interrupt lookups

        for speaker, start, end in turns:
            if speaker != "self":
                continue
            for other_speaker, other_start, other_end in context:
                if other_speaker == "self":
                    continue
                # An "other" turn already in progress that self cuts into,
                # with at least INTERRUPT_MIN_LEAD_S left before it would
                # naturally have ended.
                if other_start < start < other_end and (other_end - start) >= INTERRUPT_MIN_LEAD_S:
                    overlap = other_end - start
                    level = 3 if overlap >= 2.0 else 2 if overlap >= 1.0 else 1
                    events.append(VectorEvent(
                        vector="interrupting", level=level, t=start, value=overlap,
                        detail=f"self started {overlap:.1f}s before the other's turn ended",
                    ))

        self._turns.extend(turns)
        events.extend(self._airtime_events())
        return events

    def _airtime_events(self) -> list[VectorEvent]:
        # Per design spec §4.1, airtime is "your share of speaking time" —
        # the denominator is total SPEECH time within the trailing 120s
        # window, not wall-clock time. Window *membership* is still by
        # wall-clock recency (last 120s of turn activity); only the ratio
        # excludes silence, so a wearer who dominates every exchange but
        # with long silences in between still reads as high-share.
        if not self._turns:
            return []
        now = max(end for _, _, end in self._turns)
        window_start = max(0.0, now - AIRTIME_WINDOW_S)

        self_speech = 0.0
        other_speech = 0.0
        for speaker, start, end in self._turns:
            overlap = min(end, now) - max(start, window_start)
            if overlap <= 0:
                continue
            if speaker == "self":
                self_speech += overlap
            else:
                other_speech += overlap

        total_speech = self_speech + other_speech
        if total_speech <= 0:
            return []

        share = self_speech / total_speech
        level = _level_for(share, AIRTIME_LEVELS)
        if not level:
            return []
        return [VectorEvent(
            vector="airtime", level=level, t=now, value=share,
            detail=f"self {share:.0%} of speech in the last {AIRTIME_WINDOW_S:.0f}s",
        )]
