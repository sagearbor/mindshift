# Ported from gauge@2157433 server/tests/test_vectors.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
import numpy as np
import pytest
from watch.models import EnrollmentBaseline
from watch.vectors import VectorEngine


def pcm(amplitude: float, freq: float = 150.0, seconds: float = 1.0, sr: int = 16000) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16).tobytes()


BASE = EnrollmentBaseline(account_id="a", rms_db=-30.0, f0_median=150.0, updated_at="2026-07-31T00:00:00Z")


def test_quiet_speech_no_events():
    eng = VectorEngine(BASE)
    assert eng.push_pcm(pcm(0.03)) == []          # ~ -30 dBFS ≈ baseline


def test_yelling_levels_scale_with_loudness():
    eng = VectorEngine(BASE)
    evs = eng.push_pcm(pcm(0.32))                  # ≈ -10 dBFS = +20 dB over baseline
    ys = [e for e in evs if e.vector == "yelling"]
    assert ys and ys[0].level == 3 and ys[0].value > 14


def test_aggressive_tone_requires_sustained_high_pitch_and_loud():
    eng = VectorEngine(BASE)
    for _ in range(4):
        evs = eng.push_pcm(pcm(0.15, freq=220.0))  # loud + 1.47x baseline F0
    assert any(e.vector == "aggressive_tone" and e.level >= 2 for e in evs)


def test_hr_spike():
    eng = VectorEngine(BASE)
    for bpm in (64, 65, 66, 65, 64):
        assert eng.push_hr(bpm, t=0.0) == []
    evs = eng.push_hr(95, t=10.0)                  # +31 over resting 64
    assert evs and evs[0].vector == "hr_spike" and evs[0].level == 2


def test_airtime_and_interrupting():
    eng = VectorEngine(BASE)
    evs = eng.push_diarization([("self", 0.0, 50.0), ("other", 49.0, 55.0), ("self", 54.0, 90.0)])
    names = {e.vector for e in evs}
    assert "interrupting" in names                 # self restarts 1.0s before other ends
    assert any(e.vector == "airtime" and e.level >= 2 for e in evs)  # ~87% share


def test_airtime_excludes_silence_from_denominator():
    # Per design spec §4.1, airtime is self's share of SPEECH time, not
    # wall-clock time — long silence between turns must not dilute the
    # share. Window membership (last 120s) is still wall-clock, but the
    # ratio itself is self-speech / (self-speech + other-speech).
    eng = VectorEngine(BASE)
    eng.push_diarization([("self", 0.0, 50.0), ("other", 50.0, 55.0)])
    evs = eng.push_diarization([("self", 110.0, 115.0)])   # 55s of silence before this turn
    airtime = [e for e in evs if e.vector == "airtime"]
    assert airtime and airtime[0].level == 3               # 55s self / 60s total speech ≈ 0.92
    assert airtime[0].value == pytest.approx(55 / 60, abs=0.01)
