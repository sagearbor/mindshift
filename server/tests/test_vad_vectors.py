"""Golden-vector contract test for server/watch/diarize.py's speech_segments.

The cases live in server/tests/fixtures/policy_vectors/vad_segments.json (see
the README next to it). Each case describes its PCM as constant-loudness
stretches rather than shipping sample data, so the fixture stays small,
diff-able and generatable by any language: this file's ``synthesize`` is the
reference generator the "_schema.signal.generator" text describes — a port
that reproduces it will get bit-identical int16 input.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from watch.diarize import speech_segments
from watch.vectors import rms_dbfs

VECTORS_PATH = Path(__file__).parent / "fixtures" / "policy_vectors" / "vad_segments.json"

INT16_FULL_SCALE = 32768.0


def _load_cases() -> list[dict]:
    with VECTORS_PATH.open() as f:
        doc = json.load(f)
    assert doc["_schema"]["version"] == 1
    return doc["cases"]


CASES = _load_cases()


def synthesize(case: dict) -> np.ndarray:
    """Reference PCM generator for a case's ``signal`` spec (int16 mono)."""
    sr = case["sample_rate"]
    tone_hz = case.get("tone_hz", 150.0)
    parts: list[np.ndarray] = []
    for stretch in case["signal"]:
        n = int(stretch["seconds"] * sr)
        if stretch["dbfs"] is None:
            parts.append(np.zeros(n, dtype=np.int16))
            continue
        # RMS of a sine is amplitude/sqrt(2); solve for the amplitude that
        # lands the stretch on the requested dBFS. Phase restarts at 0 per
        # stretch (the spec says so — it keeps generators trivially identical).
        amplitude = INT16_FULL_SCALE * (10.0 ** (stretch["dbfs"] / 20.0)) * np.sqrt(2.0)
        t = np.arange(n) / sr
        parts.append((np.sin(2 * np.pi * tone_hz * t) * amplitude).astype(np.int16))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case_segments_match(case):
    pcm = synthesize(case)
    got = speech_segments(pcm, case["sample_rate"], **case["config"])
    want = [(s["start_s"], s["end_s"]) for s in case["expected"]]
    tol = case["tolerance_s"]

    assert len(got) == len(want), f"{case['name']}: got {got}, want {want}"
    for (gs, ge), (ws, we) in zip(got, want):
        assert abs(gs - ws) <= tol, f"{case['name']}: start {gs} vs {ws}"
        assert abs(ge - we) <= tol, f"{case['name']}: end {ge} vs {we}"


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_generator_hits_requested_loudness(case):
    """The fixture's contract is 'this stretch IS N dBFS'. Verify the reference
    generator honours it (within 0.5 dB — sub-cycle truncation on stretches
    that aren't a whole number of periods) so a port that copies the formula
    can trust the numbers, and digital silence really is silence."""
    sr = case["sample_rate"]
    pcm = synthesize(case)
    offset = 0
    for stretch in case["signal"]:
        n = int(stretch["seconds"] * sr)
        chunk = pcm[offset:offset + n]
        offset += n
        measured = rms_dbfs(chunk)
        if stretch["dbfs"] is None:
            assert measured == float("-inf")
        else:
            assert abs(measured - stretch["dbfs"]) <= 0.5, f"{case['name']}: {measured} vs {stretch['dbfs']}"
    assert offset == pcm.size


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_fixture_is_well_formed(case):
    assert case["sample_rate"] > 0
    assert set(case["config"]) == {"floor_dbfs", "frame_seconds", "merge_gap_seconds", "min_seconds"}
    assert case["signal"], "a case has at least one stretch"
    for stretch in case["signal"]:
        assert isinstance(stretch["seconds"], float) and stretch["seconds"] > 0
        assert stretch["dbfs"] is None or isinstance(stretch["dbfs"], float)
    last_end = 0.0
    for span in case["expected"]:
        assert isinstance(span["start_s"], float) and isinstance(span["end_s"], float)
        assert span["start_s"] >= last_end, "spans are ordered and non-overlapping"
        assert span["end_s"] > span["start_s"]
        assert span["end_s"] - span["start_s"] >= case["config"]["min_seconds"], "no span survives under min_seconds"
        last_end = span["end_s"]


def test_coverage_of_required_scenarios():
    names = {c["name"] for c in CASES}
    assert len(names) == len(CASES), "case names are unique"
    assert len(CASES) >= 6
    required = {
        "silence_only",
        "one_burst",
        "two_bursts_merged_across_short_gap",
        "two_bursts_kept_apart_across_long_gap",
        "burst_too_short_dropped",
        "mixed_conversation_shape",
    }
    assert required <= names
