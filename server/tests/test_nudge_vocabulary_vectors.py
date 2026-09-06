"""Golden-vector driver for the nudge vocabulary.

Replays ``server/tests/fixtures/policy_vectors/nudge_vocabulary.json`` against
:mod:`server.nudge_vocabulary`. The TypeScript mirror
(``apps/mobile/__tests__/nudgeVocabulary.test.ts``) and the Kotlin mirror
(``apps/watch/shared/src/jvmTest/.../NudgeVocabularyVectorsTest.kt``) replay the
SAME file, so an emoji or a millisecond can only change in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nudge_vocabulary import (
    HAPTIC_GAP_EXCEPTIONS,
    MIN_AMPLITUDE,
    MIN_GAP_MS,
    MIN_ON_MS,
    NUDGE_VOCABULARY,
    POSITIVE_CAP_S,
    PositiveNudgeGate,
    code_for_vectors,
    haptic_for,
    icon_for,
    vocabulary_for,
    vocabulary_for_code,
)

FIXTURE = Path(__file__).parent / "fixtures" / "policy_vectors" / "nudge_vocabulary.json"
DOC = json.loads(FIXTURE.read_text())
CASES = {c["name"]: c for c in DOC["cases"]}


def test_schema_version_and_constants():
    assert DOC["_schema"]["version"] == 1
    assert DOC["constants"]["min_gap_ms"] == MIN_GAP_MS
    assert DOC["constants"]["positive_cap_s"] == POSITIVE_CAP_S
    assert DOC["constants"]["min_on_ms"] == MIN_ON_MS
    assert DOC["constants"]["min_amplitude"] == MIN_AMPLITUDE
    assert tuple(DOC["_schema"]["haptic_encoding"]["haptic_gap_exceptions"]) == HAPTIC_GAP_EXCEPTIONS


def test_module_is_byte_for_byte_the_fixture():
    """Every field of every entry, in order — the whole point of the file."""
    assert len(DOC["vocabulary"]) == len(NUDGE_VOCABULARY)
    for spec, entry in zip(DOC["vocabulary"], NUDGE_VOCABULARY, strict=True):
        assert entry.code == spec["code"]
        assert entry.vector == spec["vector"]
        assert entry.icon == spec["icon"]
        assert entry.color == spec["color"]
        assert entry.name == spec["name"]
        assert entry.meaning == spec["meaning"]
        assert entry.polarity == spec["polarity"]
        assert list(entry.sources) == spec["sources"]
        assert entry.flash_text == spec["flash_text"]
        assert entry.summary_only == spec["summary_only"]
        assert entry.watch_only == spec["watch_only"]
        assert list(entry.levels) == spec["levels"]
        if spec["haptic"] is None:
            assert entry.haptic is None
            continue
        assert entry.haptic is not None
        assert sorted(entry.haptic) == sorted(int(k) for k in spec["haptic"])
        for level_key, wave in spec["haptic"].items():
            got = entry.haptic[int(level_key)]
            assert list(got.timings_ms) == wave["timings_ms"], (entry.code, level_key)
            assert list(got.amplitudes) == wave["amplitudes"], (entry.code, level_key)


def test_every_code_is_unique_and_stable():
    expected = CASES["every_code_is_unique_and_stable"]["expected_codes"]
    assert [e.code for e in NUDGE_VOCABULARY] == expected
    assert len(set(expected)) == len(expected)
    assert len({e.vector for e in NUDGE_VOCABULARY}) == len(expected)


def test_alert_codes_have_three_levels_positives_have_one():
    case = CASES["alert_codes_have_three_levels_positives_have_one"]
    alerts = [e.code for e in NUDGE_VOCABULARY if e.polarity == "alert"]
    positives = [e.code for e in NUDGE_VOCABULARY if e.polarity == "positive"]
    assert alerts == case["expected_alert_codes"]
    assert positives == case["expected_positive_codes"]
    for e in NUDGE_VOCABULARY:
        assert e.levels == ((1, 2, 3) if e.polarity == "alert" else (1,)), e.code


@pytest.mark.parametrize("entry", [e for e in NUDGE_VOCABULARY if e.haptic], ids=lambda e: e.code)
def test_waveforms_are_well_formed(entry):
    for level, wave in entry.haptic.items():
        t, a = wave.timings_ms, wave.amplitudes
        assert len(t) == len(a) and len(t) % 2 == 0 and len(t) >= 2, (entry.code, level)
        assert t[0] == 0, "the pattern must open with a zero delay (RN's Vibration shape)"
        for i, (ms, amp) in enumerate(zip(t, a, strict=True)):
            if i % 2 == 0:  # OFF slot
                assert amp == 0, (entry.code, level, i)
                if i > 0:
                    floor = 0 if entry.vector in HAPTIC_GAP_EXCEPTIONS else MIN_GAP_MS
                    assert ms >= floor, f"{entry.code} L{level} gap {ms}ms merges two taps"
            else:  # ON slot
                assert ms > 0 and 1 <= amp <= 255, (entry.code, level, i)
                # The measured perceptibility floor. A cue nobody can feel is
                # not a soft cue, it is a missing one — so this binds the soft
                # positives too, not just the alerts.
                assert ms >= MIN_ON_MS, f"{entry.code} L{level} tap {ms}ms is under the floor"
                assert amp >= MIN_AMPLITUDE, f"{entry.code} L{level} amplitude {amp} is under the floor"


def test_level_is_carried_by_rhythm():
    """Total ON time strictly increases with the level, so the cue is
    discriminable with amplitude switched off — which is what a phone feels."""
    expected = CASES["level_is_carried_by_rhythm"]["expected_on_ms"]
    for code, on_ms in expected.items():
        got = [haptic_for(code, lvl).on_ms for lvl in (1, 2, 3)]
        assert got == on_ms, code
        assert got[0] < got[1] < got[2], code


def test_positive_cap_is_one_per_two_minutes():
    case = CASES["positive_cap_is_one_per_two_minutes"]
    gate = PositiveNudgeGate()
    admitted = [o for o in case["offers"] if gate.admit(o["t"])]
    assert admitted == case["expected_admitted"]


def test_lookup_helpers():
    assert icon_for("yelling") == "📈"  # detector name -> its family's icon
    assert icon_for("heated") == "📈"  # vocabulary id
    assert icon_for("nonesuch") is None  # never a fallback emoji
    assert vocabulary_for("aggressive_tone").code == "H"
    assert vocabulary_for_code("Z") is None
    # Worst-first tie-breaking on the owner's order.
    assert code_for_vectors(["airtime", "yelling"]) == "H"
    assert code_for_vectors(["airtime", "interrupting"]) == "C"
    assert code_for_vectors(["nonesuch"]) is None


def test_positives_are_unleveled_and_K_never_buzzes():
    for code in ("D", "E", "R"):
        one = haptic_for(code, 1)
        assert one is not None
        assert haptic_for(code, 3) == one, "a 'level 2 well done' is not a thing"
    assert haptic_for("K", 1) is None
    assert vocabulary_for_code("K").summary_only is True


def test_alert_levels_out_of_range_are_silent_not_clamped():
    """A bad level must produce no cue rather than a wrong one — silence is the
    safe failure for a haptic (the watch's fail direction)."""
    assert haptic_for("H", 0) is None
    assert haptic_for("H", 4) is None
