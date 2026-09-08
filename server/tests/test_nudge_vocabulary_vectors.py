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
        assert len(t) == len(a) and len(t) >= 2, (entry.code, level)
        assert t[0] == 0 and a[0] == 0, "the pattern must open with a zero delay"
        for i, (ms, amp) in enumerate(zip(t, a, strict=True)):
            if i == 0:
                continue
            assert ms > 0, (entry.code, level, i)
            assert 0 <= amp <= 255, (entry.code, level, i)
        # Per RUN — a swell is ONE felt buzz, so the perceptibility floor binds
        # its total length and its PEAK, not each segment inside it.
        for run_ms, peak in wave.runs:
            assert run_ms >= MIN_ON_MS, f"{entry.code} L{level} run {run_ms}ms is under the floor"
            assert peak >= MIN_AMPLITUDE, f"{entry.code} L{level} peak {peak} is under the floor"
        # Silences BETWEEN runs must not let two buzzes smear into one.
        floor = 0 if entry.vector in HAPTIC_GAP_EXCEPTIONS else MIN_GAP_MS
        seen_run = False
        for ms, amp in zip(t[1:], a[1:], strict=True):
            if amp > 0:
                seen_run = True
            elif seen_run:
                assert ms >= floor, f"{entry.code} L{level} gap {ms}ms merges two buzzes"


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


def _taps(w):
    """What a PHONE feels as separate buzzes — a swell counts as one."""
    return [ms for ms, _ in w.runs]


def _gaps(w):
    """Silences between those buzzes."""
    pattern = w.phone_pattern
    return pattern[2::2]


def _confusable(a, b, gap_ms: int, ratio: float) -> bool:
    """Would a PHONE be unable to tell these two cues apart?

    Amplitude is removed on purpose: React Native can only switch an Android
    motor on and off, so strength is not a channel there at all.
    """
    ta, tb = _taps(a), _taps(b)
    if len(ta) != len(tb):
        return False
    if any(abs(x - y) >= gap_ms for x, y in zip(_gaps(a), _gaps(b))):
        return False
    return all(max(x, y) / min(x, y) < ratio for x, y in zip(ta, tb))


def test_no_two_cues_are_confusable():
    """The case the owner found by hand on 2026-09-06.

    Shipped 📈 level 3 was three 100 ms taps and 📉 was three 70 ms taps — they
    differed ONLY in amplitude, which a phone ignores, so both reached the motor
    as "three taps 170 ms apart" and read as the same cue. 👂 and 🤝 were
    byte-for-byte identical. No two cues may be confusable again.
    """
    case = CASES["no_two_cues_are_confusable"]
    gap_ms, ratio = case["confusable_gap_ms"], case["confusable_tap_ratio"]
    cues = [
        (f"{e.code} L{level}", wave)
        for e in NUDGE_VOCABULARY
        if e.haptic
        for level, wave in sorted(e.haptic.items())
    ]
    pairs = [
        [a, b]
        for i, (a, wa) in enumerate(cues)
        for (b, wb) in cues[i + 1:]
        if _confusable(wa, wb, gap_ms, ratio)
    ]
    assert pairs == case["expected_confusable_pairs"]


def test_the_rising_and_falling_ramps_are_opposites_in_tap_length():
    """H rises and D falls in the one dimension every device can play. Encoding
    that in amplitude alone is what produced "heated L3 = de-escalated"."""
    rising = _taps(haptic_for("H", 3))
    falling = _taps(haptic_for("D", 1))
    assert list(rising) == sorted(rising), "H must get longer tap by tap"
    assert list(falling) == sorted(falling, reverse=True), "D must get shorter tap by tap"
    assert list(falling) == list(reversed(rising)), "the two must be exact mirrors"
    # ...and the difference has to be big enough to feel, not a few milliseconds.
    assert max(rising) / min(rising) >= 3.0


def test_positives_are_swells_and_alerts_are_taps():
    """The texture split the owner asked for (2026-09-07, on a Pixel Watch):
    "i would want positive to be very diff than negative". A positive is one
    continuous buzz that rises and falls INSIDE itself; an alert is discrete
    taps at a flat level. On a watch that is felt immediately, before any
    counting; on a phone the swell survives as a smoother, longer buzz."""
    def has_swell(wave):
        # A run built from more than one segment, with the strength changing.
        segments, changed, count = 0, False, 0
        last = None
        for t, amp in zip(wave.timings_ms[1:], wave.amplitudes[1:]):
            if amp > 0:
                count += 1
                if last is not None and amp != last:
                    changed = True
                last = amp
            else:
                segments = max(segments, count)
                count, last = 0, None
        return changed and max(segments, count) > 1

    for entry in NUDGE_VOCABULARY:
        if entry.haptic is None or entry.code == "D":
            continue  # D mirrors H's ramp by design; see its haptic_note.
        for wave in entry.haptic.values():
            assert has_swell(wave) == (entry.polarity == "positive"), entry.code


def test_phone_pattern_merges_a_swell_into_one_buzz():
    """What React Native actually sends. A swell has no internal shape there,
    so it must arrive as ONE buzz of the run's length — not as three taps,
    which is what passing the raw timings would produce."""
    listened = haptic_for("E", 1)
    assert listened.phone_pattern == [0, 180, 170, 180]
    assert [ms for ms, _ in listened.runs] == [180, 180]
    # An alert is unchanged by merging — it has no consecutive vibrating slots.
    assert haptic_for("C", 1).phone_pattern == list(haptic_for("C", 1).timings_ms)
