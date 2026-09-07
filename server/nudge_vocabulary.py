"""The nudge VOCABULARY — what a firing behaviour is CALLED, how it LOOKS and how it FEELS.

Owner-approved 2026-09-06. The executable contract is
``server/tests/fixtures/policy_vectors/nudge_vocabulary.json`` (driven by
``server/tests/test_nudge_vocabulary_vectors.py``); the mirrors are
``apps/mobile/src/live/nudgeVocabulary.ts`` and
``apps/watch/shared/src/commonMain/kotlin/app/gauge/shared/NudgeVocabulary.kt``.

Deliberately separate from :mod:`server.nudge_policy`: the policy decides WHEN
something fires and at what level, this decides what the user is shown and
what the wrist plays. A threshold change must never silently restyle a cue,
and a restyle must never move a threshold.

Haptics are encoded once, in a form both runtimes render verbatim:
``timings_ms`` alternates OFF, ON, OFF, ON, … starting with an initial delay
(always 0) — React Native's Android ``Vibration.vibrate(pattern)`` shape — and
``amplitudes`` (0 in the OFF slots, 1..255 in the ON slots) is what Wear OS's
``VibrationEffect.createWaveform`` uses. RN cannot vary amplitude, so the phone
feels the RHYTHM only; that is why every LEVEL difference is a rhythm
difference and never an intensity one (Brown & Brewster: rhythm identified
~93% of the time, intensity ~61%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Polarity = Literal["alert", "positive"]
Color = Literal["red", "green", "neutral"]

#: Pattern-wide never-merge silence floor (the watch's ``HapticPatterns.MIN_GAP_MS``).
MIN_GAP_MS = 170

#: The softest cue a wrist can actually feel: 50 ms at amplitude 180. Not a
#: style choice — it is this app's own measured device finding (the watch's
#: v0.1–v0.2.3 raw waveforms at 40 ms / amplitude 120–180 proved barely
#: perceptible on a real Pixel Watch). Every ON slot in the vocabulary is at or
#: above it, positives included: a cue nobody can feel is not a soft cue, it is
#: a missing one.
MIN_ON_MS = 50
MIN_AMPLITUDE = 180

#: At most one positive haptic per this many seconds, across D/E/R together —
#: so praise can never become its own nag.
POSITIVE_CAP_S = 120.0

#: The one vector whose intra-cue gap is deliberately under :data:`MIN_GAP_MS`:
#: a lub-dub only reads as a heartbeat when the two beats nearly merge.
HAPTIC_GAP_EXCEPTIONS: tuple[str, ...] = ("pulse",)


@dataclass(frozen=True)
class HapticWaveform:
    """One playable cue. See the module docstring for the encoding."""

    timings_ms: tuple[int, ...]
    amplitudes: tuple[int, ...]

    @property
    def on_ms(self) -> int:
        """Total vibrating time — what an amplitude-blind phone actually feels."""
        return sum(self.timings_ms[1::2])


@dataclass(frozen=True)
class NudgeVocabularyEntry:
    code: str
    vector: str
    icon: str
    color: Color
    name: str
    meaning: str
    polarity: Polarity
    #: Detector vector names (``server/watch/vectors.py``) that raise this code.
    #: Empty for the positive codes, whose detectors live in
    #: :mod:`server.positive_nudges`.
    sources: tuple[str, ...]
    flash_text: str | None
    #: Never fires live — a badge on the post-session summary (K).
    summary_only: bool
    #: The phone has no sensor for it (P: heart rate).
    watch_only: bool
    levels: tuple[int, ...]
    #: Level -> waveform, or ``None`` for a code that never buzzes (K).
    haptic: dict[int, HapticWaveform] | None = field(default=None)


def _w(timings: list[int], amps: list[int]) -> HapticWaveform:
    return HapticWaveform(tuple(timings), tuple(amps))


#: Owner order (2026-09-06) — also worst-first, which is how ties break in
#: :func:`code_for_vectors`.
NUDGE_VOCABULARY: tuple[NudgeVocabularyEntry, ...] = (
    NudgeVocabularyEntry(
        code="H",
        vector="heated",
        icon="📈",
        color="red",
        name="Heated",
        meaning=(
            "You got loud, or your words got hot. Yelling and aggressive tone are one "
            "family to the user (the detectors stay separate underneath)."
        ),
        polarity="alert",
        sources=("yelling", "aggressive_tone", "activation"),
        flash_text="Take it down a notch",
        summary_only=False,
        watch_only=False,
        levels=(1, 2, 3),
        # A RISING ramp: the cue builds, and the LEVEL is how many taps it takes
        # to get there. This IS the shipped watch channel-A ladder.
        haptic={
            1: _w([0, 200], [0, 255]),
            2: _w([0, 70, 170, 150], [0, 210, 0, 255]),
            3: _w([0, 60, 170, 110, 170, 200], [0, 200, 0, 230, 0, 255]),
        },
    ),
    NudgeVocabularyEntry(
        code="D",
        vector="de_escalated",
        icon="📉",
        color="green",
        name="De-escalated",
        meaning="Heat or loudness dropped a level within two turns of a spike — you pulled it back.",
        polarity="positive",
        sources=(),
        flash_text="Nice recovery",
        summary_only=False,
        watch_only=False,
        levels=(1,),
        # A FALLING ramp — the mirror of H, and the only cue that fades.
        haptic={
            1: _w([0, 200, 170, 110, 170, 60], [0, 230, 0, 200, 0, 180]),
        },
    ),
    NudgeVocabularyEntry(
        code="C",
        vector="cut_in",
        icon="✂️",
        color="red",
        name="Cut in",
        meaning=(
            "You started talking over them and kept going — sustained overlap, not the "
            "brief overlap of ordinary engagement."
        ),
        polarity="alert",
        sources=("interrupting",),
        flash_text="Let them finish",
        summary_only=False,
        watch_only=False,
        levels=(1, 2, 3),
        # `• —`, `• • —`, `• • • —`: short tap(s) then one long "stop" buzz. The
        # short taps are 75 ms, not 60 — a "short" tap under the perceptibility
        # floor turns `• —` into a plain buzz.
        haptic={
            1: _w([0, 60, 170, 300], [0, 255, 0, 200]),
            2: _w([0, 60, 170, 60, 170, 300], [0, 255, 0, 255, 0, 200]),
            3: _w([0, 60, 170, 60, 170, 60, 170, 300], [0, 255, 0, 255, 0, 255, 0, 200]),
        },
    ),
    NudgeVocabularyEntry(
        code="A",
        vector="hogging",
        icon="🎤",
        color="red",
        name="Hogging",
        meaning="You have been taking most of the airtime over the last two minutes.",
        polarity="alert",
        sources=("airtime",),
        flash_text="Give them the floor",
        summary_only=False,
        watch_only=False,
        levels=(1, 2, 3),
        # Slow `— — —`: long, unhurried buzzes — "you're talking a lot" is not an
        # emergency, so it must not feel like H's crisp taps.
        haptic={
            1: _w([0, 250, 300, 250], [0, 180, 0, 180]),
            2: _w([0, 250, 300, 250, 300, 250], [0, 180, 0, 180, 0, 180]),
            3: _w([0, 250, 300, 250, 300, 250, 300, 250], [0, 180, 0, 180, 0, 180, 0, 180]),
        },
    ),
    NudgeVocabularyEntry(
        code="E",
        vector="listened",
        icon="👂",
        color="green",
        name="Listened",
        meaning="You let them finish a long turn without cutting in.",
        polarity="positive",
        sources=(),
        flash_text=None,
        summary_only=False,
        watch_only=False,
        levels=(1,),
        # Soft `••` — deliberately the same cue as R: the wrist says "that was
        # good", the screen says which good thing.
        haptic={
            1: _w([0, 50, 170, 50], [0, 180, 0, 180]),
        },
    ),
    NudgeVocabularyEntry(
        code="R",
        vector="repair",
        icon="🤝",
        color="green",
        name="Repair",
        meaning="You validated or apologised and their tone softened on the next turn.",
        polarity="positive",
        sources=(),
        flash_text=None,
        summary_only=False,
        watch_only=False,
        levels=(1,),
        haptic={
            1: _w([0, 50, 170, 50, 170, 50], [0, 180, 0, 180, 0, 180]),
        },
    ),
    NudgeVocabularyEntry(
        code="K",
        vector="calm_streak",
        icon="🧘",
        color="green",
        name="Calm streak",
        meaning="N minutes with no escalation at all.",
        polarity="positive",
        sources=(),
        flash_text=None,
        summary_only=True,
        watch_only=False,
        levels=(1,),
        # No cue, ever — buzzing someone to say nothing happened is a nag.
        haptic=None,
    ),
    NudgeVocabularyEntry(
        code="P",
        vector="pulse",
        icon="❤️",
        color="red",
        name="Pulse",
        meaning="Your heart rate jumped well over your resting rate (+15 / +25 / +35 bpm).",
        polarity="alert",
        sources=("hr_spike",),
        flash_text=None,
        summary_only=False,
        watch_only=True,
        levels=(1, 2, 3),
        # Lub-dub: a short beat then a longer one 120 ms apart — under MIN_GAP_MS
        # on purpose, because the near-merge is the heartbeat.
        haptic={
            1: _w([0, 90, 100, 150], [0, 190, 0, 240]),
            2: _w([0, 90, 100, 150, 400, 90, 100, 150], [0, 190, 0, 240, 0, 190, 0, 240]),
            3: _w([0, 90, 100, 150, 400, 90, 100, 150, 400, 90, 100, 150], [0, 190, 0, 240, 0, 190, 0, 240, 0, 190, 0, 240]),
        },
    ),
)

_BY_CODE = {e.code: e for e in NUDGE_VOCABULARY}
_BY_VECTOR = {e.vector: e for e in NUDGE_VOCABULARY}
_BY_SOURCE = {s: e for e in NUDGE_VOCABULARY for s in e.sources}
_RANK = {e.code: i for i, e in enumerate(NUDGE_VOCABULARY)}


def vocabulary_for_code(code: str) -> NudgeVocabularyEntry | None:
    return _BY_CODE.get(code)


def vocabulary_for(vector: str) -> NudgeVocabularyEntry | None:
    """By vocabulary id (``"heated"``) OR by the detector vector that raises it
    (``"yelling"``, ``"aggressive_tone"``, …) — callers hold both kinds of name."""
    return _BY_VECTOR.get(vector) or _BY_SOURCE.get(vector)


def icon_for(vector: str) -> str | None:
    """The icon, or ``None`` when unknown — never a fallback emoji, so a missing
    mapping is visible instead of silently mislabelled."""
    entry = vocabulary_for(vector)
    return entry.icon if entry else None


def haptic_for(code: str, level: int) -> HapticWaveform | None:
    """The cue for a code at a level, or ``None`` when that code never buzzes (K)
    or the level is out of range. Positives are unleveled — a "level 2 well
    done" is not a thing — so any level maps to their single cue."""
    entry = _BY_CODE.get(code)
    if entry is None or entry.haptic is None:
        return None
    if entry.polarity == "positive":
        return entry.haptic.get(1)
    return entry.haptic.get(int(level))


def code_for_vectors(vectors: list[str] | tuple[str, ...]) -> str | None:
    """The strongest code a set of firing detector vectors maps to (a
    ``NudgeEvent.vectors`` list). Ties break on the owner's order, which is
    worst-first."""
    best: str | None = None
    best_rank = len(NUDGE_VOCABULARY)
    for v in vectors:
        entry = vocabulary_for(v)
        if entry is None:
            continue
        rank = _RANK[entry.code]
        if rank < best_rank:
            best, best_rank = entry.code, rank
    return best


class PositiveNudgeGate:
    """At most one positive cue per ``cap_s`` across D/E/R together.

    ``>=`` (not ``>``) so a caller ticking at exactly the cadence fires on the
    tick — the same direction as the watch's ``NudgeHapticSchedule.reminderDue``.
    """

    def __init__(self, cap_s: float = POSITIVE_CAP_S) -> None:
        self.cap_s = cap_s
        self._last_t: float | None = None

    def admit(self, t: float) -> bool:
        """True (and the clock resets) when this positive offer may be delivered."""
        if self._last_t is not None and t - self._last_t < self.cap_s:
            return False
        self._last_t = t
        return True

    def wait_s(self, t: float) -> float:
        """Seconds until the next positive may fire; 0 when one may fire now."""
        if self._last_t is None:
            return 0.0
        return max(0.0, self.cap_s - (t - self._last_t))
