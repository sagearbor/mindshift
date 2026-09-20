# Ported from gauge@2157433 server/nudge_policy.py; adapted per docs/plans/2026-08-15-phase1-one-repo-one-engine.md
# Moved from server/watch/nudge_policy.py (Foundation A, 2026-08-24) so the
# phone's realtime path (server/audio_pipeline.py, later track) can share the
# SAME escalation brain as the watch instead of growing its own. The old
# module path is kept as a thin re-export — every existing import still works.
"""Shared nudge/escalation policy: vector events -> per-channel haptic levels.

Mirror contract: this is the server-side counterpart of the on-watch
NudgeStateMachine at
apps/watch/shared/src/commonMain/kotlin/app/gauge/shared/NudgeStateMachine.kt
— thresholds/semantics (per-channel hysteresis, cooldown, half-up rounding)
must stay identical between the two. The cross-language golden vectors in
server/tests/fixtures/policy_vectors/nudge_policy.json (driven here by
server/tests/test_nudge_policy_vectors.py) are the executable form of that
contract; a Kotlin/TypeScript port should consume the same file.

Channel model: a "channel" is an independent escalation lane with its own
level and its own cooldown clock (the watch uses "A" = voice vectors, "B" =
heart-rate). The policy is parameterized by which channels it runs — the
default ``("A", "B")`` is exactly the shipped watch behaviour; a second
caller that only has one lane (e.g. the phone's realtime coaching path) can
run ``channels=("A",)`` and gets identical single-channel semantics without
a phantom idle "B" in its ``current()`` output. Channel *labels* are still
constrained to ``watch.models.Channel`` because ``NudgeEvent.channel`` is
typed by it — a new lane means a new Literal member there, not an ad-hoc
string here.
"""

from __future__ import annotations

import os
from typing import Sequence

from watch.models import Channel, NudgeEvent, VectorEvent, VectorSubscription

# The shipped watch's two lanes, in emission order. Emission order matters
# for byte-for-byte compatibility: ``on_events`` returns at most one nudge per
# channel, in THIS order, and server/tests/watch/test_nudge_policy.py's
# assertions (and the watch's WS consumer) rely on A before B.
DEFAULT_CHANNELS: tuple[Channel, ...] = ("A", "B")

# --- hold-N hysteresis on the loudness ladder (2026-09-20) -------------------
# The ladder used to escalate off a SINGLE 1 s window that cleared +6 dB over
# the wearer's baseline. Measured over 32 h of real conversation (AMI + SBCSAE,
# scripts/heat_map.py) that instantaneous crossing is most of the dose: the
# median AMI meeting delivered 97 buzzes/hour and the median SBCSAE recording
# 80 — in conversations where nobody is angry. Requiring the first rung to HOLD
# for three consecutive windows before the ladder may climb halves it (46.4 and
# 36.0 respectively) without any new signal, because a single loud window is a
# laugh, a cough or a door, and three in a row is a raised voice.
#
# Reference implementation and the measurement: ``sustained()`` in
# scripts/conversation_audit.py. ``LoudnessHold`` below reproduces it exactly
# when fed one 1 s window per call.
#
# The window the watch/server PCM paths run on: SentinelController's
# MAX_SLICE_BYTES 32000 / 2 bytes / 16 kHz.
WINDOW_S = 1.0
#: Seconds the first rung must hold before ANY upward escalation on the
#: loudness lane. 0 or 1 reproduces the pre-2026-09-20 behaviour exactly (one
#: qualifying window is enough), which is how the old golden vectors are pinned.
HEAT_HOLD_S_DEFAULT = 3.0
#: Server-side override, read once per policy construction.
HEAT_HOLD_ENV = "MINDSHIFT_HEAT_HOLD_S"
#: The vector the hold gates. Only the LOUDNESS lane: ``aggressive_tone`` is
#: about the words, is a better signal, and rides the same channel untouched.
HOLD_VECTOR = "yelling"


def heat_hold_s() -> float:
    """Configured hold, in seconds. ``MINDSHIFT_HEAT_HOLD_S`` overrides
    [HEAT_HOLD_S_DEFAULT]; an unparseable or negative value falls back to the
    default rather than disabling the gate silently."""
    raw = (os.getenv(HEAT_HOLD_ENV) or "").strip()
    if not raw:
        return HEAT_HOLD_S_DEFAULT
    try:
        value = float(raw)
    except ValueError:
        return HEAT_HOLD_S_DEFAULT
    return value if value >= 0.0 else HEAT_HOLD_S_DEFAULT


class LoudnessHold:
    """Has the ladder's first rung held long enough to escalate?

    Exactly ``scripts/conversation_audit.py``'s ``sustained()`` when fed one
    1 s window per call: a run of consecutive qualifying observations, reset by
    the first that does not qualify, and the gate opens once the run reaches
    ``hold_s``.

    Turn-driven callers (the phone's fast loop, ``watch/relay.py``) observe once
    per TURN rather than once per window, so an observation carries how many
    seconds of audio it covers — a 4 s turn that read as loud is four windows of
    hold, not one. Every observation is worth at least one window, which is what
    makes ``hold_s <= 1`` byte-identical to the pre-hold behaviour on every path.
    """

    def __init__(self, hold_s: float | None = None, window_s: float = WINDOW_S):
        self.hold_s = heat_hold_s() if hold_s is None else float(hold_s)
        self.window_s = window_s
        self._run_s = 0.0

    @property
    def run_s(self) -> float:
        """Seconds of consecutive qualifying observation so far."""
        return self._run_s

    def _next_run(self, loud: bool, observed_s: float) -> float:
        if not loud:
            return 0.0
        return self._run_s + max(observed_s, self.window_s)

    def peek(self, loud: bool, observed_s: float = WINDOW_S) -> bool:
        """Would this observation open the gate? Pure — the phone's instant
        haptic tier asks before the same turn's policy tick advances it."""
        return self._next_run(loud, observed_s) >= self.hold_s

    def observe(self, loud: bool, observed_s: float = WINDOW_S) -> bool:
        """Advance the run and report whether this observation may escalate."""
        self._run_s = self._next_run(loud, observed_s)
        return self._run_s >= self.hold_s


class NudgePolicy:
    """Transforms vector events into per-channel haptic nudges with hysteresis.

    Channel level = max over subscribed+haptics-on vectors of round(level × sensitivity) clamped 0–3.
    Emits NudgeEvent only when a channel's level changes.
    De-escalates (drops one level) after cooldown_s seconds with no event ≥ current level.
    Sustained events at current level refresh the decay clock without emitting.

    Hold-N hysteresis (2026-09-20): a ``yelling`` event may only raise a channel
    once the first rung has held for ``hold_s`` seconds — see [LoudnessHold].
    A gated-out loudness observation counts as "not loud" for THIS call, so it
    neither escalates nor refreshes the decay clock, exactly as the measured
    replay models it (``conversation_audit.replay``'s ``gate``).
    """

    def __init__(
        self,
        subs: list[VectorSubscription],
        cooldown_s: float = 20.0,
        channels: Sequence[Channel] = DEFAULT_CHANNELS,
        hold_s: float | None = None,
    ):
        if not channels:
            raise ValueError("NudgePolicy needs at least one channel")
        self.subs = subs
        self.cooldown_s = cooldown_s
        #: The loudness lane's hysteresis. Reads MINDSHIFT_HEAT_HOLD_S once,
        #: here, so a test can set the env and construct a policy to pin either
        #: behaviour.
        self.hold = LoudnessHold(hold_s)
        # Ordered, de-duplicated: emission order is the caller's order, and a
        # repeated label would otherwise double-emit for that lane.
        self.channels: tuple[Channel, ...] = tuple(dict.fromkeys(channels))
        # Current level for each channel
        self._levels: dict[Channel, int] = {c: 0 for c in self.channels}
        # Last time we observed a qualifying event (≥ current level) for each channel
        self._last_qualifying_t: dict[Channel, float] = {c: 0.0 for c in self.channels}

    def on_events(
        self,
        events: list[VectorEvent],
        t: float,
        observed_s: float | None = WINDOW_S,
    ) -> list[NudgeEvent]:
        """Process vector events and return nudge events for level changes or de-escalation.

        Semantics: per channel, E = max scaled level from THIS call's events (0 if none).
        - If E > current: set level=E, emit, refresh qualifying time
        - Elif E == current and current > 0: refresh qualifying time, no emit (sustain clock)
        - Else (E < current): never snap down; if > cooldown_s since last qualifying event: drop exactly ONE level, emit, refresh time

        Events whose subscription routes to a channel this policy doesn't run
        are ignored (a single-channel caller subscribed to "hr_spike" simply
        never nudges on it) — they can't escalate a lane that doesn't exist.

        Args:
            events: List of VectorEvent objects to process
            t: Current timestamp
            observed_s: how many seconds of audio this call's observations
                cover. One 1 s window by default — the cadence the watch and
                the server's own PCM path tick on. A turn-driven caller
                (watch/relay.py) passes the turn's duration so the loudness
                hold counts seconds of speech, not calls.
                ``None`` means "this call carries NO loudness observation" and
                leaves the hold's run untouched. Required for every non-audio
                tick: an ``hr`` frame arriving mid-shout must not break a run
                the wearer's voice is still building, or the wrist would stay
                silent through a raised voice because a heart-rate sample
                landed between two windows.

        Returns:
            List of NudgeEvent objects, at most one per channel per call
        """
        nudges = []

        # Hold-N hysteresis, advanced exactly once per call that heard audio —
        # including quiet ones, since a window under the first rung is what
        # BREAKS a run. The raw, unscaled level is what clears the rung:
        # sensitivity is the wearer's preference about being told, not physics.
        if observed_s is None:
            loudness_may_escalate = True    # no loudness in this call to gate
        else:
            loud = any(e.vector == HOLD_VECTOR and e.level >= 1 for e in events)
            loudness_may_escalate = self.hold.observe(loud, observed_s)

        # Create a mapping of vector -> subscription for haptics-on subscriptions
        sub_by_vector: dict[str, VectorSubscription] = {}
        for sub in self.subs:
            if sub.haptics:  # Only consider haptics-on subscriptions
                sub_by_vector[sub.vector] = sub

        # Compute max scaled level from THIS call's events, grouped by channel
        event_max: dict[Channel, tuple[int, list[str]]] = {c: (0, []) for c in self.channels}
        for event in events:
            if event.vector == HOLD_VECTOR and not loudness_may_escalate:
                # Loud, but the rung has not held long enough yet. The call
                # then reads as E=0 for this lane, so it neither escalates nor
                # refreshes the sustain clock — the decay below runs exactly as
                # it would have on a quiet window.
                continue
            if event.vector in sub_by_vector:
                sub = sub_by_vector[event.vector]
                channel = sub.channel
                if channel not in event_max:
                    continue
                # Scale level by sensitivity using half-up rounding and clamp to 0-3
                scaled_level = self._round_half_up(event.level * sub.sensitivity)
                scaled_level = min(3, max(0, scaled_level))

                current_max, current_vecs = event_max[channel]
                if scaled_level > current_max:
                    event_max[channel] = (scaled_level, [event.vector])
                elif scaled_level == current_max and scaled_level > 0:
                    current_vecs.append(event.vector)

        # Process each channel
        for channel in self.channels:
            E, event_vectors = event_max[channel]
            current = self._levels[channel]

            if E > current:
                # Event(s) with higher level: escalate, emit, refresh clock
                self._levels[channel] = E
                self._last_qualifying_t[channel] = t
                nudges.append(NudgeEvent(
                    channel=channel,
                    level=E,
                    t=t,
                    vectors=sorted(set(event_vectors))
                ))
            elif E == current and current > 0:
                # Sustained event at current level: refresh clock, no emit
                self._last_qualifying_t[channel] = t
            else:
                # E < current or no events: check for de-escalation
                # Only drop if no qualifying event in more than cooldown_s seconds (strict >)
                if current > 0 and t - self._last_qualifying_t[channel] > self.cooldown_s:
                    # Drop exactly one level after cooldown with no high event
                    new_level = current - 1
                    self._levels[channel] = new_level
                    self._last_qualifying_t[channel] = t
                    nudges.append(NudgeEvent(
                        channel=channel,
                        level=new_level,
                        t=t,
                        vectors=[]
                    ))

        return nudges

    def current(self) -> dict[str, int]:
        """Return current level for each channel this policy runs."""
        return {c: self._levels[c] for c in self.channels}

    @staticmethod
    def _round_half_up(x: float) -> int:
        """Round using half-up method (match Kotlin Math.round), not banker's rounding.

        For x >= 0: int(x + 0.5)
        For x < 0: int(x - 0.5)
        This ensures 0.5 rounds up to 1, 1.5 to 2, etc., matching Kotlin Math.round (Task 10 mirror).
        """
        if x >= 0:
            return int(x + 0.5)
        else:
            return int(x - 0.5)
