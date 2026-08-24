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

from typing import Sequence

from watch.models import Channel, NudgeEvent, VectorEvent, VectorSubscription

# The shipped watch's two lanes, in emission order. Emission order matters
# for byte-for-byte compatibility: ``on_events`` returns at most one nudge per
# channel, in THIS order, and server/tests/watch/test_nudge_policy.py's
# assertions (and the watch's WS consumer) rely on A before B.
DEFAULT_CHANNELS: tuple[Channel, ...] = ("A", "B")


class NudgePolicy:
    """Transforms vector events into per-channel haptic nudges with hysteresis.

    Channel level = max over subscribed+haptics-on vectors of round(level × sensitivity) clamped 0–3.
    Emits NudgeEvent only when a channel's level changes.
    De-escalates (drops one level) after cooldown_s seconds with no event ≥ current level.
    Sustained events at current level refresh the decay clock without emitting.
    """

    def __init__(
        self,
        subs: list[VectorSubscription],
        cooldown_s: float = 20.0,
        channels: Sequence[Channel] = DEFAULT_CHANNELS,
    ):
        if not channels:
            raise ValueError("NudgePolicy needs at least one channel")
        self.subs = subs
        self.cooldown_s = cooldown_s
        # Ordered, de-duplicated: emission order is the caller's order, and a
        # repeated label would otherwise double-emit for that lane.
        self.channels: tuple[Channel, ...] = tuple(dict.fromkeys(channels))
        # Current level for each channel
        self._levels: dict[Channel, int] = {c: 0 for c in self.channels}
        # Last time we observed a qualifying event (≥ current level) for each channel
        self._last_qualifying_t: dict[Channel, float] = {c: 0.0 for c in self.channels}

    def on_events(self, events: list[VectorEvent], t: float) -> list[NudgeEvent]:
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

        Returns:
            List of NudgeEvent objects, at most one per channel per call
        """
        nudges = []

        # Create a mapping of vector -> subscription for haptics-on subscriptions
        sub_by_vector: dict[str, VectorSubscription] = {}
        for sub in self.subs:
            if sub.haptics:  # Only consider haptics-on subscriptions
                sub_by_vector[sub.vector] = sub

        # Compute max scaled level from THIS call's events, grouped by channel
        event_max: dict[Channel, tuple[int, list[str]]] = {c: (0, []) for c in self.channels}
        for event in events:
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
