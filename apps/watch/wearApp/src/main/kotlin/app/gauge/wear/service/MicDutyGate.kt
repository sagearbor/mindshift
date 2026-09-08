package app.gauge.wear.service

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.MicDutyCycle

/**
 * THE decision [SentinelService]'s tick makes at its duty-cycle consult point: should this tick
 * pause the mic (skip the window and sleep until the schedule's next ON phase) instead of
 * capturing? Extracted as a pure function — same discipline as [notificationText]/
 * `PulseChainDecision` — so the service-level guarantees stay directly unit-testable on the JVM
 * (no emulator in this repo's CI loop, per CLAUDE.md):
 *
 *  - ONLY `ARMED` ever pauses. STREAMING (an episode in flight) and COOLDOWN (which may still
 *    return to one) always capture continuously, regardless of how deep the [dutyCycle]'s own
 *    schedule currently sits — including the DEEP tier's up-to-28s OFF phases.
 *  - COMPANION never touches the mic at all, so it never pauses it either (the tick's separate
 *    companion branch never resumes it in the first place).
 *  - Everything else defers to [MicDutyCycle.shouldCapture], which carries the fail-open
 *    doctrine (ambiguity always captures).
 *
 * [lastPublishedMode]/[lastPublishedSentinel] are the service's own handler-thread bookkeeping
 * (the last published snapshot's mode/state), nullable exactly as those fields are.
 */
fun micPauseDue(
    lastPublishedMode: Mode?,
    lastPublishedSentinel: SentinelState?,
    dutyCycle: MicDutyCycle,
    nowMs: Long,
): Boolean =
    lastPublishedMode != Mode.COMPANION &&
        lastPublishedSentinel == SentinelState.ARMED &&
        !dutyCycle.shouldCapture(nowMs)
