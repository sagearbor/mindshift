package app.gauge.wear.service

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.MicDutyCycle
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private const val TEN_MIN = 600_000L

/**
 * The service-level guarantee at the tick's duty-cycle consult point ([micPauseDue] — the exact
 * predicate SentinelService.tickRunnable runs): duty cycling, including the motion-gated DEEP
 * tier, only ever pauses the mic while ARMED. STREAMING/COOLDOWN capture every window untouched,
 * and COMPANION never pauses (it never resumed the mic to begin with).
 */
class MicDutyGateTest {

    /** A cycle driven deep into the DEEP tier (quiet + still from t=0; deep anchor = 600_000). */
    private fun deepCycle(): MicDutyCycle =
        MicDutyCycle().also { it.onObservation(voiced = false, nowMs = 0L, still = true) }

    @Test
    fun streamingIsUntouchedEvenMidDeepOffPhase() {
        val d = deepCycle()
        val offPhase = TEN_MIN + 15_000L // mid the deep tier's 28s OFF stretch
        assertFalse(d.shouldCapture(offPhase)) // sanity: the schedule itself says OFF right now
        // ...and yet a STREAMING sentinel never pauses:
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.STREAMING, d, offPhase))
    }

    @Test
    fun cooldownAndDisarmedNeverPauseEither() {
        val d = deepCycle()
        val offPhase = TEN_MIN + 15_000L
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.COOLDOWN, d, offPhase))
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.DISARMED, d, offPhase))
        assertFalse(micPauseDue(Mode.STANDARD, null, d, offPhase))
    }

    @Test
    fun companionNeverPausesRegardlessOfSchedule() {
        val d = deepCycle()
        assertFalse(micPauseDue(Mode.COMPANION, SentinelState.ARMED, d, TEN_MIN + 15_000L))
    }

    @Test
    fun armedPausesExactlyWhenTheScheduleSaysOff() {
        val d = deepCycle()
        // Deep ON phase: capture, no pause.
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.ARMED, d, TEN_MIN + 1_000L))
        // Deep OFF phase: pause.
        assertTrue(micPauseDue(Mode.STANDARD, SentinelState.ARMED, d, TEN_MIN + 15_000L))
        // 20% tier (still run broken by movement): its own 10s cycle governs again.
        d.onObservation(voiced = false, nowMs = TEN_MIN + 30_500L, still = false)
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.ARMED, d, TEN_MIN + 30_500L))
        assertTrue(micPauseDue(Mode.STANDARD, SentinelState.ARMED, d, TEN_MIN + 35_000L))
    }

    @Test
    fun armedContinuousNeverPauses() {
        val fresh = MicDutyCycle() // no quiet run at all: continuous, fail open
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.ARMED, fresh, 0L))
        assertFalse(micPauseDue(Mode.STANDARD, SentinelState.ARMED, fresh, 10 * TEN_MIN))
    }
}
