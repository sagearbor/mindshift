package app.gauge.wear.haptics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * P4-3: [PulseEngine] is the pure decision core for the local proportional pulse-train haptic —
 * see its own KDoc for the full contract this exercises (bands, interval gating, the "never
 * merge" clamp, "off", and the below-threshold reset).
 */
class PulseEngineTest {

    private fun engine(intervalMs: Long? = 250L, clock: () -> Long): PulseEngine =
        PulseEngine(intervalMs = { intervalMs }, nowMs = clock)

    // --- band boundaries (dB over the *trigger threshold*, not over baseline) ---------------

    @Test
    fun gentleBandAtZeroOver() {
        val e = engine { 0L }
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0)) // exactly 0 over
    }

    @Test
    fun gentleBandJustBelowThreeOver() {
        val e = engine { 0L }
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 12.9, thresholdDb = 10.0)) // 2.9 over
    }

    @Test
    fun moderateBandAtExactlyThreeOver() {
        val e = engine { 0L }
        assertEquals(Pulse(60L, 205), e.onWindow(dbOver = 13.0, thresholdDb = 10.0)) // exactly 3
    }

    @Test
    fun moderateBandJustBelowSixOver() {
        val e = engine { 0L }
        assertEquals(Pulse(60L, 205), e.onWindow(dbOver = 15.9, thresholdDb = 10.0)) // 5.9 over
    }

    @Test
    fun aggressiveBandAtExactlySixOver() {
        val e = engine { 0L }
        assertEquals(Pulse(70L, 230), e.onWindow(dbOver = 16.0, thresholdDb = 10.0)) // exactly 6
    }

    @Test
    fun aggressiveBandJustBelowNineOver() {
        val e = engine { 0L }
        assertEquals(Pulse(70L, 230), e.onWindow(dbOver = 18.9, thresholdDb = 10.0)) // 8.9 over
    }

    @Test
    fun maxBandAtExactlyNineOver() {
        val e = engine { 0L }
        assertEquals(Pulse(80L, 255), e.onWindow(dbOver = 19.0, thresholdDb = 10.0)) // exactly 9
    }

    @Test
    fun maxBandWellAboveNineOver() {
        val e = engine { 0L }
        assertEquals(Pulse(80L, 255), e.onWindow(dbOver = 40.0, thresholdDb = 10.0))
    }

    // --- below threshold -----------------------------------------------------------------

    @Test
    fun belowThresholdReturnsNull() {
        val e = engine { 0L }
        assertNull(e.onWindow(dbOver = 9.9, thresholdDb = 10.0))
    }

    @Test
    fun belowThresholdResetsIntervalClockSoNextOverThresholdFiresImmediately() {
        var clock = 0L
        val e = engine(intervalMs = 250L) { clock }
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0)) // fires at clock=0

        clock = 50 // still well inside the 250ms interval
        assertNull(e.onWindow(dbOver = 5.0, thresholdDb = 10.0)) // drop below threshold -> resets clock

        clock = 60 // only 10ms after the dip, but the reset means this must fire immediately
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0))
    }

    // --- interval gating -------------------------------------------------------------------

    @Test
    fun secondCallWithinIntervalIsGated() {
        var clock = 0L
        val e = engine(intervalMs = 250L) { clock }
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0))
        clock = 100
        assertNull(e.onWindow(dbOver = 10.0, thresholdDb = 10.0))
    }

    @Test
    fun callAfterIntervalElapsesFiresAgain() {
        var clock = 0L
        val e = engine(intervalMs = 250L) { clock }
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0))
        clock = 250
        assertEquals(Pulse(50L, 180), e.onWindow(dbOver = 10.0, thresholdDb = 10.0))
    }

    // --- "never merge" clamp ---------------------------------------------------------------

    @Test
    fun intervalIsClampedToPulseDurationPlusSilenceFloor() {
        var clock = 0L
        // A max-band pulse is 80ms; the clamp floor is 80 + 170 = 250ms, well above this
        // deliberately-too-short configured 50ms interval.
        val e = engine(intervalMs = 50L) { clock }
        assertEquals(Pulse(80L, 255), e.onWindow(dbOver = 40.0, thresholdDb = 10.0))

        clock = 100 // past the configured 50ms interval but inside the 250ms clamp floor
        assertNull(e.onWindow(dbOver = 40.0, thresholdDb = 10.0))

        clock = 250 // exactly at the clamp floor
        assertEquals(Pulse(80L, 255), e.onWindow(dbOver = 40.0, thresholdDb = 10.0))
    }

    // --- off ---------------------------------------------------------------------------------

    @Test
    fun offIntervalAlwaysReturnsNull() {
        val e = engine(intervalMs = null) { 0L }
        assertNull(e.onWindow(dbOver = 40.0, thresholdDb = 10.0))
        assertNull(e.onWindow(dbOver = 0.0, thresholdDb = 10.0))
    }

    // --- pure band mapping -------------------------------------------------------------------

    @Test
    fun bandForIsExposedForDirectBoundaryTesting() {
        assertEquals(Pulse(50L, 180), PulseEngine.bandFor(0.0))
        assertEquals(Pulse(60L, 205), PulseEngine.bandFor(3.0))
        assertEquals(Pulse(70L, 230), PulseEngine.bandFor(6.0))
        assertEquals(Pulse(80L, 255), PulseEngine.bandFor(9.0))
    }

    // --- effectiveIntervalMs (P4-3 review fix: shared clamp, engine and service must agree) -----

    @Test
    fun effectiveIntervalMsClampsShortConfiguredIntervalToPulseDurationPlusSilenceFloor() {
        assertEquals(250L, PulseEngine.effectiveIntervalMs(Pulse(80L, 255), 50L)) // 80 + 170
        assertEquals(210L, PulseEngine.effectiveIntervalMs(Pulse(40L, 120), 100L)) // 40 + 170
    }

    @Test
    fun effectiveIntervalMsUsesConfiguredIntervalWhenItAlreadyExceedsTheFloor() {
        assertEquals(500L, PulseEngine.effectiveIntervalMs(Pulse(40L, 120), 500L))
        assertEquals(1000L, PulseEngine.effectiveIntervalMs(Pulse(80L, 255), 1000L))
    }
}
