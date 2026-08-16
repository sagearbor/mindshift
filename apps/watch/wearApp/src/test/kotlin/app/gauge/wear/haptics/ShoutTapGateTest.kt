package app.gauge.wear.haptics

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ShoutTapGateTest {

    @Test
    fun firstLoudWindowTaps() {
        val g = ShoutTapGate()
        assertTrue(g.onLoudWindow(0L))
    }

    @Test
    fun tapsInsideTheMinIntervalAreSuppressed() {
        val g = ShoutTapGate(minIntervalMs = 2_000L)
        assertTrue(g.onLoudWindow(0L))
        assertFalse(g.onLoudWindow(1L))
        assertFalse(g.onLoudWindow(1_999L))
    }

    @Test
    fun tapFiresAgainExactlyAtTheMinInterval() {
        val g = ShoutTapGate(minIntervalMs = 2_000L)
        assertTrue(g.onLoudWindow(0L))
        assertTrue(g.onLoudWindow(2_000L))
    }

    @Test
    fun suppressedCallsDoNotRestartTheClock() {
        val g = ShoutTapGate(minIntervalMs = 2_000L)
        assertTrue(g.onLoudWindow(0L))
        assertFalse(g.onLoudWindow(1_000L)) // must NOT push the next allowed tap to 3_000
        assertTrue(g.onLoudWindow(2_000L))
    }

    @Test
    fun resetAllowsAnImmediateTap() {
        val g = ShoutTapGate(minIntervalMs = 2_000L)
        assertTrue(g.onLoudWindow(0L))
        g.reset()
        assertTrue(g.onLoudWindow(1L))
    }

    @Test
    fun backwardsClockFailsTowardSilence() {
        // Opposite fail direction from MicDutyCycle, deliberately: a skipped nicety tap costs
        // nothing; a haptic firing storm during a clock anomaly costs trust.
        val g = ShoutTapGate(minIntervalMs = 2_000L)
        assertTrue(g.onLoudWindow(100_000L))
        assertFalse(g.onLoudWindow(0L))
        assertFalse(g.onLoudWindow(99_999L))
        assertTrue(g.onLoudWindow(102_000L))
    }

    @Test
    fun customIntervalIsHonoured() {
        val g = ShoutTapGate(minIntervalMs = 500L)
        assertTrue(g.onLoudWindow(0L))
        assertFalse(g.onLoudWindow(499L))
        assertTrue(g.onLoudWindow(500L))
    }
}
