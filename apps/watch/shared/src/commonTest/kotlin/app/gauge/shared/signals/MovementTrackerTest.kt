package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class MovementTrackerTest {
    @Test fun noThresholdUntilFiveReadings() {
        val t = MovementTracker()
        repeat(4) { assertNull(t.observe(0.5).threshold) }
        assertNotNull(t.observe(0.5).threshold)
    }

    @Test fun spikeOverThresholdIsOver() {
        val t = MovementTracker(); repeat(10) { t.observe(0.5) }
        assertFalse(t.observe(0.5).over)
        assertTrue(t.observe(3.0).over)
    }

    @Test fun neverOverWhileBaselineNull() {
        val t = MovementTracker(); assertFalse(t.observe(10.0).over)
    }

    @Test fun baselineDoesNotChaseSustainedSpike() {
        val t = MovementTracker(); repeat(10) { t.observe(0.5) }
        var last = false
        repeat(20) { last = t.observe(3.0).over }
        assertTrue(last)
    }
}
