package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HrTrackerTest {
    @Test fun noThresholdUntilFiveReadings() {
        val t = HrTracker()
        repeat(4) { assertNull(t.observe(70.0).threshold) }
        assertNotNull(t.observe(70.0).threshold)
    }

    @Test fun spikeOverBaselinePlus15IsOver() {
        val t = HrTracker(); repeat(10) { t.observe(70.0) }
        assertFalse(t.observe(80.0).over)      // +10 — under
        assertTrue(t.observe(90.0).over)       // +20 — over
    }

    @Test fun neverOverWhileBaselineNull() {
        val t = HrTracker(); assertFalse(t.observe(200.0).over)
    }

    @Test fun baselineDoesNotChaseSustainedSpike() {
        val t = HrTracker(); repeat(10) { t.observe(70.0) }
        var last = false
        repeat(20) { last = t.observe(95.0).over }
        assertTrue(last)
    }
}
