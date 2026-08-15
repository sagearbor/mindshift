package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SpeakingRateTrackerTest {
    @Test fun noThresholdUntilFiveReadings() {
        val t = SpeakingRateTracker()
        repeat(4) { assertNull(t.observe(2.0).threshold) }
        assertNotNull(t.observe(2.0).threshold)
    }

    @Test fun spikeOverBaselinePlus1_5IsOver() {
        val t = SpeakingRateTracker(); repeat(10) { t.observe(2.0) }
        assertFalse(t.observe(3.0).over)      // bar 3.5 — under
        assertTrue(t.observe(4.0).over)       // over
    }

    @Test fun zerosDoNotFeedBaseline() {
        val t = SpeakingRateTracker()
        repeat(4) { t.observe(2.0) }
        repeat(20) { t.observe(0.0) }
        assertNull(t.observe(0.0).threshold)
        assertNotNull(t.observe(2.0).threshold)
    }

    @Test fun neverOverWhileBaselineNull() {
        val t = SpeakingRateTracker(); assertFalse(t.observe(50.0).over)
    }

    @Test fun baselineDoesNotChaseSustainedSpike() {
        val t = SpeakingRateTracker(); repeat(10) { t.observe(2.0) }
        var last = false
        repeat(20) { last = t.observe(4.0).over }
        assertTrue(last)
    }
}
