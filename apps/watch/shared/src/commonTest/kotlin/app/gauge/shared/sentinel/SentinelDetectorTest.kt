package app.gauge.shared.sentinel

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SentinelDetectorTest {
    private fun tone(a: Double, n: Int = 16000) =
        ShortArray(n) { (kotlin.math.sin(2 * kotlin.math.PI * 150 * it / 16000.0) * a * 32767).toInt().toShort() }

    @Test fun silenceIsUnvoicedAndNeverTriggers() {
        val d = SentinelDetector()
        repeat(10) { assertFalse(d.observe(ShortArray(16000)).let { o -> o.voiced || o.triggered }) }
    }
    @Test fun steadySpeechEstablishesBaselineWithoutTrigger() {
        val d = SentinelDetector()
        repeat(10) { assertFalse(d.observe(tone(0.05)).triggered) }   // ~ -29 dBFS steady
    }
    @Test fun sustainedLoudnessOverBaselineTriggersOnSecondWindow() {
        val d = SentinelDetector()
        repeat(6) { d.observe(tone(0.05)) }                            // baseline ~ -29 dBFS
        assertFalse(d.observe(tone(0.2)).triggered)                    // +12 dB, window 1: not yet
        assertTrue(d.observe(tone(0.2)).triggered)                     // window 2: triggered
    }
    @Test fun singleSpikeDoesNotTrigger() {
        val d = SentinelDetector()
        repeat(6) { d.observe(tone(0.05)) }
        d.observe(tone(0.2))
        assertFalse(d.observe(tone(0.05)).triggered)
    }
    @Test fun batterySaverThresholdIsStricter() {
        val d = SentinelDetector(triggerDbOverBaseline = 10.0)
        repeat(6) { d.observe(tone(0.05)) }
        d.observe(tone(0.1)); // +6 dB — below the 10 dB bar
        assertFalse(d.observe(tone(0.1)).triggered)
    }

    @Test fun baselineDoesNotChaseYelling() {
        val d = SentinelDetector()
        repeat(6) { d.observe(tone(0.05)) }                            // baseline ~ -29 dBFS
        assertFalse(d.observe(tone(0.2)).triggered)                    // +12 dB, window 1: not yet
        repeat(10) {
            // Only trigger/loud windows feed in here — baseline must NOT chase the yelling,
            // so every window from the second one onward stays triggered.
            assertTrue(d.observe(tone(0.2)).triggered)
        }
    }

    @Test fun seedBaselineUsesMedianNotFirstWindow() {
        val d = SentinelDetector()
        // Non-identical seed amplitudes: first window is the quietest
        // (~ -33.5 dBFS); the other two are ~ -29.0 dBFS. The true median
        // of the 3 seed windows is ~ -29.0 dBFS — a baseline frozen at the
        // first (quietest) window would instead sit at ~ -33.5 dBFS.
        d.observe(tone(0.03))
        d.observe(tone(0.05))
        d.observe(tone(0.05))
        // ~ -25.0 dBFS: +4 dB over the true median (~-29) — below the 6dB
        // trigger bar, must never trigger. A stale first-window baseline
        // (~-33.5) would see this as +8.5 dB — over the bar — and wrongly
        // trigger on the second window.
        assertFalse(d.observe(tone(0.08)).triggered)
        assertFalse(d.observe(tone(0.08)).triggered)
    }

    @Test fun baselineExposedAfterSeeding() {
        val d = SentinelDetector()
        assertNull(d.baseline) // no voiced window observed yet
        d.observe(tone(0.05))
        assertNotNull(d.baseline) // pinned to the first voiced window immediately (see class KDoc)
    }
}
