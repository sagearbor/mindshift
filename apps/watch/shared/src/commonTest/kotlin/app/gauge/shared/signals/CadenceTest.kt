package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertEquals

class CadenceTest {
    private fun burst(on: Boolean, ms: Int) = ShortArray(16 * ms) {
        if (on) (0.2 * 32767 * kotlin.math.sin(2 * kotlin.math.PI * 150 * it / 16000.0)).toInt().toShort() else 0
    }

    @Test fun silenceIsZero() { assertEquals(0.0, Cadence.burstsPerSecond(ShortArray(16000))) }

    @Test fun fourBurstsInOneSecondIsFour() {
        val w = burst(true, 100) + burst(false, 150) + burst(true, 100) + burst(false, 150) +
            burst(true, 100) + burst(false, 150) + burst(true, 100) + burst(false, 150)
        assertEquals(4.0, Cadence.burstsPerSecond(w), 0.51)
    }

    @Test fun continuousToneIsOneBurst() { assertEquals(1.0, Cadence.burstsPerSecond(burst(true, 1000)), 0.01) }

    private fun toneAt(dbfs: Double, ms: Int): ShortArray {
        // Sine RMS = amp/sqrt(2), so amp = 10^(dbfs/20) * sqrt(2) puts the chunk's rmsDbfs at
        // ~dbfs (within quantization). 1kHz so every 25ms chunk holds whole cycles.
        val amp = kotlin.math.sqrt(2.0) * kotlin.math.exp(dbfs / 20.0 * kotlin.math.ln(10.0))
        return ShortArray(16 * ms) {
            (amp * 32767 * kotlin.math.sin(2 * kotlin.math.PI * 1000 * it / 16000.0)).toInt().toShort()
        }
    }

    private fun envelopeAt(peakDbfs: Double): ShortArray =
        toneAt(peakDbfs, 100) + ShortArray(16 * 150) + toneAt(peakDbfs, 100) + ShortArray(16 * 150) +
            toneAt(peakDbfs, 100) + ShortArray(16 * 150) + toneAt(peakDbfs, 100) + ShortArray(16 * 150)

    // --- v0.2.4: loudness invariance ------------------------------------------------------------

    @Test
    fun identicalEnvelopeAtAnyOverallLevelYieldsIdenticalBursts() {
        // THE defect: "speaking rate still seems to be volume". The same 4-burst envelope rendered
        // at -30 / -20 / -10 dBFS peaks must count identically — the voiced floor now rides the
        // window's own level (p90 - 12dB) instead of sitting at a fixed absolute -45.
        val quiet = Cadence.burstsPerSecond(envelopeAt(-30.0))
        val normal = Cadence.burstsPerSecond(envelopeAt(-20.0))
        val loud = Cadence.burstsPerSecond(envelopeAt(-10.0))
        assertEquals(quiet, normal, 0.01)
        assertEquals(normal, loud, 0.01)
        assertEquals(4.0, loud, 0.51)
    }

    @Test
    fun floorDitherDoesNotMintBursts() {
        // Chunks alternating +-2dB around the absolute floor (-43 / -47 dBFS, 25ms each). The old
        // code counted a burst per -43 chunk (a "speaking rate" made of noise-floor flicker); with
        // >=2-unvoiced-chunk hysteresis the single-chunk dips never end the burst — one burst total.
        var w = ShortArray(0)
        repeat(20) { w = w + toneAt(-43.0, 25) + toneAt(-47.0, 25) }
        assertEquals(1.0, Cadence.burstsPerSecond(w), 0.01)
    }

    @Test
    fun genuineSyllableGapsAreStillCounted() {
        // 75ms gaps (3 chunks >= the 2-chunk hysteresis) between 100ms bursts: all four count.
        val w = toneAt(-20.0, 100) + ShortArray(16 * 75) + toneAt(-20.0, 100) + ShortArray(16 * 75) +
            toneAt(-20.0, 100) + ShortArray(16 * 75) + toneAt(-20.0, 100) + ShortArray(16 * 75)
        assertEquals(4.0, Cadence.burstsPerSecond(w) * (w.size / 16000.0), 0.51) // 4 bursts total
    }

    @Test
    fun singleChunkDipsInsideAWordDoNotSplitIt() {
        // One 25ms dip inside otherwise-continuous voicing (< the 50ms hysteresis): still 1 burst.
        val w = toneAt(-20.0, 300) + ShortArray(16 * 25) + toneAt(-20.0, 300)
        assertEquals(1.0, Cadence.burstsPerSecond(w) * (w.size / 16000.0), 0.01)
    }

    @Test
    fun quietSpeechIsNotSilencedByTheNormalizedFloor() {
        // At -40dBFS peaks, p90-12 = -52 < the absolute -45 floor, so the absolute floor still
        // governs — quiet-but-voiced speech keeps counting exactly as before this change.
        assertEquals(4.0, Cadence.burstsPerSecond(envelopeAt(-40.0)), 0.51)
    }

    @Test
    fun pureSilenceIsStillZero() {
        assertEquals(0.0, Cadence.burstsPerSecond(ShortArray(16000)))
    }
}
