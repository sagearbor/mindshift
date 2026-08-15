package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private const val FIVE_MIN = 300_000L

class MicDutyCycleTest {

    @Test
    fun capturesContinuouslyBeforeAnyQuietRunExists() {
        val d = MicDutyCycle()
        assertTrue(d.shouldCapture(0L))
        assertTrue(d.shouldCapture(10 * FIVE_MIN))
        assertEquals(CaptureMode.CONTINUOUS, d.mode(10 * FIVE_MIN))
    }

    @Test
    fun capturesContinuouslyForTheFirstFiveMinutesOfSilence() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertTrue(d.shouldCapture(1L))
        assertTrue(d.shouldCapture(FIVE_MIN - 1))
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN - 1))
    }

    @Test
    fun dutyCycleEngagesExactlyAtTheFiveMinuteBoundary() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN - 1))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN))
        assertTrue(d.shouldCapture(FIVE_MIN)) // phase 0 is inside the ON window
    }

    @Test
    fun capturesTheFirstTwoSecondsOfEachTenSecondWindow() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        // Anchor = 300_000. ON: [0,2000). OFF: [2000,10000).
        assertTrue(d.shouldCapture(FIVE_MIN + 0))
        assertTrue(d.shouldCapture(FIVE_MIN + 1_999))
        assertFalse(d.shouldCapture(FIVE_MIN + 2_000))
        assertFalse(d.shouldCapture(FIVE_MIN + 9_999))
        assertTrue(d.shouldCapture(FIVE_MIN + 10_000))
        assertTrue(d.shouldCapture(FIVE_MIN + 11_999))
        assertFalse(d.shouldCapture(FIVE_MIN + 12_000))
        assertTrue(d.shouldCapture(FIVE_MIN + 20_000))
    }

    @Test
    fun msUntilNextCaptureIsZeroWhileCapturingAndCountsDownWhileOff() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(0L, d.msUntilNextCapture(FIVE_MIN + 500))
        assertEquals(8_000L, d.msUntilNextCapture(FIVE_MIN + 2_000))
        assertEquals(1_000L, d.msUntilNextCapture(FIVE_MIN + 9_000))
        assertEquals(0L, d.msUntilNextCapture(FIVE_MIN + 10_000))
    }

    @Test
    fun msUntilNextCaptureIsZeroWhileContinuous() {
        val d = MicDutyCycle()
        assertEquals(0L, d.msUntilNextCapture(0L))
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(0L, d.msUntilNextCapture(FIVE_MIN - 1))
    }

    @Test
    fun aVoicedWindowSnapsBackToContinuousInstantly() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN + 1_000))
        // A voiced window observed during an ON phase.
        d.onObservation(voiced = true, nowMs = FIVE_MIN + 1_500)
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN + 1_501))
        assertTrue(d.shouldCapture(FIVE_MIN + 1_501))
        // ...and stays continuous through what would have been an OFF phase.
        assertTrue(d.shouldCapture(FIVE_MIN + 5_000))
        assertTrue(d.shouldCapture(FIVE_MIN + 9_999))
    }

    @Test
    fun theFiveMinuteClockRestartsAfterASnapBack() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        d.onObservation(voiced = true, nowMs = FIVE_MIN + 1_500)
        d.onObservation(voiced = false, nowMs = FIVE_MIN + 2_500)
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN + 2_500 + FIVE_MIN - 1))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN + 2_500 + FIVE_MIN))
    }

    @Test
    fun repeatedQuietWindowsDoNotRestartTheQuietRun() {
        val d = MicDutyCycle()
        for (t in 0L until 250_000L step 1_000L) d.onObservation(voiced = false, nowMs = t)
        // The run still started at t=0, so the boundary is still 300_000 — not 300_000 after the
        // LAST quiet window.
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN))
    }

    @Test
    fun scheduleIsDeterministicNotRandom() {
        // Two instances fed identical inputs must produce byte-identical capture schedules —
        // the ratified requirement is a fixed first-2s-of-each-10s window, not sampling.
        val a = MicDutyCycle()
        val b = MicDutyCycle()
        a.onObservation(voiced = false, nowMs = 0L)
        b.onObservation(voiced = false, nowMs = 0L)
        val sequenceA = (0L until 30_000L step 250L).map { a.shouldCapture(FIVE_MIN + it) }
        val sequenceB = (0L until 30_000L step 250L).map { b.shouldCapture(FIVE_MIN + it) }
        assertEquals(sequenceA, sequenceB)
        // And it really does duty-cycle rather than degenerating to always-on/always-off.
        assertTrue(sequenceA.contains(true) && sequenceA.contains(false))
    }

    @Test
    fun backwardsClockFailsOpenToContinuousCapture() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 1_000_000L)
        assertTrue(d.shouldCapture(500_000L))
        assertEquals(CaptureMode.CONTINUOUS, d.mode(500_000L))
        assertEquals(0L, d.msUntilNextCapture(500_000L))
    }

    @Test
    fun resetReturnsToContinuousCapture() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN))
        d.reset()
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN))
        assertTrue(d.shouldCapture(FIVE_MIN))
    }

    @Test
    fun customWindowsAreHonoured() {
        val d = MicDutyCycle(quietThresholdMs = 1_000L, cycleMs = 100L, onMs = 20L)
        d.onObservation(voiced = false, nowMs = 0L)
        assertTrue(d.shouldCapture(1_000L))
        assertTrue(d.shouldCapture(1_019L))
        assertFalse(d.shouldCapture(1_020L))
        assertTrue(d.shouldCapture(1_100L))
        assertEquals(80L, d.msUntilNextCapture(1_020L))
    }
}
