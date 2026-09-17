package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private const val FIVE_MIN = 300_000L
private const val TEN_MIN = 600_000L

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

    // --- DEEP tier: motion-gated 2s/30s ---------------------------------------------------------

    @Test
    fun deepeningRequiresBothQuietAndStillness() {
        // Quiet + still from t=0: still threshold (10 min) dominates, so deep engages at exactly
        // TEN_MIN — and only DUTY_CYCLED before it (quiet alone engaged the 20% tier at 5 min).
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.CONTINUOUS, d.mode(FIVE_MIN - 1))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(FIVE_MIN))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN - 1))
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
        // Stillness alone (quiet run not yet past ITS threshold) never reaches deep either —
        // before 5 min the mode is still CONTINUOUS despite the accumulating still run.
        val e = MicDutyCycle()
        e.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.CONTINUOUS, e.mode(FIVE_MIN - 1))
    }

    @Test
    fun quietWithoutMotionDataNeverDeepens() {
        // still = null (the default) is "no motion signal" — hours of quiet stay at the 20% tier.
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(10 * TEN_MIN))
        // And explicit movement (still = false) is exactly as non-deepening.
        val e = MicDutyCycle()
        e.onObservation(voiced = false, nowMs = 0L, still = false)
        assertEquals(CaptureMode.DUTY_CYCLED, e.mode(10 * TEN_MIN))
    }

    @Test
    fun aMissingMotionReadingResetsTheStillnessClock() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        // One window with no motion reading mid-run: the stillness clock restarts from the next
        // affirmative still window — fail open, never deepen on ambiguity.
        d.onObservation(voiced = false, nowMs = 60_000L, still = null)
        d.onObservation(voiced = false, nowMs = 120_000L, still = true)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(120_000L + TEN_MIN - 1))
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(120_000L + TEN_MIN))
    }

    @Test
    fun movementAloneSnapsBackToTheTwentyPercentTierAndRestartsTheStillClock() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
        // A quiet-but-moving window: back to the 20% tier (quiet run intact, still anchored at
        // the ORIGINAL quiet anchor — deterministic), and re-stilling takes a full 10 min again.
        d.onObservation(voiced = false, nowMs = TEN_MIN + 1_000, still = false)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN + 1_001))
        d.onObservation(voiced = false, nowMs = TEN_MIN + 2_000, still = true)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN + 2_000 + TEN_MIN - 1))
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN + 2_000 + TEN_MIN))
    }

    @Test
    fun aVoicedWindowSnapsBackFromDeepToContinuousAndRestartsBothClocks() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
        d.onObservation(voiced = true, nowMs = TEN_MIN + 500, still = true)
        assertEquals(CaptureMode.CONTINUOUS, d.mode(TEN_MIN + 501))
        assertTrue(d.shouldCapture(TEN_MIN + 501))
        // Both clocks restarted: quiet re-engages the 20% tier 5 min after the next quiet+still
        // window, and deep only re-engages after a FULL fresh 10 min of stillness.
        val resumeAt = TEN_MIN + 1_500
        d.onObservation(voiced = false, nowMs = resumeAt, still = true)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(resumeAt + FIVE_MIN))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(resumeAt + TEN_MIN - 1))
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(resumeAt + TEN_MIN))
    }

    @Test
    fun deepCapturesTheFirstTwoSecondsOfEachThirtySecondCycle() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        // Deep anchor = max(quiet 300_000, still 600_000) = 600_000. ON: [0,2000). OFF: [2000,30000).
        assertTrue(d.shouldCapture(TEN_MIN + 0))
        assertTrue(d.shouldCapture(TEN_MIN + 1_999))
        assertFalse(d.shouldCapture(TEN_MIN + 2_000))
        assertFalse(d.shouldCapture(TEN_MIN + 29_999))
        assertTrue(d.shouldCapture(TEN_MIN + 30_000))
        assertTrue(d.shouldCapture(TEN_MIN + 31_999))
        assertFalse(d.shouldCapture(TEN_MIN + 32_000))
        assertTrue(d.shouldCapture(TEN_MIN + 60_000))
    }

    @Test
    fun deepMsUntilNextCaptureCountsDownAgainstTheThirtySecondCycle() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(0L, d.msUntilNextCapture(TEN_MIN + 500))
        assertEquals(28_000L, d.msUntilNextCapture(TEN_MIN + 2_000))
        assertEquals(1_000L, d.msUntilNextCapture(TEN_MIN + 29_000))
        assertEquals(0L, d.msUntilNextCapture(TEN_MIN + 30_000))
    }

    @Test
    fun deepScheduleIsDeterministicNotRandom() {
        val a = MicDutyCycle()
        val b = MicDutyCycle()
        a.onObservation(voiced = false, nowMs = 0L, still = true)
        b.onObservation(voiced = false, nowMs = 0L, still = true)
        val sequenceA = (0L until 90_000L step 250L).map { a.shouldCapture(TEN_MIN + it) }
        val sequenceB = (0L until 90_000L step 250L).map { b.shouldCapture(TEN_MIN + it) }
        assertEquals(sequenceA, sequenceB)
        assertTrue(sequenceA.contains(true) && sequenceA.contains(false))
    }

    @Test
    fun backwardsClockFailsOpenFromTheDeepTier() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
        // Still-run elapsed goes short of its threshold: degrade to the 20% tier (the quiet
        // anchor is still cleared at this instant)...
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN - 200_000))
        // ...and a clock behind even the quiet threshold degrades all the way to continuous.
        assertEquals(CaptureMode.CONTINUOUS, d.mode(100_000L))
        assertTrue(d.shouldCapture(100_000L))
        assertEquals(0L, d.msUntilNextCapture(100_000L))
    }

    @Test
    fun resetClearsTheStillRunToo() {
        val d = MicDutyCycle()
        d.onObservation(voiced = false, nowMs = 0L, still = true)
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
        d.reset()
        assertEquals(CaptureMode.CONTINUOUS, d.mode(TEN_MIN))
        // Both clocks start fresh: 5 min back to the 20% tier, 10 min back to deep.
        d.onObservation(voiced = false, nowMs = TEN_MIN, still = true)
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN + FIVE_MIN))
        assertEquals(CaptureMode.DUTY_CYCLED, d.mode(TEN_MIN + TEN_MIN - 1))
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN + TEN_MIN))
    }

    @Test
    fun repeatedStillWindowsDoNotRestartTheStillRun() {
        val d = MicDutyCycle()
        for (t in 0L until 500_000L step 1_000L) d.onObservation(voiced = false, nowMs = t, still = true)
        // The still run started at t=0, so deep engages at 600_000 — not 600_000 after the LAST
        // still window (mirrors the quiet run's own anchoring rule).
        assertEquals(CaptureMode.DEEP_DUTY_CYCLED, d.mode(TEN_MIN))
    }
}
