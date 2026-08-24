package app.gauge.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** PRD §6 "Smartwatch Haptics" pinned as executable numbers — see [NudgeHapticSchedule]'s KDoc. */
class NudgeHapticScheduleTest {

    // --- score -> level ------------------------------------------------------------------------

    @Test
    fun scoreAbove70IsSilent() {
        assertEquals(0, NudgeHapticSchedule.levelForScore(100))
        assertEquals(0, NudgeHapticSchedule.levelForScore(71))
    }

    @Test
    fun score50To70IsLevel1InclusiveBothEnds() {
        assertEquals(1, NudgeHapticSchedule.levelForScore(70))
        assertEquals(1, NudgeHapticSchedule.levelForScore(60))
        assertEquals(1, NudgeHapticSchedule.levelForScore(50))
    }

    @Test
    fun score30To49IsLevel2() {
        assertEquals(2, NudgeHapticSchedule.levelForScore(49))
        assertEquals(2, NudgeHapticSchedule.levelForScore(30))
    }

    @Test
    fun scoreBelow30IsLevel3() {
        assertEquals(3, NudgeHapticSchedule.levelForScore(29))
        assertEquals(3, NudgeHapticSchedule.levelForScore(0))
    }

    @Test
    fun scoresOutsideZeroToHundredClampInsteadOfThrowing() {
        assertEquals(0, NudgeHapticSchedule.levelForScore(250))
        assertEquals(3, NudgeHapticSchedule.levelForScore(-5))
    }

    @Test
    fun levelIsMonotonicInScore() {
        var last = 3
        for (score in 0..100) {
            val level = NudgeHapticSchedule.levelForScore(score)
            assertTrue(level <= last, "level must not rise as the score improves (score=$score)")
            last = level
        }
    }

    // --- level -> plan -------------------------------------------------------------------------

    @Test
    fun level0IsNoHapticAndNeverRepeats() {
        val plan = NudgeHapticSchedule.planFor(0)
        assertEquals(NudgeHapticPattern.NONE, plan.pattern)
        assertEquals(0, plan.pulses)
        assertNull(plan.repeatIntervalMs)
    }

    @Test
    fun level1IsASingleSoftPulseEvery2Minutes() {
        val plan = NudgeHapticSchedule.planFor(1)
        assertEquals(NudgeHapticPattern.SINGLE_SOFT, plan.pattern)
        assertEquals(1, plan.pulses)
        assertEquals(120_000L, plan.repeatIntervalMs)
    }

    @Test
    fun level2IsADoublePulseEveryMinute() {
        val plan = NudgeHapticSchedule.planFor(2)
        assertEquals(NudgeHapticPattern.DOUBLE, plan.pattern)
        assertEquals(2, plan.pulses)
        assertEquals(60_000L, plan.repeatIntervalMs)
    }

    @Test
    fun level3IsAContinuousEscalatingPattern() {
        val plan = NudgeHapticSchedule.planFor(3)
        assertEquals(NudgeHapticPattern.ESCALATING, plan.pattern)
        assertEquals(3, plan.pulses)
        assertEquals(10_000L, plan.repeatIntervalMs)
        // "Escalating" means the ramp actually rises, tap over tap.
        assertEquals(plan.amplitudeRamp.sorted(), plan.amplitudeRamp)
        assertTrue(plan.amplitudeRamp.first() < plan.amplitudeRamp.last())
    }

    @Test
    fun everyPlanHasOneAmplitudePerPulseWithinVibrationEffectRange() {
        for (level in 0..3) {
            val plan = NudgeHapticSchedule.planFor(level)
            assertEquals(plan.pulses, plan.amplitudeRamp.size, "level $level")
            assertTrue(plan.amplitudeRamp.all { it in 1..255 }, "level $level amplitudes must be 1..255")
        }
    }

    @Test
    fun repeatCadenceTightensAsLevelRises() {
        val l1 = NudgeHapticSchedule.planFor(1).repeatIntervalMs!!
        val l2 = NudgeHapticSchedule.planFor(2).repeatIntervalMs!!
        val l3 = NudgeHapticSchedule.planFor(3).repeatIntervalMs!!
        assertTrue(l1 > l2 && l2 > l3)
    }

    @Test
    fun outOfRangeLevelsClamp() {
        assertEquals(NudgeHapticSchedule.planFor(3), NudgeHapticSchedule.planFor(9))
        assertEquals(NudgeHapticSchedule.planFor(0), NudgeHapticSchedule.planFor(-1))
    }

    // --- reminder timing -----------------------------------------------------------------------

    @Test
    fun reminderIsDueAtExactlyTheIntervalAndNotBefore() {
        assertFalse(NudgeHapticSchedule.reminderDue(level = 2, lastPlayedMs = 1_000L, nowMs = 60_999L))
        assertTrue(NudgeHapticSchedule.reminderDue(level = 2, lastPlayedMs = 1_000L, nowMs = 61_000L))
    }

    @Test
    fun level0IsNeverDue() {
        assertFalse(NudgeHapticSchedule.reminderDue(level = 0, lastPlayedMs = 0L, nowMs = Long.MAX_VALUE / 2))
    }

    @Test
    fun backwardsClockWaitsRatherThanFiring() {
        assertFalse(NudgeHapticSchedule.reminderDue(level = 3, lastPlayedMs = 50_000L, nowMs = 40_000L))
    }
}
