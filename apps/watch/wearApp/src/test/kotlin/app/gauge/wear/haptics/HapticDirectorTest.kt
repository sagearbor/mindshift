package app.gauge.wear.haptics

import app.gauge.shared.NudgeEvent
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.VibratorPort
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class FakeVibratorPort : VibratorPort {
    data class Call(val timingsMs: LongArray, val amplitudes: IntArray)

    val calls = mutableListOf<Call>()
    val predefinedPlayed = mutableListOf<PredefinedEffect>()
    val composedPlayed = mutableListOf<Pair<Int, Long>>()

    /** Settable so tests can exercise [HapticDirector.playPulse]'s documented fixed-amplitude
     * fallback. Defaults to true (the common case). */
    var amplitudeControl: Boolean = true

    /** v0.2.4: device-tuned-path support flags. Default FALSE on purpose — every test (including
     * SentinelControllerTest's) that doesn't opt in exercises the waveform-fallback path, which
     * is exactly what a support-less device does. */
    var supportsPredefined: Boolean = false
    var supportsComposition: Boolean = false

    /** Set to make the device-tuned paths throw — pins [HapticDirector]'s fail-soft fallback. */
    var throwOnDeviceTuned: Boolean = false

    override fun vibrate(timingsMs: LongArray, amplitudes: IntArray) {
        calls.add(Call(timingsMs.copyOf(), amplitudes.copyOf()))
    }

    override fun hasAmplitudeControl(): Boolean = amplitudeControl

    override fun playPredefined(effect: PredefinedEffect): Boolean {
        if (throwOnDeviceTuned) throw IllegalStateException("vibrator died")
        if (!supportsPredefined) return false
        predefinedPlayed.add(effect)
        return true
    }

    override fun playComposedClicks(count: Int, gapMs: Long): Boolean {
        if (throwOnDeviceTuned) throw IllegalStateException("vibrator died")
        if (!supportsComposition) return false
        composedPlayed.add(count to gapMs)
        return true
    }
}

class HapticDirectorTest {

    private lateinit var fakeVibrator: FakeVibratorPort
    private var timeMs = 0L
    private lateinit var director: HapticDirector

    private fun setup() {
        fakeVibrator = FakeVibratorPort()
        timeMs = 0L
        director = HapticDirector(fakeVibrator, nowMs = { timeMs })
    }

    private fun assertWaveformPlayed(call: FakeVibratorPort.Call, expected: HapticCue.Waveform) {
        assertTrue(call.timingsMs.contentEquals(expected.timingsMs.toLongArray()))
        assertTrue(call.amplitudes.contentEquals(expected.amplitudes.toIntArray()))
    }

    // --- device-tuned path ---------------------------------------------------------------------

    @Test
    fun channelALevel1PlaysThePredefinedClickWhenSupported() {
        setup()
        fakeVibrator.supportsPredefined = true
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        assertEquals(listOf(PredefinedEffect.CLICK), fakeVibrator.predefinedPlayed)
        assertEquals(0, fakeVibrator.calls.size) // no waveform when the tuned path landed
    }

    @Test
    fun channelALevel2PlaysTwoComposedClicksWhenSupported() {
        // Track 1 / PRD §6: level 2 is a DOUBLE pulse (was one HEAVY_CLICK).
        setup()
        fakeVibrator.supportsComposition = true
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        assertEquals(listOf(2 to 170L), fakeVibrator.composedPlayed)
        assertEquals(0, fakeVibrator.predefinedPlayed.size)
    }

    @Test
    fun channelALevel3PlaysThreeComposedClicksWhenSupported() {
        setup()
        fakeVibrator.supportsComposition = true
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0))
        assertEquals(listOf(3 to 170L), fakeVibrator.composedPlayed)
        assertEquals(0, fakeVibrator.calls.size)
    }

    // --- fallback path (default fake = support-less device) ------------------------------------

    @Test
    fun unsupportedPredefinedFallsBackToTheFullAmplitudeWaveform() {
        setup() // supportsPredefined = false
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        assertEquals(1, fakeVibrator.calls.size)
        assertWaveformPlayed(fakeVibrator.calls[0], HapticPatterns.waveformFallback("A", 1)!!)
    }

    @Test
    fun unsupportedCompositionFallsBackToTheFullAmplitudeWaveform() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0))
        assertEquals(1, fakeVibrator.calls.size)
        assertWaveformPlayed(fakeVibrator.calls[0], HapticPatterns.waveformFallback("A", 3)!!)
    }

    @Test
    fun throwingDeviceTunedPathDegradesToTheFallbackNotSilence() {
        setup()
        fakeVibrator.supportsPredefined = true
        fakeVibrator.throwOnDeviceTuned = true
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        assertEquals(1, fakeVibrator.calls.size)
        assertWaveformPlayed(fakeVibrator.calls[0], HapticPatterns.waveformFallback("A", 2)!!)
    }

    @Test
    fun channelBPlaysItsWaveformDirectlyRegardlessOfPredefinedSupport() {
        setup()
        fakeVibrator.supportsPredefined = true
        director.onNudge(NudgeEvent(channel = "B", level = 2, t = 0.0))
        assertEquals(0, fakeVibrator.predefinedPlayed.size) // channel identity: B never clicks
        assertEquals(1, fakeVibrator.calls.size)
        assertWaveformPlayed(fakeVibrator.calls[0], HapticPatterns.cue("B", 2) as HapticCue.Waveform)
    }

    // --- dedupe semantics unchanged -------------------------------------------------------------

    @Test
    fun level0IsSilent() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 0, t = 0.0))
        assertEquals(0, fakeVibrator.calls.size)
        assertEquals(0, fakeVibrator.predefinedPlayed.size)
    }

    @Test
    fun duplicateSameChannelSameLevelWithin5000msIsDeduped() {
        setup()
        fakeVibrator.supportsPredefined = true
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        timeMs = 1000
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 1.0))
        assertEquals(1, fakeVibrator.predefinedPlayed.size)
    }

    @Test
    fun sameChannelSameLevelAfter5000msPlaysAgain() {
        setup()
        fakeVibrator.supportsPredefined = true
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        timeMs = 6000
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 6.0))
        assertEquals(2, fakeVibrator.predefinedPlayed.size)
    }

    @Test
    fun levelChangeAlwaysPlays() {
        setup()
        fakeVibrator.supportsPredefined = true
        fakeVibrator.supportsComposition = true
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        timeMs = 100
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 1.0))
        // Track 1: L1 is still the predefined click; L2 is now the PRD §6 double (composed).
        assertEquals(listOf(PredefinedEffect.CLICK), fakeVibrator.predefinedPlayed)
        assertEquals(listOf(2 to 170L), fakeVibrator.composedPlayed)
    }

    @Test
    fun channelChangeAlwaysPlays() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        timeMs = 100
        director.onNudge(NudgeEvent(channel = "B", level = 2, t = 1.0))
        assertEquals(2, fakeVibrator.calls.size) // both landed as waveforms on the default fake
    }

    // --- demo seam (v0.2.4, Settings "Feel the buzzes") ----------------------------------------

    @Test
    fun demoBypassesTheDedupeWindow() {
        setup()
        director.demo("A", 2)
        director.demo("A", 2) // back-to-back, same channel+level, zero ms apart
        assertEquals(2, fakeVibrator.calls.size)
    }

    @Test
    fun demoDoesNotPolluteDedupeStateForRealNudges() {
        setup()
        director.demo("A", 2)
        // A REAL A/2 nudge immediately after the demo must still play — the demo never wrote
        // lastChannel/lastLevel.
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        assertEquals(2, fakeVibrator.calls.size)
    }

    @Test
    fun demoWithInvalidInputsIsANoOp() {
        setup()
        director.demo("A", 0)
        director.demo("C", 2)
        assertEquals(0, fakeVibrator.calls.size)
    }

    // --- PRD §6 repeat schedule (Track 1) -------------------------------------------------------

    @Test
    fun reminderIsNotDueUntilTheLevelsIntervalElapses() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0)) // double pulse, every 1 min
        assertEquals(2, director.reminderLevel())
        timeMs = 59_999
        assertNull(director.dueReminder())
        timeMs = 60_000
        val due = director.dueReminder()
        assertEquals(NudgeEvent(channel = "A", level = 2, t = 60.0, vectors = emptyList()), due)
    }

    @Test
    fun replayReminderPlaysTheSameCueAndRebasesTheClock() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0)) // single soft, every 2 min
        assertEquals(1, fakeVibrator.calls.size)
        timeMs = 120_000
        director.replayReminder(director.dueReminder()!!)
        assertEquals(2, fakeVibrator.calls.size)
        assertWaveformPlayed(fakeVibrator.calls[1], HapticPatterns.waveformFallback("A", 1)!!)
        // Rebased: not due again until another full interval.
        timeMs = 239_999
        assertNull(director.dueReminder())
        timeMs = 240_000
        assertEquals(1, director.dueReminder()!!.level)
    }

    @Test
    fun level3RepeatsEvery10SecondsAsTheContinuousPattern() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0))
        timeMs = 10_000
        assertEquals(3, director.dueReminder()!!.level)
    }

    @Test
    fun level0DeescalationOnChannelAStopsReminders() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0))
        director.onNudge(NudgeEvent(channel = "A", level = 0, t = 1.0)) // silent, but must clear
        assertEquals(0, director.reminderLevel())
        timeMs = 1_000_000
        assertNull(director.dueReminder())
    }

    @Test
    fun levelChangeRetargetsTheReminderToTheNewLevel() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        timeMs = 30_000
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 30.0))
        timeMs = 39_999
        assertNull(director.dueReminder()) // 10 s from the LEVEL-3 play, not from the level-1 one
        timeMs = 40_000
        assertEquals(3, director.dueReminder()!!.level)
    }

    @Test
    fun channelBNeverSchedulesReminders() {
        setup()
        director.onNudge(NudgeEvent(channel = "B", level = 3, t = 0.0))
        assertEquals(0, director.reminderLevel())
        timeMs = 1_000_000
        assertNull(director.dueReminder())
    }

    @Test
    fun dedupedDuplicateDoesNotRebaseTheReminderClock() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        timeMs = 3_000
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 3.0)) // swallowed by the 5 s dedupe
        assertEquals(1, fakeVibrator.calls.size)
        timeMs = 60_000 // 60 s from the play that actually happened
        assertEquals(2, director.dueReminder()!!.level)
    }

    @Test
    fun replayReminderIsANoOpIfTheLevelWasClearedMeanwhile() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        timeMs = 60_000
        val due = director.dueReminder()!!
        director.clearReminder()
        director.replayReminder(due)
        assertEquals(1, fakeVibrator.calls.size)
    }

    @Test
    fun replayReminderRefreshesTheDedupeSoAServerResendIsSwallowed() {
        setup()
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        timeMs = 60_000
        director.replayReminder(director.dueReminder()!!)
        timeMs = 61_000
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 61.0)) // same level, 1 s after the reminder
        assertEquals(2, fakeVibrator.calls.size)
    }

    // --- playPulse (unchanged contract, raised band values live in HapticPatternsTest) ----------

    @Test
    fun playPulseSendsExactSingleTapArrays() {
        setup()
        director.playPulse(Pulse(durationMs = 60L, amplitude = 205))
        assertEquals(1, fakeVibrator.calls.size)
        val call = fakeVibrator.calls[0]
        assertTrue(call.timingsMs.contentEquals(longArrayOf(0L, 60L)))
        assertTrue(call.amplitudes.contentEquals(intArrayOf(0, 205)))
    }

    @Test
    fun playPulseFallsBackToFixedAmplitudeWithoutAmplitudeControl() {
        setup()
        fakeVibrator.amplitudeControl = false
        director.playPulse(Pulse(durationMs = 50L, amplitude = 180))
        assertEquals(1, fakeVibrator.calls.size)
        assertTrue(fakeVibrator.calls[0].amplitudes.contentEquals(intArrayOf(0, 255)))
    }

    @Test
    fun playPulseIsNotDedupedLikeOnNudge() {
        setup()
        director.playPulse(Pulse(durationMs = 50L, amplitude = 180))
        director.playPulse(Pulse(durationMs = 50L, amplitude = 180))
        assertEquals(2, fakeVibrator.calls.size)
    }

    // --- telemetry breadcrumb (v0.2.4 review round 1: which physical path actually played) -----

    @Test
    fun repeatedSameOutcomeAcrossPlaysLogsOnlyOnce() {
        setup() // fakeVibrator.supportsPredefined = false -> every A/1 play falls back to waveform
        val log = mutableListOf<String>()
        director = HapticDirector(fakeVibrator, nowMs = { timeMs }, diag = DiagLog { l, t, m -> log.add("$l/$t/$m") })

        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0))
        timeMs = 6000 // past onNudge's own 5s dedupe so play() genuinely re-runs each time
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 6.0))
        timeMs = 12000
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 12.0))

        assertEquals(1, log.count { it == "info/HapticDirector/haptic path: A L1 = waveform-fallback" })
    }

    @Test
    fun distinctNewOutcomeForSameChannelLevelLogsAgain() {
        setup()
        val log = mutableListOf<String>()
        director = HapticDirector(fakeVibrator, nowMs = { timeMs }, diag = DiagLog { l, t, m -> log.add("$l/$t/$m") })

        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0)) // unsupported -> fallback
        fakeVibrator.supportsPredefined = true
        timeMs = 6000
        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 6.0)) // now supported -> predefined

        assertEquals(1, log.count { it == "info/HapticDirector/haptic path: A L1 = waveform-fallback" })
        assertEquals(1, log.count { it == "info/HapticDirector/haptic path: A L1 = predefined" })
    }

    @Test
    fun differentChannelsAndLevelsTrackTheirOwnOutcomesIndependently() {
        setup()
        val log = mutableListOf<String>()
        director = HapticDirector(fakeVibrator, nowMs = { timeMs }, diag = DiagLog { l, t, m -> log.add("$l/$t/$m") })

        director.onNudge(NudgeEvent(channel = "A", level = 1, t = 0.0)) // A/1 -> waveform-fallback
        director.onNudge(NudgeEvent(channel = "B", level = 1, t = 0.0)) // B/1 -> native waveform

        assertTrue(log.contains("info/HapticDirector/haptic path: A L1 = waveform-fallback"))
        assertTrue(log.contains("info/HapticDirector/haptic path: B L1 = waveform"))
    }
}
