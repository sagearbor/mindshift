package app.gauge.wear.haptics

import app.gauge.shared.NudgeHapticSchedule
import app.gauge.shared.NudgeVocabulary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HapticPatternsTest {

    // --- channel A: device-tuned clicks --------------------------------------------------------

    @Test
    fun channelALevel1IsAPredefinedClick() {
        assertEquals(HapticCue.Predefined(PredefinedEffect.CLICK), HapticPatterns.cue("A", 1))
    }

    @Test
    fun channelALevel2IsADoubleClickPerPrdSection6() {
        // Track 1: was a single HEAVY_CLICK; PRD §6 says "double pulse" and a count is what a
        // wrist can actually tell apart from a single tap.
        assertEquals(HapticCue.ComposedClicks(count = 2, gapMs = 170L), HapticPatterns.cue("A", 2))
    }

    @Test
    fun channelALevel3IsThreeComposedClicksAtTheSilenceFloorGap() {
        assertEquals(HapticCue.ComposedClicks(count = 3, gapMs = 170L), HapticPatterns.cue("A", 3))
    }

    @Test
    fun channelATapCountsMatchTheSharedScheduleAtEveryLevel() {
        // The one-source-of-truth guard: HapticPatterns must render exactly the pulse count
        // NudgeHapticSchedule (PRD §6) prescribes, on both the device-tuned and fallback paths.
        for (level in 1..3) {
            val plan = NudgeHapticSchedule.planFor(level)
            val tuned = when (val cue = HapticPatterns.cue("A", level)) {
                is HapticCue.Predefined -> 1
                is HapticCue.ComposedClicks -> cue.count
                is HapticCue.Waveform -> cue.amplitudes.count { it > 0 }
                null -> -1
            }
            assertEquals(plan.pulses, tuned, "A/$level tuned cue pulse count")
            val fallback = HapticPatterns.waveformFallback("A", level)!!
            assertEquals(plan.pulses, fallback.amplitudes.count { it > 0 }, "A/$level fallback pulse count")
        }
    }

    // --- channel B: long smooth buzzes, waveform-only (channel identity) -----------------------

    @Test
    fun channelBLevelsAreWaveformsNeverPredefined() {
        for (level in 1..3) {
            assertTrue(
                HapticPatterns.cue("B", level) is HapticCue.Waveform,
                "B/$level must stay a waveform so the two channels remain distinguishable by feel",
            )
        }
    }

    @Test
    fun channelBLevel1IsOneLongBuzz() {
        assertEquals(
            HapticCue.Waveform(listOf(0L, 250L), listOf(0, 220)),
            HapticPatterns.cue("B", 1),
        )
    }

    @Test
    fun channelBLevel3IsThreeMaxBuzzes() {
        assertEquals(
            HapticCue.Waveform(listOf(0L, 400L, 170L, 400L, 170L, 400L), listOf(0, 255, 0, 255, 0, 255)),
            HapticPatterns.cue("B", 3),
        )
    }

    // --- fallbacks -----------------------------------------------------------------------------

    @Test
    fun channelAFallbacksAreTheHeatedVocabularyVerbatim() {
        // Channel A IS 📈 Heated, so its fallback is the contract's waveform, not a restatement.
        // Restating it is exactly how the two drifted apart on 2026-09-06.
        for (level in 1..3) {
            val h = NudgeVocabulary.hapticFor("H", level)!!
            assertEquals(HapticCue.Waveform(h.timingsMs, h.amplitudes), HapticPatterns.waveformFallback("A", level))
        }
    }

    @Test
    fun channelAFallbacksStayConsistentWithTheSharedSchedule() {
        // The schedule owns the tap COUNT and the amplitude ramp; the vocabulary owns the tap
        // LENGTHS. Both have to keep agreeing or one of the two is lying about the cue.
        for (level in 1..3) {
            val plan = NudgeHapticSchedule.planFor(level)
            val wave = HapticPatterns.waveformFallback("A", level)!!
            val taps = wave.timingsMs.filterIndexed { i, _ -> i % 2 == 1 }
            assertEquals(plan.pulses, taps.size, "level $level tap count")
            assertEquals(plan.amplitudeRamp, wave.amplitudes.filterIndexed { i, _ -> i % 2 == 1 }, "level $level ramp")
            assertTrue(taps.all { it >= NudgeVocabulary.MIN_ON_MS }, "level $level taps must be perceptible")
        }
    }

    @Test
    fun channelBFallbackIsItsOwnCue() {
        for (level in 1..3) {
            assertEquals(HapticPatterns.cue("B", level), HapticPatterns.waveformFallback("B", level))
        }
    }

    @Test
    fun invalidInputsReturnNull() {
        assertNull(HapticPatterns.cue("A", 0))
        assertNull(HapticPatterns.cue("A", 4))
        assertNull(HapticPatterns.cue("C", 1))
        assertNull(HapticPatterns.waveformFallback("A", 0))
        assertNull(HapticPatterns.waveformFallback("C", 2))
    }

    @Test
    fun everyMultiTapGapRespectsTheNeverMergeFloor() {
        // The 170ms silence floor is a pattern-wide rule now, not just the pulse train's. Gap
        // entries in a waveform are the odd-indexed timings after the first (off, on, off, on...).
        for (channel in listOf("A", "B")) {
            for (level in 1..3) {
                val wf = HapticPatterns.waveformFallback(channel, level) ?: continue
                // timings alternate [initialDelay, on, gap, on, gap, on] — gaps are indices 2, 4, ...
                for (i in 2 until wf.timingsMs.size step 2) {
                    assertTrue(
                        wf.timingsMs[i] >= HapticPatterns.MIN_GAP_MS,
                        "$channel/$level gap ${wf.timingsMs[i]}ms < ${HapticPatterns.MIN_GAP_MS}ms",
                    )
                }
            }
        }
    }

    // --- pulse bands: raised floor, kept scaling ------------------------------------------------

    @Test
    fun pulseBandsAreRaisedAndMonotonic() {
        assertEquals(Pulse(50L, 180), HapticPatterns.pulseBandFor(0.0))
        assertEquals(Pulse(50L, 180), HapticPatterns.pulseBandFor(2.9))
        assertEquals(Pulse(60L, 205), HapticPatterns.pulseBandFor(3.0))
        assertEquals(Pulse(70L, 230), HapticPatterns.pulseBandFor(6.0))
        assertEquals(Pulse(80L, 255), HapticPatterns.pulseBandFor(9.0))
        assertEquals(Pulse(80L, 255), HapticPatterns.pulseBandFor(40.0))
    }

    @Test
    fun maxBandDurationIsUnchangedSoTheClampFloorMathHolds() {
        // 80 + 170 == 250 == the fastest "Pulse speed" preference; raising the max band's duration
        // would silently slow the fastest preset via PulseEngine.effectiveIntervalMs.
        assertEquals(80L, HapticPatterns.pulseBandFor(9.0).durationMs)
    }

    @Test
    fun armedShoutTapIsTheFloorBand() {
        assertEquals(HapticPatterns.pulseBandFor(0.0), HapticPatterns.ARMED_SHOUT_TAP)
    }

    // --- the nudge vocabulary ------------------------------------------------------------------

    @Test
    fun heatedKeepsTheOemTunedChannelCue() {
        // H must NOT degrade to a raw waveform: predefined/composed clicks are tuned to the real
        // actuator, and a waveform would feel worse (the v0.2.4 device finding).
        for (level in 1..3) {
            assertEquals(HapticPatterns.cue("A", level), HapticPatterns.cueFor("A", level, "H"))
            assertEquals(HapticPatterns.cue("A", level), HapticPatterns.cueFor("A", level, null))
        }
    }

    @Test
    fun aCodeOnItsHomeChannelPlaysItsOwnRhythm() {
        for (code in listOf("C", "A")) {
            val wave = NudgeVocabulary.hapticFor(code, 2)!!
            assertEquals(
                HapticCue.Waveform(wave.timingsMs, wave.amplitudes),
                HapticPatterns.cueFor("A", 2, code),
                "code $code on channel A",
            )
        }
        val pulse = NudgeVocabulary.hapticFor("P", 2)!!
        assertEquals(
            HapticCue.Waveform(pulse.timingsMs, pulse.amplitudes),
            HapticPatterns.cueFor("B", 2, "P"),
        )
    }

    @Test
    fun aCodeOnTheWrongLaneKeepsThatLanesFeel() {
        // The two channels must stay distinguishable by feel alone: a channel-A code relayed on
        // channel B plays B's smooth buzz, never A's crisp rhythm.
        assertEquals(HapticPatterns.cue("B", 2), HapticPatterns.cueFor("B", 2, "C"))
        assertEquals(HapticPatterns.cue("A", 2), HapticPatterns.cueFor("A", 2, "P"))
    }

    @Test
    fun aCodeThatMustNeverBuzzStaysSilentEvenOnAValidChannel() {
        // 🧘 K has no cue by CONTRACT — falling through to channel A's click would
        // buzz someone to tell them nothing happened, which is the nag the code exists to avoid.
        for (level in 1..3) {
            assertNull(HapticPatterns.cueFor("A", level, "K"), "K L$level must be silent")
            assertNull(HapticPatterns.waveformFallbackFor("A", level, "K"), "K L$level fallback")
        }
    }

    @Test
    fun anUnknownCodeFallsBackToTheChannelCue() {
        // Not the same thing: an unrecognised label means "this build doesn't know that code",
        // and losing a real nudge over a label would be worse than playing a generic one.
        assertEquals(HapticPatterns.cue("A", 2), HapticPatterns.cueFor("A", 2, "Z"))
        assertEquals(HapticPatterns.cue("A", 2), HapticPatterns.cueFor("A", 2, null))
        assertNull(HapticPatterns.cueFor("A", 0, "C"))
        assertNull(HapticPatterns.cueFor("A", 4, "C"))
    }
}
