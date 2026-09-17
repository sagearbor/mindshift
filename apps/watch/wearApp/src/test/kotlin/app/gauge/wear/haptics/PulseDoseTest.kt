package app.gauge.wear.haptics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * DOSE, not correctness — the question no other test in this repo asks.
 *
 * Every other haptic test here checks that the *right* cue plays for the right
 * input. None of them ever counted how OFTEN the wrist is asked to buzz, and on
 * 2026-09-10 the owner reported it "buzzing all over the place" during a real
 * session. The cause was not a wrong cue: it was the proportional pulse train
 * doing exactly what it was designed to do — one tap per 1 s audio window, for
 * as long as the wearer stays >= 6 dB over their own baseline.
 *
 * The envelopes below are MEASURED, by porting [app.gauge.shared.sentinel.
 * SentinelDetector] exactly (same seeding, same silence floor, same
 * "loud windows never feed the baseline" rule) over the two committed REAL
 * recordings. Reproduce with `python scripts/pulse_dose.py`.
 *
 * Read what those two recordings are before reading the numbers:
 * `family_real` is a father and son talking about bike camp, containing one
 * mildly raised turn. `poker6_real` is a poker game — nobody is arguing at all.
 */
class PulseDoseTest {

    /** Drives [PulseEngine] the way SentinelController does: one call per 1 s window. */
    private fun buzzes(envelope: List<Double>, intervalMs: Long?): Int {
        var now = 0L
        val engine = PulseEngine(intervalMs = { intervalMs }, nowMs = { now })
        var count = 0
        for (db in envelope) {
            if (engine.onWindow(db, SESSION_PULSE_THRESHOLD_DB) != null) count++
            now += WINDOW_MS
        }
        return count
    }

    private fun perMinute(envelope: List<Double>, intervalMs: Long?): Double =
        buzzes(envelope, intervalMs) * 60.0 / envelope.size

    @Test
    fun `the default shipped until 2026-09-10 buzzed 14 times a minute through an ordinary chat`() {
        assertEquals(7, buzzes(FAMILY_REAL, 250L))
        assertTrue(perMinute(FAMILY_REAL, 250L) > 14.0, "measured 14.5/min, got ${perMinute(FAMILY_REAL, 250L)}")
    }

    @Test
    fun `and 36 times a minute through a poker game, where nobody is arguing`() {
        assertEquals(18, buzzes(POKER6_REAL, 250L))
        assertTrue(perMinute(POKER6_REAL, 250L) > 35.0, "measured 36/min, got ${perMinute(POKER6_REAL, 250L)}")
    }

    @Test
    fun `the one thing a wearer could do about it did nothing`() {
        // "Pulse speed" offers 0.25 s / 0.5 s / 1 s. Every one of them is <= the 1 s WINDOW, so
        // the spacing clamp never binds and all three deliver the identical dose. Someone who
        // felt nagged and slowed the pulse down would have felt no difference whatsoever.
        for (env in listOf(FAMILY_REAL, POKER6_REAL)) {
            assertEquals(buzzes(env, 250L), buzzes(env, 500L))
            assertEquals(buzzes(env, 250L), buzzes(env, 1000L))
        }
    }

    @Test
    fun `off means silent — the new default`() {
        assertEquals(0, buzzes(FAMILY_REAL, null))
        assertEquals(0, buzzes(POKER6_REAL, null))
    }

    @Test
    fun `the detector's seeding rule already protects the cold start`() {
        // Worth pinning because it is the obvious suspect and it is NOT guilty: the first two
        // windows report 0.0 (unvoiced) and the seed windows are judged against the first voiced
        // window, so starting to talk cannot read as a shout. An earlier pass of this
        // investigation blamed the cold start using the SERVER's baseline (watch/vectors.py
        // running_median, which returns a baseline from a single sample) — that is a different
        // algorithm on a different device, and it does not describe the wrist.
        val seeded = FAMILY_REAL.take(6)
        assertTrue(seeded.all { it < SESSION_PULSE_THRESHOLD_DB }, "cold start should not pulse: $seeded")
    }

    @Test
    fun `sustained loudness never habituates, by design — which is why dose has to be capped elsewhere`() {
        // The baseline deliberately excludes windows over the trigger bar, so that sustained
        // yelling cannot drag the bar up and silence the detector. Correct for DETECTION, and
        // exactly why the pulse train cannot be left to self-limit: the longer the argument, the
        // longer it buzzes, once a second, forever.
        val tail = FAMILY_REAL.drop(20)
        assertTrue(tail.count { it >= SESSION_PULSE_THRESHOLD_DB } >= 5,
            "the raised stretch stays over the bar rather than being absorbed: $tail")
    }

    private companion object {
        /** SentinelController's window: MAX_SLICE_BYTES 32000 / 2 bytes / 16 kHz. */
        const val WINDOW_MS = 1000L

        /** pulseThresholdDbFor(SESSION) == Mode.STANDARD's triggerDbOverBaseline. */
        const val SESSION_PULSE_THRESHOLD_DB = 6.0

        /** Measured, not hand-written — `python scripts/pulse_dose.py`. Two decimals on
         *  purpose: the last window here is 5.98, i.e. 0.02 dB under the bar, so rounding this
         *  data to one decimal silently adds a buzz that the device never delivers. */
        val FAMILY_REAL = listOf(
            0.00, 0.00, 1.67, 0.81, 2.85, 2.04, -5.96, -6.70, -15.85, -12.11,
            4.64, 4.79, 7.71, -0.22, -0.89, -8.66, -2.68, -3.82, -6.17, -11.00,
            9.50, 11.00, 11.55, 8.56, 10.40, 6.22, 4.69, 3.15, 5.98,
        )

        val POKER6_REAL = listOf(
            0.00, 0.00, 6.78, -0.72, -3.33, -4.26, 6.02, -1.36, 4.32, 5.91,
            8.33, 8.63, 12.81, 9.83, 11.72, 4.07, 14.79, 11.45, 13.62, 8.13,
            15.92, 19.18, 15.10, 8.13, -3.84, 9.51, 17.81, 10.83, 0.09, -0.78,
        )
    }
}

/**
 * The SECOND dose defect, and the one that survives turning the pulse train off: PRD §6's
 * reminder repeat.
 *
 * Level 3 repeated every 10 s while the level held, and [app.gauge.shared.NudgePolicy]
 * de-escalates only one level per 20 s of quiet — so a sustained argument, the exact situation
 * level 3 exists for, buzzed without end. These cases pin the arithmetic before and after the
 * back-off, using the same "count what the wrist FEELS" method as [PulseDoseTest].
 */
class ReminderDoseTest {
    private var fakeVibrator = FakeVibratorPort()
    private var timeMs = 0L
    private var director = HapticDirector(fakeVibrator, nowMs = { timeMs })

    /** Escalate once to [level], then hold it for [seconds], ticking once a second like the
     *  service does. Returns every moment the wrist buzzed, in seconds.
     *
     *  Builds a FRESH director and clock each time: a reminder is deliberately silent under a
     *  backwards-stepping clock, so reusing one across two runs would measure that safety rule
     *  rather than the back-off. */
    private fun sustained(level: Int, seconds: Int): List<Int> {
        fakeVibrator = FakeVibratorPort()
        timeMs = 0L
        director = HapticDirector(fakeVibrator, nowMs = { timeMs })
        val felt = mutableListOf<Int>()
        director.onNudge(app.gauge.shared.NudgeEvent(channel = "A", level = level, t = 0.0))
        felt.add(0)
        for (s in 1..seconds) {
            timeMs = s * 1000L
            val due = director.dueReminder() ?: continue
            director.replayReminder(due)
            felt.add(s)
        }
        return felt
    }

    @Test
    fun `a three-minute level-3 stretch now costs five buzzes, not eighteen`() {
        val felt = sustained(level = 3, seconds = 180)
        // 10s, then 20, 40, 80 — then capped at level 1's gentle two minutes.
        assertEquals(listOf(0, 10, 30, 70, 150), felt)
        assertTrue(felt.size < 6, "measured 18 before the back-off; got ${felt.size}")
    }

    @Test
    fun `the first repeat is still prompt — urgency is not what was wrong`() {
        // The complaint was never "it told me too soon", it was "it never stopped". Level 3 must
        // still speak up within 10 s of escalating.
        assertEquals(10, sustained(level = 3, seconds = 60)[1])
    }

    @Test
    fun `a NEW escalation resets the back-off, so things getting worse is still reported`() {
        sustained(level = 3, seconds = 180)
        val before = fakeVibrator.calls.size + fakeVibrator.composedPlayed.size
        // A fresh level-3 detection after the lane has gone quiet-ish.
        timeMs += 10_000L
        director.onNudge(app.gauge.shared.NudgeEvent(channel = "A", level = 3, t = 190.0, vectors = listOf("yelling")))
        timeMs += 10_000L
        val due = director.dueReminder()
        assertTrue(due != null, "a new escalation must restart the prompt cadence, not inherit the backed-off one")
        assertTrue(fakeVibrator.calls.size + fakeVibrator.composedPlayed.size > before)
    }

    @Test
    fun `levels 1 and 2 are barely touched — they were never the problem`() {
        // Level 1 is already at the cap, so its cadence is unchanged; level 2 backs off from one
        // minute to the same two-minute floor rather than to silence.
        assertEquals(listOf(0, 120), sustained(level = 1, seconds = 240).take(2))
        assertEquals(listOf(0, 60, 180), sustained(level = 2, seconds = 240).take(3))
    }
}
