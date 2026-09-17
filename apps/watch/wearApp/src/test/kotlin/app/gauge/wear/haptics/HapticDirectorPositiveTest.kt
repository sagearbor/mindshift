package app.gauge.wear.haptics

import app.gauge.shared.NudgeEvent
import app.gauge.shared.NudgeVocabulary
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * [HapticDirector.playPositive] — praise on the wrist.
 *
 * The whole point of these cases is the NEGATIVE space. Before this path existed the relay carried
 * alert vectors only, so a wearer with the phone in a pocket felt every complaint and no praise;
 * the wrist was a pure complaint channel. The fix must not overshoot in the other direction: a
 * positive is unleveled by contract, so it must never move channel A's level, never arm or rebase
 * the PRD §6 reminder, and never be re-fired by it. A wrist that repeats "well done" every two
 * minutes is worse than one that never said it.
 */
class HapticDirectorPositiveTest {
    private val fakeVibrator = FakeVibratorPort()
    private var timeMs = 0L
    private val director = HapticDirector(fakeVibrator, nowMs = { timeMs })

    @Test
    fun `plays the vocabulary swell for a positive code`() {
        director.playPositive("E")
        assertEquals(1, fakeVibrator.calls.size)
        val expected = NudgeVocabulary.hapticFor("E", 1)!!
        assertEquals(expected.timingsMs, fakeVibrator.calls[0].timingsMs.toList())
        assertEquals(expected.amplitudes, fakeVibrator.calls[0].amplitudes.toList())
    }

    @Test
    fun `each positive code has its own distinguishable gesture`() {
        val waves = listOf("D", "E", "R").map { code ->
            fakeVibrator.calls.clear()
            director.playPositive(code)
            assertEquals(1, fakeVibrator.calls.size, "$code played nothing")
            fakeVibrator.calls[0].timingsMs.toList() to fakeVibrator.calls[0].amplitudes.toList()
        }
        assertEquals(waves.size, waves.toSet().size, "two positives feel identical: $waves")
    }

    @Test
    fun `a positive never arms the reminder`() {
        director.playPositive("E")
        assertEquals(0, director.reminderLevel())
        timeMs += 10 * 60_000L
        assertNull(director.dueReminder())
    }

    @Test
    fun `a positive leaves an armed reminder untouched`() {
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0))
        val armedAt = timeMs
        timeMs += 5_000L
        director.playPositive("E")
        assertEquals(3, director.reminderLevel(), "praise changed the escalation level")

        // The L3 repeat is due 10 s after the nudge — not 10 s after the praise. If playPositive
        // had rebased the reminder clock this would still be null.
        timeMs = armedAt + 10_000L + 1L
        val due = director.dueReminder()
        assertEquals(3, due?.level, "praise rebased or cleared the PRD §6 reminder clock")
    }

    @Test
    fun `a positive does not become the reminder's cue`() {
        director.onNudge(NudgeEvent(channel = "A", level = 3, t = 0.0, vectors = listOf("yelling")))
        director.playPositive("E")
        fakeVibrator.calls.clear()
        timeMs += 10_001L
        director.replayReminder(director.dueReminder()!!)
        val praise = NudgeVocabulary.hapticFor("E", 1)!!
        assertTrue(
            fakeVibrator.calls.none { it.timingsMs.toList() == praise.timingsMs },
            "the reminder replayed the praise gesture",
        )
    }

    @Test
    fun `silence-by-contract and non-positive codes play nothing`() {
        // K is the calm streak: buzzing someone to say nothing happened is the definition of a nag.
        // H is an ALERT — routing it here would launder a complaint as praise.
        for (code in listOf("K", "H", "C", "A", "P", "", "zzz")) {
            director.playPositive(code)
        }
        assertEquals(0, fakeVibrator.calls.size)
        assertEquals(0, fakeVibrator.predefinedPlayed.size)
        assertEquals(0, fakeVibrator.composedPlayed.size)
        assertEquals(0, director.reminderLevel())
    }

    @Test
    fun `a positive does not swallow the next real nudge via dedupe`() {
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 0.0))
        val afterNudge = fakeVibrator.calls.size + fakeVibrator.composedPlayed.size
        timeMs += 1_000L
        director.playPositive("E")
        timeMs += 1_000L
        // Same channel+level inside the 5 s window: still a duplicate, because a positive must not
        // touch the dedupe state in EITHER direction.
        director.onNudge(NudgeEvent(channel = "A", level = 2, t = 2.0))
        assertEquals(
            afterNudge + 1,
            fakeVibrator.calls.size + fakeVibrator.composedPlayed.size,
            "the positive disturbed onNudge's dedupe",
        )
    }
}
