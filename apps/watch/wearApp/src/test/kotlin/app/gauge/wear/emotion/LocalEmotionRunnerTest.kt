package app.gauge.wear.emotion

import app.gauge.wear.ui.emotionLocalSummary
import java.nio.file.Files
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private fun probs(vararg v: Float) = floatArrayOf(*v)
private val NEUTRAL = probs(0.02f, 0.02f, 0.02f, 0.02f, 0.9f, 0.02f)
private val ANGRY = probs(0.7f, 0.1f, 0.05f, 0.05f, 0.05f, 0.05f)
private val WEAK_ANGRY = probs(0.4f, 0.1f, 0.1f, 0.1f, 0.2f, 0.1f)
private val HAPPY = probs(0.05f, 0.05f, 0.05f, 0.7f, 0.1f, 0.05f)

class NudgeGateTest {
    @Test fun `defaults match the phone`() {
        val c = NudgeConfig()
        assertEquals(0.5, c.minProb); assertEquals(2, c.confirmWindows)
        assertEquals(4, c.maxPerHour); assertEquals(120, c.minGapS)
    }

    @Test fun `neutral never buzzes, weak never buzzes`() {
        val g = NudgeGate()
        assertEquals("neutral", g.decide(NEUTRAL, 0).reason)
        assertEquals("below-threshold", g.decide(WEAK_ANGRY, 3000).reason)
    }

    @Test fun `needs the same class twice in a row`() {
        val g = NudgeGate()
        assertEquals("unconfirmed", g.decide(ANGRY, 0).reason)
        assertTrue(g.decide(ANGRY, 3000).nudge)
        val g2 = NudgeGate()
        g2.decide(ANGRY, 0)
        assertEquals("unconfirmed", g2.decide(HAPPY, 3000).reason) // streak restarts on a different class
        assertTrue(g2.decide(HAPPY, 6000).nudge)
    }

    @Test fun `a neutral window breaks the streak`() {
        val g = NudgeGate()
        g.decide(ANGRY, 0); g.decide(NEUTRAL, 3000)
        assertEquals("unconfirmed", g.decide(ANGRY, 6000).reason)
    }

    @Test fun `min gap 120 s and max 4 per rolling hour`() {
        val g = NudgeGate()
        var t = 0L
        fun fire(): NudgeDecision {
            val first = g.decide(ANGRY, t)
            if (first.nudge) return first
            t += 3000
            return g.decide(ANGRY, t)
        }
        assertTrue(fire().nudge)                       // buzz 1 at ~3 s
        t += 3000
        assertEquals("rate-capped", fire().reason)     // < 120 s later
        t += 120_000
        assertTrue(fire().nudge)                       // buzz 2
        t += 120_000; assertTrue(fire().nudge)         // 3
        t += 120_000; assertTrue(fire().nudge)         // 4
        t += 120_000; assertEquals("rate-capped", fire().reason) // 5th inside the hour
        t += 3_600_000; assertTrue(fire().nudge)       // the hour has rolled
    }
}

class LocalEmotionRunnerTest {
    private class FakeEngine(var next: FloatArray = NEUTRAL, var boom: Boolean = false) : EmotionEngine {
        var calls = 0
        override fun classify(mel: LogMel.Result): FloatArray {
            calls++
            if (boom) error("bad model")
            return next
        }
        override fun close() {}
    }

    private fun speech(n: Int = 96000) = ShortArray(n) { (8000 * sin(it / 9.0)).toInt().toShort() }

    private fun runner(engine: EmotionEngine, buzzes: MutableList<Long>, now: () -> Long = { 0L }): Pair<LocalEmotionRunner, WindowLog> {
        val log = WindowLog(Files.createTempDirectory("emo").toFile())
        return LocalEmotionRunner(engine, log, buzz = { buzzes += now() }, batteryPercent = { 77 }) to log
    }

    @Test fun `silence is skipped without touching the model`() {
        val e = FakeEngine(); val buzzes = mutableListOf<Long>()
        val (r, log) = runner(e, buzzes)
        r.onPcm(ShortArray(96000))
        assertTrue(r.tick(1000) is LocalEmotionRunner.Tick.Skipped)
        assertEquals(0, e.calls); assertEquals(1, r.skippedSilent); assertEquals(0, log.all().size)
    }

    @Test fun `speech is classified, logged with numbers only, and two angry windows buzz once`() {
        val e = FakeEngine(next = ANGRY); val buzzes = mutableListOf<Long>()
        var now = 0L
        val (r, log) = runner(e, buzzes) { now }
        r.onPcm(speech())
        now = 3000; val t1 = r.tick(now) as LocalEmotionRunner.Tick.Classified
        assertFalse(t1.row.nudged); assertEquals("unconfirmed", t1.row.reason)
        now = 6000; val t2 = r.tick(now) as LocalEmotionRunner.Tick.Classified
        assertTrue(t2.row.nudged)
        assertEquals(listOf(6000L), buzzes)
        val rows = log.all()
        assertEquals(2, rows.size)
        assertEquals(listOf(0.7, 0.1, 0.05, 0.05, 0.05, 0.05), rows[1].p)
        assertEquals(77, rows[1].battery)
        assertEquals(2, r.windows); assertEquals(1, r.nudges)
    }

    @Test fun `a model failure is reported, not thrown, and never buzzes`() {
        val e = FakeEngine(boom = true); val buzzes = mutableListOf<Long>()
        val (r, log) = runner(e, buzzes)
        r.onPcm(speech())
        val t = r.tick(1)
        assertTrue(t is LocalEmotionRunner.Tick.Failed)
        assertTrue(buzzes.isEmpty()); assertEquals(0, log.all().size)
    }

    @Test fun `ring keeps only the newest six seconds`() {
        val (r, _) = runner(FakeEngine(), mutableListOf())
        r.onPcm(ShortArray(96000) { 1 })
        r.onPcm(ShortArray(16000) { 2 })
        val s = r.snapshot()
        assertEquals(96000, s.size)
        assertEquals(1, s[0].toInt()); assertEquals(2, s[96000 - 1].toInt()); assertEquals(2, s[80000].toInt())
        assertEquals(1, s[79999].toInt())
    }

    @Test fun `rolling log trims, tracks the sync cursor, and builds the upload body`() {
        val dir = Files.createTempDirectory("emo").toFile()
        val log = WindowLog(dir, maxBytes = 2_000)
        for (i in 1..100) log.append(WindowRow(i * 1000L, listOf(0.1, 0.1, 0.1, 0.1, 0.5, 0.1), 12, 80, false, "neutral"))
        val all = log.all()
        assertTrue(all.size in 10..99, "trimmed, got ${all.size}")
        assertEquals(100_000L, all.last().t)
        assertEquals(all.size, log.unsent(Int.MAX_VALUE).size)
        log.markSent(all[all.size - 3].t)
        assertEquals(listOf(all[all.size - 2].t, all.last().t), log.unsent().map { it.t })
        // A fresh instance on the same folder resumes from the persisted cursor.
        assertEquals(2, WindowLog(dir, 2_000).unsent().size)
        val body = WindowLog.syncBody(log.unsent())
        assertTrue(body.contains("\"rows\"") && body.contains("\"p\":[0.1,0.1,0.1,0.1,0.5,0.1]"))
        assertFalse(body.contains("audio"))
    }

    @Test fun `settings summary is plain language`() {
        val s = LocalEmotionStatus(running = true, text = "Listening", windows = 4, skippedSilent = 2, nudges = 1, lastLabel = "anger", lastProb = 0.71, avgMs = 40, maxMs = 90, unsynced = 3, lastSyncOk = false)
        val t = emotionLocalSummary(s)
        assertTrue(t.contains("4 heard, 2 silent, 1 buzzes") && t.contains("40 ms avg") && t.contains("anger 71%") && t.contains("3 unsynced (offline)"), t)
        assertContentEquals(listOf("Off"), listOf(emotionLocalSummary(LocalEmotionStatus())))
    }
}
