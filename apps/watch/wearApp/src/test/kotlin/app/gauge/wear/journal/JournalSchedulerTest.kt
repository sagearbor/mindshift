package app.gauge.wear.journal

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

private const val T0 = 1_000_000L
private const val INTERVAL = JOURNAL_UPLOAD_INTERVAL_MS

class JournalSchedulerTest {

    private fun scheduler() = JournalScheduler()

    private fun JournalScheduler.tick(
        now: Long,
        enabled: Boolean = true,
        consent: Boolean = true,
        streaming: Boolean = false,
    ): Double? = onTick(now, journalEnabled = enabled, consentConfirmed = consent, streaming = streaming)

    @Test
    fun firstEnabledTickOnlyAnchorsAndIsNeverDue() {
        val s = scheduler()
        assertNull(s.tick(T0))
    }

    @Test
    fun notDueBeforeTheInterval() {
        val s = scheduler()
        s.tick(T0)
        assertNull(s.tick(T0 + INTERVAL - 1))
    }

    @Test
    fun dueOnceTheIntervalElapsesAndReportsTheElapsedSeconds() {
        val s = scheduler()
        s.tick(T0)
        val due = s.tick(T0 + INTERVAL)
        assertEquals(INTERVAL / 1000.0, assertNotNull(due))
        // The anchor advanced: nothing due again until another full interval.
        assertNull(s.tick(T0 + INTERVAL + 1_000))
        assertNotNull(s.tick(T0 + 2 * INTERVAL))
    }

    @Test
    fun streamingDefersTheUploadAndCatchesUpAfterTheEpisode() {
        // A 60s interval so the caught-up elapsed stretch stays visibly BELOW the 300s ring
        // ceiling (with the real 5-min interval any deferral immediately hits the clamp —
        // that case is pinned separately below).
        val s = JournalScheduler(intervalMs = 60_000L)
        s.tick(T0)
        // Due, but an episode is live — the WS path already has this audio: defer, don't reset.
        assertNull(s.tick(T0 + 60_000, streaming = true))
        assertNull(s.tick(T0 + 75_000, streaming = true))
        // Episode over: the deferred upload fires and covers the WHOLE elapsed stretch.
        val due = s.tick(T0 + 90_000)
        assertEquals(90.0, assertNotNull(due))
    }

    @Test
    fun elapsedIsClampedToTheRingsHonest300sCeiling() {
        val s = scheduler()
        s.tick(T0)
        // A very long deferral (e.g. a long episode): the ring only ever holds 300s — asking
        // for more would be asking for audio that has already fallen off, so the request clamps.
        val due = s.tick(T0 + 20 * 60_000L)
        assertEquals(JOURNAL_MAX_SNAPSHOT_SECONDS, assertNotNull(due))
    }

    @Test
    fun disabledTicksResetTheAnchorSoReEnableStartsAFreshInterval() {
        val s = scheduler()
        s.tick(T0)
        assertNull(s.tick(T0 + INTERVAL, enabled = false)) // toggled off right at due time
        // Back on: no instant fire off the stale anchor — a fresh interval starts now.
        assertNull(s.tick(T0 + INTERVAL + 1_000))
        assertNull(s.tick(T0 + 2 * INTERVAL))
        assertNotNull(s.tick(T0 + INTERVAL + 1_000 + INTERVAL))
    }

    @Test
    fun withoutConsentNothingIsEverDue() {
        val s = scheduler()
        s.tick(T0, consent = false)
        assertNull(s.tick(T0 + 10 * INTERVAL, consent = false))
    }

    @Test
    fun backwardsClockReAnchorsInsteadOfFiring() {
        val s = scheduler()
        s.tick(T0)
        assertNull(s.tick(T0 - 1)) // clock stepped back: fail closed, re-anchor
        assertNull(s.tick(T0 + INTERVAL - 2)) // one interval from the NEW anchor, not the old
        assertNotNull(s.tick(T0 - 1 + INTERVAL))
    }

    @Test
    fun resetForgetsTheAnchor() {
        val s = scheduler()
        s.tick(T0)
        s.reset()
        assertNull(s.tick(T0 + 2 * INTERVAL)) // only re-anchors
    }
}
