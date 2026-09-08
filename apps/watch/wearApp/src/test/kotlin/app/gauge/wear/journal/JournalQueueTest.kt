package app.gauge.wear.journal

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

private fun snap(tag: Byte) = JournalQueue.Snapshot(
    pcm = byteArrayOf(tag), durationS = 1.0, intervalS = 300.0, capturedAtIso = "2026-08-30T10:00:00Z",
)

class JournalQueueTest {

    @Test
    fun takeOnEmptyIsNull() {
        assertNull(JournalQueue().take())
    }

    @Test
    fun offerThenTakeRoundTripsTheSnapshotAndEmptiesTheQueue() {
        val q = JournalQueue()
        val s = snap(1)
        assertFalse(q.offer(s), "first offer drops nothing")
        assertEquals(1, q.size)
        assertSame(s, q.take())
        assertEquals(0, q.size)
        assertNull(q.take())
    }

    @Test
    fun capacityIsOneAndTheOlderSnapshotIsReportedDropped() {
        val q = JournalQueue()
        q.offer(snap(1))
        val newer = snap(2)
        assertTrue(q.offer(newer), "displacing the pending snapshot is a drop")
        // The NEWEST failed snapshot is the one kept — older audio has already fallen off the
        // 300s ring by the next interval anyway, so keeping it would be pretend-retention.
        assertSame(newer, q.take())
    }
}
