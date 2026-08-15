package app.gauge.shared.capture

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RetroCaptureBufferTest {

    private fun window(seconds: Double = 1.0): ByteArray =
        // 16kHz mono PCM16: 2 bytes/sample, matching MicReader's real window shape.
        ByteArray((16000 * 2 * seconds).toInt())

    @Test
    fun startsEmpty() {
        val b = RetroCaptureBuffer()
        assertEquals(0.0, b.availableSeconds())
        assertEquals(0, b.snapshot(60.0).size)
    }

    @Test
    fun accumulatesAcrossPushes() {
        val b = RetroCaptureBuffer()
        b.push(window(1.0))
        b.push(window(1.0))
        assertEquals(2.0, b.availableSeconds(), absoluteTolerance = 0.001)
    }

    @Test
    fun capsAtFiveMinutes() {
        val b = RetroCaptureBuffer()
        repeat(301) { b.push(window(1.0)) } // 301s pushed, cap is 300s
        assertTrue(b.availableSeconds() <= 300.0)
        assertTrue(b.availableSeconds() > 299.0)
    }

    @Test
    fun snapshotClampsToWhatsActuallyAvailable() {
        val b = RetroCaptureBuffer()
        b.push(window(1.0)) // only 1s recorded
        val snap = b.snapshot(120.0) // asked for 2 minutes
        assertEquals(16000 * 2, snap.size) // gets exactly the 1s that exists, not zero-padded
    }

    @Test
    fun snapshotReturnsTheMostRecentNSeconds() {
        val b = RetroCaptureBuffer()
        val first = ByteArray(16000 * 2) { 1 }
        val second = ByteArray(16000 * 2) { 2 }
        b.push(first)
        b.push(second)
        val snap = b.snapshot(1.0) // want only the last second
        assertEquals(16000 * 2, snap.size)
        assertEquals(2.toByte(), snap[0]) // it's `second`'s data, not `first`'s
    }

    @Test
    fun clearEmptiesTheBuffer() {
        val b = RetroCaptureBuffer()
        b.push(window(1.0))
        b.clear()
        assertEquals(0.0, b.availableSeconds())
    }
}
