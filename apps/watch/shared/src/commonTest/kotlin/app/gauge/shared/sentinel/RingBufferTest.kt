package app.gauge.shared.sentinel

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertContentEquals

class RingBufferTest {
    @Test fun keepsOnlyCapacity() {
        val rb = RingBuffer(capacityBytes = 4)
        rb.push(byteArrayOf(1, 2, 3)); rb.push(byteArrayOf(4, 5, 6))
        assertContentEquals(byteArrayOf(3, 4, 5, 6), rb.snapshot())
        assertEquals(4, rb.sizeBytes)
    }
    @Test fun snapshotBeforeFullReturnsAll() {
        val rb = RingBuffer(8); rb.push(byteArrayOf(9, 9))
        assertContentEquals(byteArrayOf(9, 9), rb.snapshot())
    }
    @Test fun oversizedChunkKeepsTail() {
        val rb = RingBuffer(2); rb.push(byteArrayOf(1, 2, 3, 4, 5))
        assertContentEquals(byteArrayOf(4, 5), rb.snapshot())
    }
    @Test fun clearEmpties() {
        val rb = RingBuffer(4); rb.push(byteArrayOf(1)); rb.clear()
        assertEquals(0, rb.sizeBytes); assertContentEquals(byteArrayOf(), rb.snapshot())
    }
}
