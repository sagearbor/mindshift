package app.gauge.shared.telemetry

import kotlin.test.Test
import kotlin.test.assertEquals

class DebugRingTest {
    @Test
    fun keepsOnlyCapacityNewest() {
        val r = DebugRing(capacity = 2)
        r.add("info", "t", "m1", "ts1")
        r.add("info", "t", "m2", "ts2")
        r.add("info", "t", "m3", "ts3")
        assertEquals(listOf("m2", "m3"), r.snapshot().map { it.message })
    }

    @Test
    fun snapshotIsStableCopy() {
        val r = DebugRing(4)
        r.add("warn", "t", "m", "ts")
        val snap = r.snapshot()
        r.clear()
        assertEquals(1, snap.size)
        assertEquals(0, r.snapshot().size)
    }

    @Test
    fun carriesStackAndLevel() {
        val r = DebugRing(4)
        r.add("crash", "Handler", "boom", "ts", stack = "trace")
        val e = r.snapshot().single()
        assertEquals("crash", e.level)
        assertEquals("trace", e.stack)
    }

    @Test
    fun zeroCapacityRingNeverCrashesAndStaysEmpty() {
        val r = DebugRing(capacity = 0)
        r.add("info", "t", "m1", "ts1")
        assertEquals(emptyList(), r.snapshot())
    }

    @Test
    fun negativeCapacityRingNeverCrashesAndStaysEmpty() {
        val r = DebugRing(capacity = -5)
        r.add("info", "t", "m1", "ts1")
        assertEquals(emptyList(), r.snapshot())
    }
}
