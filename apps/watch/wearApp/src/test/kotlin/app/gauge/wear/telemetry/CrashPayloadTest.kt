package app.gauge.wear.telemetry

import app.gauge.shared.telemetry.DebugRing
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CrashPayloadTest {
    @Test fun crashEventAppendedAfterRingSnapshot() {
        val ring = DebugRing(10); ring.add("info", "svc", "armed", "t1")
        val events = crashEvents(IllegalStateException("kaput"), "OkHttp Dispatcher", ring, "t2")
        assertEquals(2, events.size)
        assertEquals("armed", events[0].message)
        val crash = events[1]
        assertEquals("crash", crash.level)
        assertTrue(crash.message.contains("IllegalStateException"))
        assertTrue(crash.message.contains("OkHttp Dispatcher"))
        assertTrue(crash.stack!!.contains("kaput"))
    }
    @Test fun stackIsTruncatedTo20k() {
        val deep = RuntimeException("x".repeat(50_000))
        val crash = crashEvents(deep, "main", DebugRing(1), "t").last()
        assertTrue(crash.stack!!.length <= 20_000)
    }

    @Test
    fun startBannerNamesVersionAndCode() {
        assertEquals("app start v0.2.4 (code 7)", startBanner("0.2.4", 7))
    }

    @Test
    fun startBannerIsGreppableByPrefix() {
        // The agent-side contract: `curl .../telemetry | grep "app start"` answers "which build is
        // the watch running" — the prefix is load-bearing, pin it.
        assertTrue(startBanner("9.9.9", 123).startsWith("app start v"))
    }
}
