package app.gauge.wear.control

import kotlin.test.Test
import kotlin.test.assertEquals

/** Tier B: the deterministic per-day+account companion session id — see CompanionSession.kt. */
class CompanionSessionIdTest {
    @Test fun deterministicPerUtcDayAndAccount() {
        assertEquals("companion-19700101-alice", companionSessionId("alice", nowMs = 0L))
        // Any two instants inside the same UTC day agree — a whole day of reconnects shares one id.
        assertEquals(
            companionSessionId("alice", nowMs = 1_000L),
            companionSessionId("alice", nowMs = 82_800_000L), // 23:00 the same UTC day
        )
    }

    @Test fun crossesToTheNextUtcDay() {
        assertEquals("companion-19700102-alice", companionSessionId("alice", nowMs = 24L * 3_600_000L))
    }

    @Test fun sanitizesUnsafeAccountIdsForTheWsPath() {
        assertEquals("companion-19700101-ab-c_1", companionSessionId("a/b?-c_1&", nowMs = 0L))
        assertEquals("companion-19700101-anon", companionSessionId("///", nowMs = 0L))
    }
}
