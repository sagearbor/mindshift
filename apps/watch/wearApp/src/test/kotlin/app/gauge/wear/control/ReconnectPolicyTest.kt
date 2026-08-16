package app.gauge.wear.control

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * P4-5: pure unit tests for [ReconnectPolicy] — no controller/mic/WS harness needed since this
 * class's whole point is to carry all the "when to attempt / how backoff progresses / when to
 * give up" decision logic in one directly-testable place (mirrors why PulseChainDecision was
 * split out from SentinelService in P4-3).
 */
class ReconnectPolicyTest {

    @Test
    fun backoffProgressesTwoFourEightSixteenThenCapsAtThirty() {
        val policy = ReconnectPolicy()

        assertEquals(2_000L, policy.onFailure(nowMs = 0L))
        assertEquals(4_000L, policy.onFailure(nowMs = 0L))
        assertEquals(8_000L, policy.onFailure(nowMs = 0L))
        assertEquals(16_000L, policy.onFailure(nowMs = 0L))
        assertEquals(30_000L, policy.onFailure(nowMs = 0L), "5th failure must cap at 30s, not 32s")
    }

    @Test
    fun backoffStaysCappedAtThirtyForManyFurtherFailures() {
        val policy = ReconnectPolicy()
        repeat(10) { policy.onFailure(nowMs = 0L) }

        assertEquals(30_000L, policy.onFailure(nowMs = 0L))
        assertEquals(30_000L, policy.onFailure(nowMs = 0L))
    }

    @Test
    fun resetRestartsBackoffLadderFromInitialDelay() {
        val policy = ReconnectPolicy()
        policy.onFailure(nowMs = 0L) // 2s
        policy.onFailure(nowMs = 2_000L) // 4s

        policy.reset()

        assertEquals(2_000L, policy.onFailure(nowMs = 10_000L), "a failure after reset must start the ladder over")
    }

    @Test
    fun resetClearsAnyPendingScheduledAttempt() {
        val policy = ReconnectPolicy()
        policy.onFailure(nowMs = 0L) // arms next attempt at 2000

        policy.reset()

        assertFalse(
            policy.isAttemptDue(streaming = true, nowMs = 999_999L),
            "reset must clear the armed attempt — no failure has been recorded since",
        )
    }

    @Test
    fun isAttemptDueFalseBeforeDelayElapsed() {
        val policy = ReconnectPolicy()
        policy.onFailure(nowMs = 0L) // due at 2000

        assertFalse(policy.isAttemptDue(streaming = true, nowMs = 1_999L))
    }

    @Test
    fun isAttemptDueTrueOnceDelayElapsed() {
        val policy = ReconnectPolicy()
        policy.onFailure(nowMs = 0L) // due at 2000

        assertTrue(policy.isAttemptDue(streaming = true, nowMs = 2_000L))
        assertTrue(policy.isAttemptDue(streaming = true, nowMs = 5_000L), "still due — nothing consumes it but onFailure/reset")
    }

    @Test
    fun isAttemptDueFalseWithoutAnyArmedFailure() {
        val policy = ReconnectPolicy()

        assertFalse(policy.isAttemptDue(streaming = true, nowMs = Long.MAX_VALUE))
    }

    @Test
    fun stopOnNotStreamingEvenWhenAttemptWouldOtherwiseBeDue() {
        val policy = ReconnectPolicy()
        policy.onFailure(nowMs = 0L) // due at 2000

        assertFalse(
            policy.isAttemptDue(streaming = false, nowMs = 10_000L),
            "COOLDOWN/disarm must give up on reconnecting regardless of elapsed backoff time",
        )
    }
}
