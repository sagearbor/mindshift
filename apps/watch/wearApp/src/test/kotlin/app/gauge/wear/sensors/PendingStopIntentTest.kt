package app.gauge.wear.sensors

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Pins the P4-10 review round-2 fix at the pure-logic layer: HrSource's stop() can only tell
 * [app.gauge.wear.sensors.SensorLifecycleGate] about the stop once the SDK's async unregister
 * confirms — see HrSource's own KDoc trace — which opened a window where a start() arriving
 * during that unconfirmed window was silently dropped (suppressed as "already registered" by the
 * gate, with no record kept, so the eventual confirmation flipped the gate inactive even though
 * the caller's most recent request was "on"). [PendingStopIntent] is the extracted, JVM-testable
 * state machine that remembers and replays that intent; these tests model HrSource's exact call
 * pattern without needing an Android SDK mock.
 */
class PendingStopIntentTest {

    @Test
    fun startArrivedReturnsFalseWhenNoStopIsPending() {
        val p = PendingStopIntent()
        assertFalse(p.startArrived())
    }

    @Test
    fun stopArrivedReturnsFalseWhenNoStopIsPending() {
        val p = PendingStopIntent()
        assertFalse(p.stopArrived())
    }

    @Test
    fun beginStopMarksPending() {
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.isPending)
    }

    // Scenario (a) from the review: stop-pending -> start suppressed -> onSuccess -> start
    // re-issued (final state registered).
    @Test
    fun startArrivedWhilePendingIsReplayedOnConfirm() {
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.startArrived()) // deferred: caller must not register yet
        assertTrue(p.confirm()) // onSuccess: replay requested
        assertFalse(p.isPending)
    }

    // Scenario (b) from the review: stop-pending -> start -> stop again -> onSuccess -> stays
    // unregistered, flag cleared.
    @Test
    fun stopArrivedAfterADeferredStartClearsTheReplayIntent() {
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.startArrived()) // deferred
        assertTrue(p.stopArrived()) // latest intent flips back to "off"; already pending, so the
        // caller must skip issuing a second SDK unregister
        assertFalse(p.confirm()) // no replay — stays unregistered
        assertFalse(p.isPending)
    }

    @Test
    fun confirmWithNoDeferredStartReturnsFalse() {
        val p = PendingStopIntent()
        p.beginStop()
        assertFalse(p.confirm())
        assertFalse(p.isPending)
    }

    @Test
    fun abandonClearsPendingAndDiscardsAnyDeferredStartIntent() {
        // Models onFailure/a synchronous throw issuing the unregister: the SDK call never took
        // effect, so the source is unchanged (still registered) and any deferred start intent is
        // already satisfied — must not linger and cause a stale replay on some later cycle.
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.startArrived())
        p.abandon()
        assertFalse(p.isPending)
    }

    @Test
    fun cyclesDoNotLeakStateBetweenIndependentStops() {
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.startArrived())
        assertTrue(p.confirm()) // first cycle: replay requested and consumed

        // A fresh, independent stop cycle must not see a leftover intent from the first one.
        p.beginStop()
        assertFalse(p.confirm())
    }

    @Test
    fun abandonThenFreshCycleDoesNotLeakState() {
        val p = PendingStopIntent()
        p.beginStop()
        assertTrue(p.startArrived())
        p.abandon()

        p.beginStop()
        assertFalse(p.confirm())
    }
}
