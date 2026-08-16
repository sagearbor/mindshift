package app.gauge.wear.sensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SensorLifecycleGateTest {

    @Test
    fun firstStartRegisters() {
        val g = SensorLifecycleGate()
        assertTrue(g.onStart(0L))
        assertTrue(g.isActive)
    }

    @Test
    fun duplicateStartIsSuppressed() {
        val g = SensorLifecycleGate()
        assertTrue(g.onStart(0L))
        assertFalse(g.onStart(10L))
        assertFalse(g.onStart(20L))
        assertTrue(g.isActive)
    }

    @Test
    fun stopAfterStartUnregisters() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        assertTrue(g.onStop(100L))
        assertFalse(g.isActive)
    }

    @Test
    fun stopWithoutStartIsSuppressed() {
        val g = SensorLifecycleGate()
        assertFalse(g.onStop(0L))
        assertFalse(g.isActive)
    }

    @Test
    fun duplicateStopIsSuppressed() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        assertTrue(g.onStop(100L))
        assertFalse(g.onStop(200L))
    }

    @Test
    fun restartAfterStopRegistersAgain() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        g.onStop(100L)
        assertTrue(g.onStart(200L))
        assertTrue(g.isActive)
    }

    @Test
    fun suppressedCallsDoNotCountAsChurn() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        repeat(10) { g.onStart(it + 1L) } // all duplicates
        assertEquals(1, g.transitionCount(50L))
        assertFalse(g.churnDetected(50L))
    }

    @Test
    fun churnDetectedAfterFourRealTransitionsInsideTheWindow() {
        // Exactly the preview<->service handoff thrash the v0.2.2 device pull showed: two full
        // register/unregister cycles inside one second.
        val g = SensorLifecycleGate()
        g.onStart(0L)
        g.onStop(200L)
        g.onStart(400L)
        g.onStop(900L)
        assertEquals(4, g.transitionCount(900L))
        assertTrue(g.churnDetected(900L))
    }

    @Test
    fun churnNotDetectedWhenTransitionsAreSpacedOut() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        g.onStop(60_000L)
        g.onStart(120_000L)
        g.onStop(180_000L)
        assertEquals(1, g.transitionCount(180_000L))
        assertFalse(g.churnDetected(180_000L))
    }

    @Test
    fun churnWindowSlidesSoOldTransitionsAgeOut() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        g.onStop(200L)
        g.onStart(400L)
        g.onStop(900L)
        assertTrue(g.churnDetected(900L))
        // 5s + 1ms after the oldest: all four have aged out.
        assertEquals(0, g.transitionCount(5_901L))
        assertFalse(g.churnDetected(5_901L))
    }

    @Test
    fun resetClearsActiveStateAndChurnHistory() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        g.onStop(200L)
        g.reset()
        assertFalse(g.isActive)
        assertEquals(0, g.transitionCount(300L))
        assertTrue(g.onStart(300L)) // a fresh start is not treated as a duplicate
    }

    @Test
    fun backwardsClockNeverPrunesTheFuture() {
        // Defensive: a clock stepping backwards must not silently erase the churn history (nor
        // throw). Transitions recorded "in the future" simply stay until time catches up.
        val g = SensorLifecycleGate()
        g.onStart(10_000L)
        g.onStop(10_100L)
        assertEquals(2, g.transitionCount(0L))
    }

    // --- P4-10 review round 1 (Critical fix): pins the contract HrSource.stop() relies on for its
    // async-unregister retry pattern. HrSource does NOT call onStop() synchronously when stop() is
    // invoked — only from the SDK's eventual FutureCallback.onSuccess — so the gate must stay
    // "active" (and duplicate onStart() calls must stay suppressed) for the entire time an
    // unregister is in flight, or has failed and not yet been retried. See HrSource.stop()'s KDoc
    // for the full state-machine trace this backs.

    @Test
    fun gateStaysActiveWhileAConfirmingCallerHasNotYetCalledOnStop() {
        // Models an unregister that is in flight, or that already failed: the caller (HrSource)
        // deliberately withholds onStop() until the SDK confirms success, so isActive must still
        // read true and a concurrent/duplicate start() must still be suppressed.
        val g = SensorLifecycleGate()
        g.onStart(0L)
        assertTrue(g.isActive)
        assertFalse(g.onStart(50L))
        assertTrue(g.isActive)
    }

    @Test
    fun onStopArrivingAfterAWithheldConfirmationStillSucceeds() {
        // The retry path: a first stop() attempt fails (caller never calls onStop()), a later
        // retry succeeds and the caller confirms it then — exactly once, not duplicated.
        val g = SensorLifecycleGate()
        g.onStart(0L)
        // First attempt "fails": no onStop() call here at all.
        assertTrue(g.isActive)
        assertTrue(g.onStop(500L)) // retry succeeds and confirms
        assertFalse(g.isActive)
    }

    @Test
    fun onceConfirmedByOnStopAFurtherStopIsSuppressed() {
        val g = SensorLifecycleGate()
        g.onStart(0L)
        assertTrue(g.onStop(100L)) // onSuccess confirms
        assertFalse(g.onStop(200L)) // a second stop() no-ops
        assertFalse(g.isActive)
    }
}
