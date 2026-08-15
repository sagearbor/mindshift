package app.gauge.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class NudgeStateMachineTest {
    @Test
    fun escalatesAndCoolsDown() {
        val sm = NudgeStateMachine(cooldownS = 20.0)
        assertEquals(2, sm.onLocalLoudness(11.0, t = 1.0))
        assertNull(sm.onLocalLoudness(11.0, t = 2.0))          // unchanged (sustain, refreshes clock to t=2.0)
        assertNull(sm.onLocalLoudness(0.0, t = 10.0))          // within cooldown (10-2=8 <= 20)
        // Note: bumped from the brief's illustrative t=22.0 to t=23.0. The
        // sustain call above refreshes the qualifying clock to t=2.0 (this
        // is required — see sustainedLoudnessRefreshesClock below, mirroring
        // server/tests/test_nudge_policy.py's test_sustained_qualifying_event
        // "Critical 2" fix), so t=22.0 would be an *exact* tie
        // (22-2=20, not >20 under the server's strict-greater-than cooldown
        // rule in nudge_policy.py) and would correctly return null, not a
        // de-escalation. t=23.0 (23-2=21>20) is unambiguous.
        assertEquals(1, sm.onLocalLoudness(0.0, t = 23.0))     // de-escalate
    }

    @Test
    fun sustainedLoudnessRefreshesClock() {
        // Mirrors server/tests/test_nudge_policy.py::test_sustained_qualifying_event:
        // a qualifying observation at the current level refreshes the decay
        // clock, so a subsequent quiet reading within cooldownS of *that*
        // refresh must NOT de-escalate, even though it's well past cooldownS
        // since the original escalation.
        val sm = NudgeStateMachine(cooldownS = 20.0)

        assertEquals(3, sm.onLocalLoudness(20.0, t = 1.0))     // escalate to level 3
        assertNull(sm.onLocalLoudness(20.0, t = 15.0))         // sustain at level 3, refresh clock to t=15
        assertNull(sm.onLocalLoudness(0.0, t = 22.0))          // 22-15=7, not >20: no drop
        assertEquals(3, sm.currentLevel())
    }

    @Test
    fun stepwiseDeescalation2To1To0() {
        // Mirrors server/tests/test_nudge_policy.py::test_stepwise_deescalation:
        // de-escalation drops exactly one level per cooldown expiry, never
        // snaps straight to 0.
        val sm = NudgeStateMachine(cooldownS = 20.0)

        assertEquals(2, sm.onLocalLoudness(11.0, t = 1.0))     // escalate to level 2
        assertEquals(1, sm.onLocalLoudness(0.0, t = 22.0))     // 22-1=21>20: drop to 1, clock rebased to t=22
        assertEquals(0, sm.onLocalLoudness(0.0, t = 43.0))     // 43-22=21>20: drop to 0
    }
}
