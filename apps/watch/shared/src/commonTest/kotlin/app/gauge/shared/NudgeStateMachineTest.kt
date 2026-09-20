package app.gauge.shared

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class NudgeStateMachineTest {
    // hold-3s: these three cases pin the COOLDOWN/SUSTAIN machinery, which is unchanged. They are
    // built at holdS = 1.0 — the pre-2026-09-20 gate — so they keep asserting exactly what they
    // were written for instead of quietly re-testing the new hysteresis. The hold itself is pinned
    // by NudgeStateMachineHoldTest below and by nudge_policy.json's `hold_*` cases.
    private fun legacy(cooldownS: Double = 20.0) = NudgeStateMachine(cooldownS = cooldownS, holdS = 1.0)

    @Test
    fun escalatesAndCoolsDown() {
        val sm = legacy()
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
        val sm = legacy()

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
        val sm = legacy()

        assertEquals(2, sm.onLocalLoudness(11.0, t = 1.0))     // escalate to level 2
        assertEquals(1, sm.onLocalLoudness(0.0, t = 22.0))     // 22-1=21>20: drop to 1, clock rebased to t=22
        assertEquals(0, sm.onLocalLoudness(0.0, t = 43.0))     // 43-22=21>20: drop to 0
    }
}

/**
 * Hold-N hysteresis on the loudness ladder (2026-09-20) — the SHIPPED default, not a knob setting.
 *
 * The behaviour and the numbers behind it are in [LoudnessHold]'s KDoc; these cases are the wrist's
 * copy of what `server/tests/fixtures/policy_vectors/nudge_policy.json`'s `hold_*` cases pin for
 * all three runtimes, plus the seconds-vs-calls rule that file cannot express (its steps are all
 * one window long).
 */
class NudgeStateMachineHoldTest {

    @Test
    fun twoLoudWindowsThenQuietNeverBuzzes() {
        val sm = NudgeStateMachine()
        assertNull(sm.onLocalLoudness(7.0, t = 1.0))
        assertNull(sm.onLocalLoudness(7.0, t = 2.0))
        assertNull(sm.onLocalLoudness(0.0, t = 3.0))           // the run resets here
        assertNull(sm.onLocalLoudness(7.0, t = 4.0))
        assertNull(sm.onLocalLoudness(7.0, t = 5.0))
        assertEquals(0, sm.currentLevel())
    }

    @Test
    fun theThirdConsecutiveWindowIsTheFirstThatMayClimb() {
        val sm = NudgeStateMachine()
        assertNull(sm.onLocalLoudness(7.0, t = 1.0))
        assertNull(sm.onLocalLoudness(7.0, t = 2.0))
        assertEquals(1, sm.onLocalLoudness(7.0, t = 3.0))
        // Once the run is long enough the ladder is its old self: the next window at +11 dB climbs
        // with no further wait.
        assertEquals(2, sm.onLocalLoudness(11.0, t = 4.0))
    }

    @Test
    fun aQuietWindowMakesTheNextRungEarnTheHoldAgain() {
        val sm = NudgeStateMachine()
        sm.onLocalLoudness(7.0, t = 1.0)
        sm.onLocalLoudness(7.0, t = 2.0)
        assertEquals(1, sm.onLocalLoudness(7.0, t = 3.0))
        assertNull(sm.onLocalLoudness(0.0, t = 4.0))           // breaks the run; 4-3=1, far inside the cooldown
        assertNull(sm.onLocalLoudness(11.0, t = 5.0))
        assertNull(sm.onLocalLoudness(11.0, t = 6.0))
        assertEquals(2, sm.onLocalLoudness(11.0, t = 7.0))
    }

    @Test
    fun aGatedOutWindowLetsTheLevelDecayExactlyAsAQuietOneWould() {
        // The gate only ever REMOVES an escalation. It must not refresh the sustain clock, or a
        // burst of gated-out loud windows would hold a stale level up forever.
        val sm = NudgeStateMachine()
        sm.onLocalLoudness(7.0, t = 1.0)
        sm.onLocalLoudness(7.0, t = 2.0)
        assertEquals(1, sm.onLocalLoudness(7.0, t = 3.0))      // clock at t=3
        assertNull(sm.onLocalLoudness(0.0, t = 4.0))           // run broken
        assertNull(sm.onLocalLoudness(9.0, t = 20.0))          // loud, gated out (run of 1)
        assertEquals(0, sm.onLocalLoudness(9.0, t = 24.0))     // 24-3=21>20: decays, as if it had been quiet
    }

    @Test
    fun holdOfOneOrZeroIsThePreHysteresisLadder() {
        for (holdS in listOf(0.0, 1.0)) {
            val sm = NudgeStateMachine(holdS = holdS)
            assertEquals(1, sm.onLocalLoudness(7.0, t = 1.0), "holdS=$holdS")
            assertNull(sm.onLocalLoudness(7.0, t = 2.0), "holdS=$holdS")
            assertEquals(2, sm.onLocalLoudness(11.0, t = 3.0), "holdS=$holdS")
        }
    }

    @Test
    fun theHoldCountsSecondsOfAudioNotCalls() {
        // The watch's own sentinel calls once per 1 s window, but a turn-driven caller (the phone's
        // fast loop, watch/relay.py) observes once per TURN — so one 4 s loud turn is four windows
        // of hold, and a sub-second one is still worth a whole window.
        assertTrue(LoudnessHold(HEAT_HOLD_S).observe(true, observedS = 4.0))
        val short = LoudnessHold(HEAT_HOLD_S)
        assertTrue(!short.observe(true, observedS = 0.2))
        assertTrue(!short.observe(true, observedS = 0.2))
        assertTrue(short.observe(true, observedS = 0.2))
    }

    @Test
    fun peekDoesNotAdvanceTheRun() {
        // The phone's instant haptic tier asks before the same turn's policy tick; asking must be
        // free, or the turn would count twice.
        val hold = LoudnessHold(HEAT_HOLD_S)
        assertTrue(hold.peek(true, observedS = 5.0))
        assertTrue(hold.peek(true, observedS = 5.0))
        assertEquals(0.0, hold.runSeconds())
    }

    @Test
    fun theShippedHoldIsThreeSeconds() {
        // The number is load-bearing: it is what the 32 h corpus replay was measured at
        // (server/tests/test_conversation_dose.py). Changing it means re-running that gate.
        assertEquals(3.0, HEAT_HOLD_S)
    }
}
