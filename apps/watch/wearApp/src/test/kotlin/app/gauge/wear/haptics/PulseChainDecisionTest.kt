package app.gauge.wear.haptics

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * P4-3 review fix: [PulseChainDecision.decide] is the pure logic behind SentinelService's
 * restart-avoidance — see its own KDoc for why minimizing chain restarts is itself part of the
 * race fix, not just [PulseChainGate]'s token check.
 */
class PulseChainDecisionTest {

    private val gentle = Pulse(40L, 120)
    private val aggressive = Pulse(70L, 220)

    @Test
    fun noChainAndNoPulseIsNoOp() {
        val decision = PulseChainDecision.decide(
            currentChain = null,
            pulse = null,
            streaming = false,
            configuredIntervalMs = 250L,
        )
        assertEquals(PulseChainDecision.NoOp, decision)
    }

    @Test
    fun freshPulseWithNoRunningChainStarts() {
        val decision = PulseChainDecision.decide(
            currentChain = null,
            pulse = gentle,
            streaming = true,
            configuredIntervalMs = 250L,
        )
        assertEquals(PulseChainDecision.Start(gentle, PulseEngine.effectiveIntervalMs(gentle, 250L)), decision)
    }

    @Test
    fun identicalVerdictToRunningChainContinuesWithoutRestart() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = gentle,
            streaming = true,
            configuredIntervalMs = 250L,
        )
        // The teeth of the restart-avoidance fix: an unchanged verdict must NOT produce a new
        // Start — restarting every tick is exactly what created the original race window.
        assertEquals(PulseChainDecision.Continue, decision)
    }

    @Test
    fun bandChangeWhileRunningRestartsTheChain() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = aggressive, // intensity increased mid-episode
            streaming = true,
            configuredIntervalMs = 250L,
        )
        assertEquals(
            PulseChainDecision.Start(aggressive, PulseEngine.effectiveIntervalMs(aggressive, 250L)),
            decision,
        )
    }

    @Test
    fun intervalPrefChangeWhileRunningRestartsTheChain() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = gentle,
            streaming = true,
            configuredIntervalMs = 500L, // wearer changed "Pulse speed" mid-episode
        )
        assertEquals(PulseChainDecision.Start(gentle, PulseEngine.effectiveIntervalMs(gentle, 500L)), decision)
    }

    @Test
    fun droppingBelowThresholdStopsARunningChain() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = null, // this window's Observation is no longer over threshold
            streaming = true,
            configuredIntervalMs = 250L,
        )
        assertEquals(PulseChainDecision.Stop, decision)
    }

    @Test
    fun leavingStreamingStopsARunningChainEvenIfPulseNonNull() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = gentle,
            streaming = false, // episode ended
            configuredIntervalMs = 250L,
        )
        assertEquals(PulseChainDecision.Stop, decision)
    }

    @Test
    fun turningPulsesOffStopsARunningChain() {
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        val decision = PulseChainDecision.decide(
            currentChain = gentle to effective,
            pulse = gentle,
            streaming = true,
            configuredIntervalMs = null, // wearer switched "Pulse speed" to Off
        )
        assertEquals(PulseChainDecision.Stop, decision)
    }

    @Test
    fun noChainAndNoPulseWhileOffIsNoOp() {
        val decision = PulseChainDecision.decide(
            currentChain = null,
            pulse = null,
            streaming = true,
            configuredIntervalMs = null,
        )
        assertEquals(PulseChainDecision.NoOp, decision)
    }

    @Test
    fun sustainedIdenticalVerdictAcrossManyTicksNeverRestartsTheChain() {
        // Simulates ~10 consecutive ~1s ticks all reporting the same gentle-band verdict — the
        // scenario that used to restart (and therefore race) the physical chain every single
        // tick. With the fix, only the FIRST tick starts a chain; every tick after it just
        // continues the same running chain, so there is only ever one Runnable's worth of
        // scheduling in flight — no window boundary can ever pit two independently-scheduled
        // chains against each other, which is what makes "no two pulses closer than
        // pulseMs + 170ms" hold across a simulated window boundary, not just within one window.
        val effective = PulseEngine.effectiveIntervalMs(gentle, 250L)
        var currentChain: Pair<Pulse, Long>? = null
        var starts = 0
        repeat(10) {
            when (val decision = PulseChainDecision.decide(currentChain, gentle, true, 250L)) {
                is PulseChainDecision.Start -> {
                    starts++
                    currentChain = decision.pulse to decision.effectiveIntervalMs
                }
                PulseChainDecision.Continue -> Unit
                else -> error("unexpected decision: $decision")
            }
        }
        assertEquals(1, starts, "an unchanged verdict across ticks must start the chain exactly once")
        assertEquals(gentle to effective, currentChain)
    }
}
