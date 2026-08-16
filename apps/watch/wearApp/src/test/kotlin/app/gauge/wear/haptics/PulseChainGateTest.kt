package app.gauge.wear.haptics

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * P4-3 review fix: proves [PulseChainGate]'s structural guarantee that a stale/superseded token
 * can never read as current again — the property SentinelService's `Runnable`s rely on to
 * self-neuter (no fire, no re-arm) regardless of whether `Handler.removeCallbacks` won its own
 * race against the looper. See [PulseChainGate]'s own KDoc for the full incident.
 */
class PulseChainGateTest {

    @Test
    fun freshlyStartedTokenIsCurrent() {
        val gate = PulseChainGate()
        val token = gate.start()
        assertTrue(gate.isCurrent(token))
    }

    @Test
    fun startingAgainSupersedesTheOldToken() {
        val gate = PulseChainGate()
        val first = gate.start()
        val second = gate.start()

        assertNotEquals(first, second)
        assertFalse(gate.isCurrent(first), "a chain-superseded token must no longer be current — no fire, no re-arm")
        assertTrue(gate.isCurrent(second))
    }

    @Test
    fun cancelInvalidatesTheCurrentToken() {
        val gate = PulseChainGate()
        val token = gate.start()
        gate.cancel()

        assertFalse(gate.isCurrent(token), "a cancelled chain's token must die — no fire, no re-arm")
    }

    @Test
    fun cancelWithNoChainRunningIsHarmless() {
        val gate = PulseChainGate()
        gate.cancel() // no start() yet — must not throw, and token 0 (the initial state) must not
        // read as current afterward either.
        assertFalse(gate.isCurrent(0))
    }

    @Test
    fun atMostOneTokenCanEverBeCurrentAtOnce() {
        // The structural guarantee that makes a double-fire impossible by construction, not by
        // timing luck: however many chains have started, only the LATEST token ever validates —
        // simulates a whole sequence of restarts (the once-per-tick churn the original bug had)
        // and checks every earlier token is dead the instant a later one starts.
        val gate = PulseChainGate()
        val tokens = (1..10).map { gate.start() }
        tokens.dropLast(1).forEach { stale ->
            assertFalse(gate.isCurrent(stale), "token $stale must not be current once a later chain started")
        }
        assertTrue(gate.isCurrent(tokens.last()))
    }
}
