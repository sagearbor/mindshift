package app.gauge.shared.sentinel

import kotlin.test.Test
import kotlin.test.assertEquals

class SentinelStateMachineTest {
    @Test fun standardFlowArmTriggerStreamQuietCooldownRearm() {
        val sm = SentinelStateMachine(Mode.STANDARD, quietSecondsToStop = 3, cooldownSeconds = 2)
        assertEquals(SentinelState.ARMED, sm.onArm())
        assertEquals(SentinelState.ARMED, sm.onWindow(triggered = false, voicedLoud = false))
        assertEquals(SentinelState.STREAMING, sm.onWindow(triggered = true, voicedLoud = true))
        repeat(2) { assertEquals(SentinelState.STREAMING, sm.onWindow(false, false)) } // quiet 2 of 3
        assertEquals(SentinelState.COOLDOWN, sm.onWindow(false, false))                // quiet 3rd
        assertEquals(SentinelState.COOLDOWN, sm.onWindow(false, false))                // cooldown 1
        assertEquals(SentinelState.ARMED, sm.onWindow(false, false))                   // cooldown 2 → armed
    }
    @Test fun loudWindowKeepsStreamOpen() {
        val sm = SentinelStateMachine(Mode.STANDARD, quietSecondsToStop = 2)
        sm.onArm(); sm.onWindow(true, true)
        sm.onWindow(false, false)
        assertEquals(SentinelState.STREAMING, sm.onWindow(false, true)) // loud resets quiet count
        assertEquals(SentinelState.STREAMING, sm.onWindow(false, false))
    }
    @Test fun companionModeStreamsImmediatelyUntilDisarmAndNeverLeavesOnWindows() {
        // Tier B: COMPANION is continuous like SESSION — arm() goes straight to STREAMING (the
        // open-socket state), no window (there are none: no mic) can ever end it, only disarm.
        val sm = SentinelStateMachine(Mode.COMPANION)
        assertEquals(SentinelState.STREAMING, sm.onArm())
        repeat(50) { assertEquals(SentinelState.STREAMING, sm.onWindow(false, false)) }
        assertEquals(SentinelState.DISARMED, sm.onDisarm())
    }
    @Test fun companionModeParamsUseNoMic() {
        assertEquals(false, Mode.COMPANION.params().usesMic)
        assertEquals(true, Mode.COMPANION.params().continuous)
        // Every mic-using mode keeps usesMic = true (the default) — pinned so a future param
        // shuffle can't silently turn a listening mode into a deaf one.
        for (m in listOf(Mode.STANDARD, Mode.BATTERY_SAVER, Mode.SESSION)) {
            assertEquals(true, m.params().usesMic, "mode ${'$'}m must use the mic")
        }
    }
    @Test fun sessionModeStreamsImmediatelyUntilDisarm() {
        val sm = SentinelStateMachine(Mode.SESSION)
        assertEquals(SentinelState.STREAMING, sm.onArm())
        repeat(100) { assertEquals(SentinelState.STREAMING, sm.onWindow(false, false)) }
        assertEquals(SentinelState.DISARMED, sm.onDisarm())
    }
    @Test fun disarmedIgnoresOnWindow() {
        val sm = SentinelStateMachine(Mode.STANDARD)
        assertEquals(SentinelState.DISARMED, sm.state)
        assertEquals(SentinelState.DISARMED, sm.onWindow(triggered = true, voicedLoud = true))
        assertEquals(SentinelState.DISARMED, sm.onWindow(triggered = false, voicedLoud = false))
    }
    @Test fun secondCycleBehavesIdenticallyToFirst() {
        val sm = SentinelStateMachine(Mode.STANDARD, quietSecondsToStop = 3, cooldownSeconds = 2)
        fun cycle() {
            assertEquals(SentinelState.ARMED, sm.onArm())
            assertEquals(SentinelState.STREAMING, sm.onWindow(triggered = true, voicedLoud = true))
            repeat(2) { assertEquals(SentinelState.STREAMING, sm.onWindow(false, false)) }
            assertEquals(SentinelState.COOLDOWN, sm.onWindow(false, false))
            assertEquals(SentinelState.COOLDOWN, sm.onWindow(false, false))
            assertEquals(SentinelState.ARMED, sm.onWindow(false, false))
        }
        cycle()
        cycle()
    }
}
