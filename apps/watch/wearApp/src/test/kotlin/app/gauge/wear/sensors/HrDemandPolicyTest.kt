package app.gauge.wear.sensors

import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalKind
import kotlin.test.Test
import kotlin.test.assertEquals

class HrDemandPolicyTest {

    private fun eval(
        p: HrDemandPolicy,
        selected: SignalKind = SignalKind.VOLUME,
        sentinel: SentinelState = SentinelState.ARMED,
        subscribed: Boolean = true,
        registered: Boolean = false,
        nowMs: Long = 0L,
    ) = p.evaluate(selected, sentinel, subscribed, registered, nowMs)

    @Test
    fun registersWhenHrIsTheSelectedSignal() {
        val p = HrDemandPolicy()
        assertEquals(HrAction.REGISTER, eval(p, selected = SignalKind.HEART_RATE))
    }

    @Test
    fun noActionWhenHrIsSelectedAndAlreadyRegistered() {
        val p = HrDemandPolicy()
        assertEquals(HrAction.NONE, eval(p, selected = SignalKind.HEART_RATE, registered = true))
    }

    @Test
    fun registersWhileStreamingWithASubscribedHrVectorEvenWithAnotherSignalSelected() {
        val p = HrDemandPolicy()
        assertEquals(
            HrAction.REGISTER,
            eval(p, selected = SignalKind.VOLUME, sentinel = SentinelState.STREAMING, subscribed = true),
        )
    }

    @Test
    fun doesNotRegisterWhileStreamingWhenTheHrVectorIsNotSubscribed() {
        val p = HrDemandPolicy()
        assertEquals(
            HrAction.NONE,
            eval(p, selected = SignalKind.VOLUME, sentinel = SentinelState.STREAMING, subscribed = false),
        )
    }

    @Test
    fun doesNotRegisterWhileMerelyArmedWithANonHrSignal() {
        val p = HrDemandPolicy()
        assertEquals(HrAction.NONE, eval(p, selected = SignalKind.MOVEMENT, sentinel = SentinelState.ARMED))
    }

    @Test
    fun unregistersOnlyAfterTheReleaseGraceElapses() {
        val p = HrDemandPolicy(releaseGraceMs = 30_000L)
        // Demand ends at t=1000 while registered.
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.COOLDOWN, registered = true, nowMs = 1_000L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.COOLDOWN, registered = true, nowMs = 30_999L))
        assertEquals(HrAction.UNREGISTER, eval(p, sentinel = SentinelState.COOLDOWN, registered = true, nowMs = 31_000L))
    }

    @Test
    fun backToBackEpisodesNeverThrashTheSensor() {
        // The whole point of the grace window: STREAMING -> COOLDOWN -> ARMED -> STREAMING inside
        // 30s must not produce a single unregister.
        val p = HrDemandPolicy(releaseGraceMs = 30_000L)
        assertEquals(HrAction.REGISTER, eval(p, sentinel = SentinelState.STREAMING, nowMs = 0L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.COOLDOWN, registered = true, nowMs = 5_000L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 10_000L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.STREAMING, registered = true, nowMs = 15_000L))
        // Grace was cancelled by the returning demand, so the clock restarts from the NEXT drop.
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 20_000L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 49_999L))
        assertEquals(HrAction.UNREGISTER, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 50_000L))
    }

    @Test
    fun noActionWhenNeitherDemandedNorRegistered() {
        val p = HrDemandPolicy()
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = false, nowMs = 99_999L))
    }

    @Test
    fun unregisterIsEmittedOnceThenTheCallerReportsNotRegistered() {
        val p = HrDemandPolicy(releaseGraceMs = 1_000L)
        eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 0L)
        assertEquals(HrAction.UNREGISTER, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 1_000L))
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = false, nowMs = 2_000L))
    }

    @Test
    fun disarmedWithHrSelectedStillDemandsItForThePreviewMeter() {
        // The picker's selection is what drives the preview meter — a DISARMED sentinel with HR
        // selected must not be a reason to drop the sensor.
        val p = HrDemandPolicy()
        assertEquals(
            HrAction.REGISTER,
            eval(p, selected = SignalKind.HEART_RATE, sentinel = SentinelState.DISARMED),
        )
    }

    @Test
    fun backwardsClockKeepsItRegisteredRatherThanUnregisteringEarly() {
        val p = HrDemandPolicy(releaseGraceMs = 30_000L)
        eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 100_000L)
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 0L))
    }

    @Test
    fun resetClearsAPendingGraceClock() {
        val p = HrDemandPolicy(releaseGraceMs = 1_000L)
        eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 0L)
        p.reset()
        // The clock restarts from this call, so 1_000ms after the ORIGINAL drop is still too early.
        assertEquals(HrAction.NONE, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 1_000L))
        assertEquals(HrAction.UNREGISTER, eval(p, sentinel = SentinelState.ARMED, registered = true, nowMs = 2_000L))
    }
}
