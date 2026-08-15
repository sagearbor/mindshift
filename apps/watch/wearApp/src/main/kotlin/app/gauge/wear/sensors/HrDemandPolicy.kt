package app.gauge.wear.sensors

import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalKind

/** What [HrDemandPolicy] wants the caller to do with the HR measure callback right now. */
enum class HrAction { NONE, REGISTER, UNREGISTER }

/**
 * P5-3 (C1): lazy HR registration. The optical HR sensor is one of the most expensive things this
 * app can leave running, and until now it was registered for the entire armed session regardless
 * of whether anything consumed it.
 *
 * Demand is exactly two things (ratified): HR is the picker-selected signal (the live meter needs
 * it — including while DISARMED, where the preview drives it), OR an episode is streaming and the
 * hr_spike vector is subscribed (the backend needs it).
 *
 * [releaseGraceMs] exists because demand oscillates naturally: STREAMING -> COOLDOWN -> ARMED ->
 * STREAMING inside one conversation would otherwise unregister and re-register the sensor every
 * few seconds — the exact churn P4-10 is separately measuring. Demand returning inside the grace
 * cancels the pending release outright.
 *
 * Pure and clock-free by construction ([nowMs] is always a parameter). Not thread-safe: driven
 * from the service's single loop thread.
 */
class HrDemandPolicy(private val releaseGraceMs: Long = 30_000L) {

    /** When demand dropped while still registered, or `null` when demand is live (or nothing is
     * registered to release). */
    private var demandEndedAtMs: Long? = null

    fun evaluate(
        selectedSignal: SignalKind,
        sentinel: SentinelState,
        hrVectorSubscribed: Boolean,
        registered: Boolean,
        nowMs: Long,
    ): HrAction {
        val demanded = selectedSignal == SignalKind.HEART_RATE ||
            (sentinel == SentinelState.STREAMING && hrVectorSubscribed)

        if (demanded) {
            demandEndedAtMs = null
            return if (registered) HrAction.NONE else HrAction.REGISTER
        }
        if (!registered) {
            demandEndedAtMs = null
            return HrAction.NONE
        }
        val endedAt = demandEndedAtMs ?: nowMs.also { demandEndedAtMs = it }
        // A backwards-stepping clock yields a negative elapsed, which never clears the grace —
        // deliberately fails toward "stay registered": costing battery is recoverable, dropping
        // the wearer's signal mid-conversation is not.
        return if (nowMs - endedAt >= releaseGraceMs) {
            demandEndedAtMs = null
            HrAction.UNREGISTER
        } else {
            HrAction.NONE
        }
    }

    fun reset() {
        demandEndedAtMs = null
    }
}
