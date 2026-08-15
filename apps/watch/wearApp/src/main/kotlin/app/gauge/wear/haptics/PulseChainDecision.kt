package app.gauge.wear.haptics

/**
 * P4-3 review fix: pure decision for whether [app.gauge.wear.service.SentinelService]'s
 * pulse-repeat chain should (re)start, stop, or continue unchanged this tick.
 *
 * Extracted specifically because the *original* bug wasn't just the `Handler` race itself
 * ([PulseChainGate] fixes that structurally) — it was that the service used to cancel and
 * reschedule the chain unconditionally on EVERY tick, even when nothing about the verdict had
 * changed. That meant a fresh chain restart roughly once a second, which is also roughly once a
 * second that the old chain's tail end and the new chain's immediate first fire could land on top
 * of each other. Minimizing restarts to only the ticks where something actually changed (the
 * train just started, the band changed, pulses got turned off, etc.) shrinks that race window to
 * the rare case it should be, on top of [PulseChainGate] making even that rare case safe.
 *
 * [SentinelService.managePulseRepeat] is the only caller — it owns turning [Start] into an actual
 * [PulseChainGate]-guarded `Handler` chain and [Stop] into tearing one down.
 */
sealed interface PulseChainDecision {
    /** No pulse is due this tick (below threshold, not STREAMING, or pulses are off) and no chain
     * was running — nothing to do. */
    data object NoOp : PulseChainDecision

    /** A chain WAS running but must stop now. */
    data object Stop : PulseChainDecision

    /** A chain must (re)start at [pulse]/[effectiveIntervalMs] — either none was running, or the
     * running one's spec no longer matches this tick's verdict. */
    data class Start(val pulse: Pulse, val effectiveIntervalMs: Long) : PulseChainDecision

    /** The already-running chain still matches this tick's verdict exactly — leave its own
     * independently-scheduled cadence alone. */
    data object Continue : PulseChainDecision

    companion object {
        /**
         * [currentChain] is the chain currently running, if any — the exact `(pulse,
         * effectiveIntervalMs)` pair it was last [Start]ed with, `null` if none. [pulse]/
         * [streaming]/[configuredIntervalMs] are this tick's fresh verdict: [configuredIntervalMs]
         * is the raw, unclamped "Pulse speed" preference (`null` means "Off").
         */
        fun decide(
            currentChain: Pair<Pulse, Long>?,
            pulse: Pulse?,
            streaming: Boolean,
            configuredIntervalMs: Long?,
        ): PulseChainDecision {
            if (pulse == null || !streaming || configuredIntervalMs == null) {
                return if (currentChain != null) Stop else NoOp
            }
            val effective = PulseEngine.effectiveIntervalMs(pulse, configuredIntervalMs)
            val key = pulse to effective
            return if (key == currentChain) Continue else Start(pulse, effective)
        }
    }
}
