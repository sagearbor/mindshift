package app.gauge.shared.sentinel

/** The sentinel's episode lifecycle states. */
enum class SentinelState { DISARMED, ARMED, STREAMING, COOLDOWN }

/**
 * Drives when the watch starts and stops a capture episode.
 *
 * Two hysteresis pitfalls this implementation is deliberately written to
 * avoid (project history: an earlier server-side state machine shipped both
 * bugs at once):
 *  - STREAMING must NOT drop to COOLDOWN just because the *current* window
 *    is quiet — it holds open as long as any of the last
 *    [quietSecondsToStop] windows was loud. A `voicedLoud` window resets the
 *    consecutive-quiet-window counter to zero; the counter must *reach*
 *    [quietSecondsToStop] (not merely equal it after an off-by-one) on the
 *    same [onWindow] call that flips the state.
 *  - COOLDOWN counts a fixed number of windows — every window counts,
 *    loud or quiet — and only releases to ARMED on the call that completes
 *    the count. Counting starts fresh the instant COOLDOWN is entered: the
 *    window whose quiet-streak triggered the STREAMING → COOLDOWN
 *    transition does not itself count as cooldown window 1.
 *
 * [Mode.SESSION] short-circuits both: [onArm] goes straight to STREAMING,
 * [onWindow] never leaves STREAMING, and only [onDisarm] returns to
 * DISARMED.
 */
class SentinelStateMachine(
    val mode: Mode,
    private val quietSecondsToStop: Int = 30,
    private val cooldownSeconds: Int = 10,
) {
    private val continuous = mode.params().continuous

    var state: SentinelState = SentinelState.DISARMED
        private set

    private var quietWindowCount = 0
    private var cooldownWindowCount = 0

    fun onArm(): SentinelState {
        resetCounters()
        state = if (continuous) SentinelState.STREAMING else SentinelState.ARMED
        return state
    }

    fun onDisarm(): SentinelState {
        resetCounters()
        state = SentinelState.DISARMED
        return state
    }

    fun onWindow(triggered: Boolean, voicedLoud: Boolean): SentinelState {
        when (state) {
            SentinelState.DISARMED -> Unit // ignored — no episode to advance
            SentinelState.ARMED -> if (triggered) {
                quietWindowCount = 0
                state = SentinelState.STREAMING
            }
            SentinelState.STREAMING -> if (!continuous) {
                if (voicedLoud) {
                    quietWindowCount = 0
                } else {
                    quietWindowCount++
                    if (quietWindowCount >= quietSecondsToStop) {
                        cooldownWindowCount = 0
                        state = SentinelState.COOLDOWN
                    }
                }
            }
            SentinelState.COOLDOWN -> {
                cooldownWindowCount++
                if (cooldownWindowCount >= cooldownSeconds) {
                    resetCounters()
                    state = SentinelState.ARMED
                }
            }
        }
        return state
    }

    private fun resetCounters() {
        quietWindowCount = 0
        cooldownWindowCount = 0
    }
}
