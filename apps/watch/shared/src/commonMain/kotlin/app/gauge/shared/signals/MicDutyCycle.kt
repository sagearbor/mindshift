package app.gauge.shared.signals

/** Whether the mic should be capturing every window, or only the duty-cycled slice of each cycle. */
enum class CaptureMode { CONTINUOUS, DUTY_CYCLED }

/**
 * P5-3 (C2): quiet-ambient mic duty cycle. An always-on 16kHz `AudioRecord` is the sentinel's
 * single largest standing power draw, and most of a wearer's day is silence the sentinel has
 * nothing to do with.
 *
 * Ratified rule: after [quietThresholdMs] (5 min) of CONTINUOUS below-threshold windows, capture
 * only the first [onMs] (2 s) of each [cycleMs] (10 s) window. The phase is anchored
 * DETERMINISTICALLY at `quietSince + quietThresholdMs` — not sampled at random — so the schedule
 * is reproducible, reviewable, and identical across instances fed the same inputs.
 *
 * Snap-back is absolute: a single voiced window returns to continuous capture immediately and
 * restarts the 5-minute clock. Everything ambiguous fails OPEN (capture): no quiet run yet, the run
 * shorter than the threshold, or a clock that stepped backwards. Missing a conversation is the one
 * failure this feature is not allowed to cause.
 *
 * Caller contract: only consult this while ARMED. STREAMING/COOLDOWN are always continuous — call
 * [reset] on leaving ARMED so a fresh quiet run starts when the sentinel comes back to it.
 *
 * Pure and clock-free ([nowMs] is always a parameter); not thread-safe (single caller thread).
 */
class MicDutyCycle(
    private val quietThresholdMs: Long = 5 * 60_000L,
    private val cycleMs: Long = 10_000L,
    private val onMs: Long = 2_000L,
) {
    /** When the current unbroken run of unvoiced windows began, or `null` if there isn't one. */
    private var quietSinceMs: Long? = null

    /** Feed every window the sentinel actually processed. `voiced = true` is the snap-back. */
    fun onObservation(voiced: Boolean, nowMs: Long) {
        if (voiced) {
            quietSinceMs = null
            return
        }
        if (quietSinceMs == null) quietSinceMs = nowMs
    }

    fun mode(nowMs: Long): CaptureMode =
        if (anchorMs(nowMs) == null) CaptureMode.CONTINUOUS else CaptureMode.DUTY_CYCLED

    /** Whether the mic should read a window right now. Always `true` while [CaptureMode.CONTINUOUS]. */
    fun shouldCapture(nowMs: Long): Boolean {
        val anchor = anchorMs(nowMs) ?: return true
        return phaseMs(nowMs, anchor) < onMs
    }

    /** `0` whenever capture is due right now (including the whole continuous mode); otherwise the
     * wait until this cycle's next ON phase begins — what the service sleeps for. */
    fun msUntilNextCapture(nowMs: Long): Long {
        val anchor = anchorMs(nowMs) ?: return 0L
        val phase = phaseMs(nowMs, anchor)
        return if (phase < onMs) 0L else cycleMs - phase
    }

    fun reset() {
        quietSinceMs = null
    }

    /** The instant duty-cycling began (`quietSince + quietThresholdMs`), or `null` while capture is
     * continuous. A backwards clock produces a negative elapsed, which is never `>=` the threshold
     * — i.e. it fails open. */
    private fun anchorMs(nowMs: Long): Long? {
        val since = quietSinceMs ?: return null
        if (nowMs - since < quietThresholdMs) return null
        return since + quietThresholdMs
    }

    private fun phaseMs(nowMs: Long, anchorMs: Long): Long {
        val delta = nowMs - anchorMs
        return if (delta < 0) 0L else delta % cycleMs
    }
}
