package app.gauge.shared.signals

/** Whether the mic should be capturing every window, or only the duty-cycled slice of each cycle.
 * [DEEP_DUTY_CYCLED] is the motion-gated deeper tier of [DUTY_CYCLED] — same contract, longer
 * cycle (see [MicDutyCycle]'s KDoc). */
enum class CaptureMode { CONTINUOUS, DUTY_CYCLED, DEEP_DUTY_CYCLED }

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
 * DEEPER tier (motion-gated): while already duty-cycled, if the wrist has ALSO been still —
 * every observed window's movement reading below the wearer's own [MovementTracker] threshold —
 * for [stillThresholdMs] (10 min), capture drops further to the first [deepOnMs] (2 s) of each
 * [deepCycleMs] (30 s) cycle (~6.7%). The deep phase is anchored just as deterministically, at
 * the instant both conditions first held: `max(quietSince + quietThresholdMs,
 * stillSince + stillThresholdMs)`. Because a voiced window resets BOTH clocks, the still run can
 * never outlive the quiet run — deep structurally implies duty-cycled, never a jump straight
 * from continuous.
 *
 * Snap-back is absolute: a single voiced window returns to continuous capture immediately and
 * restarts BOTH clocks. A single moving window (or a window with NO motion reading at all —
 * `still = null`) drops back from deep to the 2s/10s tier and restarts the stillness clock; the
 * quiet run is untouched by movement alone. Everything ambiguous fails OPEN (capture, or at
 * worst the shallower tier): no quiet run yet, the run shorter than its threshold, a clock that
 * stepped backwards, or missing/unavailable motion data — no motion signal NEVER deepens.
 * Missing a conversation is the one failure this feature is not allowed to cause.
 *
 * Caller contract: only consult this while ARMED. STREAMING/COOLDOWN are always continuous — call
 * [reset] on leaving ARMED so a fresh quiet run starts when the sentinel comes back to it. Note
 * that snap-backs are observation-driven: during an OFF phase the mic reads no windows, so a
 * voiced/moving snap-back lands on the first window of the next ON phase — inherent to any duty
 * cycle, and exactly the same bound the 20% tier has always had (just up to [deepCycleMs] now).
 *
 * Pure and clock-free ([nowMs] is always a parameter); not thread-safe (single caller thread).
 */
class MicDutyCycle(
    private val quietThresholdMs: Long = 5 * 60_000L,
    private val cycleMs: Long = 10_000L,
    private val onMs: Long = 2_000L,
    private val stillThresholdMs: Long = STILL_THRESHOLD_MS,
    private val deepCycleMs: Long = 30_000L,
    private val deepOnMs: Long = 2_000L,
) {
    /** When the current unbroken run of unvoiced windows began, or `null` if there isn't one. */
    private var quietSinceMs: Long? = null

    /** When the current unbroken run of BELOW-MOVEMENT-THRESHOLD windows began, or `null` if
     * there isn't one — `null` also whenever the last window carried no honest motion reading
     * (fail open: no motion signal never deepens). */
    private var stillSinceMs: Long? = null

    /**
     * Feed every window the sentinel actually processed. `voiced = true` is the snap-back to
     * continuous — it restarts BOTH clocks. [still] is the window's movement verdict from
     * [MovementTracker]'s existing semantics: `true` = reading below the wearer's own threshold,
     * `false` = at/over it, `null` = no honest reading exists (no accel source, no bucket yet,
     * or no established baseline/threshold). Anything but an affirmative `true` resets the
     * stillness clock — ambiguity fails open, toward MORE capture.
     */
    fun onObservation(voiced: Boolean, nowMs: Long, still: Boolean? = null) {
        if (voiced) {
            quietSinceMs = null
            stillSinceMs = null
            return
        }
        if (quietSinceMs == null) quietSinceMs = nowMs
        if (still != true) {
            // Movement, or missing/unavailable motion data: never deepen, restart the clock.
            stillSinceMs = null
        } else if (stillSinceMs == null) {
            stillSinceMs = nowMs
        }
    }

    fun mode(nowMs: Long): CaptureMode = when {
        deepAnchorMs(nowMs) != null -> CaptureMode.DEEP_DUTY_CYCLED
        anchorMs(nowMs) != null -> CaptureMode.DUTY_CYCLED
        else -> CaptureMode.CONTINUOUS
    }

    /** Whether the mic should read a window right now. Always `true` while [CaptureMode.CONTINUOUS]. */
    fun shouldCapture(nowMs: Long): Boolean {
        deepAnchorMs(nowMs)?.let { return phaseMs(nowMs, it, deepCycleMs) < deepOnMs }
        val anchor = anchorMs(nowMs) ?: return true
        return phaseMs(nowMs, anchor, cycleMs) < onMs
    }

    /** `0` whenever capture is due right now (including the whole continuous mode); otherwise the
     * wait until this cycle's next ON phase begins — what the service sleeps for. */
    fun msUntilNextCapture(nowMs: Long): Long {
        deepAnchorMs(nowMs)?.let { anchor ->
            val phase = phaseMs(nowMs, anchor, deepCycleMs)
            return if (phase < deepOnMs) 0L else deepCycleMs - phase
        }
        val anchor = anchorMs(nowMs) ?: return 0L
        val phase = phaseMs(nowMs, anchor, cycleMs)
        return if (phase < onMs) 0L else cycleMs - phase
    }

    fun reset() {
        quietSinceMs = null
        stillSinceMs = null
    }

    /** The instant duty-cycling began (`quietSince + quietThresholdMs`), or `null` while capture is
     * continuous. A backwards clock produces a negative elapsed, which is never `>=` the threshold
     * — i.e. it fails open. */
    private fun anchorMs(nowMs: Long): Long? {
        val since = quietSinceMs ?: return null
        if (nowMs - since < quietThresholdMs) return null
        return since + quietThresholdMs
    }

    /** The instant the DEEP tier began — the moment both the quiet run and the still run had
     * cleared their thresholds: `max(quietAnchor, stillSince + stillThresholdMs)` — or `null`
     * while the deep tier doesn't apply. Same backwards-clock fail-open as [anchorMs]: a
     * negative still-elapsed never clears the threshold, degrading to the 20% tier (whose own
     * anchor degrades to continuous the same way). */
    private fun deepAnchorMs(nowMs: Long): Long? {
        val quietAnchor = anchorMs(nowMs) ?: return null
        val still = stillSinceMs ?: return null
        if (nowMs - still < stillThresholdMs) return null
        return maxOf(quietAnchor, still + stillThresholdMs)
    }

    private fun phaseMs(nowMs: Long, anchorMs: Long, periodMs: Long): Long {
        val delta = nowMs - anchorMs
        return if (delta < 0) 0L else delta % periodMs
    }

    companion object {
        /** How long the wrist must have been continuously still (on top of an already-engaged
         * quiet duty cycle) before the DEEP tier may engage. */
        const val STILL_THRESHOLD_MS = 10 * 60_000L
    }
}
