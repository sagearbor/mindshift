package app.gauge.wear.journal

/** Journal auto-upload cadence: one retro-buffer snapshot every 5 minutes. */
const val JOURNAL_UPLOAD_INTERVAL_MS: Long = 5 * 60_000L

/** The retro buffer's own hard ceiling ([app.gauge.shared.capture.RetroCaptureBuffer] holds at
 * most 300 s) — a snapshot request is never for more than the ring can honestly hold. */
const val JOURNAL_MAX_SNAPSHOT_SECONDS: Double = 300.0

/**
 * Pure, clock-free scheduler for the journal's periodic retro-capture upload (A/B journal mode).
 * Driven from [app.gauge.wear.service.SentinelService]'s existing ~1s tick loop — this class only
 * ever answers "is an upload due on THIS tick, and for how many seconds of buffer?"; it does no
 * I/O and reads no clock of its own ([nowMs] is always a parameter), mirroring
 * [app.gauge.wear.sensors.HrDemandPolicy] / [app.gauge.shared.signals.MicDutyCycle]'s
 * pure-policy pattern. Not thread-safe: single caller thread (the service's handler thread).
 *
 * Rules (task-ratified):
 *  - Disabled or unconsented → never due, and the anchor resets so re-enabling starts a fresh
 *    interval rather than firing immediately off a stale one.
 *  - The first enabled tick only anchors the clock — the first upload comes one full interval
 *    later (nothing meaningful has accumulated *for the journal* before that).
 *  - While a STREAMING episode is active, a due upload is DEFERRED, not skipped-and-rescheduled:
 *    the live WS path already has that audio, so the journal waits and then snapshots the whole
 *    elapsed stretch (clamped to the ring's honest 300 s ceiling — anything older has already
 *    fallen off the ring and is inherently lost; see [JournalQueue]'s KDoc for the same honesty
 *    note on the retry side).
 *  - A backwards-stepping clock re-anchors (fails closed: never due off a negative elapsed).
 */
class JournalScheduler(private val intervalMs: Long = JOURNAL_UPLOAD_INTERVAL_MS) {

    /** When the current interval started (last upload, or enable time), or `null` when idle. */
    private var anchorMs: Long? = null

    fun reset() {
        anchorMs = null
    }

    /**
     * Returns the number of seconds of retro buffer to snapshot-and-upload on this tick
     * (`min(elapsed, 300)`), or `null` when nothing is due.
     */
    fun onTick(
        nowMs: Long,
        journalEnabled: Boolean,
        consentConfirmed: Boolean,
        streaming: Boolean,
    ): Double? {
        if (!journalEnabled || !consentConfirmed) {
            anchorMs = null
            return null
        }
        val anchor = anchorMs
        if (anchor == null || nowMs < anchor) {
            anchorMs = nowMs
            return null
        }
        val elapsedMs = nowMs - anchor
        if (elapsedMs < intervalMs) return null
        if (streaming) return null // defer: the live episode path already carries this audio
        anchorMs = nowMs
        return (elapsedMs / 1000.0).coerceAtMost(JOURNAL_MAX_SNAPSHOT_SECONDS)
    }
}
