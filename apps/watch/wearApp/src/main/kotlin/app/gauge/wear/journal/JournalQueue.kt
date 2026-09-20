package app.gauge.wear.journal

/**
 * Capacity-ONE retry queue for failed journal uploads — deliberately tiny and honest about it:
 * the retro ring holds at most 300 s, so by the time the *next* 5-minute interval comes around,
 * any snapshot older than the one most recently taken describes audio the ring has already
 * overwritten anyway. Keeping exactly the last failed snapshot (and counting everything older as
 * dropped — see [offer]'s return value, surfaced as `journal_drops` in telemetry) is the whole
 * retention story; there is no hidden multi-item buffer pretending otherwise.
 *
 * Pure Kotlin, no Android imports. Thread-safe ([take] runs on the service's loop thread while
 * [offer] runs on the upload coroutine's IO thread — a slow upload straddling the next tick must
 * not corrupt the slot).
 */
class JournalQueue {

    private val lock = Any()

    /** One snapshot's worth of upload work: the PCM16 bytes plus the metadata the captures API
     * needs. [intervalS] is the interval the snapshot covers (what the `interval_s` label
     * reports); [durationS] is the audio actually in [pcm] (≤ [intervalS] — the ring clamps
     * honestly, and mic duty-cycling can thin a quiet stretch further). */
    class Snapshot(
        val pcm: ByteArray,
        val durationS: Double,
        val intervalS: Double,
        val capturedAtIso: String,
    )

    private var pending: Snapshot? = null

    val size: Int get() = synchronized(lock) { if (pending != null) 1 else 0 }

    /** Stores [snapshot] as THE pending retry, returning `true` when an older pending snapshot
     * was dropped to make room (the caller counts that as a drop). */
    fun offer(snapshot: Snapshot): Boolean = synchronized(lock) {
        val dropped = pending != null
        pending = snapshot
        dropped
    }

    /** Removes and returns the pending snapshot, or `null` when there is none. */
    fun take(): Snapshot? = synchronized(lock) {
        val s = pending
        pending = null
        s
    }
}
