package app.gauge.wear.sensors

/**
 * Register/unregister discipline for a single sensor source instance (P4-10).
 *
 * Two jobs, both pure so they're testable off-device:
 * 1. **Dedupe** — [onStart]/[onStop] return `false` for a call that would repeat the state the
 *    source is already in, so the caller skips the SDK call entirely rather than relying on the
 *    SDK to be idempotent (Health Services' `registerMeasureCallback` is fire-and-forget: a
 *    duplicate registration reports nothing at all, which is precisely the blind spot P4-7 hit).
 * 2. **Measure the churn that's left** — the v0.2.2 device pull showed register+unregister pairs
 *    landing inside 1s during the preview<->service handoff. The structural fix is elsewhere
 *    (P4-8's per-signal preview acquisition + P5-3's lazy HR); this counts real transitions inside
 *    a sliding [churnWindowMs] so the *next* device pull can state honestly whether any survived,
 *    instead of us guessing.
 *
 * P4-10 review round 1 (Important): despite the "single caller" contract above, MainActivity's
 * preview loop runs its start/stop cadence on `Dispatchers.Default` while a lifecycle edge can
 * call `previewHrSource?.stop()` from the main thread (coroutine cancellation is cooperative, so
 * the race window is real, not theoretical) — meaning a single [HrSource] instance's gate CAN see
 * concurrent callers in practice. Every public method below is therefore `@Synchronized` (cheap:
 * these calls are infrequent — one per arm/disarm edge, not per tick) and [isActive] is
 * `@Volatile` so a plain read from a second thread (no synchronized block needed to just read it)
 * observes the latest write. This makes each method atomic and visible across threads; it does
 * NOT make compound caller-side sequences (e.g. HrSource's read-isActive-then-decide-to-call-
 * onStop pattern, see that class's [app.gauge.wear.sensors.HrSource.stop] KDoc) atomic across the
 * gap between two separate gate calls — callers that need that additionally use their own state
 * (e.g. `stopPending`).
 */
class SensorLifecycleGate(
    private val churnWindowMs: Long = 5_000L,
    private val churnThreshold: Int = 4,
) {
    @Volatile
    var isActive: Boolean = false
        private set

    private val transitions = ArrayDeque<Long>()

    @Synchronized
    fun onStart(nowMs: Long): Boolean {
        if (isActive) return false
        isActive = true
        recordTransition(nowMs)
        return true
    }

    @Synchronized
    fun onStop(nowMs: Long): Boolean {
        if (!isActive) return false
        isActive = false
        recordTransition(nowMs)
        return true
    }

    @Synchronized
    fun transitionCount(nowMs: Long): Int {
        prune(nowMs)
        return transitions.size
    }

    /** `>=` [churnThreshold] real transitions inside the sliding window — deliberately
     * unconditional on whether those transitions' underlying SDK calls actually succeeded: a
     * registration that keeps failing on every retry is still churn (the caller is hammering the
     * SDK either way), so this must never be gated on register/unregister success. */
    @Synchronized
    fun churnDetected(nowMs: Long): Boolean = transitionCount(nowMs) >= churnThreshold

    @Synchronized
    fun reset() {
        isActive = false
        transitions.clear()
    }

    private fun recordTransition(nowMs: Long) {
        transitions.addLast(nowMs)
        prune(nowMs)
    }

    /** Drops transitions older than [churnWindowMs]. A backwards-stepping clock produces a
     * negative age, which is never `> churnWindowMs` — so nothing is wrongly erased. */
    private fun prune(nowMs: Long) {
        while (transitions.isNotEmpty() && nowMs - transitions.first() > churnWindowMs) {
            transitions.removeFirst()
        }
    }
}
