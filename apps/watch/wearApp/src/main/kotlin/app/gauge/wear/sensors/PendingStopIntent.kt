package app.gauge.wear.sensors

/**
 * Remembers whether the caller's intent flipped back to "on" while a stop was awaiting async SDK
 * confirmation (P4-10 review round 2).
 *
 * Context: [SensorLifecycleGate.onStop] can only be told about a stop once
 * [androidx.health.services.client.MeasureClient.unregisterMeasureCallbackAsync]'s `Future`
 * actually confirms — see [HrSource]'s own KDoc trace for why the gate is deliberately NOT
 * flipped synchronously on `stop()`. That fix (round 1) opened this class's reason to exist: while
 * the gate still reports "active" during that unconfirmed window, a `start()` arriving in the
 * meantime was suppressed by the gate as a plain duplicate, with no record kept anywhere — so once
 * the stop eventually confirmed, HR silently landed off even though the most recent request was
 * "on", and nothing re-armed it.
 *
 * This class is the extracted, pure, JVM-testable state machine for that bookkeeping — kept
 * separate from [SensorLifecycleGate] itself so the gate's already-reviewed, already-locked public
 * contract (its 12 spec tests + round 1's 3 additive tests) stays byte-unmodified; this is
 * HrSource-specific orchestration state, not a generic register/unregister concept every
 * [app.gauge.wear.control.ScalarSource] needs.
 *
 * Every public method is a single atomic "check current state and record the new fact" operation
 * (never a bare read followed by a separate mutation) specifically so a caller cannot race itself
 * across two calls — the same cross-thread exposure that forced [SensorLifecycleGate]'s round-1
 * `@Synchronized` hardening applies here too (MainActivity's preview [HrSource] instance is driven
 * from `Dispatchers.Default` while a lifecycle edge can call `stop()` from the main thread).
 *
 * Not thread-safe in the "multiple stops in flight at once" sense — by design there is only ever
 * one pending stop at a time (a second `stop()` call while one is already pending is suppressed by
 * [stopArrived] itself, matching [HrSource.stop]'s contract).
 */
class PendingStopIntent {
    @Volatile private var pending = false
    @Volatile private var startRequested = false

    /** True while a stop is awaiting SDK confirmation. Read-only convenience (logging/tests) —
     * callers making a real start()/stop() decision must go through [startArrived]/[stopArrived]/
     * [beginStop] instead of racing a separate read against a separate mutation. */
    val isPending: Boolean get() = pending

    /** Call when a fresh stop() is about to issue its SDK unregister (i.e. after [stopArrived] has
     * already confirmed no stop is currently pending). */
    @Synchronized
    fun beginStop() {
        pending = true
        startRequested = false
    }

    /** Call when `start()` arrives. Returns `true` ("a stop is pending — do not register yet, this
     * call already recorded your intent for replay once it confirms") if a stop is currently
     * pending; `false` ("no stop pending, proceed with your normal registration immediately")
     * otherwise. */
    @Synchronized
    fun startArrived(): Boolean {
        if (!pending) return false
        startRequested = true
        return true
    }

    /** Call when `stop()` arrives. Returns `true` ("a stop is already in flight — skip issuing a
     * second SDK unregister call"; this call also clears any deferred start intent, since the
     * caller's latest request is "stop" again, overriding an earlier deferred start) if a stop is
     * already pending; `false` ("no stop pending yet — the caller should [beginStop] and issue the
     * SDK unregister") otherwise. */
    @Synchronized
    fun stopArrived(): Boolean {
        if (!pending) return false
        startRequested = false
        return true
    }

    /** Call from the stop's async success confirmation. Clears the pending window and returns
     * whether a `start()` had been requested during it — the caller should replay its registration
     * exactly when this returns `true`. */
    @Synchronized
    fun confirm(): Boolean {
        pending = false
        val wanted = startRequested
        startRequested = false
        return wanted
    }

    /** Call from the stop's async failure (or a synchronous throw issuing the SDK call): the
     * unregister never actually took effect, so the source is unchanged (still registered) — any
     * deferred start intent is therefore already satisfied and is discarded, not carried forward
     * to a later, unrelated stop cycle. */
    @Synchronized
    fun abandon() {
        pending = false
        startRequested = false
    }
}
