package app.gauge.wear.control

import kotlin.math.min

/**
 * P4-5: pure decision for [SentinelController]'s mid-episode WS reconnect — whether an attempt is
 * due right now, and how the backoff ladder progresses across repeated failures. Extracted as its
 * own stateful, directly-testable class for the same reason [app.gauge.wear.haptics.
 * PulseChainDecision] was split out of [app.gauge.wear.service.SentinelService] in P4-3: all of
 * "when to attempt," "how backoff progresses," and "when to give up" belong together and don't
 * need a controller/mic/WS harness to verify — see [ReconnectPolicyTest] for that coverage.
 * [SentinelController] itself stays thin: it only calls [onFailure] when a WS dies mid-episode,
 * [isAttemptDue] once per STREAMING window to decide whether to actually retry, and [reset] on a
 * confirmed-ish reconnect (or a disarm/fresh-episode boundary) — no backoff arithmetic lives in
 * the controller itself.
 *
 * Progression: 2s, 4s, 8s, 16s, capped at 30s (matches the controller-ratified P4-5 design). Not
 * thread-safe on its own — [SentinelController] guards every call to this class with its own
 * `lock`, the same discipline it already applies to the `ws`/`online` fields this backoff state
 * travels alongside (see [SentinelController]'s own KDoc for why: WS listener callbacks arrive on
 * OkHttp's reader thread, a different thread than the one driving `tick()`).
 */
class ReconnectPolicy(
    private val initialDelayMs: Long = INITIAL_DELAY_MS,
    private val maxDelayMs: Long = MAX_DELAY_MS,
) {
    private var attempt = 0
    private var nextAttemptAtMs: Long? = null

    /**
     * Records a WS failure/drop while STREAMING (the original failure, or a failed reconnect
     * attempt itself) — arms the next attempt at [nowMs] plus this call's backoff delay, advances
     * the ladder for any subsequent failure, and returns the delay just armed (ms), purely so
     * callers can log it without re-deriving it from [isAttemptDue]'s internals.
     */
    fun onFailure(nowMs: Long): Long {
        val delay = min(initialDelayMs shl attempt, maxDelayMs)
        attempt = min(attempt + 1, MAX_SHIFT)
        nextAttemptAtMs = nowMs + delay
        return delay
    }

    /**
     * Resets the backoff ladder to its first rung AND clears any pending scheduled attempt — call
     * after a reconnect that opened without an immediate failure (see [SentinelController]'s own
     * "publish before open()" ordering discipline for why that's the earliest honest signal
     * available, same as the original episode-start optimism), on a fresh episode's own
     * [SentinelController]-internal `startStreaming()`, and on `disarm()` — so a later, unrelated
     * failure starts the ladder over rather than continuing a stale one left by a previous
     * episode or connection attempt.
     */
    fun reset() {
        attempt = 0
        nextAttemptAtMs = null
    }

    /**
     * Whether a reconnect attempt is due right now. `false` whenever [streaming] is `false` — the
     * give-up condition (COOLDOWN/disarm): even a fully-elapsed backoff delay must not fire once
     * the episode itself is no longer STREAMING. `false` also when no failure has been armed yet
     * ([nextAttemptAtMs] `null`, e.g. after [reset] or before any [onFailure] call) or the armed
     * delay hasn't elapsed. Does NOT itself clear the armed attempt — a caller acting on `true`
     * should immediately call [onFailure] (the attempt failed too) or [reset] (it succeeded) to
     * advance state, same as consuming any one-shot signal.
     */
    fun isAttemptDue(streaming: Boolean, nowMs: Long): Boolean {
        if (!streaming) return false
        val at = nextAttemptAtMs ?: return false
        return nowMs >= at
    }

    companion object {
        const val INITIAL_DELAY_MS = 2_000L
        const val MAX_DELAY_MS = 30_000L

        // Guards `initialDelayMs shl attempt` against ever growing unbounded across a
        // pathologically long-lived offline streak — irrelevant to the observable delay (which
        // caps at maxDelayMs long before this), just keeps `attempt` itself bounded.
        private const val MAX_SHIFT = 8
    }
}
