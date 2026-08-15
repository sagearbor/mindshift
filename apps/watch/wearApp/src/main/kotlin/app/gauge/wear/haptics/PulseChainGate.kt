package app.gauge.wear.haptics

import java.util.concurrent.atomic.AtomicInteger

/**
 * P4-3 review fix: race-proof "is this still the current pulse-repeat chain" gate for
 * [app.gauge.wear.service.SentinelService]'s Handler-based repeat loop.
 *
 * The incident this exists to prevent: that loop schedules a self-re-posting `Runnable` on a
 * `Handler` from [app.gauge.wear.control.SentinelController]'s handler thread, while the
 * `Runnable` itself executes on a *different* thread (the pulse `Handler`'s own looper). Every
 * tick that supersedes or cancels the running chain calls `Handler.removeCallbacks` to try to
 * drop the old `Runnable` from the queue — but `removeCallbacks` can lose a genuine race against
 * the looper having *already* dequeued that `Runnable` and being about to run it. Without a
 * secondary guard, a `Runnable` that loses that race would still fire: (a) a stray pulse landing
 * right next to the new chain's own first fire, close enough to violate [PulseEngine]'s "never
 * merge" guarantee, and (b) — worse — since it then unconditionally re-posts itself, an orphaned
 * loop that nothing can ever cancel again, because the field that would have let a future
 * `removeCallbacks` find it has already been overwritten to point at the newer chain's `Runnable`.
 *
 * [start] mints a fresh token, invalidating any earlier one, and is called on every chain
 * start/restart, not just the first. [cancel] also mints a fresh token — there is deliberately no
 * separate "cancelled" state, since a cancelled chain and a superseded chain look identical from a
 * stale `Runnable`'s point of view: its captured token simply no longer matches. A `Runnable` must
 * call [isCurrent] with its own captured token immediately before EVERY side effect (vibrating,
 * re-posting) — not just once at the top of `run()` — so it self-neuters no matter which side of
 * a `Handler` race it lands on, independent of whether `removeCallbacks` itself succeeded.
 *
 * Thread-safe via [AtomicInteger]: correct invalidation semantics hold even under concurrent
 * start/cancel calls from more than one thread (the tick loop's handler thread and the service's
 * main-thread lifecycle callbacks both call into this) — a benign lost increment under a genuine
 * simultaneous race still leaves the token different from anything captured before either call,
 * which is all correctness here requires.
 */
class PulseChainGate {
    private val generation = AtomicInteger(0)

    /** Mints and returns a new token, invalidating any token captured before this call. */
    fun start(): Int = generation.incrementAndGet()

    /** Invalidates the current chain (if any) without starting a new one. */
    fun cancel() {
        generation.incrementAndGet()
    }

    /** Whether [token] (captured from an earlier [start]) is still the live chain. */
    fun isCurrent(token: Int): Boolean = generation.get() == token
}
