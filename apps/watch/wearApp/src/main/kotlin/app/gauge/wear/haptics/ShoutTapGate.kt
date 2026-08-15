package app.gauge.wear.haptics

/**
 * v0.2.4: rate limiter for the ARMED shout-tap — the single band-1 pulse that fires when the
 * wearer goes loud while the sentinel is merely armed (no episode open yet), so the product is
 * *felt* before the two-consecutive-loud-window trigger ever fires.
 *
 * Contract: the caller invokes [onLoudWindow] ONLY for windows that already passed the exact
 * baseline-relative loud test the trigger path uses (`voicedLoud` in SentinelController's
 * processWindow) — this class decides nothing about loudness, only spacing. `true` means "tap
 * now" and restarts the clock; a suppressed call does NOT restart it (otherwise sustained
 * near-loud speech could postpone taps forever).
 *
 * Fail direction, ratified: SILENCE. A backwards-stepping clock yields a negative elapsed, which
 * is `< minIntervalMs`, so the tap is suppressed until real time passes the last tap again — the
 * deliberate opposite of MicDutyCycle's fail-open, because a missed nicety tap costs nothing
 * while a tap storm costs the wearer's trust in the channel.
 *
 * Pure and clock-free ([nowMs] is always a parameter); not thread-safe (single caller — the
 * controller's core thread).
 */
class ShoutTapGate(private val minIntervalMs: Long = 2_000L) {
    private var lastTapAtMs: Long? = null

    fun onLoudWindow(nowMs: Long): Boolean {
        val last = lastTapAtMs
        if (last != null && nowMs - last < minIntervalMs) return false
        lastTapAtMs = nowMs
        return true
    }

    fun reset() {
        lastTapAtMs = null
    }
}
