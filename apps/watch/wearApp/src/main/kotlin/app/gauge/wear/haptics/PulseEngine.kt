package app.gauge.wear.haptics

/**
 * A single haptic tap in the local proportional pulse train (channel A) — see [PulseEngine] KDoc.
 * [durationMs] is how long the tap lasts; [amplitude] is its `VibrationEffect`-scale amplitude
 * (1-255) — [HapticDirector.playPulse] falls back to a fixed value when the device has no
 * amplitude control rather than sending an amplitude the hardware can't honor.
 */
data class Pulse(val durationMs: Long, val amplitude: Int)

/**
 * Pure, platform-free decision engine for Gauge's proportional pulse-train haptic (P4-3 spec):
 * while the wearer's own volume is over their mode's trigger threshold, the watch taps out one
 * short pulse per [intervalMs] on channel A, with duration/amplitude scaling by how far over the
 * threshold the current window is (see [bandFor]). This REPLACES channel-A's old discrete
 * level-pattern haptic ([HapticDirector.onNudge] / [HapticPatterns]) while pulses are enabled
 * ([intervalMs] non-null) — [app.gauge.wear.control.SentinelController] is the one caller,
 * driving [onWindow] once per ~1s mic window with that window's own
 * `Observation.dbOverBaseline` (works identically online and offline, since it never depends on
 * server connectivity — see the spec's own note).
 *
 * [intervalMs] is a live supplier (not a fixed constructor value) so a mid-session change to the
 * "Pulse speed" setting takes effect on the very next window, same live-supplier pattern used
 * throughout this app for prefs-backed values (e.g. [app.gauge.wear.ui.GaugeViewModel]'s own
 * `selectedSignal`). `null` means "Off" — the pulse train is disabled entirely and [onWindow]
 * always returns `null` (the caller falls back to the legacy level-pattern haptic instead).
 *
 * [nowMs] is an injected wall-clock supplier (real `System.currentTimeMillis()` in production,
 * fake in tests) driving the interval gating below. This class holds no Android dependency and is
 * meant to be called at whatever cadence the caller ticks at (nominally once per second here —
 * see [app.gauge.wear.control.SentinelController]'s own KDoc for why the *intra*-window repeat at
 * the full [intervalMs] cadence is the service's own job, not this class's).
 *
 * Interval gating: a pulse only fires once at least the *effective* interval has elapsed since
 * the previous one — `max(intervalMs, currentBand.durationMs + `[SILENCE_FLOOR_MS]`)` — the clamp
 * that guarantees pulses never audibly merge into one another even at a very short configured
 * interval or the longest (max-band, 80ms) pulse. Dropping below threshold resets the gating
 * clock entirely: the very next over-threshold window always fires immediately rather than
 * inheriting a stale cooldown from an earlier, unrelated loud stretch.
 */
class PulseEngine(
    private val intervalMs: () -> Long?,
    private val nowMs: () -> Long,
) {
    private var lastPulseAtMs: Long? = null

    // I2: the effective (clamped) interval that governed the most recently emitted pulse, kept
    // alongside lastPulseAtMs (reset to null in lockstep with it) so a caller can tell not just
    // *when* the pulse train last fired but *how fast* it was running then — see
    // [app.gauge.wear.control.SentinelController]'s "actively covering" check, which needs both
    // to decide whether the train is still plausibly covering channel A right now.
    private var lastEffectiveIntervalMs: Long? = null

    fun onWindow(dbOver: Double, thresholdDb: Double): Pulse? {
        val interval = intervalMs()
        if (interval == null) {
            lastPulseAtMs = null
            lastEffectiveIntervalMs = null
            return null
        }
        if (dbOver < thresholdDb) {
            lastPulseAtMs = null
            lastEffectiveIntervalMs = null
            return null
        }

        val band = bandFor(dbOver - thresholdDb)
        val effectiveIntervalMs = effectiveIntervalMs(band, interval)
        val now = nowMs()
        val last = lastPulseAtMs
        if (last != null && now - last < effectiveIntervalMs) {
            return null
        }
        lastPulseAtMs = now
        lastEffectiveIntervalMs = effectiveIntervalMs
        return band
    }

    /**
     * I2 (channel-A suppression narrowing, plan-owner ratified): wall-clock time of the most
     * recently emitted pulse, or `null` if none yet this "streak" — reset alongside a pulses-off
     * interval or a below-threshold window (see [onWindow]), same as the private field it mirrors.
     * Exposed purely for [app.gauge.wear.control.SentinelController]'s own "is the pulse train
     * actively covering channel A right now" check; this class has no opinion on what a caller
     * does with it.
     */
    fun lastPulseAtMs(): Long? = lastPulseAtMs

    /** The effective (post-[effectiveIntervalMs]-clamp) interval that governed the pulse
     * [lastPulseAtMs] reports, or `null` alongside it. See [lastPulseAtMs] KDoc for why this is
     * exposed. */
    fun lastEffectiveIntervalMs(): Long? = lastEffectiveIntervalMs

    companion object {
        /** Minimum silence after a pulse before the next one may start — see class KDoc's "never
         * merge" guarantee. */
        private const val SILENCE_FLOOR_MS = 170L

        /** Pure band mapping — delegates to [HapticPatterns.pulseBandFor] (v0.2.4: all pattern
         * constants live in HapticPatterns; see its KDoc for the raised floor band). Kept as a
         * seam here so PulseEngine's own tests/KDoc still name it. */
        fun bandFor(dbOverThreshold: Double): Pulse = HapticPatterns.pulseBandFor(dbOverThreshold)

        /**
         * P4-3 review fix: the actual interval a pulse train enforces between successive taps once
         * [pulse]'s own duration is accounted for — `max(configuredIntervalMs, pulse.durationMs +`
         * [SILENCE_FLOOR_MS]`)`. Exposed (not just inlined in [onWindow]) so [app.gauge.wear.
         * service.SentinelService]'s physical repeat loop schedules off the exact same clamped
         * value this engine gates its own verdicts on, rather than re-deriving it — or worse,
         * scheduling off the raw, unclamped preference. With today's 250/500/1000ms preference
         * choices the two happened to already agree (250 == 80 + 170), but that was accidental,
         * not guaranteed: a future faster preset could silently violate the "never merge" guarantee
         * in a caller that didn't share this function.
         */
        fun effectiveIntervalMs(pulse: Pulse, configuredIntervalMs: Long): Long =
            maxOf(configuredIntervalMs, pulse.durationMs + SILENCE_FLOOR_MS)
    }
}
