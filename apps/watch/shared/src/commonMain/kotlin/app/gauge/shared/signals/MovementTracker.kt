package app.gauge.shared.signals

import kotlin.math.abs

/**
 * On-device movement/fidget tracker (accelerometer window stddev) with a
 * rolling-median baseline over the wearer's own recent readings.
 *
 * Threshold is baseline + 2x the median absolute deviation (MAD) of the
 * same history — a robust spread measure. When history is perfectly steady
 * (MAD == 0), that would pin the threshold at the baseline itself, so a
 * floor kicks in: `baseline*2 + 0.1`.
 *
 * Honest degradation: [baseline]/[threshold] stay `null` — and
 * [SignalReading.over] stays `false` — until at least
 * [MIN_READINGS_FOR_BASELINE] readings have been observed.
 *
 * Anti-poisoning (mirrors [app.gauge.shared.sentinel.SentinelDetector]'s
 * non-negotiable rule): once a baseline is established, only readings that
 * do NOT come out `over` feed it forward — sustained elevated movement must
 * not drag the baseline up and self-suppress future detection.
 */
class MovementTracker(private val baselineWindow: Int = 30) {
    private val history = ArrayDeque<Double>()

    fun observe(windowStddev: Double): SignalReading {
        val hasEstablishedBaseline = history.size >= MIN_READINGS_FOR_BASELINE

        // Seeding: before a baseline exists, there's nothing yet to poison —
        // accumulate unconditionally.
        if (!hasEstablishedBaseline) {
            history.addLast(windowStddev)
            while (history.size > baselineWindow) history.removeFirst()
        }

        val baseline = if (history.size >= MIN_READINGS_FOR_BASELINE) median(history) else null
        val threshold = baseline?.let { b ->
            val mad = median(ArrayDeque(history.map { abs(it - b) }))
            if (mad == 0.0) b * 2.0 + 0.1 else b + 2.0 * mad
        }
        val over = threshold != null && windowStddev >= threshold

        // Anti-poisoning: once established, only non-over readings feed the
        // baseline forward.
        if (hasEstablishedBaseline && !over) {
            history.addLast(windowStddev)
            while (history.size > baselineWindow) history.removeFirst()
        }

        return SignalReading(
            kind = SignalKind.MOVEMENT,
            value = windowStddev,
            baseline = baseline,
            threshold = threshold,
            over = over,
        )
    }

    companion object {
        private const val MIN_READINGS_FOR_BASELINE = 5
    }
}
