package app.gauge.shared.signals

/**
 * On-device heart-rate tracker with a rolling-median baseline over the
 * wearer's own recent readings (mirrors
 * [app.gauge.shared.sentinel.SentinelDetector]'s bias-guard approach: never
 * an absolute/universal bpm threshold).
 *
 * Honest degradation: [baseline]/[threshold] stay `null` — and [SignalReading.over]
 * stays `false` — until at least [MIN_READINGS_FOR_BASELINE] readings have
 * been observed.
 *
 * Anti-poisoning (mirrors [app.gauge.shared.sentinel.SentinelDetector]'s
 * non-negotiable rule): once a baseline is established, only readings that
 * do NOT come out `over` feed it forward — a sustained elevated heart rate
 * must not drag the baseline up and self-suppress future detection.
 *
 * @property baselineWindow how many of the most recent readings feed the
 *   rolling median.
 * @property overBpm how many bpm above baseline counts as "over".
 */
class HrTracker(private val baselineWindow: Int = 60, private val overBpm: Double = 15.0) {
    private val history = ArrayDeque<Double>()

    fun observe(bpm: Double): SignalReading {
        val hasEstablishedBaseline = history.size >= MIN_READINGS_FOR_BASELINE

        // Seeding: before a baseline exists, there's nothing yet to poison —
        // accumulate unconditionally.
        if (!hasEstablishedBaseline) {
            history.addLast(bpm)
            while (history.size > baselineWindow) history.removeFirst()
        }

        val baseline = if (history.size >= MIN_READINGS_FOR_BASELINE) median(history) else null
        val threshold = baseline?.let { it + overBpm }
        val over = threshold != null && bpm >= threshold

        // Anti-poisoning: once established, only non-over readings feed the
        // baseline forward.
        if (hasEstablishedBaseline && !over) {
            history.addLast(bpm)
            while (history.size > baselineWindow) history.removeFirst()
        }

        return SignalReading(
            kind = SignalKind.HEART_RATE,
            value = bpm,
            baseline = baseline,
            threshold = threshold,
            over = over,
        )
    }

    companion object {
        private const val MIN_READINGS_FOR_BASELINE = 5
    }
}

internal fun median(values: ArrayDeque<Double>): Double {
    val sorted = values.sorted()
    val mid = sorted.size / 2
    return if (sorted.size % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2.0 else sorted[mid]
}
