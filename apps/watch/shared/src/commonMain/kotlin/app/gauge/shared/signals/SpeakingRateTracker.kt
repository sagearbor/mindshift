package app.gauge.shared.signals

/**
 * On-device speaking-rate (bursts/sec, from [Cadence]) tracker with a
 * rolling-median baseline over the wearer's own recent VOICED readings.
 *
 * Silent windows (`burstsPerSec <= 0`) are still returned as readings but
 * deliberately do NOT feed the baseline history — otherwise stretches of
 * silence would drag the wearer's speaking-rate baseline toward zero.
 *
 * Honest degradation: [baseline]/[threshold] stay `null` — and
 * [SignalReading.over] stays `false` — until at least
 * [MIN_READINGS_FOR_BASELINE] voiced readings have been observed.
 *
 * Anti-poisoning (mirrors [app.gauge.shared.sentinel.SentinelDetector]'s
 * non-negotiable rule): once a baseline is established, only voiced
 * readings that do NOT come out `over` feed it forward — a sustained
 * elevated speaking rate must not drag the baseline up and self-suppress
 * future detection.
 */
class SpeakingRateTracker(
    private val baselineWindow: Int = 30,
    private val overBurstsPerSec: Double = 1.5,
) {
    private val history = ArrayDeque<Double>()

    fun observe(burstsPerSec: Double): SignalReading {
        val hasEstablishedBaseline = history.size >= MIN_READINGS_FOR_BASELINE
        val voiced = burstsPerSec > 0.0

        // Seeding: before a baseline exists, there's nothing yet to poison —
        // accumulate unconditionally (still voiced-only).
        if (!hasEstablishedBaseline && voiced) {
            history.addLast(burstsPerSec)
            while (history.size > baselineWindow) history.removeFirst()
        }

        val baseline = if (history.size >= MIN_READINGS_FOR_BASELINE) median(history) else null
        val threshold = baseline?.let { it + overBurstsPerSec }
        val over = threshold != null && burstsPerSec >= threshold

        // Anti-poisoning: once established, only non-over voiced readings
        // feed the baseline forward.
        if (hasEstablishedBaseline && voiced && !over) {
            history.addLast(burstsPerSec)
            while (history.size > baselineWindow) history.removeFirst()
        }

        return SignalReading(
            kind = SignalKind.SPEAKING_RATE,
            value = burstsPerSec,
            baseline = baseline,
            threshold = threshold,
            over = over,
        )
    }

    companion object {
        private const val MIN_READINGS_FOR_BASELINE = 5
    }
}
