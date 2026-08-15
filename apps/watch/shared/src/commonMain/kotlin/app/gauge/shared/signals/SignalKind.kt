package app.gauge.shared.signals

/** Which on-device signal a [SignalReading] is reporting. */
enum class SignalKind { VOLUME, HEART_RATE, MOVEMENT, SPEAKING_RATE }

/**
 * One observation of an on-device signal, already compared against the
 * wearer's own running baseline.
 *
 * Honest degradation (non-negotiable): until a tracker has seen enough
 * history to establish a baseline, [baseline] and [threshold] are `null`
 * and [over] is `false` — never fabricate a threshold from insufficient
 * data.
 *
 * @property kind which signal this reading is for.
 * @property value the raw observed value (dbfs | bpm | accel-stddev |
 *   bursts/sec, depending on [kind]).
 * @property baseline the wearer's own running baseline, or `null` if not
 *   yet established.
 * @property threshold the over-threshold bar derived from [baseline], or
 *   `null` whenever [baseline] is `null`.
 * @property over whether [value] clears [threshold]; always `false` when
 *   [threshold] is `null`.
 */
data class SignalReading(
    val kind: SignalKind,
    val value: Double,
    val baseline: Double?,
    val threshold: Double?,
    val over: Boolean,
)
