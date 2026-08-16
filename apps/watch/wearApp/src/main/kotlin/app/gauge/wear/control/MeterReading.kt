package app.gauge.wear.control

import app.gauge.shared.signals.SignalKind

/**
 * Live-meter snapshot for whichever [SignalKind] the wearer currently has selected (Task 10/11:
 * "were you the calmer one?" main-screen redesign's green-below/red-above-threshold meter).
 *
 * Honest degradation: [SentinelController] only ever produces a [MeterReading] when [value] is a
 * real observed reading for [signal] — never a fabricated 0.0 placeholder when a sensor has no
 * reading yet (e.g. HR before Health Services delivers its first sample). [threshold] is `null`
 * whenever the underlying tracker/detector hasn't established a baseline yet, in which case [over]
 * is always `false`.
 */
data class MeterReading(
    val signal: SignalKind,
    val value: Double,
    val threshold: Double?,
    val over: Boolean,
)
