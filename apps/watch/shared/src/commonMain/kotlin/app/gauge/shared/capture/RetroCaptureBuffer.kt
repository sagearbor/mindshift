package app.gauge.shared.capture

import app.gauge.shared.sentinel.RingBuffer

private const val SAMPLE_RATE_HZ = 16000
private const val BYTES_PER_SAMPLE = 2
private const val BYTES_PER_SECOND = SAMPLE_RATE_HZ * BYTES_PER_SAMPLE
private const val MAX_RETRO_CAPTURE_SECONDS = 300 // 5 min ceiling (product: "2 min default, 5 max")

/**
 * A rolling window of the wearer's OWN recent audio, always at most
 * [MAX_RETRO_CAPTURE_SECONDS] long, so a "save the last N minutes" button has
 * something to save without ever running a second, separate mic session — it's
 * fed the exact same PCM windows [app.gauge.shared.sentinel.SentinelController]
 * already reads for the sentinel (see that class's wiring, Wave C Task 10),
 * regardless of ARMED/STREAMING/COOLDOWN state. It is NEVER fed while fully
 * DISARMED — the mic isn't running then, so there is honestly nothing to retro-
 * save, and [availableSeconds] correctly reports 0 in that case rather than
 * guessing.
 *
 * Deliberately independent of [SentinelController]'s own per-mode `ringBuffer`
 * (5s/10s Battery-Saver/Standard pre-trigger buffer) — that one exists to seed
 * an episode's preamble and is cleared/resized per mode; this one exists purely
 * so the wearer can retroactively grab a clip and never resizes.
 *
 * Pure and platform-free (reused directly by [app.gauge.shared.sentinel.RingBuffer],
 * itself already pure) so it's fully JVM-testable off-device.
 */
class RetroCaptureBuffer {
    private val ring = RingBuffer(capacityBytes = MAX_RETRO_CAPTURE_SECONDS * BYTES_PER_SECOND)

    fun push(pcmBytes: ByteArray) = ring.push(pcmBytes)

    fun availableSeconds(): Double = ring.sizeBytes.toDouble() / BYTES_PER_SECOND

    /** Returns the most recent [seconds] of audio, clamped to whatever is actually
     * available (never zero-padded, never more than what's been pushed) — honest
     * degradation: a wearer who's only been ARMED for 40s asking for "last 2 min"
     * gets 40s back, not silence stitched onto real audio. */
    fun snapshot(seconds: Double): ByteArray {
        val full = ring.snapshot()
        val wantedBytes = (seconds * BYTES_PER_SECOND).toInt().coerceAtMost(full.size)
        if (wantedBytes >= full.size) return full
        return full.copyOfRange(full.size - wantedBytes, full.size)
    }

    fun clear() = ring.clear()
}
