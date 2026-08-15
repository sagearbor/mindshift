package app.gauge.shared.signals

import app.gauge.shared.sentinel.Dsp
import app.gauge.shared.sentinel.SILENCE_FLOOR_DBFS

/**
 * Approximate speaking-cadence estimation from raw PCM — no transcription,
 * no speech recognition. A window is split into small chunks; each chunk is
 * classified voiced/unvoiced by a per-window loudness-normalized floor
 * (mirroring [app.gauge.shared.sentinel.SentinelDetector]'s silence floor
 * as the hard lower bound), and the rate of unvoiced→voiced transitions
 * approximates the rate of syllables/bursts per second.
 */
object Cadence {
    /** How far below the window's own ~90th-percentile chunk loudness a chunk may sit and still
     * count as voiced. 12dB ≈ the natural intra-word dynamic range of speech — deep enough to
     * keep unstressed syllables voiced, shallow enough that room noise under loud speech is not. */
    private const val LOUDNESS_DROP_DB = 12.0

    /** A burst only ENDS after this many consecutive unvoiced chunks (2 × 25ms = 50ms): real
     * inter-syllable gaps are ≥~50ms, while single-chunk dips are amplitude flicker. */
    private const val MIN_GAP_CHUNKS = 2

    /**
     * Bursts (unvoiced→voiced transitions) per second across [window].
     *
     * v0.2.4 (loudness-decorrelation fix — this signal used to track volume): two passes.
     * Pass 1 measures every `chunkMs` chunk's dBFS; the voiced floor for THIS window is
     * `max(floorDbfs, p90 − LOUDNESS_DROP_DB)` — it rides the window's own level, so the same
     * envelope spoken louder classifies identically. [floorDbfs] (the shared absolute silence
     * floor — a documented server-parity mirror, do not change it) remains the hard lower bound,
     * so speech quieter than ~−33dBFS behaves byte-identically to the pre-v0.2.4 code.
     * Pass 2 counts bursts with gap hysteresis: a new burst is only counted after
     * [MIN_GAP_CHUNKS] consecutive unvoiced chunks, so ±dB dither around the floor cannot mint
     * bursts. The window still opens "from silence": a window that starts voiced counts one burst.
     *
     * Deliberately NOT a syllable-rate estimator (envelope peak-picking / 2–8Hz modulation energy
     * is future work) — this fix breaks the loudness correlation only.
     */
    fun burstsPerSecond(
        window: ShortArray,
        sampleRate: Int = 16000,
        chunkMs: Int = 25,
        floorDbfs: Double = SILENCE_FLOOR_DBFS,
    ): Double {
        if (window.isEmpty()) return 0.0
        val chunkSize = sampleRate * chunkMs / 1000
        if (chunkSize <= 0) return 0.0

        // Pass 1: per-chunk loudness.
        val chunkDb = ArrayList<Double>()
        var i = 0
        while (i < window.size) {
            val end = minOf(i + chunkSize, window.size)
            chunkDb.add(Dsp.rmsDbfs(window.copyOfRange(i, end)))
            i = end
        }

        val sorted = chunkDb.sorted()
        val p90 = sorted[((sorted.size - 1) * 9) / 10]
        val voicedFloor = maxOf(floorDbfs, p90 - LOUDNESS_DROP_DB)

        // Pass 2: hysteresis burst counting.
        var bursts = 0
        var inBurst = false
        var unvoicedRun = MIN_GAP_CHUNKS // window starts from silence: an opening voiced chunk counts
        for (db in chunkDb) {
            val voiced = db > voicedFloor
            if (voiced) {
                if (!inBurst && unvoicedRun >= MIN_GAP_CHUNKS) bursts++
                inBurst = true
                unvoicedRun = 0
            } else {
                unvoicedRun++
                if (unvoicedRun >= MIN_GAP_CHUNKS) inBurst = false
            }
        }

        val seconds = window.size.toDouble() / sampleRate
        return if (seconds <= 0.0) 0.0 else bursts / seconds
    }
}
