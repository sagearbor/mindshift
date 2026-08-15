package app.gauge.shared.sentinel

import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Below this window dBFS, we're looking at silence/noise floor, not voiced
 * speech. Single source of truth in this file — mirrors the server's
 * `SILENCE_FLOOR_DBFS` in `server/vectors.py` exactly; wearApp imports this
 * constant rather than redefining it.
 */
const val SILENCE_FLOOR_DBFS = -45.0

/**
 * Pure DSP helpers for on-device audio analysis. Mirrors `server/vectors.py`'s
 * `rms_dbfs` formula exactly so the on-watch sentinel and the server's
 * streaming vector engine agree on what "loud" means.
 */
object Dsp {
    /**
     * `20*log10(rms/32768)` for int16-scaled PCM. `-120.0` for an all-zero
     * (pure silence) window — mirrors the server's convention of a large
     * finite floor rather than `-inf`, which is friendlier to arithmetic
     * (comparisons, averaging) done downstream on-device.
     */
    fun rmsDbfs(pcm: ShortArray): Double {
        if (pcm.isEmpty()) return -120.0
        var sumSquares = 0.0
        for (sample in pcm) {
            val s = sample.toDouble()
            sumSquares += s * s
        }
        val rms = sqrt(sumSquares / pcm.size)
        if (rms <= 0.0) return -120.0
        return 20.0 * log10(rms / 32768.0)
    }
}

/**
 * One window's worth of sentinel analysis.
 *
 * @property dbfs the window's loudness, `20*log10(rms/32768)`.
 * @property dbOverBaseline how far [dbfs] is above the running baseline (0.0
 *   when there is no established baseline yet, i.e. before the first voiced
 *   window; also 0.0, by convention, whenever [voiced] is false).
 * @property voiced whether [dbfs] clears the silence floor.
 * @property triggered whether this window is the second consecutive
 *   over-threshold voiced window (burst debounce).
 */
data class Observation(
    val dbfs: Double,
    val dbOverBaseline: Double,
    val voiced: Boolean,
    val triggered: Boolean,
)

/**
 * On-device always-on detector deciding when the Gauge watch starts
 * capturing.
 *
 * Bias guard (non-negotiable): every decision is measured against the
 * wearer's OWN running baseline — a running median of their last 30 voiced
 * windows' dBFS — never an absolute/universal loudness threshold.
 *
 * Baseline seeding: until 3 voiced windows have been observed, the baseline
 * is pinned to the *first* voiced window's dBFS and [Observation.triggered]
 * is always false. This prevents a cold-start window (before we have any
 * real sense of the wearer's speaking volume) from firing a trigger.
 *
 * Baseline update rule: only voiced windows that do NOT meet the trigger bar
 * feed the running median. Windows that trigger (or would trigger) are
 * deliberately excluded — otherwise sustained yelling drags the baseline up
 * over time and self-suppresses detection, defeating the point of a
 * baseline-relative bias guard.
 *
 * Trigger rule: a window is `triggered` only when it is voiced, clears
 * [triggerDbOverBaseline] over the current baseline, AND is the *second*
 * consecutive window to do so (single spikes don't trigger; the streak
 * resets the moment a qualifying window is followed by a non-qualifying
 * one).
 */
class SentinelDetector(private val triggerDbOverBaseline: Double = 6.0) {
    private val voicedHistory = ArrayDeque<Double>()

    /** The wearer's own running baseline (see class KDoc), or `null` before the first voiced window. */
    var baseline: Double? = null
        private set
    private var overThresholdStreak = 0

    fun observe(window: ShortArray): Observation {
        val dbfs = Dsp.rmsDbfs(window)
        val voiced = dbfs > SILENCE_FLOOR_DBFS

        if (!voiced) {
            overThresholdStreak = 0
            return Observation(dbfs = dbfs, dbOverBaseline = 0.0, voiced = false, triggered = false)
        }

        // Seeding: until we've seen 3 voiced windows, pin the baseline to the
        // very first voiced window and never trigger. The moment the 3rd
        // voiced window lands, immediately recompute the baseline as the
        // median of those 3 seed windows — otherwise the very next (4th)
        // window would still be judged against the stale first-window value
        // instead of the seed median, which is wrong the instant the seed
        // windows differ in loudness (real speech always does).
        if (voicedHistory.size < 3) {
            if (voicedHistory.isEmpty()) baseline = dbfs
            voicedHistory.addLast(dbfs)
            overThresholdStreak = 0
            if (voicedHistory.size >= 3) {
                baseline = median(voicedHistory)
            }
            return Observation(dbfs = dbfs, dbOverBaseline = dbfs - (baseline ?: dbfs), voiced = true, triggered = false)
        }

        val currentBaseline = baseline ?: dbfs
        val overBaseline = dbfs - currentBaseline
        val meetsTriggerBar = overBaseline >= triggerDbOverBaseline

        overThresholdStreak = if (meetsTriggerBar) overThresholdStreak + 1 else 0
        val triggered = overThresholdStreak >= 2

        // Only windows that don't meet the trigger bar feed the baseline —
        // sustained loud speech must not drag the baseline up and
        // self-suppress future detection.
        if (!meetsTriggerBar) {
            voicedHistory.addLast(dbfs)
            while (voicedHistory.size > BASELINE_WINDOW_COUNT) voicedHistory.removeFirst()
            baseline = median(voicedHistory)
        }

        return Observation(dbfs = dbfs, dbOverBaseline = overBaseline, voiced = true, triggered = triggered)
    }

    private fun median(values: ArrayDeque<Double>): Double {
        val sorted = values.sorted()
        val mid = sorted.size / 2
        return if (sorted.size % 2 == 0) (sorted[mid - 1] + sorted[mid]) / 2.0 else sorted[mid]
    }

    companion object {
        private const val BASELINE_WINDOW_COUNT = 30
    }
}
