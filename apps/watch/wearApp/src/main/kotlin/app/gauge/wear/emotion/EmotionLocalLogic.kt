package app.gauge.wear.emotion

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Pure logic of the ON-WATCH emotion mode: labels, tunables, speech gate, rate cap and nudge gate.
 * Mirrors apps/mobile/src/emotion/config.ts + logic.ts (same defaults, same behaviour) so a watch
 * decision is the decision the phone would have made on the same probabilities. No I/O, no clock of
 * its own (every function takes `nowMs`).
 */
object LocalEmotion {
    /** Model output order (input_contract.json "labels"). */
    val LABELS = listOf("anger", "disgust", "fear", "happiness", "neutral", "sadness")
    const val NEUTRAL = 4
    const val MODEL_FILE = "emotion-tiny-int8.onnx"
    /** Sanity bound on a sideloaded file (tiny is ~10 MB). */
    const val MODEL_MIN_BYTES = 5_000_000L

    const val WINDOW_S = 6
    const val HOP_S = 3
    const val SAMPLE_RATE = 16000

    // Speech-energy gate: skip windows with no speech-like energy (saves battery, avoids noise logs).
    const val GATE_FRAME_MS = 100
    const val GATE_FRAME_DBFS = -45.0
    const val GATE_MIN_ACTIVE_FRACTION = 0.3

    // Nudge (TINY only) - defaults identical to the phone's EMOTION_DEV_DEFAULTS.
    const val NUDGE_MIN_PROB = 0.5
    const val NUDGE_CONFIRM_WINDOWS = 2
    const val MAX_NUDGES_PER_HOUR = 4
    const val MIN_NUDGE_GAP_S = 120
    /** Classes that may buzz (neutral never does). */
    val NUDGE_LABELS = setOf("anger", "disgust", "fear", "happiness", "sadness")
}

data class NudgeConfig(
    val minProb: Double = LocalEmotion.NUDGE_MIN_PROB,
    val confirmWindows: Int = LocalEmotion.NUDGE_CONFIRM_WINDOWS,
    val maxPerHour: Int = LocalEmotion.MAX_NUDGES_PER_HOUR,
    val minGapS: Int = LocalEmotion.MIN_NUDGE_GAP_S,
)

fun topClass(p: FloatArray): Int {
    var best = 0
    for (i in 1 until p.size) if (p[i] > p[best]) best = i
    return best
}

/** A model output is only believable if it is a probability vector. */
fun isProbVector(p: FloatArray): Boolean {
    if (p.size != LocalEmotion.LABELS.size) return false
    var s = 0.0
    for (v in p) {
        if (!v.isFinite() || v < 0f) return false
        s += v
    }
    return abs(s - 1.0) < 1e-3
}

/** True when enough 100 ms frames of the window are at or above -45 dBFS. */
fun windowHasSpeechEnergy(pcm: ShortArray): Boolean {
    val frame = LocalEmotion.SAMPLE_RATE * LocalEmotion.GATE_FRAME_MS / 1000
    if (pcm.size < frame) return false
    val thresh = 10.0.pow(LocalEmotion.GATE_FRAME_DBFS / 20.0) * 32768.0
    var active = 0
    var frames = 0
    var off = 0
    while (off + frame <= pcm.size) {
        var s = 0.0
        for (i in 0 until frame) { val v = pcm[off + i].toDouble(); s += v * v }
        if (sqrt(s / frame) >= thresh) active++
        frames++
        off += frame
    }
    return frames > 0 && active.toDouble() / frames >= LocalEmotion.GATE_MIN_ACTIVE_FRACTION
}

/** At most [max] events in any rolling [windowMs], and [minGapMs] between them. */
class RollingRateCap(private val max: Int, private val windowMs: Long = 3_600_000L, private val minGapMs: Long = 0L) {
    private val times = ArrayDeque<Long>()

    private fun prune(nowMs: Long) { while (times.isNotEmpty() && nowMs - times.first() >= windowMs) times.removeFirst() }

    fun canFire(nowMs: Long): Boolean {
        prune(nowMs)
        if (times.size >= max) return false
        val last = times.lastOrNull()
        return last == null || nowMs - last >= minGapMs
    }

    fun tryFire(nowMs: Long): Boolean {
        if (!canFire(nowMs)) return false
        times.addLast(nowMs)
        return true
    }
}

data class NudgeDecision(val nudge: Boolean, val reason: String)

/**
 * Decides whether TINY's latest window earns a buzz: non-neutral top class in the allowed set,
 * probability >= minProb, the same class on `confirmWindows` consecutive windows, inside the
 * rolling-hour budget and the minimum gap (port of the phone's NudgeGate).
 */
class NudgeGate(private val cfg: NudgeConfig = NudgeConfig()) {
    private var streakLabel = -1
    private var streak = 0
    private val cap = RollingRateCap(cfg.maxPerHour, 3_600_000L, cfg.minGapS * 1000L)

    fun reset() { streakLabel = -1; streak = 0 }

    fun decide(p: FloatArray, nowMs: Long): NudgeDecision {
        val top = topClass(p)
        val label = LocalEmotion.LABELS[top]
        if (top == LocalEmotion.NEUTRAL || label !in LocalEmotion.NUDGE_LABELS || p[top] < cfg.minProb) {
            reset()
            return NudgeDecision(false, if (top == LocalEmotion.NEUTRAL) "neutral" else "below-threshold")
        }
        if (streakLabel == top) streak++ else { streakLabel = top; streak = 1 }
        if (streak < cfg.confirmWindows) return NudgeDecision(false, "unconfirmed")
        if (!cap.tryFire(nowMs)) return NudgeDecision(false, "rate-capped")
        reset()
        return NudgeDecision(true, "")
    }
}
