package app.gauge.wear.emotion

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** What the Settings row shows about the on-watch mode. Numbers and a status line, nothing sensitive. */
data class LocalEmotionStatus(
    val running: Boolean = false,
    /** Plain-language state: "Off", "Model missing: ...", "Listening", ... */
    val text: String = "Off",
    val windows: Int = 0,
    val skippedSilent: Int = 0,
    val nudges: Int = 0,
    val lastLabel: String? = null,
    val lastProb: Double? = null,
    val avgMs: Int = 0,
    val maxMs: Int = 0,
    val unsynced: Int = 0,
    val lastSyncOk: Boolean? = null,
)

/** Process-wide status the service writes and the Settings row reads. */
object LocalEmotionBus {
    private val _status = MutableStateFlow(LocalEmotionStatus())
    val status: StateFlow<LocalEmotionStatus> = _status.asStateFlow()
    fun update(f: (LocalEmotionStatus) -> LocalEmotionStatus) { _status.value = f(_status.value) }
}

/**
 * The on-watch pipeline minus Android: a 6 s PCM ring, and a [tick] every hop that gates on speech
 * energy, computes the log-mel, runs the [EmotionEngine], applies the [NudgeGate], buzzes and logs.
 * Everything with a side effect is injected, so the whole flow is JVM-tested with a fake engine.
 */
class LocalEmotionRunner(
    private val engine: EmotionEngine,
    private val log: WindowLog,
    private val buzz: () -> Unit,
    private val batteryPercent: () -> Int = { -1 },
    private val gate: NudgeGate = NudgeGate(),
    private val nanoClock: () -> Long = System::nanoTime,
) {
    private val ring = ShortArray(LocalEmotion.WINDOW_S * LocalEmotion.SAMPLE_RATE)
    private var head = 0
    private var filled = 0

    var windows = 0; private set
    var skippedSilent = 0; private set
    var nudges = 0; private set
    private var msSum = 0L
    var maxMs = 0; private set
    val avgMs: Int get() = if (windows == 0) 0 else (msSum / windows).toInt()

    /** Feed raw 16 kHz mono PCM16 from the mic. */
    fun onPcm(buf: ShortArray, n: Int = buf.size) {
        for (i in 0 until n) {
            ring[head] = buf[i]
            head = (head + 1) % ring.size
        }
        filled = minOf(ring.size, filled + n)
    }

    /** The newest [filled] samples (at most 6 s), oldest first. */
    fun snapshot(): ShortArray {
        val out = ShortArray(filled)
        val start = (head - filled + ring.size) % ring.size
        for (i in 0 until filled) out[i] = ring[(start + i) % ring.size]
        return out
    }

    sealed interface Tick {
        data class Skipped(val reason: String) : Tick
        data class Classified(val row: WindowRow) : Tick
        data class Failed(val error: String) : Tick
    }

    /** Classify the latest window now. Never throws: a model failure is a [Tick.Failed], not a crash. */
    fun tick(nowMs: Long): Tick {
        val pcm = snapshot()
        // Needs >= 1.5 s of audio and speech-like energy (same gate as the phone).
        if (pcm.size < LocalEmotion.SAMPLE_RATE * 3 / 2 || !windowHasSpeechEnergy(pcm)) {
            skippedSilent++
            gate.reset()
            return Tick.Skipped("silent")
        }
        return try {
            val t0 = nanoClock()
            val mel = LogMel.compute(LogMel.int16ToFloat32(pcm))
            val p = engine.classify(mel)
            val ms = ((nanoClock() - t0) / 1_000_000L).toInt()
            val d = gate.decide(p, nowMs)
            if (d.nudge) {
                nudges++
                runCatching { buzz() }
            }
            windows++
            msSum += ms
            if (ms > maxMs) maxMs = ms
            val row = WindowRow(
                t = nowMs,
                p = p.map { WindowRow.round4(it) },
                ms = ms,
                battery = batteryPercent(),
                nudged = d.nudge,
                reason = d.reason,
            )
            log.append(row)
            Tick.Classified(row)
        } catch (e: Exception) {
            gate.reset()
            Tick.Failed(e.message ?: e.javaClass.simpleName)
        }
    }
}
