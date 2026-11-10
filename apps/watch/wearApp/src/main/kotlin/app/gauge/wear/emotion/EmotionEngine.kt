package app.gauge.wear.emotion

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * The swappable inference seam on the watch (same contract as the phone's `EmotionEngine`): a
 * prepared 80x600 log-mel + valid_frames in, six probabilities in [LocalEmotion.LABELS] order out.
 */
interface EmotionEngine {
    /** Six probabilities. Throws on a bad model output - never returns a made-up vector. */
    fun classify(mel: LogMel.Result): FloatArray
    fun close()
}

/** ONNX Runtime (Android) engine for the TINY int8 model. One CPU thread: this runs every 3 s for
 *  hours, so the cheapest schedule wins over the lowest latency. */
class OnnxTinyEngine private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
) : EmotionEngine {

    override fun classify(mel: LogMel.Result): FloatArray {
        require(mel.data.size == LogMel.N_MELS * LogMel.N_FRAMES) { "log-mel has the wrong shape" }
        OnnxTensor.createTensor(
            env, FloatBuffer.wrap(mel.data), longArrayOf(1, LogMel.N_MELS.toLong(), LogMel.N_FRAMES.toLong()),
        ).use { melT ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(mel.validFrames.toLong())), longArrayOf(1)).use { vfT ->
                session.run(mapOf("log_mel" to melT, "valid_frames" to vfT)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    val rows = out[0].value as Array<FloatArray>
                    val p = rows[0].copyOf()
                    check(isProbVector(p)) { "model output is not a ${LocalEmotion.LABELS.size}-class probability vector" }
                    return p
                }
            }
        }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        fun open(modelFile: File): OnnxTinyEngine {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(1)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            return OnnxTinyEngine(env, env.createSession(modelFile.absolutePath, opts))
        }
    }
}

/** Where the (private, never-bundled) model file is looked for, in preference order. */
sealed interface ModelLookup {
    data class Ready(val file: File) : ModelLookup
    data class Missing(val reason: String) : ModelLookup
}

fun findTinyModel(dirs: List<File>): ModelLookup {
    val seen = mutableListOf<String>()
    for (dir in dirs) {
        val f = File(dir, LocalEmotion.MODEL_FILE)
        if (!f.isFile) continue
        if (f.length() < LocalEmotion.MODEL_MIN_BYTES) {
            seen += "${f.path} is only ${f.length()} bytes (truncated copy?)"
            continue
        }
        return ModelLookup.Ready(f)
    }
    return ModelLookup.Missing(
        if (seen.isNotEmpty()) seen.joinToString("; ")
        else "${LocalEmotion.MODEL_FILE} not found (looked in ${dirs.joinToString(", ") { it.path }})",
    )
}
