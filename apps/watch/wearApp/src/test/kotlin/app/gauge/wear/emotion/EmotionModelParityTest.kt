package app.gauge.wear.emotion

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import org.junit.Assume.assumeTrue

/**
 * OPTIONAL end-to-end check on a desktop JVM with the REAL tiny weights: the Kotlin log-mel -> ONNX
 * Runtime -> probabilities must reproduce the reference probabilities recorded in clip04-meta.json
 * (what the Python reference produced). The weights are private and never in the repo, so this is
 * skipped unless EMOTION_TINY_MODEL points at emotion-tiny-int8.onnx:
 *
 *   EMOTION_TINY_MODEL=/path/emotion-tiny-int8.onnx ./gradlew :wearApp:testDebugUnitTest --tests '*EmotionModelParityTest*'
 *
 * It also prints desktop latency. That is NOT a watch number - see the plan doc.
 */
class EmotionModelParityTest {
    @Test fun `kotlin pipeline reproduces the reference tiny probabilities`() {
        val path = System.getenv("EMOTION_TINY_MODEL")
        assumeTrue("EMOTION_TINY_MODEL not set; skipping (weights are private)", path != null && File(path).isFile)
        val fx = File("../../mobile/src/emotion/__fixtures__")
        val b = ByteBuffer.wrap(File(fx, "clip04-16k-int16.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        val pcm = ShortArray(b.remaining() / 2) { b.short }
        val mel = LogMel.compute(LogMel.int16ToFloat32(pcm))

        val env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
        val p = env.createSession(path!!, opts).use { s ->
            fun run(): FloatArray =
                OnnxTensor.createTensor(env, FloatBuffer.wrap(mel.data), longArrayOf(1, 80, 600)).use { m ->
                    OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(mel.validFrames.toLong())), longArrayOf(1)).use { v ->
                        s.run(mapOf("log_mel" to m, "valid_frames" to v)).use { (it[0].value as Array<FloatArray>)[0].copyOf() }
                    }
                }
            val out = run()
            val times = (1..20).map { val t0 = System.nanoTime(); run(); (System.nanoTime() - t0) / 1e6 }
            println("DESKTOP-JVM (not a watch) tiny inference ms: median=${times.sorted()[10]} max=${times.max()}")
            val t0 = System.nanoTime(); repeat(5) { LogMel.compute(LogMel.int16ToFloat32(pcm)) }
            println("DESKTOP-JVM (not a watch) log-mel ms per 2.7 s clip: ${(System.nanoTime() - t0) / 5e6}")
            out
        }
        // reference_probs.tiny from clip04-meta.json
        val ref = floatArrayOf(0.15808117f, 0.2578658f, 0.04732236f, 0.010786925f, 0.5199744f, 0.005969392f)
        val maxDiff = ref.indices.maxOf { abs(ref[it] - p[it]) }
        println("tiny probs=${p.toList()} maxDiff=$maxDiff")
        assertTrue(maxDiff < 0.02, "probabilities differ from the reference by $maxDiff")
        assertTrue(topClass(p) == topClass(ref))
    }
}
