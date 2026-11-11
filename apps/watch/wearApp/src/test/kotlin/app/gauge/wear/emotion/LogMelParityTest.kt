package app.gauge.wear.emotion

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Parity of the Kotlin log-mel against the SAME golden vector the phone's TS test pins
 * (apps/mobile/src/emotion/__fixtures__, produced by the HF WhisperFeatureExtractor from one MESS
 * clip, CC BY 4.0). If both ports match that vector, the watch feeds the model what the phone does.
 */
class LogMelParityTest {
    private val fixtures = File("../../mobile/src/emotion/__fixtures__")

    private fun leBytes(name: String): ByteBuffer =
        ByteBuffer.wrap(File(fixtures, name).readBytes()).order(ByteOrder.LITTLE_ENDIAN)

    private fun int16(name: String): ShortArray {
        val b = leBytes(name)
        return ShortArray(b.remaining() / 2) { b.short }
    }

    private fun f32(name: String): FloatArray {
        val b = leBytes(name)
        return FloatArray(b.remaining() / 4) { b.float }
    }

    @Test fun `matches the HF WhisperFeatureExtractor reference on clip-04`() {
        assertTrue(fixtures.isDirectory, "fixtures not found at ${fixtures.absolutePath}")
        val audio = LogMel.int16ToFloat32(int16("clip04-16k-int16.bin"))
        val golden = f32("clip04-logmel.f32")
        assertEquals(LogMel.N_MELS * LogMel.N_FRAMES, golden.size)

        val r = LogMel.compute(audio)
        assertEquals(135, r.validFrames) // clip04-meta.json valid_frames
        assertEquals(golden.size, r.data.size)
        var maxAbs = 0.0
        var sumAbs = 0.0
        for (i in golden.indices) {
            val d = abs((r.data[i] - golden[i]).toDouble())
            if (d > maxAbs) maxAbs = d
            sumAbs += d
        }
        // Same bar as the TS test (emotionLogMel.test.ts): well inside 1e-3 on a [-1, ~1.5] scale.
        assertTrue(maxAbs < 1e-3, "max abs diff $maxAbs")
        assertTrue(sumAbs / golden.size < 1e-4, "mean abs diff ${sumAbs / golden.size}")
    }

    @Test fun `valid_frames follows min(300, ceil(n over 320))`() {
        fun tone(n: Int) = FloatArray(n) { 0.1f * sin(it / 7.0).toFloat() }
        assertEquals(50, LogMel.compute(tone(16000)).validFrames)
        assertEquals(51, LogMel.compute(tone(16001)).validFrames)
        assertEquals(300, LogMel.compute(tone(96000)).validFrames)
    }

    @Test fun `rejects what the model contract rejects`() {
        assertNotNull(LogMel.waveformProblem(FloatArray(100)))                 // < 0.1 s
        assertNotNull(LogMel.waveformProblem(FloatArray(16000)))               // silence
        assertNotNull(LogMel.waveformProblem(FloatArray(96001) { 0.1f }))      // > 6 s
        assertNotNull(LogMel.waveformProblem(FloatArray(16000) { Float.NaN })) // non-finite
    }
}
