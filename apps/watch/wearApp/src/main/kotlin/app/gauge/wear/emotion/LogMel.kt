package app.gauge.wear.emotion

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Whisper log-mel frontend for the on-watch TINY emotion model. A line-for-line Kotlin port of
 * apps/mobile/src/emotion/logMel.ts (which is pinned against the HF `WhisperFeatureExtractor` on a
 * golden vector); LogMelParityTest pins THIS port against the very same golden file, so phone and
 * watch feed the model identical input. Change one, change both.
 *
 * Contract (eval/codex/input_contract.json): 16 kHz mono float32, NO gain normalisation; zero-pad to
 * 96000 samples; periodic Hann(400), hop 160, 200 reflected samples each side; power spectrum; 80
 * Slaney mel filters 0..8000 Hz (Slaney-normalised); log10(max(mel,1e-10)); drop the last frame
 * (601 -> 600); max(x, max(x)-8); (x+4)/4; valid_frames = min(300, ceil(n/320)).
 *
 * Watch-specific cost cuts that do not change the numbers: the mel matrix is stored sparse (each
 * triangle touches only a few of the 201 bins, the rest are exact zeros), and frames lying wholly in
 * the zero padding are not transformed (their value is exactly log10(1e-10) = -10).
 */
object LogMel {
    const val SAMPLE_RATE = 16000
    const val N_FFT = 400
    const val HOP = 160
    const val N_MELS = 80
    const val PAD_SAMPLES = 96000
    const val N_FRAMES = 600
    const val MAX_VALID_FRAMES = 300
    private const val N_FREQ = N_FFT / 2 + 1
    private const val SILENCE_RMS = 1e-5

    class Result(
        /** Row-major [80][600] float32 = the model's `log_mel` input (batch dim added by the caller). */
        val data: FloatArray,
        val validFrames: Int,
    )

    /** Why a waveform must not be fed to the model, or null when it is fine. */
    fun waveformProblem(x: FloatArray): String? {
        if (x.size < SAMPLE_RATE / 10) return "shorter than 0.1 s"
        if (x.size > PAD_SAMPLES) return "longer than 6 s (segment explicitly)"
        var sum = 0.0
        for (v in x) {
            if (!v.isFinite()) return "non-finite samples"
            sum += v.toDouble() * v
        }
        if (sqrt(sum / x.size) < SILENCE_RMS) return "silence (below -100 dBFS)"
        return null
    }

    fun int16ToFloat32(pcm: ShortArray): FloatArray = FloatArray(pcm.size) { pcm[it] / 32768f }

    // ---- tables ---------------------------------------------------------------------------------

    private fun hzToMel(f: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logstep = ln(6.4) / 27.0
        return if (f >= minLogHz) minLogMel + ln(f / minLogHz) / logstep else f / fSp
    }

    private fun melToHz(m: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logstep = ln(6.4) / 27.0
        return if (m >= minLogMel) minLogHz * exp(logstep * (m - minLogMel)) else fSp * m
    }

    /** Sparse Slaney mel filters: for filter m, bins [start[m], start[m]+weights[m].size). */
    class MelBank(val start: IntArray, val weights: Array<DoubleArray>)

    /** Dense [mel][freq] filters, exactly as the TS `buildMelFilters` (the sparse bank is cut from it). */
    fun buildDenseMelFilters(): DoubleArray {
        val melMin = hzToMel(0.0)
        val melMax = hzToMel(SAMPLE_RATE / 2.0)
        val filterFreqs = DoubleArray(N_MELS + 2) { melToHz(melMin + (melMax - melMin) * it / (N_MELS + 1)) }
        val fb = DoubleArray(N_MELS * N_FREQ)
        for (m in 0 until N_MELS) {
            val lo = filterFreqs[m]
            val mid = filterFreqs[m + 1]
            val hi = filterFreqs[m + 2]
            val enorm = 2.0 / (hi - lo)
            for (k in 0 until N_FREQ) {
                val f = k * (SAMPLE_RATE / 2.0) / (N_FREQ - 1)
                val down = -(lo - f) / (mid - lo)
                val up = (hi - f) / (hi - mid)
                fb[m * N_FREQ + k] = max(0.0, min(down, up)) * enorm
            }
        }
        return fb
    }

    private val bank: MelBank by lazy {
        val dense = buildDenseMelFilters()
        val starts = IntArray(N_MELS)
        val ws = Array(N_MELS) { DoubleArray(0) }
        for (m in 0 until N_MELS) {
            var lo = -1
            var hi = -1
            for (k in 0 until N_FREQ) if (dense[m * N_FREQ + k] != 0.0) {
                if (lo < 0) lo = k
                hi = k
            }
            if (lo < 0) { lo = 0; hi = 0 }
            starts[m] = lo
            ws[m] = DoubleArray(hi - lo + 1) { dense[m * N_FREQ + lo + it] }
        }
        MelBank(starts, ws)
    }

    private val hann: DoubleArray by lazy { DoubleArray(N_FFT) { 0.5 - 0.5 * cos(2.0 * PI * it / N_FFT) } }

    private val twRe: DoubleArray by lazy { DoubleArray(N_FFT) { cos(2.0 * PI * it / N_FFT) } }
    private val twIm: DoubleArray by lazy { DoubleArray(N_FFT) { -sin(2.0 * PI * it / N_FFT) } }
    private val FACTORS = intArrayOf(4, 4, 5, 5)

    // ---- mixed-radix FFT, N = 400 = 4*4*5*5 -----------------------------------------------------

    private fun fftRec(
        inRe: DoubleArray, inIm: DoubleArray, inOff: Int, stride: Int, n: Int, fi: Int,
        outRe: DoubleArray, outIm: DoubleArray, outOff: Int,
    ) {
        if (n == 1) {
            outRe[outOff] = inRe[inOff]
            outIm[outOff] = inIm[inOff]
            return
        }
        val p = FACTORS[fi]
        val m = n / p
        for (r in 0 until p) {
            fftRec(inRe, inIm, inOff + r * stride, stride * p, m, fi + 1, outRe, outIm, outOff + r * m)
        }
        val twStep = N_FFT / n
        val tmpRe = DoubleArray(p)
        val tmpIm = DoubleArray(p)
        for (k in 0 until m) {
            for (r in 0 until p) {
                val idx = ((r * k) % n) * twStep
                val yr = outRe[outOff + r * m + k]
                val yi = outIm[outOff + r * m + k]
                tmpRe[r] = yr * twRe[idx] - yi * twIm[idx]
                tmpIm[r] = yr * twIm[idx] + yi * twRe[idx]
            }
            for (q in 0 until p) {
                var sr = 0.0
                var si = 0.0
                for (r in 0 until p) {
                    val idx = ((r * q * m) % n) * twStep
                    sr += tmpRe[r] * twRe[idx] - tmpIm[r] * twIm[idx]
                    si += tmpRe[r] * twIm[idx] + tmpIm[r] * twRe[idx]
                }
                outRe[outOff + q * m + k] = sr
                outIm[outOff + q * m + k] = si
            }
        }
    }

    /** Power spectrum (201 bins) of one windowed 400-sample frame. */
    fun powerSpectrum(frame: DoubleArray): DoubleArray {
        val re = DoubleArray(N_FFT)
        val im = DoubleArray(N_FFT)
        fftRec(frame, DoubleArray(N_FFT), 0, 1, N_FFT, 0, re, im, 0)
        return DoubleArray(N_FREQ) { re[it] * re[it] + im[it] * im[it] }
    }

    // ---- the frontend ---------------------------------------------------------------------------

    /** Log-mel for up to 6 s of 16 kHz mono float32. Throws on a waveform the contract rejects. */
    fun compute(audio: FloatArray): Result {
        waveformProblem(audio)?.let { throw IllegalArgumentException("invalid waveform: $it") }
        val n = audio.size
        val half = N_FFT / 2
        val padded = DoubleArray(PAD_SAMPLES + N_FFT)
        for (i in 0 until n) padded[half + i] = audio[i].toDouble()
        for (i in 1..half) {
            padded[half - i] = padded[half + i]
            padded[half + PAD_SAMPLES - 1 + i] = padded[half + PAD_SAMPLES - 1 - i]
        }
        var lastNonZero = -1
        for (i in padded.size - 1 downTo 0) if (padded[i] != 0.0) { lastNonZero = i; break }

        val mb = bank
        val logSpec = DoubleArray(N_MELS * N_FRAMES)
        val frame = DoubleArray(N_FFT)
        var maxVal = Double.NEGATIVE_INFINITY
        for (t in 0 until N_FRAMES) {
            val start = t * HOP
            if (start > lastNonZero) {
                for (m in 0 until N_MELS) logSpec[m * N_FRAMES + t] = -10.0
                if (-10.0 > maxVal) maxVal = -10.0
                continue
            }
            for (i in 0 until N_FFT) frame[i] = padded[start + i] * hann[i]
            val power = powerSpectrum(frame)
            for (m in 0 until N_MELS) {
                var acc = 0.0
                val w = mb.weights[m]
                val s = mb.start[m]
                for (j in w.indices) acc += w[j] * power[s + j]
                val v = log10(max(acc, 1e-10))
                logSpec[m * N_FRAMES + t] = v
                if (v > maxVal) maxVal = v
            }
        }
        val floor = maxVal - 8.0
        val out = FloatArray(N_MELS * N_FRAMES) { ((max(logSpec[it], floor) + 4.0) / 4.0).toFloat() }
        return Result(out, min(MAX_VALID_FRAMES, Math.ceil(n / 320.0).toInt()))
    }
}
