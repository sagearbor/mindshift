package app.gauge.wear.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.ScalarSource
import kotlin.math.sqrt

/**
 * [ScalarSource] backed by the accelerometer (Task 10): tracks movement/fidget as the standard
 * deviation of the acceleration-vector magnitude within rolling 1-second buckets.
 *
 * Android shell, compile+lint gated (no unit test — no emulator/device in this repo's CI loop,
 * per CLAUDE.md).
 *
 * Honest degradation: [latest] returns `null` until the first 1-second bucket has fully elapsed —
 * never a fabricated 0.0 placeholder for a bucket that hasn't finished accumulating yet.
 */
class AccelSource(private val context: Context, private val diag: DiagLog) : ScalarSource, SensorEventListener {
    private val magnitudes = mutableListOf<Double>()
    private var bucketStartMs = 0L
    @Volatile private var lastBucketStddev: Double? = null

    private var sensorManager: SensorManager? = null

    override fun latest(): Double? = lastBucketStddev

    fun start() {
        try {
            val manager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager ?: return
            val sensor = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) ?: return
            sensorManager = manager
            bucketStartMs = System.currentTimeMillis()
            magnitudes.clear()
            manager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_UI)
        } catch (t: Throwable) {
            diag.log("error", "AccelSource", "start failed: $t")
        }
    }

    fun stop() {
        try {
            sensorManager?.unregisterListener(this)
        } catch (t: Throwable) {
            diag.log("error", "AccelSource", "stop failed: $t")
        }
        sensorManager = null
        lastBucketStddev = null
        magnitudes.clear()
    }

    override fun onSensorChanged(event: SensorEvent?) {
        try {
            val e = event ?: return
            val magnitude = sqrt(
                e.values[0] * e.values[0] + e.values[1] * e.values[1] + e.values[2] * e.values[2],
            ).toDouble()
            val now = System.currentTimeMillis()
            if (now - bucketStartMs >= BUCKET_MS) {
                // No fake data ever: an empty bucket (first callback ≥1s after start, or any
                // >1s gap between callbacks) has no samples to derive a stddev from — publish
                // nothing rather than a fabricated 0.0; latest() keeps its previous value or null.
                if (magnitudes.isNotEmpty()) {
                    lastBucketStddev = stddev(magnitudes)
                }
                magnitudes.clear()
                bucketStartMs = now
            }
            magnitudes.add(magnitude)
        } catch (t: Throwable) {
            diag.log("error", "AccelSource", "onSensorChanged failed: $t")
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    private fun stddev(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val mean = values.sum() / values.size
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        return sqrt(variance)
    }

    companion object {
        private const val BUCKET_MS = 1000L
    }
}
