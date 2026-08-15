package app.gauge.wear.haptics

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import app.gauge.wear.control.VibratorPort

/**
 * Real [VibratorPort]: adapts the platform Vibrator (VibratorManager on API 31+, legacy service
 * below it). Moved out of SentinelService.kt in v0.2.4 (it now also backs SettingsScreen's
 * "Feel the buzzes" demo) and taught the device-tuned paths.
 *
 * Support probing is honest, not optimistic: [playPredefined] returns false only when the
 * platform SAYS the effect is unsupported (VIBRATION_EFFECT_SUPPORT_NO) — an UNKNOWN answer still
 * plays, because the platform provides its own internal fallback for predefined effects and that
 * fallback is still better-tuned than our generic waveform. [playComposedClicks] requires
 * primitive support outright (compositions have no platform fallback; an unsupported primitive
 * composes to nothing). All support checks are API 30+, matching minSdk 30.
 */
class RealVibratorPort(context: Context) : VibratorPort {
    private val vibrator: Vibrator =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            manager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }

    override fun vibrate(timingsMs: LongArray, amplitudes: IntArray) {
        vibrator.vibrate(VibrationEffect.createWaveform(timingsMs, amplitudes, -1))
    }

    override fun hasAmplitudeControl(): Boolean = vibrator.hasAmplitudeControl()

    override fun playPredefined(effect: PredefinedEffect): Boolean {
        val id = when (effect) {
            PredefinedEffect.CLICK -> VibrationEffect.EFFECT_CLICK
            PredefinedEffect.HEAVY_CLICK -> VibrationEffect.EFFECT_HEAVY_CLICK
        }
        if (vibrator.areAllEffectsSupported(id) == Vibrator.VIBRATION_EFFECT_SUPPORT_NO) return false
        vibrator.vibrate(VibrationEffect.createPredefined(id))
        return true
    }

    override fun playComposedClicks(count: Int, gapMs: Long): Boolean {
        if (!vibrator.areAllPrimitivesSupported(VibrationEffect.Composition.PRIMITIVE_CLICK)) return false
        val composition = VibrationEffect.startComposition()
        repeat(count) { i ->
            composition.addPrimitive(
                VibrationEffect.Composition.PRIMITIVE_CLICK,
                /* scale = */ 1.0f,
                /* delay = */ if (i == 0) 0 else gapMs.toInt(),
            )
        }
        vibrator.vibrate(composition.compose())
        return true
    }
}
