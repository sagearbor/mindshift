package app.gauge.wear.prefs

import android.content.Context
import java.util.UUID

/**
 * Thin SharedPreferences wrapper for device-scoped settings. Android-only
 * shell (compile gate, no unit test) — mirrors the pattern of other
 * `android.*`-dependent classes in this module.
 */
object GaugePrefs {
    private const val PREFS_NAME = "gauge_prefs"
    private const val KEY_DEVICE_ID = "device_id"
    private const val KEY_SIGNAL = "signal"
    private const val DEFAULT_SIGNAL = "VOLUME"
    private const val KEY_PULSE_INTERVAL_MS = "pulse_interval_ms"
    private const val DEFAULT_PULSE_INTERVAL_MS = "250"

    /** Returns a stable per-install device id, minting and persisting one on first call. */
    fun deviceId(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val existing = prefs.getString(KEY_DEVICE_ID, null)
        if (existing != null) return existing
        val minted = UUID.randomUUID().toString()
        prefs.edit().putString(KEY_DEVICE_ID, minted).apply()
        return minted
    }

    /** Returns the user-selected live-meter signal, defaulting to [DEFAULT_SIGNAL]. Consumed by Task 11. */
    fun selectedSignal(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_SIGNAL, DEFAULT_SIGNAL) ?: DEFAULT_SIGNAL
    }

    fun setSelectedSignal(context: Context, name: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_SIGNAL, name).apply()
    }

    /** Returns the wearer's "Pulse speed" preference — one of "250"/"500"/"1000"/"off" (default
     * "250") — see [app.gauge.wear.haptics.PulseEngine]'s KDoc for what each value means.
     * Consumed by [app.gauge.wear.service.SentinelService] via a `() -> Long?` supplier (not a
     * direct GaugePrefs reference — same Android-free-controller pattern as [selectedSignal]),
     * which parses "off" to `null`. */
    fun pulseIntervalMs(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_PULSE_INTERVAL_MS, DEFAULT_PULSE_INTERVAL_MS) ?: DEFAULT_PULSE_INTERVAL_MS
    }

    fun setPulseIntervalMs(context: Context, value: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_PULSE_INTERVAL_MS, value).apply()
    }

    private const val KEY_CENTER_DISPLAY = "center_display"

    /** v0.2.4 (Addendum 2): "SPARKLINE" (default) or "DIAL" — THE single center-visualization
     * preference, replacing the removed v0.2.3-era "center_number" (METER/CALM) pref as the only
     * center-related setting. Defaulting to SPARKLINE is a deliberate default change for existing
     * installs too: this key exists on no device yet, so every current user lands on the new
     * sparkline-first center (called out in the ship notes). */
    private const val DEFAULT_CENTER_DISPLAY = "SPARKLINE"

    fun centerDisplay(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        return prefs.getString(KEY_CENTER_DISPLAY, DEFAULT_CENTER_DISPLAY) ?: DEFAULT_CENTER_DISPLAY
    }

    fun setCenterDisplay(context: Context, value: String) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_CENTER_DISPLAY, value).apply()
    }
}
