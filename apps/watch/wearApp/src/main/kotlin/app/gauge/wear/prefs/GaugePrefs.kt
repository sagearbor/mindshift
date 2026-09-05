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

    private const val KEY_JOURNAL_MODE = "journal_mode"
    private const val KEY_JOURNAL_CONSENT_TS = "journal_consent_ts"

    /** A/B journal toggle ("Journal — keep what I say"): whether auto retro-capture uploads are
     * on. `true` is only ever written together with a consent timestamp ([enableJournal]) and
     * both are cleared together ([disableJournal]) — the pair is the ON-state consent artifact
     * the service's upload path gates on ([journalConsentTs] non-null), mirroring
     * [app.gauge.wear.capture.RetroCaptureUploader]'s "no consent artifact → no upload"
     * structural rule for the manual path. */
    fun journalMode(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_JOURNAL_MODE, false)

    /** ISO timestamp of the wearer's journal consent confirmation, or `null` when journal mode
     * is off (or was never consented). Cleared by [disableJournal] — consent is session-long for
     * the toggle's ON stretch, never carried across an off/on cycle. */
    fun journalConsentTs(context: Context): String? =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_JOURNAL_CONSENT_TS, null)

    /** Turns journal mode ON, storing [consentTsIso] (the wearer's explicit "Confirm" tap time)
     * atomically with the flag — there is no way to enable journal mode without minting the
     * consent artifact, by construction. */
    fun enableJournal(context: Context, consentTsIso: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_JOURNAL_MODE, true)
            .putString(KEY_JOURNAL_CONSENT_TS, consentTsIso)
            .apply()
    }

    /** Turns journal mode OFF and clears the stored consent — the next ON requires a fresh
     * confirmation. */
    fun disableJournal(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putBoolean(KEY_JOURNAL_MODE, false)
            .remove(KEY_JOURNAL_CONSENT_TS)
            .apply()
    }
}
