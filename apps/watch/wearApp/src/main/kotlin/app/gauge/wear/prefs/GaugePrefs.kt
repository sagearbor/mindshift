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
    /**
     * "Pulse speed" default — **"off" since 2026-09-10, measured, not guessed.**
     *
     * The proportional pulse train emits one tap per 1 s audio window while the
     * wearer is >= 6 dB over their baseline. Counted against the two REAL
     * fixture recordings that is **19 buzzes/min on the owner's family
     * recording and 28/min on `poker6_real` — a poker game, not an argument**.
     * At that dose the wrist is a vibrating loudness meter, and a cue that
     * fires on ordinary conversation teaches the wearer to ignore it, which
     * costs the nudges that DO matter.
     *
     * It is not deleted, because as a live self-monitoring gauge it does
     * exactly what it was designed to do — it is just not a coaching default.
     * Anyone who wants it turns it back on in Settings -> Pulse speed.
     * apps/watch/.../PulseDoseTest.kt pins the dose arithmetic behind this.
     */
    private const val DEFAULT_PULSE_INTERVAL_MS = "off"

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

    private const val KEY_HEAT_HOLD_S = "heat_hold_s"

    /**
     * Hold-N hysteresis on the offline loudness ladder, in seconds — the wrist's half of the
     * 2026-09-20 change (mirror: server `MINDSHIFT_HEAT_HOLD_S`, phone `HEAT_HOLD_S`). The first
     * rung (+6 dB over the wearer's own baseline) must hold this long before the ladder may climb.
     *
     * **A DEBUG knob, deliberately with no Settings row.** Every other value than the shipped
     * [app.gauge.shared.HEAT_HOLD_S] is worse for the wearer — 0 or 1 restores the ladder that
     * delivered 97 buzzes/hour on the median real meeting, and a longer hold trades away real
     * escalations — so this exists to reproduce a measurement on a device, not to be chosen from a
     * menu. Set it with `adb shell` on a debug build; absent (the normal case) it reads the shared
     * constant, and an unparseable or negative value falls back to it too, which is the same
     * "a corrupt byte must pick the safe value" rule [pulseIntervalMs] follows.
     */
    fun heatHoldSeconds(context: Context): Double {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_HEAT_HOLD_S, null) ?: return app.gauge.shared.HEAT_HOLD_S
        val parsed = raw.toDoubleOrNull() ?: return app.gauge.shared.HEAT_HOLD_S
        return if (parsed >= 0.0) parsed else app.gauge.shared.HEAT_HOLD_S
    }

    fun setHeatHoldSeconds(context: Context, value: Double) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_HEAT_HOLD_S, value.toString()).apply()
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
