package app.gauge.wear.auth

import android.content.Context

/**
 * Thin SharedPreferences wrapper for the watch's signed-in identity (Wave C).
 * Separate file from [app.gauge.wear.prefs.GaugePrefs] (device-scoped UI prefs)
 * since this one holds a secret (`device_token`) with a narrower blast radius if
 * ever audited/cleared independently — same separation-of-concerns reasoning as
 * this app's existing `control`/`net`/`prefs` package split.
 *
 * `device_token` is the opaque, long-lived credential minted server-side once a
 * `POST /me/pair/claim` (phone/web side) resolves a pairing code this watch
 * started (see [app.gauge.wear.auth.DevicePairingClient]) — NOT a Firebase ID
 * token. It is sent as `Authorization: Bearer <device_token>` by
 * [app.gauge.wear.net.WatchApiClient] and verified server-side by a new
 * `TokenVerifier` implementation this plan documents but does not implement
 * (see the plan's "Open questions").
 */
object AccountPrefs {
    private const val PREFS_NAME = "gauge_account"
    private const val KEY_ACCOUNT_ID = "account_id"
    private const val KEY_DEVICE_TOKEN = "device_token"

    fun accountId(context: Context): String? =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(KEY_ACCOUNT_ID, null)

    fun deviceToken(context: Context): String? =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).getString(KEY_DEVICE_TOKEN, null)

    fun isSignedIn(context: Context): Boolean = deviceToken(context) != null

    fun setSignedIn(context: Context, accountId: String, deviceToken: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .putString(KEY_ACCOUNT_ID, accountId)
            .putString(KEY_DEVICE_TOKEN, deviceToken)
            .apply()
    }

    fun signOut(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE).edit()
            .remove(KEY_ACCOUNT_ID)
            .remove(KEY_DEVICE_TOKEN)
            .apply()
    }
}
