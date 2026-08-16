package app.gauge.wear

import android.app.Application
import app.gauge.wear.auth.AccountBus
import app.gauge.wear.auth.AccountPrefs
import app.gauge.wear.telemetry.Telemetry
import app.gauge.wear.telemetry.startBanner

/**
 * Application entry point, shared by every component in this single-process app (activity,
 * SentinelService, tile, complication). v0.1.0 had no Application subclass — this class wires up
 * the [Telemetry] singleton and crash handler before any of them can run, so ANY future crash
 * (not just the raise-voice one this task targets) is visible to an agent via `POST /telemetry`
 * instead of requiring the user to screen-capture the watch.
 */
class GaugeApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Telemetry.init(this)
        Telemetry.installCrashHandler()
        // T9-review fix (Important, Wave C): AccountBus.signedIn starts at its own hardcoded
        // `false` default and nothing was ever hydrating it from AccountPrefs (the actual
        // on-disk source of truth) at process start -- so a signed-in wearer whose process was
        // killed and restarted (routine on Wear OS: low memory, an OS-initiated restart, the app
        // simply not having run recently) would see "Sign in" and lose the couples chip until
        // SOMETHING else happened to call AccountBus.publish(true) again (only ever done from
        // SignInScreen's own successful claim). One-line fix: seed the bus from disk on every
        // process start, before any Composable can read it.
        AccountBus.publish(AccountPrefs.isSignedIn(this))
        // v0.2.4: version banner — logged once per process start and flushed IMMEDIATELY (the
        // ring otherwise only flushes on episode end / sentinel stop, which made a freshly
        // installed build invisible to the telemetry channel until its first episode). Per-event
        // app_version already rides every batch (TelemetryBatch.app_version, denormalized into
        // each stored event by server/telemetry_api.py) — this banner adds the versionCode and a
        // greppable "the process actually started" marker. Both calls are already no-ops when
        // telemetry is disabled/uninitialized.
        //
        // Night 2 hardening (Wave C final-review minor): wrapped in the same runCatching
        // convention as every other Telemetry call site in this app (SentinelService's
        // tick/hr/pulse paths, HapticDirector's reportHapticPath, ...) — this is the FIRST code
        // to run in the process, before any of GaugeApp.onCreate's own crash-handler
        // (installCrashHandler, above) has anything else installed yet to catch it; a failure in
        // either call (e.g. TelemetryClient.postAsync's synchronous setup, or a Log.println
        // edge case) must degrade to "no banner this run", never take down process start itself.
        runCatching {
            Telemetry.log("info", "GaugeApp", startBanner(BuildConfig.VERSION_NAME, BuildConfig.VERSION_CODE))
        }
        runCatching { Telemetry.flushAsync() }
    }
}
