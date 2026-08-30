package app.gauge.wear.telemetry

import android.content.Context
import android.util.Log
import app.gauge.shared.TelemetryEventOut
import app.gauge.shared.telemetry.DebugRing
import app.gauge.wear.BuildConfig
import app.gauge.wear.prefs.GaugePrefs
import java.time.Instant
import kotlinx.serialization.json.JsonObject

private const val MAX_STACK_CHARS = 20_000

/**
 * Pure, Android-free: builds the exact telemetry payload for an uncaught exception — the current
 * debug-ring snapshot followed by one synthesized "crash" event. No side effects and no
 * `android.*` imports, so it's unit-testable as a plain JVM function (see CrashPayloadTest)
 * independent of [Telemetry]'s Android wiring. Stack is truncated client-side so a pathological
 * exception (e.g. a message containing megabytes of text) can't blow up the telemetry payload.
 */
fun crashEvents(
    t: Throwable,
    threadName: String,
    ring: DebugRing,
    nowIso: String,
): List<TelemetryEventOut> {
    val crash = TelemetryEventOut(
        level = "crash",
        tag = "UncaughtException",
        message = "${t::class.simpleName ?: "Throwable"} on $threadName: ${t.message ?: ""}",
        stack = t.stackTraceToString().take(MAX_STACK_CHARS),
        ts = nowIso,
    )
    return ring.snapshot() + crash
}

/** v0.2.4: the one-line app-start banner. Pure so the exact greppable format is pinned by test —
 * an agent identifies the installed build via `GET /telemetry` + `grep "app start"`. */
fun startBanner(versionName: String, versionCode: Int): String =
    "app start v$versionName (code $versionCode)"

/**
 * Process-wide telemetry singleton. [app.gauge.wear.GaugeApp.onCreate] is the only call site for
 * [init] / [installCrashHandler] — every other component (service, activity, tile,
 * complication) shares this single process and only ever calls [log] / [flushAsync].
 *
 * This is the device→backend visibility unlock: without it, on-device crashes (like the
 * raise-voice crash this task targets) are invisible to an agent that can't screen-capture the
 * watch. [installCrashHandler] posts a best-effort crash report and then ALWAYS re-delegates to
 * whatever handler was previously installed — it must never itself be the reason the process
 * fails to die/report normally.
 */
object Telemetry {
    private const val RING_CAPACITY = 200

    @Volatile private var client: TelemetryClient? = null
    private val ring = DebugRing(RING_CAPACITY)
    @Volatile private var crashHandlerInstalled = false

    /** Idempotent; no-op when telemetry is disabled for this build. */
    fun init(context: Context) {
        if (!BuildConfig.TELEMETRY_ENABLED) return
        if (client != null) return
        client = TelemetryClient(
            baseUrl = BuildConfig.GAUGE_API_BASE,
            device = GaugePrefs.deviceId(context),
            appVersion = BuildConfig.VERSION_NAME,
        )
    }

    fun log(level: String, tag: String, message: String) {
        ring.add(level, tag, message, nowIso())
        Log.println(priorityFor(level), tag, message)
    }

    /** [log] with a structured `data` payload riding the event (server's additive
     * `TelemetryEventIn.data` — see [app.gauge.shared.TelemetryEventOut.data]). Used by the
     * journal battery/counter series; behaves exactly like [log] otherwise. */
    fun log(level: String, tag: String, message: String, data: JsonObject?) {
        ring.add(level, tag, message, nowIso(), data = data)
        Log.println(priorityFor(level), tag, message)
    }

    /** Drains the ring and fire-and-forgets it to the backend. Safe to call when disabled/uninitialized. */
    fun flushAsync() {
        val c = client ?: return
        val events = ring.snapshot()
        if (events.isEmpty()) return
        ring.clear()
        c.postAsync(events)
    }

    /**
     * Wraps whatever [Thread.getDefaultUncaughtExceptionHandler] currently returns (Android
     * always installs one) so a best-effort crash report goes out before the process dies.
     * Idempotent — calling twice (e.g. a second GaugeApp.onCreate in tests) installs only once.
     */
    fun installCrashHandler() {
        if (crashHandlerInstalled) return
        crashHandlerInstalled = true
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                client?.let { c ->
                    c.postBlocking(crashEvents(throwable, thread.name, ring, nowIso()), timeoutMs = 2000)
                }
            } catch (reportingFailure: Throwable) {
                // Telemetry reporting must never be the reason the crash itself goes unhandled.
            }
            // Never swallow: the process must still die (or be handled) exactly as it would have
            // without this handler installed.
            if (previous != null) {
                previous.uncaughtException(thread, throwable)
            } else {
                android.os.Process.killProcess(android.os.Process.myPid())
            }
        }
    }

    fun nowIso(): String = Instant.now().toString()

    private fun priorityFor(level: String): Int = when (level) {
        "crash", "error" -> Log.ERROR
        "warn" -> Log.WARN
        "debug" -> Log.DEBUG
        else -> Log.INFO
    }
}
