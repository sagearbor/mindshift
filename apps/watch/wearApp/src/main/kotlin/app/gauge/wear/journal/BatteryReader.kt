package app.gauge.wear.journal

import android.content.Context
import android.os.BatteryManager
import app.gauge.wear.telemetry.Telemetry
import java.time.Instant

/** One battery reading: `null` fields mean the platform had no honest answer (never a guess). */
data class BatteryStatus(val pct: Int?, val charging: Boolean?)

/**
 * Thin [BatteryManager] wrapper — Android-only shell (compile gate, no unit test), same pattern
 * as [app.gauge.wear.prefs.GaugePrefs]. Every read is fail-soft: a throwing/absent service
 * degrades to `BatteryStatus(null, null)` rather than ever affecting the caller (the journal
 * tick rides the sentinel loop).
 */
object BatteryReader {
    fun read(context: Context): BatteryStatus = try {
        val bm = context.getSystemService(BatteryManager::class.java)
        if (bm == null) {
            BatteryStatus(null, null)
        } else {
            val pct = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                .takeIf { it in 0..100 }
            BatteryStatus(pct = pct, charging = bm.isCharging)
        }
    } catch (t: Throwable) {
        BatteryStatus(null, null)
    }
}

/**
 * Logs a journal toggle transition with the standard journal `data` payload (battery + counters
 * — see [journalTelemetryData]) and flushes immediately so the on/off marker reaches the backend
 * without waiting for the next episode-end flush. Fail-soft like every Telemetry call site.
 */
fun logJournalToggle(context: Context, enabled: Boolean) {
    runCatching {
        val battery = BatteryReader.read(context)
        Telemetry.log(
            "info",
            "Journal",
            if (enabled) "journal on" else "journal off",
            journalTelemetryData(
                batteryPct = battery.pct,
                charging = battery.charging,
                journalUploads = JournalStats.uploads.get(),
                journalUploadFailures = JournalStats.uploadFailures.get(),
                journalDrops = JournalStats.drops.get(),
                micDutyState = JournalStats.micDutyState,
            ),
        )
        Telemetry.flushAsync()
    }
}

/** Shared ISO clock helper for journal call sites (matches [Telemetry.nowIso]'s format). */
fun journalNowIso(): String = Instant.now().toString()
