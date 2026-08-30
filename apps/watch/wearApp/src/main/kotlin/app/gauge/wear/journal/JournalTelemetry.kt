package app.gauge.wear.journal

import java.util.concurrent.atomic.AtomicInteger
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Process-wide journal counters, shared between [app.gauge.wear.service.SentinelService] (which
 * increments them on every upload tick) and the UI's toggle telemetry (which only reads them) —
 * same single-process StateFlow-less bus rationale as [app.gauge.wear.capture.RetroCaptureBus].
 * Atomics because the increments happen on the service's IO coroutines while reads can come from
 * any thread. [reset] exists for tests only.
 */
object JournalStats {
    val uploads = AtomicInteger(0)
    val uploadFailures = AtomicInteger(0)
    val drops = AtomicInteger(0)

    /** The service's last-reported mic capture mode name ("CONTINUOUS"/"DUTY_CYCLED") — written
     * by the journal tick from its own captured snapshot, read by toggle telemetry. */
    @Volatile var micDutyState: String = "CONTINUOUS"

    fun reset() {
        uploads.set(0)
        uploadFailures.set(0)
        drops.set(0)
        micDutyState = "CONTINUOUS"
    }
}

/**
 * THE journal telemetry `data` payload shape — one builder so every journal event (upload tick,
 * toggle on, toggle off) carries identical keys:
 *
 * ```json
 * {
 *   "battery_pct": 87,            // Int, or null when BatteryManager had no reading
 *   "charging": false,            // Boolean, or null when unreadable
 *   "journal_uploads": 3,         // successful uploads this process lifetime
 *   "journal_upload_failures": 1, // failed attempts this process lifetime
 *   "journal_drops": 0,           // snapshots dropped from the capacity-1 retry queue
 *   "mic_duty_state": "CONTINUOUS" // or "DUTY_CYCLED" (MicDutyCycle's CaptureMode name)
 * }
 * ```
 *
 * Pure (no Android imports) so the exact shape is pinned by a plain JVM test. Unreadable battery
 * facts are an explicit JSON `null`, never a fabricated number.
 */
fun journalTelemetryData(
    batteryPct: Int?,
    charging: Boolean?,
    journalUploads: Int,
    journalUploadFailures: Int,
    journalDrops: Int,
    micDutyState: String,
): JsonObject = buildJsonObject {
    if (batteryPct != null) put("battery_pct", batteryPct) else put("battery_pct", JsonNull)
    if (charging != null) put("charging", charging) else put("charging", JsonNull)
    put("journal_uploads", journalUploads)
    put("journal_upload_failures", journalUploadFailures)
    put("journal_drops", journalDrops)
    put("mic_duty_state", micDutyState)
}
