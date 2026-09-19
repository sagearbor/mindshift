package app.gauge.wear.journal

import app.gauge.shared.CreateCaptureRequest
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient
import java.io.ByteArrayOutputStream
import java.util.zip.GZIPOutputStream
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * Uploads one journal snapshot through the existing captures API as THREE calls, in this exact
 * order: `POST /captures` (metadata + attested consent) → `PUT /captures/{id}/labels`
 * (`{"journal": true, "interval_s": N}`) → `PUT /captures/{id}/audio` (gzip). Labels go BEFORE
 * audio on purpose: the server's journal self-filtering hook fires off the audio-upload success
 * path and only for captures already carrying the `journal` label at that moment — audio-first
 * would store a capture the server never processes. A labels failure therefore ABORTS the whole
 * attempt (no audio upload), so the caller's retry gets another shot at a fully-labeled capture
 * instead of half-uploading an unprocessable one.
 *
 * Consent hard gate, same structural rule as [app.gauge.wear.capture.RetroCaptureUploader]:
 * [consentConfirmed] is required, and `false` means NO network call at all — the caller
 * ([app.gauge.wear.service.SentinelService]) derives it from the stored journal consent artifact
 * ([app.gauge.wear.prefs.GaugePrefs.journalConsentTs]), never hardcodes `true`.
 *
 * Fail-soft end to end: every failure resolves to `false`, never a thrown exception — this runs
 * on the sentinel's service scope and must never take down the tick loop it rides beside. The
 * captures API requires a paired watch's Bearer token; with no token the underlying
 * [WatchApiClient] fails fast before the network, and the caller surfaces "pair your watch
 * first" separately.
 */
class JournalUploader(
    private val api: WatchApiClient,
    private val nowIso: () -> String,
    private val deviceId: String,
) {
    suspend fun upload(snapshot: JournalQueue.Snapshot, consentConfirmed: Boolean): Boolean {
        if (!consentConfirmed) return false
        if (snapshot.pcm.isEmpty() || snapshot.durationS <= 0.0) return false
        return try {
            val created = api.createCapture(
                CreateCaptureRequest(
                    capturedAt = snapshot.capturedAtIso,
                    durationS = snapshot.durationS,
                    trigger = "journal",
                    device = deviceId,
                    attested = consentConfirmed,
                ),
            )
            val captureId = (created as? ApiResult.Ok)?.value?.id ?: return false
            val labels = buildJsonObject {
                put("journal", true)
                put("interval_s", snapshot.intervalS)
            }
            val labeled = api.putCaptureLabels(captureId, labels.toString())
            if (labeled !is ApiResult.Ok) return false
            val uploaded = api.uploadCaptureAudio(captureId, gzip(snapshot.pcm))
            uploaded is ApiResult.Ok
        } catch (e: Exception) {
            false
        }
    }

    private fun gzip(bytes: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        GZIPOutputStream(out).use { it.write(bytes) }
        return out.toByteArray()
    }
}
