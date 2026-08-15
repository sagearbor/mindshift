package app.gauge.wear.capture

import app.gauge.shared.CreateCaptureRequest
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient
import java.io.ByteArrayOutputStream
import java.util.zip.GZIPOutputStream

/**
 * Uploads a retro-capture snapshot as the two-step `POST /captures` +
 * `PUT /captures/{id}/audio` the server expects (`server/captures_api.py`,
 * read 2026-08-04) — gzip-compressed (speech-shaped PCM16 typically shrinks
 * 2-3x), matching the server's `Content-Encoding: gzip` support. Fail-soft
 * end to end: every failure mode (network, 401, 413, 503 "storage not
 * configured") resolves to `false`, never a thrown exception — this runs off
 * a UI button tap, not a crash-tolerant service loop, so a throw here would
 * be a visible app crash for a "nice to have" feature.
 *
 * Never claims saved when the server didn't confirm: [upload] returns `true`
 * only when `PUT /captures/{id}/audio` itself came back [ApiResult.Ok] — never
 * optimistically on the strength of `POST /captures` alone. Nothing here
 * touches [app.gauge.shared.capture.RetroCaptureBuffer]/[app.gauge.wear.control.
 * SentinelController] at all — the caller owns the in-memory PCM bytes and
 * decides whether/when to retry with a fresh (or the same) snapshot; a failed
 * upload here never discards or overwrites anything on its own.
 *
 * Consent hard gate (coordinator-directed review fix, Wave C): [consentConfirmed]
 * is a REQUIRED parameter, not a hardcoded literal — this class refuses to make
 * ANY network call at all (not even `POST /captures`) unless the caller passes
 * `true`. The brief's own original design instead hardcoded `attested = true`
 * unconditionally inside [upload], trusting an inline comment ("gated upstream
 * by the consent dialog") rather than an enforced contract — meaning a caller
 * (Task 12's UI) with a bug in its own consent-dialog gating could still cause
 * this class to attest on the wearer's behalf with no backstop. Structural, not
 * documentary: "no consent artifact → no upload" now holds regardless of what
 * the caller does or forgets to do.
 */
class RetroCaptureUploader(
    private val api: WatchApiClient,
    private val nowIso: () -> String,
    private val deviceId: String,
) {
    suspend fun upload(pcm: ByteArray, durationS: Double, trigger: String, consentConfirmed: Boolean): Boolean {
        if (!consentConfirmed) return false
        return try {
            val created = api.createCapture(
                CreateCaptureRequest(
                    capturedAt = nowIso(),
                    durationS = durationS,
                    trigger = trigger,
                    device = deviceId,
                    attested = consentConfirmed,
                ),
            )
            val captureId = (created as? ApiResult.Ok)?.value?.id ?: return false
            val gzipped = gzip(pcm)
            val uploaded = api.uploadCaptureAudio(captureId, gzipped)
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
