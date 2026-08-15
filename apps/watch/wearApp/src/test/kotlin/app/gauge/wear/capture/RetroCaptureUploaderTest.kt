package app.gauge.wear.capture

import app.gauge.shared.Capture
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private fun capture(status: String = "stored") = Capture(
    id = "cap1", accountId = "acct1", capturedAt = "2026-08-04T10:00:00Z",
    receivedAt = "2026-08-04T10:00:01Z", durationS = 118.0, status = status,
)

class RetroCaptureUploaderTest {

    private class FakeApi(
        val createResult: ApiResult<Capture>,
        val uploadResult: ApiResult<Capture>,
    ) : WatchApiClient(baseUrl = "https://unused.example", deviceToken = { "tok" }) {
        var uploadedBytes: ByteArray? = null
        var createCaptureCalls = 0
        var uploadCaptureAudioCalls = 0
        var lastAttested: Boolean? = null

        override suspend fun createCapture(req: app.gauge.shared.CreateCaptureRequest): ApiResult<Capture> {
            createCaptureCalls++
            lastAttested = req.attested
            return createResult
        }

        override suspend fun uploadCaptureAudio(captureId: String, gzippedPcm: ByteArray): ApiResult<Capture> {
            uploadCaptureAudioCalls++
            uploadedBytes = gzippedPcm
            return uploadResult
        }
    }

    @Test
    fun uploadSucceedsAndGzipsThePcm() = runTest {
        val api = FakeApi(createResult = ApiResult.Ok(capture()), uploadResult = ApiResult.Ok(capture()))
        val uploader = RetroCaptureUploader(api, nowIso = { "2026-08-04T10:00:00Z" }, deviceId = "watch-1")
        val pcm = ByteArray(16000 * 2 * 5) // 5s of silence, highly compressible
        val ok = uploader.upload(pcm, durationS = 5.0, trigger = "manual", consentConfirmed = true)
        assertTrue(ok)
        assertTrue(api.uploadedBytes!!.size < pcm.size) // gzip actually shrank it
    }

    @Test
    fun uploadFailsSoftWhenCreateFails() = runTest {
        val api = FakeApi(createResult = ApiResult.Failure(401, "not signed in"), uploadResult = ApiResult.Ok(capture()))
        val uploader = RetroCaptureUploader(api, nowIso = { "2026-08-04T10:00:00Z" }, deviceId = "watch-1")
        val ok = uploader.upload(ByteArray(100), durationS = 1.0, trigger = "manual", consentConfirmed = true)
        assertFalse(ok)
    }

    @Test
    fun uploadFailsSoftWhenAudioPutFails() = runTest {
        val api = FakeApi(createResult = ApiResult.Ok(capture(status = "awaiting_audio")), uploadResult = ApiResult.Failure(503, "storage not configured"))
        val uploader = RetroCaptureUploader(api, nowIso = { "2026-08-04T10:00:00Z" }, deviceId = "watch-1")
        val ok = uploader.upload(ByteArray(100), durationS = 1.0, trigger = "manual", consentConfirmed = true)
        assertFalse(ok)
    }

    // --- Consent hard gate (coordinator-directed): "no consent artifact -> no upload,
    // structurally" -- the brief's own literal code hardcoded `attested = true` unconditionally
    // inside upload(), trusting a comment ("gated upstream by the consent dialog") rather than an
    // enforced contract. consentConfirmed is now a required parameter this class itself refuses
    // to proceed without, so a caller (Task 12's UI) that forgets to actually gate on a real
    // consent dialog can never accidentally attest on the wearer's behalf. -------------------------

    @Test
    fun consentNotConfirmedNeverTouchesTheNetworkAtAll() = runTest {
        val api = FakeApi(createResult = ApiResult.Ok(capture()), uploadResult = ApiResult.Ok(capture()))
        val uploader = RetroCaptureUploader(api, nowIso = { "2026-08-04T10:00:00Z" }, deviceId = "watch-1")
        val ok = uploader.upload(ByteArray(100), durationS = 1.0, trigger = "manual", consentConfirmed = false)
        assertFalse(ok)
        assertEquals(0, api.createCaptureCalls, "no consent artifact -> no upload, structurally: not even createCapture may fire")
        assertEquals(0, api.uploadCaptureAudioCalls)
    }

    @Test
    fun consentConfirmedIsWhatGetsSentAsAttestedNeverAHardcodedLiteral() = runTest {
        val api = FakeApi(createResult = ApiResult.Ok(capture()), uploadResult = ApiResult.Ok(capture()))
        val uploader = RetroCaptureUploader(api, nowIso = { "2026-08-04T10:00:00Z" }, deviceId = "watch-1")
        uploader.upload(ByteArray(100), durationS = 1.0, trigger = "manual", consentConfirmed = true)
        assertEquals(true, api.lastAttested)
    }
}
