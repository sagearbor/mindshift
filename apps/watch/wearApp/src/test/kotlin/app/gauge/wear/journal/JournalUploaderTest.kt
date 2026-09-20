package app.gauge.wear.journal

import app.gauge.shared.Capture
import app.gauge.wear.net.ApiResult
import app.gauge.wear.net.WatchApiClient
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.test.runTest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

private fun capture(status: String = "awaiting_audio") = Capture(
    id = "cap1", accountId = "acct1", capturedAt = "2026-08-30T10:00:00Z",
    receivedAt = "2026-08-30T10:00:01Z", durationS = 300.0, status = status,
)

private fun snapshot(pcm: ByteArray = ByteArray(16000 * 2 * 5)) = JournalQueue.Snapshot(
    pcm = pcm, durationS = pcm.size / 32000.0, intervalS = 300.0, capturedAtIso = "2026-08-30T10:00:00Z",
)

class JournalUploaderTest {

    /** Same FakeApi seam as RetroCaptureUploaderTest, extended with labels + a call-order log. */
    private class FakeApi(
        val createResult: ApiResult<Capture> = ApiResult.Ok(capture()),
        val labelsResult: ApiResult<Capture> = ApiResult.Ok(capture()),
        val uploadResult: ApiResult<Capture> = ApiResult.Ok(capture("stored")),
    ) : WatchApiClient(baseUrl = "https://unused.example", deviceToken = { "tok" }) {
        val calls = mutableListOf<String>()
        var lastLabelsJson: String? = null
        var lastAttested: Boolean? = null
        var uploadedBytes: ByteArray? = null

        override suspend fun createCapture(req: app.gauge.shared.CreateCaptureRequest): ApiResult<Capture> {
            calls += "create"
            lastAttested = req.attested
            return createResult
        }

        override suspend fun putCaptureLabels(captureId: String, labelsJson: String): ApiResult<Capture> {
            calls += "labels"
            lastLabelsJson = labelsJson
            return labelsResult
        }

        override suspend fun uploadCaptureAudio(captureId: String, gzippedPcm: ByteArray): ApiResult<Capture> {
            calls += "audio"
            uploadedBytes = gzippedPcm
            return uploadResult
        }
    }

    private fun uploader(api: FakeApi) =
        JournalUploader(api, nowIso = { "2026-08-30T10:00:00Z" }, deviceId = "watch-1")

    @Test
    fun withoutConsentNoNetworkCallAtAll() = runTest {
        val api = FakeApi()
        assertFalse(uploader(api).upload(snapshot(), consentConfirmed = false))
        assertEquals(emptyList(), api.calls, "no consent artifact -> no upload, structurally")
    }

    @Test
    fun labelsGoBeforeAudioAndCarryTheJournalMarker() = runTest {
        val api = FakeApi()
        assertTrue(uploader(api).upload(snapshot(), consentConfirmed = true))
        // Labels-before-audio is load-bearing: the server's journal hook fires off the audio
        // upload success path and only sees labels already present at that moment.
        assertEquals(listOf("create", "labels", "audio"), api.calls)
        val labels = Json.parseToJsonElement(assertNotNull(api.lastLabelsJson)).jsonObject
        assertTrue(labels.getValue("journal").jsonPrimitive.boolean)
        assertEquals(300.0, labels.getValue("interval_s").jsonPrimitive.double)
        assertEquals(true, api.lastAttested, "attested mirrors the real consent, never a literal")
    }

    @Test
    fun labelsFailureAbortsBeforeAnyAudioIsUploaded() = runTest {
        val api = FakeApi(labelsResult = ApiResult.Failure(503, "labels unavailable"))
        assertFalse(uploader(api).upload(snapshot(), consentConfirmed = true))
        assertEquals(listOf("create", "labels"), api.calls, "an unlabelable capture must not receive audio")
    }

    @Test
    fun pcmIsGzippedOnTheWire() = runTest {
        val api = FakeApi()
        val snap = snapshot() // 5s of silence, highly compressible
        assertTrue(uploader(api).upload(snap, consentConfirmed = true))
        assertTrue(assertNotNull(api.uploadedBytes).size < snap.pcm.size)
    }

    @Test
    fun createFailureIsSoftFalse() = runTest {
        val api = FakeApi(createResult = ApiResult.Failure(401, "not signed in"))
        assertFalse(uploader(api).upload(snapshot(), consentConfirmed = true))
        assertEquals(listOf("create"), api.calls)
    }

    @Test
    fun audioFailureIsSoftFalse() = runTest {
        val api = FakeApi(uploadResult = ApiResult.Failure(503, "storage not configured"))
        assertFalse(uploader(api).upload(snapshot(), consentConfirmed = true))
    }

    @Test
    fun emptySnapshotNeverTouchesTheNetwork() = runTest {
        val api = FakeApi()
        assertFalse(uploader(api).upload(snapshot(pcm = ByteArray(0)), consentConfirmed = true))
        assertEquals(emptyList(), api.calls)
    }
}
