package app.gauge.wear.net

import app.gauge.shared.CreateCaptureRequest
import kotlinx.coroutines.test.runTest
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer

class WatchApiClientTest {
    private lateinit var server: MockWebServer
    private lateinit var client: WatchApiClient

    @BeforeTest
    fun setUp() {
        server = MockWebServer()
        server.start()
        client = WatchApiClient(baseUrl = server.url("/").toString().trimEnd('/'), deviceToken = { "tok-123" })
    }

    @AfterTest
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun myStandingSendsBearerTokenAndDecodesResponse() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"account_id":"acct1","current":{"episodes":2,"calm":70.0,"nudges":1,"escalations":0},
                    "prior":{"episodes":1,"calm":60.0,"nudges":0,"escalations":0},
                    "delta_vs_self":10.0,"improving":true}""",
            ).setResponseCode(200),
        )
        val result = client.myStanding()
        assertTrue(result is ApiResult.Ok)
        assertEquals(70.0, (result as ApiResult.Ok).value.current.calm)
        val recorded = server.takeRequest()
        assertEquals("Bearer tok-123", recorded.getHeader("Authorization"))
        assertEquals("/me/standing?period_days=7", recorded.path)
    }

    @Test
    fun noTokenNeverTouchesTheNetwork() = runTest {
        val noTokenClient = WatchApiClient(baseUrl = server.url("/").toString().trimEnd('/'), deviceToken = { null })
        val result = noTokenClient.myStanding()
        assertTrue(result is ApiResult.Failure)
        assertEquals(0, server.requestCount)
    }

    @Test
    fun nonTwoxxBecomesFailureWithServerDetail() = runTest {
        server.enqueue(MockResponse().setBody("""{"detail":"not signed in"}""").setResponseCode(401))
        val result = client.myStanding()
        assertTrue(result is ApiResult.Failure)
        assertEquals("not signed in", (result as ApiResult.Failure).message)
        assertEquals(401, result.code)
    }

    @Test
    fun createCaptureSendsAttestedRequestBody() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"id":"cap1","account_id":"acct1","captured_at":"2026-08-04T10:00:00Z",
                    "received_at":"2026-08-04T10:00:01Z","duration_s":118.0,"status":"awaiting_audio"}""",
            ).setResponseCode(200),
        )
        val result = client.createCapture(
            CreateCaptureRequest(capturedAt = "2026-08-04T10:00:00Z", durationS = 118.0, attested = true),
        )
        assertTrue(result is ApiResult.Ok)
        assertEquals("cap1", (result as ApiResult.Ok).value.id)
        val recorded = server.takeRequest()
        assertEquals("POST", recorded.method)
        assertTrue(recorded.body.readUtf8().contains("\"attested\":true"))
    }

    @Test
    fun uploadCaptureAudioSetsGzipContentEncoding() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"id":"cap1","account_id":"acct1","captured_at":"2026-08-04T10:00:00Z",
                    "received_at":"2026-08-04T10:00:01Z","duration_s":118.0,"status":"stored"}""",
            ).setResponseCode(200),
        )
        val result = client.uploadCaptureAudio("cap1", byteArrayOf(1, 2, 3))
        assertTrue(result is ApiResult.Ok)
        val recorded = server.takeRequest()
        assertEquals("PUT", recorded.method)
        assertEquals("gzip", recorded.getHeader("Content-Encoding"))
        assertEquals("/captures/cap1/audio", recorded.path)
    }
}
