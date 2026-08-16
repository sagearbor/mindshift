package app.gauge.wear.auth

import app.gauge.shared.wireJson
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer

class DevicePairingClientTest {
    private lateinit var server: MockWebServer
    private lateinit var client: DevicePairingClient

    @BeforeTest
    fun setUp() {
        server = MockWebServer()
        server.start()
        client = DevicePairingClient(baseUrl = server.url("/").toString().trimEnd('/'))
    }

    @AfterTest
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun startDecodesAPairingCode() = runBlocking {
        server.enqueue(
            MockResponse().setBody(
                """{"code":"K7QP2M","pairing_id":"pid-1","expires_at":"2026-08-04T12:10:00Z"}""",
            ).setResponseCode(200),
        )
        val start = client.start()
        assertEquals("K7QP2M", start?.code)
        assertEquals("pid-1", start?.pairingId)
        val recorded = server.takeRequest()
        assertEquals("POST", recorded.method)
        assertEquals("/me/pair/start", recorded.path)
    }

    @Test
    fun startReturnsNullOnServerError() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(500))
        assertNull(client.start())
    }

    @Test
    fun pollDecodesPendingStatus() = runBlocking {
        server.enqueue(MockResponse().setBody("""{"status":"pending"}""").setResponseCode(200))
        val status = client.poll("pid-1")
        assertEquals("pending", status?.status)
        val recorded = server.takeRequest()
        assertEquals("GET", recorded.method)
        assertEquals("/me/pair/status?pairing_id=pid-1", recorded.path)
    }

    @Test
    fun pollDecodesClaimedStatusWithToken() = runBlocking {
        server.enqueue(
            MockResponse().setBody(
                """{"status":"claimed","account_id":"acct1","device_token":"secret-token"}""",
            ).setResponseCode(200),
        )
        val status = client.poll("pid-1")
        assertEquals("claimed", status?.status)
        assertEquals("secret-token", status?.deviceToken)
    }

    @Test
    fun pollReturnsNullOnMalformedJson() = runBlocking {
        server.enqueue(MockResponse().setBody("not json").setResponseCode(200))
        assertNull(client.poll("pid-1"))
    }

    @Test
    fun startReportsSwallowedTransportError() = runBlocking {
        val transportServer = MockWebServer()
        transportServer.start()
        val url = transportServer.url("/").toString().trimEnd('/')
        transportServer.shutdown() // connection refused from now on
        val errors = mutableListOf<String>()
        val transportClient = DevicePairingClient(baseUrl = url, onSwallowedError = { errors.add(it) })
        assertNull(transportClient.start())
        assertTrue(errors.single().startsWith("start "))
    }

    @Test
    fun startReportsNon2xx() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(500))
        val errors = mutableListOf<String>()
        val errorClient = DevicePairingClient(
            baseUrl = server.url("/").toString().trimEnd('/'),
            onSwallowedError = { errors.add(it) },
        )
        assertNull(errorClient.start())
        assertTrue(errors.single().contains("500"))
    }

    @Test
    fun startReportsMalformedJson() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(200).setBody("not json"))
        val errors = mutableListOf<String>()
        val errorClient = DevicePairingClient(
            baseUrl = server.url("/").toString().trimEnd('/'),
            onSwallowedError = { errors.add(it) },
        )
        assertNull(errorClient.start())
        assertTrue(errors.single().startsWith("start "))
    }
}
