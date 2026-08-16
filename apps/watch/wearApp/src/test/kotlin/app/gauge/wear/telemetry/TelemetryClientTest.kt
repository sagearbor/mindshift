package app.gauge.wear.telemetry

import app.gauge.shared.TelemetryEventOut
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer

class TelemetryClientTest {
    @Test fun postBlockingSendsBatchJson() {
        val server = MockWebServer(); server.enqueue(MockResponse().setBody("""{"stored":1,"dropped":0}"""))
        server.start()
        val c = TelemetryClient(server.url("/").toString().trimEnd('/'), "dev-1", "0.1.1")
        val ok = c.postBlocking(listOf(TelemetryEventOut("crash", "Handler", "boom", "trace", "ts1")))
        assertTrue(ok)
        val req = server.takeRequest()
        assertEquals("/telemetry", req.path)
        val body = req.body.readUtf8()
        assertTrue(body.contains(""""device":"dev-1""""))
        assertTrue(body.contains(""""app_version":"0.1.1""""))
        assertTrue(body.contains(""""stack":"trace""""))
        server.shutdown()
    }
    @Test fun postBlockingNeverThrowsOnDeadServer() {
        val c = TelemetryClient("http://127.0.0.1:1", "d", "v")   // nothing listens
        assertFalse(c.postBlocking(listOf(TelemetryEventOut("info", "t", "m", null, "ts")), timeoutMs = 500))
    }
    @Test fun emptyListIsNoop() {
        val c = TelemetryClient("http://127.0.0.1:1", "d", "v")
        assertTrue(c.postBlocking(emptyList()))    // no network touched, success
    }
}
