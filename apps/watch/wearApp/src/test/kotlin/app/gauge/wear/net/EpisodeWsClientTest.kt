package app.gauge.wear.net

import app.gauge.shared.NudgeEvent
import app.gauge.shared.VectorEvent
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.ByteString

class EpisodeWsClientTest {
    @Test fun sendsBinaryAndDecodesNudgeAndSaved() {
        val server = MockWebServer()
        val received = ArrayDeque<String>()
        val nudges = mutableListOf<NudgeEvent>(); var saved: String? = null
        val latch = CountDownLatch(2)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onMessage(ws: WebSocket, bytes: ByteString) {
                received.add("binary:${bytes.size}")
                ws.send("""{"type":"nudge","channel":"A","level":2,"t":1.0,"vectors":["yelling"]}""")
            }
            override fun onMessage(ws: WebSocket, text: String) {
                received.add(text)
                if (text.contains("\"end\"")) ws.send("""{"type":"live_session_saved","live_session_id":"e1","status":"captured"}""")
            }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e1", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) { nudges.add(n); latch.countDown() }
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onFailure(t: Throwable) {} ; override fun onClosed() {}
        })
        client.sendPcmWindow(ByteArray(32000))
        client.end()
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        // Task A3: the client must speak the renamed live-session WS path, not the old episode one.
        val recorded = server.takeRequest()
        assertEquals("/ws/live-session/e1?account=acct", recorded.path)
        assertEquals(2, nudges.first().level); assertEquals("A", nudges.first().channel)
        assertEquals("e1", saved)
        assertTrue(received.any { it.startsWith("binary:32000") })
        server.shutdown()
    }

    @Test fun unknownFrameTypeIsIgnored() {
        val server = MockWebServer(); var saved: String? = null; val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"future_thing","x":1}"""); ws.send("""{"type":"live_session_saved","live_session_id":"e2","status":"captured"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e2", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS)); assertEquals("e2", saved)
        server.shutdown()
    }

    @Test fun sendsHrFrameAsJson() {
        val server = MockWebServer()
        val hrFrames = mutableListOf<String>()
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onMessage(ws: WebSocket, text: String) {
                hrFrames.add(text)
                latch.countDown()
            }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e1", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) {}
            override fun onFailure(t: Throwable) {}
            override fun onClosed() {}
        })
        client.sendHr(120.0, 2.0)
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        val obj = Json.parseToJsonElement(hrFrames.first()).jsonObject
        assertEquals("hr", obj.getValue("type").jsonPrimitive.content)
        assertEquals(120.0, obj.getValue("bpm").jsonPrimitive.double)
        assertEquals(2.0, obj.getValue("t").jsonPrimitive.double)
        server.shutdown()
    }

    @Test fun malformedFramesAreToleratedAndSocketStaysOpen() {
        val server = MockWebServer()
        var nudgeCalls = 0; var saved: String? = null; var failures = 0
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""not json{{{""")
                ws.send("""{"type":"nudge","level":"not-a-number"}""")
                ws.send("""{"type":"live_session_saved","live_session_id":"e3","status":"captured"}""")
            }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e3", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) { nudgeCalls++ }
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onFailure(t: Throwable) { failures++ }
            override fun onClosed() {}
        })
        // The teeth of this test: if a malformed frame killed the reader loop, this
        // live_session_saved (sent after the two bad frames, on the same still-open socket)
        // would never arrive and the latch would time out.
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("e3", saved)
        assertEquals(0, nudgeCalls)
        assertEquals(0, failures)
        server.shutdown()
    }

    // Task 7: a Listener implementation that itself throws must not kill the WebSocket — the
    // teeth of this test (as with malformedFramesAreToleratedAndSocketStaysOpen above) is that
    // live_session_saved, sent right after the throwing nudge frame on the same still-open socket,
    // still arrives.
    @Test fun listenerThrowInOnNudgeDoesNotKillSocket() {
        val server = MockWebServer()
        var saved: String? = null
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"nudge","channel":"A","level":2,"t":1.0,"vectors":["yelling"]}""")
                ws.send("""{"type":"live_session_saved","live_session_id":"e4","status":"captured"}""")
            }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e4", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) {
                throw RuntimeException("listener boom")
            }
            override fun onEpisodeSaved(id: String) {
                saved = id
                latch.countDown()
            }
            override fun onFailure(t: Throwable) {}
            override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("e4", saved)
        server.shutdown()
    }

    private fun listenerStub() = object : EpisodeWsClient.Listener {
        override fun onVectorEvent(e: VectorEvent) {}
        override fun onNudge(n: NudgeEvent) {}
        override fun onEpisodeSaved(id: String) {}
        override fun onFailure(t: Throwable) {}
        override fun onClosed() {}
    }

    // Tier B: a paired device token upgrades the URL to the server-preferred `?token=` form —
    // URL-encoded, since the token is an opaque string that may carry reserved characters.
    @Test fun tokenAuthUrlPreferredOverLegacyAccount() {
        val server = MockWebServer()
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {}))
        val client = EpisodeWsClient(
            server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct", token = "tok+en/1",
        )
        client.open("e6", listenerStub())
        val recorded = server.takeRequest()
        assertEquals("/ws/live-session/e6?token=tok%2Ben%2F1", recorded.path)
        client.cancel()
        server.shutdown()
    }

    // Tier B: no token (unpaired watch) keeps the legacy `?account=` URL byte-identical.
    @Test fun noTokenFallsBackToLegacyAccountUrl() {
        val server = MockWebServer()
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {}))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e6", listenerStub())
        assertEquals("/ws/live-session/e6?account=acct", server.takeRequest().path)
        client.cancel()
        server.shutdown()
    }

    @Test fun companionHelloAndHeartbeatFramesAreSentAsJson() {
        val server = MockWebServer()
        val frames = mutableListOf<String>()
        val latch = CountDownLatch(2)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onMessage(ws: WebSocket, text: String) {
                frames.add(text)
                latch.countDown()
            }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e7", listenerStub())
        client.sendCompanionHello()
        client.sendHeartbeat()
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("""{"type":"companion"}""", frames[0])
        assertEquals("""{"type":"heartbeat"}""", frames[1])
        client.cancel()
        server.shutdown()
    }

    // Task 12.5: an HTTP-level handshake rejection (no WebSocket upgrade at all) must surface its
    // status code in the Throwable reaching the Listener — this is the exact diagnostic payload
    // Task 12.5 exists to produce (see EpisodeWsClient.onFailure's "handshake ${response.code}"
    // wrap).
    @Test fun handshakeRejectionIncludesStatusCodeInFailureMessage() {
        val server = MockWebServer()
        server.enqueue(MockResponse().setResponseCode(500)) // no .withWebSocketUpgrade: reject at handshake
        val failure = mutableListOf<Throwable>()
        val latch = CountDownLatch(1)
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        client.open("e5", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) {}
            override fun onFailure(t: Throwable) { failure.add(t); latch.countDown() }
            override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertTrue(
            failure.single().message.orEmpty().contains("handshake 500"),
            "expected the failure message to contain \"handshake 500\", got: ${failure.singleOrNull()?.message}",
        )
        server.shutdown()
    }
}
