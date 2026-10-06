package app.gauge.wear.net

import app.gauge.shared.NudgeEvent
import app.gauge.shared.PositiveEvent
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

    @Test fun decodesPositiveFrame() {
        // The praise lane. It arrives as its OWN frame type rather than a nudge with a code,
        // because a nudge carries a channel and a level and feeds the escalation machinery —
        // and praise has neither.
        val server = MockWebServer()
        val positives = mutableListOf<PositiveEvent>()
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"positive","code":"E","t":41.5}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e3", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}
            override fun onNudge(n: NudgeEvent) {}
            override fun onPositive(p: PositiveEvent) { positives.add(p); latch.countDown() }
            override fun onEpisodeSaved(id: String) {}
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("E", positives.single().code)
        assertEquals(41.5, positives.single().t)
        server.shutdown()
    }

    @Test fun positiveFrameIsIgnoredByAListenerThatDoesNotWantIt() {
        // The default no-op on the interface: an older listener keeps working and simply never
        // hears praise, instead of the client crashing on a frame it wasn't written for.
        val server = MockWebServer(); var saved: String? = null; val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"positive","code":"E","t":1.0}""")
                ws.send("""{"type":"live_session_saved","live_session_id":"e4","status":"captured"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e4", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS)); assertEquals("e4", saved)
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

    // --- 2026-09-25 protocol parity: the three frames the client used to drop ------------------

    // live_session_saved.status. The server sends "companion" for a Tier B socket that persisted
    // nothing by design, "companion_hr" for one that kept only heart rate, and "captured" for a
    // real mic episode (server/watch/routers/ws.py's `end` handling). Until now the client read
    // only the id, so a companion day and a captured session were indistinguishable on-watch.
    @Test fun liveSessionSavedStatusReachesTheListener() {
        val server = MockWebServer()
        var saved: Pair<String, String?>? = null
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"live_session_saved","live_session_id":"companion-20260925-a","status":"companion"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("companion-20260925-a", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) { throw AssertionError("the client must dispatch the status-carrying overload, not the id-only one") }
            override fun onEpisodeSaved(id: String, status: String?) { saved = id to status; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("companion-20260925-a" to "companion", saved)
        server.shutdown()
    }

    // A saved frame with NO status (the server never sends one today, but the contract is honest
    // about the field being absent) reports null rather than inventing "captured".
    @Test fun liveSessionSavedWithoutStatusReportsNull() {
        val server = MockWebServer()
        var saved: Pair<String, String?>? = null
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"live_session_saved","live_session_id":"e9"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e9", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) {}
            override fun onEpisodeSaved(id: String, status: String?) { saved = id to status; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("e9" to null, saved)
        server.shutdown()
    }

    // companion_ack: the server's answer to the {"type":"companion"} hello (ws.py's `companion`
    // branch, pinned by server/tests/watch/test_companion_ws.py). It is the only confirmation
    // that the socket is registered as a no-persistence companion; the client used to drop it
    // as an unknown type and assume success.
    @Test fun companionAckIsDispatched() {
        val server = MockWebServer()
        var acks = 0
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onMessage(ws: WebSocket, text: String) {
                if (text == """{"type":"companion"}""") ws.send("""{"type":"companion_ack"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("c1", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) {}
            override fun onCompanionAck() { acks++; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        c.sendCompanionHello()
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals(1, acks)
        c.cancel(); server.shutdown()
    }

    // error frames: {"type":"error","detail":"malformed_json"|"unknown_type"} — matched and then
    // silently discarded until 2026-09-25 (EpisodeWsClient.kt's old `"error" -> { }` branch). The
    // server keeps the socket open after one, so the teeth here are BOTH that the detail reaches
    // the listener AND that the frame after it still arrives — an error is a message, not a drop.
    @Test fun errorFrameDetailIsDispatchedAndSocketStaysOpen() {
        val server = MockWebServer()
        val errors = mutableListOf<String>(); var saved: String? = null; var failures = 0
        val latch = CountDownLatch(3)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"error","detail":"unknown_type"}""")
                ws.send("""{"type":"error","detail":"malformed_json"}""")
                ws.send("""{"type":"live_session_saved","live_session_id":"e10","status":"captured"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e10", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onServerError(detail: String) { errors.add(detail); latch.countDown() }
            override fun onFailure(t: Throwable) { failures++ }; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals(listOf("unknown_type", "malformed_json"), errors)
        assertEquals("e10", saved, "the frame AFTER the errors must still arrive: an error frame is not a disconnect")
        assertEquals(0, failures)
        server.shutdown()
    }

    // An error frame with no detail (not something the server sends, but the field is what the
    // listener keys on) still reaches the listener rather than being dropped for being malformed.
    @Test fun errorFrameWithoutDetailReportsUnknown() {
        val server = MockWebServer()
        var detail: String? = null
        val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) { ws.send("""{"type":"error"}""") }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e11", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) {}
            override fun onServerError(d: String) { detail = d; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertEquals("unknown", detail)
        server.shutdown()
    }

    // A listener written before 2026-09-25 (id-only onEpisodeSaved, no onCompanionAck, no
    // onServerError) keeps working: the defaults route the saved id through and swallow the
    // other two. The existing tests above already prove the id path; this pins the two no-ops.
    @Test fun olderListenerIgnoresAckAndErrorAndStillGetsSavedId() {
        val server = MockWebServer(); var saved: String? = null; val latch = CountDownLatch(1)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, r: Response) {
                ws.send("""{"type":"companion_ack"}""")
                ws.send("""{"type":"error","detail":"unknown_type"}""")
                ws.send("""{"type":"live_session_saved","live_session_id":"e12","status":"companion_hr"}""")
            }
        }))
        val c = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "a")
        c.open("e12", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: VectorEvent) {}; override fun onNudge(n: NudgeEvent) {}
            override fun onEpisodeSaved(id: String) { saved = id; latch.countDown() }
            override fun onFailure(t: Throwable) {}; override fun onClosed() {}
        })
        assertTrue(latch.await(5, TimeUnit.SECONDS)); assertEquals("e12", saved)
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
