package app.gauge.wear.emotion

import app.gauge.wear.haptics.FakeVibratorPort
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.net.EpisodeWsClient
import app.gauge.wear.ui.emotionFootnote
import app.gauge.wear.ui.emotionHeadline
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer

class EmotionDevTest {
    private fun parse(s: String) = parseEmotionFrame(Json.parseToJsonElement(s).jsonObject)

    @Test fun `parses a phone emotion frame`() {
        val f = parse("""{"type":"emotion","v":1,"id":"w1","t":1700000000000,"label":"anger","prob":0.71,"nudge":true,"paused":false,"suggestion":"Slow down?"}""")!!
        assertEquals("anger", f.label); assertEquals(0.71, f.prob); assertTrue(f.nudge); assertEquals("Slow down?", f.suggestion)
    }

    @Test fun `malformed frames are ignored, never shown or buzzed`() {
        assertNull(parse("""{"type":"emotion","label":"rage","prob":0.9}"""))
        assertNull(parse("""{"type":"emotion","label":"anger","prob":7}"""))
        assertNull(parse("""{"type":"emotion","label":"anger"}"""))
        assertNull(parse("""{"type":"emotion","prob":0.5}"""))
    }

    @Test fun `a nudge frame buzzes, a quiet one only updates the screen`() {
        val m = EmotionDevModel()
        assertFalse(m.onFrame(EmotionFrame("a", 1, "neutral", 0.9, nudge = false, paused = false, suggestion = null)))
        assertEquals("neutral", m.state.value.label)
        assertTrue(m.onFrame(EmotionFrame("b", 2, "anger", 0.7, nudge = true, paused = false, suggestion = "Slow down?")))
        assertEquals("b", m.state.value.lastNudgeId)
        assertEquals("Slow down?", m.state.value.suggestion)
        assertTrue(m.state.value.active)
    }

    @Test fun `a paused phone never buzzes the wrist`() {
        val m = EmotionDevModel()
        assertFalse(m.onFrame(EmotionFrame("b", 2, "anger", 0.7, nudge = true, paused = true, suggestion = null)))
        assertTrue(m.state.value.paused)
    }

    @Test fun `controls are encoded for the server queue and carry the latest nudge id`() {
        val sent = mutableListOf<String>()
        var now = 1_000_000L
        val m = EmotionDevModel(nowMs = { now })
        m.sender = { sent.add(it); true }
        m.onFrame(EmotionFrame("n1", 1, "fear", 0.8, nudge = true, paused = false, suggestion = null))
        assertTrue(m.flag(EmotionFlag.BAD))
        val j = Json.parseToJsonElement(sent.last()).jsonObject
        assertEquals("emotion_ctl", j["type"]!!.jsonPrimitive.content)
        assertEquals("flag", j["action"]!!.jsonPrimitive.content)
        assertEquals("bad", j["flag"]!!.jsonPrimitive.content)
        assertEquals("n1", j["nudge_id"]!!.jsonPrimitive.content)
        assertEquals(EmotionFlag.BAD, m.state.value.flagged)

        // ignore / good / pause / resume wire values
        m.flag(EmotionFlag.IGNORE); assertTrue(sent.last().contains("\"ignore\""))
        m.flag(EmotionFlag.GOOD); assertTrue(sent.last().contains("\"good\""))
        m.setPaused(true); assertTrue(sent.last().contains("\"pause\""))
        m.setPaused(false); assertTrue(sent.last().contains("\"resume\""))
    }

    @Test fun `a stale nudge is not attributed to a flag`() {
        val sent = mutableListOf<String>()
        var now = 0L
        val m = EmotionDevModel(nowMs = { now })
        m.sender = { sent.add(it); true }
        m.onFrame(EmotionFrame("old", 1, "fear", 0.8, nudge = true, paused = false, suggestion = null))
        now += 11 * 60_000L
        m.flag(EmotionFlag.BAD)
        assertFalse(sent.last().contains("nudge_id"))
    }

    @Test fun `with no live socket the controls report failure instead of pretending`() {
        val m = EmotionDevModel()
        assertFalse(m.setPaused(true))
        assertTrue(m.state.value.sendFailed)
        assertEquals("Not sent - no connection to the phone", emotionFootnote(m.state.value))
        m.sender = { false }
        assertFalse(m.flag(EmotionFlag.GOOD))
        assertNull(m.state.value.flagged)
    }

    @Test fun `screen text is honest before any frame and when paused`() {
        assertEquals("Waiting for the phone", emotionHeadline(EmotionUi()))
        assertEquals("Paused", emotionHeadline(EmotionUi(active = true, label = "anger", prob = 0.5, paused = true)))
        assertEquals("anger 71%", emotionHeadline(EmotionUi(active = true, label = "anger", prob = 0.71)))
    }

    @Test fun `emotion buzz is a double tap that touches no nudge state`() {
        val vib = FakeVibratorPort()
        val d = HapticDirector(vib)
        d.playEmotionNudge()
        assertEquals(1, vib.calls.size)
        assertEquals(listOf(0L, 90L, 70L, 90L), vib.calls[0].timingsMs.toList())
        assertEquals(0, d.reminderLevel())
    }

    @Test fun `the socket client decodes emotion frames and sends ctl frames`() {
        val server = MockWebServer()
        val got = ArrayDeque<String>()
        val frames = mutableListOf<EmotionFrame>()
        val latch = CountDownLatch(2)
        server.enqueue(MockResponse().withWebSocketUpgrade(object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: okhttp3.Response) {
                ws.send("""{"type":"emotion","v":1,"id":"w9","t":1,"label":"sadness","prob":0.66,"nudge":true,"paused":false,"suggestion":null}""")
                ws.send("""{"type":"emotion","label":"nonsense","prob":0.9}""")
            }
            override fun onMessage(ws: WebSocket, text: String) { got.add(text); latch.countDown() }
        }))
        val client = EpisodeWsClient(server.url("/").toString().replace("http", "ws").trimEnd('/'), "acct")
        val seen = CountDownLatch(1)
        client.open("e1", object : EpisodeWsClient.Listener {
            override fun onVectorEvent(e: app.gauge.shared.VectorEvent) {}
            override fun onNudge(n: app.gauge.shared.NudgeEvent) {}
            override fun onEmotion(e: EmotionFrame) { frames.add(e); seen.countDown() }
            override fun onEpisodeSaved(id: String) {}
            override fun onFailure(t: Throwable) {}
            override fun onClosed() {}
        })
        assertTrue(seen.await(5, TimeUnit.SECONDS))
        assertEquals(1, frames.size, "the malformed frame must be dropped")
        assertEquals("sadness", frames[0].label)
        assertTrue(client.sendEmotionCtl(EmotionCtl.flag(EmotionFlag.GOOD, "w9")))
        assertTrue(client.sendEmotionCtl(EmotionCtl.pause()))
        assertTrue(latch.await(5, TimeUnit.SECONDS))
        assertTrue(got.first().contains("\"emotion_ctl\"") && got.first().contains("\"w9\""))
        client.cancel()
        assertFalse(client.sendEmotionCtl(EmotionCtl.resume()), "no socket -> not sent")
        server.shutdown()
    }
}
