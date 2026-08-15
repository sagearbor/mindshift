package app.gauge.wear.net

import app.gauge.shared.NudgeEvent
import app.gauge.shared.VectorEvent
import app.gauge.shared.wireJson
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.IOException
import java.util.concurrent.TimeUnit
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString

/**
 * OkHttp WebSocket client that streams episode audio/HR to the Gauge backend
 * and dispatches inbound vector/nudge/episode-saved frames to a [Listener].
 *
 * This class owns no threading of its own: [open] hands control to OkHttp,
 * whose reader thread drives every [Listener] callback (see the KDoc on
 * [Listener] below). The controller (wearApp Task 7) is responsible for
 * marshalling callbacks onto whatever thread its own state needs.
 */
class EpisodeWsClient(
    private val baseWsUrl: String,
    private val account: String,
    // A default client with no timeouts would hang indefinitely on a dead/unreachable backend
    // rather than ever calling the listener's onFailure — the controller's fail-soft path (Task 7)
    // depends on onFailure actually firing to flip `online` false and fall back to local nudges.
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build(),
) {
    /**
     * Callbacks fired by [EpisodeWsClient] as inbound WebSocket frames arrive
     * and as the connection's lifecycle changes.
     *
     * IMPORTANT: every method here is called on OkHttp's WebSocket reader
     * thread, never the caller's thread and never a fixed thread across
     * calls. Implementations that touch shared/mutable state MUST
     * synchronize themselves (or hop to their own thread/handler) — this
     * client performs no synchronization or dispatching on their behalf.
     */
    interface Listener {
        fun onVectorEvent(e: VectorEvent)
        fun onNudge(n: NudgeEvent)
        fun onEpisodeSaved(id: String)
        fun onFailure(t: Throwable)
        fun onClosed()
    }

    // Written by whatever thread calls open()/cancel(), read by whatever thread calls
    // sendPcmWindow()/sendHr()/end() — @Volatile gives cross-thread visibility without
    // pulling in a lock (the controller in Task 7 owns any higher-level synchronization).
    @Volatile private var webSocket: WebSocket? = null

    /**
     * Opens the episode WebSocket and begins dispatching frames to [listener].
     *
     * Call at most once per instance — create a new [EpisodeWsClient] per episode rather than
     * reopening this one (Task 7's controller is responsible for enforcing this; calling [open]
     * twice would silently leak the first [WebSocket] since nothing here guards against it).
     */
    fun open(episodeId: String, listener: Listener) {
        val url = "$baseWsUrl/ws/episode/$episodeId?account=$account"
        val request = Request.Builder().url(url).build()
        webSocket = client.newWebSocket(
            request,
            object : WebSocketListener() {
                override fun onMessage(ws: WebSocket, text: String) {
                    dispatch(text, listener)
                }

                override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                    // Same contract as dispatch()'s own try/catch below: a throwing Listener must
                    // never escape onto OkHttp's reader thread.
                    try {
                        // OkHttp passes a non-null `response` when the failure is an HTTP-level
                        // handshake rejection (e.g. 403/404 before the socket ever upgrades) — the
                        // Listener interface only carries a Throwable, so fold the status code into
                        // its message rather than widening the interface shape.
                        val reported = if (response != null) {
                            IOException("handshake ${response.code}: ${t.message}", t)
                        } else {
                            t
                        }
                        listener.onFailure(reported)
                    } catch (e: Exception) {
                        println("EpisodeWsClient: onFailure listener dispatch failed: ${e.message}")
                    }
                }

                override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                    try {
                        listener.onClosed()
                    } catch (e: Exception) {
                        println("EpisodeWsClient: onClosed listener dispatch failed: ${e.message}")
                    }
                }
            },
        )
    }

    private fun dispatch(text: String, listener: Listener) {
        // A malformed/unparseable frame must NOT propagate out of this callback: it runs on
        // OkHttp's reader thread (see onMessage above), and any exception escaping it fails
        // the WebSocket outright, tearing down the whole episode connection over one bad
        // frame. Treat parse/decode failures the same as an unknown type: log and ignore.
        try {
            val obj = Json.parseToJsonElement(text).jsonObject
            when (obj["type"]?.jsonPrimitive?.content) {
                "vector_event" -> listener.onVectorEvent(wireJson.decodeFromString(VectorEvent.serializer(), text))
                "nudge" -> listener.onNudge(wireJson.decodeFromString(NudgeEvent.serializer(), text))
                "episode_saved" -> {
                    val id = obj["episode_id"]?.jsonPrimitive?.content
                    if (id != null) listener.onEpisodeSaved(id)
                }
                "error" -> { /* server-reported error frame; nothing to dispatch, avoid crashing. */ }
                else -> { /* unknown/future frame type: ignore silently per wire contract. */ }
            }
        } catch (e: Exception) {
            // Catches SerializationException (invalid JSON, or valid JSON that fails to
            // decode as its claimed type) as well as IllegalArgumentException (e.g. `.jsonObject`
            // on a JSON array/primitive). No android.util.Log here — this class is JVM-tested.
            println("EpisodeWsClient: ignoring unparseable inbound frame: ${e.message}")
        }
    }

    // NOTE (all three send methods below): OkHttp's WebSocket.send() enqueues and returns
    // immediately — a dropped/failed send is only observable later via Listener.onFailure/
    // onClosed, never synchronously from the call site.

    /** Sends a raw PCM16 audio window as a binary frame. */
    fun sendPcmWindow(bytes: ByteArray) {
        webSocket?.send(bytes.toByteString())
    }

    /** Sends a heart-rate sample as a `{"type":"hr",...}` text frame. */
    fun sendHr(bpm: Double, t: Double) {
        val frame = buildJsonObject {
            put("type", JsonPrimitive("hr"))
            put("bpm", JsonPrimitive(bpm))
            put("t", JsonPrimitive(t))
        }
        webSocket?.send(frame.toString())
    }

    /** Sends the `{"type":"end"}` text frame that signals episode completion. */
    fun end() {
        webSocket?.send("""{"type":"end"}""")
    }

    /** Cancels the underlying connection immediately, releasing OkHttp resources. */
    fun cancel() {
        webSocket?.cancel()
        webSocket = null
    }
}
