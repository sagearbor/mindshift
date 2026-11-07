package app.gauge.wear.emotion

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive

/**
 * Emotion dev tester (docs/plans/2026-11-07-emotion-dev-mode.md), watch side.
 *
 * The PHONE runs the two emotion models and decides on a buzz; the watch only (1) buzzes and shows
 * the latest emotion label when the phone says so, and (2) sends pause/resume and the three flag
 * verbs back. Both directions ride the existing server socket ([app.gauge.wear.net.EpisodeWsClient]):
 * an `emotion` frame comes down, an `emotion_ctl` frame goes up. Nothing here touches audio.
 *
 * Phase 2 (tiny on the watch itself) replaces the SOURCE of [EmotionFrame]s — a local engine
 * instead of the socket — and leaves [EmotionDevModel], the screen and the ctl verbs as they are.
 */
data class EmotionFrame(
    val id: String,
    val t: Long,
    val label: String,
    val prob: Double,
    val nudge: Boolean,
    val paused: Boolean,
    val suggestion: String?,
)

val EMOTION_LABELS = setOf("anger", "disgust", "fear", "happiness", "neutral", "sadness")

/** Decode the server's `emotion` frame; null (ignored) for anything malformed — a bad frame
 *  must never buzz or show a made-up emotion. */
fun parseEmotionFrame(obj: JsonObject): EmotionFrame? {
    val label = obj["label"]?.jsonPrimitive?.contentOrNull ?: return null
    if (label !in EMOTION_LABELS) return null
    val prob = obj["prob"]?.jsonPrimitive?.doubleOrNull ?: return null
    if (prob.isNaN() || prob < 0.0 || prob > 1.0) return null
    return EmotionFrame(
        id = obj["id"]?.jsonPrimitive?.contentOrNull ?: "",
        t = obj["t"]?.jsonPrimitive?.doubleOrNull?.toLong() ?: 0L,
        label = label,
        prob = prob,
        nudge = obj["nudge"]?.jsonPrimitive?.booleanOrNull ?: false,
        paused = obj["paused"]?.jsonPrimitive?.booleanOrNull ?: false,
        suggestion = obj["suggestion"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() },
    )
}

/** The three flags. `wire` is what the server queue and the phone read. */
enum class EmotionFlag(val wire: String, val label: String) {
    BAD("bad", "Wrong"),
    GOOD("good", "Good"),
    IGNORE("ignore", "Ignore"),
}

/** Builders for the `emotion_ctl` frames the watch sends up the socket. */
object EmotionCtl {
    fun pause(): String = frame("pause")
    fun resume(): String = frame("resume")
    fun flag(flag: EmotionFlag, nudgeId: String?): String = frame("flag") {
        put("flag", JsonPrimitive(flag.wire))
        if (nudgeId != null) put("nudge_id", JsonPrimitive(nudgeId))
    }

    private fun frame(action: String, extra: (kotlinx.serialization.json.JsonObjectBuilder.() -> Unit)? = null): String =
        buildJsonObject {
            put("type", JsonPrimitive("emotion_ctl"))
            put("action", JsonPrimitive(action))
            extra?.invoke(this)
        }.toString()
}

data class EmotionUi(
    /** True once a frame has arrived this process — gates the Settings entry so the normal watch UI is unchanged. */
    val active: Boolean = false,
    val label: String? = null,
    val prob: Double? = null,
    val suggestion: String? = null,
    val paused: Boolean = false,
    val lastNudgeId: String? = null,
    val lastNudgeAtMs: Long? = null,
    /** What the wearer last tapped for the latest nudge, so the screen can confirm it. */
    val flagged: EmotionFlag? = null,
    /** The last control could not be sent (offline) — shown, never hidden. */
    val sendFailed: Boolean = false,
)

/**
 * Pure state holder (JVM-testable). [sender] returns true only when the frame was actually handed to
 * a live socket; on false the UI says so instead of pretending the phone heard.
 */
class EmotionDevModel(private val nowMs: () -> Long = { System.currentTimeMillis() }) {
    private val _state = MutableStateFlow(EmotionUi())
    val state: StateFlow<EmotionUi> = _state.asStateFlow()

    @Volatile var sender: ((String) -> Boolean)? = null

    /** A nudge is attributed to a flag only while it is this fresh. */
    private val nudgeFreshMs = 10 * 60_000L

    /** Apply a phone frame. Returns true when the watch should BUZZ for it. A paused phone never buzzes. */
    fun onFrame(f: EmotionFrame): Boolean {
        _state.value = _state.value.let { s ->
            s.copy(
                active = true,
                label = f.label,
                prob = f.prob,
                suggestion = if (f.nudge) f.suggestion else s.suggestion.takeIf { s.label == f.label },
                paused = f.paused,
                lastNudgeId = if (f.nudge) f.id else s.lastNudgeId,
                lastNudgeAtMs = if (f.nudge) nowMs() else s.lastNudgeAtMs,
                flagged = if (f.nudge) null else s.flagged,
                sendFailed = false,
            )
        }
        return f.nudge && !f.paused
    }

    private fun send(json: String): Boolean {
        val ok = sender?.invoke(json) ?: false
        _state.value = _state.value.copy(sendFailed = !ok)
        return ok
    }

    /** The phone confirms the new paused state with its next frame; until then we only record the failure, never guess. */
    fun setPaused(paused: Boolean): Boolean = send(if (paused) EmotionCtl.pause() else EmotionCtl.resume())

    fun flag(flag: EmotionFlag): Boolean {
        val s = _state.value
        val fresh = s.lastNudgeAtMs?.let { nowMs() - it <= nudgeFreshMs } == true
        val ok = send(EmotionCtl.flag(flag, if (fresh) s.lastNudgeId else null))
        if (ok) _state.value = _state.value.copy(flagged = flag)
        return ok
    }
}

/** Process-wide instance the socket listener feeds and the screen reads. */
object EmotionDevBus {
    val model = EmotionDevModel()
}
