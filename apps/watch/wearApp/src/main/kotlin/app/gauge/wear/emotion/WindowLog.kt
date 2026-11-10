package app.gauge.wear.emotion

import java.io.File
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/** One classified window. Numbers only - no audio, no transcript, ever. */
data class WindowRow(
    /** Epoch ms when the window was classified. */
    val t: Long,
    /** Raw TINY probabilities, [LocalEmotion.LABELS] order, 4 decimals. */
    val p: List<Double>,
    /** Wall time of log-mel + inference for this window, ms (the watch-CPU latency measurement). */
    val ms: Int,
    /** Battery percent when classified, -1 if unknown (the battery-cost measurement). */
    val battery: Int,
    /** True when this window earned a buzz. */
    val nudged: Boolean,
    /** The gate's verdict ("" when nudged). */
    val reason: String,
) {
    fun toJson(): JsonObject = buildJsonObject {
        put("t", JsonPrimitive(t))
        put("p", buildJsonArray { p.forEach { add(JsonPrimitive(it)) } })
        put("ms", JsonPrimitive(ms))
        put("b", JsonPrimitive(battery))
        put("n", JsonPrimitive(nudged))
        if (reason.isNotEmpty()) put("why", JsonPrimitive(reason))
    }

    companion object {
        fun round4(v: Float): Double = Math.round(v * 1e4) / 1e4

        fun fromJson(o: JsonObject): WindowRow? = runCatching {
            WindowRow(
                t = o["t"]!!.jsonPrimitive.longOrNull!!,
                p = o["p"]!!.jsonArray.map { it.jsonPrimitive.doubleOrNull!! }.also { require(it.size == 6) },
                ms = o["ms"]?.jsonPrimitive?.intOrNull ?: 0,
                battery = o["b"]?.jsonPrimitive?.intOrNull ?: -1,
                nudged = o["n"]?.jsonPrimitive?.booleanOrNull ?: false,
                reason = o["why"]?.jsonPrimitive?.contentOrNull ?: "",
            )
        }.getOrNull()
    }
}

/**
 * Rolling on-watch log of per-window probabilities (`emotion-windows.jsonl`, one JSON object per
 * line). Bounded: past [maxBytes] the oldest half is dropped. A sidecar cursor file holds the `t` of
 * the last row the phone side acknowledged, so [unsent] survives a restart and a trim. Single
 * writer (the service's loop thread); [unsent] / [markSent] are called from the sync path, so every
 * method is synchronized.
 */
class WindowLog(dir: File, private val maxBytes: Long = DEFAULT_MAX_BYTES) {
    private val file = File(dir, "emotion-windows.jsonl")
    private val cursorFile = File(dir, "emotion-windows.cursor")

    init { dir.mkdirs() }

    @Synchronized
    fun append(row: WindowRow) {
        runCatching {
            file.appendText(row.toJson().toString() + "\n")
            if (file.length() > maxBytes) trim()
        }
    }

    private fun trim() {
        val lines = file.readLines()
        val keep = lines.drop(lines.size / 2)
        file.writeText(keep.joinToString("\n", postfix = "\n"))
    }

    @Synchronized
    fun all(): List<WindowRow> {
        if (!file.isFile) return emptyList()
        return file.readLines().mapNotNull { line ->
            runCatching { WindowRow.fromJson(Json.parseToJsonElement(line).jsonObject) }.getOrNull()
        }
    }

    @Synchronized
    fun cursor(): Long = runCatching { cursorFile.readText().trim().toLong() }.getOrDefault(0L)

    /** Rows newer than the acknowledged cursor, oldest first, at most [limit]. */
    @Synchronized
    fun unsent(limit: Int = SYNC_BATCH): List<WindowRow> {
        val c = cursor()
        return all().filter { it.t > c }.take(limit)
    }

    @Synchronized
    fun markSent(throughT: Long) {
        runCatching { cursorFile.writeText(throughT.toString()) }
    }

    @Synchronized
    fun clear() {
        runCatching { file.delete() }
        runCatching { cursorFile.delete() }
    }

    companion object {
        const val DEFAULT_MAX_BYTES = 1_000_000L // ~9k rows ~ 7.5 h of continuous 3 s windows
        const val SYNC_BATCH = 200

        /** The body of `POST /emotion/watch-windows`. */
        fun syncBody(rows: List<WindowRow>): String = buildJsonObject {
            put("v", JsonPrimitive(1))
            put("rows", JsonArray(rows.map { it.toJson() }))
        }.toString()
    }
}
