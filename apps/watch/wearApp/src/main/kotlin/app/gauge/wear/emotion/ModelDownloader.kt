package app.gauge.wear.emotion

import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * The pinned TINY model on Firebase Hosting (same host and mechanism as the speaker-ID ECAPA model).
 * scripts/web_deploy.sh greps [SHA256] and [FILE_NAME] out of THIS file and copies the local model
 * to dist/models/, so the pin and the deploy cannot drift. To change the model: update both
 * constants (the name carries the first 12 hex of the hash so a new model is a new URL).
 */
object TinyModelSpec {
    const val SHA256 = "ae4d75f72b2b98b281ddff7c96e9ebe469e0f30d676c22186b45d3b7cea41b70"
    const val FILE_NAME = "emotion-tiny-int8-ae4d75f72b2b.onnx"
    const val BASE_URL = "https://arborfam-hub.web.app/models/"
    val url: String get() = BASE_URL + FILE_NAME
}

fun sha256Hex(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

sealed interface DownloadResult {
    data object Ok : DownloadResult
    data class Failed(val reason: String) : DownloadResult
}

/**
 * Downloads the model into [destDir]/[LocalEmotion.MODEL_FILE] (the same place a sideload goes).
 * Streams to a temp file, then accepts it only if size > [LocalEmotion.MODEL_MIN_BYTES] AND the
 * sha256 equals the pin; a mismatch (e.g. Hosting's SPA index.html fallback) deletes the temp and
 * reports failure - a wrong file is never installed. The [open] seam makes it JVM-testable.
 */
class ModelDownloader(
    private val expectedSha256: String = TinyModelSpec.SHA256,
    private val open: (String) -> InputStream = { u ->
        (URL(u).openConnection() as HttpURLConnection).apply {
            connectTimeout = 15_000; readTimeout = 30_000
            if (responseCode != 200) throw java.io.IOException("HTTP $responseCode")
        }.inputStream
    },
) {
    fun download(destDir: File, url: String = TinyModelSpec.url, onProgress: (Long) -> Unit = {}): DownloadResult {
        destDir.mkdirs()
        val tmp = File(destDir, LocalEmotion.MODEL_FILE + ".part")
        return try {
            val md = MessageDigest.getInstance("SHA-256")
            var total = 0L
            open(url).use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n); md.update(buf, 0, n)
                        total += n; onProgress(total)
                    }
                }
            }
            if (total <= LocalEmotion.MODEL_MIN_BYTES) {
                tmp.delete(); return DownloadResult.Failed("too small ($total bytes)")
            }
            val got = md.digest().joinToString("") { "%02x".format(it) }
            if (got != expectedSha256) {
                tmp.delete(); return DownloadResult.Failed("checksum mismatch")
            }
            val dest = File(destDir, LocalEmotion.MODEL_FILE)
            dest.delete()
            if (!tmp.renameTo(dest)) { tmp.delete(); return DownloadResult.Failed("could not save file") }
            DownloadResult.Ok
        } catch (e: Exception) {
            tmp.delete()
            DownloadResult.Failed(e.message ?: e.javaClass.simpleName)
        }
    }
}
