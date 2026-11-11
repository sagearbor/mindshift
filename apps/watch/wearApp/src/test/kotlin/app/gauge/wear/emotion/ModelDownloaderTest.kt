package app.gauge.wear.emotion

import java.io.ByteArrayInputStream
import java.io.IOException
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ModelDownloaderTest {
    private val good = ByteArray(5_100_000) { (it % 251).toByte() }
    private fun dl(bytes: ByteArray, sha: String = sha256Hex(good)) =
        ModelDownloader(sha) { ByteArrayInputStream(bytes) }

    @Test fun `verified download lands under the sideload filename`() {
        val dir = Files.createTempDirectory("m").toFile()
        assertEquals(DownloadResult.Ok, dl(good).download(dir, "u"))
        assertEquals(good.size.toLong(), java.io.File(dir, LocalEmotion.MODEL_FILE).length())
        assertTrue(findTinyModel(listOf(dir)) is ModelLookup.Ready)
        assertFalse(java.io.File(dir, LocalEmotion.MODEL_FILE + ".part").exists())
    }

    @Test fun `wrong content (e g the SPA index fallback) is rejected and nothing is installed`() {
        val dir = Files.createTempDirectory("m").toFile()
        val html = "<html>".repeat(1000).toByteArray()
        assertTrue(dl(html).download(dir, "u") is DownloadResult.Failed)
        assertTrue(dl(good.copyOf().also { it[10] = 99 }).download(dir, "u") is DownloadResult.Failed) // same size, bad hash
        assertFalse(java.io.File(dir, LocalEmotion.MODEL_FILE).exists())
        assertEquals(0, dir.listFiles()!!.size)
    }

    @Test fun `network failure is a Failed result`() {
        val dir = Files.createTempDirectory("m").toFile()
        val r = ModelDownloader("x") { throw IOException("offline") }.download(dir, "u")
        assertEquals(DownloadResult.Failed("offline"), r)
    }

    @Test fun `pin is well formed`() {
        assertEquals(64, TinyModelSpec.SHA256.length)
        assertTrue(TinyModelSpec.FILE_NAME.contains(TinyModelSpec.SHA256.take(12)))
        assertTrue(TinyModelSpec.url.startsWith("https://"))
    }
}
