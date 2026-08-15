package app.gauge.wear.telemetry

import app.gauge.shared.TelemetryBatch
import app.gauge.shared.TelemetryEventOut
import app.gauge.shared.wireJson
import java.util.concurrent.TimeUnit
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody

/**
 * HTTP client for posting device telemetry (crash reports, logs) to the
 * backend's `POST /telemetry` endpoint.
 *
 * Deliberately free of `android.*` imports so [TelemetryClientTest] runs as
 * a plain JVM test under MockWebServer, matching the pattern established by
 * [app.gauge.wear.net.EpisodeWsClient].
 */
class TelemetryClient(
    private val baseUrl: String,
    private val device: String,
    private val appVersion: String,
    private val client: OkHttpClient = defaultClient(),
) {
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    /**
     * Synchronously posts [events] and blocks until the request completes or
     * times out.
     *
     * CRITICAL: this is invoked from a crash handler while the process is
     * dying, so it must NEVER throw — every failure mode (connection
     * refused, DNS failure, timeout, non-2xx response, serialization error)
     * is swallowed and reported as `false`.
     */
    fun postBlocking(events: List<TelemetryEventOut>, timeoutMs: Long = 2000): Boolean {
        if (events.isEmpty()) return true
        return try {
            val body = wireJson.encodeToString(
                TelemetryBatch.serializer(),
                TelemetryBatch(device, appVersion, events),
            )
            val request = Request.Builder()
                .url("$baseUrl/telemetry")
                .post(body.toRequestBody(jsonMediaType))
                .build()
            val callClient = client.newBuilder()
                .callTimeout(timeoutMs, TimeUnit.MILLISECONDS)
                .build()
            callClient.newCall(request).execute().use { response -> response.isSuccessful }
        } catch (e: Exception) {
            // Never propagate: this call site is a dying-process crash handler.
            false
        }
    }

    /** Fire-and-forget async post; failures are ignored (no callback). */
    fun postAsync(events: List<TelemetryEventOut>) {
        if (events.isEmpty()) return
        try {
            val body = wireJson.encodeToString(
                TelemetryBatch.serializer(),
                TelemetryBatch(device, appVersion, events),
            )
            val request = Request.Builder()
                .url("$baseUrl/telemetry")
                .post(body.toRequestBody(jsonMediaType))
                .build()
            client.newCall(request).enqueue(object : okhttp3.Callback {
                override fun onFailure(call: okhttp3.Call, e: java.io.IOException) {
                    // Ignored: best-effort telemetry, nothing to recover here.
                }
                override fun onResponse(call: okhttp3.Call, response: okhttp3.Response) {
                    response.close()
                }
            })
        } catch (e: Exception) {
            // Same never-throw contract as postBlocking, defensively applied here too.
        }
    }

    companion object {
        fun defaultClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .build()
    }
}
