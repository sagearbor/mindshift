package app.gauge.wear.auth

import app.gauge.shared.PairingStart
import app.gauge.shared.PairingStatus
import app.gauge.shared.wireJson
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response

/**
 * HTTP client for the short-code device-pairing flow (Wave C) — see this plan's
 * Task 5 KDoc for why this shape was chosen over on-wrist OAuth or a Wearable
 * Data Layer relay. Mirrors [app.gauge.wear.telemetry.TelemetryClient]'s pattern
 * exactly: no `android.*` imports, so this runs as a plain JVM test under
 * MockWebServer, and every failure mode (transport, non-2xx, malformed JSON)
 * degrades to `null` rather than throwing — a pairing screen that can't reach
 * the backend must show "couldn't reach the server, try again," never crash.
 *
 * Neither endpoint sends an Authorization header — an unclaimed watch has no
 * token yet; `pairing_id` alone (a random, unguessable id) is the capability
 * that scopes [poll] to this one pairing attempt.
 *
 * v0.3.1: both methods are `suspend` and hop to [Dispatchers.IO] INSIDE the
 * client (same posture as [app.gauge.wear.net.WatchApiClient.call]) — the
 * v0.3.0 on-device bug was exactly a call site (SignInScreen's LaunchedEffect,
 * main dispatcher) driving the old synchronous methods into
 * NetworkOnMainThreadException, which the blanket catch silently converted to
 * "Couldn't reach the server." Owning the dispatch here means no future call
 * site can reintroduce that. [onSwallowedError] receives a one-line reason for
 * every null-return (transport exception, non-2xx, empty/malformed body) so
 * the wearer-facing "try again" is never the ONLY record of what went wrong —
 * production wires it to Telemetry, tests to a list. CancellationException is
 * rethrown, never swallowed: a disposed composable must cancel cleanly.
 *
 * v0.4.0 (queued gauge fast-follow, now unblocked by the unified backend cutover): the unified
 * Cloud Run service scales to zero, so the very first pairing request after idle can eat a cold
 * start that the old 10s call timeout could clip mid-flight. [DEFAULT_CALL_TIMEOUT_SECONDS] moves
 * to 30s and [executeWithOneRetry] retries exactly once, silently, on a transport-level failure
 * (timeout, connection refused/reset — anything [IOException]) before giving up. A non-2xx
 * response or a malformed body is an honest answer from a server that IS reachable, not a
 * transport failure, so neither is retried — only [onSwallowedError] reports those, unchanged.
 */
class DevicePairingClient(
    private val baseUrl: String,
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(5, TimeUnit.SECONDS)
        .callTimeout(DEFAULT_CALL_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        .build(),
    private val onSwallowedError: (String) -> Unit = {},
) {
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    /** One silent retry on a transport-level ([IOException]) failure only — see class KDoc. If
     * the retry also throws, that exception propagates to the caller's own catch, which is the
     * only place that reports a swallowed error, so a total failure still reports exactly once. */
    private fun executeWithOneRetry(request: Request): Response =
        try {
            client.newCall(request).execute()
        } catch (e: IOException) {
            client.newCall(request).execute()
        }

    suspend fun start(): PairingStart? = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/me/pair/start")
                .post(ByteArray(0).toRequestBody(jsonMediaType))
                .build()
            executeWithOneRetry(request).use { response ->
                if (!response.isSuccessful) {
                    onSwallowedError("start http ${response.code}")
                    return@withContext null
                }
                val body = response.body?.string()
                if (body == null) {
                    onSwallowedError("start empty body")
                    return@withContext null
                }
                wireJson.decodeFromString(PairingStart.serializer(), body)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            onSwallowedError("start ${e.javaClass.simpleName}: ${e.message}")
            null
        }
    }

    suspend fun poll(pairingId: String): PairingStatus? = withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url("$baseUrl/me/pair/status?pairing_id=$pairingId")
                .get()
                .build()
            executeWithOneRetry(request).use { response ->
                if (!response.isSuccessful) {
                    onSwallowedError("poll http ${response.code}")
                    return@withContext null
                }
                val body = response.body?.string()
                if (body == null) {
                    onSwallowedError("poll empty body")
                    return@withContext null
                }
                wireJson.decodeFromString(PairingStatus.serializer(), body)
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            onSwallowedError("poll ${e.javaClass.simpleName}: ${e.message}")
            null
        }
    }

    companion object {
        /** 3x the previous 10s bound (queued gauge fast-follow) — the unified Cloud Run service
         * scales to zero, so a cold-start pairing request needs headroom the old timeout didn't
         * have. See class KDoc. */
        const val DEFAULT_CALL_TIMEOUT_SECONDS = 30L
    }
}
