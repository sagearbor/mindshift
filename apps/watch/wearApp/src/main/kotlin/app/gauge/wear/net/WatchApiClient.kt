package app.gauge.wear.net

import app.gauge.shared.Capture
import app.gauge.shared.ClaimLegacyResponse
import app.gauge.shared.CreateCaptureRequest
import app.gauge.shared.Group
import app.gauge.shared.GroupStanding
import app.gauge.shared.MemberStanding
import app.gauge.shared.wireJson
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response

/** Mirrors [app.gauge.phone.api.ApiResult]'s exact shape (androidApp's Wave B client) — kept as
 * a separate, wearApp-local type rather than a shared module (wearApp and androidApp don't share
 * code today), but matching the established shape on purpose. */
sealed interface ApiResult<out T> {
    data class Ok<T>(val value: T) : ApiResult<T>
    data class Failure(val code: Int?, val message: String) : ApiResult<Nothing>
}

private const val NOT_SIGNED_IN = "not signed in"
private val JSON_MEDIA_TYPE = "application/json".toMediaType()
private val OCTET_STREAM_MEDIA_TYPE = "application/octet-stream".toMediaType()
private val groupListSerializer = ListSerializer(Group.serializer())

/**
 * Authenticated REST client for everything Wave C's two data-facing features need:
 * the couples standing card ([myStanding]/[listGroups]/[groupStanding]) and
 * retro-capture upload ([createCapture]/[uploadCaptureAudio]). One client, one
 * auth ladder, shared by both — see this plan's Task 6 KDoc.
 *
 * Unlike [app.gauge.phone.api.HttpGaugeApi], there is no legacy `?account=`
 * fallback: `/me/standing`, `/groups`, `/captures` are `require_full_auth`-gated
 * server-side (confirmed against `server/main.py`'s router wiring), so a caller
 * with no [deviceToken] can never reach them regardless — [call] fails fast with
 * [ApiResult.Failure] before touching the network, same as [app.gauge.phone.api.
 * HttpGaugeApi]'s own "neither an override nor a token" case.
 *
 * `open` (class + every public `suspend fun` below): Task 11's `RetroCaptureUploaderTest`
 * subclasses this with a `FakeApi` that overrides individual calls to return canned
 * [ApiResult]s without touching the network — same test-seam pattern already established by
 * [app.gauge.wear.haptics.HapticDirector]'s `open fun onNudge` for `SentinelControllerTest`'s
 * `ThrowingHapticDirector`. Every method is opened uniformly (not just the two Task 11's shown
 * fake happens to override) since all six are shared across multiple downstream consumers per
 * this class's own KDoc (Task 7/8/9/11) and there's no principled reason to leave an arbitrary
 * subset final while the rest are overridable.
 */
open class WatchApiClient(
    private val baseUrl: String,
    private val deviceToken: () -> String?,
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(45, TimeUnit.SECONDS)
        .build(),
) {
    open suspend fun myStanding(periodDays: Int = 7): ApiResult<MemberStanding> =
        call("GET", "/me/standing?period_days=$periodDays") {
            wireJson.decodeFromString(MemberStanding.serializer(), it)
        }

    open suspend fun listGroups(): ApiResult<List<Group>> =
        call("GET", "/groups") { wireJson.decodeFromString(groupListSerializer, it) }

    open suspend fun groupStanding(groupId: String, periodDays: Int = 7): ApiResult<GroupStanding> =
        call("GET", "/groups/$groupId/standing?period_days=$periodDays") {
            wireJson.decodeFromString(GroupStanding.serializer(), it)
        }

    open suspend fun createCapture(req: CreateCaptureRequest): ApiResult<Capture> {
        val json = wireJson.encodeToString(CreateCaptureRequest.serializer(), req)
        return call("POST", "/captures", body = json.toRequestBody(JSON_MEDIA_TYPE)) {
            wireJson.decodeFromString(Capture.serializer(), it)
        }
    }

    /** [gzippedPcm] must already be gzip-compressed (see Task 11's [app.gauge.wear.capture.
     * RetroCaptureUploader]) — this method only sets the header, it never compresses itself, so a
     * caller that forgets to gzip would silently corrupt the upload; that risk is intentionally
     * pushed to the one call site (Task 11) that owns the compression, not duplicated here. */
    open suspend fun uploadCaptureAudio(captureId: String, gzippedPcm: ByteArray): ApiResult<Capture> =
        call(
            "PUT",
            "/captures/$captureId/audio",
            body = gzippedPcm.toRequestBody(OCTET_STREAM_MEDIA_TYPE),
            extraHeaders = mapOf("Content-Encoding" to "gzip"),
        ) { wireJson.decodeFromString(Capture.serializer(), it) }

    /** `PUT /captures/{id}/labels` — [labelsJson] must already be a serialized JSON OBJECT (the
     * server 422s anything else). The journal path PUTs its `{"journal": true, ...}` labels
     * BEFORE uploading audio, deliberately: the server's journal processing hook fires off the
     * audio-upload success path and only for captures already labeled `journal` at that moment. */
    open suspend fun putCaptureLabels(captureId: String, labelsJson: String): ApiResult<Capture> =
        call("PUT", "/captures/$captureId/labels", body = labelsJson.toRequestBody(JSON_MEDIA_TYPE)) {
            wireJson.decodeFromString(Capture.serializer(), it)
        }

    open suspend fun claimLegacy(): ApiResult<ClaimLegacyResponse> =
        call("POST", "/me/claim-legacy", body = "{}".toRequestBody(JSON_MEDIA_TYPE)) {
            wireJson.decodeFromString(ClaimLegacyResponse.serializer(), it)
        }

    private suspend fun <T> call(
        method: String,
        path: String,
        body: RequestBody? = null,
        extraHeaders: Map<String, String> = emptyMap(),
        decode: (String) -> T,
    ): ApiResult<T> = withContext(Dispatchers.IO) {
        try {
            val token = deviceToken()
            if (token == null) return@withContext ApiResult.Failure(null, NOT_SIGNED_IN)
            val requestBuilder = Request.Builder()
                .url("$baseUrl$path")
                .addHeader("Authorization", "Bearer $token")
            extraHeaders.forEach { (k, v) -> requestBuilder.addHeader(k, v) }
            when (method) {
                "GET" -> requestBuilder.get()
                "POST" -> requestBuilder.post(body ?: EMPTY_BODY)
                "PUT" -> requestBuilder.put(body ?: EMPTY_BODY)
                else -> error("unsupported method $method")
            }
            client.newCall(requestBuilder.build()).execute().use { response ->
                val text = response.body?.string().orEmpty()
                if (response.isSuccessful) {
                    ApiResult.Ok(decode(text))
                } else {
                    ApiResult.Failure(response.code, failureDetail(text, response))
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            ApiResult.Failure(null, e.message ?: e::class.simpleName ?: "unknown error")
        }
    }

    private fun failureDetail(body: String, response: Response): String {
        val detail = runCatching { Json.parseToJsonElement(body).jsonObject["detail"]?.jsonPrimitive?.contentOrNull }
            .getOrNull()
        if (detail != null) return detail
        if (body.isNotBlank()) return body.take(200)
        return "${response.code} ${response.message}".trim()
    }
}

private val EMPTY_BODY: RequestBody = ByteArray(0).toRequestBody(JSON_MEDIA_TYPE)
