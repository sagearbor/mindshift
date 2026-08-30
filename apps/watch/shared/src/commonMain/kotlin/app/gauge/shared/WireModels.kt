package app.gauge.shared

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

/**
 * Kotlin mirrors of the server's wire contract (server/models.py), decoded via
 * kotlinx.serialization. Field names are snake_case on the wire (Pydantic
 * serializes its own snake_case attribute names verbatim); Kotlin properties
 * use camelCase with `@SerialName` where the two differ.
 *
 * `pcm_b64` is intentionally NOT mirrored here: it is excluded server-side
 * (Field(exclude=True)) and never appears on the wire. `ConsentRecord` /
 * `Episode.consents` ARE mirrored as of Plan 2b (phone-side consent surface).
 * `Json.decodeFromString` below runs with `ignoreUnknownKeys = true`, so any
 * other unmirrored fields present in a real server payload are silently
 * skipped rather than causing a decode failure.
 */
val wireJson: Json = Json { ignoreUnknownKeys = true }

@Serializable
data class VectorEvent(
    val vector: String,
    val level: Int,
    val t: Double,
    val value: Double,
    val detail: String = "",
)

@Serializable
data class NudgeEvent(
    val channel: String,
    val level: Int,
    val t: Double,
    val vectors: List<String> = emptyList(),
)

@Serializable
data class VectorSubscription(
    val vector: String,
    val sensitivity: Double = 1.0,
    val haptics: Boolean = true,
    // NOTE (final-review MINOR): this "A" default only matches the server for
    // vector != "hr_spike" — server/models.py's VectorSubscription defaults
    // hr_spike's channel to "B" via a model_validator, which this Kotlin
    // field-default cannot mirror. Any Plan-2 write path (client constructing
    // a VectorSubscription to PUT /settings/vectors) must set channel
    // explicitly per vector rather than relying on this default, or an
    // hr_spike subscription will round-trip onto the wrong nudge channel.
    val channel: String = "A",
)

@Serializable
data class Participant(
    val id: String,
    val role: String,
    @SerialName("speaker_label") val speakerLabel: String,
    @SerialName("display_name") val displayName: String? = null,
    @SerialName("account_id") val accountId: String? = null,
)

@Serializable
data class EnrollmentBaseline(
    @SerialName("account_id") val accountId: String,
    @SerialName("rms_db") val rmsDb: Double,
    @SerialName("f0_median") val f0Median: Double,
    @SerialName("updated_at") val updatedAt: String,
)

@Serializable
data class ConsentRecord(
    val id: String,
    @SerialName("participant_id") val participantId: String,
    val kind: String, // "labeling" | "sharing"
    @SerialName("attested_by") val attestedBy: String,
    val confirmed: Boolean = false,
    val ts: String,
)

@Serializable
data class LabelRequest(
    @SerialName("participant_id") val participantId: String,
    @SerialName("display_name") val displayName: String,
    val attested: Boolean,
)

@Serializable
data class ShareRequest(@SerialName("with_account") val withAccount: String)

@Serializable
data class TelemetryEventOut(
    val level: String,
    val tag: String,
    val message: String,
    val stack: String? = null,
    val ts: String,
    /** Structured payload mirroring the server's additive `TelemetryEventIn.data`
     * (server/watch/routers/telemetry.py, `data: dict | None`) — the phone's
     * "Send diagnostics" pioneered it; the watch now uses it for the journal
     * battery/counter series. `null` (the default) is omitted-equivalent for
     * older readers, so every existing event construction site is unchanged. */
    val data: JsonObject? = null,
)

@Serializable
data class TelemetryBatch(
    val device: String,
    @SerialName("app_version") val appVersion: String,
    val events: List<TelemetryEventOut>,
)

// --- Wave B: identity (Task 2's POST /me/claim-legacy) + couples/groups + standing
// (server-track item 13c's GET /me/standing, server/models.py's group/standing family) +
// voice enrollment status / account lookup (server/rest_api.py). Mirrors the server
// contracts exactly (checked field-for-field against server/auth.py's Principal,
// server/rest_api.py's ClaimLegacyResponse/VoiceEnrollmentStatus/AccountLookupResponse,
// server/models.py's Group/GroupMember/GroupInvite/PeriodStats/MemberStanding/
// GroupStanding, and server/groups_api.py's CreateGroupRequest/JoinRequest — no mismatch
// found). Additions only; nothing above this line changes. ------------------------------

@Serializable
data class Me(
    @SerialName("account_id") val accountId: String,
    val email: String? = null,
    val legacy: Boolean = false,
)

@Serializable
data class ClaimLegacyResponse(
    val status: String,                          // "claimed" | "nothing_to_claim"
    @SerialName("episodes_moved") val episodesMoved: Int = 0,
    @SerialName("baseline_copied") val baselineCopied: Boolean = false,
    @SerialName("subscriptions_copied") val subscriptionsCopied: Boolean = false,
    @SerialName("speaker_profile_copied") val speakerProfileCopied: Boolean = false,
    @SerialName("previously_claimed_at") val previouslyClaimedAt: String? = null,
)

@Serializable
data class GroupMember(
    @SerialName("account_id") val accountId: String,
    @SerialName("joined_at") val joinedAt: String,
)

@Serializable
data class GroupInvite(
    val code: String,
    val email: String? = null,
    @SerialName("invited_by") val invitedBy: String,
    @SerialName("created_at") val createdAt: String,
    @SerialName("accepted_by") val acceptedBy: String? = null,
    @SerialName("accepted_at") val acceptedAt: String? = null,
)

@Serializable
data class Group(
    val id: String,
    val kind: String,                            // "pair" | "team"
    val name: String = "",
    @SerialName("created_by") val createdBy: String,
    @SerialName("created_at") val createdAt: String,
    val members: List<GroupMember> = emptyList(),
    val invites: List<GroupInvite> = emptyList(),
    val consents: List<ConsentRecord> = emptyList(),
)

@Serializable
data class PeriodStats(
    val episodes: Int,
    val calm: Double? = null,                    // null = no data, NEVER zero (server contract)
    val nudges: Int = 0,
    val escalations: Int = 0,
)

@Serializable
data class MemberStanding(
    @SerialName("account_id") val accountId: String,
    @SerialName("display_name") val displayName: String? = null,
    val current: PeriodStats,
    val prior: PeriodStats,
    @SerialName("delta_vs_self") val deltaVsSelf: Double? = null,
    val improving: Boolean? = null,
)

@Serializable
data class GroupStanding(
    @SerialName("group_id") val groupId: String,
    @SerialName("period_days") val periodDays: Int,
    @SerialName("period_start") val periodStart: String,
    @SerialName("period_end") val periodEnd: String,
    val members: List<MemberStanding> = emptyList(),
    @SerialName("both_improving") val bothImproving: Boolean = false,
    val ahead: String? = null,
)

@Serializable
data class VoiceEnrollmentStatus(
    val available: Boolean,
    val enrolled: Boolean,
    @SerialName("enroll_count") val enrollCount: Int = 0,
    val dim: Int? = null,
    val model: String? = null,
    @SerialName("updated_at") val updatedAt: String? = null,
)

@Serializable
data class AccountLookup(@SerialName("account_id") val accountId: String)

@Serializable
data class CreateGroupRequest(val kind: String = "pair", val name: String = "")

@Serializable
data class JoinGroupRequest(val code: String)

@Serializable
data class Episode(
    val id: String,
    @SerialName("owner_account") val ownerAccount: String,
    @SerialName("started_at") val startedAt: String,
    @SerialName("ended_at") val endedAt: String? = null,
    val status: String,
    val participants: List<Participant> = emptyList(),
    @SerialName("vector_events") val vectorEvents: List<VectorEvent> = emptyList(),
    @SerialName("nudge_events") val nudgeEvents: List<NudgeEvent> = emptyList(),
    val series: Map<String, List<Double>> = emptyMap(),
    val summary: String? = null,
    @SerialName("shared_with") val sharedWith: List<String> = emptyList(),
    val consents: List<ConsentRecord> = emptyList(),
)

// --- Wave C: retro-capture + device pairing ----------------------------------------
// Capture/CreateCaptureRequest mirror server/captures_api.py + server/models.py's
// Capture exactly (read 2026-08-04; `labels` intentionally NOT mirrored — same
// "unmirrored fields silently skipped" contract as pcm_b64 above). PairingStart/
// PairingStatus are this plan's OWN new contract for POST /me/pair/start and
// GET /me/pair/status — see docs/superpowers/plans/2026-08-04-gauge-wave-c-couples-
// wrist.md's Task 4 KDoc and "Open questions" section: these two endpoints are a
// server-lane dependency this plan does not implement, not yet live as of this
// writing. Additions only; nothing above this line changes. -----------------------

@Serializable
data class CreateCaptureRequest(
    @SerialName("captured_at") val capturedAt: String,
    @SerialName("duration_s") val durationS: Double,
    val trigger: String = "",
    val device: String? = null,
    @SerialName("sample_rate") val sampleRate: Int = 16000,
    val attested: Boolean,
)

@Serializable
data class Capture(
    val id: String,
    @SerialName("account_id") val accountId: String,
    val device: String? = null,
    @SerialName("captured_at") val capturedAt: String,
    @SerialName("received_at") val receivedAt: String,
    @SerialName("duration_s") val durationS: Double,
    val trigger: String = "",
    @SerialName("sample_rate") val sampleRate: Int = 16000,
    val status: String = "awaiting_audio",         // "awaiting_audio" | "stored"
    @SerialName("audio_uri") val audioUri: String? = null,
    @SerialName("audio_bytes") val audioBytes: Int? = null,
    @SerialName("upload_encoding") val uploadEncoding: String? = null,
    @SerialName("labels_updated_at") val labelsUpdatedAt: String? = null,
    val consents: List<ConsentRecord> = emptyList(),
)

/** Response of the (server-lane-dependency) `POST /me/pair/start` — no auth required, since
 * an unclaimed watch has no identity yet to authenticate as. */
@Serializable
data class PairingStart(
    val code: String,                              // short human-typeable code, e.g. "K7QP2M"
    @SerialName("pairing_id") val pairingId: String,
    @SerialName("expires_at") val expiresAt: String,
)

/** Response of the (server-lane-dependency) `GET /me/pair/status?pairing_id=...` — also no
 * auth required (the watch has no token yet; `pairing_id` itself is the capability). */
@Serializable
data class PairingStatus(
    val status: String,                            // "pending" | "claimed" | "expired"
    @SerialName("account_id") val accountId: String? = null,
    @SerialName("device_token") val deviceToken: String? = null,
)
