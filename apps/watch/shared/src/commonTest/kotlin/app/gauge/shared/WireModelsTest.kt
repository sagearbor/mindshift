package app.gauge.shared

import kotlinx.serialization.encodeToString
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Golden JSON generated from the actual server models (server/models.py),
 * NOT hand-typed, via:
 *
 *   python3 -c "
 *   from server.models import Episode, Participant, VectorEvent, NudgeEvent, ConsentRecord
 *
 *   ep = Episode(
 *       id='e1',
 *       owner_account='acct1',
 *       started_at='2026-07-31T10:00:00Z',
 *       ended_at='2026-07-31T10:05:00Z',
 *       status='analyzed',
 *       participants=[
 *           Participant(id='self', role='self', speaker_label='S1', display_name='Sophie', account_id='acct1'),
 *           Participant(id='p2', role='other', speaker_label='S2'),
 *       ],
 *       vector_events=[
 *           VectorEvent(vector='yelling', level=2, t=1.5, value=8.2, detail='loud'),
 *           VectorEvent(vector='hr_spike', level=1, t=3.0, value=95.0),
 *       ],
 *       nudge_events=[
 *           NudgeEvent(channel='A', level=2, t=1.5, vectors=['yelling']),
 *       ],
 *       series={'rms_db': [1.0, 2.0, 3.0]},
 *       summary='A short talk.',
 *       shared_with=['acct2'],
 *       consents=[ConsentRecord(id='c1', participant_id='p2', kind='labeling', attested_by='acct1', confirmed=True, ts='2026-07-31T10:00:05Z')],
 *   )
 *   print(ep.model_dump_json())
 *   "
 *
 * Note `consents` is present in the server's JSON (it's not excluded) but is
 * intentionally not mirrored as a Kotlin property in this task's scope;
 * `ignoreUnknownKeys = true` (see `wireJson` in WireModels.kt) lets us decode
 * this real payload anyway. `pcm_b64` never appears at all — it's
 * `Field(exclude=True)` server-side.
 */
private const val SERVER_GOLDEN_JSON = """{"id":"e1","owner_account":"acct1","started_at":"2026-07-31T10:00:00Z","ended_at":"2026-07-31T10:05:00Z","status":"analyzed","participants":[{"id":"self","role":"self","speaker_label":"S1","display_name":"Sophie","account_id":"acct1"},{"id":"p2","role":"other","speaker_label":"S2","display_name":null,"account_id":null}],"vector_events":[{"vector":"yelling","level":2,"t":1.5,"value":8.2,"detail":"loud"},{"vector":"hr_spike","level":1,"t":3.0,"value":95.0,"detail":""}],"nudge_events":[{"channel":"A","level":2,"t":1.5,"vectors":["yelling"]}],"series":{"rms_db":[1.0,2.0,3.0]},"summary":"A short talk.","shared_with":["acct2"],"consents":[{"id":"c1","participant_id":"p2","kind":"labeling","attested_by":"acct1","confirmed":true,"ts":"2026-07-31T10:00:05Z"}]}"""

class WireModelsTest {
    @Test
    fun decodesServerEpisodeJson() {
        val ep = wireJson.decodeFromString<Episode>(SERVER_GOLDEN_JSON)
        assertEquals("e1", ep.id)
        assertEquals(2, ep.vectorEvents.first().level)
    }

    @Test
    fun decodesSnakeCaseFieldsIntoCamelCaseProperties() {
        val ep = wireJson.decodeFromString<Episode>(SERVER_GOLDEN_JSON)

        assertEquals("acct1", ep.ownerAccount)
        assertEquals("2026-07-31T10:00:00Z", ep.startedAt)
        assertEquals("2026-07-31T10:05:00Z", ep.endedAt)
        assertEquals(listOf("acct2"), ep.sharedWith)

        val self = ep.participants.first { it.id == "self" }
        assertEquals("S1", self.speakerLabel)
        assertEquals("Sophie", self.displayName)
        assertEquals("acct1", self.accountId)

        val other = ep.participants.first { it.id == "p2" }
        assertNull(other.displayName)
        assertNull(other.accountId)

        assertEquals(1, ep.nudgeEvents.size)
        assertEquals("A", ep.nudgeEvents.first().channel)
    }

    @Test
    fun decodesVectorSubscriptionDirectly() {
        val sub = wireJson.decodeFromString<VectorSubscription>(
            """{"vector":"yelling","sensitivity":0.5,"haptics":true,"channel":"A"}"""
        )
        assertEquals("yelling", sub.vector)
        assertEquals(0.5, sub.sensitivity)
    }

    @Test
    fun telemetryBatchRoundTripsAndUsesSnakeCaseAppVersion() {
        val batch = TelemetryBatch(
            device = "d",
            appVersion = "0.1.1",
            events = listOf(TelemetryEventOut("info", "t", "m", null, "ts")),
        )

        val encoded = wireJson.encodeToString(batch)
        assertEquals(true, encoded.contains("\"app_version\":\"0.1.1\""))

        val decoded = wireJson.decodeFromString<TelemetryBatch>(encoded)
        assertEquals(batch, decoded)
    }

    @Test
    fun enrollmentBaselineRoundTrips() {
        val b = EnrollmentBaseline("acct-1", -31.5, 148.0, "2026-08-02T12:00:00+00:00")
        val json = wireJson.encodeToString(EnrollmentBaseline.serializer(), b)
        assertTrue(json.contains("\"account_id\":\"acct-1\""))
        assertTrue(json.contains("\"rms_db\":-31.5"))
        assertEquals(b, wireJson.decodeFromString(EnrollmentBaseline.serializer(), json))
    }

    @Test
    fun labelRequestUsesSnakeCase() {
        val json = wireJson.encodeToString(LabelRequest.serializer(), LabelRequest("p1", "Mom", true))
        assertTrue(json.contains("\"participant_id\":\"p1\""))
        assertTrue(json.contains("\"display_name\":\"Mom\""))
        assertTrue(json.contains("\"attested\":true"))
    }

    @Test
    fun shareRequestUsesSnakeCase() {
        assertTrue(
            wireJson.encodeToString(ShareRequest.serializer(), ShareRequest("acct-2"))
                .contains("\"with_account\":\"acct-2\"")
        )
    }

    @Test
    fun episodeDecodesConsents() {
        val json = """{"id":"e1","owner_account":"a","started_at":"t","ended_at":null,"status":"captured",
            "participants":[],"vector_events":[],"nudge_events":[],
            "consents":[{"id":"c1","participant_id":"self","kind":"sharing","attested_by":"a",
                         "confirmed":false,"ts":"t"}]}"""
        val ep = wireJson.decodeFromString(Episode.serializer(), json)
        assertEquals(1, ep.consents.size)
        assertEquals("sharing", ep.consents.first().kind)
        assertEquals("a", ep.consents.first().attestedBy)
    }

    @Test
    fun episodeWithoutConsentsStillDecodes() { // wear-track regression guard
        val json = """{"id":"e1","owner_account":"a","started_at":"t","ended_at":null,"status":"live",
            "participants":[],"vector_events":[],"nudge_events":[]}"""
        assertEquals(emptyList(), wireJson.decodeFromString(Episode.serializer(), json).consents)
    }

    // --- Wave B: me/claim-legacy, groups, standing, voice status (server-track item 13c
    // + Wave B Task 2's POST /me/claim-legacy) --------------------------------------------

    @Test fun meDecodesPrincipalShape() {
        val me = wireJson.decodeFromString(Me.serializer(),
            """{"account_id":"uid-1","email":"a@example.com","legacy":false}""")
        assertEquals("uid-1", me.accountId)
        assertEquals("a@example.com", me.email)
        assertFalse(me.legacy)
    }

    @Test fun claimLegacyResponseDecodesAllFields() {
        val r = wireJson.decodeFromString(ClaimLegacyResponse.serializer(),
            """{"status":"claimed","episodes_moved":7,"baseline_copied":true,
                "subscriptions_copied":false,"speaker_profile_copied":true,
                "previously_claimed_at":"2026-08-02T12:00:00+00:00"}""")
        assertEquals("claimed", r.status)
        assertEquals(7, r.episodesMoved)
        assertTrue(r.baselineCopied)
        assertEquals("2026-08-02T12:00:00+00:00", r.previouslyClaimedAt)
    }

    @Test fun groupDecodesMembersInvitesAndConsents() {
        val g = wireJson.decodeFromString(Group.serializer(),
            """{"id":"g1","kind":"pair","name":"","created_by":"uid-1","created_at":"t",
                "members":[{"account_id":"uid-1","joined_at":"t"}],
                "invites":[{"code":"abcd1234","email":null,"invited_by":"uid-1","created_at":"t",
                            "accepted_by":null,"accepted_at":null}],
                "consents":[{"id":"c1","participant_id":"uid-1","kind":"mutual_visibility",
                             "attested_by":"uid-1","confirmed":true,"ts":"t"}]}""")
        assertEquals("pair", g.kind)
        assertEquals(listOf("uid-1"), g.members.map { it.accountId })
        assertEquals("abcd1234", g.invites.single().code)
        assertNull(g.invites.single().acceptedBy)
        assertEquals("mutual_visibility", g.consents.single().kind)
    }

    @Test fun groupStandingDecodesNullCalmAndNullAheadHonestly() {
        val s = wireJson.decodeFromString(GroupStanding.serializer(),
            """{"group_id":"g1","period_days":7,"period_start":"a","period_end":"b",
                "members":[{"account_id":"uid-1","display_name":null,
                            "current":{"episodes":0,"calm":null,"nudges":0,"escalations":0},
                            "prior":{"episodes":2,"calm":75.0,"nudges":1,"escalations":0},
                            "delta_vs_self":null,"improving":null}],
                "both_improving":false,"ahead":null}""")
        assertNull(s.ahead)
        assertNull(s.members.single().current.calm)
        assertEquals(75.0, s.members.single().prior.calm)
        assertNull(s.members.single().deltaVsSelf)
    }

    @Test
    fun decodesGroupStandingFromServerShapedJson() {
        // Golden JSON shaped like server/models.py's GroupStanding.model_dump_json() — the
        // fully-populated, both-improving counterpart to
        // groupStandingDecodesNullCalmAndNullAheadHonestly's all-null case above.
        val json = """
            {"group_id":"g1","period_days":7,"period_start":"2026-07-28T00:00:00Z",
             "period_end":"2026-08-04T00:00:00Z","both_improving":true,"ahead":"acct1",
             "members":[
               {"account_id":"acct1","display_name":"Sophie",
                "current":{"episodes":3,"calm":72.5,"nudges":4,"escalations":1},
                "prior":{"episodes":2,"calm":60.0,"nudges":6,"escalations":2},
                "delta_vs_self":12.5,"improving":true}
             ]}
        """.trimIndent()
        val standing = wireJson.decodeFromString(GroupStanding.serializer(), json)
        assertEquals("g1", standing.groupId)
        assertEquals(true, standing.bothImproving)
        assertEquals("acct1", standing.ahead)
        assertEquals(1, standing.members.size)
        assertEquals(72.5, standing.members[0].current.calm)
    }

    @Test
    fun memberStandingCalmNullNeverZero() {
        // Server contract: calm is null (no data), never a fabricated zero.
        val json = """{"episodes":0,"calm":null,"nudges":0,"escalations":0}"""
        val stats = wireJson.decodeFromString(PeriodStats.serializer(), json)
        assertNull(stats.calm)
        assertEquals(0, stats.episodes)
    }

    @Test fun voiceStatusDecodesUnenrolledMinimalShape() {
        val v = wireJson.decodeFromString(VoiceEnrollmentStatus.serializer(),
            """{"available":false,"enrolled":false}""")
        assertFalse(v.available); assertFalse(v.enrolled)
        assertEquals(0, v.enrollCount); assertNull(v.updatedAt)
    }

    @Test fun joinGroupRequestEncodesCode() {
        assertTrue(wireJson.encodeToString(JoinGroupRequest.serializer(), JoinGroupRequest("abcd1234"))
            .contains("\"code\":\"abcd1234\""))
    }

    // --- Wave C: capture + device-pairing wire models (T2) --------------------------------

    @Test
    fun decodesCaptureFromServerShapedJson() {
        // Golden JSON shaped like server/models.py's Capture.model_dump_json(). `labels` is
        // deliberately NOT mirrored (opaque dashboard-only payload, same "unmirrored fields are
        // silently skipped" contract this file already documents for pcm_b64/consents).
        val json = """
            {"id":"cap1","account_id":"acct1","device":"watch-abc","captured_at":"2026-08-04T10:00:00Z",
             "received_at":"2026-08-04T10:00:05Z","duration_s":118.0,"trigger":"manual",
             "sample_rate":16000,"status":"stored","audio_uri":"gs://bucket/captures/acct1/cap1.pcm",
             "audio_bytes":3776000,"upload_encoding":"gzip","labels":{"note":"ignored"},
             "labels_updated_at":null,"consents":[]}
        """.trimIndent()
        val cap = wireJson.decodeFromString(Capture.serializer(), json)
        assertEquals("cap1", cap.id)
        assertEquals("stored", cap.status)
        assertEquals(118.0, cap.durationS)
        assertEquals("gzip", cap.uploadEncoding)
    }

    @Test
    fun encodesCreateCaptureRequestWithAttestedTrue() {
        val req = CreateCaptureRequest(
            capturedAt = "2026-08-04T10:00:00Z",
            durationS = 120.0,
            trigger = "manual",
            device = "watch-abc",
            attested = true,
        )
        val json = wireJson.encodeToString(CreateCaptureRequest.serializer(), req)
        assertEquals(true, json.contains("\"attested\":true"))
        assertEquals(true, json.contains("\"duration_s\":120.0"))
    }

    @Test
    fun decodesPairingStatusClaimed() {
        val json = """{"status":"claimed","account_id":"acct1","device_token":"opaque-secret"}"""
        val status = wireJson.decodeFromString(PairingStatus.serializer(), json)
        assertEquals("claimed", status.status)
        assertEquals("opaque-secret", status.deviceToken)
    }

    @Test
    fun decodesPairingStatusPendingHasNoToken() {
        val json = """{"status":"pending"}"""
        val status = wireJson.decodeFromString(PairingStatus.serializer(), json)
        assertEquals("pending", status.status)
        assertNull(status.deviceToken)
        assertNull(status.accountId)
    }
}
