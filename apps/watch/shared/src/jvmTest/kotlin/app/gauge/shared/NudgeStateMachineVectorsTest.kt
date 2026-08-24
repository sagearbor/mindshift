package app.gauge.shared

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Track 1 (2026-08-24): drives [NudgeStateMachine] from the SAME golden vectors the Python
 * reference (`server/nudge_policy.py`, driver `server/tests/test_nudge_policy_vectors.py`) replays
 * — `server/tests/fixtures/policy_vectors/nudge_policy.json`, synced onto this test classpath by
 * `shared/build.gradle.kts`'s `syncPolicyVectors` task (never a hand-edited copy; see the comment
 * there). The JSON is the contract: if a case fails here, the fix is to the DIVERGING runtime, not
 * to the fixture (see the README next to it).
 *
 * Only cases tagged `applies_to: ["…", "watch", …]` are replayed — per the fixture's own `_schema`,
 * those are shaped for this single-channel machine: channel "A" only, sensitivity 1.0, haptics
 * on, and at most ONE `yelling` event per step carrying `db_over_baseline`. An empty step is fed
 * as `0.0 dB`, and `onLocalLoudness` must return `nudges[0].level` when the step emitted a nudge
 * and `null` otherwise; `currentLevel()` must equal `levels["A"]` after every step.
 */
class NudgeStateMachineVectorsTest {

    private fun loadDoc(): JsonObject {
        val stream = NudgeStateMachineVectorsTest::class.java.getResourceAsStream("/policy_vectors/nudge_policy.json")
        assertNotNull(stream, "policy_vectors/nudge_policy.json missing from the test classpath — did syncPolicyVectors run?")
        val text = stream.bufferedReader().use { it.readText() }
        return Json.parseToJsonElement(text).jsonObject
    }

    private fun watchCases(doc: JsonObject): List<JsonObject> =
        doc["cases"]!!.jsonArray.map { it.jsonObject }.filter { case ->
            case["applies_to"]!!.jsonArray.any { it.jsonPrimitive.content == "watch" }
        }

    @Test
    fun fixtureIsTheServerSchemaVersionThisTestUnderstands() {
        val doc = loadDoc()
        assertEquals(1, doc["_schema"]!!.jsonObject["version"]!!.jsonPrimitive.int)
        // A future fixture edit that drops every watch case would leave this test vacuously green;
        // the Python driver requires the same named scenarios, so pin them here too.
        val names = watchCases(doc).map { it["name"]!!.jsonPrimitive.content }.toSet()
        for (required in listOf(
            "below_threshold_no_nudge",
            "single_nudge_then_sustain_is_silent",
            "cooldown_is_strictly_greater_than",
            "sustained_observation_refreshes_clock",
            "stepwise_deescalation_3_to_0",
            "full_decay_then_fresh_escalation",
        )) {
            assertTrue(required in names, "watch-tagged case '$required' missing from nudge_policy.json")
        }
    }

    @Test
    fun everyWatchTaggedCaseReplaysIdentically() {
        val doc = loadDoc()
        val cases = watchCases(doc)
        assertTrue(cases.isNotEmpty(), "no watch-tagged cases in nudge_policy.json")

        for (case in cases) {
            val name = case["name"]!!.jsonPrimitive.content
            val config = case["config"]!!.jsonObject
            // Shape guard (mirrors the Python driver's
            // test_watch_applicable_cases_match_kotlin_machine_shape): a mis-tagged case must fail
            // loudly here rather than be silently mis-fed to the single-channel machine.
            assertEquals(listOf("A"), config["channels"]!!.jsonArray.map { it.jsonPrimitive.content }, "$name: watch cases are channel A only")
            for (sub in config["subscriptions"]!!.jsonArray.map { it.jsonObject }) {
                assertEquals(1.0, sub["sensitivity"]!!.jsonPrimitive.double, "$name: watch cases run at sensitivity 1.0")
                assertEquals(true, sub["haptics"]!!.jsonPrimitive.content.toBoolean(), "$name: watch cases have haptics on")
            }

            val sm = NudgeStateMachine(cooldownS = config["cooldown_s"]!!.jsonPrimitive.double)
            val inputs = case["inputs"]!!.jsonArray.map { it.jsonObject }
            val expected = case["expected"]!!.jsonArray.map { it.jsonObject }
            assertEquals(inputs.size, expected.size, "$name: one expected entry per input step")

            inputs.zip(expected).forEachIndexed { i, (step, want) ->
                val t = step["t"]!!.jsonPrimitive.double
                val events = step["events"]!!.jsonArray.map { it.jsonObject }
                assertTrue(events.size <= 1, "$name step $i: at most one event per watch step")
                val db = events.firstOrNull()?.let { e ->
                    assertEquals("yelling", e["vector"]!!.jsonPrimitive.content, "$name step $i: watch steps are yelling-only")
                    e["db_over_baseline"]!!.jsonPrimitive.double
                } ?: 0.0

                val got = sm.onLocalLoudness(db, t)

                val nudges = want["nudges"]!!.jsonArray.map { it.jsonObject }
                if (nudges.isEmpty()) {
                    assertNull(got, "$name step $i (t=$t, db=$db): expected no nudge, got level $got")
                } else {
                    assertEquals(1, nudges.size, "$name step $i: single-channel machine emits at most one nudge")
                    assertEquals(
                        nudges[0]["level"]!!.jsonPrimitive.int, got,
                        "$name step $i (t=$t, db=$db): emitted level",
                    )
                }
                assertEquals(
                    want["levels"]!!.jsonObject["A"]!!.jsonPrimitive.int, sm.currentLevel(),
                    "$name step $i (t=$t, db=$db): current level",
                )
            }
        }
    }
}
