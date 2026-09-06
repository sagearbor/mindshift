package app.gauge.shared

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Drives [NudgeVocabulary] from `server/tests/fixtures/policy_vectors/nudge_vocabulary.json` —
 * the same file `server/tests/test_nudge_vocabulary_vectors.py` and
 * `apps/mobile/__tests__/nudgeVocabulary.test.ts` replay, synced onto this test classpath by
 * `shared/build.gradle.kts`'s `syncPolicyVectors` task. The JSON is the contract: if a case fails
 * here, the fix is to the DIVERGING runtime, not to the fixture.
 */
class NudgeVocabularyVectorsTest {

    private fun loadDoc(): JsonObject {
        val stream = NudgeVocabularyVectorsTest::class.java
            .getResourceAsStream("/policy_vectors/nudge_vocabulary.json")
        assertNotNull(stream, "policy_vectors/nudge_vocabulary.json missing from the test classpath — did syncPolicyVectors run?")
        return Json.parseToJsonElement(stream.bufferedReader().use { it.readText() }).jsonObject
    }

    private fun cases(doc: JsonObject): Map<String, JsonObject> =
        doc["cases"]!!.jsonArray.associate { it.jsonObject["name"]!!.jsonPrimitive.content to it.jsonObject }

    @Test
    fun schemaVersionAndConstantsMatchTheModule() {
        val doc = loadDoc()
        assertEquals(1, doc["_schema"]!!.jsonObject["version"]!!.jsonPrimitive.int)
        val constants = doc["constants"]!!.jsonObject
        assertEquals(NudgeVocabulary.MIN_GAP_MS, constants["min_gap_ms"]!!.jsonPrimitive.long)
        assertEquals(NudgeVocabulary.POSITIVE_CAP_S, constants["positive_cap_s"]!!.jsonPrimitive.double)
        assertEquals(NudgeVocabulary.MIN_ON_MS, constants["min_on_ms"]!!.jsonPrimitive.long)
        assertEquals(NudgeVocabulary.MIN_AMPLITUDE, constants["min_amplitude"]!!.jsonPrimitive.int)
        val exceptions = doc["_schema"]!!.jsonObject["haptic_encoding"]!!.jsonObject["haptic_gap_exceptions"]!!
            .jsonArray.map { it.jsonPrimitive.content }
        assertEquals(NudgeVocabulary.HAPTIC_GAP_EXCEPTIONS, exceptions)
    }

    @Test
    fun theModuleIsFieldForFieldTheFixtureInOrder() {
        val spec = loadDoc()["vocabulary"]!!.jsonArray.map { it.jsonObject }
        assertEquals(spec.size, NudgeVocabulary.ALL.size)
        spec.forEachIndexed { i, s ->
            val e = NudgeVocabulary.ALL[i]
            assertEquals(s["code"]!!.jsonPrimitive.content, e.code)
            assertEquals(s["vector"]!!.jsonPrimitive.content, e.vector)
            assertEquals(s["icon"]!!.jsonPrimitive.content, e.icon, "icon for ${e.code}")
            assertEquals(s["color"]!!.jsonPrimitive.content, e.color.name.lowercase(), "colour for ${e.code}")
            assertEquals(s["name"]!!.jsonPrimitive.content, e.name)
            assertEquals(s["meaning"]!!.jsonPrimitive.content, e.meaning, "meaning for ${e.code}")
            assertEquals(s["polarity"]!!.jsonPrimitive.content, e.polarity.name.lowercase())
            assertEquals(s["sources"]!!.jsonArray.map { it.jsonPrimitive.content }, e.sources)
            val flash = s["flash_text"]!!
            assertEquals(if (flash is JsonNull) null else flash.jsonPrimitive.content, e.flashText)
            assertEquals(s["summary_only"]!!.jsonPrimitive.boolean, e.summaryOnly)
            assertEquals(s["watch_only"]!!.jsonPrimitive.boolean, e.watchOnly)
            assertEquals(s["levels"]!!.jsonArray.map { it.jsonPrimitive.int }, e.levels)

            val haptic = s["haptic"]!!
            if (haptic is JsonNull) {
                assertNull(e.haptic, "${e.code} must have no cue")
                return@forEachIndexed
            }
            val cues = haptic.jsonObject
            assertNotNull(e.haptic)
            assertEquals(cues.keys.map { it.toInt() }.sorted(), e.haptic!!.keys.sorted())
            for ((levelKey, wave) in cues) {
                val got = e.haptic!!.getValue(levelKey.toInt())
                val obj = wave.jsonObject
                assertEquals(
                    obj["timings_ms"]!!.jsonArray.map { it.jsonPrimitive.long },
                    got.timingsMs,
                    "${e.code} L$levelKey timings",
                )
                assertEquals(
                    obj["amplitudes"]!!.jsonArray.map { it.jsonPrimitive.int },
                    got.amplitudes,
                    "${e.code} L$levelKey amplitudes",
                )
            }
        }
    }

    @Test
    fun everyCodeIsUniqueAndStable() {
        val expected = cases(loadDoc()).getValue("every_code_is_unique_and_stable")["expected_codes"]!!
            .jsonArray.map { it.jsonPrimitive.content }
        assertEquals(expected, NudgeVocabulary.ALL.map { it.code })
        assertEquals(expected.size, NudgeVocabulary.ALL.map { it.vector }.toSet().size)
    }

    @Test
    fun alertCodesHaveThreeLevelsPositivesHaveOne() {
        val case = cases(loadDoc()).getValue("alert_codes_have_three_levels_positives_have_one")
        fun expected(key: String) = case[key]!!.jsonArray.map { it.jsonPrimitive.content }
        assertEquals(
            expected("expected_alert_codes"),
            NudgeVocabulary.ALL.filter { it.polarity == NudgePolarity.ALERT }.map { it.code },
        )
        assertEquals(
            expected("expected_positive_codes"),
            NudgeVocabulary.ALL.filter { it.polarity == NudgePolarity.POSITIVE }.map { it.code },
        )
        for (e in NudgeVocabulary.ALL) {
            val want = if (e.polarity == NudgePolarity.ALERT) listOf(1, 2, 3) else listOf(1)
            assertEquals(want, e.levels, "levels for ${e.code}")
        }
    }

    @Test
    fun everyWaveformIsWellFormed() {
        for (e in NudgeVocabulary.ALL) {
            val haptic = e.haptic ?: continue
            for ((level, wave) in haptic) {
                val t = wave.timingsMs
                val a = wave.amplitudes
                assertEquals(t.size, a.size, "${e.code} L$level length")
                assertTrue(t.size % 2 == 0 && t.size >= 2, "${e.code} L$level must be OFF/ON pairs")
                assertEquals(0L, t[0], "${e.code} L$level must open with a zero delay")
                t.forEachIndexed { i, ms ->
                    if (i % 2 == 0) {
                        assertEquals(0, a[i], "${e.code} L$level slot $i is OFF")
                        if (i > 0) {
                            val floor = if (e.vector in NudgeVocabulary.HAPTIC_GAP_EXCEPTIONS) 0L else NudgeVocabulary.MIN_GAP_MS
                            assertTrue(ms >= floor, "${e.code} L$level gap ${ms}ms merges two taps")
                        }
                    } else {
                        assertTrue(a[i] <= 255, "${e.code} L$level amplitude ${a[i]}")
                        // The measured perceptibility floor — it binds the soft positives too: a
                        // cue nobody can feel is not a soft cue, it is a missing one.
                        assertTrue(ms >= NudgeVocabulary.MIN_ON_MS, "${e.code} L$level tap ${ms}ms is under the floor")
                        assertTrue(a[i] >= NudgeVocabulary.MIN_AMPLITUDE, "${e.code} L$level amplitude ${a[i]} is under the floor")
                    }
                }
            }
        }
    }

    @Test
    fun theLevelIsCarriedByRhythm() {
        val expected = cases(loadDoc()).getValue("level_is_carried_by_rhythm")["expected_on_ms"]!!.jsonObject
        for ((code, arr) in expected) {
            val want = arr.jsonArray.map { it.jsonPrimitive.long }
            val got = (1..3).map { NudgeVocabulary.hapticFor(code, it)!!.onMs }
            assertEquals(want, got, "ON ms for $code")
            assertTrue(got[0] < got[1] && got[1] < got[2], "$code must grow by rhythm alone")
        }
    }

    @Test
    fun thePositiveCapAdmitsOnePerTwoMinutes() {
        val case = cases(loadDoc()).getValue("positive_cap_is_one_per_two_minutes")
        val gate = PositiveNudgeGate()
        val admitted = case["offers"]!!.jsonArray.map { it.jsonObject }.filter {
            gate.admit(it["t"]!!.jsonPrimitive.double)
        }.map { it["t"]!!.jsonPrimitive.double to it["code"]!!.jsonPrimitive.content }
        val expected = case["expected_admitted"]!!.jsonArray.map { it.jsonObject }
            .map { it["t"]!!.jsonPrimitive.double to it["code"]!!.jsonPrimitive.content }
        assertEquals(expected, admitted)
    }

    @Test
    fun lookupsResolveDetectorNamesAndNeverInventAnIcon() {
        assertEquals("📈", NudgeVocabulary.iconFor("yelling"))
        assertEquals("📈", NudgeVocabulary.iconFor("heated"))
        assertNull(NudgeVocabulary.iconFor("nonesuch"))
        assertEquals("H", NudgeVocabulary.forVector("aggressive_tone")?.code)
        assertNull(NudgeVocabulary.forCode("Z"))
        assertEquals("H", NudgeVocabulary.codeForVectors(listOf("airtime", "yelling")))
        assertEquals("C", NudgeVocabulary.codeForVectors(listOf("airtime", "interrupting")))
        assertNull(NudgeVocabulary.codeForVectors(listOf("nonesuch")))
        assertNull(NudgeVocabulary.codeForVectors(emptyList()))
    }

    @Test
    fun positivesAreUnleveledKNeverBuzzesAndABadLevelIsSilent() {
        for (code in listOf("D", "E", "R")) {
            assertEquals(NudgeVocabulary.hapticFor(code, 1), NudgeVocabulary.hapticFor(code, 3), code)
        }
        assertNull(NudgeVocabulary.hapticFor("K", 1))
        assertTrue(NudgeVocabulary.forCode("K")!!.summaryOnly)
        assertNull(NudgeVocabulary.hapticFor("H", 0))
        assertNull(NudgeVocabulary.hapticFor("H", 4))
    }
}
