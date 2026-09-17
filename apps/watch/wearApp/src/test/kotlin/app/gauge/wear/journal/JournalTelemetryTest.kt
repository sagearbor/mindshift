package app.gauge.wear.journal

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive

class JournalTelemetryTest {

    @Test
    fun payloadCarriesEveryBatteryAndCounterField() {
        val data = journalTelemetryData(
            batteryPct = 87,
            charging = false,
            journalUploads = 3,
            journalUploadFailures = 1,
            journalDrops = 2,
            micDutyState = "DUTY_CYCLED",
        )
        assertEquals(87, data.getValue("battery_pct").jsonPrimitive.int)
        assertEquals(false, data.getValue("charging").jsonPrimitive.boolean)
        assertEquals(3, data.getValue("journal_uploads").jsonPrimitive.int)
        assertEquals(1, data.getValue("journal_upload_failures").jsonPrimitive.int)
        assertEquals(2, data.getValue("journal_drops").jsonPrimitive.int)
        assertEquals("DUTY_CYCLED", data.getValue("mic_duty_state").jsonPrimitive.content)
    }

    @Test
    fun unreadableBatteryIsAnExplicitNullNeverAFabricatedNumber() {
        val data = journalTelemetryData(
            batteryPct = null,
            charging = null,
            journalUploads = 0,
            journalUploadFailures = 0,
            journalDrops = 0,
            micDutyState = "CONTINUOUS",
        )
        assertEquals(JsonNull, data.getValue("battery_pct"))
        assertEquals(JsonNull, data.getValue("charging"))
        // The keys are still present — a reader can tell "unreadable" from "not reported".
        assertTrue("battery_pct" in data && "charging" in data)
    }
}
