package app.gauge.wear.tile

import app.gauge.shared.sentinel.Mode
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * [tileLayout] itself builds real `androidx.wear.protolayout`/`androidx.wear.tiles` builder
 * objects — its own KDoc (and [TileStrings]'s) explain why calling `toLayoutElementProto()` on
 * the result isn't usable from a plain JVM unit test here (protolayout's generated proto message
 * classes are a runtime-only, not compile-time, transitive dependency on the unit-test
 * classpath). Per the task brief's documented fallback, this TDDs [tileStrings] instead — the
 * pure string/id builder [tileLayout] is built from — which covers exactly the same content
 * ("On · Standard"/"Off" status text, and the correct click id) without touching the
 * protolayout builder API at all.
 */
class TileLayoutTest {

    @Test
    fun onStandardShowsOnStatusAndTurnOffClick() {
        val strings = tileStrings(armed = true, mode = Mode.STANDARD)

        assertEquals("On · Standard", strings.statusText)
        assertEquals("Turn off", strings.buttonText)
        assertEquals(CLICK_ID_DISARM, strings.clickId)
    }

    @Test
    fun offBatterySaverShowsOffAndTurnOnClick() {
        val strings = tileStrings(armed = false, mode = Mode.BATTERY_SAVER)

        assertEquals("Off", strings.statusText)
        assertEquals("Turn on", strings.buttonText)
        assertEquals(CLICK_ID_ARM, strings.clickId)
    }

    /**
     * P4-4 review round 2: `SentinelService.pushFaceUpdates` caches the last-pushed [tileStrings]
     * output and only calls `TileService.getUpdater(...).requestUpdate` when a fresh call's result
     * differs from that cache (same push-on-change reasoning as `ComplicationContentTest`'s own
     * stability tests for `complicationValue`) — correct only because [tileStrings] is pure. This
     * locks that guarantee in directly rather than leaving it an unstated assumption.
     */
    @Test
    fun sameInputsProduceEqualStringsAcrossCalls() {
        val first = tileStrings(armed = true, mode = Mode.SESSION)
        val second = tileStrings(armed = true, mode = Mode.SESSION)

        assertEquals(first, second)
    }

    @Test
    fun tileLayoutBuildsWithoutThrowingForBothStates() {
        // Smoke check that the real protolayout builder call graph (LayoutElementBuilders /
        // ModifiersBuilders / ActionBuilders) succeeds end to end for both states — the actual
        // content assertions live above, against tileStrings, per this file's own KDoc.
        tileLayout(armed = true, mode = Mode.STANDARD)
        tileLayout(armed = false, mode = Mode.BATTERY_SAVER)
    }
}
