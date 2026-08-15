package app.gauge.wear.complication

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.wear.control.ControllerState
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * [complicationValue] is the pure value/text pair [ArmedComplicationService] builds its
 * RANGED_VALUE complication from — see that class's own KDoc for why this, not
 * [ArmedComplicationService] itself, is what's unit tested (same fallback the tile module already
 * documents in TileLayoutTest: the real `androidx.wear.watchface.complications.data` builder call
 * graph is exercised indirectly by `assembleDebug`/`lintDebug`, not a dedicated unit test).
 */
class ComplicationContentTest {

    private fun state(
        sentinel: SentinelState,
        channelLevels: Map<String, Int> = emptyMap(),
    ): ControllerState = ControllerState(
        sentinel = sentinel,
        mode = Mode.STANDARD,
        online = true,
        channelLevels = channelLevels,
        lastVector = null,
        sparkline = emptyList(),
    )

    @Test
    fun disarmedIsZeroAndOff() {
        val (value, text) = complicationValue(state(SentinelState.DISARMED))

        assertEquals(0f, value)
        assertEquals("Off", text)
    }

    @Test
    fun armedWithNoEpisodeIsZeroAndOn() {
        val (value, text) = complicationValue(state(SentinelState.ARMED))

        assertEquals(0f, value)
        assertEquals("On", text)
    }

    @Test
    fun cooldownWithNoEpisodeIsZeroAndOn() {
        // COOLDOWN is "on" by the same rule ArmedComplicationService's old shortTextData used
        // (armed := sentinel != DISARMED) — only STREAMING counts as an episode for the value.
        val (value, text) = complicationValue(state(SentinelState.COOLDOWN))

        assertEquals(0f, value)
        assertEquals("On", text)
    }

    @Test
    fun streamingWithNoChannelsYetIsLevelZero() {
        val (value, text) = complicationValue(state(SentinelState.STREAMING))

        assertEquals(0f, value)
        assertEquals("Level 0", text)
    }

    @Test
    fun streamingReportsWorstChannelLevel() {
        val (value, text) = complicationValue(
            state(SentinelState.STREAMING, channelLevels = mapOf("A" to 1, "B" to 2)),
        )

        assertEquals(2f, value)
        assertEquals("Level 2", text)
    }

    @Test
    fun streamingClampsAnOutOfRangeLevelToThree() {
        val (value, text) = complicationValue(
            state(SentinelState.STREAMING, channelLevels = mapOf("A" to 7)),
        )

        assertEquals(3f, value)
        assertEquals("Level 3", text)
    }

    @Test
    fun streamingClampsANegativeLevelToZero() {
        val (value, text) = complicationValue(
            state(SentinelState.STREAMING, channelLevels = mapOf("A" to -1)),
        )

        assertEquals(0f, value)
        assertEquals("Level 0", text)
    }

    /**
     * P4-4 review round 2: `SentinelService.pushFaceUpdates` caches the last-pushed
     * [complicationValue] output and only calls `requestUpdateAll()` when a fresh call's result
     * differs from that cache — a push-on-change optimization needed because `tick()` runs on a
     * ~1s cadence for the entire armed session, not just episodes. That optimization is only
     * correct because [complicationValue] is a pure function of its input: two calls given
     * `equals()` [ControllerState] values MUST return `equals()` results, or the cache comparison
     * would be comparing apples to oranges. This test documents/locks in that guarantee directly,
     * rather than leaving it as an unstated assumption behind `SentinelService`'s dedup logic.
     */
    @Test
    fun sameStateProducesEqualProjectionsAcrossCalls() {
        val first = state(SentinelState.STREAMING, channelLevels = mapOf("A" to 1, "B" to 2))
        val second = state(SentinelState.STREAMING, channelLevels = mapOf("A" to 1, "B" to 2))

        assertEquals(complicationValue(first), complicationValue(second))
    }

    /**
     * P4-4 review round 2 (companion to [sameStateProducesEqualProjectionsAcrossCalls]): the push
     * dedup in `SentinelService` compares [complicationValue]'s *output*, not the raw
     * [ControllerState] — deliberately, since most fields on [ControllerState] (sparkline,
     * lastVector, online, ...) have no bearing on the complication face at all. Two
     * states that differ ONLY in fields [complicationValue] never reads must still project to the
     * same value/text pair, or comparing projections instead of whole snapshots would be unsound
     * (it would under-count real changes only if projection equality could mask a display
     * difference — this test is the guarantee that it can't, for the fields that vary
     * tick-to-tick regardless of escalation state).
     */
    @Test
    fun projectionIgnoresFieldsOutsideItsOwnInputs() {
        val streamingLowChurn = ControllerState(
            sentinel = SentinelState.STREAMING,
            mode = Mode.STANDARD,
            online = true,
            channelLevels = mapOf("A" to 1),
            lastVector = null,
            sparkline = emptyList(),
        )
        val streamingHighChurn = ControllerState(
            sentinel = SentinelState.STREAMING,
            mode = Mode.SESSION,
            online = false,
            channelLevels = mapOf("A" to 1),
            lastVector = "yelling",
            sparkline = listOf(1.0, 2.0, 3.0),
        )

        assertEquals(complicationValue(streamingLowChurn), complicationValue(streamingHighChurn))
    }
}
