package app.gauge.wear.service

import app.gauge.wear.control.ControllerState
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import kotlin.test.Test
import kotlin.test.assertEquals

private fun state(
    sentinel: SentinelState,
    mode: Mode = Mode.STANDARD,
    online: Boolean = true,
) = ControllerState(
    sentinel = sentinel,
    mode = mode,
    online = online,
    channelLevels = emptyMap(),
    lastVector = null,
    sparkline = emptyList(),
)

class NotificationTextTest {

    @Test
    fun disarmedIsOff() {
        assertEquals("Off", notificationText(state(SentinelState.DISARMED)))
    }

    @Test
    fun onShowsModeLabel() {
        assertEquals(
            "On · Standard",
            notificationText(state(SentinelState.ARMED, mode = Mode.STANDARD)),
        )
        assertEquals(
            "On · Battery Saver",
            notificationText(state(SentinelState.ARMED, mode = Mode.BATTERY_SAVER)),
        )
        assertEquals(
            "On · Session",
            notificationText(state(SentinelState.ARMED, mode = Mode.SESSION)),
        )
    }

    @Test
    fun streamingOnlineShowsEpisodeActive() {
        assertEquals("Episode active", notificationText(state(SentinelState.STREAMING, online = true)))
    }

    @Test
    fun streamingOfflineShowsLocalNudgesNotice() {
        assertEquals(
            "Episode active · offline · local nudges",
            notificationText(state(SentinelState.STREAMING, online = false)),
        )
    }

    @Test
    fun cooldownShowsCoolingDown() {
        assertEquals("Cooling down", notificationText(state(SentinelState.COOLDOWN)))
    }
}
