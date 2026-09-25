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
    companionAcked: Boolean = false,
) = ControllerState(
    sentinel = sentinel,
    mode = mode,
    online = online,
    channelLevels = emptyMap(),
    lastVector = null,
    sparkline = emptyList(),
    companionAcked = companionAcked,
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

    // 2026-09-25: "phone listens" is a claim about the SERVER (it registered the companion), so it
    // waits for the companion_ack the client used to drop. Before the ack the socket is merely open.
    @Test
    fun companionSaysConnectingUntilAckedThenPhoneListens() {
        assertEquals(
            "Companion · connecting",
            notificationText(state(SentinelState.STREAMING, mode = Mode.COMPANION, online = true)),
        )
        assertEquals(
            "Companion · phone listens",
            notificationText(state(SentinelState.STREAMING, mode = Mode.COMPANION, online = true, companionAcked = true)),
        )
        assertEquals(
            "Companion · offline",
            notificationText(state(SentinelState.STREAMING, mode = Mode.COMPANION, online = false, companionAcked = true)),
        )
    }
}
