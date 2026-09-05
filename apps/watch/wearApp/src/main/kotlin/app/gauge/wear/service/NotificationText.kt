package app.gauge.wear.service

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.wear.control.ControllerState
import app.gauge.wear.ui.displayLabel

/**
 * Pure text builder for [SentinelService]'s ongoing foreground notification. Kept free of any
 * Android dependency so it's testable on the plain JVM (see NotificationTextTest) — the service
 * itself just calls this once per tick and hands the result to NotificationManager.
 */
fun notificationText(state: ControllerState): String = when (state.sentinel) {
    SentinelState.DISARMED -> "Off"
    SentinelState.ARMED -> "On · ${modeLabel(state.mode)}"
    SentinelState.STREAMING -> when {
        // Companion (Tier B): no mic, no episode — the phone listens; the watch renders nudges.
        state.mode == Mode.COMPANION && state.online -> "Companion · phone listens"
        state.mode == Mode.COMPANION -> "Companion · offline"
        state.online -> "Episode active"
        else -> "Episode active · offline · local nudges"
    }
    SentinelState.COOLDOWN -> "Cooling down"
}

// Was a byte-identical duplicate of [Mode.displayLabel]; delegates to keep the mode -> label
// mapping in one source.
private fun modeLabel(mode: Mode): String = mode.displayLabel()
