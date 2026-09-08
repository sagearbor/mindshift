package app.gauge.wear.ui

import android.content.Context
import android.content.Intent
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.service.SentinelService

/**
 * The only place `ui/` builds [SentinelService] intents — every action/extra name here is
 * imported from [SentinelService]'s own constants, never retyped as a string literal, so a rename
 * on the service side can't silently desync from the UI side.
 *
 * ARM promotes the service to foreground (it isn't running yet, or wasn't foregrounded), so it
 * goes through [Context.startForegroundService]. DISARM and SET_MODE target an already-running
 * service and don't need to (re)claim foreground state, so they use the plain
 * [Context.startService].
 */
internal fun sendArm(context: Context) {
    val intent = Intent(context, SentinelService::class.java).setAction(SentinelService.ACTION_ARM)
    context.startForegroundService(intent)
}

internal fun sendDisarm(context: Context) {
    val intent = Intent(context, SentinelService::class.java).setAction(SentinelService.ACTION_DISARM)
    context.startService(intent)
}

internal fun sendSetMode(context: Context, mode: Mode) {
    val intent = Intent(context, SentinelService::class.java)
        .setAction(SentinelService.ACTION_SET_MODE)
        .putExtra(SentinelService.EXTRA_MODE, mode.name)
    context.startService(intent)
}

/** Display label shared by [GlanceScreen]'s mode Chip and [ModeScreen]'s three Chips. */
internal fun Mode.displayLabel(): String = when (this) {
    Mode.STANDARD -> "Standard"
    Mode.BATTERY_SAVER -> "Battery Saver"
    Mode.SESSION -> "Session"
    Mode.COMPANION -> "Companion — phone listens"
}

/** Display label shared by [GlanceScreen]'s "Signal: <label> ▾" Chip and [SignalScreen]'s four
 * Chips (Task 11). */
internal fun SignalKind.displayLabel(): String = when (this) {
    SignalKind.VOLUME -> "Volume"
    SignalKind.HEART_RATE -> "Heart Rate"
    SignalKind.MOVEMENT -> "Movement"
    SignalKind.SPEAKING_RATE -> "Speaking Rate"
}
