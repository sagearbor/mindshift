package app.gauge.wear.complication

import android.app.PendingIntent
import android.content.Intent
import androidx.wear.watchface.complications.data.ComplicationData
import androidx.wear.watchface.complications.data.ComplicationType
import androidx.wear.watchface.complications.data.PlainComplicationText
import androidx.wear.watchface.complications.data.RangedValueComplicationData
import androidx.wear.watchface.complications.datasource.ComplicationRequest
import androidx.wear.watchface.complications.datasource.SuspendingComplicationDataSourceService
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.wear.control.ControllerState
import app.gauge.wear.control.ControllerStateBus
import app.gauge.wear.ui.MainActivity

/**
 * Watch-face complication: an escalation gauge (P4-4). RANGED_VALUE is the only type declared
 * (see manifest) — the arc fills 0..[MAX_LEVEL] with the worst nudge level reported across
 * [ControllerState.channelLevels] while an episode ([SentinelState.STREAMING]) is live, and sits
 * at 0 the rest of the time, with the on-face text/content-description carrying the On/Off state
 * instead (see [complicationValue]'s own KDoc for the exact mapping).
 *
 * Reads the exact same [ControllerStateBus] snapshot [app.gauge.wear.tile.GaugeTileService] and
 * [app.gauge.wear.ui.GaugeViewModel] do, and "on" means the same thing here it means everywhere
 * else in this app: any [SentinelState] other than DISARMED (see [app.gauge.wear.ui.GaugeViewModel]'s
 * `isOn`) — not just the ARMED state specifically.
 *
 * UPDATE FREQUENCY: the manifest still declares a periodic 600s (`UPDATE_PERIOD_SECONDS`) poll as
 * a backstop, but P4-4 also wires the previously-documented-as-future-work push path —
 * `SentinelService.publishAndNotify` calls
 * [androidx.wear.watchface.complications.datasource.ComplicationDataSourceUpdateRequester]
 * `.requestUpdateAll()` — so the arc tracks escalation live instead of up to 10 minutes late.
 * PUSH ON CHANGE ONLY (review fix, round 2): `SentinelService` does NOT call this on every
 * republish — `tick()` runs on a ~1s cadence for the entire armed session (not just episodes), so
 * an unconditional push would fire 3600+ times per armed hour against this rate-limited API.
 * `SentinelService` instead compares each tick's [complicationValue] output against the last value
 * it actually pushed and only calls `requestUpdateAll()` when that pure projection changed — see
 * `SentinelService.pushFaceUpdates`'s own KDoc for the full reasoning.
 */
class ArmedComplicationService : SuspendingComplicationDataSourceService() {

    override suspend fun onComplicationRequest(request: ComplicationRequest): ComplicationData =
        rangedValueData(ControllerStateBus.state.value)

    override fun getPreviewData(type: ComplicationType): ComplicationData? {
        if (type != ComplicationType.RANGED_VALUE) return null
        return rangedValueData(PREVIEW_STATE)
    }

    private fun rangedValueData(state: ControllerState): ComplicationData {
        val (value, label) = complicationValue(state)
        val description = when {
            state.sentinel == SentinelState.STREAMING -> "Gauge escalation level ${value.toInt()}"
            state.sentinel != SentinelState.DISARMED -> "Gauge on, ${state.mode.displayNameForDescription()}"
            else -> "Gauge off"
        }

        return RangedValueComplicationData.Builder(
            value = value,
            min = MIN_LEVEL,
            max = MAX_LEVEL,
            contentDescription = PlainComplicationText.Builder(description).build(),
        )
            .setText(PlainComplicationText.Builder(label).build())
            .setTapAction(launchMainActivityPendingIntent())
            .build()
    }

    private fun launchMainActivityPendingIntent(): PendingIntent {
        val intent = Intent(this, MainActivity::class.java)
        return PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_IMMUTABLE)
    }

    private companion object {
        // Preview shown in the watch-face editor: a mid-escalation episode snapshot — shows the
        // arc actually filling, the most compelling "this is what it looks like when it matters"
        // state for a RANGED_VALUE complication (an On/Off-only preview would look identical to a
        // static badge and wouldn't demonstrate the gauge at all).
        val PREVIEW_STATE = ControllerState(
            sentinel = SentinelState.STREAMING,
            mode = Mode.STANDARD,
            online = true,
            channelLevels = mapOf("A" to 2),
            lastVector = null,
            sparkline = emptyList(),
        )
    }
}

private fun Mode.displayNameForDescription(): String = when (this) {
    Mode.STANDARD -> "standard mode"
    Mode.BATTERY_SAVER -> "battery saver mode"
    Mode.SESSION -> "session mode"
}

/** Bounds of the RANGED_VALUE arc — see [complicationValue]'s KDoc. */
internal const val MIN_LEVEL = 0f
internal const val MAX_LEVEL = 3f

/**
 * Pure value/text pair [rangedValueData] builds the escalation-gauge complication from — kept
 * free of every `androidx.wear.watchface.complications.data`/Android class so it's plain-JVM unit
 * testable (see ComplicationContentTest; same fallback [app.gauge.wear.tile.TileStrings] documents
 * for the equivalent tile-layout split).
 *
 * - While an episode is live ([SentinelState.STREAMING]): value is the worst level across
 *   [ControllerState.channelLevels] (missing/empty -> 0), clamped to `[`[MIN_LEVEL]`,`[MAX_LEVEL]`]`
 *   since channel levels are reported by the server (`onNudge`) and aren't guaranteed in-range —
 *   text is `"Level <n>"`, so the arc fill and the on-face number always agree.
 * - Otherwise (ARMED/COOLDOWN/DISARMED — no episode to show escalation for): value is always 0
 *   (empty arc), text carries the On/Off state instead — "On" for any non-DISARMED state (matching
 *   the same on/off rule used everywhere else in this app), "Off" for DISARMED.
 */
internal fun complicationValue(state: ControllerState): Pair<Float, String> {
    if (state.sentinel != SentinelState.STREAMING) {
        val text = if (state.sentinel != SentinelState.DISARMED) "On" else "Off"
        return MIN_LEVEL to text
    }
    val level = (state.channelLevels.values.maxOrNull() ?: 0).coerceIn(MIN_LEVEL.toInt(), MAX_LEVEL.toInt())
    return level.toFloat() to "Level $level"
}
