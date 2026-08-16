package app.gauge.wear.ui

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.ScalingLazyColumn
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.rememberScalingLazyListState
import app.gauge.wear.control.DiagLog
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.haptics.HapticPatterns
import app.gauge.wear.haptics.RealVibratorPort
import app.gauge.wear.prefs.GaugePrefs
import app.gauge.wear.telemetry.Telemetry
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * The settings screen (P4-3): three sections — "Feel the buzzes" (v0.2.4 demo, see below),
 * "Center display" ([CENTER_DISPLAY_OPTIONS] -> [GaugePrefs.setCenterDisplay] — v0.2.4 Addendum 2's
 * sparkline-vs-dial switcher; this Settings row is the ONLY way to switch — the center tap itself
 * is always the arm/disarm toggle on both views, see [GlanceScreen]'s own gesture-contract KDoc),
 * and "Pulse speed" ([PULSE_OPTIONS] -> [GaugePrefs.setPulseIntervalMs], read every STREAMING
 * window by [app.gauge.wear.control.SentinelController]'s `pulseIntervalMs` supplier — see
 * [app.gauge.wear.haptics.PulseEngine] KDoc for what each value means). Unlike [ModeScreen]/
 * [SignalScreen] (single selection, navigate straight back), both pickers live on this screen and
 * stay put after a tap so the wearer can review/change either without re-opening the screen — the
 * currently selected chip renders as [ChipDefaults.primaryChipColors] (unselected: [ChipDefaults.
 * secondaryChipColors]). The demo section has no selection state at all — every row is a one-shot
 * play. (v0.2.4 also removed the old "Center number" METER/CALM section entirely — see
 * [GaugeViewModel]'s KDoc; "Center display" is an unrelated, newer setting with the same
 * `Center:` label prefix.)
 */
@Composable
fun SettingsScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val listState = rememberScalingLazyListState()

    var pulseInterval by remember { mutableStateOf(GaugePrefs.pulseIntervalMs(context)) }
    var centerDisplay by remember { mutableStateOf(GaugePrefs.centerDisplay(context)) }

    // v0.2.4 "Feel the buzzes": a demo HapticDirector of this screen's own, on the real vibrator.
    // Uses demo()/playPulse() (never onNudge), so nothing here touches the service's nudge dedupe
    // or the sentinel — this screen can buzz freely whether the sentinel is on or off.
    //
    // Review fix (Important, round 1): same diag as SentinelService's own HapticDirector (see its
    // onCreate) so this screen's reportHapticPath breadcrumb reaches the same telemetry channel —
    // without it, demo taps produced zero haptic-path telemetry, defeating the breadcrumb's whole
    // purpose of confirming the physical path without screen-capturing the device.
    val demoDiag = remember { DiagLog { l, t, m -> runCatching { Telemetry.log(l, t, m) } } }
    val demoDirector = remember { HapticDirector(RealVibratorPort(context.applicationContext), diag = demoDiag) }
    val demoScope = rememberCoroutineScope()
    var sampleJob by remember { mutableStateOf<Job?>(null) }

    ScalingLazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
    ) {
        item { Text(text = "Feel the buzzes", style = MaterialTheme.typography.caption1) }
        for ((level, label) in DEMO_NUDGE_LEVELS) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(label) },
                    colors = ChipDefaults.secondaryChipColors(),
                    onClick = { demoDirector.demo("A", level) },
                )
            }
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Pulse train sample") },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = {
                    // One band per step, gentle -> max, 300ms apart (> the 250ms never-merge
                    // effective floor, so sample pulses can never smear). The isActive guard keeps
                    // impatient re-taps from overlapping two trains.
                    if (sampleJob?.isActive != true) {
                        sampleJob = demoScope.launch {
                            for (db in listOf(0.0, 3.0, 6.0, 9.0)) {
                                demoDirector.playPulse(HapticPatterns.pulseBandFor(db))
                                delay(300L)
                            }
                        }
                    }
                },
            )
        }
        item {
            Chip(
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Partner cue (B)") },
                colors = ChipDefaults.secondaryChipColors(),
                onClick = { demoDirector.demo("B", 2) },
            )
        }
        item { Text(text = "Center display", style = MaterialTheme.typography.caption1) }
        for ((value, label) in CENTER_DISPLAY_OPTIONS) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(label) },
                    colors = if (value == centerDisplay) {
                        ChipDefaults.primaryChipColors()
                    } else {
                        ChipDefaults.secondaryChipColors()
                    },
                    onClick = {
                        GaugePrefs.setCenterDisplay(context, value)
                        centerDisplay = value
                    },
                )
            }
        }
        item { Text(text = "Pulse speed", style = MaterialTheme.typography.caption1) }
        for ((value, label) in PULSE_OPTIONS) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(label) },
                    colors = if (value == pulseInterval) {
                        ChipDefaults.primaryChipColors()
                    } else {
                        ChipDefaults.secondaryChipColors()
                    },
                    onClick = {
                        GaugePrefs.setPulseIntervalMs(context, value)
                        pulseInterval = value
                    },
                )
            }
        }
    }
}

private val DEMO_NUDGE_LEVELS = listOf(
    1 to "Nudge · level 1",
    2 to "Nudge · level 2",
    3 to "Nudge · level 3",
)

private val PULSE_OPTIONS = listOf(
    "250" to "Pulse: 0.25s",
    "500" to "Pulse: 0.5s",
    "1000" to "Pulse: 1s",
    "off" to "Pulse: Off",
)

private val CENTER_DISPLAY_OPTIONS = listOf(
    "SPARKLINE" to "Center: Sparkline",
    "DIAL" to "Center: Dial",
)
