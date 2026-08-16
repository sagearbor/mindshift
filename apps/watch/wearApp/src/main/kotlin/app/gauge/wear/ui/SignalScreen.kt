package app.gauge.wear.ui

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.wear.compose.material.Chip
import androidx.wear.compose.material.ChipDefaults
import androidx.wear.compose.material.ScalingLazyColumn
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.rememberScalingLazyListState
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.prefs.GaugePrefs

/**
 * The live-meter signal picker (Task 11): one Chip per [SignalKind]. Picking one persists the
 * choice via [GaugePrefs.setSelectedSignal] — read back every processing window by
 * `SentinelController`'s `selectedSignal` supplier (see `SentinelService`'s wiring of it) — and
 * hands control back to [onSignalSelected] to navigate away, same shape as [ModeScreen].
 *
 * Picking [SignalKind.HEART_RATE] additionally requests `BODY_SENSORS` right here: this is
 * deliberately the *only* place in the app that asks for it — `SentinelService`'s own arm path
 * never requests it (see its KDoc), instead degrading honestly to "no HR reading" when it's
 * missing. A denial here just means the meter (and hr_spike vector) stay off; nothing crashes or
 * re-prompts on a loop.
 *
 * P4-6 (Issue B fix, review round 2): [onRequestBodySensors] is invoked instead of this composable
 * owning a `rememberLauncherForActivityResult` of its own. The original version launched the
 * request AND synchronously called `onSignalSelected` (which pops this screen off the back stack)
 * in the same click handler — that disposes this composable, and with it the launcher's
 * `ActivityResultRegistry` registration, within milliseconds, before the system permission dialog
 * the wearer is still looking at can return a result. The grant/deny outcome was silently dropped.
 * [MainActivity] now hosts the launcher itself, one level up in its own composition (outside the
 * `NavHost`, so it isn't torn down by navigation) — see its own KDoc — and passes the `launch` call
 * down as [onRequestBodySensors], so the result reaches [MainActivity]'s callback regardless of
 * whether this screen is still on screen by the time the dialog closes. Also more robust for any
 * future screen that needs the same grant, since the launcher isn't tied to this one composable's
 * lifetime.
 *
 * P4-9: the currently-selected chip renders as [ChipDefaults.primaryChipColors] (unselected:
 * [ChipDefaults.secondaryChipColors]) — same idiom [SettingsScreen]'s pickers already use.
 * Selection is read from [GaugePrefs.selectedSignal] at composition and updated locally on every
 * pick (not re-read from prefs after each pick — this screen pops itself off the back stack right
 * after, same as before), so the wearer can always tell which signal is currently driving the live
 * meter before picking a different one.
 */
@Composable
fun SignalScreen(
    onSignalSelected: (SignalKind) -> Unit,
    onRequestBodySensors: () -> Unit = {},
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val listState = rememberScalingLazyListState()
    var selected by remember { mutableStateOf(GaugePrefs.selectedSignal(context)) }

    ScalingLazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
    ) {
        for (kind in SIGNAL_ORDER) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(kind.displayLabel()) },
                    colors = if (kind.name == selected) {
                        ChipDefaults.primaryChipColors()
                    } else {
                        ChipDefaults.secondaryChipColors()
                    },
                    onClick = {
                        GaugePrefs.setSelectedSignal(context, kind.name)
                        selected = kind.name
                        if (kind == SignalKind.HEART_RATE) {
                            onRequestBodySensors()
                        }
                        onSignalSelected(kind)
                    },
                )
            }
        }
    }
}

private val SIGNAL_ORDER = listOf(
    SignalKind.VOLUME,
    SignalKind.HEART_RATE,
    SignalKind.MOVEMENT,
    SignalKind.SPEAKING_RATE,
)
