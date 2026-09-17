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
import app.gauge.shared.sentinel.Mode
import app.gauge.wear.control.ControllerStateBus

/**
 * The mode picker: one Chip per [Mode]. Picking one sends `ACTION_SET_MODE` (via [sendSetMode] —
 * the shared intent builder, so this never retypes the action/extra strings) and hands control
 * back to [onModeSelected] to navigate away. No mode-behavior logic lives here — what each mode
 * actually does is entirely owned by shared's `ModePolicy`/`SentinelController`.
 *
 * P4-9 (same gap SignalScreen had, sanctioned to fix here too): the currently-active mode's chip
 * renders as [ChipDefaults.primaryChipColors] (unselected: [ChipDefaults.secondaryChipColors]) —
 * same idiom [SettingsScreen]/[SignalScreen] use. Unlike [SignalScreen] (an
 * [app.gauge.wear.prefs.GaugePrefs] value), the current mode lives on [ControllerStateBus] — the
 * controller's own state, not a preference — so this reads [ControllerStateBus.state]'s mode at
 * composition and updates the
 * local highlight on every pick (the bus itself only catches up once `SentinelService` processes
 * the `ACTION_SET_MODE` intent on its handler thread — see [app.gauge.wear.tile.GaugeTileService]'s
 * KDoc for the same read-your-own-write race elsewhere in this app).
 */
@Composable
fun ModeScreen(
    onModeSelected: (Mode) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val listState = rememberScalingLazyListState()
    var selected by remember { mutableStateOf(ControllerStateBus.state.value.mode) }

    ScalingLazyColumn(
        modifier = modifier.fillMaxWidth(),
        state = listState,
    ) {
        for (mode in listOf(Mode.STANDARD, Mode.BATTERY_SAVER, Mode.SESSION, Mode.COMPANION)) {
            item {
                Chip(
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text(mode.displayLabel()) },
                    colors = if (mode == selected) {
                        ChipDefaults.primaryChipColors()
                    } else {
                        ChipDefaults.secondaryChipColors()
                    },
                    onClick = {
                        sendSetMode(context, mode)
                        selected = mode
                        onModeSelected(mode)
                    },
                )
            }
        }
    }
}
