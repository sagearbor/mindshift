package app.gauge.wear.control

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalAvailability
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * The safe channel for reading [SentinelController.state] from a third thread (the UI, T9).
 *
 * [SentinelController.state] itself is fine to poll from the service's own loop thread (it takes
 * the controller's internal lock), but nothing about the *service* guarantees a UI thread ever
 * sees a coherent read — [SentinelService] is the only thing driving [SentinelController], so it
 * publishes a fresh snapshot into this bus after every [SentinelController.tick] (and after every
 * arm/disarm/setMode). Consumers just collect [state]; they never touch [SentinelController]
 * directly.
 */
object ControllerStateBus {
    private val _state = MutableStateFlow(
        ControllerState(
            sentinel = SentinelState.DISARMED,
            mode = Mode.STANDARD,
            online = true,
            channelLevels = emptyMap(),
            lastVector = null,
            sparkline = emptyList(),
        ),
    )

    val state: StateFlow<ControllerState> = _state.asStateFlow()

    fun publish(newState: ControllerState) {
        _state.value = newState
    }
}

/**
 * The UI feed for [app.gauge.wear.ui.PreviewMeterEngine]'s live meter while the sentinel is OFF
 * (Task 12) — a second, independent channel from [ControllerStateBus.state] since the preview
 * engine runs entirely outside [SentinelController] (no controller instance exists while
 * DISARMED with the app just sitting on screen). [app.gauge.wear.ui.GaugeViewModel] combines this
 * with [ControllerStateBus.state]'s own `meter`, service meter always winning when both are
 * present — see its own KDoc.
 */
object MeterBus {
    val preview = MutableStateFlow<MeterReading?>(null)

    /** P4-10: the preview engine's own availability, mirroring [preview] above — `null`/no-source
     * cases surface as [SignalAvailability.UNKNOWN] (the default), same honest-degradation
     * contract as [preview] itself. Nothing writes this yet in this task (see [GaugeViewModel]'s
     * constructor KDoc); a future task wires the preview engine to it. */
    val previewAvailability = MutableStateFlow(SignalAvailability.UNKNOWN)

    /** v0.2.4 (Addendum 2): the preview engine's own sparkline series — raw signal-unit values
     * for the CURRENTLY selected signal, oldest first, capped at 30. Cleared (published empty) on
     * a signal switch, on service yield, and on engine close — see PreviewMeterEngine.emit/
     * clearSparkline. Empty means "no honest series to draw", never "draw a flat fake". */
    val previewSparkline = MutableStateFlow<List<Double>>(emptyList())
}
