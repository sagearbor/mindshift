package app.gauge.wear.capture

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Outcome of the most recent `ACTION_RETRO_CAPTURE` request — `null` until the
 * first attempt this process lifetime. Same StateFlow-bus pattern as
 * [app.gauge.wear.control.ControllerStateBus]. */
enum class RetroCaptureResult { SAVED, FAILED }

object RetroCaptureBus {
    private val _lastResult = MutableStateFlow<RetroCaptureResult?>(null)
    val lastResult: StateFlow<RetroCaptureResult?> = _lastResult.asStateFlow()

    fun publish(result: RetroCaptureResult) {
        _lastResult.value = result
    }

    /** Clears the last result — called once the UI has shown it, so a re-open of the glance
     * screen doesn't replay a stale "Saved" toast from a previous, unrelated tap. */
    fun clear() {
        _lastResult.value = null
    }
}
