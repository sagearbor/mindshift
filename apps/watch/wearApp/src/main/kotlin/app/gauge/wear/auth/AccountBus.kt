package app.gauge.wear.auth

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * The UI feed for "is this watch signed in right now" (Wave C) — same StateFlow-
 * bus pattern as [app.gauge.wear.control.ControllerStateBus]/[app.gauge.wear.ui.
 * MeterBus], so [app.gauge.wear.ui.GaugeViewModel] can combine it the same way it
 * already combines those. [AccountPrefs] is the source of truth on disk; this bus
 * is just the live signal a Composable can `collectAsState()` without touching
 * `SharedPreferences` (and its Context requirement) directly from Compose.
 */
object AccountBus {
    private val _signedIn = MutableStateFlow(false)
    val signedIn: StateFlow<Boolean> = _signedIn.asStateFlow()

    fun publish(signedIn: Boolean) {
        _signedIn.value = signedIn
    }
}
