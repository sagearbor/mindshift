package app.gauge.wear.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.navigation.SwipeDismissableNavHost
import androidx.wear.compose.navigation.composable
import androidx.wear.compose.navigation.rememberSwipeDismissableNavController
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.control.ControllerStateBus
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.MeterBus
import app.gauge.wear.control.MicSource
import app.gauge.wear.control.ScalarSource
import app.gauge.wear.prefs.GaugePrefs
import app.gauge.wear.sensors.AccelSource
import app.gauge.wear.sensors.HrSource
import app.gauge.wear.service.MicReader
import app.gauge.wear.telemetry.Telemetry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * Entry point for the Gauge Wear OS app. THIN by design: wires [GaugeViewModel] to
 * [ControllerStateBus] (the only UI feed — see its own KDoc) and hosts navigation between the
 * glance screen, the mode picker, the live-meter signal picker (Task 11), and (P4-3) the settings
 * screen. All state -> UI
 * mapping lives in [GaugeViewModel]; all sentinel-behavior decisions live in
 * shared/`SentinelController` — nothing here re-decides either.
 *
 * Task 12: also drives [PreviewMeterEngine] — the live meter while the sentinel is OFF — on its
 * own mic, entirely separate from [app.gauge.wear.service.SentinelService]'s. [onStart]/[onStop]
 * (not [onCreate]/`onDestroy`) own the outer lifetime: the preview mic must only exist while this
 * screen is actually visible, and must be fully released the instant it isn't (screen off, app
 * backgrounded, navigated away to another app) — never left running unattended. That's a *distinct*
 * concern from the never-two-readers guarantee (round-2 hardening): tapping the ring to arm keeps
 * this screen visible and STARTED the whole time, so onStart/onStop alone would leave the preview
 * `AudioRecord` open in RECORDING state throughout an entire armed session, racing
 * [app.gauge.wear.service.SentinelService]'s own `AudioRecord` for the platform's capture priority.
 *
 * P4-8: [PreviewMeterEngine]'s `onAcquire`/`onRelease` callbacks fire per-[SignalKind], not
 * per-active/inactive-transition — [acquirePreviewSensor]/[releasePreviewSensor] acquire/release
 * exactly ONE source at a time (whichever the wearer currently has selected), swapping it the
 * instant the picker selection changes (never on every step while it holds steady), and releasing
 * it entirely the instant the engine observes the service go active. A fresh source is acquired
 * the instant it observes DISARMED again with a selection in hand. Both are invoked synchronously
 * inside `step()` on the polling loop's own thread, so there is never a concurrent read in flight
 * on the newly-released source when the release happens (see [PreviewMeterEngine]'s own KDoc).
 * [onStop]'s release-everything-unconditionally remains as the outer backstop for the "screen not
 * visible at all" case — every underlying `stop()`/`release()` call is itself idempotent.
 *
 * P4-6 (Issue B fix): the preview engine used to be wired with `hr = null, accel = null`, so
 * Heart Rate/Movement never showed anything while the sentinel was off — the wearer couldn't set
 * those signals up without arming first. [previewAccelSource]/[previewManagedHr] (P4-8: now only
 * ever the ONE currently-selected source, not all three at once — see above) extend the mic's
 * acquire/release treatment to a real [AccelSource] (always constructed — no permission needed)
 * and a lazily-created [HrSource] (only once BODY_SENSORS is observed granted), mirroring
 * [app.gauge.wear.service.SentinelService]'s own "stop sensors before yielding the mic" posture.
 * [previewManagedHr] is the same kind of mutable indirection as [ManagedMic] — it lets
 * [refreshHrPreviewSource] wire in a freshly created [HrSource] the moment BODY_SENSORS is granted
 * mid-visit without recreating [previewEngine] itself.
 *
 * P4-6 review round 2: the BODY_SENSORS `rememberLauncherForActivityResult` lives in [onCreate]'s
 * `setContent` block, one level up from the `NavHost` (see its own placement below) — NOT inside
 * [SignalScreen] itself. [SignalScreen]'s Heart Rate chip both launches that request AND
 * synchronously pops itself off the back stack in the same click handler; a launcher registered
 * *inside* [SignalScreen] would have its `ActivityResultRegistry` registration torn down by that
 * pop within milliseconds — before the system permission dialog the wearer is still looking at can
 * return a result — silently dropping the grant. Hosting it in the composition that contains the
 * `NavHost` (never disposed by in-app navigation) means the result always reaches
 * [refreshHrPreviewSource], regardless of which screen is on top by the time the dialog closes.
 */
class MainActivity : ComponentActivity() {

    /** Mutable indirection over the live preview [MicReader] instance so [PreviewMeterEngine]'s
     * onAcquire/onRelease callbacks can swap it out (release while the mic isn't the selected
     * signal or the service is active, recreate once VOLUME/SPEAKING_RATE is selected and DISARMED
     * again) without recreating [previewEngine] itself — preserving its trackers/baseline across a
     * quick arm/disarm cycle instead of resetting them every time. */
    private class ManagedMic : MicSource {
        var reader: MicReader? = null
        override fun readWindow(): ShortArray? = reader?.readWindow()
    }

    /** Same indirection as [ManagedMic], for the preview's [HrSource] (P4-6/Issue B): [source] is
     * `null` until BODY_SENSORS is first observed granted (see [refreshHrPreviewSource]), so
     * [PreviewMeterEngine] can hold a permanent non-null [ScalarSource] reference from the moment
     * it's constructed and simply see `null` readings — same honest-degradation shape as every
     * other missing-source case — until a grant wires one in. */
    private class ManagedScalar : ScalarSource {
        @Volatile var source: ScalarSource? = null
        override fun latest(): Double? = source?.latest()
    }

    private var previewManagedMic: ManagedMic? = null
    private var previewManagedHr: ManagedScalar? = null
    private var previewAccelSource: AccelSource? = null

    // M2 (cheap visibility hardening, review-sanctioned): both fields are written from the
    // BODY_SENSORS permission launcher's callback (main-thread Compose recomposition path) and
    // read from acquirePreviewSensor/releasePreviewSensor (also main-thread today, but on the
    // engine's own onAcquire/onRelease seam — see class KDoc) — @Volatile costs nothing on a
    // reference/boolean field and removes any doubt about a stale cached read across that seam.
    @Volatile private var previewHrSource: HrSource? = null
    private var previewEngine: PreviewMeterEngine? = null
    private var previewJob: Job? = null

    /** Which [SignalKind] (if any) this visit's preview currently has acquired — the last
     * [acquirePreviewSensor]/[releasePreviewSensor] edge, per P4-8: exactly one source is ever held
     * at a time, so this is a single kind rather than a blanket boolean. P4-6 review round 2's
     * rationale still applies: set directly by those two methods rather than inferred from
     * [ManagedMic.reader] being non-null, since a failed [MicReader.start] (busy `AudioRecord`,
     * etc.) would otherwise make [refreshHrPreviewSource] think nothing was acquired even though
     * accel/HR are in fact running for a non-mic selection. */
    @Volatile private var previewAcquiredKind: SignalKind? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val viewModel = GaugeViewModel(
            bus = ControllerStateBus.state,
            scope = lifecycleScope,
            selectedSignal = { selectedSignalPref() },
            centerDisplayPref = { GaugePrefs.centerDisplay(applicationContext) },
        )

        setContent {
            MaterialTheme {
                val navController = rememberSwipeDismissableNavController()
                val uiState by viewModel.uiState.collectAsState()

                // P4-6 review round 2: deliberately hosted here, not inside SignalScreen — see this
                // class's own KDoc for why a launcher registered inside the screen that also pops
                // itself off the back stack in the same click handler loses the permission result.
                val bodySensorsLauncher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission(),
                ) {
                    // Honest degradation either way — HrSource.latest() just returns null if
                    // denied. Always re-check so a grant obtained here takes effect immediately.
                    refreshHrPreviewSource()
                }

                SwipeDismissableNavHost(
                    navController = navController,
                    startDestination = ROUTE_GLANCE,
                ) {
                    composable(ROUTE_GLANCE) {
                        GlanceScreen(
                            uiState = uiState,
                            onOpenModeScreen = { navController.navigate(ROUTE_MODE) },
                            onOpenSignalScreen = { navController.navigate(ROUTE_SIGNAL) },
                            onOpenSettingsScreen = { navController.navigate(ROUTE_SETTINGS) },
                            onOpenSignInScreen = { navController.navigate(ROUTE_SIGN_IN) },
                            onOpenCouplesScreen = { navController.navigate(ROUTE_COUPLES) },
                        )
                    }
                    composable(ROUTE_MODE) {
                        ModeScreen(
                            onModeSelected = { navController.popBackStack() },
                        )
                    }
                    composable(ROUTE_SIGNAL) {
                        SignalScreen(
                            onSignalSelected = { navController.popBackStack() },
                            onRequestBodySensors = {
                                bodySensorsLauncher.launch(Manifest.permission.BODY_SENSORS)
                            },
                        )
                    }
                    composable(ROUTE_SETTINGS) {
                        SettingsScreen()
                    }
                    composable(ROUTE_SIGN_IN) {
                        SignInScreen(onSignedIn = { navController.popBackStack() })
                    }
                    composable(ROUTE_COUPLES) {
                        CouplesScreen(onNeedsSignIn = { navController.navigate(ROUTE_SIGN_IN) })
                    }
                }
            }
        }
    }

    /**
     * Starts the preview engine + polling loop, but only when RECORD_AUDIO is already granted —
     * same fail-soft posture as everywhere else RECORD_AUDIO is touched in this app (see
     * [MicReader.start]'s own "requires RECORD_AUDIO to already be granted" contract): a missing
     * permission just means no live preview this visit, not a crash. Re-checked on every onStart
     * (not just the first), so a grant obtained via [GlanceScreen]'s own permission prompt takes
     * effect the next time this screen becomes visible.
     *
     * Deliberately does NOT start a [MicReader] here — [PreviewMeterEngine]'s first `step()` call
     * does that itself via [PreviewMeterEngine.onAcquire] the moment it observes the service isn't
     * active AND VOLUME/SPEAKING_RATE is selected (see class KDoc's "same code path as reacquire"
     * note), so there's exactly one place that ever calls [MicReader.start] for the preview mic,
     * not two. [previewAccelSource]/[previewManagedHr] (P4-6/Issue B) follow the identical
     * deferred-start posture: constructed here, but only actually started from
     * [acquirePreviewSensor] on that same first `step()`'s acquire edge, for whichever signal is
     * actually selected (P4-8).
     */
    override fun onStart() {
        super.onStart()
        if (previewJob != null) return // already running (e.g. a config change didn't tear us down)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return
        }

        previewAcquiredKind = null
        val managedMic = ManagedMic()
        previewManagedMic = managedMic
        val managedHr = ManagedScalar()
        previewManagedHr = managedHr
        val accel = AccelSource(applicationContext, previewDiag())
        previewAccelSource = accel
        refreshHrPreviewSource() // lazily wires previewHrSource in if BODY_SENSORS is already granted

        val engine = PreviewMeterEngine(
            mic = managedMic,
            hr = managedHr,
            accel = accel,
            selectedSignal = { selectedSignalPref() },
            publish = { MeterBus.preview.value = it },
            isServiceActive = { ControllerStateBus.state.value.sentinel != SentinelState.DISARMED },
            onAcquire = { acquirePreviewSensor(it, managedMic, accel) },
            onRelease = { releasePreviewSensor(it, managedMic, accel) },
            publishAvailability = { MeterBus.previewAvailability.value = it },
            publishSparkline = { MeterBus.previewSparkline.value = it },
        )
        previewEngine = engine

        previewJob = lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                engine.step()
                // Paces every selection, not just the mic-backed ones: reading a ScalarSource
                // (HEART_RATE/MOVEMENT) never blocks the way mic.readWindow() does, so without an
                // explicit delay a selection with no source wired (or yielding to an active
                // service) would spin this Dispatchers.Default thread instead of idling.
                delay(STEP_INTERVAL_MS)
            }
        }
    }

    /** [GaugePrefs.selectedSignal] parsed into a [SignalKind], falling back to [SignalKind.VOLUME]
     * on a corrupt/unrecognized stored value — same shape [SentinelService] and [PreviewMeterEngine]
     * itself both already use for this exact pref, and now also what [GaugeViewModel]'s
     * `selectedSignal` supplier reads for the P4-6 signal-chip fix. */
    private fun selectedSignalPref(): SignalKind =
        runCatching { SignalKind.valueOf(GaugePrefs.selectedSignal(applicationContext)) }
            .getOrDefault(SignalKind.VOLUME)

    private fun previewDiag(): DiagLog = DiagLog { l, t, m -> runCatching { Telemetry.log(l, t, m) } }

    /**
     * (Re)acquires the preview mic. Idempotent by construction — [managedMic] only ever holds at
     * most one live [MicReader] at a time, and the engine only ever calls this on a genuine
     * acquire edge (never redundantly on every step — see [PreviewMeterEngine]'s own KDoc), so
     * there's no double-start to guard against here.
     */
    private fun startPreviewMic(managedMic: ManagedMic) {
        val reader = MicReader()
        try {
            reader.start()
        } catch (t: Throwable) {
            // Fail-soft (mirrors SentinelService's own startStreaming()/arm() posture): a busy or
            // misbehaving AudioRecord here must never crash the activity — just means no live
            // preview meter until the next acquire edge (next disarm, or next onStart).
            runCatching { Telemetry.log("error", TAG, "preview mic.start() failed: $t") }
            return
        }
        managedMic.reader = reader
    }

    /**
     * Releases the preview mic. Idempotent — [MicReader.release] is itself idempotent (safe even
     * if [managedMic] never held a reader, e.g. [startPreviewMic] failed above), and clearing
     * [ManagedMic.reader] first means a `step()` that races this call on this same thread (there is
     * no such race — see [PreviewMeterEngine]'s KDoc — but this is also the exact same method
     * [onStop] uses as its backstop from the *main* thread) sees a clean `null` mic rather than a
     * half-released one.
     */
    private fun releasePreviewMic(managedMic: ManagedMic) {
        val reader = managedMic.reader
        managedMic.reader = null
        reader?.release()
    }

    /**
     * [PreviewMeterEngine.onAcquire] (P4-8): starts ONLY [kind]'s underlying source — the mic for
     * VOLUME/SPEAKING_RATE, [accel] for MOVEMENT, or the preview [HrSource] for HEART_RATE (lazily
     * created by [refreshHrPreviewSource] the moment BODY_SENSORS is granted) — never all three
     * regardless of selection, which is the whole battery point of P4-8. [AccelSource.start]/
     * [HrSource.start] are already individually fail-soft (each catches and diag-logs its own
     * `Throwable` internally — see their own KDocs), so no extra `try/catch` is needed here.
     */
    private fun acquirePreviewSensor(kind: SignalKind, managedMic: ManagedMic, accel: AccelSource) {
        when (kind) {
            SignalKind.VOLUME, SignalKind.SPEAKING_RATE -> startPreviewMic(managedMic)
            SignalKind.MOVEMENT -> accel.start()
            SignalKind.HEART_RATE -> {
                refreshHrPreviewSource() // lazily creates one if BODY_SENSORS is granted
                previewHrSource?.start()
            }
        }
        previewAcquiredKind = kind
    }

    /**
     * [PreviewMeterEngine.onRelease] (P4-8): stops ONLY [kind]'s underlying source — mirrors
     * [acquirePreviewSensor]'s per-kind dispatch, and [app.gauge.wear.service.SentinelService]'s
     * own sensor lifecycle so the preview never leaves a listener registered for a signal that
     * isn't (or is no longer) selected. [AccelSource.stop]/[HrSource.stop] are already individually
     * fail-soft, same as their `start()` counterparts.
     */
    private fun releasePreviewSensor(kind: SignalKind, managedMic: ManagedMic, accel: AccelSource) {
        previewAcquiredKind = null
        when (kind) {
            SignalKind.VOLUME, SignalKind.SPEAKING_RATE -> releasePreviewMic(managedMic)
            SignalKind.MOVEMENT -> accel.stop()
            SignalKind.HEART_RATE -> previewHrSource?.stop()
        }
    }

    /**
     * Re-checks BODY_SENSORS and lazily creates+wires this visit's [HrSource] the first time it's
     * seen granted (P4-6/Issue B) — a no-op once [previewHrSource] already exists, or if the
     * preview isn't running this visit at all ([previewManagedHr] is only non-null between
     * [onStart] and [onStop]/RECORD_AUDIO-denied-bail-out). Called from [onStart] itself AND from
     * the BODY_SENSORS launcher's result callback (see its `setContent` wiring and this class's own
     * KDoc for why that launcher result, not an Activity lifecycle callback, is the hook — and why
     * the launcher itself is hosted outside [SignalScreen], not inside it), so a grant obtained
     * mid-visit — right there on the signal picker — takes effect immediately instead of waiting for
     * the next `onStart`.
     *
     * If the preview currently has HEART_RATE as its acquired kind ([previewAcquiredKind] — P4-8:
     * only true when HR is actually the selected signal, since that's now the only case anything
     * ever acquires it in the first place), the freshly wired source is started right away rather
     * than waiting for the next acquire edge, which wouldn't come until the next selection change
     * or arm/disarm cycle.
     */
    private fun refreshHrPreviewSource() {
        val managedHr = previewManagedHr ?: return
        if (previewHrSource != null) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        // I1: distinct tag from the service's own HrSource instance (default "HrSource") — both
        // can be diag-logging in the same device session (this preview while disarmed, the
        // service's own instance once armed), and a shared literal tag would make a telemetry
        // pull ambiguous about which HR lifecycle a given log line belongs to.
        val hr = HrSource(applicationContext, previewDiag(), tag = "HrSourcePreview")
        previewHrSource = hr
        managedHr.source = hr
        if (previewAcquiredKind == SignalKind.HEART_RATE) hr.start()
    }

    /** Tears down the preview loop, mic, and sensors the instant the screen isn't visible — the
     * backstop for the "screen not visible at all" case; see class KDoc for the separate per-kind
     * acquire/release edges ([acquirePreviewSensor]/[releasePreviewSensor]) that handle it while
     * the screen IS visible. Releases everything unconditionally (mic AND accel AND hr) rather than
     * only [previewAcquiredKind]'s source — deliberately belt-and-suspenders: every underlying
     * `stop()`/`release()` call is itself idempotent (and [HrSource]'s own [SensorLifecycleGate]
     * now dedupes it), so there's no cost to also stopping sources that were never acquired this
     * visit. */
    override fun onStop() {
        super.onStop()
        previewJob?.cancel()
        previewJob = null
        previewEngine?.close()
        previewEngine = null
        previewManagedMic?.let(::releasePreviewMic)
        previewManagedMic = null
        previewAccelSource?.stop()
        previewAccelSource = null
        previewHrSource?.stop()
        previewHrSource = null
        previewManagedHr = null
        previewAcquiredKind = null
    }

    private companion object {
        const val ROUTE_GLANCE = "glance"
        const val ROUTE_MODE = "mode"
        const val ROUTE_SIGNAL = "signal"
        const val ROUTE_SETTINGS = "settings"
        const val ROUTE_SIGN_IN = "sign_in"
        const val ROUTE_COUPLES = "couples"
        const val STEP_INTERVAL_MS = 1000L
        const val TAG = "MainActivity"
    }
}
