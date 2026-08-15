package app.gauge.wear.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.wear.tiles.TileService
import androidx.wear.watchface.complications.datasource.ComplicationDataSourceUpdateRequester
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.CaptureMode
import app.gauge.shared.signals.MicDutyCycle
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.BuildConfig
import app.gauge.wear.R
import app.gauge.wear.complication.ArmedComplicationService
import app.gauge.wear.complication.complicationValue
import app.gauge.wear.control.ControllerState
import app.gauge.wear.control.ControllerStateBus
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.EpisodeIdFactory
import app.gauge.wear.control.EpisodeWs
import app.gauge.wear.control.RealEpisodeWs
import app.gauge.wear.control.SentinelController
import app.gauge.wear.control.WsFactory
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.haptics.Pulse
import app.gauge.wear.haptics.PulseChainDecision
import app.gauge.wear.haptics.PulseChainGate
import app.gauge.wear.haptics.RealVibratorPort
import app.gauge.wear.prefs.GaugePrefs
import app.gauge.wear.sensors.AccelSource
import app.gauge.wear.sensors.HrAction
import app.gauge.wear.sensors.HrDemandPolicy
import app.gauge.wear.sensors.HrSource
import app.gauge.wear.telemetry.Telemetry
import app.gauge.wear.tile.GaugeTileService
import app.gauge.wear.tile.TileStrings
import app.gauge.wear.tile.tileStrings
import app.gauge.wear.ui.MainActivity
import java.util.UUID
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Foreground (microphone-type) service hosting [SentinelController]. THIN BY DESIGN: every method
 * here either (a) forwards an intent action straight to a controller call, or (b) does Android
 * plumbing (notification, foreground state, mic lifecycle) that has no decision content of its
 * own. All sentinel logic lives in SentinelController / shared — nothing here re-decides it.
 *
 * Threading: [SentinelController] requires arm()/disarm()/setMode()/tick() to all be driven by a
 * single thread (see its own KDoc). That thread is [handlerThread] here — every controller call,
 * including the ones triggered by ACTION_ARM/DISARM/SET_MODE intents arriving on the main thread,
 * is posted onto it rather than invoked directly.
 */
class SentinelService : Service() {

    private var handlerThread: HandlerThread? = null
    private var handler: Handler? = null
    private var micReader: MicReader? = null
    private var controller: SentinelController? = null
    private var hrSource: HrSource? = null
    private var accelSource: AccelSource? = null

    // Night 2 hardening (T12 review minor): service-owned, cancellable scope for retro-capture
    // uploads. Previously onActionRetroCapture created a fresh, unscoped
    // CoroutineScope(Dispatchers.IO) per call that nothing ever cancelled — including on
    // onDestroy — so an in-flight upload from a torn-down service kept running (and could still
    // publish to RetroCaptureBus) after the service itself was gone, unlike every other
    // handler-thread-owned resource here (mic/hr/accel), all of which stop in onDestroy.
    // SupervisorJob so one upload's failure/cancellation can't cancel a concurrent one launched
    // from a second retro-capture request on the same scope. Created in onCreate, cancelled in
    // onDestroy — same teardown discipline as every other field in this class.
    private var retroCaptureScope: CoroutineScope? = null

    // P5-3: lazy HR registration policy — see manageHrRegistration's own KDoc. Handler-thread-only,
    // same contract as loopRunning/hrRegistered below.
    private val hrDemandPolicy = HrDemandPolicy()

    // Handler-thread-only: whether hrSource is currently registered (mirrors HrDemandPolicy's own
    // `registered` parameter back to it each tick). Never read/written off the loop thread.
    private var hrRegistered = false

    // Review fix (P5-3 round 1): without BODY_SENSORS granted, hrRegistered never flips true, so
    // HrDemandPolicy re-emits REGISTER on every ~1s tick for the whole demand stretch (HR_VECTOR_
    // SUBSCRIBED is unconditionally true today, so every STREAMING episode alone demands it) —
    // logging on every one of those ticks would flood the fixed-size, no-dedup Telemetry
    // DebugRing, this app's only on-device debugging channel, evicting every other diagnostic
    // within minutes. Set on the first permission-denied REGISTER attempt, reset whenever demand
    // is no longer blocked on that specific reason (a real REGISTER succeeds, an UNREGISTER fires,
    // or an evaluation comes back NONE — i.e. demand ended or is already satisfied) so the next
    // distinct stretch still warns once. Handler-thread-only, same contract as hrRegistered.
    private var hrPermissionWarnedThisDemand = false

    // P4-3: the same HapticDirector wired into `controller` below, kept as its own field so the
    // pulse-repeat loop (startPulseChain) can reuse it directly rather than reaching back through
    // the controller — SentinelController itself never calls HapticDirector.playPulse (see its own
    // KDoc): this is the SOLE place any pulse is physically played. pulseHandler is a SEPARATE
    // Handler from `handler` above, bound to the main looper rather than handlerThread's:
    // handlerThread's own thread spends most of its time blocked inside MicReader's real
    // (synchronous) AudioRecord.read() call for the duration of each ~1s window, so a repeat
    // scheduled on that same handler would never actually interleave sub-second pulses during that
    // block — this needs a thread of its own.
    private var haptics: HapticDirector? = null
    private var pulseHandler: Handler? = null
    private var pulseRunnable: Runnable? = null

    // M2 (teardown guard): set true FIRST in onDestroy, before any of that method's own cleanup
    // runs — pulseHandler is bound to the main looper (see its own field KDoc), a different
    // thread than handlerThread's, so a pulse `Runnable` already queued there (or one racing in
    // via managePulseRepeat -> startPulseChain from handlerThread, mid-teardown) could otherwise
    // observe fields onDestroy is in the middle of nulling out. @Volatile so a check on either
    // thread always sees the latest write, not a stale cached one.
    @Volatile private var destroyed = false

    // Review fix (P4-3 round 2): race-proofing for the pulse-repeat loop — see PulseChainGate's
    // own KDoc for the incident this exists to prevent (a stale Runnable that Handler.
    // removeCallbacks failed to intercept firing anyway, or worse, orphaning itself into a
    // permanently self-re-arming loop). Handler-thread-only bookkeeping (like loopRunning above):
    // the (pulse, effectiveIntervalMs) pair the currently-running chain was started with, or null
    // if none is running — compared against each tick's fresh verdict (PulseChainDecision.decide)
    // so the chain only restarts when something actually changed, not unconditionally every tick
    // (unconditional restarts were the other half of the original race: a fresh restart roughly
    // once a second is also roughly once a second the old chain's tail and the new chain's
    // immediate first fire could land on top of each other).
    private val pulseChainGate = PulseChainGate()
    private var currentPulseChain: Pair<Pulse, Long>? = null

    // Handler-thread-only: whether the tick loop is currently scheduled. Only ever read/written
    // from the handler thread, so (like SentinelController's own core fields) needs no locking.
    private var loopRunning = false

    // P5-3: duty-cycle schedule state — handler-thread-only, same contract as loopRunning. Reset
    // (alongside lastCaptureMode below) in stopSentinel()/onDestroy() so a re-arm always starts
    // continuous rather than resuming mid-cycle from a previous armed session.
    private val micDutyCycle = MicDutyCycle()

    // Handler-thread-only, same contract as loopRunning — push-on-change cache for
    // reportCaptureMode's telemetry, mirroring lastPushedComplication/lastPushedTileStrings' own
    // discipline below.
    private var lastCaptureMode: CaptureMode = CaptureMode.CONTINUOUS

    // Handler-thread-only: the last SentinelState passed through publishAndNotify. Used only to
    // detect the STREAMING -> COOLDOWN transition (episode-end) for the telemetry flush below —
    // no decision content, mirrors loopRunning's threading contract.
    private var lastPublishedSentinelState: SentinelState? = null

    // Review fix (P4-4 round 2): push-on-change caches for pushFaceUpdates, mirroring
    // lastPublishedSentinelState's own threading contract (handler-thread-only, no locking). tick()
    // runs on this ~1s cadence for the ENTIRE armed session (ARMED/COOLDOWN/STREAMING alike, not
    // just episodes — see MicReader's own 1s window), so calling requestUpdateAll/requestUpdate
    // unconditionally on every publishAndNotify would fire 3600+ times per armed hour against two
    // rate-limited Wear APIs whose documented usage is push-on-data-change, even though
    // complicationValue(snapshot) (5 possible outputs) and tileStrings(armed, mode) are constant on
    // the overwhelming majority of those ticks. Comparing against these before pushing is what
    // makes push-on-change correct rather than just "less often": both projections are pure
    // functions of `snapshot` alone (see ComplicationContentTest / TileLayoutTest), so an unchanged
    // projection provably means an unchanged face — nothing to push. Reset to null in
    // stopSentinel()/onDestroy() so the next arm always pushes fresh rather than comparing against
    // a stale value from a previous session.
    private var lastPushedComplication: Pair<Float, String>? = null
    private var lastPushedTileStrings: TileStrings? = null

    private val tickRunnable = object : Runnable {
        override fun run() {
            val c = controller ?: return
            try {
                val now = System.currentTimeMillis()
                // P5-3 (C2): duty-cycling only ever applies while ARMED — an episode in flight, or
                // a cooldown that may still return to one, always captures continuously. Reading
                // the sentinel state from the last published snapshot (not the controller) keeps
                // this on the same handler-thread-only bookkeeping as loopRunning.
                val armedOnly = lastPublishedSentinelState == SentinelState.ARMED
                if (armedOnly && !micDutyCycle.shouldCapture(now)) {
                    micReader?.pause()
                    reportCaptureMode(CaptureMode.DUTY_CYCLED)
                    // Sleep exactly until this cycle's next ON phase rather than re-polling every
                    // second — that wait is the actual power saving. An ACTION_DISARM posted from
                    // the main thread still runs promptly (it's a separate, undelayed message on
                    // this same handler, dispatched before this delayed one regardless of how much
                    // of the sleep remains) — see onActionDisarm, which additionally cancels this
                    // exact pending message and posts an immediate tick, so the actual STOP
                    // (mic release, notification teardown, stopSelf) converges within about one
                    // cycle too, not just the disarm() call itself, rather than waiting out
                    // whatever's left of this up-to-8s sleep.
                    handler?.postDelayed(this, micDutyCycle.msUntilNextCapture(now).coerceAtLeast(MIN_DUTY_SLEEP_MS))
                    return
                }
                // Round-1 review fix (Important 1, P5-3): a controller already DISARMED by the time
                // this fires (an ACTION_DISARM landed while this exact message was pending — see
                // onActionDisarm's cancel-and-repost above, and the case-5 hand-trace in this
                // task's report for the orderings this closes) must NOT physically restart the mic
                // — a resume() with nobody about to read from it is a visible "mic re-lit right
                // after Off" trust regression, not just wasted power. c.tick() below still runs (a
                // no-op on a DISARMED controller — see SentinelController.tick()'s own guard), so
                // this invocation still converges to stopSentinel() exactly as before; stopSentinel()
                // -> micReader.release() cleanly stops+releases the record whether or not it was
                // ever resumed (AudioRecord.stop() on an already-stopped record is a no-op).
                if (lastPublishedSentinelState != SentinelState.DISARMED) {
                    // Unconditional and idempotent — see MicReader.resume()'s KDoc for why this must
                    // happen before every capturing tick, not just on the OFF -> ON edge.
                    micReader?.resume()
                    reportCaptureMode(if (armedOnly) micDutyCycle.mode(now) else CaptureMode.CONTINUOUS)
                }
                c.tick()
                // The post-tick state read/publish/notify path is inside the same try as tick()
                // itself (round-2 hardening): ControllerStateBus.publish or the NotificationManager
                // call inside publishAndNotify() (e.g. a SecurityException from a revoked
                // POST_NOTIFICATIONS permission) run unguarded on this HandlerThread just like
                // tick() does, and must degrade the same way rather than killing the process.
                val snapshot = c.state
                // Feed the observed window back in: `true` is the snap-back to continuous capture.
                micDutyCycle.onObservation(snapshot.lastWindowVoiced, now)
                if (snapshot.sentinel != SentinelState.ARMED) micDutyCycle.reset()
                publishAndNotify(snapshot)
                managePulseRepeat(snapshot)
                // v0.2.4: ARMED shout-tap — a one-shot, deliberately NOT a chain. Mutual exclusion
                // with the pulse-repeat chain is by construction: managePulseRepeat only ever
                // starts chains while snapshot.sentinel == STREAMING, and armedShoutTap is only
                // ever non-null while it stayed ARMED — the two can never fire for the same tick.
                snapshot.armedShoutTap?.let { tap -> runCatching { haptics?.playPulse(tap) } }
                if (snapshot.sentinel == SentinelState.DISARMED) {
                    // Converges explicit ACTION_DISARM and mic-loss auto-disarm (SentinelController.
                    // tick() disarms itself when the mic returns null) onto one cleanup path.
                    stopSentinel()
                } else {
                    handler?.post(this)
                }
            } catch (t: Throwable) {
                // Throwable, not Exception: an uncaught Error (e.g. from a misbehaving native/SDK
                // call somewhere in the trigger path) on this HandlerThread would otherwise kill
                // the whole process (no custom UncaughtExceptionHandler is installed) — mid-mic-
                // session, with no cleanup. Stop the sentinel cleanly instead and do NOT repost.
                Log.e(TAG, "tick failed; stopping sentinel", t)
                runCatching {
                    Telemetry.log(
                        "error",
                        TAG,
                        "tick failed; stopping sentinel: ${t.message}\n${t.stackTraceToString().take(MAX_TICK_STACK_CHARS)}",
                    )
                }
                // Best-effort, and deliberately not re-entrant with this same catch: if the
                // exception above actually came from stopSentinel() itself (called from the try
                // block's DISARMED branch), this second call must not be allowed to throw again
                // and kill the thread — runCatching absorbs that rather than recursing.
                runCatching { stopSentinel() }
            }
        }
    }

    /** Logs a duty-cycle mode transition exactly once per change (never once per tick). */
    private fun reportCaptureMode(mode: CaptureMode) {
        if (mode == lastCaptureMode) return
        lastCaptureMode = mode
        runCatching { Telemetry.log("info", TAG, "mic capture mode: $mode") }
    }

    override fun onCreate() {
        super.onCreate()
        val thread = HandlerThread("SentinelServiceLoop").apply { start() }
        handlerThread = thread
        handler = Handler(thread.looper)
        // Round-1 review fix (Important 3, P5-3): diag must exist before MicReader is constructed
        // so a failed pause()/resume() leaves a telemetry breadcrumb (see MicReader's own KDoc).
        val diag = DiagLog { l, t, m -> runCatching { Telemetry.log(l, t, m) } }
        val mic = MicReader(diag)
        micReader = mic
        val hr = HrSource(applicationContext, diag)
        hrSource = hr
        val accel = AccelSource(applicationContext, diag)
        accelSource = accel
        // Night 2 hardening (T12 review minor): see the field's own KDoc — created here alongside
        // every other per-instance resource, cancelled in onDestroy below.
        retroCaptureScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        val wsBase = wsBaseUrlFromApiBase(BuildConfig.GAUGE_API_BASE)
        // Round-1 review fix (Important, v0.2.4): same diag as mic/hr/accel above, so
        // HapticDirector's reportHapticPath breadcrumb reaches the same telemetry channel.
        val hapticDirector = HapticDirector(RealVibratorPort(applicationContext), diag = diag)
        haptics = hapticDirector
        pulseHandler = Handler(Looper.getMainLooper())
        controller = SentinelController(
            mode = Mode.STANDARD,
            mic = mic,
            wsFactory = object : WsFactory {
                override fun create(): EpisodeWs =
                    RealEpisodeWs(wsBase, app.gauge.wear.ui.currentAccountId(applicationContext))
            },
            haptics = hapticDirector,
            ids = EpisodeIdFactory { UUID.randomUUID().toString() },
            diag = diag,
            hr = hr,
            accel = accel,
            selectedSignal = { selectedSignalPref() },
            pulseIntervalMs = { pulseIntervalMsPref() },
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_ARM -> onActionArm(intent)
            ACTION_DISARM -> onActionDisarm()
            ACTION_SET_MODE -> onActionSetMode(intent)
            ACTION_RETRO_CAPTURE -> onActionRetroCapture(intent)
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        // M2 (teardown guard): set FIRST, before any other cleanup below — see the field's own
        // KDoc for why. startPulseChain() checks this and bails out rather than arming a fresh
        // repeat chain once teardown has begun.
        destroyed = true
        // Best-effort cleanup for the "service torn down without an explicit DISARM" path (task
        // switcher kill, low-memory kill, etc). Capture the current controller/mic as locals —
        // the posted lambda runs asynchronously on the handler thread, after this method has
        // already nulled out the fields below — quitSafely() lets it finish before the loop
        // stops accepting new messages.
        val c = controller
        val mic = micReader
        val hr = hrSource
        val accel = accelSource
        handler?.post {
            runCatching { c?.disarm() }
            mic?.release()
            runCatching { hr?.stop() }
            runCatching { accel?.stop() }
        }
        // Night 2 hardening (T12 review minor): cancel any in-flight (or future) retro-capture
        // upload — see the field's own KDoc. Safe to call from onDestroy's own thread (main):
        // Job.cancel() is thread-safe and doesn't require being on the scope's own dispatcher.
        runCatching { retroCaptureScope?.cancel() }
        retroCaptureScope = null
        // P4-4 (review round 2): same reset as stopSentinel() — belt-and-suspenders here since a
        // fresh SentinelService instance would start with null caches anyway, but keeps both
        // teardown paths visibly symmetric rather than relying on that implicitly.
        lastPushedComplication = null
        lastPushedTileStrings = null
        // P5-3: same belt-and-suspenders symmetry for the HR demand policy's own state.
        hrRegistered = false
        hrPermissionWarnedThisDemand = false
        hrDemandPolicy.reset()
        // P5-3: same belt-and-suspenders symmetry for the duty-cycle schedule — a fresh
        // SentinelService instance would start with a fresh MicDutyCycle/CONTINUOUS anyway, but
        // keeps both teardown paths visibly symmetric rather than relying on that implicitly.
        micDutyCycle.reset()
        lastCaptureMode = CaptureMode.CONTINUOUS
        handlerThread?.quitSafely()
        handlerThread = null
        handler = null
        controller = null
        micReader = null
        hrSource = null
        accelSource = null
        // P4-3: pulseHandler is bound to the main looper (this method's own thread) — safe to
        // clean up directly here rather than posting, unlike the handlerThread-bound work above.
        cancelPulseRepeat()
        pulseHandler = null
        haptics = null
        super.onDestroy()
    }

    // --- intent handlers (main thread) ---------------------------------------------------------

    private fun onActionArm(intent: Intent) {
        // Choke point for every arm entry point (tile, complication, third-party intents): on
        // targetSdk 34, startForeground(FOREGROUND_SERVICE_TYPE_MICROPHONE) throws
        // SecurityException if RECORD_AUDIO isn't already granted — e.g. a fresh install where the
        // user adds the tile and taps Arm before ever opening the app. Check BEFORE any
        // startForeground call and bail out to the UI's existing permission flow instead of
        // crashing.
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            Log.w(TAG, "ACTION_ARM without RECORD_AUDIO; routing to MainActivity for permission")
            startActivity(
                Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            )
            // cancels the startForegroundService obligation — without this the system kills the
            // app (~5s) via ForegroundServiceDidNotStartInTimeException
            stopSelf()
            return
        }
        val mode = modeExtra(intent)
        // Must call startForeground() promptly from onStartCommand, independent of the handler
        // thread's own timing — use whatever the bus last held as a harmless placeholder text; the
        // handler-thread arm below refreshes it within one tick.
        startForegroundNotification(ControllerStateBus.state.value)
        handler?.post { armOnLoopThread(mode) }
    }

    private fun onActionDisarm() {
        // Latency bound here is 1-2 read cycles, not 1, REGARDLESS of duty-cycle phase: if this
        // lands while tickRunnable is mid-flight (blocked in mic.readWindow()), it queues behind
        // the in-flight tick — but *which* side of that tick's own self-repost it lands on is a
        // narrow, genuine race (both are fast, non-blocking handler.post() calls from different
        // threads). If it wins the race, disarm() runs right after the current tick finishes (~1
        // cycle). If it loses, the self-repost goes first and a whole extra ~1s blocking read
        // happens before this even gets a chance to run (~2 cycles).
        //
        // Round-1 review fix (Important 2, P5-3): a duty-cycle OFF-phase repost can be scheduled
        // up to ~8s out (MicDutyCycle's cycleMs - onMs) — without the cancel-and-repost below, an
        // Off tap landing during that sleep would queue behind it and the actual STOP (mic
        // release, notification teardown, stopSelf) would defer up to that whole ~8s, even though
        // this block's own disarm()/publishAndNotify/cancelPulseRepeat calls (none of which touch
        // the mic) still run immediately. removeCallbacks + an undelayed re-post collapses that
        // back down to the same ~1-2 cycle bound as every other disarm: it cancels whatever's
        // pending (a duty-cycle sleep or an ordinary ~1s repost alike) and replaces it with a
        // fresh immediate tick that observes the just-published DISARMED state and converges
        // through tickRunnable's own DISARMED branch on its very next run.
        handler?.post {
            controller?.let { c ->
                c.disarm()
                runCatching { Telemetry.log("info", TAG, "disarmed") }
                publishAndNotify(c.state)
                // M1: an explicit Off tap must stop the pulse train immediately, same as
                // stopSentinel()'s own cancelPulseRepeat() — without this, a pulse chain started
                // just before the tap keeps firing on pulseHandler's separate main-looper thread
                // until whatever else eventually calls cancelPulseRepeat() (or nothing does, if
                // loopRunning was already false and stopSentinel() below is skipped).
                cancelPulseRepeat()
                if (loopRunning) {
                    // Cancel whatever's pending (an up-to-8s duty-cycle sleep, or an ordinary ~1s
                    // repost) and replace it with an immediate tick — see this method's own KDoc
                    // above. tickRunnable's `lastPublishedSentinelState != DISARMED` guard
                    // (Important 1) is what keeps THIS immediate re-run from physically resuming
                    // the mic before converging to stopSentinel().
                    handler?.removeCallbacks(tickRunnable)
                    handler?.post(tickRunnable)
                } else {
                    // If the loop was never running (e.g. DISARM with no prior ARM), there's
                    // nothing to converge through tickRunnable's own DISARMED branch — stop here
                    // instead.
                    stopSentinel()
                }
            }
        }
    }

    private fun onActionSetMode(intent: Intent) {
        val mode = modeExtra(intent) ?: return
        handler?.post {
            controller?.let { c ->
                c.setMode(mode)
                publishAndNotify(c.state)
            }
        }
    }

    private fun modeExtra(intent: Intent): Mode? =
        intent.getStringExtra(EXTRA_MODE)?.let { runCatching { Mode.valueOf(it) }.getOrNull() }

    /** Snapshots the last 2 minutes of the wearer's own audio and uploads it (Wave C) — off the
     * main thread (this whole method runs via [handler].post, the service's existing single
     * HandlerThread), fire-and-forget from the caller's perspective: the result reaches the UI
     * via [app.gauge.wear.capture.RetroCaptureBus], never a blocking return through the Intent
     * that triggered it.
     *
     * [EXTRA_CONSENT_CONFIRMED] (post-Task-11 correction — see that constant's own KDoc): the
     * wearer's actual dialog confirmation, threaded from Task 13's "Confirm & save" tap through
     * this Intent and on into [app.gauge.wear.capture.RetroCaptureUploader.upload]'s own
     * required, structurally-enforced `consentConfirmed` parameter — the ONE place this is ever
     * enforced. Defaults to `false` (fail-closed) if the extra is somehow missing; this method
     * never assumes consent just because the action fired, and never hardcodes `true`. */
    private fun onActionRetroCapture(intent: Intent?) {
        val consentConfirmed = intent?.getBooleanExtra(EXTRA_CONSENT_CONFIRMED, false) ?: false
        val c = controller ?: run {
            app.gauge.wear.capture.RetroCaptureBus.publish(app.gauge.wear.capture.RetroCaptureResult.FAILED)
            return
        }
        handler?.post {
            val pcm = runCatching { c.captureRetroSnapshot(RETRO_CAPTURE_DEFAULT_SECONDS) }.getOrNull()
            if (pcm == null) {
                app.gauge.wear.capture.RetroCaptureBus.publish(app.gauge.wear.capture.RetroCaptureResult.FAILED)
                return@post
            }
            val deviceId = app.gauge.wear.prefs.GaugePrefs.deviceId(applicationContext)
            val token = app.gauge.wear.auth.AccountPrefs.deviceToken(applicationContext)
            if (token == null) {
                app.gauge.wear.capture.RetroCaptureBus.publish(app.gauge.wear.capture.RetroCaptureResult.FAILED)
                return@post
            }
            val api = app.gauge.wear.net.WatchApiClient(baseUrl = BuildConfig.GAUGE_API_BASE, deviceToken = { token })
            val uploader = app.gauge.wear.capture.RetroCaptureUploader(
                api = api,
                nowIso = { java.time.Instant.now().toString() },
                deviceId = deviceId,
            )
            // Night 2 hardening (T12 review minor): launched on the service-owned retroCaptureScope
            // (see its own KDoc) rather than a fresh, unscoped CoroutineScope per call — cancelled
            // by onDestroy if the service is torn down mid-upload, instead of leaking past the
            // service's own lifetime. A null scope here means onDestroy already ran (or onCreate
            // somehow never did) — same fail-closed FAILED publish as every other guard above.
            val scope = retroCaptureScope ?: run {
                app.gauge.wear.capture.RetroCaptureBus.publish(app.gauge.wear.capture.RetroCaptureResult.FAILED)
                return@post
            }
            scope.launch {
                val ok = runCatching {
                    uploader.upload(
                        pcm,
                        durationS = pcm.size / (16000.0 * 2),
                        trigger = "manual",
                        consentConfirmed = consentConfirmed,
                    )
                }.getOrDefault(false)
                app.gauge.wear.capture.RetroCaptureBus.publish(
                    if (ok) app.gauge.wear.capture.RetroCaptureResult.SAVED else app.gauge.wear.capture.RetroCaptureResult.FAILED,
                )
            }
        }
    }

    // --- loop-thread-only ----------------------------------------------------------------------

    private fun armOnLoopThread(mode: Mode?) {
        val c = controller ?: return
        if (loopRunning) {
            // Already armed/streaming: apply a mode change if any, otherwise no-op (arm() itself
            // is a no-op once past DISARMED).
            if (mode != null) c.setMode(mode)
            c.arm()
            publishAndNotify(c.state)
            return
        }
        val mic = micReader ?: return
        try {
            mic.start()
        } catch (e: Exception) {
            // Mic unavailable (permission not granted, hardware busy, ...): nothing to arm.
            runCatching {
                Telemetry.log("error", TAG, "mic start failed; stopping sentinel: ${e.message}")
            }
            stopSentinel()
            return
        }
        // P5-3: HR registration is no longer unconditional here — demand (selected signal, or
        // streaming with an HR-subscribed vector) decides via manageHrRegistration, called from
        // publishAndNotify below on this same tick. accelSource stays unconditional: it's cheap
        // and always the movement signal's source.
        accelSource?.start()

        if (mode != null) c.setMode(mode)
        c.arm()
        loopRunning = true
        runCatching { Telemetry.log("info", TAG, "armed mode=${c.state.mode}") }
        publishAndNotify(c.state)
        handler?.post(tickRunnable)
    }

    /** Idempotent: releases the mic, drops the foreground notification, and stops the service. */
    private fun stopSentinel() {
        loopRunning = false
        // Wave-final review, Important 2: force a disarmed snapshot through the SAME publish path
        // (ControllerStateBus + pushFaceUpdates) as every other stop, BEFORE the push-on-change
        // caches below get cleared. Without this, a path that reaches stopSentinel() without ever
        // having published a disarmed snapshot for THIS transition — chiefly tickRunnable's catch
        // block, which stops the sentinel after `c.tick()` (or the state-read/publish/notify that
        // follows it) throws, with no publish having happened for that tick at all — leaves
        // ControllerStateBus (and therefore the complication, which reads it on its own 600s poll
        // plus this push) retaining the last STREAMING snapshot forever: a red "Level 3" arc with
        // no episode actually running.
        //
        // controller.disarm() is idempotent (no-ops once already DISARMED — see its own guard), so
        // this is a harmless repeat on every path that already published a disarmed snapshot itself
        // (onActionDisarm, tickRunnable's own DISARMED branch): re-reading `c.state` returns the
        // identical values (channelLevels/meter/activePulse already cleared by that first disarm()
        // — see SentinelController's own Important-1 fix), so ControllerStateBus's StateFlow
        // equality check skips the re-emission and pushFaceUpdates' own lastPushedComplication/
        // lastPushedTileStrings caches (still populated at this point, cleared further below) skip
        // the requester calls too — a genuine no-op, not just a harmless one.
        //
        // Wrapped in runCatching end to end (both the disarm() call and the publish) — this method
        // must stay fail-soft on every caller, very much including the tick-catch path, which is
        // already mid-recovery from one exception and must not throw a second.
        runCatching {
            controller?.let { c ->
                c.disarm()
                publishAndNotify(c.state)
            }
        }
        micReader?.release()
        hrSource?.stop()
        hrRegistered = false
        hrPermissionWarnedThisDemand = false
        hrDemandPolicy.reset()
        // P5-3: reset the duty-cycle schedule so a re-arm always starts continuous rather than
        // resuming mid-cycle (or mid-quiet-run) from this now-ended session.
        micDutyCycle.reset()
        lastCaptureMode = CaptureMode.CONTINUOUS
        accelSource?.stop()
        // P4-4 (review round 2): clear the push-on-change caches so a subsequent re-arm always
        // pushes its first projection fresh, rather than comparing against a stale value left over
        // from this session (a re-arm could otherwise land on the exact same complication/tile
        // projection the previous session ended on and wrongly skip the push).
        lastPushedComplication = null
        lastPushedTileStrings = null
        // P4-3: no episode running past this point means no pulse train either — Handler.
        // removeCallbacks is safe to call from any thread (this runs on handlerThread, not
        // pulseHandler's own main-looper thread).
        cancelPulseRepeat()
        // Best-effort: flush whatever's left in the ring on every stop path (explicit disarm,
        // mic loss, tick error) so nothing sits un-sent past the service's own lifetime. Never
        // allowed to affect sentinel shutdown itself.
        runCatching { Telemetry.flushAsync() }
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    // --- P4-3: pulse-train repeat loop (round 2: race/orphan fix) -------------------------------

    /**
     * Reconciles the pulse-repeat loop against this tick's [ControllerState.activePulse] verdict
     * via the pure [PulseChainDecision.decide] — see its own KDoc for why minimizing restarts to
     * only the ticks where the verdict actually changed matters (not just [PulseChainGate]'s token
     * check) — then acts on the result: [PulseChainDecision.Start] (re)starts the physical chain,
     * [PulseChainDecision.Stop] tears it down, [PulseChainDecision.Continue]/[PulseChainDecision.
     * NoOp] leave it alone. Called every tick, right after [publishAndNotify].
     */
    private fun managePulseRepeat(snapshot: ControllerState) {
        val decision = PulseChainDecision.decide(
            currentChain = currentPulseChain,
            pulse = snapshot.activePulse,
            streaming = snapshot.sentinel == SentinelState.STREAMING,
            configuredIntervalMs = pulseIntervalMsPref(),
        )
        when (decision) {
            is PulseChainDecision.NoOp, PulseChainDecision.Continue -> Unit
            PulseChainDecision.Stop -> cancelPulseRepeat()
            is PulseChainDecision.Start -> {
                currentPulseChain = decision.pulse to decision.effectiveIntervalMs
                startPulseChain(decision.pulse, decision.effectiveIntervalMs)
            }
        }
    }

    /**
     * Starts a fresh, [pulseChainGate]-guarded repeating [Runnable] chain on [pulseHandler] — a
     * Handler bound to the main looper, deliberately separate from [handler]/[handlerThread] (see
     * their own field KDoc) — that plays [pulse] via [haptics] every [intervalMs], starting
     * immediately (posted with no delay: this IS the "on an over-threshold window, emit the due
     * pulse immediately" behavior now — see [SentinelController]'s own KDoc for why the controller
     * itself no longer fires this first pulse synchronously).
     *
     * Every `Runnable` captures its own [PulseChainGate] token from [pulseChainGate.start] and
     * checks [PulseChainGate.isCurrent] immediately before EVERY side effect — both the vibrate
     * call and the re-post — so a `Runnable` this method (or [cancelPulseRepeat]) failed to reach
     * via `removeCallbacks` (a genuine race against the looper already having dequeued it — see
     * [PulseChainGate]'s own KDoc for the full incident) still self-neuters instead of firing a
     * stray pulse or, worse, becoming an orphaned loop nothing can cancel. The `removeCallbacks`
     * call below remains as best-effort housekeeping (avoids leaving an inert message sitting in
     * the queue for one extra no-op cycle) — correctness no longer depends on it succeeding.
     */
    private fun startPulseChain(pulse: Pulse, intervalMs: Long) {
        // M2 (teardown guard): never arm a fresh repeat chain once onDestroy has started tearing
        // things down — see `destroyed`'s own KDoc for the race this closes (a managePulseRepeat
        // call from handlerThread landing mid-teardown, after pulseHandler/haptics are already
        // being nulled out but before this specific check would otherwise have caught it via the
        // `?: return` below alone).
        if (destroyed) return
        val hdl = pulseHandler ?: return
        pulseRunnable?.let { runCatching { hdl.removeCallbacks(it) } }
        val token = pulseChainGate.start()
        val runnable = object : Runnable {
            override fun run() {
                if (!pulseChainGate.isCurrent(token)) return
                runCatching { haptics?.playPulse(pulse) }
                if (!pulseChainGate.isCurrent(token)) return
                runCatching { hdl.postDelayed(this, intervalMs) }
            }
        }
        pulseRunnable = runnable
        runCatching { hdl.post(runnable) }
    }

    /** Idempotent: invalidates [pulseChainGate]'s current token (dead regardless of whether
     * `removeCallbacks` below actually intercepts the queued `Runnable` — see [startPulseChain]'s
     * KDoc), best-effort-removes it from [pulseHandler]'s queue, and clears [currentPulseChain] so
     * the next [managePulseRepeat] call starts a fresh chain rather than seeing a stale match. */
    private fun cancelPulseRepeat() {
        pulseChainGate.cancel()
        val hdl = pulseHandler
        val runnable = pulseRunnable
        if (hdl != null && runnable != null) {
            runCatching { hdl.removeCallbacks(runnable) }
        }
        pulseRunnable = null
        currentPulseChain = null
    }

    /** Parses [GaugePrefs.pulseIntervalMs] ("250"/"500"/"1000"/"off") into the `Long?` shape
     * [SentinelController]/[app.gauge.wear.haptics.PulseEngine] expect — `null` for "off",
     * falling back to the 250ms default for any unrecognized value rather than crashing on a
     * corrupt pref. */
    private fun pulseIntervalMsPref(): Long? {
        val raw = GaugePrefs.pulseIntervalMs(applicationContext)
        if (raw == "off") return null
        return raw.toLongOrNull() ?: SentinelController.DEFAULT_PULSE_INTERVAL_MS
    }

    /** Parses [GaugePrefs.selectedSignal], falling back to [SignalKind.VOLUME] for an absent or
     * corrupt pref. Single parser for the service: both the controller's own `selectedSignal` seam
     * and [manageHrRegistration]'s demand check call this rather than each re-implementing it. */
    private fun selectedSignalPref(): SignalKind =
        runCatching { SignalKind.valueOf(GaugePrefs.selectedSignal(applicationContext)) }
            .getOrDefault(SignalKind.VOLUME)

    // --- notification ----------------------------------------------------------------------------

    private fun publishAndNotify(snapshot: ControllerState) {
        // Episode-end flush: STREAMING -> COOLDOWN is the moment an episode's stream just ended,
        // independent of whether the sentinel itself keeps running afterward (COOLDOWN still
        // leads back to ARMED, not necessarily a stopSentinel() call) — so this can't be covered
        // by the stopSentinel() flush alone.
        if (lastPublishedSentinelState == SentinelState.STREAMING &&
            snapshot.sentinel == SentinelState.COOLDOWN
        ) {
            runCatching { Telemetry.flushAsync() }
        }
        lastPublishedSentinelState = snapshot.sentinel
        ControllerStateBus.publish(snapshot)
        if (loopRunning) updateNotification(snapshot)
        pushFaceUpdates(snapshot)
        manageHrRegistration(snapshot)
    }

    /**
     * P5-3 (C1): applies [HrDemandPolicy] to this tick's snapshot. Called from [publishAndNotify]'s
     * caller path (once per ~1s tick, and right after arm/disarm), so a signal-picker change or an
     * episode start/end takes effect within one window without any extra scheduling of its own.
     * Handler-thread-only, same contract as [loopRunning].
     *
     * BODY_SENSORS is checked at the REGISTER edge, never requested here (the UI's job — see
     * [app.gauge.wear.ui.SignalScreen]); a missing grant degrades honestly to "no HR reading",
     * exactly as before. Every branch is runCatching-wrapped: HR lifecycle management must never be
     * the thing that takes down the sentinel.
     */
    private fun manageHrRegistration(snapshot: ControllerState) {
        val hr = hrSource ?: return
        val action = runCatching {
            hrDemandPolicy.evaluate(
                selectedSignal = selectedSignalPref(),
                sentinel = snapshot.sentinel,
                hrVectorSubscribed = HR_VECTOR_SUBSCRIBED,
                registered = hrRegistered,
                nowMs = System.currentTimeMillis(),
            )
        }.getOrDefault(HrAction.NONE)
        when (action) {
            HrAction.NONE -> hrPermissionWarnedThisDemand = false
            HrAction.REGISTER -> {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.BODY_SENSORS) !=
                    PackageManager.PERMISSION_GRANTED
                ) {
                    if (!hrPermissionWarnedThisDemand) {
                        runCatching { Telemetry.log("info", TAG, "hr skipped: BODY_SENSORS not granted") }
                        hrPermissionWarnedThisDemand = true
                    }
                    return
                }
                runCatching { hr.start() }
                hrRegistered = true
                hrPermissionWarnedThisDemand = false
                runCatching { Telemetry.log("info", TAG, "hr registered (demand)") }
            }
            HrAction.UNREGISTER -> {
                runCatching { hr.stop() }
                hrRegistered = false
                hrPermissionWarnedThisDemand = false
                runCatching { Telemetry.log("info", TAG, "hr unregistered (idle grace elapsed)") }
            }
        }
    }

    /**
     * P4-4: the sanctioned push path [ArmedComplicationService]'s own KDoc has documented as
     * future work since it landed — every [ControllerStateBus] republish now also nudges the
     * escalation-gauge complication and the swipe-right tile to refresh immediately, instead of
     * the complication's 600s poll / the tile's own [app.gauge.wear.tile.GaugeTileService]
     * 1.5s-delayed click-only refresh (see that class's KDoc) being the only way either ever
     * updates.
     *
     * PUSH ON CHANGE ONLY (review fix, round 2): `tick()` — and therefore `publishAndNotify` —
     * runs on a ~1s cadence for the entire armed session (ARMED/COOLDOWN/STREAMING alike, not just
     * episodes), so pushing unconditionally would fire 3600+ `requestUpdateAll`/`requestUpdate`
     * calls per armed hour against two rate-limited Wear APIs whose documented usage is
     * push-on-data-change — worst case, the system throttles/penalizes this app as a source and
     * the face freezes, exactly the failure this feature exists to avoid. [complicationValue] and
     * [tileStrings] are both pure functions of `snapshot` alone (see ComplicationContentTest /
     * TileLayoutTest), so comparing their output against [lastPushedComplication]/
     * [lastPushedTileStrings] and skipping the requester call when unchanged is provably correct,
     * not just "less often": an unchanged projection means an unchanged face, full stop. Each
     * requester call remains independently `runCatching`-wrapped (telemetry-never-breaks-sentinel
     * discipline, same posture as every other best-effort call in this class) — a
     * `ComplicationDataSourceUpdateRequester`/`TileService.getUpdater` failure (e.g. no
     * complication/tile currently added to a face) must never take down the trigger/publish path
     * it's piggybacking on, and a failed push leaves the cache un-updated so the next real change
     * still retries it.
     */
    private fun pushFaceUpdates(snapshot: ControllerState) {
        val complication = complicationValue(snapshot)
        if (complication != lastPushedComplication) {
            runCatching {
                ComplicationDataSourceUpdateRequester.create(
                    applicationContext,
                    ComponentName(applicationContext, ArmedComplicationService::class.java),
                ).requestUpdateAll()
            }.onSuccess {
                lastPushedComplication = complication
            }.onFailure { t ->
                runCatching { Telemetry.log("error", TAG, "complication push failed: ${t.message}") }
            }
        }

        val strings = tileStrings(armed = snapshot.sentinel != SentinelState.DISARMED, mode = snapshot.mode)
        if (strings != lastPushedTileStrings) {
            runCatching {
                TileService.getUpdater(applicationContext).requestUpdate(GaugeTileService::class.java)
            }.onSuccess {
                lastPushedTileStrings = strings
            }.onFailure { t ->
                runCatching { Telemetry.log("error", TAG, "tile push failed: ${t.message}") }
            }
        }
    }

    private fun startForegroundNotification(initial: ControllerState) {
        startForeground(
            NOTIFICATION_ID,
            buildNotification(notificationText(initial)),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
        )
    }

    private fun updateNotification(snapshot: ControllerState) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIFICATION_ID, buildNotification(notificationText(snapshot)))
    }

    private fun buildNotification(text: String): Notification {
        ensureChannel()
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.app_name))
            .setContentText(text)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()
    }

    private fun ensureChannel() {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Gauge Sentinel", NotificationManager.IMPORTANCE_LOW),
            )
        }
    }

    companion object {
        const val ACTION_ARM = "app.gauge.wear.action.ARM"
        const val ACTION_DISARM = "app.gauge.wear.action.DISARM"
        const val ACTION_SET_MODE = "app.gauge.wear.action.SET_MODE"
        const val ACTION_RETRO_CAPTURE = "app.gauge.wear.ACTION_RETRO_CAPTURE"
        const val EXTRA_MODE = "mode"
        /** Post-Task-11 correction (plan update): the ACTUAL wearer confirmation from the consent
         * dialog (Task 13), never assumed. [onActionRetroCapture] reads this fail-closed
         * (`getBooleanExtra(..., false)`) on a missing/malformed extra and threads it straight
         * into [app.gauge.wear.capture.RetroCaptureUploader.upload]'s own required
         * `consentConfirmed` parameter — that's the ONE place this ever gets enforced (never
         * hardcoded `true` here), so the fail-closed guarantee stays structural end to end rather
         * than relocating T11's "trust an intermediate layer" fix to a new boundary a future
         * Tile/complication/debug entry point could bypass. Task 13 must set this `true` ONLY
         * from the dialog's own affirmative "Confirm & save" tap — dismiss/back must never send
         * this intent with this extra `true` (or, more simply, must never send this intent at
         * all). */
        const val EXTRA_CONSENT_CONFIRMED = "app.gauge.wear.EXTRA_CONSENT_CONFIRMED"

        private const val TAG = "SentinelService"
        private const val CHANNEL_ID = "sentinel"
        private const val NOTIFICATION_ID = 1

        // Keeps a pathologically deep stack trace from ballooning a single telemetry ring event.
        private const val MAX_TICK_STACK_CHARS = 4000

        // P5-3: floor for the duty-cycled repost delay — a msUntilNextCapture() of 0 in a mode
        // race (e.g. the OFF-phase boundary lands exactly on `now`) must never spin the handler
        // with a zero-delay repost loop.
        private const val MIN_DUTY_SLEEP_MS = 250L

        private const val RETRO_CAPTURE_DEFAULT_SECONDS = 120.0 // "the last 2 minutes" — product default

        // P5-3 (ratified): every current backend nudging vector subscribes hr_spike, so this is
        // `true` unconditionally today. Seam for when per-account/per-mode vector subscriptions
        // land (Plan 2b+) — at that point this becomes a real lookup instead of a literal, and
        // HrDemandPolicy's `hrVectorSubscribed` parameter needs no changes at all to consume it.
        private const val HR_VECTOR_SUBSCRIBED = true

        private fun wsBaseUrlFromApiBase(apiBase: String): String =
            apiBase.replaceFirst("https://", "wss://").replaceFirst("http://", "ws://")
    }
}
