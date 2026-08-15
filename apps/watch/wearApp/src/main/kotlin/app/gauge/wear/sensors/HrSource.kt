package app.gauge.wear.sensors

import android.content.Context
import androidx.health.services.client.HealthServices
import androidx.health.services.client.MeasureCallback
import androidx.health.services.client.data.Availability
import androidx.health.services.client.data.DataPointContainer
import androidx.health.services.client.data.DataType
import androidx.health.services.client.data.DeltaDataType
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.signalAvailabilityFrom
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.ScalarSource
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import com.google.common.util.concurrent.MoreExecutors

/**
 * [ScalarSource] backed by Wear Health Services' live heart-rate measurement stream (Task 10).
 *
 * Android shell, compile+lint gated (no unit test — no emulator/device in this repo's CI loop,
 * per CLAUDE.md). Only ever constructed/started by [app.gauge.wear.service.SentinelService] after
 * it has confirmed BODY_SENSORS is granted (this class never requests permissions itself — that's
 * the UI's job, Task 11). [app.gauge.wear.ui.MainActivity] also constructs its own instance for the
 * disarmed-state live preview meter (P4-6); both call sites share this class but supply their own
 * [DiagLog] (wired to [app.gauge.wear.telemetry.Telemetry] in production) — this class itself has
 * no hard `Telemetry` dependency, only the [diag] seam.
 *
 * I1: [tag] disambiguates which instance a diag/telemetry entry came from — the service's live
 * instance and MainActivity's preview instance both log under the same literal `"HrSource"` tag
 * before this fix, making a device telemetry pull genuinely ambiguous about which HR lifecycle
 * (armed-service vs. disarmed-preview) a given "measure callback registered"/"first hr reading"
 * line belongs to whenever both have been active in the same session. The service keeps the
 * default; [app.gauge.wear.ui.MainActivity] passes `"HrSourcePreview"`.
 *
 * Honest degradation: [latest] returns `null` until the first [MeasureCallback.onDataReceived]
 * delivers a heart-rate sample — never a fabricated 0.0 placeholder.
 *
 * P4-7: on the real Pixel Watch, HR never delivers a reading even with BODY_SENSORS granted and
 * the sentinel armed, and telemetry showed ZERO HR errors — meaning the pipeline was failing
 * *silently*, not loudly. Root cause of the blind spot: [MeasureCallback.onRegistered] and
 * [MeasureCallback.onRegistrationFailed] are default no-op methods in the health-services-client
 * SDK (registration itself, per [androidx.health.services.client.MeasureClient.
 * registerMeasureCallback]'s own signature, is fire-and-forget — success/failure only ever arrives
 * via those two callbacks), and this class never overrode either one; [onAvailabilityChanged] was
 * also a bare `= Unit`. So a registration that silently failed, or that succeeded but then sat at
 * ACQUIRING/UNAVAILABLE forever, would have produced literally zero telemetry either way. Every
 * lifecycle edge below is now diag-logged (tag "HrSource") so the next wear session pinpoints
 * exactly where delivery stops, the same pattern that root-caused the WebSocket failure in an
 * earlier session. No functional fix is applied here — see this task's own instructions: instrument
 * first, honestly. (Decompiled the health-services-client 1.0.0 AAR to confirm this: the two-arg
 * `registerMeasureCallback(dataType, callback)` overload this class already used internally
 * delegates to `ContextCompat.getMainExecutor(context)`, so there's no missing-executor or
 * wrong-thread bug either — the gap was purely the un-overridden callbacks.)
 *
 * P4-10: live telemetry also showed rapid register/unregister churn — [start]/[stop] pairs landing
 * inside 1s during the preview<->service handoff. [gate] dedupes duplicate calls (skipped with a
 * `debug` diag line instead of hitting the SDK) and counts real transitions inside a sliding
 * window, reported to telemetry as a `warn` line when churn is detected. Fail-soft ordering rule:
 * if the SDK call inside [start] throws, [gate] has already flipped to active — that's intentional
 * (a failed registration must not be retried on every tick), and the next [stop] correctly clears
 * it. Every [gate] call is itself wrapped in `try`/`catch (Throwable)`: a broken gate must degrade
 * to "always attempt the SDK call" (the pre-P4-10 behavior), never to "silently stop calling the
 * SDK at all" — the fail-soft posture applies to the gate itself, not just the sensor SDK calls it
 * guards.
 *
 * P4-10 review round 1 (Critical fix): [androidx.health.services.client.MeasureClient.
 * unregisterMeasureCallbackAsync] is asynchronous — a "did this actually take effect" answer only
 * exists once its [FutureCallback] fires — so [stop] does NOT call [SensorLifecycleGate.onStop]
 * synchronously the way [start] calls [SensorLifecycleGate.onStart]. Calling it eagerly (the
 * original round-0 shape) meant a failed unregister left [gate] reporting "inactive" while the SDK
 * listener was, in reality, still registered: every belt-and-suspenders repeat [stop] call
 * (`SentinelService.stopSentinel`/`onDestroy` both call it) would then be suppressed as a
 * "duplicate", leaking the HR listener registered for the process lifetime — worse than the churn
 * this class exists to fix. Fixed shape: [gate] is only told about the stop from
 * [FutureCallback.onSuccess]; [pendingStop] separately guards against firing a second concurrent
 * unregister while one is already in flight. Traced states:
 * - stop() while idle (gate active, no stop pending) → SDK call issued, [pendingStop.beginStop].
 * - stop() while pending (unregister in flight) → suppressed (`stop ignored: unregister already
 *   in flight`), no second SDK call.
 * - [FutureCallback.onFailure] → [pendingStop.abandon], [gate] stays active (nothing confirmed) →
 *   the *next* stop() call is NOT suppressed and genuinely retries the SDK call.
 * - [FutureCallback.onSuccess] → [pendingStop.confirm], [gate] flips inactive via [onStop] → a
 *   further stop() call now IS suppressed (`stop ignored: not registered`), a real no-op.
 * - [start] called while a stop is pending or has failed-and-not-yet-retried → [gate] still
 *   reports active, so [SensorLifecycleGate.onStart] suppresses it (`start ignored: already
 *   registered`) — correct, since the SDK's own registration is still live in both cases.
 *
 * P4-10 review round 2 (Important fix): round 1 above still had a gap — a [start] arriving while a
 * stop was pending-unconfirmed was suppressed by [gate] with NO record kept anywhere, so the
 * eventual confirmation landed HR off even though the caller's most recent request was "on" (real
 * call sites: `SentinelService`'s fast disarm→rearm around its `arm()`/`stopSentinel()` calls, and
 * `MainActivity`'s preview onPause/onResume edges around `stop()`/`start()`). [pendingStop] (a
 * [PendingStopIntent] — see its own KDoc) now remembers that a start arrived during the pending
 * window and [doStart] is replayed from [FutureCallback.onSuccess] when it did, so the final state
 * always matches the *latest* request rather than whichever one happened to be in flight first.
 * The mirror edge — a second stop() arriving before the first confirms, overriding an
 * already-deferred start — clears the deferred intent (see [PendingStopIntent.stopArrived]).
 * [pendingStop] is placed in this class, not [SensorLifecycleGate]: it is HrSource-specific async
 * orchestration state (the gate stays a generic, SDK-agnostic register/unregister dedupe/churn
 * component with an already-reviewed, locked public contract — this keeps that contract
 * untouched).
 */
class HrSource(
    private val context: Context,
    private val diag: DiagLog,
    private val tag: String = "HrSource",
    private val nowMs: () -> Long = { System.currentTimeMillis() },
) : ScalarSource {
    @Volatile private var latestBpm: Double? = null

    @Volatile private var latestAvailability: SignalAvailability = SignalAvailability.UNKNOWN

    private val gate = SensorLifecycleGate()

    // Guards against firing a second concurrent unregister while one is already in flight, and
    // remembers a start() that arrives during that window so it can be replayed once the stop
    // confirms — see the class KDoc's "P4-10 review round 1"/"round 2" traces and PendingStopIntent's
    // own KDoc. Not part of [gate] itself: this is HrSource-specific async orchestration state,
    // orthogonal to the gate's register/unregister dedupe/churn contract.
    private val pendingStop = PendingStopIntent()

    // Set false at the top of every start() so "first hr reading" logs once per start(), not once
    // ever per HrSource instance (an instance can be started/stopped across multiple arm cycles —
    // see class KDoc's "one unregister per disarm" note on stop()).
    @Volatile private var loggedFirstReading = false

    private val callback = object : MeasureCallback {
        override fun onRegistered() {
            diag.log("info", tag, "measure callback registered")
        }

        override fun onRegistrationFailed(throwable: Throwable) {
            diag.log(
                "error",
                tag,
                "measure callback registration failed: ${throwable::class.simpleName}: ${throwable.message}",
            )
        }

        override fun onAvailabilityChanged(dataType: DeltaDataType<*, *>, availability: Availability) {
            // PRIME SUSPECT (see class KDoc): Health Services reports ACQUIRING / AVAILABLE /
            // UNAVAILABLE / UNAVAILABLE_DEVICE_OFF_BODY / UNKNOWN transitions here. A stream stuck
            // at ACQUIRING forever, or one that jumps straight to UNAVAILABLE, is exactly what
            // "armed, permission granted, but never a reading" looks like from this callback's
            // point of view — and until this task, nothing observed it at all.
            diag.log("info", tag, "availability changed: $availability")
            try {
                val mapped = signalAvailabilityFrom(availability.toString())
                latestAvailability = mapped
                // P4-10 review round 1 (Critical fix): a registered-but-now-off-body/unavailable
                // stream must not go on reporting the LAST sample onDataReceived ever delivered —
                // onDataReceived only ever sets latestBpm, never clears it, so without this a
                // wearer who removed the watch mid-episode would see a frozen, honest-looking-but-
                // stale bpm number sitting right next to an honest "off-body" caption. Cleared
                // whenever the new availability isn't AVAILABLE; the next onDataReceived after a
                // real re-acquire naturally repopulates it.
                if (mapped != SignalAvailability.AVAILABLE) {
                    latestBpm = null
                }
            } catch (t: Throwable) {
                diag.log("error", tag, "availability mapping failed: $t")
            }
        }

        override fun onDataReceived(data: DataPointContainer) {
            try {
                val sample = data.getData(DataType.HEART_RATE_BPM).lastOrNull() ?: return
                latestBpm = sample.value
                if (!loggedFirstReading) {
                    loggedFirstReading = true
                    diag.log("info", tag, "first hr reading: ${sample.value}")
                }
            } catch (t: Throwable) {
                diag.log("error", tag, "onDataReceived failed: $t")
            }
        }
    }

    override fun latest(): Double? = latestBpm

    override fun availability(): SignalAvailability = latestAvailability

    /** Registers the live HR measure callback. Gate-guarded (P4-10): a duplicate call is skipped
     * with a `debug` diag line instead of hitting the SDK — matching [stop]'s one unregister per
     * disarm. If a stop is currently pending SDK confirmation, the request is deferred instead
     * (see [PendingStopIntent] / the class KDoc's "review round 2" trace) rather than attempting a
     * register while an unregister might still land. */
    fun start() {
        val deferred = try {
            pendingStop.startArrived()
        } catch (t: Throwable) {
            diag.log("error", tag, "pendingStop startArrived failed: $t")
            false // fail open: proceed with the normal registration path immediately rather than
            // silently drop the request because the bookkeeping itself broke.
        }
        if (deferred) {
            diag.log("debug", tag, "start deferred: unregister in flight, will re-arm once confirmed")
            return
        }
        doStart()
    }

    /** The actual register attempt — shared by [start] and the "replay a deferred start once its
     * blocking stop confirms" path in [stop]'s [FutureCallback.onSuccess]. Every failure mode
     * inside is already caught and diag-logged, so calling this a second time (the replay) can
     * never crash the callback thread it runs on. */
    private fun doStart() {
        val shouldRegister = try {
            gate.onStart(nowMs())
        } catch (t: Throwable) {
            diag.log("error", tag, "gate onStart failed: $t")
            true // fail open: without a working gate we can't tell duplicate from fresh — behave
            // as pre-P4-10 (always attempt), the safer default for a signal source.
        }
        if (!shouldRegister) {
            diag.log("debug", tag, "start ignored: already registered")
            return
        }
        loggedFirstReading = false
        latestAvailability = SignalAvailability.UNKNOWN
        try {
            diag.log("info", tag, "registering measure callback")
            val measureClient = HealthServices.getClient(context).measureClient
            measureClient.registerMeasureCallback(DataType.HEART_RATE_BPM, callback)
        } catch (t: Throwable) {
            diag.log("error", tag, "start failed: ${t::class.simpleName}: ${t.message}")
        }
        try {
            // Deliberately unconditional on the register attempt above having succeeded — see
            // SensorLifecycleGate.churnDetected's own KDoc: a registration that keeps failing on
            // every retry is still churn.
            if (gate.churnDetected(nowMs())) {
                diag.log("warn", tag, "registration churn: ${gate.transitionCount(nowMs())} transitions in 5s")
            }
        } catch (t: Throwable) {
            diag.log("error", tag, "gate churnDetected failed: $t")
        }
    }

    /** Unregisters the live HR measure callback. See the class KDoc's "P4-10 review round 1"/
     * "round 2" traces for the full state machine: [gate] is only told about the stop once the
     * SDK's async unregister actually confirms via [FutureCallback.onSuccess]; [pendingStop]
     * suppresses a second concurrent unregister while one is already in flight (without blocking a
     * genuine retry after [FutureCallback.onFailure]) and remembers/replays a [start] that arrived
     * during the pending window. */
    fun stop() {
        val stillRegistered = try {
            gate.isActive
        } catch (t: Throwable) {
            diag.log("error", tag, "gate isActive check failed: $t")
            true // fail open: never silently skip an unregister because the gate itself broke.
        }
        if (!stillRegistered) {
            diag.log("debug", tag, "stop ignored: not registered")
            return
        }
        val alreadyPending = try {
            pendingStop.stopArrived()
        } catch (t: Throwable) {
            diag.log("error", tag, "pendingStop stopArrived failed: $t")
            false // fail open: attempt the unregister rather than silently skip it.
        }
        if (alreadyPending) {
            diag.log("debug", tag, "stop ignored: unregister already in flight")
            return
        }
        try {
            pendingStop.beginStop()
        } catch (t: Throwable) {
            diag.log("error", tag, "pendingStop beginStop failed: $t")
        }
        try {
            diag.log("info", tag, "unregistering measure callback")
            val measureClient = HealthServices.getClient(context).measureClient
            val future = measureClient.unregisterMeasureCallbackAsync(DataType.HEART_RATE_BPM, callback)
            Futures.addCallback(
                future,
                object : FutureCallback<Void?> {
                    override fun onSuccess(result: Void?) {
                        val replayStart = try {
                            pendingStop.confirm()
                        } catch (t: Throwable) {
                            diag.log("error", tag, "pendingStop confirm failed: $t")
                            false
                        }
                        try {
                            gate.onStop(nowMs())
                        } catch (t: Throwable) {
                            diag.log("error", tag, "gate onStop failed: $t")
                        }
                        diag.log("info", tag, "unregister completed")
                        if (replayStart) {
                            diag.log("info", tag, "re-arming: start() was requested while unregister was pending")
                            doStart()
                        }
                    }

                    override fun onFailure(unregisterError: Throwable) {
                        // Deliberately does NOT call gate.onStop(): the unregister did not take
                        // effect, so the gate must keep reporting active — the next stop() call
                        // (SentinelService's belt-and-suspenders repeat stops) genuinely retries
                        // instead of being suppressed as a duplicate. abandon() also discards any
                        // deferred start intent — moot, since the source never actually went off.
                        try {
                            pendingStop.abandon()
                        } catch (t: Throwable) {
                            diag.log("error", tag, "pendingStop abandon failed: $t")
                        }
                        diag.log(
                            "error",
                            tag,
                            "unregister failed: ${unregisterError::class.simpleName}: ${unregisterError.message}",
                        )
                    }
                },
                MoreExecutors.directExecutor(),
            )
        } catch (t: Throwable) {
            // Synchronous throw from the SDK call itself: same as onFailure above — leave the
            // gate active so a subsequent stop() retries, and discard any deferred start intent.
            try {
                pendingStop.abandon()
            } catch (t2: Throwable) {
                diag.log("error", tag, "pendingStop abandon failed: $t2")
            }
            diag.log("error", tag, "stop failed: $t")
        }
        latestBpm = null
        latestAvailability = SignalAvailability.UNKNOWN
    }
}
