package app.gauge.wear.ui

import app.gauge.shared.sentinel.SentinelDetector
import app.gauge.shared.signals.Cadence
import app.gauge.shared.signals.HrTracker
import app.gauge.shared.signals.MovementTracker
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.shared.signals.SignalReading
import app.gauge.shared.signals.SpeakingRateTracker
import app.gauge.wear.control.MeterReading
import app.gauge.wear.control.MicSource
import app.gauge.wear.control.ScalarSource

/**
 * Live meter while the sentinel is OFF (Task 12): reads whichever [SignalKind] the wearer
 * currently has selected and publishes a [MeterReading] for it, purely so the main screen's
 * green/red meter is alive even before the wearer arms — "were you the calmer one?" starts the
 * instant the app is on screen, not just mid-episode.
 *
 * Never-two-readers rule (non-negotiable, locked in the Phase 3 plan): [SentinelController] and
 * this engine must never both hold the mic open at once. [step] checks [isServiceActive] FIRST,
 * before touching [mic] or either [ScalarSource].
 *
 * P4-8: the acquire/release contract used to be signal-agnostic (`onYield`/`onResume`, firing on
 * every active/inactive transition regardless of which signal was selected) — this engine held
 * the mic (and only the mic; HR/accel were separately wired directly in [MainActivity]) even while
 * HEART_RATE or MOVEMENT was selected, and switching the picker never released anything. Now
 * exactly ONE signal's source is ever held: [acquiredFor] tracks which [SignalKind] the caller
 * currently has acquired on this engine's behalf, and an acquisition edge is (kind changed) OR
 * (service active/inactive changed) — [onAcquire]/[onRelease] fire only on those edges, never on
 * every step while steady state holds. The very first `step()` call on an inactive service is
 * itself an acquire edge ([acquiredFor] starts `null`), so "initial acquire" and "reacquire after
 * disarm" are literally the same code path — no separate pre-acquire step for the caller to get
 * wrong. This is the battery point of P4-8: the preview costs exactly one sensor at a time, not
 * three (mic + accel + HR all idle-polling regardless of what's on screen).
 *
 * The never-two-readers rule is unchanged and still checked FIRST, before any source is touched:
 * an active service releases whatever this engine currently holds and acquires nothing while it
 * stays active, mirroring the old `onYield`/`onResume` edges but per-signal now.
 *
 * [MainActivity] is expected to poll this on a background loop only while the activity is STARTED
 * and mic permission is granted (see its own wiring) — [isServiceActive] alone is not a substitute
 * for that outer gating, since AudioRecord access itself needs the permission check. Because
 * `step()` is only ever driven by that single-threaded loop, an edge is detected and its callback
 * invoked entirely between two `step()` calls — never concurrently with a [mic] read in progress on
 * this same object (see [safeReadWindow]'s KDoc for the separate, genuinely cross-thread race this
 * does NOT cover: [MainActivity]'s `onStop()` releasing the mic from the *main* thread while a
 * `step()` is still in flight on the polling loop's thread).
 *
 * Own tracker instances, deliberately independent of [app.gauge.wear.control.SentinelController]'s
 * (which only run while armed): this engine needs a baseline/trigger picture of its own for the
 * OFF window, and must not perturb (or be perturbed by) the service's episode-scoped state.
 *
 * Honest degradation (mirrors [MeterReading]'s own KDoc): [publish] only ever receives a real
 * observed reading or `null` — never a fabricated placeholder — whenever the service is active,
 * the selected signal's source is missing, or that source has no reading yet.
 *
 * P4-10 in the preview: [publishAvailability] mirrors [MeterBus.previewAvailability][
 * app.gauge.wear.control.MeterBus.previewAvailability] — only ever non-[SignalAvailability.UNKNOWN]
 * while HEART_RATE is the selected signal (publishing e.g. [SignalAvailability.OFF_BODY] while
 * VOLUME is selected would put a false caption under a mic meter). Also mirrors
 * [app.gauge.wear.control.SentinelController.updateMeter]'s structural honesty guarantee: a
 * HEART_RATE availability that has something honest to SAY (ACQUIRING/OFF_BODY/UNAVAILABLE — i.e.
 * [SignalAvailability.statusText] != null and it isn't "yes, available") suppresses the reading
 * itself, never letting a stale/fabricated bpm sit next to an honest "off-body" caption.
 * [SignalAvailability.UNKNOWN] is deliberately NOT gated, for the same reason as the controller's
 * own version: a [ScalarSource] with no real availability API must keep behaving exactly as before
 * this existed.
 */
class PreviewMeterEngine(
    private val mic: MicSource,
    private val hr: ScalarSource?,
    private val accel: ScalarSource?,
    private val selectedSignal: () -> SignalKind,
    private val publish: (MeterReading?) -> Unit,
    private val isServiceActive: () -> Boolean,
    private val onAcquire: (SignalKind) -> Unit = {},
    private val onRelease: (SignalKind) -> Unit = {},
    private val publishAvailability: (SignalAvailability) -> Unit = {},
    private val publishSparkline: (List<Double>) -> Unit = {},
) {
    private val detector = SentinelDetector(TRIGGER_DB_OVER_BASELINE)
    private val hrTracker = HrTracker()
    private val movementTracker = MovementTracker()
    private val speakingRateTracker = SpeakingRateTracker()

    /** v0.2.4 (Addendum 2): the preview sparkline's own accumulated series — see [emit]/
     * [clearSparkline]. [sparklineKind] tracks which signal the series currently holds values for
     * (mirrors [acquiredFor]'s edge-detection shape but is independent of it — a series switch is
     * keyed off the actual reading's own [MeterReading.signal], not the selection supplier, so a
     * stale reading from a just-superseded selection can never bleed into the new series). */
    private val sparklineSeries = ArrayDeque<Double>()
    private var sparklineKind: SignalKind? = null

    /** Which [SignalKind] the caller currently has acquired on this engine's behalf, per the last
     * edge this engine observed — see class KDoc. Starts `null` ("nothing acquired yet"), which is
     * what makes the very first inactive [step] behave as an acquire edge. */
    private var acquiredFor: SignalKind? = null

    /** Reads exactly what the currently selected signal needs (never more — e.g. never touches
     * [mic] for a HEART_RATE/MOVEMENT selection) and republishes the meter for it. Returns
     * immediately, publishing `null`/[SignalAvailability.UNKNOWN], when [isServiceActive] — see
     * class KDoc. */
    fun step() {
        val kind = selectedSignal()

        if (isServiceActive()) {
            acquiredFor?.let {
                onRelease(it)
                acquiredFor = null
            }
            publish(null)
            publishAvailability(SignalAvailability.UNKNOWN)
            clearSparkline()
            return
        }

        if (acquiredFor != kind) {
            acquiredFor?.let { onRelease(it) }
            onAcquire(kind)
            acquiredFor = kind
        }

        val availability = when (kind) {
            SignalKind.VOLUME -> {
                stepVolume()
                SignalAvailability.UNKNOWN
            }
            SignalKind.HEART_RATE -> stepHeartRate()
            SignalKind.MOVEMENT -> {
                stepScalar(accel) { movementTracker.observe(it) }
                SignalAvailability.UNKNOWN
            }
            SignalKind.SPEAKING_RATE -> {
                stepSpeakingRate()
                SignalAvailability.UNKNOWN
            }
        }
        publishAvailability(availability)
    }

    /** Clears whatever this engine last published, and releases whatever signal is currently held
     * (idempotent) — called when the caller stops driving [step] (activity backgrounded / mic
     * permission revoked) so a stale reading/acquisition doesn't linger. */
    fun close() {
        acquiredFor?.let {
            onRelease(it)
            acquiredFor = null
        }
        publish(null)
        publishAvailability(SignalAvailability.UNKNOWN)
        clearSparkline()
    }

    private fun stepVolume() {
        val window = safeReadWindow() ?: return emit(null)
        val obs = detector.observe(window)
        emit(
            MeterReading(
                signal = SignalKind.VOLUME,
                value = obs.dbfs,
                threshold = detector.baseline?.plus(TRIGGER_DB_OVER_BASELINE),
                over = obs.voiced && obs.dbOverBaseline >= TRIGGER_DB_OVER_BASELINE,
            ),
        )
    }

    private fun stepSpeakingRate() {
        val window = safeReadWindow() ?: return emit(null)
        val reading = speakingRateTracker.observe(Cadence.burstsPerSecond(window))
        emit(MeterReading(signal = SignalKind.SPEAKING_RATE, value = reading.value, threshold = reading.threshold, over = reading.over))
    }

    /** HEART_RATE's per-signal body (P4-10 in the preview): computes availability first — same
     * order as [app.gauge.wear.control.SentinelController.updateMeter] — and structurally refuses
     * to publish a reading whenever that availability has something honest to say (see class
     * KDoc). Returns the availability so [step] can [publishAvailability] it without calling
     * [ScalarSource.availability] a second time. */
    private fun stepHeartRate(): SignalAvailability {
        val availability = safeAvailability(hr)
        val blocksReading = availability != SignalAvailability.AVAILABLE && availability != SignalAvailability.UNKNOWN
        if (blocksReading) {
            emit(null)
        } else {
            stepScalar(hr) { hrTracker.observe(it) }
        }
        return availability
    }

    private fun stepScalar(source: ScalarSource?, observe: (Double) -> SignalReading) {
        val value = safeLatest(source) ?: return emit(null)
        val reading = observe(value)
        emit(MeterReading(signal = reading.kind, value = reading.value, threshold = reading.threshold, over = reading.over))
    }

    /** v0.2.4 (Addendum 2): the single choke point every per-signal body publishes through —
     * forwards to [publish] AND maintains the preview sparkline series. Real readings append
     * (capped at [SPARKLINE_LENGTH]); a `null` reading leaves the series untouched (an honest
     * gap, never a fabricated 0-point); a signal switch clears it first (values from different
     * signals must never share one series — mixed units drawn as one line would be fabricated
     * shape). */
    private fun emit(reading: MeterReading?) {
        publish(reading)
        if (reading == null) return
        if (sparklineKind != reading.signal) {
            sparklineSeries.clear()
            sparklineKind = reading.signal
        }
        sparklineSeries.addLast(reading.value)
        while (sparklineSeries.size > SPARKLINE_LENGTH) sparklineSeries.removeFirst()
        publishSparkline(sparklineSeries.toList())
    }

    /** Yield/close hygiene: clears the series AND publishes an explicit empty, so GaugeViewModel
     * can never keep drawing a stale preview line once the service owns the screen (or the
     * activity is gone). */
    private fun clearSparkline() {
        sparklineSeries.clear()
        sparklineKind = null
        publishSparkline(emptyList())
    }

    /** `null` on either "mic gone" (a clean [MicSource.readWindow] `null`) or a thrown exception —
     * fail-soft (mirrors [app.gauge.wear.control.SentinelController]'s own trigger-path posture,
     * Task 7): a misbehaving/concurrently-released [MicReader] (e.g. [MainActivity]'s onStop
     * racing a step() already blocked in a read) must degrade this window to "no reading", never
     * propagate out of [step] and crash the polling loop's thread. */
    private fun safeReadWindow(): ShortArray? = try {
        mic.readWindow()
    } catch (_: Throwable) {
        null
    }

    /** Same fail-soft contract as [safeReadWindow], for [ScalarSource.latest]. */
    private fun safeLatest(source: ScalarSource?): Double? {
        if (source == null) return null
        return try {
            source.latest()
        } catch (_: Throwable) {
            null
        }
    }

    /** [SignalAvailability.UNKNOWN] on either "no source configured" or a thrown exception — never
     * propagates out of [step], mirroring [app.gauge.wear.control.SentinelController.
     * safeAvailability]'s own fail-soft contract. */
    private fun safeAvailability(source: ScalarSource?): SignalAvailability {
        if (source == null) return SignalAvailability.UNKNOWN
        return try {
            source.availability()
        } catch (_: Throwable) {
            SignalAvailability.UNKNOWN
        }
    }

    private companion object {
        /** Matches [app.gauge.shared.sentinel.ModePolicy]'s STANDARD-mode trigger bar — this
         * engine runs entirely while DISARMED (no live [app.gauge.shared.sentinel.Mode] to read
         * one from), so STANDARD's value is the reasonable default rather than fabricating a
         * mode-specific one. */
        const val TRIGGER_DB_OVER_BASELINE = 6.0

        /** Matches [app.gauge.wear.control.SentinelController]'s own `SPARKLINE_LENGTH`. */
        const val SPARKLINE_LENGTH = 30
    }
}
