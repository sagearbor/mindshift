package app.gauge.wear.control

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.haptics.PredefinedEffect
import app.gauge.wear.haptics.Pulse
import app.gauge.wear.net.EpisodeWsClient

interface VibratorPort {
    fun vibrate(timingsMs: LongArray, amplitudes: IntArray)

    /** Whether the underlying vibrator hardware supports variable amplitude — see
     * [app.gauge.wear.haptics.HapticDirector.playPulse]'s honest documented fallback to a fixed
     * amplitude when this is `false`. Defaults to `true` so existing callers/fakes are unaffected
     * unless they explicitly opt into testing the fallback. */
    fun hasAmplitudeControl(): Boolean = true

    /** Plays a device-tuned predefined system effect. Return false when the device/port can't
     *  honor it — the caller ([app.gauge.wear.haptics.HapticDirector]) then plays the documented
     *  waveform fallback. Defaults to false so ports (and fakes) that never opt in behave exactly
     *  as before. */
    fun playPredefined(effect: PredefinedEffect): Boolean = false

    /** Plays [count] full-scale primitive clicks [gapMs] apart as ONE composed effect.
     *  Return false when primitives are unsupported — caller falls back to a waveform. */
    fun playComposedClicks(count: Int, gapMs: Long): Boolean = false
}

/**
 * A single on-device scalar sensor reading source (Task 10): HR bpm, accelerometer
 * window-stddev, etc. `null` means no reading yet (honest degradation — never fabricate a
 * value). Implementations must not throw from [latest] in production, but
 * [SentinelController] wraps every call in try/catch -> [DiagLog] anyway, so a misbehaving
 * source degrades to "no reading" rather than crashing the trigger path.
 */
interface ScalarSource {
    fun latest(): Double?

    /** Why [latest] is (or isn't) producing. Defaults to UNKNOWN — only [app.gauge.wear.sensors.
     * HrSource] has a real availability API; UNKNOWN renders no status string, so other sources
     * look exactly as before. */
    fun availability(): SignalAvailability = SignalAvailability.UNKNOWN
}

/**
 * Device→backend diagnostic sink for [SentinelController]'s fail-soft trigger path (Task 7): every
 * caught exception on that path is reported here rather than left silent. Kept as a tiny seam
 * (rather than a direct `Telemetry` dependency) so `control/` stays testable off-device — tests
 * inject a recording [DiagLog], production wires it to `Telemetry.log` (itself `runCatching`-
 * wrapped by the caller, since diagnostic reporting must never be the thing that throws).
 */
fun interface DiagLog {
    fun log(level: String, tag: String, message: String)
}

/** No-op [DiagLog] — the default for tests/call sites that don't care about diagnostics. */
val NOOP_DIAG = DiagLog { _, _, _ -> }

/** Supplies one mic window per call. `null` means the mic is gone (Task 7 treats this as mic loss → disarm). */
interface MicSource {
    fun readWindow(): ShortArray?
}

/** Mints a fresh, globally-unique episode id. One call per episode — never reused across episodes. */
fun interface EpisodeIdFactory {
    fun newId(): String
}

/**
 * Seam over [EpisodeWsClient] so wearApp's controller (Task 7) can be tested with a fake
 * WebSocket instead of a real OkHttp connection. [EpisodeWsClient] is a concrete class (it owns
 * a real `OkHttpClient` and does real network I/O in [EpisodeWsClient.open]), so this interface
 * mirrors its public surface one-for-one; [RealEpisodeWs] is the thin adapter that satisfies it
 * in production, and tests substitute their own fake implementation.
 */
interface EpisodeWs {
    fun open(episodeId: String, listener: EpisodeWsClient.Listener)
    fun sendPcmWindow(bytes: ByteArray)
    fun sendHr(bpm: Double, t: Double)
    fun end()
    fun cancel()
}

/** Production [EpisodeWs]: delegates straight through to a real [EpisodeWsClient]. */
class RealEpisodeWs(baseWsUrl: String, account: String) : EpisodeWs {
    private val client = EpisodeWsClient(baseWsUrl, account)

    override fun open(episodeId: String, listener: EpisodeWsClient.Listener) = client.open(episodeId, listener)
    override fun sendPcmWindow(bytes: ByteArray) = client.sendPcmWindow(bytes)
    override fun sendHr(bpm: Double, t: Double) = client.sendHr(bpm, t)
    override fun end() = client.end()
    override fun cancel() = client.cancel()
}

/**
 * Builds a brand-new [EpisodeWs] per episode. [SentinelController] calls this once per triggered
 * episode (never reuses an instance across episodes — see [EpisodeWs] KDoc / [EpisodeWsClient]'s
 * own "call open() at most once" contract).
 */
interface WsFactory {
    fun create(): EpisodeWs
}

/** Snapshot of [SentinelController]'s state, read synchronously by callers after [SentinelController.tick]. */
data class ControllerState(
    val sentinel: SentinelState,
    val mode: Mode,
    val online: Boolean,
    val channelLevels: Map<String, Int>,
    val lastVector: String?,
    val sparkline: List<Double>,
    val meter: MeterReading? = null,
    /** P4-3: the pulse due *this* window, if any — see [app.gauge.wear.haptics.PulseEngine] KDoc.
     * `null` whenever no pulse is due this tick (below threshold, pulses off, or gated by the
     * engine's own interval). [app.gauge.wear.service.SentinelService] reads this to drive its own
     * intra-window repeat loop (the ~1s tick cadence is coarser than the pulse interval itself —
     * see [app.gauge.wear.control.SentinelController]'s KDoc). */
    val activePulse: Pulse? = null,
    /** Availability of the CURRENTLY SELECTED signal (P4-10). Only HEART_RATE ever reports
     * anything but [SignalAvailability.UNKNOWN] today — Health Services is the only source in the
     * app with an availability API — and UNKNOWN renders no status string, so every other signal
     * looks exactly as it did before. Never a substitute for [meter]: an off-body HR publishes
     * `meter = null` AND `availability = OFF_BODY`, i.e. a reason with no fabricated number.
     *
     * ENFORCED, not merely conventional (review fix, P4-10 round 1, Critical): [SentinelController.
     * updateMeter] structurally refuses to publish a HEART_RATE [meter] value whenever this field
     * is [SignalAvailability.ACQUIRING], [SignalAvailability.OFF_BODY], or [SignalAvailability.
     * UNAVAILABLE] — a guarantee that holds at this combine point regardless of whether the
     * underlying [ScalarSource] itself remembers to clear its own last reading on an availability
     * drop (see [app.gauge.wear.sensors.HrSource.onAvailabilityChanged]'s own matching clear, kept
     * as belt-and-suspenders, not a substitute for this check). [SignalAvailability.UNKNOWN] is
     * deliberately NOT gated — a source with no real availability API (most tests, and any future
     * ScalarSource that never overrides [ScalarSource.availability]) must keep behaving exactly as
     * it did before this field existed. */
    val availability: SignalAvailability = SignalAvailability.UNKNOWN,
    /** P5-3: whether the most recently processed window was voiced/above-threshold — the one fact
     * [app.gauge.wear.service.SentinelService] needs to drive [app.gauge.shared.signals.
     * MicDutyCycle] without reaching into the controller's internals (`obs`/`detector` are private
     * to [SentinelController]). Set from `obs.voiced` at the top of `processWindow`; cleared to
     * `false` in [SentinelController.disarm] next to `meter`/`availability` — a disarmed sentinel
     * reports nothing about a window it isn't reading. */
    val lastWindowVoiced: Boolean = false,
    /** v0.2.4: a single band-1 tap due THIS window because the wearer went loud while merely
     * ARMED (before any episode opened). Set only when the sentinel both started AND stayed ARMED
     * this tick — the ARMED→STREAMING transition tick never taps (the pulse chain's own first
     * fire owns that moment), and STREAMING/COOLDOWN never tap. Rate-limited ≥2s by
     * [app.gauge.wear.haptics.ShoutTapGate]. [app.gauge.wear.service.SentinelService] plays it as
     * ONE direct playPulse — never a chain, never through PulseChainDecision. */
    val armedShoutTap: Pulse? = null,
    /** v0.2.5/Wave C: seconds of the wearer's own audio currently available to retro-capture —
     * see [app.gauge.shared.capture.RetroCaptureBuffer]. Grows on every window the mic actually
     * reads, regardless of ARMED/STREAMING/COOLDOWN (never while fully DISARMED — no window
     * exists to grow it then). Defaulted so every existing ControllerState construction site
     * keeps compiling untouched. */
    val retroCaptureAvailableSeconds: Double = 0.0,
    /** v0.3.1: which SignalKind the [sparkline] series currently holds values for — the armed
     * series follows the wearer's signal selection now (volume: dbOverBaseline; others: the
     * meter's raw units). Consumers normalize per-kind; see GaugeViewModel.resolveSparkline. */
    val sparklineSignal: SignalKind = SignalKind.VOLUME,
)
