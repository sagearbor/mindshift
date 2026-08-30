package app.gauge.wear.ui

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.sentinel.params
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.shared.signals.statusText
import app.gauge.wear.auth.AccountBus
import app.gauge.wear.control.ControllerState
import app.gauge.wear.control.MeterBus
import app.gauge.wear.control.MeterReading
import java.util.Locale
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

/**
 * Everything a glance screen needs to render, pre-computed from [ControllerState] — no
 * composable in `ui/` is allowed to re-derive any of this from raw sentinel/mode/level values;
 * see [GaugeViewModel]'s mapping rules for where each field comes from.
 *
 * Task 11 (main-screen redesign) additions: [isOn] replaces the old `isArmed` name (the ring is
 * now a plain on/off toggle, not a security-panel "arm"); [meterValue]/[meterFraction]/
 * [meterOver]/[meterHasThreshold]/[signal] mirror [ControllerState.meter] for the live green/red
 * threshold meter.
 *
 * v0.2.4: [centerText] is the ring's center string, precomputed here (not in [GlanceRing]) — the
 * center number is always the live meter (calm score is gone from the product entirely; see
 * [GaugeViewModel]'s `toGlanceUi` for the exact fallback rule).
 *
 * P4-6: [signal] is the wearer's TRUE selection (from [GaugeViewModel]'s `selectedSignal` supplier),
 * never derived from whatever the meter happens to be showing — see that constructor's KDoc for the
 * chip-lying bug this fixes. [signalHasReading] is `true` only when a combined meter reading exists
 * AND its own `signal` matches [signal] — [GlanceScreen]'s chip uses it to add an honest
 * "(no reading)" suffix rather than ever claiming a reading that isn't there.
 */
data class GlanceUi(
    val armedLabel: String,
    val ringColor: Long,
    val vectorIcon: String?,
    /** v0.2.4 (Addendum 2): pre-normalized to the honest fixed scale — see [resolveSparkline];
     * never min/max auto-normalized: the same y always means the same distance from the wearer's
     * own threshold, so flat draws flat. [sparklineThresholdFrac] is the threshold line's y on
     * that same scale. The only sparkline shape [GlanceScreen] ever sees. */
    val sparklineNorm: List<Float>,
    val sparklineThresholdFrac: Float,
    val mode: Mode,
    val showEpisode: Boolean,
    val isOn: Boolean,
    val meterValue: Double?,
    val meterFraction: Float?,
    val meterOver: Boolean,
    val meterHasThreshold: Boolean,
    val signal: SignalKind,
    val signalHasReading: Boolean,
    val centerText: String,
    /** P4-10: the honest "why there's no number" line shown directly under the ring, or `null`
     * when there's nothing honest to add — see [SignalAvailability.statusText]. Never a substitute
     * for [meterValue]: an off-body HR reading publishes `meterValue = null` AND a non-null
     * [signalStatus], never a fabricated number alongside the caption. */
    val signalStatus: String?,
    /** v0.2.4 (Addendum 2): which visualization [GlanceScreen] renders in the center this frame —
     * see [CenterDisplay]'s own KDoc. */
    val centerDisplay: CenterDisplay,
    /** Wave C Task 7: mirrors [AccountBus.signedIn] — defaulted so every existing `GlanceUi(...)`
     * construction site (including both `@Preview` functions in `GlanceScreen.kt`) keeps compiling
     * untouched, same convention as [SignalAvailability]-backed [signalStatus] before it. */
    val signedIn: Boolean = false,
    /** Wave C Task 13: mirrors [ControllerState.retroCaptureAvailableSeconds] (Task 10) directly —
     * defaulted so every existing `GlanceUi(...)` construction site keeps compiling untouched,
     * same convention as [signedIn]. Unlike [signedIn] (a separate [AccountBus] StateFlow), this
     * one rides the existing `state: ControllerState` the combine block already carries, so no
     * new combine input was needed — [ControllerState.toGlanceUi] reads it straight off its own
     * receiver. */
    val retroCaptureAvailableSeconds: Double = 0.0,
)

/** v0.2.4 (Addendum 2): which single visualization owns the main screen's center. Exactly one at
 * a time — never both (founder-ratified). */
enum class CenterDisplay { SPARKLINE, DIAL }

/** Parses [GaugePrefs.centerDisplay]'s stored value, falling back to [CenterDisplay.SPARKLINE]
 * (the deliberate v0.2.4 default — see that pref's own KDoc) for anything unrecognized, including
 * an absent/corrupt pref. Case- and whitespace-insensitive so a hand-edited or legacy value still
 * resolves sanely rather than silently losing the wearer's own view. */
internal fun parseCenterDisplay(raw: String): CenterDisplay =
    runCatching { CenterDisplay.valueOf(raw.trim().uppercase()) }.getOrDefault(CenterDisplay.SPARKLINE)

/**
 * Maps [ControllerState] (as published on [app.gauge.wear.control.ControllerStateBus], the only
 * UI feed — see its own KDoc: never poll [app.gauge.wear.control.SentinelController] directly)
 * into [GlanceUi]. Takes the bus as a plain `StateFlow` (not the singleton itself) so tests can
 * substitute a [kotlinx.coroutines.flow.MutableStateFlow] instead of touching the real bus.
 *
 * Task 12: [previewMeter] is [app.gauge.wear.control.MeterBus.preview] — [app.gauge.wear.ui.
 * PreviewMeterEngine]'s live meter while the sentinel is DISARMED. The meter actually shown is
 * `bus.meter ?: previewMeter`: the service's own meter always wins whenever an episode/arm makes
 * one available, and the preview only ever fills the gap while it's OFF (never overrides a real
 * in-progress reading — see [PreviewMeterEngine]'s own "yields to active service" contract, which
 * is what keeps `bus.meter` and `previewMeter` from ever being simultaneously "live" for the same
 * window in the first place).
 *
 * [scope] drives [uiState]'s `stateIn` — production callers pass `lifecycleScope` from
 * [MainActivity]'s `ComponentActivity`; tests pass a `TestScope`'s `backgroundScope`, which is
 * torn down automatically at the end of `runTest`.
 *
 * P4-6 (Issue A fix): [selectedSignal] is the same `() -> SignalKind` supplier shape as
 * [SentinelController]'s/[PreviewMeterEngine]'s own — [MainActivity] passes
 * `{ SignalKind.valueOf(GaugePrefs.selectedSignal(context)) }`, tests pass a lambda. Read fresh on
 * every emission (not captured once) so a mid-session selection change takes effect immediately,
 * same as [previewMeter] and [bus] themselves. [GlanceUi.signal] used to be derived as
 * `meter?.signal ?: SignalKind.VOLUME`, which silently reverted the signal chip to "Volume" any time
 * the wearer's actual selection had no live reading (HR just picked, no Health Services sample yet;
 * anything while off except mic-backed signals before Task 12 preview coverage) — the wearer had no
 * way to tell their tap didn't land. [GlanceUi.signal] now always reports this supplier's value
 * (the true selection); [GlanceUi.signalHasReading] separately says whether the combined meter
 * actually matches that selection, so the chip can add an honest "(no reading)" suffix instead of
 * lying about the selection itself.
 */
class GaugeViewModel(
    bus: StateFlow<ControllerState>,
    scope: CoroutineScope,
    previewMeter: StateFlow<MeterReading?> = MeterBus.preview,
    previewAvailability: StateFlow<SignalAvailability> = MeterBus.previewAvailability,
    previewSparkline: StateFlow<List<Double>> = MeterBus.previewSparkline,
    selectedSignal: () -> SignalKind = { SignalKind.VOLUME },
    centerDisplayPref: () -> String = { CenterDisplay.SPARKLINE.name },
    accountSignedIn: StateFlow<Boolean> = AccountBus.signedIn,
) {
    val uiState: StateFlow<GlanceUi> =
        combine(
            bus,
            previewMeter,
            previewAvailability,
            previewSparkline,
            accountSignedIn,
        ) { state, preview, previewAvail, previewSpark, signedIn ->
            val meter = state.meter ?: preview
            state.withMeter(meter)
                .withAvailability(state.resolvedAvailability(previewAvail))
                .toGlanceUi(
                    selectedSignal(),
                    resolveSparkline(state, meter, previewSpark),
                    parseCenterDisplay(centerDisplayPref()),
                    signedIn,
                )
        }.stateIn(
            scope = scope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = run {
                val meter = bus.value.meter ?: previewMeter.value
                bus.value.withMeter(meter)
                    .withAvailability(bus.value.resolvedAvailability(previewAvailability.value))
                    .toGlanceUi(
                        selectedSignal(),
                        resolveSparkline(bus.value, meter, previewSparkline.value),
                        parseCenterDisplay(centerDisplayPref()),
                        accountSignedIn.value,
                    )
            },
        )
}

private fun ControllerState.withMeter(meter: MeterReading?): ControllerState = copy(meter = meter)

private fun ControllerState.withAvailability(availability: SignalAvailability): ControllerState =
    copy(availability = availability)

/** P4-10 resolution rule, mirroring the existing meter rule (service wins when it has something
 * to say): the service's own [ControllerState.availability] is used whenever it isn't
 * [SignalAvailability.UNKNOWN] (a real availability read for the current selection); otherwise
 * fall back to the preview engine's own reading — see [GaugeViewModel]'s constructor KDoc. */
private fun ControllerState.resolvedAvailability(previewAvail: SignalAvailability): SignalAvailability =
    if (availability != SignalAvailability.UNKNOWN) availability else previewAvail

/** Green/red meter colors — same green as [ringColorFor]'s level-0 case, kept as its own
 * constant here since the meter's over/under decision is independent of nudge levels. */
private const val METER_COLOR_GREEN = 0xFF2E7D32L
private const val METER_COLOR_RED = 0xFFC62828L

/** v0.2.4: the window a signal's ring-meter arc sweeps — [below] units under the threshold to
 * [over] units above it. Per-signal because the old global 12/6 was a dB scale wearing a
 * one-size-fits-all costume: on bursts/sec (plausible range ~0–4) it froze the arc. */
internal data class MeterSpan(val below: Double, val over: Double)

internal fun meterSpanFor(signal: SignalKind): MeterSpan = when (signal) {
    SignalKind.VOLUME -> MeterSpan(12.0, 6.0)        // dB — unchanged
    SignalKind.HEART_RATE -> MeterSpan(12.0, 6.0)    // bpm — unchanged (addendum-ratified)
    SignalKind.SPEAKING_RATE -> MeterSpan(2.0, 1.0)  // b/s: threshold = baseline+1.5, range ~0–4
    SignalKind.MOVEMENT -> MeterSpan(2.0, 1.0)       // accel window-stddev units (MAD-scale threshold)
}

/** v0.2.4: dB of over-threshold headroom above the sparkline's threshold line — the same 6 units
 * the ring meter's over-span uses, so "pinned at the top of the sparkline" and "pinned at the red
 * end of the meter" mean the same thing. */
internal const val SPARK_OVER_HEADROOM_DB = 6.0

/** One resolved, ready-to-draw sparkline: pre-normalized 0..1 points + the threshold line's
 * y-fraction on the SAME scale. The only sparkline shape GlanceScreen ever sees. */
internal data class SparkView(val norm: List<Float>, val thresholdFrac: Float)

/** The trigger bar to DISPLAY for [mode]. SESSION's own bar is 0.0 (streams unconditionally —
 * no trigger concept), which would put the threshold line on the baseline; it borrows STANDARD's
 * +6dB instead — the exact carve-out (and rationale) of SentinelController.pulseThresholdDbFor. */
internal fun displayThresholdDbFor(mode: Mode): Double =
    if (mode == Mode.SESSION) Mode.STANDARD.params().triggerDbOverBaseline
    else mode.params().triggerDbOverBaseline

/** Fixed-scale normalization for the SERVICE series (values are dbOverBaseline): 0 = the wearer's
 * own baseline, 1 = threshold + headroom. Clamped, never rescaled — no min/max auto-normalize. */
internal fun normalizeSparkline(values: List<Double>, thresholdDb: Double): List<Float> {
    val top = thresholdDb + SPARK_OVER_HEADROOM_DB
    if (top <= 0.0) return values.map { 0f }
    return values.map { (it.coerceIn(0.0, top) / top).toFloat() }
}

/** The threshold line's 0..1 y-fraction on that same service scale. */
internal fun sparklineThresholdFrac(thresholdDb: Double): Float {
    val top = thresholdDb + SPARK_OVER_HEADROOM_DB
    if (top <= 0.0) return 1f
    return (thresholdDb / top).toFloat()
}

/** Fixed-scale normalization for the PREVIEW series (values are raw signal units): the window is
 * the signal's own [MeterSpan] around [threshold] — identical anchoring to the ring meter's arc,
 * so the sparkline and the meter always agree on what "near the line" means. */
internal fun normalizeMeterSeries(values: List<Double>, threshold: Double, span: MeterSpan): List<Float> {
    val low = threshold - span.below
    val width = span.below + span.over
    if (width <= 0.0) return emptyList()
    return values.map { ((it - low) / width).toFloat().coerceIn(0f, 1f) }
}

/** The threshold line's 0..1 y-fraction on that same preview scale. */
internal fun meterThresholdFrac(span: MeterSpan): Float {
    val width = span.below + span.over
    if (width <= 0.0) return 1f
    return (span.below / width).toFloat()
}

/**
 * v0.2.4 (Addendum 2), revised v0.3.1: THE sparkline resolution rule, mirroring the meter's own
 * service-wins shape. Sentinel on -> the controller's series follows [ControllerState.
 * sparklineSignal] (the wearer's own selection, overriding v0.2.4's loudness-only decision):
 * VOLUME stays on the fixed loudness scale (dbOverBaseline, byte-identical to pre-v0.3.1
 * behavior); every other signal normalizes on that signal's own [MeterSpan] around [meter]'s live
 * threshold, same anchoring the ring meter's arc uses — no threshold yet -> an EMPTY SparkView
 * (honest blank, see armedNonVolumeSparklineWithoutThresholdDrawsNothing). Sentinel off -> the
 * preview engine's per-signal series anchored to [meter]'s live threshold and span; no threshold
 * yet -> an EMPTY SparkView (honest blank — see offSparklineWithoutAThresholdDrawsNothing). The
 * thresholdFrac of an empty view is unused by GlanceScreen (it draws no line for an empty series).
 */
internal fun resolveSparkline(
    state: ControllerState,
    meter: MeterReading?,
    previewSeries: List<Double>,
): SparkView {
    if (state.sentinel != SentinelState.DISARMED) {
        if (state.sparklineSignal == SignalKind.VOLUME) {
            val th = displayThresholdDbFor(state.mode)
            return SparkView(normalizeSparkline(state.sparkline, th), sparklineThresholdFrac(th))
        }
        val threshold = meter?.threshold ?: return SparkView(emptyList(), 0.5f)
        val span = meterSpanFor(state.sparklineSignal)
        return SparkView(normalizeMeterSeries(state.sparkline, threshold, span), meterThresholdFrac(span))
    }
    val threshold = meter?.threshold ?: return SparkView(emptyList(), 0.5f)
    val span = meterSpanFor(meter.signal)
    return SparkView(normalizeMeterSeries(previewSeries, threshold, span), meterThresholdFrac(span))
}

private fun ControllerState.toGlanceUi(
    selectedSignal: SignalKind,
    spark: SparkView,
    centerDisplay: CenterDisplay,
    signedIn: Boolean,
): GlanceUi {
    val showEpisode = sentinel == SentinelState.STREAMING
    val isOn = sentinel != SentinelState.DISARMED
    val armedLabelText = armedLabel()
    val meterText = meter?.let { formatMeterValue(it.value, it.signal) }
    // v0.2.4: the center number is always the meter — calm score is gone. "Off" only when BOTH
    // disarmed AND no reading exists; otherwise the live reading, else the armed label.
    val centerText = when {
        // Tier B: COMPANION has no mic, so no meter ever exists — the center shows the phone-
        // relayed nudge level instead ("what my wrist is being told right now"), falling back to
        // the "Companion" label while everything is calm.
        mode == Mode.COMPANION && sentinel == SentinelState.STREAMING ->
            (channelLevels.values.maxOrNull() ?: 0).let { if (it > 0) "Level $it" else armedLabelText }
        !isOn && meterText == null -> "Off"
        meterText != null -> meterText
        else -> armedLabelText
    }
    return GlanceUi(
        armedLabel = armedLabelText,
        ringColor = when {
            // Nudge-level color always wins during an episode — a live escalation matters more
            // than where the meter happens to sit at that instant.
            showEpisode -> ringColorFor(channelLevels.values.maxOrNull() ?: 0)
            meter != null -> if (meter.over) METER_COLOR_RED else METER_COLOR_GREEN
            else -> ringColorFor(channelLevels.values.maxOrNull() ?: 0)
        },
        vectorIcon = lastVector?.let(::vectorIconFor),
        sparklineNorm = spark.norm,
        sparklineThresholdFrac = spark.thresholdFrac,
        mode = mode,
        showEpisode = showEpisode,
        isOn = isOn,
        meterValue = meter?.value,
        meterFraction = meter?.threshold?.let { threshold ->
            meterFraction(meter.value, threshold, meterSpanFor(meter.signal))
        },
        meterOver = meter?.over ?: false,
        meterHasThreshold = meter?.threshold != null,
        // P4-6 (Issue A fix): always the TRUE selection, never derived from whatever the meter
        // happens to hold — see [GaugeViewModel]'s constructor KDoc for the chip-lying bug this
        // replaces. signalHasReading tells the composable, separately, whether that selection
        // currently has a live reading to show.
        signal = selectedSignal,
        signalHasReading = meter != null && meter.signal == selectedSignal,
        centerText = centerText,
        signalStatus = availability.statusText(),
        centerDisplay = centerDisplay,
        signedIn = signedIn,
        retroCaptureAvailableSeconds = retroCaptureAvailableSeconds,
    )
}

/** Ring color mapping — max level across all channels; 0 (or no channels reporting) is green. */
private fun ringColorFor(maxLevel: Int): Long = when (maxLevel) {
    0 -> 0xFF2E7D32 // green
    1 -> 0xFFF9A825 // amber
    2 -> 0xFFEF6C00 // orange
    else -> 0xFFC62828 // red (level 3+)
}

private fun ControllerState.armedLabel(): String = when (sentinel) {
    SentinelState.DISARMED -> "Off"
    SentinelState.ARMED -> "On"
    SentinelState.STREAMING -> when {
        // Tier B: COMPANION's STREAMING is "socket open, phone listening", not an episode.
        mode == Mode.COMPANION && online -> "Companion"
        mode == Mode.COMPANION -> "Companion · offline"
        online -> "Episode"
        else -> "Episode · offline"
    }
    SentinelState.COOLDOWN -> "Cooling down"
}

/**
 * 0..1 position of [value] inside `[threshold - span.below, threshold + span.over]` — the window
 * the live meter sweeps across, e.g. for VOLUME's dB scale `[threshold-12, threshold+6]`. Coerced
 * to `0f..1f` so a value outside that window (a very quiet or very loud reading) still draws a
 * valid arc rather than an out-of-range sweep.
 */
private fun meterFraction(value: Double, threshold: Double, span: MeterSpan): Float {
    val low = threshold - span.below
    val width = span.below + span.over
    return ((value - low) / width).toFloat().coerceIn(0f, 1f)
}

private fun vectorIconFor(vector: String): String? = when (vector) {
    "yelling" -> "📢"
    "aggressive_tone" -> "🔥"
    "interrupting" -> "✂️"
    "airtime" -> "🎤"
    "hr_spike" -> "❤️"
    else -> null
}

/** "-24.0dB" / "72.0bpm" / "3.5" / "1.2b/s" — one decimal place, unit suffix per [SignalKind]
 * (MOVEMENT's accelerometer stddev has no natural unit, so its suffix is empty). Moved here from
 * `GlanceScreen.kt` (P4-2) since [ControllerState.toGlanceUi] now needs it to precompute
 * [GlanceUi.centerText]. */
private fun formatMeterValue(value: Double, signal: SignalKind): String {
    val unit = when (signal) {
        SignalKind.VOLUME -> "dB"
        SignalKind.HEART_RATE -> "bpm"
        SignalKind.MOVEMENT -> ""
        SignalKind.SPEAKING_RATE -> "b/s"
    }
    return String.format(Locale.US, "%.1f", value) + unit
}
