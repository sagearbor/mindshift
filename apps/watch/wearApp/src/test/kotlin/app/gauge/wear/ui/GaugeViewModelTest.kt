package app.gauge.wear.ui

import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.control.ControllerState
import app.gauge.wear.control.MeterReading
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

private fun state(
    sentinel: SentinelState = SentinelState.DISARMED,
    mode: Mode = Mode.STANDARD,
    online: Boolean = true,
    channelLevels: Map<String, Int> = emptyMap(),
    lastVector: String? = null,
    sparkline: List<Double> = emptyList(),
    meter: MeterReading? = null,
    availability: SignalAvailability = SignalAvailability.UNKNOWN,
    retroCaptureAvailableSeconds: Double = 0.0,
    sparklineSignal: SignalKind = SignalKind.VOLUME,
) = ControllerState(
    sentinel = sentinel,
    mode = mode,
    online = online,
    channelLevels = channelLevels,
    lastVector = lastVector,
    sparkline = sparkline,
    meter = meter,
    retroCaptureAvailableSeconds = retroCaptureAvailableSeconds,
    availability = availability,
    sparklineSignal = sparklineSignal,
)

/** Shorthand matching the brief's own `ui(state)` helper name — builds a [GaugeViewModel] from
 * [state] and returns its synchronously-available initial [GlanceUi]. [selectedSignal] defaults to
 * [SignalKind.VOLUME], matching [GaugeViewModel]'s own default. [previewAvailability] defaults to
 * [SignalAvailability.UNKNOWN], matching [MeterBus.previewAvailability]'s own default.
 * [previewMeter]/[previewSparkline] default to `null`/empty — previously [previewMeter] fell back
 * to the real [app.gauge.wear.control.MeterBus.preview] singleton; a fresh `MutableStateFlow(null)`
 * default is behaviorally identical and keeps every test isolated from that shared state.
 * [accountSignedIn] defaults to `false`, matching [app.gauge.wear.auth.AccountBus.signedIn]'s own
 * default — wrapped in its own isolated `MutableStateFlow` (never the real singleton), same
 * isolation reasoning as [previewMeter]/[previewAvailability]/[previewSparkline] above. */
@OptIn(ExperimentalCoroutinesApi::class)
private fun kotlinx.coroutines.test.TestScope.ui(
    state: ControllerState,
    selectedSignal: () -> SignalKind = { SignalKind.VOLUME },
    previewAvailability: SignalAvailability = SignalAvailability.UNKNOWN,
    previewMeter: MeterReading? = null,
    previewSparkline: List<Double> = emptyList(),
    centerDisplayPref: () -> String = { CenterDisplay.SPARKLINE.name },
    accountSignedIn: Boolean = false,
): GlanceUi =
    GaugeViewModel(
        bus = MutableStateFlow(state),
        scope = backgroundScope,
        previewMeter = MutableStateFlow(previewMeter),
        previewAvailability = MutableStateFlow(previewAvailability),
        previewSparkline = MutableStateFlow(previewSparkline),
        selectedSignal = selectedSignal,
        centerDisplayPref = centerDisplayPref,
        accountSignedIn = MutableStateFlow(accountSignedIn),
    ).uiState.value

/**
 * Mapping tests for [GaugeViewModel]. [GaugeViewModel.uiState]'s initial value is computed
 * synchronously from the bus's current value (see its `stateIn` call), so `.value` is correct
 * immediately — no need to advance a dispatcher to observe it, but we still construct the
 * ViewModel inside [runTest]'s [kotlinx.coroutines.test.TestScope] (via `backgroundScope`) per the
 * brief, since that's the scope `stateIn` needs and it's torn down automatically at test end.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class GaugeViewModelTest {

    @Test
    fun ringColorGreenAtLevelZero() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(channelLevels = mapOf("A" to 0))), backgroundScope)
        assertEquals(0xFF2E7D32, vm.uiState.value.ringColor)
    }

    @Test
    fun ringColorGreenWithNoChannels() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state()), backgroundScope)
        assertEquals(0xFF2E7D32, vm.uiState.value.ringColor)
    }

    @Test
    fun ringColorAmberAtLevelOne() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(channelLevels = mapOf("A" to 1))), backgroundScope)
        assertEquals(0xFFF9A825, vm.uiState.value.ringColor)
    }

    @Test
    fun ringColorOrangeAtLevelTwo() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(channelLevels = mapOf("A" to 2))), backgroundScope)
        assertEquals(0xFFEF6C00, vm.uiState.value.ringColor)
    }

    @Test
    fun ringColorRedAtLevelThree() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(channelLevels = mapOf("A" to 3))), backgroundScope)
        assertEquals(0xFFC62828, vm.uiState.value.ringColor)
    }

    @Test
    fun ringColorUsesMaxAcrossChannels() = runTest {
        val vm = GaugeViewModel(
            MutableStateFlow(state(channelLevels = mapOf("A" to 1, "B" to 3))),
            backgroundScope,
        )
        assertEquals(0xFFC62828, vm.uiState.value.ringColor)
    }

    @Test
    fun armedLabelDisarmedIsOff() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(sentinel = SentinelState.DISARMED)), backgroundScope)
        assertEquals("Off", vm.uiState.value.armedLabel)
    }

    @Test
    fun armedLabelArmedIsOn() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(sentinel = SentinelState.ARMED)), backgroundScope)
        assertEquals("On", vm.uiState.value.armedLabel)
    }

    @Test
    fun armedLabelStreamingOnlineIsEpisode() = runTest {
        val vm = GaugeViewModel(
            MutableStateFlow(state(sentinel = SentinelState.STREAMING, online = true)),
            backgroundScope,
        )
        assertEquals("Episode", vm.uiState.value.armedLabel)
    }

    @Test
    fun armedLabelStreamingOfflineShowsOfflineSuffix() = runTest {
        val vm = GaugeViewModel(
            MutableStateFlow(state(sentinel = SentinelState.STREAMING, online = false)),
            backgroundScope,
        )
        assertEquals("Episode · offline", vm.uiState.value.armedLabel)
    }

    @Test
    fun armedLabelCooldownIsCoolingDown() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(sentinel = SentinelState.COOLDOWN)), backgroundScope)
        assertEquals("Cooling down", vm.uiState.value.armedLabel)
    }

    @Test
    fun vectorIconMapsYelling() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "yelling")), backgroundScope)
        assertEquals("📢", vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconMapsAggressiveTone() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "aggressive_tone")), backgroundScope)
        assertEquals("🔥", vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconMapsInterrupting() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "interrupting")), backgroundScope)
        assertEquals("✂️", vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconMapsAirtime() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "airtime")), backgroundScope)
        assertEquals("🎤", vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconMapsHrSpike() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "hr_spike")), backgroundScope)
        assertEquals("❤️", vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconNullWhenNoLastVector() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = null)), backgroundScope)
        assertNull(vm.uiState.value.vectorIcon)
    }

    @Test
    fun vectorIconNullForUnknownVector() = runTest {
        val vm = GaugeViewModel(MutableStateFlow(state(lastVector = "unknown_vector")), backgroundScope)
        assertNull(vm.uiState.value.vectorIcon)
    }

    @Test
    fun showEpisodeTrueOnlyWhileStreaming() = runTest {
        for (s in SentinelState.entries) {
            val vm = GaugeViewModel(MutableStateFlow(state(sentinel = s)), backgroundScope)
            if (s == SentinelState.STREAMING) {
                assertTrue(vm.uiState.value.showEpisode, "expected showEpisode for $s")
            } else {
                assertFalse(vm.uiState.value.showEpisode, "expected !showEpisode for $s")
            }
        }
    }

    @Test
    fun sparklinePassthrough() = runTest {
        // SESSION displays STANDARD's 6.0 bar (see displayThresholdDbFor) -> scale top 12.
        val ui = ui(state(mode = Mode.SESSION, sparkline = listOf(1.0, 2.0, 3.0), sentinel = SentinelState.ARMED))
        assertEquals(Mode.SESSION, ui.mode)
        assertEquals(listOf((1.0 / 12).toFloat(), (2.0 / 12).toFloat(), (3.0 / 12).toFloat()), ui.sparklineNorm)
    }

    @Test
    fun isOnFalseOnlyWhenDisarmed() = runTest {
        for (s in SentinelState.entries) {
            val vm = GaugeViewModel(MutableStateFlow(state(sentinel = s)), backgroundScope)
            if (s == SentinelState.DISARMED) {
                assertFalse(vm.uiState.value.isOn, "expected !isOn for $s")
            } else {
                assertTrue(vm.uiState.value.isOn, "expected isOn for $s")
            }
        }
    }

    // --- Task 11: live meter mapping -------------------------------------------------------------

    @Test
    fun meterUnderThresholdIsGreenAndFraction() = runTest {
        val ui = ui(state(meter = MeterReading(SignalKind.VOLUME, value = -30.0, threshold = -24.0, over = false)))
        assertFalse(ui.meterOver)
        assertEquals(0xFF2E7D32, ui.ringColor)
        // fraction = (value - (threshold - 12)) / (12 + 6) = (-30 - (-36)) / 18 = 6 / 18
        assertEquals(0.333f, ui.meterFraction!!, 0.01f)
    }

    @Test
    fun meterOverThresholdIsRed() = runTest {
        val ui = ui(state(meter = MeterReading(SignalKind.VOLUME, -20.0, -24.0, true)))
        assertTrue(ui.meterOver)
        assertEquals(0xFFC62828, ui.ringColor)
    }

    @Test
    fun noThresholdRendersGreenWithoutFraction() = runTest {
        val ui = ui(state(meter = MeterReading(SignalKind.HEART_RATE, 72.0, null, false)))
        assertFalse(ui.meterHasThreshold)
        assertNull(ui.meterFraction)
        assertFalse(ui.meterOver)
        assertEquals(0xFF2E7D32, ui.ringColor)
    }

    @Test
    fun episodeNudgeColorStillWinsOverMeterColor() = runTest {
        val ui = ui(
            state(
                sentinel = SentinelState.STREAMING,
                channelLevels = mapOf("A" to 3),
                meter = MeterReading(SignalKind.VOLUME, -30.0, -24.0, false),
            ),
        )
        assertEquals(0xFFC62828, ui.ringColor) // level-3 red beats meter green
    }

    // --- P4-10: honest status under the meter ---------------------------------------------------

    @Test
    fun signalStatusOffBodyWhenTheServiceReportsOffBody() = runTest {
        val ui = ui(state(availability = SignalAvailability.OFF_BODY))
        assertEquals("off-body — wear snug", ui.signalStatus)
    }

    @Test
    fun signalStatusAcquiring() = runTest {
        assertEquals("acquiring…", ui(state(availability = SignalAvailability.ACQUIRING)).signalStatus)
    }

    @Test
    fun signalStatusUnavailable() = runTest {
        assertEquals("unavailable", ui(state(availability = SignalAvailability.UNAVAILABLE)).signalStatus)
    }

    @Test
    fun signalStatusNullWhenAvailableOrUnknown() = runTest {
        assertNull(ui(state(availability = SignalAvailability.AVAILABLE)).signalStatus)
        assertNull(ui(state(availability = SignalAvailability.UNKNOWN)).signalStatus)
    }

    @Test
    fun offBodyShowsAStatusAndStillNoFabricatedReading() = runTest {
        val ui = ui(
            state(meter = null, availability = SignalAvailability.OFF_BODY),
            selectedSignal = { SignalKind.HEART_RATE },
        )
        assertEquals("off-body — wear snug", ui.signalStatus)
        assertNull(ui.meterValue)
        assertFalse(ui.signalHasReading)
        assertEquals(SignalKind.HEART_RATE, ui.signal)
    }

    @Test
    fun previewAvailabilityUsedWhenTheServiceHasNothingToSay() = runTest {
        // Sentinel OFF: the preview engine is the only thing reading HR, so its availability is
        // the one that matters — same shape as the existing `state.meter ?: preview` rule.
        val ui = ui(
            state(availability = SignalAvailability.UNKNOWN),
            previewAvailability = SignalAvailability.OFF_BODY,
        )
        assertEquals("off-body — wear snug", ui.signalStatus)
    }

    @Test
    fun serviceAvailabilityWinsOverPreviewWhenKnown() = runTest {
        val ui = ui(
            state(availability = SignalAvailability.AVAILABLE),
            previewAvailability = SignalAvailability.OFF_BODY,
        )
        assertNull(ui.signalStatus) // AVAILABLE -> nothing to say; the preview's stale status loses
    }

    // --- Task 12: preview-meter combine (live meter while OFF) -----------------------------------

    @Test
    fun previewMeterShowsWhenServiceMeterIsNull() = runTest {
        val previewMeter = MutableStateFlow<MeterReading?>(MeterReading(SignalKind.VOLUME, -20.0, -24.0, true))
        val vm = GaugeViewModel(MutableStateFlow(state(meter = null)), backgroundScope, previewMeter)
        val ui = vm.uiState.value
        assertEquals(SignalKind.VOLUME, ui.signal)
        assertTrue(ui.signalHasReading)
        assertEquals(-20.0, ui.meterValue)
        assertTrue(ui.meterOver)
        assertEquals(0xFFC62828, ui.ringColor)
    }

    @Test
    fun serviceMeterWinsWhenBothExist() = runTest {
        val previewMeter = MutableStateFlow<MeterReading?>(MeterReading(SignalKind.VOLUME, -20.0, -24.0, true))
        val vm = GaugeViewModel(
            bus = MutableStateFlow(state(meter = MeterReading(SignalKind.HEART_RATE, 72.0, null, false))),
            scope = backgroundScope,
            previewMeter = previewMeter,
            selectedSignal = { SignalKind.HEART_RATE },
        )
        val ui = vm.uiState.value
        assertEquals(SignalKind.HEART_RATE, ui.signal)
        assertTrue(ui.signalHasReading)
        assertEquals(72.0, ui.meterValue)
        assertFalse(ui.meterOver)
    }

    // --- v0.2.4: the center number is always the meter — calm score is gone --------------------

    @Test
    fun centerTextDuringAnEpisodeIsAlwaysTheMeter() = runTest {
        val ui = ui(
            state(
                sentinel = SentinelState.STREAMING,
                meter = MeterReading(SignalKind.VOLUME, -18.0, -24.0, true),
            ),
        )
        assertEquals("-18.0dB", ui.centerText)
    }

    @Test
    fun centerTextFallsBackToArmedLabelWhenOnWithNoReading() = runTest {
        assertEquals("On", ui(state(sentinel = SentinelState.ARMED, meter = null)).centerText)
    }

    @Test
    fun centerTextOffWhenDisarmedWithNoReading() = runTest {
        assertEquals("Off", ui(state(sentinel = SentinelState.DISARMED, meter = null)).centerText)
    }

    // Review fix: PreviewMeterEngine keeps a meter alive on MeterBus while the sentinel is
    // DISARMED (app open, mic permission granted) — that's the common case, not an edge case, and
    // "Off" must not clobber a live reading. "Off" is reserved for disarmed-with-no-meter-at-all.
    @Test
    fun centerTextShowsMeterValueWhileDisarmedWithLiveReading() = runTest {
        val ui = ui(
            state(
                sentinel = SentinelState.DISARMED,
                meter = MeterReading(SignalKind.VOLUME, -32.5, -24.0, false),
            ),
        )
        assertEquals("-32.5dB", ui.centerText)
    }

    // --- P4-6 (Issue A): the signal chip must always show the TRUE selection, never silently
    // revert to VOLUME just because the selected signal has no reading yet -------------------------

    @Test
    fun signalReflectsTrueSelectionEvenWithNoMeterReading() = runTest {
        // HR selected, sentinel off, no reading of any kind yet (no previewMeter either) — the old
        // `meter?.signal ?: VOLUME` derivation would have reported VOLUME here, lying about the
        // wearer's actual selection.
        val ui = ui(state(meter = null), selectedSignal = { SignalKind.HEART_RATE })
        assertEquals(SignalKind.HEART_RATE, ui.signal)
        assertFalse(ui.signalHasReading)
    }

    @Test
    fun signalHasReadingTrueWhenMeterMatchesSelection() = runTest {
        val ui = ui(
            state(meter = MeterReading(SignalKind.HEART_RATE, 72.0, null, false)),
            selectedSignal = { SignalKind.HEART_RATE },
        )
        assertEquals(SignalKind.HEART_RATE, ui.signal)
        assertTrue(ui.signalHasReading)
    }

    @Test
    fun signalHasReadingFalseWhenMeterIsForADifferentSignalThanSelection() = runTest {
        // Defensive case: a stale/mismatched meter reading (e.g. a preview reading left over from a
        // just-changed selection) must never be reported as satisfying the current selection.
        val ui = ui(
            state(meter = MeterReading(SignalKind.VOLUME, -30.0, -24.0, false)),
            selectedSignal = { SignalKind.HEART_RATE },
        )
        assertEquals(SignalKind.HEART_RATE, ui.signal)
        assertFalse(ui.signalHasReading)
    }

    @Test
    fun signalChipReflectsSupplierChangeOnNextEmissionWithoutAnyMeter() = runTest {
        // "Supplier-driven": GaugeViewModel must read `selectedSignal()` fresh on every combine
        // emission, so a mid-session selection change shows up the moment the bus/previewMeter next
        // emits — never captured once at construction time.
        // The combine flow only runs (past its synchronously-computed initial value) while
        // something is actively collecting [GaugeViewModel.uiState] — WhileSubscribed(5_000) — so
        // this test collects it explicitly, same as the real app's `collectAsState()` does.
        var current = SignalKind.VOLUME
        val busFlow = MutableStateFlow(state(meter = null))
        val vm = GaugeViewModel(busFlow, backgroundScope, selectedSignal = { current })
        backgroundScope.launch { vm.uiState.collect {} }
        runCurrent()
        assertEquals(SignalKind.VOLUME, vm.uiState.value.signal)

        current = SignalKind.MOVEMENT
        // Force a fresh (distinct) emission on the bus to drive recombination — the sparkline is
        // otherwise irrelevant to this test, just a cheap way to make the new state != the old one.
        busFlow.value = state(meter = null, sparkline = listOf(99.0))
        runCurrent()

        assertEquals(SignalKind.MOVEMENT, vm.uiState.value.signal)
        assertFalse(vm.uiState.value.signalHasReading)
    }

    // --- v0.2.4: per-signal meter spans ---------------------------------------------------------

    @Test
    fun meterSpansArePerSignal() {
        assertEquals(MeterSpan(12.0, 6.0), meterSpanFor(SignalKind.VOLUME))
        assertEquals(MeterSpan(12.0, 6.0), meterSpanFor(SignalKind.HEART_RATE))
        assertEquals(MeterSpan(2.0, 1.0), meterSpanFor(SignalKind.SPEAKING_RATE))
        assertEquals(MeterSpan(2.0, 1.0), meterSpanFor(SignalKind.MOVEMENT))
    }

    @Test
    fun speakingRateArcActuallySweepsItsPlausibleRange() = runTest {
        // Baseline speech ~2.0 b/s with threshold 3.5 (SpeakingRateTracker's baseline+1.5). Under
        // the old global 12/6-dB span, this whole plausible range collapsed into ~0.08 of arc
        // (0.583 -> 0.666); with the 2.0/1.0 span it sweeps half the dial.
        fun frac(value: Double): Float = ui(
            state(meter = MeterReading(SignalKind.SPEAKING_RATE, value, 3.5, value >= 3.5)),
            selectedSignal = { SignalKind.SPEAKING_RATE },
        ).meterFraction!!
        assertEquals(0.5f, frac(3.0), 0.01f)      // (3.0 - 1.5) / 3.0
        assertEquals(0.667f, frac(3.5), 0.01f)    // at threshold
        assertEquals(1.0f, frac(4.5), 0.01f)      // threshold + full over-span
        assertTrue(frac(3.5) - frac(2.0) > 0.4f)  // the sweep the old span never had
    }

    @Test
    fun volumeMeterFractionIsUnchanged() = runTest {
        val ui = ui(state(meter = MeterReading(SignalKind.VOLUME, -24.0, -18.0, false)))
        // (-24 - (-18 - 12)) / 18 = 6/18
        assertEquals(0.333f, ui.meterFraction!!, 0.01f)
    }

    // --- v0.2.4 (Addendum 2): one resolved honest sparkline scale -------------------------------

    @Test
    fun displayThresholdUsesEachModesOwnTriggerBar() {
        assertEquals(6.0, displayThresholdDbFor(Mode.STANDARD))
        assertEquals(10.0, displayThresholdDbFor(Mode.BATTERY_SAVER))
    }

    @Test
    fun sessionBorrowsStandardsBarForDisplay() {
        // SESSION's own trigger bar is 0.0 (it streams unconditionally) — a threshold line at the
        // baseline would be meaningless. Same carve-out, same rationale, as the controller's
        // pulseThresholdDbFor.
        assertEquals(6.0, displayThresholdDbFor(Mode.SESSION))
    }

    @Test
    fun normalizeAnchorsBaselineThresholdAndHeadroom() {
        // Scale for threshold 6.0 runs 0..12: baseline -> 0, threshold -> 0.5, threshold+6 -> 1.
        val norm = normalizeSparkline(listOf(0.0, 3.0, 6.0, 12.0), thresholdDb = 6.0)
        assertEquals(listOf(0.0f, 0.25f, 0.5f, 1.0f), norm)
    }

    @Test
    fun normalizeClampsOutOfRangeValuesInsteadOfRescaling() {
        // A -4dB (quieter than baseline) or +20dB point must clamp, NOT stretch the scale — the
        // no-fake-data rule applied to pixels: rescaling would redraw history to fit the outlier.
        val norm = normalizeSparkline(listOf(-4.0, 20.0), thresholdDb = 6.0)
        assertEquals(listOf(0.0f, 1.0f), norm)
    }

    @Test
    fun flatQuietAudioDrawsFlatNotStretched() {
        // The exact min/max-auto-normalize bug this replaces: identical near-baseline values used
        // to be stretched across the full canvas height by (v - min) / (max - min).
        val norm = normalizeSparkline(listOf(0.5, 0.6, 0.5, 0.6), thresholdDb = 6.0)
        for (v in norm) assertTrue(v < 0.06f, "near-baseline point drawn at $v — was stretched")
    }

    @Test
    fun thresholdFracMatchesTheSameScale() {
        assertEquals(0.5f, sparklineThresholdFrac(6.0))
        assertEquals(0.625f, sparklineThresholdFrac(10.0))
    }

    @Test
    fun armedSparklineUsesTheServiceSeriesOnTheLoudnessScale() = runTest {
        val ui = ui(
            state(sentinel = SentinelState.ARMED, mode = Mode.STANDARD, sparkline = listOf(0.0, 6.0, 12.0)),
            previewSparkline = listOf(999.0), // must be ignored while the sentinel is on
        )
        assertEquals(listOf(0.0f, 0.5f, 1.0f), ui.sparklineNorm)
        assertEquals(0.5f, ui.sparklineThresholdFrac)
    }

    // --- v0.3.1: armed sparkline follows the selected signal (was hardcoded to loudness) -------

    @Test
    fun armedNonVolumeSparklineNormalizesOnTheSignalsMeterSpan() = runTest {
        val meter = MeterReading(signal = SignalKind.HEART_RATE, value = 90.0, threshold = 100.0, over = false)
        val ui = ui(
            state(
                sentinel = SentinelState.ARMED,
                sparkline = listOf(80.0, 90.0, 100.0),
                sparklineSignal = SignalKind.HEART_RATE,
                meter = meter,
            ),
        )
        val span = meterSpanFor(SignalKind.HEART_RATE)
        assertEquals(normalizeMeterSeries(listOf(80.0, 90.0, 100.0), 100.0, span), ui.sparklineNorm)
        assertEquals(meterThresholdFrac(span), ui.sparklineThresholdFrac)
    }

    @Test
    fun armedNonVolumeSparklineWithoutThresholdDrawsNothing() = runTest {
        val ui = ui(
            state(
                sentinel = SentinelState.ARMED,
                sparkline = listOf(80.0),
                sparklineSignal = SignalKind.HEART_RATE,
                meter = null, // e.g. HR availability just went OFF_BODY
            ),
        )
        assertTrue(ui.sparklineNorm.isEmpty())
    }

    @Test
    fun armedVolumeSparklineKeepsTheLoudnessScale() = runTest {
        // regression pin: VOLUME branch must be byte-identical to pre-v0.3.1 behavior
        val ui = ui(
            state(
                sentinel = SentinelState.ARMED,
                mode = Mode.STANDARD,
                sparkline = listOf(2.0, 8.0),
                sparklineSignal = SignalKind.VOLUME,
                meter = null,
            ),
        )
        val th = displayThresholdDbFor(Mode.STANDARD)
        assertEquals(normalizeSparkline(listOf(2.0, 8.0), th), ui.sparklineNorm)
        assertEquals(sparklineThresholdFrac(th), ui.sparklineThresholdFrac)
    }

    @Test
    fun offSparklineComesFromThePreviewSeriesOnThePerSignalScale() = runTest {
        // VOLUME preview with threshold -24: MeterSpan(12, 6) -> window [-36, -18].
        val ui = ui(
            state(sentinel = SentinelState.DISARMED),
            previewMeter = MeterReading(SignalKind.VOLUME, -30.0, -24.0, false),
            previewSparkline = listOf(-36.0, -30.0, -24.0, -18.0),
        )
        assertEquals(4, ui.sparklineNorm.size)
        assertEquals(0.0f, ui.sparklineNorm[0], 0.01f)
        assertEquals(0.333f, ui.sparklineNorm[1], 0.01f)
        assertEquals(0.667f, ui.sparklineNorm[2], 0.01f)
        assertEquals(1.0f, ui.sparklineNorm[3], 0.01f)
        assertEquals(0.667f, ui.sparklineThresholdFrac, 0.01f) // 12 / (12+6)
    }

    @Test
    fun offSparklineWithoutAThresholdDrawsNothing() = runTest {
        // No baseline yet -> no threshold -> no honest scale exists. An empty line is honest; an
        // auto-normalized one would be fabricated shape.
        val ui = ui(
            state(sentinel = SentinelState.DISARMED),
            previewMeter = MeterReading(SignalKind.VOLUME, -30.0, null, false),
            previewSparkline = listOf(-30.0, -29.0),
        )
        assertTrue(ui.sparklineNorm.isEmpty())
    }

    @Test
    fun offSparklineWithNoPreviewMeterDrawsNothing() = runTest {
        val ui = ui(state(sentinel = SentinelState.DISARMED), previewSparkline = listOf(1.0, 2.0))
        assertTrue(ui.sparklineNorm.isEmpty())
    }

    // --- v0.2.4 (Addendum 2): center display switcher -------------------------------------------

    @Test
    fun centerDisplayDefaultsToSparkline() = runTest {
        assertEquals(CenterDisplay.SPARKLINE, ui(state()).centerDisplay)
    }

    @Test
    fun centerDisplayDialWhenThePrefSaysDial() = runTest {
        assertEquals(CenterDisplay.DIAL, ui(state(), centerDisplayPref = { "DIAL" }).centerDisplay)
    }

    @Test
    fun corruptCenterDisplayPrefFallsBackToSparkline() {
        assertEquals(CenterDisplay.SPARKLINE, parseCenterDisplay("banana"))
        assertEquals(CenterDisplay.SPARKLINE, parseCenterDisplay(""))
        assertEquals(CenterDisplay.DIAL, parseCenterDisplay(" dial "))
        assertEquals(CenterDisplay.SPARKLINE, parseCenterDisplay("sparkline"))
    }

    // --- Wave C Task 7: signed-in state -----------------------------------------------------------

    @Test
    fun glanceUiDefaultsToNotSignedIn() = runTest {
        assertFalse(ui(state()).signedIn)
    }

    @Test
    fun glanceUiReflectsAccountBusWhenSignedIn() = runTest {
        assertTrue(ui(state(), accountSignedIn = true).signedIn)
    }

    // --- Wave C Task 13: retro-capture availability passthrough ------------------------------------

    @Test
    fun glanceUiDefaultsToNoRetroCaptureAvailable() = runTest {
        assertEquals(0.0, ui(state()).retroCaptureAvailableSeconds)
    }

    @Test
    fun glanceUiPassesThroughRetroCaptureAvailableSecondsFromControllerState() = runTest {
        // Straight passthrough (Task 10's ControllerState.retroCaptureAvailableSeconds already
        // rides the existing `state` the combine block carries -- no new bus/combine input needed,
        // unlike signedIn's separate AccountBus) -- pinned so the field can't silently regress to
        // its 0.0 default if a future refactor drops the wiring.
        assertEquals(87.5, ui(state(retroCaptureAvailableSeconds = 87.5)).retroCaptureAvailableSeconds)
    }
}
