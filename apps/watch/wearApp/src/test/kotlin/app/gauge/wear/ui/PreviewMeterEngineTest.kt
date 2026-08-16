package app.gauge.wear.ui

import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.FakeScalar
import app.gauge.wear.ScriptedMic
import app.gauge.wear.control.MeterReading
import app.gauge.wear.control.MicSource
import app.gauge.wear.control.ScalarSource
import app.gauge.wear.tone
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Task 12: [PreviewMeterEngine] is the live meter while the sentinel is OFF — same signal-reading
 * machinery as [app.gauge.wear.control.SentinelController]'s meter, but driven off its own tracker
 * instances and gated so it never contends with the service for the mic (see [PreviewMeterEngine]'s
 * own KDoc on the never-two-readers rule).
 */
class PreviewMeterEngineTest {
    @Test
    fun publishesVolumeReadingFromMic() {
        var last: MeterReading? = MeterReading(SignalKind.VOLUME, 0.0, null, false)
        val e = PreviewMeterEngine(
            mic = ScriptedMic(List(6) { tone(0.05) }),
            hr = null,
            accel = null,
            selectedSignal = { SignalKind.VOLUME },
            publish = { last = it },
            isServiceActive = { false },
        )
        repeat(6) { e.step() }
        assertEquals(SignalKind.VOLUME, last!!.signal)
        assertTrue(last!!.value > -40.0 && last!!.value < -20.0)
    }

    @Test
    fun yieldsToActiveService() {
        var last: MeterReading? = MeterReading(SignalKind.VOLUME, 0.0, null, false)
        var micReads = 0
        val mic = object : MicSource {
            override fun readWindow(): ShortArray {
                micReads++
                return tone(0.05)
            }
        }
        val e = PreviewMeterEngine(mic, null, null, { SignalKind.VOLUME }, { last = it }, isServiceActive = { true })
        e.step()
        assertEquals(0, micReads)
        assertNull(last)
    }

    @Test
    fun hrPreviewUsesScalarSource() {
        var last: MeterReading? = null
        val e = PreviewMeterEngine(
            ScriptedMic(emptyList()),
            hr = FakeScalar(70.0),
            accel = null,
            selectedSignal = { SignalKind.HEART_RATE },
            publish = { last = it },
            isServiceActive = { false },
        )
        repeat(6) { e.step() }
        assertEquals(70.0, last!!.value)
    }

    // --- P4-6 (Issue B): all signals — not just mic-backed ones — must preview while OFF ---------

    @Test
    fun movementPreviewUsesScalarSource() {
        var last: MeterReading? = null
        val e = PreviewMeterEngine(
            ScriptedMic(emptyList()),
            hr = null,
            accel = FakeScalar(3.5),
            selectedSignal = { SignalKind.MOVEMENT },
            publish = { last = it },
            isServiceActive = { false },
        )
        repeat(6) { e.step() }
        assertEquals(SignalKind.MOVEMENT, last!!.signal)
        assertEquals(3.5, last!!.value)
    }

    @Test
    fun hrPreviewPublishesNullWhenSourceAbsent() {
        var last: MeterReading? = MeterReading(SignalKind.HEART_RATE, 0.0, null, false)
        val e = PreviewMeterEngine(
            ScriptedMic(emptyList()),
            hr = null,
            accel = null,
            selectedSignal = { SignalKind.HEART_RATE },
            publish = { last = it },
            isServiceActive = { false },
        )
        e.step()
        assertNull(last)
    }

    @Test
    fun movementPreviewPublishesNullWhenSourceAbsent() {
        var last: MeterReading? = MeterReading(SignalKind.MOVEMENT, 0.0, null, false)
        val e = PreviewMeterEngine(
            ScriptedMic(emptyList()),
            hr = null,
            accel = null,
            selectedSignal = { SignalKind.MOVEMENT },
            publish = { last = it },
            isServiceActive = { false },
        )
        e.step()
        assertNull(last)
    }

    // --- P4-8: only the selected signal's source is ever acquired -------------------------------

    private class Acquisitions {
        val acquired = mutableListOf<SignalKind>()
        val released = mutableListOf<SignalKind>()
    }

    private fun engine(
        acq: Acquisitions,
        selected: () -> SignalKind,
        active: () -> Boolean = { false },
        mic: MicSource = ScriptedMic(fillWith = tone(0.05)),
        hr: ScalarSource? = FakeScalar(70.0),
        accel: ScalarSource? = FakeScalar(3.5),
        publish: (MeterReading?) -> Unit = {},
        publishAvailability: (SignalAvailability) -> Unit = {},
        publishSparkline: (List<Double>) -> Unit = {},
    ) = PreviewMeterEngine(
        mic = mic,
        hr = hr,
        accel = accel,
        selectedSignal = selected,
        publish = publish,
        isServiceActive = active,
        onAcquire = { acq.acquired.add(it) },
        onRelease = { acq.released.add(it) },
        publishAvailability = publishAvailability,
        publishSparkline = publishSparkline,
    )

    @Test
    fun acquiresOnlyTheSelectedSignalsSource() {
        val acq = Acquisitions()
        val e = engine(acq, selected = { SignalKind.VOLUME })
        repeat(3) { e.step() }
        assertEquals(listOf(SignalKind.VOLUME), acq.acquired) // once, not once per step
        assertEquals(emptyList(), acq.released)
    }

    @Test
    fun heartRateSelectionNeverTouchesTheMic() {
        val acq = Acquisitions()
        var micReads = 0
        val mic = object : MicSource {
            override fun readWindow(): ShortArray {
                micReads++
                return tone(0.05)
            }
        }
        val e = engine(acq, selected = { SignalKind.HEART_RATE }, mic = mic)
        repeat(5) { e.step() }
        assertEquals(0, micReads)
        assertEquals(listOf(SignalKind.HEART_RATE), acq.acquired)
    }

    @Test
    fun switchingSelectionReleasesTheOldSourceAndAcquiresTheNew() {
        val acq = Acquisitions()
        var selected = SignalKind.VOLUME
        val e = engine(acq, selected = { selected })
        e.step()
        selected = SignalKind.HEART_RATE
        e.step()
        assertEquals(listOf(SignalKind.VOLUME, SignalKind.HEART_RATE), acq.acquired)
        assertEquals(listOf(SignalKind.VOLUME), acq.released)
    }

    @Test
    fun everyAcquireIsPairedWithExactlyOneReleaseAcrossManySwitches() {
        val acq = Acquisitions()
        var selected = SignalKind.VOLUME
        val e = engine(acq, selected = { selected })
        for (kind in listOf(SignalKind.VOLUME, SignalKind.MOVEMENT, SignalKind.HEART_RATE, SignalKind.SPEAKING_RATE, SignalKind.VOLUME)) {
            selected = kind
            e.step()
            e.step() // a second step on the same selection must not re-acquire
        }
        assertEquals(
            listOf(SignalKind.VOLUME, SignalKind.MOVEMENT, SignalKind.HEART_RATE, SignalKind.SPEAKING_RATE, SignalKind.VOLUME),
            acq.acquired,
        )
        // Every acquire but the currently-held one has a matching release, in order.
        assertEquals(acq.acquired.dropLast(1), acq.released)
    }

    @Test
    fun activeServiceReleasesTheHeldSignalAndAcquiresNothing() {
        val acq = Acquisitions()
        var active = false
        val e = engine(acq, selected = { SignalKind.MOVEMENT }, active = { active })
        e.step()
        assertEquals(listOf(SignalKind.MOVEMENT), acq.acquired)
        active = true
        e.step()
        assertEquals(listOf(SignalKind.MOVEMENT), acq.released)
        e.step() // still active: no redundant release, no acquire
        assertEquals(listOf(SignalKind.MOVEMENT), acq.released)
        assertEquals(listOf(SignalKind.MOVEMENT), acq.acquired)
    }

    @Test
    fun reacquiresTheCurrentSelectionOnceTheServiceGoesInactiveAgain() {
        val acq = Acquisitions()
        var active = true
        val e = engine(acq, selected = { SignalKind.HEART_RATE }, active = { active })
        e.step() // active from the start: nothing was held, so nothing to release
        assertEquals(emptyList(), acq.released)
        assertEquals(emptyList(), acq.acquired)
        active = false
        e.step()
        assertEquals(listOf(SignalKind.HEART_RATE), acq.acquired)
        e.step()
        assertEquals(listOf(SignalKind.HEART_RATE), acq.acquired) // no redundant acquire
    }

    @Test
    fun closeReleasesTheHeldSignalExactlyOnce() {
        val acq = Acquisitions()
        val e = engine(acq, selected = { SignalKind.VOLUME })
        e.step()
        e.close()
        e.close()
        assertEquals(listOf(SignalKind.VOLUME), acq.released)
    }

    // --- P4-10 in the preview: availability published even with no reading ----------------------

    @Test
    fun publishesHrAvailabilityEvenWhenThereIsNoReading() {
        val acq = Acquisitions()
        var lastMeter: MeterReading? = MeterReading(SignalKind.HEART_RATE, 0.0, null, false)
        var lastAvail = SignalAvailability.AVAILABLE
        val e = engine(
            acq,
            selected = { SignalKind.HEART_RATE },
            hr = FakeScalar(null, SignalAvailability.OFF_BODY),
            publish = { lastMeter = it },
            publishAvailability = { lastAvail = it },
        )
        e.step()
        assertNull(lastMeter)
        assertEquals(SignalAvailability.OFF_BODY, lastAvail)
    }

    @Test
    fun publishesUnknownAvailabilityForNonHrSelections() {
        val acq = Acquisitions()
        var lastAvail = SignalAvailability.OFF_BODY
        val e = engine(acq, selected = { SignalKind.VOLUME }, publishAvailability = { lastAvail = it })
        e.step()
        assertEquals(SignalAvailability.UNKNOWN, lastAvail)
    }

    @Test
    fun publishesUnknownAvailabilityWhileYieldingToTheService() {
        val acq = Acquisitions()
        var lastAvail = SignalAvailability.OFF_BODY
        val e = engine(
            acq,
            selected = { SignalKind.HEART_RATE },
            active = { true },
            hr = FakeScalar(null, SignalAvailability.OFF_BODY),
            publishAvailability = { lastAvail = it },
        )
        e.step()
        assertEquals(SignalAvailability.UNKNOWN, lastAvail)
    }

    @Test
    fun throwingAvailabilitySourceDegradesToUnknownWithoutEscapingStep() {
        val acq = Acquisitions()
        var lastAvail = SignalAvailability.OFF_BODY
        val hr = object : ScalarSource {
            override fun latest(): Double = 70.0
            override fun availability(): SignalAvailability = throw IllegalStateException("sensor died")
        }
        val e = engine(acq, selected = { SignalKind.HEART_RATE }, hr = hr, publishAvailability = { lastAvail = it })
        e.step()
        assertEquals(SignalAvailability.UNKNOWN, lastAvail)
    }

    // --- v0.2.4 (Addendum 2): preview sparkline series ------------------------------------------

    @Test
    fun previewSparklineAccumulatesReadingsInOrder() {
        val acq = Acquisitions()
        var series: List<Double> = emptyList()
        val accel = FakeScalar(1.0)
        val e = engine(acq, selected = { SignalKind.MOVEMENT }, accel = accel, publishSparkline = { series = it })
        e.step()
        accel.v = 2.0
        e.step()
        accel.v = 3.0
        e.step()
        assertEquals(listOf(1.0, 2.0, 3.0), series)
    }

    @Test
    fun previewSparklineCapsAtThirtyPoints() {
        val acq = Acquisitions()
        var series: List<Double> = emptyList()
        val accel = FakeScalar(0.0)
        val e = engine(acq, selected = { SignalKind.MOVEMENT }, accel = accel, publishSparkline = { series = it })
        repeat(35) { i ->
            accel.v = i.toDouble()
            e.step()
        }
        assertEquals(30, series.size)
        assertEquals(5.0, series.first())
        assertEquals(34.0, series.last())
    }

    @Test
    fun previewSparklineClearsOnSignalSwitchNeverMixingUnits() {
        val acq = Acquisitions()
        var series: List<Double> = emptyList()
        var selected = SignalKind.MOVEMENT
        val e = engine(
            acq,
            selected = { selected },
            accel = FakeScalar(3.5),
            hr = FakeScalar(70.0),
            publishSparkline = { series = it },
        )
        e.step()
        e.step()
        assertEquals(listOf(3.5, 3.5), series)
        selected = SignalKind.HEART_RATE
        e.step()
        // bpm never appended onto a stddev series — mixed units drawn as one line would be
        // fabricated shape (ratified).
        assertEquals(listOf(70.0), series)
    }

    @Test
    fun nullReadingsLeaveTheSeriesUnchanged() {
        val acq = Acquisitions()
        var series: List<Double> = emptyList()
        val hr = FakeScalar(70.0)
        val e = engine(acq, selected = { SignalKind.HEART_RATE }, hr = hr, publishSparkline = { series = it })
        e.step()
        hr.v = null // no reading this step: an honest gap, never a fabricated 0-point
        e.step()
        assertEquals(listOf(70.0), series)
    }

    @Test
    fun yieldingToTheServicePublishesAnEmptySeries() {
        val acq = Acquisitions()
        var series: List<Double> = listOf(999.0)
        var active = false
        val e = engine(
            acq,
            selected = { SignalKind.MOVEMENT },
            accel = FakeScalar(1.0),
            active = { active },
            publishSparkline = { series = it },
        )
        e.step()
        assertEquals(listOf(1.0), series)
        active = true
        e.step() // the service's own sparkline takes over; a stale preview line must not linger
        assertEquals(emptyList(), series)
    }

    @Test
    fun closePublishesAnEmptySeries() {
        val acq = Acquisitions()
        var series: List<Double> = listOf(999.0)
        val e = engine(acq, selected = { SignalKind.MOVEMENT }, accel = FakeScalar(1.0), publishSparkline = { series = it })
        e.step()
        e.close()
        assertEquals(emptyList(), series)
    }
}
