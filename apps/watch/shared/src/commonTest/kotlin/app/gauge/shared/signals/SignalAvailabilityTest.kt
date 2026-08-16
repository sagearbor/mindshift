package app.gauge.shared.signals

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class SignalAvailabilityTest {

    @Test
    fun mapsAvailable() {
        assertEquals(SignalAvailability.AVAILABLE, signalAvailabilityFrom("AVAILABLE"))
    }

    @Test
    fun mapsAcquiring() {
        assertEquals(SignalAvailability.ACQUIRING, signalAvailabilityFrom("ACQUIRING"))
    }

    @Test
    fun mapsPlainUnavailable() {
        assertEquals(SignalAvailability.UNAVAILABLE, signalAvailabilityFrom("UNAVAILABLE"))
    }

    @Test
    fun offBodyWinsOverThePlainUnavailablePrefix() {
        // The exact string the v0.2.2 device telemetry pull reported. A naive
        // startsWith("UNAVAILABLE") check would swallow it into UNAVAILABLE and lose the one
        // piece of information the wearer can actually act on.
        assertEquals(SignalAvailability.OFF_BODY, signalAvailabilityFrom("UNAVAILABLE_DEVICE_OFF_BODY"))
    }

    @Test
    fun unrecognizedNameIsUnknownNotAGuess() {
        assertEquals(SignalAvailability.UNKNOWN, signalAvailabilityFrom("SOME_FUTURE_SDK_STATE"))
        assertEquals(SignalAvailability.UNKNOWN, signalAvailabilityFrom(""))
    }

    @Test
    fun toleratesWhitespaceAndCaseFromToString() {
        assertEquals(SignalAvailability.ACQUIRING, signalAvailabilityFrom("  acquiring "))
        assertEquals(SignalAvailability.OFF_BODY, signalAvailabilityFrom("unavailable_device_off_body"))
    }

    @Test
    fun statusTextOffBodyTellsTheWearerWhatToDo() {
        assertEquals("off-body — wear snug", SignalAvailability.OFF_BODY.statusText())
    }

    @Test
    fun statusTextAcquiring() {
        assertEquals("acquiring…", SignalAvailability.ACQUIRING.statusText())
    }

    @Test
    fun statusTextUnavailable() {
        assertEquals("unavailable", SignalAvailability.UNAVAILABLE.statusText())
    }

    @Test
    fun statusTextIsNullWhenThereIsNothingHonestToSay() {
        assertNull(SignalAvailability.AVAILABLE.statusText())
        assertNull(SignalAvailability.UNKNOWN.statusText())
    }

    @Test
    fun noStatusTextEverContainsADigit() {
        // No fake data: a status string is a *status*, never a stand-in reading. If any future
        // edit sneaks a number into one of these, it would render in the exact spot the wearer
        // reads live values from.
        for (a in SignalAvailability.entries) {
            val text = a.statusText() ?: continue
            assertEquals(false, text.any { it.isDigit() }, "status text for $a must not contain digits: $text")
        }
    }
}
