package app.gauge.shared.signals

/**
 * SDK-agnostic availability of an on-device signal source.
 *
 * Exists because a source can be perfectly registered and still deliver nothing: the v0.2.2 device
 * session had HR registering cleanly while Health Services reported UNAVAILABLE_DEVICE_OFF_BODY
 * (watch worn loose), and the UI could only say "(no reading)" — true, but useless. Availability is
 * the *why*, and it is never a substitute for a reading (see [statusText]).
 */
enum class SignalAvailability { UNKNOWN, ACQUIRING, AVAILABLE, OFF_BODY, UNAVAILABLE }

/**
 * Maps a Health Services `Availability.toString()` name onto [SignalAvailability]. Takes a plain
 * `String` (not the SDK type) so this mapping stays KMP-pure and unit-testable with no androidx
 * dependency — `HrSource` passes `availability.toString()`.
 *
 * OFF_BODY is matched first: `UNAVAILABLE_DEVICE_OFF_BODY` contains `UNAVAILABLE`, and off-body is
 * the only one of the two the wearer can actually fix. Anything unrecognized degrades to
 * [SignalAvailability.UNKNOWN] — a future SDK state must render as "we don't know", never as a
 * confidently wrong status.
 */
fun signalAvailabilityFrom(raw: String): SignalAvailability {
    val name = raw.trim().uppercase()
    return when {
        name.contains("OFF_BODY") -> SignalAvailability.OFF_BODY
        name == "AVAILABLE" -> SignalAvailability.AVAILABLE
        name == "ACQUIRING" -> SignalAvailability.ACQUIRING
        name == "UNAVAILABLE" -> SignalAvailability.UNAVAILABLE
        else -> SignalAvailability.UNKNOWN
    }
}

/**
 * The short line shown under the live meter, or `null` when there is nothing honest to add.
 *
 * `AVAILABLE` -> null (the reading itself is the message). `UNKNOWN` -> null (we genuinely don't
 * know yet; the signal chip's existing "(no reading)" suffix already says the honest thing). These
 * strings are status only and never contain a number — see this file's own test.
 */
fun SignalAvailability.statusText(): String? = when (this) {
    SignalAvailability.AVAILABLE, SignalAvailability.UNKNOWN -> null
    SignalAvailability.ACQUIRING -> "acquiring…"
    SignalAvailability.OFF_BODY -> "off-body — wear snug"
    SignalAvailability.UNAVAILABLE -> "unavailable"
}
