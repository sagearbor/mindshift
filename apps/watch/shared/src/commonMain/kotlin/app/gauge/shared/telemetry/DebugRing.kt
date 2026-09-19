package app.gauge.shared.telemetry

import app.gauge.shared.TelemetryEventOut
import kotlinx.serialization.json.JsonObject

/**
 * Fixed-capacity, thread-safe ring buffer of telemetry events, kept entirely
 * in memory on the device. Oldest events fall off once `capacity` is
 * exceeded. This is KMP common code, so it has no clock of its own —
 * callers (Android side) supply an ISO-formatted timestamp string per event.
 */
class DebugRing(capacity: Int = 200) {
    // Non-positive capacity is nonsensical for a ring buffer, but this class
    // exists to report crashes, so it must never itself crash on bad input —
    // clamp rather than propagate an IllegalArgumentException/NoSuchElementException.
    private val capacity: Int = capacity.coerceAtLeast(0)
    private val lock = Any()
    private val events = ArrayDeque<TelemetryEventOut>(this.capacity)

    fun add(level: String, tag: String, message: String, ts: String, stack: String? = null, data: JsonObject? = null) {
        if (capacity <= 0) return
        synchronized(lock) {
            if (events.size >= capacity) {
                events.removeFirst()
            }
            events.addLast(
                TelemetryEventOut(level = level, tag = tag, message = message, stack = stack, ts = ts, data = data),
            )
        }
    }

    fun snapshot(): List<TelemetryEventOut> = synchronized(lock) { events.toList() }

    fun clear() {
        synchronized(lock) { events.clear() }
    }
}
