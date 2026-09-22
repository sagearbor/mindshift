package app.gauge.shared.telemetry

/**
 * The smallest possible mutual-exclusion primitive, because `synchronized` is a
 * JVM intrinsic and this module now also compiles for watchOS (2026-09-22).
 *
 * Deliberately `expect`/`actual` rather than "drop the lock on Apple": the only
 * caller is [DebugRing], whose whole job is to survive a crash and report what
 * led to it. A telemetry buffer that corrupts itself under concurrent access is
 * worse than no telemetry, and "the watch probably only touches it from one
 * thread" is an assumption nobody would be able to check when it mattered.
 *
 * Kept `internal`: this is plumbing for DebugRing, not a general-purpose lock
 * for the rest of the module to reach for.
 */
internal expect class RingLock() {
    fun <T> withLock(block: () -> T): T
}
