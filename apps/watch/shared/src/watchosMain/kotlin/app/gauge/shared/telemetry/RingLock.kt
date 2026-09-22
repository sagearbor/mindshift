package app.gauge.shared.telemetry

import platform.Foundation.NSLock

/**
 * watchOS: NSLock, which ships with the Foundation interop Kotlin/Native
 * already provides — so this costs no new dependency. Not reentrant, which is
 * fine: [DebugRing] never calls one locked method from inside another.
 */
internal actual class RingLock actual constructor() {
    private val lock = NSLock()
    actual fun <T> withLock(block: () -> T): T {
        lock.lock()
        try {
            return block()
        } finally {
            lock.unlock()
        }
    }
}
