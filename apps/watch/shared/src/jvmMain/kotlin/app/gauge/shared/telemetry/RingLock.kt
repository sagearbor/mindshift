package app.gauge.shared.telemetry

/** JVM/Android: the monitor this class replaced, unchanged in behaviour. */
internal actual class RingLock actual constructor() {
    private val monitor = Any()
    actual fun <T> withLock(block: () -> T): T = synchronized(monitor) { block() }
}
