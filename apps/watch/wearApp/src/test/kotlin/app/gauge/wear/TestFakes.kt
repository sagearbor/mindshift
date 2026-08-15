package app.gauge.wear

import app.gauge.shared.signals.SignalAvailability
import app.gauge.wear.control.MicSource
import app.gauge.wear.control.ScalarSource
import kotlin.math.PI
import kotlin.math.sin

/**
 * Shared test fakes for [MicSource]/[ScalarSource] consumers ([app.gauge.wear.control.
 * SentinelControllerTest], [app.gauge.wear.ui.PreviewMeterEngineTest]) — kept in one place (rather
 * than duplicated per test file) since both exercise the same "one second of PCM16 @ 16kHz per
 * window" and "scalar source that may have no reading yet" contracts.
 */

/** One second of PCM16 audio at the 16kHz sample rate SentinelController/PreviewMeterEngine both
 * assume (1 window == 1 second, matching SentinelStateMachine's window-counted seconds). */
fun tone(a: Double, n: Int = 16000) =
    ShortArray(n) { (sin(2 * PI * 150 * it / 16000.0) * a * 32767).toInt().toShort() }

fun silence(n: Int = 16000) = ShortArray(n)

/** Feeds a scripted sequence of windows to [MicSource.readWindow]; `null` simulates mic loss. Once
 * the scripted queue is exhausted, keeps returning [fillWith] (default: silence) so tests don't
 * have to enumerate every tick explicitly. */
class ScriptedMic(initial: List<ShortArray?> = emptyList(), private val fillWith: ShortArray? = silence()) : MicSource {
    private val queue = ArrayDeque(initial)
    override fun readWindow(): ShortArray? = if (queue.isNotEmpty()) queue.removeFirst() else fillWith
    fun enqueue(window: ShortArray?) = queue.addLast(window)
    fun enqueueMany(count: Int, window: ShortArray?) = repeat(count) { queue.addLast(window) }
}

/** Trivial [ScalarSource] fake: `v` is settable/nullable so tests can simulate "no reading yet";
 * `avail` is settable so tests can simulate off-body/acquiring/unavailable independently of
 * whether a reading exists (the P4-10 case is precisely "registered, no reading, off-body"). */
class FakeScalar(
    var v: Double?,
    var avail: SignalAvailability = SignalAvailability.UNKNOWN,
) : ScalarSource {
    override fun latest(): Double? = v
    override fun availability(): SignalAvailability = avail
}
