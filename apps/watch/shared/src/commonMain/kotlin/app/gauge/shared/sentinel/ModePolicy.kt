package app.gauge.shared.sentinel

/**
 * The wearer-selected sentinel operating mode. Drives how aggressively the
 * watch buffers audio and how loud a window must be (relative to the
 * wearer's own baseline — see [SentinelDetector]) before it counts as a
 * trigger.
 */
enum class Mode { BATTERY_SAVER, STANDARD, SESSION, COMPANION }

/**
 * @property bufferSeconds how many seconds of pre-roll audio are kept ready
 *   to prepend to a capture once it starts (0 for [Mode.SESSION], which
 *   streams continuously and has no notion of pre-roll).
 * @property triggerDbOverBaseline the dB-over-baseline bar [SentinelDetector]
 *   uses to decide a window is `triggered` (unused in [Mode.SESSION], which
 *   never evaluates a trigger bar — it streams unconditionally).
 * @property continuous whether the mode holds an open session once armed
 *   (true for [Mode.SESSION] and [Mode.COMPANION]) rather than waiting for
 *   a trigger.
 * @property usesMic whether the mode reads the watch mic at all. False only
 *   for [Mode.COMPANION] (Tier B): the PHONE listens; the watch keeps the
 *   episode WebSocket open purely to receive relayed nudges and render them
 *   as haptics — no mic, no PCM, no trigger evaluation.
 */
data class ModeParams(
    val bufferSeconds: Int,
    val triggerDbOverBaseline: Double,
    val continuous: Boolean,
    val usesMic: Boolean = true,
)

/** Per-[Mode] tuning. Single source of truth for mode → parameter mapping. */
fun Mode.params(): ModeParams = when (this) {
    Mode.BATTERY_SAVER -> ModeParams(bufferSeconds = 5, triggerDbOverBaseline = 10.0, continuous = false)
    Mode.STANDARD -> ModeParams(bufferSeconds = 10, triggerDbOverBaseline = 6.0, continuous = false)
    Mode.SESSION -> ModeParams(bufferSeconds = 0, triggerDbOverBaseline = 0.0, continuous = true)
    Mode.COMPANION -> ModeParams(bufferSeconds = 0, triggerDbOverBaseline = 0.0, continuous = true, usesMic = false)
}
