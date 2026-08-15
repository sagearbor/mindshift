package app.gauge.shared

/**
 * Offline, single-channel (channel A only) mirror of the server's
 * `NudgePolicy` (server/nudge_policy.py), for use when the watch/phone can't
 * reach the backend (spec §8). There is no sensitivity scaling here (no
 * subscriptions to consult locally), so thresholds map directly to levels:
 *
 *   dbOverBaseline >= 14 -> level 3
 *   dbOverBaseline >= 10 -> level 2
 *   dbOverBaseline >= 6  -> level 1
 *   otherwise            -> level 0
 *
 * Semantics mirror `NudgePolicy.on_events` exactly for a single channel fed
 * one observation per call:
 *  - If the new level (E) is higher than the current level: escalate, emit
 *    the new level, and refresh the qualifying-event clock to `t`.
 *  - Else if E equals the current level and the current level is > 0: this is
 *    a "sustained" qualifying observation — refresh the clock to `t` but
 *    don't emit (this is what stops a long, steady loud stretch from
 *    de-escalating early; see server/tests/test_nudge_policy.py's
 *    test_sustained_qualifying_event, a.k.a. the "Critical 2" fix).
 *  - Else (E is lower, or both are 0): if the current level is > 0 and more
 *    than `cooldownS` seconds (strict `>`) have elapsed since the last
 *    qualifying observation, drop exactly one level, emit it, and rebase the
 *    clock to `t`. Otherwise, no change.
 *
 * `onLocalLoudness` returns the new level only when it changes; `null`
 * otherwise.
 */
class NudgeStateMachine(private val cooldownS: Double = 20.0) {
    private var level: Int = 0
    private var lastQualifyingT: Double = 0.0

    fun onLocalLoudness(dbOverBaseline: Double, t: Double): Int? {
        val e = levelFor(dbOverBaseline)
        return when {
            e > level -> {
                level = e
                lastQualifyingT = t
                level
            }
            e == level && level > 0 -> {
                lastQualifyingT = t
                null
            }
            else -> {
                if (level > 0 && t - lastQualifyingT > cooldownS) {
                    level -= 1
                    lastQualifyingT = t
                    level
                } else {
                    null
                }
            }
        }
    }

    fun currentLevel(): Int = level

    private fun levelFor(dbOverBaseline: Double): Int = when {
        dbOverBaseline >= 14.0 -> 3
        dbOverBaseline >= 10.0 -> 2
        dbOverBaseline >= 6.0 -> 1
        else -> 0
    }
}
