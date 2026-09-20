package app.gauge.shared

/**
 * Hold-N hysteresis on the loudness ladder (2026-09-20).
 *
 * The ladder used to climb off a SINGLE 1 s window that cleared +6 dB over the wearer's baseline.
 * Measured over 32 h of real conversation (AMI + SBCSAE, `scripts/heat_map.py`), that instantaneous
 * crossing is most of the dose: the median AMI meeting delivered 97 buzzes/hour and the median
 * SBCSAE recording 80, in conversations where nobody is angry. Requiring the first rung to HOLD for
 * three consecutive windows before the ladder may climb halves it (46.4 and 36.0) with no new
 * signal, because one loud window is a laugh, a cough or a door, and three in a row is a raised
 * voice.
 *
 * Seconds, not windows: a turn-driven caller observes once per TURN rather than once per 1 s
 * window, so an observation says how much audio it covers. Every observation is worth at least one
 * window, which is what makes [HEAT_HOLD_S] <= 1 byte-identical to the pre-hold ladder on every
 * path.
 *
 * Reference implementation and the measurement: `sustained()` in scripts/conversation_audit.py.
 * Pinned across the three runtimes by `server/tests/fixtures/policy_vectors/nudge_policy.json`
 * (schema v2's `config.hold_s`).
 */
const val HEAT_HOLD_S: Double = 3.0

/** The window the watch's sentinel runs on: SentinelController MAX_SLICE_BYTES 32000 / 2 / 16 kHz. */
const val HEAT_WINDOW_S: Double = 1.0

/**
 * "Has the ladder's first rung held long enough to escalate?" — a run of consecutive qualifying
 * observations, reset by the first that does not qualify. `holdS` of 0.0 or 1.0 opens the gate on
 * the first qualifying observation, i.e. the pre-2026-09-20 ladder.
 */
class LoudnessHold(private val holdS: Double = HEAT_HOLD_S, private val windowS: Double = HEAT_WINDOW_S) {
    private var runS: Double = 0.0

    /** Seconds of consecutive qualifying observation so far. */
    fun runSeconds(): Double = runS

    private fun nextRun(loud: Boolean, observedS: Double): Double =
        if (loud) runS + maxOf(observedS, windowS) else 0.0

    /** Would this observation open the gate? Pure — nothing is advanced. */
    fun peek(loud: Boolean, observedS: Double = HEAT_WINDOW_S): Boolean = nextRun(loud, observedS) >= holdS

    /** Advance the run and report whether this observation may escalate. */
    fun observe(loud: Boolean, observedS: Double = HEAT_WINDOW_S): Boolean {
        runS = nextRun(loud, observedS)
        return runS >= holdS
    }
}

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
 * On top of that, hold-N hysteresis (2026-09-20): a window may only raise the level once the first
 * rung has held for `holdS` seconds ([LoudnessHold]). A gated-out loud window reads as E = 0 for
 * that call, so it neither escalates nor refreshes the sustain clock — the decay runs exactly as it
 * would have on a quiet window, which is precisely how the measured replay models it
 * (`conversation_audit.replay`'s `gate`).
 *
 * `onLocalLoudness` returns the new level only when it changes; `null`
 * otherwise.
 */
class NudgeStateMachine(
    private val cooldownS: Double = 20.0,
    holdS: Double = HEAT_HOLD_S,
) {
    private var level: Int = 0
    private var lastQualifyingT: Double = 0.0
    private val hold = LoudnessHold(holdS)

    /**
     * One loudness observation. [observedS] is how many seconds of audio it covers — the watch's
     * sentinel calls once per 1 s window and leaves the default.
     */
    fun onLocalLoudness(dbOverBaseline: Double, t: Double, observedS: Double = HEAT_WINDOW_S): Int? {
        val raw = levelFor(dbOverBaseline)
        val mayEscalate = hold.observe(raw >= 1, observedS)
        val e = if (mayEscalate) raw else 0
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
