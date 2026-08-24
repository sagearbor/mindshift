package app.gauge.shared

/**
 * The shape of the wrist cue for one nudge level — PRD §6 "Smartwatch Haptics", named.
 * [NONE] is level 0 (score > 70: the conversation is fine, say nothing).
 */
enum class NudgeHapticPattern { NONE, SINGLE_SOFT, DOUBLE, ESCALATING }

/**
 * Everything a haptic layer needs to render one nudge level, platform-free:
 *
 * - [pulses]: how many discrete taps one cue is made of (1 / 2 / 3).
 * - [repeatIntervalMs]: how often the cue REPEATS while the level is held — PRD §6's "every 2 min"
 *   / "every 1 min" / "continuous" — or `null` when it never repeats (level 0). A single nudge on
 *   level change was the pre-Track-1 behaviour; the repeat is what turns "you got a tap once" into
 *   "you're still in the yellow" without the server having to re-send anything.
 * - [amplitudeRamp]: per-pulse relative amplitudes (1..255), [pulses] entries. Flat 255 for the
 *   plain patterns; rising for [NudgeHapticPattern.ESCALATING] so the cue itself feels like it's
 *   building, not just repeating faster.
 */
data class NudgeHapticPlan(
    val level: Int,
    val pattern: NudgeHapticPattern,
    val pulses: Int,
    val repeatIntervalMs: Long?,
    val amplitudeRamp: List<Int>,
)

/**
 * Track 1 (2026-08-24): the ONE place PRD §6's haptic schedule is encoded, as a pure mapping both
 * the wear app and its JVM tests can call:
 *
 * ```
 *   Score > 70     -> level 0: no haptic
 *   Score 50–70    -> level 1: single soft pulse every 2 min
 *   Score 30–50    -> level 2: double pulse every 1 min
 *   Score < 30     -> level 3: continuous escalating pattern ("you're in the red")
 * ```
 *
 * Two mappings live here, deliberately separated:
 *
 * 1. [levelForScore] — the PRD speaks in a 0–100 "calm score" (higher = calmer); the watch's
 *    nudge machinery (server `NudgePolicy`, on-watch [NudgeStateMachine], the WS `nudge` frame)
 *    speaks in escalation levels 0–3 (higher = worse). The PRD's four bands ARE the four levels,
 *    so the score->level step is just band lookup. Boundary calls (the PRD's ranges overlap at
 *    50 and 30): a boundary score belongs to the CALMER band — 70 and 50 are level 1, 30 is
 *    level 2 — because a nudge should need clear evidence, and "exactly on the line" isn't it.
 *
 * 2. [planFor] — level -> pattern. This is what the wear app's `HapticPatterns` renders and what
 *    `HapticDirector` re-fires on the [NudgeHapticPlan.repeatIntervalMs] cadence.
 *
 * "Continuous" for level 3 is rendered as a repeat every [LEVEL_3_REPEAT_MS] (10 s), not a literal
 * unbroken buzz: a watch actuator running continuously drains the battery in minutes, and a cue
 * that never stops is the fastest way to teach a wearer to take the watch off. 10 s is short enough
 * that it never feels like the watch gave up, and each repeat is the rising 3-tap ramp.
 */
object NudgeHapticSchedule {
    /** PRD §6 score bands (calm score, 0..100). A score ABOVE this is level 0. */
    const val NO_HAPTIC_ABOVE_SCORE = 70
    /** Scores in [LEVEL_1_MIN_SCORE, NO_HAPTIC_ABOVE_SCORE] are level 1. */
    const val LEVEL_1_MIN_SCORE = 50
    /** Scores in [LEVEL_2_MIN_SCORE, LEVEL_1_MIN_SCORE) are level 2; below is level 3. */
    const val LEVEL_2_MIN_SCORE = 30

    /** PRD §6 cadences. */
    const val LEVEL_1_REPEAT_MS = 2 * 60_000L
    const val LEVEL_2_REPEAT_MS = 60_000L
    const val LEVEL_3_REPEAT_MS = 10_000L

    /** Rising per-pulse amplitudes for the escalating cue. The floor (200) is well above the
     * "barely perceptible through a band" threshold the wear app's v0.2.4 device finding set
     * (its fallback taps are 255), so even the first tap of the ramp is unmistakable. */
    val ESCALATING_RAMP: List<Int> = listOf(200, 230, 255)

    /** Calm score (0..100, clamped) -> nudge level 0..3. See class KDoc for the boundary rule. */
    fun levelForScore(score: Int): Int {
        val s = score.coerceIn(0, 100)
        return when {
            s > NO_HAPTIC_ABOVE_SCORE -> 0
            s >= LEVEL_1_MIN_SCORE -> 1
            s >= LEVEL_2_MIN_SCORE -> 2
            else -> 3
        }
    }

    /** Nudge level -> cue plan. Out-of-range levels clamp to 0..3 rather than throw: a haptic
     * layer must never crash the process over a bad level (Phase-3 rule). */
    fun planFor(level: Int): NudgeHapticPlan = when (level.coerceIn(0, 3)) {
        0 -> NudgeHapticPlan(0, NudgeHapticPattern.NONE, pulses = 0, repeatIntervalMs = null, amplitudeRamp = emptyList())
        1 -> NudgeHapticPlan(1, NudgeHapticPattern.SINGLE_SOFT, pulses = 1, repeatIntervalMs = LEVEL_1_REPEAT_MS, amplitudeRamp = listOf(255))
        2 -> NudgeHapticPlan(2, NudgeHapticPattern.DOUBLE, pulses = 2, repeatIntervalMs = LEVEL_2_REPEAT_MS, amplitudeRamp = listOf(255, 255))
        else -> NudgeHapticPlan(3, NudgeHapticPattern.ESCALATING, pulses = 3, repeatIntervalMs = LEVEL_3_REPEAT_MS, amplitudeRamp = ESCALATING_RAMP)
    }

    /**
     * Whether a level's cue should be re-fired now, given when it last played. `>=` (not `>`) so a
     * caller ticking at exactly the cadence fires on the tick, and a backwards-stepping clock
     * (negative elapsed) simply waits — silence is the safe failure for a reminder (mirrors the
     * wear app's ShoutTapGate fail direction). Level 0 is never due.
     */
    fun reminderDue(level: Int, lastPlayedMs: Long, nowMs: Long): Boolean {
        val interval = planFor(level).repeatIntervalMs ?: return false
        return nowMs - lastPlayedMs >= interval
    }
}
