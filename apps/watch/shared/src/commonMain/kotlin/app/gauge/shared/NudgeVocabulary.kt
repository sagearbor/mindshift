package app.gauge.shared

/**
 * The nudge VOCABULARY — what a firing behaviour is CALLED, how it LOOKS and how it FEELS.
 *
 * Owner-approved 2026-09-06. The executable contract is
 * `server/tests/fixtures/policy_vectors/nudge_vocabulary.json`, replayed here by
 * `NudgeVocabularyVectorsTest` (jvmTest) and, from the same file, by
 * `server/tests/test_nudge_vocabulary_vectors.py` and
 * `apps/mobile/__tests__/nudgeVocabulary.test.ts`. An emoji or a millisecond can only change in
 * one place.
 *
 * Deliberately separate from [NudgeStateMachine] / the server `NudgePolicy`: the policy decides
 * WHEN something fires and at what level, this decides what the wearer is shown and what the
 * wrist plays. A threshold change must never silently restyle a cue, and a restyle must never
 * move a threshold.
 *
 * Haptics are encoded once, in a form both runtimes render verbatim: [HapticWaveform.timingsMs]
 * alternates OFF, ON, OFF, ON, … starting with an initial delay (always 0) — React Native's
 * Android `Vibration.vibrate(pattern)` shape — and [HapticWaveform.amplitudes] (0 in the OFF
 * slots, 1..255 in the ON slots) is what `VibrationEffect.createWaveform` uses. RN cannot vary
 * amplitude, so the PHONE feels the rhythm only; that is why every LEVEL difference is a rhythm
 * difference and never an intensity one (Brown & Brewster: rhythm identified ~93% of the time,
 * intensity ~61%).
 *
 * Rendering note for the wear app: `HapticPatterns` stays the only file that hands a
 * `VibrationEffect` to the platform, and its channel-A cues are the H family here — see
 * `HapticPatternsTest.channelACuesAreTheHeatedVocabulary`.
 */

/** Whether a code is something to correct or something the wearer did well. */
enum class NudgePolarity { ALERT, POSITIVE }

/** The semantic colour, not a hex value — each surface maps it to its own palette. */
enum class NudgeColor { RED, GREEN, NEUTRAL }

/** One playable cue. See [NudgeVocabulary]'s file KDoc for the encoding. */
data class HapticWaveform(val timingsMs: List<Long>, val amplitudes: List<Int>) {
    /** Total vibrating time — what an amplitude-blind phone actually feels. */
    val onMs: Long get() = timingsMs.filterIndexed { i, _ -> i % 2 == 1 }.sum()
}

data class NudgeVocabularyEntry(
    /** One uppercase letter, stable forever — the compact form in reports, chart markers and
     * telemetry. */
    val code: String,
    val vector: String,
    val icon: String,
    val color: NudgeColor,
    val name: String,
    val meaning: String,
    val polarity: NudgePolarity,
    /** Detector vector names (`server/watch/vectors.py`) that raise this code; empty for the
     * positive codes, whose detectors live in the positive-nudge pass. */
    val sources: List<String>,
    /** The on-screen line, or null when the icon alone is the message. */
    val flashText: String?,
    /** Never fires live — a badge on the post-session summary (K). */
    val summaryOnly: Boolean,
    /** The phone has no sensor for it (P: heart rate). */
    val watchOnly: Boolean,
    val levels: List<Int>,
    /** Level -> waveform, or null for a code that never buzzes (K). */
    val haptic: Map<Int, HapticWaveform>?,
)

object NudgeVocabulary {

    /** Pattern-wide never-merge silence floor; mirrors the wear app's `HapticPatterns.MIN_GAP_MS`. */
    const val MIN_GAP_MS = 170L

    /** The softest cue a wrist can actually feel: 50 ms at amplitude 180. Not a style choice — it
     * is this app's own measured device finding (the v0.1–v0.2.3 raw waveforms at 40 ms /
     * amplitude 120–180 proved barely perceptible on a real Pixel Watch, and `HapticPatterns`'
     * pulse floor band was raised to exactly this). Every ON slot in the vocabulary is at or above
     * it, positives included: a cue nobody can feel is not a soft cue, it is a missing one. */
    const val MIN_ON_MS = 50L
    const val MIN_AMPLITUDE = 180

    /** At most one positive cue per this many seconds, across D/E/R together — so praise can
     * never become its own nag. */
    const val POSITIVE_CAP_S = 120.0

    /** The one vector whose intra-cue gap is deliberately under [MIN_GAP_MS]: a lub-dub only
     * reads as a heartbeat when the two beats nearly merge. */
    val HAPTIC_GAP_EXCEPTIONS = listOf("pulse")

    private fun w(timings: List<Long>, amps: List<Int>) = HapticWaveform(timings, amps)

    /** Owner order (2026-09-06) — also worst-first, which is how ties break in [codeForVectors]. */
    val ALL: List<NudgeVocabularyEntry> = listOf(
        NudgeVocabularyEntry(
            code = "H",
            vector = "heated",
            icon = "📈",
            color = NudgeColor.RED,
            name = "Heated",
            meaning = "You got loud, or your words got hot. Yelling and aggressive tone are one " +
                "family to the user (the detectors stay separate underneath).",
            polarity = NudgePolarity.ALERT,
            sources = listOf("yelling", "aggressive_tone", "activation"),
            flashText = "Take it down a notch",
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1, 2, 3),
            // A RISING ramp: the cue builds, and the LEVEL is how many taps it takes to get
            // there. This IS the shipped channel-A ladder.
            haptic = mapOf(
                1 to w(listOf(0L, 200L), listOf(0, 255)),
                2 to w(listOf(0L, 70L, 170L, 150L), listOf(0, 210, 0, 255)),
                3 to w(listOf(0L, 60L, 170L, 110L, 170L, 200L), listOf(0, 200, 0, 230, 0, 255)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "D",
            vector = "de_escalated",
            icon = "📉",
            color = NudgeColor.GREEN,
            name = "De-escalated",
            meaning = "Heat or loudness dropped a level within two turns of a spike — you pulled it back.",
            polarity = NudgePolarity.POSITIVE,
            sources = emptyList(),
            flashText = "Nice recovery",
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1),
            // A FALLING ramp — the mirror of H, and the only cue that fades.
            haptic = mapOf(
                1 to w(listOf(0L, 200L, 170L, 110L, 170L, 60L), listOf(0, 230, 0, 200, 0, 180)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "C",
            vector = "cut_in",
            icon = "✂️",
            color = NudgeColor.RED,
            name = "Cut in",
            meaning = "You started talking over them and kept going — sustained overlap, not the " +
                "brief overlap of ordinary engagement.",
            polarity = NudgePolarity.ALERT,
            sources = listOf("interrupting"),
            flashText = "Let them finish",
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1, 2, 3),
            // `• —`, `• • —`, `• • • —`: short tap(s) then one long "stop" buzz. The short taps
            // are 75 ms, not 60 — a "short" tap under the perceptibility floor turns `• —` into a
            // plain buzz.
            haptic = mapOf(
                1 to w(listOf(0L, 60L, 170L, 300L), listOf(0, 255, 0, 200)),
                2 to w(listOf(0L, 60L, 170L, 60L, 170L, 300L), listOf(0, 255, 0, 255, 0, 200)),
                3 to w(listOf(0L, 60L, 170L, 60L, 170L, 60L, 170L, 300L), listOf(0, 255, 0, 255, 0, 255, 0, 200)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "A",
            vector = "hogging",
            icon = "🎤",
            color = NudgeColor.RED,
            name = "Hogging",
            meaning = "You have been taking most of the airtime over the last two minutes.",
            polarity = NudgePolarity.ALERT,
            sources = listOf("airtime"),
            flashText = "Give them the floor",
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1, 2, 3),
            // Slow `— — —`: long, unhurried buzzes — "you're talking a lot" is not an emergency,
            // so it must not feel like H's crisp taps.
            haptic = mapOf(
                1 to w(listOf(0L, 250L, 300L, 250L), listOf(0, 180, 0, 180)),
                2 to w(listOf(0L, 250L, 300L, 250L, 300L, 250L), listOf(0, 180, 0, 180, 0, 180)),
                3 to w(listOf(0L, 250L, 300L, 250L, 300L, 250L, 300L, 250L), listOf(0, 180, 0, 180, 0, 180, 0, 180)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "E",
            vector = "listened",
            icon = "👂",
            color = NudgeColor.GREEN,
            name = "Listened",
            meaning = "You let them finish a long turn without cutting in.",
            polarity = NudgePolarity.POSITIVE,
            sources = emptyList(),
            flashText = null,
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1),
            // Soft `••` — deliberately the same cue as R: the wrist says "that was good", the
            // screen says which good thing.
            haptic = mapOf(
                1 to w(listOf(0L, 50L, 170L, 50L), listOf(0, 180, 0, 180)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "R",
            vector = "repair",
            icon = "🤝",
            color = NudgeColor.GREEN,
            name = "Repair",
            meaning = "You validated or apologised and their tone softened on the next turn.",
            polarity = NudgePolarity.POSITIVE,
            sources = emptyList(),
            flashText = null,
            summaryOnly = false,
            watchOnly = false,
            levels = listOf(1),
            haptic = mapOf(
                1 to w(listOf(0L, 50L, 170L, 50L, 170L, 50L), listOf(0, 180, 0, 180, 0, 180)),
            ),
        ),
        NudgeVocabularyEntry(
            code = "K",
            vector = "calm_streak",
            icon = "🧘",
            color = NudgeColor.GREEN,
            name = "Calm streak",
            meaning = "N minutes with no escalation at all.",
            polarity = NudgePolarity.POSITIVE,
            sources = emptyList(),
            flashText = null,
            summaryOnly = true,
            watchOnly = false,
            levels = listOf(1),
            // No cue, ever — buzzing someone to say nothing happened is a nag.
            haptic = null,
        ),
        NudgeVocabularyEntry(
            code = "P",
            vector = "pulse",
            icon = "❤️",
            color = NudgeColor.RED,
            name = "Pulse",
            meaning = "Your heart rate jumped well over your resting rate (+15 / +25 / +35 bpm).",
            polarity = NudgePolarity.ALERT,
            sources = listOf("hr_spike"),
            flashText = null,
            summaryOnly = false,
            watchOnly = true,
            levels = listOf(1, 2, 3),
            // Lub-dub: a short beat then a longer one 120 ms apart — under MIN_GAP_MS on purpose,
            // because the near-merge is the heartbeat.
            haptic = mapOf(
                1 to w(listOf(0L, 90L, 100L, 150L), listOf(0, 190, 0, 240)),
                2 to w(listOf(0L, 90L, 100L, 150L, 400L, 90L, 100L, 150L), listOf(0, 190, 0, 240, 0, 190, 0, 240)),
                3 to w(listOf(0L, 90L, 100L, 150L, 400L, 90L, 100L, 150L, 400L, 90L, 100L, 150L), listOf(0, 190, 0, 240, 0, 190, 0, 240, 0, 190, 0, 240)),
            ),
        ),
    )

    private val byCode = ALL.associateBy { it.code }
    private val byVector = ALL.associateBy { it.vector }
    private val bySource = ALL.flatMap { e -> e.sources.map { it to e } }.toMap()
    private val rank = ALL.withIndex().associate { (i, e) -> e.code to i }

    fun forCode(code: String): NudgeVocabularyEntry? = byCode[code]

    /** By vocabulary id ("heated") OR by the detector vector that raises it ("yelling", …) —
     * callers hold both kinds of name. */
    fun forVector(vector: String): NudgeVocabularyEntry? = byVector[vector] ?: bySource[vector]

    /** The icon, or null when unknown — never a fallback emoji, so a missing mapping is visible
     * instead of silently mislabelled. */
    fun iconFor(vector: String): String? = forVector(vector)?.icon

    /** The cue for a code at a level, or null when that code never buzzes (K) or the level is out
     * of range — silence is the safe failure for a haptic. Positives are unleveled: a "level 2
     * well done" is not a thing. */
    fun hapticFor(code: String, level: Int): HapticWaveform? {
        val entry = byCode[code] ?: return null
        val haptic = entry.haptic ?: return null
        return if (entry.polarity == NudgePolarity.POSITIVE) haptic[1] else haptic[level]
    }

    /** The strongest code a set of firing detector vectors maps to (a `NudgeEvent.vectors` list).
     * Ties break on the owner's order, which is worst-first. */
    fun codeForVectors(vectors: List<String>): String? =
        vectors.mapNotNull { forVector(it) }.minByOrNull { rank.getValue(it.code) }?.code
}

/**
 * The positive-nudge rate limit: at most one positive cue per [capS] across D/E/R together.
 *
 * `>=` (not `>`) so a caller ticking at exactly the cadence fires on the tick — the same
 * direction as [NudgeHapticSchedule.reminderDue].
 */
class PositiveNudgeGate(private val capS: Double = NudgeVocabulary.POSITIVE_CAP_S) {
    private var lastT: Double? = null

    /** True (and the clock resets) when this positive offer may be delivered. */
    fun admit(t: Double): Boolean {
        val last = lastT
        if (last != null && t - last < capS) return false
        lastT = t
        return true
    }

    /** Seconds until the next positive may fire; 0 when one may fire now. */
    fun waitS(t: Double): Double {
        val last = lastT ?: return 0.0
        return maxOf(0.0, capS - (t - last))
    }
}
