package app.gauge.wear.haptics

import app.gauge.shared.NudgeHapticSchedule
import app.gauge.shared.NudgeVocabulary

/**
 * Device-tuned system effects channel A's cues map onto. SDK-agnostic on purpose (no android
 * import in this file): the mapping stays plain-JVM testable, and [app.gauge.wear.haptics.
 * RealVibratorPort] owns the translation to `VibrationEffect` constants.
 */
enum class PredefinedEffect { CLICK, HEAVY_CLICK }

/**
 * One playable haptic cue (v0.2.4). The v0.1–v0.2.3 raw waveforms (40ms taps at amplitude
 * 120–180) were tuned to no actuator at all and proved barely perceptible on the real Pixel
 * Watch — the fix is to prefer the effects the OEM tuned to the actual hardware:
 *
 * - [Predefined]: one system-tuned effect (EFFECT_CLICK / EFFECT_HEAVY_CLICK).
 * - [ComposedClicks]: N full-scale primitive clicks [ComposedClicks.gapMs] apart, composed as ONE
 *   platform effect (VibrationEffect.Composition) so the platform, not a Handler loop, owns the
 *   intra-pattern timing.
 * - [Waveform]: a raw createWaveform pattern — channel B's native form, and every cue's fallback
 *   when the device can't honor predefined/composed effects. Lists (not arrays) so data-class
 *   equality works in tests.
 */
sealed interface HapticCue {
    data class Predefined(val effect: PredefinedEffect) : HapticCue
    data class ComposedClicks(val count: Int, val gapMs: Long) : HapticCue
    data class Waveform(val timingsMs: List<Long>, val amplitudes: List<Int>) : HapticCue
}

/**
 * THE single home of every haptic pattern constant in this app (v0.2.4 rule — rationale lives
 * next to the numbers, and no other file may define pattern timings/amplitudes):
 *
 * - Channel A ("you" — the wearer's own escalation) is crisp clicks, and its SHAPE per level is
 *   PRD §6's schedule as encoded in the shared [NudgeHapticSchedule] (Track 1, 2026-08-24): L1 a
 *   single soft click, L2 a DOUBLE click (two composed clicks — was one heavy click before Track
 *   1; the PRD's "double pulse" is a count, and one heavy tap is not distinguishable from one
 *   soft tap through a band), L3 three composed clicks whose fallback waveform RAMPS
 *   ([NudgeHapticSchedule.ESCALATING_RAMP]) — "continuous escalating". Tap counts come from
 *   [NudgeHapticSchedule.planFor]'s `pulses`, so this file can't drift from the schedule the
 *   tests pin; the REPEAT cadence per level (every 2 min / 1 min / 10 s) is owned by
 *   [HapticDirector]'s reminder, not by these one-shot patterns. Crispness is the channel's
 *   identity.
 * - Channel B ("partner/paired cue") is long smooth buzzes (250–400ms) and deliberately NEVER
 *   uses predefined click effects — the two channels must stay distinguishable by feel alone.
 * - Every gap between taps in any multi-tap pattern is [MIN_GAP_MS] (170ms) — the same
 *   never-merge silence floor [PulseEngine] enforces for the pulse train, promoted to a
 *   pattern-wide rule in v0.2.4 (the old 80/150ms gaps could smear into one long buzz).
 * - Fallback waveforms (played when a device can't honor predefined/composed effects, and what
 *   every JVM test's FakeVibratorPort sees by default) are LONG (75–100ms) FULL-amplitude (255)
 *   taps: a generic linear actuator needs roughly that to be clearly felt through a watch band —
 *   this is the direct fix for the "raw 40ms/120–180 is imperceptible" device finding.
 * - The pulse train's bands ([pulseBandFor]) keep their proportional-amplitude scaling but the
 *   floor band rises from (40ms,120) — imperceptible — to (50ms,180). The max band's 80ms
 *   duration is deliberately unchanged: 80 + 170 = 250ms is the fastest "Pulse speed" preference,
 *   and PulseEngine.effectiveIntervalMs clamps against exactly that sum.
 */
object HapticPatterns {

    /** Never-merge silence floor between taps, pattern-wide. Mirrors [PulseEngine]'s own
     * SILENCE_FLOOR_MS (170) — kept as two constants because PulseEngine's is private and governs
     * inter-pulse gating, while this one shapes intra-pattern gaps; both are pinned by tests. */
    const val MIN_GAP_MS = 170L

    /** The device-tuned cue for a channel/level, or null for level 0 / out-of-range / unknown
     * channel (level 0 de-escalation stays silent — see [HapticDirector.onNudge]). */
    fun cue(channel: String, level: Int): HapticCue? {
        if (level < 1 || level > 3) return null
        return when (channel) {
            "A" -> when (level) {
                // One pulse: the OEM-tuned soft click (PRD §6 "single soft pulse").
                1 -> HapticCue.Predefined(PredefinedEffect.CLICK)
                // Two / three pulses: N full-scale clicks composed as one platform effect, N from
                // the shared schedule (2 for DOUBLE, 3 for ESCALATING).
                else -> HapticCue.ComposedClicks(count = NudgeHapticSchedule.planFor(level).pulses, gapMs = MIN_GAP_MS)
            }
            "B" -> when (level) {
                1 -> HapticCue.Waveform(listOf(0L, 250L), listOf(0, 220))
                2 -> HapticCue.Waveform(listOf(0L, 250L, MIN_GAP_MS, 250L), listOf(0, 220, 0, 220))
                else -> HapticCue.Waveform(
                    listOf(0L, 400L, MIN_GAP_MS, 400L, MIN_GAP_MS, 400L),
                    listOf(0, 255, 0, 255, 0, 255),
                )
            }
            else -> null
        }
    }

    /** The raw-waveform fallback for a channel/level (see class KDoc for why these are long and
     * full-amplitude), or null for the same invalid inputs as [cue]. For channel B the cue IS a
     * waveform, so the fallback is the cue itself — one source of truth, no drift. */
    fun waveformFallback(channel: String, level: Int): HapticCue.Waveform? {
        if (level < 1 || level > 3) return null
        return when (channel) {
            // Channel A IS the Heated family, so its fallback is READ from the shared vocabulary
            // rather than restated here. It used to be restated, and on 2026-09-06 that drifted:
            // the vocabulary moved H's rising ramp into tap LENGTH (60 -> 110 -> 200 ms) because a
            // phone cannot play amplitude, and this file kept three equal 100 ms taps. Delegating
            // is the only way the wrist and the phone can be the same gesture.
            "A" -> NudgeVocabulary.hapticFor("H", level)?.let { HapticCue.Waveform(it.timingsMs, it.amplitudes) }
            "B" -> cue("B", level) as? HapticCue.Waveform
            else -> null
        }
    }

    /**
     * The cue for a nudge that carries a VOCABULARY code (nudge_vocabulary.json, mirrored by
     * [app.gauge.shared.NudgeVocabulary]) — so the wrist says WHICH behaviour, not just how bad:
     * a cut-in is `• —`, hogging is a slow `— — —`, a heart-rate spike is a lub-dub.
     *
     * **H** (Heated) deliberately keeps the shipped channel cues rather than a raw waveform: H IS
     * channel A's ladder, and its predefined/composed clicks are OEM-tuned to the actual actuator
     * (the v0.2.4 device finding) — a raw waveform would feel worse, not better. A null or
     * unrecognised code falls back to [cue] for the same reason: never lose a nudge over a label.
     */
    fun cueFor(channel: String, level: Int, code: String?): HapticCue? {
        if (level < 1 || level > 3) return null
        if (isSilentByContract(code)) return null
        val wave = vocabularyWave(channel, level, code) ?: return cue(channel, level)
        return HapticCue.Waveform(wave.timingsMs, wave.amplitudes)
    }

    /** The raw-waveform fallback matching [cueFor]. A vocabulary cue IS a waveform, so it is its
     * own fallback — one source of truth, no drift (same rule as channel B in [waveformFallback]). */
    fun waveformFallbackFor(channel: String, level: Int, code: String?): HapticCue.Waveform? {
        if (level < 1 || level > 3) return null
        if (isSilentByContract(code)) return null
        val wave = vocabularyWave(channel, level, code) ?: return waveformFallback(channel, level)
        return HapticCue.Waveform(wave.timingsMs, wave.amplitudes)
    }

    /** A code the vocabulary says must NEVER buzz (🧘 K: "buzzing someone to tell them nothing
     * happened is the definition of a nag"). Distinct from "this code has no waveform override",
     * which falls through to the channel cue — silence-by-contract must stay silent, and
     * [cueFor] is public enough that a future caller will eventually hand it a K. */
    private fun isSilentByContract(code: String?): Boolean {
        val entry = code?.let { NudgeVocabulary.forCode(it) } ?: return false
        return entry.haptic == null
    }

    /** The vocabulary's own waveform for this cue, or null when the channel cue should be used
     * instead. Null for: no code; **H**, which IS channel A's OEM-tuned ladder (see [cueFor]);
     * a code with no cue (K) or an unknown one; and — the subtle case — a code arriving on the
     * WRONG lane. The two channels must stay distinguishable by feel alone (class KDoc), so a
     * channel-A code relayed on channel B plays B's smooth buzz, not A's crisp rhythm. Each
     * code's home lane is B for the watch-only ones (P: heart rate) and A for the rest. */
    private fun vocabularyWave(channel: String, level: Int, code: String?): app.gauge.shared.HapticWaveform? {
        if (code == null || code == "H") return null
        val entry = NudgeVocabulary.forCode(code) ?: return null
        val homeChannel = if (entry.watchOnly) "B" else "A"
        if (homeChannel != channel) return null
        return NudgeVocabulary.hapticFor(code, level)
    }

    /**
     * Proportional pulse-train band: dB over the *trigger threshold* (not over baseline) →
     * (duration, amplitude). Owned here (v0.2.4) so every pattern constant lives in one file;
     * [PulseEngine.bandFor] delegates. Floor band raised to a perceptible minimum — see class
     * KDoc; scaling and the 80ms max-band duration are unchanged.
     */
    fun pulseBandFor(dbOverThreshold: Double): Pulse = when {
        dbOverThreshold < 3.0 -> Pulse(50L, 180)
        dbOverThreshold < 6.0 -> Pulse(60L, 205)
        dbOverThreshold < 9.0 -> Pulse(70L, 230)
        else -> Pulse(80L, 255)
    }

    /** The ARMED shout-tap (Task 2): always the floor band — a deliberate pre-episode nicety,
     * never proportional (proportionality is the STREAMING pulse train's job). */
    val ARMED_SHOUT_TAP: Pulse get() = pulseBandFor(0.0)
}
