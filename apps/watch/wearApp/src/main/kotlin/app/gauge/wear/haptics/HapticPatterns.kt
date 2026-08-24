package app.gauge.wear.haptics

import app.gauge.shared.NudgeHapticSchedule

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
            "A" -> when (level) {
                1 -> HapticCue.Waveform(listOf(0L, 75L), listOf(0, 255))
                2 -> HapticCue.Waveform(listOf(0L, 75L, MIN_GAP_MS, 75L), listOf(0, 255, 0, 255))
                // Escalating: three 100ms taps whose amplitudes rise tap over tap (200 -> 230 ->
                // 255, from the shared schedule) so the cue itself builds — the composed-click
                // path above can't scale per primitive, so the ramp lives in the fallback only.
                else -> {
                    val ramp = NudgeHapticSchedule.planFor(3).amplitudeRamp
                    HapticCue.Waveform(
                        listOf(0L, 100L, MIN_GAP_MS, 100L, MIN_GAP_MS, 100L),
                        listOf(0, ramp[0], 0, ramp[1], 0, ramp[2]),
                    )
                }
            }
            "B" -> cue("B", level) as? HapticCue.Waveform
            else -> null
        }
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
