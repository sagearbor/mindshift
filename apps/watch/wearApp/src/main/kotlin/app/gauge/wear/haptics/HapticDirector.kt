package app.gauge.wear.haptics

import app.gauge.shared.NudgeEvent
import app.gauge.shared.NudgeHapticSchedule
import app.gauge.wear.control.DiagLog
import app.gauge.wear.control.NOOP_DIAG
import app.gauge.wear.control.VibratorPort

/**
 * Not internally thread-safe: [onNudge]'s dedupe fields (`lastChannel`/`lastLevel`/
 * `lastVibrationTimeMs`) are plain vars with no synchronization. Callers that may invoke
 * [onNudge] from more than one thread concurrently (e.g. wearApp's `SentinelController`, which
 * calls it from both its own core thread and an `EpisodeWsClient.Listener` callback thread) MUST
 * synchronize their own calls externally — this class deliberately doesn't take a lock itself so
 * single-threaded callers pay no synchronization cost.
 *
 * v0.2.4: plays [HapticCue]s predefined-first — the device-tuned effect when the port can honor
 * it ([VibratorPort.playPredefined]/[VibratorPort.playComposedClicks] return true), the documented
 * full-amplitude [HapticPatterns.waveformFallback] otherwise. Honest degradation, never silence:
 * an unsupported or throwing device-tuned path always falls through to the raw waveform.
 *
 * Round-1 review fix (Important, P5-3/v0.2.4): [diag] exists so which physical path a cue actually
 * took is remotely diagnosable — confirming the founder's Pixel Watch lands on the OEM-tuned
 * predefined/composed path rather than silently falling back to the waveform tier is this task's
 * whole point, and without a breadcrumb there was no way to check that without screen-capturing
 * the watch. See [reportHapticPath].
 */
open class HapticDirector(
    private val vibrator: VibratorPort,
    private val nowMs: () -> Long = { System.currentTimeMillis() },
    private val diag: DiagLog = NOOP_DIAG,
) {
    private var lastChannel: String? = null
    private var lastLevel: Int? = null
    private var lastVibrationTimeMs: Long = 0L

    // Round-1 review fix: push-on-change cache for reportHapticPath, keyed by (channel, level) —
    // mirrors SentinelService.reportCaptureMode's single-value dedupe, widened to a map because
    // there are 6 independent (channel, level) combos here rather than one global mode. See
    // reportHapticPath's own KDoc.
    private val lastHapticPath = mutableMapOf<Pair<String, Int>, String>()

    // Track 1 (PRD §6 repeat schedule): the wearer's CURRENT channel-A level and when its cue last
    // physically played. Before Track 1 a level was felt exactly once, on the change — a wearer
    // who stayed at level 2 for five minutes got one double tap five minutes ago. PRD §6 says the
    // cue repeats "every 2 min / every 1 min / continuous" while the level holds; the shared
    // NudgeHapticSchedule owns those cadences, [dueReminder]/[replayReminder] apply them, and
    // SentinelController drives them once per mic window. Channel A only: §6 is about the
    // wearer's own score; channel B (partner cue) keeps its single-shot behaviour. Same
    // no-internal-locking contract as the dedupe fields above (see class KDoc).
    private var reminderLevel: Int = 0
    private var reminderLastPlayedMs: Long = 0L

    // open: lets tests exercise SentinelController's own listener-callback containment in
    // isolation from this class's internal runCatching — see SentinelControllerTest's
    // ThrowingHapticDirector.
    open fun onNudge(n: NudgeEvent) {
        // Level 0 → no vibration (de-escalation is silent) — and, for channel A, no more
        // reminders either: "score > 70: no haptic" (PRD §6) must stop the repeat, not just the
        // one-shot. Cleared BEFORE the early return so a silent de-escalation still lands.
        if (n.level == 0) {
            if (n.channel == REMINDER_CHANNEL) reminderLevel = 0
            return
        }

        val now = nowMs()
        val isDuplicate = (n.channel == lastChannel && n.level == lastLevel &&
                (now - lastVibrationTimeMs) < 5000L)
        if (isDuplicate) return

        val cue = HapticPatterns.cue(n.channel, n.level) ?: return
        play(n.channel, n.level, cue, HapticPatterns.waveformFallback(n.channel, n.level))

        lastChannel = n.channel
        lastLevel = n.level
        lastVibrationTimeMs = now
        if (n.channel == REMINDER_CHANNEL) {
            reminderLevel = n.level
            reminderLastPlayedMs = now
        }
    }

    /**
     * PRD §6 repeat: the channel-A reminder that is due right now, or `null`. Pure check, no side
     * effects — the caller (SentinelController) decides whether to actually play it (it may be
     * suppressed while the local pulse train is already covering channel A, exactly like a fresh
     * nudge would be) and then calls [replayReminder]. The returned event's `t` is the wall clock
     * in seconds and its `vectors` is empty: this is honestly a replay, not a new detection.
     */
    fun dueReminder(): NudgeEvent? {
        val level = reminderLevel
        if (level == 0) return null
        val now = nowMs()
        if (!NudgeHapticSchedule.reminderDue(level, reminderLastPlayedMs, now)) return null
        return NudgeEvent(channel = REMINDER_CHANNEL, level = level, t = now / 1000.0, vectors = emptyList())
    }

    /**
     * Plays a reminder [dueReminder] returned, bypassing [onNudge]'s 5 s dedupe (the reminder
     * cadence is already >= 10 s, and the dedupe exists to swallow server re-sends, not
     * schedule-driven repeats) and rebasing the reminder clock. Also refreshes the dedupe state,
     * so a server re-send of the same level arriving right after a reminder is still swallowed.
     * No-op if the level was cleared between the check and this call (a de-escalation raced in).
     */
    fun replayReminder(n: NudgeEvent) {
        if (n.channel != REMINDER_CHANNEL || reminderLevel == 0 || n.level != reminderLevel) return
        val cue = HapticPatterns.cue(n.channel, n.level) ?: return
        play(n.channel, n.level, cue, HapticPatterns.waveformFallback(n.channel, n.level))
        val now = nowMs()
        reminderLastPlayedMs = now
        lastChannel = n.channel
        lastLevel = n.level
        lastVibrationTimeMs = now
    }

    /** Forget the current channel-A level entirely (episode ended / sentinel disarmed): a watch
     * that is no longer listening must not keep buzzing about a conversation that's over. */
    fun clearReminder() {
        reminderLevel = 0
    }

    /** Test/diagnostic seam: the channel-A level reminders are currently scheduled for (0 = none). */
    fun reminderLevel(): Int = reminderLevel

    /**
     * v0.2.4 (Settings "Feel the buzzes"): plays a channel/level cue immediately, BYPASSING
     * [onNudge]'s 5s channel/level dedupe — a demo is user-invoked, and suppressing the second
     * tap of "let me feel that again" would defeat the screen's whole purpose. Deliberately does
     * NOT touch the dedupe state either: a real nudge arriving right after a demo must still
     * play. Same fail-soft posture as every other play path here.
     */
    fun demo(channel: String, level: Int) {
        val cue = HapticPatterns.cue(channel, level) ?: return
        play(channel, level, cue, HapticPatterns.waveformFallback(channel, level))
    }

    /**
     * P4-3: fires a single local proportional pulse (channel A's pulse train — see [PulseEngine]
     * KDoc), completely independent of [onNudge]'s channel/level dedupe above. Falls back to a
     * fixed max amplitude when the device has no amplitude control (honest degradation — see
     * [VibratorPort.hasAmplitudeControl]'s KDoc). Fail-soft: a throwing platform Vibrator must
     * never propagate to the caller.
     */
    fun playPulse(pulse: Pulse) {
        val amplitude = if (vibrator.hasAmplitudeControl()) pulse.amplitude else FIXED_AMPLITUDE_NO_CONTROL
        runCatching { vibrator.vibrate(longArrayOf(0L, pulse.durationMs), intArrayOf(0, amplitude)) }
    }

    /** Predefined/composed-first with waveform fallback. A port that returns false OR throws on
     * the device-tuned path degrades to [fallback]; a [HapticCue.Waveform] cue plays directly and
     * never "falls back" to itself. Every vibrator call is fail-soft (may run on OkHttp's reader
     * thread via SentinelController's WS listener — see [onNudge]'s original Task-7 rationale). */
    private fun play(channel: String, level: Int, cue: HapticCue, fallback: HapticCue.Waveform?) {
        val played = runCatching {
            when (cue) {
                is HapticCue.Predefined -> vibrator.playPredefined(cue.effect)
                is HapticCue.ComposedClicks -> vibrator.playComposedClicks(cue.count, cue.gapMs)
                is HapticCue.Waveform -> {
                    vibrator.vibrate(cue.timingsMs.toLongArray(), cue.amplitudes.toIntArray())
                    true
                }
            }
        }.getOrDefault(false)
        val path = when {
            played -> when (cue) {
                is HapticCue.Predefined -> "predefined"
                is HapticCue.ComposedClicks -> "composed"
                is HapticCue.Waveform -> "waveform"
            }
            // A Waveform cue IS its own fallback (waveformFallback("B", level) returns the same
            // cue) — replaying it would just repeat the exact same failing call, so there's
            // nothing left to try. Distinct label from "waveform" so a device that can't even play
            // its native waveform is honestly distinguishable from one that played it fine.
            cue is HapticCue.Waveform -> "waveform-failed"
            fallback != null -> {
                runCatching { vibrator.vibrate(fallback.timingsMs.toLongArray(), fallback.amplitudes.toIntArray()) }
                "waveform-fallback"
            }
            // Shouldn't happen in practice — cue()/waveformFallback() are always defined together
            // for the same valid (channel, level) — but named honestly rather than silently
            // reporting nothing if it ever does.
            else -> "unplayable"
        }
        reportHapticPath(channel, level, path)
    }

    /**
     * Round-1 review fix (Important, v0.2.4): remote breadcrumb for which physical path a cue
     * actually took ("predefined" / "composed" / "waveform" native / "waveform-fallback" /
     * "waveform-failed" / "unplayable") — the founder's only way to confirm the OEM-tuned path is
     * landing on the real Pixel Watch without screen-capturing it (this task's whole point; see
     * CLAUDE.md's telemetry-channel rationale). Push-on-change per (channel, level), mirroring
     * [app.gauge.wear.service.SentinelService.reportCaptureMode]'s dedupe discipline: a repeated
     * same-outcome play must NOT re-log (the DebugRing is small, fixed-size, and no-dedup — see
     * [app.gauge.wear.telemetry.Telemetry]), but a genuinely new outcome for that channel/level
     * (e.g. a device that stops honoring EFFECT_CLICK mid-session) always gets its own line.
     */
    private fun reportHapticPath(channel: String, level: Int, path: String) {
        val key = channel to level
        if (lastHapticPath[key] == path) return
        lastHapticPath[key] = path
        diag.log("info", TAG, "haptic path: $channel L$level = $path")
    }

    private companion object {
        const val FIXED_AMPLITUDE_NO_CONTROL = 255
        const val TAG = "HapticDirector"
        /** The lane PRD §6's repeat schedule applies to — the wearer's own escalation. */
        const val REMINDER_CHANNEL = "A"
    }
}
