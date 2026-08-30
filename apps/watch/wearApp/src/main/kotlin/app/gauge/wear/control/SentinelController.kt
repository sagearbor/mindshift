package app.gauge.wear.control

import app.gauge.shared.NudgeEvent
import app.gauge.shared.NudgeStateMachine
import app.gauge.shared.VectorEvent
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.Observation
import app.gauge.shared.sentinel.RingBuffer
import app.gauge.shared.sentinel.SentinelDetector
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.sentinel.SentinelStateMachine
import app.gauge.shared.sentinel.params
import app.gauge.shared.signals.Cadence
import app.gauge.shared.signals.HrTracker
import app.gauge.shared.signals.MovementTracker
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.shared.signals.SpeakingRateTracker
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.haptics.HapticPatterns
import app.gauge.wear.haptics.Pulse
import app.gauge.wear.haptics.PulseEngine
import app.gauge.wear.haptics.ShoutTapGate
import app.gauge.wear.net.EpisodeWsClient
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.min

/**
 * The orchestrator wiring mic -> ring buffer -> detector -> state machine -> WebSocket -> haptics.
 *
 * Threading model: [tick], [arm], [disarm], and [setMode] are all meant to be driven by a single
 * caller thread (the service's own loop thread) and touch the "core" fields (detector, ring
 * buffer, state machine, mode, window counters) without any locking — see [EpisodeWsClient]'s own
 * KDoc: its [EpisodeWsClient.Listener] callbacks arrive on OkHttp's reader thread, a *different*
 * thread than the one driving [tick]. The handful of fields both sides touch ([ws], [online],
 * [channelLevels], [lastVector]) are guarded by [lock] at every read/write, on both threads. Both
 * call sites that invoke [HapticDirector.onNudge] (the WS listener's `onNudge`, and the local
 * offline-ladder path in `streamWindow`) also hold [lock] for the call itself, since
 * [HapticDirector]'s own dedupe state isn't internally synchronized (see its KDoc).
 *
 * Window/second convention: every mic window is assumed to be exactly one second of audio at
 * [SAMPLE_RATE_HZ] (matching [SentinelStateMachine]'s window-counted `quietSecondsToStop`/
 * `cooldownSeconds` and shared's `SentinelDetectorTest` `tone()` helper's default `n = 16000`).
 *
 * P4-3: [pulseEngine] drives the local proportional pulse-train haptic — see its own KDoc for the
 * full contract. [emitPulseIfDue] calls it once per STREAMING window (both online and offline,
 * unconditional on [online] — the whole point is it never depends on server connectivity) and
 * publishes the verdict as [ControllerState.activePulse]. This REPLACES channel-A's old discrete
 * level-pattern haptic while pulses are enabled ([pulseIntervalMs] non-null — see
 * [playChannelHapticIfApplicable]); [nowMs] is the pulse engine's own wall-clock supplier (real
 * time in production, fake in tests).
 *
 * Review fix (P4-3 round 2): this controller does NOT itself call [HapticDirector.playPulse] —
 * [app.gauge.wear.service.SentinelService]'s repeat loop is the SOLE place any pulse is physically
 * played. The original design had this controller fire the first pulse of a window synchronously
 * (on its own handler thread) *in addition to* the service's independently-scheduled repeat chain
 * (on a different thread) — two independent physical-timing sources for the same channel, which
 * is exactly what let a stale tail-end repeat and a fresh synchronous fire land on top of each
 * other at a window boundary, violating [PulseEngine]'s "never merge" guarantee. Collapsing to one
 * physical-timing source removes that race by construction rather than by timing luck; see
 * [PulseChainGate]/[PulseChainDecision] for how the service's own single-source loop stays
 * race-safe against a superseded/late `Runnable`.
 */
class SentinelController(
    mode: Mode,
    private val mic: MicSource,
    private val wsFactory: WsFactory,
    private val haptics: HapticDirector,
    private val ids: EpisodeIdFactory,
    private val diag: DiagLog = NOOP_DIAG,
    private val hr: ScalarSource? = null,
    private val accel: ScalarSource? = null,
    private val selectedSignal: () -> SignalKind = { SignalKind.VOLUME },
    private val pulseIntervalMs: () -> Long? = { DEFAULT_PULSE_INTERVAL_MS },
    /** Tier B: the session id COMPANION mode opens its socket under (see [companionSessionId] —
     *  deterministic per day+account in production, so a whole day of reconnects lands on one id).
     *  Null falls back to [ids] like every other episode. Mic-using modes never read this. */
    private val companionSessionId: (() -> String)? = null,
    nowMs: () -> Long = { System.currentTimeMillis() },
) {
    private val lock = Any()

    private var mode: Mode = mode
    private var pendingMode: Mode? = null

    private var detector = SentinelDetector(mode.params().triggerDbOverBaseline)
    private var stateMachine = SentinelStateMachine(mode)
    private var ringBuffer = RingBuffer(capacityBytesFor(mode))

    // Wave C Task 10: unlike ringBuffer above (ARMED-only, mode-sized, exists to seed an episode's
    // preamble), this one is a SECOND, independent 5-minute-ceiling buffer that grows on every
    // window regardless of sentinel state — see RetroCaptureBuffer's own KDoc for why. Never
    // resized/replaced (unlike ringBuffer, which rebuilds per mode in rebuildForMode) — its
    // capacity is fixed for the life of this controller instance.
    private val retroCaptureBuffer = app.gauge.shared.capture.RetroCaptureBuffer()

    private val nudgeStateMachine = NudgeStateMachine()
    private val pulseEngine = PulseEngine(intervalMs = pulseIntervalMs, nowMs = nowMs)

    // I2: kept alongside pulseEngine so playChannelHapticIfApplicable can ask "what time is it
    // right now" without a second wall-clock source drifting from the one PulseEngine itself
    // gates its own verdicts on.
    private val nowMsSupplier: () -> Long = nowMs

    // Task 10: per-signal baseline trackers, rebuilt alongside [detector] in [rebuildForMode] so a
    // mode change (fresh episode sizing) also starts these signals' baselines fresh.
    private var hrTracker = HrTracker()
    private var movementTracker = MovementTracker()
    private var speakingRateTracker = SpeakingRateTracker()

    // Core-thread-only bookkeeping for the *current* episode (reset in startStreaming()).
    private var windowIndex: Long = 0
    private var wsEndSent = false

    // Tier B (COMPANION): when the last JSON heartbeat went out — core-thread-only, same contract
    // as windowIndex. Stamped on every (re)connect so a fresh socket waits a full interval.
    private var lastHeartbeatAtMs: Long = 0L

    // Core-thread-only (same contract as `mode`/`stateMachine` above — written by tick(), read by
    // the `state` getter, both always on the single core thread): the live meter for whichever
    // signal `selectedSignal()` currently names. `null` whenever that signal has no reading yet
    // (honest degradation — see MeterReading KDoc).
    private var meter: MeterReading? = null

    // Core-thread-only, same contract as `meter` above (P4-10): availability of the CURRENTLY
    // SELECTED signal — see ControllerState.availability's own KDoc. Computed only for
    // SignalKind.HEART_RATE (the only source with a real availability API today; see
    // safeAvailability()); every other selection leaves this at UNKNOWN.
    private var availability: SignalAvailability = SignalAvailability.UNKNOWN

    // Core-thread-only, same contract as `meter` above (P5-3): whether the most recently processed
    // window was voiced/above-threshold — the one fact SentinelService needs to drive MicDutyCycle
    // without reaching into this controller's private `obs`/`detector`. Set from `obs.voiced` at
    // the top of processWindow(); cleared in disarm() beside `meter`/`availability`.
    private var lastWindowVoiced: Boolean = false

    // Core-thread-only, same contract as `meter` above (P4-3): the pulse-train verdict for the
    // *current* window — the pulse due this tick, or null (below threshold, pulses off, or gated
    // by PulseEngine's own interval). Reset at the top of every processWindow() call and only
    // ever (re)set by emitPulseIfDue() when streamWindow() is actually reached this tick.
    private var activePulse: Pulse? = null

    // Core-thread-only, same contract as `meter`/`activePulse` above (v0.2.4): the ARMED shout-tap
    // due THIS window, or null — see ControllerState.armedShoutTap's own KDoc. shoutTapGate rate-
    // limits it independently of activePulse's own PulseEngine gating.
    private var armedShoutTap: Pulse? = null
    private val shoutTapGate = ShoutTapGate()

    // Guarded by [lock] — read by tick()/state getter on the core thread, written by both the
    // core thread (startStreaming/endEpisodeStream/disarm) and the WS listener callbacks
    // (onEpisodeSaved/onFailure) on OkHttp's thread.
    private var ws: EpisodeWs? = null
    private var online: Boolean = true
    private val channelLevels = mutableMapOf<String, Int>()
    private var lastVector: String? = null
    private val sparkline = ArrayDeque<Double>()

    // v0.3.1: which SignalKind `sparkline` currently holds values for — see recordSparkline's own
    // KDoc for the "armed sparkline follows the selected signal" contract this backs.
    private var sparklineKind: SignalKind = SignalKind.VOLUME

    // P4-5: mid-episode WS reconnect. `reconnectPolicy` carries the same cross-thread contract as
    // `ws`/`online` above (touched by the core thread via streamWindow()/attemptReconnect() AND
    // by the WS listener's onFailure on OkHttp's thread) — every access is under `lock`.
    private val reconnectPolicy = ReconnectPolicy()

    // Generation token for EpisodeWs [EpisodeWsClient.Listener] callbacks — mirrors
    // [app.gauge.wear.haptics.PulseChainGate]'s own discipline (see its KDoc for the analogous
    // Handler-race incident this class was built to prevent), applied here for the same reason
    // P4-5 makes it newly relevant: before reconnect, at most one EpisodeWs existed per STREAMING
    // run, so a listener callback could only ever be "the" listener. A mid-episode reconnect can
    // now create a second (or third...) EpisodeWs within the same STREAMING run, and every
    // Listener instance's callbacks mutate the SAME shared `ws`/`online`/`channelLevels`/
    // `lastVector` fields regardless of which physical client they came from — a stale callback
    // from a client already superseded by a later reconnect (or by disarm()) must not clobber
    // that newer state. [buildListener] mints a fresh token per client; every callback checks it
    // first and no-ops if superseded.
    private val wsListenerGeneration = AtomicInteger(0)

    val state: ControllerState
        get() = synchronized(lock) {
            ControllerState(
                sentinel = stateMachine.state,
                mode = mode,
                online = online,
                channelLevels = channelLevels.toMap(),
                lastVector = lastVector,
                sparkline = sparkline.toList(),
                meter = meter,
                activePulse = activePulse,
                availability = availability,
                lastWindowVoiced = lastWindowVoiced,
                armedShoutTap = armedShoutTap,
                retroCaptureAvailableSeconds = retroCaptureBuffer.availableSeconds(),
                sparklineSignal = sparklineKind,
            )
        }

    fun arm() {
        // Contract note 2: never call stateMachine.onArm() while already armed/streaming — arming
        // an already-armed/streaming controller is a no-op.
        if (stateMachine.state != SentinelState.DISARMED) return
        applyPendingModeIfAny()
        val newState = stateMachine.onArm()
        if (newState == SentinelState.STREAMING) {
            // Mode.SESSION: onArm() jumps straight to STREAMING with no ARMED->STREAMING tick to
            // hook into, so start the episode here.
            startStreaming()
        }
    }

    fun disarm() {
        if (stateMachine.state == SentinelState.DISARMED) return
        synchronized(lock) {
            // Review fix (P4-5 round 2): bump INSIDE this same lock acquisition, before any of the
            // state below — not after the block, as the original round-1 version had it. A WS
            // listener callback's own currency check now happens under this SAME `lock` (see
            // [buildListener]/[goOfflineAndArmReconnectIfCurrent]), so the two are mutually
            // exclusive: whichever of {this disarm() call, a racing stale callback} acquires `lock`
            // second is guaranteed to observe everything the first one did — including this bump.
            // A bump placed after the block (as round 1 had it) left a real window: a callback that
            // read the OLD generation, got descheduled, and resumed — after this whole block had
            // already released the lock and returned true from `online = true` — could still race
            // the bump itself (a plain `AtomicInteger.incrementAndGet()` outside any lock has no
            // ordering relationship with the callback's own lock-protected read) and mutate state
            // disarm() had just cleaned up. Mirrors PulseChainGate's own documented requirement:
            // currency must be checked immediately before EVERY side effect, under the same
            // synchronization domain as whatever invalidates it — not via an isolated pre-check.
            wsListenerGeneration.incrementAndGet()
            ws?.cancel()
            ws = null
            // State hygiene: a stale `online == false` left over from a prior episode's failure
            // would misreport "offline" to the UI while sitting DISARMED/ARMED with no connection
            // attempt in flight at all. The next startStreaming() sets it explicitly anyway, so
            // this doesn't change behavior — it just stops the interim misreport.
            online = true
            // P4-5: disarm is a give-up condition for any in-flight reconnect backoff — without
            // this, a backoff armed by this episode's failure would otherwise survive into a
            // later re-arm and (a) potentially fire a reconnect attempt for an episode that no
            // longer exists, were it not for isAttemptDue()'s own streaming-only gate, and (b)
            // still incorrectly continue the ladder from wherever it left off instead of starting
            // a genuinely new episode's failure back at the initial delay.
            reconnectPolicy.reset()
            // Wave-final review, Important 1: a disarmed sentinel must not keep reporting the last
            // episode's channel levels — see [startStreaming]'s own comment for the fuller
            // "stale across episodes" story this and that clear both close. Cleared under the same
            // `lock` acquisition as the rest of this block, same contract as every other field
            // touched here.
            channelLevels.clear()
            // Track 1: a disarmed watch stops repeating the last episode's cue (PRD §6 reminders
            // are for a conversation that's still happening). Under `lock` like every other
            // HapticDirector call in this class.
            haptics.clearReminder()
        }
        // C1: `meter`/`activePulse` are core-thread-only fields (same contract as the rest of
        // this section's KDoc — written by tick()/processWindow(), read by the `state` getter,
        // all on the single core thread), so they don't need `lock` themselves; reset here so a
        // disarmed sentinel's `state` doesn't keep reporting a stale meter/pulse reading from the
        // episode that just ended — that reading would otherwise mask MainActivity's live preview
        // meter (P4-6) from showing "no reading yet" honestly until the *next* window overwrites
        // it, which may never come if the wearer stays disarmed.
        meter = null
        activePulse = null
        // v0.2.4: a disarmed sentinel must not keep a tap or a stale rate-limit clock.
        armedShoutTap = null
        shoutTapGate.reset()
        // P4-10: same "stale across episodes" rationale as `meter` above — a disarmed sentinel
        // must not keep reporting the last episode's off-body/acquiring status.
        availability = SignalAvailability.UNKNOWN
        // P5-3: same "stale across episodes" rationale — a disarmed sentinel reports nothing about
        // a window it isn't reading.
        lastWindowVoiced = false
        // Review fix (T6, v0.2.4): same "stale across episodes" rationale as meter/activePulse/
        // availability above — missed in the original pass. Without this, rearming showed up to
        // SPARKLINE_LENGTH ticks of the PREVIOUS session's loudness trace as if it were current,
        // now user-visible now that Task 7 no longer gates the sparkline behind an open episode.
        sparkline.clear()
        // Wave C Task 10: same "nothing stale survives a disarm" rule — a disarmed sentinel has no
        // audio left to retro-save either.
        retroCaptureBuffer.clear()
        stateMachine.onDisarm()
        wsEndSent = false
        applyPendingModeIfAny()
    }

    /** Wave C Task 10: returns the most recent [seconds] of the wearer's own audio, or `null` if
     * nothing has been recorded yet (mirrors [app.gauge.shared.capture.RetroCaptureBuffer.
     * snapshot]'s own honest clamping — never zero-padded, never fabricated). Callable regardless
     * of sentinel state. */
    fun captureRetroSnapshot(seconds: Double): ByteArray? {
        val available = retroCaptureBuffer.availableSeconds()
        if (available <= 0.0) return null
        return retroCaptureBuffer.snapshot(seconds)
    }

    fun setMode(m: Mode) {
        if (stateMachine.state == SentinelState.DISARMED) {
            mode = m
            rebuildForMode(m)
        } else {
            // Don't disrupt an in-flight episode's detector/state-machine/ring-buffer sizing;
            // apply on the next return to DISARMED (see applyPendingModeIfAny()).
            pendingMode = m
        }
    }

    fun tick() {
        if (stateMachine.state == SentinelState.DISARMED) return

        // Tier B: COMPANION never touches the mic — the phone listens; this tick only keeps the
        // nudge socket alive (heartbeat + reconnect) and replays due PRD §6 reminders.
        if (mode == Mode.COMPANION) {
            companionTick()
            return
        }

        val window = mic.readWindow()
        if (window == null) {
            // Round-1 review fix (Important 3, P5-3): this is the ONLY place a null window turns
            // into an auto-disarm — without a breadcrumb here, a silent mic dropout (including a
            // MicReader.resume() failure this same window's KDoc warns about) is indistinguishable
            // from an explicit user Off tap in the telemetry channel, this app's only on-device
            // debugging tool per CLAUDE.md.
            diag.log("warn", "SentinelController", "mic lost (readWindow returned null); auto-disarming")
            disarm() // mic lost: treat as disarm, stop cleanly.
            return
        }

        processWindow(window)
    }

    /**
     * Task 10 test seam: processes a caller-supplied window instead of reading one from [mic].
     * Production always goes through the no-arg [tick] above (real audio comes from the mic);
     * this overload exists so controller tests can drive specific tone windows directly without
     * having to pre-load a [ScriptedMic] queue.
     */
    fun tick(window: ShortArray) {
        if (stateMachine.state == SentinelState.DISARMED) return
        processWindow(window)
    }

    private fun processWindow(window: ShortArray) {
        val obs = detector.observe(window)
        // P5-3: the one fact SentinelService needs to drive MicDutyCycle — see the field's own
        // KDoc for why this lives here rather than the service reaching into `obs`/`detector`.
        lastWindowVoiced = obs.voiced
        val voicedLoud = obs.voiced && obs.dbOverBaseline >= mode.params().triggerDbOverBaseline
        windowIndex++
        // Track 1 / PRD §6: re-fire the current channel-A cue on its schedule. Once per window is
        // the right cadence — the shortest interval (level 3) is 10 s, so a 1 s tick lands within
        // one window of due, and this is the same thread that already owns every other haptic
        // decision in this class.
        replayDueReminder()
        // P4-3: reset every tick; only overwritten below when streamWindow() is actually reached
        // this tick (see emitPulseIfDue()) — any tick that doesn't (ARMED-only, the ARMED->
        // STREAMING transition tick itself, or COOLDOWN) reports no active pulse.
        activePulse = null
        // v0.2.4: reset every tick; only overwritten below when this tick qualifies as an ARMED
        // shout-tap window (see the block after the `when` further down).
        armedShoutTap = null

        // Read the HR source at most once per window (its raw bpm feeds both the meter below and
        // the STREAMING hr_spike send further down) — see updateMeter()/trySendHr() KDoc.
        val hrBpm = safeLatest(hr, "hr")
        updateMeter(window, obs, voicedLoud, hrBpm)

        val prevState = stateMachine.state
        val pcmBytes = pcm16LittleEndian(window)

        if (prevState == SentinelState.ARMED) {
            ringBuffer.push(pcmBytes)
        }
        // v0.2.5/Wave C: unlike the trigger-side ringBuffer above (ARMED-only, mode-sized), this
        // one grows on every window regardless of sentinel state — see RetroCaptureBuffer's own
        // KDoc for why. Fail-soft (Phase-3 rule: trigger paths never crash the process) — a
        // RetroCaptureBuffer failure must degrade to "this window didn't get retro-saved," never
        // take down the trigger path riding the same tick.
        safeRetroCapturePush(pcmBytes)

        val newState = stateMachine.onWindow(obs.triggered, voicedLoud)

        when {
            prevState == SentinelState.ARMED && newState == SentinelState.STREAMING -> {
                // This window's bytes are already in the ring buffer (pushed above) and will go
                // out as the tail of the preamble — nothing more to send for it this tick.
                startStreaming()
            }
            newState == SentinelState.STREAMING -> {
                streamWindow(pcmBytes, obs)
            }
            prevState == SentinelState.STREAMING && newState == SentinelState.COOLDOWN -> {
                endEpisodeStream()
            }
            prevState == SentinelState.COOLDOWN && newState == SentinelState.ARMED -> {
                // Full episode lifecycle complete, naturally cycling back to ARMED (no arm()/
                // disarm() call in between) — this is the only other point besides arm() where a
                // queued setMode() can safely take effect, since applyPendingModeIfAny() rebuilds
                // the detector/state machine from scratch (same as arm() does).
                if (pendingMode != null) {
                    applyPendingModeIfAny()
                    val armed = stateMachine.onArm()
                    if (armed == SentinelState.STREAMING) {
                        startStreaming()
                    }
                }
            }
            else -> Unit // COOLDOWN counting further, or ARMED with no transition: nothing to send.
        }

        // v0.2.4: ARMED shout-tap — the product must be *felt* before an episode ever opens. Fires
        // only when the window was loud (the EXACT voicedLoud the trigger uses — baseline-relative,
        // threshold = this mode's trigger bar) AND the sentinel both started and stayed ARMED this
        // tick: the ARMED->STREAMING transition tick is excluded so this one-shot can never land on
        // top of the pulse chain's own immediate first fire, and STREAMING/COOLDOWN never tap.
        // Fail-soft (the codebase's standing "Task 7 posture" from the Phase-3 plan — see the
        // surrounding methods' identical wording): nothing here may escape processWindow().
        if (prevState == SentinelState.ARMED && newState == SentinelState.ARMED && voicedLoud) {
            armedShoutTap = try {
                if (shoutTapGate.onLoudWindow(nowMsSupplier())) HapticPatterns.ARMED_SHOUT_TAP else null
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "shout-tap gating failed: $t")
                null
            }
        }

        // Task 10: hr_spike vector goes live server-side once a real bpm reading rides along on
        // every window while an episode is actually in flight and reachable — matches
        // streamWindow()'s own "only while STREAMING and online" contract for PCM frames.
        // Independent of `selectedSignal` — this sends whenever the device HAS an hr source and
        // reading, regardless of which signal the wearer has the meter showing.
        if (newState == SentinelState.STREAMING) {
            trySendHr(hrBpm)
        }
    }

    // --- Task 10: on-device signals + live meter ------------------------------------------------

    /**
     * Recomputes every available signal's reading this window and republishes [meter] for
     * whichever [SignalKind] [selectedSignal] currently names. Fail-soft (mirrors the rest of the
     * trigger path, Task 7): a throwing [accel] source degrades to "no reading" via [safeLatest]
     * rather than propagating out of [processWindow]; [hrBpm] is already fail-soft-computed by the
     * caller.
     */
    private fun updateMeter(window: ShortArray, obs: Observation, voicedLoud: Boolean, hrBpm: Double?) {
        val hrReading = hrBpm?.let { hrTracker.observe(it) }
        val accelStddev = safeLatest(accel, "accel")
        val movementReading = accelStddev?.let { movementTracker.observe(it) }
        val speakingReading = speakingRateTracker.observe(Cadence.burstsPerSecond(window))

        val selected = selectedSignal()
        availability = if (selected == SignalKind.HEART_RATE) safeAvailability(hr) else SignalAvailability.UNKNOWN
        // Review fix (P4-10 round 1, Critical): a structural guarantee at THIS combine point,
        // independent of whatever the ScalarSource implementation does or forgets to do — see
        // HrSource.onAvailabilityChanged's own matching fix (belt-and-suspenders, not either/or).
        // Only ACQUIRING/OFF_BODY/UNAVAILABLE (availability has something honest to SAY, i.e.
        // SignalAvailability.statusText() != null, and it isn't "yes, available") suppress the
        // number — UNKNOWN must still pass a reading through unchanged, exactly like before this
        // task, for every ScalarSource that has no real availability API at all (see ScalarSource.
        // availability()'s own KDoc: "UNKNOWN renders no status string, so other sources look
        // exactly as before" — the meter side of that same contract).
        val hrAvailabilityBlocksReading = selected == SignalKind.HEART_RATE &&
            availability != SignalAvailability.AVAILABLE &&
            availability != SignalAvailability.UNKNOWN

        meter = when (selected) {
            SignalKind.VOLUME -> MeterReading(
                signal = SignalKind.VOLUME,
                value = obs.dbfs,
                threshold = detector.baseline?.plus(mode.params().triggerDbOverBaseline),
                over = voicedLoud,
            )
            SignalKind.HEART_RATE -> if (hrAvailabilityBlocksReading) {
                null
            } else {
                hrReading?.let {
                    MeterReading(signal = SignalKind.HEART_RATE, value = it.value, threshold = it.threshold, over = it.over)
                }
            }
            SignalKind.MOVEMENT -> movementReading?.let {
                MeterReading(signal = SignalKind.MOVEMENT, value = it.value, threshold = it.threshold, over = it.over)
            }
            SignalKind.SPEAKING_RATE -> MeterReading(
                signal = SignalKind.SPEAKING_RATE,
                value = speakingReading.value,
                threshold = speakingReading.threshold,
                over = speakingReading.over,
            )
        }
        recordSparkline(selected, obs.dbOverBaseline)
    }

    /** `null` on either "no source configured", "no reading yet" (source returns `null`), or a
     * thrown exception — reported to [diag] in the last case. Never propagates. */
    private fun safeLatest(source: ScalarSource?, tag: String): Double? {
        if (source == null) return null
        return try {
            source.latest()
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "$tag.latest() failed: $t")
            null
        }
    }

    /** Wave C Task 10: fail-soft wrap for [retroCaptureBuffer]'s write side, same posture as
     * [safeLatest]/[safeAvailability] — [RetroCaptureBuffer.push] is pure/dependency-free and
     * shouldn't realistically throw, but this rides the SAME tick as the trigger path (Phase-3
     * rule: trigger paths never crash the process), so a hypothetical failure here degrades to
     * "this window didn't get retro-saved" rather than risking the tick that also drives
     * detection/streaming. */
    private fun safeRetroCapturePush(pcmBytes: ByteArray) {
        try {
            retroCaptureBuffer.push(pcmBytes)
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "retroCaptureBuffer.push failed: $t")
        }
    }

    /** [SignalAvailability.UNKNOWN] on either "no source configured" or a thrown exception —
     * reported to [diag] in the latter case, same fail-soft contract as [safeLatest]. Never
     * propagates. */
    private fun safeAvailability(source: ScalarSource?): SignalAvailability {
        if (source == null) return SignalAvailability.UNKNOWN
        return try {
            source.availability()
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "availability() failed: $t")
            SignalAvailability.UNKNOWN
        }
    }

    /** Sends this window's already-fail-soft-computed HR reading (if any) while STREAMING and
     * online — see the call site's comment for why this is unconditional on [selectedSignal]. */
    private fun trySendHr(bpm: Double?) {
        if (bpm == null) return
        val (client, isOnline) = synchronized(lock) { ws to online }
        if (client == null || !isOnline) return
        try {
            client.sendHr(bpm, windowIndex.toDouble())
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "sendHr failed: $t")
            val delay = goOfflineAndArmReconnect()
            diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
        }
    }

    // --- Tier B: companion (no-mic) session --------------------------------------------------

    /**
     * One ~1 s companion tick, driven by SentinelService's own postDelayed cadence (there is no
     * blocking mic read to pace the loop in this mode). Three jobs, all fail-soft:
     * reminder replays (the PRD §6 repeat schedule works exactly as in a mic episode — nudges
     * arrive via the WS listener and set [HapticDirector]'s reminder level), the reconnect
     * backoff (same [ReconnectPolicy]/[maybeAttemptReconnect] as a mic episode — COMPANION is
     * always STREAMING while armed), and the 20 s JSON heartbeat that keeps a frames-free socket
     * alive through NATs/idle reapers. Never reads the mic, never sends PCM.
     */
    private fun companionTick() {
        replayDueReminder()
        maybeAttemptReconnect()
        val now = nowMsSupplier()
        val (client, isOnline) = synchronized(lock) { ws to online }
        if (client == null || !isOnline) return
        if (now - lastHeartbeatAtMs < HEARTBEAT_INTERVAL_MS) return
        try {
            client.sendHeartbeat()
            lastHeartbeatAtMs = now
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "sendHeartbeat failed: $t")
            val delay = goOfflineAndArmReconnect()
            diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
        }
    }

    /** The id this session's socket opens under: COMPANION reuses the caller-supplied
     *  deterministic id on every (re)connect (a companion session persists nothing server-side, so
     *  one id per day reads better in logs than a UUID per reconnect); every other mode mints a
     *  fresh episode id exactly as before. */
    private fun nextEpisodeId(): String =
        if (mode == Mode.COMPANION) (companionSessionId?.invoke() ?: ids.newId()) else ids.newId()

    /** Post-open companion handshake: announce `{"type":"companion"}` (the server suppresses
     *  persistence for this socket) and start the heartbeat clock. No-op for mic modes. */
    private fun sendCompanionHelloIfApplicable(client: EpisodeWs) {
        if (mode != Mode.COMPANION) return
        client.sendCompanionHello()
        lastHeartbeatAtMs = nowMsSupplier()
    }

    // --- episode lifecycle -------------------------------------------------------------------

    private fun startStreaming() {
        // v0.2.4: a fresh episode clears the pre-episode tap clock, so the post-episode return to
        // ARMED starts clean.
        shoutTapGate.reset()
        wsEndSent = false
        // P4-5: a fresh episode start gives up on any reconnect backoff a *previous* episode's
        // failure armed — see reconnectPolicy.reset()'s own KDoc for why leaving it armed would
        // wrongly carry that episode's ladder position into this one.
        //
        // Wave-final review, Important 1: a fresh episode also starts with a clean channelLevels —
        // otherwise a wearer who stays armed across back-to-back conversations (the natural
        // COOLDOWN -> ARMED -> STREAMING cycle, no disarm()/arm() in between) would have episode
        // N+1's very first STREAMING tick publish episode N's last level (e.g. "Level 3"/a red
        // arc on the complication, P4-4) before episode N+1 has had a single nudge of its own.
        // Deliberately NOT cleared in [attemptReconnect] — a mid-episode reconnect is the SAME
        // conversation continuing after a connectivity gap, so carrying levels across it is
        // correct (see [attemptReconnect]'s own KDoc and SentinelControllerTest's
        // `midEpisodeReconnectPreservesChannelLevels`).
        synchronized(lock) {
            reconnectPolicy.reset()
            channelLevels.clear()
            // Track 1: same "fresh episode, clean slate" rule for the PRD §6 reminder — episode
            // N+1 must not inherit episode N's repeat cadence any more than its level.
            haptics.clearReminder()
        }
        // Fail-soft (Task 7): factory.create()/open()/the preamble send are all real I/O (or, on a
        // fresh install missing INTERNET/RECORD_AUDIO wiring, a SecurityException) that must never
        // reach the caller — tick()'s state machine has *already* transitioned to STREAMING by the
        // time this runs (see tick()'s STREAMING branch), so any failure here just means this
        // episode degrades to the offline local nudge ladder rather than losing the whole process.
        try {
            // Every triggered episode gets a fresh WS attempt, regardless of whether a *previous*
            // episode ended offline — "online" only ever reflects *this* episode's connection state
            // (set false by onFailure below), never a standing decision to skip future episodes.
            val client = wsFactory.create()
            val token = wsListenerGeneration.incrementAndGet()
            // Publish ws/online BEFORE calling open(), not after: open() is async and its listener
            // callbacks (in particular onFailure, on a real connection refusal) can fire on OkHttp's
            // thread before open() even returns. Writing ws/online post-open() would silently
            // clobber a same-or-later onFailure's ws=null/online=false back to "healthy" — this
            // ordering guarantees onFailure, however fast, is always the last writer.
            synchronized(lock) { ws = client; online = true }
            client.open(nextEpisodeId(), buildListener(token))
            sendCompanionHelloIfApplicable(client)

            val preamble = ringBuffer.snapshot()
            if (preamble.isNotEmpty()) {
                sendInSlices(client, preamble)
            }
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "startStreaming failed: $t")
            val delay = goOfflineAndArmReconnect()
            diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
        } finally {
            // Unconditional, regardless of success/failure/where in the try an exception hit:
            // this episode's ring-buffer contents are either already sent (success path) or
            // orphaned by a failed create()/open() (failure path) — either way they must not
            // survive to bleed into the *next* episode's preamble as stale audio.
            ringBuffer.clear()
        }
    }

    /**
     * P4-5: consults [reconnectPolicy] for whether a mid-episode reconnect attempt is due right
     * now (this episode is STREAMING, currently offline, and enough backoff time has elapsed
     * since the last failure/attempt) and, if so, performs it via [attemptReconnect]. Called once
     * per STREAMING window from [streamWindow], before that window's own send/offline-ladder
     * handling — so a reconnect that succeeds this tick gets this same window's PCM flowing on
     * the fresh client immediately rather than losing it. Contributes no decision logic of its
     * own — see [ReconnectPolicy] KDoc for the backoff/give-up rules this defers to entirely.
     */
    private fun maybeAttemptReconnect() {
        val due = synchronized(lock) {
            // `streaming = true` is hardcoded here, not derived from `stateMachine.state`: this
            // method is ONLY ever called from `streamWindow()`, which `processWindow()` only
            // reaches while `state == STREAMING` (see the `when` block in `processWindow()`) — so
            // there is nothing else this call site could pass. `isAttemptDue`'s `streaming`
            // parameter still earns its keep on `ReconnectPolicy` itself: it's what makes the
            // give-up-on-COOLDOWN/disarm rule directly unit-testable in isolation (see
            // ReconnectPolicyTest's `stopOnNotStreamingEvenWhenAttemptWouldOtherwiseBeDue`)
            // without needing a controller/mic/WS harness — mirrors why PulseChainDecision.decide()
            // takes an explicit `streaming: Boolean` too, even though SentinelService's own call
            // site always derives it the same way.
            !online && reconnectPolicy.isAttemptDue(streaming = true, nowMs = nowMsSupplier())
        }
        if (due) attemptReconnect()
    }

    /**
     * P4-5: performs one mid-episode WS reconnect attempt — a fresh [EpisodeWs] against a
     * brand-new episode id. Server-side resume doesn't exist yet, so this is a deliberate v1
     * semantics choice (controller-ratified): the new episode doc honestly represents the
     * connectivity gap rather than pretending to continue the old one. Deliberately does NOT
     * touch [ringBuffer] or send any preamble — unlike [startStreaming]'s original trigger (which
     * captures pre-trigger ARMED-state audio), the ring buffer is never pushed to while STREAMING
     * (see [processWindow]'s `prevState == ARMED` guard around `ringBuffer.push`), so by the time
     * an episode needs reconnecting it's already empty; fabricating a preamble here would
     * misrepresent the gap as captured audio. Fail-soft, same contract as [startStreaming]: any
     * I/O exception here degrades back to offline + a re-armed backoff rather than escaping to
     * the caller (this runs on the same core thread as `tick()`, inside [streamWindow]).
     */
    private fun attemptReconnect() {
        // Minor 1 (wave-final review): declared outside the try only so the catch block below can
        // still report which episode id (if any) this attempt got as far as minting — the mint
        // itself and its own diag.log now happen INSIDE the try, matching startStreaming()'s own
        // fail-soft contract (its equivalent id-mint/preamble-prep work is likewise inside the
        // guarded block). `ids.newId()` is a caller-supplied [EpisodeIdFactory]; a throwing
        // implementation must degrade the same as every other failure here (offline + a re-armed
        // backoff) rather than escaping to the caller — staying null here means it never got that
        // far.
        var newId: String? = null
        try {
            newId = nextEpisodeId()
            diag.log("info", "EpisodeWs", "reconnect attempt: episode=$newId")
            val client = wsFactory.create()
            val token = wsListenerGeneration.incrementAndGet()
            // Same write-before-open() ordering as startStreaming() (see its own comment for the
            // hazard this avoids) for ws/online specifically: publish them BEFORE calling open(),
            // so a synchronous (or fast-racing) onFailure from the listener is always the last
            // writer for those two fields.
            synchronized(lock) { ws = client; online = true }
            client.open(newId, buildListener(token))
            sendCompanionHelloIfApplicable(client)
            // Reached only if open() itself didn't throw synchronously (a throw here is caught
            // below, using reconnectPolicy's state as of BEFORE this attempt — see the catch
            // block). Reset the backoff ladder now, but NOT unconditionally: only if `online` is
            // STILL true, i.e. no listener onFailure has already raced in (synchronously, during
            // the open() call above) and armed a fresh backoff of its own via
            // goOfflineAndArmReconnectIfCurrent(). Resetting unconditionally here would silently
            // wipe out that just-armed backoff and — worse — leave `online=false` with no scheduled
            // retry at all, the exact "permanently stranded offline" failure mode P4-5 exists to
            // fix. See SentinelControllerTest's dedicated race coverage for this exact guard.
            val stillOnline = synchronized(lock) {
                if (online) reconnectPolicy.reset()
                online
            }
            if (stillOnline) {
                // Review fix (P4-5 round 2): the earliest honest "this attempt didn't immediately
                // die" signal available (no onOpen callback exists in this design — see class
                // KDoc) — same optimism as `online = true` itself. Telemetry needs this positive
                // confirmation as much as the failure logs below: an agent debugging a device
                // session must be able to see a reconnect actually landed, not just that one was
                // attempted.
                diag.log("info", "EpisodeWs", "reconnect succeeded: episode=$newId")
            }
        } catch (t: Throwable) {
            diag.log("error", "EpisodeWs", "reconnect attempt failed: episode=$newId: $t")
            val delay = goOfflineAndArmReconnect()
            diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
        }
    }

    /**
     * Marks the current episode offline and arms [reconnectPolicy]'s next backoff attempt — the
     * choke point for every CORE-THREAD-initiated place `online` flips false mid-episode (a
     * synchronous send failure in [streamWindow]/[trySendHr], or a failed connect/reconnect
     * attempt in [startStreaming]/[attemptReconnect]'s own catch block). Returns the delay just
     * armed (ms) purely so callers can diag-log it. Must NOT be called while already holding
     * [lock] — it takes the lock itself.
     *
     * NOT used by the WS listener's own `onFailure` (a *different* thread) — see
     * [goOfflineAndArmReconnectIfCurrent] for why that caller needs its currency check and its
     * mutation to be one atomic operation rather than composed from this method plus a separate
     * pre-check.
     */
    private fun goOfflineAndArmReconnect(): Long = synchronized(lock) {
        ws = null
        online = false
        reconnectPolicy.onFailure(nowMsSupplier())
    }

    /**
     * Review fix (P4-5 round 2): the listener-safe variant of [goOfflineAndArmReconnect] — checks
     * [wsListenerGeneration] against [token] ATOMICALLY INSIDE [lock], immediately before mutating
     * anything, rather than via an earlier `isCurrent()`-style pre-check performed outside the
     * lock. The pre-check version had a genuine check-then-act race: a concurrent [disarm] can
     * complete — including its own generation bump, now also inside `lock` (see its own comment)
     * — in the window between an outer currency check and this method's lock acquisition, letting
     * a callback that read "still current" a moment ago go on to mutate state disarm() had just
     * cleaned up (flip `online` back to false, re-arm a backoff for an episode that no longer
     * exists). Mirrors [app.gauge.wear.haptics.PulseChainGate]'s own documented requirement:
     * currency must be checked immediately before EVERY side effect, under the same
     * synchronization domain as whatever invalidates it. Returns the delay armed, or `null` if
     * [token] was already stale (nothing was mutated).
     */
    private fun goOfflineAndArmReconnectIfCurrent(token: Int): Long? = synchronized(lock) {
        if (wsListenerGeneration.get() != token) return@synchronized null
        ws = null
        online = false
        reconnectPolicy.onFailure(nowMsSupplier())
    }

    private fun streamWindow(pcmBytes: ByteArray, obs: Observation) {
        emitPulseIfDue(obs)
        // P4-5: consult the reconnect backoff BEFORE this window's own send — a due attempt that
        // succeeds here means this same window's PCM flows on the fresh client below rather than
        // falling back to the offline ladder for one more window than necessary.
        maybeAttemptReconnect()

        val (client, isOnline) = synchronized(lock) { ws to online }
        // A send failure (e.g. the socket died between the read above and this call) degrades the
        // same as an already-known-offline episode — never let it propagate out of tick().
        val stillOnline = if (client == null) {
            isOnline
        } else {
            try {
                client.sendPcmWindow(pcmBytes)
                isOnline
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "sendPcmWindow failed: $t")
                val delay = goOfflineAndArmReconnect()
                diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
                false
            }
        }
        if (!stillOnline) {
            val newLevel = nudgeStateMachine.onLocalLoudness(obs.dbOverBaseline, windowIndex.toDouble())
            if (newLevel != null) {
                // Synthetic, offline-only nudge — spec: "no fabricated events in state beyond
                // channel levels", so this never touches lastVector (reserved for real
                // server-reported vector events via onVectorEvent) and never fabricates a
                // `vectors` label (HapticDirector never reads it; leaving it empty keeps this
                // honestly local rather than claiming a detection we didn't make).
                // HapticDirector isn't internally thread-safe (see its KDoc) — both call sites
                // that invoke it (here, and the WS listener's onNudge below) hold `lock`.
                synchronized(lock) {
                    channelLevels["A"] = newLevel
                    playChannelHapticIfApplicable(
                        NudgeEvent(channel = "A", level = newLevel, t = windowIndex.toDouble(), vectors = emptyList()),
                    )
                }
            }
        }
    }

    /**
     * Computes this window's pulse-train verdict ([PulseEngine.onWindow]) and publishes it as
     * [activePulse] — driven purely off this window's own [Observation], never off server
     * round-trips, and unconditional on [online]/offline status. Review fix (P4-3 round 2): does
     * NOT itself call [HapticDirector.playPulse] — see this class's own KDoc for why physically
     * playing the pulse is [app.gauge.wear.service.SentinelService]'s job alone now. Fail-soft
     * (Task 7 posture): a throwing [pulseEngine] must never propagate out of
     * [streamWindow]/[processWindow].
     */
    private fun emitPulseIfDue(obs: Observation) {
        val pulse = try {
            pulseEngine.onWindow(obs.dbOverBaseline, pulseThresholdDbFor(mode))
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "pulseEngine.onWindow failed: $t")
            null
        }
        activePulse = pulse
    }

    /**
     * Review fix (P4-3 round 2, plan-owner ratified): the pulse train's own threshold, distinct
     * from [Mode.params]' [app.gauge.shared.sentinel.ModeParams.triggerDbOverBaseline] that
     * [SentinelDetector]/[updateMeter] use for their own "triggered"/"over" decisions. [Mode.
     * SESSION]'s own trigger bar is 0.0 (it streams unconditionally — there's no trigger concept
     * at all, see [app.gauge.shared.sentinel.ModeParams] KDoc), which would make the pulse train
     * fire on almost every voiced window during ordinary conversation. SESSION borrows STANDARD's
     * +6dB bar here instead, so a pulse still means "the wearer is noticeably louder than their
     * own baseline," not "the wearer is speaking at all." BATTERY_SAVER/STANDARD are unaffected —
     * this is purely a SESSION carve-out.
     */
    private fun pulseThresholdDbFor(m: Mode): Double =
        if (m == Mode.SESSION) Mode.STANDARD.params().triggerDbOverBaseline else m.params().triggerDbOverBaseline

    /**
     * P4-3 / I2 (round 2, plan-owner ratified — narrowed from the original blanket rule): channel
     * A's old discrete level-pattern haptic ([HapticPatterns] via [HapticDirector.onNudge]) is
     * replaced by the local pulse train ([emitPulseIfDue]) only while that train is actually,
     * currently covering channel A — see [pulseTrainActivelyCovering] — not merely "whenever
     * pulses are enabled." The original blanket suppression (any channel-A nudge, pulses enabled
     * or not, always silenced) was too broad: a channel-A nudge that arrives during a quiet
     * stretch — no pulse recently emitted, e.g. a future interrupting/airtime vector that isn't
     * volume-driven — would have gone both unsuppressed-by-a-pulse AND unplayed, a silent gap.
     * Narrowing the suppression to "only while the train is plausibly still audible" closes that
     * gap: such a nudge now falls back to its legacy pattern instead. Channel B is never affected
     * either way. Picking "Off" for the "Pulse speed" setting still restores the legacy
     * level-pattern behavior for channel A unconditionally (pulses can't be "actively covering"
     * anything if they're off). Both call sites ([buildListener]'s `onNudge` and the offline
     * ladder branch above) already hold [lock] when calling this.
     */
    private fun playChannelHapticIfApplicable(n: NudgeEvent) {
        if (n.channel == "A" && pulseIntervalMs() != null && pulseTrainActivelyCovering()) return
        haptics.onNudge(n)
    }

    /**
     * Track 1 / PRD §6 ("single soft pulse every 2 min" / "double pulse every 1 min" /
     * "continuous escalating"): asks [HapticDirector.dueReminder] whether the wearer's current
     * channel-A level is due a repeat and, if so, plays it — under the SAME pulse-train
     * suppression rule as a fresh nudge ([playChannelHapticIfApplicable]): while the local pulse
     * train is actively covering channel A, the wearer is already feeling proportional taps, and
     * a reminder on top would just be noise. A suppressed reminder stays due and fires on the
     * first window after the train goes quiet. Holds [lock] around the director call (its
     * reminder/dedupe state isn't synchronized — class KDoc). Fail-soft like every other haptic
     * path: a throwing vibrator must never take down the tick.
     */
    private fun replayDueReminder() {
        try {
            synchronized(lock) {
                val due = haptics.dueReminder() ?: return
                if (pulseIntervalMs() != null && pulseTrainActivelyCovering()) return
                haptics.replayReminder(due)
            }
        } catch (t: Throwable) {
            diag.log("error", "SentinelController", "reminder replay failed: $t")
        }
    }

    /**
     * I2: true only when [pulseEngine] emitted a pulse recently enough that it's still plausibly
     * audible/covering channel A right now — "recently enough" being within `2 ×` the
     * [PulseEngine.lastEffectiveIntervalMs] that governed that last pulse (double its own cadence,
     * so a single missed/gated window right at the boundary doesn't spuriously flip this to
     * "not covering" mid-stream). `false` whenever no pulse has been emitted yet this streak
     * ([PulseEngine.lastPulseAtMs] `null` — pulses off, never over threshold, or reset by a
     * below-threshold window) — see [PulseEngine.lastPulseAtMs] KDoc for the exact reset contract
     * this mirrors.
     */
    private fun pulseTrainActivelyCovering(): Boolean {
        val lastAt = pulseEngine.lastPulseAtMs() ?: return false
        val lastEffectiveInterval = pulseEngine.lastEffectiveIntervalMs() ?: return false
        return nowMsSupplier() - lastAt <= 2 * lastEffectiveInterval
    }

    private fun endEpisodeStream() {
        // Track 1: the episode is over, so its level is over — the server sends no de-escalation
        // frames once the socket closes, and without this a level-3 wearer would keep getting the
        // "you're in the red" ramp every 10 s through COOLDOWN and back into ARMED.
        synchronized(lock) { haptics.clearReminder() }
        val client = synchronized(lock) { ws }
        if (client != null && !wsEndSent) {
            try {
                client.end()
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "end failed: $t")
                synchronized(lock) { ws = null; online = false }
            }
            wsEndSent = true
        }
        // Keep the client reference alive until live_session_saved or failure drops it (see listener).
    }

    /**
     * P4-5: [token] is this client's [wsListenerGeneration] snapshot, minted by the caller
     * ([startStreaming]/[attemptReconnect]) right before [EpisodeWs.open]. Review fix (P4-5 round
     * 2): every callback below re-checks [wsListenerGeneration] against [token] ATOMICALLY INSIDE
     * `lock`, immediately before the mutation it guards — never via a standalone check performed
     * before acquiring the lock. An earlier version checked currency first and mutated in a
     * separate, later lock acquisition; that's a genuine check-then-act race (a concurrent
     * [disarm] — which now also bumps [wsListenerGeneration] inside its own lock, see its own
     * comment — could complete entirely in the gap between the two, letting a callback that read
     * "still current" a moment ago mutate state disarm() had just cleaned up). So a callback from
     * a client already superseded by a later reconnect (or invalidated by [disarm]) self-neuters
     * instead of clobbering that newer state — mirrors [app.gauge.wear.haptics.PulseChainGate]'s
     * own documented requirement to check immediately before EVERY side effect.
     */
    private fun buildListener(token: Int): EpisodeWsClient.Listener = object : EpisodeWsClient.Listener {
        override fun onVectorEvent(e: VectorEvent) {
            try {
                synchronized(lock) {
                    if (wsListenerGeneration.get() == token) lastVector = e.vector
                }
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "onVectorEvent failed: $t")
            }
        }

        override fun onNudge(n: NudgeEvent) {
            // Holds `lock` around the HapticDirector call too — see the matching comment in
            // streamWindow() for why (HapticDirector's own dedupe state isn't synchronized).
            // Every callback here runs on OkHttp's reader thread (see EpisodeWsClient's own
            // KDoc) — an exception escaping any of them (including from HapticDirector/the
            // vibrator port) must never propagate back into OkHttp.
            try {
                synchronized(lock) {
                    if (wsListenerGeneration.get() == token) {
                        channelLevels[n.channel] = n.level
                        playChannelHapticIfApplicable(n)
                    }
                }
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "onNudge failed: $t")
            }
        }

        override fun onEpisodeSaved(id: String) {
            try {
                synchronized(lock) {
                    if (wsListenerGeneration.get() == token) ws = null
                }
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "onEpisodeSaved failed: $t")
            }
        }

        override fun onFailure(t: Throwable) {
            try {
                val delay = goOfflineAndArmReconnectIfCurrent(token)
                if (delay == null) {
                    diag.log("info", "EpisodeWs", "stale ws failure ignored (superseded): ${t::class.simpleName}")
                    return
                }
                diag.log("error", "EpisodeWs", "ws failure: ${t::class.simpleName}: ${t.message}")
                diag.log("info", "EpisodeWs", "reconnect backoff armed: next attempt in ${delay}ms")
            } catch (inner: Throwable) {
                diag.log("error", "SentinelController", "onFailure handling failed: $inner")
            }
        }

        override fun onClosed() {
            try {
                val wasCurrent = synchronized(lock) {
                    val current = wsListenerGeneration.get() == token
                    if (current) ws = null
                    current
                }
                if (wasCurrent) diag.log("info", "EpisodeWs", "ws closed")
            } catch (t: Throwable) {
                diag.log("error", "SentinelController", "onClosed failed: $t")
            }
        }
    }

    // --- helpers -------------------------------------------------------------------------------

    private fun applyPendingModeIfAny() {
        val m = pendingMode ?: return
        pendingMode = null
        mode = m
        rebuildForMode(m)
    }

    private fun rebuildForMode(m: Mode) {
        detector = SentinelDetector(m.params().triggerDbOverBaseline)
        stateMachine = SentinelStateMachine(m)
        ringBuffer = RingBuffer(capacityBytesFor(m))
        hrTracker = HrTracker()
        movementTracker = MovementTracker()
        speakingRateTracker = SpeakingRateTracker()
        windowIndex = 0
        wsEndSent = false
        // P4-5: belt-and-suspenders alongside disarm()/startStreaming()'s own resets — every path
        // that rebuilds episode-scoped state should leave no reconnect backoff armed for a
        // not-yet-started episode.
        synchronized(lock) { reconnectPolicy.reset() }
    }

    /** v0.3.1: the armed sparkline follows the SELECTED signal (wearer feedback overriding
     * v0.2.4's loudness-only decision). Volume records dbOverBaseline (unchanged scale);
     * every other signal records the just-computed meter's raw value, skipping windows with
     * no honest reading (never a fake 0). A selection change clears the series — mixed-unit
     * points must never coexist (same rule as PreviewMeterEngine.emit). */
    private fun recordSparkline(selected: SignalKind, dbOverBaseline: Double) {
        if (selected != sparklineKind) {
            sparkline.clear()
            sparklineKind = selected
        }
        val value = when (selected) {
            SignalKind.VOLUME -> dbOverBaseline
            else -> meter?.value ?: return
        }
        sparkline.addLast(value)
        while (sparkline.size > SPARKLINE_LENGTH) sparkline.removeFirst()
    }

    private fun sendInSlices(client: EpisodeWs, data: ByteArray) {
        var offset = 0
        while (offset < data.size) {
            val end = min(offset + MAX_SLICE_BYTES, data.size)
            client.sendPcmWindow(data.copyOfRange(offset, end))
            offset = end
        }
    }

    private fun pcm16LittleEndian(window: ShortArray): ByteArray {
        val bytes = ByteArray(window.size * 2)
        for (i in window.indices) {
            val v = window[i].toInt()
            bytes[i * 2] = (v and 0xFF).toByte()
            bytes[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
        }
        return bytes
    }

    private fun capacityBytesFor(m: Mode): Int = m.params().bufferSeconds * SAMPLE_RATE_HZ * BYTES_PER_SAMPLE

    companion object {
        private const val SAMPLE_RATE_HZ = 16000
        private const val BYTES_PER_SAMPLE = 2
        private const val MAX_SLICE_BYTES = 32000
        private const val SPARKLINE_LENGTH = 30

        /** Matches [app.gauge.wear.prefs.GaugePrefs]'s own "250" default for the "Pulse speed"
         * setting — kept here too so a caller that doesn't wire a prefs-backed supplier (e.g. a
         * test) still gets pulses on by default, same posture as [selectedSignal]'s VOLUME default. */
        const val DEFAULT_PULSE_INTERVAL_MS = 250L

        /** Tier B: companion JSON-heartbeat cadence — matches [EpisodeWsClient]'s own OkHttp
         * `pingInterval` (20 s), the existing keepalive rhythm of this socket. */
        const val HEARTBEAT_INTERVAL_MS = 20_000L
    }
}
