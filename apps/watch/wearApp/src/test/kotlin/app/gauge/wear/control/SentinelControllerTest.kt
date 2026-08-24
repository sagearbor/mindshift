package app.gauge.wear.control

import app.gauge.shared.NudgeEvent
import app.gauge.shared.sentinel.Dsp
import app.gauge.shared.sentinel.Mode
import app.gauge.shared.sentinel.SentinelState
import app.gauge.shared.sentinel.params
import app.gauge.shared.signals.SignalAvailability
import app.gauge.shared.signals.SignalKind
import app.gauge.wear.FakeScalar
import app.gauge.wear.ScriptedMic
import app.gauge.wear.haptics.FakeVibratorPort
import app.gauge.wear.haptics.HapticDirector
import app.gauge.wear.haptics.HapticPatterns
import app.gauge.wear.haptics.Pulse
import app.gauge.wear.haptics.PulseEngine
import app.gauge.wear.net.EpisodeWsClient
import app.gauge.wear.silence
import app.gauge.wear.tone
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Little-endian PCM16 byte count for one [tone]/[silence] window: 16000 samples * 2 bytes. */
private const val WINDOW_BYTES = 32000

/** Records every frame sent and every lifecycle call; exposes the injected [EpisodeWsClient.Listener] so tests
 * can synchronously fire server callbacks (per Task 7 brief: "tests may call listener injections synchronously"). */
class FakeWs : EpisodeWs {
    var openedEpisodeId: String? = null
        private set
    var listener: EpisodeWsClient.Listener? = null
        private set
    val sentWindows = mutableListOf<ByteArray>()
    val hrSent = mutableListOf<Pair<Double, Double>>()
    var endCallCount = 0
        private set
    var cancelCallCount = 0
        private set

    override fun open(episodeId: String, listener: EpisodeWsClient.Listener) {
        openedEpisodeId = episodeId
        this.listener = listener
    }

    override fun sendPcmWindow(bytes: ByteArray) {
        sentWindows.add(bytes)
    }

    override fun sendHr(bpm: Double, t: Double) {
        hrSent.add(bpm to t)
    }

    override fun end() {
        endCallCount++
    }

    override fun cancel() {
        cancelCallCount++
    }
}

/** Creates and records a fresh [FakeWs] per call, mirroring the real "one client per episode" contract. */
class FakeWsFactory : WsFactory {
    val created = mutableListOf<FakeWs>()
    override fun create(): EpisodeWs = FakeWs().also { created.add(it) }
}

/** Worst-case async-failure race double: [open] synchronously invokes `listener.onFailure` before
 * returning — a connection failure so fast it beats even a fully-synchronous caller back to the
 * return point. This is the sharpest version of the `startStreaming()` write-ordering hazard: any
 * implementation that publishes `ws`/`online` *after* calling `open()` will have this failure's
 * writes clobbered by the post-open() write, however small the window. */
class FailingFakeWs : EpisodeWs {
    var listener: EpisodeWsClient.Listener? = null
        private set
    val sentWindows = mutableListOf<ByteArray>()

    override fun open(episodeId: String, listener: EpisodeWsClient.Listener) {
        this.listener = listener
        listener.onFailure(java.io.IOException("immediate connection refusal"))
    }

    override fun sendPcmWindow(bytes: ByteArray) {
        sentWindows.add(bytes)
    }

    override fun sendHr(bpm: Double, t: Double) = Unit
    override fun end() = Unit
    override fun cancel() = Unit
}

class FailingFakeWsFactory : WsFactory {
    val created = mutableListOf<FailingFakeWs>()
    override fun create(): EpisodeWs = FailingFakeWs().also { created.add(it) }
}

/** Task 7: simulates a factory whose [create] itself throws (e.g. missing INTERNET permission
 * surfacing as a SecurityException from the OkHttpClient/socket layer before any WS work starts).
 * P4-5 round 2: tracks [callCount] so reconnect tests can assert an attempt actually FIRED (a
 * second [create] call happened) without needing a [create] call that ever succeeds — every call
 * throws, but a growing [callCount] still proves the armed backoff didn't just silently vanish. */
class ThrowingWsFactory : WsFactory {
    var callCount = 0
        private set
    override fun create(): EpisodeWs {
        callCount++
        throw SecurityException("Permission denied (missing INTERNET permission?)")
    }
}

/** Task 7: simulates a WS whose [open] itself throws synchronously, distinct from [FailingFakeWs]
 * (which opens fine and then reports failure via the listener). */
class OpenThrowingFakeWs : EpisodeWs {
    override fun open(episodeId: String, listener: EpisodeWsClient.Listener) {
        throw IllegalStateException("open failed")
    }

    override fun sendPcmWindow(bytes: ByteArray) = Unit
    override fun sendHr(bpm: Double, t: Double) = Unit
    override fun end() = Unit
    override fun cancel() = Unit
}

class OpenThrowingFakeWsFactory : WsFactory {
    override fun create(): EpisodeWs = OpenThrowingFakeWs()
}

/** Task 7 round 2: throws on the FIRST [create] call only (simulating a transient failure, e.g. a
 * permission that gets granted between episode 1 and episode 2), then produces real [FakeWs]
 * instances on every subsequent call — lets a test drive one failed episode followed by one that
 * actually opens, to check the ring buffer doesn't bleed stale audio across that boundary. */
class ThrowOnceThenWorkingWsFactory : WsFactory {
    private var thrown = false
    val created = mutableListOf<FakeWs>()
    override fun create(): EpisodeWs {
        if (!thrown) {
            thrown = true
            throw SecurityException("first attempt fails")
        }
        return FakeWs().also { created.add(it) }
    }
}

/** Task 7 round 2: overrides [HapticDirector.onNudge] to throw directly, bypassing that class's own
 * internal `runCatching` around `vibrator.vibrate(...)` entirely — proves SentinelController's own
 * listener-callback containment (not HapticDirector's) is what's catching this. */
class ThrowingHapticDirector(vibrator: VibratorPort) : HapticDirector(vibrator, nowMs = { 0L }) {
    override fun onNudge(n: NudgeEvent) {
        throw RuntimeException("haptics blew up")
    }
}

/** P4-5: succeeds on every [create] call except call number [failOnCall] (1-indexed), which
 * throws — lets a test drive a successful episode trigger, then a failed reconnect attempt (or
 * vice versa across multiple attempts), without a WsFactory that fails every single time. */
class FailOnceAtCallWsFactory(private val failOnCall: Int) : WsFactory {
    private var callCount = 0
    val created = mutableListOf<FakeWs>()
    override fun create(): EpisodeWs {
        callCount++
        if (callCount == failOnCall) throw java.io.IOException("call $callCount refused")
        return FakeWs().also { created.add(it) }
    }
}

/** P4-5 round 2: produces a normal [FakeWs] for every call BEFORE [failFromCall], then a
 * [FailingFakeWs] (synchronously fires `listener.onFailure` from inside [EpisodeWs.open] itself,
 * before `open()` returns — see [FailingFakeWs]'s own KDoc for why that's the sharpest possible
 * race) for every call at/after it. Lets a test drive a normal first episode, then a reconnect
 * attempt that races a synchronous listener failure exactly the way a real fast connection
 * refusal can — this is what exercises [SentinelController.attemptReconnect]'s "only reset the
 * backoff ladder if still online after open() returns" guard, which no other existing fake reaches
 * (a create()-throw skips that line entirely; a plain non-racing [FakeWs] never fires synchronously
 * at all). */
class FailingFromCallWsFactory(private val failFromCall: Int) : WsFactory {
    var callCount = 0
        private set
    var firstCreated: FakeWs? = null
        private set
    override fun create(): EpisodeWs {
        callCount++
        return if (callCount >= failFromCall) {
            FailingFakeWs()
        } else {
            FakeWs().also { if (firstCreated == null) firstCreated = it }
        }
    }
}

/** Sequential episode ids: "e1", "e2", ... — lets tests assert uniqueness across episodes. */
class FakeEpisodeIdFactory : EpisodeIdFactory {
    private var counter = 0
    val issued = mutableListOf<String>()
    override fun newId(): String {
        counter++
        return "e$counter".also { issued.add(it) }
    }
}

class SentinelControllerTest {

    // Set by `controller()`'s WsFactory on the (single, per-test) episode it creates — Task 10's
    // hr/meter tests only ever drive one episode each, so "the most recently created FakeWs" is
    // unambiguous.
    private lateinit var fakeWs: FakeWs

    /** Task 10: lightweight helper for the hr/accel/meter tests — hides mic/ws/vibrator plumbing
     * behind sensible defaults so those tests can focus on the new constructor params. Distinct
     * from [newController] (kept as-is for the pre-Task-10 test suite above/below). */
    private fun controller(
        mode: Mode = Mode.STANDARD,
        hr: ScalarSource? = null,
        accel: ScalarSource? = null,
        selectedSignal: () -> SignalKind = { SignalKind.VOLUME },
    ): SentinelController {
        val mic = ScriptedMic(fillWith = silence())
        val wsFactory = object : WsFactory {
            override fun create(): EpisodeWs = FakeWs().also { fakeWs = it }
        }
        val vibrator = FakeVibratorPort()
        val haptics = HapticDirector(vibrator, nowMs = { 0L })
        return SentinelController(
            mode = mode,
            mic = mic,
            wsFactory = wsFactory,
            haptics = haptics,
            ids = FakeEpisodeIdFactory(),
            hr = hr,
            accel = accel,
            selectedSignal = selectedSignal,
        )
    }

    // --- Task 10: hr/movement/speaking-rate sources + live meter state ----------------------

    @Test
    fun meterFollowsSelectedSignal() {
        val hr = FakeScalar(72.0)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        assertEquals(SignalKind.HEART_RATE, c.state.meter!!.signal)
        assertEquals(72.0, c.state.meter!!.value)
    }

    @Test
    fun volumeMeterThresholdIsBaselinePlusTrigger() {
        val c = controller() // STANDARD: +6 dB
        c.arm(); repeat(6) { c.tick(tone(0.05)) } // baseline ~ -29 dBFS
        val m = c.state.meter!!
        assertEquals(SignalKind.VOLUME, m.signal)
        assertEquals(m.threshold!!, m.value + 6.0, 1.5) // quiet steady tone sits ~threshold-6
        assertFalse(m.over)
    }

    @Test
    fun hrSentDuringOnlineStreaming() {
        val hr = FakeScalar(95.0)
        val c = controller(hr = hr) // FakeWs records sendHr calls
        c.arm(); repeat(6) { c.tick(tone(0.05)) }; repeat(2) { c.tick(tone(0.2)) } // trigger
        c.tick(tone(0.2))
        assertTrue(fakeWs.hrSent.isNotEmpty())
    }

    @Test
    fun throwingScalarSourceIsContained() {
        val c = controller(
            hr = object : ScalarSource { override fun latest(): Double = throw IllegalStateException("sensor died") },
            selectedSignal = { SignalKind.HEART_RATE },
        )
        c.arm(); repeat(3) { c.tick(tone(0.05)) } // must not throw; meter falls back to null value semantics
        assertEquals(SentinelState.ARMED, c.state.sentinel)
    }

    @Test
    fun noHrReadingMeansNoMeterValueNotFakeValue() {
        val c = controller(hr = FakeScalar(null), selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); c.tick(tone(0.05))
        assertNull(c.state.meter) // honest: no reading, no meter — never 0.0
    }

    // --- v0.3.1: armed sparkline follows the selected signal (was hardcoded to loudness) ------

    @Test
    fun armedSparklineFollowsSelectedSignal() {
        val hr = FakeScalar(72.0)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm()
        c.tick(tone(0.05)) // any voiced window
        val state = c.state
        assertEquals(SignalKind.HEART_RATE, state.sparklineSignal)
        assertEquals(listOf(72.0), state.sparkline) // bpm, NOT dbOverBaseline
    }

    @Test
    fun sparklineSeriesClearsWhenSelectionChanges() {
        var selected = SignalKind.VOLUME
        val accel = FakeScalar(1.0)
        val c = controller(accel = accel, selectedSignal = { selected })
        c.arm()
        c.tick(tone(0.2))
        assertTrue(c.state.sparkline.isNotEmpty())

        selected = SignalKind.MOVEMENT
        c.tick(tone(0.2))
        val state = c.state
        assertEquals(SignalKind.MOVEMENT, state.sparklineSignal)
        // old volume points gone; at most this window's movement reading remains
        assertTrue(state.sparkline.size <= 1)
    }

    @Test
    fun armedSparklineSkipsWindowsWithNoReading() {
        val c = controller(hr = FakeScalar(null), selectedSignal = { SignalKind.HEART_RATE })
        c.arm()
        c.tick(tone(0.05))
        assertEquals(emptyList<Double>(), c.state.sparkline) // honest blank, no fake 0.0
    }

    // --- C1: disarm() must not leave a stale meter/pulse masking the preview meter ------------

    @Test
    fun disarmClearsStaleMeterAndActivePulse() {
        val c = controller() // VOLUME selectedSignal by default, pulses on by default (250ms)
        c.arm()
        repeat(6) { c.tick(tone(0.05)) } // baseline
        repeat(2) { c.tick(tone(0.2)) } // 2nd loud window triggers ARMED -> STREAMING (streamWindow() not yet called)
        c.tick(tone(0.2)) // first streamWindow() call: populates activePulse

        // Sanity: both fields are actually populated before disarm, so this test would fail red
        // without the fix rather than passing vacuously.
        assertNotNull(c.state.meter, "sanity: meter should be populated before disarm")
        assertNotNull(c.state.activePulse, "sanity: activePulse should be populated before disarm")

        c.disarm()

        assertNull(c.state.meter, "C1: a stale meter must not survive disarm() — it would mask the disarmed-state preview meter")
        assertNull(c.state.activePulse, "C1: a stale activePulse must not survive disarm()")
    }

    // --- T6 review fix: disarm() must not leave a stale sparkline trace from the PREVIOUS
    // session — missed in the original C1 pass (meter/activePulse/availability/lastWindowVoiced/
    // channelLevels were all cleared; sparkline wasn't) ------------------------------------------

    @Test
    fun disarmClearsSparklineSoARearmDoesNotShowThePreviousSessionsTrace() {
        val c = controller()
        c.arm()
        repeat(6) { c.tick(tone(0.05)) } // baseline
        repeat(3) { c.tick(tone(0.3)) } // distinctly loud windows: populates a non-trivial trace

        // Sanity: the trace is actually populated before disarm, so this test would fail red
        // without the fix rather than passing vacuously.
        assertTrue(c.state.sparkline.isNotEmpty(), "sanity: sparkline should be populated before disarm")

        c.disarm()

        assertTrue(
            c.state.sparkline.isEmpty(),
            "a stale sparkline must not survive disarm() — it would render as the CURRENT " +
                "session's trace the moment the wearer rearms",
        )

        c.arm()
        c.tick(tone(0.05))
        assertEquals(
            1,
            c.state.sparkline.size,
            "a rearm must start the trace fresh, not resume appending to the previous session's points",
        )
    }

    // --- wave-final review, Important 1: stale channelLevels must not survive disarm() or a
    // fresh episode's startStreaming(), but MUST survive a mid-episode reconnect ----------------

    @Test
    fun disarmClearsChannelLevels() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, ws opens
        val ws = wsFactory.created.single()
        ws.listener!!.onNudge(NudgeEvent(channel = "A", level = 2, t = 8.0, vectors = listOf("yelling")))
        assertEquals(2, controller.state.channelLevels["A"], "sanity: channelLevels populated before disarm")

        controller.disarm()

        assertTrue(
            controller.state.channelLevels.isEmpty(),
            "Important 1: disarm() must clear channelLevels — a stale non-empty map here would let the " +
                "complication keep publishing the last episode's level while fully DISARMED",
        )
    }

    @Test
    fun startStreamingClearsChannelLevelsFromPriorEpisode() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // episode 1: reach STREAMING, ws1 created
        val ws1 = wsFactory.created.single()
        ws1.listener!!.onNudge(NudgeEvent(channel = "A", level = 3, t = 8.0, vectors = listOf("yelling")))
        assertEquals(3, controller.state.channelLevels["A"], "sanity: episode 1's nudge populated channelLevels")

        // Drive episode 1 to natural completion: quiet stretch -> COOLDOWN -> cooldownSeconds ->
        // ARMED, with NO disarm()/arm() call in between — this is exactly the "no episode boundary
        // hook" gap Important 1 targets: a wearer who stays armed across back-to-back conversations
        // never touches disarm(), so only startStreaming() itself can clear the stale map.
        mic.enqueueMany(30, silence())
        repeat(30) { controller.tick() }
        assertEquals(SentinelState.COOLDOWN, controller.state.sentinel)
        mic.enqueueMany(10, silence())
        repeat(10) { controller.tick() }
        assertEquals(SentinelState.ARMED, controller.state.sentinel)

        // Episode 2 triggers (baseline from episode 1 persists, so 2 loud windows are enough).
        mic.enqueueMany(2, tone(0.2))
        repeat(2) { controller.tick() }
        assertEquals(SentinelState.STREAMING, controller.state.sentinel)

        // Important 1: episode 1's "Level 3" must not leak into episode 2's first STREAMING tick.
        // P4-4's complication reads channelLevels while STREAMING, so a stale map here would
        // publish episode 1's red arc immediately, before episode 2 has had a single nudge.
        assertTrue(
            controller.state.channelLevels.isEmpty(),
            "startStreaming() must clear channelLevels for a fresh episode",
        )
    }

    @Test
    fun midEpisodeReconnectPreservesChannelLevels() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, ws1 opens
        val ws1 = wsFactory.created.single()
        ws1.listener!!.onNudge(NudgeEvent(channel = "A", level = 3, t = 8.0, vectors = listOf("yelling")))
        assertEquals(3, controller.state.channelLevels["A"], "sanity: channelLevels populated before the failure")

        ws1.listener!!.onFailure(java.io.IOException("connection reset"))
        assertFalse(controller.state.online)

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // cross the armed 2s backoff boundary -> reconnect attempt fires and succeeds
        controller.tick()

        assertEquals(2, wsFactory.created.size, "the armed 2s backoff must fire a reconnect attempt")
        assertTrue(controller.state.online, "the reconnect attempt must have succeeded")
        // Deliberate carve-out (Important 1): attemptReconnect() must NOT clear channelLevels — a
        // mid-episode reconnect is the SAME conversation, so the level set before the connectivity
        // gap must still be showing on the watch face once the gap closes, not reset to zero.
        assertEquals(
            3,
            controller.state.channelLevels["A"],
            "attemptReconnect() must preserve channelLevels across a mid-episode reconnect",
        )
    }

    private fun newController(
        mode: Mode,
        mic: ScriptedMic,
        wsFactory: WsFactory = FakeWsFactory(),
        vibrator: VibratorPort = FakeVibratorPort(),
        ids: FakeEpisodeIdFactory = FakeEpisodeIdFactory(),
        diag: DiagLog = NOOP_DIAG,
        pulseIntervalMs: () -> Long? = { 250L },
        nowMs: () -> Long = { 0L },
    ): SentinelController {
        val haptics = HapticDirector(vibrator, nowMs = { 0L })
        return SentinelController(
            mode = mode,
            mic = mic,
            wsFactory = wsFactory,
            haptics = haptics,
            ids = ids,
            diag = diag,
            pulseIntervalMs = pulseIntervalMs,
            nowMs = nowMs,
        )
    }

    // The exact recipe SentinelDetectorTest's sustainedLoudnessOverBaselineTriggersOnSecondWindow uses:
    // 6 baseline-establishing windows, then the 2nd loud window is `triggered`.
    private fun quietThenTriggerWindows(): List<ShortArray> =
        List(6) { tone(0.05) } + List(2) { tone(0.2) }

    @Test
    fun armedDoesNotTransmit() {
        val mic = ScriptedMic(quietThenTriggerWindows().take(6)) // baseline only, never crosses the trigger bar
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(6) { controller.tick() }

        assertEquals(SentinelState.ARMED, controller.state.sentinel)
        assertTrue(wsFactory.created.isEmpty(), "no WS should ever be created while only quiet windows arrive")
        assertTrue(vibrator.calls.isEmpty())
    }

    @Test
    fun triggerSendsPreambleFirst() {
        val windows = quietThenTriggerWindows() // 6 quiet + 2 loud (2nd loud triggers)
        val mic = ScriptedMic(windows)
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // through the triggering window

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertEquals(1, wsFactory.created.size)
        val ws = wsFactory.created.single()

        // All 8 ARMED-state windows (none evicted: 8 * 32000 = 256000 < STANDARD's 10s = 320000-byte capacity)
        // were pushed to the ring buffer and sent as the preamble before streaming went live. Each window is
        // exactly one 32000-byte slice, so this is exactly 8 frames.
        assertEquals(8, ws.sentWindows.size)
        val preambleBytes = ws.sentWindows.sumOf { it.size }
        assertEquals(8 * WINDOW_BYTES, preambleBytes)
        ws.sentWindows.forEach { assertEquals(WINDOW_BYTES, it.size) } // slice boundary == window boundary here

        // Next tick streams live: exactly one more frame, still window-sized.
        mic.enqueue(tone(0.2))
        controller.tick()
        assertEquals(9, ws.sentWindows.size)
        assertEquals(WINDOW_BYTES, ws.sentWindows.last().size)
    }

    // --- P4-3: proportional pulse-train haptics -----------------------------------------------

    /** The exact pulse the production PulseEngine must compute for a [tone] `0.2` window
     * immediately following a baseline of six `tone(0.05)` windows under [Mode.STANDARD] — mirrors
     * the production formula ([Dsp.rmsDbfs] over baseline, minus the mode's trigger threshold, fed
     * through [PulseEngine.bandFor]) rather than hardcoding a fragile literal dB value. */
    private fun expectedPulseForLoudToneAfterQuietBaseline(): Pulse {
        val overBaseline = Dsp.rmsDbfs(tone(0.2)) - Dsp.rmsDbfs(tone(0.05))
        return PulseEngine.bandFor(overBaseline - Mode.STANDARD.params().triggerDbOverBaseline)
    }

    @Test
    fun overThresholdWindowsDriveBandedSinglePulses() {
        // Review fix (P4-3 round 2): the controller no longer calls vibrator.vibrate() itself for
        // pulses — SentinelController.playPulse's sole caller is now SentinelService's repeat
        // loop — so this asserts on ControllerState.activePulse (the verdict the controller DOES
        // publish) rather than on vibrator.calls.
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING via trigger — streamWindow() not yet called
        assertNull(controller.state.activePulse, "no pulse before the first STREAMING window is processed")
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")

        mic.enqueue(tone(0.2))
        controller.tick() // first streamWindow() call: the pulse train's first-ever window, always fires

        val expected = expectedPulseForLoudToneAfterQuietBaseline()
        assertEquals(expected, controller.state.activePulse)
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")
    }

    @Test
    fun nudgeFromServerChannelASuppressedWhilePulseTrainActivelyCovering() {
        // I2 (round 2, plan-owner ratified): the local pulse train owns channel A's haptic ONLY
        // while it's actively, currently covering it — this drives a pulse first (so the train
        // has just fired), then fires a server nudge in the same instant (default test clock is
        // constant at 0L) and asserts it's still suppressed, matching the old blanket-suppression
        // behavior for the case that's actually still valid: the train is genuinely covering.
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open — streamWindow() not yet called
        mic.enqueue(tone(0.2))
        controller.tick() // first streamWindow() call: emits a pulse via the local train
        assertNotNull(controller.state.activePulse, "sanity: the pulse train must have fired for this test to mean anything")

        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)
        listener.onNudge(NudgeEvent(channel = "A", level = 2, t = 9.0, vectors = listOf("yelling")))

        assertTrue(vibrator.calls.isEmpty(), "a channel-A nudge arriving while the pulse train is actively covering must stay suppressed")
        assertEquals(2, controller.state.channelLevels["A"])
    }

    @Test
    fun nudgeFromServerChannelAPlaysPatternWhenNoRecentPulseCovering() {
        // I2 (b): pulses are enabled, but the local train has never actually fired this episode
        // (streamWindow() hasn't been reached yet) — a channel-A nudge arriving during this quiet
        // stretch (e.g. a future interrupting/airtime vector that isn't volume-driven) must still
        // play its legacy pattern rather than going silently unsuppressed-and-unplayed.
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open — streamWindow() not yet called
        assertNull(controller.state.activePulse, "sanity: the pulse train must not have fired yet for this test to mean anything")
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onNudge(NudgeEvent(channel = "A", level = 2, t = 8.0, vectors = listOf("yelling")))

        assertEquals(1, vibrator.calls.size)
        val expected = HapticPatterns.waveformFallback("A", 2)!!
        assertTrue(vibrator.calls[0].timingsMs.contentEquals(expected.timingsMs.toLongArray()))
        assertTrue(vibrator.calls[0].amplitudes.contentEquals(expected.amplitudes.toIntArray()))
        assertEquals(2, controller.state.channelLevels["A"])
    }

    @Test
    fun nudgeFromServerChannelAPlaysPatternWhenLastPulseIsStale() {
        // I2 (b, stronger form): the train DID fire earlier, but time has since moved on well
        // past 2x that pulse's own effective interval — the train has plausibly gone quiet, so a
        // channel-A nudge arriving now must not be suppressed just because a pulse happened at
        // some point in the past.
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open — streamWindow() not yet called
        mic.enqueue(tone(0.2))
        controller.tick() // first streamWindow() call: emits a pulse at clock=0
        val emittedPulse = requireNotNull(controller.state.activePulse)

        clock += 2 * PulseEngine.effectiveIntervalMs(emittedPulse, 250L) + 1 // just past the 2x window

        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)
        listener.onNudge(NudgeEvent(channel = "A", level = 2, t = 9.0, vectors = listOf("yelling")))

        assertEquals(1, vibrator.calls.size, "a stale pulse (older than 2x its own effective interval) must not suppress channel A")
        assertEquals(2, controller.state.channelLevels["A"])
    }

    @Test
    fun channelANudgeRepeatsOnThePrdSection6ScheduleWhileStreaming() {
        // Track 1 / PRD §6: a level-2 nudge ("double pulse every 1 min") is felt once on arrival
        // and then again every minute the level holds, driven by the controller's own window tick
        // — no server re-send involved. Pulses "Off" so the pulse-train suppression rule can't
        // interfere; the director shares the controller's clock so reminders can actually come due.
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        var clock = 0L
        val haptics = HapticDirector(vibrator, nowMs = { clock })
        val controller = SentinelController(
            mode = Mode.STANDARD, mic = mic, wsFactory = wsFactory, haptics = haptics,
            ids = FakeEpisodeIdFactory(), pulseIntervalMs = { null }, nowMs = { clock },
        )

        controller.arm()
        repeat(8) { controller.tick() } // STREAMING, WS open
        val listener = requireNotNull(wsFactory.created.single().listener)
        listener.onNudge(NudgeEvent(channel = "A", level = 2, t = 9.0, vectors = listOf("yelling")))
        assertEquals(1, vibrator.calls.size, "felt once on arrival")

        clock = 59_000
        controller.tick()
        assertEquals(1, vibrator.calls.size, "not yet due")

        clock = 60_000
        controller.tick()
        assertEquals(2, vibrator.calls.size, "repeated at the 1 min cadence by the window tick")

        clock = 120_000
        controller.tick()
        assertEquals(3, vibrator.calls.size)

        // Disarm ends the conversation: no more reminders, however long we wait.
        controller.disarm()
        clock = 10_000_000
        controller.arm()
        controller.tick()
        assertEquals(3, vibrator.calls.size, "a disarmed/re-armed sentinel forgets the old level")
    }

    @Test
    fun nudgeFromServerChannelBStillVibrates() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onNudge(NudgeEvent(channel = "B", level = 2, t = 8.0, vectors = listOf("interrupting")))

        assertEquals(1, vibrator.calls.size)
        val expected = HapticPatterns.waveformFallback("B", 2)!!
        assertTrue(vibrator.calls[0].timingsMs.contentEquals(expected.timingsMs.toLongArray()))
        assertTrue(vibrator.calls[0].amplitudes.contentEquals(expected.amplitudes.toIntArray()))
        assertEquals(2, controller.state.channelLevels["B"])
    }

    @Test
    fun pulseIntervalOffRestoresLegacyChannelALevelPatternHaptic() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory,
            vibrator,
            pulseIntervalMs = { null }, // "Off" pref
        )

        controller.arm()
        repeat(8) { controller.tick() }
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onNudge(NudgeEvent(channel = "A", level = 2, t = 8.0, vectors = listOf("yelling")))

        assertEquals(1, vibrator.calls.size)
        val expected = HapticPatterns.waveformFallback("A", 2)!!
        assertTrue(vibrator.calls[0].timingsMs.contentEquals(expected.timingsMs.toLongArray()))
        assertTrue(vibrator.calls[0].amplitudes.contentEquals(expected.amplitudes.toIntArray()))
        assertEquals(2, controller.state.channelLevels["A"])
    }

    @Test
    fun sessionModeUsesStandardThresholdForPulses() {
        // Review fix (P4-3 round 2, plan-owner ratified): Mode.SESSION's own trigger bar is 0.0
        // (it streams unconditionally — see ModeParams' KDoc), which would make the pulse train
        // fire on nearly every voiced window during ordinary conversation if it used that bar
        // directly. The pulse train borrows STANDARD's +6dB bar for SESSION instead.
        val quiet = tone(0.05)
        val ordinary = tone(0.0629465) // ~+2dB over `quiet`'s baseline — ordinary conversation
        val loud = tone(0.1119361) // ~+7dB over `quiet`'s baseline — noticeably louder than usual

        val baselineDbfs = Dsp.rmsDbfs(quiet)
        val ordinaryOver = Dsp.rmsDbfs(ordinary) - baselineDbfs
        val loudOver = Dsp.rmsDbfs(loud) - baselineDbfs
        // Sanity-check the fixture actually straddles STANDARD's 6.0dB bar the way the test name
        // and comments above claim, rather than silently testing nothing.
        assertTrue(ordinaryOver < 6.0, "fixture should be below STANDARD's threshold, was $ordinaryOver")
        assertTrue(loudOver >= 6.0, "fixture should be at/above STANDARD's threshold, was $loudOver")

        val mic = ScriptedMic(List(3) { quiet }) // 3 windows: exactly enough to seed SESSION's baseline
        val controller = newController(Mode.SESSION, mic, FakeWsFactory())

        controller.arm() // SESSION jumps straight to STREAMING
        repeat(3) { controller.tick() } // seed the baseline at `quiet`'s dbfs

        controller.tick(ordinary)
        assertNull(controller.state.activePulse, "ordinary conversation (+2dB) must not pulse in SESSION mode")

        controller.tick(loud)
        assertEquals(PulseEngine.bandFor(loudOver - 6.0), controller.state.activePulse)
    }

    // --- Task 12.5: log ws failure/close reasons to diag/telemetry --------------------------

    @Test
    fun onFailureLogsReasonToDiag() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val diag = mutableListOf<String>()
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory = wsFactory,
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
        )

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onFailure(java.io.IOException("boom"))

        assertFalse(controller.state.online)
        assertTrue(
            diag.any { it.startsWith("error/EpisodeWs") && it.contains("IOException") && it.contains("boom") },
            "expected an error/EpisodeWs entry naming IOException and its message, got: $diag",
        )
    }

    @Test
    fun onClosedLogsToDiag() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val diag = mutableListOf<String>()
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory = wsFactory,
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
        )

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onClosed()

        assertTrue(
            diag.any { it.startsWith("info/EpisodeWs") && it.contains("ws closed") },
            "expected an info/EpisodeWs \"ws closed\" entry, got: $diag",
        )
    }

    @Test
    fun wsFailureFallsBackToLocalNudges() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val vibrator = FakeVibratorPort()
        val controller = newController(Mode.STANDARD, mic, wsFactory, vibrator)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        listener.onFailure(java.io.IOException("connection reset"))
        assertFalse(controller.state.online)
        val sentBeforeFailureFollowup = ws.sentWindows.size

        // Subsequent loud ticks (same +12dB-over-baseline tone as the trigger) must still drive the
        // offline NudgeStateMachine ladder (+12dB over baseline -> level 2) for channelLevels/UI
        // purposes, without touching the dead WS and without throwing. Channel-A haptics themselves
        // no longer come from that ladder (P4-3) — the local pulse train (driven directly off this
        // window's own Observation, online or offline) is what actually pulses now. Review fix
        // (P4-3 round 2): the controller only ever PUBLISHES the pulse verdict (activePulse) — it
        // never calls the vibrator itself for pulses (SentinelService's repeat loop does), so
        // that's what this asserts on rather than vibrator.calls.
        mic.enqueue(tone(0.2))
        controller.tick()

        assertEquals(2, controller.state.channelLevels["A"], "offline ladder still tracks channel level")
        val expected = expectedPulseForLoudToneAfterQuietBaseline()
        assertEquals(expected, controller.state.activePulse, "local pulse train should still compute a verdict while offline")
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")
        // No new frames sent to the (now-dead) WS after failure.
        assertEquals(sentBeforeFailureFollowup, ws.sentWindows.size)
    }

    @Test
    fun quietStretchEndsEpisode() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING
        val ws = wsFactory.created.single()

        // SentinelStateMachine's default quietSecondsToStop is 30: 30 consecutive non-loud windows to flip to COOLDOWN.
        mic.enqueueMany(30, silence())
        repeat(30) { controller.tick() }

        assertEquals(SentinelState.COOLDOWN, controller.state.sentinel)
        assertEquals(1, ws.endCallCount)

        // Further quiet ticks while COOLDOWN counts down must not call end() again.
        mic.enqueueMany(5, silence())
        repeat(5) { controller.tick() }
        assertEquals(1, ws.endCallCount)
    }

    @Test
    fun sessionModeStreamsWithoutTrigger() {
        val mic = ScriptedMic()
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.SESSION, mic, wsFactory)

        controller.arm()
        assertEquals(SentinelState.STREAMING, controller.state.sentinel) // SESSION jumps straight to STREAMING

        mic.enqueue(tone(0.05))
        controller.tick()

        assertEquals(1, wsFactory.created.size)
        val ws = wsFactory.created.single()
        assertEquals(1, ws.sentWindows.size) // first tick already streams live — no trigger wait, no preamble
        assertEquals(WINDOW_BYTES, ws.sentWindows.single().size)
    }

    @Test
    fun micLossDisarmsCleanly() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING
        val ws = wsFactory.created.single()

        mic.enqueue(null) // mic gone
        controller.tick()

        assertEquals(SentinelState.DISARMED, controller.state.sentinel)
        assertEquals(1, ws.cancelCallCount)

        // A subsequent tick while disarmed must not crash or read the mic again.
        controller.tick()
        assertEquals(SentinelState.DISARMED, controller.state.sentinel)
    }

    @Test
    fun freshWsAttemptOnNextEpisodeAfterMidEpisodeFailure() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // episode 1: reach STREAMING, ws1 created
        assertEquals(1, wsFactory.created.size)
        wsFactory.created.single().listener!!.onFailure(java.io.IOException("connection reset"))
        assertFalse(controller.state.online)

        // Let episode 1 run to natural completion: quiet stretch -> COOLDOWN -> (cooldownSeconds
        // default 10) -> back to ARMED.
        mic.enqueueMany(30, silence())
        repeat(30) { controller.tick() }
        assertEquals(SentinelState.COOLDOWN, controller.state.sentinel)
        mic.enqueueMany(10, silence())
        repeat(10) { controller.tick() }
        assertEquals(SentinelState.ARMED, controller.state.sentinel)

        // Episode 2 triggers (baseline from episode 1 persists, so 2 loud windows are enough).
        // Even though `online` was left false by episode 1's failure, this must be a completely
        // fresh WS attempt, not a skip.
        mic.enqueueMany(2, tone(0.2))
        repeat(2) { controller.tick() }

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertEquals(2, wsFactory.created.size, "episode 2 must attempt a fresh WS, not skip because episode 1 failed")
        assertTrue(controller.state.online, "a freshly (successfully-opened, no failure yet) attempted episode reports online")
    }

    @Test
    fun freshWsAttemptAfterFailureThenManualDisarmRearm() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // episode 1: reach STREAMING, ws1 created
        assertEquals(1, wsFactory.created.size)
        wsFactory.created.single().listener!!.onFailure(java.io.IOException("connection reset"))
        assertFalse(controller.state.online)

        // Abrupt manual stop/restart instead of letting the episode finish naturally — this is
        // the "stuck offline forever" case: online must not get wedged false across this cycle.
        controller.disarm()
        controller.arm()

        mic.enqueueMany(2, tone(0.2)) // baseline persists (detector isn't reset by disarm/arm)
        repeat(2) { controller.tick() }

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertEquals(2, wsFactory.created.size, "re-armed episode must attempt a fresh WS, not stay stuck offline")
    }

    @Test
    fun setModeAppliesOnNaturalCooldownToArmedCycle() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val controller = newController(Mode.STANDARD, mic, wsFactory)

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING under STANDARD
        assertEquals(SentinelState.STREAMING, controller.state.sentinel)

        controller.setMode(Mode.BATTERY_SAVER) // queued: controller is not DISARMED
        assertEquals(Mode.STANDARD, controller.state.mode, "queued mode must not apply mid-episode")

        // Drive the episode to natural completion: quiet stretch -> COOLDOWN -> cooldownSeconds -> ARMED.
        mic.enqueueMany(30, silence())
        repeat(30) { controller.tick() }
        assertEquals(SentinelState.COOLDOWN, controller.state.sentinel)
        mic.enqueueMany(10, silence())
        repeat(10) { controller.tick() }

        assertEquals(SentinelState.ARMED, controller.state.sentinel)
        assertEquals(Mode.BATTERY_SAVER, controller.state.mode, "queued mode must apply on the natural COOLDOWN->ARMED cycle")
    }

    @Test
    fun asyncFailureRacingOpenReturnStillDisablesOnline() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FailingFakeWsFactory()
        val vibrator = FakeVibratorPort()
        val haptics = HapticDirector(vibrator, nowMs = { 0L })
        val controller = SentinelController(
            mode = Mode.STANDARD,
            mic = mic,
            wsFactory = wsFactory,
            haptics = haptics,
            ids = FakeEpisodeIdFactory(),
        )

        controller.arm()
        repeat(8) { controller.tick() } // trigger; FailingFakeWs.open() synchronously fires onFailure mid-startStreaming()

        // If startStreaming() publishes ws/online AFTER calling open(), that post-open() write
        // clobbers onFailure's ws=null/online=false back to a "healthy" (but actually dead)
        // client — silently defeating honest degradation. onFailure must win.
        assertFalse(controller.state.online, "an onFailure that races open() must still end up disabling online")

        mic.enqueue(tone(0.2))
        controller.tick()

        // Review fix (P4-3 round 2): the controller only publishes the pulse verdict now — it
        // never calls the vibrator itself (SentinelService's repeat loop is the sole caller).
        val expected = expectedPulseForLoudToneAfterQuietBaseline()
        assertEquals(expected, controller.state.activePulse, "local pulse train fallback must engage after a race-losing failure")
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")
    }

    // --- Task 7: fail-soft trigger path -----------------------------------------------------

    @Test
    fun wsFactoryThrowSurvivesAndDegradesOffline() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val vibrator = FakeVibratorPort()
        val diag = mutableListOf<String>()
        var clock = 0L
        val wsFactory = ThrowingWsFactory()
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory = wsFactory,
            vibrator = vibrator,
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
            nowMs = { clock },
        )

        controller.arm()
        repeat(8) { controller.tick() } // trigger -> startStreaming() throws internally, caught

        assertEquals(SentinelState.STREAMING, controller.state.sentinel) // still capturing
        assertFalse(controller.state.online) // honest offline
        assertTrue(diag.any { it.startsWith("error/SentinelController") })
        assertEquals(1, wsFactory.callCount, "the initial startStreaming() failure is call 1")

        // loud offline windows -> local ladder nudges + pulse verdicts. Clock advances a full
        // second per tick (matching the real ~1s window cadence) so PulseEngine's own interval
        // gating never masks the final tick's verdict — see the PulseEngineTest suite for that
        // gating's own dedicated coverage. Crucially (P4-5 round 2), 3 ticks crosses the armed 2s
        // backoff boundary (clock reaches 2000 on tick 2), so this also pins that the v0.1.1-style
        // "WS died before a single frame went out" failure arms a reconnect that actually FIRES —
        // not just that the episode stays offline and stops there.
        mic.enqueueMany(3, tone(0.3))
        repeat(3) { clock += 1000; controller.tick() }
        assertEquals(2, wsFactory.callCount, "the armed 2s backoff must have fired a reconnect attempt")
        // Review fix (P4-3 round 2): the controller never calls the vibrator for pulses itself —
        // it still publishes a verdict, which is what "local haptics still fire" now means here.
        assertNotNull(controller.state.activePulse, "local pulse verdict still computed while offline")
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")
    }

    @Test
    fun openThrowSurvives() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val vibrator = FakeVibratorPort()
        val diag = mutableListOf<String>()
        var clock = 0L
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory = OpenThrowingFakeWsFactory(),
            vibrator = vibrator,
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
            nowMs = { clock },
        )

        controller.arm()
        repeat(8) { controller.tick() } // trigger -> open() throws internally, caught

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertFalse(controller.state.online)
        assertTrue(diag.any { it.startsWith("error/SentinelController") })

        mic.enqueueMany(3, tone(0.3))
        repeat(3) { clock += 1000; controller.tick() }
        assertNotNull(controller.state.activePulse, "local pulse verdict still computed while offline")
        assertTrue(vibrator.calls.isEmpty(), "the controller itself must never call the vibrator for pulses")
    }

    @Test
    fun listenerCallbackThrowIsContained() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        val diag = mutableListOf<String>()
        // ThrowingHapticDirector throws directly from onNudge, BEFORE/INSTEAD of ever reaching
        // the real HapticDirector.vibrate() call — so its own internal runCatching around
        // vibrator.vibrate(...) never gets a chance to absorb this. Only SentinelController's own
        // buildListener().onNudge try/catch can be containing it; deleting that catch would make
        // this test throw out of listener.onNudge(...) below and fail.
        val controller = SentinelController(
            mode = Mode.STANDARD,
            mic = mic,
            wsFactory = wsFactory,
            haptics = ThrowingHapticDirector(FakeVibratorPort()),
            ids = FakeEpisodeIdFactory(),
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
        )

        controller.arm()
        repeat(8) { controller.tick() } // reach STREAMING, WS open
        val ws = wsFactory.created.single()
        val listener = requireNotNull(ws.listener)

        // Channel B, not A (P4-3): with pulses enabled (the default), channel-A nudges never even
        // reach HapticDirector.onNudge (see playChannelHapticIfApplicable) — using channel A here
        // would make this a no-op test that never actually exercises the throw. Channel B is
        // unaffected by that suppression and still always calls onNudge.
        // Fired directly from the test thread (standing in for OkHttp's reader thread).
        listener.onNudge(NudgeEvent(channel = "B", level = 2, t = 1.0, vectors = listOf("yelling")))

        assertEquals(2, controller.state.channelLevels["B"])
        assertTrue(diag.any { it.startsWith("error/SentinelController") && it.contains("onNudge") })
    }

    @Test
    fun ringBufferClearedAfterStartStreamingFailureNoStaleAudioBleedsIntoNextEpisode() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = ThrowOnceThenWorkingWsFactory()
        val diag = mutableListOf<String>()
        val controller = newController(
            Mode.STANDARD,
            mic,
            wsFactory = wsFactory,
            diag = DiagLog { l, t, m -> diag.add("$l/$t/$m") },
        )

        controller.arm()
        repeat(8) { controller.tick() } // episode 1 triggers; create() throws -> caught, must clear ring buffer

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertFalse(controller.state.online)
        assertTrue(diag.any { it.startsWith("error/SentinelController") })

        // Drive episode 1 to natural completion: quiet stretch -> COOLDOWN -> cooldownSeconds -> ARMED.
        mic.enqueueMany(30, silence())
        repeat(30) { controller.tick() }
        assertEquals(SentinelState.COOLDOWN, controller.state.sentinel)
        mic.enqueueMany(10, silence())
        repeat(10) { controller.tick() }
        assertEquals(SentinelState.ARMED, controller.state.sentinel)

        // Episode 2 triggers (baseline from episode 1 persists, so 2 loud windows are enough — same
        // recipe as freshWsAttemptOnNextEpisodeAfterMidEpisodeFailure). This time create() succeeds.
        mic.enqueueMany(2, tone(0.2))
        repeat(2) { controller.tick() }

        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        assertEquals(1, wsFactory.created.size, "only the second (successful) create() call produces a FakeWs")
        val ws = wsFactory.created.single()
        // The teeth of this test: if the failed episode 1's ring buffer wasn't cleared, this
        // preamble would carry ~10s of stale pre-failure audio (30 + 10 quiet windows worth of
        // capacity, up to the ring's cap) on top of episode 2's own 2 windows.
        val preambleBytes = ws.sentWindows.sumOf { it.size }
        assertEquals(
            2 * WINDOW_BYTES,
            preambleBytes,
            "stale preamble from the failed episode 1 must not bleed into episode 2",
        )
    }

    // --- P4-5: mid-episode WS reconnect with backoff -----------------------------------------

    @Test
    fun wsFailureWhileStreamingGoesOfflineAndArmsReconnectPolicy() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // episode triggers, ws1 opens
        val ws1 = wsFactory.created.single()

        ws1.listener!!.onFailure(java.io.IOException("connection reset"))
        assertFalse(controller.state.online, "a WS failure while STREAMING must flip offline immediately")

        // The policy was consulted/armed, not skipped: before its 2s delay elapses, no attempt fires.
        mic.enqueueMany(1, tone(0.2))
        clock += 1_999
        controller.tick()
        assertEquals(1, wsFactory.created.size, "1.999s elapsed — still short of the armed 2s delay")

        // Once the armed delay elapses, an attempt DOES fire — proving the failure actually armed
        // a scheduled retry rather than just going offline and stopping there.
        mic.enqueueMany(1, tone(0.2))
        clock += 1
        controller.tick()
        assertEquals(2, wsFactory.created.size, "the armed 2s backoff must fire a reconnect attempt once elapsed")
    }

    @Test
    fun reconnectSucceedsWithFreshEpisodeIdOnlineTrueAndServerNudgesResume() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() }
        val ws1 = wsFactory.created.single()
        ws1.listener!!.onFailure(java.io.IOException("connection reset"))

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // cross the armed 2s backoff boundary
        controller.tick()

        assertEquals(2, wsFactory.created.size)
        val ws2 = wsFactory.created[1]
        assertTrue(ws2.openedEpisodeId != ws1.openedEpisodeId, "reconnect must open a FRESH episode id, not resume the old one")
        assertTrue(controller.state.online, "a reconnect that opens without an immediate failure reports online")

        ws2.listener!!.onNudge(NudgeEvent(channel = "B", level = 1, t = 1.0, vectors = listOf("interrupting")))
        assertEquals(1, controller.state.channelLevels["B"], "server nudges must resume flowing through the new client's listener")
    }

    @Test
    fun reconnectAttemptFailureKeepsOfflineAndContinuesBackoffLadder() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        // Call 1 (the original trigger) succeeds; call 2 (the first reconnect attempt) fails.
        val wsFactory = FailOnceAtCallWsFactory(failOnCall = 2)
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // episode triggers, call 1 opens ws1
        val ws1 = wsFactory.created.single()
        ws1.listener!!.onFailure(java.io.IOException("connection reset")) // arms 2s backoff (attempt 1)

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // crosses the 2s boundary -> reconnect call 2 fires and throws
        controller.tick()

        assertFalse(controller.state.online, "a failed reconnect attempt must stay offline")
        assertEquals(1, wsFactory.created.size, "the failed reconnect call must not have produced a FakeWs")

        // Next rung is 4s, not another 2s — 2s further (4s total since the ORIGINAL failure, but
        // only 2s since the FAILED reconnect attempt at clock=2000) must not be due yet if the
        // ladder correctly continued rather than resetting.
        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // clock = 4000
        controller.tick()
        assertEquals(1, wsFactory.created.size, "2s after the failed reconnect attempt is not yet due — next rung is 4s")

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // clock = 6000 -> 4s elapsed since the failed attempt at 2000
        controller.tick()
        assertEquals(2, wsFactory.created.size, "4s after the failed reconnect attempt, the next attempt must fire")
    }

    @Test
    fun disarmResetsReconnectBackoffLadderForNextEpisode() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // episode 1 triggers
        wsFactory.created.single().listener!!.onFailure(java.io.IOException("connection reset")) // arms 2s (attempt 1)

        controller.disarm() // must reset the backoff ladder, not just cancel this episode
        controller.arm()

        // Episode 2 triggers (baseline persists — 2 loud windows are enough, same recipe as
        // freshWsAttemptAfterFailureThenManualDisarmRearm).
        mic.enqueueMany(2, tone(0.2))
        repeat(2) { controller.tick() }
        assertEquals(SentinelState.STREAMING, controller.state.sentinel)
        val ws2 = wsFactory.created[1]
        ws2.listener!!.onFailure(java.io.IOException("connection reset again")) // episode 2's OWN first failure

        // If disarm() hadn't reset the ladder, this failure would continue from attempt=1 (a 4s
        // delay) instead of restarting at the initial 2s rung — advancing by exactly 2s (not 4s)
        // distinguishes the two.
        mic.enqueueMany(1, tone(0.2))
        clock += 2_000
        controller.tick()

        assertEquals(3, wsFactory.created.size, "a reset ladder must fire a reconnect attempt after only 2s, not 4s")
    }

    @Test
    fun staleListenerFromSupersededReconnectAttemptIsIgnored() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() }
        val ws1 = wsFactory.created.single()
        ws1.listener!!.onFailure(java.io.IOException("connection reset"))

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000
        controller.tick() // reconnect succeeds -> ws2 is now the live client
        assertTrue(controller.state.online)

        // ws1's listener is now stale (superseded by ws2's reconnect) — a late callback from it
        // must be ignored rather than clobbering ws2's live state.
        ws1.listener!!.onFailure(java.io.IOException("late failure from the dead client"))
        assertTrue(controller.state.online, "a stale (superseded) client's onFailure must not flip online back to false")

        ws1.listener!!.onNudge(NudgeEvent(channel = "A", level = 3, t = 99.0, vectors = listOf("yelling")))
        assertNull(controller.state.channelLevels["A"], "a stale client's onNudge must not publish a channel level either")
    }

    // --- P4-5 round 2 (review fixes): reset-ordering guard + token check-then-act race --------

    @Test
    fun reconnectRacingSynchronousListenerFailureKeepsBackoffArmedNotReset() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        // Call 1 (the original trigger) opens normally; call 2 (the reconnect attempt) synchronously
        // fires listener.onFailure from within open() itself — the sharpest race FailingFakeWs
        // exists to simulate (see its own KDoc), now exercised against attemptReconnect()'s own
        // "reset the ladder only if still online after open() returns" guard rather than
        // startStreaming()'s. This is the test the P4-5 review flagged as missing: the guard at
        // SentinelController.attemptReconnect() had zero coverage — every prior test either hit a
        // create()-throw (skips the guarded line entirely) or a plain FakeWs (never fires
        // synchronously, so `online` was always still true when the guard ran).
        val wsFactory = FailingFromCallWsFactory(failFromCall = 2)
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() } // episode triggers, call 1 opens normally
        val ws1 = requireNotNull(wsFactory.firstCreated)
        ws1.listener!!.onFailure(java.io.IOException("connection reset")) // arms 2s backoff (attempt 1)
        assertEquals(1, wsFactory.callCount)

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // crosses the 2s boundary -> reconnect attempt (call 2) fires and races a
        // synchronous listener onFailure from inside open() itself (FailingFakeWs), arming a 4s
        // backoff (attempt 2) BEFORE attemptReconnect()'s own "if (online) reset()" line runs.
        controller.tick()

        assertEquals(2, wsFactory.callCount, "the reconnect attempt (call 2) must have fired")
        assertFalse(controller.state.online, "a reconnect that races a synchronous listener failure must stay offline")

        // The teeth of this test: if attemptReconnect() reset() the ladder UNCONDITIONALLY after
        // open() returns (rather than only when still online), the racing onFailure's own 4s
        // backoff would get silently wiped back to the initial 2s rung — worse, wiped to "no
        // attempt scheduled at all" if the wipe raced the online=false write the wrong way. 2s
        // further must NOT be enough to fire a 3rd attempt if the guard held.
        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // 2s since the racing failure — not enough for the 4s rung it should have armed
        controller.tick()
        assertEquals(
            2,
            wsFactory.callCount,
            "2s after the racing failure is not due yet — the ladder must have continued to 4s, not reset to 2s",
        )

        // ...but 4s IS enough, proving the ladder is still alive and correctly continued.
        mic.enqueueMany(1, tone(0.2))
        clock += 2_000 // 4s since the racing failure
        controller.tick()
        assertEquals(
            3,
            wsFactory.callCount,
            "4s after the racing failure the next attempt must fire, proving the ladder correctly continued rather than reset or vanishing",
        )
    }

    // --- P4-10: availability is published honestly alongside (never instead of) a reading -------

    @Test
    fun availabilityReportsTheHrSourcesStatusWhenHrIsSelected() {
        val hr = FakeScalar(72.0, SignalAvailability.AVAILABLE)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        assertEquals(SignalAvailability.AVAILABLE, c.state.availability)
    }

    @Test
    fun offBodyHrPublishesNoMeterButAnHonestAvailability() {
        // The exact v0.2.2 device case: registered fine, availability UNAVAILABLE_DEVICE_OFF_BODY,
        // therefore no samples. A number here would be fabricated; a status is not.
        val hr = FakeScalar(null, SignalAvailability.OFF_BODY)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        assertNull(c.state.meter)
        assertEquals(SignalAvailability.OFF_BODY, c.state.availability)
    }

    @Test
    fun availabilityIsUnknownForNonHrSignals() {
        val hr = FakeScalar(72.0, SignalAvailability.OFF_BODY)
        val c = controller(hr = hr, selectedSignal = { SignalKind.VOLUME })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        assertEquals(SignalAvailability.UNKNOWN, c.state.availability)
    }

    @Test
    fun throwingAvailabilitySourceDegradesToUnknownWithoutCrashingTheTickPath() {
        val hr = object : ScalarSource {
            override fun latest(): Double = 72.0
            override fun availability(): SignalAvailability = throw IllegalStateException("sensor died")
        }
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(3) { c.tick(tone(0.05)) }
        assertEquals(SentinelState.ARMED, c.state.sentinel)
        assertEquals(SignalAvailability.UNKNOWN, c.state.availability)
    }

    @Test
    fun disarmClearsAvailabilityAlongsideTheMeter() {
        val hr = FakeScalar(72.0, SignalAvailability.AVAILABLE)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        c.disarm()
        assertNull(c.state.meter)
        assertEquals(SignalAvailability.UNKNOWN, c.state.availability)
    }

    @Test
    fun offBodyTransitionClearsAStaleMeterReadingEvenIfTheSourceForgetsTo() {
        // Regression for the P4-10 review round-1 Critical fix: updateMeter's HEART_RATE branch
        // must structurally refuse to publish a meter value once availability says otherwise — a
        // guarantee independent of whether the ScalarSource itself remembers to clear its own
        // last reading (see ControllerState.availability's "ENFORCED, not merely conventional"
        // KDoc). This FakeScalar deliberately keeps returning the STALE 72.0 bpm from latest()
        // even after `.avail` flips to OFF_BODY, standing in for a ScalarSource implementation
        // that forgets to self-clear — exactly the gap HrSource itself had before this fix.
        val hr = FakeScalar(72.0, SignalAvailability.AVAILABLE)
        val c = controller(hr = hr, selectedSignal = { SignalKind.HEART_RATE })
        c.arm(); repeat(6) { c.tick(tone(0.05)) }
        assertEquals(72.0, c.state.meter!!.value)
        assertEquals(SignalAvailability.AVAILABLE, c.state.availability)

        hr.avail = SignalAvailability.OFF_BODY // v is left at 72.0 on purpose — see comment above
        c.tick(tone(0.05))
        assertNull(c.state.meter)
        assertEquals(SignalAvailability.OFF_BODY, c.state.availability)
    }

    @Test
    fun staleOnFailureAfterDisarmDoesNotResurrectOfflineOrReconnectBackoff() {
        val mic = ScriptedMic(quietThenTriggerWindows())
        val wsFactory = FakeWsFactory()
        var clock = 0L
        val controller = newController(Mode.STANDARD, mic, wsFactory, nowMs = { clock })

        controller.arm()
        repeat(8) { controller.tick() }
        val ws1 = wsFactory.created.single()

        controller.disarm() // generation bumped; ws1's listener token is now stale

        // Regression coverage for the check-then-act race the P4-5 review caught: an isCurrent()
        // check performed OUTSIDE `lock`, with the actual mutation in a separate later lock
        // acquisition, could let a stale callback "win" against a concurrent disarm() that had
        // already reset everything. This test drives the deterministic-ordering version the
        // review suggested (call the stale listener AFTER disarm() has fully completed) — true
        // thread interleaving isn't reproducible with this synchronous fake harness, but the fix
        // itself (currency re-checked atomically inside the SAME lock disarm()'s own generation
        // bump now lives in) is what this end state depends on: if either half of that fix
        // regressed (the bump moved back outside disarm()'s lock, or a callback's mutation moved
        // back outside its own currency check), this stale call would still be harmless here too
        // — so the assertions below are necessary-but-not-sufficient regression coverage, not a
        // full proof of the concurrency property (see the round-2 report for that caveat).
        ws1.listener!!.onFailure(java.io.IOException("late failure racing disarm()"))
        assertTrue(controller.state.online, "a stale post-disarm onFailure must not flip online back to false")

        // Prove the backoff ladder is genuinely untouched (not merely "online happens to read
        // true") by re-arming and re-failing: a fresh episode's first failure must still get the
        // INITIAL 2s delay, not a continued/advanced one — if the stale onFailure above had wrongly
        // gone through, it would have advanced reconnectPolicy's internal ladder past its initial
        // rung.
        controller.arm()
        mic.enqueueMany(2, tone(0.2))
        repeat(2) { controller.tick() }
        val ws2 = wsFactory.created[1]
        ws2.listener!!.onFailure(java.io.IOException("this episode's own first failure"))

        mic.enqueueMany(1, tone(0.2))
        clock += 2_000
        controller.tick()
        assertEquals(
            3,
            wsFactory.created.size,
            "the ladder must still be at its initial 2s rung — a stale onFailure that had wrongly advanced it would push this past 2s",
        )
    }

    // --- P5-3: the one fact SentinelService needs to drive MicDutyCycle ------------------------

    @Test
    fun lastWindowVoicedTracksTheMostRecentWindow() {
        val c = controller()
        c.arm()
        c.tick(tone(0.05))
        assertTrue(c.state.lastWindowVoiced)
        c.tick(silence())
        assertFalse(c.state.lastWindowVoiced)
        c.tick(tone(0.05))
        assertTrue(c.state.lastWindowVoiced)
    }

    @Test
    fun disarmClearsLastWindowVoiced() {
        val c = controller()
        c.arm()
        c.tick(tone(0.05))
        c.disarm()
        assertFalse(c.state.lastWindowVoiced)
    }

    // --- v0.2.4: ARMED shout-tap ----------------------------------------------------------------

    private fun shoutController(clock: () -> Long): SentinelController {
        val wsFactory = object : WsFactory {
            override fun create(): EpisodeWs = FakeWs().also { fakeWs = it }
        }
        return SentinelController(
            mode = Mode.STANDARD,
            mic = ScriptedMic(fillWith = silence()),
            wsFactory = wsFactory,
            haptics = HapticDirector(FakeVibratorPort(), nowMs = { 0L }),
            ids = FakeEpisodeIdFactory(),
            nowMs = clock,
        )
    }

    /** Seeds the detector baseline with quiet voiced windows (3 seed + 2 settle). */
    private fun seedBaseline(c: SentinelController) {
        repeat(5) { c.tick(tone(0.05)) }
    }

    @Test
    fun armedLoudWindowPublishesASingleBandOneShoutTap() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.5)) // ~20dB over the quiet baseline: voicedLoud, first over-threshold window
        assertEquals(SentinelState.ARMED, c.state.sentinel) // trigger needs 2 consecutive — still armed
        assertEquals(HapticPatterns.ARMED_SHOUT_TAP, c.state.armedShoutTap)
    }

    @Test
    fun theTriggerTransitionTickDoesNotShoutTap() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.5))
        clock = 20_000L // far past the 2s rate limit — only the transition rule can suppress this
        c.tick(tone(0.5)) // second consecutive loud window: ARMED -> STREAMING
        assertEquals(SentinelState.STREAMING, c.state.sentinel)
        assertNull(c.state.armedShoutTap) // the pulse chain owns this moment
    }

    @Test
    fun shoutTapIsRateLimitedToTwoSeconds() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        // Alternating loud/quiet keeps the trigger streak at 1 (never two consecutive), so the
        // sentinel stays ARMED throughout — exactly the sustained-almost-arguing case.
        clock = 10_000L
        c.tick(tone(0.5))
        assertEquals(HapticPatterns.ARMED_SHOUT_TAP, c.state.armedShoutTap)
        clock = 10_500L
        c.tick(silence())
        clock = 11_000L
        c.tick(tone(0.5)) // only 1s after the tap: suppressed
        assertNull(c.state.armedShoutTap)
        clock = 11_500L
        c.tick(silence())
        clock = 12_000L
        c.tick(tone(0.5)) // 2s after the tap: fires again
        assertEquals(HapticPatterns.ARMED_SHOUT_TAP, c.state.armedShoutTap)
        assertEquals(SentinelState.ARMED, c.state.sentinel)
    }

    @Test
    fun quietArmedWindowsNeverShoutTap() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.05))
        assertNull(c.state.armedShoutTap)
        c.tick(silence())
        assertNull(c.state.armedShoutTap)
    }

    @Test
    fun streamingWindowsNeverShoutTap() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.5))
        clock = 20_000L
        c.tick(tone(0.5)) // -> STREAMING
        clock = 30_000L
        c.tick(tone(0.5)) // loud mid-episode: the pulse train's territory, not the shout-tap's
        assertEquals(SentinelState.STREAMING, c.state.sentinel)
        assertNull(c.state.armedShoutTap)
    }

    @Test
    fun disarmClearsTheShoutTapAlongsideTheMeter() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.5))
        assertNotNull(c.state.armedShoutTap)
        c.disarm()
        assertNull(c.state.armedShoutTap)
    }

    @Test
    fun aShoutTapIsPublishedForExactlyOneWindow() {
        var clock = 0L
        val c = shoutController { clock }
        c.arm()
        seedBaseline(c)
        clock = 10_000L
        c.tick(tone(0.5))
        assertNotNull(c.state.armedShoutTap)
        clock = 10_500L
        c.tick(silence()) // very next window is quiet: the tap must not linger in state
        assertNull(c.state.armedShoutTap)
    }

    // --- Wave C Task 10: RetroCaptureBuffer wiring ------------------------------------------------

    @Test
    fun retroCaptureAvailableSecondsGrowsEveryWindowRegardlessOfState() {
        val c = controller() // existing test helper
        c.arm()
        c.tick(tone(0.05))
        c.tick(silence())
        assertEquals(2.0, c.state.retroCaptureAvailableSeconds, absoluteTolerance = 0.001)
    }

    @Test
    fun captureRetroSnapshotReturnsNullBeforeAnyWindow() {
        val c = controller()
        assertNull(c.captureRetroSnapshot(120.0))
    }

    @Test
    fun captureRetroSnapshotReturnsRecordedPcm() {
        val c = controller()
        c.arm()
        c.tick(tone(0.05))
        val snap = c.captureRetroSnapshot(120.0)
        assertEquals(16000 * 2, snap?.size) // one window's worth (2 bytes/sample), clamped honestly
    }

    @Test
    fun retroCaptureKeepsRecordingThroughAnEpisode() {
        val c = controller()
        c.arm()
        repeat(5) { c.tick(tone(0.05)) } // seed baseline
        c.tick(tone(0.5))
        c.tick(tone(0.5)) // -> STREAMING
        assertEquals(SentinelState.STREAMING, c.state.sentinel)
        assertTrue(c.state.retroCaptureAvailableSeconds >= 7.0) // still growing mid-episode
    }

    @Test
    fun disarmClearsRetroCaptureAlongsideEverythingElseEpisodeScoped() {
        // Same "nothing stale survives a disarm" rule already applied to meter/activePulse/
        // availability/lastWindowVoiced/sparkline/armedShoutTap in this method.
        val c = controller()
        c.arm()
        c.tick(tone(0.05))
        assertTrue(c.state.retroCaptureAvailableSeconds > 0.0, "sanity: buffer should be populated before disarm")
        c.disarm()
        assertEquals(0.0, c.state.retroCaptureAvailableSeconds, absoluteTolerance = 0.001)
        assertNull(c.captureRetroSnapshot(120.0))
    }
}
