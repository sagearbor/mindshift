import Foundation
import SwiftUI

/// The one place watch state lives. Views are purely presentational: nothing in
/// `UI/` decides anything, exactly as `GlanceScreen.kt` takes every value it
/// draws straight off `GlanceUi` and makes no sentinel/mode mapping decisions of
/// its own.
///
/// Scope, and why it stops where it does — `docs/plans/2026-09-21-watchos-parity-spec.md`
/// §19's build order, steps 2 and 3:
///
/// * **Step 2, pairing + auth + `/me/standing`** — built. Pure HTTP, FULL
///   parity, needs no mic, no HealthKit and no haptics, and is testable end to
///   end in the simulator.
/// * **Step 3, COMPANION mode** — built. Socket, `{"type":"companion"}` hello,
///   20 s heartbeat, nudge frames, the PRD §6 reminder ladder, and the reduced
///   `playHaptic` mapping.
/// * **Step 4, a user-started mic session** — *not* built. It needs
///   `AVAudioEngine` → 16 kHz PCM16 → binary WS frames under an
///   `HKWorkoutSession`, and the single question it depends on — whether
///   `playHaptic` fires at all with the screen off and the wrist down — is
///   UNVERIFIABLE WITHOUT HARDWARE (spec §7.3). Shipping the socket lane first
///   means nothing has to be un-built if the answer is no.
/// * **The all-day armed sentinel and the pulse train** — never (spec §6.3,
///   §7.2 item 3). Absent, not broken.
///
/// The honest limit of what IS built: a nudge arriving on the socket reaches the
/// wrist only while this app is frontmost. `playHaptic` has no documented
/// background guarantee (`WKInterfaceDevice.h:104`) and the only non-frontmost
/// repeating-haptic API is an *alarm* facility that caps its repeat at 60 s and
/// shows a full-screen system alert (`WKExtendedRuntimeSession.h:142-148`),
/// which cannot express level 1's 2-minute cadence. `ListeningView` says this on
/// screen rather than leaving the wearer to discover it.
@MainActor
final class WatchStore: ObservableObject {
    enum Phase: Equatable {
        case loading
        case signIn
        case idle
        case listening
    }

    // MARK: Published state

    @Published private(set) var phase: Phase = .loading

    /// Sign-in
    @Published private(set) var pairingCode: String?
    @Published private(set) var pairingError: String?

    /// Session
    @Published private(set) var channelALevel: Int = 0
    @Published private(set) var lastNudge: NudgeDisplay?
    @Published private(set) var socketState: SocketState = .closed
    @Published private(set) var standing: MemberStanding?
    @Published private(set) var phoneReachable = false
    @Published private(set) var hapticLog: [HapticAttempt] = []
    @Published private(set) var reminderCountdownS: Int?
    /// 2026-09-25: the plain-language line for the most recent server `error`
    /// frame this session, or `nil`. `WatchWireText.serverError` — the same
    /// strings `GaugeViewModel.kt`'s `serverErrorText` shows on Wear. Cleared
    /// when a session starts and when it stops.
    @Published private(set) var lastServerError: String?
    /// 2026-09-25: what the server said it kept when the session ended
    /// (`live_session_saved.status` → `WatchWireText.sessionOutcome`), or `nil`
    /// — including for status "companion", which persisted nothing by design.
    @Published private(set) var lastSessionOutcome: String?

    enum SocketState: Equatable {
        case closed
        case connecting
        case open
        case reconnecting(inS: Int)
        case failed(String)

        var label: String {
            switch self {
            case .closed: return "Off"
            case .connecting: return "Connecting…"
            case .open: return "Listening"
            case .reconnecting(let s): return "Retrying in \(s)s"
            case .failed(let why): return why
            }
        }
    }

    /// What the screen shows for one nudge. The screen has to carry the code's
    /// identity because on watchOS the wrist cannot: five of the eight cues are
    /// indistinguishable by feel (`WatchHapticVocabulary`).
    struct NudgeDisplay: Equatable {
        let code: String
        let icon: String
        let name: String
        /// The vocabulary's own flash line, or `nil` — never invented.
        let flashText: String?
        let level: Int
        let isPositive: Bool
        let fidelity: WatchHapticVocabulary.Fidelity
        let at: Date
    }

    // MARK: Collaborators

    private let api: WatchApi
    private let tokens: TokenStore
    private let haptics: WristHaptics
    private let socket: CompanionSocket
    private let schedule: NudgeScheduling
    private let vocabulary: NudgeVocabularyLookup
    private let positiveGate = PositiveNudgeGate()
    private let reconnect = ReconnectPolicy()
    private var phoneBridge: PhoneBridging?

    /// Reminder bookkeeping — the PRD §6 ladder. A reminder replays the SAME
    /// vocabulary code the level was raised by, never a generic buzz
    /// (`HapticDirector.kt:54-59`, `:130`), and reminders are channel A only:
    /// channel B stays single-shot (`HapticDirector.kt:45-52`, `:269-270`).
    private var reminderCode: String?
    private var reminderLastPlayedMs: Int64 = 0
    private var reminderRepeats = 0
    /// The 5 s duplicate-suppression window on fresh nudges
    /// (`HapticDirector.kt:83-85`).
    private var lastVibrationMs: Int64 = 0

    private var pollTask: Task<Void, Never>?
    private var tickTask: Task<Void, Never>?
    private var pairingId: String?

    init(
        api: WatchApi = WatchApi(),
        tokens: TokenStore = TokenStore(),
        haptics: WristHaptics? = nil,
        socket: CompanionSocket = CompanionSocket(),
        schedule: NudgeScheduling = SharedPolicy.schedule,
        vocabulary: NudgeVocabularyLookup = SharedPolicy.vocabulary
    ) {
        self.api = api
        self.tokens = tokens
        #if targetEnvironment(simulator)
        // The simulator has no actuator, so there is nothing to ask. Recording
        // the request is the honest behaviour and makes the haptic log on the
        // diagnostics screen real evidence rather than decoration.
        self.haptics = haptics ?? RecordingHaptics()
        #else
        self.haptics = haptics ?? WatchKitHaptics()
        #endif
        self.socket = socket
        self.schedule = schedule
        self.vocabulary = vocabulary
    }

    var policyBacking: String { SharedPolicy.backing.rawValue }
    /// Non-empty only when the KMP framework is linked AND the Swift fallback
    /// disagrees with it. Rendered on the diagnostics screen: a cross-runtime
    /// drift should be visible on the wrist, not buried in a test log.
    lazy var policyParityFailures: [String] = SharedPolicy.parityFailures()

    // MARK: Lifecycle

    func start() {
        logPolicyBacking()
        #if DEBUG
        if let preview = DebugPreview.requested {
            applyPreview(preview)
            return
        }
        #endif
        attachPhoneBridge()
        if tokens.isSignedIn {
            phase = .idle
            Task { await refreshStanding() }
        } else {
            phase = .signIn
            beginPairing()
        }
    }

    /// One startup line naming which policy implementation is live and whether
    /// it agrees with the Kotlin.
    ///
    /// This is deliberately a log line and not only a screen: the simulator is
    /// the only test surface, `simctl` cannot scroll a watch list to reach the
    /// diagnostics screen's Policy section, and a cross-runtime drift needs to
    /// be checkable by a script rather than by someone squinting at a 42 mm
    /// screenshot. `scripts/verify_watchos_policy.sh` greps for this prefix.
    private func logPolicyBacking() {
        let failures = policyParityFailures
        NSLog("MINDSHIFT_WATCH_POLICY backing=%@ parity_failures=%d %@",
              SharedPolicy.backing.rawValue,
              failures.count,
              failures.prefix(3).joined(separator: " | "))
    }

    private func attachPhoneBridge() {
        #if canImport(WatchConnectivity)
        let bridge = PhoneBridge { [weak self] event in
            Task { @MainActor in self?.handlePhone(event) }
        }
        #else
        let bridge = UnavailablePhoneBridge()
        #endif
        phoneBridge = bridge
        bridge.activate()
        phoneReachable = bridge.isReachable
    }

    private func handlePhone(_ event: PhoneEvent) {
        switch event {
        case .reachabilityChanged(let reachable):
            phoneReachable = reachable
        case .credentials(let accountId, let deviceToken):
            // Additive: the phone saved us the 6-character dance. It never
            // replaces it — if this never arrives, the code path stands.
            guard !tokens.isSignedIn else { return }
            tokens.save(accountId: accountId, deviceToken: deviceToken)
            pollTask?.cancel()
            phase = .idle
            Task { await refreshStanding() }
        case .signedOut:
            signOut()
        case .frame(let frame):
            // One pipeline for both transports, by design.
            apply(frame)
        }
    }

    // MARK: Pairing

    /// `SignInScreen.kt`'s loop, ported: mint a code, poll `/me/pair/status`
    /// every 3 s, give up after 20 minutes.
    ///
    /// The Wear version's poller dies with the screen (`SignInScreen.kt:37-40`),
    /// which is *more* of a problem on a watch whose display sleeps in ~20 s
    /// (spec §11.2). This one is owned by the store rather than a
    /// `LaunchedEffect`, so it survives the screen blanking, and
    /// `resumePairingIfNeeded()` re-mints only when there is genuinely nothing
    /// in flight.
    func beginPairing() {
        pollTask?.cancel()
        pairingError = nil
        pairingCode = nil
        pollTask = Task { [weak self] in await self?.pairingLoop() }
    }

    func resumePairingIfNeeded() {
        guard phase == .signIn, pollTask == nil || pollTask?.isCancelled == true else { return }
        beginPairing()
    }

    private func pairingLoop() async {
        let poller = PairingPoller(now: { Self.nowMs() })
        do {
            let start = try await api.startPairing()
            pairingCode = start.code
            pairingId = start.pairingId
        } catch {
            pairingError = "Can't reach MindShift. Check Wi-Fi."
            return
        }
        guard let pairingId else { return }
        while !Task.isCancelled {
            try? await Task.sleep(nanoseconds: UInt64(PairingPoller.pollIntervalMs) * 1_000_000)
            if Task.isCancelled { return }
            // `nil` here means transport failure ONLY, never a lifecycle state
            // — that distinction is the whole reason `PairingPoller` exists.
            let status = try? await api.pollPairing(pairingId: pairingId)
            switch poller.onPollResult(status) {
            case .keepPolling:
                continue
            case .signedIn(let accountId, let deviceToken):
                tokens.save(accountId: accountId, deviceToken: deviceToken)
                // Fire-and-forget, exactly as `SignInScreen.kt:88-95`.
                Task { [api] in await api.claimLegacy(token: deviceToken) }
                phase = .idle
                await refreshStanding()
                return
            case .failed(let message):
                pairingError = message
                pairingCode = nil
                return
            }
        }
    }

    func signOut() {
        tokens.clearForUnauthorized()
        stopListening()
        phase = .signIn
        standing = nil
        beginPairing()
    }

    // MARK: Standing

    func refreshStanding() async {
        guard let token = tokens.deviceToken else { return }
        do {
            standing = try await api.standing(token: token)
        } catch WatchApiError.unauthorized {
            // 401: the token is dead. Clear it and route to sign-in — the watch
            // must KNOW it is signed out rather than retrying forever
            // (`CouplesLoad.kt:19-24`).
            signOut()
        } catch WatchApiError.serverUnavailable {
            // 503: a transient verifier outage. KEEP the token. The server
            // answers 503 rather than 401 for exactly this reason
            // (`server/watch/auth.py:156-162`); treating it as a 401 logs the
            // wearer out on every Firestore blip.
        } catch {
            // Any other failure leaves the last known standing on screen rather
            // than blanking it — a stale number beats no number.
        }
    }

    // MARK: Companion session

    func startListening() {
        guard tokens.isSignedIn else { return }
        phase = .listening
        // A fresh session starts the back-off ladder over
        // (`ReconnectPolicy.kt:44-55` — reset on a fresh episode and on disarm).
        reconnect.reset()
        // ...and a clean slate for the server's verdicts on the PREVIOUS session
        // (`SentinelController.startStreaming` clears the same two fields).
        lastServerError = nil
        lastSessionOutcome = nil
        openSocket()
        startTicking()
    }

    /// Opens (or re-opens) the companion socket. Deliberately separate from
    /// `startListening()` and deliberately does NOT touch the reconnect ladder:
    /// a retry that resets the ladder is not a ladder, it is a 2-second poll.
    private func openSocket() {
        guard let token = tokens.deviceToken, let accountId = tokens.accountId else { return }
        socketState = .connecting
        let sessionId = companionSessionId(accountId: accountId)
        Task { [socket, api] in
            let wsBase = await api.wsBaseUrl
            await socket.open(wsBaseUrl: wsBase, sessionId: sessionId, token: token) { [weak self] event in
                Task { @MainActor in self?.handleSocket(event) }
            }
        }
    }

    func stopListening() {
        tickTask?.cancel()
        tickTask = nil
        Task { [socket] in
            await socket.end()
            await socket.close()
        }
        socketState = .closed
        clearReminder()
        // Disarm reports nothing about a socket it no longer has
        // (`SentinelController.disarm` clears the same fields on Wear).
        lastServerError = nil
        lastSessionOutcome = nil
        if phase == .listening { phase = tokens.isSignedIn ? .idle : .signIn }
    }

    private func handleSocket(_ event: CompanionSocket.Event) {
        switch event {
        case .opened:
            // The task resumed; the server has confirmed nothing yet. Wear
            // publishes optimistically here ("publish before open()", the
            // earliest honest signal OkHttp offers), but this protocol has a
            // real ack — `companion_ack` — so the ladder waits for it rather
            // than resetting on a socket that may still be refused.
            socketState = .connecting
        case .frame(let frame):
            apply(frame)
        case .closed(let reason):
            guard phase == .listening else { return }
            let delay = reconnect.onFailure(nowMs: Self.nowMs())
            socketState = .reconnecting(inS: Int(delay / 1000))
            _ = reason
        }
    }

    /// The tick. `COMPANION_TICK_MS = 1_000` on Wear (`SentinelService.kt:1091`)
    /// and the tick is `postDelayed`, not paced by a blocking mic read — which
    /// is the one part of the Wear execution model that ports cleanly, because
    /// there is no mic read here at all.
    private func startTicking() {
        tickTask?.cancel()
        tickTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self else { return }
                // No `await`: `WatchStore` is `@MainActor`, so this `Task`
                // inherits main-actor isolation and `tick()` is already local.
                self.tick()
            }
        }
    }

    private func tick() {
        guard phase == .listening else { return }
        let now = Self.nowMs()

        // Replay a due reminder.
        if let code = reminderCode, channelALevel > 0 {
            if schedule.reminderDue(
                level: channelALevel,
                lastPlayedMs: reminderLastPlayedMs,
                nowMs: now,
                repeatIndex: reminderRepeats
            ) {
                play(code: code, level: channelALevel, isReminder: true)
                reminderLastPlayedMs = now
                reminderRepeats += 1
            }
            if let interval = schedule.reminderIntervalMs(level: channelALevel, repeatIndex: reminderRepeats) {
                reminderCountdownS = max(0, Int((reminderLastPlayedMs + interval - now) / 1000))
            }
        } else {
            reminderCountdownS = nil
        }

        // Reconnect if the ladder says it is due.
        if reconnect.isAttemptDue(streaming: phase == .listening, nowMs: now) {
            // Arm the NEXT rung before attempting, because the attempt itself
            // may fail — consuming a one-shot signal without advancing it is how
            // a retry loop turns into a busy loop (`ReconnectPolicy.kt:66-70`).
            reconnect.onFailure(nowMs: now)
            openSocket()
        } else if case .reconnecting = socketState, let at = reconnect.armedDelayRemainingMs {
            socketState = .reconnecting(inS: max(0, Int((at - now) / 1000)))
        }

        hapticLog = currentHapticLog()
    }

    // MARK: The nudge pipeline

    private func apply(_ frame: ServerFrame) {
        switch frame {
        case .companionAck:
            // A real round trip: the socket authenticated and the server agreed
            // to persist nothing. This — not the task resuming — is what resets
            // the back-off ladder.
            socketState = .open
            reconnect.reset()
        case .nudge(let channel, let level, _, let vectors):
            applyNudge(channel: channel, level: level, vectors: vectors)
        case .positive(let code, let t):
            applyPositive(code: code, t: t)
        case .vectorEvent:
            break // Nothing on this screen shows raw vectors yet.
        case .liveSessionSaved(let id, let status):
            // The status is the whole point: "companion" means the server kept
            // NOTHING by design, "companion_hr" only heart rate, "captured" a
            // full episode. Logged with the `MINDSHIFT_WATCH_` prefix family so
            // a simulator run can be checked by grep, like the policy line.
            NSLog("MINDSHIFT_WATCH_WS live_session_saved id=%@ status=%@", id, status ?? "unknown")
            lastSessionOutcome = WatchWireText.sessionOutcome(status: status)
        case .error(let detail):
            // Surfaced, not acted on: the server keeps the socket open after an
            // error frame, so this must not touch `socketState` or the ladder.
            // It IS the only way the server can say our frames are wrong
            // (an older server answers the companion hello with `unknown_type`
            // and then treats this socket as a mic session), so it is logged at
            // error level and shown on the listening screen — it used to be
            // matched here and dropped, the same bug as `EpisodeWsClient.kt:140`.
            NSLog("MINDSHIFT_WATCH_WS error detail=%@", detail)
            lastServerError = WatchWireText.serverError(detail: detail)
        case .other:
            break
        }
    }

    private func applyNudge(channel: String, level: Int, vectors: [String]) {
        // A reminder replays the code the level was RAISED by, so the code has
        // to be resolved from the firing vectors rather than assumed to be H.
        let code = vocabulary.code(forVectors: vectors) ?? "H"
        guard let entry = vocabulary.entry(forCode: code) else { return }

        if channel == "A" {
            let escalated = level > channelALevel
            channelALevel = level
            if level == 0 {
                // Level 0 is silent AND clears the reminder
                // (`HapticDirector.kt:69-80`).
                clearReminder()
                return
            }
            reminderCode = code
            if escalated {
                // A NEW escalation resets the repeat count so a genuinely
                // worsening situation is reported promptly again
                // (`HapticDirector.kt:98-103`, reset at `:102`).
                reminderRepeats = 0
            }
        }

        let now = Self.nowMs()
        // The 5 s duplicate-suppression window (`HapticDirector.kt:83-85`).
        if now - lastVibrationMs < 5_000 {
            show(entry: entry, level: level)
            return
        }
        show(entry: entry, level: level)
        play(code: code, level: level, isReminder: false)
        if channel == "A" { reminderLastPlayedMs = now }
    }

    private func applyPositive(code: String, t: Double) {
        guard let entry = vocabulary.entry(forCode: code) else { return }
        // At most one positive per 120 s across D/E/R together
        // (`NudgeVocabulary.kt:118`, `:323-338`). A rate-limited positive still
        // shows on screen — the cap is about the wrist, not the record.
        let admitted = positiveGate.admit(t: t)
        show(entry: entry, level: 1)
        guard admitted else { return }
        // A positive NEVER arms the reminder and NEVER moves a level:
        // `playPositive` bypasses `onNudge` entirely (`HapticDirector.kt:190-196`)
        // because "a wrist that repeats 'well done' every two minutes is worse
        // than one that never said it".
        play(code: code, level: 1, isReminder: false)
    }

    private func show(entry: VocabularyEntry, level: Int) {
        let fidelity = WatchHapticVocabulary.table[entry.code]?.fidelity ?? .collapsed
        lastNudge = NudgeDisplay(
            code: entry.code,
            icon: entry.icon,
            name: entry.name,
            flashText: entry.flashText,
            level: level,
            isPositive: entry.isPositive,
            fidelity: fidelity,
            at: Date()
        )
    }

    private func play(code: String, level: Int, isReminder: Bool) {
        // K never buzzes, ever — and this check is deliberately distinct from
        // "no mapping for this code", exactly as
        // `HapticPatterns.isSilentByContract` (`HapticPatterns.kt:134-141`)
        // keeps the two apart.
        if WatchHapticVocabulary.isSilentByContract(code: code) { return }
        guard haptics.play(code: code, level: level, isReminder: isReminder) != nil else { return }
        lastVibrationMs = Self.nowMs()
        hapticLog = currentHapticLog()
    }

    /// `clearReminder()` on session end / disarm — "a watch that is no longer
    /// listening must not keep buzzing about a conversation that's over"
    /// (`HapticDirector.kt:140-146`).
    private func clearReminder() {
        reminderCode = nil
        reminderRepeats = 0
        reminderLastPlayedMs = 0
        reminderCountdownS = nil
        channelALevel = 0
    }

    private func currentHapticLog() -> [HapticAttempt] {
        switch haptics {
        case let recording as RecordingHaptics: return Array(recording.attempts.suffix(8).reversed())
        case let real as WatchKitHaptics: return Array(real.attempts.suffix(8).reversed())
        default: return []
        }
    }

    // MARK: Test seam

    /// Injects a frame as if the server had sent it. Exists because the only
    /// test surface available is a simulator that cannot produce a real nudge:
    /// no mic, no phone, no relay. `ListeningView`'s debug row calls this, so
    /// the nudge presentation and the reminder ladder can be *seen* rather than
    /// asserted from a document. Guarded to DEBUG so it cannot ship enabled.
    #if DEBUG
    func injectForTest(_ frame: ServerFrame) { apply(frame) }

    /// Drives the store into one named screen for a simulator screenshot. See
    /// `DebugLaunchArguments.swift` for why this seam exists and what it refuses
    /// to do (no fake tokens, no fake sockets, no fabricated server data on the
    /// real paths).
    private func applyPreview(_ preview: DebugPreview) {
        // `.open` is set directly rather than by opening a socket: a socket
        // needs a token, and minting a fake one would put a lie in the Keychain.
        switch preview {
        case .idle:
            phase = .idle
        case .diagnostics:
            phase = .idle
        case .listening:
            phase = .listening
            socketState = .open
        case .nudgeH1:
            phase = .listening
            socketState = .open
            apply(.nudge(channel: "A", level: 1, t: 0, vectors: ["yelling"]))
        case .nudgeH3:
            phase = .listening
            socketState = .open
            apply(.nudge(channel: "A", level: 3, t: 0, vectors: ["yelling"]))
        case .nudgeCutIn:
            phase = .listening
            socketState = .open
            apply(.nudge(channel: "A", level: 2, t: 0, vectors: ["interrupting"]))
        case .positiveE:
            phase = .listening
            socketState = .open
            apply(.positive(code: "E", t: 0))
        }
        if phase == .listening { startTicking() }
    }
    #endif

    static func nowMs() -> Int64 { Int64(Date().timeIntervalSince1970 * 1000) }
}
