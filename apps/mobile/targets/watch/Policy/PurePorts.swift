import Foundation

/// Swift ports of the Wear app's small, pure decision classes.
///
/// These live in `apps/watch/wearApp/` rather than the shared KMP module, so
/// they are *not* in `MindShiftShared.xcframework` and cannot be reused even
/// once the framework is linked. They were extracted on Android precisely
/// because the Compose shells around them are compile-gated only
/// (`PairingPoller.kt`'s own KDoc says so) — the same reasoning applies double
/// on a platform whose shells cannot be exercised at all without hardware.
///
/// Each port keeps the original's fail direction, because in every one of them
/// the fail direction was the decision. If these ever grow, the right move is to
/// promote them into `apps/watch/shared/src/commonMain` so Android and watchOS
/// share one copy — see the report's recommendation.

// MARK: - PairingPoller

/// Port of `apps/watch/wearApp/.../auth/PairingPoller.kt`.
///
/// The contract that matters: every pairing lifecycle state — pending / claimed
/// / expired — arrives as a decoded, non-`nil` status. `nil` means **transport
/// failure only**, never a lifecycle state (`PairingPoller.kt:13-16`). Without
/// the independent give-up bound, a network outage returning `nil` forever would
/// keep the screen polling forever, because a transport failure never reaches
/// the server long enough to hear an honest "expired".
enum PairingPollOutcome: Equatable {
    case keepPolling
    case signedIn(accountId: String, deviceToken: String)
    case failed(message: String)
}

final class PairingPoller {
    /// 2x the server's 10-minute code TTL (`PairingPoller.kt:69`).
    static let defaultMaxPollingDurationMs: Int64 = 1_200_000
    /// `SignInScreen.kt:32` — and note the Wear behaviour this port keeps: the
    /// poll is driven by the screen, so it stops when the screen leaves. On a
    /// watch whose display sleeps in ~20 s that is *more* of a problem than on
    /// Wear, which is why `WatchStore` re-polls on `onAppear` rather than
    /// assuming the loop survived.
    static let pollIntervalMs: Int64 = 3_000

    private let now: () -> Int64
    private let maxDurationMs: Int64
    private let startedAtMs: Int64

    init(now: @escaping () -> Int64, maxDurationMs: Int64 = PairingPoller.defaultMaxPollingDurationMs) {
        self.now = now
        self.maxDurationMs = maxDurationMs
        self.startedAtMs = now()
    }

    /// Call once per poll attempt, including a `nil` transport failure. Never
    /// sleeps and never makes the network call — that is the caller's job.
    func onPollResult(_ status: PairingStatus?) -> PairingPollOutcome {
        if let status {
            switch status.status {
            case "claimed":
                if let accountId = status.accountId, let deviceToken = status.deviceToken {
                    return .signedIn(accountId: accountId, deviceToken: deviceToken)
                }
                // Contract violation: the server should never send "claimed"
                // without both fields. An honest terminal failure beats a
                // silently stuck screen.
                return .failed(message: "Something went wrong finishing sign-in. Try again.")
            case "expired":
                return .failed(message: "Code expired. Reopen this screen for a new one.")
            default:
                break // "pending", or anything unknown: fall through to the bound.
            }
        }
        if now() - startedAtMs >= maxDurationMs {
            return .failed(message: "Code expired — get a new one.")
        }
        return .keepPolling
    }
}

// MARK: - ReconnectPolicy

/// Port of `apps/watch/wearApp/.../control/ReconnectPolicy.kt`.
/// Ladder: 2 s, 4 s, 8 s, 16 s, 30 s (`:36-41`, `:73-74`).
final class ReconnectPolicy {
    static let initialDelayMs: Int64 = 2_000
    static let maxDelayMs: Int64 = 30_000
    /// Bounds `attempt`, not the delay (`ReconnectPolicy.kt:79`) — keeps the
    /// shift from growing unbounded across a pathologically long outage.
    private static let maxShift = 8

    private let initial: Int64
    private let maxDelay: Int64
    private var attempt = 0
    private var nextAttemptAtMs: Int64?

    init(initialDelayMs: Int64 = ReconnectPolicy.initialDelayMs,
         maxDelayMs: Int64 = ReconnectPolicy.maxDelayMs) {
        self.initial = initialDelayMs
        self.maxDelay = maxDelayMs
    }

    /// Records a drop while streaming and arms the next attempt. Returns the
    /// delay just armed, purely so callers can log it.
    @discardableResult
    func onFailure(nowMs: Int64) -> Int64 {
        let delay = min(initial << Int64(attempt), maxDelay)
        attempt = min(attempt + 1, Self.maxShift)
        nextAttemptAtMs = nowMs + delay
        return delay
    }

    /// Back to the first rung, pending attempt cleared. Call on a reconnect that
    /// opened, on a fresh session, and on disarm.
    func reset() {
        attempt = 0
        nextAttemptAtMs = nil
    }

    /// `false` whenever `streaming` is false — the give-up condition. Even a
    /// fully elapsed delay must not fire once the session is over.
    func isAttemptDue(streaming: Bool, nowMs: Int64) -> Bool {
        guard streaming, let at = nextAttemptAtMs else { return false }
        return nowMs >= at
    }

    var armedDelayRemainingMs: Int64? { nextAttemptAtMs }
}

// MARK: - PositiveNudgeGate

/// Port of `PositiveNudgeGate` (`apps/watch/shared/.../NudgeVocabulary.kt:323-338`).
/// At most one positive per 120 s **across D/E/R together** (`:118`).
///
/// This one IS in the XCFramework, but it is 12 lines of arithmetic and the seam
/// does not currently surface it; when the framework is linked, prefer wiring it
/// through `SharedPolicy` over keeping this copy.
final class PositiveNudgeGate {
    static let positiveCapS: Double = 120.0

    private let capS: Double
    private var lastT: Double?

    init(capS: Double = PositiveNudgeGate.positiveCapS) { self.capS = capS }

    /// True (and the clock resets) when this positive may be delivered.
    /// `>=` not `>`, so a caller ticking at exactly the cadence fires.
    func admit(t: Double) -> Bool {
        if let last = lastT, t - last < capS { return false }
        lastT = t
        return true
    }

    /// Seconds until the next positive may fire; 0 when one may fire now.
    func waitS(t: Double) -> Double {
        guard let last = lastT else { return 0 }
        return max(0, capS - (t - last))
    }
}

// MARK: - ShoutTapGate

/// Port of `apps/watch/wearApp/.../haptics/ShoutTapGate.kt`.
///
/// Present but **unused in this build**: it gates the ARMED shout-tap, which
/// fires off the wearer's own microphone, and this app has no microphone path
/// (spec §7.2 rates the all-day armed sentinel IMPOSSIBLE on watchOS and §19
/// puts a user-started mic session at step 4). It is here because it is 20 lines
/// and because the mic lane, if it is ever built, must not re-derive the fail
/// direction: a suppressed call does **not** restart the clock, so sustained
/// near-loud speech cannot postpone taps forever (`ShoutTapGate.kt:26-30`), and
/// a backwards-stepping clock yields silence rather than a tap storm (`:16-20`).
final class ShoutTapGate {
    private let minIntervalMs: Int64
    private var lastTapAtMs: Int64?

    init(minIntervalMs: Int64 = 2_000) { self.minIntervalMs = minIntervalMs }

    func onLoudWindow(nowMs: Int64) -> Bool {
        if let last = lastTapAtMs, nowMs - last < minIntervalMs { return false }
        lastTapAtMs = nowMs
        return true
    }

    func reset() { lastTapAtMs = nil }
}

// MARK: - companionSessionId

/// Port of `apps/watch/wearApp/.../control/CompanionSession.kt`.
///
/// `companion-YYYYMMDD-<account>`: one deterministic live-session id per account
/// per UTC day, so a day of reconnects reuses one id and the server logs stay
/// coherent. The account segment is sanitized to URL-path-safe characters
/// because it rides in the WS path — it only disambiguates logs, it is **not**
/// the auth (the `?token=` query param is).
func companionSessionId(accountId: String, now: Date = Date()) -> String {
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd"
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.locale = Locale(identifier: "en_US_POSIX")
    let day = formatter.string(from: now)
    let safe = String(accountId.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }.prefix(24))
    return "companion-\(day)-\(safe.isEmpty ? "anon" : safe)"
}
