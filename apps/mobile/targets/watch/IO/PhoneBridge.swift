import Foundation
#if canImport(WatchConnectivity)
import WatchConnectivity
#endif

/// WatchConnectivity, as an **additive** channel — never a dependency.
///
/// ## Read this before extending it
///
/// `docs/plans/2026-09-21-watchos-parity-spec.md` §10.2 is explicit:
///
/// > `WatchConnectivity.framework` is present … **but nothing in this protocol
/// > needs it.** Do not introduce a WatchConnectivity dependency; it would make
/// > the watch app *less* standalone than the Wear one.
///
/// That ruling is correct and is respected here. The Wear app is genuinely
/// standalone — verified by exhaustive grep: no `MessageClient`, no `DataClient`,
/// no Play Services anywhere in `apps/watch/**` — and pairing is two
/// unauthenticated HTTP calls plus a 6-character code a human reads off the
/// watch face (`ui/SignInScreen.kt:113`). All of that ports at FULL fidelity
/// (spec §11.2) and remains the only path this app *requires*.
///
/// So what is this file for? Two things HTTP cannot do, both optional:
///
/// 1. **Skipping the code dance when the phone is right there.** An Apple Watch
///    cannot be set up without an iPhone at all (spec §10.2), so unlike Wear OS
///    the phone is guaranteed to exist. If the phone app is signed in it can
///    hand the wrist its `account_id` + `device_token` directly. If it does not,
///    or is unreachable, the 6-character code path still works unchanged — that
///    is what makes this additive rather than a dependency.
/// 2. **A second delivery path for relayed nudges.** In COMPANION mode the phone
///    is the one listening. The server relay already does the job over the
///    socket, and that stays primary; this is the fallback for the case where
///    the wrist's own socket is down but the phone's is up.
///
/// ## Status: implemented on this side, dark on the other
///
/// **There is no iOS-side counterpart yet.** The React Native app has no
/// WatchConnectivity bridge (no `react-native-watch-connectivity`, no native
/// `WCSessionDelegate` in `apps/mobile/ios/MindShift/`), so nothing will ever
/// send these messages until that is built — `docs/plans/2026-09-21-watchos-mechanism.md`
/// P2 budgets 2–4 days for exactly that, including an untested RN 0.86 pairing.
/// This side is therefore written, wired, and **unexercised**: `reachable` will
/// read false and `lastMessageAt` will stay nil on a real device today. Do not
/// read "it compiles" as "the round trip works".
///
/// Everything crossing this boundary is plist-safe (`String`, `Double`, `Bool`,
/// `Int`) because `WCSession` silently fails on anything else.
protocol PhoneBridging: AnyObject {
    var isSupported: Bool { get }
    var isReachable: Bool { get }
    func activate()
}

/// What the phone can tell the wrist. Keys are the wire contract with the
/// unwritten iOS side; keep them snake_case to match the server's own wire.
enum PhoneMessage {
    static let keyKind = "kind"
    /// `{"kind": "credentials", "account_id": …, "device_token": …}`
    static let kindCredentials = "credentials"
    /// `{"kind": "nudge", "channel": "A", "level": 2, "t": 12.5, "vectors": [...]}`
    static let kindNudge = "nudge"
    /// `{"kind": "positive", "code": "E", "t": 12.5}`
    static let kindPositive = "positive"
    /// `{"kind": "signed_out"}` — the phone signed out; the wrist must too.
    static let kindSignedOut = "signed_out"
}

/// What the bridge hands upward. Deliberately the same shapes `CompanionSocket`
/// produces, so `WatchStore` has ONE nudge pipeline rather than two.
enum PhoneEvent {
    case credentials(accountId: String, deviceToken: String)
    case frame(ServerFrame)
    case signedOut
    case reachabilityChanged(Bool)
}

#if canImport(WatchConnectivity)
final class PhoneBridge: NSObject, PhoneBridging, WCSessionDelegate {
    private let onEvent: (PhoneEvent) -> Void
    private(set) var lastMessageAt: Date?

    init(onEvent: @escaping (PhoneEvent) -> Void) {
        self.onEvent = onEvent
        super.init()
    }

    var isSupported: Bool { WCSession.isSupported() }

    var isReachable: Bool {
        guard WCSession.isSupported() else { return false }
        return WCSession.default.isReachable
    }

    func activate() {
        guard WCSession.isSupported() else { return }
        let session = WCSession.default
        session.delegate = self
        session.activate()
        // An `applicationContext` already set by the phone survives the watch
        // app being closed, so a credentials handoff sent yesterday is still
        // here today. `didReceiveApplicationContext` only fires on *changes*,
        // which is why the current value is read directly on activation.
        handle(session.receivedApplicationContext)
    }

    // MARK: WCSessionDelegate

    func session(
        _ session: WCSession,
        activationDidCompleteWith activationState: WCSessionActivationState,
        error: Error?
    ) {
        onEvent(.reachabilityChanged(session.isReachable))
    }

    func sessionReachabilityDidChange(_ session: WCSession) {
        onEvent(.reachabilityChanged(session.isReachable))
    }

    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        handle(message)
    }

    func session(_ session: WCSession, didReceiveApplicationContext applicationContext: [String: Any]) {
        handle(applicationContext)
    }

    func session(_ session: WCSession, didReceiveUserInfo userInfo: [String: Any] = [:]) {
        handle(userInfo)
    }

    // MARK: Decoding

    private func handle(_ payload: [String: Any]) {
        guard let kind = payload[PhoneMessage.keyKind] as? String else { return }
        lastMessageAt = Date()
        switch kind {
        case PhoneMessage.kindCredentials:
            guard let accountId = payload["account_id"] as? String,
                  let token = payload["device_token"] as? String,
                  !accountId.isEmpty, !token.isEmpty
            else { return }
            onEvent(.credentials(accountId: accountId, deviceToken: token))
        case PhoneMessage.kindNudge:
            onEvent(.frame(.nudge(
                channel: payload["channel"] as? String ?? "A",
                level: (payload["level"] as? NSNumber)?.intValue ?? 0,
                t: (payload["t"] as? NSNumber)?.doubleValue ?? 0,
                vectors: payload["vectors"] as? [String] ?? []
            )))
        case PhoneMessage.kindPositive:
            guard let code = payload["code"] as? String else { return }
            onEvent(.frame(.positive(code: code, t: (payload["t"] as? NSNumber)?.doubleValue ?? 0)))
        case PhoneMessage.kindSignedOut:
            onEvent(.signedOut)
        default:
            break
        }
    }
}
#endif

/// Stands in wherever WatchConnectivity is absent (and in previews). Reports
/// honestly rather than pretending a phone is there.
final class UnavailablePhoneBridge: PhoneBridging {
    var isSupported: Bool { false }
    var isReachable: Bool { false }
    func activate() {}
}
