import Foundation
#if canImport(WatchKit)
import WatchKit
#endif

/// A record of one asked-for cue. `HapticDirector.reportHapticPath`
/// (`HapticDirector.kt:214-264`) exists on Android because the vibrator can tell
/// you which physical path a cue took — predefined, composed, raw waveform,
/// failed. **watchOS has exactly one path and no observability of it**:
/// `playHaptic` returns `Void`, there is no `areAllEffectsSupported`, and there
/// is no way to learn whether anything reached the actuator.
///
/// So this is deliberately not a port of `reportHapticPath`. It records the only
/// thing that will ever be true — *we asked for N taps of type T at time t* —
/// which is worth recording precisely because it is the only evidence that will
/// ever exist (spec §4.1's closing note).
struct HapticAttempt: Identifiable {
    let id = UUID()
    let at: Date
    let code: String
    let level: Int
    let taps: [WatchHapticVocabulary.TapType]
    let fidelity: WatchHapticVocabulary.Fidelity
    /// `true` when this attempt was a PRD §6 reminder repeat rather than a fresh
    /// escalation.
    let isReminder: Bool

    var summary: String {
        let names = taps.map(\.rawValue).joined(separator: "+")
        return "\(code)\(level > 0 ? String(level) : "") \(names)\(isReminder ? " (repeat)" : "")"
    }
}

/// The wrist, behind a protocol.
///
/// Everything above this line is pure and testable; everything below it is a
/// platform shell that cannot be exercised without hardware — the simulator
/// plays no haptics at all (spec §2, finding 2). Keeping the boundary here is
/// the same discipline `Ports.kt:11-117` follows on Wear, and it is the only
/// reason any of this can be reasoned about on a machine with no Apple Watch.
protocol WristHaptics: AnyObject {
    /// Plays one cue. Returns the attempt it recorded, or `nil` when the code is
    /// silent by contract / unmapped / level 0.
    @discardableResult
    func play(code: String, level: Int, isReminder: Bool) -> HapticAttempt?
}

/// The real thing: `WKInterfaceDevice.playHaptic(_:)`.
///
/// Three properties of this implementation are consequences of the platform, not
/// choices, and each is a downgrade from Wear:
///
/// 1. **The app owns intra-cue timing.** On Wear a multi-tap cue is ONE
///    `VibrationEffect.Composition` and the platform renders it atomically
///    (`HapticPatterns.kt:20-22`). Here it is N separate `playHaptic` calls with
///    our own `Task.sleep` between them, so a suspended app delivers a
///    *truncated* cue — two taps of a three-tap level 3, and no way to know.
/// 2. **No amplitude, no duration.** There is nothing to degrade gracefully to;
///    `FIXED_AMPLITUDE_NO_CONTROL` (`HapticDirector.kt:267`) has no analogue
///    because there is no amplitude to fix.
/// 3. **Frontmost only.** `playHaptic` has no documented background guarantee
///    (`WKInterfaceDevice.h:104`). This app therefore does not pretend to buzz
///    when it is not on screen; `WatchStore` says so on screen instead.
final class WatchKitHaptics: WristHaptics {
    /// Everything asked for this session, newest last. Bounded — a ring, not a
    /// leak: an hour of level-3 reminders is ~30 entries, but a bug that fires
    /// in a loop should not grow memory without bound.
    private(set) var attempts: [HapticAttempt] = []
    private static let attemptRingSize = 64

    private let interTapNanos: UInt64

    init(interTapMs: UInt64 = WatchHapticVocabulary.interTapMs) {
        self.interTapNanos = interTapMs * 1_000_000
    }

    @discardableResult
    func play(code: String, level: Int, isReminder: Bool) -> HapticAttempt? {
        guard let plan = WatchHapticVocabulary.plan(forCode: code, level: level) else {
            // K is silent by contract and an unknown code has no mapping. Both
            // fail to silence; only the first is intentional, and
            // `isSilentByContract` is what tells them apart for logging.
            return nil
        }
        let attempt = HapticAttempt(
            at: Date(),
            code: plan.code,
            level: plan.level,
            taps: plan.taps,
            fidelity: plan.fidelity,
            isReminder: isReminder
        )
        record(attempt)

        #if canImport(WatchKit)
        let taps = plan.taps
        let gap = interTapNanos
        Task { @MainActor in
            let device = WKInterfaceDevice.current()
            for (index, tap) in taps.enumerated() {
                if index > 0 {
                    // 170 ms is Wear's never-merge floor (`HapticPatterns.kt:65`).
                    // Whether the actuator renders two `.click`s 170 ms apart as
                    // two taps or one smear is UNVERIFIABLE WITHOUT HARDWARE —
                    // spec §2, finding 3, and the first thing to measure.
                    try? await Task.sleep(nanoseconds: gap)
                }
                device.play(tap.wkType)
            }
        }
        #endif
        return attempt
    }

    private func record(_ attempt: HapticAttempt) {
        attempts.append(attempt)
        if attempts.count > Self.attemptRingSize {
            attempts.removeFirst(attempts.count - Self.attemptRingSize)
        }
    }
}

/// Records what it was asked for and plays nothing. This is what runs in the
/// simulator whether you pick it or not — the simulator has no actuator — so
/// making that explicit beats pretending a silent `playHaptic` succeeded.
final class RecordingHaptics: WristHaptics {
    private(set) var attempts: [HapticAttempt] = []

    @discardableResult
    func play(code: String, level: Int, isReminder: Bool) -> HapticAttempt? {
        guard let plan = WatchHapticVocabulary.plan(forCode: code, level: level) else { return nil }
        let attempt = HapticAttempt(at: Date(), code: plan.code, level: plan.level,
                                    taps: plan.taps, fidelity: plan.fidelity, isReminder: isReminder)
        attempts.append(attempt)
        return attempt
    }
}

#if canImport(WatchKit)
extension WatchHapticVocabulary.TapType {
    /// The 9-member fixed enum, `WKInterfaceDevice.h:16-33`. Only these three
    /// are used: the navigation types need an active navigation session and the
    /// underwater ones a depth session (`:26`, `:30`), so asking for them
    /// outside those sessions is a silent no-op.
    var wkType: WKHapticType {
        switch self {
        case .click: return .click
        case .success: return .success
        case .notification: return .notification
        }
    }
}
#endif
