#if DEBUG
import Foundation

/// Launch arguments that drive the app into a named screen, for the simulator.
///
/// **Why this exists.** The owner has no Apple Watch and no iPhone, so the
/// simulator is the entire test surface (spec §2.1), and the simulator can
/// produce exactly none of the inputs that move this app past its first screen:
/// no paired phone to claim a pairing code, no relay to push a nudge, no
/// microphone lane at all. Without a seam, "verified in the simulator" would
/// mean "the sign-in screen renders", and every other screen would ship
/// unlooked-at.
///
/// So these arguments put the store into a state and stop. They do **not**
/// fabricate server data into the normal paths: nothing here writes a token to
/// the Keychain, mints an account, or opens a socket. `previewSignedIn` marks
/// the store signed-in *for display only*, which is why `startListening()` is
/// bypassed in that mode rather than being fed a fake token.
///
/// Compiled out of Release builds entirely by the `#if DEBUG` around the whole
/// file, so it cannot reach TestFlight or the App Store.
///
/// Usage:
/// ```
/// xcrun simctl launch booted com.sagearbor.mindshift.app.watchkitapp -uiPreview idle
/// xcrun simctl launch booted com.sagearbor.mindshift.app.watchkitapp -uiPreview listening
/// xcrun simctl launch booted com.sagearbor.mindshift.app.watchkitapp -uiPreview nudge-h3
/// ```
enum DebugPreview: String {
    case idle
    case listening
    case nudgeH1 = "nudge-h1"
    case nudgeH3 = "nudge-h3"
    case nudgeCutIn = "nudge-c2"
    case positiveE = "positive-e"
    case diagnostics

    /// `-uiPreview <value>`, read from the launch arguments. `nil` in normal use.
    static var requested: DebugPreview? {
        let args = ProcessInfo.processInfo.arguments
        guard let flagIndex = args.firstIndex(of: "-uiPreview"),
              args.index(after: flagIndex) < args.endIndex
        else { return nil }
        return DebugPreview(rawValue: args[args.index(after: flagIndex)])
    }
}
#endif
