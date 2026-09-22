import SwiftUI

/// Entry point for the MindShift watchOS companion.
///
/// Scope is `docs/plans/2026-09-21-watchos-parity-spec.md` §19 steps 2 and 3 —
/// pairing/auth over plain HTTP, and COMPANION mode (the phone listens, the
/// wrist receives relayed nudges). Steps 4–6 (a mic session under
/// `HKWorkoutSession`, widgets, retro-capture) are not built, and the all-day
/// armed sentinel and the proportional pulse train never will be: the spec rates
/// both IMPOSSIBLE on this platform and says to ship them absent rather than
/// broken.
///
/// `WatchStore` is created once here and injected, so the socket, the pairing
/// poller and the reminder ladder survive view identity changes — on a watch
/// whose screen sleeps in ~20 seconds, anything scoped to a view's lifetime is
/// scoped to about twenty seconds.
@main
struct MindShiftWatchApp: App {
    @StateObject private var store = WatchStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
    }
}
