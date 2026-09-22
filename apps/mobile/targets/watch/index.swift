import SwiftUI

/// Entry point for the MindShift watchOS companion.
///
/// Deliberately minimal: this target exists to prove the build/signing path
/// (P0/P1 of docs/plans/2026-09-21-watchos-mechanism.md). The real wrist UI and
/// the WatchConnectivity session land in P2/P3.
@main
struct MindShiftWatchApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
