import SwiftUI

/// Placeholder wrist screen.
///
/// Shows the bundle's own short version string so a TestFlight build can be
/// eyeballed against the phone app's version — the ITMS-90473 class of
/// mismatch is otherwise invisible until Apple rejects the upload.
struct ContentView: View {
    private var shortVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }

    private var buildNumber: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
    }

    var body: some View {
        VStack(spacing: 6) {
            Text("MindShift")
                .font(.headline)
            Text("v\(shortVersion) (\(buildNumber))")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
