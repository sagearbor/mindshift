import SwiftUI

/// The 6-character pairing code, and nothing else.
///
/// The Wear original reads *"Enter $it on your phone or gauge.app/pair"*
/// (`ui/SignInScreen.kt:113`) — a human carries the code to any signed-in
/// client, which is why the handshake needs no phone-side code at all and ports
/// at FULL fidelity (spec §11.1–§11.2). Code alphabet is
/// `ABCDEFGHJKMNPQRSTUVWXYZ23456789`: no 0/O/1/I/L, because someone is reading
/// this off a 42 mm screen (`server/watch/routers/pairing.py:166-167`).
struct SignInView: View {
    @EnvironmentObject private var store: WatchStore

    var body: some View {
        ScrollView {
            VStack(spacing: 10) {
                Text("Pair this watch")
                    .font(.headline)

                if let code = store.pairingCode {
                    Text(code)
                        .font(.system(.title2, design: .monospaced))
                        .fontWeight(.semibold)
                        .tracking(2)
                        .padding(.vertical, 6)
                        .padding(.horizontal, 10)
                        .background(Color.accentColor.opacity(0.18), in: RoundedRectangle(cornerRadius: 8))
                        // Read aloud one character at a time; "MindShift" is not
                        // a word VoiceOver should try on a pairing code.
                        .accessibilityLabel(code.map(String.init).joined(separator: " "))

                    Text("Enter it on your phone, or mindshift.app/pair")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    Text("Expires in 10 minutes")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                } else if let error = store.pairingError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .multilineTextAlignment(.center)
                    Button("Try again") { store.beginPairing() }
                        .buttonStyle(.borderedProminent)
                } else {
                    ProgressView()
                        .padding(.vertical, 12)
                }
            }
            .padding(.horizontal, 4)
        }
        // The display sleeps in ~20 s, and the Wear poller dies with its screen
        // (`SignInScreen.kt:37-40`). This store-owned poller survives that, and
        // this only re-mints when nothing is genuinely in flight (spec §11.2).
        .onAppear { store.resumePairingIfNeeded() }
    }
}
