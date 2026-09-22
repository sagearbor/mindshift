import SwiftUI

/// Routes between the three states the wrist can be in. Owns no decisions of its
/// own: `WatchStore.phase` is the only thing it reads.
///
/// The version line stays on the idle screen deliberately. It was the whole
/// point of the original placeholder — a TestFlight build has to be eyeballable
/// against the phone app's version, because the ITMS-90473 class of mismatch is
/// otherwise invisible until Apple rejects the upload, and
/// `apps/mobile/plugins/withWatchTargetVersion.js` is what keeps the two in step.
struct ContentView: View {
    @EnvironmentObject private var store: WatchStore

    private var shortVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
    }

    private var buildNumber: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
    }

    var body: some View {
        NavigationStack {
            #if DEBUG
            // The simulator has no way to tap into a NavigationLink from
            // `simctl`, so the diagnostics screen is otherwise unscreenshottable
            // — and it is the screen that carries the haptic degradation, which
            // is the one thing about this port that most needs looking at.
            if DebugPreview.requested == .diagnostics {
                DiagnosticsView()
            } else {
                phaseView
            }
            #else
            phaseView
            #endif
        }
        .onAppear { store.start() }
    }

    @ViewBuilder
    private var phaseView: some View {
        Group {
            switch store.phase {
            case .loading:
                ProgressView()
            case .signIn:
                SignInView()
            case .idle:
                IdleView(shortVersion: shortVersion, buildNumber: buildNumber)
            case .listening:
                ListeningView()
            }
        }
    }
}

/// Signed in, not listening. One button, one honest sentence about what the
/// button does, and the version.
struct IdleView: View {
    @EnvironmentObject private var store: WatchStore
    let shortVersion: String
    let buildNumber: String

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {
                Text("MindShift").font(.headline)

                Button("Start listening") { store.startListening() }
                    .buttonStyle(.borderedProminent)

                // COMPANION mode, named for what it is: the PHONE listens, the
                // wrist keeps a socket open purely to receive relayed nudges and
                // render them — no mic, no PCM, no trigger evaluation
                // (`ModePolicy.kt:21-24`). It is the highest-fidelity mode
                // watchOS can host (spec §10.3) precisely because it removes
                // every part this platform cannot do.
                Text("Your phone listens. This watch buzzes.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                NavigationLink("Details") { DiagnosticsView() }
                    .font(.caption2)

                Text("v\(shortVersion) (\(buildNumber))")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .padding(.top, 4)
            }
            .padding(.horizontal, 4)
        }
        .task { await store.refreshStanding() }
    }
}

#Preview("Sign in") {
    SignInView().environmentObject(WatchStore())
}

#Preview("Listening") {
    ListeningView().environmentObject(WatchStore())
}
