import SwiftUI

/// What the wrist can and cannot say, plus what it was actually asked to play.
///
/// This screen exists because of a specific ruling in the parity spec (§4.3):
///
/// > The watchOS build must either **(a)** ship with the vocabulary explicitly
/// > reduced to level-only cues **and say so on screen**, or **(b)** get its own
/// > `watchos` lane in the contract file … Option (b) is the honest one and is
/// > the recommendation.
///
/// This build does (a) now and is set up for (b): `WatchHapticVocabulary` is the
/// watchOS column, written in one place with a `lost` line per code, and the
/// recommended next step is promoting it into
/// `server/tests/fixtures/policy_vectors/nudge_vocabulary.json` so the same
/// cross-runtime vector test that pins the Android mapping pins this one too.
/// That is a change to a file three other runtimes replay, so it is a
/// deliberate, separate decision rather than something to slip in.
///
/// The haptic log below is the other half. `HapticDirector.reportHapticPath`
/// (`HapticDirector.kt:214-264`) tells the owner remotely which physical path a
/// cue took on Android; watchOS has one path and no observability of it, so the
/// only thing that can ever be reported is *what we asked for*. That is what
/// this shows — and in the simulator, where there is no actuator at all, it is
/// the ONLY evidence a cue happened.
struct DiagnosticsView: View {
    @EnvironmentObject private var store: WatchStore

    var body: some View {
        List {
            Section("Wrist vocabulary") {
                Text(WatchHapticVocabulary.honestSummary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(WatchHapticVocabulary.codeOrder, id: \.self) { code in
                    if let row = WatchHapticVocabulary.table[code] {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 4) {
                                Text(code).font(.caption).fontWeight(.bold)
                                Text(fidelityBadge(row.fidelity))
                                    .font(.system(size: 9))
                                    .padding(.horizontal, 4)
                                    .padding(.vertical, 1)
                                    .background(badgeColor(row.fidelity).opacity(0.25),
                                                in: Capsule())
                                Spacer()
                                Text(tapSummary(code))
                                    .font(.system(size: 10))
                                    .foregroundStyle(.secondary)
                            }
                            if !row.lost.isEmpty {
                                Text(row.lost)
                                    .font(.system(size: 10))
                                    .foregroundStyle(.tertiary)
                            }
                        }
                        .padding(.vertical, 1)
                    }
                }
            }

            Section("Asked to play") {
                if store.hapticLog.isEmpty {
                    Text("Nothing yet.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                } else {
                    ForEach(store.hapticLog) { attempt in
                        Text(attempt.summary)
                            .font(.system(size: 11, design: .monospaced))
                    }
                }
                #if targetEnvironment(simulator)
                Text("Simulator: no actuator. These are requests, not buzzes.")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange)
                #endif
            }

            Section("Policy") {
                LabeledRow(label: "Source", value: store.policyBacking)
                if store.policyParityFailures.isEmpty {
                    Text("Swift and Kotlin agree.")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                } else {
                    // A cross-runtime drift belongs on the wrist, not in a log
                    // nobody reads. The ramp already drifted once this way
                    // (`HapticPatterns.kt:97-101`, 2026-09-06).
                    ForEach(store.policyParityFailures, id: \.self) { failure in
                        Text(failure)
                            .font(.system(size: 10))
                            .foregroundStyle(.red)
                    }
                }
            }

            Section("Account") {
                LabeledRow(label: "Phone", value: store.phoneReachable ? "Reachable" : "Not reachable")
                if let standing = store.standing {
                    if let sessions = standing.sessions {
                        LabeledRow(label: "Sessions", value: "\(sessions)")
                    }
                    if let calm = standing.calmScore {
                        LabeledRow(label: "Calm", value: String(format: "%.0f", calm))
                    }
                }
                Button("Sign out", role: .destructive) { store.signOut() }
                    .font(.caption)
            }

            #if DEBUG
            // The simulator cannot produce a real nudge: no mic, no phone, no
            // relay. These inject server frames verbatim so the presentation and
            // the reminder ladder can be seen rather than asserted from a
            // document. DEBUG-only, so it cannot ship enabled.
            Section("Debug: inject a frame") {
                Button("H level 1") { store.injectForTest(.nudge(channel: "A", level: 1, t: 0, vectors: ["yelling"])) }
                Button("H level 3") { store.injectForTest(.nudge(channel: "A", level: 3, t: 0, vectors: ["yelling"])) }
                Button("Cut in (C)") { store.injectForTest(.nudge(channel: "A", level: 2, t: 0, vectors: ["interrupting"])) }
                Button("Listened (E)") { store.injectForTest(.positive(code: "E", t: 0)) }
                Button("Calm streak (K)") { store.injectForTest(.positive(code: "K", t: 0)) }
                Button("Clear (level 0)") { store.injectForTest(.nudge(channel: "A", level: 0, t: 0, vectors: ["yelling"])) }
            }
            .font(.caption2)
            #endif
        }
        .navigationTitle("Details")
    }

    private func fidelityBadge(_ fidelity: WatchHapticVocabulary.Fidelity) -> String {
        fidelity.shortLabel
    }

    private func badgeColor(_ fidelity: WatchHapticVocabulary.Fidelity) -> Color {
        switch fidelity {
        case .full: return .green
        case .countOnly: return .yellow
        case .approximate: return .orange
        case .collapsed: return .red
        }
    }

    private func tapSummary(_ code: String) -> String {
        guard let plan = WatchHapticVocabulary.plan(forCode: code, level: 3) else { return "silent" }
        return plan.taps.map(\.rawValue).joined(separator: "+")
    }
}

private struct LabeledRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Spacer()
            Text(value).font(.caption2)
        }
    }
}
