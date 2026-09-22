import SwiftUI

/// The armed / listening screen: the one at-a-glance answer to "is it on?".
///
/// Structure follows `GlanceScreen.kt`'s v0.3.1 design — a status ring whose
/// colour answers on/off independently of the nudge colour inside it — reduced
/// to what watchOS can honestly show. What is deliberately absent:
///
/// * the **sparkline** and the **meter**, because both are drawn from the
///   wearer's own microphone and this app has no mic path (spec §7.2);
/// * the **pulse train**, which is IMPOSSIBLE here and should ship absent
///   rather than broken (spec §6.3);
/// * any claim to be listening while off screen — see the footer line.
struct ListeningView: View {
    @EnvironmentObject private var store: WatchStore

    private var ringColor: Color {
        switch store.socketState {
        case .open:
            // The nudge level owns the ring once a level is up; otherwise the
            // "on" green (`GlanceScreen.kt`'s STATUS_ON_GREEN 0xFF2E7D32).
            switch store.channelALevel {
            case 3: return Color(red: 0.78, green: 0.16, blue: 0.16)
            case 2: return Color(red: 0.90, green: 0.49, blue: 0.13)
            case 1: return Color(red: 0.95, green: 0.77, blue: 0.22)
            default: return Color(red: 0.18, green: 0.49, blue: 0.20)
            }
        case .connecting, .reconnecting:
            return Color(red: 0.55, green: 0.55, blue: 0.55)
        case .closed, .failed:
            return Color(red: 0.78, green: 0.16, blue: 0.16) // STATUS_OFF_RED
        }
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {
                ZStack {
                    Circle()
                        .stroke(ringColor, lineWidth: 6)
                        .frame(width: 86, height: 86)
                    VStack(spacing: 0) {
                        if let nudge = store.lastNudge {
                            Text(nudge.icon).font(.system(size: 30))
                            Text(nudge.name)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else {
                            Text(store.socketState == .open ? "On" : "Off")
                                .font(.title3)
                                .fontWeight(.semibold)
                        }
                    }
                }
                .padding(.top, 2)
                .accessibilityElement(children: .combine)

                Text(store.socketState.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if let nudge = store.lastNudge {
                    NudgeFlashView(nudge: nudge)
                }

                if let countdown = store.reminderCountdownS, store.channelALevel > 0 {
                    // The PRD §6 ladder, visible. Level 3 fires at 10 s, 20, 40,
                    // 80, then settles at 120 — a back-off, because flat
                    // repetition measured at 18 buzzes in three minutes
                    // (`NudgeHapticSchedule.kt:109-124`).
                    Text("Level \(store.channelALevel) · repeats in \(countdown)s")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                Button(store.phase == .listening ? "Stop" : "Start") {
                    if store.phase == .listening { store.stopListening() } else { store.startListening() }
                }
                .buttonStyle(.borderedProminent)
                .tint(store.phase == .listening ? .red : .accentColor)

                // The honest limit, on the screen rather than only in a doc.
                // `playHaptic` has no documented background guarantee
                // (`WKInterfaceDevice.h:104`) and the only non-frontmost
                // repeating-haptic path is a scheduled alarm capped at a 60 s
                // repeat (`WKExtendedRuntimeSession.h:142-148`), which cannot
                // express level 1's two-minute cadence.
                if store.phase == .listening {
                    Text("Nudges reach your wrist while this screen is on.")
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                        .padding(.top, 2)
                }

                NavigationLink("Details") { DiagnosticsView() }
                    .font(.caption2)
            }
            .padding(.horizontal, 4)
        }
    }
}

/// One nudge, as the screen must say it.
///
/// This view carries more weight on watchOS than on Wear, and that is the whole
/// point: five of the eight cues cannot be told apart by feel here
/// (`WatchHapticVocabulary`), so the identity of what fired has to live on the
/// display. The vocabulary already anticipated this for E and R —
/// *"the wrist says 'that was good', the screen says which good thing"*
/// (`NudgeVocabulary.kt:225-226`) — and on watchOS that bargain extends to D, C,
/// A and P as well.
struct NudgeFlashView: View {
    let nudge: WatchStore.NudgeDisplay

    var body: some View {
        VStack(spacing: 2) {
            if let flash = nudge.flashText {
                // Only the vocabulary's own line, never an invented one: three
                // codes (E, R, K) define no flash text and must show none.
                Text(flash)
                    .font(.footnote)
                    .fontWeight(.medium)
                    .multilineTextAlignment(.center)
            }
            Text(nudge.isPositive ? "Well done" : "Level \(nudge.level)")
                .font(.caption2)
                .foregroundStyle(nudge.isPositive ? .green : .secondary)
            if nudge.fidelity != .full {
                Text("wrist: \(nudge.fidelity.rawValue)")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 2)
    }
}
