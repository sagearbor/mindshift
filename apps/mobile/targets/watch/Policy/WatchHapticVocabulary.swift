import Foundation

/// **The watchOS haptic degradation, written down.**
///
/// The product's nudge design is *"every LEVEL difference is a rhythm
/// difference"* (`apps/watch/shared/.../NudgeVocabulary.kt:18-24`): eight codes,
/// eight distinct rhythms, so the wrist says WHICH behaviour fired, not only how
/// bad it was. Wear OS can express that — `VibrationEffect.createWaveform` takes
/// arbitrary timings and per-segment amplitudes (`RealVibratorPort.kt:32-34`).
///
/// watchOS cannot. `WKInterfaceDevice.playHaptic(_:)` is the only playback API,
/// it takes a member of a fixed 9-value enum, and it has **no duration
/// parameter, no amplitude parameter, no timings array, and no composition
/// builder** (`WKInterfaceDevice.h:16-33`, `:104`). `CoreHaptics.framework` is
/// not in the watchOS SDK at all, so `CHHapticPattern` — the iOS API that could
/// have expressed these waveforms — does not exist on the wrist. There is also
/// no support-probing call: `playHaptic` returns `Void`, so unlike
/// `HapticDirector.reportHapticPath` (`HapticDirector.kt:214-264`) there is no
/// way to know whether anything played.
///
/// So this table is deliberately *not* "one buzz reused for eight codes with the
/// labels still on". Each entry records the cue we can actually play, the
/// fidelity that cue has against the designed one, and **what the wearer loses**.
/// `WatchHapticFidelityView` renders this table verbatim, so the degradation is
/// on the wrist's own screen rather than only in a document.
///
/// Reference: `docs/plans/2026-09-21-watchos-parity-spec.md` §4.1–§4.5, which
/// rates this row IMPOSSIBLE (capability #3) and works through it code by code.
enum WatchHapticVocabulary {
    /// How close the watchOS cue gets to the designed one.
    enum Fidelity: String {
        /// The designed cue survives exactly. Exactly one code is here.
        case full = "full"
        /// The level's tap COUNT survives; the amplitude/length ramp does not.
        case countOnly = "count only"
        /// A different-feeling cue in the right category, but not the design.
        case approximate = "approximate"
        /// Cannot be distinguished from another code by feel. The screen has to
        /// carry the identity.
        case collapsed = "collapsed"

        var shortLabel: String {
            switch self {
            case .full: return "FULL"
            case .countOnly: return "COUNT"
            case .approximate: return "APPROX"
            case .collapsed: return "SCREEN"
            }
        }
    }

    /// The watchOS haptic types this app will ever ask for. Deliberately a
    /// narrow subset: the three navigation types require an active navigation
    /// session and the two underwater ones a depth session
    /// (`WKInterfaceDevice.h:26`, `:30`), so asking for them here would be a
    /// silent no-op.
    enum Cue: String {
        /// No cue, ever. K's contract.
        case silence
        /// `WKHapticType.click`, `count` times, `interTapMs` apart.
        case clicks
        /// `WKHapticType.success` once — categorically unlike an alert, so
        /// praise does not read as an alarm (spec §4.5).
        case success
    }

    /// The three `WKHapticType` members this app asks for, kept as a plain enum
    /// so the pure policy in this file never imports WatchKit (and so the
    /// vector-replay script can run it on macOS).
    enum TapType: String {
        case click
        case success
        case notification
    }

    /// A concrete, playable cue: the exact sequence of `playHaptic` calls, each
    /// separated by `interTapMs`.
    struct Plan {
        let code: String
        let level: Int
        let taps: [TapType]
        let fidelity: Fidelity
        /// One line naming what this plan cannot say, for telemetry and the
        /// fidelity screen. Empty for K's silence.
        let lost: String
    }

    /// The `playHaptic` sequence for a code at a level, or `nil` when nothing
    /// may play (K, an unknown code, or level 0).
    ///
    /// Level 3's first tap is `.notification` rather than `.click`: the designed
    /// L3 cue is a *rising* three-tap ramp and watchOS can vary neither length
    /// nor amplitude, so the only remaining way to mark the worst level as
    /// categorically worse — rather than merely one tap longer — is a different
    /// enum member. Whether a wearer can actually feel `.notification` as
    /// different from `.click` through a band is **UNVERIFIABLE WITHOUT
    /// HARDWARE** (spec §2, finding 3), which is why it is one line here and not
    /// a design the rest of the app depends on.
    static func plan(forCode code: String, level: Int) -> Plan? {
        guard let row = table[code] else { return nil }
        switch row.cue {
        case .silence:
            return nil
        case .success:
            // Positives are unleveled by contract: a "level 2 well done" is not
            // a thing (`NudgeVocabulary.hapticFor`). One cue, always.
            return Plan(code: code, level: 1, taps: [.success], fidelity: row.fidelity, lost: row.lost)
        case .clicks:
            let count = tapCount(forLevel: level)
            guard count > 0 else { return nil }
            var taps: [TapType] = Array(repeating: .click, count: count)
            if count == 3 { taps[0] = .notification }
            return Plan(code: code, level: level, taps: taps, fidelity: row.fidelity, lost: row.lost)
        }
    }

    struct WatchCue {
        /// Vocabulary code: H D C A E R K P.
        let code: String
        /// What we play per level (index 1...3; positives are unleveled and use
        /// level 1 — `NudgeVocabulary.hapticFor` does the same).
        let cue: Cue
        let fidelity: Fidelity
        /// What the designed cue was, in one line, for the on-wrist explanation.
        let designed: String
        /// What the wearer loses versus Wear OS. Empty only for K.
        let lost: String
    }

    /// Worst-first, matching `SwiftWatchVocabulary.all` / `NudgeVocabulary.ALL`.
    static let codeOrder = ["H", "D", "C", "A", "E", "R", "K", "P"]

    /// The whole degradation, one row per code.
    static let table: [String: WatchCue] = [
        "H": WatchCue(
            code: "H", cue: .clicks, fidelity: .countOnly,
            designed: "Rising: •, • •, • • • — tap length climbs 60→110→200 ms so the cue BUILDS.",
            lost: "The ramp. L2 and L3 become two and three identical taps; "
                + "\"the cue builds\" was the design (NudgeVocabulary.kt:142-148)."
        ),
        "D": WatchCue(
            code: "D", cue: .success, fidelity: .collapsed,
            designed: "A falling ramp — the only cue in the vocabulary that fades.",
            lost: "Everything that made it D. Nothing fades on watchOS, so D feels "
                + "identical to E and R; only the screen's \"Nice recovery\" tells them apart."
        ),
        "C": WatchCue(
            code: "C", cue: .clicks, fidelity: .collapsed,
            designed: "• —, • • —, • • • — : short taps then one long 300 ms \"stop\" buzz.",
            lost: "The long buzz. watchOS has no buzz of any length, so C is "
                + "indistinguishable from H by feel."
        ),
        "A": WatchCue(
            code: "A", cue: .clicks, fidelity: .collapsed,
            designed: "Slow — — — : 250 ms buzzes, 300 ms apart, soft. Deliberately unhurried.",
            lost: "The unhurriedness. Clicks 300 ms apart read as urgent taps — the "
                + "opposite of \"you're talking a lot is not an emergency\"."
        ),
        "E": WatchCue(
            code: "E", cue: .success, fidelity: .approximate,
            designed: "Soft •• — two swells, a continuous buzz whose strength changes.",
            lost: "The swell. .success is at least categorically unlike an alert, "
                + "which is the part that matters for praise."
        ),
        "R": WatchCue(
            code: "R", cue: .success, fidelity: .collapsed,
            designed: "The E swell, three times — count 3 of a positive.",
            lost: "The count. R and E are the same cue on watchOS. The vocabulary "
                + "already anticipated this for these two (NudgeVocabulary.kt:225-226: "
                + "\"the wrist says that was good, the screen says which good thing\")."
        ),
        "K": WatchCue(
            code: "K", cue: .silence, fidelity: .full,
            designed: "Never buzzes, ever (haptic = null).",
            lost: ""
        ),
        "P": WatchCue(
            code: "P", cue: .clicks, fidelity: .collapsed,
            designed: "Lub-dub: 90 ms then 150 ms, 100 ms apart — under the 170 ms merge "
                + "floor on purpose, because the near-merge IS the heartbeat.",
            lost: "The heartbeat. Its identity is a sub-merge-floor gap between two "
                + "UNEQUAL taps; watchOS gives neither unequal taps nor controllable timing."
        ),
    ]

    /// Target gap between taps within one cue. `MIN_GAP_MS = 170` is a hard,
    /// pattern-wide never-merge floor on Wear (`HapticPatterns.kt:65`, mirrored
    /// `NudgeVocabulary.kt:106`) and is kept here as the target — but the
    /// difference matters: on Wear the PLATFORM owns intra-pattern timing
    /// because the pattern is one `VibrationEffect.Composition`; here the app
    /// owns it, on a `Task` it has to keep alive, and a suspended app delivers a
    /// truncated cue. There is no atomic multi-tap primitive on watchOS.
    static let interTapMs: UInt64 = 170

    /// Tap count for a channel-A level — the one property of the designed cue
    /// that survives, and the one Brown & Brewster found identifiable ~93% of
    /// the time (`NudgeVocabulary.kt:18-24`). Level 0 is silent *and* clears the
    /// reminder (`HapticDirector.kt:69-80`).
    static func tapCount(forLevel level: Int) -> Int {
        switch min(max(level, 0), 3) {
        case 1: return 1
        case 2: return 2
        case 3: return 3
        default: return 0
        }
    }

    /// The cue for a code at a level, or `nil` when nothing may play.
    ///
    /// `nil` covers two different things that must stay different, exactly as
    /// `HapticPatterns.isSilentByContract` (`HapticPatterns.kt:134-141`) keeps
    /// them different: K is *silent by contract* and an unknown code has *no
    /// mapping*. Both fail to silence — silence is the safe failure for a
    /// haptic — but `isSilentByContract` reports which.
    static func cue(forCode code: String, level: Int) -> WatchCue? {
        guard let row = table[code], row.cue != .silence else { return nil }
        if tapCount(forLevel: level) == 0 && row.cue == .clicks { return nil }
        return row
    }

    /// True when this code never buzzes by design (K). Distinct from "no
    /// mapping for this code", which returns false.
    static func isSilentByContract(code: String) -> Bool {
        table[code]?.cue == .silence
    }

    /// One-line summary for the sign-in / diagnostics screen: how much of the
    /// vocabulary this platform can actually say.
    static var honestSummary: String {
        let counts = Dictionary(grouping: table.values, by: { $0.fidelity })
            .mapValues(\.count)
        let full = counts[.full] ?? 0
        let collapsed = counts[.collapsed] ?? 0
        return "\(full) of \(table.count) cues survive intact; \(collapsed) can't be told apart by feel."
    }
}
