import Foundation

/// The seam between this watchOS app and the cross-runtime nudge policy.
///
/// ## Why a seam and not a direct `import MindShiftShared`
///
/// The policy has ONE implementation for the whole product — the KMP module at
/// `apps/watch/shared/`, which is what `server/tests/fixtures/policy_vectors/*.json`
/// pins across Python, TypeScript and Kotlin. `docs/plans/2026-09-21-watchos-parity-spec.md`
/// §18 is explicit that a *hand-transcribed* fourth copy is how the channel-A
/// ramp already drifted once (`HapticPatterns.kt:97-101`, 2026-09-06), so the
/// preferred wiring is: Swift calls the Kotlin.
///
/// The obstacle is the build, not the language. `MindShiftShared.xcframework` is
/// produced by Gradle (`./gradlew :shared:assembleMindShiftSharedXCFramework`),
/// and EAS cloud builds of the iOS app do not run Gradle — the iOS build image
/// has no Android SDK, and `apps/watch/shared/build.gradle.kts` cannot even
/// *configure* without one (it applies the Android library plugin). So the
/// framework has to arrive in the cloud build some other way. See the seam
/// decision recorded in `docs/plans/2026-09-21-watchos-parity-spec.md` §18 and
/// the recommendation in this session's report.
///
/// Until that lands, this file keeps the app compiling and correct **both ways**:
///
/// * `#if canImport(MindShiftShared)` — every policy question is answered by the
///   Kotlin, and `SharedPolicy.parityFailures` re-checks the Swift fallback's
///   constants against the Kotlin ones at runtime so a drift is *visible on the
///   wrist* rather than silent.
/// * otherwise — the pure-Swift `SwiftNudgeSchedule` / `SwiftWatchVocabulary`
///   answer, and they are pinned by `scripts/watchos_policy_vectors.swift`
///   replaying the same `server/tests/fixtures/policy_vectors/*.json` the other
///   three runtimes replay. That is the spec's own non-negotiable fallback
///   (§18: *"replay the same JSON vector fixtures from Swift tests"*).
///
/// Nothing in `UI/` or `IO/` may import `MindShiftShared` directly. Every call
/// goes through `SharedPolicy.schedule` / `SharedPolicy.vocabulary`, so swapping
/// the backing implementation is a one-file change.
#if canImport(MindShiftShared)
import MindShiftShared
#endif

// MARK: - Protocols

/// PRD §6's repeat ladder — `apps/watch/shared/.../NudgeHapticSchedule.kt`.
protocol NudgeScheduling {
    /// Channel-A level 0...3 for a heat score, or 0 when no haptic is warranted.
    func level(forScore score: Int) -> Int
    /// Milliseconds between repeats of a level-`level` reminder on its
    /// `repeatIndex`-th repeat, or `nil` when that level never repeats.
    func reminderIntervalMs(level: Int, repeatIndex: Int) -> Int64?
    /// Whether a reminder is due. Pure; the caller decides whether to play.
    func reminderDue(level: Int, lastPlayedMs: Int64, nowMs: Int64, repeatIndex: Int) -> Bool
}

/// The `code → what the user is told` table —
/// `apps/watch/shared/.../NudgeVocabulary.kt`, pinned by
/// `server/tests/fixtures/policy_vectors/nudge_vocabulary.json`.
protocol NudgeVocabularyLookup {
    /// The strongest code a `NudgeEvent.vectors` list maps to, worst-first.
    func code(forVectors vectors: [String]) -> String?
    /// Icon / name / flash text for a code, or `nil` for an unknown code.
    func entry(forCode code: String) -> VocabularyEntry?
}

/// The subset of a vocabulary entry a wrist screen can actually use. The full
/// Kotlin entry carries a `HapticWaveform` map that watchOS cannot play at all
/// (see `WatchHapticVocabulary`), so it is deliberately not surfaced here.
struct VocabularyEntry: Equatable {
    let code: String
    let vector: String
    let icon: String
    let name: String
    /// The line the screen flashes, or `nil` when the vocabulary defines none
    /// (E, R, K, P) — never invent one.
    let flashText: String?
    /// `true` for D / E / R / K: a positive never moves a level and never arms
    /// the reminder (`HapticDirector.kt:180-196`).
    let isPositive: Bool
    /// Levels this code can fire at: `[1, 2, 3]` for alerts, `[1]` for positives.
    let levels: [Int]
    /// The detector vector names that raise this code — `NudgeVocabulary.kt`'s
    /// `sources`. A `nudge` frame's `vectors` list carries THESE
    /// ("yelling", "interrupting", "airtime", "hr_spike"), not the vocabulary
    /// id, which is why `NudgeVocabulary.forVector` looks in both maps
    /// (`NudgeVocabulary.kt:294-296`).
    let sources: [String]
}

// MARK: - Resolution

enum SharedPolicy {
    /// Where the policy answers are coming from, shown on the diagnostics screen
    /// so "which implementation is live" is never a guess.
    enum Backing: String {
        case kotlinShared = "KMP MindShiftShared"
        case swiftFallback = "Swift fallback (vector-pinned)"
    }

    #if canImport(MindShiftShared)
    static let backing: Backing = .kotlinShared
    static let schedule: NudgeScheduling = KotlinNudgeSchedule()
    static let vocabulary: NudgeVocabularyLookup = KotlinWatchVocabulary()
    #else
    static let backing: Backing = .swiftFallback
    static let schedule: NudgeScheduling = SwiftNudgeSchedule()
    static let vocabulary: NudgeVocabularyLookup = SwiftWatchVocabulary()
    #endif

    /// Human-readable descriptions of every place the Swift fallback disagrees
    /// with the Kotlin it mirrors. Empty when the two agree, and empty (not
    /// "unknown") when the Kotlin is not linked — there is nothing to compare
    /// against then, and the JSON vector replay is the proof instead.
    ///
    /// This exists because a silent divergence is the exact failure this repo
    /// has already paid for once. When the framework IS linked, both
    /// implementations are constructed and compared at launch; it costs
    /// microseconds and turns a drift into a visible line of text.
    static func parityFailures() -> [String] {
        #if canImport(MindShiftShared)
        var failures: [String] = []
        let kotlin = KotlinNudgeSchedule()
        let swift = SwiftNudgeSchedule()
        for score in stride(from: 0, through: 100, by: 1) {
            let k = kotlin.level(forScore: score)
            let s = swift.level(forScore: score)
            if k != s { failures.append("levelForScore(\(score)): kotlin \(k) != swift \(s)") }
        }
        // -1 and 4...7 on purpose: `planFor` clamps rather than throwing, so an
        // out-of-range level is a real (and easy to get wrong) case.
        for level in -1...7 {
            for repeatIndex in -1...18 {
                let k = kotlin.reminderIntervalMs(level: level, repeatIndex: repeatIndex)
                let s = swift.reminderIntervalMs(level: level, repeatIndex: repeatIndex)
                if k != s {
                    failures.append(
                        "reminderIntervalMs(\(level), \(repeatIndex)): kotlin \(k.map(String.init) ?? "nil")"
                            + " != swift \(s.map(String.init) ?? "nil")"
                    )
                }
            }
        }
        let kVocab = KotlinWatchVocabulary()
        let sVocab = SwiftWatchVocabulary()
        for code in WatchHapticVocabulary.codeOrder {
            let k = kVocab.entry(forCode: code)
            let s = sVocab.entry(forCode: code)
            if k != s { failures.append("vocabulary[\(code)]: kotlin \(String(describing: k)) != swift \(String(describing: s))") }
        }
        return failures
        #else
        return []
        #endif
    }
}

// MARK: - Swift fallback: the reminder ladder

/// Straight mirror of `apps/watch/shared/.../NudgeHapticSchedule.kt`.
/// Every constant below carries the line it came from; change one here without
/// changing it there and `scripts/watchos_policy_vectors.swift` is the thing
/// that is supposed to catch you.
struct SwiftNudgeSchedule: NudgeScheduling {
    // NudgeHapticSchedule.kt:59-68
    static let noHapticAboveScore = 70
    static let level1MinScore = 50
    static let level2MinScore = 30
    static let level1RepeatMs: Int64 = 120_000
    static let level2RepeatMs: Int64 = 60_000
    static let level3RepeatMs: Int64 = 10_000
    /// NudgeHapticSchedule.kt:107 — the back-off ceiling IS level 1's cadence.
    static let reminderBackoffCapMs: Int64 = level1RepeatMs
    /// NudgeHapticSchedule.kt:151 — bounds the shift, not the delay.
    static let backoffMaxDoublings = 16

    func level(forScore score: Int) -> Int {
        // `coerceIn(0, 100)` first, exactly as NudgeHapticSchedule.kt:82 does.
        let s = min(max(score, 0), 100)
        if s > Self.noHapticAboveScore { return 0 }
        if s >= Self.level1MinScore { return 1 }
        if s >= Self.level2MinScore { return 2 }
        return 3
    }

    func reminderIntervalMs(level: Int, repeatIndex: Int) -> Int64? {
        // `planFor` CLAMPS the level to 0...3 rather than throwing
        // (NudgeHapticSchedule.kt:93) — "a haptic layer must never crash the
        // process over a bad level". So level 7 is level 3's cadence here too,
        // not nil; getting that wrong would make a garbage level silent on the
        // wrist and loud on Android.
        let base: Int64
        switch min(max(level, 0), 3) {
        case 1: base = Self.level1RepeatMs
        case 2: base = Self.level2RepeatMs
        case 3: base = Self.level3RepeatMs
        default: return nil // level 0 is silent AND clears the reminder.
        }
        // NudgeHapticSchedule.kt:126-135 — double per repeat, capped. Level 3
        // therefore fires at 10 s, 20 s, 40 s, 80 s, then settles at 120 s:
        // the measured fix for "18 buzzes in three minutes" (:109-124).
        let doublings = min(max(repeatIndex, 0), Self.backoffMaxDoublings)
        let scaled = base &* (Int64(1) << Int64(doublings))
        return min(scaled, Self.reminderBackoffCapMs)
    }

    func reminderDue(level: Int, lastPlayedMs: Int64, nowMs: Int64, repeatIndex: Int) -> Bool {
        guard let interval = reminderIntervalMs(level: level, repeatIndex: repeatIndex) else { return false }
        // `>=` so a caller ticking at exactly the cadence fires on the tick —
        // same direction as PositiveNudgeGate.admit.
        return nowMs - lastPlayedMs >= interval
    }
}

// MARK: - Swift fallback: the vocabulary

/// Mirror of the eight `NudgeVocabulary.ALL` entries
/// (`NudgeVocabulary.kt:127-285`), reduced to the fields a wrist screen uses.
///
/// The *haptics* are deliberately absent: see `WatchHapticVocabulary`, which is
/// where the watchOS degradation is written down rather than pretended away.
struct SwiftWatchVocabulary: NudgeVocabularyLookup {
    /// Worst-first, i.e. `NudgeVocabulary.rank`'s order — a tie between two
    /// firing vectors resolves to the earlier entry here.
    static let all: [VocabularyEntry] = [
        VocabularyEntry(code: "H", vector: "heated", icon: "📈", name: "Heated",
                        flashText: "Take it down a notch", isPositive: false, levels: [1, 2, 3],
                        sources: ["yelling", "aggressive_tone", "activation"]),
        VocabularyEntry(code: "D", vector: "de_escalated", icon: "📉", name: "De-escalated",
                        flashText: "Nice recovery", isPositive: true, levels: [1], sources: []),
        VocabularyEntry(code: "C", vector: "cut_in", icon: "✂️", name: "Cut in",
                        flashText: "Let them finish", isPositive: false, levels: [1, 2, 3],
                        sources: ["interrupting"]),
        VocabularyEntry(code: "A", vector: "hogging", icon: "🎤", name: "Hogging",
                        flashText: "Give them the floor", isPositive: false, levels: [1, 2, 3],
                        sources: ["airtime"]),
        VocabularyEntry(code: "E", vector: "listened", icon: "👂", name: "Listened",
                        flashText: nil, isPositive: true, levels: [1], sources: []),
        VocabularyEntry(code: "R", vector: "repair", icon: "🤝", name: "Repair",
                        flashText: nil, isPositive: true, levels: [1], sources: []),
        VocabularyEntry(code: "K", vector: "calm_streak", icon: "🧘", name: "Calm streak",
                        flashText: nil, isPositive: true, levels: [1], sources: []),
        VocabularyEntry(code: "P", vector: "pulse", icon: "❤️", name: "Pulse",
                        flashText: nil, isPositive: false, levels: [1, 2, 3], sources: ["hr_spike"]),
    ]

    private static let byCode: [String: VocabularyEntry] =
        Dictionary(uniqueKeysWithValues: all.map { ($0.code, $0) })
    /// Vocabulary id -> entry, then source vector -> entry, in that precedence:
    /// `NudgeVocabulary.kt:296` is `byVector[vector] ?: bySource[vector]`.
    private static let byVector: [String: VocabularyEntry] =
        Dictionary(uniqueKeysWithValues: all.map { ($0.vector, $0) })
    private static let bySource: [String: VocabularyEntry] =
        Dictionary(all.flatMap { e in e.sources.map { ($0, e) } }, uniquingKeysWith: { first, _ in first })
    private static let rank: [String: Int] =
        Dictionary(uniqueKeysWithValues: all.enumerated().map { ($0.element.code, $0.offset) })

    func entry(forCode code: String) -> VocabularyEntry? { Self.byCode[code] }

    func code(forVectors vectors: [String]) -> String? {
        // `NudgeVocabulary.codeForVectors`: the strongest code the firing
        // vectors map to, ties breaking on the owner's worst-first order. A
        // vector with no vocabulary entry contributes nothing rather than
        // producing a nameless buzz.
        vectors
            .compactMap { Self.byVector[$0] ?? Self.bySource[$0] }
            .min { (Self.rank[$0.code] ?? .max) < (Self.rank[$1.code] ?? .max) }?
            .code
    }
}

// MARK: - Kotlin-backed implementations

#if canImport(MindShiftShared)
/// Thin adapter over `NudgeHapticSchedule` — the Kotlin object is the authority.
struct KotlinNudgeSchedule: NudgeScheduling {
    private let shared = NudgeHapticSchedule.shared

    func level(forScore score: Int) -> Int {
        Int(shared.levelForScore(score: Int32(score)))
    }

    func reminderIntervalMs(level: Int, repeatIndex: Int) -> Int64? {
        shared.reminderIntervalMs(level: Int32(level), repeatIndex: Int32(repeatIndex))?.int64Value
    }

    func reminderDue(level: Int, lastPlayedMs: Int64, nowMs: Int64, repeatIndex: Int) -> Bool {
        shared.reminderDue(
            level: Int32(level),
            lastPlayedMs: lastPlayedMs,
            nowMs: nowMs,
            repeatIndex: Int32(repeatIndex)
        )
    }
}

/// Thin adapter over `NudgeVocabulary`.
struct KotlinWatchVocabulary: NudgeVocabularyLookup {
    private let shared = NudgeVocabulary.shared

    func code(forVectors vectors: [String]) -> String? {
        shared.codeForVectors(vectors: vectors)
    }

    func entry(forCode code: String) -> VocabularyEntry? {
        guard let e = shared.forCode(code: code) else { return nil }
        return VocabularyEntry(
            code: e.code,
            vector: e.vector,
            icon: e.icon,
            name: e.name,
            flashText: e.flashText,
            isPositive: e.polarity == NudgePolarity.positive,
            levels: e.levels.map { Int(truncating: $0) },
            sources: e.sources
        )
    }
}
#endif
