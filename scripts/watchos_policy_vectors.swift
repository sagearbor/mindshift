// Replays server/tests/fixtures/policy_vectors/nudge_vocabulary.json against
// the watchOS app's own Swift policy files.
//
// WHY THIS EXISTS
//
// `docs/plans/2026-09-21-watchos-parity-spec.md` §18 is blunt about it: a
// watchOS port adds a THIRD hand-written copy of the nudge vocabulary to a
// product whose whole discipline is that "an emoji or a millisecond can only
// change in one place" (NudgeVocabulary.kt:10-11). The ramp already drifted
// once exactly this way (HapticPatterns.kt:97-101, 2026-09-06). The spec's
// ruling:
//
//   > If KMP is not wired up, the fallback is non-negotiable: replay the same
//   > JSON vector fixtures from Swift tests.
//
// Three runtimes already replay these fixtures — NudgeStateMachineVectorsTest.kt,
// server/tests/test_nudge_vocabulary_vectors.py, and
// apps/mobile/__tests__/nudgeVocabulary.test.ts. This is the fourth.
//
// WHAT IT COVERS, AND WHAT IT DELIBERATELY DOES NOT
//
// It pins the VOCABULARY (codes, icons, names, flash text, polarity, levels,
// source vectors, the 120 s positive cap) and the watchOS haptic column's
// completeness. It does NOT pin the reminder SCHEDULE, because no fixture
// describes it — that is covered instead by the live Kotlin/Swift parity check
// (`SharedPolicy.parityFailures`), which runs 900+ comparisons on the simulator
// and logs the result. `scripts/verify_watchos_policy.sh` runs both lanes.
//
// Run: swift scripts/watchos_policy_vectors.swift   (from the repo root)
// Or via the wrapper, which compiles it together with the app's own sources:
//   ./scripts/verify_watchos_policy.sh

import Foundation

// MARK: - Harness

var failures: [String] = []
var checks = 0

func expect(_ condition: Bool, _ message: @autoclosure () -> String) {
    checks += 1
    if !condition { failures.append(message()) }
}

func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ what: String) {
    checks += 1
    if actual != expected {
        failures.append("\(what): got \(actual), fixture says \(expected)")
    }
}

// MARK: - Fixture

let repoRoot: URL = {
    // The script lives at <root>/scripts/, and is run from the root.
    let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    if FileManager.default.fileExists(
        atPath: cwd.appendingPathComponent("server/tests/fixtures/policy_vectors/nudge_vocabulary.json").path
    ) { return cwd }
    return cwd.deletingLastPathComponent()
}()

let fixtureUrl = repoRoot.appendingPathComponent("server/tests/fixtures/policy_vectors/nudge_vocabulary.json")

guard let fixtureData = try? Data(contentsOf: fixtureUrl),
      let fixture = try? JSONSerialization.jsonObject(with: fixtureData) as? [String: Any]
else {
    FileHandle.standardError.write(Data("cannot read \(fixtureUrl.path)\n".utf8))
    exit(2)
}

let constants = fixture["constants"] as? [String: Any] ?? [:]
let vocabulary = fixture["vocabulary"] as? [[String: Any]] ?? []
let cases = fixture["cases"] as? [[String: Any]] ?? []

func caseNamed(_ name: String) -> [String: Any]? {
    cases.first { ($0["name"] as? String) == name }
}

// MARK: - 1. Every code, in the owner's worst-first order

if let expectedCodes = caseNamed("every_code_is_unique_and_stable")?["expected_codes"] as? [String] {
    expectEqual(SwiftWatchVocabulary.all.map(\.code), expectedCodes,
                "SwiftWatchVocabulary order")
    expectEqual(WatchHapticVocabulary.codeOrder, expectedCodes,
                "WatchHapticVocabulary.codeOrder")
    // Every code must have a watchOS haptic row. A code with no row would fall
    // through to silence with nobody noticing — the exact failure the spec's
    // "must say so on screen" ruling is trying to prevent.
    for code in expectedCodes {
        expect(WatchHapticVocabulary.table[code] != nil,
               "WatchHapticVocabulary.table is missing code \(code)")
    }
} else {
    failures.append("fixture case every_code_is_unique_and_stable is missing")
}

// MARK: - 2. Polarity and levels

if let alertCase = caseNamed("alert_codes_have_three_levels_positives_have_one") {
    let alerts = Set(alertCase["expected_alert_codes"] as? [String] ?? [])
    let positives = Set(alertCase["expected_positive_codes"] as? [String] ?? [])
    for entry in SwiftWatchVocabulary.all {
        if alerts.contains(entry.code) {
            expect(!entry.isPositive, "\(entry.code) should be an ALERT")
            expectEqual(entry.levels, [1, 2, 3], "\(entry.code).levels")
        } else if positives.contains(entry.code) {
            expect(entry.isPositive, "\(entry.code) should be a POSITIVE")
            expectEqual(entry.levels, [1], "\(entry.code).levels")
        } else {
            failures.append("\(entry.code) appears in neither alert nor positive list")
        }
    }
}

// MARK: - 3. Field-by-field against the vocabulary table

let lookup = SwiftWatchVocabulary()
for row in vocabulary {
    guard let code = row["code"] as? String else { continue }
    guard let entry = lookup.entry(forCode: code) else {
        failures.append("Swift vocabulary has no entry for \(code)")
        continue
    }
    expectEqual(entry.vector, row["vector"] as? String ?? "", "\(code).vector")
    expectEqual(entry.icon, row["icon"] as? String ?? "", "\(code).icon")
    expectEqual(entry.name, row["name"] as? String ?? "", "\(code).name")
    // flash_text is `null` for E, R, K, P and must stay nil — never an invented
    // line, which is why the screen renders nothing for those.
    expectEqual(entry.flashText, row["flash_text"] as? String, "\(code).flashText")
    expectEqual(entry.levels, row["levels"] as? [Int] ?? [], "\(code).levels")
    expectEqual(entry.sources, row["sources"] as? [String] ?? [], "\(code).sources")
    expectEqual(entry.isPositive, (row["polarity"] as? String) == "positive", "\(code).isPositive")

    // Every SOURCE vector must resolve back to its code. A nudge frame carries
    // source names ("yelling", "interrupting", "airtime", "hr_spike"), not the
    // vocabulary id, so this is the lookup the wrist actually performs.
    for source in entry.sources {
        expectEqual(lookup.code(forVectors: [source]), code, "code(forVectors: [\(source)])")
    }
    expectEqual(lookup.code(forVectors: [entry.vector]), code, "code(forVectors: [\(entry.vector)])")
}

// MARK: - 4. Worst-first tie-breaking

// "A" (hogging) and "H" (heated) firing together must resolve to H: the owner's
// order is worst-first and ties break on it.
expectEqual(lookup.code(forVectors: ["airtime", "yelling"]), "H",
            "code(forVectors:) ties break worst-first")
expectEqual(lookup.code(forVectors: ["not_a_vector"]), nil,
            "an unmapped vector contributes nothing rather than a nameless buzz")

// MARK: - 5. The positive cap, replayed

if let capCase = caseNamed("positive_cap_is_one_per_two_minutes"),
   let offers = capCase["offers"] as? [[String: Any]],
   let expectedAdmitted = capCase["expected_admitted"] as? [[String: Any]] {
    let capS = (constants["positive_cap_s"] as? NSNumber)?.doubleValue ?? 120.0
    expectEqual(PositiveNudgeGate.positiveCapS, capS, "POSITIVE_CAP_S")

    let gate = PositiveNudgeGate(capS: capS)
    var admitted: [String] = []
    for offer in offers {
        let t = (offer["t"] as? NSNumber)?.doubleValue ?? 0
        let code = offer["code"] as? String ?? "?"
        if gate.admit(t: t) { admitted.append("\(code)@\(t)") }
    }
    let expected = expectedAdmitted.map { offer in
        "\(offer["code"] as? String ?? "?")@\((offer["t"] as? NSNumber)?.doubleValue ?? 0)"
    }
    expectEqual(admitted, expected, "PositiveNudgeGate admissions")
}

// MARK: - 6. The watchOS column's own claims

// K is the one cue that survives intact, and it survives by being silence.
expect(WatchHapticVocabulary.isSilentByContract(code: "K"),
       "K must be silent by contract")
expect(WatchHapticVocabulary.plan(forCode: "K", level: 1) == nil,
       "K must never produce a playable plan")
expect(!WatchHapticVocabulary.isSilentByContract(code: "ZZ"),
       "an unknown code is NOT silent-by-contract — that distinction is the point")
expectEqual(WatchHapticVocabulary.table["K"]?.fidelity, .full, "K fidelity")

// Level is carried by tap COUNT — the one property of the design that survives.
expectEqual(WatchHapticVocabulary.plan(forCode: "H", level: 1)?.taps.count, 1, "H L1 tap count")
expectEqual(WatchHapticVocabulary.plan(forCode: "H", level: 2)?.taps.count, 2, "H L2 tap count")
expectEqual(WatchHapticVocabulary.plan(forCode: "H", level: 3)?.taps.count, 3, "H L3 tap count")
expect(WatchHapticVocabulary.plan(forCode: "H", level: 0) == nil,
       "level 0 is silent and clears the reminder")

// Positives are unleveled: a "level 2 well done" is not a thing.
expectEqual(WatchHapticVocabulary.plan(forCode: "E", level: 3)?.level, 1, "E is unleveled")
expectEqual(WatchHapticVocabulary.plan(forCode: "E", level: 1)?.taps, [.success], "E cue")
expectEqual(WatchHapticVocabulary.plan(forCode: "R", level: 1)?.taps, [.success], "R cue")

// The degradation must stay ADMITTED, not quietly papered over. Exactly one
// code may claim full fidelity, and the collapsed ones must each say what they
// lost — if someone "fixes" this by reusing one buzz everywhere and marking it
// fine, this is what fails.
let fullCount = WatchHapticVocabulary.table.values.filter { $0.fidelity == .full }.count
expectEqual(fullCount, 1, "exactly one cue (K) may claim full fidelity on watchOS")
for (code, row) in WatchHapticVocabulary.table where row.fidelity != .full {
    expect(!row.lost.isEmpty, "\(code) is degraded but names nothing it lost")
}

// The never-merge floor is carried over as the inter-tap target.
expectEqual(Int(WatchHapticVocabulary.interTapMs),
            (constants["min_gap_ms"] as? NSNumber)?.intValue ?? 170,
            "inter-tap gap == MIN_GAP_MS")

// MARK: - Report

if failures.isEmpty {
    print("watchOS policy vectors: \(checks) checks passed against \(fixtureUrl.lastPathComponent)")
    exit(0)
} else {
    print("watchOS policy vectors: \(failures.count) of \(checks) checks FAILED")
    for failure in failures { print("  - \(failure)") }
    exit(1)
}
