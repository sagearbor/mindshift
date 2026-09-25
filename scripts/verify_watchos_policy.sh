#!/usr/bin/env bash
# Proves the watchOS app's nudge policy is not a fourth, drifting hand-copy.
#
# Two lanes, because no single artifact covers both halves of the contract:
#
#   1. VECTORS  — replays server/tests/fixtures/policy_vectors/nudge_vocabulary.json
#                 against the app's own Swift sources, on macOS, with no
#                 simulator and no Xcode project. This is the lane the parity
#                 spec calls non-negotiable (§18) and the lane that keeps
#                 working if the KMP framework is never wired into the cloud
#                 build. Fast: ~2 s.
#
#   2. PARITY   — builds the real watch target against MindShiftShared.xcframework,
#                 runs it on a watchOS simulator, and reads back the one startup
#                 line the app logs: which policy implementation is live, and how
#                 many places the Swift disagrees with the Kotlin. This is the
#                 lane that catches a drift the fixtures do not describe — the
#                 PRD §6 reminder ladder in particular, which no fixture pins.
#                 Slower (~2 min) and skipped automatically when the framework
#                 has not been built.
#
# Usage:
#   ./scripts/verify_watchos_policy.sh            # both lanes (parity skipped if no framework)
#   ./scripts/verify_watchos_policy.sh --vectors  # lane 1 only
#
# Build the framework for lane 2 with:
#   cd apps/watch && JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
#     ./gradlew :shared:assembleMindShiftSharedXCFramework

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WATCH_SRC="apps/mobile/targets/watch"
FRAMEWORK="apps/watch/shared/build/XCFrameworks/release/MindShiftShared.xcframework/watchos-arm64-simulator"
SIM_NAME="${MINDSHIFT_WATCH_SIM:-Apple Watch Series 11 (46mm)}"
BUNDLE_ID="com.sagearbor.mindshift.app.watchkitapp"

# ---------------------------------------------------------------- lane 1
echo "== lane 1: JSON vectors vs the Swift policy =="
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Only the platform-free files (CompanionSocket.swift is Foundation-only — its
# `ServerFrame` decoder is what section 7 of the vectors exercises).
# SharedPolicySeam's `#if canImport(MindShiftShared)`
# is false here, so this compiles and exercises the SWIFT FALLBACK — which is
# exactly the implementation that ships when the framework is not linked, and
# therefore exactly the one the fixtures need to pin.
# Copied to main.swift because Swift only allows top-level statements in a file
# with that name once more than one file is being compiled.
cp scripts/watchos_policy_vectors.swift "$WORK/main.swift"
swiftc -O \
  "$WATCH_SRC/Policy/SharedPolicySeam.swift" \
  "$WATCH_SRC/Policy/WatchHapticVocabulary.swift" \
  "$WATCH_SRC/Policy/PurePorts.swift" \
  "$WATCH_SRC/IO/WatchApi.swift" \
  "$WATCH_SRC/IO/CompanionSocket.swift" \
  "$WORK/main.swift" \
  -o "$WORK/vectors" 2>&1 | grep -v 'warning: .*never used' || true

if [ ! -x "$WORK/vectors" ]; then
  echo "FAIL: the Swift policy sources did not compile for the host"
  exit 1
fi
"$WORK/vectors"

if [ "${1:-}" = "--vectors" ]; then
  exit 0
fi

# ---------------------------------------------------------------- lane 2
echo
echo "== lane 2: Kotlin vs Swift, on a watchOS simulator =="

if [ ! -d "$FRAMEWORK" ]; then
  echo "SKIP: $FRAMEWORK not built."
  echo "      This lane needs Gradle. See the header for the command."
  exit 0
fi

if ! xcrun simctl list devices available | grep -q "$SIM_NAME"; then
  echo "SKIP: no '$SIM_NAME' simulator available."
  exit 0
fi

DD="$(mktemp -d)"
trap 'rm -rf "$WORK" "$DD"' EXIT

# `-target`, not `-scheme`: the scheme pulls in the React Native app, whose
# CocoaPods sandbox check fails on any machine that has not run `pod install`.
# The watch target itself has no pod dependencies.
xcodebuild -project apps/mobile/ios/MindShift.xcodeproj \
  -scheme MindShiftWatch -configuration Debug -sdk watchsimulator \
  -destination "platform=watchOS Simulator,name=$SIM_NAME" \
  -derivedDataPath "$DD" CODE_SIGNING_ALLOWED=NO \
  FRAMEWORK_SEARCH_PATHS="$ROOT/$FRAMEWORK" \
  OTHER_LDFLAGS="-framework MindShiftShared" \
  build > "$DD/build.log" 2>&1 || true

APP="$DD/Build/Products/Debug-watchsimulator/MindShiftWatch.app"
if [ ! -d "$APP" ]; then
  echo "FAIL: MindShiftWatch.app was not produced. Swift errors:"
  grep -E '\.swift:[0-9]+:[0-9]+: error' "$DD/build.log" | head -20
  exit 1
fi
if grep -qE '\.swift:[0-9]+:[0-9]+: error' "$DD/build.log"; then
  echo "FAIL: Swift errors while linking the KMP framework:"
  grep -E '\.swift:[0-9]+:[0-9]+: error' "$DD/build.log" | head -20
  exit 1
fi

xcrun simctl boot "$SIM_NAME" >/dev/null 2>&1 || true
xcrun simctl bootstatus "$SIM_NAME" -b >/dev/null 2>&1 || true
xcrun simctl terminate "$SIM_NAME" "$BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl install "$SIM_NAME" "$APP"
xcrun simctl launch "$SIM_NAME" "$BUNDLE_ID" -uiPreview idle >/dev/null
sleep 5

LINE="$(xcrun simctl spawn "$SIM_NAME" log show --last 2m \
  --predicate 'eventMessage CONTAINS "MINDSHIFT_WATCH_POLICY"' --style compact 2>/dev/null \
  | grep 'MINDSHIFT_WATCH_POLICY backing=' | tail -1 || true)"
# `backing=` rather than the bare prefix: `log show`'s own invocation echoes the
# predicate string back into the output, so a bare grep matches the logger.

if [ -z "$LINE" ]; then
  echo "FAIL: the app did not log MINDSHIFT_WATCH_POLICY."
  exit 1
fi
echo "$LINE"

case "$LINE" in
  *"backing=KMP MindShiftShared"*) ;;
  *) echo "FAIL: the KMP framework linked but the Swift fallback is still live."; exit 1 ;;
esac
case "$LINE" in
  *"parity_failures=0"*) echo "PASS: Kotlin and Swift agree." ;;
  *) echo "FAIL: the Swift policy has drifted from the Kotlin."; exit 1 ;;
esac
