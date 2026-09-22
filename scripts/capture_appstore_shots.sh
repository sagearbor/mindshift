#!/bin/bash
# Capture App Store screenshots of the REAL app on an iOS Simulator.
#
# Produces 1320x2868 PNGs (iPhone 6.9"), which is what Apple currently requires
# and what the API models as the APP_IPHONE_67 screenshotDisplayType. Feed them
# to scripts/asc_upload_screenshots.py, which asks the API for that enum rather
# than trusting this comment to stay current.
#
# WHY idb AND NOT A GUI CLICKER
#   simctl can install, launch, screenshot and open URLs, but it cannot touch
#   the screen. Everything that drives the screen through the window server --
#   AppleScript, cliclick -- needs macOS Automation/Accessibility permission,
#   which is a GUI grant that cannot be given on a headless or remote run; both
#   fail with an AppleEvent timeout or "Accessibility privileges not enabled".
#   idb talks to CoreSimulator directly and needs neither.
#
# SETUP (once)
#   brew trust facebook/fb && brew install idb-companion
#   python3 -m venv tmp/idbvenv && tmp/idbvenv/bin/pip install fb-idb
#
# BUILD (the app itself)
#   eas build --platform ios --profile ios-simulator
#   ...then download the .tar.gz artifact and untar it next to this script's
#   APP_PATH. A local `npx expo run:ios` works too.
#
#   NOTE: as of 2026-09 the ios-simulator profile fails while the watchOS
#   companion target (apps/mobile/targets/watch) is in the build, and the
#   phone screenshots do not need it. Temporarily dropping
#   "./plugins/withWatchTargetVersion.js" and "@bacons/apple-targets" from
#   app.json's plugins builds a phone-only app with identical phone UI.
#
# COORDINATES are in POINTS (440x956 on this device), not pixels. The captured
# PNG is 3x that. They are tied to the current layout; if a tap lands wrong,
# screenshot first and re-measure rather than nudging blindly.
set -euo pipefail

DEVICE="${DEVICE:-}"
BUNDLE_ID="${BUNDLE_ID:-com.sagearbor.mindshift.app}"
APP_PATH="${APP_PATH:-tmp/simbuild/MindShift.app}"
OUT_DIR="${OUT_DIR:-tmp/appstore-shots}"
IDB="${IDB:-tmp/idbvenv/bin/idb}"
GRPC_PORT="${GRPC_PORT:-10882}"

export PATH="/opt/homebrew/bin:$PATH"

if [ -z "$DEVICE" ]; then
  # The largest available iPhone is the one that matches the required 6.9"
  # size; a smaller device produces a PNG App Store Connect will reject.
  DEVICE=$(xcrun simctl list devices available | awk '/iPhone 17 Pro Max/ {print $NF; exit}' | tr -d '()')
fi
[ -n "$DEVICE" ] || { echo "no iPhone 17 Pro Max simulator found" >&2; exit 1; }
echo "device: $DEVICE"

mkdir -p "$OUT_DIR"

xcrun simctl boot "$DEVICE" 2>/dev/null || true
xcrun simctl install "$DEVICE" "$APP_PATH"

# The marketing-standard status bar. This overrides only the status bar, not
# anything the app draws.
xcrun simctl status_bar "$DEVICE" override \
  --time "9:41" --dataNetwork wifi --wifiMode active --wifiBars 3 \
  --cellularMode active --cellularBars 4 --batteryState charged --batteryLevel 100

# Live Coach asks for the microphone; granting it up front keeps the system
# permission alert out of the screenshots.
xcrun simctl privacy "$DEVICE" grant microphone "$BUNDLE_ID" || true

# The companion is a gRPC server; it has to outlive the taps below.
idb_companion --udid "$DEVICE" --grpc-port "$GRPC_PORT" --log-level info \
  > tmp/logs/idb-companion.log 2>&1 &
COMPANION_PID=$!
trap 'kill $COMPANION_PID 2>/dev/null || true' EXIT
sleep 5
"$IDB" connect localhost "$GRPC_PORT"

tap()  { "$IDB" ui tap --udid "$DEVICE" "$1" "$2"; sleep "${3:-4}"; }
shot() { xcrun simctl io "$DEVICE" screenshot "$OUT_DIR/$1"; }

xcrun simctl terminate "$DEVICE" "$BUNDLE_ID" 2>/dev/null || true
xcrun simctl launch "$DEVICE" "$BUNDLE_ID"
sleep 9

# 1. Sign-in, showing guest mode alongside the real sign-in options. This is
#    the launch screen with no stored session, so it needs no interaction.
shot "04-sign-in-guest.png"

# Guest mode is the only path that needs no credentials. It creates a real
# Firebase anonymous session, which then persists across relaunches.
tap 220 720 8   # Continue as guest
tap 396 100 5   # Skip the onboarding carousel

tap 78  895 5 ; shot "01-live-coach.png"           # Live Coach tab
tap 220 895 5 ; shot "02-analyze-conversation.png" # Analyze a Conversation tab
tap 40  143 4 ; shot "03-everything.png"           # hamburger -> full catalog

echo
echo "wrote to $OUT_DIR:"
ls -1 "$OUT_DIR"/0*.png
echo
echo "NOT capturable on the simulator: the in-session Live Coach screen."
echo "Tapping Start Listening aborts the process --"
echo "  AVAudioIONodeImpl: associating with audio session (0x0), error -10879"
echo "  coreaudio: Initialize: RPC timeout. Apparently deadlocked. Aborting now."
echo "The simulator has no real audio input to bind AVAudioEngine to. That"
echo "screen needs a device, so do not substitute another screen for it."
