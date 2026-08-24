#!/usr/bin/env bash
# Publish an OTA (EAS Update) with the SAME EXPO_PUBLIC_* env the production
# build bakes in. `eas update` bundles JS on the machine running it — without
# these vars the client's API URL falls back to http://localhost:8000, which
# shipped a broken bundle to production on 2026-08-14 (found on-device:
# "Failed to connect to localhost/127.0.0.1:8000"). Never run `eas update`
# directly; use this.
#
# USAGE: ./scripts/ota_publish.sh "update message"
#
#   OTA_CHANNEL=preview ./scripts/ota_publish.sh "msg"
#       Publish to another channel (default: production). `preview` is the
#       channel the internal-distribution APK / ad-hoc builds from
#       `eas build --profile preview` listen on.
#   OTA_DRY_RUN=1 ./scripts/ota_publish.sh "msg"
#       Prove the plumbing without publishing: checks `eas whoami`, the
#       project id, the channel, then exports the bundle locally with the
#       baked env and greps the compiled JS for the production API URL (the
#       exact failure mode above). Nothing is uploaded.
#
# Runtime isolation: app.json's runtimeVersion policy is `appVersion`, so an
# update is only ever delivered to builds whose `expo.version` matches the
# checkout you publish from. Bump `expo.version` whenever a native module is
# added/removed (done for 1.17.0 on 2026-08-24) — otherwise an OTA could land
# JS that requires native code the installed binary lacks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$SCRIPT_DIR/../apps/mobile"
MSG="${1:?usage: ota_publish.sh \"update message\"}"
CHANNEL="${OTA_CHANNEL:-production}"

# Source of truth: eas.json's production profile env — read, not duplicated.
eval "$(node -e '
  const env = require(process.argv[1] + "/eas.json").build.production.env;
  for (const [k, v] of Object.entries(env)) {
    console.log(`export ${k}=${JSON.stringify(v)}`);
  }
' "$MOBILE_DIR")"

echo "→ OTA publish to channel '$CHANNEL' with baked env:"
env | grep "^EXPO_PUBLIC_" | sed "s/^/   /"
cd "$MOBILE_DIR"

# Prefer the globally installed CLI (already logged in); fall back to npx.
if command -v eas >/dev/null 2>&1; then EAS=(eas); else EAS=(npx eas-cli); fi

if [[ "${OTA_DRY_RUN:-0}" == "1" ]]; then
  echo "→ DRY RUN: auth / project / runtime check"
  echo "   eas user:        $("${EAS[@]}" whoami 2>/dev/null | grep -v '^★\|^To upgrade\|^npm install\|^Proceeding\|^$' | head -1)"
  echo "   project id:      $(node -e 'console.log(require("./app.json").expo.extra.eas.projectId)')"
  echo "   runtime version: $(node -e 'const a=require("./app.json").expo; console.log(a.runtimeVersion.policy==="appVersion" ? a.version : JSON.stringify(a.runtimeVersion))')"
  echo "   channel:         $CHANNEL"
  if "${EAS[@]}" channel:view "$CHANNEL" --non-interactive >/dev/null 2>&1; then
    echo "   channel exists:  yes"
  else
    echo "   channel exists:  NO (eas update would create it)"
  fi
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/ota-dry-run.XXXXXX")"
  echo "→ DRY RUN: exporting bundle to $OUT (not uploaded)"
  if ! npx expo export --platform android --platform ios --output-dir "$OUT" >"$OUT/export.log" 2>&1; then
    tail -40 "$OUT/export.log"
    echo "✗ expo export failed"
    exit 1
  fi
  if grep -rqF "$EXPO_PUBLIC_API_URL" "$OUT/_expo/static/js"; then
    echo "✓ baked EXPO_PUBLIC_API_URL found in the exported JS bundle(s)"
  else
    echo "✗ EXPO_PUBLIC_API_URL NOT found in exported bundle — the env did not bake in"
    exit 1
  fi
  echo "✓ dry run OK — nothing published"
  exit 0
fi

"${EAS[@]}" update --channel "$CHANNEL" --environment production \
  --message "$MSG" --non-interactive
