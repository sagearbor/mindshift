#!/usr/bin/env bash
# Publish an OTA (EAS Update) with the SAME EXPO_PUBLIC_* env the production
# build bakes in. `eas update` bundles JS on the machine running it — without
# these vars the client's API URL falls back to http://localhost:8000, which
# shipped a broken bundle to production on 2026-08-14 (found on-device:
# "Failed to connect to localhost/127.0.0.1:8000"). Never run `eas update`
# directly; use this.
#
# USAGE: ./scripts/ota_publish.sh "update message"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$SCRIPT_DIR/../apps/mobile"
MSG="${1:?usage: ota_publish.sh \"update message\"}"

# Source of truth: eas.json's production profile env — read, not duplicated.
eval "$(node -e '
  const env = require(process.argv[1] + "/eas.json").build.production.env;
  for (const [k, v] of Object.entries(env)) {
    console.log(`export ${k}=${JSON.stringify(v)}`);
  }
' "$MOBILE_DIR")"

echo "→ OTA publish with baked env:"
env | grep "^EXPO_PUBLIC_" | sed "s/^/   /"
cd "$MOBILE_DIR"
npx eas-cli update --channel production --environment production \
  --message "$MSG" --non-interactive
