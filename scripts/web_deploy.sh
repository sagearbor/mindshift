#!/usr/bin/env bash
# Build the Expo web app with the PRODUCTION env baked in and deploy it to
# Firebase Hosting (https://arborfam-hub.web.app, config: firebase.json ->
# apps/mobile/dist).
#
# Why a script (like ota_publish.sh): `expo export` bakes EXPO_PUBLIC_* into
# the bundle at build time. A bare export without them ships a client that
# talks to http://localhost:8000 — exactly the bug that once hit an OTA. The
# env values are read from apps/mobile/eas.json (build.production.env) so
# the web build and the store builds can never disagree.
#
# Usage:
#   scripts/web_deploy.sh              # build, verify, deploy
#   scripts/web_deploy.sh --dry-run    # build + verify only (no firebase)
#   WEB_DRY_RUN=1 scripts/web_deploy.sh
#
# Needs: node (>=20), the workspace installed (`npm ci` at the repo root),
# and for a real deploy `firebase` on PATH and logged in (`firebase login`).
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"

DRY_RUN="${WEB_DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE="$ROOT/apps/mobile"
DIST="$MOBILE/dist"

# Production env from eas.json — the single source of truth.
read_env() {
  node -e '
    const eas = require(process.argv[1]);
    const env = (eas.build && eas.build.production && eas.build.production.env) || {};
    const v = env[process.argv[2]];
    if (!v) { console.error("eas.json build.production.env." + process.argv[2] + " is missing"); process.exit(1); }
    process.stdout.write(v);
  ' "$MOBILE/eas.json" "$1"
}
export EXPO_PUBLIC_API_URL="${EXPO_PUBLIC_API_URL:-$(read_env EXPO_PUBLIC_API_URL)}"
export EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID="${EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID:-$(read_env EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID)}"

echo "web_deploy: API      = $EXPO_PUBLIC_API_URL"
echo "web_deploy: client   = ${EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID:0:12}…"
echo "web_deploy: dry-run  = $DRY_RUN"

# 1. Build (copies the onnxruntime-web runtime into public/ort first).
( cd "$MOBILE" && npm run build:web )

# 2. Verify the export before anything ships.
if [ ! -f "$DIST/index.html" ]; then
  echo "web_deploy: no $DIST/index.html — export failed" >&2; exit 1
fi
BUNDLES=$(find "$DIST/_expo/static/js/web" -name '*.js' 2>/dev/null || true)
if [ -z "$BUNDLES" ]; then
  echo "web_deploy: no JS bundle under $DIST/_expo/static/js/web" >&2; exit 1
fi
if ! grep -q -- "$EXPO_PUBLIC_API_URL" $BUNDLES; then
  echo "web_deploy: the bundle does not contain $EXPO_PUBLIC_API_URL — env not baked in" >&2; exit 1
fi
if grep -q "localhost:8000" $BUNDLES; then
  # The fallback literal is allowed only as the `||` default; a bundle that
  # ALSO carries the production URL is fine. Flag it loudly anyway.
  echo "web_deploy: note — bundle still carries the localhost:8000 fallback literal (expected: it is the || default)"
fi
for f in ort/ort.wasm.min.js ort/ort-wasm-simd-threaded.wasm ort/ort-wasm-simd-threaded.mjs; do
  if [ ! -f "$DIST/$f" ]; then
    echo "web_deploy: $DIST/$f missing — public/ort was not copied" >&2; exit 1
  fi
done
if ! ls "$DIST"/assets/*silero* "$DIST"/assets/**/*silero* >/dev/null 2>&1 && ! find "$DIST/assets" -name '*.onnx' | grep -q .; then
  echo "web_deploy: the Silero VAD asset (.onnx) is not in the export" >&2; exit 1
fi
echo "web_deploy: export verified ($(du -sh "$DIST" | cut -f1))"

# 2b. Headless-Chrome smoke (mount + self-hosted ONNX Runtime) when Chrome exists.
if [ -n "${CHROME:-}" ] || [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] || command -v google-chrome >/dev/null 2>&1; then
  if [ "${WEB_SKIP_SMOKE:-0}" = "1" ]; then
    echo "web_deploy: WEB_SKIP_SMOKE=1 — skipping the headless Chrome smoke"
  else
    node "$ROOT/scripts/web_smoke.mjs"
  fi
else
  echo "web_deploy: no Chrome found — skipping the headless smoke (set CHROME=/path/to/chrome to run it)"
fi

# 3. Deploy.
if [ "$DRY_RUN" = "1" ]; then
  echo "web_deploy: dry run — skipping 'firebase deploy'. To ship:"
  echo "  (cd $ROOT && firebase deploy --only hosting)"
  exit 0
fi
command -v firebase >/dev/null 2>&1 || { echo "web_deploy: firebase CLI not on PATH (npm i -g firebase-tools)" >&2; exit 1; }
( cd "$ROOT" && firebase deploy --only hosting )
echo "web_deploy: live at https://arborfam-hub.web.app"
