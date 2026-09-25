#!/usr/bin/env bash
# Build the Android AND iOS production binaries from the same commit, so the
# two stores can never drift. The only sanctioned way to start a store build.
#
# Why both, always: the JS is one bundle shared over the air; the binaries
# are the only thing that can differ, and they differ exactly when a native
# input changes (a native module, a config plugin, an app.json permission, the
# watch target). apps/mobile/__tests__/nativeParity.test.ts fails when those
# inputs change without an expo.version bump; this script refuses to build
# until that test passes, then builds both platforms in one EAS run.
#
# Usage:
#   scripts/release_both.sh                 # check, then build both (no-wait)
#   scripts/release_both.sh --dry-run       # check only
#   scripts/release_both.sh --profile preview
#
# After the builds finish: `eas submit` per platform as the playbooks say
# (~/.claude/playbooks/), then scripts/ota_publish.sh for JS-only follow-ups.
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE="$ROOT/apps/mobile"
PROFILE="production"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run|-n) DRY=1 ;;
    --profile) PROFILE="$2"; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done

cd "$MOBILE"
echo "release_both: native parity check"
node scripts/nativeFingerprint.js
node_modules/.bin/jest --ci --silent __tests__/nativeParity.test.ts >/dev/null
echo "release_both: parity OK"

if [ -n "$(git -C "$ROOT" status --porcelain -- apps/mobile/app.json apps/mobile/eas.json apps/mobile/plugins apps/mobile/targets)" ]; then
  echo "release_both: uncommitted native inputs — commit first so both builds come from one SHA" >&2
  exit 1
fi

VERSION=$(node -p "require('./app.json').expo.version")
echo "release_both: expo.version $VERSION @ $(git -C "$ROOT" rev-parse --short HEAD), profile $PROFILE, platforms android+ios"
if [ "$DRY" = 1 ]; then echo "release_both: dry run — not building"; exit 0; fi

eas build --platform all --profile "$PROFILE" --non-interactive --no-wait
echo "release_both: both builds queued. Watch with: eas build:list --limit 2"
