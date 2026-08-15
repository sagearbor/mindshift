#!/usr/bin/env bash
# Generate TypeScript types for the mobile app from the unified backend's
# OpenAPI schema (server/main.py — both the pre-existing MindShift domain
# and the ported Gauge/watch domain live on the same FastAPI app as of
# B12). Re-runnable and safe to call from any cwd; wraps `npm run gen:api`.
#
# SCOPE NOTE — Kotlin generation is DELIBERATELY DEFERRED. The watch app
# keeps its hand-written `WireModels.kt` as the source of the mirror
# contract until the Phase 2/3 watch release cycle; this script emits
# TypeScript only. Do not add a Kotlin codegen step here without first
# updating docs/plans/2026-08-15-phase1-one-repo-one-engine.md.
#
# ADOPTION NOTE — the mobile app does not switch over to these generated
# types in Phase 1. Adoption is incremental starting Phase 3, new/ported
# endpoints first; existing hand-rolled request/response types in
# apps/mobile/src/api/client.ts are untouched by this task.
#
# DEVDEP LOCATION — `openapi-typescript` is installed as a devDependency
# of the REPO-ROOT package.json (`npm install --save-dev
# openapi-typescript` run from repo root), not apps/mobile/package.json.
# Rationale: this script's own working directory for the `npx` step is
# the repo root (see below), npm workspaces hoist devDependencies to the
# root node_modules/.bin by default, and the tool serves a repo-wide
# codegen concern (spec comes from server/, output lands under
# apps/mobile/) rather than being a mobile app runtime/build dependency.
#
# USAGE: ./scripts/gen_api_types.sh   (or: npm run gen:api)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_DIR="$REPO_ROOT/server"
OUT_FILE="$REPO_ROOT/apps/mobile/src/api/generated/openapi.d.ts"

SPEC_FILE="$(mktemp -t mindshift-openapi.XXXXXX.json)"
trap 'rm -f "$SPEC_FILE"' EXIT

echo "→ Emitting OpenAPI spec from server/main.py ..."
(
  cd "$SERVER_DIR"
  python3 -c "import json, main; print(json.dumps(main.app.openapi()))"
) > "$SPEC_FILE"

echo "→ Generating TypeScript types → ${OUT_FILE#"$REPO_ROOT"/} ..."
mkdir -p "$(dirname "$OUT_FILE")"
(
  cd "$REPO_ROOT"
  npx --no-install openapi-typescript "$SPEC_FILE" -o "$OUT_FILE"
)

echo "→ Done."
