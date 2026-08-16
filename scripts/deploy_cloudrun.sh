#!/usr/bin/env bash
# Deploy the MindShift FastAPI backend (live WebSocket audio pipeline) to
# Google Cloud Run from source. Cloud Run is used because it supports
# long-lived WebSocket connections (unlike most static hosts) and can build
# the repo's Dockerfile with Cloud Build — no local Docker required.
#
# WHAT THIS DOES
#   1. Reads ANTHROPIC_API_KEY + DEEPGRAM_API_KEY from the repo-root .env at
#      runtime and passes them to Cloud Run as service env vars. Secrets are
#      never hardcoded here or committed.
#   2. Enables the Run + Cloud Build APIs (idempotent).
#   3. Deploys `--source .` so Cloud Build builds the Dockerfile.
#   4. Prints the public HTTPS URL and the wss:// WebSocket URL to paste into
#      the app's EXPO_PUBLIC_API_URL.
#
# USAGE
#   ./scripts/deploy_cloudrun.sh <PROJECT_ID> [REGION] [SERVICE]
#   # or set env: GCP_PROJECT, GCP_REGION, GCP_SERVICE
#
#   ./scripts/deploy_cloudrun.sh my-gcp-project
#   ./scripts/deploy_cloudrun.sh my-gcp-project us-central1 mindshift-api
#
# PREREQS (see docs/DEPLOY.md): gcloud installed + `gcloud auth login`, a GCP
# project with BILLING ENABLED, and a filled repo-root .env. This script does
# not create credentials or a project.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (this script lives in <repo>/scripts/).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Config (positional args win over env vars, which win over defaults).
# ---------------------------------------------------------------------------
PROJECT="${1:-${GCP_PROJECT:-}}"
REGION="${2:-${GCP_REGION:-us-central1}}"
SERVICE="${3:-${GCP_SERVICE:-mindshift-api}}"

if [[ -z "$PROJECT" ]]; then
  echo "ERROR: no GCP project. Pass it as the first arg or set GCP_PROJECT." >&2
  echo "  usage: ./scripts/deploy_cloudrun.sh <PROJECT_ID> [REGION] [SERVICE]" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found. Install it: brew install --cask google-cloud-sdk" >&2
  echo "  then: gcloud auth login && gcloud config set project $PROJECT" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Load the two required secrets from repo-root .env (KEY=VALUE lines).
# We parse only the keys we need so an arbitrary .env can't inject env into
# this shell. Real environment variables take precedence over the .env file.
# ---------------------------------------------------------------------------
ENV_FILE="$REPO_ROOT/.env"

read_env() {
  # read_env KEY -> echoes the value from $ENV_FILE, or empty. Strips optional
  # surrounding quotes and inline `export `; ignores comments/blank lines.
  local key="$1"
  [[ -f "$ENV_FILE" ]] || return 0
  # Last matching assignment wins.
  local line
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ENV_FILE" | tail -n 1 || true)"
  [[ -n "$line" ]] || return 0
  local val="${line#*=}"
  # Trim surrounding whitespace and matching quotes.
  val="${val#"${val%%[![:space:]]*}"}"
  val="${val%"${val##*[![:space:]]}"}"
  if [[ "$val" == \"*\" ]]; then val="${val%\"}"; val="${val#\"}"; fi
  if [[ "$val" == \'*\' ]]; then val="${val%\'}"; val="${val#\'}"; fi
  printf '%s' "$val"
}

# Prefer an already-exported env var; otherwise read from .env.
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$(read_env ANTHROPIC_API_KEY)}"
DEEPGRAM_API_KEY="${DEEPGRAM_API_KEY:-$(read_env DEEPGRAM_API_KEY)}"
missing=()
[[ -n "$ANTHROPIC_API_KEY" ]] || missing+=("ANTHROPIC_API_KEY")
[[ -n "$DEEPGRAM_API_KEY" ]] || missing+=("DEEPGRAM_API_KEY")
if (( ${#missing[@]} > 0 )); then
  echo "ERROR: missing required key(s): ${missing[*]}" >&2
  echo "  Set them in $ENV_FILE (copy env.example -> .env and fill them in)," >&2
  echo "  or export them in this shell before running." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build the --set-env-vars list. Cloud Run wants a single comma-delimited
# string; use ^@^ as the delimiter so values may safely contain commas.
# ---------------------------------------------------------------------------
ENV_VARS="ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}@DEEPGRAM_API_KEY=${DEEPGRAM_API_KEY}"
# --set-env-vars REPLACES the service's entire env set on every deploy, so any
# var not listed here is silently wiped. That bit us once: the WS browser
# origin allowlist was added via `gcloud run services update` and the next
# deploy erased it, re-breaking the web/iPhone mic. Default it here (it is
# public config, not a secret) so the web app's origin always survives; a
# MINDSHIFT_ALLOWED_ORIGINS in .env / the environment still overrides.
MINDSHIFT_ALLOWED_ORIGINS="${MINDSHIFT_ALLOWED_ORIGINS:-$(read_env MINDSHIFT_ALLOWED_ORIGINS)}"
MINDSHIFT_ALLOWED_ORIGINS="${MINDSHIFT_ALLOWED_ORIGINS:-https://arborfam-hub.web.app}"
# Same defaulting for the recordings bucket: --set-env-vars REPLACES the whole
# env set each deploy, so without a default here a redeploy would silently WIPE
# the bucket and disable recording storage. Default it (public config, not a
# secret); a MINDSHIFT_RECORDINGS_BUCKET in .env / the environment overrides.
MINDSHIFT_RECORDINGS_BUCKET="${MINDSHIFT_RECORDINGS_BUCKET:-$(read_env MINDSHIFT_RECORDINGS_BUCKET)}"
MINDSHIFT_RECORDINGS_BUCKET="${MINDSHIFT_RECORDINGS_BUCKET:-arborfam-hub-mindshift-recordings}"
# Word-level diarization cross-check: owner-approved ON for this deployment
# (2026-08-13) — every upload gets our own speaker verification even when the
# vendor transcript already hears 2+ speakers. Same default-in-script pattern
# as the allowlist above, for the same reason. Override via .env / env var.
MINDSHIFT_DIARIZE_CROSSCHECK="${MINDSHIFT_DIARIZE_CROSSCHECK:-$(read_env MINDSHIFT_DIARIZE_CROSSCHECK)}"
MINDSHIFT_DIARIZE_CROSSCHECK="${MINDSHIFT_DIARIZE_CROSSCHECK:-1}"
# Watch-domain defaults (I1, final whole-branch review 2026-08-15): same
# --set-env-vars-REPLACES-everything trap as the allowlist above — without a
# default here for these four, a redeploy silently WIPES them and the watch
# domain falls back to in-memory stores (server/watch/store.py's
# MemoryLiveSessionStore) and a None blob store (server/watch/blobs.py) in
# PROD, losing every live session / pairing / capture on the next restart or
# rollout. Same defaulting pattern as MINDSHIFT_ALLOWED_ORIGINS/
# MINDSHIFT_RECORDINGS_BUCKET above; mirrors gauge's own
# GAUGE_FIRESTORE_PROJECT/GAUGE_CAPTURE_BUCKET/GAUGE_ALLOW_LEGACY_ACCOUNT
# defaults in gauge's scripts/deploy_cloudrun.sh.
MINDSHIFT_FIRESTORE_PROJECT="${MINDSHIFT_FIRESTORE_PROJECT:-$(read_env MINDSHIFT_FIRESTORE_PROJECT)}"
MINDSHIFT_FIRESTORE_PROJECT="${MINDSHIFT_FIRESTORE_PROJECT:-$PROJECT}"
MINDSHIFT_CAPTURE_BUCKET="${MINDSHIFT_CAPTURE_BUCKET:-$(read_env MINDSHIFT_CAPTURE_BUCKET)}"
MINDSHIFT_CAPTURE_BUCKET="${MINDSHIFT_CAPTURE_BUCKET:-arborfam-hub-mindshift-captures}"
MINDSHIFT_ALLOW_LEGACY_ACCOUNT="${MINDSHIFT_ALLOW_LEGACY_ACCOUNT:-$(read_env MINDSHIFT_ALLOW_LEGACY_ACCOUNT)}"
MINDSHIFT_ALLOW_LEGACY_ACCOUNT="${MINDSHIFT_ALLOW_LEGACY_ACCOUNT:-true}"
MINDSHIFT_WATCH_STT="${MINDSHIFT_WATCH_STT:-$(read_env MINDSHIFT_WATCH_STT)}"
MINDSHIFT_WATCH_STT="${MINDSHIFT_WATCH_STT:-whisper}"

# Optional config: forwarded to Cloud Run only when present (in .env or a real
# env var). This is what makes MINDSHIFT_MODEL, STT_PROVIDER, etc. genuinely
# switch-in-.env — no code change needed as models/config evolve.
for k in MINDSHIFT_MODEL STT_PROVIDER MINDSHIFT_UPLOAD_STT WHISPER_MODEL MINDSHIFT_ALLOWED_ORIGINS MINDSHIFT_RECORDINGS_BUCKET LOG_LEVEL RATE_LIMIT_ENABLED RATE_LIMIT_PER_MINUTE MINDSHIFT_DIARIZE_CROSSCHECK MINDSHIFT_DIARIZE_MAX_POOLED_COSINE MINDSHIFT_DIARIZE_SPLIT_MIN_MARGIN MINDSHIFT_FIRESTORE_PROJECT MINDSHIFT_CAPTURE_BUCKET MINDSHIFT_ALLOW_LEGACY_ACCOUNT MINDSHIFT_WATCH_STT; do
  v="${!k:-}"
  [[ -n "$v" ]] || v="$(read_env "$k")"
  if [[ -n "$v" ]]; then
    ENV_VARS="${ENV_VARS}@${k}=${v}"
    echo "   config  : ${k}=${v}"
  fi
done

echo "──────────────────────────────────────────────────────────────"
echo " MindShift → Cloud Run"
echo "   project : $PROJECT"
echo "   region  : $REGION"
echo "   service : $SERVICE"
echo "   source  : $REPO_ROOT (Dockerfile via Cloud Build)"
echo "   secrets : ANTHROPIC_API_KEY, DEEPGRAM_API_KEY loaded (not printed)"
echo "──────────────────────────────────────────────────────────────"

echo "→ Setting active project"
gcloud config set project "$PROJECT" >/dev/null

echo "→ Enabling required APIs (run, cloudbuild) — idempotent"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  --project "$PROJECT"

echo "→ Deploying (this builds the image; first deploy takes a few minutes)"
# --allow-unauthenticated : the app has no login yet (auth is deferred).
# --timeout 3600          : WebSocket audio sessions are long-lived.
# --no-cpu-throttling     : analysis runs as a BACKGROUND task after the 202;
#                           with request-based throttling that background work
#                           got ~zero CPU and our local ECAPA diarization
#                           crawled for many minutes (observed live 2026-08-14).
#                           Always-allocated CPU while an instance exists.
# --min-instances 0       : with always-allocated CPU a warm instance bills
#                           continuously (~$150+/mo at 4cpu/2Gi) — owner chose
#                           scale-to-zero instead (2026-08-14): a few $/mo, at
#                           the cost of ~10-15s to start the first live session
#                           after idle.
# --memory 2Gi            : the media pipeline (video download + PCM decode +
#                           prosody arrays + ffmpeg transcode) OOM-killed the
#                           default 512Mi container mid-request (surfacing as
#                           malformed 502/503s). Pinned here so redeploys
#                           can't silently shrink it back.
# --port 8080             : matches the Dockerfile's EXPOSE/uvicorn port.
# --cpu 4                 : software HEVC-10bit (phone HDR video) decode needs
#                           real cores — at the default allocation the 360p
#                           transcode of a 48s phone clip exceeded the 600s cap
#                           (~125s on a laptop). Billed per-second only while
#                           requests run, so cost impact is small. Pinned so
#                           redeploys can't silently shrink it back.
gcloud run deploy "$SERVICE" \
  --source "$REPO_ROOT" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --timeout 3600 \
  --no-cpu-throttling \
  --min-instances 0 \
  --memory 2Gi \
  --cpu 4 \
  --port 8080 \
  --set-env-vars "^@^${ENV_VARS}"

# ---------------------------------------------------------------------------
# Report the URLs.
# ---------------------------------------------------------------------------
URL="$(gcloud run services describe "$SERVICE" \
  --region "$REGION" --format 'value(status.url)')"
WSS="${URL/https:\/\//wss:\/\/}"

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "✓ Deployed."
echo "   Service URL (HTTPS/health) : $URL"
echo "   WebSocket base (wss)       : $WSS"
echo "   Live audio endpoint        : ${WSS}/ws/session/{session_id}"
echo ""
echo "   Point the app at the backend with the HTTPS host:"
echo "     EXPO_PUBLIC_API_URL=$URL"
echo "   (The app rewrites http→ws itself and appends /ws/session/<id>.)"
echo "──────────────────────────────────────────────────────────────"
