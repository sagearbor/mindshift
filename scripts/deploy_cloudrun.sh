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
# Optional config: forwarded to Cloud Run only when present (in .env or a real
# env var). This is what makes MINDSHIFT_MODEL, STT_PROVIDER, etc. genuinely
# switch-in-.env — no code change needed as models/config evolve.
for k in MINDSHIFT_MODEL STT_PROVIDER WHISPER_MODEL MINDSHIFT_ALLOWED_ORIGINS MINDSHIFT_RECORDINGS_BUCKET LOG_LEVEL RATE_LIMIT_ENABLED RATE_LIMIT_PER_MINUTE; do
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
# --min-instances 1       : avoid cold starts dropping a live session.
# --memory 2Gi            : the media pipeline (video download + PCM decode +
#                           prosody arrays + ffmpeg transcode) OOM-killed the
#                           default 512Mi container mid-request (surfacing as
#                           malformed 502/503s). Pinned here so redeploys
#                           can't silently shrink it back.
# --port 8080             : matches the Dockerfile's EXPOSE/uvicorn port.
gcloud run deploy "$SERVICE" \
  --source "$REPO_ROOT" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --timeout 3600 \
  --min-instances 1 \
  --memory 2Gi \
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
