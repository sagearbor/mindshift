# MindShift — Deployment Runbook

Ship the backend to **Google Cloud Run** (WebSocket-capable) and the app to
**Google Play internal testing** via **EAS Build**. This is the step-by-step
morning runbook. Steps marked **[INTERACTIVE — owner only]** cannot be
scripted (they need a human in a browser / your Google account).

There are two goals, and you can stop after whichever you need:

- **Fastest path (visit today):** deploy the backend, build a `preview` **APK**,
  send Dad the EAS install link. **No Play Console needed at all.** Jump to
  [§4 Build the app](#4-build-the-app) → preview, then
  [§6 Fastest path](#6-fastest-path-for-the-visit-no-play-console).
- **Play internal testing:** everything below, ending with the `.aab` uploaded
  to the Play internal track and Dad added as a tester.

---

## 0. What you need once

- A Google account (the one that owns the GCP project + Play Console).
- **Anthropic** API key (`console.anthropic.com`) and **Deepgram** API key
  (`console.deepgram.com`). Claude 3 Haiku is the default model and is cheap.
- Node 18+ and Python 3.11+ locally.
- For Play only: a **Google Play Developer account** ($25 one-time) — this is
  the owner's account; it can't be scripted.

---

## 1. Prereqs — install & authenticate  ·  [INTERACTIVE — owner only]

```bash
# Google Cloud SDK
brew install --cask google-cloud-sdk
gcloud auth login                     # opens a browser

# Pick / confirm a GCP project WITH BILLING ENABLED.
# You may reuse the GCP project behind an existing Firebase app
# (Firebase projects *are* GCP projects). List what you have:
gcloud projects list

# If you need billing: https://console.cloud.google.com/billing  (link a project)

gcloud config set project <PROJECT_ID>
```

> Cloud Run source deploys build with Cloud Build and need billing enabled.
> The deploy script enables the `run` and `cloudbuild` APIs for you.

---

## 2. Fill `.env` with the two keys

```bash
cd <repo-root>
cp .example.env .env      # (or: cp env.example .env)
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...
#   DEEPGRAM_API_KEY=...
```

`.env` is git-ignored — the keys never get committed. The deploy script reads
them from `.env` at runtime and passes them to Cloud Run as service env vars.

---

## 3. Deploy the backend to Cloud Run

```bash
./scripts/deploy_cloudrun.sh <PROJECT_ID>            # region defaults to us-central1
# or: ./scripts/deploy_cloudrun.sh <PROJECT_ID> us-central1 mindshift-api
```

First deploy takes a few minutes (it builds the Docker image). When it finishes
it prints:

```
Service URL (HTTPS/health) : https://mindshift-api-XXXX.us-central1.run.app
WebSocket base (wss)       : wss://mindshift-api-XXXX.us-central1.run.app
```

**Copy the HTTPS Service URL** — that is your `EXPO_PUBLIC_API_URL`. The app
rewrites `http(s)`→`ws(s)` itself and appends `/ws/session/<id>` for the live
audio stream.

Quick smoke test (optional):

```bash
curl https://mindshift-api-XXXX.us-central1.run.app/health
```

---

## 4. Build the app  ·  [INTERACTIVE — owner only for login]

EAS builds run in Expo's cloud. Log in once:

```bash
npm install -g eas-cli          # if not already installed
cd apps/mobile
npx eas-cli login               # or set EXPO_TOKEN in the environment for CI
```

**Point the build at your backend.** The Cloud Run HTTPS URL must be baked in
at build time. Set it in `apps/mobile/eas.json` under the profile's
`env.EXPO_PUBLIC_API_URL` (it ships as `http://localhost:8000` — replace it):

```jsonc
// apps/mobile/eas.json → build.preview.env  (and build.production.env)
"env": { "EXPO_PUBLIC_API_URL": "https://mindshift-api-XXXX.us-central1.run.app" }
```

Then build:

```bash
# Fast APK to sideload to Dad TODAY (internal distribution, no Play Console):
eas build -p android --profile preview

# OR the Play-ready App Bundle (.aab) for internal testing:
eas build -p android --profile production
```

When a build finishes, EAS prints a build page URL and (for `internal`
distribution) an **install link / QR code**. Download the `.aab` (production) or
grab the APK install link (preview) from that page.

> First Android build also prompts to **generate an Android Keystore** — say
> **yes** and let EAS manage it. It reuses the same key on later builds (needed
> for Play to accept updates).

---

## 5. Publish to Google Play internal testing  ·  [INTERACTIVE + scripted]

### 5a. Create the app in Play Console  ·  [INTERACTIVE — owner only]

1. Go to <https://play.google.com/console> → **Create app**.
2. Name **MindShift**, app/game = App, free, accept declarations.
3. Package name must be **`com.sagearbor.mindshift.app`** (matches `app.json`).
4. Left nav → **Testing → Internal testing** → create a release track.
5. Complete the minimum store-listing / content items Play nags you for
   (these gates are unavoidable and owner-only).

### 5b. Create a Play service account + grant it access  ·  [INTERACTIVE — owner only]

This is the classic **two-place** gotcha — miss the second step and uploads
fail with a permissions error:

1. **Google Cloud** (same project is fine):
   <https://console.cloud.google.com/iam-admin/serviceaccounts> → create a
   service account → **create a JSON key** → save it somewhere safe, e.g.
   `~/.config/play/mindshift-sa.json`. **Do not commit it.**
2. **Enable the API:** in that GCP project enable **Google Play Android
   Developer API**.
3. **Play Console → Users & permissions → Invite new user** → paste the service
   account's email → grant **admin (or release)** access to this app. *(This is
   the step people forget — the key alone is not enough; Play must also grant it
   permission.)*

### 5c. Upload the `.aab` to the internal track  ·  [scripted]

```bash
pip install google-api-python-client google-auth   # one time
python3 scripts/play_publish.py \
    --aab <path-to-downloaded-app-release.aab> \
    --service-account ~/.config/play/mindshift-sa.json \
    --notes "first internal build"
# defaults: --package com.sagearbor.mindshift.app  --track internal  --status completed
```

> A brand-new app's first upload may need to be finished/reviewed in the Console
> the first time; the script handles the "changes sent for review" fallback
> automatically. Internal-testing releases reach testers immediately.

### 5d. Add Dad as an internal tester  ·  [INTERACTIVE — owner only]

1. **Testing → Internal testing → Testers** → add Dad's **Gmail** address (or a
   tester list containing it).
2. Copy the **opt-in URL** and send it to Dad. He opens it on his Android phone,
   taps **Become a tester**, then installs MindShift from Play.

---

## 6. Fastest path for the visit (no Play Console)

If you just want it on Dad's phone today, skip §5 entirely:

1. Do §1–§3 (backend on Cloud Run).
2. Set `EXPO_PUBLIC_API_URL` in `eas.json` → `build.preview.env`.
3. `eas build -p android --profile preview`
4. Send Dad the **EAS install link / QR** from the finished build page. He taps
   it on his Android phone and installs the APK directly. Done.

(The `preview` profile builds a directly-installable APK with internal
distribution — no keystore-for-Play, no review, no testers list.)

---

## 7. Platform notes / honest limitations

- **iOS:** not possible without a paid **Apple Developer account** ($99/yr) and
  a Mac/EAS credentials setup. Not covered here.
- **Web build:** the web target has **no microphone capture wired up yet**, so a
  hosted web app can't do live coaching today (separate task). If/when a web
  build lands, set `MINDSHIFT_ALLOWED_ORIGINS` (see `.example.env`) to its
  origin so the browser is allowed to open the WebSocket — native apps send no
  Origin and are always allowed.
- **Database on Cloud Run:** the SQLite file lives on the instance and resets on
  redeploy — fine for a tester demo; move to a mounted volume / managed DB for
  anything durable.
- **Cost/keys:** the deployed service holds your Anthropic + Deepgram keys as
  env vars. `--allow-unauthenticated` means anyone with the URL can reach it
  (auth is deferred); don't post the URL publicly.

---

## Quick reference — the morning sequence

```bash
# 1. Auth + project (interactive: gcloud auth login)
gcloud auth login
gcloud config set project <PROJECT_ID>

# 2. Keys
cp .example.env .env && $EDITOR .env      # set ANTHROPIC_API_KEY + DEEPGRAM_API_KEY

# 3. Backend → Cloud Run  (copy the printed HTTPS URL)
./scripts/deploy_cloudrun.sh <PROJECT_ID>

# 4. App: set EXPO_PUBLIC_API_URL in apps/mobile/eas.json, then build
cd apps/mobile
npx eas-cli login
eas build -p android --profile preview       # fast APK for Dad today
# and/or:
eas build -p android --profile production     # .aab for Play

# 5. Play (after the interactive Console + service-account setup in §5a/§5b)
pip install google-api-python-client google-auth
python3 ../../scripts/play_publish.py --aab <app-release.aab> \
    --service-account ~/.config/play/mindshift-sa.json --notes "first internal build"
```
