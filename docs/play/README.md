# Play Console answer packs — how to reuse this for the other 7 apps

The Google Play "Set up your app" checklist has **no API**. Every one of those
forms has to be clicked by a human or a browser agent (Claude in Chrome, Cowork).
Browser agents are fine at clicking and terrible at deciding — they stall, or
worse, they guess. So the work splits in two:

1. **A code-reading agent** (no browser) produces an answer pack: every field,
   with its literal value, derived from what the app actually does.
2. **A browser agent** walks the pack top-to-bottom and never has to think.

This directory holds step 1's output for MindShift:

| File | What it is |
|---|---|
| [`play-answers-mindshift.yaml`](play-answers-mindshift.yaml) | The full answer pack. Every Play Console field, its value, why, and a `file:line` that proves it. |
| `../../apps/mobile/public/privacy/index.html` | The privacy policy. Deploys to <https://arborfam-hub.web.app/privacy>. |
| `../../apps/mobile/public/delete-account/index.html` | The data-deletion page Play requires for any app with accounts. Deploys to <https://arborfam-hub.web.app/delete-account>. |

A gitignored working copy also lives at `tmp/play-answers-mindshift.yaml`; the
committed copy here is the source of truth.

---

## The rule that makes this work

**Never write a data-safety answer from the app's description. Write it from the
code.** A mismatch between the Data safety declaration and what the binary does
is the single most common way an app gets flagged, and it is the one thing a
generic template cannot get right. Every claim in the pack carries a `file:line`
so the next person can re-check it in ten seconds instead of re-deriving it.

The corollary: **the privacy policy and the data-safety matrix must say the same
thing.** Write them in the same sitting, from the same evidence. If you change
one later, change the other in the same commit.

---

## What stays constant across all 8 apps

These do not need re-deriving. Copy them.

| Field | Value |
|---|---|
| Play developer account | `7699975610134980668` (Sage Arbor) |
| Contact email | `sagearbor@gmail.com` |
| Government app | No |
| Ads | No — none of these apps has an ad SDK. **Verify per app** with the dependency-list check below; do not assume. |
| Financial features | None |
| News app | No |
| COVID-19 app | No |
| Target audience | 18 and over only, unless the app is genuinely built for minors |
| Store listing appeals to children | No |
| Encrypted in transit | Yes (any app on Cloud Run + HTTPS/wss) |
| Deletion mechanism | Same shape: in-app deletes for individual items, a self-serve `DELETE /me` behind a type-to-confirm flow in Settings → Account, and a `/delete-account` page whose email route is only for someone who can no longer sign in |
| Agent hard rules | Identical — see `notes_for_the_agent` in the YAML |

Also constant for any app sharing MindShift's architecture (Expo + Firebase Auth
+ FastAPI on Cloud Run + GCS/Firestore):

- **Personal info → Email address**: collected, required, App functionality +
  Account management.
- **Personal info → User IDs**: collected, required — the Firebase uid is the
  tenancy key.
- **App info and performance → Crash logs / Diagnostics**: collected, required
  if the app auto-sends on error (MindShift does).
- **Device or other IDs**: *not* collected — Expo apps do not pull an
  advertising ID unless something adds `AD_ID` to the merged manifest.
- **Location / Contacts / Calendar / Web browsing / Installed apps**: not
  collected.

---

## What must change per app

Work through this list. Each item is a decision that cannot be inherited.

1. **Package name, current version, and whether the Play record already exists.**
2. **Is there a login?** If yes, App access needs a demo account — and the owner
   must supply it. Never invent credentials. Never hand Play a real account: a
   reviewer signing in sees that account's data.
3. **The data-type matrix.** Re-derive it. The questions to answer with code:
   - What leaves the device, on which endpoint, to whom? Grep for `fetch(`,
     `axios`, WebSocket URLs, SDK constructors.
   - What is stored server-side, in which bucket/collection, under what key?
   - Which third-party API keys does the deploy script set? That is the real
     list of processors, and it is usually shorter or longer than the README's.
   - Is there any always-on / auto-sent telemetry? If it is auto-sent, it is
     **Required**, not Optional.
   - Are there any biometrics (voiceprints, face embeddings)? Play has no
     biometrics category — declare them under **Personal info → Other info** and
     name them explicitly in the policy.
4. **Ads, definitively.** Read the whole dependency list, not a grep:
   ```bash
   node -e 'const p=require("./apps/mobile/package.json");console.log(Object.keys({...p.dependencies,...p.devDependencies}).join("\n"))'
   ```
   Then check the manifest for `com.google.android.gms.permission.AD_ID`. A
   string match for "sentry"/"analytics" inside a Jest `transformIgnorePatterns`
   regex is not a dependency — MindShift has exactly that false positive.
5. **Content rating category and the user-interaction question.** Does the app
   let users reach *each other*? Chat, calls, shared content, comments — any of
   those makes the interaction answer Yes, which produces a "Users Interact"
   descriptor. That descriptor is correct; do not tune answers to remove it.
   Then: can *strangers* reach each other? If discovery/matching exists, the
   answer changes and so does the rating.
6. **Store listing copy.** Name ≤30, short description ≤80, full description
   ≤4000. Write the full description for the actual buyer, not for a keyword
   crawler.
7. **Category.** Pick the honest one, but know the consequences: *Health &
   Fitness* and *Medical* pull the app into Play's Health apps declaration and
   health-content policy. If the app is not a medical tool, do not put it in a
   medical category.
8. **Graphics.** Icon 512×512, feature graphic 1024×500, and 2–8 phone
   screenshots (each side 320–3840 px). Screenshots must show the real app, not
   a marketing render. Say in the pack exactly which screens to capture and how
   to get each into the right state — that is the part an agent cannot invent.
9. **Privacy policy URL and deletion URL.** Both must actually resolve before
   the forms are filled in, or the console rejects them.

---

## The pipeline, end to end

```
# 1. Derive the pack (a code-reading agent, no browser, ~20 min)
#    -> docs/play/play-answers-<app>.yaml
#    -> <web public dir>/privacy/index.html
#    -> <web public dir>/delete-account/index.html

# 2. Publish the two pages FIRST — Play validates the URLs.
scripts/web_deploy.sh --dry-run     # must pass
scripts/web_deploy.sh
curl -s -o /dev/null -w '%{http_code}\n' -L https://<host>/privacy
curl -s -o /dev/null -w '%{http_code}\n' -L https://<host>/delete-account

# 3. Hand the pack to a browser agent (prompt below).

# 4. The owner presses the two Submit buttons the agent is forbidden to touch.
```

Step 2 before step 3 is not optional: Play rejects a privacy policy URL that
404s, and a browser agent that hits that rejection will start improvising.

---

## The prompt to hand a browser agent

Copy this verbatim, changing only the two bracketed values.

> You are completing the Google Play Console **"Set up your app"** checklist for
> **[com.sagearbor.mindshift.app]** on the developer account **7699975610134980668**.
> Do not do anything else in the console.
>
> Your answers are in **[docs/play/play-answers-mindshift.yaml]**. Read the whole
> file before you touch the browser. It is authoritative: every field you need is
> in there, with the literal value to enter. Work through it in order —
> `app_access`, `ads`, `content_rating`, `target_audience`, `data_safety`,
> `government_apps`, `other_declarations`, `store_listing`, `graphics_checklist`.
>
> **Hard rules. Breaking any of these is a failure, not a judgement call.**
> 1. Do **not** create, edit, promote, roll out or publish a release — not to
>    production, not to a testing track, not as a draft. Nothing under "Release"
>    or "Testing" is in scope.
> 2. Do **not** press **Submit** on the content-rating questionnaire. Answer
>    every question, reach the review/summary screen, screenshot it, stop.
> 3. Do **not** press **Submit for review** on the Data safety form. Save the
>    draft and stop at the summary screen.
> 4. Do **not** invent any value. Any field the pack marks `OWNER_MUST_SUPPLY`
>    is left blank and reported — this includes the demo-account credentials for
>    App access.
> 5. Do **not** change developer account settings, payment profile, users &
>    permissions, API access, or accept any new Play agreement.
>
> **When Play asks something the pack does not cover:** do not guess. Record the
> exact question text and every option offered, leave it unanswered, and include
> it in your report. Play's wording changes constantly; a wrong Data safety
> answer is much worse than an unfinished form.
>
> Press **Save** on each section when it is offered — saving drafts is expected
> and fine. Only the final Submit buttons are off-limits.
>
> **Report back with:** every section you completed and the state you left it in;
> every field left blank and why; the exact text of any question the pack did not
> cover; and screenshots of the content-rating summary screen and the Data safety
> summary screen.

---

## Known gaps worth fixing before the other 7

These bit MindShift and will bite the rest of the fleet, because they are
architectural rather than app-specific:

- ~~**No self-serve account deletion.**~~ **Closed.** `DELETE /me`
  (`server/routers/account.py` + `server/account_deletion.py`) erases every
  storage tier for the authenticated uid — GCS prefixes, Firestore documents,
  SQLite rows, capture blobs — and then the Firebase Auth user, last, so a
  mid-way failure leaves the account retryable rather than orphaning data. The
  UI is Settings → Account → "Delete my account" (one Expo screen, so phone and
  web get it together). **Reuse this across the other 7:** the tier walk is a
  standalone module that takes stores as arguments, the guards (a freshly
  issued token via `auth.get_fresh_uid`, a `{"confirm": "DELETE"}` body, a
  3/min per-IP budget) are worth copying verbatim, and the scope + shared-data
  rule are documented once in `server/account_deletion.py`'s module docstring,
  which the privacy policy, the `/delete-account` page and the answer pack all
  restate.
- **No provider-side retention controls.** No zero-data-retention header on the
  LLM client, no `redact`/`no_store` on the STT vendor. Each vendor's account
  default governs. Setting them once in a shared client would let every app's
  policy make a stronger, still-honest claim.
- **No retention limit on stored user content.** "Until you delete it" is the
  truthful answer, and it is what the policy says — but a GCS lifecycle rule
  would be a better one.
