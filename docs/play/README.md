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
| [`play-answers-mindshift.yaml`](play-answers-mindshift.yaml) | **Phone/web listing** — `com.sagearbor.mindshift.app`. Every Play Console field, its value, why, and a `file:line` that proves it. |
| [`play-answers-mindshift-wear.yaml`](play-answers-mindshift-wear.yaml) | **Watch (Wear OS) listing** — `com.sagearbor.gauge.wear`, a *separate* Play record with a standalone form factor. Same schema. Its answers deliberately differ from the phone's in three places: it declares **Health info** (heart rate from the body sensor), it declares a **device ID** (an app-minted install UUID on every telemetry batch), and its **App access** is a partial restriction rather than a login wall — the watch runs unpaired. Never import one listing's answers into the other. |
| `../../apps/mobile/public/privacy/index.html` | The privacy policy. Deploys to <https://arborfam-hub.web.app/privacy>. Covers **both** listings — the watch has no policy of its own. |
| `../../apps/mobile/public/delete-account/index.html` | The data-deletion page Play requires for any app with accounts. Deploys to <https://arborfam-hub.web.app/delete-account>. Also the watch's deletion URL: there is no delete control on the wrist. |
| `../../server/tests/test_play_answer_packs.py` | The gate. Asserts both packs parse, carry every schema section, and that **every `proof:` path still resolves to a real file**. A rename that silently orphans a proof fails CI instead of quietly turning the pack back into a guess. |

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
  advertising ID unless something adds `AD_ID` to the merged manifest. **This
  one does not survive contact with a native module.** The Wear OS app mints
  its own per-install UUID and sends it on every telemetry batch, so its pack
  declares the type. Check for an app-generated install id, not just for an
  advertising id.
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

## Refresh checklist — run this after every release

A pack is a snapshot of the code on one commit. It goes stale the same way a
comment does, and it goes stale *silently*: the YAML still reads fine while the
binary has moved. Run this before any release that changes what the app does,
and always before re-submitting a Data safety form.

Start by bumping `_schema.verified_against_commit` and `_schema.last_updated`,
and mark every leaf you change with a `changed_<date>:` sibling note — that
marker is what lets a browser agent re-enter *only the changed fields* instead
of walking the whole console again. The 2026-09-20 refresh is the worked
example.

Then work the five questions below. Each is a diff, not a re-read — that is what
makes this twenty minutes instead of a re-derivation.

1. **Permissions diff.** `git diff <last verified commit>..HEAD -- apps/mobile/app.json apps/watch/wearApp/src/main/AndroidManifest.xml`.
   A new permission almost always means a new Data safety type, and a new
   *foreground service type* means a new justification form on Android 14+.
   Check the merged manifest of the built AAB too, not just the source — a
   dependency can inject `AD_ID` and flip the advertising-ID answer without
   anyone editing a file.
2. **New data stored.** Grep the diff for new persistence, not new features:
   new Firestore fields, new GCS prefixes, new columns, new `series` keys, a
   new `put_*`/`save_*` call. Ask of each one: is it a new Play *type*, or more
   of an existing one? Two real examples from 2026-09-20 — `hr_bpm_series`
   was a whole new type (Health info); the heat judge's
   `arousal`/`valence`/`judge` series was an inference about the user that
   landed under an existing one (Personal info → Other info).
3. **New SDKs and new outbound hosts.** Read the *whole* dependency list, not a
   grep (`package.json` for the phone, the `dependencies {}` block for the
   watch), and diff it. Then grep the diff for new `fetch(`/`axios`/WebSocket
   URLs/SDK constructors and for new keys in the deploy script's env list —
   that env list is the real processor roster. A model that runs *in our own
   container* (`server/tone_id.py`) is not a new processor; a model behind
   someone's API is.
4. **New user-to-user surface.** Anything that lets one account see another
   account's data changes the content rating's "users interact" answer and
   usually adds a `shared: true`. Check both directions — the Wear couples card
   was a receive-only surface and still counted.
5. **Policy parity.** Re-read <https://arborfam-hub.web.app/privacy> against the
   matrix you just refreshed. The policy and the declaration must say the same
   thing, and the policy is the half that rots first because nothing tests it.
   If you changed a data type in step 2, the policy paragraph changes in the
   same commit — not the next one.

Finally, `pytest server/tests/test_play_answer_packs.py`. It will not tell you
an answer is wrong, but it will tell you a proof no longer points at anything,
which is the failure mode that turns a checkable pack back into an assertion.

Then re-run the browser agent with the prompt below, telling it explicitly to
**only** re-enter fields carrying the new `changed_<date>:` marker.

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
