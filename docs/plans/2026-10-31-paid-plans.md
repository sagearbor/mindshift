# Paid plans: phase 1 (shipped in this PR) and phase 2 (plan only)

Status 2026-10-31. Phase 1 is code in the repo and needs no store account.
Phase 2 is a design plus exact owner console steps; nothing in it is built.
The reference is the owner's `oralhistory_timeline` (docs/SUBSCRIPTIONS.md,
docs/GIFT_PLANS.md, functions/src/{plans,entitlements,revenuecat,gift-plans}.ts),
mirrored here for FastAPI + Firestore.

## Phase 1 (done): plans, limits, redeem codes

| Piece | Where |
|---|---|
| Levels, limits table, effective-plan rule, redeem rules, stores | `server/plans.py` |
| `GET /plan`, `POST /plan/redeem` | `server/routers/plan.py` |
| Caps come from the user's plan | `server/usage_meter.py` (`check`, `prime`) |
| Library gate (one call) | `server/routers/library.py` -> `plans.check_library_create` |
| Mint / list / revoke / grant | `scripts/mint_code.py` |
| Settings "Plan" card, redeem box, upgrade message | `apps/mobile/src/components/PlanCard.tsx`, `PlanLimitSheet.tsx`, `src/api/plan.ts`, `src/store/planLimitStore.ts` |

Firestore (project `MINDSHIFT_FIRESTORE_PROJECT`, same one the library uses):
`ms_plans/{uid}` = `plan`, `plan_expires_at`, `plan_source`
(`revenuecat|admin|promo`), `plan_updated_at`; `ms_plans/{uid}/redemptions/{code}`;
`ms_plan_codes/{CODE}` = `plan, days, max_uses, uses, expires_at, revoked, note,
created_by`. (The `ms_` prefix avoids the shared GCP project's other apps.)
Add a Firestore rule that denies all client access to both collections (the
server uses the Admin/ADC path, which bypasses rules).

Behaviour, in short:

- **Free = today's numbers** (usage_meter's `DAILY_*` constants and their
  `MINDSHIFT_DAILY_*` env vars), so nobody regresses. Pro is a row in
  `plans.LIMITS`; any value can be overridden with
  `MINDSHIFT_PLAN_<LEVEL>_<KEY>` (e.g. `MINDSHIFT_PLAN_PRO_LIBRARY_ITEMS=300`).
  Adding a level = one entry each in `LEVELS`, `LABELS`, `LIMITS`.
- The stored level counts only while `plan_expires_at` is in the future (or null).
- Over a limit never deletes anything. Daily caps keep their existing degrade
  behaviour; the library gate refuses only *new* items with HTTP 429
  `{type:"plan_limit", limit, used, cap, plan, upgrade_plan, message}`;
  `quota_notice` now also carries `plan` and `upgrade_plan`. The app shows the
  server's sentence in `PlanLimitSheet`.
- Redeem: auth required, no guests (a guest uid is disposable), per-IP limiter
  plus 8 failures/hour/uid, one generic error for unknown/revoked/expired/used-up
  codes, atomic use counting in a Firestore transaction, idempotent per
  user+code (a 7-day code is `mint_code.py mint --plan pro --days 7`, tested),
  extends a same-level grant (`expires = max(current, now) + days`),
  never shortens, and does not spend a use when the user already has Pro with no
  expiry or via a paying subscription.
- If `MINDSHIFT_FIRESTORE_PROJECT` is unset, plans are unavailable: everyone is
  free, `/plan/redeem` answers 503, library plan caps are not enforced. No
  fake in-memory plan in production.
- `room_mode` and `heads_up_cues` are flags in the table (both true for free so
  nothing changes). They are reported by `/plan` but not enforced; making either a
  Pro perk means flipping the table value and adding an `effective_cached(uid)`
  check at the (other agent's) call site.
- `DELETE /me` removes `ms_plans/{uid}`, its redemptions, the user's referral
  code and referral record (`plan_records` in the deletion counts).
- Known gap: the per-uid redeem failure limiter is per process.

### Levels, listening hours and cost (owner decision 2026-10-31)

Levels (ordered): `free`, `basic`, `pro`, `max`. The unit is **listening
hours**. Worst case per hour: **$0.51** = cloud STT $0.45 + about 33k AI tokens
at Claude Haiku 4.5 ($1/M in, $5/M out, blended $1.80/M) = $0.06. Token caps are
derived from hours (`TOKENS_PER_HOUR` = 40k, i.e. 33k measured plus ~20%
headroom). Everything is a constant at the top of `server/plans.py`
(`ALLOWANCE_HOURS`, `PRICE_USD`, `LIBRARY`, env override
`MINDSHIFT_PLAN_<LEVEL>_<KEY>`); adding a level is one entry in each. Daily STT
and AI caps are abuse backstops at 25% of the monthly allowance.

| Level | Price/mo | Allowance | Max cost to us (all cloud, $0.51/h) | Net after 15% store fee | Worst-case profit and margin | Profit if half used |
|---|---|---|---|---|---|---|
| Free | $0 | 1 h once; 5 items | $0.51 once | n/a | -$0.51 once | n/a |
| Free + referral max | $0 | 2 h once (1 h + 1 h bonus); 5 items | $1.02 once | n/a | -$1.02 once | n/a |
| Basic | $10 ($100/yr) | 8 h / month; 25 items | $4.08 | $8.50 | $4.42 (52%) | $6.46 |
| Pro | $25 ($250/yr) | 20 h / month; 200 items | $10.20 | $21.25 | $11.05 (52%) | $16.15 |
| Max | $100 ($1,000/yr) | 80 h / month; 1000 items | $40.80 | $85.00 | $44.20 (52%) | $64.60 |

With STT on-device the cost is only AI tokens ($0.06/h), so margins rise above
90%. Annual prices are 10x monthly (two months free); net figures are per month.

If 1,000 free users each use their whole allowance: **$510 once** all-cloud
($60 if all on-device STT); **$1,020** if every one also earned the referral
bonus. That is a one-time total, not per month. Every paid tier keeps about a 52% margin even in the
all-cloud worst case; a typical user uses a fraction of the allowance. Token
headroom adds at most $0.013 per allowance hour.

**What a session consumes** (the allowance meters only what costs us):

| Session type | `stt.seconds` (cloud STT) | AI tokens (`llm.*`) | `live.minutes` |
|---|---|---|---|
| All-cloud (phone streams audio to Deepgram) | consumed | consumed | recorded, NOT part of the allowance |
| Phone-only STT (local-first; the server skips vendor audio, `ctx.local_first` in `audio_pipeline.py`) | **none** | consumed | recorded, NOT part of the allowance |

So the binding listening time is the smaller of remaining STT seconds and
remaining AI tokens converted at 40k tokens per hour. Wall-clock live minutes
only have a generous daily backstop (8 h free, 12 h paid), because on-device
listening costs us nothing. Counters live on `ms_plans/{uid}` (`lifetime`,
`monthly`), survive restarts and day rollover, and are never touched by a
redeem. Plans must be configured (Firestore project set) for the windows to apply.

**Countdown.** `GET /plan` returns `listening` = `{remaining_seconds, period
(lifetime|month|day), limit, allowance_seconds}` for the binding limit. Live
Coach shows a pill ("Free time left: 42 min", or "6.5 h left this month" for
paid), ticking locally and re-syncing every 3 minutes, amber at 10 min, red at 5,
with one gentle banner at 10 / 5 / 1 min. At zero the server pauses cloud
coaching/transcription (the socket and on-device features keep running, one
`quota_notice` is sent) and the app opens the out-of-time sheet (Redeem a code,
Upgrade: coming soon). Nothing is deleted. Known limit: cloud transcription
stops at the cap check on the next audio frame, not at the end of the current
sentence.

### Referrals (built): bonus free listening hours, not paid days

Every signed-in (non-guest) user has a personal share code (8 chars, e.g.
`ABCD-2345`; Settings > Plan, with a Share button; created lazily on
`GET /plan`). A new user types it into the "Have a code?" box. Card copy:
"Invite a friend: you both get 1 more free hour."

- The new user gets **+1 h** immediately (1 h -> 2 h of free listening; the
  bonus raises the STT and AI-token lifetime caps together).
- The referrer gets **+1 h** only after the new user has done >= 10 minutes of
  real live listening (counted by the usage flusher: server-side evidence),
  once per new account.
- Total bonus per account is capped at **+1 h** (so a free account tops out at
  2 h). Rewards are never paid days nor percent discounts. Admin/mint codes can
  grant any level (`mint_code.py mint --plan basic|pro|max --days N`); a store
  subscriber redeeming one is "covered" (no use spent, store expiry untouched).
- Records: `ms_referrals/{new_uid}` = `{referrer, status: pending|rewarded|capped,
  created_at, closed_at}`; `ms_referral_codes/{CODE}` = `{uid}`. Opaque uids
  only, no email or device data; all removed by `DELETE /me`.

Abuse rules: guests cannot redeem; one referral per account, ever; own code
refused; the 10-minute listening gate means a fake account must really listen
(each is itself capped at the free allowance); referrer bonus hard-capped (then
`capped`); per-IP and 8 failures/hour/uid redeem limits apply.

## Phase 2 (plan): RevenueCat + store subscriptions

### Pricing proposal

Basic $10, Pro $25, Max $100 per month (store tiers). Reasoning:

- Variable cost is $0.51 per listening hour worst case (cloud STT $0.45 plus
  about 33k AI tokens). Each paid tier is priced at the same hourly rate
  ($1.25/h for Basic and Pro, $1.25/h for Max, before the store fee), so every
  tier keeps about a 52% margin after a 15% store fee even if the whole
  allowance is used all-cloud, and more when STT is on-device or usage is partial.
- Apple/Google take 15% (small business program / subscriptions after year 1)
  to 30%; at 30% the worst-case margin falls to about 40%.
- Revisit prices and allowances with `scripts/usage_report.py` data. A new tier
  is one entry each in `LEVELS`, `LABELS`, `ALLOWANCE_HOURS`, `PRICE_USD`, `LIBRARY`.

### Products (one subscription group, mirrors oral history)

| Product id | Period | Price |
|---|---|---|
| `mindshift_basic_monthly` | 1 month | $10 |
| `mindshift_basic_annual` | 1 year | $100 |
| `mindshift_pro_monthly` | 1 month | $25 |
| `mindshift_pro_annual` | 1 year | $250 |
| `mindshift_max_monthly` | 1 month | $100 |
| `mindshift_max_annual` | 1 year | $1,000 |
All three share one subscription group ranked max > pro > basic.

Entitlement ids `basic`, `pro`, `max`. Level resolution is by entitlement or product-id prefix
(`mindshift_<level>_...`), highest wins, same as `eventPlanLevel` in
`revenuecat.ts`. Adding a level later = new products + new entitlement + entries in `plans.LEVELS` / `ALLOWANCE_HOURS`.

### Server design

`POST /webhooks/revenuecat` in a new `server/routers/revenuecat.py`:

1. Auth: constant-time compare of `Authorization` (verbatim or `Bearer <v>`)
   against secret `REVENUECAT_WEBHOOK_AUTH`; unset secret -> 503.
2. `app_user_id` == Firebase uid (the app calls `Purchases.logIn(uid)` before any
   purchase). Ignore `$RCAnonymousID:*`; validate uid shape.
3. Event mapping (identical to `mapEventToChanges`):
   - INITIAL_PURCHASE, RENEWAL, UNCANCELLATION, NON_RENEWING_PURCHASE,
     SUBSCRIPTION_EXTENDED, TEMPORARY_ENTITLEMENT_GRANT, REFUND_REVERSED:
     `plan=<level>`, `plan_expires_at=expiration_at_ms`, `plan_source=revenuecat`,
     status active.
   - CANCELLATION: keep plan to `plan_expires_at`, status cancelled, no renew
     (refund => expiry = refund time => free at once by the expiry rule).
   - BILLING_ISSUE: expiry = grace period end, status billing_issue.
   - SUBSCRIPTION_PAUSED: no renew; EXPIRATION follows.
   - EXPIRATION: `plan=free`, status expired.
   - TRANSFER: losing uid -> free; receiving uid synced via REST refresh.
   - TEST, PRODUCT_CHANGE, others: 200 ignored.
4. Out-of-order protection: store `plan_last_event_ms`; skip older events, in a
   Firestore transaction. Skip negative events for a product the user has since
   moved off (`isSupersededProductEvent`).
5. A promo/admin grant must not be clobbered by a stray RevenueCat event: if the
   stored source is `promo|admin` with a later expiry than the event implies,
   keep the later expiry and the higher level (the server stores the max).
6. `POST /plan/refresh` (auth): re-reads the subscriber from RevenueCat REST
   (`REVENUECAT_SECRET_API_KEY`) after a purchase or Restore; no key -> no-op.
   Covers webhook delay and TRANSFER.

Client: add `react-native-purchases`, `Purchases.logIn(uid)`, a paywall using
the store's localized price (never hard-coded), Restore Purchases, link to
Terms (EULA) and Privacy (Apple 3.1.2). The "Coming soon" button in `PlanCard`
becomes the purchase button; `plan.purchase.available` from `/plan` flips it.
This adds a native module, so bump `expo.version` and `android.versionCode`
before the next OTA (AGENTS.md runtime-version rule) and ship via the normal
release scripts (`ship-mobile-app` skill), Android and iOS together.

### Owner console steps (exact, in order)

1. **App Store Connect**: Agreements, Tax, and Banking -> sign the Paid Apps
   Agreement, add bank + tax forms (status Active). App -> Monetization ->
   Subscriptions -> create group `MindShift`; create the six product ids above (monthly $10/$25/$100, annual $100/$250/$1,000), localized name/description, review
   screenshot of the paywall, all countries. Users and Access -> Integrations ->
   In-App Purchase -> generate a key (.p8). Update App Privacy for Purchases.
   First IAPs ship with the next app version.
2. **Google Play Console**: Settings -> Payments profile (merchant account).
   App -> Monetize -> Subscriptions -> create the same six product ids (base plans `monthly` and `annual`), activate. Create a
   Cloud service account for RevenueCat with "View financial data" and "Manage
   orders and subscriptions"; download its JSON. Enable real-time developer
   notifications with the Pub/Sub topic RevenueCat gives. License testing: add
   tester Gmails. (A build with the billing library must be on a track first.)
   Wear OS has its own track; subscriptions are sold on the phone app.
3. **RevenueCat**: new project MindShift; add the App Store app (bundle id +
   .p8 + key id + issuer id), the Play app (package + service-account JSON).
   Entitlement `pro` with both products attached. Default offering with two
   packages. Integrations -> Webhooks: URL
   `https://<cloud-run-host>/webhooks/revenuecat`, Authorization value from
   `openssl rand -hex 32`. Copy the public SDK keys (`appl_`, `goog_`) and a
   secret API key (`sk_`). There is already `~/.config/revenuecat` on the owner
   machine from the sibling app; do not reuse its project.
4. **Secrets**: `REVENUECAT_WEBHOOK_AUTH` and `REVENUECAT_SECRET_API_KEY` as Cloud
   Run secrets; public SDK keys as `EXPO_PUBLIC_*` build env. Then send a TEST
   event (expect HTTP 200, ignored).
5. Sandbox-test purchase, renewal, cancel, expiry, restore on iOS TestFlight and
   Play internal testing before submitting.

### Policy notes

- Apple (3.1.1) and Google Play Billing policy: digital features sold inside the
  app (Pro) must use the store's in-app purchase. Do not link out to a web
  checkout or tell users where to pay cheaper from inside the app.
- Promo codes minted by the owner and given away, and admin grants, are fine:
  nothing is *sold* through them. Do not sell redeem codes inside the app or
  advertise a way to buy them in the app. Codes sold on the owner's own website
  outside the app are a grey area under Apple's anti-steering rules; avoid until
  a reader-app or external-link entitlement is clear.
- Store-provided offer codes (Apple offer codes, Play promo codes) are the
  policy-clean way to hand out free months through the store itself once phase 2
  exists; our own codes remain the simple path for hand-outs now.
- Subscription terms (price, period, auto-renew, how to cancel) must be on the
  paywall; Restore Purchases is required (Apple 3.1.1).

### Purchasable gift or coupon codes: recommendation

Oral history sells "a year for N family members" with consumable products
(28 product ids, orders, refund sweeps, recipient emails, ~1,700 lines in
`gift-plans.ts`). **Do not port it now.** MindShift buyers are individuals
paying for their own coaching; there is no family-buys-for-family motive, the
28-product SKU matrix and refund-clawback machinery are large, and the admin
codes from phase 1 already cover the owner's real need (hand a friend, a
pilot customer, or a reviewer a month free). Revisit only if sales asks for
corporate or gifted seats; the cheaper first step then is store-native offer codes
(Apple/Play), which cost nothing to build, not a custom gift system.
