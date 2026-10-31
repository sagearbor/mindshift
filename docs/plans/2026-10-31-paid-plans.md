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

### Speech-to-text allowance (owner decision 2026-10-31)

Free gets **2 hours of cloud transcription as a LIFETIME total per account**
(one constant: `plans.FREE_STT_LIFETIME_SECONDS`), not per day. Pro gets a
calendar-month allowance (`stt_monthly_seconds`, 60 h by default). The counters
live on `ms_plans/{uid}` (`lifetime`, `monthly`), are persisted by the usage
flusher, survive restarts and day rollover, and are never touched by a redeem
or grant. Over the limit: the live session keeps the on-device transcript and
coaching, one notice says "You've used your 2 free hours of transcription",
nothing is deleted. Other metrics keep their daily caps. Like the library caps,
the windows are enforced only when plans are configured (Firestore project set).
Worst-case free cost is now a one-time ~$0.50-0.90 per account.

### Referrals (built)

Every signed-in (non-guest) user has a personal share code (8 chars, e.g.
`ABCD-2345`; shown in Settings > Plan with a Share button, created lazily on
`GET /plan`). A new user types it into the same "Have a code?" box.

- The new user gets **+7 days Pro immediately**.
- The referrer gets **+7 days Pro only after the new user has done >= 10 minutes
  of real live listening** (counted by the same usage flusher, so it is server-side
  evidence, not a client claim). Credited exactly once per new account.
- Rewards are Pro **days**, never percent discounts (store subscriptions cannot
  take a dynamic % off).
- Records: `ms_referrals/{new_uid}` = `{referrer, status: pending|rewarded|capped,
  created_at, closed_at}`; `ms_referral_codes/{CODE}` = `{uid}`. Opaque uids only,
  no email, name or device data. All removed by `DELETE /me` for that user's own
  records.

Abuse rules: guests cannot redeem (disposable uids); one referral per account,
ever (`referred_by` + the `ms_referrals/{uid}` doc, written in a transaction); own
code refused; the 10-minute listening gate makes throwaway-account farming cost
real time (and each fake account still hits the 2-hour lifetime STT cap); at most
**10 rewards per referrer (70 days)**, further ones are logged as `capped`; the
usual per-IP and 8 failures/hour/uid redeem limits apply. Known limit: a person
can make a second real account and listen 10 minutes; that is accepted, it costs
us about $0.05 of STT.

Users already on a paid store subscription: `plans.grant_days` returns `banked`
and adds the days to `credit_days` on their plan doc instead of changing
`plan_expires_at` (a promo must not overwrite the store's expiry). **Phase 2 must
apply `credit_days`**: when RevenueCat sends EXPIRATION (or the subscription is
cancelled and lapses), if `credit_days > 0`, set `plan=pro`, `plan_source=promo`,
`plan_expires_at=now + credit_days`, and zero the field. Same rule for admin
codes if you ever want them to bank. In phase 1 nobody has a store subscription,
so banking is unit-tested but not yet exercised.

## Phase 2 (plan): RevenueCat + store subscriptions

### Pricing proposal

Pro at **$12/month and $99/year** (store tiers; inside the owner's $10-15
band, annual about 31% off). Reasoning:

- Variable cost is STT $0.25-0.45 per listening hour plus coach LLM tokens
  (about 40k tokens per measured 30-minute coached call). A user who listens
  1 h/day costs $7.50-13.50/month in STT alone. A $12 price only works if
  Pro's STT cap is bounded, so the shipped Pro row (12 h/day) is a *ceiling*
  for abuse, not the expected use. Recommend setting Pro STT to about 2 h/day
  (60 h/month) before launch: worst case ~$27/month, typical (20 h) ~$5-9.
- Apple/Google take 15% (small business program / subscriptions after year 1)
  to 30%, so $12 nets about $10.2 (15%) to $8.4 (30%).
- $10 would leave almost nothing for heavy users; $15 fits the market for
  "AI meeting coach" but is a harder first price. Start at $12, revisit with
  `scripts/usage_report.py` data. A cheaper tier later is a one-line table row
  (`plus`), not a rewrite.

### Products (one subscription group, mirrors oral history)

| Product id | Period | Price |
|---|---|---|
| `mindshift_pro_monthly` | 1 month | $12 |
| `mindshift_pro_annual` | 1 year | $99 |

Entitlement id `pro`. Level resolution is by entitlement or product-id prefix
(`mindshift_<level>_...`), highest wins, same as `eventPlanLevel` in
`revenuecat.ts`. Adding a level later = new products + new entitlement + a row
in `plans.LIMITS`.

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
   Subscriptions -> create group `MindShift Pro`; create `mindshift_pro_monthly`
   ($12) and `mindshift_pro_annual` ($99), localized name/description, review
   screenshot of the paywall, all countries. Users and Access -> Integrations ->
   In-App Purchase -> generate a key (.p8). Update App Privacy for Purchases.
   First IAPs ship with the next app version.
2. **Google Play Console**: Settings -> Payments profile (merchant account).
   App -> Monetize -> Subscriptions -> create `mindshift_pro_monthly` (base plan
   `monthly`) and `mindshift_pro_annual` (base plan `annual`), activate. Create a
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
