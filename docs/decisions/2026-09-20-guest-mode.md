# Decision: ship "Continue as guest" (Firebase Anonymous Auth)

**Date:** 2026-09-20
**Status:** implemented, on by default. Requires Anonymous sign-in to be
enabled in the `arborfam-hub` Firebase project — if it is off, the button
reports "This sign-in method isn't enabled for the app yet" and nothing else
breaks.
**Decides:** whether the app keeps its hard login gate.
**Code:** `apps/mobile/src/screens/LoginScreen.tsx`,
`apps/mobile/src/store/authStore.ts`,
`apps/mobile/src/components/GuestBanner.tsx`, `server/guest_quota.py`,
`server/auth.py` (`Identity`), `server/watch/auth.py` (`Principal.is_guest`).
Tests: `apps/mobile/__tests__/GuestMode.test.tsx` (11),
`server/tests/test_guest_mode.py` (37).

---

## The decision

The login screen gets a "Continue as guest" button that calls
`signInAnonymously`. A guest gets the whole product — Live Coach in every mode
but Call, recordings, voice profile, watch pairing — on a real Firebase uid, so
every existing uid-scoped store, share rule and delete path works unchanged. A
persistent one-line banner says where that data lives and offers an inline
"Create account" that **links** an email/password or Google credential onto the
same uid, so nothing recorded as a guest is stranded by signing up. In-app
Calls are hidden for guests, because a call needs a second real account on the
other phone — there is nothing a credential on *this* phone could unlock.
Server-side, an anonymous token is marked `is_guest` on the account record and
bounded by a quota: **3 live sessions per UTC day, 10 minutes per session**
(`MINDSHIFT_GUEST_MAX_SESSIONS_PER_DAY`, `MINDSHIFT_GUEST_MAX_SESSION_MIN`),
enforced both on the WebSocket that actually spends Deepgram/Anthropic money
(close 4429, one `guest_limit` frame first) and on `POST /sessions/live`, which
buys the batch analysis (HTTP 429). Both refusals carry the same sentence:
*"Guest limit reached — create a free account to continue."*

## Why

Two reasons, and the second is the bigger one. **Play review:** App access
previously demanded a demo account the owner had to create and hand over, which
means a reviewer signing in can see that account's recordings, transcripts and
voiceprints — an uncomfortable thing to be required to do for a
therapy-adjacent app, and a recurring source of review friction when the
credentials rot. Both answer packs now say "All functionality is available
without special access" with empty credential fields. **First-run friction:**
asking someone to create an account before they have heard the product coach a
single sentence is the steepest part of the funnel, and the thing being asked
for (an account) is worth the least to a person who does not yet know whether
they want it.

## The abuse posture, stated plainly

The quota is a **cost brake, not a security control**, and
`server/guest_quota.py` says so at the top rather than only here. Two holes,
both known and both accepted:

1. **The counter is per process.** Cloud Run runs several instances, so a
   determined guest's real ceiling across a fleet of N instances is up to N ×
   the limit.
2. **Signing out mints a fresh uid with a fresh allowance, forever.** No
   per-uid counter can fix that; only device attestation or a payment signal
   could, and neither is worth adding to a free try-before-you-sign-up path.

What the quota *does* buy is the failure mode that actually costs money in
practice: no single guest session can run unbounded. The per-session minute cap
is hard, in-process, and applies to the socket doing the spending, so a phone
left on a desk streaming all afternoon, or a client that reconnect-loops, is
capped rather than open-ended. Deliberate, repeated abuse is bounded by that
per-session cap plus the existing global `MAX_WS_SESSIONS` and per-IP rate
limiters — not by the daily counter.

## Deletion

`DELETE /me` already worked: the walk in `server/account_deletion.py` is
uid-scoped and never reads how the uid signed in, so a guest's data and their
Firebase user are erased by the same in-app control, with the same
type-to-confirm and fresh-token guards. That is now a tested promise rather
than an accident (`test_delete_me_works_for_an_anonymous_account`), because
Play requires an in-app delete for any account the app creates, and a guest
account is one — created silently, but created.

## TODO, not done

* **Persist the counter.** Firestore `guest_usage/<uid>/<day>` would survive a
  restart and hold across instances. Not done because the per-session cap is
  the part with teeth and it needs no storage.
* **Nightly cleanup.** Anonymous accounts idle for N days, and the data under
  them, are never swept. A guest who walks away leaves a uid and their
  recordings behind indefinitely. The retention the privacy policy already
  states applies to them like anyone else; a dedicated sweep is still worth
  adding.

## Reproduce

```bash
pytest server/tests/test_guest_mode.py -q            # 37
cd apps/mobile && npx jest __tests__/GuestMode.test.tsx   # 11
```
