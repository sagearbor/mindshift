# Demo runbook — in-app 3-way call: Sage (Pixel 10) + Dad (Android) + Mom (iPhone Safari, therapist)

> Checklists, not prose. Supersedes [2026-08-24-demo-runbook.md](2026-08-24-demo-runbook.md)
> (its "second device carries the call" advice is rejected — one device per person, the app IS the
> call). Server contract: [in-app calls](2026-08-25-in-app-calls.md) (`main` ≥ b5b2fbe, #164).
> Background: [three-track handoff](2026-08-24-realtime-three-tracks-handoff.md) §6–§11,
> [Safari fast loop](2026-08-24-web-safari-fast-loop.md).
>
> **†** = a UI string from the planned client PR (`feat/calls-client`, not open when this was
> written). The server-side names (roles, join code, `/call/<code>`, `?role=therapist`) are fixed;
> the exact tap labels are **verify on device** and may differ slightly. Strings without † are
> the current `apps/mobile/src` labels at #164.

## 0. Why this works on one device (the old runbook's §0 no longer applies)

Android 10+ and iOS hand third-party apps *silence* while the phone is on a cellular/VoIP call —
that is why the Pixel could not phone Mom and listen at the same time. In an in-app call there is
no OS call: MindShift owns the mic on every device, audio goes phone-to-phone over WebRTC (the
server only relays signaling; `AudioManager` never enters `MODE_IN_CALL`), and each phone
transcribes **only its owner** on-device and sends `turn_local` exactly as a solo session does.
The server merges the three streams into one shared transcript, coaches each *participant* with
the full context, and hands the *therapist* a read-only copy. Attribution is structural (member =
speaker), so no speaker-ID, no server STT and no diarization of the remote voices is needed.
Nobody dials anybody: **if a real phone call is ringing or answered on any of the three phones,
that phone's side goes silent — DND on all three.**

## 1. Cast

| Who | Device | Role | Joins via | Coached? |
|---|---|---|---|---|
| Sage | Pixel 10, preview APK | `participant`, host, slot A ("You") | **Start a call**† | yes — nudges on own turns, suggestions after Dad's turns |
| Dad | his Android, same preview APK | `participant`, slot B ("Speaker B" → named "Dad") | **Join with code**† or the `…/call/<CODE>` link | yes — mirror image of Sage |
| Mom | iPhone, Safari at https://arborfam-hub.web.app | `therapist` observer, slot C ("Mom (therapist)") | link `…/call/<CODE>?role=therapist` → **Answer**† | **never** — sees both transcripts, both suggestion streams, scoreboard |

## 2. T-1 day

Installs (both Android phones — the old preview APK `555b04e3…` has **no call UI**):
- [ ] Pixel + Dad's phone: install `<APK link from morning message>` (Play Protect "install anyway",
      ~600 MB free). A Play-store or debug-signed MindShift must be uninstalled first (signature
      mismatch); the previous EAS `preview` APK uses the same upload key and can be installed over.
      If the morning message says "OTA only", launch the existing APK twice on Wi-Fi instead.
- [ ] Both phones: launch on Wi-Fi once so the `preview` OTA and `ecapa.onnx` (~80 MB) download.
- [ ] Both phones: Settings → Apps → MindShift → Microphone *Allow*, Notifications on, Battery
      *Unrestricted*; screen timeout ≥ 10 min (a locked screen may stop buffers; #153).
- [ ] Both phones: on-device speech pack: Settings → System → Languages → On-device speech
      recognition → English (US) downloaded. Pre-flight "On-device speech ✓" proves it.
- [ ] Dad: Google account signed in to the app (any Google account; nothing else needed).

Accounts and links:
- [ ] Mom: signs in at https://arborfam-hub.web.app in Safari (Google popup — not a Private tab;
      Safari, not the Gmail in-app browser). Therapist Dashboard loads.
- [ ] Therapist link Mom↔Sage: Pixel → Settings → **Therapist** → *My therapist* → her email →
      link, *Share sessions automatically* **on**. Mom: Settings → **Therapist Dashboard** → "Wants
      to share sessions with you" → **Accept**. (The call grants her both episodes even without a
      link — the link is what makes Sage appear in her Patients list and auto-shares future sessions.)
- [ ] Tell Dad: his half of the call becomes *his* episode and is shared with Mom automatically
      because she was on the call (`shared_with` on the end card).

Voices and toggles:
- [ ] Sage: enroll your voice on the Pixel, in the demo room, phone where it will lie: Settings →
      Voice → *Voice profile* → 4 phrases, **Record** each. Dad: same on his phone (optional for the
      call — attribution is by member — but the 60-s solo test and future **In person**† sessions
      use it). Do NOT pre-enroll each other; name people mid-call.
- [ ] Live Coach on both phones: **On-device coaching** on; **Scoreboard** switch on (needs
      on-device coaching). Mode: **Call**† for the demo, **In person**† (was *Speaker-phone*) for the
      solo test; the mode is remembered per account.
- [ ] 60-s solo test, each phone, mode **In person**†: Start Listening, talk to yourself in two
      voices, Stop. Expect: labelled lines, ≥1 suggestion tagged *on-device* or *cloud*, summary card,
      `[fastLoop] N turns, median segment-end→speak X ms`. Your Day shows the row; Replay shows "what
      you could have said" within ~1 min; Mom's Dashboard → Patients → Sage → the session.
- [ ] 3-min call rehearsal with the three real devices, same Wi-Fi, following §4. This is the only
      way to learn whether WebRTC connects on your network (§4 fallbacks) and whether Safari grants
      speech + mic together (the one unverified iOS item).
- [ ] Optional: `adb logcat -s ReactNativeJS > ~/Desktop/demo-logcat.txt` over USB during the
      rehearsal — latency table plus any `call_join:` / `rtc_signal:` error frames.

## 3. T-1 hour

- [ ] All three: Wi-Fi (same network if you can — see §4 fallbacks), battery > 50 %, volume up,
      **Do Not Disturb on** on all three phones, screen timeout long. Mom: iPhone unlocked, Safari
      foreground, no other audio app, Silent switch irrelevant (WebRTC audio plays anyway — verify).
- [ ] Sage: open the app once (warms Cloud Run — `--max-instances 1`, cold start is tens of seconds;
      one instance also means every call socket lands on the same process, which the registry needs).
- [ ] Pixel Live Coach idle screen, mode **Call**†: pre-flight reads **On-device speech ✓ ·
      Speaker-ID ✓/✗ (irrelevant in a call) · model cached · Suggestions <provider or cloud> · Turn
      detection Silero VAD**. Screenshot it. Dad's phone: same, screenshot.
- [ ] Do **not** create the call the night before — an open call expires after `ttl_minutes`
      (default 180); create it ≤ 15 min before people join.
- [ ] Agree the script: Sage speaks first, then Dad; include one deliberately sharp line each
      ("that's ridiculous, you never listen") so a NUDGE and an escalation are guaranteed on both
      sides, and one calm repair line so Mom sees a suggestion land.

## 4. During the call — exact taps

**Sage (host, Pixel):**
1. Live Coach → mode **Call**† → **Start a call**†. The card shows the 6-char join code (letters
   A–Z minus O/I, digits 2–9; case, spaces and dashes don't matter) and the link
   `https://arborfam-hub.web.app/call/<CODE>`. Display name: "Sage"†.
2. Send Dad the link or read him the code. Send Mom the link **with `?role=therapist`**† (the
   therapist variant — if she opens the plain link she joins as a participant and takes Dad's seat:
   409 `call already has two participants` for Dad → she must leave and reopen the right link).
3. Wait for the "who's here" strip: *Dad · connected*, *Mom (therapist) · connected*
   (`call_state.connected: true`). Audio starts when the strip goes green — say "can you hear me".
4. Say your first full sentence. Your lines render locally under **You**; Dad's arrive as *Speaker B*
   (`transcript` events from his phone); Mom's as *Mom (therapist)* (or her email until she types a
   name).
5. Name Dad: tap his speaker chip → **Who is this?** → **New person…** → "Dad" → save. This is your
   naming only (per viewer); Mom names him on her side if she wants. The **You: … ⇄** flip chip is
   meaningless in a call (attribution is structural) — ignore it.
6. 8–12 min. Hang up with **Stop Listening** (or **End call**†, which ends it for everyone). The
   call ends when the last participant socket leaves; each phone then shows the summary card with
   the call's episode.

**Dad (Android):**
1. Live Coach → mode **Call**† → **Join with code**† → type the code → display name "Dad"† → tap
   **Answer**†. (Tapping the `…/call/<CODE>` link on Android may open the browser instead of the
   app — verify; if so, use Join with code.)
2. Speak normally; his phone hears only him. He sees Sage's lines as *Speaker A* / "Sage", gets his
   own suggestions after Sage's turns and NUDGEs on his own sharp lines.
3. When told: **Stop Listening**.

**Mom (iPhone Safari, therapist):**
1. Open the therapist link in Safari (signed in). The page shows the call card with the code and
   role *Therapist*.
2. Tap **Answer**† **once**. Everything Safari gates on a user gesture starts inside that tap:
   `getUserMedia`, the AudioContext, and Web Speech recognition. Two prompts follow — *speech
   recognition*, then *microphone* — **Allow both**. No second tap is needed; if she dismisses a
   prompt, reload and tap Answer again.
3. Keep Safari foreground and the screen unlocked for the whole call (iOS releases the mic on
   lock/app switch — the page shows a banner and she must tap Answer again).
4. What she sees: both transcripts (two-column), each participant's suggestion cards tagged with
   whom they are for, NUDGEs as they fire, the kindness scoreboard over everyone (from the tone
   fields on each `transcript` — verify it renders for the observer). Nothing is ever spoken to her
   and nothing is generated *for* her. She may talk — her lines are transcribed and merged, and
   the participants get suggestions about them like any other member's turns.
5. She does not need to hang up; the call ends when Sage and Dad stop. Her end card lists both
   episodes (`episodes` on `call_ended`) and she has read access to both.

**What good looks like:** Sage hears Dad and Mom through the Pixel; a suggestion card on Sage's
screen within ~1–3 s after *Dad's* turn ends (tag *on-device* or *cloud*, cloud p50 ≈ 0.9 s), spoken
only while everyone is silent; the same on Dad's phone after Sage's turns; a NUDGE banner + haptic
on the sharp line, escalation counter ticks; Mom's page shows every line from both phones within
~1 s of it being finalized, plus every suggestion card, without lag on her side.

**If a peer can't connect** (strip stays *connected* but no audio, or "connecting…" forever):
- Only STUN is configured; **TURN is not** (`MINDSHIFT_TURN_URLS`, `MINDSHIFT_TURN_USERNAME`,
  `MINDSHIFT_TURN_CREDENTIAL` are unset). Two phones both behind carrier NAT may never reach each
  other. Fix order: (1) put all three devices on the same home Wi-Fi; (2) Pixel hotspot for Dad's
  phone; (3) set the three TURN env vars on the Cloud Run service (any coturn/Metered/Twilio TURN;
  they are read per request — no image rebuild, no restart) and re-join.
- **Transcript and coaching do not depend on WebRTC.** Every phone's turns go to the server over its
  own WebSocket, so even with no audio path the merged transcript, suggestions and Mom's page keep
  working. If the three of you are in the same house, that is the demo: sit in one room, skip the
  audio, watch the screens.
- A phone that drops: re-open **Join with code**† with the same code — re-join is idempotent, a new
  socket replaces the old one, and the client re-offers when `call_state` shows the peer connected.
- Dad can't join at all (410 expired → create a new call; 409 seat taken → someone used the wrong
  link). Worst case run the demo 2-way (Sage + Mom therapist): the call works with two members.

**If suggestions lag:** status dot (WS), the tag on the last card (`via=` says who served), the
*interject* slider, and whether someone is still talking (the coach never speaks over speech).
Remember participants are coached on the **other participant's** turns — Sage gets nothing after his
own lines. Nano's first use can download AICore for minutes; budgets are 1.5 s/4 s then the next
rung answers, so cloud should still land. Nothing at all → **On-device coaching** off: server STT
fallback (Deepgram + cloud suggestions) still merges and coaches inside the call.

**If nudges don't fire:** they fire only on *your own* turns judged escalated (text-tone from the
LLM; audio tone is dark). In a call your turns are always yours (no chip to fix) — so say the sharp
line plainly and let it finish (a NUDGE needs a finalized turn). Frozen transcript → Stop, re-join.

## 5. After

- [ ] Sage + Dad: summary card (duration, turns per person, Escalations, First words median/best,
      "shared with Mom"). Tap **Review this conversation →**.
- [ ] Your Day: one row each, mode `call`, titled *Call with Dad and Mom (therapist)* — **two rows for
      one call is a bug** (the phone must not POST `/sessions/live` for a call session). Open it →
      Replay → "what you could have said" (polls 5 s × 12).
- [ ] Growth: the new point; per-person dimensions show *Dad* once named.
- [ ] Settings → Voice → People: *Dad* listed (voiceprint attached only if you named him via a
      person with a print; otherwise a name only).
- [ ] Scoreboard: both lines on the Pixel; the same board in Session detail.
- [ ] Mom: Therapist Dashboard → pull to refresh → Patients → Sage → the call episode (escalation
      markers, named people, "Kindness scoreboard (patient's session)"); Dad's episode appears too
      (direct grant, no link). Write a private note on each.
- [ ] Ask her the therapist question: were the "could have said" lines ones she'd offer? And: did
      seeing both sides' suggestions live add anything over the replay?

## 6. What to screenshot / save for the next Claude session

1. Pixel + Dad's phone: pre-flight card (idle) and the call card with code/link after **Start a call**†.
2. The "who's here" strip once all three show *connected*; the Android/Safari permission prompts.
3. Mom's Safari page: right after **Answer**† (status line/banners), and mid-call with both columns.
4. Transcript after ~30 s on each phone (labels), the "Who is this?" sheet when naming Dad.
5. First suggestion card on each phone with its tag; the first NUDGE banner on each; Mom's view of
   the same suggestion (the `for_uid` copy).
6. Session summary card + latency headline on each phone; the end card on Mom's page.
7. `adb logcat -s ReactNativeJS > ~/Desktop/demo-logcat.txt` from the Pixel (lines
   `#n end= spk= stt= pros= llm= speak= via=`, plus any `call_join:` / `rtc_signal:` errors).
8. Your Day rows (one per phone), Replay reflections, Growth; Mom's Dashboard patient list and both
   episode details.
9. Server: Cloud Run logs for the window (`call_state`, `latency_summary`, reconnects, 409/410s).
10. If audio never connected: which networks each phone was on (Wi-Fi vs carrier) and the
    ICE state the app showed — that decides whether TURN goes in next.
