> **SUPERSEDED (2026-08-25).** Use [2026-08-25-demo-runbook-in-app-call.md](2026-08-25-demo-runbook-in-app-call.md).
> The "second device carries the call" advice below (§0 ranked setups a/b) is withdrawn — the owner
> rejected any two-device setup. The demo is now an **in-app 3-way call** (Sage + Dad on the preview
> APK, Mom as therapist observer in Safari): the app owns the mic, so the OS in-call restriction in
> §0 no longer applies. §1 pre-flight rows, enrollment, §5 screenshots and the `adb logcat` tip were
> carried over into the new runbook.

# Demo runbook — Sage (Pixel 10) coaches a call with Mom (therapist, iPhone/Safari)

> Checklists, not prose. Companion to the
> [three-track handoff](2026-08-24-realtime-three-tracks-handoff.md) (§6–§11),
> [Safari fast loop](2026-08-24-web-safari-fast-loop.md) and
> [live e2e](2026-08-24-live-e2e.md). UI labels below are the exact strings in
> `apps/mobile/src` at #157. Research date 2026-08-24.

## 0. The verdict on "mic during a call" (read first — it changes the setup)

**Same-phone cellular speaker-phone does NOT work. The Pixel 10 must not be the phone on the call.**

- Since **Android 10 (API 29)** the framework shares the mic by *silencing*, not blocking: an ordinary
  app's `AudioRecord` opens fine and delivers **zeros**. While a voice call is active — defined as
  `AudioManager.getMode()` = `MODE_IN_CALL` (cellular) **or** `MODE_IN_COMMUNICATION` (VoIP) — "the
  call always receives audio"; the only apps that still get mic audio are an **accessibility service**,
  and only a **privileged pre-installed app with `CAPTURE_AUDIO_OUTPUT`** may capture the call itself
  (`AudioSource.VOICE_CALL` / `VOICE_UPLINK` / `VOICE_DOWNLINK`, device `TYPE_TELEPHONY`).
  [developer.android.com/media/platform/sharing-audio-input](https://developer.android.com/media/platform/sharing-audio-input),
  [source.android.com/docs/core/audio/concurrent](https://source.android.com/docs/core/audio/concurrent).
- The `AudioSource` an ordinary app picks is irrelevant in-call: `MIC`, `VOICE_COMMUNICATION` and
  `VOICE_RECOGNITION` are all "ordinary" and all get silence. expo-audio's `AudioStream` (what
  `useAudioStream` consumes) hard-codes `MediaRecorder.AudioSource.MIC`
  (`node_modules/expo-audio/android/src/main/java/expo/modules/audio/AudioStream.kt:141`); there is
  no option that changes the outcome. Being the **default dialer** is not an exemption for *capture* —
  the dialer *is* the call path; Google's Play policy (2022-05-11) additionally bans Accessibility-API
  call recording for third parties ([The Register](https://www.theregister.com/2022/04/22/google_banning_thirdparty_callrecording_apps/),
  [GrapheneOS #3401](https://github.com/GrapheneOS/os-issue-tracker/issues/3401)). MindShift is neither.
- Android 11 only added `setPrivacySensitive()` (assistant/voice-comm priority *outside* calls);
  14/15 added the foreground-service `microphone` type. The in-call rule is unchanged through the
  current doc, i.e. **Android 16 on the Pixel 10 (and 17)** — no version makes this work.
- **Speaker-phone changes nothing** for the policy. Mom's voice from the Pixel's own speaker *would* be
  room audio the mic hears — but capture is silenced, so neither voice arrives.
- **VoIP on the same phone is the same story**: WhatsApp/Meet/Duo set `MODE_IN_COMMUNICATION`, which
  the doc counts as "voice call active". A VoIP app that merely holds the mic without that mode is
  the non-call concurrent-capture case (most-recent / privacy-sensitive app wins, the other gets
  silence) — still not MindShift.
- Observed with expo-audio: silence during the call, and on Android 12+ it **stays silent after
  hang-up** until recording is restarted ([expo/expo#20198](https://github.com/expo/expo/issues/20198),
  [#20208](https://github.com/expo/expo/issues/20208)). So if the Pixel takes *any* call mid-session:
  Stop, then Start again.
- **iOS (Mom):** a cellular or FaceTime call interrupts every third-party `AVAudioSession`; nothing can
  record through it (Apple engineer's answer: file an enhancement request —
  [forums/thread/784782](https://developer.apple.com/forums/thread/784782); interruption model:
  [Responding to Interruptions](https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioInterruptions/HandlingAudioInterruptions.html)).
  Safari's `getUserMedia` track goes muted/ended and the Web Speech recognizer ends
  ([twilio-video.js#941](https://github.com/twilio/twilio-video.js/issues/941)). **Safari on the
  iPhone that is on the call cannot run Live Coach.** Her review path (Therapist Dashboard) is unaffected.

### What will happen on the same phone, exactly (option c — do not do this)
Start Listening → pre-flight fine → dial Mom on speaker → at "connected" `AudioRecord` returns zeros →
Silero VAD never opens a turn → no `turn_local`, no server transcript (server gets the same zeros) →
on-device STT (Google's recognizer service) may or may not hear — unverified, but without a VAD turn
nothing is coached anyway → screen shows "live", transcript frozen. After hang-up, likely still frozen
(#20198) until Stop/Start.

### Ranked setups
1. **(a) Second device carries the call; Pixel only listens — use this.** Son's phone (or any spare)
   calls Mom on cellular speaker, volume high, lying next to the Pixel. Pixel: DND on, *Speaker-phone*
   mode, Start Listening before dialing. Most robust: no OS policy involved, Mom's voice is plain room
   audio.
2. **(b) VoIP from the Mac** (Google Meet link; Mom joins from Safari or the Meet app on her iPhone,
   or FaceTime audio if you ever have an Apple ID on the Mac). Same principle as (a) with a louder
   speaker; bonus: Mom can *also* open https://arborfam-hub.web.app in Safari on a Mac/iPad (not the
   phone on the call) in Therapist mode and watch her side transcribed.
3. **(c) Same-phone cellular speaker-phone — fails as described above.** Only useful to *demonstrate*
   the OS limit (screenshot the frozen transcript for the record).

Mom's equivalent: the iPhone that is on the call can only be the phone; her Safari session, if any,
runs on a second Apple device, foreground only, screen unlocked.

## 1. T-1 day

- [ ] Pixel: uninstall any Play-store MindShift (different signing key), install the `preview` APK:
      https://expo.dev/accounts/sagearbor/projects/mindshift/builds/555b04e3-9a28-4efb-8bbf-e9a63e2dcac7
      (Play Protect "install anyway"; ~600 MB free). Launch on Wi-Fi so the `preview` OTA and the
      ~80 MB `ecapa.onnx` download once.
- [ ] Pixel: Android Settings → Apps → MindShift → Permissions → Microphone *Allow*, Notifications on;
      Battery *Unrestricted*. Keep screen timeout ≥ 10 min (no `AppState` handling — a locked screen
      may stop buffers; #153 PLAUSIBLE).
- [ ] Pixel: Android on-device speech pack present: Settings → System → Languages → (Speech /
      On-device speech recognition) → English (US) downloaded. Pre-flight "On-device speech ✓" proves it.
- [ ] Sign in on the Pixel with your Google account; Mom creates/signs into hers at
      https://arborfam-hub.web.app in Safari (Google sign-in popup — no private-browsing tab).
- [ ] Therapist link: Pixel → Settings → **Therapist** → *My therapist* card → her account email →
      link; leave *Share sessions automatically* **on**. Mom: Settings → **Therapist Dashboard** →
      "Wants to share sessions with you" → **Accept**. (Pending still auto-shares; accepted is cleaner.)
- [ ] Enroll YOUR voice on the Pixel, in the demo room, phone where it will lie: Settings → Voice →
      *Voice profile* → train (4 phrases, tap **Record** each). #149: same-room/same-phone prints match;
      old prints from another device don't. Do NOT pre-enroll Mom — name her mid-call (below).
- [ ] Live Coach: pick **Speaker-phone** mode (remembered per account), **On-device coaching** on,
      **Scoreboard** on if you want the kindness race in the demo (it needs on-device coaching).
- [ ] 60-s solo rehearsal: Start Listening, talk to yourself in two voices for a minute, Stop. Expect:
      transcript lines labelled, ≥1 suggestion tagged *on-device* or *cloud*, session summary card,
      latency headline `[fastLoop] N turns, median segment-end→speak X ms`. Then: Your Day shows the row
      badged `· live · speaker`; Replay shows "what you could have said" within ~1 min; Mom's Therapist
      Dashboard → Patients → you → the session with its live badge.
- [ ] Full rehearsal of setup (a) with the son's phone for 2 min. If the transcript freezes when the
      call connects, the Pixel is on the call — it must not be.
- [ ] Optional: `adb logcat -s ReactNativeJS` over USB during rehearsal captures the full latency table.

## 2. T-1 hour

- [ ] Pixel: Wi-Fi + signal ok, battery > 50 %, **Do Not Disturb on** (an answered call on the Pixel
      silences capture; ringing alone does not), volume up, screen timeout long, app opened once
      (warms the Cloud Run instance; `--max-instances 1`, cold start is tens of seconds).
- [ ] Second device (call carrier): charged, volume max, speaker-phone tested, placed 20–30 cm from the
      Pixel, Pixel mic (bottom edge) facing it and you.
- [ ] Live Coach idle screen: pre-flight panel reads **On-device speech ✓ ready · Speaker-ID ✓ N
      enrolled · model cached · Suggestions <provider or cloud> · Turn detection Silero VAD**, and the
      "who's here" strip lists *You*. Screenshot it.
- [ ] Mom: Safari open at the site, signed in, Therapist Dashboard loads. If she has a Mac/iPad: Live
      Coach → **Therapist** mode → keep the tab foreground; expect two permission prompts on Start.
- [ ] Agree the script: you speak first (self-voice lock-in); include one deliberately sharp line
      ("that's ridiculous, you never listen") so a NUDGE and an escalation are guaranteed.

## 3. During the call

Sage (Pixel):
1. Live Coach → mode **Speaker-phone** → **Start Listening**. Status dot green, "On-device: …" line
   lists what loaded. Say a full sentence *before* dialing.
2. Dial Mom from the second device; put it on speaker next to the Pixel.
3. Good looks like: your lines labelled *You*, hers *Speaker B*; a suggestion card within ~1–3 s of
   *her* turn ending, tag *on-device* (Nano/bundled) or *cloud* (p50 ≈ 0.9 s); the coach speaks only
   in silence (it holds while anyone talks); a **NUDGE** banner + haptic after your sharp line; the
   escalation counter in the session strip ticks.
4. Name her: tap her speaker chip → "Who is this?" → **New person…** → "Mom" → save. Later lines read
   *Mom*; the voice is remembered (#157).
5. If the chip shows "You: Speaker B" for your own lines, tap **You: … ⇄** once to flip.
6. Keep the call to 8–12 min. **Stop Listening** *before* hanging up (a held suggestion is dropped;
   the summary and latency headline render).

Mom (Safari, Therapist mode, only on a second Apple device): Start → allow speech + mic → two-column
transcript, nothing spoken. If she gets a banner "speech recognition failed … transcript from the
server", that's the known iOS-Safari limit — keep going. Her phone stays a plain phone.

If something is off:
- **Pre-flight shows On-device speech ✗** → the reason is printed; toggle **On-device coaching** off:
  the legacy server path (Deepgram + cloud suggestions, server TTS) still runs the demo. Fix the
  speech pack later.
- **Speaker-ID ✗** → reason is `0 enrolled` (enroll) or model not cached (wait, Wi-Fi). Without it,
  voices are labelled by speaking order — speak first, flip the chip if wrong. Nudges then rely on
  the chip, so get it right.
- **No suggestion within 3 s** → check the status dot (WS), the tag on the last card (`via=` at the
  end tells who served), the *interject* slider (raise it), and whether someone is still talking (the
  coach never speaks over speech). Nano's first use can download AICore for minutes: budgets are
  1.5 s/4 s, then the next rung answers, so cloud should still land. If *nothing* — the Pixel is on a
  call or the screen locked: Stop, Start.
- **Nudges never fire** → they only fire on *your* turns judged escalated (text-tone from the LLM;
  audio tone is dark). Make sure your lines are labelled *You*, then say the sharp line, plainly.
- **Transcript frozen mid-session** → Stop, Start (mic silenced by a call, or focus lost).

## 4. After

- [ ] Pixel: session summary card (duration, turns per person, Escalations, First words median/best,
      "shared with your therapist automatically"). Tap **Review this conversation →**.
- [ ] Your Day: the row `· live · speaker`; open it → Replay → wait for "what you could have said"
      (polls 5 s × 12).
- [ ] Growth: the new point; per-person dimensions show *Mom* once named.
- [ ] Settings → Voice → People: *Mom* listed with seconds of learned speech.
- [ ] Scoreboard (if on): compare the two lines; the same board shows in Session detail.
- [ ] Mom: Therapist Dashboard → pull to refresh → Patients → you → the session: escalation markers,
      named people, "Kindness scoreboard (patient's session)", write a private note (only she sees it).
- [ ] Ask her the therapist question: were the "could have said" lines ones she'd offer?

## 5. What to screenshot / save for the next Claude session

1. Pre-flight panel (idle, before Start) and the "On-device: …" status line right after Start.
2. Transcript after ~30 s (labels + the "You: … ⇄" chip) and the "Who is this?" sheet when naming Mom.
3. First suggestion card with its *on-device*/*cloud* tag; the first NUDGE banner.
4. Session summary card + the latency headline under it.
5. Full latency table: `adb logcat -s ReactNativeJS > ~/Desktop/demo-logcat.txt` (lines
   `#n end= spk= stt= pros= llm= speak= via=`), or a screen recording of the console if untethered.
6. Your Day row, Replay reflections, Growth point.
7. Mom's side: Dashboard patient list, session detail, and her Safari Live Coach status line/banners
   if she ran Therapist mode.
8. Server: Cloud Run logs for the session window (look for "Skipping audio tone … recovered 0.00s",
   `latency_summary`, reconnects).
9. If anything froze: note the wall-clock time of the freeze vs. call connect/hang-up/screen lock.
