# Apple Watch (watchOS) parity specification

**Date:** 2026-09-21
**Branch this was derived on:** `ship/heat-judge-20260920`
**Source of truth:** the shipped Wear OS app at `apps/watch/` (versionCode 17, versionName 0.5.1 —
`apps/watch/wearApp/build.gradle.kts:25-26`), the shared KMP module at `apps/watch/shared/`, and the
server it speaks to (`server/watch/`, `server/nudge_policy.py`).

This document exists so a watchOS app can be implemented **from the code, not from memory**. Every
behavioural claim about the existing app carries a `file:line` citation. Every claim about watchOS
is either backed by a header read out of the installed SDK (cited as a path inside
`WatchOS26.2.sdk`) or explicitly marked **UNVERIFIABLE WITHOUT HARDWARE**.

> **Read the "What cannot be honest here" section before budgeting this project.** The single most
> important finding is not a missing API — it is that watchOS's haptic vocabulary is a **9-member
> fixed enum with no duration, no amplitude and no timing control**, while this product's entire
> nudge design is *"every LEVEL difference is a rhythm difference"*
> (`apps/watch/shared/src/commonMain/kotlin/app/gauge/shared/NudgeVocabulary.kt:18-24`). The
> vocabulary does not port. It has to be redesigned for watchOS, and the redesign will be worse.

---

## 1. Capability summary

FULL = the shipped behaviour is reproducible on watchOS with no loss of meaning.
PARTIAL = reproducible in degraded form; the degradation is specified in the linked section.
IMPOSSIBLE = the platform forbids it; no workaround preserves the product intent.

| # | Capability | watchOS | Section |
|---|---|---|---|
| 1 | Arbitrary haptic waveform (`VibrationEffect.createWaveform`) | **IMPOSSIBLE** | [§4.1](#41-what-the-platform-gives-you) |
| 2 | Per-tap amplitude / rising & falling ramps / swells | **IMPOSSIBLE** | [§4.1](#41-what-the-platform-gives-you) |
| 3 | The 8-code nudge vocabulary as 8 distinguishable rhythms | **IMPOSSIBLE** | [§4.3](#43-the-vocabulary-does-not-port-code-by-code) |
| 4 | Channel A vs channel B distinguishable by feel alone | **IMPOSSIBLE** | [§4.4](#44-two-channels-one-enum) |
| 5 | Channel-A 3-level escalation ladder (L1/L2/L3 tellable apart) | PARTIAL | [§4.2](#42-the-nearest-honest-mapping) |
| 6 | Positive cues (D / E / R) felt as praise, not alarm | PARTIAL | [§4.5](#45-positives-and-silence-by-contract) |
| 7 | "K never buzzes" silence-by-contract | **FULL** | [§4.5](#45-positives-and-silence-by-contract) |
| 8 | PRD §6 reminder repeat (2 min / 1 min / 10 s, doubling to a 2 min cap) | PARTIAL | [§5](#5-the-reminder-ladder-and-the-back-off) |
| 9 | Proportional pulse train (4 dB bands × duration/amplitude, 250 ms cadence) | **IMPOSSIBLE** | [§6](#6-the-pulse-train) |
| 10 | ARMED shout-tap (one band-1 tap, ≥2 s apart) | PARTIAL | [§6.2](#62-the-armed-shout-tap) |
| 11 | Haptic delivered while the app is not frontmost / wrist is down | **IMPOSSIBLE** (general case) | [§7.3](#73-can-the-wrist-buzz-when-the-app-is-not-frontmost) |
| 12 | Live heart rate stream with availability states (incl. off-body) | PARTIAL | [§8](#8-heart-rate) |
| 13 | HR baseline + `+15/+25/+35` spike ladder | **FULL** | [§8.3](#83-what-ports-unchanged) |
| 14 | Lazy HR registration with a 30 s release grace | **FULL** | [§8.3](#83-what-ports-unchanged) |
| 15 | 16 kHz mono PCM16 mic capture in 1 s windows | PARTIAL | [§9](#9-audio-capture-and-streaming) |
| 16 | Raw PCM streamed to the server over a WebSocket | PARTIAL | [§9.3](#93-streaming-pcm-to-the-server) |
| 17 | Mic duty-cycling (2 s/10 s, motion-gated 2 s/30 s) | PARTIAL | [§9.4](#94-duty-cycling) |
| 18 | All-day armed sentinel (hours of background mic) | **IMPOSSIBLE** | [§7](#7-execution-model-the-real-blocker) |
| 19 | Persistent "ongoing" foreground-service notification | **IMPOSSIBLE** | [§7.4](#74-the-ongoing-notification) |
| 20 | Standalone operation with no phone | PARTIAL | [§10](#10-standalone-vs-companion) |
| 21 | COMPANION mode (no mic; phone listens, wrist buzzes) | PARTIAL | [§10.3](#103-companion-mode-tier-b) |
| 22 | Short-code device pairing over plain HTTP | **FULL** | [§11](#11-pairing-and-auth) |
| 23 | Bearer-token storage | **FULL** (better than today) | [§11.3](#113-token-storage) |
| 24 | WebSocket reconnect back-off ladder | **FULL** | [§12.1](#121-reconnect) |
| 25 | Offline nudge ladder (`NudgeStateMachine`, hold-3 hysteresis) | **FULL** | [§12.2](#122-the-offline-ladder) |
| 26 | Retro-capture 300 s ring + gzip PCM upload | PARTIAL | [§13](#13-retro-capture-and-journal) |
| 27 | Journal auto-upload every 5 minutes | PARTIAL | [§13.2](#132-the-journal-cadence-does-not-survive) |
| 28 | Telemetry ring + flush + crash handler | PARTIAL | [§14](#14-telemetry) |
| 29 | Tile (status + one-tap arm/disarm) | PARTIAL | [§15.1](#151-tile--widgetkit) |
| 30 | Complication (RANGED_VALUE 0..3 arc) | PARTIAL | [§15.2](#152-complication--accessory-widget) |
| 31 | Accelerometer window-stddev movement signal | PARTIAL | [§16](#16-motion) |
| 32 | Battery percent + charging state | **FULL** | [§17](#17-battery) |
| 33 | Reuse of the `shared/` KMP module | PARTIAL | [§18](#18-code-reuse-the-shared-module) |

**Tally: 33 capabilities — 8 FULL · 17 PARTIAL · 8 IMPOSSIBLE.**

The 8 FULL rows are #7, 13, 14, 22, 23, 24, 25, 32 — and note what they have in common: every one
of them is either pure arithmetic or plain HTTP. **Nothing that touches the wrist itself is FULL.**
The 8 IMPOSSIBLE rows are #1, 2, 3, 4, 9, 11, 18, 19 — and note what *they* have in common: six of
the eight are haptics, and the other two are background execution.

---

## 2. What cannot be honest here

Three things in this document are **assertions from documentation and SDK headers only**, and the
owner has no Apple Watch and no iPhone. They cannot be verified on the test surface available.
They are listed here rather than buried, because two of them can kill the project on their own.

1. **Whether an Apple Watch app can capture microphone audio while it is not the frontmost app.**
   The watchOS SDK ships `AVAudioEngine`/`AVAudioRecorder`
   (`WatchOS26.2.sdk/System/Library/Frameworks/AVFAudio.framework/Headers/AVAudioRecorder.h`,
   `AVAudioEngine.h`), and `AVAudioSession` supports `Record`/`PlayAndRecord`
   (`.../AVAudioSession.h:250`). Nothing in the SDK says whether a `HKWorkoutSession`-backed
   background app may hold a recording session for hours. **UNVERIFIABLE WITHOUT HARDWARE.** The
   simulator will happily record and prove nothing — simulator audio has no power model, no
   `AVAudioSessionInterruption` from a real phone call, and no wrist-down state.
2. **Whether `WKInterfaceDevice.playHaptic(_:)` actually fires when the app is backgrounded but a
   workout/extended-runtime session is live.** The API
   (`.../WatchKit.framework/Headers/WKInterfaceDevice.h:104`) is documented as frontmost-dependent;
   the only documented non-frontmost haptic path is
   `WKExtendedRuntimeSession.notifyUserWithHaptic:repeatHandler:`, which its own header restricts to
   sessions "scheduled with `startAtDate:`" and running
   (`.../WatchKit.framework/Headers/WKExtendedRuntimeSession.h:148`).
   **UNVERIFIABLE WITHOUT HARDWARE.** The simulator plays no haptics at all.
3. **How the physical actuator renders any sequence of `playHaptic` calls.** Whether two
   `.click`s 170 ms apart are felt as two taps or one smear, whether the system rate-limits or
   coalesces them, and whether `.notification` is stronger than `.click` through a band — all of
   that is exactly the class of finding this repo already paid for once on Android
   ("the v0.1–v0.2.3 raw waveforms at 40 ms / amplitude 120–180 proved barely perceptible on a real
   Pixel Watch", `NudgeVocabulary.kt:108-114`). **UNVERIFIABLE WITHOUT HARDWARE**, and the whole
   haptic design depends on it.

### 2.1 The test surface that does exist

Verified on this machine, 2026-09-21:

- `xcodebuild -version` → **Xcode 26.3** (build 17C529).
- `xcodebuild -showsdks` → **watchOS 26.2** device SDK and **watchsimulator26.2** are installed.
- `xcrun simctl list devicetypes` → Apple Watch Series 9/10/11, SE 3, Ultra 2/3 device types exist.
- `xcrun simctl list runtimes` → **only `iOS 26.3` is installed. There is no watchOS simulator
  runtime.** Before anyone can launch a watchOS simulator at all, the watchOS 26.2 runtime must be
  downloaded (Xcode → Settings → Components).

So the achievable test surface is: **compile, unit-test, SwiftUI preview, and simulator UI
interaction.** Not haptics, not real HR, not background execution, not power. Write the watchOS app
so that every decision lives in pure, injectable code with the platform behind a protocol — exactly
the discipline the Wear app already follows (`apps/watch/wearApp/src/main/kotlin/app/gauge/wear/control/Ports.kt:11-117`)
— because unit tests on the simulator are the only proof available.

---

## 3. Decision ownership (unchanged by platform)

This table is the contract. A watchOS implementation that moves a row is not a port.

| Decision | Owner today | Citation |
|---|---|---|
| Is this 1 s window "voiced"? "loud"? | **Watch** (on-device DSP) | `shared/.../sentinel/SentinelDetector.kt:12`, `:82`, `:90` |
| Should an episode start / stop / cool down? | **Watch** | `shared/.../sentinel/SentinelStateMachine.kt:28-33` (`quietSecondsToStop = 30`, `cooldownSeconds = 10`) |
| Which nudge vector fired, at what level? | **Server** | `server/watch/vectors.py:37` (`YELLING_LEVELS`), `:50` (`HR_SPIKE_LEVELS`), `:63`, `:70` |
| Channel level 0..3, hysteresis, de-escalation | **Server** (authoritative), watch mirrors offline | `server/nudge_policy.py:143` (`cooldown_s = 20.0`), `:245-271`; mirror `shared/.../NudgeStateMachine.kt:86-131` |
| Hold-3 hysteresis before the ladder may climb | **Both** (must stay in lockstep) | `server/nudge_policy.py:62` (`HEAT_HOLD_S_DEFAULT = 3.0`), `shared/.../NudgeStateMachine.kt:23` |
| Tone/heat veto of an escalation | **Server** | `server/watch/heat_judge.py:191-193`, `:121` (`VALENCE_VETO_MAX = 0.48`) |
| Which code/emoji/rhythm a firing behaviour is | **Shared contract file** | `server/tests/fixtures/policy_vectors/nudge_vocabulary.json`, mirrored `shared/.../NudgeVocabulary.kt:127-285` |
| What the wrist physically plays, and when it repeats | **Watch** | `wearApp/.../haptics/HapticPatterns.kt:60`, `wearApp/.../haptics/HapticDirector.kt:113-146` |
| Proportional pulse train | **Watch only** (never needs the network) | `wearApp/.../haptics/PulseEngine.kt:11-40` |
| Positive cue rate limit (1 per 120 s across D/E/R) | **Shared** | `shared/.../NudgeVocabulary.kt:118` (`POSITIVE_CAP_S = 120.0`) |

Nothing on the watch side of this table may move to the server. The rationale is already written
down and is a product-defining constraint:
`docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md:20` — *"Never 'simplify' by moving
trigger detection server-side — round-trips, cold starts, and dead zones would kill the product's
defining feature."*

---

## 4. Haptics — the part that does not port

### 4.1 What the platform gives you

Read out of the installed SDK,
`WatchOS26.2.sdk/System/Library/Frameworks/WatchKit.framework/Headers/WKInterfaceDevice.h:16-33`:

```objc
typedef NS_ENUM(NSInteger, WKHapticType) {
    WKHapticTypeNotification,            // :17
    WKHapticTypeDirectionUp,             // :18
    WKHapticTypeDirectionDown,           // :19
    WKHapticTypeSuccess,                 // :20
    WKHapticTypeFailure,                 // :21
    WKHapticTypeRetry,                   // :22
    WKHapticTypeStart,                   // :23
    WKHapticTypeStop,                    // :24
    WKHapticTypeClick,                   // :25
    // "can only be used while the app has an active navigation session running"  :26
    WKHapticTypeNavigationLeftTurn,      // :27
    WKHapticTypeNavigationRightTurn,     // :28
    WKHapticTypeNavigationGenericManeuver, // :29
    // "can only be used while the app has an active underwater depth session running"  :30
    WKHapticTypeUnderwaterDepthPrompt,          // :31
    WKHapticTypeUnderwaterDepthCriticalPrompt,  // :32
};
```

The only playback API is `- (void)playHaptic:(WKHapticType)type;`
(`WKInterfaceDevice.h:104`). **There is no duration parameter, no amplitude parameter, no
timings/amplitudes array, no composition builder, and no support-probing call.**

And, decisively: **`CoreHaptics.framework` is not present in the watchOS SDK.** A full listing of
`WatchOS26.2.sdk/System/Library/Frameworks/` contains `AVFAudio`, `CoreMotion`, `HealthKit`,
`SoundAnalysis`, `WatchConnectivity`, `WatchKit`, `WidgetKit`, `ClockKit`, `BackgroundTasks`,
`CoreML`, `Network` — and **no `CoreHaptics`, no `AudioToolbox`, no `Speech`, no `SensorKit`**. So
`CHHapticPattern` / `CHHapticEngine` — the iOS API that *could* have expressed these waveforms — is
unavailable on the wrist.

**9 usable haptic types** (the 3 navigation ones require an active navigation session; the 2
underwater ones require a depth session). Against them, the Wear app currently plays:

| Wear OS capability | Where | watchOS equivalent |
|---|---|---|
| `VibrationEffect.createWaveform(timings, amplitudes, -1)` — arbitrary rhythm + per-segment amplitude | `wearApp/.../haptics/RealVibratorPort.kt:32-34` | **none** |
| `VibrationEffect.createPredefined(EFFECT_CLICK / EFFECT_HEAVY_CLICK)` — OEM-tuned | `RealVibratorPort.kt:38-46` | `playHaptic(.click)` — nearest, but no `HEAVY_CLICK` analogue and no second tier |
| `VibrationEffect.Composition` + `PRIMITIVE_CLICK`, N clicks with explicit `delay` | `RealVibratorPort.kt:48-60` | **none** — you must issue N separate `playHaptic` calls with your own `Task.sleep`, whose real-world rendering is UNVERIFIABLE WITHOUT HARDWARE |
| `vibrator.hasAmplitudeControl()` | `RealVibratorPort.kt:36` | **none** — no amplitude exists to control |
| `areAllEffectsSupported(...)` / `areAllPrimitivesSupported(...)` honest support probing | `RealVibratorPort.kt:43`, `:49` | **none** — `playHaptic` returns `Void`; there is no way to know whether anything played |

That last row deletes a whole safety mechanism. `HapticDirector` today plays predefined-first and
falls back to a documented raw waveform when the device says no, and it *reports which physical path
a cue took* to telemetry so the owner can confirm remotely that the OEM path is landing
(`wearApp/.../haptics/HapticDirector.kt:214-264`, path labels `"predefined"` / `"composed"` /
`"waveform"` / `"waveform-fallback"` / `"waveform-failed"` / `"unplayable"` at `:225-244`). On
watchOS there is exactly one path and no observability of it. **Do not port `reportHapticPath`;
there is nothing for it to report.** What *can* be logged is "we asked for N taps of type T at time
t" — which is worth logging, since it is the only evidence that will ever exist.

### 4.2 The nearest honest mapping

The shipped channel-A ladder, from `HapticPatterns.cue("A", level)`
(`wearApp/.../haptics/HapticPatterns.kt:69-89`), with tap counts read from the shared schedule
(`shared/.../NudgeHapticSchedule.kt:95-100`):

| Level | Wear OS cue | Fallback waveform (= the **H** vocabulary entry) | Proposed watchOS cue |
|---|---|---|---|
| 0 | silence | — | silence (`HapticDirector.kt:69-80` — level 0 is silent *and* clears the reminder) |
| 1 | `Predefined(CLICK)` — one OEM soft click (`HapticPatterns.kt:74`) | `[0, 200 ms] @ [0, 255]` (`NudgeVocabulary.kt:145`) | `playHaptic(.click)` × 1 |
| 2 | `ComposedClicks(count = 2, gapMs = 170)` (`HapticPatterns.kt:77`) | `[0,70,170,150] @ [0,210,0,255]` — a **rising** ramp (`NudgeVocabulary.kt:146`) | `playHaptic(.click)` × 2, ~170 ms apart |
| 3 | `ComposedClicks(count = 3, gapMs = 170)` | `[0,60,170,110,170,200] @ [0,200,0,230,0,255]` — a 3-step rising ramp (`NudgeVocabulary.kt:147`) | `playHaptic(.notification)` then `.click` × 2, or `.click` × 3 |

`MIN_GAP_MS = 170` is a hard, pattern-wide never-merge floor on Wear
(`HapticPatterns.kt:65`, mirrored in `NudgeVocabulary.kt:106`). Keep 170 ms as the target inter-tap
gap on watchOS, but understand the difference: on Wear the **platform** owns the intra-pattern
timing because the whole thing is one `VibrationEffect.Composition`
(`HapticPatterns.kt:20-22`); on watchOS **your app** owns it, on a `Task` you have to keep alive,
and if the app is suspended mid-sequence the wearer feels a truncated cue. There is no atomic
multi-tap primitive.

**What survives:** level is still a tap *count* (1 / 2 / 3), which is the property Brown & Brewster
found identifiable ~93% of the time and which the vocabulary explicitly leans on
(`NudgeVocabulary.kt:18-24`).
**What is lost:** the ramp. L2 and L3 are currently *rising* gestures — the rise is encoded in tap
LENGTH (60 → 110 → 200 ms) precisely so a phone that cannot vary amplitude still feels the shape
(`NudgeVocabulary.kt:142-143`, `HapticPatterns.kt:97-101`). watchOS can vary neither length nor
amplitude, so L2 and L3 become "two identical taps" and "three identical taps". That is a real
downgrade, not a cosmetic one: "the cue builds" was the design.

### 4.3 The vocabulary does not port, code by code

`NudgeVocabulary.ALL` (`shared/.../NudgeVocabulary.kt:127-285`) defines 8 codes, each with a
distinct rhythm, so *the wrist says WHICH behaviour, not just how bad*
(`HapticPatterns.kt:108-117`). Here is every one of them against the watchOS enum. "Distinct?" means
"can a wearer tell this apart from the others by feel alone on watchOS?"

| Code | Name | Wear rhythm (cited) | What makes it that rhythm | Nearest watchOS | Distinct? |
|---|---|---|---|---|---|
| **H** | Heated 📈 | rising `•`, `• •`, `• • •` | tap count + rising amplitude/length (`NudgeVocabulary.kt:144-148`) | `.click` ×1/×2/×3 | count only — **ramp lost** |
| **D** | De-escalated 📉 | falling ramp `[0,200,170,110,170,60] @ [0,230,0,200,0,180]` — "the only cue that fades" (`NudgeVocabulary.kt:163-166`) | *decreasing* tap length + amplitude | nothing fades on watchOS | **NO** |
| **C** | Cut in ✂️ | `• —`, `• • —`, `• • • —` — short tap(s) then one long 300 ms "stop" buzz (`NudgeVocabulary.kt:182-189`) | contrast between a 60 ms tap and a 300 ms buzz | no long buzz exists | **NO** |
| **A** | Hogging 🎤 | slow `— — —`, 250 ms buzzes 300 ms apart, amplitude 180 — deliberately *unhurried*, "not an emergency" (`NudgeVocabulary.kt:204-210`) | long, soft, slow | `.click` spaced 300 ms reads as urgent taps, not a slow swell | **NO** |
| **E** | Listened 👂 | soft `••` = `[0,60,60,60,170,60,60,60] @ [0,120,190,120,0,120,190,120]` — two *swells* (`NudgeVocabulary.kt:225-229`) | consecutive vibrating segments = one continuous buzz whose strength changes (`NudgeVocabulary.kt:38-42`) | `.success` ×1 | partially — see §4.5 |
| **R** | Repair 🤝 | the same swell, three times (`NudgeVocabulary.kt:244-246`) | count 3 of the E swell | `.success` ×1 (E and R must then be the same cue) | **NO** (E vs R) |
| **K** | Calm streak 🧘 | **never buzzes, ever** (`NudgeVocabulary.kt:261-262`, `haptic = null`) | absence | absence | **FULL** |
| **P** | Pulse ❤️ | lub-dub: 90 ms then 150 ms, **100 ms apart — deliberately under the 170 ms merge floor, "because the near-merge is the heartbeat"** (`NudgeVocabulary.kt:120-122`, `:277-282`) | a sub-floor gap between two unequal taps | two `.click`s as fast as the API will fire them | **NO** — the heartbeat *is* the timing |

So of 8 codes, exactly **one** (K, silence) ports at full fidelity, one (H) ports as a count without
its ramp, one (E) ports approximately, and **five do not port at all**. The file's own framing —
*"a threshold change must never silently restyle a cue, and a restyle must never move a threshold"*
(`NudgeVocabulary.kt:14-16`) — means this cannot be papered over by quietly reusing H's cue for
everything. **The watchOS build must either (a) ship with the vocabulary explicitly reduced to
level-only cues and say so on screen, or (b) get its own `nudge_vocabulary` lane in the contract
file with a documented watchOS column.** Option (b) is the honest one and is the recommendation:
add a `watchos` key to `server/tests/fixtures/policy_vectors/nudge_vocabulary.json` so the reduced
mapping is pinned by the same cross-runtime vector test that pins the Android one
(`apps/watch/shared/src/jvmTest/kotlin/app/gauge/shared/NudgeVocabularyVectorsTest.kt`).

Note also `ICON-LEGEND.md:138-139`, which already says the ramps are only reinforced by amplitude
*on the watch*: **"Test both: they do not feel the same, and only the watch reinforces the ramps
with amplitude."** On Apple Watch, neither surface reinforces them.

### 4.4 Two channels, one enum

Channel A ("you") is crisp clicks; channel B ("partner / paired cue") is long smooth buzzes of
250–400 ms and **deliberately never uses predefined click effects**, so that *"the two channels must
stay distinguishable by feel alone"* (`HapticPatterns.kt:38-48`, cues at `:79-86`). The routing rule
is enforced in code: a code arriving on the wrong lane plays that lane's generic cue rather than its
own rhythm, and each code's home lane is B for the watch-only ones (P: heart rate) and A for the rest
(`HapticPatterns.kt:143-155`).

watchOS has no long smooth buzz. The only remotely "different texture" options in the enum are
`.notification`, `.directionUp`/`.directionDown`, `.start`/`.stop`, `.retry`, `.failure`. A
two-channel distinction could be *attempted* as, say, A = `.click` family and B = `.directionUp`
family — but whether a wearer can actually tell those apart on the wrist is
**UNVERIFIABLE WITHOUT HARDWARE**, and the design premise (texture, not identity) is gone. Rated
IMPOSSIBLE.

### 4.5 Positives and silence-by-contract

Two rules here port cleanly and must be preserved:

- **K never buzzes.** `HapticPatterns.isSilentByContract` (`HapticPatterns.kt:134-141`) checks the
  vocabulary's `haptic == null` and returns no cue, *distinct from* "this code has no override".
  Port this check verbatim. **FULL.**
- **A positive never arms the reminder and never moves a level.** `playPositive`
  (`HapticDirector.kt:190-196`) deliberately bypasses `onNudge` entirely: *"a wrist that repeats
  'well done' every two minutes is worse than one that never said it"* (`:180-189`). Same for the
  server-side `PositiveEvent`, which carries no channel and no level
  (`shared/.../WireModels.kt:40-55`). Port this separation verbatim.
- **At most one positive per 120 s across D/E/R together** — `PositiveNudgeGate`
  (`NudgeVocabulary.kt:323-338`), `POSITIVE_CAP_S = 120.0` (`:118`). Pure code; ports unchanged.
- Positives are also suppressed while the pulse train is covering channel A
  (`wearApp/.../control/SentinelController.kt:1038-1041`) — moot on watchOS, since there is no pulse
  train (§6).

`.success` is the obvious watchOS cue for a positive and is at least *categorically* different from
`.click`/`.notification`, so praise will not read as an alarm. But D, E and R then collapse to one
cue, and the screen has to carry which good thing happened — which the vocabulary already
anticipates for E and R (`NudgeVocabulary.kt:225-226`: *"the wrist says 'that was good', the screen
says which good thing"*) but **not** for D, which has its own falling ramp and its own flash text
"Nice recovery" (`NudgeVocabulary.kt:160-166`). Rated PARTIAL.

---

## 5. The reminder ladder and the back-off

### 5.1 What it does today

PRD §6's repeat schedule, encoded once in `shared/.../NudgeHapticSchedule.kt`:

| Constant | Value | Citation |
|---|---|---|
| `NO_HAPTIC_ABOVE_SCORE` | 70 | `NudgeHapticSchedule.kt:59` |
| `LEVEL_1_MIN_SCORE` | 50 | `:61` |
| `LEVEL_2_MIN_SCORE` | 30 | `:63` |
| `LEVEL_1_REPEAT_MS` | 120 000 (2 min) | `:66` |
| `LEVEL_2_REPEAT_MS` | 60 000 (1 min) | `:67` |
| `LEVEL_3_REPEAT_MS` | 10 000 (10 s) | `:68` |
| `REMINDER_BACKOFF_CAP_MS` | = `LEVEL_1_REPEAT_MS` = 120 000 | `:107` |
| `BACKOFF_MAX_DOUBLINGS` | 16 | `:151` |
| `DOUBLE_RAMP` | `[210, 255]` | `:80` |
| `ESCALATING_RAMP` | `[200, 230, 255]` | `:73` |

`reminderIntervalMs(level, repeatIndex)` doubles the base cadence per repeat, capped
(`:126-135`), so level 3 fires at 10 s, 20 s, 40 s, 80 s, then settles at 120 s. The reason is
measured, not stylistic: flat repetition *"buzzed **18 times in three minutes** and never stopped…
The same three minutes now cost 5 buzzes rather than 18"* (`:109-124`). A **new** escalation resets
`reminderRepeats` so a genuinely worsening situation is reported promptly again
(`HapticDirector.kt:98-103`, reset at `:102`).

Driving and clearing:
- `dueReminder()` is a pure check with no side effects; the caller decides whether to play
  (`HapticDirector.kt:113-119`).
- `replayReminder()` bypasses the 5 s dedupe and rebases the clock (`:128-138`).
- `clearReminder()` on episode end / disarm — *"a watch that is no longer listening must not keep
  buzzing about a conversation that's over"* (`:140-146`), called from
  `SentinelController.endEpisodeStream` (`SentinelController.kt:970`).
- A reminder replays the **same** vocabulary code the level was raised by, not a generic buzz
  (`HapticDirector.kt:54-59`, `:130`).
- The 5 s duplicate-suppression window on fresh nudges: `(now - lastVibrationTimeMs) < 5000L`
  (`HapticDirector.kt:83-85`).
- Reminders are channel A only; channel B stays single-shot (`HapticDirector.kt:45-52`,
  `REMINDER_CHANNEL = "A"` at `:269-270`).

### 5.2 On watchOS

All of the *scheduling logic* is pure and ports unchanged — `NudgeHapticSchedule` has no platform
dependency and is exactly the kind of file that should move into the shared module's Apple target
(§18). What does not port is the ability to *fire* on that schedule.

- **Frontmost:** a `Timer`/`Task.sleep` loop calling `playHaptic` reproduces the ladder exactly.
- **Not frontmost:** the only documented repeating-haptic API is
  `WKExtendedRuntimeSession.notifyUserWithHaptic:repeatHandler:`, and its header imposes hard
  limits that break this ladder:
  - the repeat interval *"must be > 0.0 and <= 60.0"* seconds
    (`WKExtendedRuntimeSession.h:142`) — so **level 1's 2-minute cadence and the 2-minute back-off
    cap are both unrepresentable**;
  - *"If the app is not active, this will result in a system alert UI"* (`:144`) — a full-screen
    system alarm, not a silent wrist tap;
  - it continues until the user taps a stop button or the app invalidates the session (`:145-146`);
  - and it *"can only be called on a WKExtendedRuntimeSession that was scheduled with `startAtDate:`
    and currently has a state of Running. If it is called outside that time, it will be ignored."*
    (`:148`) — i.e. it is an **alarm** facility, scheduled in advance, not a reactive one.

  `startAtDate:` itself is alarm-mode-only and must be called while foreground
  (`WKExtendedRuntimeSession.h:126`, error `MustBeActiveToStartOrSchedule = 3` at `:58`), and
  sessions cannot be scheduled more than 36 hours ahead (`:57`).

So: the back-off ladder is reproducible while the app is on screen, and reduces to "a local
notification, or a system alarm alert" otherwise. Rated PARTIAL. See §7.3.

---

## 6. The pulse train

### 6.1 What it does today

While the wearer's own volume is over their mode's trigger threshold, the watch taps out one short
pulse per configured interval on channel A, with duration **and** amplitude scaling by how far over
threshold the window is (`wearApp/.../haptics/PulseEngine.kt:11-40`).

Bands — `HapticPatterns.pulseBandFor(dbOverThreshold)` (`HapticPatterns.kt:163-168`):

| dB over threshold | duration | amplitude |
|---|---|---|
| < 3.0 | 50 ms | 180 |
| < 6.0 | 60 ms | 205 |
| < 9.0 | 70 ms | 230 |
| ≥ 9.0 | 80 ms | 255 |

- Never-merge silence floor between pulses: `SILENCE_FLOOR_MS = 170`
  (`PulseEngine.kt:97`), and the effective interval is
  `max(configuredIntervalMs, pulse.durationMs + 170)` (`PulseEngine.kt:115-116`).
- Configured cadence comes from a live preference supplier: `"250"` / `"500"` / `"1000"` / `"off"`
  (`wearApp/.../ui/SettingsScreen.kt:191-194`), and `250 == 80 + 170` is the documented tightest
  legal value (`HapticPatterns.kt:56-58`).
- The controller default is 250 ms (`SentinelController.kt:1159`) but the **shipped preference
  default is `"off"` since 2026-09-10, measured** — at 250 ms the train delivered *"19 buzzes/min on
  the owner's family recording and 28/min on `poker6_real` — a poker game, not an argument"*
  (`wearApp/.../prefs/GaugePrefs.kt:17-33`).
- Amplitude degradation is honest: no amplitude control → a fixed 255
  (`HapticDirector.kt:205-208`, `FIXED_AMPLITUDE_NO_CONTROL = 255` at `:267`).
- The physical repeat runs on a separate main-looper `Handler` because the mic thread blocks inside
  `AudioRecord.read()` for a whole second (`SentinelService.kt:117-128`), guarded by a
  generation-token gate against `removeCallbacks` races (`haptics/PulseChainGate.kt:5-34`) and a
  pure restart decision (`haptics/PulseChainDecision.kt:42-54`).
- While the train is actively covering channel A, fresh nudges, reminders **and** positives are all
  suppressed (`SentinelController.kt:922-925`, `:938-948`, `:1038-1041`), "actively covering" being
  within 2× the last effective interval (`:960-964`).

### 6.2 The ARMED shout-tap

A single band-1 tap when the wearer goes loud while merely ARMED, before any episode opens — *"so
the product is felt before the two-consecutive-loud-window trigger ever fires"*
(`haptics/ShoutTapGate.kt:5-7`). Always the floor band, never proportional
(`HapticPatterns.ARMED_SHOUT_TAP` = `pulseBandFor(0.0)` = 50 ms @ 180, `HapticPatterns.kt:170-172`).
Rate-limited to one per **2 000 ms** (`ShoutTapGate.kt:22`), and a suppressed call does *not* restart
the clock (`:26-30`); the fail direction is silence (`:16-20`). Played as one direct `playPulse`,
never a chain (`Ports.kt:168-174`, `SentinelService.kt:264-268`).

### 6.3 On watchOS

**The pulse train is IMPOSSIBLE.** Every axis it varies — duration (50/60/70/80 ms) and amplitude
(180/205/230/255) — is a parameter `playHaptic` does not have. All four bands collapse to the same
`.click`, which means the train stops being *proportional* and becomes a metronome — precisely the
failure mode the 2026-09-10 measurement already rejected on Android
(`GaugePrefs.kt:22-27`). Compounding it, the train's whole point is firing every 250 ms, and whether
watchOS will even render four `playHaptic` calls per second (rather than coalescing or
rate-limiting them) is **UNVERIFIABLE WITHOUT HARDWARE**.

**Recommendation: do not build the pulse train on watchOS.** Ship it absent, not broken. The
shipped Android default is already `"off"`, so this costs nothing the owner is currently using, and
it simplifies the port considerably: `PulseEngine`, `PulseChainGate`, `PulseChainDecision`, the
separate pulse `Handler`, and every "is the train covering channel A" suppression check
(`SentinelController.kt:960-964` and its three call sites) all disappear.

**The ARMED shout-tap, by contrast, should port** — it is one tap, always the same band, gated at
2 s. It becomes `playHaptic(.click)` behind the ported `ShoutTapGate` (pure, 30 lines, keep
verbatim). Rated PARTIAL only because the tap is no longer the soft floor-band tap it was designed
to be; on watchOS it is indistinguishable from a level-1 nudge.

---

## 7. Execution model — the real blocker

### 7.1 What the Wear app is

A **foreground service of type `microphone`**:
`AndroidManifest.xml:38-41` (`android:foregroundServiceType="microphone"`), started with
`startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)`
(`SentinelService.kt:1026-1030`), holding `FOREGROUND_SERVICE_MICROPHONE` and `WAKE_LOCK`
(`AndroidManifest.xml:10-11`).

It runs a dedicated `HandlerThread` whose tick is **paced by a blocking 1-second mic read**
(`SentinelService.kt:199-300`; the loop re-posts undelayed at `:279` because
`MicReader.readWindow()` blocks for the window). It stays up for the **entire armed session**, which
is intended to be all day: the duty cycle exists precisely because *"most of a wearer's day is
silence the sentinel has nothing to do with"* (`shared/.../signals/MicDutyCycle.kt:9-12`), and the
deep tier engages after **5 minutes** of quiet plus **10 minutes** of stillness
(`MicDutyCycle.kt:44-49`, `STILL_THRESHOLD_MS` at `:141`).

That is: an app that holds the microphone, in the background, indefinitely, for hours.

### 7.2 What watchOS offers instead

| Mechanism | Duration | Mic? | Notes |
|---|---|---|---|
| Frontmost app | while on screen; screen sleeps in ~15–70 s | yes | the only unambiguous state |
| `HKWorkoutSession` (+ `HKLiveWorkoutBuilder`) | long / open-ended | **UNVERIFIABLE** (§2.1) | `HealthKit.framework/Headers/HKWorkoutSession.h:55-56`; needs the workout-processing background mode, HealthKit entitlement, and **records a workout in the user's Health app** |
| `WKExtendedRuntimeSession` — mindfulness / self-care / physical-therapy | ~1 h per session | no documented mic guarantee | `WKExtendedRuntimeSession.h:76-84`; invalidation reason `ResignedFrontmost` = *"The application has lost frontmost status, so the session ended"* (`:39-45`) |
| `WKExtendedRuntimeSession` — alarm (`startAtDate:`) | seconds, at a scheduled time | no | `:126`; the only non-frontmost haptic path (§5.2) |
| `WKApplicationRefreshBackgroundTask` + `scheduleBackgroundRefreshWithPreferredDate:` | seconds of CPU, roughly hourly, system-budgeted | no | `WatchKit.framework/Headers/WKBackgroundTask.h:58`, `:128` |
| `BGTaskScheduler` | opportunistic | no | `BackgroundTasks.framework` is present |
| `WKURLSessionRefreshBackgroundTask` | wakes on a background `URLSession` completion | no | `WKBackgroundTask.h:88` — useful for uploads (§13) |

So the all-day armed sentinel has **no watchOS equivalent**. The honest shapes for a watchOS build
are, in order of decreasing product fidelity:

1. **A session the user starts** — "Coach this conversation" → the app goes frontmost, starts an
   `HKWorkoutSession` (activity type `.mindAndBody` or `.other`), captures, streams, buzzes, and
   ends when the user ends it. Bounded, explicit, and the user sees a workout in Health. This is the
   closest thing to `Mode.SESSION` (`shared/.../sentinel/ModePolicy.kt:37`: `continuous = true`,
   no trigger bar) and is the recommended v1.
2. **COMPANION mode** — the iPhone listens, the watch only receives relayed nudges (§10.3). No mic
   on the wrist at all, which removes the hardest constraint. The Wear app already has this mode
   and it already works over the same socket.
3. **All-day ARMED / BATTERY_SAVER / duty-cycled sentinel** — **do not attempt.** Rated IMPOSSIBLE.

### 7.3 Can the wrist buzz when the app is not frontmost?

This is the question the product lives or dies on, and the answer from the SDK is: **not in the
general case.**

- `playHaptic` has no documented background guarantee (`WKInterfaceDevice.h:104`).
- `notifyUserWithHaptic:` works only inside a *scheduled alarm* session, caps its repeat at 60 s,
  and *"if the app is not active, this will result in a system alert UI"*
  (`WKExtendedRuntimeSession.h:142-148`).
- A `UNNotificationRequest` (UserNotifications.framework is present) will tap the wrist — with the
  **system notification haptic**, not ours, and subject to the wearer's notification settings and
  Focus modes.
- Whether `playHaptic` fires during a live `HKWorkoutSession` while the screen is off and the wrist
  is down is **UNVERIFIABLE WITHOUT HARDWARE** and is the single most important thing to test the
  day an Apple Watch arrives. Write that test first.

Wrist-down specifically: the screen is off, the app is not frontmost by any normal definition, and
the watch is in its lowest-power state. Assume nothing works there until measured.

### 7.4 The "ongoing" notification

The Wear app shows a low-importance, ongoing notification for the whole armed session —
`NotificationChannel(CHANNEL_ID, "Gauge Sentinel", IMPORTANCE_LOW)` (`SentinelService.kt:1051-1054`),
`.setOngoing(true)` (`:1045`), text from the pure `notificationText` helper
(`wearApp/.../service/NotificationText.kt`). This is both a legal requirement of the foreground
service and the wearer's visible assurance that the mic is live.

watchOS has **no persistent-notification concept**. A `UNNotificationRequest` is a transient alert.
The nearest equivalents are (a) the app being frontmost, (b) the green/orange mic indicator the
system shows for an active recording session, and (c) a WidgetKit complication showing "On" (§15.2).
None of them is an ongoing notification. Rated IMPOSSIBLE — and note this is a **trust** regression,
not just a technical one: the wearer loses the persistent "this is recording" affordance.

---

## 8. Heart rate

### 8.1 What the Wear app does

`HrSource` (`wearApp/.../sensors/HrSource.kt`) wraps Wear **Health Services**'
`MeasureClient.registerMeasureCallback(DataType.HEART_RATE_BPM, callback)`
(`HrSource.kt:224-225`), an explicit **live measurement stream** — the same API a workout app uses,
available while merely armed. Key behaviours:

- Availability is a first-class signal: `onAvailabilityChanged` maps `ACQUIRING` / `AVAILABLE` /
  `UNAVAILABLE` / `UNAVAILABLE_DEVICE_OFF_BODY` / `UNKNOWN`
  (`HrSource.kt:141-164`), and a non-`AVAILABLE` transition **clears the last bpm** so the UI never
  shows a frozen number next to an "off-body" caption (`:151-160`).
- `latest()` returns `null` until a real sample lands — *"never a fabricated 0.0 placeholder"*
  (`:36-37`, `:180`).
- The combine point structurally refuses to publish an HR meter value while availability is
  `ACQUIRING` / `OFF_BODY` / `UNAVAILABLE` (`control/Ports.kt:140-149`).
- Registration is async and carefully sequenced: `unregisterMeasureCallbackAsync` only tells the
  lifecycle gate about the stop from `FutureCallback.onSuccess` (`HrSource.kt:280-297`), with
  `PendingStopIntent` guarding a second concurrent unregister and replaying a `start()` that arrived
  during the pending window (`:121`, `:189-202`, `:293-296`).
- Churn detection: `SensorLifecycleGate` dedupes duplicate start/stop and warns on >N transitions in
  5 s (`HrSource.kt:233-235`) — added because live telemetry showed register/unregister pairs
  landing inside 1 s during the preview↔service handoff (`:56-65`).
- Demand is exactly two things: HR is the picker-selected signal, **or** an episode is STREAMING and
  the `hr_spike` vector is subscribed (`sensors/HrDemandPolicy.kt:39-40`). Release grace is
  **30 000 ms** (`HrDemandPolicy.kt:26`), and the fail direction is "stay registered"
  (`:51-53`).
- HR rides the episode socket as `{"type":"hr","bpm":…,"t":…}` (`net/EpisodeWsClient.kt:160-168`),
  every window while STREAMING (`SentinelController.kt:557-568`) and every **5 000 ms** in COMPANION
  mode (`COMPANION_HR_INTERVAL_MS`, `SentinelController.kt:1168`, gate at `:615`).
- The server ladder is `HR_SPIKE_LEVELS = ((35.0,3),(25.0,2),(15.0,1))` bpm over resting
  (`server/watch/vectors.py:50`), default resting 65.0 (`:48`), and `hr_spike` maps to vocabulary
  code **P**, whose home channel is **B** because it is the one `watchOnly` code
  (`NudgeVocabulary.kt:264-283`, `watchOnly = true` at `:275`; lane rule
  `HapticPatterns.kt:148-153`).
- On-device, `HrTracker` keeps a rolling-median baseline over the last 60 readings with an
  anti-poisoning rule and `+15.0` bpm "over" bar (`shared/.../signals/HrTracker.kt:22`,
  `MIN_READINGS_FOR_BASELINE = 5` at `:56`).

### 8.2 What watchOS gives you, and what it costs

`HealthKit.framework` is present, including `HKWorkoutSession.h`, `HKLiveWorkoutBuilder.h`,
`HKLiveWorkoutDataSource.h`. `CoreMotion` additionally ships `CMHighFrequencyHeartRateData.h` and
`CMBatchedSensorManager` (`CoreMotion.framework/Headers/CMBatchedSensorManager.h:20-21`,
watchOS 10+).

| Wear behaviour | watchOS equivalent | Verdict |
|---|---|---|
| Live HR stream while merely armed, no workout | **none** | Outside a workout, HealthKit delivers HR samples sporadically (minutes apart). A ~1 Hz stream requires an active `HKWorkoutSession` + `HKLiveWorkoutBuilder`, or `CMBatchedSensorManager` high-frequency HR (also workout-gated). |
| `MeasureClient` register / unregister | `HKAnchoredObjectQuery` with an `updateHandler`, or `HKLiveWorkoutBuilder` delegate callbacks | start/stop map cleanly onto `healthStore.execute/stop(query)` |
| `onAvailabilityChanged` → ACQUIRING / AVAILABLE / UNAVAILABLE / **OFF_BODY** | **no equivalent.** HealthKit gives you samples or nothing. There is no off-body state, no acquiring state, no "registered but not producing" signal. | The `SignalAvailability` enum (`shared/.../signals/SignalAvailability.kt`) must degrade to `UNKNOWN`/`AVAILABLE` on watchOS, and the honest UI caption "off body" disappears. |
| Registration success/failure callbacks (`onRegistered`, `onRegistrationFailed`) | `HKHealthStore.execute` + query `resultsHandler` error | comparable |
| `BODY_SENSORS` permission | `HKHealthStore.requestAuthorization(toShare:read:)` with `NSHealthShareUsageDescription` | comparable, but **HealthKit read authorization is deliberately opaque** — a denied read is indistinguishable from "no data", so the "permission denied" diagnostic path (`SentinelService.kt:106-115`) cannot be reproduced honestly |

The workout requirement is a product decision, not a detail: to get the HR cadence this app already
depends on, the watchOS app must **start a workout the user can see in their Health app**. That
changes what the product *is* on Apple Watch. Rated PARTIAL.

### 8.3 What ports unchanged

- `HrTracker` (rolling median, anti-poisoning, `+15` bar) — pure Kotlin, no platform deps.
  **FULL.**
- `HrDemandPolicy` (30 s grace, fail-toward-registered) — pure. **FULL.** The two demand conditions
  map directly: "HR is the selected signal" and "STREAMING && hr_spike subscribed"
  (`HrDemandPolicy.kt:39-40`).
- `SensorLifecycleGate` + `PendingStopIntent` — pure; still worth keeping, because
  `HKHealthStore.stop(query)` is also effectively asynchronous.
- The server-side `hr_spike` ladder and the P vocabulary entry — server-side, unchanged.

---

## 9. Audio capture and streaming

### 9.1 What the Wear app captures

`MicReader` (`wearApp/.../service/MicReader.kt`):

| Parameter | Value | Citation |
|---|---|---|
| Sample rate | 16 000 Hz | `MicReader.kt:149` |
| Channels | mono (`CHANNEL_IN_MONO`) | `:48` |
| Encoding | `ENCODING_PCM_16BIT` | `:49` |
| Window | 16 000 samples = exactly 1 second | `:150` |
| Source | `MediaRecorder.AudioSource.MIC` | `:47` |
| Buffer | ≥ 2 full windows of headroom | `:45` |
| `readWindow()` | blocking, loops until the window is full; `null` = mic gone | `:73-83` |
| `pause()` / `resume()` | `AudioRecord.stop()` without release, then `startRecording()` — the duty-cycle OFF/ON edges | `:121-146` |

`readWindow()` returning `null` is treated as **mic loss → auto-disarm** by the controller
(`MicReader.kt:106-108`), which is why nothing may call it between `pause()` and `resume()`.

### 9.2 On watchOS

`AVAudioEngine` with a tap on `inputNode` is the equivalent (`AVFAudio.framework/Headers/
AVAudioEngine.h`, `AVAudioIONode.h`), or `AVAudioRecorder` for file-based capture
(`AVAudioRecorder.h`). `AVAudioSession` supports the `Record` and `PlayAndRecord` categories
(`AVAudioSession.h:250`). `WKInterfaceDevice.supportsAudioStreaming` reports whether the device can
stream audio at all — *"supported on Apple Watch Series 3 and later"*
(`WKInterfaceDevice.h` comment above the property).

Differences that matter:

- **The hardware format is not 16 kHz.** The input node's native format will be 44.1/48 kHz float.
  You must run an `AVAudioConverter` (`AVAudioConverter.h` is present) to reach 16 kHz mono
  PCM16 — and it must be exactly that, because the server derives dB from the raw bytes and expects
  16 kHz PCM16LE (§9.3).
- **There is no `pause()` that keeps the object alive.** `AVAudioEngine.pause()`/`stop()` exist, but
  the duty-cycle's power model ("the platform stops delivering and powering audio during an OFF
  phase", `MicReader.kt:99-101`) is not something you can verify.
  **UNVERIFIABLE WITHOUT HARDWARE.**
- **`NSMicrophoneUsageDescription` is required**, and permission is requested via
  `AVAudioApplication.requestRecordPermission` (`AVAudioApplication.h`).
- **Interruptions are a first-class concern** the Wear app does not have. An incoming call or Siri
  will interrupt the session; `AVAudioSession` explicitly refuses a `Record`/`PlayAndRecord`
  activation while another app hosts a call, failing with
  `AVAudioSessionErrorCodeInsufficientPriority` (`AVAudioSession.h:247-252`). The watchOS port needs
  an interruption handler with a documented policy; today's `null`-means-mic-loss auto-disarm
  (`SentinelController.tick`) is a reasonable model to reuse.
- **Background capture: see §2.1 item 1 and §7.** This is the blocker.

Rated PARTIAL: capture itself is straightforward; capture *when the app is not on screen* is not.

### 9.3 Streaming PCM to the server

The socket contract (verified against both ends):

- URL: `{baseWsUrl}/ws/live-session/{episodeId}?token={urlencoded deviceToken}`
  (`net/EpisodeWsClient.kt:80-87`), server route `server/watch/routers/ws.py:212`, params parsed at
  `:216-217`, auth failure closes with **1008** before accept (`:219-223`).
  Auth is in the **query string**, not a header — `server/watch/auth.py:187-199` synthesizes
  `Authorization: Bearer <token>` internally. Convenience, not a constraint
  (`auth.py:194-196` notes an OkHttp WS *could* set headers), and it ports to
  `URLSessionWebSocketTask` unchanged.
- **Audio is sent as raw PCM16LE in binary WebSocket frames, with no envelope and no header.**
  Client: `EpisodeWsClient.kt:155-158`. Server: `ws.py:345-368`, which buffers the bytes and derives
  the dB series **server-side** (`ws.py:361`, `_rms_dbfs` at `:118-130`).
  The watch sends **no derived features at all** over this socket — no dB, no voiced flag, no F0.
- Chunk size: `MAX_SLICE_BYTES = 32 000` bytes = exactly 1.0 s of 16 kHz mono PCM16
  (`SentinelController.kt:1153`), slicing loop `:1129-1135`, steady-state send `:844`.
- Server retention cap: `MAX_LIVE_SESSION_PCM_BYTES = 57_600_000` (30 min @ 32 KB/s),
  `ws.py:84` — past the cap detection continues but audio is no longer retained (`:352-360`).
- Other watch→server frames: `{"type":"hr","bpm":…,"t":…}` (`EpisodeWsClient.kt:160-168`),
  `{"type":"end"}` (`:170-173`), `{"type":"companion"}` (`:175-180`),
  `{"type":"heartbeat"}` (`:182-187`). The `hr.t` value is **ignored** by the server, which
  re-stamps on its own clock (`ws.py:404`, `:409-414`).
- Server→watch frames: `vector_event`, `nudge`, `positive`, `companion_ack`, `live_session_saved`,
  `error` (`ws.py:261`, `:271`, `:281`, `:391`, `:434-438`/`:453-457`/`:481-485`, `:377`/`:489`).
  Client dispatch `EpisodeWsClient.kt:133-141`. Models: `VectorEvent`
  (`shared/.../WireModels.kt:23-30`), `NudgeEvent` (`:32-38`), `PositiveEvent` (`:50-55`).
  JSON is snake_case with `ignoreUnknownKeys = true` (`WireModels.kt:21`).
- OkHttp tuning to reproduce: `connectTimeout 5 s`, `pingInterval 20 s`
  (`EpisodeWsClient.kt:41-44`).

**watchOS equivalent: `URLSessionWebSocketTask`**, available since watchOS 6.0 —
`Foundation.framework/Headers/NSURLSession.h:227`, `:233`, `:239`. Binary frames via
`.data(Data)` messages; `sendPing` for keepalive (the 20 s OkHttp `pingInterval` must be
reimplemented as your own ping timer, since `URLSessionWebSocketTask` has no automatic ping
interval). Rated PARTIAL only because the socket cannot outlive the app's execution window (§7).

**Three protocol bugs found while reading, worth fixing in *both* clients** (they are client-side
omissions, not platform issues):

1. `live_session_saved.status` distinguishes `"companion"` / `"companion_hr"` / `"captured"`
   server-side (`ws.py:437`, `:456`, `:484`) but the watch reads only `live_session_id`
   (`EpisodeWsClient.kt:136-139`), so a companion socket that persisted nothing looks identical
   on-watch to a fully captured session.
2. `companion_ack` is sent explicitly so a client can confirm mode registration
   (`ws.py:388-391`) but the watch drops it (`EpisodeWsClient.kt:141`) and assumes success
   (`SentinelController.kt:630-632`).
3. `error` frames are matched and then **silently discarded** (`EpisodeWsClient.kt:140`) — no
   listener callback, no log. A repeated `malformed_json` / `unknown_type` is invisible to the watch
   and to telemetry. The watchOS client should log and telemeter these from day one; it is the only
   channel that will tell you your PCM format is wrong.

Also note `VectorEvent.participant_id` exists server-side (`server/watch/models.py:37`) but is
absent from the Kotlin model (`WireModels.kt:23-30`), surviving only on `ignoreUnknownKeys`. Include
it in the Swift model.

### 9.4 Duty cycling

`MicDutyCycle` (`shared/.../signals/MicDutyCycle.kt`) is pure and would port unchanged:

| Parameter | Value | Citation |
|---|---|---|
| Quiet before duty-cycling | 5 min | `MicDutyCycle.kt:44` |
| Cycle / ON | 10 s / 2 s (20%) | `:45-46` |
| Stillness before the deep tier | 10 min | `:47`, `:141` |
| Deep cycle / ON | 30 s / 2 s (~6.7%) | `:48-49` |
| Snap-back | a single voiced window → continuous, both clocks reset | `:67-72` |
| Fail direction | open (capture) on any ambiguity, incl. a backwards clock or missing motion | `:28-32`, `:113-131` |
| Gate | only `ARMED` ever pauses; STREAMING/COOLDOWN always continuous; COMPANION never touches the mic | `service/MicDutyGate.kt:25-33` |
| Min sleep between duty ticks | 250 ms | `SentinelService.kt:1087` |

The *logic* ports (FULL, as pure code). The *point* of it does not: duty-cycling exists to make an
all-day background mic affordable, and there is no all-day background mic on watchOS (§7). In a
user-started session, capture is continuous by definition (`Mode.SESSION`:
`shared/.../sentinel/ModePolicy.kt:37`). **Port the class for its tests and for a future
BATTERY_SAVER-like mode, but do not build a v1 around it.** Rated PARTIAL.

---

## 10. Standalone vs companion

### 10.1 What the Wear app is today

Genuinely standalone. Declared:
`<meta-data android:name="com.google.android.wearable.standalone" android:value="true"/>`
(`AndroidManifest.xml:25-27`), with `android.hardware.type.watch`
(`:4`).

And it is not merely declared — it is architecturally true. Verified by exhaustive grep across
`apps/watch/**`: **there is no `MessageClient`, `DataClient`, `NodeClient`, `CapabilityClient`,
`Wearable.*`, `WearableListenerService`, `com.google.android.gms`, Play Services dependency,
Firebase dependency, or `google-services` plugin anywhere in the watch module's sources or
`build.gradle.kts`.** The only matches are prose in comments and unconsumed version-catalog entries
from the phone lane. The watch talks to `https://mindshift-api-…run.app`
(`wearApp/build.gradle.kts:28-32`) over its own `INTERNET` permission and nothing else.

The phone's only role in pairing is a **human** reading a 6-character code off the watch face and
typing it into any signed-in client — the sign-in screen literally offers a web alternative:
`"Enter $it on your phone or gauge.app/pair"` (`ui/SignInScreen.kt:113`). This design was a
deliberate choice over both on-wrist OAuth and a Data Layer relay
(`auth/DevicePairingClient.kt:18-20`).

So: the watch needs **network**, not a phone.

### 10.2 On watchOS

Two different senses of "standalone", and they must not be conflated:

- **Independent app** (no iOS companion binary required): supported since watchOS 6, and this app
  should be built that way — it needs no iPhone-side native code, since pairing is pure HTTP.
- **Usable with no iPhone at all**: **not possible.** An Apple Watch must be paired to an iPhone to
  be set up and to receive software; a cellular model can then operate away from the phone, but the
  phone must exist. There is no Apple Watch equivalent of "buy a Wear OS watch, pair it to nothing,
  use the app".

That is a hard product constraint for this owner specifically: **the owner has no iPhone**
(`docs/plans/2026-09-20-ios-readiness.md:4-6`, and the repo's own device notes). An Apple Watch
cannot be set up without one. Rated PARTIAL.

`WatchConnectivity.framework` is present and there is a
`WKWatchConnectivityRefreshBackgroundTask` (`WKBackgroundTask.h:93`) — but **nothing in this
protocol needs it.** Do not introduce a WatchConnectivity dependency; it would make the watch app
*less* standalone than the Wear one. The one place it is genuinely tempting is COMPANION mode
(§10.3), and even there the server relay already does the job.

### 10.3 COMPANION mode (Tier B)

`Mode.COMPANION` (`shared/.../sentinel/ModePolicy.kt:9`, params at `:38`:
`bufferSeconds = 0, triggerDbOverBaseline = 0.0, continuous = true, usesMic = false`) — *"the PHONE
listens; the watch keeps the episode WebSocket open purely to receive relayed nudges and render them
as haptics — no mic, no PCM, no trigger evaluation"* (`ModePolicy.kt:21-24`).

Mechanics:
- `companionTick()` (`SentinelController.kt:581-604`): replay due reminders, attempt reconnect, send
  HR every 5 s, send `{"type":"heartbeat"}` every **20 000 ms**
  (`HEARTBEAT_INTERVAL_MS`, `:1163`; matches the OkHttp `pingInterval`).
- Tick cadence is `postDelayed`, not blocking-read-paced: `COMPANION_TICK_MS = 1_000L`
  (`SentinelService.kt:1091`, posted `:277`).
- One deterministic session id per account per UTC day: `companion-YYYYMMDD-<account>`
  (`control/CompanionSession.kt:21-29`), so a day of reconnects reuses one id.
- `{"type":"companion"}` hello tells the server to persist nothing (`ws.py:386-392`).
- Nudges arrive as ordinary `nudge` frames and set `HapticDirector`'s reminder level exactly as a
  mic episode would (`SentinelController.kt:574-577`).

**This is the highest-fidelity watchOS mode available, and it should be v1 alongside the
user-started session.** It removes the mic, removes the duty cycle, removes the PCM stream, and
reduces the watch to "a socket and a buzzer" — which is the one thing a watchOS app can nearly do.
It is rated PARTIAL only because of §7.3: a nudge arriving on the socket at 14:32 cannot reliably
reach the wrist unless the app is frontmost or an extended-runtime/workout session is live. An
all-day COMPANION socket is not a thing watchOS supports.

---

## 11. Pairing and auth

### 11.1 The handshake

Entirely server-mediated HTTP. **This is the single cleanest part of the port: it works on watchOS
as-is.**

| Thing | Value | Citation |
|---|---|---|
| Code length | 6 | `server/watch/routers/pairing.py:165` |
| Code alphabet | `ABCDEFGHJKMNPQRSTUVWXYZ23456789` (31 chars, no 0/O/1/I/L) | `pairing.py:166-167` |
| Code minting | server-side, `secrets.choice` per char | `pairing.py:231-232` |
| Pairing TTL | **10 minutes** | `pairing.py:164` |
| `pairing_id` | `secrets.token_urlsafe(24)` = 192 bits | `pairing.py:175`, minted `:275` |
| Device token | `secrets.token_urlsafe(32)` = 256 bits | `pairing.py:172`, minted `:235-236` |
| Max plaintext-token reads | **5**, then the raw token is nulled | `pairing.py:180`, enforced `:316-325` |
| Claim lockout | 15 failed claims per account, 24 h sliding | `pairing.py:187`, `:193`, `:200-228` |
| Hash at rest | bare SHA-256 hex, unsalted (high-entropy secrets) | `server/watch/pairing_store.py:37-46` |

Endpoints:
- `POST /me/pair/start` — **no auth**, returns `{code, pairing_id, expires_at}`
  (`pairing.py:270-285`, response model `:243-246`). Stores only `code_hash` (`:280`).
- `GET /me/pair/status?pairing_id=…` — **no auth** (`pairing_id` is the capability), **always
  HTTP 200**; unknown and expired are both reported as `status:"expired"` deliberately
  (`pairing.py:287-330`, `:291-299`). Returns `status` ∈ `pending|claimed|expired`, plus optional
  `account_id` and `device_token` (`:249-252`).
- `POST /me/pair/claim` — the **phone/web** side, full auth required, never a legacy `?account=`
  caller (`pairing.py:332-394`, `:334-344`). The watch never calls this.

Watch side:
- `DevicePairingClient` — plain OkHttp, no `Authorization` header on either call by design
  (`auth/DevicePairingClient.kt:26-28`); `connectTimeout 5 s` (`:53`), `callTimeout 30 s`
  (`:126`, raised from 10 s for Cloud Run cold starts); **exactly one silent retry, `IOException`
  only** (`:63-68`). Every failure degrades to `null` with the reason handed to `onSwallowedError`.
- Poll interval **3 000 ms** (`ui/SignInScreen.kt:32`, applied `:82`), driven by a one-shot
  `LaunchedEffect` — **polling stops the moment the screen leaves; there is no background poller**
  (`SignInScreen.kt:37-40`).
- Give-up bound **1 200 000 ms = 20 minutes** (2× the server TTL)
  (`auth/PairingPoller.kt:69`, checked `:58-60`).
- `null` from `poll()` means transport failure only, never a lifecycle state
  (`PairingPoller.kt:13-16`); `"claimed"` missing either field → `Failed` (`:48-49`).
- Wire models: `PairingStart` (`shared/.../WireModels.kt:290-294`), `PairingStatus` (`:299-303`).
- On success: store, publish to `AccountBus`, then fire-and-forget `POST /me/claim-legacy`
  (`SignInScreen.kt:88-95`).

### 11.2 Why this ports at full fidelity

Nothing in it is Android. Two unauthenticated HTTP calls, a 3 s poll, a 20 min bound, and a bearer
token. `URLSession` covers it entirely. The pure decision class `PairingPoller` (79 lines, no
platform deps) should be ported verbatim or moved into the shared module. Rated **FULL**.

One thing to change on watchOS, not because you must but because you should: the poller dying with
the screen (`SignInScreen.kt:37-40`) is *more* of a problem on a watch whose screen sleeps in
~20 seconds. Consider keeping the code on screen with `WKExtendedRuntimeSession` (self-care type) or
accepting a re-poll on `onAppear`. The 20-minute bound gives you room.

### 11.3 Token storage

Today: plain `SharedPreferences`, file `"gauge_account"`, keys `"account_id"` / `"device_token"` —
**unencrypted, no Keystore, no EncryptedSharedPreferences**
(`auth/AccountPrefs.kt:21-23`). Sent as `Authorization: Bearer <token>`
(`net/WatchApiClient.kt:124-128`), with a fail-fast `NOT_SIGNED_IN` before any network call when the
token is null (`:125`).

On watchOS: **use the Keychain** (`Security.framework` is present) for the device token and
`UserDefaults` for the non-secret `account_id` and the preferences in `GaugePrefs`. This is a
straight improvement over the Android side and costs nothing. Rated **FULL**.

Two auth behaviours to preserve exactly:
- A **401 clears the stored device token** and routes back to sign-in — the watch must know it is
  signed out rather than retrying forever (`control/CouplesLoad.kt:19-24`).
- A **503 must not** do that. The server deliberately returns 503 rather than 401 for a transient
  verifier outage precisely so the watch does not sign itself out on a Firestore blip
  (`server/watch/auth.py:156-162`, `:263-277`). Getting this backwards silently logs the user out
  during every server hiccup.

---

## 12. Connectivity and offline behaviour

### 12.1 Reconnect

`control/ReconnectPolicy.kt` — pure, ports unchanged:

| Constant | Value | Citation |
|---|---|---|
| `INITIAL_DELAY_MS` | 2 000 | `ReconnectPolicy.kt:73` |
| `MAX_DELAY_MS` | 30 000 | `:74` |
| `MAX_SHIFT` | 8 (bounds `attempt`, not the delay) | `:79` |
| Ladder | `min(initial shl attempt, max)` → 2 s, 4 s, 8 s, 16 s, 30 s | `:36-41` |
| Give-up condition | `isAttemptDue` returns `false` whenever `streaming == false` | `:66-70` |
| Reset | on confirmed reconnect / fresh episode / disarm | `:44-55` |

Also note `EpisodeWsClient` itself holds **no** reconnect state — it is create-one-per-episode
(`net/EpisodeWsClient.kt:76-79`), and the controller mints a listener generation token so a
superseded client's callbacks self-neuter (`SentinelController.kt:984-997`). Reproduce that
pattern on watchOS; `URLSessionWebSocketTask` has the same supersession hazard. Rated **FULL**.

### 12.2 The offline ladder

When the socket is down, the watch computes levels itself — `NudgeStateMachine`
(`shared/.../NudgeStateMachine.kt:86-131`), invoked only from the offline branch
(`SentinelController.kt:849-855`):

| Parameter | Value | Citation |
|---|---|---|
| dB→level | ≥14 → 3, ≥10 → 2, ≥6 → 1, else 0 | `NudgeStateMachine.kt:126-131` |
| Cooldown | 20.0 s, strict `>` | `:87`, `:113` |
| De-escalation | exactly one level per cooldown, never a snap to 0 | `:113-119` |
| Sustain | `E == level && level > 0` refreshes the clock and emits nothing | `:108-111` |
| Hold-3 | `HEAT_HOLD_S = 3.0`, window `HEAT_WINDOW_S = 1.0` | `:23`, `:26`, `LoudnessHold` `:33-50` |

The hold-3 number is measured, not chosen: the single-window ladder delivered *"97 buzzes/hour"* on
the median AMI meeting and 80 on SBCSAE, halved to 46.4 / 36.0 by requiring three consecutive
windows (`NudgeStateMachine.kt:6-14`). The server mirrors it
(`server/nudge_policy.py:62`, `HOLD_VECTOR = "yelling"` at `:67`), and the two are pinned only by
`server/tests/fixtures/policy_vectors/nudge_policy.json` (schema v2 `config.hold_s`).

**A watchOS port adds a third runtime to that lockstep.** Either compile the shared Kotlin (§18) or
replay the same JSON vectors from a Swift test — do not hand-transcribe the ladder. Rated **FULL**
(it is pure arithmetic), with the caveat already documented on Android: there is **no sensitivity
scaling offline** (`NudgeStateMachine.kt:55-56`), so a wearer with sensitivity ≠ 1.0 feels a
different ladder when the socket drops. Same on watchOS.

---

## 13. Retro-capture and journal

### 13.1 What exists

- `RetroCaptureBuffer` — a rolling ring of the wearer's own recent audio, ceiling
  **300 s** (`shared/.../capture/RetroCaptureBuffer.kt:8`, `:30`), fed the same PCM windows the
  sentinel reads, never while DISARMED (`:14-19`). Honest clamp: asking for 2 min after 40 s armed
  returns 40 s, never zero-padded (`:36-45`). At 32 KB/s that is a **9.6 MB** in-memory buffer.
- Product default for a manual grab: `RETRO_CAPTURE_DEFAULT_SECONDS = 120.0`
  (`SentinelService.kt:1093`).
- Upload is two-step, gzipped: `POST /captures` then `PUT /captures/{id}/audio` with
  `Content-Type: application/octet-stream` + `Content-Encoding: gzip`
  (`capture/RetroCaptureUploader.kt:43-62`, gzip `:64-68`; paths
  `net/WatchApiClient.kt:83-100`). Returns `true` **only** when the audio PUT itself came back OK —
  never optimistically on the `POST` alone (`RetroCaptureUploader.kt:19-25`, `:55-58`).
- **Consent is a structural gate, not a comment**: `consentConfirmed` is a required parameter and the
  class refuses to make *any* network call without it (`RetroCaptureUploader.kt:27-36`, `:44`).
- Journal mode writes the consent timestamp atomically with the flag, and clears both together
  (`prefs/GaugePrefs.kt:117-154`).

### 13.2 The journal cadence does not survive

`JournalScheduler`: upload interval **5 minutes** (`journal/JournalScheduler.kt:4`), snapshot ceiling
300 s (`:8`), deferred (not rescheduled) while STREAMING (`:60`), catch-up clamped to 300 s (`:62`),
driven off the ~1 s tick. `JournalQueue` is a deliberate **capacity-one** retry queue
(`journal/JournalQueue.kt:15`, `:30`) because the ring only holds 300 s anyway (`:3-9`). The service
uploads oldest-first, at most two per tick (`SentinelService.kt:357`, `:375`), and the three-call
order is mandatory — `POST /captures` → `PUT …/labels` → `PUT …/audio` — because the server's journal
hook fires off the audio-success path and only for already-labelled captures
(`journal/JournalUploader.kt:13-19`, `:51-57`).

On watchOS:
- The 5-minute cadence requires a process that is alive every 5 minutes. `WKApplicationRefresh`
  budgets are system-determined and nearer *hourly*
  (`WKBackgroundTask.h:128`), and background refresh tasks get seconds of CPU, not a 9.6 MB gzip and
  upload.
- The 300 s ring means every missed slot is **unrecoverable data**, by design. A 5-minute ring
  polled hourly keeps ~8% of the audio.
- Uploads themselves are fine: use a **background `URLSession`** so the system can finish the
  transfer after the app suspends, and handle the completion in
  `WKURLSessionRefreshBackgroundTask` (`WKBackgroundTask.h:88`).

**Honest watchOS shape:** journal captures only within a user-started session, uploading at session
end (or on the 5-minute schedule *while the session is live*), via a background `URLSession`. There
is no all-day journal. Rated PARTIAL. `JournalScheduler`, `JournalQueue`, `RetroCaptureBuffer` and
`journalTelemetryData` are all pure and port unchanged; their *driver* does not.

Also check memory: a 9.6 MB PCM ring plus a gzip buffer plus an `AVAudioEngine` on a watch is a real
risk. `WKExtendedRuntimeSessionErrorExceededResourceLimits = 5`
(`WKExtendedRuntimeSession.h:60`) exists for a reason. Consider halving the ceiling for watchOS and
saying so.

---

## 14. Telemetry

Today: a fixed-size in-memory ring, `RING_CAPACITY = 200` (`telemetry/Telemetry.kt:54`), no dedupe —
which is why so many call sites in this codebase log push-on-change rather than per-tick
(`HapticDirector.reportHapticPath` at `:259-264`, `SentinelService.reportCaptureMode`,
`MicReader.pauseFailureWarned` at `:28-34`). `flushAsync` **drains** the ring
(`Telemetry.kt:85-91`). Endpoint `POST /telemetry` with a `TelemetryBatch(device, app_version,
events)` (`telemetry/TelemetryClient.kt:44-46`, `:66-68`; model
`shared/.../WireModels.kt:124-129`), `connectTimeout 5 s`, no call timeout on the async path
(`TelemetryClient.kt:84-86`).

A crash handler wraps the previous default `Thread.UncaughtExceptionHandler`, posts **blocking with a
2 000 ms timeout**, then always re-delegates or kills the process
(`Telemetry.kt:98-118`), truncating the stack to `MAX_STACK_CHARS = 20_000` (`:12`).

Flush triggers: every journal tick (`SentinelService.kt:408`), the journal toggle
(`journal/BatteryReader.kt:53`), episode end on STREAMING→COOLDOWN (`SentinelService.kt:905`), every
sentinel stop path (`:789`), and both terminal sign-in failures
(`ui/SignInScreen.kt:76`, `:102`).

On watchOS:
- The ring, the batch model, the flush discipline and the push-on-change conventions all port
  unchanged (pure code, and `TelemetryClient` is deliberately free of `android.*` so it runs as a
  plain JVM test — `TelemetryClient.kt:16-18`).
- **The crash handler does not.** `NSSetUncaughtExceptionHandler` catches Objective-C exceptions but
  **not Swift runtime traps** (a force-unwrap of nil, an array bound, a `fatalError`), which is the
  dominant Swift crash class. `MetricKit` is **not in the watchOS SDK** (verified: absent from the
  framework listing). So the on-device crash breadcrumb this repo relies on has no faithful
  equivalent. Plan on Xcode Organizer crash logs plus aggressive pre-crash breadcrumb flushing
  instead. Rated PARTIAL.
- Use a **background `URLSession`** for flushes so a flush issued as the app suspends actually
  completes.

---

## 15. Glanceable surfaces

### 15.1 Tile → WidgetKit

Today: `androidx.wear.tiles.TileService` (`tile/GaugeTileService.kt:10`, `:147`) showing exactly two
text elements — `"On · <mode>"` / `"Off"` and a button labelled with the action about to happen,
`"Turn off"` / `"Turn on"` (`tileStrings` at `:62-66`, layout `:75-104`). Clicks use
`ActionBuilders.LoadAction` because nothing in the tiles API can start a foreground service from a
`Clickable` (`:89-93`), round-tripping through `onTileRequest` via
`requestParams.state?.lastClickableId` (`:152`). Freshness **60 000 ms** (`:168`, `:199`), plus a
one-shot post-click refresh at **1 500 ms** (`:200`, posted `:190-195`), plus the real push path from
the service.

watchOS: the analogue is a **WidgetKit accessory widget** (`WidgetKit.framework` is present);
`androidx.wear.tiles` / `protolayout` have no counterpart. Differences:
- Tap-to-toggle requires an **App Intent** (`AppIntents.framework` is present) rather than a
  `Clickable`; interactive widget buttons are supported and this is arguably cleaner than the
  `LoadAction` round-trip.
- There is no `setFreshnessIntervalMillis`. You publish a **timeline** and call
  `WidgetCenter.shared.reloadTimelines(ofKind:)` on change — which maps well onto the existing
  push-on-change design (§15.3), but reloads are system-budgeted, so the 1 500 ms post-click refresh
  is not guaranteed.
- An arm/disarm toggle from a widget can only start work the app is *allowed* to start — see §7.
  "Turn on" from a widget cannot launch an all-day mic session.

Rated PARTIAL. `tileStrings` is a pure function with its own test
(`wearApp/src/test/kotlin/app/gauge/wear/tile/TileLayoutTest.kt`); port it as-is.

### 15.2 Complication → accessory widget

Today: `SuspendingComplicationDataSourceService`
(`complication/ArmedComplicationService.kt:10`, `:41`), **RANGED_VALUE only**
(`AndroidManifest.xml:68-70`), arc bounds `MIN_LEVEL = 0f` / `MAX_LEVEL = 3f`
(`ArmedComplicationService.kt:99-100`). Content: while STREAMING, the value is the worst level across
`channelLevels` clamped 0..3 with text `"Level <n>"`; otherwise value 0 and `"On"` / `"Off"`
(`complicationValue` at `:116-123`). Periodic poll `UPDATE_PERIOD_SECONDS = 600` is an explicit
backstop, not the primary path (`AndroidManifest.xml:66-73`).

watchOS: `accessoryCircular` with a SwiftUI `Gauge(value:in: 0...3)` is a close visual match, and
`accessoryCorner` / `accessoryInline` cover the other slots. Rated PARTIAL: the shape ports, the
push model does not (below), and there is no direct `RANGED_VALUE` type — you get a SwiftUI view and
must draw the arc yourself.

`complicationValue` is pure with its own test (`ComplicationContentTest.kt`); port as-is.

### 15.3 The push-on-change discipline — port this exactly

`pushFaceUpdates` (`SentinelService.kt:999-1021`, called from `publishAndNotify` at `:911`) calls
`ComplicationDataSourceUpdateRequester.requestUpdateAll()` **only when `complicationValue(snapshot)`
changed** (`:1000-1006`) and `TileService.getUpdater(...).requestUpdate(...)` **only when
`tileStrings(...)` changed** (`:1015-1017`). The reason is arithmetic: the tick runs at ~1 s for the
whole armed session, so an ungated push would be **3600+ calls/hour against two rate-limited
APIs** (`:983-987`). Both calls are `runCatching`-wrapped and only update the cache on success
(`:1007-1013`, `:1018-1020`), so a failed push retries on the next real change.

WidgetKit reload budgets are, if anything, **tighter** than Wear's. Port this gating verbatim; it is
the difference between a working complication and a silently throttled one.

---

## 16. Motion

`AccelSource` (`sensors/AccelSource.kt:22-50`): `SensorManager` +
`Sensor.TYPE_ACCELEROMETER` at `SENSOR_DELAY_UI` (`:38`), reporting the **standard deviation of the
acceleration-vector magnitude within rolling 1-second buckets** (`:13-14`), `null` until the first
bucket completes (`:19-20`). `MovementTracker`
(`shared/.../signals/MovementTracker.kt:23-40`) keeps a rolling-median baseline over 30 readings
with threshold `baseline + 2·MAD`, floored at `baseline·2 + 0.1` when MAD is 0, and the same
anti-poisoning rule as HR.

Consumed for exactly one thing: gating `MicDutyCycle`'s DEEP tier, where **`null` never deepens**
(`MicDutyCycle.kt:54-57`, `:74-79`).

watchOS: `CoreMotion` is present. Two options — `CMMotionManager` (classic, delegate/handler-based)
or `CMBatchedSensorManager`, watchOS 10+, which delivers **batched** accelerometer data with a
reported `accelerometerDataFrequency`
(`CMBatchedSensorManager.h:20-21`, `:47-57`, `:64-77`) and is the right tool for a 1-second-bucket
stddev. Batching is actually a better fit than `SENSOR_DELAY_UI` polling.

Caveats: `CMBatchedSensorManager.authorizationStatus` (`:29`) means motion is a
**permission-gated** signal on watchOS (`NSMotionUsageDescription`) where on Wear it is not, and
high-rate batched motion outside a workout session is subject to the same background limits as
everything else (§7). `MovementTracker` itself is pure and ports unchanged. Rated PARTIAL — and
note it becomes moot if you skip the duty cycle (§9.4), which is the recommendation.

---

## 17. Battery

`BatteryReader` (`journal/BatteryReader.kt:19-29`): `BatteryManager.getIntProperty(
BATTERY_PROPERTY_CAPACITY)` filtered to 0..100, plus `bm.isCharging`; any failure yields
`BatteryStatus(null, null)` and the telemetry payload emits explicit JSON `null` rather than a
fabricated number (`journal/JournalTelemetry.kt:51-65`, `:59-60`).

watchOS: `WKInterfaceDevice.currentDevice().batteryLevel` (0.0–1.0, **-1.0 when unknown**) and
`.batteryState` (`Unknown` / `Unplugged` / `Charging` / `Full`), both watchOS 4.0+ —
`WKInterfaceDevice.h`, in the property block above `playHaptic`. One gotcha:
`isBatteryMonitoringEnabled` **defaults to NO** and must be set before the values are meaningful
(same header). Map `-1.0` and `Unknown` to the existing honest-`null` path. Rated **FULL**.

---

## 18. Code reuse: the `shared/` module

`docs/plans/2026-08-15-unification-mindshift-absorbs-gauge.md:21` claims the shared KMP module
*"compiles to Kotlin/Native watchOS targets later — same brains, SwiftUI skin."*

**That is currently aspirational, and the reader should know it.**
`apps/watch/shared/build.gradle.kts:36-38` declares exactly two targets:

```kotlin
kotlin {
    jvm()
    androidTarget()
}
```

No `watchosArm64`, no `watchosSimulatorArm64`, no `iosArm64`. The dependencies are, however,
friendly: commonMain uses only `kotlinx-serialization-json` and `kotlinx-coroutines-core`
(`shared/build.gradle.kts:43-44`), both of which ship Apple/watchOS artifacts. So adding the targets
is plausible work, not a rewrite — but it is *unstarted* work, and the KMP-to-XCFramework toolchain
(plus Gradle's `embedAndSignAppleFrameworkForXcode`) is its own project.

What is worth reusing, ranked by value ÷ effort:

| Priority | Files | Why |
|---|---|---|
| **Highest** | `NudgeVocabulary.kt`, `NudgeHapticSchedule.kt`, `NudgeStateMachine.kt` | These are the cross-runtime contracts already pinned by `server/tests/fixtures/policy_vectors/*.json`. A third hand-transcription is how the ramp already drifted once — see `HapticPatterns.kt:97-101`, where channel A's fallback silently diverged from the vocabulary on 2026-09-06. |
| High | `HrTracker`, `MovementTracker`, `SentinelDetector`, `SentinelStateMachine`, `MicDutyCycle`, `RetroCaptureBuffer`, `RingBuffer` | Pure, tested, non-trivial arithmetic |
| Medium | `PairingPoller`, `ReconnectPolicy`, `ShoutTapGate`, `JournalScheduler`, `JournalQueue`, `HrDemandPolicy`, `SensorLifecycleGate`, `PendingStopIntent`, `tileStrings`, `complicationValue`, `companionSessionId`, `micPauseDue`, `notificationText` | Pure and already extracted specifically *because* the Android shells are compile-gated only — the same reasoning applies double on a platform whose shells cannot be tested at all |
| Do not port | `PulseEngine`, `PulseChainGate`, `PulseChainDecision` | §6: the pulse train should not exist on watchOS |
| Rewrite | everything in `wearApp/ui/`, `RealVibratorPort`, `MicReader`, `HrSource`, `AccelSource`, `SentinelService`, `GaugePrefs`, `GaugeTileService`, `ArmedComplicationService` | Platform shells |

**If KMP is not wired up, the fallback is non-negotiable: replay the same JSON vector fixtures from
Swift tests.** `server/tests/fixtures/policy_vectors/nudge_policy.json` and
`nudge_vocabulary.json` are already replayed by three runtimes
(`NudgeStateMachineVectorsTest.kt`, `server/tests/test_nudge_vocabulary_vectors.py`,
`apps/mobile/__tests__/nudgeVocabulary.test.ts`). Add a fourth. *"An emoji or a millisecond can
only change in one place"* (`NudgeVocabulary.kt:10-11`) has to keep being true.

Rated PARTIAL.

---

## 19. Recommended build order

1. **Add the watchOS simulator runtime** (Xcode → Settings → Components). Nothing can be run until
   this exists (§2.1).
2. **Pairing + auth + `GET /me/standing`.** Pure HTTP, FULL parity, proves the account and the
   Keychain end to end, needs no mic, no HealthKit, no haptics. Testable entirely on the simulator.
3. **COMPANION mode.** Socket, `{"type":"companion"}` hello, heartbeat at 20 s, nudge frames,
   reminder ladder from the ported `NudgeHapticSchedule`, `playHaptic` mapping from §4.2. This is
   the highest-fidelity mode watchOS can host, and it needs no microphone.
4. **A user-started coaching session** (`Mode.SESSION`-shaped): `AVAudioEngine` → `AVAudioConverter`
   → 16 kHz mono PCM16 → 32 000-byte binary frames on `URLSessionWebSocketTask`, with
   `HKWorkoutSession` for HR and background runtime.
5. **Widget + complication** using the ported pure projections and the push-on-change gating.
6. **Retro-capture / journal** on a background `URLSession`, session-scoped.
7. **Never:** all-day ARMED sentinel, the pulse train, the full 8-code haptic vocabulary.

### The three tests to run the hour an Apple Watch is available

1. Does `playHaptic` fire while the screen is off / the wrist is down, during a live
   `HKWorkoutSession`? (Decides whether the product exists on this platform at all.)
2. How does the actuator render N `.click` calls 170 ms apart — N taps, or one smear? And are
   2 taps distinguishable from 3? (Decides whether the level ladder survives at all.)
3. Can an `AVAudioSession` in `.record` stay active for 30 minutes in the background under a workout
   session, and what does it cost in battery? (Decides whether mic sessions are usable.)

Until those three are answered on hardware, every PARTIAL in this document's haptic and execution
rows should be read as "PARTIAL at best".
