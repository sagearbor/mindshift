# Icon legend — tagging features across modalities (2026-09-05)

Every feature line in an hlist, report, plan, or status doc opens with a
**four-column tag, fixed order, pipe-separated, exactly one icon per column**
(ordered most → least important to the owner):

```
WHERE DISPLAYED | METHOD | LATENCY | WHERE COMPUTED
```

Put this legend at the top of any doc that uses the tags. In HTML it is an
expandable `<details class="legend">` (tap to open on a phone — hover does not
exist there) whose summary shows the four column names and the latency icons
with **tooltips** (`title=`): ⚡ ≈ 1 s · 🧠 ≈ 3–7 s · ⏳ ≈ minutes after the
session · 🌙 ≈ hours/overnight. The open body lists every icon. Reference
markup: tmp/hlist-20260905-abcde.html. Every hlist/report also opens with a
"How to test" card. Markdown legend line:

> **displayed | method | latency | computed** — see docs/ICON-LEGEND.md

## 1 — WHERE DISPLAYED (the device the user experiences it on)

| icon | meaning |
|---|---|
| 🎧 | earpiece / earbud |
| ⌚ | watch |
| 📱 | phone |
| 💻 | computer (web app in a browser) |
| – | not displayed anywhere (research / infrastructure) |

## 2 — METHOD (the channel; pick the PRIMARY one, name extras in the detail text)

| icon | meaning |
|---|---|
| 🗣️ | voice (spoken, TTS) |
| 📳 | vibrate / haptic |
| ✨ | screen flash |
| 📝 | on-screen text |
| 📄 | report / summary (post-session, replay, growth) |
| 👓 | Developer-mode-only text (a "dark" feature: measuring, not nudging) |
| 🔕 | not shown — numbers only (an experiment) |
| 🚧 | not yet (planned) |

## 3 — LATENCY

| icon | meaning |
|---|---|
| ⚡ | instant — ~1 s, on-device acoustics |
| 🧠 | medium — 3–7 s, the LLM's words / nuance |
| ⏳ | post-session — analysis, re-analyze, summary, replay |
| 🌙 | batch / research — overnight runs, corpus mining, fixtures |

## 4 — WHERE COMPUTED (which codebase to open when it misbehaves)

| icon | meaning |
|---|---|
| 📱 | phone app (apps/mobile — on-device VAD, ECAPA, prosody, instant tier, local LLM) |
| ⌚ | watch (apps/watch) |
| 🌐 | browser — client-side web build (same loop code as the phone; talks to ☁️ like the phone does) |
| ☁️ | our cloud server (server/ on Cloud Run: diarization, analysis, call merging, watch relay) |
| 🔀 | hybrid — the computation itself is split (on-device LLM with cloud fallback; phone STT + cloud coach). Escape hatch, not a default |

## Session MODE is NOT a column

Earpiece / in-person / call / journal / therapist-view is stated in the detail
text when a feature is mode-specific ("call mode only"). Most features apply in
every mode.

## Examples

| tag | feature |
|---|---|
| `📱 \| 📳 \| ⚡ \| 📱` | loudness buzz (instant nudge tier) |
| `📱 \| 📳 \| 🧠 \| ☁️` | steamroll nudge — call mode only; also wrist + "Let them finish" flash |
| `🎧 \| 🗣️ \| 🧠 \| 🔀` | the nuanced LLM nudge (on-device model, cloud fallback) |
| `📱 \| 👓 \| ⚡ \| 📱` | in-person overlap probe (⟂ tag) · vocal-intensity classifier (⚡% tag) |
| `📱 \| 📄 \| ⏳ \| 📱` | pre/post mood check · session dynamics block (dev mode) |
| `📱 \| 📄 \| ⏳ \| ☁️` | voice separation on recordings (windows-first) |
| `⌚ \| 📳 \| ⚡ \| ⌚` | watch journal sentinel |
| `– \| 🔕 \| 🌙 \| ☁️` | CANDOR corpus mining |

## The nudge VOCABULARY is a different thing (2026-09-06)

The four tags above describe a FEATURE, for a reader of a doc. The nudge
**vocabulary** describes a BEHAVIOUR, for the user, on their wrist and screen.
They share emoji space, so keep them apart: 📱 in column 1 means "shown on the
phone", 📈 in a nudge means "you got heated".

The vocabulary is owner-approved and lives in ONE file —
`server/tests/fixtures/policy_vectors/nudge_vocabulary.json` — replayed by the
phone (`apps/mobile/src/live/nudgeVocabulary.ts`), the watch
(`app.gauge.shared.NudgeVocabulary`) and the server
(`server/nudge_vocabulary.py`). Never hand-type one of these emoji anywhere
else; read it from the contract, so the wrist, the screen and a replay report
cannot disagree.

| icon | code | name | what it means | haptic |
|---|---|---|---|---|
| 📈 | H | Heated | you got loud, or your words got hot (yelling + aggressive tone are ONE family to the user; the detectors stay separate) | taps that get LONGER: 60→110→200 ms |
| 📉 | D | De-escalated | heat dropped a level within two turns of a spike | taps that get SHORTER: 200→110→60 ms |
| ✂️ | C | Cut in | sustained talking-over, not ordinary brief overlap | `• —`, `• • —`, `• • • —` |
| 🎤 | A | Hogging | most of the airtime over the last two minutes | slow `— — —` |
| 👂 | E | Listened | you let them finish a long turn with no cut-in | soft `••` |
| 🤝 | R | Repair | you validated or apologised and their tone softened | soft `•••` |
| 🧘 | K | Calm streak | N minutes with no escalation — **summary badge, never live** | none, ever |
| ❤️ | P | Pulse | heart rate well over resting (+15/25/35 bpm) — **watch only** | lub-dub |

Rules that are part of the contract, not styling:

- **Rhythm carries EVERYTHING — level and identity.** Brown & Brewster: rhythm
  is identified ~93% of the time, intensity ~61%. More bluntly, an Android
  phone cannot vary vibration strength *at all* — React Native can only switch
  the motor on and off — so any two cues that differ only in strength are
  literally the same cue there. That is not theory: 📈 level 3 and 📉 originally
  differed only in amplitude, and the first time they were felt on a real Pixel
  the verdict was *"heated L3 = de-escalated, I can't tell a diff"*. Rising and
  falling are now expressed in tap LENGTH; amplitude is a bonus the watch gets
  on top. **No two cues may be confusable** with amplitude removed — same tap
  count, gaps within 50 ms, every tap within 1.5× — and that is a test on all
  three runtimes, not a guideline.
- **Every tap clears the measured perceptibility floor** — 50 ms at amplitude
  180. The watch's v0.1–v0.2.3 cues (40 ms at 120–180) proved *barely
  perceptible* on a real Pixel Watch, so the soft positives sit exactly at the
  floor rather than below it: a cue nobody can feel is not a soft cue, it is a
  missing one.
- **Positives are soft, unleveled and capped at one per two minutes** across
  D/E/R together. A withheld positive still reaches the session summary: the
  cap silences a cue, it does not erase what the user did.
- **Alert codes are H C A P; positive codes are D E R K.** The order above is
  the owner's, and it is also worst-first — which is how ties break when
  several vectors fire at once.
- 👂 and 🤝 were originally the same soft `••`, on the theory that the wrist
  says "that was good" and the screen says which. On the device they were
  indistinguishable — and a phone in a pocket has no screen to check — so 🤝
  gained a third tap. Same family, same softness, its own rhythm.

To feel all eight: **Settings → Feel the patterns** — on the phone (Developer
mode) *and* on the watch. Test both: they do not feel the same, and only the
watch reinforces the ramps with amplitude.

