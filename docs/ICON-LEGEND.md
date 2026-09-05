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
