# Thanksgiving test plan + where nudges stand (2026-11-22)

Written from a cloud Claude chat (no device available) by reading this repo.
Shareable copy for the owner's mom (a therapist): the Claude doc
"MindShift — Thanksgiving test sheet". This file is the copy a local session
can read. Sources: `docs/research/2026-08-30-nudge-quality/README.md`,
`docs/plans/2026-09-18-real-conversation-audit.md`,
`docs/plans/2026-09-10-heat-rubric-and-buzz-dose.md`,
`docs/plans/2026-11-07-emotion-dev-mode.md`,
`docs/plans/2026-11-10-emotion-on-watch-phase2.md`,
`docs/plans/2026-08-25-cost-model.md`, `apps/mobile/src/live/fastLoop.ts`.

## Where things stand

Nudge wording is good; timing is not. Spoken lines arrived ~16 s late in the
only measured session, and both emotion models have never run on a device.

| From -> to | Delay | Quality | Measured? |
|---|---|---|---|
| Phone AI (Gemini Nano) -> earbud | ~16 s median (LLM 5.5 s) | 4.1/5 lab (prompt v2), 3.0/5 real stored lines | y, 2026-08-26 |
| Phone -> cloud -> earbud | 4 s wasted on Nano timeout + cloud | 4.05/5 replayed | partly |
| Watch mic -> buzz (loudness) | ~1 s | buzzes on ~46% of happy speech (~32% with valence) | recordings only |
| Phone emotion model (TINY) -> watch | >= 6 s (2 x 3 s windows) + ~0.3-2 s relay | none | n |
| Watch-only emotion model | unmeasured (desktop ~24 ms compute) | none | n |

Quality = Haiku judge 1-5 (specific, actionable, not preachy).
False-positive nudges on calm/repair turns: 36% -> 9% after the 08-30 fixes.
Watch dose: ~85 buzzes/h on ordinary talk; ~20/h with a 5 s hold; target ~12/h.

## Why 16 s, and why cloud can be faster

| Step | Phone AI path | Cloud path |
|---|---|---|
| Turn end (VAD pause) | ~0.3 s+ | same |
| STT finalizes | ~1-2 s | same |
| Write the line | ~5.5 s (Nano) | ~1-2 s (Haiku) |
| Hold while anyone talks | several s | same |
| Total | ~16 s median | not measured end-to-end |

- 08-26 sessions tried Nano first and waited its full 4 s timeout before cloud.
- Since 10-29 `earlyTurnLocal` sends the turn to cloud before Nano runs; first
  answer is voiced. Replay-measured only, not device-verified.
- PRD target < 0.8 s mic -> earpiece. Realistic next target ~3 s cloud-first.

## Voice training (identity)

- 09-18 audit: ECAPA is fine (AUC 0.891), threshold was too strict.
  Self-recall in a real room: 0.65 -> ~36%, 0.60 (shipped 09-18) -> ~48%,
  0.55 -> ~60% (0.5% false accept). Scenes at 0.60: self 7/7, 3/5, 4/5.
- All 8 real 08-26 sessions had `is_self = null` on every turn, so the nudge
  path never ran in production.
- Unmatched turns fall back to "Speaker A = you", which is near-random in groups.
- Open: reach >= 55% self-recall; decide silent-vs-guess for unmatched turns.

## Tests to run at Thanksgiving

1. **Scripted emotion test (phone, ~5 min).** Settings -> Emotion dev tester ->
   Scripted test; 3 rounds per emotion; Live Coach closed. Record asked vs guessed.
2. **Watch-only model (~15 min normal talk).** Watch Settings -> Emotion on
   watch -> on. Tap ✗ / ✓ / Ignore after each buzz. Note avg/max ms and
   battery % at start and end. Pull `emotion-windows.jsonl` later (see 11-10 plan).
3. **Spoken nudges, timed (Live Coach, earpiece, ~10 min).** Confirm the
   "You: Speaker A" chip first. Per nudge: seconds from end of sentence to
   whisper, and a 1-5 score.
4. **Mom labels clips (Review mode, ~10 min).** Labeller "Mom", mode
   "Emotion + nudge", guess hidden.

| Test | Tries | Right | Delay | Notes |
|---|---|---|---|---|
| 1 Scripted: anger | | | – | |
| 1 Scripted: happy | | | – | |
| 1 Scripted: sad | | | – | |
| 1 Scripted: neutral | | | – | |
| 2 Watch buzzes (✓ vs ✗) | | | avg ms: | |
| 2 Watch battery | | – | start/end %: | |
| 3 Spoken nudges | | avg score: | avg s: | |
| 3 "You" detected right | | | – | |
| 4 Mom's clip labels | | agree w/ model: | – | |

Good = scripted >= 2/3 per emotion; watch mostly ✓ and <= ~4/h; spoken nudges
< ~5 s and scoring 4+.

## Questions for Mom

- Is a buzz 5-10 s late still useful, or worse than none?
- Unsure who spoke: stay silent or guess?
- Silence after a sincere apology (no praise): right, or should repair be reinforced?
- Instruction ("ease up") vs words to say ("I hear you"): which lands?
- Nudges per hour before people tune out (current targets ~12/h coach, 4/h watch)?
- Is a wrong buzz during loud laughter a deal-breaker?
- Would she use it with clients, and what would she need to see first?

## Next: cost, privacy, efficacy

- Cost: cloud coaching ≈ $0.38 per 30-min 3-way call (Haiku); whole call
  ≈ $0.59 local-first. The 08-30 offline eval cost ≈ $1.03.
- Privacy: per `fastLoop.ts` the WebSocket keeps receiving PCM during Live
  Coach, so the server already gets audio; verify before deciding. Option: a
  "private mode" that is on-device only (slower).
- Efficacy plan: replay the same recordings through the on-device prompt and
  the cloud coach with `docs/research/2026-08-30-nudge-quality/grade.py`
  (quality + latency side by side, ~$1); emotion models and watch latency
  only from on-device logs (the Thanksgiving tests). Then choose cloud-first,
  phone-first, or a private-mode switch from the numbers.
