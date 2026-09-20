# Decision: hysteresis on the ladder, WavLM+SpeechBrain as the cloud "heated" judge

*2026-09-20. Owner: "i think im good with your 1-4 suggestions, probably
including SpeechBrain." Evidence: `docs/research/2026-09-20-tone-model-options.md`,
`docs/plans/2026-09-19-heat-map-rounds.md`.*

## What was decided

1. **Hold-5s hysteresis on the loudness ladder.** The +6 dB first rung must
   hold for 5 consecutive 1 s windows before the ladder may escalate; reminders
   and decay unchanged. Measured dose: AMI 97 → 18/h, SBCSAE 80 → 12/h. 3 s is
   the fallback knob if the wrist feels sluggish. This is a three-runtime
   contract change (watch Kotlin `NudgePolicy`/`SentinelDetector`, phone
   `fastLoop.ts`, server `relay.py`) and must land in all three in one PR with
   the shared fixture updated.
2. **Flip Cloud Run `MINDSHIFT_TONE_AUDIO` from `off` to `dark`.** Logs
   arousal/valence/dominance beside every relayed session; no user-visible
   change. **Prerequisite:** bake the pinned 1.27 GB snapshot into the
   `INSTALL_VOICE=1` image layer (or GCS mount) so cold starts do not
   re-download it.
3. **Cloud "heated" judge on the relayed lane: WavLM odyssey_dim + SpeechBrain
   IEMOCAP.** Extend the PR #186 veto from valence-only to an arousal+valence
   confirm/veto within ~3 s of the instant tap. Stack AUC 0.941 / recall 0.66
   vs 0.897 / 0.59 for WavLM alone; +360 MB, +~100 ms. Gate: CONFER agreement
   must stay ≥ +0.55 and AMI/SBCSAE dose must not rise.
4. **On-device tier later:** eGeMAPS linear model (AUC 0.83, 36 ms) on the
   phone; distil our own Wav2Small-class student (MIT/Apache teachers only)
   for the watch. Published Wav2Small weights do not exist and are NC-licensed.

## Not decided / owner's call

- Streaming watch audio to the cloud when no phone is present (privacy posture).
- $50 OpenAI top-up to bench gpt-audio as a labeller (~$25 for a full bench).

## Order of work

1 (dose win, zero model risk) → 2 (data starts flowing) → 3 (quality) → 4.
Each step is verified from files: replay report + jest/pytest gate, never the
owner (`verify-from-files-never-owner`).

## Next steps and where each runs (2026-09-20, owner asked for the short form)

Latency = speech event → nudge on that device. Watch column assumes a phone is
present (relayed lane); watch-alone gets only rows 1 and 6. "5 s" earlier
meant two different things: the hold-N hysteresis rule (a deliberate delay on
the FIRST buzz of an episode, tunable 3 s) and the tone model's audio window
(2 s is enough: 130 ms compute measured; it was trained on 3–11 s podcast
segments, shorter than 2 s untested).

| # | step | watch | phone | web/cloud | CC effort |
|---|---|---|---|---|---|
| 1 | Hold-3s hysteresis on the loudness ladder, all runtimes | 3 s | 3 s | 3 s | ~4 h (3 runtimes + shared fixture + dose gate) |
| 2 | Bake WavLM weights into image; flip cloud flag off → dark | NO (logs only) | NO | ~0.5 s compute, logging | ~2 h (Dockerfile layer, build, deploy, verify logs) |
| 3 | Sliding 2 s tone window every 1 s; arousal+valence confirm/veto; flag dark → on | ~3 s via cloud | ~3 s via cloud | ~2.5 s | ~1 day (relay, gate on CONFER ≥ +0.55 and no dose rise) |
| 4 | Stack SpeechBrain angry-vs-happy vote on the judge | ~3 s via cloud | ~3 s via cloud | ~2.7 s | ~3 h (backend exists; wire + bench gate) |
| 5 | eGeMAPS + linear model on device (instant tier) | NO until Kotlin port (~2 s) | ~2 s | ~2 s | ~3 days phone (no openSMILE in RN: port features) + ~2 days watch |
| 6 | Distil 72 K student on our 31 h; run on watch | ~2 s | ~2 s | ~2 s | ~5 days (teacher labelling, GPU rental, ONNX on watch) |
| 7 | ✅ Regenerate CHiME-6 speaker labels from transcripts | — | — | offline | done, 2026-09-19 |
| 8 | $50 OpenAI top-up; bench gpt-audio as labeller | NO | NO | 3–5 s, offline only | ~3 h after credits |
| 9 | Review and push eval/real-conversation-audit | — | — | — | ~30 min + owner review |

"Dark" is the middle of a three-state flag (`MINDSHIFT_TONE_AUDIO`): **off** =
not computed; **dark** = computed and logged beside every session, never
changes a buzz; **on** = drives the confirm/veto. Step 2 goes off → dark so
real-session data accumulates with zero user-visible risk; step 3 goes dark → on.

**Step 7 result (2026-09-19):** the official CHiME-6 transcript JSONs
(`CHiME6_transcriptions.tar.gz`, OpenSLR resource 150) fixed the labels —
speech share is now balanced (max 18.1% of a participant, not 85%) and
overlap turns out to be much higher than assumed (43.5% vs AMI's 8.2%), which
changes the step-4/5 overlap-gate expectation for a dinner-party setting: see
"CHiME-6 corrected (transcript truth)" in
`docs/plans/2026-09-19-heat-map-rounds.md` for the full numbers. Nothing here
changes steps 1–6, 8 or 9 — CHiME-6 was never part of the hysteresis/tone-model
gates, only of the identity/overlap ceiling measurements on the graph.
