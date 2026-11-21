# Wearer-identity decision rule (plan 2026-11-05, Sprint Step 3) - result: MISSED

Measured with `scripts/recreplay/identity_eval.py` (phone side: Silero + ECAPA labeler over cached
recordings, held-out voiceprints; 5 "honest" DEV items) and the `replay.nudgeReport` jest gates.
$0 spend. Held-out was looked at once (burned) for the shipped config.

## DEV, one change per run

| run | change | turn recall | recall, turns >= 1.5 s | recall, seconds | false-self |
|---|---|---|---|---|---|
| s3base | shipped labeler | 0.478 | 0.647 | 0.717 | 8.8% |
| r1 | `minJudgeSeconds` 1.5 | 0.478 | 0.647 | 0.717 | 8.8% |
| r2 | `windowVote` (2 s windows, old veto) = overlap exclusion | 0.370 | 0.500 | 0.489 | 3.6% |
| r3 | `clusterSelfVote` 8 s / 0.45 (seconds-weighted) | 0.522 | 0.706 | 0.771 | 9.8% |
| r4 | windowVote + `vetoOtherFrac` 0.25 | 0.261 | 0.353 | 0.221 | 3.1% |
| r6 | r2 + r3 | 0.391 | 0.529 | 0.498 | 3.6% |
| r8 | clusterSelfVote strict (10 s, 0.52, 3 turns) | 0.478 | 0.647 | 0.717 | 8.8% |

Held-out, shipped config (`{}`): turn recall 0.329, >= 1.5 s 0.535, seconds 0.554, false-self 4.6%,
time to first confirm 300 s.

## Reading

* No knob reaches false-self <= 2.5% with seconds-recall >= 0.70: overlap exclusion buys false-self
  (8.8% -> 3.6%) by giving up ~23 points of seconds-recall; the seconds-weighted cluster vote buys
  recall (+5 points) but admits false-self. The residual false-self is mostly the wearer's voice truly
  present under another speaker (CHiME-6, AMI TS3005a: long turns scoring 0.65-0.78 vs the print).
* `minJudgeSeconds` 1.5 changed no recall/false-self number, but dropped the pinned
  `scene_meeting4` strong-shout hit in `replay.nudgeReport` (short non-matching turns stop joining a
  cluster), so it ships OFF.
* The floor-cue gate (>= 4/6 held-out runs) was NOT measured: the cue (`floorTick.ts`) scores
  windows against the print directly and does not read the labeler verdict these changes touch.

## What ships

* `fastLoop.ts`: no "Speaker A is you" guess by default (`selfSpeakerFallback` default null);
  only a voiceprint verdict or the user's explicit You-chip choice makes a turn wearer-addressed.
* `server/speaker_id.py`: `MIN_MATCH_SECONDS` 1.0 -> 1.5 (env `MINDSHIFT_VOICE_MIN_JUDGE_SECONDS`);
  `audio_pipeline` reads it at call time. Server-side effect not replay-measured.
* `speakerId.ts`: opt-in `minJudgeSeconds`, `clusterSelfVote`, `windowVote.vetoOtherFrac` (all OFF).
* Not done (files owned by another agent this sprint): seconds-weighted wearer estimate and overlap
  exclusion inside `audio_pipeline._resolve_wearer` / `_confirm_wearer_by_voiceprint`.
* Harness: `identity_eval.py` now reports turn-recall >= 1.5 s and seconds-weighted recall, and runs
  under tsx via `scripts/recreplay/tsconfig.replay.json` (react-native / expo-secure-store stubs)
  because `languagePrefs.ts` made `fastLoop.ts` unimportable from node.
