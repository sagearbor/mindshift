# Poker6 sliding-window diarization — round 3 research artifacts

Archived reference material from the round-3 investigation (branch
`feat/sliding-window-diarization`) into whether a sliding-window,
voice-only speaker-change detector can find all 6 speakers in
`server/tests/fixtures/audio/test_recording_poker6_real.wav`, where the
existing pipeline only finds 4.

**These scripts are not meant to be run as-is** — `run_v3.py` hardcodes
absolute paths from the machine/worktree it was developed on. They're
kept for reference: the calibration approach (`centroid_calibrate*.py`),
the batching benchmark (`bench_batch.py`, `bench_batch_n8.txt`), and the
scoring logic (`score.py`).

## Final measured results (clean, idle-machine run, 2026-08-23)

- **poker6** (`v3_result_poker6.json`): num_speakers=6 (correct, first
  time), pooled_cosine=0.291 (passes the production 0.45 threshold),
  per_turn_accuracy=0.714 (5/7). Cost: sliding-window stage 19.8s +
  diarize_turns stage 32.5s = **52.3s total** for a 30s clip.
- **family_real** (`v3_result_family.json`): 100% accuracy, num_speakers=2
  — safety check confirming the new approach doesn't regress the easy
  case.

The identical measurement run the night before, under heavy concurrent
CPU load from other jobs on the same machine, read as ~29.7 minutes
total — a ~34x difference purely from contention, not the algorithm.
Cost is not the blocker; accuracy is.

## Two remaining failure modes (poker6)

1. One speaker's continuous speech got split into two different
   predicted speakers (over-segmentation).
2. The final new speaker (last to appear in the recording) got folded
   into an earlier speaker's cluster instead of recognized as new.

## Status

Not production-ready by round 3's own bar (~90%+ accuracy target).
Not wired into `server/diarize_local.py` or the live pipeline. Next
step under consideration: evaluate pyannote.audio (blocked earlier only
on missing HuggingFace credentials, not on capability) as a
higher-leverage alternative to further hand-tuning this bespoke
approach.
