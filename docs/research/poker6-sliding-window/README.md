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

## Status (round 3)

Not production-ready by round 3's own bar (~90%+ accuracy target).
Not wired into `server/diarize_local.py` or the live pipeline.

## Round 4 (2026-08-24): RESOLVED — root cause was a threshold, not the algorithm

The sliding-window approach above and pyannote.audio (evaluated this round,
see below) both plateaued around 71-83% on poker6. Root cause turned out to
be much simpler: `diarize_local.py`'s existing, already-automatic k-search
(no speaker count needed, tries k=2..6 and validates each) was rejecting the
genuine 6th voice by a hair — its marginal-split cosine (0.301) and anchor
cosine (0.231) each missed the old `STRONG_SEPARATION_COSINE=0.30` /
`NEW_VOICE_ANCHOR_COSINE=0.20` bars by a small margin. Recalibrated both
constants (0.30->0.32, 0.20->0.24) using poker6 as a third real data point
alongside the existing calibration fixtures — see `diarize_local.py`'s
comments on both constants for the full before/after numbers and what was
re-verified before changing them (every real, checked-in, listenable
fixture: openai, gptaudio, family_real, poker6 — NOT the two fixtures that
turned out to be invalid/unverifiable, see below).

**Result: poker6 now measures 6/6 = 100% exact per-turn accuracy**, zero new
dependencies, zero latency/cost increase, zero manual speaker-count input
(pinned as `test_ecapa_clustering_poker6_fixture_full_accuracy` in
`server/tests/test_diarize_regression_ladder.py`).

**A side-finding worth keeping**: two tests were "protecting" against a
regression using fixtures that turned out to be compromised —
`tmp/test_recording.wav`, which `server/tests/fixtures/audio/README.md`
already documented as invalid for diarization testing (physics-modulated
gain/tempo on ONE flat TTS voice, not real acted speech — "Deepgram Aura...
cannot act"), and a synthetic unit test whose numbers partly traced to a
"couple recording" and "3-person recording" that don't exist as files
anywhere in the repo, only as numbers in old comments. Both were repaired
(repointed to the real, checked-in `gptaudio.wav`; recalibrated to the new
thresholds) rather than deleted — see `server/tests/test_diarize_local_live.py`
and `test_diarize_local.py::TestKSelection::test_unanchored_split_does_not_become_a_third_speaker`.

## pyannote.audio evaluation (round 4, investigated then abandoned)

Before finding the threshold root-cause, this round also evaluated
pyannote.audio 3.3.2 as a wholesale replacement engine (prototype:
`diarize_pyannote_prototype.py`, results: `pyannote_result_*.json`,
`pyannote_summary.json`, eval scripts: `eval_pyannote.py`,
`eval_production_metric.py`, `tune_one.py`).

**Findings:**
- Needs its own pinned dependency set incompatible with `requirements-voice.txt`'s
  speechbrain setup (see `requirements-pyannote.txt`'s comment for the exact
  torch/torchaudio/huggingface_hub version conflicts) — a real deployment
  cost (~800MB+, a second HF-gated model, an HF_TOKEN).
- With `clustering.min_cluster_size` tuned from pyannote's default (12) down
  to 3 (short clips don't have enough embedding windows per speaker to clear
  a higher floor), it DID correctly auto-detect all 6 poker6 speakers with
  no speaker-count hint needed — matching the "never require manual input"
  constraint. Per-utterance accuracy: 83.3% (worse than the 100% the
  threshold fix achieves).
- Auto-detection at that same tuning badly over-segments a normal 2-speaker
  recording (`family_real.wav`: 5 phantom clusters, 25% accuracy) — a real
  regression risk for the app's primary 2-4-person use case. Not fixed by
  bounding min/max speakers.
- **Abandoned in favor of the threshold fix**: same or better accuracy,
  zero new dependencies, zero latency/cost increase, no risk to the
  already-100%-accurate 2-4-speaker path. Kept here for reference in case a
  genuinely hard future case needs a real second embedding model as an
  automatic cross-check (see the parent conversation's "second opinion
  model" discussion) — this prototype is a reasonable starting point for
  that, not a dead end.
