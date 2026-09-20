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
| 7 | Regenerate CHiME-6 speaker labels from transcripts | — | — | offline | ~2 h |
| 8 | $50 OpenAI top-up; bench gpt-audio as labeller | NO | NO | 3–5 s, offline only | ~3 h after credits |
| 9 | Review and push eval/real-conversation-audit | — | — | — | ~30 min + owner review |

"Dark" is the middle of a three-state flag (`MINDSHIFT_TONE_AUDIO`): **off** =
not computed; **dark** = computed and logged beside every session, never
changes a buzz; **on** = drives the confirm/veto. Step 2 goes off → dark so
real-session data accumulates with zero user-visible risk; step 3 goes dark → on.

## Step 4 built (2026-09-20)

`server/tone_id.angry_vote(pcm, sr) -> float | None` and
`stacked_heat_score(dims, angry_p) -> float` (worktree
`worktree-agent-a6a390549be893455`, branch merged from
`eval/real-conversation-audit`). The judge agent calls these two functions:

```python
def angry_vote(pcm: np.ndarray, sr: int = TARGET_SR) -> float | None: ...
def stacked_heat_score(dims: dict, angry_p: float | None) -> float: ...
```

`angry_vote` loads the `iemocap` backend directly (bypassing
`MINDSHIFT_TONE_BACKEND`) so it and the default `odyssey_dim` backend are
BOTH resident in the same process at once — `_load_model` already caches per
backend name, so no structural change was needed there, just calling it with
an explicit name. Formula: `P(angry) - P(happy)` on the IEMOCAP 4-way
softmax (range `[-1, 1]`); `None` — never fabricated — when the flag is off,
deps are missing, or the pinned snapshot isn't cached, logged ONCE per
process. `dims` is the `{"arousal","dominance","valence"}` dict odyssey_dim
already produces.

**HF repo id + revision to bake** (confirmed against `BACKEND_INFO` in
`server/tone_id.py`, unchanged from step 2/3):

```
speechbrain/emotion-recognition-wav2vec2-IEMOCAP @ 117a9c3dff08be81a3628eecf6a66b547ec1659b   (iemocap)
facebook/wav2vec2-base @ 0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8                              (iemocap's backbone config pin)
3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes @ 00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9      (odyssey_dim, already baked)
microsoft/wavlm-large @ c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c                                (odyssey_dim's backbone config pin)
```

**Coefficients** (fit 2026-09-19 by `scripts/fit_stacked_heat.py` on
`tmp/feature-bank/{index,tone,sbiemocap}.parquet`, 2,940 CREMA-D+RAVDESS
clips; StandardScaler + LogisticRegression(C=3.0), trained on all of
CREMA-D's 6 emotions, folded into raw-unit coefficients — see the script for
the scaled-space numbers):

```
STACKED    score = sigmoid(-7.357533 - 24.091263*arousal - 4.679289*valence + 35.493496*dominance + 0.828883*angry_p)
DIMS-ONLY  score = sigmoid(-8.198175 - 26.743169*arousal - 5.105053*valence + 41.252174*dominance)   # angry_p unavailable fallback
```

**Gate** (`server/tests/test_stacked_heat.py`, cross-corpus: train CREMA-D,
test RAVDESS — same protocol as `scripts/feature_bench.py`):

| combo | auc(angry-vs-happy) | recall@5%FA | gate |
|---|---|---|---|
| STACKED (dims + angry_p) | 0.935 | 0.651 | ≥0.93 / ≥0.62 ✓ |
| DIMS-ONLY fallback | 0.926 | 0.615 | ≥0.88 ✓ |
| odyssey_dim alone (step 3, reference) | 0.897 | 0.590 | — |
| iemocap alone (step 1 round, reference) | 0.867 | 0.500 | — |

Plus a 20-row fixture (`server/tests/fixtures/stacked_heat_fixture.json`)
pinning that the hard-coded constants reproduce the fit exactly
(`test_stacked_heat_score_reproduces_the_fixture`).

**Smoke test** (`test_smoke_both_backends_resident_angry_vote_and_classify_pcm`):
both backends loaded and classified 4 real clips (CREMA-D speaker 1003
angry/happy/neutral + one RAVDESS angry clip) in one process.
`angry_vote`: angry=1.000, happy≈0.000, neutral≈0.000, RAVDESS angry=1.000
(speaker 1001 was tried first and rejected for this smoke test — its angry
and happy clips both saturate to ~1.0, the round-1 "reads voice identity"
failure the module docstring already documents; speaker 1003 shows the
clean split).

**Latency** (`test_latency_angry_vote_2s_and_5s_clips`, CPU, `torch.
set_num_threads(2)`, one call after a one-time warm-up):

| clip length | measured |
|---|---|
| 2 s | 59.1 ms |
| 5 s | 128.6 ms |

Within the "~50-100 ms" estimate at 2 s; the 5 s clip runs a bit past 100 ms
(transformer attention scales with sequence length) but stays comfortably
under the ~3 s latency budget step 3/4 already committed to in the table
above.
