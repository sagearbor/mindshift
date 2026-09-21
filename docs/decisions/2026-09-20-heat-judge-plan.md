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

   > **Shipped 2026-09-20 at N = 3, not 5** — the fallback this item already
   > names, taken as the default. Dose: AMI 97 → 46.4/h, SBCSAE 80 → 36.1/h.
   > The reason is the cost side, which was not measurable until the chain was
   > built: at 5 s a +29.8 dB shout lasting 2.4 s and a 3-second raised voice
   > both go unbuzzed, and the scripted-scene and RAVDESS fixtures show real
   > escalations starting to disappear. 3 s takes roughly half the available
   > reduction and keeps the cue responsive; **5 s is a one-line change** —
   > `MINDSHIFT_HEAT_HOLD_S=5` on the server, `HEAT_HOLD_S` on phone and watch
   > — and the dose gate measures both. Full write-up and the measured costs:
   > `docs/plans/2026-09-20-hold3-hysteresis.md`.
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

**Step 7 result (2026-09-19):** the official CHiME-6 transcript JSONs
(`CHiME6_transcriptions.tar.gz`, OpenSLR resource 150) fixed the labels —
speech share is now balanced (max 18.1% of a participant, not 85%) and
overlap turns out to be much higher than assumed (43.5% vs AMI's 8.2%), which
changes the step-4/5 overlap-gate expectation for a dinner-party setting: see
"CHiME-6 corrected (transcript truth)" in
`docs/plans/2026-09-19-heat-map-rounds.md` for the full numbers. Nothing here
changes steps 1–6, 8 or 9 — CHiME-6 was never part of the hysteresis/tone-model
gates, only of the identity/overlap ceiling measurements on the graph.

## Step 2 done (2026-09-20, worktree-agent-ad96e8f5269cd684b)

Baked `odyssey_dim` + `iemocap` into the `INSTALL_VOICE=1` image layer and
flipped Cloud Run `MINDSHIFT_TONE_AUDIO` **off → dark** — not off → on.

**Deviation from the dispatched instructions, on file evidence, not opinion.**
The task this branch was given said to set `MINDSHIFT_TONE_AUDIO=on`, reading
it as "safe" because the only thing it drives is the PR #186 valence veto
(`docs/decisions/2026-09-17-valence-veto.md`), which sits behind its own,
separately-off flag (`MINDSHIFT_HEAT_VALENCE_GATE`). That undersells what
`tone_id.surface_allowed()` (`mode() == "on"`) actually gates:

1. `server/audio_pipeline.py`'s `_enrich_tone()` — when `surface_allowed()`,
   sends a real `ToneFlagEvent(source="audio")` over the websocket to the
   turn's own client AND fans it out to every other call participant
   (`server/calls.py:fan_out`). `apps/mobile/src/hooks/useAudioStream.ts`
   already has a live `type === "tone_flag"` handler — this channel is real
   and currently dormant, not inert. In "dark" mode the same classification
   runs and is logged, but this function returns `None` and nothing is sent.
2. `server/watch/relay.py`'s `tone_level()` — only reachable via (1) — folds
   the flag into the watch's `aggressive_tone` escalation vector when
   `confidence >= 0.5`. Inert today because `odyssey_dim` (dimensional) always
   reports `confidence == 0.0` by construction, but would go live the moment
   `MINDSHIFT_TONE_BACKEND` is ever switched to a categorical backend
   (`iemocap` / `superb_er`).
3. `server/live_sessions.py`'s `audio_tone_allowed()` (same gate) feeds
   `turn_tone_rows(..., audio_allowed=True)`, writing `audio_label` /
   `audio_escalated` into the STORED session `analysis.json` — the Growth /
   Replay / YourDay recap. This is a second, independent surfacing path with
   nothing to do with live nudges.

`tone_id.py`'s own module docstring (2026-09-07 status note) says the same
thing independently: the per-speaker delta has only been measured on ACTED
fixtures, and going "on" needs it re-confirmed against a REAL recording
(`server/tests/fixtures/audio/test_recording_family_real.wav`) first — exactly
what this decision doc's step 3 (arousal+valence confirm/veto, CONFER-gated,
"another agent... later") is scoped to do. Flipping to "on" today would
surface an unvalidated per-turn signal to real users on real calls, which is
what step 3 exists to make safe. So this deploy does what the table above
already says for step 2 — off → dark — and nothing more. Flagged to `main`
(the coordinating session) before routing traffic; agreed.

**What was built (independent of on vs. dark, needed either way):**

- `server/tone_id.py`: added `prefetch(name)` — the same `_snapshot()` calls
  each backend's loader makes, split out so they can run without constructing
  a model — plus a `python tone_id.py BACKEND [BACKEND ...]` CLI entry point.
- `Dockerfile`: inside the `INSTALL_VOICE=1` layer, `COPY server/tone_id.py`
  (only that file, not the full `server/` tree — keeps this layer cached
  across unrelated server changes) then `python tone_id.py odyssey_dim
  iemocap`, with `ENV MINDSHIFT_TONE_CACHE=/app/server/.tone_cache` set
  explicitly. Confirmed from the Cloud Build log: both backends printed
  `snapshot_present=True` at build time. `superb_er` deliberately NOT baked —
  not production-selectable today (see `tone_id.py`'s docstring: its delta is
  no better than volume alone); `server/tests/test_dockerfile_tone_prefetch.py`
  fails the build the moment that stops being true without a matching bake.
  ECAPA (`server/speaker_id.py`) needed no new step — it was already baked as
  a side effect of the pre-existing ECAPA-ONNX-export build step.
- `server/main.py`: added `_tone_status()` (`{mode, backend, weights_present}`,
  never fabricates `weights_present=True`) — logged in one line at startup
  (`tone_mode=... tone_backend=... tone_weights_present=...`) and added to the
  `/health` (`/healthz`) JSON under `"tone"`.
- `server/tests/test_dockerfile_tone_prefetch.py`: five tests pinning the
  Dockerfile text (every production-selectable backend named in the bake
  line, `MINDSHIFT_TONE_CACHE` set explicitly, the tone `COPY` precedes the
  full `COPY server/ ./server/`, and a backend added to `tone_id.TONE_BACKENDS`
  with no bake/exclude decision recorded fails loudly).

**Deploy record:** merged `eval/real-conversation-audit` into this worktree
branch first (coordinator instruction — this worktree had branched from
`main` at the same commit, not from `eval/real-conversation-audit` as
originally briefed; merge was clean apart from an add/add conflict on this
very file, resolved by keeping this section). Built via `gcloud run deploy
mindshift-api --source . --no-traffic --tag tone` (Cloud Build, same
mechanism `scripts/deploy_cloudrun.sh` uses) from the pre-existing revision
`mindshift-api-00089-jgr` (4 vCPU / 8Gi, `MINDSHIFT_TONE_AUDIO=off`, image
`sha256:4fe024ba...`). No `--set-env-vars` / `--update-env-vars` was passed to
the build step so every existing env var carried over unchanged onto the
tagged revision `mindshift-api-00093-mif` (image `sha256:e660f568...`) —
confirmed `weights_present=true` for both `odyssey_dim` and `iemocap` from its
own startup log line and `/health`. Env was then flipped with
`gcloud run services update --update-env-vars MINDSHIFT_TONE_AUDIO=dark`,
producing `mindshift-api-00094-nug` (same image `sha256:e660f568...` as
`00093-mif` — confirmed by digest — env diff against `00089-jgr` is
`MINDSHIFT_TONE_AUDIO: off -> dark` and nothing else). `00094-nug`'s startup
log: `tone_mode=dark tone_backend=odyssey_dim tone_weights_present=True`; its
`/health` matches. Traffic routed 100% to `mindshift-api-00094-nug`. See the
session report for the full verification trail.

## Step 5 built (phone), fitted 2026-09-19 — acoustic instant tier on device

Row 5 above is built and dark on the phone. What shipped is not "eGeMAPS on
device": openSMILE is a native dependency we will not add to the RN bundle, so
the question became *how few of its 88 functionals do we actually need, out of
the ones a person can honestly re-implement in TypeScript?*

**Files.** `scripts/instant_tier_select.py` (selection, fit, fixtures),
`apps/mobile/src/live/instantTier.ts` (the extractor + scorer, 660 lines, no
native deps), `apps/mobile/src/live/instantTier.model.json` (the fitted
numbers), `scripts/instant_tier_eval.ts` (the real-audio gate), and three jest
suites: `instantTier.test.ts`, `instantTier.parity.test.ts`,
`instantTier.eval.test.ts`, `instantTier.replay.test.ts`.

### The subset

Greedy forward selection over the 46 implementable eGeMAPS functionals,
maximising angry-vs-HAPPY AUC under speaker-grouped 5-fold CV **on CREMA-D
alone** — the cross-corpus numbers below were never seen during selection.
Formants and H1-H2/H1-A3 were excluded from the pool on purpose: LPC root
solving and harmonic peak picking are not things we want to write twice and
keep in sync across three runtimes.

Sixteen descriptors, eight of them loudness shape:

| | descriptor | why it is cheap |
|---|---|---|
| 1–8 | loudness mean, 20th pctl, 20–80 range, stddevNorm, mean+sd of rising slope, mean+sd of falling slope | one compressed mel-band sum per frame |
| 9 | F0 80th percentile (semitones from 27.5 Hz) | YIN over the 55–500 Hz lag range |
| 10 | spectral flux (mean) | L2 distance between consecutive magnitude spectra |
| 11–12 | MFCC 2 and 4 (mean) | the mel bank we already computed, plus a 4-row DCT |
| 13–14 | alpha ratio, voiced and unvoiced | two band sums in dB |
| 15 | Hammarberg index, unvoiced | two band peaks in dB |
| 16 | spectral slope 0–500 Hz, voiced | a least-squares fit over 17 bins |

Candidates measured (cross-corpus CREMA-D → RAVDESS, angry vs happy):

| subset | k | CREMA-D CV | xcorp all | **xcorp angry-vs-happy** | recall @5% FA |
|---|---|---|---|---|---|
| loudness only (shipped instant signal) | 1 | 0.788 | 0.831 | **0.713** | 0.42 |
| greedy-8 | 8 | 0.830 | 0.848 | **0.779** | 0.48 |
| greedy-12 | 12 | 0.840 | 0.884 | **0.832** | 0.52 |
| **greedy-16 (chosen)** | 16 | 0.843 | 0.891 | **0.842** | 0.56 |
| greedy-20 | 20 | 0.844 | 0.867 | **0.828** | 0.52 |
| all 46 implementable | 46 | 0.841 | 0.879 | **0.828** | 0.49 |
| eGeMAPS-88 (reference ceiling) | 88 | 0.854 | 0.867 | **0.832** | 0.51 |

Sixteen hand-writable descriptors beat all 88. That is not a subset trick — it
is what 88 correlated descriptors do when the train and test corpora have
different rooms: the extra 72 fit CREMA-D, not anger.

### What the phone actually measures

25 ms frames / 10 ms hop for energy and spectrum, a 512-point radix-2 FFT
(the app's first — there was no FFT anywhere in `apps/mobile` before this),
26 mel bands 20–8000 Hz, and YIN with an absolute CMND threshold on a 25 ms
window every 20 ms for F0 and the voiced/unvoiced split.

Two definitions were got wrong first and fixed by measurement, not by reading:

- **spectral flux** on energy-normalised spectra tracks openSMILE at ρ 0.17;
  on raw magnitudes, ρ 0.999. Most of what the descriptor carries is how
  violently the loudness itself is moving.
- **spectral slope 0–500 Hz** fitted on linear magnitude tracks at ρ 0.31; in
  dB, ρ 0.907. Tilt lives in the log domain.

Per-descriptor agreement with openSMILE on the identical 2 s windows now runs
0.895 → 1.000 (`feature_fidelity_spearman_vs_opensmile` in the model file).
20 ms frames were tried to match eGeMAPS's LLD geometry exactly and were
slightly *worse* on two descriptors, so 25 ms stayed.

### The coefficients, and why they are a distillation

Two fits were measured on the same windows before one was picked:

| | cross-corpus angry-vs-happy | angry-vs-all | rank agreement with the reference |
|---|---|---|---|
| fit the anger labels directly | 0.795 | 0.854 | 0.958 |
| **ridge-fit the openSMILE reference model's logit** | **0.795** | **0.857** | **0.974** |

The second ships. Same accuracy, and it says the honest thing about what the
phone is doing: not learning anger a second time off a hand-rolled feature
set, but carrying the model the bench selected as far as a pure-TypeScript
extractor can carry it. The accuracy claim never rests on that choice — it
rests on RAVDESS, which neither fit ever trains on.

### Measured, end to end, through the shipped TypeScript

`npx tsx scripts/instant_tier_eval.ts` decodes every RAVDESS angry+happy clip
and 500 CREMA-D ones from the corpora and runs `instantTier.ts` over them.
Result, committed as `apps/mobile/__tests__/fixtures/instantTier.eval.json`
and gated in jest at ≥ 0.78:

- **cross-corpus angry-vs-happy AUC 0.795** over 384 RAVDESS clips, 0 unscored
  (loudness alone: 0.72 on the same split; openSMILE-88: 0.87)
- within-corpus 0.840 over 500 CREMA-D clips — the corpus it was fitted on,
  labelled as such and not a generalisation claim
- rank parity with the openSMILE reference: **Spearman 0.942** over a 147-clip
  fixture spanning both corpora, three emotions and 79 speakers; the fixture's
  own angry-vs-happy AUC is 0.890
- **latency 10.6 ms per 2 s window** (node v26, dev Mac; 200 windows benched)
  against a 40 ms budget

The gap between the openSMILE subset (0.842 on whole clips, 0.820 on the same
2 s windows) and the phone (0.795) is the price of hand-written extraction.
Roughly half of it is the 2 s window itself, half the extractor.

### Wired dark

`fastLoop.tickHeat()` scores the trailing 2 s once a second on the VAD queue —
only windows that contain speech, because an anger score over room tone is a
number, not a measurement. Each one reaches `deps.onHeat`, `ReplayResult.heat`
and, summarised as a per-turn maximum, `LocalTurn.instantHeat` →
`FragmentRow.instantHeat` / `TurnRow.instantHeatMax` in the nudge report, right
beside the loudness rung the ladder actually fired on. **Nothing escalates on
it.** `deps.instantHeat: false` turns the arithmetic off entirely. The scene
pack's nudge pins are unchanged, which is itself asserted.

### What a Kotlin port would need

The watch column in row 5 stays NO until this is ported. The port is the
extractor, not the model — `instantTier.model.json` is 16 names, 16 means, 16
sds, 16 coefficients and an intercept, and it is read at runtime, so Kotlin
loads the same file. What has to be written again:

1. A radix-2 FFT (512-point, real input) — or `JTransforms`/`KissFFT` via NDK.
2. The mel filterbank (26 bands, 20–8000 Hz) and a 4-row DCT-II.
3. YIN with the cumulative mean normalised difference and parabolic
   interpolation, 55–500 Hz.
4. The functionals: percentiles with linear interpolation, `stddevNorm`, the
   3-frame moving average, and the local-extrema rising/falling slope rule.
   These are where a port silently diverges; they carry four of the sixteen
   descriptors.
5. The null discipline: an unmeasurable descriptor must land on the training
   mean (z = 0), never on 0.0.

The parity fixture is the port's gate, not a new corpus run: the six clips
carrying base64 PCM give byte-identical input, and Kotlin must land within
1e-4 of the stored feature vectors and reproduce the fixture's Spearman.
Budget the same 40 ms; Kotlin on a wrist-class CPU should beat the phone's
10.6 ms of JavaScript, but that is an assumption until someone measures it.

---

# Step 3 built (2026-09-20)

`server/watch/heat_judge.py`, calibrated by `scripts/heat_judge_calibrate.py`,
pinned by `server/tests/test_heat_judge.py` (44 tests). The phone's relayed
audio is buffered into a rolling **2 s window, scored once per second**,
exponentially smoothed over the last **3** scores, and the verdict confirms or
vetoes the loudness ladder's escalation.

## The rules, and the number behind each

| constant | value | evidence |
|---|---|---|
| `WINDOW_S` / `HOP_S` | 2.0 s / 1.0 s | 130 ms of CPU per 2 s clip (measured); 1 s is `SentinelDetector`'s own window cadence, so a verdict always lines up with the dB window that asked for it |
| `SMOOTHING_N` / `ALPHA` | 3 / 0.5 | weights 0.571 / 0.286 / 0.143 — the current window still dominates, but one odd window cannot flip a verdict on its own |
| `MIN_SCORES` | 2 | under this the judge says `unknown`, and `unknown` escalates |
| `VALENCE_VETO_MAX` | 0.48 | inherited unchanged from PR #186 (leave-one-speaker-out over 85 held-out CREMA-D speakers) |
| `CALM_AROUSAL_FLOOR` | **0.40** | highest floor that still keeps ≥ 90% of over-rung acted anger |
| `CONFIRM_AROUSAL` | **0.78** | above the 90th percentile of over-rung acted HAPPY arousal (0.756), far below CONFER's human-rated heated mean (0.839 ± 0.048) |
| `JUDGE_WAIT_S` | 2.0 s | one hop plus one model call, rounded up; only the FIRST rung ever waits |

**Verdict order — and the one place this departs from the brief.** `confirm`
is evaluated **before** the valence veto, not after it. Measured reason: the
inherited 0.48 valence bar, applied veto-first, vetoes **20.6%** of CONFER's
human-rated heated windows, and *no choice of either arousal knob can move that
number* because the veto never consults them. Letting a window at CONFER-heated
activation escape a borderline pleasantness reading takes it to **8.8%** and
costs 0.7 points of the happy veto. Pinned by
`test_the_rule_order_is_what_the_confer_numbers_chose`, against the corpus that
measured it.

## Calibration (`scripts/heat_judge_calibrate.py`)

Three gates at once; **77 of 1,140** threshold pairs clear all three. Chosen:
the pair vetoing the most acted HAPPY.

| gate | bar | at (0.40, 0.78) |
|---|---|---|
| CONFER human-rated heated windows vetoed | ≤ 10% | **8.8%** (89.7% confirmed) |
| over-rung acted HAPPY vetoed | ≥ 50% | **56.9%** |
| over-rung acted ANGRY vetoed (PR #186's recall gate) | ≤ 10% | **9.6%** — 90.4% of anger kept |

CONFER calm windows confirm at 26.7% against heated at 89.7% (separation 0.63).

Corpora, and the caveats kept in the open:

* **CONFER** — 345 scored 5 s windows over 24 TV-debate recordings, 68 heated
  (ten raters' mean conflict ≥ 400/1000). **Zero** of those 68 are also ≥ +6 dB
  over baseline, i.e. the shipped ladder nudges on none of them — consistent
  with round 2's "0 of 25 human-rated spans caught". So the CONFER gate bounds
  a cost the ladder cannot currently incur; it is kept as the guard against a
  future ladder that can, and as the only human-rated y-axis we have.
* **CREMA-D** — the same two artefacts PR #186 used (`tmp/corpora/valence_probe.json`
  + `tmp/corpora/heat_rubric_cremad.csv`), so the numbers are directly
  comparable: 116 happy and 208 angry clips clear the +6 dB rung. Valence alone
  = 60.3% happy / 9.1% angry, exactly reproducing that record. (The feature
  bank carries the same dims but its own dB estimator, which puts 99 happy
  clips over the rung instead of 116 — worth knowing before anyone re-derives
  these numbers from `tmp/feature-bank/` and finds different ones.)
* The cached reference windows are **5 s** and the judge runs on **2 s**. Both
  are inside the model's trained span, and the smoother gives the judge ~4 s of
  context, but the thresholds should be re-checked against real sessions once
  `dark` has accumulated `arousal`/`valence` series from live lanes — which is
  exactly why those series are now persisted (below).
* CONFER is smoothed along contiguous runs (a time series, which is what the
  judge sees); CREMA-D clips are treated as one steady state (the judge's three
  windows would all be that clip). The asymmetry biases the calibration toward
  vetoing more acted-happy than heated conversation — the safe direction.

## Dose: AMI and SBCSAE, replayed

Full audio, shipped nudge chain, gate built from the cached 5 s tone series
(unscored windows left OPEN — fail-open, same rule as the shipped veto):

| corpus | n | median buzzes/h before | after | fell | rose |
|---|---|---|---|---|---|
| AMI | 12 | 97.0 | **72.0** | 11 | 1 (`IS1000a`, 102.4 → 104.6) |
| SBCSAE | 60 | 80.2 | **72.2** | 50 | 6 |

A recording *can* come out marginally worse: removing an escalation can leave
the ladder at a lower level that later re-escalates — a second first-rung buzz
with a fresh PRD §6 reminder cycle, where the ungated replay sat at one high
level and reminded slowly. `test_the_judge_gate_never_raises_the_dose` bounds
this: median must not rise, total must not rise, risers must stay under a
quarter of fallers.

This does **not** reach the 12/h product target on its own; step 1's hold-3s
hysteresis is still the dominant lever (AMI 97 → 18/h).

## Mode × rung × verdict → action

`MINDSHIFT_TONE_AUDIO` is the only switch. **Production is `dark` now; the
merge deploy sets `on`.** Everything that changes at that flip is in this table.

### Wrist — the relayed loudness lane (`watch/relay.py`)

| mode | rung | verdict | action |
|---|---|---|---|
| `off` | any | — (no judge) | escalate exactly as today |
| `dark` | any | computed + logged | escalate exactly as today — **nothing the wearer can feel changes** |
| `on` | 1 (+6 dB) | `confirm` | escalate, no wait |
| `on` | 1 | `unknown` | **wait up to 2 s** for a verdict, then escalate anyway |
| `on` | 1 | `veto` | **no escalation this window** |
| `on` | 2 (+10 dB) / 3 (+14 dB) | `confirm` / `unknown` | escalate immediately — higher rungs are **never delayed** |
| `on` | 2 / 3 | `veto` | no escalation (vetoed, but never delayed) |

`aggressive_tone` — the words — is never touched, in any mode. A model that is
missing, cold, or broken yields `unknown`, which escalates: **a model outage
can never mute the product.** The waited escalation keeps its ORIGINAL stream
clock, so `NudgePolicy`'s cooldown arithmetic still sees the moment the ladder
crossed its rung.

### Phone + in-call surfacing (`audio_pipeline._enrich_tone`)

| mode | backend | verdict | `tone_flag` to the phone, call fan-out, and on to the relay |
|---|---|---|---|
| `off` | any | — | nothing computed |
| `dark` | any | computed + logged | **never sent** |
| `on` | dimensional (`odyssey_dim`, the default) | `confirm` | sent |
| `on` | dimensional | `unknown` / `veto` | **not sent** — logged only, exactly like `dark` |
| `on` | categorical (`superb_er`, `iemocap`) | n/a | sent, unchanged — a 4-class softmax has no arousal/valence axis, so the judge cannot have an opinion about one, and gating it to death would switch off a working configuration nobody asked to change |

This gate **fails closed** for a dimensional reading: no judge module, or a
judge that raised, means no surfacing. Showing a user "you sounded angry" is a
claim the server has to be able to support; escalating the wrist without a
verdict only costs a buzz they can attribute to their own volume.

Note the practical consequence of `CONFIRM_AROUSAL = 0.78`: roughly a quarter
of over-rung acted anger reaches it, so surfaced tone flags will be **rare**.
That is the intended posture for a first `on` deploy, not an accident.

### Stored analysis (`live_sessions.turn_tone_rows` → Growth / Replay / YourDay)

| `audio_allowed` | flag's `heat_verdict` | `audio_label` | `audio_escalated` |
|---|---|---|---|
| False | any | `None` | `None` |
| True | `confirm` | the model's label | per the label |
| True | `unknown` / `veto` / absent | the model's label | **False** |

The verdict travels on the flag's `scores` dict (`heat_verdict`: −1 veto /
0 unknown / +1 confirm), so a session stored tonight can still be told apart
from an unjudged one months from now. Dims are always kept; only the
escalation verdict needs a confirm.

### `MINDSHIFT_HEAT_VALENCE_GATE` — leave it OFF

The judge **supersedes** PR #186's per-turn `valence_veto`. Both read the same
valence axis at the same 0.48 bar, but the judge reads a rolling 2 s window
instead of one whole turn and carries the arousal floor as well, so when it has
an opinion the per-turn veto has nothing to add. One mechanism, not two: the
deploy must **not** set `MINDSHIFT_HEAT_VALENCE_GATE`. (`valence_veto` stays in
the code and stays tested — it is still the fallback for a build with no judge.)

`relay.tone_level` is inert for dimensional backends by construction: the
dimensional result reports `confidence 0.0`, below `TONE_FLAG_MIN_CONFIDENCE`,
so flipping the flag to `on` cannot make AUDIO tone drive the `aggressive_tone`
lane. Pinned by `test_tone_level_stays_inert_for_a_dimensional_backend`.

## Where the judge runs, and what it costs

Attached by `relay.register_live_session` — the judge exists to confirm or veto
a buzz on a **wrist**, so it runs exactly while a wrist is live, and never for a
phone-only session. It is also a no-op when the pinned snapshot is not on disk
(a filesystem check, no download): a server that would have to fetch 1.3 GB
first gets no judge rather than a queue of stalled windows, which makes step 2's
baked image layer a hard prerequisite for step 3 rather than a nicety. One score
at a time — a model slower than the hop drops windows rather than queueing them,
because a 4 s-old verdict is worse than no verdict. Scoring runs on a worker
thread; the phone's receive loop only appends bytes. One structured log line per
verdict (arousal, valence, verdict, latency ms), which in `dark` mode IS the
feature.

## Per-second series, persisted

`watch/routers/ws.py`'s `series` now carries `arousal`, `valence` and `judge`
beside `rms_db` / `hr_bpm` / `hr_t` — one sample per second, the judge's whole
stream and not just the moments it acted. `judge` is a code (−1 / 0 / +1)
because `LiveSession.series` is typed `dict[str, list[float]]`; the ordering
veto < unknown < confirm is meaningful and plots. The keys carry no `_series`
suffix, matching the existing `rms_db` / `hr_bpm` convention. No keys at all are
added when the flag is off or nothing was judged, so a stored document stays
byte-identical to today's.

This is the answer to item 4 of the 2026-09-17 record's "what I would need to
be more confident": live shadow data, so the next calibration can use real
sessions instead of CONFER and CREMA-D.

## Step 4 (SpeechBrain) — designed for, deliberately not wired

`tone_id.angry_vote` / `stacked_heat_score` landed on a parallel branch while
this was being built. The judge stores the full dims dict per window and
`verdict_from` ignores unknown keys, so the second vote is a scorer change plus
one constant. It is **not** wired here for one measurable reason: the cached
CONFER reference series carry arousal and valence but **no dominance**, and
`stacked_heat_score` needs all three — so the stack cannot be evaluated against
the human-rated corpus without re-running the model over 24 recordings. Wiring
a signal that has not been measured on the strongest y-axis in the repo is
exactly what the rest of this document argues against. The `dark`-mode series
above will supply that data.

## Reproduce

```bash
python scripts/heat_judge_calibrate.py --dose     # every number above
pytest server/tests/test_heat_judge.py -q         # 44 tests
```

The corpus-scale tests skip cleanly without the gitignored `tmp/` data (that
includes CI); the rules, thresholds, wiring, degrade and latency all run
everywhere.

## Shipped 2026-09-20 (overnight, integration branch `ship/heat-judge-20260920`)

| what | where | state |
|---|---|---|
| Server: hold-3s ladder + heat judge + baked weights, `MINDSHIFT_TONE_AUDIO=on` | Cloud Run `mindshift-api-00096-ruy` (image from build 39f7c886) | **serving 100%**; `/health` → `tone: mode=on, weights_present=true`. Rollback: `00094-nug` (dark, same weights) or `00089-jgr` (pre-tone). |
| Phone: hold-3s, instant tier (dark), iOS config, versionCode 38 | EAS production build `01cb945c` | building → `eas submit` lands a Production **draft**; owner rolls out in Play Console |
| Watch: 0.5.0 (vc16) hold-3s, praise, HR logging, reminder back-off | Play `com.sagearbor.gauge.wear`, track `wear:production` | **uploaded as draft**; owner reviews listing title / data safety, then rolls out |
| Watch: 0.5.1 (vc17) — targetSdk 35 for Play (2026-09-21) | Play `com.sagearbor.gauge.wear`, track `wear:production` | **uploaded as draft** (replaces vc16 draft); Play's API-35 rejection fixed, no manifest/permission changes needed — `compileSdk`/`targetSdk` bump alone compiled clean, 582 tests pass |

Merged-tree suites: pytest 2,267 + 44 judge tests (one live Deepgram diarization
test fails identically on the untouched eval branch — Deepgram-side synthetic-voice
limitation), jest 2,181, watch `:shared:jvmTest :wearApp:testDebugUnitTest` 582
(JDK 17 via `/opt/homebrew/opt/openjdk@17`; the Android Studio JBR is JDK 25 and
Gradle 8.9 refuses it). Not shipped: step 5's escalation policy (instant tier
stays dark until the judge has real-session data); step 6 student (training on
CPU tonight; GPU quota request `gpus-all-regions-1` pending on arborfam-hub).
