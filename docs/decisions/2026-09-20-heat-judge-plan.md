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
