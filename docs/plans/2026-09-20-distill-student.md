# Distilling a watch-sized "heat" student — 2026-09-20

Step 6 of `docs/decisions/2026-09-20-heat-judge-plan.md`'s roadmap ("distil
our own Wav2Small-class student (MIT/Apache teachers only) for the watch —
published Wav2Small weights do not exist and are NC-licensed"): teacher-label
our real-conversation audio with the two cloud models already in
`server/tone_id.py`, train a first tiny CPU student on those labels, and
measure it honestly against the teacher and the shipped signal. No GPU
rental (a quota request is pending — see *What a GPU run would add*, below),
no push, no deploy. Everything here is a first pass, time-boxed by a CPU-only
worktree; the verdict at the end says plainly whether it earns the GPU.

## Teacher and licence

Two models already resident in `server/tone_id.py`, both used for every
other step of the heat-judge plan — never audeering's CC-BY-NC-SA model:

| model | licence | output used |
|---|---|---|
| `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes` (`odyssey_dim`) | MIT | `classify_pcm(pcm, 16000)["scores"]` → arousal / dominance / valence, each ≈0..1 |
| SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP` (`iemocap`) | Apache-2.0 | `angry_vote(pcm, sr)` → P(angry) − P(happy) ∈ [−1, 1] |

`stacked_heat_score(dims, angry_p)` folds both into one 0..1 "how heated"
number via the fixed logistic fit already hard-coded in `tone_id.py` (2026-09-19,
`scripts/fit_stacked_heat.py`). That's the per-window label distilled below.

## Data: `scripts/distill_teacher_labels.py`

2 s windows, 1 s hop, gated on speech: a manifest that carries per-speaker
intervals (AMI, SBCSAE, CHiME-6) uses them directly (≥50% of the window must
be someone talking); CONFER (no speaker truth) and CREMA-D (single-utterance
clips, no manifest at all) fall back to the −45 dBFS silence floor
`scripts/conversation_audit.py`'s `SentinelDetector` already ships. Per
recording: one parquet (`window_start`, `arousal`, `valence`, `dominance`,
`angry_p`, `heat`, `rms_db`, `audio_path`), written atomically and skipped on
a re-run the moment it exists — resumable at recording granularity, the same
pattern the corpus loaders (`ami_corpus.py`, `sbcsae_corpus.py`, ...) use.

**Breadth over the brief's literal order.** The brief's priority was "SBCSAE
+ CHiME-6 + CONFER + CREMA-D, AMI last" — but SBCSAE alone is ~23 h of audio,
enough on its own to consume the whole 3 CPU-hour budget (measured ≈190 ms/
window: `classify_pcm` ~130 ms + `angry_vote` ~59 ms,
`docs/decisions/2026-09-20-heat-judge-plan.md`) and leave CONFER — the one
corpus with human-rated ground truth — and CREMA-D — the one corpus with
true angry/happy labels — untouched. So recordings are scheduled
ROUND-ROBIN across corpora in that stated priority order, one recording at a
time, so every corpus gets *some* coverage before any one corpus can exhaust
the clock. Deviation stated on file evidence, not opinion, per this repo's
convention (see `docs/decisions/2026-09-20-heat-judge-plan.md`'s own step 2
write-up for the same discipline).

### What actually got labelled

Six foreground invocations (the 600 s/10 min-per-command ceiling this
worktree runs under, plus two deliberately killed early — see below), ~0.36
logged CPU-hours of successful labelling out of the 3-hour budget (real
elapsed time across all invocations, including the two killed ones, is
roughly double that; a killed run's IN-PROGRESS recording is discarded
cleanly — no partial parquet, confirmed by inspecting for `*.tmp.parquet`
leftovers after each kill — but its wall time isn't credited to the
resumable progress log, only completed recordings are):

| corpus | recordings labelled | of total | windows | ≈ hours of speech |
|---|---|---|---|---|
| SBCSAE | 3 | of 60 | 4,514 | 1.25 h |
| CHiME-6 | 2 | of 18 | 1,680 | 0.47 h |
| CONFER | **24** | **of 24 (complete)** | 1,750 | 0.49 h |
| CREMA-D | 942 | of 7,442 | 1,287 | 0.36 h |
| AMI | 1 | of 12 | 967 | 0.27 h |
| **total** | **972** | | **10,198** | **2.83 h** |

CONFER was finished completely (6 → 24 of 24) in a dedicated follow-up round
once the metrics below made its partial coverage the obvious next move (see
*Metrics*, below — a 6-clip sample of CONFER's human-rated agreement turned
out to disagree noticeably with the full 24-clip number, which is exactly
the kind of thing a 3-hour budget spread thin across five corpora produces,
and exactly why this doc reports the FULL-coverage numbers, not the
first-pass ones, everywhere below).

**Two rounds were killed deliberately, not crashed.** AMI meetings run
30–40 min of real audio — at ~190 ms/window that is ~7–8 minutes of teacher
inference for ONE recording, and the round-robin scheduler (by design: it
advances one recording per corpus per pass, so it can't check the time
budget again until the whole pass finishes) can't interrupt a recording
mid-flight. Two rounds ran well past their `--max-seconds` budget stuck
finishing a single AMI or SBCSAE recording; both were killed from the
outside once that became clear, and the atomic write design (tmp file,
rename only on completion) meant nothing was lost or corrupted — confirmed
by checking for stray `.tmp.parquet` files after each kill (none). The
lesson worth keeping: the round-robin buys BREADTH across corpora but not
BOUNDED latency per pass when corpora differ this much in per-recording
cost; a future version could budget per-corpus TIME per pass rather than
one recording per pass.

**CREMA-D got its own dedicated round** (`--corpora cremad`) after the
round-robin's "one recording per corpus per pass" rule proved too coarse
for a corpus this cheap and this valuable (true angry/happy labels, the
whole point of the RAVDESS comparison) — 904 clips in 280 s, 12 actors
(1001–1012), full six-emotion coverage. This is also where a real
correctness bug was caught before it reached training: `distill_student.py`
originally split train/val by RECORDING, which for CREMA-D means by CLIP —
letting the SAME actor's voice appear in both train and val (different
emotion, same speaker). Fixed to split by ACTOR ID for CREMA-D specifically
(see `build_index`'s `speaker_key` column) before the real training run
below used this data.

**What's left, if this were resumed**: SBCSAE 57, CHiME-6 16, CREMA-D 6,500,
AMI 11 recordings (CONFER now complete) —
`python scripts/distill_teacher_labels.py` is idempotent and resumable at
recording granularity, so continuing this is purely a matter of more
foreground time.

## Student: `scripts/distill_student.py`

A Wav2Small-class network (arXiv 2408.13920, Kounadis-Bastian et al.):
strided-conv front-end on raw 16 kHz PCM (downsamples ~32×), three
depthwise-separable conv blocks (each halves the time axis again),
attention pooling over whatever time length remains, a shared trunk, four
heads (arousal / valence / dominance regression, one heat classification
logit). Global pooling means the SAME graph accepts any input length — the
parity fixture below exploits this to stay small.

Channel widths (48, 72, 104, 152, hidden 192) were chosen by grid search over
a handful of configurations to land within 0.1% of the paper's 72 K-param
target — measured, not asserted:

**72,077 parameters** (`count_params()`, measured, not estimated) — channels
(48, 72, 104, 152), attention pool over channel 152, shared trunk width 192.

Training: MSE on the three dimensions + BCE-with-logits on `heat > 0.5`,
mixup α=0.3 on both inputs and targets, Adam, speaker-disjoint split by
SPEAKER (a whole recording — not a window — goes to train or val, so no
speaker's voice crosses the boundary; every corpus keeps at least one val
speaker). **CREMA-D is the one place "recording" and "speaker" differ**: its
91 actors each recorded many single-utterance clips, and this pipeline's
`distill_teacher_labels.py` treats each clip as its own "recording" —
splitting on that filename would let the SAME actor's voice appear in both
train and val (different emotions, same speaker), exactly the leak
"speaker-disjoint" exists to prevent. `distill_student.py`'s `build_index()`
derives a `speaker_key` — the CREMA-D filename's leading actor id for that
one corpus, the recording id everywhere else — and splits on THAT. Caught by
re-reading my own split logic before the real training run, not by a test;
worth a regression test if this script gets a second life. Checkpointed
every epoch (`tmp/distill/checkpoints/student.pt`, atomic write) so
`--max-seconds` (≤60 min/invocation, per the brief) can be hit repeatedly
without losing progress — the same resumability discipline as the labeller.

### Training curve

**10,198 windows, 7,399 train / 2,799 val** (34 train speaker-keys / 8 val
speaker-keys — a CREMA-D actor counts as one speaker-key, so of 42
speaker-keys total this is roughly 3 SBCSAE + 2 CHiME-6 + 24 CONFER
recordings + 1 AMI recording + 12 CREMA-D actors, split 34/8). A first
13-epoch run on the SBCSAE/CHiME-6/6-of-24-CONFER/CREMA-D-partial dataset
was discarded (`--fresh`) once CONFER finished labelling completely, so the
numbers below are from ONE retrain on the final dataset — 11 epochs in
~840 s wall (stopped mid-epoch, past its own 700 s budget, for the same
reason the labeller's long recordings run over: the time check only fires
at epoch boundaries):

| epoch | train loss (dim + heat) | val loss |
|---|---|---|
| 0 | 0.1175 | 0.0590 |
| 2 | 0.0780 | 0.0492 |
| 4 | 0.0695 | 0.0497 |
| 5 | 0.0628 | **0.0465** (best) |
| 6 | 0.0571 | 0.0503 |
| 8 | 0.0507 | 0.0565 |
| 10 (final) | **0.0451** | 0.0576 |

Both losses fall smoothly this time — the full-CONFER val set (2,799
windows over 8 speaker-keys, vs 2,592 over 5 before) is enough to stop the
epoch-to-epoch val-loss swings the first run showed. Val loss bottoms out
around epoch 5 and creeps back up slightly while train loss keeps falling —
a small amount of overfitting starting, consistent with a 72 K-param model
given only ~11 passes over 7,399 windows. The checkpoint used for every
metric below is epoch 10 (whatever the time budget landed on, not
cherry-picked for the lowest val loss).

## Metrics

Three questions, three protocols, teacher and student measured the same way
so the comparison is apples-to-apples:

1. **CCC per dimension** (arousal / valence / dominance) against the
   teacher's own labels, on HELD-OUT recordings (never seen in training).
   This asks "did the student learn to imitate the teacher", not "is the
   teacher right".
2. **Angry-vs-happy AUC on RAVDESS**, cross-corpus, TRUE labels (RAVDESS's
   own emotion codes, never the teacher's) — the exact protocol
   `scripts/feature_bench.py` and `docs/decisions/2026-09-20-heat-judge-plan.md`
   use for the teacher (0.897) and the shipped dB-over-baseline signal
   (0.733, `tmp/feature-bank/bench.json`). RAVDESS is never trained on by
   either the teacher-label pipeline (which only ever produces training
   targets from AMI/SBCSAE/CHiME-6/CONFER/CREMA-D) or the student.
3. **CONFER human agreement**, `heat_map.py::human_conflict_agreement`'s
   protocol: Spearman rank correlation against the ten human raters'
   `conflict_per_second`, split WITHIN-clip (each recording's own time
   series, correlated, then averaged across recordings — the "does the
   shape track a real argument's rise and fall" question) and ACROSS-clip
   (every scored window from every recording pooled into one correlation,
   which also captures "is this whole clip calmer/hotter than that one" —
   usually the larger number). Computed for the teacher's cached `heat`
   score and the student's predicted heat probability on the identical
   windows.

| metric | student | teacher | shipped loudness |
|---|---|---|---|
| CCC — arousal (held-out, vs teacher) | **0.617** | 1.0 (by definition) | — |
| CCC — valence (held-out, vs teacher) | **0.085** | 1.0 (by definition) | — |
| CCC — dominance (held-out, vs teacher) | **0.677** | 1.0 (by definition) | — |
| RAVDESS angry-vs-happy AUC (n=384, true labels) | **0.655** | 0.897 | 0.733 |
| CONFER within-clip Spearman (n=24 clips, full corpus) | 0.112 | 0.089 | — |
| CONFER across-clip Spearman (n=24 clips, full corpus) | 0.297 | 0.185 | — |

Reading this honestly, not hopefully:

- **Valence is the weak dimension, and got WORSE with more data** (CCC 0.085,
  down from a 6-CONFER-clip run's 0.174) — exactly the axis the whole
  heat-judge project keeps finding is hardest
  (`docs/plans/2026-09-19-heat-map-rounds.md`: "the missing axis is
  valence", the literature's own finding, independently rediscovered a third
  time here). Arousal and dominance both hold up (0.62, 0.68). 11 epochs on
  7,399 windows is not enough signal to fix valence; a GPU run with the full
  corpora would be the first thing to re-check, and might not fix it either
  — this could be a genuine architecture/capacity limit at 72 K params on a
  4× compressed conv front-end, not just a data problem.
- **RAVDESS AUC (0.655) is below BOTH references** — below the teacher
  (0.897, expected: the student is compressing two transformer models into
  72 K params off ~0.36 CPU-hours of successful labelling) and below the
  shipped loudness signal (0.733). This is the central, unflattering number
  this whole exercise exists to produce, and it is reported as measured, not
  softened. Given the training scale (11 epochs, 7,399 windows, still
  dominated by SBCSAE and 12 CREMA-D actors rather than the corpus's full
  91), an AUC clearing "better than a coin flip but below a hand-rolled
  loudness heuristic" is the expected shape of a first pass, not a surprise.
- **CONFER agreement, at full 24-clip coverage, is lower for BOTH teacher
  and student than the brief's stated reference (+0.51 / +0.85) and lower
  than `docs/decisions/2026-09-20-heat-judge-plan.md`'s own measured +0.55**
  (arousal-only, on cached 5 s reference windows — a different protocol from
  this run's 2 s windows scored fresh). Worth flagging plainly: this run's
  teacher-within-clip number (0.089) is a REAL measurement on the full
  corpus, not a partial sample any more, and it disagrees with the other
  measurement in this repo — the two protocols (this run's fresh 2 s
  windows vs. the cached 5 s reference series `heat_judge_calibrate.py`
  uses) are close enough in spirit but not identical, and reconciling them
  is future work, not something to paper over here. **The student narrowly
  BEATS the teacher on both CONFER numbers** (0.112 vs 0.089 within-clip,
  0.297 vs 0.185 across-clip) — read this as noise from a small, heavily
  mixed-up-regularised model landing in a lucky spot, not as "the student
  is better at reading human conflict than the model it was distilled
  from"; nothing in this pipeline would predict that outcome, and it isn't
  claimed as a finding.

## ONNX export

`torch.onnx.export(..., dynamo=False)` — the new (2.9+) dynamo-based
exporter mishandled this model's dynamic time axis (a shape-inference
conflict onnxruntime's quantizer caught downstream, not a dynamo bug this
doc needs to relitigate); the legacy TorchScript-tracing exporter exports
and `onnxruntime.quantization.quantize_dynamic` quantizes cleanly on the
first try. Both the fp32 and int8-dynamic-quantized graphs are benchmarked
with `onnxruntime` (`CPUExecutionProvider`, `intra_op_num_threads=1`, 100
runs after a 5-run warm-up) on one 2 s window:

| | size | latency / 2 s window |
|---|---|---|
| fp32 | 290,600 bytes (283.8 KB) | 0.59–0.65 ms |
| **int8 dynamic-quantized** | **91,819 bytes (89.7 KB)** | **1.6–1.8 ms** |

(Ranges across three separate export runs during this session — same
architecture, different trained weights each time; weights don't change
FLOPs, so the spread is measurement noise, not a real difference.)

Both numbers are smaller/lower than the paper's own Wav2Small (120 KB
quantized, ~9 ms — confirmed against the paper's abstract via web search,
this repo holds no local copy of the exact benchmark table), but that
comparison isn't apples-to-apples: this student is a different, from-scratch
architecture at the SAME 72 K-param budget, not a re-implementation of theirs,
and 100-run CPU benchmarks on different hardware (paper unspecified, this run
an M-series Mac laptop) don't transfer directly either way.

**int8 is SLOWER than fp32 here (1.6–1.8 ms vs 0.59 ms) — a real measurement,
not a bug.** At 72 K params and a 2 s input, this network's total FLOPs are
tiny enough that `onnxruntime`'s dynamic quantization overhead (packing/
unpacking int8 tensors around each op at inference time — "dynamic" means
the quantization scale is computed per-inference, not calibrated once ahead
of time) costs more than the reduced compute saves. The 120 KB the paper
reports is int8 for the FILE SIZE win (a real 3× reduction here too: 283.8 KB
→ 89.7 KB), not necessarily a latency win at this parameter count — worth
re-measuring with static/calibrated quantization before assuming int8 is the
right choice for a watch deployment; fp32 at under 1 ms is already
comfortably inside any real-time budget this product needs.

## Parity fixture

`server/tests/fixtures/distill_student_parity.json` — 20 real speech
snippets (base64 int16 PCM) plus the int8 ONNX model's own outputs on them,
for a future Kotlin `onnxruntime-android` port to check bit-for-bit (within
int8 quantization tolerance, ~1e-3) numeric agreement against. **Deliberately
short (0.3 s, not the production 2 s window)** — the brief caps this fixture
at 300 KB, and 20 full 2 s windows at 16-bit PCM is ~1.25 MB before even
base64-encoding it. The model pools globally over time, so ANY input length
exercises the identical graph (conv → depthwise-separable blocks → attention
pool → heads) — this fixture tests "does the runtime reproduce the same
numbers given the same bytes", not "does 2 s of context feel right", which
the real app validates with real audio. Measured: **257.1 KB, 20 windows**
(0.3 s / 4,800 samples each) — comfortably under the 300 KB cap.

## What a GPU run would add (the paper's own recipe)

Wav2Small's paper trains 72 K-parameter students (Wav2Small itself, plus four
MobileNet variants) by distilling a large transformer teacher's
arousal/dominance/valence predictions — not human labels — over a corpus
large enough that the teacher itself sets a new state of the art on MSP-Podcast
(valence CCC 0.676). That is the shape of the gap between this run and a
GPU run:

- **Scale.** This CPU pass labelled **2.83 hours** of real conversation in
  **~0.36 logged CPU-hours** (against a 3-hour budget — most of the budget
  went unused not because labelling is fast enough to finish early, but
  because AMI/SBCSAE's long recordings made the round-robin's per-pass
  granularity a poor fit for a fixed session-time budget; see *What
  actually got labelled*) and trained for **~28 minutes across two runs**
  (13 discarded epochs on a smaller dataset, then 11 kept epochs on the
  final 10,198-window dataset) on CPU. A GPU run removes both ceilings:
  label the FULL corpora (SBCSAE's 23.3 h alone, CHiME-6's dev
  split, all of CREMA-D, AMI in full) at whatever hop resolution the bench
  wants (denser than 1 s), and train for many more epochs with batch sizes
  this CPU box cannot hold RAM for.
- **Mixed precision + larger batches.** The paper's own students converge on
  GPU-scale batches; this run's batch size (32) and float32-only CPU training
  is the ceiling a laptop-class CPU allows, not a design choice.
- **A real train/val/test split with enough recordings per corpus** to make
  the CCC and AUC numbers tight rather than measured on a handful of
  held-out recordings, which is what a 3-CPU-hour labelling budget affords
  today.
- **Longer sequences / more of each recording.** This run's speech gate and
  1 s hop already use most of what's speech-covered, but a GPU run could
  also try longer training windows (the paper notes Wav2Small was NOT
  tested on inputs shorter than 2 s here, matching this run's window) and a
  denser hop for more overlapping supervision per second of audio.

None of this changes the STUDENT ARCHITECTURE decision — 72 K params, raw
waveform in, four heads out, global time pooling — only how well the fixed
architecture is trained. The GPU quota request already pending
(`docs/decisions/2026-09-20-heat-judge-plan.md`'s "not decided / owner's
call") is what unlocks it.

## Kotlin port plan (watch)

Simpler than the instant tier's port (`docs/decisions/2026-09-20-heat-judge-plan.md`,
"What a Kotlin port would need") for one reason: **there is no hand-written
feature extractor to reproduce.** The instant tier's port has to re-implement
an FFT, a mel filterbank, YIN pitch tracking, and four functionals bit-for-bit
in Kotlin because it hand-rolled eGeMAPS-style features in TypeScript first.
This student takes raw PCM straight into an ONNX graph — the ONLY thing a
Kotlin port needs to get right is feeding the SAME bytes into the SAME
`.onnx` file `onnxruntime-android` already knows how to run.

1. **Model file.** Ship `student_int8.onnx` (int8-dynamic-quantized,
   89.7 KB / 91,819 bytes on disk) as a watch app asset — comfortably inside
   a Wear OS APK's asset budget.
2. **Ring buffer.** A 2 s / 16 kHz PCM ring buffer on the watch mic lane —
   the SAME window the phone's `SentinelDetector`/`NudgePolicy` already
   reason about (`HOP_S`/`WINDOW_S` in `server/watch/heat_judge.py`), scored
   once per second, same cadence as the cloud judge it would eventually sit
   beside or replace on a phone-absent (watch-alone) session.
3. **Inference call.** `onnxruntime-android`'s `OrtSession.run()` with the
   int16→float32 PCM (`/32768.0`, no other normalization — the model's own
   `BatchNorm1d` layers absorb scale, confirmed by this run's fp32/int8
   parity) — one call, four floats out (arousal, valence, dominance,
   heat_logit; `sigmoid` the last one).
4. **Gate.** `server/tests/fixtures/distill_student_parity.json` — 20
   snippets, expected outputs, ~1e-3 tolerance. This is the ENTIRE Kotlin
   port's correctness gate; there is no feature-parity Spearman check to
   port because there are no hand-written features.
5. **Hold-3s gate stays put.** The student's heat probability plugs into the
   SAME `confirm`/`unknown`/`veto` decision `server/watch/heat_judge.py`
   already makes for the cloud judge — same `CALM_AROUSAL_FLOOR` /
   `CONFIRM_AROUSAL` thresholds apply to `arousal`/`valence` output IF this
   student's own dims are recalibrated against them first (its raw scale is
   not guaranteed to match `odyssey_dim`'s — this run's CCC numbers above
   are exactly the "how close" measurement that recalibration would start
   from), never assumed compatible out of the box.

## Verdict

**Not yet — the architecture and pipeline are worth a GPU run, but this
CHECKPOINT is not worth shipping, and that's a data problem, not an
architecture problem.** The evidence for both halves of that sentence:

- The pipeline WORKS end to end and is cheap to scale: teacher labelling,
  training, ONNX export + int8 quantization, and a parity fixture all run
  correctly on a laptop CPU, are fully resumable, and produced a real
  72,077-parameter model that measurably learned SOMETHING (arousal CCC
  0.62, dominance CCC 0.68 — nowhere near the teacher, but nowhere near
  random either). That's the scaffolding a GPU run would need, and it's
  built and gated (`server/tests/test_distill_student.py`).
- The MODEL is not there: RAVDESS AUC 0.655 loses to the shipped
  dB-over-baseline loudness signal (0.733) — the one comparison that
  matters for "should this replace or augment anything shipped" — and
  valence (CCC 0.085) is closer to noise than to the teacher. This is
  measured on ~0.36 CPU-hours of successful labelling (2.83 h of audio,
  dominated by 3 SBCSAE recordings and 12 of CREMA-D's 91 actors) and 11
  kept training epochs. Nothing here rules out the architecture; it rules
  out THIS amount of data and training time as sufficient.
- **The single highest-leverage next step is more CPU labelling before any
  GPU request**, specifically CREMA-D (6,500 of 7,442 clips still
  unlabelled, and it's the corpus with the true angry/happy contrast this
  whole exercise is being scored against) and the rest of SBCSAE (57 of 60
  recordings unlabelled, the most naturalistic corpus held). Both are
  cheap, resumable, and don't need a quota approval to run.
- **A GPU request is justified once CPU labelling is closer to the 3-hour
  budget's actual ceiling** (this run used ~12% of it) and a few more CPU
  training passes on the fuller dataset still leave valence stuck — at that
  point the paper's own recipe (train on the FULL teacher-labelled set,
  more epochs, larger batches) is the right lever, and this run's
  `scripts/distill_teacher_labels.py` / `scripts/distill_student.py` are
  already the harness that run would use, unchanged.

## Reproduce

```bash
MINDSHIFT_TONE_CACHE=/Users/sagearbor/projects/githubs/mindshift/server/.tone_cache \
  python scripts/distill_teacher_labels.py --max-seconds 540   # repeat to extend coverage
python scripts/distill_student.py --max-seconds 3300           # repeat to extend training
pytest server/tests/test_distill_student.py -q
```
