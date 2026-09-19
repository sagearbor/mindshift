# Approach C — pyannote.audio as the external baseline (2026-08-29)

**Question.** Our production diarizer (`server/diarize_local.py`: ECAPA + average-linkage + hand-tuned
thresholds) merges voices that sound obviously different (father + 12-year-old son; six poker-night men).
Is the strongest off-the-shelf diarizer better, and if it is, is the gain in its *segmentation* or in its
*clustering*? Everything here is scored with the shared `../score.py` (10 ms frame accuracy inside GT
speech, best one-to-one label mapping, `k_pred/k_true`, owner purity).

**Models actually evaluated.** `pyannote/speaker-diarization-3.1` (pyannote.audio 3.3.2, CPU).
`speaker-diarization-3.0` and `speaker-diarization-community-1` are **gated and this HF token has not accepted
their licence** (`probe_models.py`: `Pipeline.from_pretrained` returns `None`); community-1 additionally
needs pyannote.audio 4.x, which the venv does not have. Not evaluated, stated rather than guessed.

## Results

`frame_accuracy (k_pred/k_true / owner_purity)`; scene_* are the three TTS scene fixtures, everything else
is real audio except openai/gptaudio (2-voice TTS). maggiano3 = the private restaurant clip, scored against
the owner's listen-through rubric (dad/mom/asher, overlap frames credit either speaker).

### End-to-end pyannote 3.1 (transcript-free)

| variant | family_real | poker6 | maggiano3 | openai | gptaudio | scene_couple | scene_family3 | scene_meeting4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| 3.1 default (thr 0.7046) | 0.552 (4/2 / 0.99) | 0.355 (4/6) | 0.522 (4/3 / 0.72) | 0.910 (2/2) | 0.834 (3/2) | 0.950 (2/2 / 1.00) | 0.904 (3/3 / 1.00) | 0.771 (3/4 / 1.00) | 0.725 |
| 3.1 `num_speakers=k_true` (oracle) | 0.877 (2/2 / 0.99) | 0.355 (4/6) | 0.569 (3/3 / 0.72) | 0.910 (2/2) | 0.923 (2/2) | 0.950 (2/2 / 1.00) | 0.904 (3/3 / 1.00) | 0.700 (4/4 / 1.00) | 0.773 |
| 3.1 `min/max_speakers` 2..6 | 0.552 (4/2 / 0.99) | 0.355 (4/6) | 0.522 (4/3 / 0.72) | 0.910 (2/2) | 0.834 (3/2) | 0.950 (2/2 / 1.00) | 0.904 (3/3 / 1.00) | 0.771 (3/4 / 1.00) | 0.725 |
| 3.1 tuned thr (scene-only tuning → 0.70) | 0.552 (4/2 / 0.99) | 0.355 (4/6) | 0.522 (4/3 / 0.72) | 0.910 (2/2) | 0.649 (5/2) | 0.950 (2/2 / 1.00) | 0.904 (3/3 / 1.00) | 0.771 (3/4 / 1.00) | 0.702 |

Even with the true speaker count handed to it, pyannote cannot find 6 poker players (4/6, "Found only 4
clusters" — its `min_cluster_size=12` floor, same as the prior round found) and mislabels half of maggiano3.

### Decomposition A — keep pyannote's segmentation, swap the embedder / clustering

Units = pyannote's own (10 s chunk × local speaker) segments, exactly what its clustering consumes. "OUR
avg-link" = `diarize_local`'s recipe (average linkage on cosine, merge to exactly k), everything else through
pyannote's assign/reconstruct tail so only the clustering differs.

| variant | family_real | poker6 | maggiano3 | openai | gptaudio | scene_couple | scene_family3 | scene_meeting4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| **seg ceiling: ORACLE unit labels** | 0.878 (2/2) | **0.559** (6/6) | **0.544** (3/3) | 0.910 (2/2) | 0.923 (2/2) | 0.950 (2/2) | 0.904 (3/3) | 0.895 (4/4) | 0.820 |
| wespeaker + pyannote clust (stock) | 0.552 (4/2) | 0.355 (4/6) | 0.522 (4/3) | 0.910 | 0.834 (3/2) | 0.950 | 0.904 | 0.771 (3/4) | 0.725 |
| wespeaker + pyannote clust, oracle k | 0.877 | 0.355 (4/6) | 0.569 | 0.910 | 0.923 | 0.950 | 0.904 | 0.700 | 0.773 |
| wespeaker + OUR avg-link, oracle k | 0.877 | 0.549 (5/6) | 0.546 | 0.910 | 0.923 | 0.950 | 0.326 (1/3) | 0.570 (2/4) | 0.706 |
| ECAPA + pyannote clust (thr 0.7046) | 0.552 (4/2) | 0.447 (5/6) | 0.576 (3/3) | 0.910 | 0.649 (5/2) | 0.950 | **0.948** | 0.754 (3/4) | 0.723 |
| ECAPA + pyannote clust, oracle k | 0.877 | 0.447 (5/6) | 0.576 | 0.910 | 0.923 | 0.950 | 0.948 | 0.684 | 0.789 |
| ECAPA + pyannote clust, bounds 2..6 | 0.552 (4/2) | 0.447 (5/6) | 0.576 | 0.910 | 0.649 (5/2) | 0.950 | 0.948 | 0.754 | 0.723 |
| ECAPA + OUR avg-link, oracle k | 0.877 | 0.524 (6/6) | 0.566 | 0.910 | 0.923 | 0.950 | 0.948 | 0.760 (3/4) | 0.807 |
| ECAPA + pyannote clust, tuned thr (0.70) | 0.552 | 0.447 | 0.576 | 0.910 | 0.649 | 0.950 | 0.948 | 0.754 | 0.723 |

### Decomposition B — GT segmentation (clustering-only)

The GT intervals themselves are the segments; only the embedder + clustering are tested.

| variant | family_real | poker6 | maggiano3 | openai | gptaudio | scene_couple | scene_family3 | scene_meeting4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| GT seg + wespeaker, pyannote thr | 0.346 (8/2) | **1.000** (6/6) | 0.432 (1/3) | 1.000 | 1.000 | 1.000 | 0.685 (2/3) | 0.818 (3/4) | 0.785 |
| GT seg + wespeaker, oracle k | 0.621 (2/2 / 0.61) | 1.000 | 0.640 (3/3 / 0.59) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.908 |
| GT seg + wespeaker, OUR avg-link k | **0.980** (2/2 / 0.97) | 1.000 | 0.611 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.949 |
| GT seg + ECAPA, pyannote thr | 0.346 (8/2) | 1.000 | 0.432 (1/3) | 1.000 | 1.000 | 0.503 (9/2) | 0.685 (2/3) | 0.513 (2/4) | 0.685 |
| GT seg + ECAPA, oracle k | 0.621 (2/2 / 0.61) | 1.000 | 0.432 (1/3) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.882 |
| GT seg + ECAPA, OUR avg-link k | **0.980** (2/2 / 0.97) | 1.000 | 0.608 (3/3 / 0.57) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.949 |

### Segmentation-only diagnostics (`seg_diagnostic.py`, `results_segdiag.json`)

| fixture | VAD miss (GT speech pyannote hears nobody) | unit purity | GT speech lost into a unit whose majority is another speaker |
|---|---|---|---|
| family_real | 0.103 | 0.973 | Asher 0.02, Sage 0.03 |
| poker6 | **0.320** | 0.761 | P1 0.20, **P2 0.56**, P3 0.07, P4 0.27, P5 0.24, P6 0.11 |
| maggiano3 | 0.131 | **0.649** | **asher 0.76**, dad 0.21, mom 0.35 |
| openai / gptaudio / scene_couple | 0.05-0.09 | ≥0.999 | ≤0.002 |
| scene_family3 | 0.056 | 0.910 | A 0.02, B 0.10, C 0.16 |
| scene_meeting4 | 0.053 | 0.856 | A 0.00, B 0.29, C 0.11, D 0.22 |

Gap-fill re-scoring (`gapfill_eval.py`: every unlabelled frame takes the temporally nearest predicted
label, i.e. what utterance-level labelling would do) shows how much is just tight VAD edges:
openai 0.910→0.988, gptaudio(oracle) 0.923→0.995, family_real(oracle) 0.877→0.964, scene_couple 0.950→1.000;
but poker6 default 0.355→0.408 and maggiano3 default 0.522→0.590 — those are not edge effects.

## Diagnosis: the gap is in SEGMENTATION, not clustering

1. **With perfect segments, clustering is essentially solved — by either embedder.** GT segments + pyannote's
   own centroid clustering *at its stock threshold* gets poker6 to 1.000 (6/6), and oracle-k gets every TTS
   fixture to 1.000. Our pinned ECAPA does exactly as well as wespeaker there (and our average-linkage
   beats pyannote's centroid linkage on family_real, 0.980 vs 0.621: the son's turns are heterogeneous and
   centroid linkage + pyannote's `min_cluster_size` handling absorbs them into the father).
2. **With pyannote's segments, even ORACLE labels cap out at 0.559 (poker6) and 0.544 (maggiano3).** That
   ceiling is the segmenter: on poker6 it hears nobody during 32 % of the speech (Player2's whole turn is
   94 % missed); on maggiano3 its local speaker tracks put 76 % of the child's speech into the same
   track as a parent inside the 10 s window, so no clustering downstream can ever separate him (unit
   purity 0.65). family_real's ceiling is 0.878 for the same reason (10 % VAD miss, mostly the son).
3. **Swapping in ECAPA behind pyannote's segmenter changes little** (mean 0.723 vs 0.725 at the stock
   threshold; 0.789 vs 0.773 with oracle k). It helps where the segments are clean (scene_family3 0.948 vs
   0.904, maggiano3 0.576 vs 0.522, poker6 0.447 vs 0.355) and hurts on gptaudio (0.649, over-split — a
   wespeaker-calibrated threshold is not an ECAPA threshold). None of it approaches 0.9 on the real fixtures.
4. **The threshold knob is flat.** Sweeping 0.30-1.10 on the three scene fixtures gives an identical plateau
   at 0.65-0.85 for wespeaker (mean 0.875) and 0.675-0.875 for ECAPA (0.884); the stock 0.7046 already sits
   on it, the closest-to-stock tie (0.70) was kept, and the held-out real fixtures are unchanged. There is no
   hidden gain in pyannote's clustering threshold. (`min_cluster_size` was deliberately not touched — one knob;
   the prior round showed lowering it to 3 finds 6 poker players but shatters family_real into 5 clusters.)

## Per-fixture failure analysis (everything under 0.9)

- **family_real 0.552 default (k=4).** Auto-count over-splits both speakers (Asher recall 0.51, Sage 0.58);
  oracle k fixes it to 0.877, the remaining loss is the 10 % VAD miss (0.964 gap-filled). The son's turns
  are the least self-similar of any speaker here (ECAPA intra-cosine 0.0-0.45, vs 0.5-0.7 for the father).
- **poker6 0.355 (k=4 in every variant, even oracle).** 32 % VAD miss (Player2 94 % missed, Player1/5 ~50 %),
  unit purity 0.76, and 46 short units (median 3.0 s active) cannot clear `min_cluster_size` — the
  pipeline prints "Found only 4 clusters". GT is ±1-2 s approximate, so ~0.9 is the honest ceiling; with GT
  segments both embedders hit 1.000, so nothing about these six voices is hard for the embedder.
- **maggiano3 0.522 (k=4).** Restaurant noise + a child: 13 % VAD miss and the segmenter fuses asher with
  a parent 76 % of the time. Even GT segments only reach 0.61-0.64 (mom recall 0.25-0.32): 21 short, often
  <1 s intervals with overlap; this clip is hard for clustering too, but the segmenter loses first.
- **gptaudio 0.834 (k=3).** One phantom third cluster on Speaker A (recall 0.74); oracle k → 0.923, gap-filled
  0.995. Pure over-counting plus tight edges.
- **scene_meeting4 0.771 (k=3) / 0.700 oracle.** Speaker D never gets a cluster (recall 0.0) even when k=4 is
  forced: pyannote's local segmentation mixes B/D (29 %/22 % lost into other-majority units), so the fourth
  cluster it makes with oracle k is a split of A, not D. GT segments → 1.000.
- **openai 0.910 / scene_family3 0.904.** Labels are right; loss is the 5-9 % VAD trim (0.988 / 0.955 gap-filled).

## Deployment notes

- **Size.** `segmentation-3.0` 5.6 MB + `wespeaker-voxceleb-resnet34-LM` 25 MB (+ pipeline yaml); our pinned
  ECAPA checkpoint is 79 MB. pyannote.audio 3.3.2 itself pulls pytorch-lightning, torchmetrics, asteroid-filterbanks,
  torch-audiomentations, pyannote.{core,database,metrics,pipeline} (~150 MB on top of torch).
- **Licence / gating.** Code MIT. `speaker-diarization-3.1` + `segmentation-3.0` are MIT but **gated** (per-user
  licence acceptance on HF + an `HF_TOKEN` at image-build time to bake the weights); wespeaker is CC-BY-4.0,
  ungated. 3.0 / community-1 are gated and not accepted on this token.
- **Can it run in our Docker image?** Not as-is. `Dockerfile` (repo root) installs the latest CPU torch (2.13 in
  `tmp/venv-voice`) + speechbrain 1.1 + transformers 4.57. pyannote.audio **3.3.2** needs torchaudio<2.9
  (`torchaudio.AudioMetaData`) and huggingface_hub<0.26 (`use_auth_token`), which transformers 4.57 rejects —
  the prior round's `requirements-pyannote.txt` pins (torch 2.8 / hub 0.24.6) only work in a second venv/image
  layer. pyannote.audio **4.0.7** wants torch≥2.8, torchaudio≥2.8, hub≥0.28 and **torchcodec≥0.7**, i.e. FFmpeg
  shared libs in `python:3.11-slim` (apt, ~100 MB) — plausibly co-installable with our stack, untested here.
- **Latency.** CPU, 4 torch threads on this Mac while three sibling experiments ran (load avg 8-28): 30 s clip
  → 24 s, 70 s clip → 72-124 s. Breakdown: segmentation is cheap (2-5 s per 70 s), **wespeaker embedding is
  the cost** (60-110 s per 70 s: 125 units × 10 s windows through a ResNet34). A 4 vCPU Cloud Run instance is
  slower per core; expect ≥2× real time, ~3 min for a 70 s upload, and it would compete with Whisper for
  the same 8 Gi/4 vCPU. No GPU/MPS was used (Cloud Run has none).
- **Segmenter-only adoption** (the hybrid) would cost 2-5 s + our ECAPA per unit (~0.25 s each, 30-40 s per
  70 s clip single-call, batchable) — affordable, but the tables above show it does not fix the real cases.

## Verdict

pyannote 3.1 is **not** a better baseline on the recordings we care about: family_real 0.55, poker6 0.36,
maggiano3 0.52 out of the box (0.88 / 0.36 / 0.57 with the true speaker count, which production never has),
versus the ~0.95 it reaches on clean TTS. The prior round's "71-83 %" on poker6 were per-turn metrics (sliding-window 71 %, pyannote 83 % only
after lowering `min_cluster_size` to 3); with the shared frame scorer and stock settings it is 0.355. Its weakness is exactly
the one we were hoping it would fix: the **segmentation model merges a child with an adult and drops quiet
speakers in noise**, so the clustering never sees separable units. Where the segments are right, clustering is a
solved problem for *both* embedders — our ECAPA + average linkage scores 1.000 on poker6 and 0.980 on
family_real with GT segments. **Do not adopt pyannote (whole or segmenter-only).** The lever is our own
segmentation (transcript/word boundaries, gap-free labelling) feeding the clustering we already have; the
"gap-filled" numbers above show utterance-level labelling alone is worth 5-9 points on every clean fixture.

## Files

- `run_pipeline.py` (venv-pyannote) → `results_pipeline.json`, `preds/pred_<fx>__p31_*.json`, `cache/` (segmentation,
  counts, wespeaker embeddings, per-unit audio, GT-interval audio). `embed_ecapa.py` (venv-voice; `speaker_id.embed_pcm`)
  → `cache/<fx>_ecapa.npz`. `cluster_hybrid.py` + `hybrid_lib.py` → `results_hybrid.json`. `tune_threshold.py` →
  `results_tuned.json`. `seg_diagnostic.py`, `gapfill_eval.py`. `build_results.py` → **`results.json`** (every variant,
  every fixture, runtimes, sweep). `probe_models.py`: which pipelines this token can load.
- `pred_<fixture>.json` = the stock 3.1 default run; all other variants under `preds/`.
- `maggiano3_16k.wav` and `cache/` contain the private clip's audio (decoded here with ffmpeg as instructed) and
  are git-ignored by the local `.gitignore`; delete them when the experiment is archived.
- Run order: `run_pipeline.py` → `embed_ecapa.py` → `cluster_hybrid.py` → `tune_threshold.py` → `seg_diagnostic.py`
  → `gapfill_eval.py` → `build_results.py`. `set -a; source .env; set +a` first (HF_TOKEN); the scripts also read
  `.env` themselves without printing it.
