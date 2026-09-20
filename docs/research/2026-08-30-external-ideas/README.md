# External-report ideas, measured — 2026-08-30

Three offline tests of the top-ranked items in
`../2026-08-30-external-research-assessment.md` (AS-Norm for self-ID,
overlap masking, denoising), each scored with the bake-off's shared scorer
(`../2026-08-29-voice-separation/score.py`: 10 ms frame accuracy under the
best one-to-one label mapping over GT speech; overlap frames credit either
speaker) on the checked-in fixtures plus the owner's PRIVATE 3-person
restaurant clip `maggiano3` against his own rubric. Approach B = the
bake-off's transcript-free window engine (1.5 s / 0.25 s grid, spectral +
eigengap k at p=0.80, `B-sliding-window/run_b.py` reused unchanged, cache
redirected here). Production = `server/diarize_local.diarize_turns` fed the
GT boundaries (optimistic) and, for maggiano3, the two real Deepgram
transcripts. No production code was changed; every hook is a monkeypatch
inside the experiment process. All numbers are in `results.json`.

Reference points (bake-off, unchanged today): maggiano3 B 0.761 (k 3, owner
purity 0.80), window-level ceiling 0.837, production 0.833 / 0.702 / 0.671
(GT / 7-utt / 8-utt transcripts); poker6 B 0.809 (k 7/6), production 1.00.

Environment: `tmp/venv-voice` (torch 2.13, speechbrain 1.1, pinned ECAPA) for
everything ECAPA; `tmp/venv-pyannote` for pyannote segmentation-3.0;
`tmp/venv-dfn` (**Python 3.11**, torch 2.5, deepfilternet 0.5.6) for
DeepFilterNet3 — `pip install deepfilternet` fails in venv-voice because
DeepFilterLib ships no cp312 wheel and this Mac has no Rust toolchain, so a
`python@3.11` venv with the prebuilt arm64 wheel was used (`brew install
python@3.11`). Torch threads = 4 throughout.

Verdicts in one line each:

| # | idea | verdict | the number that decides it |
|---|---|---|---|
| 1 | AS-Norm for self-ID | **drop** (keep the multi-setting print + contrast match) | stored 3-setting print: raw gap +0.19 (2.5 sd of the non-owner spread) vs AS-Norm N=20 +4.7 z (1.7 sd); single-setting prints get no usable gap either way |
| 2 | pyannote overlap mask → B / production | **drop** | 6.97 s flagged on maggiano3, 0.67 s of them inside the rubric's 1.6 s of overlap (recall 0.42); B 0.761 → 0.722 at the spec'd 30 % mask, production 8-utt 0.671 → 0.574 |
| 3 | DeepFilterNet3 before embedding | **drop for production; at most a window-pass-only device trial of the 12 dB-limited variant** | full DFN: ceiling 0.837 → 0.918 but B collapses to k=1 (0.432) and production GT 0.833 → 0.571 (k 5); 12 dB: B +0.015, production GT −0.115 |

---

## 1. AS-Norm for self-identification

`exp1_asnorm.py` (+ `exp1_libri_cohort.py`). AS-Norm (Matejka et al. 2017):
`s_norm = ½·((s−μ_e)/σ_e + (s−μ_t)/σ_t)` with μ/σ over each side's top-N
cosines to a cohort, N ∈ {10, 20, 30, all} (N = all is plain S-norm).

**Cohorts.** (a) *fixtures*: every non-owner voice in the checked-in
fixtures — poker P1–P5, the family son, the five TTS voices (onyx / coral /
ballad / nova / marin, deduplicated across the openai / gptaudio / scene
fixtures) — one pooled vector per (fixture, speaker) + up to 20
non-overlapping 1.5 s speech windows per voice = **113 vectors, 11 voices**.
(b) *fixtures + LibriSpeech dev-clean* (40 public read-speech speakers, 30 s
pooled + 6 windows each) = **393 vectors, 51 voices**. Scoring is
leave-one-voice-out (a non-owner probe's own voice is removed from the
cohort); the owner is never in a cohort.

**Prints** (enrollment side): single-setting = maggiano-only (rubric dad
audio pooled; and the 3 stored app samples' centroid, i.e. what the app
enrolled from its own Speaker A), the stored guided sample, family-only,
poker-only; multi-setting = a 2-setting maggiano+family blend and the
**stored 5-sample blend** (`speaker_id.blend_samples`: maggiano ×3 → one
centroid, guided, family = 3 settings). **Probes**: the owner pooled in each
of the 3 recordings (in-sample probes — the recording a print was built
from — are excluded from "owner min"), and 21 non-owner pooled speakers
(maggiano mom / asher, family Asher, poker P1–P5, TTS per fixture).

| print (settings) | raw: owner min / non-owner max → gap (gap ÷ sd of non-owner scores) | AS-Norm N=20, fixtures cohort (11 voices) | AS-Norm N=20, +LibriSpeech (51 voices) | S-norm N=all, 51 voices | in-recording contrast winner = owner (maggiano / family / poker), raw |
|---|---|---|---|---|---|
| maggiano_only (rubric dad pooled) (1) | 0.37 / 0.24 → **+0.13** (+1.5) | 4.32 / 7.30 → **−2.98** (−0.9) | 3.36 / 1.54 → **+1.81** (+0.7) | 3.83 / 2.75 → **+1.08** (+0.9) | Y/Y/Y |
| maggiano_only (3 stored app samples) (1) | 0.23 / 0.40 → **−0.17** (−1.7) | 1.85 / 10.94 → **−9.10** (−2.2) | −0.08 / 6.37 → **−6.45** (−2.0) | 2.29 / 4.59 → **−2.30** (−1.7) | Y/Y/Y |
| guided_only (stored guided sample) (1) | 0.30 / 0.23 → **+0.07** (+0.7) | 4.18 / 2.02 → **+2.17** (+0.7) | 3.39 / 0.86 → **+2.53** (+0.7) | 3.25 / 2.73 → **+0.52** (+0.4) | Y/Y/Y |
| family_only (Sage pooled) (1) | 0.23 / 0.21 → **+0.02** (+0.4) | 2.16 / 2.86 → **−0.70** (−0.3) | −0.29 / −0.10 → **−0.19** (−0.1) | 2.18 / 2.21 → **−0.03** (−0.0) | Y/Y/Y |
| poker_only (Player6 pooled) (1) | 0.23 / 0.30 → **−0.07** (−0.8) | 2.16 / 3.16 → **−0.99** (−0.4) | −0.29 / 2.44 → **−2.73** (−0.9) | 2.18 / 3.19 → **−1.01** (−0.9) | Y/Y/Y |
| blend maggiano+family (2) | 0.36 / 0.23 → **+0.13** (+2.1) | 4.92 / 7.18 → **−2.25** (−0.7) | 3.04 / 0.92 → **+2.12** (+0.9) | 3.59 / 2.49 → **+1.10** (+1.2) | Y/Y/Y |
| **stored blend (5 samples / 3 settings, app)** (3) | **0.42 / 0.23 → +0.19 (+2.5)** | 5.26 / 6.55 → **−1.29** (−0.4) | 5.67 / 0.98 → **+4.70** (+1.7) | 4.37 / 2.51 → **+1.86** (+1.8) | Y/Y/Y |

Non-owner max is almost always the **same-room stranger** (maggiano mom
0.23–0.24 raw against every maggiano-containing print; the app's stored
maggiano samples even put asher at 0.40 — production's Speaker A pooled
some of the kids' speech, which is why that print is unusable in any
space). The owner's lowest cross-setting score is always poker (0.42 with
the stored blend, 0.23–0.37 with one-setting prints). Window-level (1.5 s)
scores never separate: owner-window p10 < non-owner-window p90 for every
print and normalisation (`results.json` → `owner_window_p10` /
`nonowner_window_p90`).

Single-threshold test — the interval of thresholds that clears every print
with a positive gap: raw **none** (family_only's owner min 0.234 sits under
mom's 0.236, so a global raw threshold covers 4 prints, not 5); AS-Norm N=20
with the 51-voice cohort: (1.54, 3.04) covering 4 of 7 prints (the same 4
raw covers, minus family_only). The stored blend's z-scale is ≈24 z per raw
cosine unit, so the ±0.05 run-to-run variance `speaker_id.py` budgets for is
≈ ±1.2 z — the 0.75 z half-margin of that interval is inside the noise,
while the raw threshold 0.32 for the stored blend has a ±0.10 margin (2× the
noise).

**Verdict — drop (keep what ships).**
1. AS-Norm does not open a threshold that works everywhere: it never turns a
   negative raw gap positive for a single-setting print, and it narrows the
   multi-setting print's separation relative to the score spread (stored
   blend 2.5 sd raw → 1.7 sd at N=20, 51 voices).
2. With a cohort we can actually ship on a phone today (the 11 fixture
   voices), top-N AS-Norm is actively harmful: every multi-setting print's
   gap flips negative (stored blend +0.19 raw → −1.29 z) because the
   same-room stranger shares a channel the cohort does not contain and gets
   inflated more than the owner in another room.
3. It only stops hurting with 51 voices (LibriSpeech), and even then plain
   S-norm (N=all) beats top-N on the multi-setting prints — the "adaptive"
   part needs hundreds of diverse voices, which we do not have and cannot
   bundle cheaply.
4. What does separate the owner is what ships: the per-recording blended
   print (raw gap −0.17…+0.13 for one setting → +0.19 for 3 settings) plus
   the in-recording contrast rule — the owner wins every recording under
   every print and every normalisation (last column), with raw margins
   0.54 / 0.66 / 0.22 for the stored blend (`CROSS_MATCH_MARGIN` 0.15).
5. The stored blend at threshold 0.32 (± 0.10) is the deployable number;
   `CROSS_MATCH_THRESHOLD` 0.40 sits above its poker score 0.419 by only
   0.02, so if anything should move it is that constant, not the scoring.
6. Room, not identity, is the residual: the top non-owner is the spouse in
   the same restaurant in 5 of 7 prints. A cohort that *contained* the
   owner's own recordings' other speakers would model that channel — but
   that is the contrast match again, expressed differently.
7. Revisit only with a real cohort (≥ 200 speakers, several rooms/mics) and
   a measured run-to-run variance in z-space; not before.

## 2. Overlap masking → B (and production's window pass)

`exp2_overlap_pyannote.py` (venv-pyannote: segmentation-3.0 run directly on
10 s chunks / 2.5 s step; p_overlap = summed posterior of the 2-speaker
powerset classes, averaged over covering chunks; 0.8 s CPU for the 42 s
clip) → `exp2_overlap_score.py` (venv-voice). A 1.5 s window is masked when
≥ 10 / **30** / 50 % of its frames have p_overlap > 0.5, then B re-clusters
on the remaining grid (the masked span inherits its nearest kept window's
label, like any VAD gap). Production: `speaker_id.speech_mask` wrapped so
the diarizer's window pass treats overlapped 30 ms frames as non-speech;
transcript path unchanged.

| fixture | pyannote flagged (p>0.5) | inside rubric overlap (27–28, 29.8–30.4 s) | B unmasked | B mask ≥10 % | B mask ≥30 % (spec) | B mask ≥50 % | production unmasked | production masked |
|---|---|---|---|---|---|---|---|---|
| maggiano3 | **6.97 s** | **0.67 s of 1.6 s (recall 0.42)**; 6.3 s flagged outside | 0.761 (k3, 0/162 masked) | 0.661 (k3, 63/162) | **0.722** (k3, 42/162) | 0.765 (k3, 19/162) | GT 0.833 (k3, pur 0.84); 7utt 0.702 (k3, 0.80); 8utt 0.671 (k3, 0.79) | GT 0.833 (k3, 0.84); 7utt 0.702 (k3, 0.80); **8utt 0.574 (k3, 0.59)** |
| scene_family3 | 0.0 s | — (no overlap) | 0.990 (k3) | 0.990 | 0.990 | 0.990 | GT 1.000 (k3) | GT 1.000 (k3) |
| scene_meeting4 | 0.0 s | — | 0.809 (k3) | 0.809 | 0.809 | 0.809 | GT 0.597 (k2) | GT 0.597 (k2) |
| family_real | 0.0 s | — | 0.959 (k2) | 0.959 | 0.959 | 0.959 | GT 1.000 (k2) | GT 1.000 (k2) |
| poker6 | 0.17 s | — | 0.809 (k7) | 0.772 (k8, 5/109) | 0.809 | 0.809 | GT 1.000 (k6) | GT 1.000 (k6) |

maggiano3 flagged runs (s): 12.7–12.9, 21.5–22.7, 25.2–25.4, 25.5–25.9,
**27.5–29.9**, 32.3–32.8, 34.1–34.8, 38.0–39.5. The one true overlap it
finds (27.5–29.9) is smeared over the child's 28–29 s turn; the rest are
rapid turn-takes and the father's 37–39 s joke, i.e. the child/parent
confusion C-pyannote already measured, now expressed as "overlap".

**Verdict — drop.**
1. The mask is mostly wrong: 4.4× more seconds flagged than the rubric has
   overlap, and it catches 42 % of the real overlap.
2. At the spec'd setting (p > 0.5, ≥ 30 % of the window) B loses 0.04
   (0.761 → 0.722) and owner purity 0.80 → 0.79; at 10 % it loses 0.10.
3. The only non-negative setting (≥ 50 %, +0.004) masks 19 windows and is
   within run-to-run noise.
4. Production: no change on GT boundaries or the 7-utt transcript, and the
   8-utt transcript drops 0.671 → 0.574 — the window pass lost the proposals
   that were splitting a welded utterance.
5. The "must not hurt" condition holds on the no-overlap fixtures only
   because nothing gets flagged there (0 s on all three TTS scenes and on
   family_real; 0.17 s on poker6, which still costs 0.04 at the 10 %
   setting).
6. The abstain-then-recluster idea was sound; the detector is not — the
   Unknown experiment's finding stands (overlapped speech mints phantoms),
   but pyannote's OSD on this audio cannot tell us where it is.
7. Not worth a device build; if overlap ever matters, an in-house
   two-speaker energy/pitch cue on the rubric's segments would need to be
   validated first, and 1.6 s of overlap in 42 s is not where the accuracy
   is being lost.

## 3. Denoising → B and production

`exp3_prepare.py` (production-decoded PCM → `cache/*_src.wav`) →
`exp3_denoise_dfn.py` (venv-dfn: 16 → 48 kHz, DeepFilterNet3 full
attenuation `dfn` and 12 dB-limited `dfn12`, → 16 kHz, same sample count;
residual lag measured by cross-correlation = **0 samples**, so window and
transcript timings hold) → `exp3_score.py` (venv-voice). `rmsnorm` = no
model, every chunk the embedder sees gain-normalised to RMS 0.05.

DeepFilterNet3 cost on this Mac (4 threads): **0.019 wall s / 0.075 CPU s
per audio second** (42.6 s clip in 0.8 s wall), model load 0.07 s from the
local checkpoint cache. It
removes a lot: maggiano3 RMS 0.072 → 0.038 (noise-floor p10 frame RMS
0.0013 → 0.00009), poker6 0.042 → 0.034.

| audio | B (eigengap k) | B, k given | ceiling: window → nearest GT window-centroid (window acc / frame acc) | ceiling: window → nearest POOLED print | within-speaker window cos (mean) / cross max | closest pooled-print pair | production |
|---|---|---|---|---|---|---|---|
| maggiano3 orig | 0.761 (k3, pur 0.80) | 0.761 | **0.837** / 0.844 | 0.721 / 0.746 | 0.202 / 0.108 | 0.236 (dad–mom) | GT 0.833 (k3, 0.84); 7utt 0.702 (k3, 0.80); 8utt 0.671 (k3, 0.79) |
| maggiano3 dfn | **0.432 (k1)** | 0.708 (pur 0.79) | **0.918** / 0.813 | 0.818 / 0.751 | 0.247 / 0.133 | **0.378** (dad–mom) | **GT 0.571 (k5, 0.62)**; 7utt 0.453 (k2, 0.45); 8utt 0.583 (k3, 0.57) |
| maggiano3 dfn12 | 0.776 (k3, pur 0.81) | 0.776 | 0.843 / 0.854 | 0.740 / 0.786 | 0.209 / 0.119 | 0.219 (dad–mom) | GT 0.718 (k3, 0.66); 7utt 0.703 (k3, 0.80); 8utt 0.572 (k3, 0.59) |
| maggiano3 rmsnorm | 0.761 (k3, pur 0.80) | 0.761 | 0.837 / 0.844 | 0.721 / 0.746 | 0.202 / 0.108 | 0.236 | identical to orig |
| poker6 orig | 0.809 (k7, pur 1.00) | 0.809 | 0.927 / 0.904 | 0.844 / 0.867 | 0.389 / 0.142 | 0.301 (P2–P6) | GT 1.000 (k6, 1.00) |
| poker6 dfn | 0.835 (k7, pur 0.87) | 0.901 | 0.904 / 0.909 | 0.892 / 0.901 | 0.515 / 0.221 | 0.337 (P2–P6) | **GT 0.834 (k5, 0.51)** |
| poker6 dfn12 | 0.867 (k6, pur 0.90) | 0.867 | 0.926 / 0.933 | 0.884 / 0.875 | 0.441 / 0.189 | 0.313 (P2–P6) | GT 0.834 (k5, 0.51) |
| poker6 rmsnorm | 0.809 (k7, pur 1.00) | 0.809 | 0.927 / 0.904 | 0.844 / 0.867 | 0.389 / 0.142 | 0.301 | GT 1.000 (k6, 1.00) |

**Verdict — drop for production; a window-pass-only trial of `dfn12` on
device is the most this earns.**
1. The ceiling does rise where it was asked to: maggiano3 window→centroid
   accuracy 0.837 → 0.918 with full DFN, within-speaker window cosine
   0.20 → 0.25 (poker 0.39 → 0.52).
2. But different speakers move closer just as fast: cross-speaker max
   0.108 → 0.133 (poker 0.142 → 0.221) and the pooled dad–mom print cosine
   0.236 → **0.378** — denoised audio shares a "DFN channel" the way a room
   shares one, so every k-decision built on pooled cosines gets worse.
3. Full DFN breaks the count: B's eigengap says **k=1** on maggiano3
   (0.432) and production finds k=5 on GT boundaries (0.833 → 0.571) and
   k=5 on poker6 (1.00 → 0.834); even with k given, B drops 0.761 → 0.708.
4. The 12 dB-limited variant is the only one that does not break things:
   B +0.015 on maggiano3 and +0.06 on poker6 (k=6 found, the first time B
   gets poker's count right), ceiling +0.006 / +0.01 — but production still
   loses (GT 0.833 → 0.718 on maggiano3, 1.00 → 0.834 on poker6).
5. The VAD sees the change too: full DFN drops 26 of 162 maggiano3 windows
   (the quiet child's) and 26 of 109 poker windows — the noise-floor gate is
   now the 0.003 absolute floor and quiet speech went with the noise.
6. Per-window RMS gain normalisation changes nothing, to three decimals, on
   every metric: ECAPA's feature normalisation already makes it
   gain-invariant, so "the child is 16 dB quieter" is not the mechanism of
   the ceiling.
7. Cost is not the objection — 0.02 s per audio second on this Mac, ~8.5 MB
   ONNX on device would be fine.
8. If tried at all: `dfn12` feeding only the window pass (spectral
   proposals + eigengap lower bound) while the transcript-utterance
   embeddings and pooled prints stay on the raw audio, scored on the same
   fixtures first; the on-device number to beat is B 0.761 → 0.776.

## Files

* `common.py` — scorer / run_b / speaker_id glue, B pipeline with a window
  mask, window ceiling, production wrapper.
* `exp1_asnorm.py`, `exp1_libri_cohort.py` — experiment 1.
* `exp2_overlap_pyannote.py` (venv-pyannote), `exp2_overlap_score.py` — experiment 2.
* `exp3_prepare.py`, `exp3_denoise_dfn.py` (venv-dfn), `exp3_score.py` — experiment 3.
* `results.json` — every number above (`exp1` / `exp2` / `exp3`), plus the
  full per-probe score tables and DFN timings.
* `cache/` (gitignored) — window embeddings, cohort embeddings, pyannote
  overlap frames, enhanced WAVs, logs. `maggiano3_16k.wav` (gitignored) —
  the private clip decoded for pyannote; never copied elsewhere.
  LibriSpeech dev-clean lives in `tmp/external-ideas/libri/`.
