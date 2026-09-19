# Approach A — hand-crafted acoustic features (pitch / timbre / MFCC)

**Question tested:** the owner's hypothesis that voices which sound *obviously*
different (a 12-year-old vs. his dad; six poker-night men) must be separable by
something simple — pitch, timbre — that the ECAPA pipeline is missing.

**Answer, in one line:** pitch/formants DO carry the adult-vs-child split
(family_real 0.89 frame accuracy, unsupervised) but NOTHING simple separates
six adult men (poker6 tops out at **0.46 even when told k=6**). What our ear
does on poker6 is not a scalar feature; it is exactly what a speaker-embedding
model is for. Acoustic features are a plausible *second discriminator* for
adult/child and male/female merges, not a replacement.

Everything below is scored with the shared `../score.py` (frame-level,
best one-to-one mapping, 10 ms frames inside GT speech). Runtime figures are
on the owner's Mac (Apple Silicon), single-threaded.

## Method

`run_acoustic.py` (all paths derived from `__file__`; run with
`tmp/venv-voice/bin/python run_acoustic.py [fixture ...]`):

1. **Decode** to 16 kHz mono (`scipy.io.wavfile` for the .wav fixtures;
   `server/audio_ingest.decode_to_pcm_16k` for the private .m4a).
2. **Energy VAD** on a 10 ms grid: frame RMS in dB, threshold =
   10th-percentile floor + 0.22 × (95th-percentile peak − floor), then a
   110 ms median filter. (First attempt used 0.35 and silently dropped half of
   the child in family_real and 74 % of Player1 in poker6 — see failure
   analysis.)
3. **Frame tracks computed once per file**: Praat F0 (parselmouth,
   70–500 Hz, 10 ms), Praat formants F1–F3 (Burg, 5.5 kHz), librosa
   MFCC-13, spectral centroid, 85 % rolloff, spectral tilt (dB/kHz slope of
   the log spectrum 100 Hz–4 kHz), RMS.
4. **Windows** 0.75 s, hop 0.20 s, kept if ≥ 40 % of frames are VAD-speech.
   Per window: F0 median (semitones re 100 Hz), F0 IQR (semitones, clipped at
   one octave so Praat octave errors can't dominate), voiced fraction,
   mean centroid / tilt / rolloff, RMS dB, median F1–F3 over voiced frames,
   MFCC mean + std (26 dims).
5. **Cluster**: winsorize at 2/98 %, z-score, Ward agglomerative.
   Pitch-bearing variants cluster only windows with ≥ 15 % voiced frames and
   propagate labels to the rest by nearest-in-time window.
   * `auto` k: silhouette over k = 2..8, taking the *smallest* k within 0.03
     of the best (plain arg-max over-fragmented: silhouettes on these features
     are flat, 0.13–0.18, for anything beyond 2 speakers).
   * `oracle` k = k_true (upper bound: "if we knew how many people").
6. **Smooth**: majority vote over ±2 windows, then merge label runs shorter
   than 0.6 s into the longer neighbour; windows → segments; unlabelled gaps
   ≤ 1 s between segments are filled (same label → joined, different → split at
   the midpoint).

Feature variants (ablation): `pitch` = {F0 med, F0 IQR}; `pitch_spec` = pitch
+ {centroid, tilt, rolloff}; `pitch_form` = pitch + {F1, F2, F3}; `mfcc` = 26
MFCC stats; `full` = everything + RMS (35 dims).

`ablation.py` additionally scores every *single* scalar feature on its own at
oracle k, and computes a GT-supervised **Fisher ratio** (variance of
per-speaker window means ÷ mean within-speaker variance) — the ceiling on how
separable the speakers are along that one axis regardless of clustering.

## Results — auto k (what production would actually see)

Cell = frame_accuracy (k_pred/k_true, owner_purity; `–` = no owner in GT).

| fixture | pitch | pitch_spec | pitch_form | mfcc | full |
|---|---|---|---|---|---|
| family_real | 0.700 (2/2, 0.68) | 0.573 (3/2, 1.00) | 0.884 (2/2, 0.86) | 0.754 (3/2, 0.99) | **0.894** (2/2, 0.99) |
| poker6 | 0.299 (3/6, –) | 0.225 (2/6, –) | **0.422** (7/6, 0.52) | 0.217 (2/6, 0.54) | 0.409 (5/6, 0.55) |
| maggiano3 | 0.587 (5/3, 0.91) | 0.546 (2/3, 0.79) | 0.586 (5/3, 0.98) | 0.547 (2/3, 0.62) | **0.598** (2/3, 0.50) |
| openai | 0.821 (3/2, –) | 0.968 (2/2, –) | 0.973 (2/2, –) | **0.994** (2/2, –) | 0.993 (2/2, –) |
| gptaudio | 0.615 (3/2, –) | 0.790 (2/2, –) | 0.945 (2/2, –) | **0.993** (2/2, –) | 0.993 (2/2, –) |
| scene_couple | **1.000** (2/2, 1.00) | 0.835 (2/2, 0.76) | 0.982 (2/2, 0.97) | 1.000 (2/2, 1.00) | 1.000 (2/2, 1.00) |
| scene_family3 | **0.764** (3/3, 0.97) | 0.669 (2/3, 0.85) | 0.671 (2/3, 0.99) | 0.576 (2/3, 0.70) | 0.663 (2/3, 1.00) |
| scene_meeting4 | **0.626** (3/4, 0.93) | 0.544 (2/4, 0.99) | 0.578 (2/4, 1.00) | 0.547 (2/4, 0.50) | 0.595 (2/4, 0.98) |

## Results — oracle k = k_true (upper bound if speaker count were known)

| fixture | pitch | pitch_spec | pitch_form | mfcc | full |
|---|---|---|---|---|---|
| family_real | 0.700 (2/2, 0.68) | 0.656 (2/2, 1.00) | 0.884 (2/2, 0.86) | **0.973** (2/2, 0.99) | 0.894 (2/2, 0.99) |
| poker6 | 0.426 (6/6, 0.31) | 0.327 (6/6, 0.23) | 0.411 (6/6, 0.00) | 0.454 (6/6, 0.47) | **0.455** (6/6, 0.55) |
| maggiano3 | 0.530 (3/3, 0.50) | 0.517 (3/3, 0.77) | 0.636 (3/3, 0.71) | 0.607 (3/3, 0.62) | **0.708** (3/3, 0.73) |
| openai | 0.910 (2/2, –) | 0.968 (2/2, –) | 0.973 (2/2, –) | **0.994** (2/2, –) | 0.993 (2/2, –) |
| gptaudio | 0.706 (2/2, –) | 0.790 (2/2, –) | 0.945 (2/2, –) | **0.993** (2/2, –) | 0.993 (2/2, –) |
| scene_couple | **1.000** (2/2, 1.00) | 0.835 (2/2, 0.76) | 0.982 (2/2, 0.97) | 1.000 (2/2, 1.00) | 1.000 (2/2, 1.00) |
| scene_family3 | 0.764 (3/3, 0.97) | 0.653 (3/3, 0.82) | 0.694 (3/3, 0.99) | 0.824 (3/3, 0.68) | **0.901** (3/3, 1.00) |
| scene_meeting4 | 0.618 (4/4, 0.93) | 0.522 (4/4, 0.99) | 0.600 (4/4, 0.98) | 0.726 (4/4, 1.00) | **0.745** (4/4, 0.98) |

Reference (from `server/tests/fixtures/audio/README.md`): the current ECAPA
`diarize_local` scores 100 % per-turn on family_real, openai, gptaudio,
scene_couple, scene_family3; finds 4/6 on poker6; 2/4 on meeting4.

## Runtime

| fixture | audio s | windows | speech frac | decode s | features s | slowest cluster+score s |
|---|---|---|---|---|---|---|
| family_real | 29.6 | 112 | 0.64 | 0.00 | 0.55 | 0.04 |
| poker6 | 30.1 | 122 | 0.75 | 0.00 | 0.17 | 0.01 |
| maggiano3 | 42.6 | 189 | 0.84 | 0.29 (ffmpeg) | 0.28 | 0.02 |
| openai | 70.4 | 309 | 0.81 | 0.01 | 0.45 | 0.02 |
| gptaudio | 74.7 | 310 | 0.74 | 0.01 | 0.45 | 0.02 |
| scene_couple | 69.6 | 320 | 0.77 | 0.00 | 0.43 | 0.02 |
| scene_family3 | 67.1 | 312 | 0.75 | 0.00 | 0.42 | 0.03 |
| scene_meeting4 | 82.9 | 370 | 0.75 | 0.00 | 0.51 | 0.03 |

Real-time factor ≈ 0.006 (Praat pitch + formants dominate). The first
family_real number (0.55 s) includes library warm-up.

## Ablation — which single feature does the most?

Cell = Fisher ratio (GT-supervised separability, higher = better; > 1 means
between-speaker spread exceeds within-speaker spread) / oracle-k accuracy
clustering on that ONE feature.

| fixture | f0_med | f0_iqr | centroid | tilt | rolloff | rms | f1 | f2 | f3 | mfcc1 | best single |
|---|---|---|---|---|---|---|---|---|---|---|---|
| family_real | 0.31 / 0.67 | 0.10 / 0.70 | 0.61 / 0.62 | 0.33 / 0.71 | 0.69 / 0.69 | 0.42 / 0.64 | **1.20 / 0.88** | 0.07 / 0.56 | 0.04 / 0.57 | 0.72 / 0.77 | f1 0.88 |
| poker6 | **0.78 / 0.44** | 0.17 / 0.37 | 0.08 / 0.33 | 0.23 / 0.34 | 0.05 / 0.34 | 0.24 / 0.44 | 0.22 / 0.36 | 0.42 / 0.35 | 0.10 / 0.38 | 0.25 / 0.30 | f0_med 0.44 |
| maggiano3 | **1.52 / 0.71** | 0.03 / 0.38 | 0.73 / 0.63 | 0.74 / 0.58 | 0.39 / 0.56 | 0.26 / 0.43 | 1.21 / 0.65 | 0.32 / 0.49 | 0.16 / 0.42 | 1.02 / 0.64 | f0_med 0.71 |
| openai | **8.28 / 0.99** | 0.12 / 0.65 | 0.37 / 0.71 | 0.40 / 0.81 | 0.46 / 0.74 | 0.01 / 0.56 | 0.04 / 0.56 | 0.10 / 0.68 | 1.71 / 0.94 | 0.13 / 0.74 | f0_med 0.99 |
| gptaudio | 2.09 / 0.79 | 0.00 / 0.56 | 0.25 / 0.66 | 0.71 / 0.77 | 0.28 / 0.68 | 0.26 / 0.74 | 0.19 / 0.62 | 0.07 / 0.67 | **1.09 / 0.88** | 0.63 / 0.79 | f3 0.88 |
| scene_couple | **6.55 / 1.00** | 0.14 / 0.74 | 0.27 / 0.81 | 0.54 / 0.84 | 0.42 / 0.86 | 0.24 / 0.82 | 0.06 / 0.60 | 0.09 / 0.57 | 0.86 / 0.89 | 0.37 / 0.82 | f0_med 1.00 |
| scene_family3 | **2.53 / 0.79** | 0.19 / 0.49 | 0.38 / 0.52 | 1.22 / 0.71 | 0.53 / 0.57 | 0.40 / 0.53 | 0.18 / 0.49 | 0.12 / 0.45 | 1.49 / 0.75 | 0.93 / 0.65 | f0_med 0.79 |
| scene_meeting4 | **2.33 / 0.59** | 0.26 / 0.41 | 0.29 / 0.46 | 0.82 / 0.49 | 0.38 / 0.43 | 0.46 / 0.41 | 0.21 / 0.41 | 0.18 / 0.36 | 0.63 / 0.58 | 0.67 / 0.46 | f0_med 0.59 |

Per-speaker medians on the two real fixtures (from `features_*.csv`):

| family_real | F0 Hz | F0 IQR Hz | centroid Hz | tilt dB/kHz | level dB |
|---|---|---|---|---|---|
| Sage (owner) | 148 | 23 | 1084 | −7.5 | −22 |
| Asher (son, 12) | 194–207 | 86 | 1392 | −6.2 | **−38** |

| poker6 | F0 Hz | F1 | F2 | F3 | level dB |
|---|---|---|---|---|---|
| Player1 | 99 | 693 | 1411 | 2491 | **−38** (at noise floor) |
| Player2 | 137 | 542 | 1674 | 2674 | −34 |
| Player3 | 128 | 426 | 1807 | 2552 | −30 |
| Player4 | 180 | 495 | 1677 | 2727 | −29 |
| Player5 | 114 | 400 | 1745 | 2878 | −27 |
| Player6 (owner) | 116 | 402 | 1858 | 2534 | −30 |

**Findings:**

1. **Median F0 is the single most useful feature — but only when the voices
   differ in pitch by ≳ 40 %.** It is near-perfect on the TTS pairs (Fisher
   6–8: `onyx` at 93 Hz vs. a female voice at 180–240 Hz) and it is the best
   single axis on maggiano3 (owner 154 / son 215 / wife 302 Hz). That is why
   the hypothesis *feels* true: on couple/family audio, pitch really is the
   obvious thing.
2. **On poker6, no single feature reaches Fisher 1.** F0 is the best at 0.78
   (Fisher) and 0.44 (accuracy): four of the six men sit within 114–137 Hz
   (Player5 114 vs. the owner 116). Formants, tilt, centroid, MFCC-1 are all
   ≤ 0.42. The 35-dim `full` vector at oracle k gets 0.455. There is no
   "obvious" scalar here — what the owner's ear hears is joint spectral
   detail that these summaries do not preserve.
3. **For the adult/child case the best single feature is F1 (0.88), not
   pitch (0.67).** Asher's F0 (≈200 Hz) overlaps the top of Sage's range
   (Sage's windows span 124–227 Hz because he raises pitch at phrase ends),
   while the child's shorter vocal tract pushes F1 up consistently. MFCC
   captures the same thing better (0.973 oracle). Note the child is 16 dB
   quieter than the owner — level, not voice, is the main reason both this and
   the original ECAPA pipeline struggle with him.
4. **Adding spectral centroid/tilt/rolloff to pitch HURTS on every real
   fixture** (`pitch_spec` < `pitch`): those features track *what is being
   said* (vowel vs. fricative, loud vs. soft) more than *who says it*.
5. **Auto-k is the Achilles heel.** Silhouette on these features is flat
   beyond k = 2 (0.13–0.18), so the unsupervised run collapses family3,
   meeting4 and maggiano3 to 2 clusters, and fragments pitch-only runs to
   3–5. The oracle-vs-auto gap on `full` is 0.24 on family3, 0.15 on meeting4,
   0.11 on maggiano3. Any production use would have to take k from elsewhere
   (e.g. ECAPA's validated k).

## Failure analysis (every fixture under 0.9)

**poker6 — 0.42 auto / 0.46 oracle.** Root cause is the feature space, not
the clustering: see finding 2. Two aggravating factors: (a) Player1 is
recorded at the noise floor (median −38 dB vs. a −50 dB floor; Praat finds
voicing in 15 % of his frames), so he contributes 10 windows, 8 with usable
pitch; (b) turns are ±1–2 s approximate, which caps the achievable score at
≈0.9 anyway. Even the best pairing (owner Player6 vs. Player5) differs by
2 Hz in F0 and by < 60 Hz in F1/F2. Confusions at oracle k are spread across
all pairs (recalls 0.35–0.59), i.e. the clusters are cutting on something
other than speaker. **Verdict: hand-crafted features cannot do six adult men.**

**maggiano3 — 0.60 auto / 0.71 oracle.** Restaurant floor + two quiet
speakers. Pitch alone is the best axis (wife 302 Hz is recovered at 0.90
recall in `full_oracle`) but the owner and son overlap: owner F0 IQR is 58 Hz
(he's animated), the son's median 215 Hz sits inside that range. Auto-k picks
2 and merges the son entirely into the owner (`son` recall 0.0 in
`full_auto`) — the same merge the owner reported from the ECAPA pipeline,
reproduced by a totally different method, which points at the *audio* (quiet
child, noise) rather than at ECAPA specifically. With k = 3 given, `full`
gets owner 0.57 / son 0.73 / wife 0.90.

**scene_meeting4 — 0.60 auto / 0.75 oracle.** Self (onyx, 93 Hz) is isolated
perfectly on every variant (owner_purity 0.98–1.00) — same as ECAPA. The three
colleagues (marin/ballad/nova at 162/167/197 Hz, centroids within 300 Hz) are
not separable by these features even at oracle k (recalls 0.45–0.76); auto-k
gives 2. Pitch-only at least finds k = 3.

**scene_family3 — 0.66 auto / 0.90 oracle.** Oracle `full` is at 0.90
(A 0.93 / B 1.00 / C 0.76): B (ballad, 184 Hz, IQR 82 — the acted teen) and
C (nova, 160 Hz) overlap in pitch when B is calm. Auto-k picks 2 and merges C
into A. `pitch` auto is the only variant that finds k = 3 (0.764).

**family_real — 0.89 auto (`full`), 0.97 oracle (`mfcc`).** Just under the
bar. The residual errors are the first ≈1 s of each of Asher's turns (labelled
Sage) — the child starts each turn quietly and the 0.75 s window straddles
the boundary — plus two Sage windows at 200–227 Hz (phrase-final rises) that
go to the child. The first run (VAD at 0.35, raw-Hz IQR) scored 0.59 on pitch
with Asher recall 0.02: two windows with a 250 Hz octave-error IQR were the
whole first Ward split. The semitone/clipped features and lower VAD fixed it;
the lesson is that this route is brittle to exactly the quiet-child case it
is supposed to help with.

**gptaudio pitch 0.62 / pitch_spec 0.79** (mfcc/full are 0.99): the acted
voices swing F0 by > an octave inside a turn (Speaker A IQR 114 Hz), so
pitch on 0.75 s windows is a poor speaker cue; F3 is the best single feature
here (0.88). Timbre (MFCC) still separates them cleanly.

## Verdict

**Replace ECAPA? No.** On the fixtures where ECAPA is at 100 % this route is
at 0.89–1.00 with the *true* k supplied and 0.66–1.00 without; on the two
fixtures where ECAPA is known to fall short (poker6 4/6, meeting4 2/4) this
route is worse (0.46 / 0.75 oracle) and does not find k on its own. Its
cluster-count discovery is strictly weaker than the pooled-cosine validation
in `diarize_local`.

**Augment ECAPA? Yes, narrowly — as a cheap second opinion on a proposed
MERGE, not as a clusterer.** The specific value is where a *single* scalar
carries a big, interpretable, human-meaningful gap:

* *Adult vs. child / male vs. female*: if two ECAPA clusters (or two halves of
  one cluster) differ in median F0 by > ~5 semitones (≈ 35 %) **and** in
  median F1 by > ~100 Hz, with Fisher > 1 on pooled windows, that is strong
  independent evidence they are two people. On family_real those numbers are
  F0 148 vs. 200 Hz (5.2 st) and F1 differing by ≈ 150 Hz; on maggiano3
  owner/son/wife are 154/215/302 Hz. This could be used to *relax*
  `STRONG_SEPARATION_COSINE` / the duration floor for a split that pitch and
  F1 both endorse — i.e. as a tie-breaker in the k-validation step, where the
  owner's reported failures (son merged into owner) live.
* *Owner-purity check*: compute the owner's enrolled F0/F1 envelope at
  enrollment time; utterances assigned to the owner that fall > 6 st outside
  it are candidates for re-assignment. Cheap and explainable in logs.

It **cannot**: separate same-sex adults of similar pitch (poker6, meeting4's
colleagues), choose k, or cope with a speaker at the noise floor. Also do not
expect it to rescue quiet children: the 16 dB level gap on family_real is the
underlying problem for both methods, and gain-normalising per window before
feature extraction would be the first thing to try in either pipeline.

**To productionise as an augmentation:** deps are `praat-parselmouth`
(binary wheels, ~10 MB, no system Praat) and `librosa` (pure-python +
numba); or drop librosa and use `numpy` STFT + a hand-rolled MFCC (the only
librosa pieces used are `rms`, `stft`, `mfcc`, `spectral_*`). Cost ≈ 6 ms per
second of audio on the Mac, i.e. < 0.5 s on a 60 s clip, well inside the
existing ECAPA cross-check budget. It would live as a small helper called
from `diarize_local`'s k-validation with the pooled cluster audio, returning
`{f0_med_semitones, f1_med, fisher_f0, fisher_f1}` per cluster pair. On
Android/on-device the same features are straightforward (F0 via YIN/pYIN,
formants via LPC) but that is a separate port.

## Files

* `run_acoustic.py` — the experiment; writes everything below.
* `ablation.py` — single-feature Fisher ratios + oracle-k accuracies → `ablation.json`.
* `results.json` — scorer output per fixture × variant × {auto, oracle} with runtimes and silhouettes.
* `pred_<fixture>.json` — the `full` variant at auto k (headline prediction);
  `pred_<fixture>_<variant>_<auto|oracle>.json` — every run.
* `features_<fixture>.csv` — columns exactly `time, speaker_gt, f0, centroid, tilt, rms`
  (f0 in Hz, blank when the window has no voiced frames; one row per 0.75 s / 0.2 s-hop speech window).

Reproduce: `export PATH=/opt/homebrew/bin:$PATH; tmp/venv-voice/bin/python docs/research/2026-08-29-voice-separation/A-acoustic/run_acoustic.py && ... /ablation.py`
(needs `pip install librosa praat-parselmouth` in that venv — done 2026-08-29).
