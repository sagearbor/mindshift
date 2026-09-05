# Approach B — transcript-free sliding-window diarization (2026-08-29)

One of four parallel experiments in the 2026-08-29 voice-separation bake-off.
Scored with the shared `../score.py` (frame-level, best one-to-one mapping,
GT speech frames only; maggiano3 = the owner's per-second rubric with
overlap segments crediting either speaker).

Files: `run_b.py` (pipeline; caches window embeddings under `cache/`),
`make_report.py` (tables + `pred_<fixture>.json` from `results.json`),
`separability.py` (window-level oracle probe → `separability.json`),
`results.json` (every fixture × variant × stage, with timings),
`pred_<fixture>.json` (best single global transcript-free variant).

## Method

1. **Decode** through `audio_ingest.decode_to_pcm_16k` (openai/gptaudio WAVs
   are natively 24 kHz).
2. **VAD** — energy, 30 ms frames, **noise-floor-relative** gate
   `max(0.003, 1.5 × p10(frame RMS))`; a window is kept if ≥ 30 % of its
   frames pass. This calibration mattered: poker6's quietest player has a
   median frame RMS of 0.0036 against a clip floor of 0.0032, so
   `speaker_id`'s absolute 0.01 gate (or any peak-relative gate) silently
   drops him — the first attempt (25 % of p90) kept 41/113 windows on
   family_real and only 3 of the child's. With the floor-relative gate every
   speaker keeps ≥ 76 % of his windows (poker6 Player1 13/17, the family_real
   child 39/39) and the TTS gaps (digital silence) are still dropped.
3. **Windows** 1.5 s / hop 0.25 s (headline) and 1.0 s / hop 0.5 s; every
   window embedded with the pinned ECAPA in ONE `speaker_id.embed_pcm_batch`
   call (`diarize_sliding_window._window_slices` reused for the slicing).
4. **Clustering** on the cosine affinity:
   * (a) average-linkage agglomerative, cosine-distance threshold swept
     0.60–0.90 (one global value reported; no per-fixture tuning);
   * (b) spectral with eigengap k (Wang et al. 2018 refinement: diag = row
     max, row-wise percentile threshold `p`, symmetrize, diffusion `A·Aᵀ`,
     row-max normalise, `k = argmax λ_k/λ_{k+1}`, k-means on the top-k
     eigenvectors), `p` swept 0.95/0.90/0.80/0.70;
   * oracle-k versions of both as the upper bound.
5. **Smoothing** — mode filter over ±2 hops of temporal neighbours, then
   runs < 0.5 s absorbed into the longer neighbour; labels → intervals by
   nearest window centre at 10 ms (gaps inherit the nearest window, so
   nothing is left unlabelled).
6. **Hybrid (production-shaped)** — two stages, both on POOLED audio:
   *refine*: embed each segment and each cluster's pooled PCM, re-assign
   every segment to its nearest pooled centroid (one pass);
   *pooled-merge*: starting from the window partition, average-linkage-merge
   clusters whose pooled centroids are ≥ 0.45 cosine
   (`diarize_local.MAX_POOLED_COSINE`), re-pool, ≤ 3 rounds, then the
   refine pass.
7. **Coherence** — per cluster, mean (and p10) pairwise cosine of its
   windows, tabulated against the cluster's majority GT speaker and whether
   it is a phantom (its majority speaker is also another cluster's majority,
   or it sits on unlabelled audio).

Timing: `torch.set_num_threads(4)` to approximate Cloud Run's 4 vCPU, but
the runs were made on a shared Mac with three sibling experiments running
(load average 17–38 on 10 cores) — see the runtime table's CPU column and
the caveat below before reading the wall clocks as Cloud Run latency.

## Results — frame accuracy (k_pred/k_true, owner purity), 1.5 s / 0.25 s grid unless noted

`—` = k so far off k_true that the shared scorer's permutation search does not
finish (k=9+ vs 6); the row's mean counts it as 0. All numbers are from
`results.json`; `make_report.py` regenerates the table.

| variant | family_real | poker6 | maggiano3 | openai | gptaudio | scene_couple | scene_family3 | scene_meeting4 | mean |
|---|---|---|---|---|---|---|---|---|---|
| agg t=0.80 | 0.921 (4/2, 1.00) | — (9/6) | 0.514 (10/3, 0.91) | 0.989 (3/2) | 0.795 (7/2) | 0.947 (3/2, 0.99) | 0.972 (4/3, 0.96) | 0.782 (6/4, 0.99) | 0.740 |
| agg t=0.85 | 0.952 (2/2, 0.96) | 0.809 (8/6, 1.00) | 0.551 (7/3, 0.80) | 0.993 (2/2) | 0.990 (2/2) | 0.982 (2/2, 1.00) | 0.672 (2/3, 0.96) | 0.593 (3/4, 0.97) | 0.818 |
| agg t=0.90 | 0.952 (2/2, 0.96) | 0.564 (5/6, 0.07) | 0.694 (4/3, 0.80) | 0.535 (1/2) | 0.990 (2/2) | 0.525 (1/2, 0.53) | 0.672 (2/3, 0.96) | 0.305 (1/4, 0.30) | 0.655 |
| spectral, eigengap p=0.95 | 0.322 (8/2, 1.00) | 0.788 (8/6, 1.00) | 0.432 (1/3, 0.43) | 0.994 (2/2) | 0.984 (2/2) | 0.989 (2/2, 0.98) | 0.990 (3/3, 0.99) | 0.814 (3/4, 0.99) | 0.789 |
| **spectral, eigengap p=0.80** | **0.959 (2/2, 0.97)** | **0.809 (7/6, 1.00)** | **0.761 (3/3, 0.80)** | **0.994 (2/2)** | **0.984 (2/2)** | **0.986 (2/2, 0.97)** | **0.990 (3/3, 0.99)** | **0.809 (3/4, 0.98)** | **0.911** |
| agg t=0.85 + pooled-merge/refine | 0.952 (2/2) | 0.809 (8/6) | 0.551 (7/3) | 0.993 (2/2) | 0.990 (2/2) | 0.982 (2/2) | 0.672 (2/3) | 0.593 (3/4) | 0.818 |
| spectral p=0.80 + pooled-merge/refine | 0.959 (2/2) | 0.809 (7/6) | 0.761 (3/3) | 0.994 (2/2) | 0.984 (2/2) | 0.986 (2/2) | 0.990 (3/3) | 0.809 (3/4) | 0.911 |
| agg ORACLE k | 0.952 | 0.705 (0.07) | 0.676 (0.66) | 0.994 | 0.990 | 0.982 | 0.975 | 0.798 | 0.884 |
| spectral ORACLE k | 0.937 | 0.647 (0.54) | 0.694 (0.82) | 0.994 | 0.984 | 0.989 | 0.990 | 0.969 | 0.900 |
| spectral p=0.80, 1.0 s / 0.5 s grid | 0.973 (2/2) | 0.805 (7/6) | 0.432 (1/3) | 0.985 (2/2) | 0.993 (2/2) | 0.998 (2/2) | 0.982 (3/3) | 0.813 (3/4) | 0.873 |
| agg ORACLE k, 1.0 s / 0.5 s grid | 0.987 | 0.655 | 0.454 (2/3) | 0.985 | 0.993 | 0.998 | 0.681 | 0.305 | 0.757 |

`pred_<fixture>.json` = the bold row (one global setting, no per-fixture
tuning). Per-speaker recall of that row: family_real Asher 0.95 / Sage 0.96;
poker6 P1 1.00, P2 0.97, P3 0.77, P4 0.85, P5 0.63, P6 0.64; maggiano3 asher
0.70, dad 0.74, mom 0.84; scene_meeting4 A/B/C ≥ 0.97, D 0.00.

Window-level ceiling (`separability.json`: assign every window to its nearest
ORACLE GT centroid): family_real 0.960, poker6 0.927, maggiano3 **0.837**,
openai 1.000, gptaudio 0.993, couple 0.992, family3 0.996, meeting4 0.983.
Within-speaker window cosine averages 0.30–0.43 on every fixture except
maggiano3 (0.20; restaurant noise), cross-speaker 0.04–0.17 except
meeting4's B–D pair (0.316).

### What the sweeps say

* **Agglomerative has no global threshold.** Same-voice 1.5 s windows only
  cohere at ~0.3 cosine, so the cut has to sit at distance 0.85–0.90 — but
  0.85 already welds two of scene_family3's three voices (0.672) and three
  of meeting4's four, while 0.80 leaves 3–10 phantom pieces on the real
  recordings. The two real-voice fixtures and the TTS ones want different
  cuts; there is no value that serves both.
* **Spectral + eigengap works once the affinity is dense enough.** Wang's
  p=0.95 keeps ~5 neighbours per row, and on a 0.25 s hop those are all the
  *temporal* neighbours, so the graph decomposes into time blocks (8
  clusters on family_real). At p=0.80 (≈20 % of rows kept) the eigengap
  finds the true k on 6/8 fixtures and is the best single transcript-free
  setting: **mean 0.911, ≥ 0.959 on every 2–3-voice fixture except
  maggiano3**.
* **Oracle k does not help much** (0.884 / 0.900): the residual error is in
  the embeddings, not in k — see poker6 and meeting4 below.
* **The pooled-centroid hybrid is a no-op on top of a good partition** and
  cannot rescue a bad one. One-pass re-assignment moved 0 segments on every
  fixture (a segment is always closest to the pool that contains it).
  Merging at production's 0.45 pooled bar never fired on any headline
  partition (pooled max off-diagonal 0.16–0.39); on a deliberately
  over-clustered partition (agg t=0.60, 17 pieces on family_real) same-voice
  pools of 2–3 s only reach 0.40–0.51, so three rounds got 17 → 12 pieces.
  Production's "same voice pooled ≈ 0.73" holds for long pools only.

## Coherence — is a phantom cluster less self-similar than a real voice?

**No.** Mean pairwise window cosine within each cluster (full table in
`results.json["fixtures"][*]["coherence"]`, headline rows below):

| fixture / partition | real-voice clusters | phantom (over-split) clusters |
|---|---|---|
| poker6, spectral p=0.80 (k=7) | P1 0.498, P2 0.378, P3 0.519, P4 0.460, P6 0.538 | Player5 halves **0.577** and **0.410** |
| poker6, agg t=0.85 (k=8) | 0.378–0.534 | P5 halves 0.577 / 0.536; P6 halves 0.538 / 0.368 |
| maggiano3, agg t=0.85 (k=7) | dad 0.271 | six pieces of mom/asher **0.329–0.619** |
| maggiano3, agg t=0.80 (k=10) | dad 0.308 | 0.336–0.665 |
| gptaudio, agg t=0.80 (k=7) | (2 true voices) 0.320 / 0.407 | 0.352–0.573 |
| family_real, agg t=0.80 (k=4) | Sage 0.303, Asher 0.338 | 0.411, 0.663 |
| maggiano3 overlap 25–33 s (spectral k=3) | asher 0.254, dad 0.279, **mom 0.183** (holds most of the dad/mom + asher/mom overlap windows) | — |

Phantom pieces are *smaller and more homogeneous*, so their coherence is
usually HIGHER than the real voice they were carved from — the opposite of
the hypothesis. The one signal that does show up is the reverse case: an
UNDER-split cluster is less coherent than a pure one (meeting4 agg t=0.85's
B+C+D cluster 0.280 vs pure A 0.345; family3's B+C 0.284 vs A 0.383;
maggiano3's mom cluster 0.183, the one carrying the overlapped speech). So
window coherence is a weak "this cluster may hold two voices" cue, never a
"this cluster is fake" cue. Not recommended as a second k-validation
discriminator; production's problem is under-counting, and for that the
pooled-cosine bar already measures the same thing more directly.

## Failure analysis (every fixture under 0.9 on the headline row)

* **poker6 0.809 (k=7).** All six men are found (P1 1.00, P2 0.97), but
  Player5 splits into two clusters whose POOLED embeddings sit at cosine
  **0.044** — ECAPA hears his two registers as two different men, so no
  pooled-cosine bar (0.45, 0.33) can merge them and production's k-validation
  would keep both too. The remaining loss is boundary slop: P3 0.77, P4
  0.85 on ±1–2 s approximate GT (the scorer docstring's ~0.9 ceiling).
  Oracle k=6 is WORSE (0.705/0.647) because average linkage merges Player6
  into Player2 before it reunites Player5. Bottom line: the sliding window
  beats the transcript path's old 4-of-6 but not its recalibrated 6/6.
* **scene_meeting4 0.809 (k=3).** Speaker D is absorbed into B — their
  pooled cosine is 0.322, the exact pair `diarize_local` already documents
  as its ceiling (marginal 0.339 vs the 0.33 bar). The eigengap sees
  λ3/λ4 = 3.0 vs λ4/λ5 = 1.5 and stops at 3; given k=4 the same embeddings
  score 0.969, so this is a k-estimation miss on a genuinely close pair.
* **maggiano3 0.761 (k=3, right count).** The window-level oracle ceiling is
  0.837: restaurant noise pulls within-speaker window cosine down to 0.20
  and every pair of speakers to 0.05–0.11, so ~16 % of windows are closer
  to the wrong centroid before any clustering. The overlap region (25–33 s)
  lands in the mom cluster wholesale; asher recall 0.70. Longer pools
  (production's utterance-level embeddings) are the right tool here; the
  window path is only useful to *propose* boundaries.

## Runtime (this Mac, `torch.set_num_threads(4)`)

| fixture | clip s | speech s | windows | embed wall s | embed CPU s | cluster (all variants) s | pooled merge+refine s |
|---|---|---|---|---|---|---|---|
| family_real (idle machine, load ≈ 4) | 29.6 | 20.5 | 110 | **59.6** | **236.7** | 0.6 | 0.9 |
| poker6 | 30.1 | 24.9 | 109 | 84.7 † | — | 0.5 | 1.1 |
| maggiano3 | 42.6 | 35.9 | 162 | 268.9 † | — | 1.7 | 6.1 |
| openai | 70.4 | 47.9 | 266 | 335.3 † | — | 0.5 | 5.3 |
| gptaudio | 74.7 | 51.6 | 286 | 277.2 † | — | 2.0 | 4.4 |
| scene_couple | 69.6 | 46.6 | 268 | 271.7 † | — | 1.0 | 4.3 |
| scene_family3 | 67.1 | 47.3 | 263 | 169.7 † | — | 0.2 | 2.8 |
| scene_meeting4 | 82.9 | 58.5 | 326 | 260.2 † | — | 0.5 | 5.1 |

† embedded while three sibling experiments were running (load average 17–38
on 10 cores): 3–5× slower than the idle family_real measurement. Use the
family_real row: **≈ 0.54 s wall (2.2 CPU-s) per 1.5 s window at 4 threads,
i.e. the 1.5/0.25 grid costs ≈ 2× the clip's duration in wall time** (the
0.25 s hop re-embeds every second of audio six times); the 1.0/0.5 grid
costs ≈ 0.7× (54 windows in 21 s). Everything after embedding is
< 2 s for clustering + smoothing and 1–6 s for the pooled pass. The prior
round measured 19.8 s for a 1.5/0.5 grid on the same 30 s poker6 clip
(`docs/research/poker6-sliding-window/README.md`), consistent with this.
Cloud Run 4 vCPU (x86, no AVX-512 guarantee) was not measured; ECAPA on
this Mac is already known to be slow (`speaker_id.embed_pcm` docstring), so
treat 1–2× clip duration as the planning number for the dense grid and
budget a 5-minute recording at 5–10 min of embedding unless the hop is
widened to 0.5 s.

## Verdict — replace or augment `diarize_local.py`?

**Do not replace.** The transcript-free path reaches 0.91 mean and is
robust on 2–3 voices, but on the two hardest real recordings it is at or
below what production already does: poker6 0.81 vs production's 6/6 exact
turns after the 2026-08-24 recalibration; maggiano3 0.76 with a 0.84
window-level ceiling because 1.5 s windows in a noisy room simply do not
carry enough voice. Oracle k barely helps, so the limit is the
window-embedding SNR, not the clustering — exactly what production's
pooled-utterance design already sidesteps.

**Augment, with two specific pieces:**

1. **Boundary proposals for the transcriber-welded case.** Run the 1.5/0.25
   window pass + spectral (p=0.80) ONLY inside utterances longer than
   `SPLIT_MIN_UTTERANCE_SECONDS`, and hand the resulting change points to
   the existing `split_long_utterances` machinery instead of the current
   margin scan. Cost is proportional to the long utterances, not the
   recording. Every voice change in family_real / openai / gptaudio /
   family3 was recovered this way (recall ≥ 0.95 per speaker) with zero
   phantom clusters.
2. **Eigengap as a *lower bound* on k.** The p=0.80 eigengap never
   over-counted on a 2–4-voice fixture (it under-counts meeting4 and
   over-counts only poker6, by the genuine Player5 register split), so
   `_select_k` could take `max(k_eigengap, ...)` as the smallest k to
   validate rather than starting at 2 — cheap insurance against the
   "father and son merged" failure when the transcript's utterance
   boundaries are poor. It will not fix meeting4-class pairs (pooled 0.32),
   which sit under the same ceiling either way.

Not recommended: the within-cluster coherence check (see above) and the
pooled-merge-at-0.45 step for short pieces (same-voice 2–3 s pools measure
0.40–0.51, so the production bar is only meaningful for pools of the size
`diarize_local` already builds).
