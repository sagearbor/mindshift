# E — the "on device" point: the phone's live clustering, scored like the bake-off

No phone was touched. Two questions, same scorer (`../score.py`) and same eight
fixtures as A/B/C:

1. **How good is the speaker clustering the phone ships today?**
   `apps/mobile/src/live/speakerId.ts` `SpeakerLabeler` (online greedy
   centroids, `CLUSTER_THRESHOLD` 0.48, `MIN_CLUSTER_SECONDS` 1.5) fed by the
   real fast loop, replayed offline through the app's own harness
   (`apps/mobile/src/live/replay/sceneReplay.ts`): 100 ms frames → Silero VAD
   (`silero_vad.onnx`) → `StreamingSegmenter` (merge gap 0.3 s, min 0.6 s) →
   ECAPA ONNX on the last ≤10 s of the turn (`MAX_EMBED_SECONDS`) → labeler.
   The ECAPA model is the served export (`server/.ecapa_cache/ecapa_<rev>.onnx`)
   under `onnxruntime-node`; a torch spot-check on the family_real GT segments
   gives cosine(torch, onnx-node) = 1.00000 on all 8, so these embeddings ARE
   the phone's. No voiceprints enrolled (the bake-off approaches are all
   unsupervised) — this is the "two strangers open the app" case.
2. **What would approach B's window embeddings cost on device?** Measured on
   this Mac with the same ONNX file, extrapolated with a stated factor.

Files: `prep_fixtures.py` (GT + private clip → `tmp/e-on-device/`, gitignored),
`replay_live.ts` (the replay; run from `apps/mobile` with `npx tsx`),
`bench_ecapa.ts` (latency), `score_all.py` (→ `results.json`),
`pred_<fixture>_<variant>_t048|t055.json`, `segments_<fixture>.json`
(every segment's start/end/embedding per variant), `replay_summary.json`,
`bench_ecapa.json`.

## Part 1 — frame accuracy (k found) [owner-cluster purity]

Three segmentations × two thresholds. `live` = the shipped loop as above;
`gt` = the ground-truth utterance boundaries fed as the segments (a perfect
segmenter, same embed + labeler); `energy` = the phone's fallback
`EnergyVad` segmentation (−45 dBFS, 0.25 s frames, same merge/min rules).
"Unknown" turns (the 1.5 s founding guard) claim no identity and count as
unlabelled = wrong, exactly as the scorer treats any uncovered GT frame.

| fixture (k) | **live @0.48** | live @0.55 | gt segs @0.48 | gt segs @0.55 | energy segs @0.48 | production, GT bounds | B (1.5 s/0.25 s spectral) | A auto-k | C pyannote |
|---|---|---|---|---|---|---|---|---|---|
| family_real (2) | **0.68** (3) [0.85] | 0.48 (4) [0.78] | 0.73 (4) [1.00] | 0.73 (4) [1.00] | 0.60 (1) [0.61] | 1.00 (2) [1.00] | 0.96 (2) [0.97] | 0.89 (2) | 0.55 (4) |
| poker6 (6) | **0.55** (6) [1.00] | 0.55 (6) [1.00] | 1.00 (6) [1.00] | 1.00 (6) [1.00] | 0.33 (3) [0.23] | 1.00 (6) [1.00] | 0.81 (7) [1.00] | 0.41 (5) | 0.36 (4) |
| openai (2) | **0.67** (4) | 0.62 (5) | 1.00 (2) | 1.00 (2) | 0.80 (3) | 1.00 (2) | 0.99 (2) | 0.99 (2) | 0.91 |
| gptaudio (2) | **0.66** (3) | 0.49 (5) | 1.00 (2) | 0.90 (3) | 0.74 (3) | 1.00 (2) | 0.98 (2) | 0.99 (2) | 0.83 |
| scene_couple (2) | **0.70** (2) [1.00] | 0.50 (5) [1.00] | 1.00 (2) [1.00] | 1.00 (2) [1.00] | 0.63 (4) [0.83] | 1.00 (2) [1.00] | 0.99 (2) [0.98] | — | 0.95 |
| scene_family3 (3) | **0.43** (5) [1.00] | 0.41 (6) [1.00] | 1.00 (3) [1.00] | 1.00 (3) [1.00] | 0.41 (3) [1.00] | 1.00 (3) [1.00] | 0.99 (3) [1.00] | 0.66 (2) | 0.90 (3) |
| scene_meeting4 (4) | **0.63** (5) [1.00] | 0.74 (6) [1.00] | 0.82 (3) [1.00] | 0.82 (3) [1.00] | 0.43 (5) [0.59] | 0.60 (2) [1.00] | 0.81 (3) [0.98] | 0.60 (2) | 0.77 (3) |
| maggiano3 (3, private) | **0.37** (3) [1.00] | 0.33 (5) [1.00] | 0.39 (10) [1.00] | 0.39 (10) [1.00] | 0.38 (2) [0.40] | 0.83 (3) [0.84]; transcripts 0.70 / 0.67 | 0.76 (3) [0.80] | 0.62 (2) | 0.52 (4) |
| **mean of 8** | **0.585** | 0.514 | 0.868 | 0.855 | 0.540 | 0.89 | 0.91 | 0.77 | 0.73 |

Production/A/B/C columns are copied from the sibling folders' `results.json`
(the README table's rows; B = `w1.5_h0.25/spec_eigengap_p0.80`, merged stage).

Two decompositions of the live number (all in `results.json`):

* **Coverage vs. clustering.** Silero-trimmed turns leave 15–37 % of GT speech
  frames uncovered (pauses inside utterances, sub-0.6 s fragments, the merge
  gap). Accuracy *on the frames the loop labelled* is 0.79 mean (family 0.80,
  poker 0.87, openai 0.94, gptaudio 0.88, couple 1.00, family3 0.59,
  meeting4 0.75, maggiano3 0.50). Filling each gap by the nearest labelled
  turn (`live_filled`, what a contiguous session transcript would do) lifts
  the mean to **0.75** (poker 0.84, couple 0.97, openai 0.88, gptaudio 0.86,
  meeting4 0.76; family3 0.52, maggiano3 0.45).
* **Segmenter vs. labeler.** Given the true boundaries the same labeler scores
  0.87 mean — 1.00 on poker6 / openai / gptaudio / couple / family3 — so the
  online greedy clustering is not the weak part on clean, long segments. It IS
  the weak part on maggiano3: the rubric's 1–3 s restaurant segments found
  **10** clusters (10 of 21 too short to found one → Unknown), matching B's
  finding that within-speaker window cosine there is ~0.20, far below 0.48.
  family_real's 0.73 (k 4) is the son's two ≤0.7 s GT turns (Unknown) plus
  one 2.6 s turn that founded its own cluster.

What the live loop actually does wrong, per fixture (`replay_summary.json`
has every turn): family_real — the son's first reply is merged into the
owner's opening turn (no pause; `replay.real.test.ts` pins this), his last is
a 1.7 s turn that founded "Speaker C"; poker6 — 5/6 players correct but a
1.3 s fragment is Unknown and the approximate ±1–2 s GT plus VAD trimming
costs 37 % coverage; scene_family3 — two of three voices fragment across
five clusters (recall 0.27–0.67); scene_meeting4 — production merges B/D
(cosine 0.72) so the loop's 5 clusters actually beat production's 2 (0.63 vs
0.60), and 0.55 does better still (0.74); maggiano3 — 7 turns, two Unknown,
dad and asher share a cluster (recall 0.26–0.54).

**Threshold sensitivity.** 0.55 (the pre-2026-08-26 value) is worse on 6/8:
mean 0.514 vs 0.585, k inflates (couple 2→5, gptaudio 3→5, maggiano3 3→5),
exactly the "same voice founds fresh clusters" failure the 2026-08-26 tuning
note describes. The one gainer is scene_meeting4 (0.63→0.74), the fixture
that note already called out as the cost of 0.48. On GT segments the two
thresholds tie (0.868 vs 0.855): the sensitivity is a short-segment effect.

**Energy VAD** (the degraded-mode fallback) is worse than Silero on 6/8
(mean 0.54): it welds across speaker changes (family_real 1 cluster, poker6
3 segments for 6 players) because a −45 dBFS floor rarely goes quiet
between two people talking.

## Part 2 — on-device cost of approach B (window embeddings)

Measured with `onnxruntime-node` 1.24.3 on this Mac (Apple M4, 10 cores,
macOS 24.6, Node 26) with the served export (84.2 MB), real speech input,
batch 1, 100 timed runs after 10 warm-ups (`bench_ecapa.json`):

| | intra-op 1 thread | ORT default threads |
|---|---|---|
| ECAPA, 1.5 s window @16 kHz | **14.9 ms** mean, 15.0 median, 15.1 p90 (14.7–16.1) | 15.3 mean, 14.8 median, 17.6 p90 |
| 0.5 / 1 / 3 / 5 / 10 s clip | 7.6 / 10.7 / 24.9 / 39.4 / 78.8 ms | 9.6 / 13.0 / 24.4 / 50.3 / 87.8 ms |
| Silero VAD, one 512-sample (32 ms) frame | 0.09 ms → 1,875 frames/min = 0.17 s compute/min | — |

Extrapolation per **minute of audio** (1-thread numbers; the model is
~linear in clip length, ~7 ms fixed + ~7 ms/s):

| hop | windows/min | ECAPA compute / min (Mac) | 30-min session (Mac) |
|---|---|---|---|
| 0.25 s (B as run) | 240 | **3.6 s** (p90 3.6 s) | 1.8 min |
| 0.5 s | 120 | **1.8 s** | 0.9 min |
| batched 10 s clips (upper bound on pooling) | 6 | 0.5 s | 0.24 min |

**What the live loop already spends** (from the replay, `replay_summary.json`):
one embedding per finalized turn — 10–26 turns/min across the fixtures
(median ~19), each on the last ≤10 s of the turn, i.e. 41–49 s of audio
embedded per minute of session, costing **~0.4 s of Mac ECAPA compute per
minute** (0.36–0.47 s) plus Silero (0.17 s/min for the raw kernel; ~1 s/min
through the harness's tensor seam). B at 0.25 s hop is ~9× that
ECAPA work; at 0.5 s hop ~4.5×.

**Pixel 10 factor — not measured.** These are Mac CPU numbers. The only
public anchor I found is CPU benchmark ratios: Geekbench 6 puts the Tensor G5
at ~62–63 % of an M4 single-core (M4 ≈ 3,770–3,810), so ≥1.6× slower on one
big core; multi-core is ~2.3× (M4 ≈ 14,500 vs. G5 ≈ 6,300); add thermal
throttling and ORT's Android CPU/XNNPACK kernels vs. Apple-tuned ones and
**2–4× is the plausible range**, so B post-session ≈ **7–15 s of CPU per
minute of audio at 0.25 s hop (3.5–7 s at 0.5 s hop)** — a 30-minute session
would take roughly 4–7 minutes (0.25 s hop) or 2–4 minutes (0.5 s hop) of
background CPU. The docs' earlier estimate of "a few hundred ms per turn"
single-threaded on a phone (`docs/plans/2026-08-24-web-safari-fast-loop.md`)
is consistent with a ≤10 s clip at ~3× the Mac's 79 ms. No ECAPA-on-Android
figure exists in the repo or in the search results; the replay harness's
`speakerCostMs: 40` is a guess, not a measurement. Sources:
[cpu-monkey M4 vs Tensor G5](https://www.cpu-monkey.com/en/compare_cpu-apple_m4_10_cpu-vs-google_tensor_g5),
[Notebookcheck M4 Geekbench](https://www.notebookcheck.net/Apple-M4-New-Geekbench-listing-highlights-25-single-core-performance-improvement-over-Apple-M3.835472.0.html),
[nanoreview Tensor G5](https://nanoreview.net/en/soc/google-tensor-g5).

## Verdict

1. The phone's live clustering, scored the bake-off's way, is **0.585 mean
   (0.75 gap-filled)** — below production-on-GT-boundaries (0.89), B (0.91)
   and A (0.77), above pyannote (0.73) on four of eight; it lands at the
   bottom-left of the accuracy-vs-latency scatter (real-time, ~15 ms/turn on
   a Mac).
2. Its embeddings and greedy centroids are not the problem: given true
   boundaries the same code scores 0.87 and is perfect on five fixtures.
3. The loss is segmentation: Silero turns weld two people who don't pause
   (family_real) and fragment one who does, 15–37 % of speech is never
   labelled, and 1–3 s turns are too short for a 0.48 cosine (maggiano3 → 10
   clusters on true boundaries).
4. 0.48 beats 0.55 on 6/8 (0.585 vs 0.514); keep it. scene_meeting4 is the
   known exception.
5. Energy-VAD fallback mode is materially worse (0.54): don't score sessions
   recorded in degraded mode as if they were Silero sessions.
6. **B on device, post-session, is plausible**: ~2–7 min of background CPU
   for a 30-min session at the extrapolated 2–4× Pixel factor, less with a
   0.5 s hop or VAD-gated windows, and the ECAPA ONNX + ORT are already on
   the phone. The unported pieces are B's spectral clustering with eigengap
   k over N×N window affinities (N ≈ 3,600–7,200 per 30 min at 0.5/0.25 s
   hop — needs a JS/native eigensolver or a per-chunk approximation) and its
   smoothing/merge passes.
7. B live (per 0.25 s hop) would be ~9× the loop's current ECAPA work — only
   viable if the Pixel factor comes in at the low end; measure before trying.
8. **An A-on-device port** would need a pitch tracker (Praat/pYIN-class, not
   the loop's current prosody F0 alone), formant (LPC) and MFCC front ends
   in JS/native, a k-means/agglomerative clusterer, and still cannot choose
   k (A's auto-k mean 0.77, poker 0.41) — it adds nothing over the ECAPA
   already running, except as independent "two people" evidence.
9. Cheapest real gain for the live loop is in the segmenter, not the labeler:
   a speaker-change check inside long turns (B's boundary proposals, run on
   the turn's own windows — 4–8 embeddings for a 10 s turn ≈ 60–120 ms Mac)
   and labelling the trimmed gaps by nearest turn.
10. To pin the Pixel factor: time `EcapaEmbedder.embed` on a 1.5 s window
    in the running app (the `speakerMs` field of `TurnLatency` already exists
    in `fastLoop.ts` — one diagnostics dump from a real session is enough).
