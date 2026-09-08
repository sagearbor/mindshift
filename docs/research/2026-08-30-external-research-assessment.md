# External deep-research report — assessment (2026-08-30)

Source: a Gemini deep-research report the owner commissioned from our brief
(`tmp/voice-separation-research-brief-20260830.md`). Title: "Architecting
Edge-Optimized Speaker Diarization: A KISS-Driven Blueprint for Live
Conversational Coaching". This file records what in it is worth acting on,
ranked by how directly it addresses a failure we have MEASURED, and what we
are skeptical of. Nothing here is adopted until it scores on our fixtures
(`docs/research/2026-08-29-voice-separation/score.py`).

## Its blueprint in one line

Denoise (DeepFilterNet3) → transcript-free segmentation with overlap detection
(pyannote powerset) → a lighter/better embedder (CAM++, ideally trained with
Sub-Center ArcFace so registers stay one speaker) → bounded online clustering
with EMA centroids and re-labelling of the last 5–10 s → AS-Norm instead of an
absolute threshold for "is this the user" → spectral clustering + label
propagation for sub-second fragments in batch → ORT with XNNPACK/NNAPI,
2–4 threads, INT8.

## Ranked: act on / test first

1. **AS-Norm for self-identification (HIGH, cheap).** Directly targets our
   measured problem: same person across settings 0.24–0.45 vs a one-setting
   print, different people 0.11–0.28 — no absolute threshold works. AS-Norm
   rescales the raw cosine against each side's top-N most similar "cohort"
   speakers, so a room that suppresses every score is normalized away. Pure
   math on embeddings we already have; needs a cohort (a few hundred
   background speaker embeddings — from the TTS voices, the checked-in
   fixtures' non-owner speakers, and public speaker data). Can run on the
   phone. **Test offline first** on the owner's real probes (maggiano/family/
   poker owner vs non-owners) — the harness from 2026-08-27 (`vm_measure`)
   has everything needed.
2. **Overlap detection → abstain from embedding overlapped frames (HIGH,
   measurable).** Matches our finding exactly: overlapped speech embeds far
   from everyone and mints phantom clusters; the Unknown experiment failed
   because it abstained *before* re-clustering. pyannote's segmentation
   scored poorly for *speech* detection on our audio (32 % VAD miss on
   poker), but its overlap output may still be usable purely as a mask.
   **Test:** mask pyannote-OSD frames out of B's window grid on maggiano3
   and re-score.
3. **Speech enhancement before embedding (MEDIUM-HIGH, measurable).** Attacks
   the 0.84 ceiling (restaurant noise pulls same-speaker window cosine to
   ~0.20; the child is 16 dB quieter). DeepFilterNet3: ~8.5 MB ONNX, 10–40 ms
   latency, MIT. **Test:** denoise maggiano3 with the `deepfilternet` package,
   re-run production + B, and re-measure the window-level ceiling. If the
   ceiling rises, it earns an on-device trial.
4. **Bounded online clustering with EMA centroids + re-labelling the recent
   past (MEDIUM, design).** The right shape for the live loop's measured
   weakness (0.585 mean; greedy running means can't revise). Adopt as the
   design for the live-path rewrite, once 1–3 are measured.
5. **ORT settings (MEDIUM, cheap).** XNNPACK / NNAPI, `intra_op` 2–4 threads,
   INT8 static quantization. Our measured 65 ms/window on the Pixel is the
   whole bill of B on device; worth a benchmark. NOTE: the report's
   "zero-copy JNI" point is moot for the post-session engine (one buffer) but
   relevant live.
6. **Strict DER alongside our metric (LOW effort).** Add missed / false-alarm /
   confusion breakdown, no collar, overlap penalized, to score.py.

## Skeptical / not now

- **CAM++ + Sub-Center ArcFace.** The EER table (ECAPA 1.07 % vs CAM++ 0.89 %)
  is VoxCeleb, not noisy family audio; the register-robustness claim rests on
  *retraining* with SCAF, which we can't do (no data, no budget). A pretrained
  CAM++ (WeSpeaker / 3D-Speaker) is worth one offline score on the fixtures
  as a 4th embedder, nothing more until then.
- **"Collapsed to two speakers ⇒ embedding vulnerability."** Our measurement
  says otherwise: the same embeddings score 0.98–1.00 given correct
  segments; the collapse was average-linkage's partition + a duration gate,
  and the spectral route fixed it (0.52 → 0.83). The report generalized from
  the brief.
- **Label propagation with pitch priors for sub-second interjections.** We
  measured nearest-window labelling at 1/11 and pitch is weak on same-sex
  adults; plausible for parent/child, unproven. Low priority.
- **EEND dismissal** — agreed, for now.

## What we already have that the report assumes we don't

Transcript-free window pass with spectral clustering + eigengap k (shipped
2026-08-29, on the phone with exact parity 2026-08-30); a noise-floor-relative
speech gate; per-recording print blending + a contrast match on the server.

## Outcome (measured the same day — `docs/research/2026-08-30-external-ideas/`)

| idea | result | decision |
|---|---|---|
| AS-Norm for self-ID | With an 11-voice cohort it flips every multi-setting print NEGATIVE (the spouse shares the room's channel, the cohort doesn't); with 51 voices (+LibriSpeech) it is positive but the owner/non-owner margin is 1.7 sd vs **2.5 sd for raw cosine with the 3-setting blend**; never rescues a single-setting print. Our contrast rule already wins on all three recordings. | **drop** |
| pyannote overlap mask → B | Flags 7.0 s on maggiano's, only 0.67 s of the rubric's 1.6 s overlap; B 0.761 → 0.66–0.77 depending on mask strength; production 8-utt 0.67 → 0.57. | **drop** |
| DeepFilterNet3 denoising | Raises the window ceiling 0.84 → 0.92 but pulls DIFFERENT speakers together (dad–mom pooled 0.24 → 0.38): B collapses to k=1, production 0.83 → 0.57, poker 1.00 → 0.83. A 12 dB-limited variant: B +0.015 (poker k=6), production −0.12. Per-window gain normalization: no effect (ECAPA is gain-invariant). | **drop** (a window-pass-only 12 dB device trial is the most that survives) |
