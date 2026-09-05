# F — approach B ported to the phone, replayed on the fixtures (2026-08-30)

The bake-off's winner on maggiano3 (B: 1.5 s / 0.25 s windows → ECAPA →
refined cosine affinity → eigengap k → spectral clustering → mode filter →
runs) now runs **on the phone, post-hoc, over a stored recording's audio**
(`apps/mobile/src/live/diarizeWindows.ts`, wired by
`live/deviceDiarization.ts` + `components/DeviceDiarizationRow.tsx` behind
Advanced → "Experimental voice engine"). This folder is the offline check
that the port IS the shipped algorithm, scored the bake-off's way.

What was ported is the **production** numpy (`server/diarize_sliding_window.py`
`refine_affinity` / `eigengap_k` / `spectral_labels` / `mode_filter` /
`window_label_runs` and `diarize_local._WindowPass`'s grid: 1.5 s / 0.25 s,
≤ 600 windows with hop auto-widening, `speaker_id`'s noise-floor-relative
speech gate), not `run_b.py`. Line-for-line: numpy's percentile
interpolation, tie-breaking, and — via `live/numpyRandom.ts`, a bit-exact
`SeedSequence` + `PCG64` + Lemire `integers` / `random` / `choice` port —
the SAME seeded k-means++ draws, so the phone's partition is the server's
partition on the same embeddings (`parity_*.json`: identical label ids, not
just the same clustering). The one piece without a numpy twin is the
symmetric eigensolver (Householder + implicit QL); eigenvalues agree to
~1e-6 here and k-means is basis-invariant within an eigenspace.

Files: `replay_b.ts` (the port over the eight fixtures with the served ECAPA
ONNX under onnxruntime-node, batch 1 — the phone's shape; run from
`apps/mobile` with `npx tsx`), `score_f.py` (→ `results.json` + the table),
`dump_parity.py` (→ `parity_<fixture>.json`, `parity_rng.json` for
`apps/mobile/__tests__/diarizeWindows.parity.test.ts`), `pred_<fixture>_b8.json`
(B's k policy: eigengap over 1..8) and `pred_<fixture>_prod6.json` (the same
embeddings under production's k clamp: eigengap over 1..6, floor 2),
`replay_summary.json` (windows / gate / k / eigenvalues / timings).

## Result — the port reproduces B exactly

Frame accuracy (k found) [owner purity], `score.py`, maggiano3 = the owner's
private per-second rubric. "Python B" = `../B-sliding-window/results.json`
(`w1.5_h0.25/spec_eigengap_p0.80`, smoothed).

| fixture (k) | phone port (b8) | prod6 k-clamp | Python B | Δ vs B | wall s (Mac M4) | ECAPA ms/window mean / p90 | windows kept/total |
|---|---|---|---|---|---|---|---|
| family_real (2) | 0.959 (2) [0.97] | 0.959 (2) | 0.959 (2) [0.97] | +0.000 | 1.7 | 15.1 / 15.5 | 110/113 |
| poker6 (6) | 0.809 (7) [1.00] | 0.597 (4) | 0.809 (7) [1.00] | +0.000 | 1.7 | 15.2 / 15.9 | 109/115 |
| openai (2) | 0.994 (2) | 0.994 (2) | 0.994 (2) | +0.000 | 4.2 | 15.5 / 16.5 | 266/276 |
| gptaudio (2) | 0.984 (2) | 0.984 (2) | 0.984 (2) | +0.000 | 4.5 | 15.3 / 15.7 | 286/293 |
| scene_couple (2) | 0.986 (2) [0.98] | 0.986 (2) | 0.986 (2) [0.98] | +0.000 | 4.2 | 15.3 / 16.2 | 268/273 |
| scene_family3 (3) | 0.990 (3) [1.00] | 0.990 (3) | 0.990 (3) [1.00] | +0.000 | 4.1 | 15.3 / 15.8 | 263/263 |
| scene_meeting4 (4) | 0.809 (3) [0.98] | 0.809 (3) | 0.809 (3) [0.98] | +0.000 | 5.1 | 15.4 / 15.9 | 326/326 |
| **maggiano3 (3)** | **0.761 (3) [0.81]** | 0.761 (3) | 0.761 (3) [0.80] | +0.000 | 2.5 | 15.5 / 16.6 | 162/165 |
| mean of 8 | **0.911** | | 0.911 | | | | |

* Window counts and gates match B's run to the window (family_real 110/113 at
  gate 0.0052, poker6 109/115 at 0.0047, maggiano3 162/165), the eigengap k
  matches on every fixture, and every frame accuracy is identical to three
  decimals — the port is B.
* **k policy.** B's numbers come from the raw eigengap over 1..8; production's
  `_select_k` narrows the eigengap to `MAX_SPEAKERS_LOCAL` (6) with a floor of
  2 (`_WindowPass.run_global`). On the same embeddings that clamp costs
  poker6 (k 4, 0.597: Player5's register split is real and forcing ≤ 6
  merges the wrong pair) and changes nothing else, so the phone defaults to
  B's policy (`maxSpeakers: 8, minSpeakers: 1`; `{6, 2}` is one option away).
* **Cost.** Clustering (N ≤ 326 here; ≤ 600 by design) is 17–122 ms in JS on
  the Mac — the N×N eigensolver is not the bill; ECAPA is: ~15 ms per 1.5 s
  window on the M4 (E's number), i.e. ≈ 3.6 s of model time per minute of
  speech at the 0.25 s hop. The phone's own number is what the
  `device_diarization` diagnostics event reports (`embed_ms_mean` / `p90`);
  E's 2–4× Pixel factor puts a 3-minute clip at ~30–60 s of embedding.

## Scoring a phone run against a rubric

The replay row posts `device_diarization` (recording id, k, segments,
windows, hop, timings, model revision, device) as its own diagnostics
record and prints its `dx-…` id; then, on the Mac:

    python scripts/diagnostics_tail.py --id dx-XXXX-XXXX \
        --score-rubric tmp/private_fixtures/maggiano3/rubric.json --owner dad

prints the record and its frame accuracy / k / owner purity / per-speaker
recall under the bake-off scorer — the number to put next to the 0.761 above.

## 2026-08-30 — the first Pixel 10 runs said "1 voice found (eigengap 1)"

Both recordings (maggiano's 162/165 windows, poker night 108/115 — the window
counts and gates matched this replay exactly) came back as one voice. Root
cause, reproduced in node against this replay's model and fixture
(`meanPairwiseCos` 0.187 / k = 2 with a normal session; **1.0000 / k = 1,
eigenvalues [110, 0, 0, …]** with a session that behaves like the phone's):

* `diarizeWindows` handed the embedder `pcm.subarray(start, start + 24000)`
  — views into the whole recording's Float32Array.
* `onnxruntime-react-native` 1.24.3's JSI binding (`cpp/TensorUtils.cpp`,
  `createOrtValueFromJSTensor`) builds the `Ort::Value` from
  `tensor.data.buffer` and never reads `byteOffset`: every window's tensor
  pointed at sample 0 of the recording, so all 162 windows embedded the same
  first 1.5 s — one vector, an all-ones affinity, eigengap 1.
* `onnxruntime-node` honours the view, and this replay copied anyway
  (`Float32Array.from(c)`), so the parity harness could not see it. The model
  was not at fault: the served `ecapa_<rev>.onnx` (ETag `960224a6…`, the
  revision the phone printed) is byte-identical to the export replayed here.

Fix: `diarizeWindows` now `slice`s each window (an owned, zero-offset buffer;
96 KB per window), `deviceDiarization` re-guards at the native seam
(`ownedFloat32`), the event carries `mean_pairwise_cosine`, and the Replay row
shows a warning instead of a voice count when that cosine is > 0.95.
`apps/mobile/__tests__/deviceDiarization.ecapa.test.ts` runs the production
orchestration over `family_real` with the real export — once normally and once
through a byteOffset-dropping session wrapper — and asserts k = 2 both ways.

The same `subarray` shape exists in the live loop (`fastLoop.ts`: the
oversized-turn tail in `poolSpeakerAudio` and `finalizeTurn`'s
`pcm.subarray(pcm.length - maxEmbedSamples)`): on the phone those embed the
HEAD of the same turn instead of its tail — still that speaker's voice, which
is why live separation worked while this did not. The durable fix is at the
seam (`ort.ts wrapOrtRuntime` or `EcapaEmbedder.embed`: copy when
`byteOffset !== 0 || byteLength !== buffer.byteLength`).

**Pixel timing.** The phone's ECAPA cost was 62–68 ms per 1.5 s window
(p90 70 ms) against ~15 ms on the M4 here — 4.3×. At the 0.25 s hop that is
240 windows per minute of speech ≈ 16 s of model time per audio minute
(maggiano's 42.6 s → 10.1 s of embedding inside the 11.8 s total; download
was 1.5 s for 1.4 MB). The 600-window cap (hop widening) bounds any clip at
~40 s of embedding, so a 10-minute recording costs the same 40 s as a
2.5-minute one — at a coarser hop. Cheaper paths, untested: batch the
windows as one `[B, 24000]` tensor if the export's first dim is dynamic, or the
XNNPACK execution provider.
