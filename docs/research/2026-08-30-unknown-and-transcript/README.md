# "Unknown" speaker + transcript-label influence — 2026-08-30

Two questions against the 2026-08-29 scorer (`../2026-08-29-voice-separation/score.py`:
10 ms frame accuracy under the best one-to-one label mapping over ground-truth
speech) and the owner's PRIVATE `maggiano3` clip with his per-second rubric
(`tmp/private_fixtures/maggiano3/`, never committed — see `.gitignore`).

Scripts (run from this directory with `tmp/venv-voice`):

* `calibrate_unclaimed.py` — per-word best cosine to any pooled centroid
  (round-1 k-selection + window-pass spectral), tagged right/wrong by the
  rubric, on both Deepgram transcripts → `out/calibration_maggiano3.log`.
* `run_measure.py <tag> [fixture…]` — production diarizer + an **Unknown
  column**: turns labelled `Unknown` are REMOVED before scoring (= unlabelled
  = wrong); `unknown_s` reports the seconds they claimed → `out/<tag>/`.
* `transcript_influence.py` — Deepgram ×3 per recording vs ours vs what
  `main.py`'s never-reduce guard would keep → `out/transcript_influence.json`.

## 1. Unknown speaker for unclaimed word runs (`MINDSHIFT_DIARIZE_UNKNOWN`)

Rule as built (`server/diarize_local.py`, "Unclaimed speech"): a run of ≥ 2
words lasting ≥ 0.8 s whose best cosine to EVERY pooled centroid (k-selection
winner's + the window pass's spectral centroids) is under `UNCLAIMED_COSINE`
(0.12) becomes its own piece labelled `Unknown` — excluded from every
k-selection pass, never inherited from, not a speaker (talk share, heat stats /
report cards, coupling, enrollment matching skip it). Same floor on every
embeddable turn after the final refinement and on self-attributed short turns.

### Calibration (per-word best cosine, maggiano3, both transcripts)

| words | n | p5 | p10 | p50 |
|---|---|---|---|---|
| production RIGHT, 7utt | 77 | 0.096 | 0.141 | 0.410 |
| production WRONG, 7utt | 18 | 0.133 (min 0.129) | 0.147 | 0.327 |
| production RIGHT, 8utt | 75 | 0.099 | 0.153 | 0.394 |
| production WRONG, 8utt | 20 | 0.132 (min 0.102) | 0.139 | 0.362 |

Words under a floor: 0.10 → 9 (0 wrong); 0.12 → 13 (1 wrong); 0.15 → 21 (5);
0.18 → 34 (7). The mislabelled words are **claimed confidently by the wrong
centroid**, not unclaimed; the low-cosine words are the OVERLAPPED chant at
28–32 s (both voices at once). The son's quiet "Because I wanna do my Duolingo,
dad" is claimed at 0.74 / 0.71 by today's pipeline (the 2026-08-29 window pass
fixed it). Every embeddable final turn sits ≥ 0.22 from its centroid — the
whole-turn rule has nothing to fire on.

### Measurement (`out/before`, `out/unknown012`, `out/unknown_0.15`)

Unknown counts as unlabelled (wrong). Purity = dad-cluster purity.

| fixture / variant | flag OFF acc | purity | flag ON (0.12) acc | purity | Unknown s | ON (0.15) acc / purity / Unknown s |
|---|---|---|---|---|---|---|
| family_real (gt) | 1.000 | 1.00 | 1.000 | 1.00 | 0 | — |
| poker6 (gt) | 1.000 | 1.00 | 1.000 | 1.00 | 0 | — |
| openai (gt) | 1.000 | — | 1.000 | — | 0 | — |
| gptaudio (gt) | 1.000 | — | 1.000 | — | 0 | — |
| scene_couple (gt) | 1.000 | 1.00 | 1.000 | 1.00 | 0 | — |
| scene_family3 (gt) | 1.000 | 1.00 | 1.000 | 1.00 | 0 | — |
| scene_meeting4 (gt) | 0.597 (k=2/4) | 1.00 | 0.597 | 1.00 | 0 | — |
| maggiano3 rubric boundaries | 0.833 | 0.84 | 0.833 | 0.84 | 0 | 0.833 / 0.90 / 1.48 |
| maggiano3 transcript_7utt | 0.702 | 0.796 | **0.491** | **0.508** | 0.85 | 0.491 / 0.504 / 0.85 |
| maggiano3 transcript_8utt | 0.671 | 0.792 | **0.592** | **0.609** | 0.85 | 0.592 / 0.609 / 0.85 |

The 0.85 s Unknown on both transcripts is "I wanna go. No." (30.04–30.89 s,
mom's chant under the son — rubric: mom / overlap). The damage is not that
piece: cutting it re-shapes the neighbouring pieces, the post-split
re-selection validates a DIFFERENT linkage k=3 whose third cluster is the
2.1 s overlapped-chant piece (marginal 0.119, anchor 0.062 — the 2026-08-27
phantom again) and mom folds into dad (`out/unknown012/results.json`,
`k_evaluated`).

**Ship criterion** — purity +0.05 on both transcripts, accuracy −0.03 at most,
no pinned fixture gaining Unknown seconds: pinned fixtures pass (all identical,
0 s), maggiano3 fails badly (purity −0.29 / −0.18, accuracy −0.21 / −0.08).
**Decision: flag defaults OFF** (`UNKNOWN_DEFAULT = False`); the code path is
fully unit-tested (`server/tests/test_diarize_local.py::TestUnknownSpeaker`,
`test_diarize_unknown_downstream.py`, `test_analyze_upload.py`) for a
recording that really holds an unfound voice.

## 2. Does the transcript's own labelling push us wrong?

The owner's three stored recordings (GCS `audio.m4a`; maggiano byte-identical
to `maggiano3`, poker 30.123 s = fixture 30.123 s, family 29.568 s = fixture
29.568 s), each transcribed three times by Deepgram via
`audio_ingest.transcribe_upload` (cached as `transcript_run<i>.json` beside the
audio, private). Deepgram's OWN labels scored with `score.py`; ours =
`diarize_turns` on that transcript; "winner" = `main.py`'s cross-check block
with `MINDSHIFT_DIARIZE_CROSSCHECK=1`.

| recording | run | utt | Deepgram k | Deepgram acc | ours k | ours acc | winner | final acc | guard hurt? |
|---|---|---|---|---|---|---|---|---|---|
| maggiano3 | 1 | 8 | 1 | 0.397 | 3 | 0.671 | local (fallback) | 0.671 | no |
| maggiano3 | 2 | 8 | 2 | 0.384 | 3 | 0.671 | local (3 > 2, changed) | 0.671 | no |
| maggiano3 | 3 | 8 | 1 | 0.397 | 3 | 0.671 | local (fallback) | 0.671 | no |
| poker6 | 1–3 | 7 | 1 | 0.146 | 4 | 0.467 | local (fallback) | 0.467 | no |
| family_real | 1–3 | 5 | 1 | 0.601 | 2 | 0.974 | local (fallback) | 0.974 | no |

Today's Deepgram (nova-3) hears ONE speaker on 8 of 9 runs of these three
recordings (the poker fixture's 2026-08-21 baseline recorded 4 — the vendor has
regressed further); the one 2-speaker maggiano run scores 0.384 vs our 0.671
and our k=3 exceeds its 2, so the never-reduce guard does not fire. **The guard
never kept a worse Deepgram labelling on these recordings; it is left
unchanged.** (Poker's 0.467 is the known welded-utterance / 4-of-6 shortfall,
not a guard effect.)
