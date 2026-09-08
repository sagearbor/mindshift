# Voice-separation bake-off — 2026-08-29

Four parallel experiments against one scorer (`score.py`: 10 ms frame accuracy
under the best one-to-one label mapping, over ground-truth speech; overlap
segments credit either speaker) on eight fixtures — six checked-in
(`server/tests/fixtures/audio/`) plus the owner's PRIVATE 3-person restaurant
clip `maggiano3` scored against **his own per-second listen-through rubric**
(`tmp/private_fixtures/maggiano3/rubric.json`, not committed).

Trigger: the shipped diarizer (`server/diarize_local.py`) merged the owner's
son into the owner and — per the rubric — "finds three voices but mixes them"
(frame accuracy 0.64 / 0.57 on the two Deepgram transcript variants; fed the
rubric's own boundaries it collapses to k=2 with mom inside dad).

## Results (frame accuracy; k found / k true)

| approach | family_real (2) | poker6 (6) | maggiano3 (3) | scene_family3 (3) | scene_meeting4 (4) | mean of 8 |
|---|---|---|---|---|---|---|
| **production, transcript utterances** (what ships) | — | — | 0.64 (3) / 0.57 (3) | — | — | — |
| production, clean GT boundaries (`baseline/`) | 1.00 (2) | 1.00 (6) | 0.52 (**2**) | 1.00 (3) | 0.60 (2) | 0.89 |
| A — acoustic features (pitch/F1/MFCC), auto k | 0.89 (2) | 0.41 (5) | 0.62 (2) | 0.66 (2) | 0.60 (2) | 0.77 |
| A — same, k given | 0.89 | 0.46 (6) | 0.62 (3) | 0.90 (3) | 0.75 (4) | 0.80 |
| B — sliding window 1.5 s/0.25 s + spectral eigengap p=0.80 (transcript-free) | 0.96 (2) | 0.81 (7) | **0.76 (3)** | 0.99 (3) | 0.81 (3) | **0.91** |
| C — pyannote 3.1 default (transcript-free) | 0.55 (4) | 0.36 (4) | 0.52 (4) | 0.90 (3) | 0.77 (3) | 0.73 |
| C — our ECAPA + avg-link on correct segments, k given | 0.98 | 1.00 | 0.61 | 1.00 | 1.00 | 0.95 |

Owner-cluster purity on maggiano3: production 0.76 / 0.59; B 0.80.

## What the four found (details in each folder's README)

* **The voice model is not the problem.** Given correct segments, the pinned
  ECAPA + average linkage scores 1.00 on the six poker men and 0.98 on father
  + son (C's control arm), and the dashboard's voiceprint PC1 is a clean
  per-player staircase on poker where pitch is not (Player 1 and Player 5 share
  a 111 Hz median). Simple acoustic features (A) separate the *family* voices
  (pitch 145 / 324 / ~215 Hz; the child's F1) but not six similar men, and
  cannot choose k.
* **pyannote is worse, and its failure is segmentation** (32 % VAD miss on
  poker; fuses the child with a parent 76 % of the time on maggiano3). Can't go
  in the production image either (torch/torchaudio conflict).
* **B is the only approach that beats production on maggiano3** (0.76 vs
  0.64, k=3 found, purity 0.80) — transcript-free windows + spectral
  clustering with eigengap-k — while trailing production on poker (0.81 vs
  1.00; a ±1–2 s GT slop file where one player splits at pooled cosine 0.04).
  Its ceiling on maggiano3 is 0.84: restaurant noise pulls within-speaker
  window cosine down to 0.20.
* **maggiano3's production failure is a partition + gate problem, not a
  threshold**: all three pooled voiceprints are ≤0.24 cosine apart (clearly
  different people, mom's pitch is 2.2× dad's), yet average linkage's 3-way
  split peels a 1-second sliver instead of separating mom, and the duration
  floor rightly rejects that sliver. The k it wants is right; the partition it
  proposes is wrong.
* **Within-cluster coherence is NOT a phantom discriminator** (B measured it:
  phantoms are *more* coherent, being smaller); rejected.
* **VAD gate** (B): `speaker_id`'s absolute 0.01 RMS speech gate drops poker's
  quietest player (RMS 0.0036); a noise-floor-relative gate keeps everyone.

## 2026-08-30 update — B ships as the default engine

`server/diarize_local.diarize_windows_first` runs approach B end to end as
the speaker labelling (window pass → spectral labels at the eigengap k, max
k 8 → mode filter → runs → segments; the transcript's words regrouped by
those segments, uncovered speech labelled by the same timeline) and is
production's default (`MINDSHIFT_DIARIZE_ENGINE=windows`; `utterances` is the
engine above). `baseline/run.py --engine both` produces the two-engine table
(`baseline/results.json` = utterances, `baseline/results_windows.json`), now
including the owner's REAL Deepgram transcripts for poker and family.
Frame accuracy, windows vs utterances (windows' raw segment timeline in
brackets), 4 torch threads:

| input | windows | utterances |
|---|---|---|
| poker, real Deepgram transcript (1 speaker heard, 7 utt) | **0.720** k=7, 2.2 s [0.809] | 0.447 k=4, 4.5 s |
| maggiano3, transcript 7utt / 8utt | 0.694 / 0.681 k=3, purity 0.775, 3 s [0.761] | 0.702 / 0.671 k=3, purity 0.80 / 0.79, 7.6 s |
| family, real Deepgram transcript (1 speaker heard, 5 utt) | 0.949 k=2, 2.1 s [0.959] | 0.974 k=2, 4.7 s |
| maggiano3, rubric boundaries | **0.865** k=3, 3.1 s | 0.833 k=3, 5.2 s |
| family_real / poker6, GT boundaries | 0.980 / 1.000 | 1.000 / 1.000 |
| openai / gptaudio / couple / family3, GT | 1.000 each, 5 s | 1.000 each, 7.5-8 s |
| scene_meeting4, GT | **0.818** k=3, 6.2 s | 0.597 k=2, 9.8 s |

maggiano3's transcripts stay under B's 0.76 because 7-8 % of the rubric's
speech has no words and sits under the speech gate or in sub-0.4 s gaps —
no turn can carry it; the engine's own timeline reproduces B to the frame.

## Decision

Keep the ECAPA model and the k-validation rules. Augment `diarize_local.py`
with B's two pieces plus one fix:

1. **Boundary proposals inside long utterances** from the window pass +
   spectral clustering — a transcript-free change-point source that attacks
   the welded-utterance failure directly (every voice change in the 2–3-voice
   fixtures recovered, zero phantoms), costing time only proportional to the
   long utterances.
2. **Eigengap k as a lower bound** in `_select_k` (it never over-counted a
   2–4-voice fixture).
3. **Better k-way partitions**: when the k-way average-linkage split fails
   validation on a sliver, try the spectral partition at the same k before
   giving up — the maggiano3 case.
4. Noise-floor-relative speech gate in `speaker_id`.

Not adopted: pyannote (C), acoustic features as a replacement (A; a pitch/F1
gap could later serve as independent "two people" evidence), coherence.

Dashboard: `tmp/voice-dashboard-20260829.html` (`D-dashboard/`), raw feature
lines per recording with the rubric as an overlay.
