# Two literature surveys, 2026-09-20 — what detects anger in speech, and conflict in conversation

Two parallel research passes, each verifying links and model ids rather than
working from memory. Condensed; every claim flagged "unverified" by the
researchers is kept flagged. Feeds `scripts/feature_bank.py` /
`feature_bench.py` and the conversation-level knobs in
`scripts/conversation_audit.py`.

## A. Anger vs happy/excited from a single utterance

**The headline reframes our problem.** Our loudness ladder's AUC of ~0.80 on
angry-vs-happy is a *published, structural* limitation of energy+pitch
features, not a bug: GeMAPS/eGeMAPS-style descriptors confuse happiness and
anger because both are high-arousal; the discriminating signal is mostly
**valence**, which correlates weakly with low-level acoustics. The fix is a
valence-carrying signal (a dimensional SER model) or an SSL embedding, not a
better hand-crafted feature.

**Cross-corpus (acted → different acted, and acted → spontaneous):** SSL
embeddings degrade ~0.3–0.6% cross-corpus; eGeMAPS ~17.8–18.5% (Pepino et
al., Interspeech 2021). Our own bench reproduced the *direction* but not the
size: eGeMAPS alone did **not** degrade CREMA-D→RAVDESS (0.854→0.869), while
our prosody features did.

### Candidates, ranked for a CPU night

| # | model / feature set | licence | output | verified number | notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `superb/wav2vec2-base-superb-er` | Apache-2.0 | 4-class | 62.6% IEMOCAP | ~95M params; SSL generalisation |
| 2 | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | Apache-2.0 | 4-class | **78.7%** IEMOCAP | highest cleanly-licensed |
| 3 | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | **CC-BY-NC-SA** | arousal/valence/dominance | — | the valence experiment; *not shippable*; generic `pipeline()` gives garbage — use the card's own `EmotionModel` |
| 4 | `iic/emotion2vec_plus_base` (funasr) | custom FunASR licence | 9-class | unverified (radar charts only) | purpose-built; licence unclear for commercial |
| 5 | eGeMAPSv02 via `opensmile` 2.6.0 | BSD-ish (openSMILE research/commercial split) | 88 functionals | — | 24 ms/clip measured here |
| 6 | Wav2Small (arXiv 2408.13920) | ? | arousal/valence | 9 ms / 5 s on Xeon; valence CCC 0.37 OOD | 72K params, 60 KB — the only watch-plausible one; weights availability unverified |
| 7 | prosody+SSL fusion (INTERSPEECH 2025 naturalistic SER) | — | — | fusion beat either alone | direction evidence only |

What we already ship covers #3's role with a **permissive** model: `tone_id`'s
default backend (Odyssey WavLM, MIT) returns arousal/valence/dominance.

### Hand-crafted features vs angry-vs-happy

No single feature separates them. Loudness/energy percentiles are anger's
top SHAP features but happy is loud too; F0 and rate rise in both; jitter/
shimmer are *not* anger-specific (shimmer peaks for surprise); HNR points to
fear. The one family with a head-to-head number is nonlinear energy operators
fused with MFCC/LPCC (93.8% angry / 91.6% happy, multiclass, single study).
The whole eGeMAPS bundle is documented to systematically confuse the pair
(PMC9571288) — our bench's 0.87 cross-corpus is therefore *better* than the
literature would predict and deserves scrutiny before trust.

### Could not verify
No published binary angry-vs-happy AUC for any method — we may be setting a
benchmark rather than matching one. No per-feature ablation isolating the pair.
No CPU latency for the ~95M-param models (estimate: a few hundred ms per 2–5 s
window). No canonical Whisper-encoder SER checkpoint.

## B. Conflict / escalation at the *conversation* level

**Best evidence:** Kim, Valente, Filippone & Vinciarelli (2014), IEEE TAC —
on the SSPNet Conflict Corpus, **loudness + overlapping speech** feed a GP
regressor reaching ~0.8 correlation with humans' continuous conflict ratings.
Schuller et al. (2013) ComParE baseline 80.8% UAR on the same corpus.

### Ranked rules to test (all now knobs in `conversation_audit.py` / `heat_map.py`)

1. **Overlap / double-talk rate** — the strongest cue, and identity-free in
   principle (telecom double-talk detection). Our own single-mic probe scored
   56%, so tonight's `overlap_ceiling` uses ground truth and is the *ceiling*.
2. **Rising trend** of loudness over a trailing window, not a peak (O'Dwyer,
   Flynn & Murray 2017: 3 s windows / 1 s hop, arousal r=0.52).
3. **Relative loudness** — wearer vs the other voice present, not vs own
   baseline (needs weak speaker-change tracking).
4. **Turn-taking latency / negative gaps** (CONFER framing; coefficients
   paywalled).
5. **Sustained-duration hysteresis + refractory cooldown** — no affect paper
   specifies parameters; justified from alarm design (Cvach 2012: 80–99% of
   instantaneous-threshold hospital alarms are clinically insignificant;
   driver-drowsiness wearables reach ~1.7 false positives/hour with windowed
   criteria). **Target order of magnitude: low single digits per hour.**
6. **Laughter veto** — Gillick et al. 2021 for detection; the veto itself has
   *no* engineering precedent. Tonight's hypothesis, not prior art.
7. Speaking-rate change trend (weak evidence).
8. Prosodic entrainment/divergence (Lee et al. 2010, couples) — defer; needs
   per-speaker separation.

**Recommended design:** 3–5 s feature windows, ~1 s hop; alarm decision over
a rolling 20–30 s requiring rising trend *and* sustained crossing, then a
cooldown.

**Explicitly deprioritised — HR/HRV fusion.** Zhang et al. 2022: voice-only
stress detection 83.0% vs ECG-only 74.1%; adding ECG+face to voice gains
+2.1 points. HRV needs ~5-minute windows; wrist-PPG error grows with motion
(6.8 bpm MAE on a watch at *low* motion). Worth logging (done), not building.

### Could not verify
Exact winning score of the 2013 Conflict sub-challenge; Kim et al.'s feature
ranking; CONFER's per-feature coefficients; couples-therapy per-code accuracy;
any validated "reciprocal vs one-sided escalation" feature (a genuine gap);
laughter-detector primary-source F1; PPG contamination during *talking*.
