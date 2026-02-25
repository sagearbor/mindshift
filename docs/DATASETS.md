# Emotion Speech Datasets for Tone Detection

**Author:** Agent C (Sophie)
**Date:** 2026-02-25
**Context:** MindShift Tier 2 testing — ground-truth labels for tone detection model evaluation

---

## Purpose

MindShift's tone detection system (V3) needs ground-truth labeled audio to:
1. Evaluate model accuracy (confusion matrices, F1 scores)
2. Fine-tune open-source models (SpeechBrain, wav2vec2)
3. Validate the Pleasantness Score dimensions against established emotion labels
4. Run automated agent validation loops (PRD §7, Tier 2 tests)

---

## Dataset Comparison

| Dataset | Size | Speakers | Emotion Labels | Modalities | Style | Access | License/Cost |
|---------|------|----------|---------------|------------|-------|--------|-------------|
| **IEMOCAP** | ~12 hrs, 10,039 utterances | 10 actors (5M/5F) | 9 categories (angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted) + valence/activation/dominance dimensions | Audio + Video + Text | Scripted + Improvised dyadic | Request from USC (academic agreement) | Free for research; no commercial use |
| **MELD** | ~13 hrs, 13,708 utterances | 304 speakers (Friends TV) | 7 emotions (anger, disgust, sadness, joy, neutral, surprise, fear) + 3 sentiments (pos/neg/neutral) | Audio + Video + Text | Naturalistic TV dialogue | [Public GitHub](https://github.com/declare-lab/MELD) | Free (CC BY-NC-SA 4.0) |
| **MSP-Podcast** | 400+ hrs, 100K+ utterances | 1,000+ speakers | Categorical emotions + valence/activation/dominance (continuous) | Audio + Text | Naturalistic podcast speech | Request from UTD (academic agreement) | Free for research (Common License audio sources) |
| **RAVDESS** | ~7 min per actor × 24 actors, 7,356 files | 24 actors (12M/12F) | 8 emotions (calm, happy, sad, angry, fearful, surprise, disgust, neutral) × 2 intensity levels | Audio + Video + Song | Acted, controlled | [Zenodo (public)](https://zenodo.org/records/1188976) | CC BY-NC-SA 4.0 |
| **CMU-MOSI** | ~2 hrs, 2,199 utterances | 93 speakers | Sentiment intensity (−3 to +3 continuous) + subjectivity | Audio + Video + Text | Naturalistic YouTube opinions | [CMU MultiComp](http://multicomp.cs.cmu.edu/) | Free for research |

---

## Detailed Notes

### IEMOCAP (Interactive Emotional Dyadic Motion Capture)
- **Strengths:** Gold standard for speech emotion research. Dyadic conversations are closest to MindShift's couple use case. Both scripted and improvised sessions. Dimensional annotations (valence/activation/dominance) map to Pleasantness Score dimensions. Most cited dataset in SER literature.
- **Weaknesses:** Only 10 speakers — limited speaker diversity. Acted emotions may not reflect real couple dynamics. Access requires formal request to USC SAIL lab (1–2 week turnaround). No commercial license.
- **MindShift fit:** Essential for benchmarking. The dyadic conversation format directly mirrors our use case. Use for model evaluation and initial fine-tuning.

### MELD (Multimodal EmotionLines Dataset)
- **Strengths:** Large, multi-party dialogue with natural turn-taking. 7 emotion + 3 sentiment labels. Text transcripts included. Publicly available on GitHub. Multi-speaker conversations capture group dynamics.
- **Weaknesses:** TV dialogue (Friends) — not naturalistic real conversation. Audio quality varies. Some emotion labels may not match real therapeutic contexts. CC BY-NC-SA license restricts commercial use.
- **MindShift fit:** Good for testing multi-turn conversation analysis. The dialogue structure (multi-party, turn-taking) is relevant. Use as supplementary training data alongside IEMOCAP.

### MSP-Podcast
- **Strengths:** Largest dataset by far (400+ hrs). Highly diverse speakers (1,000+). Naturalistic speech from real podcasts. Both categorical and dimensional annotations. Most representative of real-world audio conditions (varying recording quality, background noise).
- **Weaknesses:** Podcast speech ≠ intimate couple conversation. Access requires formal request to UTD MSP lab. Not all segments have emotion labels. Speaker overlap is rare (mostly monologue).
- **MindShift fit:** Best for robust model training due to scale and diversity. Naturalistic audio conditions will improve model generalization. Use as primary training set for fine-tuning wav2vec2/SpeechBrain models.

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Strengths:** Clean, controlled recordings. Two intensity levels per emotion. Includes both speech and song. Easily accessible on Zenodo (no application required). Good for quick model prototyping.
- **Weaknesses:** Acted, single-sentence utterances — not conversational. Small scale. Does not represent real interaction patterns. Controlled studio conditions don't match real-world audio.
- **MindShift fit:** Useful for initial model sanity checks and unit tests. Clean labels make it good for automated testing pipelines. Not sufficient for production model training.

### CMU-MOSI (Multimodal Opinion Sentiment Intensity)
- **Strengths:** Continuous sentiment scores (not just categories). Multimodal with aligned audio/video/text. Real YouTube speakers. Good for fine-grained sentiment analysis.
- **Weaknesses:** Small (2 hrs). Opinion/sentiment focused, not emotion-focused. Monologue format — no dialogue interaction. Sentiment ≠ emotion (different constructs).
- **MindShift fit:** Useful for calibrating the continuous Pleasantness Score scale (0–100). The −3 to +3 sentiment intensity maps to our scoring system. Supplementary only — too small and too focused on opinion rather than interpersonal emotion.

---

## Recommendations

### For Model Evaluation (Benchmarking)
**Primary:** IEMOCAP — the dyadic conversation format is the closest match to MindShift's couple use case. Use as the standard benchmark for comparing model variants.

### For Model Training (Fine-Tuning)
**Primary:** MSP-Podcast — scale and diversity make it the best training set.
**Secondary:** IEMOCAP + MELD combined — adds conversational dynamics.

### For Automated Testing Pipelines
**Primary:** RAVDESS — clean, labeled, easy to download, good for CI/CD test suites.
**Secondary:** MELD — publicly available with text transcripts for integration tests.

### For Pleasantness Score Calibration
**Primary:** CMU-MOSI — continuous sentiment scores help calibrate our 0–100 scale.
**Secondary:** IEMOCAP dimensional annotations (valence/activation/dominance).

### Acquisition Priority
1. **RAVDESS** — download immediately (public, no application) → use for test fixtures
2. **MELD** — clone GitHub repo → use for development testing
3. **IEMOCAP** — submit access request to USC SAIL → primary benchmark
4. **MSP-Podcast** — submit access request to UTD MSP → primary training data
5. **CMU-MOSI** — download from CMU MultiComp → Pleasantness Score calibration

### Custom Dataset (Future)
For production accuracy on couple conversations, we will eventually need a custom labeled dataset:
- Record consenting therapy sessions (with IRB approval)
- Label with MindShift-specific dimensions (warmth, defensiveness, sarcasm, constructiveness)
- Target: 50–100 hours for meaningful fine-tuning
- This is a V4+ goal — use public datasets for V3

---

## References

- [IEMOCAP — USC SAIL](https://sail.usc.edu/iemocap/)
- [MELD — GitHub](https://github.com/declare-lab/MELD)
- [MSP-Podcast — UTD](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
- [RAVDESS — Zenodo](https://zenodo.org/records/1188976)
- [CMU-MOSI — MultiComp](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)
- [SER Datasets Collection](https://github.com/SuperKogito/SER-datasets)
