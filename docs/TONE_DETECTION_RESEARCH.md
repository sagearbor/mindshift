# Tone Detection Library Research

**Author:** Agent C (Sophie)
**Date:** 2026-02-25
**Context:** MindShift V3 (Tier 2) — audio tone analysis for empathy coaching

---

## Purpose

MindShift needs to detect emotional tone (frustration, sarcasm, defensiveness, sadness, warmth) from speech audio to power the Pleasantness Score and adjust coaching suggestions. This document compares six candidate libraries/APIs identified in the PRD.

---

## Library Comparison

| Library | License | Cost | Latency | Accuracy | Integration Effort | Emotion Granularity |
|---------|---------|------|---------|----------|-------------------|---------------------|
| **Hume AI API** | Proprietary (SaaS) | Free tier: 5 min EVI/mo; Starter $3/mo; Growth $25/mo; Scale $225/mo; Business $500/mo | <300ms (EVI 3) | High — trained on large-scale expression data; proprietary benchmarks | Low — REST API, Python SDK | 48 emotion dimensions (FACS-based), continuous scores |
| **GPT-4o Audio** | Proprietary (OpenAI API) | Per-token pricing (~$2.50/1M input tokens for audio) | ~500ms–1s (API round-trip) | High for sentiment/emotion in context; not purpose-built for acoustic emotion | Low — OpenAI SDK, multimodal input | Free-form text labels; no standardized emotion taxonomy |
| **SpeechBrain** | Apache 2.0 | Free (self-hosted compute) | ~100–300ms per utterance (GPU); ~500ms–1s (CPU) | 78.7% on IEMOCAP (wav2vec2 model) | Medium — HuggingFace model hub, PyTorch dependency | 4–9 categorical emotions (model-dependent) |
| **wav2vec2 (fine-tuned)** | Apache 2.0 / MIT (model-dependent) | Free (self-hosted compute) | ~100–300ms per utterance (GPU) | 78–80% on IEMOCAP; varies by fine-tuning | Medium — HuggingFace Transformers, requires fine-tuning pipeline | Categorical (angry, happy, sad, neutral); customizable with fine-tuning |
| **pyAudioAnalysis** | Apache 2.0 | Free | <50ms (feature extraction); classification adds ~10–50ms | ~65–72% with traditional ML classifiers on standard benchmarks | Low — pure Python, pip install | Configurable via classifier training; typically 4–6 categories |
| **openSMILE** | Dual: free for research/education; commercial license required for products | Free for research; commercial license fee (contact audeering) | <50ms (feature extraction) | Feature extractor only — accuracy depends on downstream classifier; widely used in SOTA systems | Medium — C++ core with Python bindings; features feed into separate classifier | N/A (feature extractor); enables any taxonomy via downstream model |

---

## Detailed Notes

### Hume AI API
- **Strengths:** Purpose-built for emotion from speech/face/text. 48-dimensional continuous emotion output is the most granular option. EVI 3 achieves sub-300ms latency. Expression Measurement API allows batch analysis of audio clips. Active development with Octave 2 (50% cost reduction in Oct 2025).
- **Weaknesses:** Proprietary — vendor lock-in risk. Cost scales with usage. Requires internet connectivity. Emotion taxonomy is Hume-specific (not directly compatible with academic emotion labels).
- **MindShift fit:** Best for production V3. The 48-dimension output maps well to our Pleasantness Score dimensions (warmth, calmness, respect). API-first design means minimal backend changes.

### GPT-4o Audio
- **Strengths:** Multimodal — can reason about tone AND content simultaneously. Already in our stack (LLM layer). Can generate nuanced, contextual emotion descriptions.
- **Weaknesses:** Not optimized for acoustic emotion classification. Higher latency than dedicated audio models. Expensive at scale for continuous audio processing. No structured emotion taxonomy — outputs are free-text.
- **MindShift fit:** Good for Tier 0/1 text+audio hybrid scoring. Not ideal as primary acoustic emotion classifier but useful as a second opinion or for complex contextual tone analysis.

### SpeechBrain
- **Strengths:** Apache 2.0, fully open source. Pre-trained wav2vec2 model on IEMOCAP available on HuggingFace. Active community. Good baseline accuracy (78.7%). Can be fine-tuned on custom data.
- **Weaknesses:** Limited to categorical emotions out-of-the-box. Requires GPU for real-time inference. PyTorch dependency adds deployment complexity. IEMOCAP-trained model may not generalize well to naturalistic couple conversations.
- **MindShift fit:** Strong open-source baseline. Recommended for development/testing and as fallback if Hume AI is too expensive. Fine-tune on MSP-Podcast or custom couple data for production.

### wav2vec2 (standalone fine-tuned)
- **Strengths:** Same underlying architecture as SpeechBrain model but can be customized freely. Multiple pre-trained emotion models on HuggingFace. Can target specific emotions relevant to MindShift (defensiveness, sarcasm).
- **Weaknesses:** Requires ML expertise for fine-tuning. Model selection is fragmented across HuggingFace. No unified framework — more DIY.
- **MindShift fit:** Best for custom emotion dimensions not covered by off-the-shelf models. Consider if we need defensiveness/sarcasm detectors specifically trained for couple conversations.

### pyAudioAnalysis
- **Strengths:** Lightweight, pure Python, Apache 2.0. Fast feature extraction. Good for prototyping and feature exploration. Low resource requirements.
- **Weaknesses:** Lower accuracy than deep learning approaches. Traditional ML classifiers (SVM, RF) cap out around 65–72% on emotion recognition. Not actively maintained (last major update 2020).
- **MindShift fit:** Useful for quick prototyping and as a lightweight feature extractor. Not recommended as primary classifier for production.

### openSMILE
- **Strengths:** Research-grade feature extraction used in INTERSPEECH challenges. Extremely fast. eGeMAPS and ComParE feature sets are industry standards. Python bindings available.
- **Weaknesses:** Dual license — commercial use requires paid license. Feature extractor only, not end-to-end. Adds pipeline complexity (extract → classify). Steep learning curve for configuration.
- **MindShift fit:** Excellent as a feature front-end feeding into a custom classifier. Combine with SpeechBrain or a simple NN for best open-source accuracy. License may be problematic if MindShift commercializes.

---

## Recommendations

### MVP / Development Phase (Now)
1. **Primary:** Use Claude (text-based) tone scoring — already planned for Tier 0
2. **Audio prototyping:** SpeechBrain wav2vec2 model for quick baseline experiments
3. **Feature exploration:** pyAudioAnalysis or openSMILE to understand acoustic features

### Production V3 (Tier 2)
1. **Primary:** Hume AI API — best accuracy, lowest integration effort, purpose-built
2. **Fallback:** SpeechBrain wav2vec2 fine-tuned on MSP-Podcast + custom data
3. **Hybrid:** openSMILE features + custom classifier for dimensions Hume doesn't cover

### Cost-Sensitive Alternative
If Hume AI pricing is prohibitive at scale:
1. Fine-tune wav2vec2 on combination of IEMOCAP + MSP-Podcast + custom labeled data
2. Use openSMILE eGeMAPS features as supplementary signal
3. Deploy on GPU instances (estimated $0.50–2/hr depending on provider)

### Architecture Decision
Recommend building a **ToneAnalyzer abstraction layer** in the FastAPI backend that supports swappable backends (Hume API, SpeechBrain, custom model). This lets us:
- Start with text-based Claude scoring (Tier 0)
- Add Hume API when ready (Tier 2)
- Fall back to open-source if needed
- A/B test different backends

---

## References

- [Hume AI Pricing](https://www.hume.ai/pricing)
- [Hume AI Developer Docs](https://dev.hume.ai/intro)
- [SpeechBrain wav2vec2 IEMOCAP Model](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)
- [openSMILE Documentation](https://audeering.github.io/opensmile/)
- [openSMILE Python Bindings](https://audeering.github.io/opensmile-python/)
- [pyAudioAnalysis Paper](https://pubmed.ncbi.nlm.nih.gov/26656189/)
- [OpenAI GPT-4o Audio](https://platform.openai.com/docs/guides/audio)
