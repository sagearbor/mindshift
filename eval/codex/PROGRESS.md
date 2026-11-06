# Execution ledger — eval/codex/PLAN.md

- 2026-11-05: Created codex/emotion-eval from refreshed origin/docs/emotion-nudge-plan. Clean starting checkout. 128 GiB Apple Silicon Mac; 2.7 TiB free. Fresh clone lacks old ignored data/models.
- Ruling: native execution with final independent review; no implementation delegation needed. User explicitly requested autonomous work, so no extra design gate.
- Ruling: begin with transparent MIT Whisper encoders and official CREMA-D audio-only human ratings. Broad model audit informs alternatives; do not delay a runnable baseline for a catalogue.
- Ruling: all six emotions, English first, phone primary; watch only a compression comparison. No product code or nudge work.

- Data validation: reconstructed filtered human audio ratings agree with all 7,442 official summary winner sets. Initial unfiltered chat counts corrected in licence audit.
- Source defect: `1076_MTI_SAD_XX.wav` contains all-zero samples. Extraction stopped loudly. Preserved manifest-v1, created manifest-v2 excluding this ONE train clip and recording its hash. 4,489 train / 1,476 dev / 1,476 test; same actor assignment. No model-selection/test information used for exclusion.
- 2026-11-05 user steering: preserve actor intent and listener labels; compare training on actor/listener agreement against all listener labels. Implemented as train-only cohort; broad dev/test unchanged.
- Frozen-feature comparisons, development macro F1: tiny/all 0.5282, tiny/agreement 0.5049, base/all 0.5379, base/agreement 0.5276. Agreement cohort has 1,849 training clips; all cohort 4,489. These are development results, not final test claims. Further encoder fine-tuning authorized and running.
- Ruling: official Qwen2.5-Omni local teacher comparison uses next-token forced-choice scores rather than generated prose. No token completion cap or empty reasoning output. It is an additional benchmark candidate, not automatically a trusted labeller. Its prior CREMA-D exposure is unknown.
- Full local Qwen audio development benchmark: macro F1 0.38772858; transcript-only 0.12645260. No qualifying teacher. No mass machine-label pool generated; guarded admission utilities are not represented as a completed collection/labelling pipeline.
- Independent review corrected all-input audio hashing, feature-cache provenance checks, finite/complete teacher gates, and composed pseudo-label admission. A second review added pre-test hashes and presence checks for checkpoints, controls and ONNX graphs, plus export validation and saved-report provenance checks. Added regression tests failed before the corresponding fixes and passed afterward; 45 targeted tests pass.
- Full tiny/base/small baseline exports include encoder and head, not just a classifier. Tiny fine-tuned int8 is 9,942,329 bytes; base 23,588,349; small frozen baseline 94,862,824. Real phone/watch latency, memory and battery remain unmeasured.
- Real-audio end-user smoke: infer CLI on 1001_IEO_NEU_XX.wav with tiny int8 returned finite six-class probabilities and experimental_prediction. Silence path returns no emotion. No microphone/app integration was performed.
- Plan scope refinement: linear soft-vote heads plus full encoder fine-tuning were implemented; a separate MLP or handcrafted acoustic comparator was not run. Pruning/distillation and watch deployment were not attempted because recognition quality remains the limiting gate. Transcript and training-prior controls are implemented.
- Model-selection protocol will be frozen only after the final small fine-tune and its export validation complete. All final test outputs remain absent as of this entry.

- Final small fine-tune stopped at 15 epochs, selected epoch 9 on development only (macro F1 0.59168299). Int8 development macro F1 0.59319436; full float/quantization validation passed.
- Frozen protocol committed as 0635886; nine candidates evaluated once. Selected small int8 test macro F1 0.63363193, accuracy 0.73527218, actor-bootstrap 95% interval 0.60476129–0.65610899. No subsequent model tuning.
- Held-out cohort comparison reverses development: agreement-only beats all-vote frozen heads for tiny/base/small. Retain both training options and full labels; do not claim all-vote training universally wins. Selected fine-tuned all-vote small checkpoint still has highest measured F1 in this fixed set.
- High-calibre goal remains incomplete: disgust/fear/sadness recall below provisional 60%; no natural-speech or real-device validation, no trustworthy automatic teacher, no mass pseudo-label pool. Next round requires stronger candidates and independent human natural-speech evidence, plus actual device measurements.
- Final targeted suite: 46 passed. Frozen portable index verified for all 7,441 clips. Independent final review matched all nine reports to raw predictions/metrics; clarified distinct tiny-size versus small-accuracy cards. HTML structure checked; browser visual preview blocked by local-file URL policy, not bypassed. App file preview queued.

## 2026-11-06 03:34 ET continuation (supersedes earlier stopping state)

- Device testing remains excluded by user direction. New development experiments
  are authorized; original frozen test reuse is explicitly regression evidence.
- Agreement-only fine-tunes completed for tiny/base/small. All three were worse
  on development than corresponding all-clip fine-tunes; old-test changes mixed.
- Local native 120-clip screens: Qwen3-Omni MLX macro F1 .307907, Liquid .048309,
  MERaLiON .569148. MERaLiON trained on CREMA-D, so its number is overlap-only.
  No automatic teacher qualified. EmotionThinker still running.
- Added existing human-rated Italian corpus: 6,902 recordings, five ratings each.
  Frozen 953-clip diagnostic with unchanged small int8 scored .262938 macro F1.
- Added MESS English: 1,800 recordings, six actors, human A/C/H/S distributions,
  publisher CC BY 4.0. Three-shared-emotion protocol frozen in 3a32962 before
  inference: 1,270 clips, exclude 530 calm/tied; no calm-to-neutral substitution.
  Existing small int8: 230 correct (18.11%), macro F1 .212916. Full-precision
  sanity check matched int8 choices on 15/15 fixed examples. Transfer is poor.
- Independent review confirmed cohorts, raw metric recomputation, audio mapping,
  resampling and frontend settings. Preserve actor intentions and all ratings.
- Tiny balanced CPU run failed to improve even its starting classifier; retained
  epoch-zero fallback, not promoted. Small weighted MPS run queued behind teacher.
- Latest 58 targeted tests passed; three valid report SVGs and 16 collapsed
  accordions. New top chart makes cross-corpus weakness explicit. Browser preview
  still blocked; no browser-rendering claim. Main report path unchanged.
- Full live job/session details and next work in OVERNIGHT-20261106.md. Research
  remains incomplete: finish teacher and small balancing, review/report results.
