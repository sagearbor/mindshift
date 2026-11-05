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
