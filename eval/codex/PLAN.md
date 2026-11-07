# Commercial phone emotion model — implementation plan

Goal: reproducible, human-audited emotion evaluation and a small local model, without nudge policy or usefulness work. User authorized autonomous execution on 2026-11-05.

Constraints: no hosted inference or paid APIs; no merge/deploy; PR targets docs/emotion-nudge-plan. Only commercially permitted model/data candidates enter training. All experiment code/data/artifacts stay under eval/codex; audit under docs/research. Python in repository .venv. No full repository test suite during data jobs. Never claim desktop timing proves phone performance.

Design: six CREMA-D categories (anger, disgust, fear, happiness, neutral, sadness), human AUDIO-only vote distributions; intended acted emotion retained separately. Deterministic actor-disjoint train/development/test splits frozen before model fitting. Ties remain in distribution scoring, excluded explicitly from single-label metrics. Machine labels never replace human labels. Original audio SHA256, source commit, provenance and model revision accompany results. A final fixed test set supports regression; model selection uses development only.

Provisional research gates (our choices, not proven product requirements): macro F1 >=0.75, every-class recall >=0.60 on held-out human majority labels; report clustered confidence intervals and human vote ambiguity. Do not call a model high-calibre on acted English alone; natural conversation and real device validation remain separate required gates. Export size target <=100 MB phone, <=25 MB watch stretch goal. Measure Mac CPU inference as a proxy only.

Tasks:
1. Audit licences and available runtime. Save source URLs and licence restrictions. Block unclear/noncommercial models from training. Record corrections to old notes.
2. Tests first for audio-only vote parsing, tie handling, deterministic speaker splits, data/label leakage checks, missing/nonfinite prediction rejection and correct metrics. Build immutable manifest plus test-set index; reject LFS pointers/missing audio. Verify against official summary votes.
3. Run real-audio smoke test, then extract Whisper tiny/base frozen encoder features locally with pinned model versions. Train linear/MLP heads against human distributions; compare transcript-only and low-cost acoustic controls. Select on development only, no test-driven tuning.
4. Export selected full encoder + classifier to ONNX; quantize supported operations; compare probabilities and metrics against float model. Record file sizes, CPU latency, unsupported/silent inputs, and explicit device-validation gap.
5. Evaluate final locked candidates once on test; retain raw predictions and per-emotion metrics, actor-bootstrap 95% intervals, calibration, disagreement and failure counts. Add a guarded pseudo-label pipeline: train-only pool, licensed provenance, no human label replacement, validation-derived confidence thresholds. Scale only if teacher quality qualifies; otherwise quarantine provisional labels and report the failed gate.
6. Run targeted tests, independent whole-branch review, write self-contained HTML results from raw artifacts, commit and open PR to planning branch. Never merge.

Review focus: audio versus video rating confusion; filenames leaking intended labels; speaker/duplicate leakage; pseudo-label confidence mistaken for accuracy; preprocessing mismatch between Python and mobile export.

Progress and rulings are recorded in PROGRESS.md. Substantive metric changes require a new versioned run, never overwriting evidence.
