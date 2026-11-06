# Authorized overnight research

User approved autonomous execution of the visual timeline on 2026-11-06,
including branches/worktrees and merging our work if needed. Work in the existing
`codex/emotion-eval` research branch; no product deployment or nudge changes.
No phone testing, paid services, or purchased human ratings. Liquid's commercial
revenue threshold is acceptable to the user; preserve the precise conditions.

## Execution plan and gates

1. Run unchanged existing fine-tuning procedure on agreement-only train cohort:
   tiny, base, small, 20 maximum epochs, development early stopping. New run names
   `SIZE-agrees-finetuned-v2`; never overwrite v1 evidence. Compare development
   scores to existing all-clip fine-tunes. Freeze new choices before regression
   evaluation on the old test; do not claim that test is fresh confirmation.
2. Overlap pinned model metadata/downloads and research with GPU work. Probe
   Liquid LFM3-Audio-2B, EmotionThinker, Qwen3-Omni in separate runtime environments
   where needed. Audit actual downloaded licence and code, not search snippets.
   Every candidate must process real audio successfully before a batch run.
   Queue large GPU jobs; let ready candidate probes run between training jobs.
3. Benchmark unchanged viable candidates on a shared development subset, then
   expand promising ones. Preserve prompt, exact revision, precision, input
   hashes, predictions, failures, and timing. No machine labels admitted to
   training merely because an inference succeeded. Reasoning generation budgets
   >=40,000 tokens or explicitly disable thinking. Forward-pass scoring has no
   generation budget, and must be labelled a different evaluation configuration.
4. Find existing human-rated natural speech online. Verify source licences and
   candidate training overlap; never describe machine labels as human ratings.
   Missing independent licensed data is a documented limitation, not a reason
   to manufacture success or purchase ratings.
5. Update graphical results and README; targeted tests only, never full repo
   tests. Fresh independent review of new code/results before completion. Commit
   and push verified work to existing draft PR 218; merging is optional and not
   necessary to preserve research artifacts.

## Live execution state

- Tiny training launched as exec session 41367; finished. Log:
  `results/overnight-20261106/tiny-training.log`.
- Base training finished (exec session 17619). Log:
  `results/overnight-20261106/base-training.log`.
- Liquid pinned metadata download session 24181, revision
  `2d6c1f99c129b3ef07a8d168bc0ff08e2bdb93f1`.
- EmotionThinker metadata session 62652, revision
  `f80d3b343ab1eaf903ffc03aa775cfef4bffa850`.
- Qwen upstream current revision from Hub API:
  `26291f793822fb6be9555850f06dfe95f2d7e695`. Hub reports licence `other`;
  inspect actual files before repeating earlier Apache claim.
- 15-minute thread heartbeat `emotion-research-overnight` created to resume work
  after interruptions. Disable when done or genuinely blocked. Never assume a
  process survived: inspect logs, checkpoint metadata, and process status.

## Rulings

- Reuse existing research branch and tested trainer rather than creating a new
  checkout: models/data are local ignored artifacts; no other edits were present.
- Original test already informed the agreement-only hypothesis. New evaluation
  is regression evidence only, even though new checkpoints are selected on dev.
- No automatic merge of model research is needed; keep the draft reviewable.

## Completion status

IN PROGRESS. Consult raw results and logs, not this document alone.

- All three fine-tunes complete. Small best epoch 4, development macro F1 0.571106. Tiny 0.532777; base 0.543625. All below all-clip fine-tuned development scores. New regression protocol frozen in reports/agreement-v2-freeze.json before evaluating these checkpoints.

## Latest checkpoint: native local probes and reporting

- All three old-test regression evaluations finished. Agreement fine-tunes:
  tiny 0.596223, base 0.620962, small 0.631244 macro F1. Development still
  favors all-clip fine-tunes, so no replacement selected. Report now includes
  a visible two-panel comparison and explicitly labels old-test reuse.
- Liquid weights complete. Isolated Transformers 5.22.1 runtime works on MPS.
  `liquid-probe-v1` constrained-token scoring had low answer mass and is not
  reliable evidence. Native `liquid-generate-v4` completed 12 balanced dev clips:
  all neutral, accuracy 2/12, macro F1 0.047619. No empty/truncated outputs.
  Earlier v2 adapter output-shape error and v3 parser issue are preserved.
  v3 also incorrectly called Liquid non-reasoning and used 1024 output tokens;
  observed thinking disproved that. v4 uses 40000; actual checkpoint text
  configuration allows 128000 positions. No machine labels admitted.
- Qwen3 MLX weights complete; mlx-vlm 0.7.12 in isolated mlx-runtime supports
  real audio. One-clip CLI smoke returned neutral. First structured probe
  returned bare E, rejected by the initial tag-only parser. Parser now accepts
  exactly one bare letter or one final answer tag after closed reasoning.
  `qwen3-mlx-probe-v2` session 97063 is the new 12-clip probe; inspect raw health.
- EmotionThinker download failed with a connection reset. Resume session 57444,
  log `results/overnight-20261106/emotionthinker-weights-retry.log`; no score yet.
- Emozionalmente publisher download session 85923, log
  `results/overnight-20261106/emozionalmente-download.log`. Public source
  https://zenodo.org/records/12616095; OpenSLR161 states CC BY 4.0. Italian,
  acted, existing human ratings. Keep separate; not natural-English acceptance.
  Expected archive MD5 398c6502dd2b0e1e810f927c12897547. Verify before extraction.
- 48 targeted tests passed before the last bare-letter parser extension;
  rerun parser tests. Full repository tests were not run.
- Required independent review dispatched to /root/review_overnight; address
  findings before committing this milestone. Existing original draft PR 218.
- Qwen upstream card uses `license: other` plus `license_name: apache-2.0`;
  metadata alone is not a newly discovered commercial restriction.

## Independent review corrections

- Reviewer recomputed regression metrics and found no leakage or numerical errors.
  Legacy frozen selection files say `test unopened`; this means the checkpoint
  had not yet been evaluated, not that this cohort was historically untouched.
  Never edit frozen evidence; protocol/report regression caveat takes precedence.
- Early smoke subset concentrated on two speakers. New selector rotates speakers
  within emotions. Qwen `qwen3-mlx-dev120-speakers-v2` covers all 18 development
  speakers, 20 clips/emotion, macro F1 0.307907, accuracy 44/120. This configuration
  does not meet the teacher goal; no machine labels admitted. Different prompts
  or precision remain unevaluated. Early probes remain smoke evidence only.
- Liquid expanded run `liquid-dev120-speakers-v1` session 43605 uses identical
  120 IDs; pending. EmotionThinker now includes publisher reasoning system prompt.
- Full targeted suite now 49 passed. MLX provenance wording corrected to
  `mlx-community`, not an official Qwen conversion. Weight inventory still pending.
- Emozionalmente reset at 97 MB; resumed with curl range support and stall timeout.
- No phone or nudge work performed. Current HTML includes new regression chart;
  user-facing browser rendering remains unverified due earlier browser restriction.

## Latest completed work and active jobs

- Liquid speaker-balanced v2 finished 120/120 with no empty, truncated or invalid
  outputs. Macro F1 0.048309, accuracy 20/120. Prior v1 failed malformed nested
  answer tags; v2 uses a simpler single-letter prompt and remains separately named.
- Emozionalmente archive checksum verified. Raw
  audit: 6902 WAVs, 34510 ratings (five each), 431 actors; all splits speaker-disjoint.
  Protocol committed c7e7f24 before inference: fixed existing small int8, 953 clips
  from 86 test actors, excludes 358 ties/surprise winners and 23 over six seconds
  from original 1334 clips/88 actors. All seven original vote categories preserved.
- External diagnostic completed: accuracy 0.366212, macro F1 0.262938. No Italian
  training. This is acted Italian, not natural-English acceptance. New source,
  labels, protocol and results are linked in the report. Independent reviewer
  reconstructed the exact 953 IDs and labels from raw archive, no blocking issue.
- EmotionThinker weights finished after standard-HTTP retry. Native 12-clip probe
  running as session 69345, log `emotionthinker-probe-v1.log`; publisher system
  prompt, 40000 token budget, no scoring of failures.
- Additional unchanged specialist MERaLiON-SER-v1 download session 21277:
  revision 7e3ee6fa4534dea8316e8ca43e377e2fbb58496b, artifact `meralion-ser`.
  Runtime `meralion-runtime` installed, Transformers4.57.6/torch2.15/peft/omegaconf.
  Official PDF contains MIT grant plus attribution/modification/redistribution
  and third-party-rights terms, not a pure unqualified MIT label. No revenue
  threshold or paid fee found in the downloaded PDF. Source code reviewed;
  requires cached whisper-medium config abdf7c39ab9d0397620ccaea8974cc764cd0953e.
  `specialist_probe.py` preserves surprise predictions as incorrect (not removed).
  One scoring test RED→GREEN. Run 12-clip smoke after EmotionThinker GPU work.
- Report now compares original Whisper and new local candidates on exactly the
  same 120 IDs and derives metrics from predictions. Review requested an assertion
  for 20/class; added. Phone-testing card removed as out of scope.
- Weight inventory verified SHA256 equals HF metadata etag for all Liquid+Qwen
  files and pinned revisions. Extend to EmotionThinker/MERaLiON once ready.
- Still needed: finish specialist probes, update report with their outcomes,
  final targeted suite + structural HTML verification, commit/push full milestone.

## Important MERaLiON training-overlap finding

- Primary paper https://arxiv.org/html/2511.04914 section4.1.2 explicitly lists
  CREMA-D in training. Its CREMA-D screening scores MUST NOT be presented as
  held-out generalization or used to qualify a teacher. Existing running
  `meralion-dev120-v1` started before this finding; retain a prominent report
  qualification and this correction. New code config records the overlap.
- MERaLiON loading clean (no missing/unexpected/mismatched weights). First adapter
  v1 failed string-key id2label; v2 corrected and completed 12/12. Score is only
  a runtime check. Expanded 120 run uses CPU, session73555. Consider the separate
  Italian corpus for diagnostic transfer, not CREMA-D for model qualification.
- All four downloaded model weight sets now hash-verified against pinned HF
  revision metadata. Inventory saved. Last suite51pass; new label-key test also
  passed, full suite should now be52.

## Current active work (supersedes earlier session states)

- All downloads finished. MERaLiON 120-screen complete: macro F1 0.569148,
  accuracy71/120; known CREMA-D training overlap means diagnostic only.
- EmotionThinker bf16 native runs looped repetitively. Original v1 interrupted
  after450s without scoring; stream-v2 showed repeated tokens, also interrupted
  unscored. Full precision plus repetition penalty1.1 completed a single real
  audio probe correctly in16.8s. These are TWO changed settings, so do not attribute
  the fix solely to precision. New run `emotionthinker-dev120-fp32-v1`, session86949,
  now progressing normally (~17–20s/clip). Retains40000 token budget and logs raw
  generation. Report automatically includes it only after successful completion.
- Started predeclared class-balancing experiment: plan commit edd8ade; power0.5,
  weights from training votes only, retain all soft votes, no test evaluation.
  Tiny CPU job `tiny-balanced-finetuned-v3`, session31668, ~136s/epoch; log
  `tiny-balanced-training.log`. CPU overlaps EmotionThinker GPU.
- Next: after GPU teacher completes, run small balanced training on MPS using
  `python -m eval.codex.finetune --source eval/codex/results/small-all-v1 --out
  eval/codex/results/small-balanced-finetuned-v3 --balance-power 0.5 --epochs 20`.
  Use repo .venv Python. If tiny CPU appears improved, confirm tiny with identical
  MPS execution before attributing changes to weighting or replacing a model.
- Independent reviewer verified class weights, split discipline and MERaLiON
  metric mapping. Added bounds validation as suggested. Latest55 targeted tests
  passed with JUnit XML in reports/tests-overnight.xml. No full repo tests.
- Current report has two valid inline SVGs,16 collapsed accordions, no external
  assets. Browser visual rendering remains blocked; source and SVG checked only.
- Continue work, not finished: complete active jobs, matched-device confirmation
  if justified, calibration comparison, report refresh, commit/push verified work.
