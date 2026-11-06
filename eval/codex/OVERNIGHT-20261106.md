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
