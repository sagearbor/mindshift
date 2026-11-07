# Open questions (2026-11-07, at Codex-machine retirement)

1. **The CREMA-D → MESS drop is mostly a label-space artefact.** MESS has no "neutral" class,
   and the model answers neutral on 97% of MESS happy clips. Three-way anger/happy/sad accuracy is
   93.1% on CREMA-D and 54.4% on MESS. **Not yet done:** fit a neutral-logit offset plus loudness
   normalisation on 2 MESS speakers, then evaluate on the other 4.
2. **Always report the baseline.** Always-neutral accuracy on the CREMA-D test split is **52.05%**.
   Report it next to the headline **73.53%** (small int8). Macro-F1 (0.634 int8) is the fairer number.
3. **No commercially usable, human-rated, natural (non-acted) English emotion set has been found**
   (search log: `reports/dataset-search-20261106.json`). All current evidence is on acted speech.
4. The v3 class-balanced runs (`tiny/small-balanced-finetuned-v3`) were compared on dev only and
   were never test-scored. Whether to freeze and score one is still open.
5. The public branch still has files with machine-local absolute paths and wording that must be
   scrubbed before anything merges to `main` (handoff rule: Claude Code does this, not Codex).
