# Handoff: move everything off the Codex computer into GitHub (read this first)

From: the owner, via Claude Code, 2026-11-06. Branch: `codex/emotion-eval` (draft PR #218).
The owner is retiring the Codex computer. Everything that exists ONLY on that machine
must be recoverable from GitHub (or a private location the owner names) so Claude Code
on another Mac can continue. Do these steps in order and report each at the end.

## Rules
- Do not touch `main`, do not merge, do not deploy. No paid services or API keys.
- This repo `sagearbor/mindshift` is **PUBLIC**. Never commit: credentials, `.env`, audio
  (except the 9 CC-BY demo clips already tracked), model weights, absolute paths to the
  Codex machine (any absolute home-directory path), or employer/institution names for that machine, in anything NEW you push.
  Do NOT rewrite or scrub files already on this branch: Claude Code will do that after
  you are off this machine.
- Answer keys: do not put labels in filenames or IDs that are public next to the audio.

## 1. Inventory (write `eval/codex/INVENTORY.md` + `eval/codex/inventory.json`)
For EVERY dataset, model, and artifact on this machine, including everything not yet
mentioned in the docs (the owner believes there is much more audio than the 9 clips):
- name, source URL, licence (quote it), hours, clip count, languages, number of speakers
- whether human-rated, how many raters per clip, label set, where the ratings file is
- local path (relative to repo root or home), size in MB, SHA-256 of each archive/file
- how to re-download it (exact script/command) and the date downloaded
- which splits use it, and whether it was used for training, tuning, or evaluation only
- every trained checkpoint/export (small, tiny, v1, v2, agreement-only, etc.): path, size,
  SHA-256, training config, training data, test metrics

## 2. Code and evidence into git
- Commit and push all scripts, configs, split manifests, freeze files, predictions (`.npz`
  / `.jsonl`), reports and tests that are not yet pushed. Run `git status --ignored` and
  explain anything ignored that matters.
- Confirm a fresh clone can reproduce splits and metrics: add `eval/codex/REPRODUCE.md`
  with exact commands, Python version and pinned requirements.

## 3. Weights and non-redownloadable data -> a PRIVATE repo (highest priority)
Target: `github.com/sagearbor/mindshift-emotion-private` (private; the owner creates it).
If it does not exist or is not private, STOP and tell the owner; do not push weights
anywhere else. Push into it, NOT into the public `mindshift` repo:
- ALL trained checkpoints and exports: tiny int8 (9.9 MB), small int8 (94.9 MB, just under
  GitHub's 100 MB per-file limit), v1/v2/agreement-only checkpoints, fine-tuned encoders,
  classifier heads, with a `SHA256SUMS` file and the exact export command per file.
  If a single file is over 95 MB, split it (`split -b 90m`) and document the join command.
- Any audio, features, embeddings or ratings that CANNOT be re-downloaded from a public
  source (owner recordings, custom clips, derived caches that took hours to compute).
- Public corpora (CREMA-D, MESS, Emozionalmente, etc.): download script + hashes only,
  in the public repo's inventory. Do not copy them.
- If the non-redownloadable material exceeds roughly 800 MB total, push the weights and the
  inventory first, then STOP and list what remains with sizes; the owner will pick storage.
- Do the weights FIRST, before anything else in this file. They are the most important item.

## 4. Open questions to leave in `eval/codex/OPEN-QUESTIONS.md`
- The CREMA-D -> MESS drop is mostly a label-space artefact (MESS has no "neutral";
  the model answers neutral on 97% of happy clips). Three-way anger/happy/sad: 93.1% on
  CREMA-D, 54.4% on MESS. The suggested next experiment (neutral-logit offset fitted on
  2 MESS speakers + loudness normalisation, evaluated on the other 4) is NOT done.
- Always-neutral accuracy on the CREMA-D test is 52.05%; report it next to 73.53%.
- No commercially usable, human-rated, natural-English emotion set has been found.

When finished, push to this branch and write a one-paragraph summary at the top of
`PROGRESS.md`. Then the Codex computer can be shut down.
