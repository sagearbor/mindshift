# TRANSFER COMPLETE (2026-11-07)

All Codex-machine work is now in GitHub. The Codex machine can be shut down.

- **Private repo** `sagearbor/mindshift-emotion-private` (verified PRIVATE): all 16 trained
  checkpoints/ONNX exports, 80 classifier-head and frontend-constant files, full run evidence, feature caches, manifests.
  Verified from a fresh clone: 96/96 weight hashes (`SHA256SUMS`) and 22,334/22,334 data hashes
  (`DATA-SHA256SUMS`) match, and all 26,527 source files are present.
  The full per-file list with bytes and SHA-256 is `PRIVATE-FILES.tsv` in that repo.
- **This branch** (`codex/emotion-eval`): code, tests (58 pass in a fresh clone), predictions,
  reports, `INVENTORY.md` + `inventory.json`, `REPRODUCE.md`, `OPEN-QUESTIONS.md`.
- **Not copied (public, re-downloadable):** CREMA-D, MESS and Emozionalmente audio, and the
  Hugging Face model weights. Commands are in `INVENTORY.md`.
- **Non-redownloadable audio:** none existed on the machine.
- **Still to do (Claude Code, other Mac):** scrub machine-local paths and banned wording from
  files that were already on this branch before anything merges to `main`.
  See `OPEN-QUESTIONS.md` item 5.
