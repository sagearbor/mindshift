# Codex-machine inventory (2026-11-07)

Machine-readable twin: `inventory.json` (every SHA-256 is there). Paths are relative to `eval/codex/`.
Weights and derived caches live in the PRIVATE repo `github.com/sagearbor/mindshift-emotion-private`
(same paths). Public corpora are NOT copied anywhere: re-download them with the commands below.

## Datasets

### CREMA-D
- Source: https://github.com/CheyneyComputerScience/CREMA-D
- Licence: Open Database License (ODbL-1.0) for the database; Database Contents License (DbCL-1.0) for contents (see licenses/CREMA-D-LICENSE.txt)
- Clips: 7441 · hours: 5.26 · languages: en · speakers: 91
- Human-rated: True · raters per clip: {'min': 4, 'mean': 9.21, 'max': 12}
- Labels: ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
- Ratings file: data/crema-d/finishedResponses.csv (+ finishedEmoResponses.csv); per-clip votes in data/manifest-v2.json
- Local: `data/crema-d (git clone incl. LFS)` (1242.4 MB) · downloaded 2026-11-05
- SHA-256: {"manifest_v1": "bd942c2d28c44639b5e8da49a4933036da99de4e80c4b395baf3b998fad58878", "manifest_v2": "a7b2e7d2e9eca057c882955072473f6b14a364589540cebcbc85a4d456b8a294", "frozen_index": "benchmarks/cremad-v2.jsonl 2b7195c43ba96e99bb39cd1fcd8421625f2411ecd0e04148590e5de79f03bd7a", "per_clip": "sha256 of every WAV in manifest-v2.json rows[].sha256 and benchmarks/cremad-v2.jsonl"}
- Re-download: `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git eval/codex/data/crema-d && GIT_LFS_SKIP_SMUDGE=1 git -C eval/codex/data/crema-d checkout 1658cd342dff90010aa843eaeebd53610a08b1dc && git -C eval/codex/data/crema-d lfs pull --include='AudioWAV/*' --exclude='' && .venv/bin/python -m eval.codex.benchmark && .venv/bin/python -m eval.codex.verify_benchmark`
- Splits / use: {'train': '4489 clips/55 speakers', 'dev': '1476/18', 'test': '1476/18', 'algorithm': 'sha256(mindshift-emotion-v1:actor), first 20% test, next 20% dev, remainder train'} — training (train), tuning/selection (dev), evaluation once after freeze (test)

### MESS (Morgan Emotional Speech Set)
- Source: https://zenodo.org/records/7378320
- Licence: CC BY 4.0 ("license": {"id": "cc-by-4.0"} in Zenodo record metadata)
- Clips: 1800 original; 1270 used · hours: not computed · languages: en · speakers: 6
- Human-rated: True · raters per clip: see reports/mess-human-ratings.json human_distribution (proportions, 10-rater granularity in source xlsx)
- Labels: ['angry', 'calm', 'happy', 'sad']
- Ratings file: data/mess/ratings.xlsx (Zenodo 'Coded stimuli - final MESS.xlsx'); parsed: reports/mess-human-ratings.json
- Local: `data/mess` (336.6 MB) · downloaded 2026-11-06
- SHA-256: {"audio.zip": "1de2336f97854a7c1891b64de32056ecf28e3257795611f233e7ecf0db37af13", "ratings.xlsx": "c541bb4ec078985187abbc8deccc2d57c7292595d8e9081aea61a76353759098"}
- Re-download: `mkdir -p eval/codex/data/mess && curl -L -o eval/codex/data/mess/audio.zip 'https://zenodo.org/api/records/7378320/files/MESS%20compressed.zip/content' && curl -L -o eval/codex/data/mess/ratings.xlsx 'https://zenodo.org/api/records/7378320/files/Coded%20stimuli%20-%20final%20MESS.xlsx/content' && curl -L -o eval/codex/data/mess/source-metadata.json https://zenodo.org/api/records/7378320 && .venv/bin/python -m eval.codex.mess_data   # mess_data checks the md5s; it refuses to overwrite frozen protocols`
- Splits / use: evaluation only (external transfer test); calm-plurality/tied clips excluded (530) — evaluation only

### Emozionalmente
- Source: https://zenodo.org/records/12616095
- Licence: CC BY 4.0 (per https://openslr.org/161/)
- Clips: 6902 samples; 953 test clips evaluated · hours: not computed · languages: it · speakers: 431
- Human-rated: True · raters per clip: 5
- Labels: anger, disgust, fear, happiness, neutral, sadness, surprise (surprise-plurality excluded)
- Ratings file: inside data/emozionalmente/emozionalmente.zip; protocol: reports/emozionalmente-protocol.json
- Local: `data/emozionalmente` (630.5 MB) · downloaded 2026-11-06
- SHA-256: {"emozionalmente.zip": "6a1fed885668374e2802ae4cfd27c6cdf1db652b8856c6e91e74fe61add3d2b9"}
- Re-download: `UNVERIFIED file name: curl -s https://zenodo.org/api/records/12616095 | python3 -c 'import json,sys;[print(f["key"],f["links"]["self"]) for f in json.load(sys.stdin)["files"]]'  then curl -L -o eval/codex/data/emozionalmente/emozionalmente.zip <that link>; verify sha256 6a1fed885668374e2802ae4cfd27c6cdf1db652b8856c6e91e74fe61add3d2b9; then .venv/bin/python -m eval.codex.external_data`
- Splits / use: official speaker splits; test only used — evaluation only

### Gemini manual demo clips (9 MESS clips)
- Source: https://doi.org/10.5281/zenodo.7378320
- Licence: CC BY 4.0
- Clips: 9 · hours: not computed · languages: en · speakers: 
- Human-rated:  · raters per clip: 
- Labels: 
- Ratings file: 
- Local: `data/gemini-manual-20261106 (tracked in public repo)` (1.1 MB) · downloaded 


- Splits / use:  — manual smoke test only

## Non-redownloadable audio

None found. All audio on this machine is from public corpora above (CREMA-D, MESS, Emozionalmente) or repo test fixtures already tracked in git. No owner recordings exist under eval/codex/.

## Public pretrained models (not copied; re-download)

Downloaded public weights, NOT copied anywhere; re-download from Hugging Face. Hashes per file: reports/local-model-weight-inventory.json

- `artifacts/emotionthinker` — 17049.9 MB
- `artifacts/gemini-runtime` — 82.6 MB
- `artifacts/hf` — 45372.0 MB
- `artifacts/lfm3-audio-2b` — 3976.9 MB
- `artifacts/lfm3-runtime` — 1052.7 MB
- `artifacts/meralion-runtime` — 1041.6 MB
- `artifacts/meralion-ser` — 2939.7 MB
- `artifacts/mlx-runtime` — 889.6 MB
- `artifacts/qwen3-omni` — 0.1 MB
- `artifacts/qwen3-omni-mlx` — 20827.9 MB

## Derived feature caches (private repo)

- `artifacts/features-base` — 30.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base-smoke.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base-smoke.npz` — 0.02 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base-smoke.pre-review.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base.npz` — 29.49 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-base.pre-review.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size base`
- `artifacts/features-small` — 44.5 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small`
- `artifacts/features-small-smoke.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small`
- `artifacts/features-small-smoke.npz` — 0.04 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small`
- `artifacts/features-small.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small`
- `artifacts/features-small.npz` — 44.03 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small`
- `artifacts/features-tiny` — 22.7 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny-smoke.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny-smoke.npz` — 0.02 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny-smoke.pre-review.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny.npz` — 22.23 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`
- `artifacts/features-tiny.pre-review.json` — 0.0 MB — regenerate: `HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size tiny`

## Trained runs (weights in private repo)

Training data: CREMA-D train split (manifest-v2). Selection on dev only; test scored once after freeze. Test metrics = CREMA-D test, all 1476 clips (always-neutral baseline accuracy 0.5205).

| Run | float acc | int8 acc | macro-F1 (float) | weight files (MB) |
|---|---|---|---|---|
| `base-agrees-finetuned-v2` | 0.7002 | — | 0.621 | `finetuned.pt` 76.26, `head.npz` 0.04, `transcript_head.npz` 0.0 |
| `base-agrees-v1` | 0.6838 | — | 0.5885 | `head.npz` 0.04, `transcript_head.npz` 0.0 |
| `base-all-v1` | 0.692 | — | 0.5837 | `head.npz` 0.04, `transcript_head.npz` 0.0 |
| `base-finetuned-v1` | 0.6853 | 0.6853 | 0.5948 | `export/emotion-fp32.onnx` 76.31, `export/emotion-int8.onnx` 22.5, `finetuned.pt` 76.26, `head.npz` 0.04, `transcript_head.npz` 0.0 |
| `small-agrees-finetuned-v2` | 0.7099 | — | 0.6312 | `finetuned.pt` 332.87, `head.npz` 0.06, `transcript_head.npz` 0.0 |
| `small-agrees-v1` | 0.7047 | — | 0.6293 | `head.npz` 0.06, `transcript_head.npz` 0.0 |
| `small-all-v1` | 0.7226 | 0.7151 | 0.6114 | `export/emotion-fp32.onnx` 332.96, `export/emotion-int8.onnx` 90.47, `head.npz` 0.06, `transcript_head.npz` 0.0 |
| `small-balanced-finetuned-v3` | dev-only | — | — | `finetuned.pt` 332.87, `head.npz` 0.06, `transcript_head.npz` 0.0 |
| `small-finetuned-v1` | 0.736 | 0.7353 | 0.6379 | `export/emotion-fp32.onnx` 332.96, `export/emotion-int8.onnx` 90.47, `finetuned.pt` 332.87, `head.npz` 0.06, `transcript_head.npz` 0.0 |
| `tiny-agrees-finetuned-v2` | 0.6667 | — | 0.5962 | `finetuned.pt` 29.6, `head.npz` 0.03, `transcript_head.npz` 0.0 |
| `tiny-agrees-v1` | 0.6711 | — | 0.5713 | `head.npz` 0.03, `transcript_head.npz` 0.0 |
| `tiny-all-v1` | 0.6719 | — | 0.5392 | `head.npz` 0.03, `transcript_head.npz` 0.0 |
| `tiny-balanced-finetuned-v3` | dev-only | — | — | `finetuned.pt` 29.6, `head.npz` 0.03, `transcript_head.npz` 0.0 |
| `tiny-finetuned-v1` | 0.6838 | 0.6786 | 0.5795 | `export/emotion-fp32.onnx` 29.63, `export/emotion-int8.onnx` 9.48, `finetuned.pt` 29.6, `head.npz` 0.03, `transcript_head.npz` 0.0 |

Training/export commands per run: private repo `README.md`; `REPRODUCE.md` here.
