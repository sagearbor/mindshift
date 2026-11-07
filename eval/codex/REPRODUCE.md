# Reproduce splits and metrics from a fresh clone

Python 3.12 (built with 3.12.13). Pinned requirements: `eval/codex/requirements.lock.txt`.
All commands run from the repo root. No API keys or paid services.

## 1. Environment
```sh
UV_NATIVE_TLS=1 uv venv --python 3.12 .venv
UV_NATIVE_TLS=1 uv pip sync --python .venv/bin/python eval/codex/requirements.lock.txt
```

## 2. Data (public; ~1.3 GB for CREMA-D)
```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/CheyneyComputerScience/CREMA-D.git eval/codex/data/crema-d
GIT_LFS_SKIP_SMUDGE=1 git -C eval/codex/data/crema-d checkout 1658cd342dff90010aa843eaeebd53610a08b1dc
git -C eval/codex/data/crema-d lfs pull --include='AudioWAV/*' --exclude=''
.venv/bin/python -m eval.codex.benchmark          # writes data/manifest-v2.json (splits by hashed actor id)
.venv/bin/python -m eval.codex.verify_benchmark   # checks every label/vote/speaker/split/audio hash vs benchmarks/cremad-v2.jsonl
```
Split algorithm: `sha256(mindshift-emotion-v1:actor)`, first 20% of actors = test, next 20% = dev,
rest = train → 4489 / 1476 / 1476 clips, 55 / 18 / 18 speakers. One silent clip is quarantined.
MESS and Emozionalmente commands: `INVENTORY.md`.

## 3. Pretrained encoders (public, pinned revisions)
```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python -c "from huggingface_hub import snapshot_download; from eval.codex.model import REVISIONS, ROOT; [snapshot_download('openai/whisper-'+name, revision=revision, allow_patterns=['config.json','preprocessor_config.json','model.safetensors'], cache_dir=ROOT/'artifacts/hf') for name,revision in REVISIONS.items()]"
```

## 4a. Use the original trained weights (fast path)
```sh
git clone git@github.com:sagearbor/mindshift-emotion-private.git ../mindshift-emotion-private   # PRIVATE
rsync -a ../mindshift-emotion-private/results/ eval/codex/results/
rsync -a ../mindshift-emotion-private/artifacts/ eval/codex/artifacts/   # feature caches, if present
cd eval/codex
while read f _; do cat "$f".part-* > "$f" && rm "$f".part-*; done < ../../../mindshift-emotion-private/SPLIT-FILES.txt
shasum -a 256 -c ../../../mindshift-emotion-private/SHA256SUMS
```

## 4b. Or retrain (new run names; run dirs are immutable)
```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m eval.codex.model --size small              # feature cache
.venv/bin/python -m eval.codex.train --size small --out eval/codex/results/small-all-repro
.venv/bin/python -m eval.codex.finetune --source eval/codex/results/small-all-repro --out eval/codex/results/small-finetuned-repro
.venv/bin/python -m eval.codex.export --run eval/codex/results/small-finetuned-repro
.venv/bin/python -m eval.codex.freeze --runs small-finetuned-repro --out eval/codex/results/repro-freeze.json
.venv/bin/python -m eval.codex.evaluate --run eval/codex/results/small-finetuned-repro --freeze eval/codex/results/repro-freeze.json
```
Expected (original `small-finetuned-v1`, CREMA-D test, 1476 clips): float accuracy 0.7360, int8 0.7353,
macro-F1 0.638. Always-neutral baseline: 0.5205. Fine-tuning on MPS is not bit-reproducible;
expect small differences on a retrain.

## 5. Tests
```sh
.venv/bin/python -m pytest eval/codex/tests -q
```
