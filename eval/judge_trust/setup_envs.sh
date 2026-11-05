#!/bin/bash
# Create the judge-trust virtualenvs under the main checkout's tmp/ (gitignored).
# venv      : API judges (google-genai, anthropic, openai) + stats
# venv-e2v  : emotion2vec+ audio-only baseline (torch + funasr)
# venv-mlx  : local open-weight text judge (mlx-lm, Qwen3)
set -u
OUT="${JT_OUT:-/Users/sagearbor/projects/githubs/mindshift/tmp/judge-trust-20261105}"
PY=/opt/homebrew/bin/python3.12
mkdir -p "$OUT/logs"
cd "$OUT" || exit 1
[ -d venv ] || { $PY -m venv venv && ./venv/bin/pip install -q google-genai anthropic openai numpy soundfile scipy; }
[ -d venv-e2v ] || $PY -m venv venv-e2v
./venv-e2v/bin/pip install -q torch torchaudio funasr modelscope soundfile numpy 2>&1 | tail -5 > logs/e2v-install.log
[ -d venv-mlx ] || $PY -m venv venv-mlx
./venv-mlx/bin/pip install -q mlx-lm 2>&1 | tail -3 > logs/mlxlm-install.log
echo done >> logs/e2v-install.log
