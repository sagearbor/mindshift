#!/bin/bash
# Full emotion-labeller run (CREMA-D 300 clips + TTS leakage set). Logs bounded with tail.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${JT_OUT:-/Users/sagearbor/projects/githubs/mindshift/tmp/judge-trust-20261105}"
PY="$OUT/venv/bin/python"
cd "$HERE" || exit 1
$PY emotion_labellers.py --models gemini-3.8-flash gemini-3.5-flash-lite \
    --conds audio_only text_only both contradict tts --workers 8 2>&1 | tail -n 200 > "$OUT/logs/emotion-gemini.log"
$PY emotion_labellers.py --models gpt-audio-1.5 --per-class 20 \
    --conds audio_only text_only contradict tts --workers 6 2>&1 | tail -n 200 > "$OUT/logs/emotion-gpt.log"
echo "all done" >> "$OUT/logs/emotion-gpt.log"
