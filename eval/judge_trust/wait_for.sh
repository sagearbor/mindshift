#!/bin/bash
# wait_for.sh FILE MIN_LINES [MAX_SECONDS] -- block until FILE has >= MIN_LINES lines (bounded wait).
f="$1"; n="$2"; max="${3:-540}"; t=0
while [ "$(wc -l < "$f" 2>/dev/null || echo 0)" -lt "$n" ] && [ "$t" -lt "$max" ]; do sleep 10; t=$((t+10)); done
wc -l < "$f"
