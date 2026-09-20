#!/usr/bin/env bash
# Download a large file, resuming for real.
#
# curl's own --retry RESTARTS the transfer from byte 0 after a stall — it does
# not re-apply -C -, so a "resumable" fetch with --retry can go BACKWARDS
# (observed 2026-09-20: SBCSAE 3.18 -> 2.85 GB). This loops curl -C - ourselves
# until the file reaches the server's Content-Length.
#
#   fetch_resume.sh <url> <out> [cookie-jar]
#
# Servers that ignore Range (ibug/CONFER) return 200 + full body on resume; curl
# detects that and errors out rather than appending, so for those the script
# falls back to a single attempt with NO stall-abort (a slow patch is not a
# failure) and reports truncation honestly.
set +u
url=$1; out=$2; jar=${3:-}
cookie=(); [ -n "$jar" ] && cookie=(-b "$jar" -c "$jar")
total=$(curl -sIL "${cookie[@]}" --max-time 60 "$url" | grep -i '^content-length' | tail -1 | awk '{print $2}' | tr -d '\r')
ranges=$(curl -sIL "${cookie[@]}" -r 0-0 --max-time 60 "$url" | grep -ci '^content-range')
echo "target $out: ${total:-?} bytes, range-capable: $ranges"
if [ "$ranges" -gt 0 ]; then
  for attempt in $(seq 1 200); do
    have=$(stat -f%z "$out" 2>/dev/null || echo 0)
    [ -n "$total" ] && [ "$have" -ge "$total" ] && { echo "complete: $have bytes"; exit 0; }
    nice -n 10 curl -sfL "${cookie[@]}" -C - --speed-limit 10000 --speed-time 300 -o "$out" "$url"
    sleep 5
  done
  echo "gave up after 200 attempts at $(stat -f%z "$out" 2>/dev/null) bytes"; exit 1
else
  nice -n 10 curl -sfL "${cookie[@]}" -o "$out.part" "$url"
  have=$(stat -f%z "$out.part" 2>/dev/null || echo 0)
  if [ -n "$total" ] && [ "$have" -ge "$total" ]; then mv "$out.part" "$out"; echo "complete: $have bytes"; exit 0; fi
  mv "$out.part" "${out%.zip}.truncated.zip" 2>/dev/null; echo "TRUNCATED at $have of ${total:-?} bytes (no Range support; kept for salvage)"; exit 2
fi
