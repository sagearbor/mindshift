#!/usr/bin/env bash
# One round of the heat map: ingest whatever corpora have landed, score the
# tone-model reference for anything new, rebuild the graph. Idempotent — every
# loader skips archives that are still downloading and work already done, so
# this can be re-run as downloads complete.
#
#   scripts/heat_round.sh            # everything that is on disk
#
set -uo pipefail
cd "$(dirname "$0")/.."
PY="nice -n 10 tmp/venv/bin/python"
echo "== ingest =="
for loader in confer_corpus meld_corpus sbcsae_corpus voxconverse_corpus chime6_corpus; do
  echo "-- $loader"
  $PY scripts/$loader.py 2>&1 | tail -3 || true
done
echo "== reference (tone model) — new recordings only =="
$PY scripts/heat_reference.py 2>&1 | grep -vE "Fetching|UserWarning|key_padding" | tail -8
echo "== graph =="
$PY scripts/heat_map.py 2>&1 | tail -2
$PY - <<'EOF'
import json, numpy as np
rows = json.load(open("tmp/heat-map/heat_map.json"))
print(f"\n{len(rows)} recordings on the graph:")
for c in sorted({r["corpus"] for r in rows}):
    g = [r for r in rows if r["corpus"] == c]
    ag = [r["agreement_loudness_vs_arousal"] for r in g if r.get("agreement_loudness_vs_arousal") is not None]
    lp = [r["loud_but_pleasant"] for r in g if r.get("loud_but_pleasant") is not None]
    print(f"  {c:18} n={len(g):3}  volatility sd {np.mean([r['db_sd'] for r in g]):4.2f}  "
          f"agreement {np.mean(ag) if ag else float('nan'):+.2f}  loud&pleasant {np.mean(lp) if lp else float('nan'):.2f}  "
          f"dose {np.mean([r['dose_per_hour'] for r in g]):6.1f}/h")
EOF
