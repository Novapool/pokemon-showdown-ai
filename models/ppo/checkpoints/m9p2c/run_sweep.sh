#!/usr/bin/env bash
# M9 Phase 2c step 1 — raw-policy sweep over the run's checkpoints, vs Random.
#
# n=500, not M7's 150: at the ~0.7 win rate these checkpoints reach, 150
# battles is +-7pp, which cannot rank a sweep whose spread is a few points.
# 500 costs ~19 s per checkpoint. See docs/EVALUATION-METHODOLOGY.md.
#
# Runs on the home box, where the intermediate (tier-2) checkpoints live.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

DIR=models/ppo/checkpoints/m9p2c
OUT=$DIR/sweep_results.txt
PY=${PY:-.venv/bin/python}
N=${N:-500}

: > "$OUT"
echo "M9 2c sweep (n=$N vs Random) — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
for ckpt in $(ls $DIR/ppo_step_*.pt | grep -v seed | sort -t_ -k3 -n); do
  echo "=== $ckpt ===" >> "$OUT"
  $PY models/evaluate.py --model ppo --obs-v3 --checkpoint "$ckpt" \
      --battles "$N" 2>&1 | grep -E "^Win rate|^Battles" >> "$OUT"
done
echo "SWEEP DONE" >> "$OUT"
