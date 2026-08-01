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

DIR=${DIR:-models/ppo/checkpoints/m9p2c}
OUT=$DIR/sweep_results.txt
PY=${PY:-.venv/bin/python}
N=${N:-500}

# Pool seeds are excluded by their filename convention (ppo_step_0_<name>.pt).
# Do NOT filter on "seed" appearing anywhere in the path: a checkpoint dir named
# e.g. "m9seed" would then match every line and the sweep would silently
# evaluate nothing while still reporting SWEEP DONE.
ckpts=$(ls $DIR/ppo_step_*.pt | grep -v '/ppo_step_0_' | sort -t_ -k3 -n)
[ -n "$ckpts" ] || { echo "no checkpoints under $DIR" >&2; exit 1; }

: > "$OUT"
echo "sweep of $DIR (n=$N vs Random) — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
for ckpt in $ckpts; do
  echo "=== $ckpt ===" >> "$OUT"
  $PY models/evaluate.py --model ppo --obs-v3 --checkpoint "$ckpt" \
      --battles "$N" 2>&1 | grep -E "^Win rate|^Battles" >> "$OUT"
done
echo "SWEEP DONE" >> "$OUT"
