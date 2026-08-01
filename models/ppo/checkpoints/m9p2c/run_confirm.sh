#!/usr/bin/env bash
# M9 Phase 2c step 2 — the gate A/B, at the pre-registered n=2000 per arm.
#
# Both baselines are re-run here at n=2000 rather than quoting their historical
# numbers, which were taken at n=150/200/500. Comparing a fresh n=2000 reading
# against an old n=200 one is the defect M9 Phase 1 exists to remove.
#
#   m5.5  bcft/ppo_step_5000000_final.pt   (v2 schema — the pre-registered gate
#                                           baseline in MILESTONES.md)
#   m7    v3/ppo_step_5000002_final.pt     (v3 — the shipping agent and the
#                                           Phase 3 ladder control arm)
#
# Each agent plays in its own native schema against the same bot opponent, so
# the schema difference is a property of the arm, not a confound in the eval.
#
# Pass the 2c candidates as arguments (bare filenames from the sweep), e.g.
#   bash models/ppo/checkpoints/m9p2c/run_confirm.sh ppo_step_5000002_final.pt
set -euo pipefail
cd "$(dirname "$0")/../../../.."

DIR=${DIR:-models/ppo/checkpoints/m9p2c}   # override to reuse for another run
OUT=$DIR/confirm_results.txt
PY=${PY:-.venv/bin/python}
N=${N:-2000}

run() {  # run <label> <checkpoint> <obs-flag>
  for opp in random damagefirst; do
    echo "=== $1 | vs $opp (n=$N) ===" >> "$OUT"
    $PY models/evaluate.py --model ppo "$3" --checkpoint "$2" \
        --opponent "$opp" --battles "$N" 2>&1 \
      | grep -E "^Win rate|^Battles" >> "$OUT"
  done
}

: > "$OUT"
echo "M9 2c confirmations (n=$N/arm) — $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT"
run m5.5 models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt --obs-v2
run m7   models/ppo/checkpoints/v3/ppo_step_5000002_final.pt   --obs-v3
# Label candidates with the run directory, not a hardcoded "2c:" — this script
# is reused across runs via DIR=, and a results file that calls m9p2d's
# checkpoints "2c:" is a misreading waiting to happen.
for ckpt in "$@"; do
  run "$(basename "$DIR"):$ckpt" "$DIR/$ckpt" --obs-v3
done
echo "CONFIRM DONE — analyse with scripts/bot_eval_ab.py" >> "$OUT"
