#!/usr/bin/env bash
# M9 Phase 2a — does format alignment matter at the BC stage?
#
# Two BC checkpoints, identical recipe, differing only in which replay corpus
# they saw: mixed (gen1randombattle + gen1ou, equal sampling weight) vs
# gen1randombattle-only. Both raw-policy, no search.
#
# n=5000, not the doc's n=2000: BC checkpoints sit near p=0.45, where binomial
# variance is at its maximum. The n=2000 standing recommendation in
# docs/EVALUATION-METHODOLOGY.md was derived at p=0.93 (the M7 operating point)
# and resolves only ~+4.5pp here. n=5000 resolves ~+2.8pp and costs ~3 min.
set -euo pipefail
cd "$(dirname "$0")/../.."

N=${N:-5000}
OUT=models/checkpoints/m9p2a_ab_results.txt

run() {  # run <label> <checkpoint> <opponent>
  echo "=== $1 vs $3 (n=$N) ===" | tee -a "$OUT"
  python3 models/evaluate.py --model ppo --obs-v3 --checkpoint "$2" \
      --opponent "$3" --battles "$N" --device cpu 2>&1 \
    | grep -E "^Win rate|^Battles|^Opp-prediction" | tee -a "$OUT"
}

: > "$OUT"
echo "M9 Phase 2a BC A/B — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT"
for opp in random damagefirst; do
  run bc-mixed models/checkpoints/bc_mlp_gen1_v3.pt     "$opp"
  run bc-rb5   models/checkpoints/bc_mlp_gen1_v3_rb5.pt "$opp"
done
echo "done — analyse with scripts/bot_eval_ab.py" | tee -a "$OUT"
