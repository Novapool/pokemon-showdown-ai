#!/bin/zsh
# M7 Job 5.2 step 1: raw-policy sweep, 150 battles vs Random per checkpoint
# (mirrors models/ppo/checkpoints/bcft/sweep_results.txt convention)
cd /Users/laithassaf/Documents/Programs/Archived/pokemon-showdown
OUT=models/ppo/checkpoints/v3/sweep_results.txt
: > "$OUT"
for ckpt in $(ls models/ppo/checkpoints/v3/ppo_step_*.pt | grep -v seed | sort -t_ -k3 -n); do
  echo "=== $ckpt ===" >> "$OUT"
  python3 models/evaluate.py --model ppo --obs-v3 --checkpoint "$ckpt" --battles 150 2>&1 | tail -5 >> "$OUT"
done
echo "SWEEP DONE" >> "$OUT"
