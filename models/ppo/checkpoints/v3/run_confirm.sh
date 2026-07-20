#!/bin/zsh
# M7 Job 5.2 step 2: raw confirmations on top sweep candidates
# (mirrors models/ppo/checkpoints/bcft/confirm_results.txt convention: 500 vs Random, 200 vs DamageFirst)
cd /Users/laithassaf/Documents/Programs/Archived/pokemon-showdown
export PATH="/Users/laithassaf/Documents/Programs/Archived/pokemon-showdown/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
PY=.venv/bin/python3
OUT=models/ppo/checkpoints/v3/confirm_results.txt
: > "$OUT"
for ckpt in ppo_step_5000002_final.pt ppo_step_4750072.pt ppo_step_2500025.pt; do
  path=models/ppo/checkpoints/v3/$ckpt
  echo "=== $path | vs random (500) ===" >> "$OUT"
  $PY models/evaluate.py --model ppo --obs-v3 --checkpoint "$path" --battles 500 2>&1 | /usr/bin/tail -4 >> "$OUT"
  echo "=== $path | vs damagefirst (200) ===" >> "$OUT"
  $PY models/evaluate.py --model ppo --obs-v3 --checkpoint "$path" --opponent damagefirst --battles 200 2>&1 | /usr/bin/tail -4 >> "$OUT"
done
echo "CONFIRM DONE" >> "$OUT"
