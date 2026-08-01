#!/usr/bin/env bash
# One-line status for a detached training run on the home box.
#
# Exists so a watcher on the Mac can poll over ssh without nesting three levels
# of shell quoting (bash -lc "..." inside ssh '...' inside a monitor script),
# which is how the first two attempts at watching M9 2c silently reported
# empty fields while training was in fact fine.
#
# Usage: homebox-run-status.sh <tmux-session> <checkpoint-dir>
# Prints: step=<n> pct=<n> ckpt=<n> done=<0|1> err=<0|1> alive=<yes|no>
set -uo pipefail
cd "$(dirname "$0")/.."

session=${1:?usage: homebox-run-status.sh <tmux-session> <checkpoint-dir>}
dir=${2:?usage: homebox-run-status.sh <tmux-session> <checkpoint-dir>}
log="$dir/train.log"

ckpt=$(ls "$dir"/ppo_step_*.pt 2>/dev/null | grep -v seed | wc -l | tr -d ' ')
line=$(grep -oE '^Step [0-9]+/[0-9]+' "$log" 2>/dev/null | tail -1)
step=${line#Step }; total=${step#*/}; step=${step%%/*}
pct=0; [ -n "${total:-}" ] && [ "${total:-0}" -gt 0 ] 2>/dev/null && pct=$((step * 100 / total))
done=0; grep -q 'Training complete' "$log" 2>/dev/null && done=1
err=0; grep -qE 'Traceback|CUDA out of memory|Killed|MemoryError' "$log" 2>/dev/null && err=1
alive=no; tmux has-session -t "$session" 2>/dev/null && alive=yes

echo "step=${step:-0} pct=$pct ckpt=$ckpt done=$done err=$err alive=$alive"
