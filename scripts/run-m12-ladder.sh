#!/usr/bin/env bash
# M12 Phase 5 ladder launcher (home box only — REPO below is an absolute path).
#
# Exists because two things must be on PATH before node starts, and neither is
# by default in any shell Claude or tmux can reach:
#   1. .venv/bin  — ladder-bot.js:143 spawns infer_server.py via a HARDCODED
#      `python3`. System python3 has no torch/numpy. No bot flag can fix this.
#   2. node 22    — node 18 has no global WebSocket, so the bot logs in, prints
#      a healthy banner, then dies at _connect with `ReferenceError: WebSocket
#      is not defined`. nvm loads from ~/.bashrc, which returns early for
#      NON-INTERACTIVE shells -- so `bash -lc` does NOT help, despite what the
#      docs said before 2026-08-07. Only the absolute path is reliable.
#
# --battles is the ABSOLUTE target, not a remainder (ladder-bot.js:36, :453):
# the bot reads the run's existing CSV rows and plays up to the target. Re-run
# this script verbatim after any interruption; do not subtract completed games.
set -euo pipefail
REPO=/home/laith/Projects/pokemon-showdown-ai
export PATH="$REPO/.venv/bin:/home/laith/.nvm/versions/node/v22.20.0/bin:$PATH"
cd "$REPO"
exec node tools/ladder-bot/ladder-bot.js \
  --login-file config/showdown_login.txt \
  --format gen1ou \
  --roster config/rosters/gen1ou-standard.txt \
  --checkpoint models/ppo/checkpoints/m12/ppo_step_5000005_final.pt \
  --run-id m12-ladder \
  --battles 356
