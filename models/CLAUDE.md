Python ML training code and models. Wraps the Node.js Pokemon Showdown gym via a subprocess bridge.

## Directory Overview

| Path | Contents |
|---|---|
| `gym_bridge.js` | Node.js stdio server wrapping `PokemonGymEnv`; spawned as subprocess by `gym_client.py` |
| `gym_client.py` | Python `GymClient` class; spawns `gym_bridge.js` and exposes `reset()`, `step()`, `valid_actions()`, `close()` |
| `vec_gym_client.py` | `VecGymClient(n_envs, structured)` — N parallel bridge subprocesses, pipelined batched `step()`, auto-reset (M3.1); used by both trainers and `evaluate.py` via `--num-envs` |
| `evaluate.py` | CLI script to evaluate a trained checkpoint against `RandomPlayerAI` |
| `q_learning/` | Tabular Q-learning agent (`q_agent.py`, `train.py`) |
| `dqn/` | Deep Q-Network agent (`dqn_agent.py`, `replay_buffer.py`, `train.py`, `checkpoints/`) |
| `ppo/` | PPO agent (`ppo_agent.py`, `trajectory_buffer.py`, `train.py`, `checkpoints/`) |
| `metamon_adapter.py` | Streams Metamon human-replay trajectories as `(obs (12,65), action, done)`; M2.5 |
| `bc_pretrain.py` | Behavior-cloning pretraining on Metamon replays (transformer, M2.5); saves to `checkpoints/bc_pretrain_gen1ou.pt` |
| `replay_adapter_cli.js` | M5.5: batch-converts scraped replay logs (`data/replays/<fmt>/`) into BC shards (`data/replay_trajs/<fmt>/shard-*.jsonl.gz`) via `sim/tools/replay-adapter.ts` (v2 obs, both seats, opp-head labels) |
| `bc_pretrain_mlp.py` | M5.5: BC of the MLP `PPOAgent` (v2 obs) on replay shards — multi-format weighted (gen1randombattle+gen1ou), `--min-rating` + tournament-unrated filter, opp-head aux CE; saves `checkpoints/bc_mlp_gen1.pt` (a normal PPO checkpoint). **Record filter (line 82): rated kept if `>= --min-rating`; unrated kept iff the battle id is NOT a plain `<format>-…` id — so every `smogtours-` tour game is already in, and lowering `--min-rating` does nothing for unrated records.** BC currently consumes 3,949,828 of 8,451,696 gen1ou records (46.7%), of which smogtours is 2.94M = **74%**. Numbers + the tiers it drops: `docs/DATA-INVENTORY.md`. **⚠️ Keep multi-format for anything that feeds RL. The corpus choice pulls in OPPOSITE directions at the two stages, both measured:** M9 Phase 2a found `--formats gen1randombattle` alone beats the mixed corpus **as a BC policy** by +5.6pp vs Random / +4.6pp vs DamageFirst (n=5,000/arm, CIs excluding 0). But M9 Phase 2c then fine-tuned it with the standard 5M-step PPO recipe and it finished **−8.3pp vs Random below the same recipe warm-started from the mixed BC** (n=2,000/arm, same machine; a seed replication put run-to-run spread under 1pp, so this is real). **A better imitator made a worse RL substrate** — the working explanation is that the gen1ou half supplies behavioural breadth that is dead weight for imitation but valuable as an exploration prior and KL anchor over 5M steps. So: `bc_mlp_gen1_v3_rb5.pt` (randbats-only) if you want the strongest *BC-only* policy; `bc_mlp_gen1_v3.pt` (mixed, the default) as a **warm-start or `--bc-anchor` for PPO**. Full numbers: `MILESTONES.md` → M9 Phase 2. |
| `transformer/` | `transformer_policy.py` — shared `TransformerPolicy` net + `load_pretrain_checkpoint()`; `transformer_agent.py` — PPO wrapper (M3); `train.py` — PPO training loop, `checkpoints/{scratch,pretrained}/` |
| `mcts/` | `mcts_agent.py` — inference-time determinized UCT over a PPO checkpoint (M4); `results/` — eval battery logs |
| `collect_value_data.py` | M8 P2: MCTS self-play collection of AlphaZero-style value targets → `.npz` shards (obs, root visits/Q, seat-perspective rewards, game outcome); `--workers` parallel, resumable |
| `value_finetune.py` | M8 P2: retrains a checkpoint's **value head only** (policy/opp head/trunk left bit-identical) on those targets — `--target outcome\|mc\|root` |
| `checkpoints/` | BC pretraining checkpoints |
| `infer_server.py` | M6: stdio JSON inference server over a PPO checkpoint (reverse of `gym_bridge.js`); spawned by `tools/ladder-bot/ladder-bot.js` |

## Bridge Architecture

Python training code cannot call Node.js APIs directly, so `gym_bridge.js` runs as a child process spawned by `gym_client.py` via `subprocess.Popen`. All communication between Python and the bridge is line-delimited JSON over stdin/stdout — one command object per line in, one response object per line out. The bridge wraps `PokemonGymEnv` from `dist/sim/tools/pokemon-gym.js`, so the TypeScript simulator must be compiled before the bridge can run. Build it once with `./build` (or `npm run build`) at the repo root before starting any training script.

**Observation modes (M2/M3.4):** `GymClient()` defaults to the structured `(12, 65)` per-Pokémon token observation. Pass `GymClient(structured=False)` (bridge: `--flat`) for the legacy 100-dim flat vector — required for `q_learning/`, `dqn/`, and `ppo/`-flat, whose networks are hardcoded to that shape. `GymClient(obs_v2=True)` (bridge: `--obs-v2`) selects **schema v2** `(12, 77)` (M3.4): v1 tokens with 12 dims appended per token — 7 boost stages (atk/def/spe/spa/accuracy/evasion/spd, stage/6) + Reflect/Light Screen/Substitute/Leech Seed flags + toxic counter, non-zero only on the two active tokens. Dims 0–64 are byte-identical to v1, so `gym_client.slice_structured_obs()` gives a v1 agent its native view of a v2 observation — this is how v1 checkpoints act as self-play opponents (pool seeding) and head-to-head opponents inside a v2 env.

**Value-head targeting (M8 Phase 2):** `collect_value_data.py` plays games with tuned MCTS driving the searching seat, recording every searched decision. `--opponent` picks who it plays: `checkpoint` (default) is dual-seat self-play against a frozen raw-policy checkpoint with seats alternating per game; `random`/`damagefirst` use the single-seat bridge bot, where the searcher is always p1 and rewards arrive already in its perspective (no sign flip). `damagefirst` matches the distribution Criterion C is scored on — the original dataset was collected against a frozen policy but evaluated against DamageFirst. Recorded per decision: the obs (already sliced to the checkpoint's `obs_size`), the search's root visit counts and root Q, the shaped gym reward accumulated to the next decision **in the searching seat's perspective**, and the game outcome. Targets are derived at training time, so one dataset serves all three: `value_finetune.py --target outcome` (final result, pure AlphaZero), `mc` (discounted return of the shaped rewards, `--gamma`), or `root` (the search's own root Q; decisions where no search ran carry NaN and are dropped). The fine-tune trains **the value head only** — trunk, policy head and opp head come out bit-identical, so the MCTS prior is unchanged and any strength delta is attributable to leaf evaluation (`--unfreeze-trunk` opts out and is not a clean A/B). Output is a normal PPO checkpoint, loadable by `evaluate.py`, `infer_server.py` and the ladder bot unchanged. Throughput: ~600 games/h per worker at `--sims 100` on CPU; `--workers N` runs N independent bridges, and shards flush every `--shard-games` games so a killed run resumes in place.

**Parallelism (M3.1):** the PPO/transformer trainers and `evaluate.py` run `--num-envs` parallel simulations (default 8, ~5x throughput) via `VecGymClient`; `--num-envs 1` reproduces the serial path. `--device {cpu,mps,cuda}` overrides device auto-detection; checkpoints never store the device and are portable Mac↔CUDA. See `docs/ML-TRAINING.md` → **Parallel Training** for benchmarks and the CUDA-machine setup note.

**Opponents & self-play (M3.3):** both trainers and `evaluate.py` take `--opponent` — `random` (legacy `RandomPlayerAI`, never voluntarily switches), `damagefirst` (highest-base-power heuristic, `sim/tools/damage-first-ai.ts`), or (trainers only) `selfplay`. Self-play runs the bridge in dual-seat mode (`gym_bridge.js --selfplay`): each rollout samples a frozen opponent from `--selfplay-pool` (default: the run's own checkpoint dir; 50% newest / 50% uniform; a frozen copy of the current policy until the first checkpoint exists). Reward and training remain p1-only. Baseline: the M2 MLP checkpoint scores 51% vs DamageFirstAI (101/200).

**Mixed-opponent training (M3.4):** `models/ppo/train.py --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2"` samples one opponent family per rollout (weights normalized; mutually exclusive with `--opponent`). The bridge accepts a per-reset opponent override (`{"cmd":"reset","opponent":...}` — `VecGymClient.set_opponent()`/`reset_all(opponent=...)`), so envs switch family at rollout boundaries without respawning; in-flight episodes are abandoned and bootstrapped, the same truncation PPO applies at every rollout end. Seed the pool by copying checkpoints into the run's checkpoint dir named `ppo_step_0_<name>.pt` (step 0 ⇒ never "newest", sampled via the uniform half; skip `ppo_step_0_*` files when sweeping checkpoints for evaluation). v1 (780-dim) seeds work inside an `--obs-v2` run via per-token slicing.

**Head-to-head (M3.3):** `evaluate.py --vs-checkpoint <path>` plays the `--checkpoint` agent (seat p1) directly against a second PPO checkpoint (seat p2) through the dual-seat bridge. PPO-only; both checkpoints must share the obs mode (`--structured` applies to both); mutually exclusive with `--opponent`. Seat bias exists (~3pp) — run both orientations and combine for a fair comparison.

**MCTS (M4):** `evaluate.py --model mcts --checkpoint <ppo.pt>` wraps a structured PPO checkpoint in determinized UCT search at inference time (no training). Per decision it clones the live battle through the bridge's sim protocol (`sim_clone`/`sim_step`/`sim_fork` — `BattleSim` in `sim/tools/battle-sim.ts`), resamples the opponent's unrevealed Pokémon per determinization, and searches with the policy head as PUCT prior and the value head at leaves; the in-search opponent model samples the same policy on the opponent's reveal-tracked obs. Flags: `--sims` (default 100, split across `--determinizations`, default 1), `--c-puct` (0.5), `--no-determinize` (omniscient diagnostic), `--mcts-seat p1|p2` (with `--vs-checkpoint`, for seat-balanced h2h). Single-env by design (`--num-envs` ignored). ~85ms/decision at 100 sims on CPU. Defaults are the post-M4 knob-sweep operating point (one deep tree, low exploration: 81% vs Random / 67% vs DamageFirst at 500 battles, vs 66%/56% for the original sims=100/det=4/c_puct=1.5); sweep logs in `models/mcts/results/sweep/`.

## Quick-Start Training

```bash
# Build the simulator first (required once per pull)
./build

# Q-learning
python models/q_learning/train.py
python models/q_learning/train.py --episodes 50000

# DQN (--steps counts environment steps, not battles — ~50 steps/battle)
python models/dqn/train.py
python models/dqn/train.py --steps 500000 --checkpoint-every 50000

# PPO (--steps counts environment steps, not battles)
# --num-envs 8 parallel sims is the default (M3.1); --device overrides cpu/mps/cuda auto-detect
python models/ppo/train.py
python models/ppo/train.py --steps 500000 --rollout-steps 512 --checkpoint-every 50000

# PPO, M2 structured (12,65)->780 obs — checkpoints go to ppo/checkpoints/structured/
python models/ppo/train.py --structured --steps 2600000 --rollout-steps 512 --checkpoint-every 250000 --num-envs 8

# BC pretraining (M2.5) — download dataset once, then train
bash scripts/download_metamon.sh
python models/bc_pretrain.py --epochs 5 --format gen1ou --checkpoint_dir models/checkpoints

# M5.5: human-replay BC for the MLP — scrape/bootstrap logs, convert, train
# NOTE: --backfill is EXHAUSTED for both gen 1 formats as of 2026-07-31. The
# gen1ou archive was scanned to "history exhausted" (103,436 entries) and we hold
# ~all of the >=1300 tier. Only --top-up (new games, ~30-44/day) still adds data.
# Read docs/DATA-INVENTORY.md before proposing any data-acquisition work.
python scripts/scrape_replays.py                     # top-up gen1randombattle+gen1ou (rated >=1300)
python scripts/bootstrap_gen1ou_replays.py           # 98k historical gen1ou logs from HF (once)
node models/replay_adapter_cli.js --format gen1randombattle --shard-size 1000
node models/replay_adapter_cli.js --format gen1ou --shard-size 1000
python models/bc_pretrain_mlp.py --epochs 5          # -> models/checkpoints/bc_mlp_gen1.pt
python models/bc_pretrain_mlp.py --max-shards 2 --epochs 1   # smoke test

# M6: ladder bot — local-server smoke (two instances, bot vs bot)
node pokemon-showdown start --no-security --port 8355   # separate terminal
node tools/ladder-bot/ladder-bot.js --server ws://localhost:8355/showdown/websocket \
    --checkpoint models/ppo/checkpoints/opp/ppo_step_5000001_final.pt \
    --name lbotalpha --battles 2 --accept-from lbotbeta
node tools/ladder-bot/ladder-bot.js --server ws://localhost:8355/showdown/websocket \
    --checkpoint models/ppo/checkpoints/opp/ppo_step_5000001_final.pt \
    --name lbotbeta --battles 2 --challenge lbotalpha
# M6: official ladder (registered account; conservative pacing — one battle at a time).
# Credentials: --login-file (never on argv/env; file lives in gitignored config/).
node tools/ladder-bot/ladder-bot.js --login-file config/showdown_login.txt \
    --checkpoint models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt --battles 10
# M6 P2: add search (determinized UCT via BattleSim.fromTracked; tuned defaults
# sims=100/det=1/c_puct=0.5; force-switches/locked states fall back to raw policy)
node tools/ladder-bot/ladder-bot.js --login-file config/showdown_login.txt \
    --checkpoint models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt --battles 10 --mcts
# Game logs land in data/replays/self_ladder/ (+ ladder_results.csv) and feed
# straight back into models/replay_adapter_cli.js.

# Transformer PPO (M3) — always structured (12,65) obs, no --structured flag.
# Both from-scratch and warm-started runs are needed to compare against the
# M2 MLP-PPO baseline (51% win rate). Checkpoints go to checkpoints/scratch/
# or checkpoints/pretrained/ depending on --pretrain_checkpoint.
python models/transformer/train.py --steps 2600000 --rollout-steps 512 --checkpoint-every 250000 --num-envs 8
python models/transformer/train.py --steps 2600000 --rollout-steps 512 --checkpoint-every 250000 --num-envs 8 \
    --pretrain_checkpoint models/checkpoints/bc_pretrain_gen1ou.pt

# Transformer PPO with the M3.2 fixes (value warmup + BC anchor) — the M3.2 decision-run recipe
python models/transformer/train.py --steps 5000000 --checkpoint-every 250000 --num-envs 8 \
    --pretrain_checkpoint models/checkpoints/bc_pretrain_gen1ou.pt \
    --bc-anchor models/checkpoints/bc_pretrain_gen1ou.pt --value-warmup-steps 200000 \
    --checkpoint-dir models/transformer/checkpoints/m32

# Self-play training (M3.3) — opponent pool = the run's own checkpoints.
# MLP-PPO is the project architecture per the M3.2 decision (transformer retired).
python models/ppo/train.py --structured --steps 5000000 --checkpoint-every 250000 --num-envs 8 \
    --opponent selfplay --checkpoint-dir models/ppo/checkpoints/selfplay

# M3.4: schema-v2 obs (12,77)->924 + mixed opponents, pool seeded with M2 + M3.3-best
# (cp <ckpt> models/ppo/checkpoints/v2/ppo_step_0_<name>.pt before starting)
python models/ppo/train.py --obs-v2 --steps 5000000 --checkpoint-every 250000 --num-envs 8 \
    --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2" --checkpoint-dir models/ppo/checkpoints/v2

# Evaluate a checkpoint (--num-envs 8 parallel battles by default; --num-envs 1 = legacy serial)
# Add --opponent damagefirst to evaluate against the heuristic attacker instead of RandomPlayerAI
python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200
python models/evaluate.py --model ppo --structured --checkpoint models/ppo/checkpoints/structured/ppo_step_2600000_final.pt --battles 200
python models/evaluate.py --model transformer --checkpoint models/transformer/checkpoints/pretrained/transformer_step_2600000_final.pt --battles 200

# Head-to-head: checkpoint vs checkpoint (M3.3) — run both seat orders and combine
python models/evaluate.py --model ppo --structured \
    --checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt \
    --vs-checkpoint models/ppo/checkpoints/structured/ppo_step_2600000_final.pt --battles 500

# MCTS (M4): determinized UCT over a PPO checkpoint — vs an AI opponent, and
# seat-balanced head-to-head vs the same checkpoint without search
python models/evaluate.py --model mcts --checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt \
    --battles 500 --sims 100 --determinizations 4 --device cpu
python models/evaluate.py --model mcts --checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt \
    --vs-checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt --battles 500 --device cpu
python models/evaluate.py --model mcts --checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt \
    --vs-checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt --mcts-seat p2 --battles 500 --device cpu

# Evaluate an --obs-v2 checkpoint; --vs-checkpoint may be a v1 checkpoint (cross-schema h2h)
python models/evaluate.py --model ppo --obs-v2 --checkpoint models/ppo/checkpoints/v2/ppo_step_5000000_final.pt --battles 500
python models/evaluate.py --model ppo --obs-v2 --checkpoint models/ppo/checkpoints/v2/ppo_step_5000000_final.pt \
    --vs-checkpoint models/ppo/checkpoints/selfplay/ppo_step_4750059.pt --battles 500

# M8 Phase 2: AlphaZero-style value-head targeting on the M7 (v3) checkpoint.
# 1) collect MCTS self-play targets (~600 games/h/worker at sims=100; resumable)
python models/collect_value_data.py --obs-v3 \
    --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
    --games 2000 --workers 4 --out-dir data/value_targets/m8_v3
# 2) fine-tune the value head only (policy/opp head/trunk stay bit-identical)
python models/value_finetune.py --target outcome \
    --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
    --data-dir data/value_targets/m8_v3 \
    --out models/ppo/checkpoints/v3_valft/ppo_v3_valft_outcome.pt
# 3) Criterion C A/B: tuned MCTS vs DamageFirst, base vs fine-tuned, 200 battles each
python models/evaluate.py --model mcts --obs-v3 --opponent damagefirst --battles 200 \
    --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt --device cpu
python models/evaluate.py --model mcts --obs-v3 --opponent damagefirst --battles 200 \
    --checkpoint models/ppo/checkpoints/v3_valft/ppo_v3_valft_outcome.pt --device cpu
```

For the full message type reference and troubleshooting, see `docs/ML-TRAINING.md` -> **Python-Node Bridge Protocol**.
