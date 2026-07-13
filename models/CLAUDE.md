Python ML training code and models. Wraps the Node.js Pokemon Showdown gym via a subprocess bridge.

## Directory Overview

| Path | Contents |
|---|---|
| `gym_bridge.js` | Node.js stdio server wrapping `PokemonGymEnv`; spawned as subprocess by `gym_client.py` |
| `gym_client.py` | Python `GymClient` class; spawns `gym_bridge.js` and exposes `reset()`, `step()`, `valid_actions()`, `close()` |
| `evaluate.py` | CLI script to evaluate a trained checkpoint against `RandomPlayerAI` |
| `q_learning/` | Tabular Q-learning agent (`q_agent.py`, `train.py`) |
| `dqn/` | Deep Q-Network agent (`dqn_agent.py`, `replay_buffer.py`, `train.py`, `checkpoints/`) |
| `ppo/` | PPO agent (`ppo_agent.py`, `trajectory_buffer.py`, `train.py`, `checkpoints/`) |
| `metamon_adapter.py` | Streams Metamon human-replay trajectories as `(obs (12,65), action, done)`; M2.5 |
| `bc_pretrain.py` | Behavior-cloning pretraining on Metamon replays; saves to `checkpoints/bc_pretrain_gen1ou.pt` |
| `transformer/` | `transformer_policy.py` — shared `TransformerPolicy` net + `load_pretrain_checkpoint()`; `transformer_agent.py` — PPO wrapper (M3); `train.py` — PPO training loop, `checkpoints/{scratch,pretrained}/` |
| `checkpoints/` | BC pretraining checkpoints |

## Bridge Architecture

Python training code cannot call Node.js APIs directly, so `gym_bridge.js` runs as a child process spawned by `gym_client.py` via `subprocess.Popen`. All communication between Python and the bridge is line-delimited JSON over stdin/stdout — one command object per line in, one response object per line out. The bridge wraps `PokemonGymEnv` from `dist/sim/tools/pokemon-gym.js`, so the TypeScript simulator must be compiled before the bridge can run. Build it once with `./build` (or `npm run build`) at the repo root before starting any training script.

**Observation modes (M2):** `GymClient()` defaults to the structured `(12, 65)` per-Pokémon token observation. Pass `GymClient(structured=False)` (bridge: `--flat`) for the legacy 100-dim flat vector — required for `q_learning/`, `dqn/`, and `ppo/`, whose networks are hardcoded to that shape. `evaluate.py` also uses `structured=False` for the same reason.

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
python models/ppo/train.py
python models/ppo/train.py --steps 500000 --rollout-steps 512 --checkpoint-every 50000

# PPO, M2 structured (12,65)->780 obs — checkpoints go to ppo/checkpoints/structured/
python models/ppo/train.py --structured --steps 2600000 --rollout-steps 512 --checkpoint-every 250000

# BC pretraining (M2.5) — download dataset once, then train
bash scripts/download_metamon.sh
python models/bc_pretrain.py --epochs 5 --format gen1ou --checkpoint_dir models/checkpoints

# Transformer PPO (M3) — always structured (12,65) obs, no --structured flag.
# Both from-scratch and warm-started runs are needed to compare against the
# M2 MLP-PPO baseline (51% win rate). Checkpoints go to checkpoints/scratch/
# or checkpoints/pretrained/ depending on --pretrain_checkpoint.
python models/transformer/train.py --steps 2600000 --rollout-steps 512 --checkpoint-every 250000
python models/transformer/train.py --steps 2600000 --rollout-steps 512 --checkpoint-every 250000 \
    --pretrain_checkpoint models/checkpoints/bc_pretrain_gen1ou.pt

# Evaluate a checkpoint
python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200
python models/evaluate.py --model ppo --structured --checkpoint models/ppo/checkpoints/structured/ppo_step_2600000_final.pt --battles 200
python models/evaluate.py --model transformer --checkpoint models/transformer/checkpoints/pretrained/transformer_step_2600000_final.pt --battles 200
```

For the full message type reference and troubleshooting, see `docs/ML-TRAINING.md` -> **Python-Node Bridge Protocol**.
