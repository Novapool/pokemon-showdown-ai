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

## Bridge Architecture

Python training code cannot call Node.js APIs directly, so `gym_bridge.js` runs as a child process spawned by `gym_client.py` via `subprocess.Popen`. All communication between Python and the bridge is line-delimited JSON over stdin/stdout — one command object per line in, one response object per line out. The bridge wraps `PokemonGymEnv` from `dist/sim/tools/pokemon-gym.js`, so the TypeScript simulator must be compiled before the bridge can run. Build it once with `./build` (or `npm run build`) at the repo root before starting any training script.

## Quick-Start Training

```bash
# Build the simulator first (required once per pull)
./build

# Q-learning
python models/q_learning/train.py
python models/q_learning/train.py --episodes 50000

# DQN
python models/dqn/train.py
python models/dqn/train.py --battles 500000 --checkpoint-every 50000

# PPO
python models/ppo/train.py
python models/ppo/train.py --battles 500000 --rollout-steps 512 --checkpoint-every 50000

# Evaluate a checkpoint
python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200
```

For the full message type reference and troubleshooting, see `docs/ML-TRAINING.md` -> **Python-Node Bridge Protocol**.
