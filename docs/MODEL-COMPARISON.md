# M2 Model Comparison

## Overview

Milestone 2 (M2) explores three reinforcement learning architectures for learning to play Pokemon Showdown Gen 1 battles. Each model is trained against `RandomPlayerAI` (the default gym opponent) and evaluated on win rate, training time, and qualitative behavior. The goal is to identify one winning architecture to carry forward into M3 large-scale training.

---

## Models Evaluated

### Q-Learning (Tabular)

A classic tabular Q-learning agent with epsilon-greedy exploration. The 828-feature observation vector (flattened 12×69 token array) is discretized into a compact 5-element state tuple (own HP bucket, active type, switch mask, opponent HP bucket, number of valid moves), enabling a lookup-table Q-function. Simple to implement and interpret, but state coverage is inherently limited by the coarse discretization — expected to plateau below DQN and PPO.

### DQN (Deep Q-Network)

A two-hidden-layer MLP (828 → 128 → 128 → 9) approximating the Q-function directly from the flattened 828-feature observation vector. Trained with experience replay (buffer size 10k) and a periodically-synced target network (sync every 1000 steps). Handles the full continuous state space and is expected to generalize significantly better than the tabular approach.

### PPO (Proximal Policy Optimization)

An Actor-Critic agent with a shared trunk (828 → 128 → 128) feeding into separate policy (128 → 9) and value (128 → 1) heads. Trained with rollout-based advantage estimation (GAE), clipped surrogate objective, and entropy regularization. On-policy learning is more sample-inefficient than DQN but often more stable; PPO is the primary candidate for M3.

---

## Results

| Model | Win Rate vs RandomPlayerAI | Training Time | Notes |
|---|---|---|---|
| Q-Learning | TBD — fill after training run | TBD | TBD |
| DQN | TBD — fill after training run | TBD | TBD |
| PPO | TBD — fill after training run | TBD | TBD |

> All result cells are placeholders. Run each model's `train.py`, then evaluate with `models/evaluate.py --battles 200` and record results here.

---

## Analysis

TBD — fill after all training runs complete.

Expected hypothesis: DQN and PPO both significantly outperform Q-Learning due to continuous state representation. PPO may show more stable convergence curves while DQN may reach competitive performance faster given sample-efficient off-policy learning.

---

## Winner Selection

TBD — will be updated once results are in. Winner advances to M3 scale training.

The winning model will be selected based on win rate vs `RandomPlayerAI`, stability of the learning curve, and wall-clock training time. The checkpoint from the best run will serve as the starting point for M3.

---

## Next Steps

Once a winner is selected, proceed to **M3: Scale Training**. M3 trains the winning model for 1M+ steps with additional evaluation against stronger baselines. See the M3 section of [MILESTONES.md](../MILESTONES.md) for the full scope.

To run evaluation for a trained model:

```bash
python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 200
python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 200
python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 200
```
