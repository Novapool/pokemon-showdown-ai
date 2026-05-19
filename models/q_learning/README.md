# Model A: Tabular Q-Learning

## Overview

Tabular Q-learning is the simplest form of reinforcement learning: a lookup table maps (state, action) pairs to expected cumulative rewards, updated online via the TD(0) rule.

We include it as a **baseline** to confirm the intuition that Gen 1 Pokemon battles contain structure beyond what a table can efficiently represent. If the win rate plateaus well below the DQN/PPO targets, it validates the need for neural function approximation. If it surprisingly converges, it tells us the effective state space is small and a table suffices.

The model uses no neural network, no PyTorch, and only stdlib + numpy.

---

## Architecture

### State Discretization

The raw observation is a 100-float32 vector produced by `feature-extractor.ts`. To make it hashable (Q-table key), we project it down to a 5-element tuple:

| Tuple element   | Source feature(s)    | Description                                    |
|-----------------|----------------------|------------------------------------------------|
| `own_hp_bucket` | `obs[0] * 4`         | Own active HP in 4 coarse buckets (0–4)        |
| `type_index`    | `obs[4] * 20`        | Own active Pokémon type1 index (0–20)          |
| `switch_mask`   | `obs[55:60] > 0.5`   | Which bench slots are available (5-bool tuple) |
| `opp_hp_bucket` | `obs[60] * 4`        | Opponent active HP in 4 coarse buckets (0–4)   |
| `n_valid_moves` | `sum(obs[15:19]>0.5)`| Number of usable moves this turn (0–4)         |

This yields a manageable state space: `5 × 21 × 32 × 5 × 5 = 84,000` theoretical states.

### Q-Table Structure

```python
q_table: defaultdict(lambda: np.zeros(9))
```

Keys are discretized state tuples (above). Values are numpy arrays of length 9, one entry per action. Actions 0–3 are moves 1–4; actions 4–8 are switches to bench slots 2–6.

### Update Rule

Standard one-step Q-learning (TD(0)):

```
Q[s][a] += lr * (reward + gamma * max(Q[s']) * (1 - done) - Q[s][a])
```

---

## Training

```bash
# Default: 10,000 episodes
python models/q_learning/train.py

# Custom episode count
python models/q_learning/train.py --episodes 50000
```

The script logs a summary to stdout every 500 episodes:

```
Episode 500/10000 | Win rate: 0.52 | Epsilon: 0.87 | States: 1234
```

The trained Q-table is saved to `models/q_learning/qtable.pkl`.

---

## Hyperparameters

| Parameter       | Value  | Description                              |
|-----------------|--------|------------------------------------------|
| `lr`            | 0.1    | Q-update learning rate                   |
| `gamma`         | 0.95   | Discount factor                          |
| `epsilon`       | 1.0    | Initial exploration rate                 |
| `epsilon_min`   | 0.05   | Minimum exploration rate                 |
| `epsilon_decay` | 0.9995 | Multiplicative decay per episode         |
| `n_actions`     | 9      | 4 moves + 5 switch slots                 |

---

## Results

TBD — fill after training run.

---

## Analysis

TBD — expected limitation: the discretized state captures only coarse HP and type info, missing move PP, stat stages, and weather. This will likely cap win rate vs Random well below DQN/PPO performance.
