# Training Commands — Quick Reference

All commands are run from the **project root** (`pokemon-showdown/`).

---

## 1. Train the models

### Model A — Tabular Q-Learning (fast, ~5 min)
```bash
python models/q_learning/train.py
```
| Flag | Default | What it does |
|---|---|---|
| `--episodes N` | 10000 | How many battles to train on. More = better (up to a ceiling). |

**Example — quick test:**
```bash
python models/q_learning/train.py --episodes 1000
```

Saves Q-table to `models/q_learning/qtable.pkl` when done.

---

### Model B — DQN (slower, ~hours)
```bash
python models/dqn/train.py
```
| Flag | Default | What it does |
|---|---|---|
| `--steps N` | 100000 | Total environment steps to train for (not battles — roughly 50 steps per battle). |
| `--checkpoint-every N` | 25000 | Save the model weights every N steps so you can resume or evaluate mid-training. |

**Example — short run to verify it works:**
```bash
python models/dqn/train.py --steps 5000 --checkpoint-every 1000
```

Saves checkpoints to `models/dqn/checkpoints/dqn_step_XXXXX.pt`.

---

### Model C — PPO (slower, ~hours)
```bash
python models/ppo/train.py
```
| Flag | Default | What it does |
|---|---|---|
| `--steps N` | 100000 | Total environment steps to train for (same convention as DQN — not battles). |
| `--rollout-steps N` | 512 | How many steps to collect before doing a weight update. Larger = more stable but slower to update. |
| `--checkpoint-every N` | 25000 | Save weights every N steps. |
| `--structured` | off | Train on the M2 (12,65)->780 flattened structured observation instead of the legacy flat 100-dim vector. Checkpoints go to `checkpoints/structured/`. |
| `--num-envs N` | 8 | Parallel battle simulations (M3.1, ~5x throughput at 8). `1` = old serial behavior. |
| `--device D` | auto | Force `cpu`, `mps`, or `cuda` (default auto-detects cuda > mps > cpu). |
| `--opponent O` | random | `random`, `damagefirst` (max-base-power heuristic), or `selfplay` (frozen past checkpoints; see `--selfplay-pool`). |
| `--selfplay-pool DIR` | own checkpoint dir | Where self-play opponents are sampled from (50% newest / 50% uniform). |
| `--checkpoint-dir DIR` | per-mode default | Override checkpoint directory so a new run never overwrites an old run's files. |

**Example — short run:**
```bash
python models/ppo/train.py --steps 5000 --rollout-steps 256 --checkpoint-every 1000
```

Saves checkpoints to `models/ppo/checkpoints/ppo_step_XXXXX.pt`.

---

### Model D — Transformer PPO (M3)
```bash
python models/transformer/train.py --steps 2600000 --checkpoint-every 250000 --num-envs 8
```
Same flags as PPO (minus `--structured` — the transformer always consumes the raw `(12,65)` tokens), plus `--pretrain_checkpoint` (BC warm-start) and `--resume`. See `models/CLAUDE.md` for the full quick-start block. Note: M3 concluded with this architecture **losing** to the MLP-PPO baseline (46% vs 51%); retraining it is gated on the M3.2 fixes in `MILESTONES.md`.

---

## 2. Evaluate a trained model

```bash
python models/evaluate.py --model MODEL --checkpoint PATH
```

| Flag | Required? | What it does |
|---|---|---|
| `--model` | Yes | Which model type: `q_learning`, `dqn`, `ppo`, or `transformer` |
| `--checkpoint` | Yes | Path to the saved file (`.pkl` for Q-learning, `.pt` for the rest) |
| `--battles N` | No (default 200) | How many test battles to run. More = more accurate win rate. |
| `--structured` | No | Evaluate a PPO checkpoint trained with `train.py --structured` (M2 verification). |
| `--num-envs N` | No (default 8) | Parallel battle simulations. `1` = old serial behavior. |
| `--device D` | No (default auto) | Force `cpu`/`mps`/`cuda` for ppo/transformer inference. |
| `--opponent O` | No (default random) | `random` or `damagefirst` (max-base-power heuristic). |

**Examples:**
```bash
# Evaluate Q-Learning
python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl

# Evaluate DQN at a specific checkpoint
python models/evaluate.py --model dqn --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 500

# Evaluate PPO
python models/evaluate.py --model ppo --checkpoint models/ppo/checkpoints/ppo_step_100000.pt
```

Prints: win rate as a fraction (e.g. `0.73 (146/200)`). Opponent is always `RandomPlayerAI`.

---

## 3. Recommended workflow

1. **Quick smoke test** — make sure everything runs before committing to a long training run:
   ```bash
   python models/q_learning/train.py --episodes 200
   python models/dqn/train.py --steps 1000 --checkpoint-every 500
   ```

2. **Full training run** — use defaults (takes hours):
   ```bash
   python models/q_learning/train.py
   python models/dqn/train.py
   python models/ppo/train.py
   ```
   Run DQN and PPO in separate terminals — they're independent.

3. **Evaluate and compare:**
   ```bash
   python models/evaluate.py --model q_learning --checkpoint models/q_learning/qtable.pkl --battles 500
   python models/evaluate.py --model dqn       --checkpoint models/dqn/checkpoints/dqn_step_100000.pt --battles 500
   python models/evaluate.py --model ppo       --checkpoint models/ppo/checkpoints/ppo_step_100000.pt --battles 500
   ```

4. **Record results** in `docs/MODEL-COMPARISON.md` and pick a winner for M3.

---

## M8 Phase 2 — value-head targeting runbook

Three steps, all on the M7 checkpoint
(`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt`). Steps 1 and 3 are the
long ones; step 2 takes a couple of minutes.

**1. Collect MCTS self-play value targets** (~600 games/h per worker at
`--sims 100` on CPU, so 2000 games ≈ 1 h at `--workers 4`; pick `--workers`
≈ physical cores, each hosts its own Node bridge):

```bash
python models/collect_value_data.py --obs-v3 \
  --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
  --games 2000 --workers 4 --out-dir data/value_targets/m8_v3
```

Shards flush every 25 games; re-running the identical command resumes from
what's on disk (`--no-resume` to start over). Expect ~40 decisions/game, so
2000 games ≈ 80k training rows (~350 MB of `.npz`, gitignored).

**2. Fine-tune the value head** (trunk, policy head and opp head come out
bit-identical — the search prior is untouched, so the A/B isolates leaf
evaluation):

```bash
python models/value_finetune.py --target outcome \
  --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
  --data-dir data/value_targets/m8_v3 \
  --out models/ppo/checkpoints/v3_valft/ppo_v3_valft_outcome.pt
```

`--target outcome` is the pure AlphaZero target (final result at every state).
`mc` (discounted shaped return, `--gamma 0.99`) and `root` (the search's own
root Q) reuse the same dataset — cheap to try all three and A/B the winner.
The script prints val MSE and sign agreement before and after, which is the
first read on whether the old value head was miscalibrated at all.

**3. Criterion C A/B** — tuned MCTS vs DamageFirstAI, 200 battles each, base
vs fine-tuned. The gate is **≥ +3pp for the fine-tuned checkpoint**:

```bash
python models/evaluate.py --model mcts --obs-v3 --opponent damagefirst --battles 200 \
  --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt --device cpu
python models/evaluate.py --model mcts --obs-v3 --opponent damagefirst --battles 200 \
  --checkpoint models/ppo/checkpoints/v3_valft/ppo_v3_valft_outcome.pt --device cpu
```

Pass → Phase 3 (full MCTS value training run). Fail → Phase 3 is skipped and
the M7 checkpoint carries into the Phase 4 ladder run.

---

## Running the ladder bot (live official server)

`tools/ladder-bot/ladder-bot.js` connects to the real Showdown ladder
(`wss://sim3.psim.us/showdown/websocket`) and plays rated games:

```bash
node tools/ladder-bot/ladder-bot.js --login-file config/showdown_login.txt \
  --checkpoint <path/to/checkpoint.pt> --battles <N> --mcts
```

**Have the user run this one themselves, not Claude.** (It *does* have websocket
reconnect logic as of M8 Phase 0, 2026-07-22 — backoff, re-login, rejoin of
in-flight battles — so the original reason for this rule is gone, but the
behavioural pattern below is not.) In practice it
runs reliably end-to-end when launched from the user's own terminal but has
repeatedly dropped connection mid-run when launched from Claude's sandboxed
background-task execution. Root cause isn't confirmed, but the pattern is
consistent enough across sessions that Claude should hand the command to the
user rather than run it directly for any live-ladder (not local-sim) work.
Everything else in this doc — training, local eval vs `RandomPlayerAI`/
`DamageFirstAI`, bot batteries — is unaffected and fine to run normally.

---

## Target win rates (M2 success criteria)

| Model | vs RandomPlayerAI | vs DamageFirstAI |
|---|---|---|
| Q-Learning | ≥ 50% | < 70% (expected to hit ceiling) |
| DQN | ≥ 80% | ≥ 60% |
| PPO | ≥ 85% | ≥ 65% |

---

## What are these models?

**Q-Learning (Tabular)**
The simplest approach. It keeps a giant lookup table: "when the game looks like X, action Y scored this well in the past." It learns by playing battles and updating scores in the table. The problem is Pokemon has too many possible game states — the table can never cover them all, so this method hits a ceiling fast. We test it mainly to confirm it's not good enough, as a baseline.

**DQN — Deep Q-Network**
Same idea as Q-Learning but instead of a lookup table, it uses a small neural network to *predict* how good each action is. The network generalizes — it can handle game states it's never seen before by learning patterns. It also uses a "replay buffer" (a memory of past moves) to learn from old experience instead of forgetting it. This is the primary model we expect to work well.

**PPO — Proximal Policy Optimization**
A more modern approach. Instead of learning "how good is each action," it directly learns a *policy* — a strategy for which action to pick. It also learns a *value estimate* (how good is this situation overall) to guide training. PPO is generally more stable than DQN and less likely to suddenly forget what it learned. Slightly slower to set up but often converges to a better final result.
