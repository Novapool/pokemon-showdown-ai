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
| `--battles N` | No (default 200) | How many test battles to run. **The default is far too small to compare two checkpoints** — see the note below. |
| `--structured` | No | Evaluate a PPO checkpoint trained with `train.py --structured` (M2 verification). |
| `--num-envs N` | No (default 8) | Parallel battle simulations. `1` = old serial behavior. |
| `--device D` | No (default auto) | Force `cpu`/`mps`/`cuda` for ppo/transformer inference. |
| `--opponent O` | No (default random) | `random` or `damagefirst` (max-base-power heuristic). |

> **How many battles?** Raw-policy evals run at **~27 battles/s** (200 battles in
> 7.4 s, 8 envs), so an A/B has no excuse for a small n: **use `--battles 2000`
> per arm (~75 s)**, which resolves a +3pp difference vs Random. Tuned MCTS runs
> at ~4.5 s/battle — **1,000/arm (~75 min)** resolves ~+4pp. At the old default of
> 200, the 95% CI is ±3.6pp vs Random and ±5.1pp vs DamageFirst, so a +3pp gate
> at n=200 fails ~2/3 of the time even when the candidate is genuinely better.
> Full tables and derivations: `docs/EVALUATION-METHODOLOGY.md` Part 2.

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

## M9 Phase 2 — data/distribution runbook

> **Read this before copying the fine-tune command below.** The commands here
> are the *procedure*, not a recommendation. The randbats-only BC they build
> won 2a as a standalone policy and then **lost 2c by −8.3pp** once fine-tuned.
> **For a PPO warm-start or `--bc-anchor`, use the mixed `bc_mlp_gen1_v3.pt`.**
> See `MILESTONES.md` → M9 Phase 2.

**Train a format-aligned BC checkpoint** (2a). The only change from the M5.5/M7
BC recipe is `--formats`; everything else is left at its default on purpose so
the comparison has exactly one variable. ~3.5 min on the Mac:

```bash
python3 models/bc_pretrain_mlp.py --epochs 5 --obs-v3 \
  --formats gen1randombattle --out bc_mlp_gen1_v3_rb5.pt
```

**A/B two checkpoints on bot evals.** Always report the *difference* with a CI:

```bash
bash models/checkpoints/run_m9p2a_ab.sh          # 4 evals at n=5000, ~13 min
python3 scripts/bot_eval_ab.py \
  --arm bc-mixed=1697/5000 --arm bc-rb5=1977/5000 --gate 3
```

Pick n from the arms' actual win rate, not habit — `--power --baseline-p 0.42`
shows a mid-range agent needs ~4,300/arm to resolve +3pp, against 906 near
p=0.93. See `docs/EVALUATION-METHODOLOGY.md`.

**Fine-tune it** (2c, home box, **~70 min** on the RTX 3080 — the "~4 h"
estimate was from Mac-era throughput; measured rate is ~21k steps/min). Push
first, then preflight. The run is M7's recipe with only the warm-start swapped
— which is exactly what makes it a clean single-variable A/B, and exactly why
its **result was negative**: swap the two `bc_mlp_gen1_v3_rb5.pt` paths below
for `bc_mlp_gen1_v3.pt` to reproduce the *stronger* arm.

```bash
git push && ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && scripts/homebox-preflight.sh"'
rsync -a models/checkpoints/bc_mlp_gen1_v3_rb5.pt homebox:Projects/pokemon-showdown-ai/models/checkpoints/
rsync -a models/ppo/checkpoints/v3/ppo_step_0_seed_*.pt \
  homebox:Projects/pokemon-showdown-ai/models/ppo/checkpoints/m9p2c/

ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && tmux new -d -s m9p2c \
  \".venv/bin/python models/ppo/train.py --obs-v3 --steps 5000000 \
     --rollout-steps 512 --num-envs 8 --checkpoint-every 250000 \
     --opponent-mix \\\"selfplay=0.5,damagefirst=0.3,random=0.2\\\" \
     --checkpoint-dir models/ppo/checkpoints/m9p2c \
     --pretrain-checkpoint models/checkpoints/bc_mlp_gen1_v3_rb5.pt \
     --bc-anchor models/checkpoints/bc_mlp_gen1_v3_rb5.pt --bc-anchor-coef 0.05 \
     --value-warmup-steps 200000 --opp-coef 0.1 2>&1 \
   | tee models/ppo/checkpoints/m9p2c/train.log\""'
```

The BC checkpoint and the pool seeds are **tier 2** (not `*_final.pt`), so they
do not travel with `git push` — rsync them or the run silently trains from
scratch with an empty pool. Then sweep and confirm on the home box:

```bash
bash models/ppo/checkpoints/m9p2c/run_sweep.sh
bash models/ppo/checkpoints/m9p2c/run_confirm.sh ppo_step_5000002_final.pt
```

Both runners take `DIR=` to point at another run, and both exclude pool seeds by
the `ppo_step_0_<name>.pt` filename convention — **not** by matching "seed" in
the path, which silently emptied the sweep for a directory named `m9seed`.

### Seed replication — how to tell a real effect from a lucky run

`train.py` does **no seeding** (no `manual_seed`, no `np.random.seed`, anywhere
in the trainer, agent or gym clients), so re-running an identical command is an
independent draw. That makes a replication a one-line job — same command, new
`--checkpoint-dir`:

```bash
DIR=models/ppo/checkpoints/m9seed bash models/ppo/checkpoints/m9p2c/run_confirm.sh \
  ppo_step_5000004_final.pt
python3 scripts/bot_eval_ab.py --arm m7=1433/2000 --arm m9seed=1421/2000
```

**Measured 2026-08-01: run-to-run spread is under 1pp** (M7 reproduced to
−0.6pp vs Random, CI including 0, across an `mps`→`cuda` change). So single-run
A/Bs here are trustworthy for effects of ~3pp and up, and a ~70-minute
replication is cheap insurance before believing a *surprising* result. Run both
arms of an A/B on the same machine.

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

**1b. Collect against the bot the A/B is judged on** (the 2026-07-29 retry).
The original dataset was collected against a *frozen policy checkpoint* while
Criterion C is scored against *DamageFirst* — one of the two standing
explanations for why the fine-tune fixed calibration without moving strength.
`--opponent damagefirst` uses the single-seat bridge, so the searcher is always
p1 and rewards need no sign flip:

```bash
python models/collect_value_data.py --opponent damagefirst --obs-v3 \
  --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
  --games 2000 --workers 10 --out-dir data/value_targets/m8_v3_df
```

Much faster than self-play collection — ~550 games/h per worker (games vs
DamageFirst are shorter and there's only one net to run), so 2000 games is
~20 min at `--workers 10`. **Known cost:** the searcher wins ~84% against
DamageFirst vs 75.5% in self-play, so the outcome labels are *more* skewed. That
cuts against the other explanation (that the MSE gain was mostly learning the
base rate), so read a win here carefully rather than as a clean fix.

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

> ⚠️ **Follow `docs/EVALUATION-METHODOLOGY.md`.** GXE from a shared account is
> not a per-run measurement, n=100 gives a ±8.8pp CI, and resolving a +10pp
> effect needs ~350 games per arm. Three milestones drew conclusions their
> numbers could not support (`docs/LADDER-MEASUREMENT.md` for why).

`tools/ladder-bot/ladder-bot.js` connects to the real Showdown ladder
(`wss://sim3.psim.us/showdown/websocket`) and plays rated games:

```bash
node tools/ladder-bot/ladder-bot.js \
  --login-file config/showdown_login_<arm>.txt \
  --checkpoint <path/to/checkpoint.pt> \
  --run-id <arm> --battles <N> --mcts
```

**`--run-id` is mandatory** — it labels the arm in
`data/replays/self_ladder/ladder_results.csv` and also makes the run resumable
(`--battles` is the absolute target; re-run the identical command to continue).
One fresh account per arm; never reuse `novapool`.

Analyse with the script, never the account JSON:

```bash
python3 scripts/ladder_analysis.py --run <arm>              # one arm
python3 scripts/ladder_analysis.py --arm <ctl> --arm <cand> # paired difference
python3 scripts/ladder_analysis.py --power                  # sample sizes
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
