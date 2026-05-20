# Pokemon Showdown AI Training — Milestones

Goal: Build a genuinely intelligent Pokemon trainer AI. Target architecture:
**Transformer-based policy + value network trained with PPO, eventually augmented
with MCTS for lookahead.**

Strategy: Structured state → transformer encoder → MCTS on top of learned value
function → opponent modeling. Each milestone unblocks the next.

---

## M0: Foundation ✅ COMPLETE

**Status:** ✅ Complete
**Artifacts:** docs suite, build system, baseline AI reference

### Deliverables
- Build system functional (`./build` → `dist/`)
- Documentation suite: 8 core reference files in `docs/`
- Verified: `RandomPlayerAI` base class at `sim/tools/random-player-ai.ts`
- Verified: `BattleStream` API works for programmatic parallel battles

### Key Files
- `sim/tools/random-player-ai.ts` — base AI class (extend for custom policies)
- `docs/SETUP.md`, `docs/SIMULATOR-API.md`, `docs/PARALLEL-SIMULATION.md`, `docs/AI-PLAYERS.md`
- `simulate.js` — **reference only** (concurrency patterns, `DamageFirstAI` example); not an ML project artifact

---

## M1: Environment & Baseline Agents ✅ COMPLETE

**Status:** ✅ Complete
**Goals:** Gym wrapper, flat-vector feature extractor, evaluation protocol, and
baseline agent implementations (tabular Q, DQN, PPO)

### Deliverables
1. `sim/tools/pokemon-gym.ts` — `PokemonGymEnv`: `reset()`, `step(action)`, `validActions()`, `destroy()`
2. `sim/tools/feature-extractor.ts` — `extractFeatures()`, `OBS_SIZE = 100` flat Float32Array
3. `sim/tools/evaluator.ts` — parallel battle evaluator, `evaluateVsRandom()`
4. `models/gym_bridge.js` + `models/gym_client.py` — Python-Node stdio bridge
5. `models/q_learning/` — tabular Q-learning (confirms tabular limitation; archived)
6. `models/dqn/` — DQN with experience replay (regression baseline)
7. `models/ppo/` — PPO actor-critic with GAE (training algorithm going forward)
8. `models/evaluate.py` — checkpoint evaluation CLI

### Observation Space (current, to be replaced in M2)
- Flat 100-dim Float32Array: own active [0–14], moves [15–54], switch mask [55–59], opponent active [60–74], padding [75–99]
- Known limitations: no bench Pokémon info, opponent bench absent, stat boosts hardcoded at 0.5, 25 bytes of padding

### Action Space
- 9 discrete actions: move 1–4 (indices 0–3), switch 1–5 (indices 4–8)
- Validity masking enforced at every step

### Reward Function
- `+0.01` per opponent KO, `-0.01` per own KO
- `+0.0001` per status inflicted on opponent
- `+1.0` win, `-1.0` loss
- `-0.001 × turns` stalling penalty
- Clipped to `[-1, 1]`

### Success Criteria ✅
- Gym loads, resets, steps without errors
- Observation shape consistent (100 features every turn)
- Reward bounds correct (never NaN, always in `[-1, 1]`)
- Battles terminate correctly via omniscient stream win detection
- All gym unit tests pass (`test/tools/gym.test.js`)

---

## M2: Structured State Representation 🔄 IN PROGRESS

**Status:** 🔄 In Progress (implementation complete — verification run pending)
**Goals:** Replace the flat 100-dim vector with a per-Pokémon tokenized
representation. The gym wrapper is unchanged; only the feature extractor changes.
Verify that PPO (with the trunk flattening the tokens) learns comparably to the
old flat-vector baseline.

### Why This Matters
The flat vector loses all relational structure between Pokémon. A transformer
(M3) cannot reason about "my Slowbro vs their Gengar" if both are smeared into
one undifferentiated vector. Structured tokens are the prerequisite for attention
to be meaningful.

### What to Build

**1. New feature extractor — `extractFeaturesStructured()`**

Returns a `(12, token_dim)` shaped array instead of a flat vector.

Token sequence (12 total for gen1 6v6):
```
[0]     own active Pokémon
[1–5]   own bench Pokémon (slots 1–5)
[6]     opponent active Pokémon
[7–11]  opponent bench Pokémon (slots 1–5; mostly unknown)
```

Per-Pokémon token features:
| Feature | Dim | Notes |
|---------|-----|-------|
| HP ratio | 1 | 0–1; 0 if fainted |
| Level / 100 | 1 | |
| Type 1 one-hot | 15 | gen1 has 15 types |
| Type 2 one-hot | 15 | same as type 1 if single-type |
| Status one-hot | 6 | brn, frz, par, psn, slp, tox |
| Active flag | 1 | 1 if currently on field |
| Unknown flag | 1 | 1 if this token is unrevealed opponent Pokémon |
| Fainted flag | 1 | 1 if fainted |
| Move 1–4 (each) | 7 | base_power/250, accuracy/100, PP_ratio, type_idx/15, category_idx/2, disabled, type_effectiveness/4 |

Total token_dim = 1+1+15+15+6+1+1+1 + 4×7 = **69**. Full tensor: **(12, 69)**.

**2. Unknown token handling**

Opponent bench slots that haven't been revealed are filled with `unknown_flag=1`
and all other features set to 0 (with the exception of `HP ratio = 1.0` to
represent "assumed full HP"). This distinguishes "not yet revealed" from
"fainted" (which has `HP ratio = 0, fainted_flag = 1`).

Do NOT use a zero vector for unknown — it looks identical to a fainted Pokémon.

**3. Stat boost tracking**

The gym's omniscient reader already sees all `|-boost|` and `|-unboost|` lines.
Add a boost accumulator to `PokemonGymEnv` that tracks boosts per slot, then
pass them into `extractFeaturesStructured()`. Boosts for the active Pokémon
replace the hardcoded 0.5 placeholder.

**4. Bridge protocol update**

`gym_bridge.js` currently serializes the obs as a flat `Array` of 100 floats.
Change the serialization to pass a `(12 × 69 = 828)`-element flat array.
`gym_client.py` reshapes it back to `(12, 69)` as a numpy array.

Backward-compat: add a `--flat` flag to `gym_bridge.js` for running the old
MLP PPO baseline against random as a regression check.

**5. Existing helpers to keep**

`parseHpRatio()`, `parseStatus()`, `fillStatusBitmask()`, `parseLevelFromDetails()`,
`typeToIndex()`, `categoryToIndex()` — all reusable as-is from `feature-extractor.ts`.

### Files to Create / Modify
| File | Action |
|------|--------|
| `sim/tools/feature-extractor.ts` | Add `extractFeaturesStructured()`, export `TOKEN_DIM=69`, `N_TOKENS=12` |
| `sim/tools/pokemon-gym.ts` | Add boost tracker; thread boosts into `extractFeaturesStructured()` |
| `models/gym_bridge.js` | Change obs serialization to 828-element flat array; add `--flat` flag |
| `models/gym_client.py` | Reshape `(828,)` → `(12, 69)` numpy array |

### Success Criteria
- `extractFeaturesStructured()` returns `Float32Array` of length `12 × 69 = 828`, shape-stable across move requests, switch requests, and end-of-episode
- Unknown opponent bench tokens have `unknown_flag=1` and `HP_ratio=1.0`
- Fainted tokens have `fainted_flag=1` and `HP_ratio=0`
- PPO with MLP trunk on flattened structured obs achieves ≥ 50% win rate vs RandomPlayerAI at 50k battles (parity with old flat-vector baseline confirms representation isn't broken)
- Stat boosts for active Pokémon are non-constant (tracked from battle log)

### Unblocks
M3 (transformer encoder needs per-Pokémon tokens as input)

---

## M3: Transformer Encoder + PPO Baseline 🔄 IN PROGRESS

**Status:** 🔄 In Progress (transformer_agent.py and train.py created, smoke tests pass — training not yet run)
**Goals:** Replace the MLP trunk with a small transformer encoder. Train with PPO
and establish a transformer win-rate baseline to beat in subsequent milestones.

### Architecture

```
Input:  (batch, 12, 69)  ← 12 Pokémon tokens, each 69-dim

Linear(69 → d_model=128)   ← project tokens into model dimension

TransformerEncoder(
  layers=2, nhead=4, d_model=128, d_ff=256, dropout=0.1
)                          ← attention over Pokémon relationships

Mean-pool over 12 tokens   ← (batch, 128) context vector

Policy head: Linear(128, 9)  → action logits
Value head:  Linear(128, 1)  → state value scalar
```

No positional encoding (Pokémon are unordered; add only if ablation shows benefit).

Unknown opponent tokens are attention-masked (key_padding_mask) so they don't
pollute the context with learned noise before the opponent's team is revealed.

Total parameters: ~500k–1M. Training target: 200k–500k battles.

### Files to Create
| File | Contents |
|------|----------|
| `models/transformer/transformer_agent.py` | `TransformerAgent(nn.Module)` with `act()`, `evaluate_actions()`, `update()`, `save()`, `load()` |
| `models/transformer/train.py` | PPO loop — reuse `TrajectoryBuffer` from `models/ppo/trajectory_buffer.py`; swap agent class only |

### Files to Modify
| File | Change |
|------|--------|
| `models/evaluate.py` | Add `--model transformer` support |

### Hyperparameters (starting point)
```
d_model=128, d_ff=256, nhead=4, n_layers=2, dropout=0.1
lr=3e-4, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01
rollout_steps=512, ppo_epochs=4, batch_size=64
```

### Training Protocol
1. Train transformer PPO for 200k battles
2. At same battle count, run MLP PPO baseline for comparison
3. Checkpoint every 25k battles
4. Log: win rate vs RandomPlayerAI, attention weight entropy (check non-uniform)

### Success Criteria
- Transformer PPO ≥ 80% win rate vs RandomPlayerAI at 200k battles
- Transformer PPO beats MLP PPO at the same battle count (demonstrates attention helps)
- Attention weight entropy > 0.5 nats (model uses attention, not degenerate uniform)
- Loss curves stable: no divergence, value loss decreasing monotonically in first 50k battles
- ≥ 65% win rate vs DamageFirstAI at 200k battles

### Unblocks
M4 (needs trained value function for MCTS leaf evaluation), M5 (opponent modeling head)

---

## M4: MCTS Integration ⬜ AFTER M3

**Status:** ⬜ Not Started
**Goals:** Layer UCT-based Monte Carlo Tree Search on top of the trained PPO
transformer. Use the learned value network for leaf evaluation instead of random
rollouts. Start with inference-time MCTS; add AlphaZero-style fine-tuning later.

### Design

**Tree nodes:** `(state_obs: np.ndarray, valid_mask: bool[9])`

**UCT selection:**
```
score(s, a) = Q(s, a) + c_puct × P(a|s) × sqrt(N(s)) / (1 + N(s, a))
```
where `P(a|s)` is the transformer policy prior and `V(s)` is the transformer
value estimate at leaf nodes.

**Simulation budget:** N=100 simulations per move decision (tune: 50–200).

### Imperfect Information

Gen1 hides the opponent's team composition and moves until revealed. Approach:

**Determinization (start here):** At each MCTS rollout, sample a plausible
opponent team from the prior distribution over gen1 Pokémon (weighted by tier
usage), conditioned on what has already been revealed. Run MCTS on the fully
observed determinized game. Simple and effective for shallow lookahead.

Belief-state MCTS (Information Set MCTS) is more principled but significantly
more complex — defer unless determinization plateaus.

### Files to Create
| File | Contents |
|------|----------|
| `models/mcts/mcts_agent.py` | `MCTSNode`, `MCTSAgent` with `search()`, `select()`, `expand()`, `backup()` |
| `models/mcts/determinizer.py` | Samples plausible opponent teams given revealed Pokémon |

### Success Criteria
- MCTS(N=100) + transformer value beats PPO-only transformer by ≥ 5% win rate vs DamageFirstAI
- Decision latency < 500ms per move (100 sims, gen1 is fast)
- Win rate vs RandomPlayerAI ≥ 90%
- Determinizer generates valid gen1randombattle-legal teams

### Unblocks
Self-play with MCTS policy (generates higher-quality training data for future fine-tuning)

---

## M5: Opponent Modeling Head ⬜ AFTER M3

**Status:** ⬜ Not Started
**Goals:** Add an auxiliary head to the transformer that predicts the opponent's
next action. Train with a multi-task loss (PPO + opponent prediction). Use to
improve the policy's anticipation of opposing moves.

### Design

The omniscient stream sees both players' choices. After each turn resolves, the
opponent's choice is observable. This provides free supervised signal.

```
Transformer context vector (128-dim)
    └─ policy head:   Linear(128, 9)   — own action logits  [PPO loss]
    └─ value head:    Linear(128, 1)   — state value        [PPO value loss]
    └─ opp_pred head: Linear(128, 9)  — opponent action     [cross-entropy]

Total loss = PPO_loss + λ × CE(opp_pred, actual_opp_action)
λ = 0.1 (tune: 0.05–0.3)
```

The opponent action label is gathered from the omniscient stream in `pokemon-gym.ts`
after each turn and passed back via the `info` dict.

### Files to Modify
| File | Change |
|------|--------|
| `sim/tools/pokemon-gym.ts` | Parse opponent's choice from omniscient stream; add `opp_action` to `info` |
| `models/transformer/transformer_agent.py` | Add `opp_action_head`, expose `opp_pred` in forward pass |
| `models/transformer/train.py` | Collect `opp_action` from `info`; add auxiliary loss term |

### Success Criteria
- Opponent action prediction accuracy > 40% (random baseline ≈ 25% for 4-move space)
- Policy trained with opponent head beats policy without it by ≥ 3% vs DamageFirstAI
- Auxiliary loss weight tuned: policy performance does not regress vs PPO-only baseline

---

## M6: Server Integration & Ladder ⬜ AFTER M4+M5

**Status:** ⬜ Not Started
**Goals:** Connect trained model to live Pokemon Showdown server, accept challenges,
track Elo rating.

### What to Build
- `server/bot-client.ts` — WebSocket connection to `sim.smogon.com` or local server
- Battle state mapper: Showdown protocol lines → structured `(12, 69)` token obs
- Inference service: load transformer checkpoint, respond within 2s latency budget
- `server/elo-ladder.ts` — per-match rating updates, CSV history

### Success Criteria
- Bot connects and accepts challenges without crashing
- Decision latency < 2s per move
- ≥ 60% win rate vs random human/bot opponents on ladder
- Runs ≥ 100 consecutive battles without crashing

---

## Architecture Reference

```
Battle state
    │
    ▼
extractFeaturesStructured()      [M2]
    │  (12 Pokémon tokens × 69 features each)
    ▼
TransformerEncoder (2L, 4H, d=128)  [M3]
    │  mean-pool → 128-dim context
    ├──▶ Policy head (128 → 9 logits)
    ├──▶ Value head  (128 → 1 scalar)
    └──▶ Opp pred   (128 → 9 logits)   [M5]
    │
    ▼  [M4]
MCTS (UCT, N=100, determinized)
    │  uses Policy as prior, Value at leaves
    ▼
Best action
```

---

## Component Reuse Guide

| Component | Status | Reuse in |
|-----------|--------|----------|
| `sim/tools/pokemon-gym.ts` | ✅ Keep as-is | All milestones |
| `sim/tools/random-player-ai.ts` | ✅ Keep as-is | Opponent in all training |
| `models/ppo/trajectory_buffer.py` | ✅ Keep as-is | M2 verification, M3 training |
| `models/ppo/ppo_agent.py` PPO update logic | ✅ Reuse | M2, M3 (new trunk only) |
| `models/evaluate.py` | ✅ Extend | M2, M3, M4 (add model types) |
| `models/gym_client.py` + `gym_bridge.js` | ✅ Update serialization | M2 |
| `feature-extractor.ts` helpers | ✅ Reuse | M2 (build on top of) |
| `models/dqn/` | ⚠️ Regression baseline | Comparison only |
| `models/q_learning/` | ❌ Archived | Tabular limitation confirmed |

---

## Summary Timeline

| Milestone | Status | Core Deliverable | Unlocks |
|-----------|--------|-----------------|---------|
| M0: Foundation | ✅ | Build system, docs | — |
| M1: Env + Baselines | ✅ | Gym, PPO, DQN, Q-learning | — |
| M2: Structured State | 🔄 | Per-Pokémon token obs (12×69) | M3 |
| M3: Transformer + PPO | 🔄 | Transformer encoder baseline | M4, M5 |
| M4: MCTS | ⬜ | UCT search over value network | M6 |
| M5: Opponent Modeling | ⬜ | Opp-prediction auxiliary head | M6 |
| M6: Server Integration | ⬜ | Live ladder bot | — |
