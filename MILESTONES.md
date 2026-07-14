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

## M2: Structured State Representation ✅ COMPLETE

**Status:** ✅ Complete and verified (2026-07-09) — 51% win rate vs RandomPlayerAI (500 battles), parity with the M1 flat baseline confirmed
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
| Move 1–4 (each) | 6 | base_power/250, accuracy/100, PP_ratio, type_idx/15, category_idx/2, disabled |

Total token_dim = 1+1+15+15+6+1+1+1 + 4×6 = **65**. Full tensor: **(12, 65)**.

**2. Unknown token handling**

Opponent bench slots that haven't been revealed are filled with `unknown_flag=1`
and all other features set to 0 (with the exception of `HP ratio = 1.0` to
represent "assumed full HP"). This distinguishes "not yet revealed" from
"fainted" (which has `HP ratio = 0, fainted_flag = 1`).

Do NOT use a zero vector for unknown — it looks identical to a fainted Pokémon.

**3. Stat boost tracking — DROPPED (2026-07-09)**

~~The gym's omniscient reader already sees all `|-boost|` and `|-unboost|`
lines. Add a boost accumulator to `PokemonGymEnv`...~~ This was written before
M2.5 locked the 65-dim token schema to match `models/metamon_adapter.py`
exactly (so the BC-pretrained checkpoint's input projection stays valid).
That schema has no boost slots, and Metamon's replay dataset doesn't encode
boosts either — there's no way to add a boost feature now without breaking
BC-checkpoint compatibility or retraining it from scratch. Boosts are out of
scope for M2; revisit only as a deliberate schema-version bump.

**4. Bridge protocol update — DONE**

`gym_bridge.js` serializes obs as a `(12 × 65 = 780)`-element flat array by
default. `gym_client.py` reshapes it back to `(12, 65)` as a numpy array.
`--flat` (bridge) / `structured=False` (`GymClient`) restore the legacy
100-float path — wired into `models/{q_learning,dqn,ppo}/train.py` and
`evaluate.py` since their networks are hardcoded to that shape.

**5. Existing helpers — reused and exported**

`parseHpRatio()`, `parseStatus()`, `fillStatusBitmask()`, `parseLevelFromDetails()`,
`typeToIndex()`, `categoryToIndex()` — now `export`ed from `feature-extractor.ts`
so `pokemon-gym.ts`'s opponent-reveal tracker can reuse them directly.
`parseLevelFromDetails()`'s no-match default was also fixed from 50 to 100
(Showdown omits the level tag entirely at level 100 — the gen1ou/gen1randombattle
norm — so the old default silently mis-encoded every level-100 Pokémon).

**6. Opponent-reveal tracker (added, not in the original plan)**

A real player's request never contains the opponent's team. `PokemonGymEnv`
now reconstructs opponent state purely from `p2`-side battle-log lines
(`|switch|`, `|drag|`, `|-damage|`, `|-heal|`, `|-status|`, `|-curestatus|`,
`|faint|`, `|move|`) — species, HP, status, fainted, and revealed moves only.
Unrevealed opponent Pokémon get the unknown-token treatment; this is what
actually populates the opponent tokens described above.

### Files Created / Modified
| File | Action |
|------|--------|
| `sim/tools/feature-extractor.ts` | Added `extractFeaturesStructured()`, `TOKEN_DIM=65`, `N_TOKENS=12`; exported existing helpers |
| `sim/tools/pokemon-gym.ts` | Added opponent-reveal tracker; added `obsMode` option (default `'structured'`) |
| `models/gym_bridge.js` | 780-element flat array by default; `--flat` flag |
| `models/gym_client.py` | Reshapes to `(12, 65)`; `structured=False` flag |
| `models/{q_learning,dqn,ppo}/train.py`, `models/evaluate.py` | Pass `structured=False` (flat-vector networks) |
| `tools/build-utils.js` | Fixed `copyOverDataJSON` to create destination dirs before copying (unblocked `./build` once `data/metamon_cache/` existed) |
| `test/tools/gym.test.js` | Structured-obs shape/unknown/fainted/bench-order tests; full-battle shape-stability smoke test |
| `models/dqn/train.py`, `models/ppo/train.py` | Renamed `--battles` → `--steps` (flag counted steps, not battles — collided in meaning with `evaluate.py`'s correct battle-counting `--battles`) |
| `models/ppo/train.py`, `models/evaluate.py` | Added `--structured` flag for training/evaluating on the M2 obs; checkpoints isolated to `checkpoints/structured/` |
| `models/evaluate.py` | **Fixed a real bug**: `_run_battles()` treated `PPOAgent.act()`'s return (`(action, log_prob, value)` tuple) as a plain int, same as `QAgent`/`DQNAgent`. Every PPO action got JSON-serialized as an array and silently rejected as illegal by the gym forever — every `evaluate.py --model ppo` run since 2026-05-18 hung on battle 1 with zero error output. Fixed + added running win-rate progress logging. |

### Success Criteria
- ✅ `extractFeaturesStructured()` returns `Float32Array` of length `12 × 65 = 780`, shape-stable across move requests, switch requests, and end-of-episode
- ✅ Unknown opponent bench tokens have `unknown_flag=1` and `HP_ratio=1.0`
- ✅ Fainted tokens have `fainted_flag=1` and `HP_ratio=0`
- ✅ PPO with MLP trunk on flattened structured obs achieves ≥ 50% win rate vs RandomPlayerAI at 50k battles — **51% (254/500) on the real evaluation, 2.6M-step training run** (`models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`)
- ❌ ~~Stat boosts for active Pokémon are non-constant~~ — dropped, see above

### Unblocks
M3 (transformer encoder needs per-Pokémon tokens as input) — fully unblocked. The structured representation is verified at parity with the M1 flat baseline.

---

## M2.5: Behavior Cloning Pretraining ✅ COMPLETE

**Status:** ✅ Complete (pending only the M3 warm-start comparison, which needs M3)
**Goals:** Warm-start the transformer policy on human gameplay before PPO ever
runs, using Metamon's parsed human replay dataset
(https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays). PPO then
fine-tunes from human-level play instead of random initialization.

### Why This Matters
PPO from scratch spends its first ~100k battles discovering basics (attack the
opponent, don't switch randomly). Tens of thousands of real gen1ou games encode
that for free. BC pretraining also de-risks M3: if the pretrained policy alone
beats RandomPlayerAI, the (12, 65) representation and transformer are known-good
before any RL debugging starts.

### Action Space Alignment (the load-bearing detail)

Metamon's `MinimalActionSpace` and our gym are both `Discrete(9)`, but index
*grounding* differs: Metamon's 0–3 are the active Pokémon's moves in
alphabetical order and 4–8 are available switches in alphabetical order, while
our gym uses request slot order and fixed bench slots. The adapter therefore
writes move features and bench tokens in Metamon's ordering, preserving the
invariant both systems share: **action k = the move in move-slot k of the own
active token; action 4+j = the Pokémon in bench token j+1.** The policy learns
that invariant, so BC weights transfer to the gym unchanged.

Consequence for M2: `extractFeaturesStructured()` must keep move slots in
request order and bench tokens in `side.pokemon[1..5]` order (already the plan).

### What Was Built
| File | Contents |
|------|----------|
| `scripts/download_metamon.sh` | Installs metamon (clone into `vendor/`, editable pip install), sets `METAMON_CACHE_DIR=./data/metamon_cache`, downloads parsed-replays for gen1ou, prints trajectory count |
| `models/metamon_adapter.py` | `MetamonDataAdapter` — streams `(obs (12,65), action, done)` from raw replay JSONs; M2 token conventions (unknown/fainted flags, monotype type duplication, status/type one-hot order); skips unreconstructable actions (−1); clear error pointing to the download script if data is missing |
| `models/transformer/transformer_policy.py` | `TransformerPolicy` — the M3 architecture (Linear 65→128, 2-layer encoder, nhead=4, d_ff=256, unknown-token attention masking, policy/value heads); `load_pretrain_checkpoint()` — tolerant loader that skips any name/shape-mismatched tensor with a warning |
| `models/bc_pretrain.py` | BC training: cross-entropy on policy head only, Adam lr=1e-3, batch 256, 5 epochs (streamed with a 50k shuffle buffer); flags `--epochs`, `--format`, `--checkpoint_dir`, `--max_files`; saves `models/checkpoints/bc_pretrain_gen1ou.pt` |

### Known Approximations
- Fainted teammates are absent from Metamon's `available_switches`, so their
  identity is lost (padded as generic fainted tokens). A live-but-trapped
  teammate (gen1 Wrap turns) also reads as fainted for those turns.
- Opponent bench uses `opponents_remaining` to split unknown-alive vs fainted
  tokens; the live extractor should mirror this from the omniscient stream.
- Move `disabled` bit is always 0 (not reconstructable from replays).

### M3 Wiring Requirement
`models/transformer/train.py` (built in M3) must accept
`--pretrain_checkpoint <path>` and call
`load_pretrain_checkpoint(model, path)` before PPO starts. Shape-mismatched
layers are skipped with a warning, not an error, so architecture experiments
never hard-fail on old BC checkpoints. M3's agent must build on
`TransformerPolicy` (same module) so checkpoint keys match without remapping.

### Success Criteria
- `download_metamon.sh` completes and reports > 0 gen1ou trajectories
  ✅ 119,536 trajectories (2026-07-02)
- Adapter output verified against M2 spec (token order, one-hot layouts,
  unknown/fainted conventions, action skipping) ✅ (synthetic-fixture tests;
  real-data streaming clean across full dataset, ≈71% moves / 29% switches)
- BC checkpoint loads into `TransformerPolicy` 30/30 tensors; deliberate
  mismatch skips cleanly with warnings ✅ (re-verified on final checkpoint)
- BC top-1 accuracy on gen1ou meaningfully above chance (~11% uniform)
  ✅ **50.5%** — loss 1.339, 4.96M samples/epoch × 5 epochs (24.8M total),
  2414s on MPS; accuracy plateaued within epoch 5 (converged at this capacity)
- After M3 exists: PPO with `--pretrain_checkpoint` reaches the M3 win-rate
  target in fewer battles than from-scratch PPO ⏳

### Unblocks
M3 (warm-started PPO; pre-validated transformer architecture)

---

## M3: Transformer Encoder + PPO Baseline ✅ COMPLETE — NEGATIVE RESULT

**Status:** ✅ Complete and verified (2026-07-13) — **transformer PPO does not beat the MLP-PPO baseline.** Best-ever win rate across ~40 evaluated checkpoints spanning two full training runs (from-scratch 2.6M steps; warm-started up to 7.6M steps; a stability-fixed warm-started retrain to 5M steps) is **46%**, against the M2 MLP-PPO baseline's **51%** at equal (2.6M-step) compute. See Results below for the full trail. Per this milestone's own success criteria and recommendation, **M4/M5/M6 should not proceed on top of this architecture** without a deliberate decision to revisit it.
**Goals:** Replace the MLP trunk with a small transformer encoder. Train with PPO
and establish a transformer win-rate baseline to beat in subsequent milestones.

### Architecture

```
Input:  (batch, 12, 65)  ← 12 Pokémon tokens, each 65-dim

Linear(65 → d_model=128)   ← project tokens into model dimension

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
| `models/transformer/transformer_agent.py` | `TransformerAgent(nn.Module)` with `act()`, `evaluate_actions()`, `update()`, `save()`, `load()` — built on `TransformerPolicy` from `transformer_policy.py` (M2.5) so BC checkpoints load without key remapping |
| `models/transformer/train.py` | PPO loop — reuse `TrajectoryBuffer` from `models/ppo/trajectory_buffer.py`; swap agent class only. Must support `--pretrain_checkpoint` (see M2.5 → M3 Wiring Requirement) |

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

### Success Criteria — ❌ NOT MET
- ❌ Transformer PPO ≥ 80% win rate vs RandomPlayerAI at 200k battles — best-ever was 46% (150-battle spot check, warm-started run 1 @ 2.5M steps ≈ 48k battles)
- ❌ Transformer PPO beats MLP PPO at the same battle count — MLP PPO scored 51% (254/500) at 2.6M steps; transformer's best at the same step count was 46% (warm-started) / 32% (from-scratch)
- ⬜ Attention weight entropy > 0.5 nats — not measured (deferred per the original M3 plan; moot given the win-rate result)
- ❌ Loss curves stable, no divergence — **directly falsified**. The uncontrolled warm-started run collapsed from 46% to 0% between 2.5M and 4.6M steps. Adding approximate-KL early-stopping + LR annealing eliminated the violent collapse but not a milder decay from an early peak (45% @ 500k → 27% @ 5.0M) — PPO fine-tuning degrades this architecture over a long horizon rather than improving it
- ⬜ ≥ 65% win rate vs DamageFirstAI at 200k battles — not tested; moot given the RandomPlayerAI result already falls well short

### Results (full experimental trail)

| Run | Steps | Result | Notes |
|---|---|---|---|
| MLP-PPO baseline (M2) | 2.6M | **51%** (254/500) | Reference baseline |
| Transformer, from-scratch | 2.6M | **32%** (158/500) | Underperforms baseline even before considering stability issues |
| Transformer, warm-started (run 1) | 2.6M | **41%** (204/500) | Warm-start helps (+9pp vs scratch) but still below baseline |
| Transformer, warm-started (run 1, extended) | 7.6M | peak **46%** @ 2.5M–3.1M, then collapsed to 0–13% (3.6M–5.6M), partial recovery to 32% (6.1M), drifting to 24% (7.6M) | Full 21-checkpoint sweep revealed violent, repeated collapse — not gradual drift. Root cause: unconstrained PPO updates over a long horizon with no LR decay/trust region; a single bad update can wreck the learned move-vs-switch balance (`RandomPlayerAI` never voluntarily switches, so over-switching policies lose almost every game) |
| Transformer, warm-started (run 2, +KL early-stop +LR annealing) | 5.0M | peak **45%** @ 500k, decaying to **27%** (135/500) @ 5.0M final | Stability fixes (approx-KL early-stopping in `TransformerAgent.update()`, linear LR annealing in `train.py`) eliminated the violent collapses, but the run still peaks almost immediately (near the BC-pretrained starting point) and degrades under continued training |

**Conclusion:** the transformer's ceiling (~46%) is below the MLP baseline (51%) regardless of warm-starting, training budget (2.6M–7.6M steps tested), or PPO stability fixes. This is a genuine negative result, not an artifact of insufficient training or an unresolved bug.

### Unblocks
M4 (needs trained value function for MCTS leaf evaluation), M5 (opponent modeling head) — **both on hold** pending a decision on whether to revisit the transformer architecture or proceed on the M2 MLP-PPO baseline instead.

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
- Battle state mapper: Showdown protocol lines → structured `(12, 65)` token obs
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
    │  (12 Pokémon tokens × 65 features each)
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
| M2: Structured State | ✅ | Per-Pokémon token obs (12×65) — verified 51% win rate vs RandomPlayerAI | M3 |
| M2.5: BC Pretraining | ✅ | Human-replay warm start (Metamon) | M3 |
| M3: Transformer + PPO | ⬜ | Transformer encoder baseline | M4, M5 |
| M4: MCTS | ⬜ | UCT search over value network | M6 |
| M5: Opponent Modeling | ⬜ | Opp-prediction auxiliary head | M6 |
| M6: Server Integration | ⬜ | Live ladder bot | — |
