# Pokemon Showdown AI Training — Milestones

Goal: Build a genuinely intelligent Pokemon trainer AI via ML training + evaluation on Pokemon Showdown simulator.

Strategy: Model exploration phase (M2) before committing to an architecture. Hypothesis: Deep RL (DQN/PPO) beats tabular and heuristic baselines. Prove it empirically.

---

## M0: Foundation ✅ COMPLETE

**Status:** ✅ Complete  
**Duration:** (Prior work)  
**Artifacts:** docs suite, baseline AI reference

> **Important:** `simulate.js` in the repo root is an unrelated gym-leader simulation script (not part of this ML project). It exists as a useful reference for concurrency patterns and `DamageFirstAI` implementation, but its output (`output/raw_battles.csv`) is not training data for this project.

### Deliverables
- Build system functional (`./build` → `dist/`)
- Documentation suite: 8 core reference files in `docs/`
- Verified: `RandomPlayerAI` base class available at `sim/tools/random-player-ai.ts`
- Verified: `BattleStream` API works for programmatic parallel battles

### Key Files
- `sim/tools/random-player-ai.ts` — base `RandomPlayerAI` class (extend this for custom policies)
- `docs/SETUP.md`, `docs/SIMULATOR-API.md`, `docs/PARALLEL-SIMULATION.md`, `docs/AI-PLAYERS.md`
- `simulate.js` — **reference only** (concurrency patterns, `DamageFirstAI` example); not a project artifact

---

## M1: Environment & Baseline Agent ✅ COMPLETE

**Status:** ✅ Complete  
**Duration:** ~1 session  
**Goals:** Define observation/action spaces, reward function, evaluation protocol

### Deliverables

1. **Gym Wrapper** (`sim/tools/pokemon-gym.ts`)
   - Class `PokemonGymEnv` wrapping `BattleStream`
   - Methods: `reset()` → `obs`, `step(action)` → `(obs, reward, done, info)`
   - Tracks: active Pokemon HP, team HP, status, boosts, held items, move PP
   - Handles: forced switches, illegal moves, game-end conditions
   - Format: `gen1randombattle` (fewest mechanics, fastest training)

2. **Observation Space**
   - Vector-based feature extraction from active request
   - Features (per active Pokémon + opponent):
     - HP ratio (0-1)
     - Level
     - Status condition (bitmask: burn, freeze, paralysis, poison, sleep, tox)
     - Stat boosts (attack, defense, spa, spd, spe, accuracy, evasion): [-6 to +6]
     - Held item ID
     - Type (bitmask, if observable)
   - Available moves (per move):
     - Base power
     - Accuracy
     - PP remaining (as ratio of max)
     - Type (bitmask)
     - Category (physical/special/status)
   - Switch options: valid switch-in Pokémon (bitmask)
   - Opponent field state: active Pokémon (species/type/level if observable)
   - **Total size:** ~100–150 features per turn
   - Format: numpy-compatible flat array

3. **Action Space**
   - Discrete 9-action space:
     - Move 1, 2, 3, 4 (indices 0–3)
     - Switch 1–5 (indices 4–8)
   - Validity masking: `env.valid_actions()` returns boolean mask
   - Agent should respect mask to avoid illegal move penalty

4. **Reward Function** (reward shaping)
   - **Per-turn reward:**
     - `+0.01` for each opponent Pokémon KO'd (normalized to [0,1])
     - `-0.01` for each friendly Pokémon KO'd
     - `+0.001` for favorable type matchup switch (heuristic)
     - `-0.001` for unfavorable switch
     - `+0.0001` for status effect inflicted (burn, paralysis, etc.)
   - **Episode reward:**
     - `+1.0` for winning the battle
     - `-1.0` for losing
     - `-0.001 × turns_taken` (penalize stalling)
   - **Clipping:** reward in [-1, +1]

5. **Evaluation Protocol**
   - Metric 1: **Win Rate vs RandomPlayerAI** (target ≥ 90% after training)
     - 1000 battles per checkpoint
     - Report: win %, KO ratio, avg battle length
   - Metric 2: **Win Rate vs DamageFirstAI** (target ≥ 70%)
     - 500 battles per checkpoint
   - Metric 3: **Battle Efficiency**
     - Avg turns per battle (should not exceed 50 for gen1)
     - Indicator of stalling / infinite loops
   - Evaluation harness: `sim/tools/evaluator.ts`

6. **Test Cases** (unit + integration)
   - Gym reset: valid initial state
   - Step: illegal move masked, legal move accepted
   - Reward: KO detection, win/loss terminal states
   - Observation shape consistency (100–150 features)
   - Deterministic battle with fixed seed reproducibility

### Success Criteria ✅
- Gym wrapper loads, resets, and steps without errors ✅
- Observation shape is consistent (100 features, `OBS_SIZE = 100`) ✅
- Reward function produces sensible values (not NaN, not exploding) ✅
- Battles terminate correctly (win detection via omniscient stream) ✅
- All 10 gym unit tests pass ✅

### Key Files Delivered
- `sim/tools/pokemon-gym.ts` — `PokemonGymEnv` class
- `sim/tools/evaluator.ts` — `evaluate()` and `evaluateVsRandom()`
- `sim/tools/feature-extractor.ts` — `extractFeatures()`, `OBS_SIZE = 100`
- `docs/AI-PLAYERS.md` — gym wrapper usage section added
- `test/tools/gym.test.js` — 10 unit tests, all passing

---

## M2: Model Exploration (Benchmark ≥3 Architectures) 🔜 UP NEXT

**Status:** 🔜 Up Next — M1 complete, Python bridge needed first  
**Duration:** ~2–3 weeks  
**Goals:** Empirically compare model architectures; select winner for M3 scale-up

### Model Candidates

| Model | Type | Framework | Entry Point | Expected Win Rate (Random) | Expected Win Rate (DamageFirst) | Est. Training Time (100k battles) | Notes |
|-------|------|-----------|-------------|------|-----|----|----|
| **Baseline: DamageFirstAI** | Heuristic | N/A | `sim/tools/damage-first-ai.ts` | 60% | — | N/A | Sanity check, already in M0 |
| **Model A: Tabular Q-Learning** | RL (Tabular) | PyTorch + Python | `models/q_learning/train.py` | ~40–50% | ~20% | 1–2h | State discretization limits utility; gen3+ state space explodes. Prove tabular won't work. |
| **Model B: DQN (Deep Q-Network)** | Deep RL | PyTorch + Python | `models/dqn/train.py` | ~80–90% | ~60–70% | 12–24h | Primary hypothesis. Fast convergence, stable. Small MLP (2×128). |
| **Model C: PPO** | Policy Gradient | PyTorch + Python | `models/ppo/train.py` | ~85–95% | ~65–75% | 18–36h | More stable than DQN; handles action space well. Actor-critic. |
| **Model D: MCTS + NN (optional)** | Tree Search + NN | PyTorch + C++ | `models/mcts/train.py` | TBD | TBD | 24–48h | Only if DQN/PPO plateau. AlphaZero-style; requires fast eval. |

### Per-Model Specifications

#### Model A: Tabular Q-Learning (`models/q_learning/`)

**State Discretization:**
- Health buckets: own HP / opponent HP (10 buckets each: 0–10%, 10–20%, ..., 90–100%)
- Move types: compress to 3 types (physical, special, status) × 3 base power tiers (low/med/high)
- Switch: yes/no (have valid switch)
- Total state space: ~10 × 10 × 3 × 3 × 2 = 1800 states

**Implementation:**
- Q-table: `{state_hash: {action: Q_value}}`
- Training loop: 10k episodes, epsilon-greedy (ε=0.1 → 0.05)
- Hyperparameters: α (learning rate) = 0.1, γ (discount) = 0.99

**Success Criteria:**
- Learns to beat RandomPlayerAI (≥ 50% win rate)
- Q-table grows to reasonable size (< 500k states encountered)
- Falls short of DamageFirstAI (confirm tabular limitation)

**Key Files:**
- Create: `models/q_learning/train.py`
- Create: `models/q_learning/q_agent.py`
- Create: `models/q_learning/README.md` (results + analysis)

---

#### Model B: DQN (Deep Q-Network) — **PRIMARY CANDIDATE**

**Architecture:**
- Input: 100–150 features (from gym wrapper)
- Hidden layers: 2 × 128 ReLU
- Output: 9 Q-values (one per action)
- Target network: copy main network every 500 steps

**Training:**
- Environment: `PokemonGymEnv(format='gen1randombattle')`
- Replay buffer: 50k capacity, batch size 32
- Exploration: ε-greedy (ε: 0.5 → 0.01 over 50k steps)
- Loss: MSE(Q_target - Q_pred)
- Optimizer: Adam, learning rate = 1e-3

**Hyperparameter Sweep:**
- Learning rates: [1e-4, 1e-3, 1e-2]
- Batch sizes: [16, 32, 64]
- Exploration schedule: linear, exponential
- Network depth: [1×128, 2×128, 3×256]

**Success Criteria:**
- Win rate ≥ 80% vs RandomPlayerAI after 100k battles
- Win rate ≥ 60% vs DamageFirstAI
- Training stable (Q-values not diverging)
- Checkpoint at 25k, 50k, 75k, 100k battles

**Key Files:**
- Create: `models/dqn/train.py`
- Create: `models/dqn/dqn_agent.py` (network + training loop)
- Create: `models/dqn/replay_buffer.py`
- Create: `models/dqn/README.md` (results + plots)

---

#### Model C: PPO (Proximal Policy Optimization)

**Architecture:**
- Actor (policy): input (100–150 features) → hidden (2 × 128) → output (9 action logits)
- Critic (value): input (100–150 features) → hidden (2 × 128) → output (scalar value estimate)
- Separate networks, shared encoder (optional)

**Training:**
- Rollout horizon: 2000 steps per update (8–16 batches of 2000-step trajectories)
- PPO clip ratio: ε = 0.2
- Entropy bonus: β = 0.01
- GAE λ: 0.95
- Optimizer: Adam, learning rate = 3e-4

**Hyperparameter Sweep:**
- Learning rates: [1e-4, 3e-4, 1e-3]
- Rollout horizons: [1000, 2000, 4000]
- Entropy bonus: [0.0, 0.01, 0.05]
- Network depth: [1×128, 2×128, 3×256]

**Success Criteria:**
- Win rate ≥ 85% vs RandomPlayerAI after 100k battles
- Win rate ≥ 65% vs DamageFirstAI
- More stable than DQN (lower variance in rollouts)
- Checkpoint at 25k, 50k, 75k, 100k battles

**Key Files:**
- Create: `models/ppo/train.py`
- Create: `models/ppo/ppo_agent.py` (actor-critic + training loop)
- Create: `models/ppo/trajectory_buffer.py`
- Create: `models/ppo/README.md` (results + plots)

---

#### Model D (Optional): MCTS + Value Network

Only pursue if DQN/PPO plateau below 90% win rate.

**Architecture:**
- Value network: 3-layer MLP, predicts win probability from board state
- MCTS: tree search over 10–20 plausible actions per turn (reduce branching)
- Simulation: NN eval (fast) + shallow rollout (default random, 5–10 moves)

**Expected Training:** 24–48h for 100k battles (slow; only if necessary)

---

### M2 Timeline & Orchestration

**Phase 1: Baseline Setup (Days 1–2)**
- Verify gym wrapper from M1 is stable
- Implement Python bridge: `PokemonGymClient` to call Node.js gym via subprocess
- Write evaluation harness in Python

**Phase 2: Model A — Tabular Q-Learning (Days 3–4)**
- Implement state discretization
- Train 10k episodes, measure win rate
- Document state space growth, performance ceiling

**Phase 3: Model B & C in Parallel (Days 5–14)**
- Model B (DQN): train 100k battles, log metrics every 5k
- Model C (PPO): train 100k battles, log metrics every 5k
- Run hyperparameter sweeps on best candidates (top 3 configs each)

**Phase 4: Comparison & Selection (Days 15–17)**
- Consolidate results: win rate, training stability, sample efficiency
- Ablation studies: layer count, learning rate sensitivity
- Write comparison report (`docs/MODEL-COMPARISON.md`)
- **Decision:** Select DQN or PPO (or both, if tie) for M3

**Phase 5 (Optional): Model D — MCTS (Days 18–25)**
- Only if winner < 90% vs RandomPlayerAI
- Implement fast NN eval + search
- Expected to take longer; justify time investment

### Success Criteria for M2
- **Model A:** Implemented, confirms tabular limitation
- **Model B (DQN):** Achieves ≥ 80% win rate, training reproducible
- **Model C (PPO):** Achieves ≥ 85% win rate, stable
- **Model D (optional):** Only if above plateau
- **Comparison:** Winner has ≥ 85% vs RandomPlayerAI, ≥ 60% vs DamageFirstAI
- **Report:** 3–5 page analysis in `docs/MODEL-COMPARISON.md`

### Key Files to Create
- `models/q_learning/` (train.py, q_agent.py, README.md)
- `models/dqn/` (train.py, dqn_agent.py, replay_buffer.py, README.md)
- `models/ppo/` (train.py, ppo_agent.py, trajectory_buffer.py, README.md)
- `models/mcts/` (optional; train.py, mcts_agent.py, README.md)
- `models/` (root): `train_env.py` (Python gym client), `evaluate.py` (shared eval logic)
- `docs/MODEL-COMPARISON.md` (results + winner selection)

---

## M3: Deep RL Training at Scale ⬜ NOT STARTED

**Status:** ⬜ Not Started  
**Duration:** ~3–4 weeks  
**Goals:** Scale winning architecture (DQN or PPO) to 1M+ battles, hyperparameter optimization, self-play

### Deliverables

1. **Scale Winner to 1M Battles**
   - Extend training from 100k → 1M battles
   - Format: `gen1randombattle` (primary), then `gen3randombattle` (stretch)
   - Checkpoints: every 50k battles
   - Metrics: win rate vs RandomPlayerAI, vs DamageFirstAI, vs self

2. **Hyperparameter Optimization**
   - Grid/random search over learning rates, batch sizes, network depth
   - Early stopping: if win rate plateaus for 100k battles, stop
   - Document all sweeps in `models/{dqn|ppo}/hyperparameter_sweep.csv`

3. **Self-Play Training**
   - After 500k battles: agent plays vs previous version (`main` vs `v1_500k`, etc.)
   - Alternate: odd iterations vs RandomPlayerAI, even vs self
   - Track ELO or win rate progression across versions

4. **Format Progression**
   - Milestone 1: Master `gen1randombattle` (current target)
   - Milestone 2 (stretch): Train on `gen3randombattle` (more mechanics, larger state space)
   - Milestone 3 (stretch): Evaluate on `gen7randombattle` (Dynamax, Terrain, etc.)

5. **Checkpointing & Resumption**
   - Save network weights, optimizer state, replay buffer every checkpoint
   - Implement `load_checkpoint(path)` to resume training mid-run
   - Allows recovery from interruptions

6. **Final Evaluation**
   - 5000 battles vs RandomPlayerAI (≥ 95% win rate)
   - 2000 battles vs DamageFirstAI (≥ 75% win rate)
   - 1000 battles vs previous champion model (if applicable)
   - Report: mean reward per episode, KO ratio, avg battle length

### Success Criteria
- Trained agent achieves ≥ 95% win rate vs RandomPlayerAI
- Trained agent achieves ≥ 75% win rate vs DamageFirstAI
- Training stable: no divergence, no reward collapse
- Format progression: eval on gen1 + gen3 at least
- Self-play: demonstrates learning curve improvement across versions

### Key Files to Create/Modify
- Modify: `models/{dqn|ppo}/train.py` → add checkpointing, self-play mode
- Create: `models/{dqn|ppo}/hyperparameter_sweep.py` (grid search harness)
- Create: `models/{dqn|ppo}/TRAINING-RESULTS.md` (logs, plots, analysis)
- Modify: `docs/ML-TRAINING.md` → add scaling guide and self-play section

---

## M4: Showdown Server Integration ⬜ NOT STARTED

**Status:** ⬜ Not Started  
**Duration:** ~2 weeks  
**Goals:** Connect trained model to live Pokemon Showdown server, accept battle challenges from humans

### Deliverables

1. **WebSocket Client to Pokemon Showdown**
   - Connect to `sim.smogon.com` or local test server
   - Authenticate with bot credentials
   - Listen for challenge events

2. **Battle Handler**
   - Receive battle state (standard Showdown protocol)
   - Map to gym observation
   - Run policy inference (forward pass through trained model)
   - Send move choice within 2-second latency budget

3. **Local Server Setup (Testing)**
   - Docker or local Node.js instance of Showdown
   - Bot account registration
   - Challenge acceptance via CLI or HTTP

4. **Inference Optimization**
   - Model quantization (if needed): FP32 → FP16 or INT8
   - Batch inference: queue moves, process multiple battles in parallel
   - Latency profiling: ensure < 2s per move

5. **Logging & Monitoring**
   - Per-battle log: teams, moves, outcome, latency
   - Per-bot metric: win rate on server, opponent diversity, response time
   - Alert on crashes or timeout

### Success Criteria
- Bot connects to Showdown and accepts challenges
- Latency < 2 seconds per move decision
- Wins ≥ 60% of matches vs random human/bot opponents
- Runs for ≥ 100 consecutive battles without crashing

### Key Files to Create
- Create: `server/bot-client.ts` (WebSocket + auth)
- Create: `server/battle-handler.ts` (state mapping, inference)
- Create: `server/config.ts` (Showdown URL, credentials, latency budget)
- Create: `server/logging.ts` (per-battle + per-bot metrics)
- Create: `docs/DEPLOYMENT.md` (local server setup, Docker guide)

---

## M5: Evaluation & Iteration ⬜ NOT STARTED

**Status:** ⬜ Not Started  
**Duration:** ~2 weeks (ongoing)  
**Goals:** Comprehensive evaluation, human playtest, document failure modes and next research directions

### Deliverables

1. **Elo Ladder**
   - Track model rating across 1000+ battles vs varied opponents
   - Update rating after each match (Glicko-2 or simplified Elo)
   - Publish rating history (CSV)

2. **Tournament Mode**
   - Round-robin: model versions play each other
   - Format: gen1, gen3, gen7 (if trained on all)
   - Output: tournament bracket, cross-version win rates

3. **Failure Analysis**
   - Identify loss patterns: specific types weak, stalling behaviors, etc.
   - Per-format analysis: gen1 vs gen3 vs gen7
   - Annotate 50–100 loss replays with decision points

4. **Human Playtest**
   - User (and optionally other testers) play against model on Showdown
   - Subjective feedback: difficult? fun? predictable?
   - Win rate over 100+ battles
   - Optional: compare vs public Showdown bots

5. **Ablation Study (if time)**
   - Feature importance: which observation features matter most?
   - Architecture sensitivity: layer count, hidden size impact
   - Reward shaping: which reward components drive learning?

6. **Research Directions Document** (`docs/FUTURE-WORK.md`)
   - What worked: DQN/PPO, architecture decisions
   - What didn't: failure modes, dead ends
   - Next steps: multi-format transfer learning, opponent adaptation, curriculum learning, etc.

### Success Criteria
- Elo ladder implemented and tracked for ≥ 500 matches
- Tournament completed, cross-version analysis published
- 50+ loss replays annotated with decision analysis
- Human playtest completed (≥ 100 matches, feedback documented)
- FUTURE-WORK.md written with 5+ concrete next-step ideas

### Key Files to Create/Modify
- Create: `server/elo-ladder.ts` (rating computation, history logging)
- Create: `server/tournament.ts` (bracket generation, match orchestration)
- Create: `eval/failure-analysis.py` (pattern detection, replay annotation)
- Create: `docs/FUTURE-WORK.md` (research directions, next experiments)
- Modify: `docs/ML-TRAINING.md` → add human playtest guide

---

## Summary Timeline

| Milestone | Status | Duration | Approx. Dates |
|-----------|--------|----------|---------------|
| M0: Foundation | ✅ Complete | (Prior) | — |
| M1: Environment & Baseline | ✅ Complete | ~1 session | — |
| M2: Model Exploration | 🔜 Up Next | ~2–3 weeks | Week 1–4 |
| M3: Scale Training | ⬜ Not Started | ~3–4 weeks | Week 5–9 |
| M4: Showdown Integration | ⬜ Not Started | ~2 weeks | Week 9–11 |
| M5: Evaluation & Iteration | ⬜ Not Started | ~2 weeks (ongoing) | Week 11–13+ |

**Total critical path:** ~11–13 weeks (assuming parallelization in M2, M3).

---

## Notes

- **M0 is already complete.** Don't redo it.
- **M1 is blocking:** gym wrapper is needed by M2 models.
- **M2 is research:** three models running in parallel, expect some to fail.
- **M3 builds on M2 winner:** only one model advances.
- **M4–M5 are integration + evaluation:** overlap possible after M3 checkpoints are stable.
- **Success metric throughout:** agent beats both RandomPlayerAI (≥ 90%) and DamageFirstAI (≥ 70%).
