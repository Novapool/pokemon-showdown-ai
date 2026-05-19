# In Progress — Pokemon Showdown AI Training

Last updated: 2026-05-19

---

## Current Work

**Milestone:** M2 (Structured State Representation) — Architecture direction locked, ready to build  
**Phase:** Replace flat 100-dim feature vector with per-Pokémon token representation

### Active Tasks
- [ ] Add `extractFeaturesStructured()` to `sim/tools/feature-extractor.ts` — returns `(12, 65)` token array
- [ ] Add boost tracker to `sim/tools/pokemon-gym.ts` — thread stat boosts into token features
- [ ] Update `models/gym_bridge.js` — serialize structured obs as 780-element flat array
- [ ] Update `models/gym_client.py` — reshape `(780,)` → `(12, 65)` numpy array
- [ ] Verify: run MLP PPO on flattened structured obs, confirm ≥ 50% win rate vs RandomPlayerAI at 50k battles

---

## Active Plan

**M2 Execution Plan:**

1. **Token schema** (in `feature-extractor.ts`)
   - 12 tokens: own active, own bench ×5, opponent active, opponent bench ×5
   - Per token (65 dims): HP_ratio, level/100, type1_onehot(15), type2_onehot(15), status_onehot(6), active_flag, unknown_flag, fainted_flag, 4×move_features(6)
   - Unknown opponent bench: `unknown_flag=1, HP_ratio=1.0`, all other features 0
   - Fainted: `fainted_flag=1, HP_ratio=0`

2. **Stat boost tracking** (in `pokemon-gym.ts`)
   - Parse `|-boost|` and `|-unboost|` lines from omniscient stream
   - Accumulate per-slot boost dict; reset on switch
   - Pass into `extractFeaturesStructured()` for the active Pokémon token

3. **Bridge update** (in `gym_bridge.js` + `gym_client.py`)
   - Serialize: flatten `(12, 65)` → 780-element array for JSON transport
   - Python side: `np.array(obs).reshape(12, 65)`
   - Keep `--flat` mode for backward compat with old MLP baseline

4. **Verification**
   - PPO with trunk `Linear(780, 128) → ReLU → Linear(128, 128)` on flattened tokens
   - Target: ≥ 50% vs RandomPlayerAI at 50k battles (parity with old flat vector confirms correctness)
   - Check token shape stability across move requests, switch requests, end-of-episode

---

## Recently Completed

✅ **Job 2.4: evaluate.py + MODEL-COMPARISON.md** (2026-05-18)
- Created `models/evaluate.py` — CLI evaluation script (`--model`, `--checkpoint`, `--battles`); loads agent from checkpoint, sets epsilon=0.0 for q_learning/dqn greedy eval, runs N battles via `GymClient`, reports win rate vs RandomPlayerAI
- Created `docs/MODEL-COMPARISON.md` — results comparison template with Overview, Models Evaluated, Results table (all TBD), Analysis placeholder, Winner Selection placeholder, Next Steps pointing to M3

✅ **Job 2.3: ppo/** (2026-05-18)
- Created `models/ppo/trajectory_buffer.py` — `TrajectoryBuffer` with GAE advantage computation, normalized advantages, `push()`/`compute_advantages()`/`get_tensors()`/`clear()` API
- Created `models/ppo/ppo_agent.py` — `PPOAgent` with shared trunk (100→128→128), policy head (128→9), value head (128→1), `act()` with valid_mask, `evaluate_actions()`, `update()` with PPO clipped surrogate + value loss + entropy bonus, `save()`/`load()` classmethod
- Created `models/ppo/train.py` — rollout-based loop collecting `rollout_steps` transitions across episode boundaries, per-rollout win-rate and loss logging, checkpoints to `models/ppo/checkpoints/ppo_step_{N}.pt`, `env.close()` in finally
- Created `models/ppo/checkpoints/` directory

✅ **Job 2.2: dqn/** (2026-05-18)
- Created `models/dqn/replay_buffer.py` — `ReplayBuffer` with deque(maxlen), `push()` converts to numpy, `sample()` returns 5 CPU torch tensors with correct dtypes
- Created `models/dqn/dqn_agent.py` — `QNetwork` MLP (100→128→128→9), `DQNAgent` with policy/target nets, ε-greedy `act()` with valid_mask, `learn()` with MSE loss, `update_target()`, `decay_epsilon()`, `save()`/`load()` classmethod
- Created `models/dqn/train.py` — step-based loop (not episode-based), rolling 500-episode win-rate/loss logging, checkpoints to `models/dqn/checkpoints/dqn_step_{N}.pt`, `env.close()` in finally
- Created `models/dqn/checkpoints/` directory

✅ **Job 2.1: q_learning/** (2026-05-18)
- Created `models/q_learning/q_agent.py` — `QAgent` with defaultdict Q-table, 5-element state discretization, epsilon-greedy `act()`, TD(0) `update()`, `decay_epsilon()`, pickle `save()`/`load()`
- Created `models/q_learning/train.py` — episode loop with `GymClient`, rolling 500-episode win rate logging, Q-table saved to `qtable.pkl` after training, `env.close()` in finally block
- Created `models/q_learning/README.md` — overview, architecture, training instructions, hyperparameters table, placeholder Results and Analysis sections

✅ **Job 1.2: gym_client.py** (2026-05-18)
- Created `models/gym_client.py` — `GymClient` class spawning `gym_bridge.js` via subprocess
- Implements `reset()`, `step()`, `valid_actions()`, `close()` with correct numpy dtypes
- Line-delimited JSON protocol: one command written per `_send()` call, one response line read back
- Error responses (`{"error":"..."}`) raise `RuntimeError`
- `__main__` block for smoke testing
- No external deps beyond `subprocess`, `json`, `numpy`, `pathlib`

✅ **Job 1.1: gym_bridge.js** (2026-05-18)
- Created `models/gym_bridge.js` — line-delimited JSON stdio server wrapping `PokemonGymEnv`
- Supports `reset`, `step`, `valid_actions`, `close` commands
- Sequential async processing preserves request/response ordering
- `obs` returned as plain Array (not Float32Array) for JSON serialization
- Unhandled exceptions written to stdout; process never crashes silently
- `models/` directory created

✅ **M1: Environment & Baseline Agent** (2026-05-10)
- **Job 3.1:** Created `sim/tools/pokemon-gym.ts` — `PokemonGymEnv` class with `reset()`, `step(action)`, `validActions()`, `destroy()`. Background omniscient reader, reward parsing, valid-action masking.
- **Job 3.2a:** Created `sim/tools/feature-extractor.ts` — 100-feature fixed-size extraction (own active, moves, switch mask, opponent, padding).
- **Job 3.2b:** Created `sim/tools/evaluator.ts` — Parallel battle runner with `evaluate()`, `evaluateVsRandom()`, up to 50 concurrent battles.
- **Job 3.1 (revised):** Added unit tests in `test/tools/gym.test.js` — reset/step validation, legal move masking, reward bounds, observation consistency, determinism check. Written as plain JS (matches project test convention) in `test/tools/` so mocharc picks it up automatically.
- **Job 3.4:** Updated `docs/AI-PLAYERS.md` — new "Gym Wrapper (PokemonGymEnv)" section covering what the gym is, quick start, observation/action space, reward function, evaluator usage, seeding.
- All code compiles under strict TypeScript with zero errors. Gym tested for 100 battles without crashes.

✅ **Job 2.2: pokemon-gym.ts** (2026-05-10)
- Created `sim/tools/pokemon-gym.ts`
- Exports `PokemonGymEnv` with `reset()`, `step(action)`, `validActions()`, `destroy()`
- Exports `GymStepResult` interface
- GymPlayer extends BattlePlayer with Promise-based request handoff
- Background omniscient reader buffers all battle lines for reward parsing
- Reward: +0.01 per opponent KO, -0.01 per own KO, +0.0001 status, ±1.0 win/loss, stalling penalty
- Compiles clean under strict TypeScript (zero errors)

✅ **Job 2.1: feature-extractor.ts** (2026-05-10)
- Created `sim/tools/feature-extractor.ts`
- Exports `extractFeatures(request, opponentRequest): Float32Array` and `OBS_SIZE = 100`
- 100-feature layout: own active (0–14), moves (15–54), switch mask (55–59), opponent (60–74), padding (75–99)
- Compiles under strict TypeScript with no errors

✅ **M0: Foundation**
- Build system (`./build`) working, `dist/` populated
- Documentation suite created (8 reference docs in `docs/`)
- Verified: `RandomPlayerAI` base class at `sim/tools/random-player-ai.ts`
- Verified: `BattleStream` API works for programmatic parallel battles

> **Note:** `simulate.js` in the repo root is an unrelated script (gym leader battles for a separate project). It is a useful code reference but is not a deliverable of this ML project. Its output in `output/` is similarly unrelated.

---

## Blockers

None. All components verified and working.

### Assumptions Made
- Python bridge for training models: new code (not integrating existing ML code)
- PyTorch preferred for neural networks (fast iteration)
- Gen1 format chosen for training (smallest state space, fastest convergence)

---

## Next Steps

### Immediate (M2 — Structured State)
1. **Build `extractFeaturesStructured()`** in `sim/tools/feature-extractor.ts`
   - 12 tokens × 65 features, schema above
   - Unknown-flag and fainted-flag semantics correct

2. **Add boost tracker** to `sim/tools/pokemon-gym.ts`
   - Parse `|-boost|` / `|-unboost|` from omniscient lines
   - Thread into token features for own/opponent active

3. **Update bridge serialization** in `gym_bridge.js` + `gym_client.py`
   - 780-element flat array over JSON; reshape to `(12, 65)` on Python side

4. **Verification run** — MLP PPO on flattened structured obs, 50k battles vs Random

### Follow-Up (M3 — Transformer)
After M2 verified:
- Build `models/transformer/transformer_agent.py` (2-layer encoder, d=128)
- Train with PPO, 200k–500k battles
- Compare vs MLP PPO baseline at same battle count

### Stretch
- Attention weight visualization to confirm non-uniform attention
- Start scoping MCTS determinizer (opponent team sampling from gen1 usage data)

---

## Key Metrics to Track

**M1 Success Indicators:**
- Gym env runs 100 battles without crashes
- Observation shape: consistent (same size every turn)
- Reward bounds: all rewards in [−1, +1]
- Baseline win rates:
  - RandomPlayerAI: ~50% (sanity check, should beat coin flip)
  - DamageFirstAI: ~60% (heuristic baseline)
- Evaluation runtime: < 5 min for 1500 battles

**M2 Success Indicators (later):**
- Model A (Tabular Q): ≥ 50% vs Random, < 70% vs DamageFirst (confirms limitation)
- Model B (DQN): ≥ 80% vs Random, ≥ 60% vs DamageFirst (primary hypothesis)
- Model C (PPO): ≥ 85% vs Random, ≥ 65% vs DamageFirst (stability)

---

## Dependencies & Ordering

**Hard blocking:** M1 must complete before M2 (gym needed by all models)

**Can overlap:**
- M2 & M3: If M2 winner is clear at 100k battles, can extrapolate training while exploring other models
- M3 & M4: Checkpoints from M3 can be loaded into M4 inference during training
- M4 & M5: Start human playtest while model still training (use stable checkpoints)

---

## Files & Directories

### Core Simulation (M0, already done)
- `sim/tools/random-player-ai.ts` — base AI class (extend for custom policies)
- `dist/sim/index.js` — compiled simulator API (`BattleStream`, `Dex`, `PRNG`)
- **Reference only (unrelated project):** `simulate.js`, `output/raw_battles.csv`

### M1 (Gym & Baseline)
- **New:** `sim/tools/pokemon-gym.ts` (gym wrapper)
- **New:** `sim/tools/feature-extractor.ts` (observation building)
- **New:** `sim/tools/evaluator.ts` (eval harness)
- **New:** `test/tools/gym.test.js` (unit tests)
- **Modify:** `docs/AI-PLAYERS.md` (gym usage section)

### M2 (Model Exploration)
- **New:** `models/` directory (root)
- **New:** `models/q_learning/` (tabular Q)
- **New:** `models/dqn/` (DQN)
- **New:** `models/ppo/` (PPO)
- **New:** `models/` base: `train_env.py`, `evaluate.py`
- **New:** `docs/MODEL-COMPARISON.md` (results & winner selection)

### M3 (Scale Training)
- **Modify:** `models/{dqn|ppo}/train.py` (checkpointing, self-play)
- **New:** `models/{dqn|ppo}/TRAINING-RESULTS.md` (logs & analysis)

### M4 (Showdown Integration)
- **New:** `server/bot-client.ts` (WebSocket)
- **New:** `server/battle-handler.ts` (inference)
- **New:** `docs/DEPLOYMENT.md` (setup guide)

### M5 (Evaluation)
- **New:** `server/elo-ladder.ts` (rating tracking)
- **New:** `docs/FUTURE-WORK.md` (research directions)

---

## Checklist for Session Start

When starting a new session:

- [ ] Review this file for current work status
- [ ] Check MILESTONES.md for milestone definitions
- [ ] If blocked, resolve blockers (see Blockers section)
- [ ] If continuing M1: check last day's progress
- [ ] Update "Recently Completed" when tasks finish
- [ ] Update "Active Tasks" when starting new work
- [ ] Commit this file after major progress

---

## Session Log

### Session 3 (2026-05-18)
- **Action:** Verified Job 3.1 — gym_bridge.js and gym_client.py integration tests
- **Status:** All three checks PASS
  - Check 1 (syntax): `node --check gym_bridge.js` → exit 0 ✓
  - Check 2 (bridge reset): reset returns valid JSON with obs array of length 100 ✓
  - Check 3 (Python client): gym_client.py smoke test runs without errors, obs shape (100,), valid_actions works, step returns correct tuple ✓
- **Notes:** numpy was not installed initially; installed via pip, then all tests passed
- **Next:** Ready to begin M2 phase training (Model A/B/C) or continue verification tasks

### Session 2 (2026-05-10)
- **Action:** Completed M1 gym wrapper, feature extractor, evaluator, unit tests, docs update
- **Status:** M1 complete. All code compiles under strict TypeScript. Gym tested 100+ battles without crashes.
- **Deliverables:** `pokemon-gym.ts`, `feature-extractor.ts`, `evaluator.ts`, `test/tools/gym.test.js`, `docs/AI-PLAYERS.md` (new Gym Wrapper section)
- **Next:** M2 model exploration — Python bridge, Model A (Tabular Q), Models B & C (DQN/PPO) in parallel

### Session 1 (2026-05-10)
- **Action:** Created docs suite (8 reference docs), MILESTONES.md, IN-PROGRESS.md
- **Status:** Project director defined 6 milestones (M0–M5), M0 complete, M1–M5 scoped
- **Clarified:** `simulate.js` is an unrelated script (not an ML project artifact)
- **Next:** Begin M1 gym wrapper implementation

