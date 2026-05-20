# In Progress — Pokemon Showdown AI Training

Last updated: 2026-05-20

---

## Current Work

**Phase:** M4 — Parallel gym + DamageFirstAI opponent  
**Goal:** 4× episode throughput via VecGymClient + sharper learning signal via DamageFirstAI

### Active Tasks
- [x] Implement `models/vec_gym_client.py` — wraps N `GymClient` instances, synchronous VecEnv API
- [x] Add `sim/tools/damage-first-ai.ts` — extends `RandomPlayerAI`, overrides `chooseMove()` to pick highest base-power non-disabled move
- [x] Update `sim/tools/pokemon-gym.ts` — accept `opponent` option (`'random'` | `'damage-first'`)
- [x] Update `models/gym_bridge.js` — pass `--opponent` flag to gym env
- [x] Update `models/gym_client.py` — expose `opponent` constructor param (added `opponent='random'` kwarg; passes `--opponent <val>` to bridge when not `'random'`)
- [x] Update `models/transformer/train.py` — use `VecGymClient`, `--n-envs`, `--opponent` args, batch `act()` calls

---

## Active Plan

**M4 Plan:**

1. **VecGymClient** (`models/vec_gym_client.py`)
   - Wraps N independent `GymClient` subprocesses
   - Synchronous: step all N envs, collect all N results
   - Auto-reset done envs inline
   - API: `reset() → (obs_batch, mask_batch)`, `step(actions) → (obs_batch, rewards, dones, infos, masks)`

2. **DamageFirstAI** (`sim/tools/damage-first-ai.ts`)
   - Extends `RandomPlayerAI`, overrides `chooseMove()`
   - Picks highest `basePower` non-disabled move; breaks ties randomly
   - Zero-setup: uses existing Dex already loaded

3. **Gym opponent selection** (`pokemon-gym.ts`, `gym_bridge.js`, `gym_client.py`)
   - `PokemonGymEnv` constructor accepts `{ opponent: 'random' | 'damage-first' }`
   - Bridge passes `--opponent damage-first` flag
   - `GymClient(opponent='damage-first')` for easy switching

4. **Training loop** (`models/transformer/train.py`)
   - Replace single `GymClient` with `VecGymClient(n_envs=4)`
   - Collect `rollout_steps` ticks across all envs (total transitions = rollout_steps × n_envs)
   - Add `--n-envs` (default 4) and `--opponent` (default `random`) CLI args
   - Batch `agent.act()` calls using `agent.batch_act(obs_batch, masks_batch)` for efficiency

---

## Recently Completed

✅ **feature-extractor.ts bug fixes** (2026-05-20)
- Edit A: Fixed 3 errors in GEN1_TYPE_CHART — Ice vs Ice now neutral (removed 0.5×), Fire vs Dragon now neutral (removed 0.5×), Ghost vs Ghost now immunity 0× (was 2×)
- Edit B: `computeEffectiveness()` now returns log2-scaled output: 0→0.0, 0.5×→0.25, 1×→0.5, 2×→0.75, 4×→1.0. Default effectiveness updated 0.25→0.5 (neutral)
- Edit C: TOKEN_DIM 69→73. `fillPokemonToken` and `fillOpponentPokemonToken` gain `boostMap` param + 4-dim boost block (atk/def/spe/spc /6). `extractFeaturesStructured` threads `boosts.ownActive` to Token 0 and `boosts.oppActive` to Token 6. Flat obs: 828→876.

✅ **M3.5: Feature extractor cleanup + type effectiveness** (2026-05-19)
- Added `GEN1_TYPE_CHART` and `computeEffectiveness()` to `feature-extractor.ts`
- Each move block gains a 7th feature: effectiveness vs opponent's active type(s), normalized /4
- Removed atk-boost-into-base-power and def-boost-into-move-slot hacks (semantic corruption)
- TOKEN_DIM: 65 → 69. Flat obs: 780 → 828. All models + gym_client updated.
- Gym client: `reshape(12, -1)` instead of hardcoded `(12, 65)`
- Build passes clean, all 4 agent smoke tests pass

✅ **M3: Transformer architecture + training runs** (2026-05-19)
- `models/transformer/transformer_agent.py` — 2-layer TransformerEncoder (d_model=128, nhead=4), token_proj Linear(69,128), learnable pos_embed, mean-pool → policy/value heads
- `models/transformer/train.py` — rollout loop, LR decay (3e-4→1e-5), `--n-envs` pending
- **Run 6** (200k, no type eff): peak 0.53, final 0.52
- **Run 7** (200k, type eff, no decay): peak 0.54, final 0.48
- **Run 8** (200k, type eff, LR decay): peak 0.53, final 0.50
- **Run 9** (500k, type eff, LR decay, cleaned obs): **peak 0.55**, final 0.52 ← best run
- Conclusion: single-env ceiling ~0.52-0.55; bottleneck is episode throughput (~9,800 episodes/500k steps)

✅ **M2: Structured State Representation** (2026-05-19)
- `extractFeaturesStructured()` in `feature-extractor.ts` — (12, 69) token array
- Bug fixed: voluntary switches removed from move-turn valid mask (4% → 47% random baseline)
- Bug fixed: obs_size defaults updated 100 → 780 → 828 across all models
- Verified ≥ 50% rolling win rate vs RandomPlayerAI

✅ **M1: Environment & Baseline Agent** (2026-05-10)
- `sim/tools/pokemon-gym.ts` — `PokemonGymEnv` with reset/step/validActions/destroy
- `sim/tools/feature-extractor.ts` — structured token extraction
- `models/gym_bridge.js` + `models/gym_client.py` — Python↔Node.js bridge
- `models/ppo/`, `models/dqn/`, `models/q_learning/` — all three model types

✅ **M0: Foundation** (2026-05-10)
- Build system, documentation suite, `RandomPlayerAI` base class verified

---

## Test Results (2026-05-20 — TOKEN_DIM=73 / boost dims verification)

**Overall verdict: PASS**

| Check | Command | Exit Code | Result |
|---|---|---|---|
| 1 — TypeScript build | `./build` | 0 | Pass |
| 2 — TOKEN_DIM constants | `node -e "... TOKEN_DIM * N_TOKENS"` | 0 | Pass |
| 3 — gym_bridge syntax | `node --check models/gym_bridge.js` | 0 | Pass |
| 4 — Python imports (ppo, dqn, transformer) | `python3 -c "importlib.util..."` | 0 | Pass |

**Check 1 output:**
```
(no output — clean build)
```

**Check 2 output:**
```
TOKEN_DIM: 73 N_TOKENS: 12 obs_size: 876
```

**Check 3 output:**
```
(no output — syntax clean)
```

**Check 4 output:**
```
ppo_agent: import OK
dqn_agent: import OK
transformer_agent: import OK
All Python imports: OK
```

---

## Previous Test Results (2026-05-19 — M4 post-build checks)

**Overall verdict: PASS**

| Check | Command | Exit Code | Result |
|---|---|---|---|
| 1 — TypeScript build | `./build` | 0 | Pass |
| 2 — TOKEN_DIM constants | `node -e "... TOKEN_DIM * N_TOKENS"` | 0 | Pass |
| 3 — VecGymClient smoke test | `python models/vec_gym_client.py` | 0 | Pass |
| 4 — gym_bridge syntax | `node --check models/gym_bridge.js` | 0 | Pass |
| 5 — Python imports + DamageFirstAI | combined node + python -c | 0 | Pass |

**Check 2 output:**
```
TOKEN_DIM: 69 N_TOKENS: 12 obs_size: 828
```

**Check 3 output:**
```
reset obs_batch.shape: (2, 12, 69)
reset len(masks_batch): 2
step obs_batch.shape: (2, 12, 69)
step rewards: [0.0, 0.0]
step dones: [False, False]
step len(masks_batch): 2
VecGymClient smoke test: OK
```

**Check 5 output:**
```
imports OK
DamageFirstAI compiled OK: function
```

---

## Blockers

None.

---

## Next Steps

### Immediate (M4)
1. Implement `VecGymClient` — 4 parallel envs, 4× episode throughput
2. Add `DamageFirstAI` opponent — better learning signal than pure random
3. Train with mixed opponents (80% Random + 20% DamageFirst initially)
4. Target: consistent WR(rolling-500) ≥ 0.57 vs RandomPlayerAI

### Follow-up
- Self-play: only after consistently beating DamageFirstAI (WR ≥ 0.65)
- Re-enable voluntary switches once opponent modeling is better
- MCTS / opponent team sampling for M5

---

## Key Architecture State

| Component | Current state |
|---|---|
| Observation | (12, 73) tokens — HP, level, types, status, flags, 4×moves(7 dims incl. effectiveness), 4×boosts |
| Action space | 9 actions — moves 0-3, force-switch slots 4-8 (voluntary switches disabled) |
| Token dim | 73 — flat obs 876 |
| Primary model | `TransformerAgent` (2-layer encoder, d=128) |
| Best checkpoint | `transformer_step_500000_final.pt` — 0.52 WR, peak 0.55 |
| Opponent | `RandomPlayerAI` (move=1.0, never switches) |
| Reward | Full: ±KO, ±status, ±1 win/loss, stalling penalty |
| Training | rollout=2048, entropy=0.05, ppo_epochs=2, lr 3e-4→1e-5 |

---

## Files & Directories

| Path | Purpose |
|---|---|
| `sim/tools/feature-extractor.ts` | Token extraction — TOKEN_DIM=73, GEN1_TYPE_CHART, boost dims [69–72] |
| `sim/tools/pokemon-gym.ts` | Battle gym env — validActions, reward, boost tracking |
| `sim/tools/random-player-ai.ts` | Base opponent AI |
| `models/gym_bridge.js` | Node.js stdio bridge |
| `models/gym_client.py` | Python GymClient |
| `models/vec_gym_client.py` | **[TODO]** Parallel env wrapper |
| `models/transformer/transformer_agent.py` | TransformerAgent — PPO Actor-Critic |
| `models/transformer/train.py` | Training loop |
| `models/transformer/checkpoints/` | Saved checkpoints |
| `docs/TRAINING-LOG.md` | Living lab notebook — all runs, bugs, decisions |
