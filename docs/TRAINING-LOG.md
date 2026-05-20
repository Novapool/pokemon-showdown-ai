# AI Training Log

Running record of training experiments, bugs found, architectural decisions, and results.
Update this file whenever training results are shared, bugs are found, or significant architectural changes are made.

---

## Environment Setup

- **Format:** Gen 1 Random Battle (`gen1randombattle`)
- **Opponent:** `RandomPlayerAI` (always picks a random valid move, never switches voluntarily — `move=1.0`)
- **Gym:** `PokemonGymEnv` (Node.js) bridged to Python via line-delimited JSON stdio
- **Observation:** (12, 73) float32 token array — 12 Pokémon tokens × 73 features
  - Tokens: own active, own bench ×5, opponent active, opponent bench ×5
  - Features: HP ratio, level, type1/type2 one-hot (15 each), status one-hot (6), active/unknown/fainted flags, 4×move features (7 each — base_power, accuracy, PP_ratio, type_idx, category_idx, disabled, type_effectiveness), 4×boost dims (atk, def, spe, spc — active Pokémon only)
- **Action space:** 9 actions — move 1–4 (indices 0–3), switch slots (indices 4–8)
  - **Voluntary switches disabled during move turns** (see Bug #1 below)
- **Reward (full):** +0.01 per opponent KO, −0.01 per own KO, +0.0001 status inflicted, ±1.0 win/loss, −0.001×turns stalling penalty, clipped to [−1, +1]
- **Reward (simple):** 0 for all non-terminal steps, ±1.0 at episode end only

---

## Bugs Found

### Bug #1 — Voluntary switches in action space (2026-05-19)

**Symptom:** Random agent achieves only 4% win rate vs RandomPlayerAI. PPO training stuck at 5–8% rolling win rate.

**Root cause:** `validActions()` in `pokemon-gym.ts` marked all 9 actions as valid during move turns — 4 moves (indices 0–3) and 5 switch slots (indices 4–8). With a uniform random policy, 55% of actions were switches. `RandomPlayerAI` has `this.move = 1.0` and **never switches voluntarily**, so the gym agent wasted ~55% of its attack turns switching while the opponent always attacked. This creates a catastrophic damage disadvantage (~45% of normal damage output vs 100% taken).

**Fix:** Removed switch actions from `validActions()` during move requests (`isMoveRequest` branch). Force-switches after KOs still work normally via the `isSwitchRequest` branch.

**Verification:** Random agent went from 4% → 47% win rate after fix.

**File changed:** `sim/tools/pokemon-gym.ts` — deleted the switch-listing block inside `isMoveRequest`.

---

### Bug #2 — Model input dim mismatch (2026-05-19)

**Symptom:** All three model files (`ppo_agent.py`, `dqn_agent.py`, `q_agent.py`) had `obs_size=100` defaults from M1. M2 changed the observation to 780 elements.

**Fix:** Updated all defaults to `obs_size=780`. Updated `train.py` files to pass `GymClient(flat_mode=True)` so agents receive (780,) flat vectors.

**Verification:** Smoke tests — all three agents `act()` on `np.zeros(780)` without shape errors.

---

## Architecture Decisions

### Action space: moves only during move turns
After Bug #1, voluntary switches are disabled during normal move turns. The agent can only choose between up to 4 moves. Force-switches (after a KO) still require picking a bench Pokémon. This makes the effective action space 4-dimensional during normal turns, matching the opponent's behavior.

**Trade-off:** Loses strategic switching (a real Pokémon skill). Acceptable for M2/M3 baselines; can be revisited for M4+ with better opponent modeling.

### Observation: structured tokens (M2+)
Flat 100-feature vector (M1) replaced by (12, 69) token array. Each token represents one Pokémon's state. Flattened to (828,) for MLP models; passed as-is (12, 69) for the Transformer (M3).

---

## Training Runs

### Run 1 — MLP PPO baseline (pre-fix, broken env)
**Date:** 2026-05-19  
**Config:** rollout=512, entropy_coef=0.01, ppo_epochs=4, simple_reward=OFF  
**Steps:** 100k  
**Result:** WR(rollout) oscillated 0–0.67 with no trend. Average ~25%. No upward trend at all; degraded after 50k steps.  
**Diagnosis:** Two compounding issues — rollout too small (512 steps = 5–15 episodes, too noisy for stable gradients) AND voluntary switches bug making the env unfair. Could not isolate which was primary.

---

### Run 2 — MLP PPO after rollout fix, before env fix
**Date:** 2026-05-19  
**Config:** rollout=2048, entropy_coef=0.01, ppo_epochs=4, simple_reward=ON  
**Steps:** ~39k (aborted)  
**Result:** WR(rolling-500) stuck at 0.04–0.08. Never improved.  
**Diagnosis:** Voluntary switches bug (Bug #1). The env was giving the gym agent a 4% win rate even with a purely random policy. PPO cannot learn from a broken reward signal.

---

### Run 3 — MLP PPO, fixed env, initial hyperparams
**Date:** 2026-05-19  
**Config:** rollout=2048, entropy_coef=0.01, ppo_epochs=4, simple_reward=ON  
**Steps:** 100k  
**Key numbers:**
| Step | WR(rolling-500) |
|------|----------------|
| 8k | 0.57 (peak) |
| 50k | 0.52 |
| 75k | 0.52 |
| 100k | 0.44 |

**Result:** Peaked at 0.57 early, then steadily degraded to 0.44 by end.  
**Diagnosis:** Policy degradation — too many PPO updates (4 epochs × 32 minibatches = 128 gradient steps) on noisy advantages (only 35–45 episodes per rollout). Low entropy coefficient (0.01) allowed the policy to collapse to near-deterministic quickly, locking in a suboptimal policy.

---

### Run 4 — MLP PPO, fixed entropy + fewer epochs
**Date:** 2026-05-19  
**Config:** rollout=2048, entropy_coef=0.05, ppo_epochs=2, simple_reward=ON  
**Steps:** 100k  
**Key numbers:**
| Step | WR(rolling-500) |
|------|----------------|
| 2k | 0.53 |
| 25k | 0.50 |
| 43k | 0.43 (dip) |
| 73k | 0.53 (recovery) |
| 100k | 0.50 |

**Result:** No catastrophic degradation. Dipped to 0.43 then recovered to 0.53, ended at 0.50. Much more stable than Run 3.  
**Diagnosis:** Increasing entropy coefficient (0.01→0.05) and halving PPO epochs (4→2) prevented policy collapse. But still oscillating — MLP cannot extract consistent signal from flattened token obs.

---

### Run 5 — MLP PPO, full rewards
**Date:** 2026-05-19  
**Config:** rollout=2048, entropy_coef=0.05, ppo_epochs=2, simple_reward=OFF  
**Steps:** 100k  
**Key numbers:**
| Step | WR(rolling-500) |
|------|----------------|
| 2k | 0.50 |
| 8k | 0.53 |
| 50k | 0.50 |
| 75k | 0.49 |
| 100k | 0.49 |

**Result:** Essentially identical to Run 4. Full rewards (per-KO signals) provided no benefit over simple terminal reward for the MLP.  
**Diagnosis:** MLP on flattened 780-dim obs has hit its ceiling at ~50% (random parity). The per-KO rewards add noise without helping the value function learn meaningful state estimates. The structured token format is wasted when flattened — positional relationships between tokens are lost.

---

### Run 6 — Transformer PPO (M3 baseline)
**Date:** 2026-05-19  
**Config:** rollout=2048, entropy_coef=0.05, ppo_epochs=2, simple_reward=OFF, battles=200k  
**Architecture:** TransformerAgent — Linear(65,128) token projection + learnable pos embeddings + 2-layer TransformerEncoder (d_model=128, nhead=4, dim_feedforward=256) + mean pool → policy/value heads  
**Input:** (12, 65) token array directly (flat_mode=False), no flattening  
**Key numbers:**

| Step | WR(rolling-500) |
|------|----------------|
| 25k | 0.48 |
| 50k | 0.49 |
| 75k | 0.51 |
| 100k | 0.50 |
| 150k | 0.49 |
| 200k | **0.52** |

**Result:** Slow upward trend from 0.44→0.52 over 200k steps. More stable than MLP — no catastrophic degradation. Ends 3 points above MLP Run 5 (0.49) but not statistically significant given rollout variance.  
**Diagnosis:** Transformer and MLP both plateau at ~50%. The architecture is not the bottleneck — the observation is. The agent knows each move's type (as a scalar typeIdx/15) but not whether it's super-effective vs the opponent. Without type effectiveness, picking between moves is nearly random. Adding type effectiveness per move is the highest-leverage next improvement.

---

## Current Conclusions

| Hypothesis | Status |
|---|---|
| Structured obs pipeline is correct | ✅ Verified — random agent gets ~47% with the fixed env |
| MLP can learn from structured obs | ⚠️ Partial — peaks at 0.53–0.57 but cannot hold above 0.50 |
| MLP ceiling vs RandomPlayerAI | ~50% (random parity) |
| Full rewards better than simple | ❌ No difference for MLP |
| Larger rollout helps stability | ✅ 512→2048 eliminated wild oscillation |
| Higher entropy prevents collapse | ✅ 0.01→0.05 eliminated degradation in later training |
| Transformer vs MLP (same training budget) | ~tied — Transformer ends slightly higher (0.52 vs 0.49) but within noise |
| Type effectiveness missing from obs | ❌ Critical gap — agent cannot distinguish super-effective from neutral moves |

---

## M3.5 — Type Effectiveness in Observation ✅ IMPLEMENTED (2026-05-19)

**Problem solved:** Move type was encoded as a scalar (typeIdx/15) which the model can't interpret as meaningful. Type *effectiveness* vs the opponent's active type is the primary strategic signal in Pokemon — always use a super-effective move when possible.

**Change applied:** Added 1 effectiveness feature per move slot on the own-active token:
- `type_effectiveness / 4` — normalized: 0 (immune), 0.25 (0.5x resisted), 0.5 (neutral 1x), 1.0 (super-effective 2x)
- Only populated for own active token; bench move slots leave this feature at 0

**Dim change:** TOKEN_DIM 65 → **69**, obs_size 780 → **828**. Each move block: 6 → 7 features.

**Files changed:**
- `sim/tools/feature-extractor.ts` — TOKEN_DIM=69, GEN1_TYPE_CHART, computeEffectiveness(), oppType threading
- `models/ppo/ppo_agent.py` — obs_size default 828
- `models/dqn/dqn_agent.py` — obs_size default 828
- `models/transformer/transformer_agent.py` — token_dim default 69
- All checkpoints invalidated (shape mismatch with prior runs)

---

### Bug #3 — Gen 1 type chart errors (2026-05-20)

**Symptom:** Type effectiveness values were wrong for several Gen 1 matchups.

**Root cause:** Three errors in GEN1_TYPE_CHART: (1) Ice was listed as resisting Ice (0.5×) — that change was made in Gen 2; Gen 1 Ice vs Ice is neutral (1×). (2) Fire was listed as resisting Dragon (0.5×) — also a Gen 2 change; Gen 1 Fire vs Dragon is neutral (1×). (3) Ghost was listed as super-effective vs Ghost (2×) — in Gen 1, Ghost-type moves fail entirely against Ghost-type targets (0×).

**Fix:** Removed `ice: 0.5` from Ice's row, removed `dragon: 0.5` from Fire's row, changed `ghost: 2` to `ghost: 0` in Ghost's row.

**File changed:** `sim/tools/feature-extractor.ts` — GEN1_TYPE_CHART

---

### Bug #4 — Effectiveness normalization was wrong (2026-05-20)

**Symptom:** The old formula `Math.min(1, (e1 * e2) / 4)` produced 0.25 for neutral (1×) and 0.5 for super-effective (2×), compressing all signal into the lower half of [0,1]. Neutral was indistinguishable from not-very-effective in terms of gradient signal.

**Fix:** Replaced with log2-based normalization: `(Math.log2(raw) + 2) / 4`. New mapping: 0.25× → 0.0, 0.5× → 0.25, 1× (neutral) → 0.5, 2× → 0.75, 4× → 1.0. Centers neutral at 0.5 with symmetric spacing in log-space.

Also updated default effectiveness value in `fillPokemonToken` from 0.25 to 0.5 (neutral).

**File changed:** `sim/tools/feature-extractor.ts` — `computeEffectiveness()`, `fillPokemonToken()`

---

## Architecture Decisions (continued)

### Boost encoding added (2026-05-20)

**Previous state:** `BoostData` was tracked in `pokemon-gym.ts` and passed to `extractFeaturesStructured` but never encoded into any token dimensions — silently ignored.

**Fix:** Added 4 dedicated boost dimensions per token: atk, def, spe, spc (indices 69–72). Values normalized as `boost_level / 6` ∈ [−1, +1]. Boosts encoded for own active (Token 0) and opponent active (Token 6) only; bench tokens have 0 for all boost dims.

**TOKEN_DIM:** 69 → 73. Flat obs size: 828 → 876 (12 × 73).

**Downstream constants updated:** `models/ppo/ppo_agent.py` (obs_size 828→876), `models/dqn/dqn_agent.py` (obs_size 828→876), `models/transformer/transformer_agent.py` (token_dim 69→73), `models/gym_bridge.js` (comment 876 floats), `models/gym_client.py` (docstring (12,73)).

**All existing checkpoints are invalidated** — obs shape changed and boost dims now carry real signal. Retraining required.
