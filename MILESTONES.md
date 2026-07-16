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
M4 (needs trained value function for MCTS leaf evaluation), M5 (opponent modeling head) — **both on hold** pending M3.2, which decides whether the transformer is revisited or retired in favor of the M2 MLP-PPO baseline. Direction decided 2026-07-14: run M3.1 (parallel training) → M3.2 (BC→PPO degradation fix) → M3.3 (self-play + opponent pool) → M3.4 (policy ceiling: obs schema v2 + mixed opponents) before touching M4+.

---

## M3.1: Parallel Training Infrastructure ✅ COMPLETE

**Status:** ✅ Complete (2026-07-14)
**Goals:** Remove the serial-training bottleneck. The M2/M3 loops ran one
`gym_bridge.js` subprocess, one battle at a time, with batch-size-1 inference
per step — the GPU sat idle waiting on a single Node event loop. Parallelize
the simulation and batch the inference so long runs stop taking days.

### What Was Built
| File | Contents |
|------|----------|
| `models/vec_gym_client.py` | `VecGymClient(n_envs, structured)` — N `gym_bridge.js` subprocesses (one Node event loop/CPU core each); pipelined step (write all N commands, then read all N responses); auto-reset on done; per-env errors reset in place and surface as `infos[i]["error"]` |
| `models/gym_client.py` | `_send()` split into `_write()`/`_read()` so the vec client can pipeline round trips |
| `models/ppo/trajectory_buffer.py` | `compute_advantages(normalize=...)` + module-level `merge_buffers()` — per-env buffers (GAE never crosses env streams), advantages normalized globally over the combined batch |
| `models/{ppo,transformer}/*_agent.py` | `act_batch()` batched inference; `update()` accepts a merged tensor dict; `device=` override kwarg + `load(path, device=...)` — device deliberately excluded from checkpoint hparams so checkpoints stay portable Mac↔CUDA |
| `models/{ppo,transformer}/train.py` | `--num-envs` (default 8) and `--device` flags; vectorized rollout collection with per-env buffers; checkpointing/`--resume`/LR-annealing unchanged (all keyed off `total_steps`) |
| `models/evaluate.py` | `--num-envs` (default 8) and `--device`; per-env battle quotas keep the count exact; q/dqn fall back to per-env `act()` |

### Device Notes (recorded so this isn't relitigated)
- `_pick_device()` auto-detects CUDA → MPS → CPU. The RTX 3080 machine needs
  no code changes: clone, `./build`, install CUDA PyTorch, same commands.
- The Apple Neural Engine is **not** usable from PyTorch (CoreML-only,
  inference-only). MPS (the GPU) is the right Mac backend and is what's used.
- For small models at small batch sizes, `--device cpu` can beat MPS
  (per-op dispatch overhead); benchmark before long runs.

### Success Criteria
- ✅ Old checkpoints unchanged/compatible: the M2 51% MLP checkpoint evaluates
  at 49% (49/100) through the parallel path
- ✅ Single-env path still works (`--num-envs 1`, same code path)
- ✅ Training throughput ≥ 5x at 8 envs vs serial (see benchmark table in
  `docs/ML-TRAINING.md`)
- ✅ Evaluation: 100 transformer battles 20.2s → 5.6s at 8 envs (3.6x)

### Unblocks
M3.2 (fast retraining runs), M3.3 (parallel self-play)

---

## M3.2: BC→PPO Degradation Fix ✅ COMPLETE — TRANSFORMER RETIRED

**Status:** ✅ Complete (2026-07-14). The fixes worked on the diagnosed
mechanism, but the transformer still does not beat the MLP baseline — **per
this milestone's own criterion, the transformer is retired and M4+ proceeds
on the MLP-PPO architecture.**

### Decision Run (5M steps, warm-started, value warmup 200k, BC anchor 0.05, real masks)

Full 20-checkpoint sweep (150 battles each) + 500-battle confirmations:

| Phase | Result |
|---|---|
| 250k–3.5M steps | **44–55%** vs Random, peak 55% @ 500k — holds at baseline parity for 3.5M steps with no collapse (old runs were at 25–35% by 2M and had cratered to 0–13% in run 1) |
| 3.75M–5M steps | Decays 35% → 29% → 25% as the KL-anchor coefficient anneals toward zero |
| Best checkpoint, confirmed | **53% (263/500)** vs Random — statistical parity with the MLP's 51% (254/500), not a win (`m32/transformer_step_500000.pt`) |
| Best checkpoint vs DamageFirstAI | **39% (77/200)** — clearly behind the MLP's 51% (101/200) |

### What this establishes
- The **diagnosis was right**: with the value-head warmup and BC anchor in
  place, PPO no longer destroys the BC policy — the model holds at its
  BC-level plateau for 3.5M steps instead of peaking immediately and decaying.
- The **anchor is load-bearing**: decay resumes almost exactly as the anchor
  coefficient anneals away (~3.75M steps onward). Unconstrained PPO on this
  architecture erodes the BC policy at any LR; it does not improve it.
- The transformer's ceiling remains the BC policy itself (~50–55% vs Random),
  and it generalizes worse than the MLP to a stronger opponent (39% vs 51%
  against DamageFirstAI). Three generations of runs (M3 × 2, M3.2 × 1) agree.

### Decision
**Retire the transformer as the policy architecture. M4 (MCTS) and M5
(opponent modeling) build on the M2 MLP-PPO baseline** (`models/ppo/`,
structured obs). The transformer remains useful as a BC study; revisit only
with a fundamentally different recipe (e.g. much larger BC dataset, or
AlphaZero-style value targets from M4 self-play rather than PPO).
**Goals:** Treat the actual failure mode M3 uncovered. Hypothesis: BC
pretraining (`models/bc_pretrain.py`) trains the **policy head only** — the
value head is random at warm-start, and PPO's value loss backpropagates
through the shared transformer encoder, scrambling the BC-learned features to
fit the value function. This matches the observed pattern exactly (peaks
at/near the BC starting point, then decays). None of the standard mitigations
were tried in M3.

### What Was Built (all three fixes, verified)
1. **Real action masks in PPO updates** — store per-step `valid_mask` in
   `TrajectoryBuffer` and use it in `evaluate_actions()` during `update()`.
   Today updates use an all-ones mask, so entropy and log-probs are computed
   over illegal actions. Applies to both agents.
2. **Value-head warmup** — `--value-warmup-steps N`: freeze `embed`/`encoder`/
   `policy_head` (`requires_grad=False`) for the first N steps and train only
   the value head, so the value function fits the BC policy *before* full PPO
   gradients flow through the encoder.
3. **KL-anchor to BC** — `--bc-anchor <checkpoint> --bc-anchor-coef 0.05`:
   keep a frozen copy of the BC policy; add `coef × KL(π_θ ‖ π_BC)` to the
   loss so PPO can't drift far from human-cloned play. Anneal the coefficient
   with the existing LR schedule.
4. **Decision run** — retrain warm-started with all three fixes (fast now, per
   M3.1), sweep checkpoints, compare against the 51% MLP baseline.

### Success Criteria
- ❌ Warm-started transformer PPO **improves** on its BC starting point instead
  of decaying, and beats the 51% MLP-PPO baseline — best confirmed 53%
  (parity), 39% vs DamageFirstAI (behind), late-run decay returns as the
  anchor anneals
- ✅ "If it still fails with these fixes: the transformer is retired, and M4+
  proceeds on the MLP-PPO architecture" — **decided: MLP-PPO is the project
  architecture going forward**

### Unblocks
M4/M5 (on the MLP-PPO architecture); M3.3 comparison runs

---

## M3.3: Self-Play + Opponent Pool ✅ COMPLETE — MIXED RESULT

**Status:** ✅ Complete (2026-07-14). Training run, fixed-opponent evals, and
head-to-head all executed. Verdict: self-play fixed training *stability*
(first run whose strength improves over training) and produced a rough peer
of the M2 agent — slightly better vs Random and head-to-head, slightly worse
vs DamageFirst — but not a decisively stronger one. The opponent-distribution
fix moves to M3.4.
**Goals:** Fix the degenerate-opponent problem. Everything before this trained
and evaluated against `RandomPlayerAI`, which never voluntarily switches
(`move: 1.0`) — a weak opponent giving a weak, exploitable learning signal.
This is the chess-engine-style ingredient the pipeline was missing: AlphaZero's
strength comes from self-play against improving copies, not a fixed random
opponent. (MCTS, the other chess ingredient, is already planned as M4.)

### What Was Built
1. **Heuristic opponent:** `sim/tools/damage-first-ai.ts` (`DamageFirstAI`,
   always picks the highest-base-power legal move; ported from the
   `simulate.js` reference). Wired as `opponent: 'damagefirst'` through
   `PokemonGymEnv`, `gym_bridge.js --opponent`, `GymClient`/`VecGymClient`,
   both trainers, and `evaluate.py --opponent`. Baseline recorded: the M2
   MLP checkpoint scores **51% (101/200)** vs DamageFirstAI.
2. **Self-play (dual-seat mode):** `PokemonGymEnv` `opponent: 'self'` +
   `resetDual()`/`stepDual()` — each call advances to the next decision point
   and returns both seats' obs/mask/needsAction (force-switches are
   single-seat decision points; request freshness is counter-tracked). The
   reveal tracker runs for both sides, so each seat sees only revealed
   opponent info. Bridge `--selfplay` dual protocol; `VecGymClient`
   `reset_all_dual()`/`step_dual()` (pipelined, auto-reset).
3. **Trainer integration (both trainers):** `--opponent selfplay` samples one
   frozen opponent per rollout from `--selfplay-pool` (default: the run's own
   checkpoint dir) — 50% newest / 50% uniform league mix; a frozen copy of
   the current policy until the first checkpoint exists. Training remains
   p1-only; rewards from opponent-only decision points accumulate into p1's
   open transition (pending-transition collection).

### Comparison Run (2026-07-14): 5M-step MLP-PPO self-play

`--opponent selfplay`, own-checkpoint pool, 8 envs, checkpoints every 250k.
20-checkpoint sweep (150 battles each vs Random) rose from 42% @ 250k into a
stable 47–61% band with **no collapse — the first run in this project whose
eval strength trends up over training instead of decaying.** Top candidates
confirmed at full battle counts:

| Checkpoint | vs Random (500) | vs DamageFirst (200) |
|---|---|---|
| M2 baseline (trained vs Random) | 51% (254/500) | 51% (101/200) |
| selfplay 2.5M | 52% (259/500) | 43% (86/200) |
| selfplay 4.25M | 53% (267/500) | 45% (90/200) |
| **selfplay 4.75M** | **57% (287/500)** | 46% (91/200) |
| selfplay 5M final | 51% (255/500) | 46% (91/200) |

Best checkpoint: `models/ppo/checkpoints/selfplay/ppo_step_4750059.pt`.

**Head-to-head** (`evaluate.py --vs-checkpoint`, dual-seat, seat-balanced):
self-play 4.75M vs the M2 checkpoint — **51% as p1 (254/500), 54% as p2
(270/500), 52.4% combined (524/1000)**. A slight edge, inside the ±3.1pp
95% CI — statistical parity, not a decisive win.

### Success Criteria
- ✅ Dual-seat battles run to completion with no illegal moves or hangs
  (TS + Python + trainer smokes, both trainers)
- 🟨 Agent trained against the pool beats the fixed-opponent-trained agent
  head-to-head — **marginal**: 52.4% over 1000 seat-balanced battles, within
  noise of 50%. The DamageFirstAI half of the criterion is **not met**:
  43–46% across all top self-play checkpoints vs the M2 agent's 51% (each
  ~±7pp at 200 battles — no self-play checkpoint transfers *better* to the
  held-out heuristic, and the trend is mildly worse)
- ✅ Win rate vs RandomPlayerAI does not regress: best 57% (287/500) vs the
  baseline's 51% (254/500), ~2σ above; final checkpoint 51% — floor held

### Reading
Self-play fixed training *stability* (monotone-ish improvement, no
collapse/decay — every fixed-opponent and transformer run eroded from its
peak) and modestly improved play vs Random, but did **not** transfer to the
held-out DamageFirst heuristic. Plausible mechanism: the pool is seeded from
an untrained policy and grows only its own descendants, so early league play
rewards exploiting weak self-like opponents rather than robust play against
pure attackers. The fix — seeding the pool with the M2 checkpoint and mixing
heuristic rollouts into training — is Part B of **M3.4**, along with the
observation-schema upgrade the M3.2 transformer retirement unlocked.

### Unblocks
Meaningful M4/M5 evaluation opponents; higher-quality training signal for
whichever architecture M3.2 selects

---

## M3.4: Raise the Policy Ceiling ✅ COMPLETE — NEGATIVE RESULT

**Status:** ✅ Complete (2026-07-15). **Neither lever raised the ceiling.** The
schema-v2 + mixed-opponent run trains stably (42–62% sweep band, no collapse)
but confirms at **54% vs Random / 46% vs DamageFirst / 48% head-to-head vs the
M3.3 best** — an M2/M3.3 peer, not an improvement. All three pre-registered
criteria unmet. See Results below.

### Decision Run (2026-07-15)

`--obs-v2 --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2"`, pool
seeded with the M2 baseline + M3.3 best, 8 envs. The run was externally killed
at 4.77M/5M steps (no crash; last checkpoint 4.75M — the last 5% doesn't
affect the conclusion; 19 periodic checkpoints cover the trajectory).

**19-checkpoint sweep (150 battles each vs Random):** stable 42–62% band, no
collapse; apparent peaks 62% @ 3.25M and 61% @ 2.25M.

**Full-battle confirmations** (peaks regress to the mean — the 60%+ sweep
readings were 150-battle noise, ±8pp):

| Checkpoint | vs Random (500) | vs DamageFirst (200) |
|---|---|---|
| M2 baseline (reference) | 51% (254/500) | 51% (101/200) |
| M3.3 best (reference) | 57% (287/500) | 46% (91/200) |
| v2 2.0M | 51% (256/500) | 42% (85/200) |
| **v2 2.25M (best combined)** | **54% (272/500)** | **46% (92/200)** |
| v2 3.25M | 53% (266/500) | 45% (89/200) |
| v2 4.0M | 54% (269/500) | 41% (81/200) |

**Head-to-head** (v2 2.25M vs M3.3 best, `--vs-checkpoint` cross-schema,
seat-balanced): 45% as p1 (227/500), 51% as p2 (253/500) → **48.0% combined
(480/1000)** — parity-to-slightly-worse, inside the ±3.1pp CI.

### Success Criteria — ❌ NOT MET
- ❌ ≥ 65% vs RandomPlayerAI (500 battles) — best confirmed **54%**
- ❌ ≥ 60% vs DamageFirstAI (200+ battles) — best **46%**, still below the
  M2 agent's 51%; the training signal again failed to transfer to the
  held-out heuristic
- ❌ Beats the M3.3 best head-to-head (≥ 55%, seat-balanced) — **48.0%**

### Reading

Richer observations (boosts/volatiles) and a fixed opponent distribution
(seeded league + heuristic/random mixing) were the two hypothesized
bottlenecks, and fixing both moved nothing outside noise. Four independent
5M-step-class runs (M2 fixed-opponent, M3.2 transformer, M3.3 self-play,
M3.4 v2+mix) now all land in the same 51–57%-vs-Random band. The remaining
suspects are more fundamental: MLP capacity at 128 hidden units, PPO sample
efficiency on this reward, and gen1randombattle team-luck variance capping
how far any policy can get without lookahead. That last one is exactly what
M4 (MCTS) attacks — and M4's criteria were already recalibrated (2026-07-14)
to be relative to the base policy, so this result does not block it.

### Recommendation for M4

Proceed on the **M3.3 best checkpoint**
(`models/ppo/checkpoints/selfplay/ppo_step_4750059.pt`, v1 schema): it has
the strongest confirmed vs-Random number (57%) and won the h2h marginally.
The v2 2.25M checkpoint (`models/ppo/checkpoints/v2/ppo_step_2250032.pt`) is
a statistical peer whose richer observation may matter more once a search is
attached — worth an A/B once the MCTS harness exists.

### What Was Built (2026-07-15)
- **Part A — schema v2:** `TOKEN_DIM_V2 = 77` in `sim/tools/feature-extractor.ts`.
  12 dims appended per token (dims 0–64 stay byte-identical to v1): 7 boost
  stages (atk/def/spe/spa/accuracy/evasion/spd — spd captures gen1 Amnesia's
  spa+spd pair), each stage/6 in [-1, 1]; Reflect / Light Screen / Substitute /
  Leech Seed flags; toxic counter (poison ticks, min(n,16)/16). Non-zero only
  on the two active tokens — gen1 resets all of it on switch. Tracked in
  `pokemon-gym.ts` (`_processVolatileLine`) from public log lines only:
  `|-boost|/|-unboost|/|-clearallboost|` (Haze), `|-start|/|-end|`
  (screens/Sub/Leech Seed), `|-status|/|-curestatus|/|-damage ... [from] psn`
  (toxic counter), reset on `|switch|/|drag|/|faint|`. New obsMode
  `'structured-v2'` → bridge `--obs-v2` → `GymClient(obs_v2=True)` → trainers
  and `evaluate.py` `--obs-v2` (924-dim flat input, checkpoints in
  `models/ppo/checkpoints/v2/`).
- **Part B — mixed opponents:** `--opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2"`
  on `models/ppo/train.py` samples one family per rollout; the bridge accepts
  a per-reset opponent override so envs switch family at rollout boundaries
  without respawning (in-flight episodes abandoned + bootstrapped — the same
  truncation PPO applies at every rollout end). Pool seeded with the M2
  baseline and M3.3 best as `ppo_step_0_seed_{m2,m33best}.pt` (step 0 ⇒ never
  "newest"). **Cross-schema play:** since v2 tokens are v1-prefixed,
  `gym_client.slice_structured_obs()` hands v1 checkpoints their native view
  of v2 observations — used for pool opponents in training and for
  `evaluate.py --vs-checkpoint` head-to-head (v2 agent vs v1 agent directly).
- Verified: 23/23 gym tests (new: v2 shape, v1-prefix byte-equality, volatile
  dims, full-battle v2 range/stability); bridge smokes (v2 single/dual seat,
  live opponent switching both directions, v1 path regression); 4k-step
  mixed-opponent training smoke with a v1 seed acting cross-schema; all four
  eval paths; v1 self-play trainer regression.
**Goals:** Close the gap between where the policy is and where M4 assumed it
would be. The best policy to date is **57% vs Random / 46% vs DamageFirst**;
M4's original criteria assumed ≥90% vs Random. MCTS multiplies the quality of
the policy/value net it searches with — layering it on a ~51–57% policy
amplifies mediocrity. Two levers, both newly unlocked:

### Part A: Observation schema v2 (boosts et al.)

The M2 schema was frozen at (12, 65) **solely** so the BC checkpoint's input
projection stayed loadable. The M3.2 decision retired the transformer — the
MLP-PPO path has no BC dependency, so the freeze is moot. The policy currently
cannot see stat boosts, which decide gen1 games (Amnesia, Swords Dance,
Agility, paralysis+speed interactions).

- Extend the active-Pokémon tokens with a boost tracker
  (`|-boost|`/`|-unboost|`/`|-clearallboost|` lines, both sides): 7 boost
  stages (atk/def/spe/spc/accuracy/evasion + a spare), normalized to
  [-1, 1] (stage/6)
- Candidates to include while the schema is open (cheap, same tracker
  pattern): Reflect/Light Screen flags, Substitute flag, Leech Seed flag,
  toxic counter. PP tracking is *own-side only* from the request JSON —
  include if trivial
- This is a deliberate schema-version bump: `TOKEN_DIM` 65 → 65+K, new
  checkpoint dir (`models/ppo/checkpoints/v2/`), old checkpoints incompatible
  by design. `extractFeaturesStructured()` keeps a v1 mode only if the tests
  need it; the flat 100-dim path is untouched
- Update `test/tools/gym.test.js` shape/consistency coverage

### Part B: Mixed-opponent training

M3.3's league contained only the run's own descendants, and the result didn't
transfer to DamageFirst (43–46% vs the M2 agent's 51%). Fix the opponent
distribution, not just the algorithm:

- **Seed the pool:** copy the M2 checkpoint (and the M3.3 best) into
  `--selfplay-pool` — already supported, zero code
- **`--opponent-mix`:** per-rollout opponent sampling across
  `selfplay/damagefirst/random` with configurable weights (e.g. 50/30/20),
  so the learner never overfits to one opponent family. Damagefirst/random
  rollouts reuse the existing single-seat path; selfplay rollouts the dual
  path — the trainer already has both loops

### Training + evaluation plan

One 5M-step run with schema v2 + seeded pool + opponent mix, 20-checkpoint
sweep vs Random, confirmations vs Random (500) / DamageFirst (200) /
head-to-head vs the M3.3 best (`evaluate.py --vs-checkpoint`, both seat
orders).

### Unblocks
M4 starts from a policy/value net actually worth searching with; M4's
success criteria re-anchored to this milestone's result

---

## M4: MCTS Integration ✅ COMPLETE — POSITIVE RESULT

**Status:** ✅ Complete (2026-07-15). **Search works — the first intervention
since M2 that clearly improves play.** MCTS (100 sims, 4 determinizations)
over the M3.3 best checkpoint beats the same checkpoint without search
**60.2% (602/1000)** seat-balanced head-to-head, improves DamageFirst
transfer to **56% (113/200)** vs the raw policy's 46% — the number no
M3.x training intervention could move — and runs at **84–88ms mean /
~102ms p95** per decision. 3 of 4 pre-registered performance criteria met;
vs-Random narrowly missed (+8.6pp vs the +10pp bar). Full results below.
**Architecture note (2026-07-14):** per the M3.2 decision, M4 builds on the
**MLP-PPO** policy/value network (`models/ppo/`, structured obs), not the
transformer. References to the transformer below predate that decision; the
MCTS design is architecture-agnostic (it needs P(a|s) and V(s), which the MLP
provides).

### Results (2026-07-15) — 100 sims / 4 determinizations, base = M3.3 best (`ppo_step_4750059.pt`), CPU

| Test | MCTS | Raw checkpoint | Criterion | Verdict |
|---|---|---|---|---|
| vs RandomPlayerAI (500) | **66.0% (330/500)** | 57.4% (287/500) | ≥ +10pp | ❌ +8.6pp — narrow miss |
| vs DamageFirstAI (200) | **56.5% (113/200)** | 45.5% (91/200) | ≥ +5pp | ✅ +11.0pp |
| Head-to-head vs raw, seat p1 (500) | **60.4% (302/500)** | 50% by construction | ≥ +5pp seat-balanced | ✅ |
| Head-to-head vs raw, seat p2 (500) | **60.0% (300/500)** | | | ✅ **60.2% (602/1000) combined** |
| Decision latency | 84–88ms mean, ~102ms p95 | — | < 500ms | ✅ (one 909ms outlier across ~30k decisions) |
| Determinizer team legality | tested (species clause, valid sets, full HP) | — | valid gen1randombattle teams | ✅ |

Artifacts: `models/mcts/results/{vs_random,vs_damagefirst,h2h_seat_p1,h2h_seat_p2,battery}.log`.

### Reading

- The head-to-head is the cleanest causal measure — same network, same
  battles, only the search differs — and it's a decisive 60/40 in both seat
  orientations. Combined with the DamageFirst gain, this confirms the M3.4
  hypothesis that *lookahead*, not observations or opponent distribution,
  was the binding constraint on this policy.
- The vs-Random criterion missed by 1.4pp (5 battles). Plausible reading:
  Random's blunders make many positions winnable by the prior alone, so
  search adds less there than against coherent opposition — the gain is
  larger exactly where opponents are stronger (DamageFirst +11pp, h2h +10pp),
  which is the direction that matters.
- Untuned knobs left on the table: sim budget (100), c_puct (1.5),
  determinization count (4), and the value head was trained under γ=0.99
  shaped rewards while backups are undiscounted. A budget/constant sweep is
  the cheapest path to closing the vs-Random gap; an A/B on the v2 2.25M
  checkpoint (richer obs) is the other pre-planned follow-up.

### Post-M4 knob sweep (2026-07-15) — new operating point: sims=100, c_puct=0.5, det=1

Two-stage sweep (OFAT then combinations, 200 battles/cell) + 500-battle
confirmation, all on the M4 base checkpoint. Two knobs beat their defaults
on both opponents, and the effects stack — both point the same way:
**concentrate the search**. Fewer determinizations at fixed total `--sims`
means one deeper tree instead of a shallow ensemble over opponent-team
samples (det=1 also fastest, ~85ms/move); lower c_puct trusts the policy
prior more (2.5 clearly hurts). Raising sims to 400 helped vs Random but
not DamageFirst at 4× latency — the least attractive lever.

Confirmed at 500 battles (c_puct=0.5/det=1 beat c_puct=1.0/det=1 on both):

| Test | Tuned MCTS | M4 default MCTS | Raw checkpoint |
|---|---|---|---|
| vs RandomPlayerAI (500) | **81.2% (406/500)** | 66.0% | 57.4% |
| vs DamageFirstAI (500) | **67.2% (336/500)** | 56.5% (200 battles) | 45.5% |

The M4 vs-Random criterion (≥ +10pp over raw), narrowly missed at the
defaults, is cleared decisively at the tuned point: **+23.8pp** (and
+21.7pp vs DamageFirst). The gap was untuned defaults, as suspected.
Defaults updated in `MCTSAgent` and `evaluate.py`; logs in
`models/mcts/results/sweep/`.

**v2-checkpoint A/B (2026-07-16, pre-planned in M3.4):** the same tuned
battery over `v2/ppo_step_2250032.pt` (obs schema v2): **82.6% (413/500)
vs Random, 70.2% (351/500) vs DamageFirst** — nominally ahead of v1 on
both (81.2%/67.2%) but within noise (~±4pp at 500 battles; DF gap ≈ 1σ).
Notable direction reversal: raw v2 was *behind* raw v1 (54%R/46%DF vs
57.4%/45.5%), yet with search attached it's at least on par — consistent
with richer obs mattering more under lookahead, though not statistically
established. Verdict: **a tie; either checkpoint is a valid M5/M6 base.**
v1 (`selfplay/ppo_step_4750059.pt`) stays the default to avoid churn; v2
is a live option for M5, where the obs schema gets revisited anyway.
Logs: `models/mcts/results/sweep/v2ab_*.log`.

### What Was Built (2026-07-15, committed `9da4d273a`)

- **Forward model** `sim/tools/battle-sim.ts` (`BattleSim`): clones the live
  gym battle via `State.serializeBattle`/`deserializeBattle` (0.8ms
  round-trip) plus the gym's tracker state (`ObservationTrackers`, extracted
  from `PokemonGymEnv`; new `snapshot()` API), steps both seats directly on
  the clone with gym-identical obs/mask/reward semantics, `fork()` for tree
  branching. **Engine gotcha fixed:** locked states (sleep, recharge,
  multi-turn moves) auto-complete a seat's choice (`side.isChoiceDone()`);
  submitting for such a seat lands one decision point ahead and desyncs the
  battle — `needsAction` now consults it (30/30 random playouts clean).
- **Determinizer** (Node-side, inside `BattleSim` — deviation from the
  original `models/mcts/determinizer.py` spec since team generation and
  legality live in Node): replaces the searcher's opponent's *unrevealed*
  slots with sets sampled from the gen1randombattle generator (seeded,
  species-clause-safe); `perspective: 'p1'|'p2'` so a p2-seated searcher
  determinizes p1. Documented approximation: revealed Pokémon keep their
  true full movesets — only unrevealed slots are resampled.
- **Bridge sim protocol**: `sim_clone`/`sim_step`/`sim_fork`/`sim_free`/
  `sim_free_all` in `gym_bridge.js` + `GymClient` wrappers (0.36ms/step over
  the wire); sims cleared on env reset.
- **`models/mcts/mcts_agent.py`**: root-parallel determinized PUCT — policy
  head as prior, opponent modeled by sampling the same policy on its
  reveal-tracked obs, value head + shaped path rewards at leaves, visit
  counts summed across determinizations, argmax-visits with Q tiebreak;
  `seat='p1'|'p2'`; raw-policy fallback for locked-choice roots.
- **`evaluate.py --model mcts`** with `--sims/--determinizations/--c-puct/`
  `--no-determinize/--mcts-seat`; h2h via `--vs-checkpoint` in either seat.
- **Tests**: `test/tools/battle-sim.test.js` (13) — obs parity with the live
  gym (v1+v2 byte-equality), determinization invariants both perspectives,
  illegal-free playouts, fork consistency; gym suite unaffected.
**Goals:** Layer UCT-based Monte Carlo Tree Search on top of the trained PPO
policy. Use the learned value network for leaf evaluation instead of random
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

### Success Criteria — 4 of 5 MET (2026-07-15)
Recalibrated 2026-07-14 — the original targets assumed a ≥90%-vs-Random base
policy that never materialized; these anchor to whatever M3.4 produces.
- ✅ MCTS(N=100) + MLP value beats the raw MLP-PPO policy (same checkpoint,
  no search) by ≥ 5pp win rate vs DamageFirstAI — **+11.0pp (56.5% vs 45.5%)**
- ✅ MCTS beats the raw policy head-to-head (`evaluate.py --vs-checkpoint`-style
  dual-seat, seat-balanced) by ≥ 5pp — **+10.2pp (60.2% over 1000 battles)**
- ✅ Decision latency < 500ms per move (100 sims, gen1 is fast) — **84–88ms mean**
- ❌ Win rate vs RandomPlayerAI ≥ base policy + 10pp — **+8.6pp (66.0% vs
  57.4%), 5 battles short of the bar; see Reading**
- ✅ Determinizer generates valid gen1randombattle-legal teams

### Unblocks
Self-play with MCTS policy (generates higher-quality training data for future
fine-tuning); M5 (opponent modeling — a better opponent model plugs directly
into the search's opponent-action sampler); M6 (the live bot should ship with
search, it's 60/40 better than the raw policy at 88ms/move)

---

## M5: Opponent Modeling Head ⬜ AFTER M4

**Status:** ⬜ Not Started
**Architecture note (2026-07-14):** per the M3.2 decision, the auxiliary head
attaches to the MLP-PPO trunk's 128-dim features, not the transformer's.
**Goals:** Add an auxiliary head to the policy network that predicts the opponent's
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
| M3: Transformer + PPO | ✅ | Negative result — transformer (peak 46%) never beat the MLP baseline (51%) | M3.1–M3.3 |
| M3.1: Parallel Training | ✅ | Vectorized envs + batched inference (`--num-envs`) | M3.2, M3.3 |
| M3.2: BC→PPO Fix | ✅ | Fixes verified; transformer still ≤ baseline → **retired; M4+ proceeds on MLP-PPO** | M4, M5 |
| M3.3: Self-Play + Opponents | ✅ | Self-play fixed training stability; peer of M2 (52.4% h2h), no DamageFirst transfer | M3.4 |
| M3.4: Policy Ceiling | ✅ | Negative result — schema v2 + opponent mix trains stably but confirms as an M2/M3.3 peer (54%/46%/48% h2h) | M4, M5 |
| M4: MCTS | ✅ | **Positive result** — determinized UCT beats the raw policy 60.2% h2h, +11pp vs DamageFirst, 88ms/move | M5, M6 |
| M5: Opponent Modeling | ⬜ | Opp-prediction auxiliary head | M6 |
| M6: Server Integration | ⬜ | Live ladder bot | — |
