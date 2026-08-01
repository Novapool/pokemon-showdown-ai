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

## M5: Opponent Modeling Head ✅ COMPLETE — THESIS NEGATIVE, NEW-BEST SIDE FINDING

**Status:** ✅ Complete (2026-07-16). **The thesis (C3) failed — sampling the
opponent's action from the trained prediction head adds nothing over the
existing policy sampler under tuned MCTS (−2.2pp vs DamageFirst, statistical
parity). But the run produced the project's best agent to date as a side
finding: the M5 checkpoint under standard policy-sampler MCTS confirms at
72.6% vs DamageFirst / 86.0% vs Random (500 each) — nominally ahead of the
prior best (v2: 70.2%/82.6%) on both, within ~1σ.** 3 of 5 pre-registered
criteria met (C2, C4, C5); C1 missed its 40% bar at 35.8%; C3 missed. Full
results below.

### Results (2026-07-16)

**Decision run:** 5M steps as pre-registered (externally stopped at 2.93M and
resumed via a newly added `--resume` on `models/ppo/train.py`, commit
`589d8bf1e`; no gap in the checkpoint sequence). Trained stably: label
coverage 78–96%, no collapse, sweep band 40–60% rising into a 50–59% plateau.
Neither pre-registered contingency fired (band never < 45% after warmup;
accuracy never < 25%).

**Sweep (21 checkpoints @ ~250k spacing, 150 battles + opp accuracy each):**
40% @ 250k → 50–59% band from 750k onward; opp accuracy vs Random flat at
~25–26% (chance-capped by Random's uniformity, as pre-registered — context
only). Full table: `models/ppo/checkpoints/opp/sweep_results.txt`.

**Raw confirmations (top 4):**

| Checkpoint | vs Random (500) | vs DamageFirst (200) | Opp acc vs DF |
|---|---|---|---|
| v2 control (M3.4, reference) | 54% (272/500) | 46% (92/200) | — |
| opp 1.25M | 53% (266/500) | 39% (77/200) | 32.1% |
| opp 2.5M | 56% (278/500) | 41% (81/200) | 34.2% |
| **opp 3.0M** (C2-clean) | 55% (273/500) | **44% (88/200)** | 29.8% |
| **opp 5.0M final** (A/B base) | **57% (285/500)** | 41.5% (83/200) | **35.8%** |

**MCTS sampler A/B (primary test):** tuned search (sims=100, c_puct=0.5,
det=1, CPU) over `opp/ppo_step_5000001_final.pt`, only the opponent sampler
differing between arms:

| Arm | vs DamageFirst (500) | vs Random (500) | Latency mean/p95 |
|---|---|---|---|
| head sampler | 70.4% (352/500) | 85.8% (429/500) | **57ms / ~100ms** |
| policy sampler | **72.6% (363/500)** | 86.0% (430/500) | 59–64ms / ~105ms |
| *prior best (v2 + policy MCTS, reference)* | *70.2% (351/500)* | *82.6% (413/500)* | *~85ms* |

Logs: `models/mcts/results/m5_ab_{head,policy}_{damagefirst,random}.log`.

### Reading

- **The head learns real signal but sampling from it doesn't help search.**
  Accuracy is meaningfully above the ~25% uniform reference vs DamageFirst
  (30–36% across checkpoints), yet the head-sampler arm is at parity-to-worse.
  Plausible mechanisms: (a) the head is trained on the *mixture* opponent
  distribution (selfplay/damagefirst/random), so against any specific opponent
  it's miscalibrated — an accurate-on-average model can be worse inside search
  than the policy sampler's implicit "opponent plays like us" assumption,
  which is adversarially robust; (b) ~33% top-1 accuracy may simply be below
  the threshold where a learned opponent model beats a strong default; (c) the
  head sees only the searcher's obs, so its distribution over unrevealed
  opponent options is noise by construction (the documented label
  approximation).
- **The aux loss appears to have helped the *policy*, not the sampler.** The
  M5 checkpoint is the fifth 5M-step-class run to land in the 51–57% raw band,
  but under search it posts the best numbers yet (72.6%/86.0%). Consistent
  with the M5 plan's secondary hypothesis (representation shaping): the trunk
  was forced to encode opponent-predictive state, and search exploits that
  even though raw play doesn't. Within ~1σ of the v2 control's search numbers,
  so — like the v1/v2 A/B — a nominal-not-statistical win, but it is nominally
  ahead on both opponents.
- **Latency win is real:** the head sampler reuses the searcher's forward
  pass (57ms vs 59–64ms mean) — kept as the implementation even though the
  sampler itself is retired as a default.

### Success Criteria — 3 of 5 MET (pre-registered 2026-07-16)
- ❌ **C1 (head learns):** ≥ 40% top-1 vs DamageFirstAI — best **35.8%**
  (5.0M, 2000/5593 unmasked). Above the 25% label-bug floor (labels were
  oracle-verified at build time: 0 mismatches vs actual submitted choices),
  so the pre-registered contingency did not fire. Reading: the 40% bar
  overestimated predictability under partial observability (unrevealed
  movesets; bench-order switch labels).
- ✅ **C2 (no raw regression — hard gate):** opp 3.0M confirms 55% vs Random
  (bar ≥50%) and 44% vs DamageFirst (bar ≥42%).
- ❌ **C3 (primary — search integration):** head-sampler vs policy-sampler
  ≥ +3pp vs DamageFirstAI — **−2.2pp (70.4% vs 72.6%)**, inside the ~±2.8pp
  1σ band for the difference; parity at best.
- ✅ **C4 (non-inferiority to standing best):** head-sampler MCTS 70.4% vs
  DamageFirst ≥ 70% bar (and the policy-sampler arm at 72.6% beats the prior
  best outright).
- ✅ **C5 (latency):** head 57ms mean ≤ policy 59–64ms; p95 ~100ms ≪ 150ms
  ceiling.

### Decision

**Retire the head *sampler* as a default (`--opp-sampler policy` stays the
default; the head mode remains available for future work). The opp head
itself stays in the training recipe** — it costs nothing, produced no
regression, and the search-amplified result is the project's best.
**New best agent: tuned MCTS (policy sampler) over
`models/ppo/checkpoints/opp/ppo_step_5000001_final.pt`** — 72.6% vs
DamageFirst / 86.0% vs Random, a statistical peer of the v2-control config
but nominally ahead on both opponents. M6 should ship this checkpoint (or
A/B it against the v2 control on the ladder if cheap).
**Architecture note:** the original M5 section predated the M3.2 decision and
referenced the retired transformer stack. Everything below targets the
**MLP-PPO** stack (`models/ppo/ppo_agent.py`, 128-dim shared trunk).

**Goals:** Add an auxiliary opponent-action-prediction head to the MLP-PPO
trunk, trained with a multi-task loss (PPO + λ·CE on the opponent's actual
resolved action — free supervised signal, observable from the omniscient
stream in `sim/tools/pokemon-gym.ts`). Two payoffs, in order of expected value:

1. **Search integration (primary).** MCTS's opponent-action sampler
   (`models/mcts/mcts_agent.py::_sample_opponent_action`) currently evaluates
   the *base policy* on the opponent's reveal-tracked obs — i.e., it assumes
   the opponent plays like us. A head trained on the opponent's *actual*
   action distribution replaces that assumption with data.
2. **Representation shaping (secondary).** The aux loss forces the shared
   trunk to encode opponent-predictive state. Given that four independent
   5M-step runs (fixed-opponent, transformer, self-play, v2+mix) all landed
   in the same 51–57%-vs-Random band, a raw-policy gain is *not* expected —
   no-regression is the gate; the search criterion is where M5 wins or loses.

### Design Decisions (made 2026-07-16)

**Base config: obs schema v2** (`--obs-v2`, 77-dim tokens with
boosts/volatiles), fresh 5M-step run with the M3.4 opponent-mix recipe
(`selfplay=0.5,damagefirst=0.3,random=0.2`). Rationale:
- The post-M4 v1-vs-v2 A/B under tuned MCTS was a tie with v2 nominally ahead
  on both opponents (82.6%/70.2% vs 81.2%/67.2%) — choosing v2 costs nothing.
- The prediction target specifically benefits from v2's extra state: boost
  stages, screens, Substitute, and the toxic counter are direct predictors of
  what the opponent does next. Training an opponent-prediction head on a
  trunk that is blind to volatiles caps head accuracy for no reason.
- The M3.4 v2 run (same schema, same opponent-mix recipe, no head) becomes
  the natural λ=0 control: `models/ppo/checkpoints/v2/ppo_step_2250032.pt`
  (raw 54%R/46%DF; tuned-MCTS 82.6%R/70.2%DF). v1
  (`selfplay/ppo_step_4750059.pt`) remains the repo-wide default elsewhere;
  M5 changes no M4 defaults.
- **Fresh run, not warm-started** from the v2 checkpoint: the hypothesized
  mechanism is representation shaping *during* training, and grafting a
  randomly initialized head onto a mature trunk risks the same
  early-updates-damage-the-trunk failure M3.2 diagnosed. A 5M-step run is
  ~2h at 8 envs — warm-starting saves nothing worth that risk.

**Head architecture** (`models/ppo/ppo_agent.py`):
```
shared trunk (obs → 128-dim features)
  ├─ policy head:  Linear(128, 9)  — own action logits      [PPO loss]
  ├─ value head:   Linear(128, 1)  — state value            [PPO value loss]
  └─ opp head:     Linear(128, 9)  — opponent action logits [CE loss]

Total loss = PPO_loss + λ · CE(opp_logits, opp_action_label)
λ = 0.1 default (tune 0.05–0.3 only if the pre-registered contingency fires)
```
CE is computed only over steps with a valid label (grounding below); masked
steps contribute zero. Checkpoint compatibility both directions: old
checkpoints load with a freshly initialized head (warn, don't fail); new
checkpoints must load cleanly in code paths that never touch the head
(`evaluate.py` raw eval, `mcts_agent.py` policy-sampler mode).

**Label grounding (the load-bearing detail).** The gym action space is
9-discrete *from the acting side's perspective*: 0–3 = moves in request-slot
order, 4–8 = switches in bench order. The label attached to each of p1's
decision points is the opponent's resolved simultaneous choice, mapped into
the **opponent's own action frame** — the exact 9-way index the opponent
would have submitted were it a gym player. That frame is precisely what the
MCTS sampler needs: it submits opponent-frame indices for the opponent seat
of the sim. Grounding, computed in `sim/tools/pokemon-gym.ts` (which has
omniscient access to both sides' requests):
- **Move:** match the resolved move ID against the opponent's active
  request's `moves` array → index 0–3.
- **Switch:** match the switched-in Pokémon against the opponent's request
  `side.pokemon` bench ordering → index 4–8.
- **Masked (label = -1, excluded from CE):** no simultaneous opponent choice
  exists at this decision point (e.g., a p1-only force-switch), Struggle/
  pass, or any choice that doesn't map cleanly to a slot (Transform/Disable
  request anomalies). Engine-auto-completed choices (sleep/recharge/locked
  moves — `side.isChoiceDone()`, the M4 gotcha) **are** labeled: they're
  real, known outcomes and exactly what the search sampler must reproduce;
  caveat recorded that they inflate headline accuracy.
- **Dual-seat self-play:** symmetric — each seat's transition is labeled
  with the other seat's simultaneous choice.
- **Documented approximation:** switch labels 4–8 index the opponent's
  *true* bench order, which p1 cannot fully observe; under determinized
  search, unrevealed-slot indices are noise by construction — the same class
  of approximation M4's determinizer already accepts. At sim time the head's
  distribution is masked to the sim opponent's *legal* actions and
  renormalized before sampling.

**MCTS integration** (`models/mcts/mcts_agent.py`): an `opp_sampler:
'policy' | 'head'` option. Head mode evaluates the opp head on the
*searcher's* obs at the node — already computed for the policy prior, so
this is *cheaper* than policy mode (which extracts and forward-passes the
opponent's reveal-tracked obs) — masks to the sim's legal opponent actions,
renormalizes, samples. Falls back to policy mode when the loaded checkpoint
has no head.

### Files to Modify
| File | Change |
|------|--------|
| `sim/tools/pokemon-gym.ts` | Capture opponent's resolved choice per decision point; map to opponent-frame 9-way label; `oppAction` (int, -1 = masked) in `info` for single-seat and dual-seat paths |
| `test/tools/gym.test.js` | Label correctness vs scripted opponents: move-slot and switch-slot mapping, force-switch masking, dual-seat symmetry, locked-choice labeling |
| `models/gym_bridge.js` | Pass `oppAction` through both single and dual (`--selfplay`) protocols |
| `models/gym_client.py` / `models/vec_gym_client.py` | Surface `opp_action` in step `info` dicts |
| `models/ppo/ppo_agent.py` | `opp_head`, forward exposure, masked aux CE in `update()`, two-way checkpoint compatibility |
| `models/ppo/trajectory_buffer.py` | Store per-step `opp_action` labels; `merge_buffers()` support |
| `models/ppo/train.py` | Collect labels from `info`; `--opp-coef` (default 0.1; 0 disables and must reproduce the old loss path); checkpoints → `checkpoints/opp/`; log CE + label coverage per rollout |
| `models/evaluate.py` | Opp-prediction top-1 accuracy reporting on headed checkpoints; `--opp-sampler head\|policy` for `--model mcts` |
| `models/mcts/mcts_agent.py` | Head-based opponent sampler (design above) |

### Build Phasing (parallelizable)
- **Phase 0 (pre-req):** commit the uncommitted post-M4 tuning working tree
  (tuned `MCTSAgent`/`evaluate.py` defaults, mcts_agent cleanup, sweep/A/B
  logs and doc updates) so M5 code starts from a clean tree.
- **Phase 1 (two parallel jobs):** **A** — gym labels + TS tests
  (`pokemon-gym.ts`, `gym.test.js`). **B** — agent head + buffer
  (`ppo_agent.py`, `trajectory_buffer.py`). Disjoint files; the interface
  between them is a single int in `info`.
- **Phase 2 (two parallel jobs, after Phase 1):** **C** — plumbing + trainer
  (`gym_bridge.js`, `gym_client.py`, `vec_gym_client.py`, `train.py`; needs
  A+B). **D** — eval + MCTS sampler (`evaluate.py`, `mcts_agent.py`; needs
  B; the accuracy-reporting half also needs A).
- **Phase 3 (smoke battery):** full TS/gym test suites green; label
  spot-check vs DamageFirstAI (its argmax-damage choice is independently
  predictable from the log); 4k-step training smoke with `--opp-coef 0.1`
  (CE decreasing, label coverage sane); `--opp-coef 0` regression smoke;
  MCTS head-sampler smoke on the smoke checkpoint. Commit code.
- **Phase 4:** decision run + eval battery (protocol below).

### Training + Evaluation Protocol (one decision run, pre-registered)
1. **Decision run:** `models/ppo/train.py --obs-v2 --opponent-mix
   "selfplay=0.5,damagefirst=0.3,random=0.2" --opp-coef 0.1
   --steps 5000000 --num-envs 8` → `models/ppo/checkpoints/opp/`.
2. **Sweep:** every checkpoint, 150 battles vs Random + opp accuracy each.
3. **Raw confirmations:** top 3–4 checkpoints at 500 vs Random / 200 vs
   DamageFirst; opp accuracy over 200 battles vs DamageFirst; optional
   seat-balanced raw h2h vs the v2 control checkpoint.
4. **MCTS battery (primary):** best checkpoint under tuned search (sims=100,
   c_puct=0.5, det=1): head-sampler vs policy-sampler A/B — 500 vs
   DamageFirst + 500 vs Random per arm; latency measured.
5. **Contingencies (pre-registered, no other reruns):** sweep band collapses
   below 45% vs Random → one rerun at λ=0.05; opp accuracy vs DamageFirst
   < 25% → treat as a label bug, fix, rerun.

### Unblocks
M6 (the live bot ships the best MCTS config, head-sampler or not);
AlphaZero-style fine-tuning (a calibrated opponent model improves the quality
of search-generated training data).

---

## M5.5: Human Replay Data + BC for the MLP ✅ COMPLETE — POSITIVE RESULT, NEW BEST AGENT

**Status:** ✅ Complete 2026-07-16. **The human-data pipeline (BC on replays →
anchored PPO fine-tune → tuned MCTS) produced the project's new best agent**,
beating the bot-trained M5 lineage on every pre-registered measure. BC alone
was decisively negative (raw 22% vs Random; 56.2%/45.0% under search); the
M3.2-fixes contingency fine-tune is what converted the human prior into a win.
**Goals:** The best agent to date (M5 ckpt + tuned MCTS) has never trained on
human data — the 119k Metamon replays (M2.5) fed only the retired transformer,
and they're gen1ou, not the competitive format. Get real human games into the
MLP-PPO stack.

**Project decision (2026-07-16):** the competitive target stays
**gen1randombattle** (ladder, eval batteries, MCTS determinizer unchanged),
but training data is **multi-format**: high-Elo gen1randombattle replays PLUS
high-level gen1ou games — "show it how pros play with real teams while keeping
it flexible but strong." Per-format sampling weights are the tuning knob if OU
data hurts randbats play. Team-specific fine-tuning (one fixed OU team) is a
noted future follow-up, out of scope.

### What Was Built (2026-07-16)
| Piece | Detail |
|---|---|
| `scripts/scrape_replays.py` | Public replay-API scraper (`search.json` + `<id>.log`), `--formats gen1randombattle,gen1ou` with per-format `=minrating` (default 1300, unrated skipped); top-up mode stops when caught up; `--backfill` resumes from the deepest cursor; gzipped logs + `manifest.csv` under `data/replays/<fmt>/` (gitignored); rating-bucket census / `--dry-run` |
| `scripts/bootstrap_gen1ou_replays.py` | Bulk-imports raw gen1 logs from HF `jakegrigsby/metamon-raw-replays` — streams only parquet shards 34–36 (where a row-group-stats scan found all gen1* rows) instead of the 5.7GB dump. **98,349 gen1ou logs imported** (10.0k rated ≥1300; 55.9k unrated incl. Smogon tournament games) |
| `sim/tools/replay-adapter.ts` | Replay log → per-decision `(obs, action)` for BOTH seats, using the live gym's own `ObservationTrackers` + `extractFeaturesStructured` (v2) — obs conventions identical to `PokemonGymEnv` by construction. Labels use the M2.5 alphabetical-slot invariant (replays have no request JSON); two-pass parse reconstructs own-side full-game rosters/movesets; own PP ≈ base PP − observed uses; `\|cant\|` decisions dropped (choice unknowable); simultaneous turn decisions cross-linked as M5 opp-head labels |
| `models/replay_adapter_cli.js` | Batch converter → `data/replay_trajs/<fmt>/shard-*.jsonl.gz` (streamed gzip, base64 float32 obs). Coverage: 91% randbats / 86% gen1ou (gap = sleep/para `\|cant\|` turns, unknowable by design) |
| `test/tools/replay-adapter.test.js` | 7 tests, incl. the load-bearing round-trip: a gym battle re-parsed from its own omniscient log yields **byte-identical opponent tokens + volatile dims** at matched decision points; move/switch label↔slot invariants vs dex data; opp-label symmetry; nicknamed OU teams |
| `models/bc_pretrain_mlp.py` | BC of `PPOAgent` (v2 obs, 924-dim): CE on policy head + opp head (λ=0.1, real human opponent labels) through the shared trunk; value head untouched; weighted multi-format interleave; per-format held-out validation accuracy; rated ≥ `--min-rating` OR tournament-prefixed unrated; saves via `PPOAgent.save()` — `evaluate.py`/MCTS consume it unchanged (verified live) |

### Pre-Registered Success Criteria — Final
- ✅ Scraper ≥ 10k high-Elo games collected — met via the gen1ou bootstrap
  (10.0k rated ≥1300 + 55.9k tournament games); randbats backfill additionally
  delivered ~12k more high-Elo randbats logs on 2026-07-16
- ✅ Adapter round-trip test green; label coverage ≥ 90% on gen1randombattle
  (91%; gen1ou's 86% is explained by unknowable `|cant|` turns, not label bugs)
- ✅ BC top-1 accuracy meaningfully above the ~11% chance floor: **49.7%
  (randbats) / 53.1% (gen1ou)** held-out validation, opp-head ~35%
- ✅ **Bar to beat — CLEARED by the fine-tuned checkpoint:** tuned-MCTS
  (sims=100, c_puct=0.5, det=1) over `bcft/ppo_step_5000000_final.pt`:
  **90.6% (453/500) vs Random / 79.2% (396/500) vs DamageFirst** vs the bars
  of 86.0%/72.6% (+4.6pp / +6.6pp, both outside noise at n=500); seat-balanced
  h2h vs the M5 best checkpoint: **78.4% (392/500) combined — 77.6% (194/250)
  as p1 (run in 2×125 fresh-process chunks, see known issue below) / 79.2%
  (198/250) as p2** — far above the 50% bar.
- ✅ Contingency exercised as pre-registered: BC-only was negative (raw 22%R /
  14.5%DF; MCTS 56.2%R / 45.0%DF — `models/mcts/results/m55_bc_mcts_*.log`),
  so the M3.2 fixes (`--pretrain-checkpoint`, `--value-warmup-steps`,
  `--bc-anchor` constant 0.05) were ported to `models/ppo/train.py` and ONE
  5M-step anchored fine-tune ran on the M5 opponent-mix recipe. No other reruns.

### Phase 4 Results (2026-07-16)
| Stage | vs Random | vs DamageFirst | Notes |
|---|---|---|---|
| BC raw (run 2, value-BC) | 22% | 14.5% | ~50% human-imitation acc doesn't transfer to raw bot play |
| BC + tuned MCTS | 56.2% (281/500) | 45.0% (225/500) | search lifts hugely, still below bars |
| Fine-tune raw (final 5M) | 54.6% (273/500) | 42.0% (84/200) | sweep band 35–58%, no collapse; first BC→RL transfer that improves |
| **Fine-tune + tuned MCTS** | **90.6% (453/500)** | **79.2% (396/500)** | **new best agent** |

Fine-tune recipe: warm-start `bc_mlp_gen1.pt` → PPO 5M steps, v2 obs,
opponent-mix selfplay/damagefirst/random 0.5/0.3/0.2 (pool seeded M2 +
M3.3-best), value warmup 200k, BC KL-anchor coef 0.05 constant, opp-coef 0.1.
Checkpoint: `models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt`.
Search latency at the new operating point: ~180–200ms mean / ~370ms p95 per
searched decision (higher than M5's ~60ms — the run shared the machine with
the replay scraper; still far inside M6's 2s budget).
Reading: neither ingredient suffices alone (BC ceiling ~56% under search;
bot-trained RL lineage ceiling ~72.6% DF) — the human prior plus anchored RL
plus search compounds. Side note confirmed again: the human-data opp head
reads DamageFirst at only ~21–26% (bot-trained M5 head: 30–36%) — mixture
miscalibration, as in M5.

### Known issue (logged 2026-07-16): h2h `sim_fork` crash
The MCTS-vs-checkpoint h2h `--mcts-seat p1` arm crashed TWICE, both times
around battle 130–150, on a Node-side `sim_fork` error ("Cannot read
properties of undefined (reading 'id')" — thrown inside
`BattleSim.fork()`'s serialize/fromSnapshot path; the bridge forwards only
the message, not the stack). Both 500-battle vs-bot battery arms and the
250-battle p2 h2h arm ran clean, so it's specific to the h2h+search p1
path and looks like slow state buildup, not a random per-decision fault.
Workaround used: run the arm in fresh-process chunks. Root-cause TODO for
M6 (the ladder bot runs long sessions): add stack forwarding to
`gym_bridge.js` error responses and a fork-crash repro harness.

### Unblocks
M6 ships whichever checkpoint wins this battery; the data pipeline also
ingests the M6 bot's own ladder games (`data/replays/self_ladder/`).

---

## M6: Server Integration & Ladder ✅ COMPLETE 2026-07-17

**Status:** ✅ Complete — **all three success criteria met.** Both phases
built and shipped; the 100-battle official-ladder criterion run finished
2026-07-17 with the MCTS config, zero mechanical failures. External
measurement (the milestone's purpose): **the agent sits at the bottom of
the human ladder** — final Elo **1017** (floor is 1000), GXE **23.9%**,
Glicko-1 **1281 ± 37**, account record 23W–96L on gen1randombattle.
Full results below. Rewritten post-M5: the original spec predated the
M3.2 transformer retirement and M4 search. **Ships the M5.5 winner
(`models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt`) with tuned
policy-sampler MCTS** (M5.5 battery beat the M5 checkpoint on every measure;
the M5 ckpt `opp/ppo_step_5000001_final.pt` remains the ladder-A/B control).
**Clarified:** the ladder needs no recruited humans — matchmaking supplies
opponents, and gen1randombattle is one of Showdown's most active ladders.
Every ladder game is also a human-opponent trajectory for the M5.5 pipeline.

### Phase 1 — Raw-policy bot (no search; trivially inside the 2s budget)
- `tools/ladder-bot/` (TS): websocket client to the official server
  (`action.php` challstr login; registered bot account; config for
  server/format/battle count). Protocol → obs is code reuse: `|request|` JSON
  gives full own-side state, public log lines feed `ObservationTrackers` —
  the same two inputs `PokemonGymEnv` consumes.
- Inference: the bot spawns `models/infer_server.py` (reverse of
  `gym_bridge.js` — Python stdio server loading `PPOAgent`; obs+mask in,
  action out), keeping PyTorch out of Node and checkpoint loading verbatim.
- Every game's log saved to `data/replays/self_ladder/`.
- Local server first (`node pokemon-showdown start --no-security`), then
  official ladder with conservative pacing (one battle at a time).

### Phase 2 — Search on ladder (the novel component; after Phase 1 is stable)
- MCTS needs a local engine `Battle` to clone; a remote game has none. Add
  `BattleSim.fromTracked(...)`: construct a local gen1randombattle battle from
  our full known team (request JSON) + determinized opponent (existing
  determinizer), then patch tracked state (HP, status, boosts/volatiles).
  Extends the determinization approximation M4 already accepts.
- `models/mcts/mcts_agent.py` then runs unchanged (57–85ms/move ≪ 2s).

### Success Criteria (replacing the stale originals)
- Bot completes ≥ 100 consecutive official-ladder battles without
  crash/timeout losses — ✅ **100/100 clean** (2026-07-17, MCTS config;
  zero invalid choices, timeouts, or crashes)
- Decision latency < 2s per move — ✅ **max 579ms observed** (cold-start,
  battle 1); warm per-battle maxima 136–540ms, typically ~150–350ms
- Elo tracked and reported — ✅ final **Elo 1017 / GXE 23.9% /
  Glicko-1 1281 ± 37** (account Novapool, gen1randombattle, 23W–96L)

### Results (criterion run, 2026-07-17)

Three rated sessions on account Novapool, all vs live matchmade humans:
shakedown 0/3 (raw policy), first criterion attempt 2/15 (raw policy,
externally stopped), full criterion run **21/100 with MCTS** (sims=100,
det=1, c_puct=0.5). Logs in `data/replays/self_ladder/`.

- **MCTS lift replicates externally:** raw policy ~13% (2/15) → MCTS 21%
  (21/100); early pace was 12/41 (~29%) before a long mid-run losing
  streak. GXE 23.9% is the settled read: ~24% expected vs a random
  ladder player.
- **Honest decomposition:** 13 of the 21 wins were ≤9-decision battles
  (opponent forfeits/disconnects — a normal part of low-ladder Elo, but
  not evidence of outplaying anyone). Full-length-battle win rate is
  ~8/87 (~9%). Bottom-of-ladder humans beat this agent ~4-in-5 games
  even with search.
- **The external measurement M0–M5 never had, delivered:** an agent at
  90.6%/79.2% vs the project's training bots and 78.4% h2h vs the prior
  best is still a ~1017-Elo (floor-adjacent) ladder player. The
  bot-relative ledger drastically overstates absolute strength; the
  human ladder is now the project's primary evaluator.
- **Qualitative failure notes (user-observed, watching live games):**
  Hypnosis into an already-asleep Pokémon; Explosion into a Ghost-type;
  repeated not-very-effective moves (e.g. Fire Blast into Slowbro); no
  evident handling of Hyper Beam recharge or Sleep Clause. Root cause
  traced to the observation schema (`feature-extractor.ts`): no type
  effectiveness, no base stats/species identity, no move secondary-effect
  flags (recharge/self-KO/status/priority), no Sleep Clause signal —
  status moves are near-indistinguishable to the net. Motivates an obs
  schema v3 as the leading next-milestone candidate.
- Mechanical: 100 consecutive battles, one at a time, ~21–61 decisions
  each, 75–90% of decisions searched (rest force-switch/locked → raw
  policy per `fromTracked`'s contract), zero search-fallback storms.

---

## M7: Observation Schema v3 ✅ COMPLETE — INCONCLUSIVE (2026-07-17 → 2026-07-20)

**Status:** ✅ All 6 phases executed. Criterion A passed, Criterion B passed decisively, Criterion C landed in the pre-registered noise band — net result is a new best agent by every bot/search metric, with an unproven (not disproven) ladder-level claim.

**Final results:**
- **Criterion A:** ✅ Pass — v3 obs shape (12, 86) valid at every decision point, no NaN/inf, Sleep Clause flag verified via unit tests (incl. Rest exclusion, faint clearing).
- **Criterion B:** ✅ Pass, decisively — tuned MCTS (sims=100, det=1, c_puct=0.5) on the v3 PPO checkpoint (`ppo_step_5000002_final.pt`): **93.0% (465/500) vs Random, 84.2% (421/500) vs DamageFirst** — both pre-registered bars cleared (≥80% R / ≥65% DF), both ahead of the prior (v2) best of 90.6%/79.2%. New best agent by bot eval.
- **Criterion C:** 🟡 Inconclusive — 100/100 consecutive rated ladder games, zero crashes, well under the 2s/move budget. Raw record 30W–70L (up from M6's 21W–79L). Account state after the run: **Elo 1034.6, GXE 28.2%** (M6 baseline: Elo 1017, GXE 23.9%). GXE landed in the pre-registered 25–34% noise band — directionally improved (+4.3pp GXE, +9pp raw win rate) but not the ≥35% needed to call it a clear win, per the pre-committed rule that 25–35% is inconclusive rather than a verdict either way.
- **Verdict:** v3's rules-understanding fixes (type effectiveness, move-effect flags, Sleep Clause) produced a measurable, non-trivial bot/search improvement and a directionally positive but statistically inconclusive ladder result. Per the pre-registered contingency, an optional 50-game follow-up run would be needed to resolve the band; not run in this session (user handed the ladder run to their own terminal after bot-side disconnects — see IN-PROGRESS.md).
- **Bug fixed along the way:** `sim/tools/battle-sim.ts` had no v3 handling (`_extractObsFor` sized non-v2 obs at the legacy 65-dim token and never passed `v3Info`), which crashed `--model mcts --obs-v3` outright. This was a real gap in M4/M5's "MCTS is obs-shape-agnostic" claim (only ever verified Python-side). Fixed and covered by the existing `battle-sim.test.js` / `gym.test.js` suites plus a smoke test.

**Motivation:** M6's 21/100 ladder win rate (9% full-battle) despite 90.6% vs bots revealed a rules-understanding gap. User-observed blunders: Hypnosis into an already-asleep Pokémon, Explosion into a Ghost-type, repeated not-very-effective moves (Fire Blast into Slowbro), no handling of Hyper Beam recharge or Sleep Clause. Root cause: v2 obs schema (65 base + 12 boosts/volatiles) lacks type effectiveness, species identity, move-effect flags, and Sleep Clause signaling. The observation schema has never been the bottleneck (parallel training, MCTS, BC, opponent modeling all improved; v2 was a wash). But this time the rules omitted are ones humans apply *every* game.

### v3 Schema Design (appendix to v1+v2 — byte-identical prefix, new dims appended)

**Current state (v2):**
- **Dims 0–64:** Per-Pokémon tokens as v1 (HP ratio, level, type one-hots, status, active/unknown/fainted flags, move 1–4 @ 6 dims each) — unchanged
- **Dims 65–76:** Active token only — boosts (7 stages), screens/Sub/Leech Seed (4 flags), toxic counter (1 dim) — 12 dims
- Total per token: 77 dims; full obs: (12, 77)

**v3 additions (per-move, affects all 12 tokens):**

| Feature | Dims | Notes |
|---------|------|-------|
| **Per-move type-effectiveness vs opponent active** | 4 | The combined multiplier vs BOTH defender types — gen1 values are {0, 0.25, 0.5, 1, 2, 4} (dual-typed defenders stack: e.g. Ice Beam vs Dragonite = 4x, Fire Blast vs Slowbro = 0.5x, Explosion vs Gengar = 0x). Encode scaled, e.g. multiplier/4 → [0, 1] (exact encoding is a Phase 0 decision). Highest-leverage fix (type chart is the hardest thing to learn from a scalar type index). One lookup per decision, pre-computed. |
| **Base stats vs opponent active** (optional A/B) | 1–2 | Speed ratio (own active speed / opp active speed, clamped) OR HP ratio (own / opp). Alternative: defer, use in M8 as a phase 1a A/B if v3 base doesn't clear the bars. |
| **Per-move effect flags** | 3–4 | Bitmask: recharge (Hyper Beam), self-KO (Explosion/Selfdestruct), priority (Quick Attack), inflicted-status. Total 3–4 flags; store as separate dims or one packed byte (space is cheap; clarity favors separate). |
| **Sleep Clause flag** | 1 | Sleep Clause Mod blocks putting a SECOND opponent Pokémon to sleep: flag = 1 while any opponent Pokémon we put to sleep is still asleep (self-induced Rest sleep does NOT count toward the clause). Derivable from the public log via the existing reveal tracker — a slept mon was necessarily active when slept, so no omniscient access is needed (project convention: observations use observable info only). |

**Dimensions breakdown (per-token proposal):**
- Move 0–3: 6 dims each (base_power, accuracy, PP, type_idx, category, disabled) → expand each move block to 6 + 4 type-eff + 1 recharge + 1 self-KO + 1 priority + 1 inflicted-status = **14 dims per move**
- Alternative (cleaner): move features stay at 6 dims, append [type-eff × 4, effect-flags × 3] to every token → **65 base + 12 volatiles + 4 type-eff + 3 effect-flags = 84 dims per token**

**Final proposal (method B, less disruptive to move indexing):**
- Dims 0–64: v1 (unchanged)
- Dims 65–76: v2 boosts/volatiles (unchanged)
- Dims 77–80: type-effectiveness (4 moves × 1 dim each, not per move)
- Dims 81–83: move effect flags (packed as 3 separate bits: recharge, self-KO, priority)
- Dim 84: inflicted-status ID (0–5, same as existing status enum, or a "carry" flag for status moves)
- Dim 85: Sleep Clause flag (1 = active opponent has sleeping Pokémon)
- **Total per token: 86 dims; full obs: (12, 86)**

Keep dims 0–76 byte-identical to v2 for cross-schema eval slicing (`slice_structured_obs`); all new info is dims 77–85. BC data and ladder evals can slice to v2 for A/B vs the current best.

### Files to Modify
| File | Change |
|------|--------|
| `sim/tools/feature-extractor.ts` | `TOKEN_DIM_V3 = 86`; `extractFeaturesStructured(volatiles, v3Info)` new param; fill type-eff (lookup table indexed by [own-move-type, opp-type]), effect-flags (parse move object), inflicted-status from move data |
| `sim/tools/pokemon-gym.ts` | Sleep Clause tracker from public log lines (`|-status|...slp` / `|-curestatus|` / `|faint|` on revealed opponent mons; exclude `[from] move: Rest`); NOT reset on switch — sleep persists on the bench. `obsMode: 'structured-v3'` |
| `sim/tools/replay-adapter.ts` | Regenerate BC trajectory data with v3; same adapter output schema (obs + action + done) but now obs is (12, 86) instead of (12, 77) |
| `models/gym_bridge.js` | `--obs-v3` flag; serialize 1032-element flat array (12 × 86) |
| `models/gym_client.py` / `models/vec_gym_client.py` | `obs_v3=` (reshape to (12, 86)), `set_obs_version()` for pool mixing |
| `models/bc_pretrain_mlp.py` | Accept v3 trajectory shards; verify input shape (924 → 1032); report per-format validation accuracy |
| `models/ppo/train.py` | `--obs-v3` flag; `--obs-size` auto-inferred; checkpoints → `checkpoints/v3/` |
| `models/evaluate.py` | `--obs-v3`; v3-vs-v2 head-to-head slicing (both obs versions from same forward pass via `slice_structured_obs`) |
| `models/mcts/mcts_agent.py` | v3-compatible (obs shape already parameterized; no changes needed if gym_client handles reshaping) |
| `tools/ladder-bot/ladder-bot.js` | Forward `--obs-v3` to bridge; no protocol changes needed |
| `test/tools/gym.test.js` | v3 shape (1032), v2-prefix byte-equality, type-eff values (known matchups incl. 4x/0.25x/immune), Sleep Clause state tracking (incl. Rest exclusion, faint clearing), full-battle v3 stability |

### Implementation Strategy (6 phases, parallelizable after Phase 0)

**Phase 0 (prerequisite):** Spec exact feature set; enumerate gen1 type chart (15×15); validate move-effect-flag mapping vs dex.

**Phase 1 (parallel):**
- **1A:** Feature-extractor logic (`sim/tools/feature-extractor.ts`) — type-eff lookup, move-effect flags, Sleep Clause signal; tests
- **1B:** Gym tracker updates (`sim/tools/pokemon-gym.ts`) — Sleep Clause state, obsMode 'structured-v3'; tests

**Phase 2 (after 1A+1B, parallel):**
- **2A:** Bridge/gym-client plumbing (`gym_bridge.js`, `gym_client.py`, `vec_gym_client.py`) — `--obs-v3` serialization, reshaping, pool mixing
- **2B:** Replay-adapter regeneration (`sim/tools/replay-adapter.ts`, `models/replay_adapter_cli.js`) — one-time batch run on existing logs; output to `data/replay_trajs/v3/` (or overwrite v2 shards after spot-check)

**Phase 3 (after 2A+2B):** Trainer wiring (`models/ppo/train.py`, `models/bc_pretrain_mlp.py`, `models/evaluate.py`, `models/infer_server.py`); smokes

**Phase 4:** BC pretrain on v3 data — typically 2h at the MLP recipe

**Phase 5:** PPO fine-tune (5M steps, opponent-mix recipe) — typically 2h at 8 envs

**Phase 6:** Ladder criterion run (≥100 games for Glicko stability)

### Pre-Registered Success Criteria

**Criterion A (hard gate — must pass):** v3 produces a valid observation shape at every decision point (smoke tests); no NaN/inf in type-eff or dims; Sleep Clause flag toggles correctly.

**Criterion B (bot evals — cheap gates before ladder spend):**
- v3 bot-trained PPO + tuned MCTS vs Random: ≥ 80% (target: within 10pp of current 90.6%), OR
- v3 bot-trained vs DamageFirst: ≥ 65% (within 10pp of current 79.2%), OR
- No regression if either is missed — if v3 hits 70–80% range vs both, proceed to ladder

**Criterion C (ladder — the true test):**
- Ladder run: ≥ 100 consecutive rated games, ≤2s per move, zero crashes
- GXE after 100 games: "clearly above" the M6 floor (23.9% ± 35 = [−11%, 59%]; Glicko-1 ± 37). Operationally: if GXE ≥ 35%, strong signal of improvement; 25–34%, noise band; < 25%, slight regression. Pre-commit: *a 100-game run that lands in the 25–35% band is inconclusive, not a win or loss*. Only GXE ≥ 35% counts as a clear win; < 25% counts as clear regression.
- Conditional: if bot evals show no regression (Criterion B met), ladder run is mandatory regardless of bot numbers (rules understanding may not show in 500-battle evals vs deterministic heuristics).

**Contingency:** if v3 lands in the 25–35% band (inconclusive) after 100 games, one optional 50-game follow-up before deciding next steps.

### Cautionary Reading

Schema v2 (boosts/volatiles) was a wash in bot evals and a statistical peer under search. The difference here: v2 added state *rarely needed* vs bots (boosts help but Amnesia/Swords Dance aren't on every randbat; volatiles even rarer). v3 adds type effectiveness — the single most frequent decision rule in any Pokemon game, across all levels. If v3 doesn't clear the bars, the binding constraint is not obs richness but something deeper (model capacity, reward signal, data quality, search depth, or inherent randbats luck variance).

### Unblocks

M8 (if v3 succeeds): further obs refinements, AlphaZero-style self-play value targets, or opponent-pool diversity experiments.

---

## M8: Value-Head Targeting + Ladder Infrastructure ⏳ SCOPED

**Status:** ⏳ Scoped 2026-07-22 (pending user approval to build)

**Thesis:** M7 produced the best agent on bot metrics (+2.4/+5.0pp over M5.5) but landed inconclusive on the ladder (28.2% GXE, inside the 25–34% noise band). The gap between 93% bot eval and ~28% ladder (9% full-battle win rate) suggests one or more binding constraints: (a) the value head was trained on PPO rewards, not search outcomes — it doesn't know what constitutes a "winning" position under lookahead; (b) obs richness still isn't sufficient; (c) opponent-pool distribution. M8 uses a **contingency escalation**: try the flagged obs refinement first (cheap, 2–3 hour payoff), escalate to a structural change (AlphaZero-style value targets) if that fails.

**Rationale for focusing on value targeting over opponent pool:** M7's cautionary reading warned that if obs richness doesn't help, the binding constraint is deeper than observations. The value head is the search's ground truth — if it's mismeasuring positions, search can't improve on a bad foundation. Opponent pool is a second-order lever; value targeting is first-order.

---

### M8 Phases (contingency gates between them)

**Phase 0 (Infra + prerequisite):** Websocket reconnect for ladder-bot — ✅ BUILT 2026-07-22 (acceptance pending a long ladder run)

- **Goal:** De-risk long ladder runs. The 100-battle M7 ladder runs fragmented into 3 sessions due to dropped connections; Phase 6 (ladder validation) requires reliable multi-hour uptime.
- **Scope:** `tools/ladder-bot/ladder-bot.js` — add reconnect logic with exponential backoff + state recovery (last battle ID). No protocol changes; transparent to the server.
- **Acceptance:** 1 clean 100+ battle session, zero manual reconnects, zero data loss.
- **Effort:** ~30 min (websocket library API, simple state machine)
- **Blocker:** None — can start immediately in parallel with Phase 1.
- **Built 2026-07-22:** exponential backoff (1s→30s cap, 15 attempts, reset on
  login), re-login on fresh `challstr`, in-flight battles rejoined via `/join`
  with the room rebuilt fresh from the server's full-log replay (repeat `|init|`
  resets an existing room — no double-processed tracker lines); untracked
  `updatesearch` games joined (covers ladder matches made while offline);
  `send()` guarded while disconnected; `[Invalid choice]` default fallback now
  ignores "too late" (stale replayed request would loop). Verified by local
  bot-vs-bot smoke through a killable TCP proxy: mid-battle kill → reconnect →
  rejoin → clean finish. Formal acceptance rides on the next 50+ game ladder run.

---

**Phase 1A (Obs refinement — fast A/B):** Base-stats + speed-ratio dims

- **Goal:** Test the M7 "flagged-for-M8" refinement (line 1301 MILESTONES.md). If obs richness *is* still a lever, this is the cheap way to find it.
- **Design:** Extend v3 schema (currently 86 dims/token) with 1–2 additional dims:
  - **Speed ratio:** own active speed / opp active speed, clamped to [0, 2] (own much slower → opp much faster). Speeds from dex via `Dex.mod('gen1').getSpecies()`.
  - Alternative (backfill during Phase 1 if speed alone underperforms): add HP ratio (own active HP / opp active HP for health-state inference).
  - **Schema:** v3 prefix unchanged (dims 0–85), append new dims → **v3-extended = 87–88 dims/token**. obs shape (12, 87/88).
- **Implementation:** Minimal changes to `sim/tools/feature-extractor.ts` (add 1 lookup at each token) + test updates. Cross-schema eval can slice v3-extended back to v3 (77 dims after v2 prefix, drop new dims) for head-to-head vs v3 control.
- **Training:** 2–3 hour quick A/B run on 1M or 2M PPO steps (vs 5M full runs) on the M7 opponent-mix recipe to check signal. If the raw policy moves >2pp on both opponents (outside the ±3–4pp noise floor for short runs), escalate to Phase 1B full run.
- **Decision gate:** "Does speed ratio help?"
  - ✅ **If yes** (>2pp improvement confirmed at 2M steps): escalate Phase 1B → full 5M run, skip Phase 2.
  - ❌ **If no** (≤2pp or negative): abort Phase 1B, escalate to Phase 2 (value targets).
- **Effort:** 4–6 hours (code, test, training, eval).
- **Infra built 2026-07-23:** `structured-v3-extended` schema (87 dims/token,
  obs 1044) — v3's 86 dims byte-identical + **dim 86 = own/opp active base-speed
  ratio** (`min(ownBaseSpe/oppBaseSpe,2)/2` → [0,1]; equal=0.5, ≥2x=1.0, unknown
  opponent→neutral 0.5), placed on all 12 tokens like the Sleep Clause flag and
  computed from pure gen1-dex base-speed lookups (no gym tracker state). Rolled
  out across the full v3 pipeline (type-chart-v3/feature-extractor/pokemon-gym/
  battle-sim + bridge/gym_client/vec/train.py `--obs-v3-extended` →
  checkpoints/v3-extended/ /evaluate.py all runners incl. cross-schema h2h/
  ladder-bot 1044 auto-detect). `./build` green, 99/99 tool tests (16 new),
  Python train/eval/h2h/MCTS smokes all clean (obs_size 1044, per-token slicing
  1044→1032 works, MCTS battle-sim path correctly sized).

- **A/B run ✅ COMPLETE 2026-07-28 (user-run) — CRITERION A FAILED, NEGATIVE
  RESULT.** 2M-step v3-extended arm trained (`--opp-coef` default nonzero
  routed the checkpoint to `checkpoints/opp/ppo_step_2000003_final.pt`
  instead of `checkpoints/v3-extended/` — a `train.py` checkpoint-dir
  routing quirk: the `elif args.opp_coef != 0.0` branch precedes the
  `--obs-v3-extended` branch, so any run without `--opp-coef 0` lands in
  `checkpoints/opp/` regardless of obs schema; obs schema itself trained
  correctly, this is a path-naming issue only). Raw-policy 150-battle eval:
  **58% (87/150) vs Random, 61% (91/150) vs DamageFirst**. v3 control
  (`v3/ppo_step_2000015.pt`, existing sweep data, no re-run needed): **61%
  (92/150) vs Random**; DamageFirst has no exact-step reading (nearest,
  step 2500025: 55%/200, not directly comparable — checkpoint transfer to
  re-run the true control was skipped as not worth the effort). **Result:
  Random −3pp (wrong direction), failing the "both opponents >+2pp" gate
  outright regardless of the DamageFirst comparison.** Per the pre-registered
  rule, Phase 1A gates negative → **Phase 1B (full 5M run) is skipped;
  escalate to Phase 2 (AlphaZero value targeting).**

**Phase 1B (if Phase 1A gates positive):** Full v3-extended PPO run

- **Goal:** Train the obs-refined model to convergence and confirm the bot-eval improvement.
- **Scope:** `models/ppo/train.py --obs-v3-extended --steps 5000000` (opponent-mix recipe as in M7), all infrastructure already in place. Checkpoints → `checkpoints/v3-extended/`.
- **Sweep & confirmation:** 20-checkpoint sweep vs Random (150 battles each), confirmations at ≥500 vs Random / 200 vs DamageFirst on top checkpoints.
- **Decision gate:** "Does v3-extended beat v3 on both opponents?"
  - ✅ **If yes** (≥+2pp on both at n=500, outside ±4.5pp noise): proceed to Phase 4 ladder validation with this checkpoint.
  - 🟨 **If tie** (+1pp to −1pp band): proceed to Phase 4 anyway (non-regression is sufficient; might transfer better to ladder).
  - ❌ **If regression** (<−2pp either side): abort, escalate to Phase 2.
- **Effort:** 2–3 hours training (parallel machine) + 2 hours eval.

---

**Phase 2 (if Phase 1 gates negative): AlphaZero value targeting**

If obs refinement doesn't move the needle, the binding constraint is likely that the value head is mismeasuring positions under lookahead. Adopt an AlphaZero-style approach: use MCTS search results (outcome + discounted return from the search tree) as training targets for the value head.

- **Goal:** Train the value head on search-quality targets, not PPO-reward targets. Improves leaf evaluation in the MCTS tree, tightens search quality.
- **Design:**
  - Generate a self-play dataset: run the current best policy (M7 checkpoint) through the MCTS search (100 sims) against a pool copy (self-play), collect 10k–20k games (~100k decisions).
  - For each decision state, run a rollout from the MCTS node, collect discounted cumulative return `G_t = R_t + γ R_{t+1} + … + γ^n V(s_n)` (use γ=0.99 to match existing training, or γ=1.0 for undiscounted final outcome, TBD). This is the "MCTS target" — better than the PPO-trained value head.
  - Freeze the policy and opponent heads, train the value head only on `MSE(V_head(s) - G_mcts)` for 1–2 epochs over this dataset.
  - No retraining the full policy; this is value-head fine-tuning only (~30 min).
- **Implementation:** Self-play collection (existing `gym_bridge.js` + MCTS with outcome tracking), value-training script (small PyTorch loop on collected trajectories).
- **Expected payoff:** Value head becomes a better leaf evaluator; MCTS should improve downstream (less dominated by luck at leaf nodes).
- **Effort:** 8–12 hours (self-play collection infrastructure, value fine-tune loop, testing).
- **Risk:** Self-play data quality depends on the policy; if the policy is weak, the value targets are weak. Mitigated by using the already-strong M7 checkpoint.
- **Infra built 2026-07-28:** two scripts plus a small MCTS instrumentation change.
  - `models/mcts/mcts_agent.py` — `act()` now also exposes `last_root_visits`
    (root visit counts) and `last_root_value` (root Q = mean backup return,
    NaN when no search ran at that decision, e.g. forced/locked choices).
    Search behaviour is unchanged.
  - `models/collect_value_data.py` — self-play collection. Tuned MCTS drives one
    seat (seats alternate per game for balance), a frozen raw-policy checkpoint
    (`--opponent-checkpoint`, default the same checkpoint) drives the other.
    Every searched decision records obs (sliced to the checkpoint's `obs_size`),
    root visits, root Q, the shaped gym reward accumulated to the next decision
    **in the searching seat's perspective** (p2 rewards negated, same
    pending-transition accounting the PPO trainer uses), and the game outcome.
    `.npz` shards flush every `--shard-games` games and re-running resumes from
    disk; `--workers N` runs N independent bridges (~600 games/h per worker at
    `--sims 100` on CPU).
  - `models/value_finetune.py` — value-head fine-tune. Targets are derived at
    train time from one dataset: `--target outcome` (final result, pure
    AlphaZero), `mc` (discounted shaped return, `--gamma`), or `root` (search
    root Q; NaN rows dropped). Train/val split is by whole game (no leakage);
    reports val MSE + sign agreement before and after. Trains **the value head
    only** — trunk/policy head/opp head come out bit-identical (verified), so
    the MCTS prior is unchanged and the Criterion C delta is attributable to
    leaf evaluation. `--unfreeze-trunk` opts out (and forfeits the clean A/B).
    Output is a normal PPO checkpoint (`evaluate.py`/`infer_server.py`/ladder-bot
    load it unchanged).
  - **Verified:** collection smokes on the v3-extended 2M checkpoint (single
    worker, 2 parallel workers, resume-after-kill), reward signs match outcomes
    on both seats, all obs finite; fine-tune smokes on all three targets;
    fine-tuned checkpoint plays through `evaluate.py --model mcts` end-to-end;
    diff of the saved checkpoint confirms only `value_head` changed. Early
    signal from the smoke data: the PPO-trained value head reads clearly
    optimistic (root Q ≈ +0.24…+0.58 in games the seat went on to lose; val
    sign agreement 0.08 vs outcomes) — consistent with the Phase 2 thesis,
    though the smoke sample is far too small to be evidence.
- **A/B run ✅ COMPLETE 2026-07-29 (user-run on the MacBook) — CRITERION C
  FAILED, NEGATIVE RESULT.** Full log:
  `models/mcts/results/m8_phase2_valft_criterionC.log`.

  **Collection:** 2000 games / 66,459 decisions / 84 shards in
  `data/value_targets/m8_v3` (6 workers, ~1100 games/h each, ~35 min wall).
  Data clean: all obs finite at 1032 dims, seats balanced 33251/33208, 3.75%
  NaN root values (forced/locked choices, expected). **Label skew: the
  searching seat won 75.5%** vs the frozen raw-policy opponent.

  **Fine-tune (`--target outcome`):** val MSE 0.7414 → 0.5907, sign agreement
  0.713 → 0.800. Read against the right baselines this is a large in-distribution
  gain: target variance is 0.860² = 0.7396 (the MSE of a constant predictor), so
  **R² went from ~0.00 to ~0.20** — and the BEFORE number was marginally *worse*
  than a constant. Base-rate sign agreement (always guess "win") is 0.755, so
  **BEFORE 0.713 was below that floor.** The PPO-trained value head had
  essentially no discriminative power over outcomes — a direct confirmation of
  hypothesis (a) in the M8 thesis.

  **Criterion C A/B (tuned MCTS vs DamageFirstAI, 200 battles each):**
  base **82.5% (165/200)**, fine-tuned **80.0% (160/200)** → **−2.5pp against a
  ≥+3pp gate.** Control sanity-checks against the M7 battery's 84.2% at n=500
  (consistent within noise). SE on the difference is ~3.9pp, so the delta is
  −2.5 ± 3.9pp — the fine-tune is *not* established as harmful, but it plainly
  fails a +3pp point gate. Per the pre-registered rule: **Phase 3 is SKIPPED;
  the M7 checkpoint carries into the Phase 4 ladder run.**

  **Substantive finding: leaf-value calibration improved a lot and none of it
  transferred to play strength.** Two candidate explanations, both actionable
  if this is retried: (1) MCTS selection depends on *relative* leaf values, and
  much of the MSE gain came from learning the base rate (+0.51) — a constant
  offset that doesn't discriminate between sibling nodes; (2) distribution
  mismatch — targets were collected against a frozen raw-policy opponent, but
  the A/B is against DamageFirst.

  **Replication run, 2026-07-30 — explanation (2) tested and eliminated.**
  Collection was repeated with `--opponent damagefirst`, so the targets come
  from exactly the distribution Criterion C is scored on: 2000 games / 57,747
  decisions / 90 shards in `data/value_targets/m8_v3_df` (10 workers, ~550
  games/h each, ~21 min wall on the home box). The `--target outcome` fine-tune
  again worked in-distribution — val MSE **0.6917 → 0.4915** against a
  constant-predictor baseline of 0.610 (**R² −0.13 → +0.19**), sign agreement
  **0.758 → 0.840**, and val MSE flat between epoch 1 (0.4908) and epoch 2
  (0.4915), so this is not undertrained. **Criterion C came back at exactly the
  same 160/200: 80.0% vs base 82.5%, −2.5pp. Failed again.** Matching the
  collection distribution to the eval distribution changed nothing.

  **This strengthens explanation (1).** These labels are far more skewed — target
  mean **+0.624**, the searcher beats DamageFirst ~84% vs 75.5% in self-play — and
  post-fine-tune sign agreement (0.840) is essentially that base rate. A value
  head that has largely learned "this seat usually wins" scores better on MSE
  while contributing a near-constant offset, and MCTS's selection rule compares
  leaf values *relatively*, so a constant offset is invisible to it. That is a
  property of how search consumes leaf values, not of which target the head is
  fit to.

  **Read the two runs as a pair.** Each delta is −2.5pp against SE ~3.9pp, i.e.
  inside noise, so neither run establishes the fine-tune as *harmful*. But two
  independent nulls from two differently-collected datasets is a much stronger
  negative than either alone, and it is sufficient grounds to stop spending on
  this thesis rather than to keep tuning it.
  Logs: `models/mcts/results/m8_phase2_df_valft_{criterionC,train}.log`;
  checkpoint `models/ppo/checkpoints/v3_valft/ppo_v3_df_valft_outcome.pt`
  (home box only — not committed).

  **Methodological note (applies project-wide): the +3pp gate at n=200 was
  underpowered.** A true +3pp effect would have been detected only ~1/3 of the
  time. Future A/Bs resolving ±3pp need ~500–800 battles per arm.

  **Untried, and the prior is now weaker:** `--target mc` and `--target root`
  reuse either dataset (~1.5h for both incl. evals). `outcome` was the strongest
  target a priori and has now failed twice, and explanation (1) — the surviving
  one — predicts the other targets fail too, since it concerns what MCTS does
  with leaf values rather than what the head is fit to. If run anyway, any arm
  clearing +3pp must be confirmed at 500 battles/arm before it counts as a pass.
  Runbook: `docs/TRAINING-COMMANDS.md` → **M8 Phase 2**.

---

**Phase 3 (if Phase 2 gates positive): Full MCTS value training run** — ❌
**SKIPPED 2026-07-29.** Phase 2 gated negative (−2.5pp vs a ≥+3pp bar), so per
the pre-registered rule this phase is not run and the M7 checkpoint carries
into Phase 4. The design below is retained for reference: it would only become
live if a future value-targeting attempt (e.g. the untried `mc`/`root` targets,
or collection against a stronger opponent) clears Criterion C.

If Phase 2's value-head fine-tuning moves the MCTS needle, scale it up: a full self-play + PPO loop where the policy trains on MCTS trajectories with value targets.

- **Goal:** Full AlphaZero recipe: self-play → MCTS search → value targeting + policy training loop.
- **Design:**
  - **Generation:** Generate K=10k self-play games via M7-policy MCTS + pool self-play (each side is a pool member or the current policy), store search outcomes and intermediate obs.
  - **Training:** PPO on the search trajectories, but with value targets `G_mcts` instead of TD-lambda estimates. Continue training the opp head (label-supervised as before). 1–2 million steps of training on the pool.
  - **Iteration:** Periodically checkpoint and re-generate self-play with the new policy. 1–2 full cycles (2–4 hours per cycle at 8 envs).
- **Expected payoff:** The policy and value head co-optimize under search, self-play avoids overfitting to a fixed opponent.
- **Effort:** 16–24 hours (infra + training + eval).
- **Risk:** High complexity, long training time, potential for instability if the value targets are bad. Pre-registered contingency: if loss diverges or win rate regresses after 500k steps, abort and use the Phase 2 checkpoint.

---

**Phase 4 (ladder validation): 100+ game ladder run** — 🟨 **COMPLETE
2026-07-31, THIRD CONSECUTIVE INCONCLUSIVE READING.**

  Ran the unchanged M7 checkpoint (Phases 1A and 2 both gated negative, so
  nothing newer existed to ship): 100 rated gen1randombattle games, tuned MCTS,
  one session 2026-07-30T21:58 → 2026-07-31T04:34 (~4 min/battle). **Raw 27/100
  (27.0%).** Account `novapool` after: **Elo 1084.0, GXE 32.9%.**

  | | M6 | M7 main | M7 follow-up (7/23) | **Phase 4** |
  |---|---|---|---|---|
  | Raw win rate | — | 30% (n=100) | 42% (n=50) | **27% (n=100)** |
  | Elo | 1017 | 1034.6 | 1101.4 | **1084.0** |
  | GXE | 23.9% | 28.2% | 32.9% | **32.9%** |

  **Verdict per Criterion E:** GXE 32.9% is inside the 25–34% band →
  **inconclusive.** Not the ≥35% win, not the <25% regression.

  **The substantive finding is about the instrument, not the agent.** Phase 4
  ran the *same M7 checkpoint* as the 7/23 follow-up — the model did not change
  — yet raw win rate went **42% → 27%** and Elo went **down 17 points**. That is
  ~1.8 SD, not formally significant, but it is a direct measurement of ladder
  noise and it invalidates the reading recorded on 2026-07-23 that the
  "monotonic 23.9 → 28.2 → 32.9 trend" was evidence of progress. It was mostly
  the ladder wandering. Two compounding defects, both now understood:

  1. **GXE was never a valid per-run gate.** It is an *account-level cumulative*
     statistic — 506 games spanning M6, M7 and M8 on one shared account. It has
     enormous inertia (it did not move at all this run: 32.9 → 32.9) and every
     reading is contaminated by prior runs.
  2. **Raw win rate is not comparable across runs at different Elo.** Climbing
     to 1101 means facing stronger opponents, so a falling win rate is partly
     the ladder working correctly, not the agent degrading.

  Every M6/M7/M8 ladder conclusion rests on these two statistics. Fixing this is
  the reason M9 Phase 1 exists and is gated ahead of any further training spend.

- **Prerequisite:** Phase 0 (reconnect) must be complete.
- **Checkpoint:** Use the best checkpoint from whichever phase gates positive (Phase 1B > Phase 2 > Phase 3).
- **Battery:** Bot evals first (vs Random, vs DamageFirst, 500 battles each) to confirm no regression from M7.
- **Ladder:** ≥100 consecutive rated gen1randombattle games via the reconnect-enabled ladder-bot, tuned MCTS (100/1/0.5).
- **Measurement:** Final Elo, GXE, Glicko-1 ± confidence; raw win rate; compare to M7 (1034.6 / 28.2%).
- **Decision gate:** "Is this better than M7?"
  - ✅ **If GXE ≥35%:** clear win; move to next milestone or stop.
  - 🟨 **If 25% ≤ GXE < 35%:** inconclusive; optional 50-game follow-up to narrow the band.
  - ❌ **If GXE <25%:** regression; write postmortem, consider whether the binding constraint is architectural.

---

### Pre-Registered Success Criteria

**Criterion A (Phase 1A gate):** Speed-ratio A/B run (1–2M steps) shows >+2pp signal vs Random and DamageFirst on the quick 150-battle evals.
- ✅ **Pass:** escalate to Phase 1B or 4 (skip 2/3).
- ❌ **Fail:** escalate to Phase 2.
- **Result 2026-07-28: FAILED.** v3-extended 58%R/61%DF vs v3 control
  61%R/~55%DF (approx) — Random moved −3pp (wrong direction). Phase 1B
  skipped; escalating to Phase 2 (Criterion C next).

**Criterion B (Phase 1B gate, if A passes):** v3-extended full run (5M) beats v3 on both bot opponents by ≥+2pp at n=500 OR ties (≤+1pp, ≥−1pp).
- ✅ **Pass or tie:** escalate to Phase 4.
- ❌ **Fail (regression):** escalate to Phase 2.

**Criterion C (Phase 2 gate, if B fails):** Value-head fine-tuning on self-play targets improves MCTS performance on the existing M7 checkpoint by ≥+3pp vs DamageFirst (tuned MCTS, 200 battles).
- ✅ **Pass:** escalate to Phase 3.
- ❌ **Fail:** abort Phase 3, use M7 checkpoint as baseline for Phase 4.

**Criterion D (Phase 3 gate, if C passes):** Full MCTS value training (after 500k steps) maintains or improves bot evals vs the M7 checkpoint.
- ✅ **Pass:** escalate to Phase 4.
- ❌ **Fail (regression or divergence):** use Phase 2 checkpoint for Phase 4.

**Criterion E (Phase 4, final):** Ladder GXE after 100+ games.
- ✅ **GXE ≥35%:** clear win; M8 complete.
- 🟨 **25–34%:** inconclusive; optional 50-game follow-up.
- ❌ **<25%:** regression; postmortem.

---

### Effort & Duration

- **Phase 0:** 30 min (parallel).
- **Phase 1A (abort on fail):** 4–6 hours (parallel machine).
- **Phase 1B (if 1A passes):** 2–3 hours training + 2 hours eval (parallel).
- **Phase 2 (if 1 fails):** 8–12 hours (includes self-play infrastructure).
- **Phase 3 (if 2 passes):** 16–24 hours (iterative self-play + training).
- **Phase 4:** 4–6 hours eval + 2–8 hours ladder (depending on checkpoint and parallelism).
- **Total (worst case):** ~48 hours spread over 3–4 days. **Most likely path:** Phase 1A → 1B or 2 → 4 = ~12–16 hours.

---

### Unblocks

Depends on result:
- **If M8 succeeds on ladder (GXE ≥35%):** M9 candidate directions — team-specific specialists, multi-format BC, opponent-pool scaling.
- **If M8 inconclusive or regresses:** postmortem on whether randbats variance or opponent-pool diversity (not obs/value targeting) is the binding constraint.

---

## M9: Evaluation Methodology + Data Distribution Hypothesis ⏳ SCOPED

**Status:** ⏳ Scoped 2026-07-31 (awaiting user approval to build)

**Context:** M8 tested two structural improvements on the M7 checkpoint (obs richness Phase 1A, value-head targeting Phase 2) and both failed. Three critical methodological defects also emerged: (1) GXE is account-level cumulative (506 games M6-M8), not per-run; (2) the same M7 checkpoint showed 42% → 27% raw win-rate variance across different ladder runs (~1.8 SD), meaning the prior "monotonic GXE trend" (23.9% → 28.2% → 32.9%) was an over-read; (3) the +3pp gate at n=200 is underpowered (SE on the difference ~3.9pp; a true +3pp effect would be detected only ~1/3 of the time). The surviving hypothesis — that opponent-pool/data distribution (constraint c from the M8 thesis) is the binding constraint — was deprioritized on reasoning alone, never tested on evidence. **M9 makes two critical changes: (1) Fix evaluation methodology so future milestones have a reliable signal; (2) Seriously investigate the data/distribution direction, including whether richer human data + RL beats bot-trained lineage on ladder.**

**Key Facts Informing Scope:**
- M5.5 (human BC + anchored PPO) was the single largest improvement in project history; BC pathway was the only one to beat the bot-trained lineage.
- Data gap: 474 MB raw human replays (gen1ou-heavy) vs 1.7 GB derived; project ladders on gen1randombattle but replays are gen1ou-focused.
- M7 achieved 93% vs Random / 84.2% vs DamageFirst (bot eval) but only 28.2% GXE on ladder — a 65pp gap suggesting format variance or opponent-pool misalignment.
- Constraints (a) and (b) from M8 are now dead. Constraint (c) is the only untested structural lever.

---

### M9 Phases

**Phase 1 (Evaluation Methodology v2):** Fix ladder measurement — ✅ **COMPLETE
2026-07-31.** Delivered `docs/EVALUATION-METHODOLOGY.md` (protocol, power tables,
runbook, reporting template), `scripts/ladder_analysis.py` (Wilson/Newcombe
intervals, session segmentation, heterogeneity test, power tables), and the
instrumentation the Phase 3 paired design requires — `ladder_results.csv` now
carries `run_id`/`account`/`checkpoint`/`opp_rating`/`own_rating`, covered by
`test/tools/ladder-results.test.js` (5 tests).

**Three findings changed the plan:**

1. **There is no measurable ladder drift.** Session heterogeneity across the five
   M7-era sessions is `chi2=4.85, df=4, p=0.303, phi=1.21` — consistent with pure
   binomial sampling. The 42%→27% swing was sample size (±13pp CI at n=50), not
   drift. The paired design's stated justification does not hold; keep it (free,
   equalises the opponent pool) but expect no reduction in required n.
2. **The 7/23 record was incomplete.** Two back-to-back 50-game sessions ran that
   day on the same checkpoint — **24.0%** and **42.0%**. Only the 42% was
   recorded, where it read as progress. A selection effect, not just noise.
3. **A well-powered M7 ladder baseline already existed in the logs:** pooling all
   387 M7-era rated games gives **30.5%, 95% CI [26.1, 35.3]** — already at the
   n≈350 target. Phase 1's planned replication ladder run is therefore
   unnecessary as a *discovery* task; the fresh-account control arm in Phase 3
   covers the design-validation purpose.

**No replication ladder run was needed. Phase 3's cost is unchanged.**

- **Goal:** Establish reliable ladder-strength evaluation so future milestones can trust their results.
- **Scope:**
  1. **Per-run GXE isolation:** Use a fresh ladder account per major checkpoint (not shared). Existing M7 account (novapool, ~506 games) becomes the M7 baseline control; M9 Phase 3 uses a new "m9" account.
  2. **Instrument validation.** Note we already have one replication datapoint, for free: the same M7 checkpoint scored **42% (n=50, 7/23)** and **27% (n=100, 7/31)** — a **15pp swing at zero true effect.** Any "expected ±4–5pp noise" assumption is already falsified; design against the observed number, not an optimistic one. One further same-checkpoint replication on the fresh accounts is worth running to confirm the paired design actually suppresses this drift (that is its whole justification), but it is a *validation of the design*, not a discovery task.
  3. **Required sample size table:** Derive from observed noise. E.g., "To resolve a true +3pp effect at 80% power, need ~400–500 games; for +2pp, ~700–900 games."
  4. **Head-to-head A/B protocol:** Alternate batches of baseline vs candidate on the same account within one session to isolate checkpoint effect from day-to-day variance.
- **Deliverable:** `docs/EVALUATION-METHODOLOGY.md` with runbooks for per-run account setup, replication protocol, and interpreting ladder results (e.g., "GXE=31% at n=150; expected noise ±7pp; lower 95% CI ≈24%; inconclusive but directional").
- **Effort:** 4–6 hours writing + 1–2 hours ladder (replication baseline).
- **Gate:** Methodology doc specifies what we're measuring and with what precision before Phase 3 ladder work starts.

---

**Phase 2 (Data/Distribution Hypothesis Test):** Investigate whether richer/better-aligned human data beats bot-trained lineage on ladder

M5.5 proved that human BC + anchored RL can beat bot-trained policies. M7 trained on bots only (M2, M3.3, M5, M5.5, M7 descendants + DamageFirst heuristic). The remaining question: does richer/better-aligned human data compound the win?

- **Scope:** 4 sub-components (pursue all if time permits; 2c is mandatory):

  1. **2a: Randbats-only BC checkpoint** — ✅ **COMPLETE 2026-07-31, POSITIVE
     RESULT. Format alignment is real, and it is the first positive evidence
     for constraint (c) in the M8 thesis.**

     `bc_mlp_gen1_v3_rb5.pt` trains on gen1randombattle shards only; every
     other knob matches the recipe behind `bc_mlp_gen1_v3.pt` (v3 schema,
     5 epochs, `--min-rating 1300`, `--opp-bc-coef 0.1`, `--value-bc-coef 0.5`).
     Raw policy, no search, n=5,000 per arm:

     | | randbats-only | mixed | difference (95% CI) |
     |---|---:|---:|---|
     | vs Random | **39.5%** (1977/5000) | 33.9% (1697/5000) | **+5.6pp [+3.7, +7.5]** |
     | vs DamageFirst | **33.2%** (1660/5000) | 28.6% (1429/5000) | **+4.6pp [+2.8, +6.4]** |
     | randbats val acc | **54.3%** | 52.7% | +1.6pp (n=60,766) |
     | opp-head acc (vs R) | **24.1%** | 21.2% | +2.9pp |

     Both CIs exclude 0. **The gen1ou half of the corpus was diluting
     gen1randombattle play rather than enriching it** — the reverse of the
     2026-07-16 multi-format decision, which was never A/B'd. Note the effect
     survives *despite* randbats-only seeing 1.10M records per epoch against
     the mixed run's 5.02M: the smaller, aligned corpus wins on less data.

     Counts in `models/checkpoints/m9p2a_ab_results.txt`; runner
     `models/checkpoints/run_m9p2a_ab.sh`; analysis `scripts/bot_eval_ab.py`.
     - Actual effort: ~25 min (BC train 201 s, four n=5,000 evals ~13 min).

  2. **2b: Richer replay corpus** — ❌ **COMPLETE 2026-07-31, NEGATIVE RESULT.
     There is no more gen 1 human data to scrape.** Both formats were mined to
     the end: gen1ou ran to `history exhausted` at page 2029 (scanned 103,436,
     downloaded **634**, skipped_existing 98,263) and randbats managed +1,387 at
     10.4% yield before the remaining archive was shown to be far too thin to
     fill a 30k cap. A full-archive census of gen1ou found **only 10,674
     replays ≥1300 in existence (10.3%)**, against 34,462 rated <1300 and
     **58,300 unrated (56.4%)** — and the corpus already held ~all of the ≥1300
     tier. **Ceiling on gen 1 human data at ≥1300: ~32k replays total** (~10.7k
     gen1ou + ~21.6k randbats). The sub-item below is retained for its
     estimates, all of which proved optimistic; treat it as superseded.
     **The proposed successor task — "harvest the 58,300 unrated gen1ou
     replays" — was investigated 2026-07-31 and is already done.**
     `bc_pretrain_mlp.py:82` keeps unrated records whose battle id is not plain
     `<format>-…`, so every `smogtours-` tour game is already in training. Over
     all 99 gen1ou shards, BC consumes 3,949,828 of 8,451,696 records (46.7%),
     of which **smogtours alone is 2,941,686 — 74% of the training data.** The
     genuinely unused remainder is the *casual* tier (26,791 unrated main-ladder
     replays / 2.12M records, plus 2.38M rated-<1300 records), dropped
     deliberately as weak play. A player-identity filter over the casual tier
     (players also seen in ≥1300 games) would add ~+18%, of uncertain quality —
     costed in IN-PROGRESS.md, not recommended as a priority. **Also relevant to any format decision:** a
     replay-volume census put gen9randombattle at 1,873/day and gen9ou at
     2,407/day versus gen1's 31–44/day (~45×) — gen 9 is where both the data and
     the ladder traffic are, at the cost of rewriting the gen1-hardcoded
     observation layer and discarding the M2→M8 lineage.
     - Run `scripts/scrape_replays.py --backfill --max-replays 50000` to top up gen1randombattle (current corpus: ~20k; backfill should add ~30k more high-Elo games).
     - Optionally: filter Metamon gen1ou replays to ≥1400 Elo only (current corpus is 10k rated ≥1300 + 55k mixed; higher Elo = higher data quality).
     - Adapter converts new logs to trajectory shards.
     - Effort: ~3 hours (scrape/filter) + 1 hour (adapt); cost is disk only, no training.

  3. **2c: BC fine-tune** — Mandatory. ✅ **COMPLETE 2026-07-31 — NEGATIVE.
     Gate passes against M5.5 (+10.5pp vs Random), but the candidate is 6.3pp
     BELOW M7 with the CI excluding 0, so the pre-registered rule sends M7 to
     Phase 3.** 5M steps in ~68 min on the RTX 3080; 20 checkpoints; sweep peak
     64% vs Random at n=500 against M7's historical 73%.

     All arms at n=2,000, same machine and session:

     | arm | vs Random | vs DamageFirst |
     |---|---:|---:|
     | M5.5 (`bcft`, v2) | 52.8% | 43.2% |
     | **M7 (`v3`, control)** | **69.7%** | **59.4%** |
     | 2c final (5.00M) | 63.3% | 51.9% |
     | 2c best-sweep (4.00M) | 62.1% | 53.7% |

     2c − M5.5 = **+10.5pp [+7.4, +13.5]** R, **+8.7pp [+5.6, +11.8]** DF.
     2c − M7 = **−6.3pp [−9.2, −3.4]** R, **−7.5pp [−10.5, −4.4]** DF.

     **M7 replicated at 69.7% (n=2,000) against its historical 70.0% (n=500)**,
     which is independent evidence the eval harness is sound.

     **The finding: 2a's BC advantage inverted rather than compounded.** The
     mixed BC went 33.9% → 69.7% (+35.8pp from RL); the randbats BC went
     39.5% → 63.3% (+23.8pp). The format-aligned checkpoint was the **better
     imitator and the worse RL substrate** — plausibly because the gen1ou half
     buys behavioural breadth that is useless for imitation but valuable as an
     exploration prior and KL anchor over 5M steps. That is a hypothesis.

     **⚠️→✅ Was confounded with training-seed variance and with backend
     (M7 ran on `mps`, 2c on `cuda`). RESOLVED 2026-08-01 by a seed
     replication** (`models/ppo/checkpoints/m9seed/`): re-running M7's recipe
     from the mixed BC with a fresh seed reproduced M7 to **−0.6pp vs Random
     [−3.4, +2.2]** and **−0.3pp vs DamageFirst**, *across the backend change
     too*. Run-to-run spread is under a point, so neither seed nor device
     explains 2c.

     **In the properly controlled A/B — both runs `cuda`, same machine, same
     recipe, BC corpus the only systematic variable — 2c is −8.3pp vs Random
     [−11.1, −5.3] and −6.7pp vs DamageFirst [−9.7, −3.6].** The regression is
     real. **"A format-aligned BC is a better imitator and a worse RL
     substrate" is now established**, and the breadth explanation (gen1ou
     widens the exploration prior and KL anchor) is the surviving hypothesis.

     Side benefit: low run-to-run variance means the project's earlier
     single-run A/Bs hold up. M7-vs-M5.5 stands, and so do the M8 negatives —
     Phase 1A's −3pp and Phase 2's −2.5pp were unlikely to be seed noise.
     Limit: one replication pair bounds the spread as small; it does not put a
     tight interval on it, and must not be quoted as "PPO variance is 0.6pp".

     **2c's premise changed twice and the surviving version is the strong one.**
     As originally written it fine-tuned on "the expanded corpus from 2b" — but
     2b is closed, so no expanded corpus exists and that version of 2c has no
     content. What it tests instead is 2a's finding carried one stage further:
     warm-start the **randbats-only** BC into the anchored-PPO recipe and see
     whether the +5.6pp BC-stage advantage survives 5M steps of RL.

     The recipe is M7's v3 run verbatim — 5M steps, `--rollout-steps 512`,
     `--num-envs 8`, `--opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2"`,
     `--bc-anchor-coef 0.05`, `--value-warmup-steps 200000`, `--opp-coef 0.1`,
     pool seeded with the same M2 and M3.3-best checkpoints. **The BC corpus is
     the only variable**, so 2c is a clean single-variable A/B against M7 rather
     than a confounded comparison. Checkpoints → `models/ppo/checkpoints/m9p2c/`.

     (Had 2a gone the other way, 2c would have warm-started from the same mixed
     BC as M7 and been a seed replication of it — worth knowing for pipeline
     variance, worthless as a test of the hypothesis.)
     - Follow M5.5 anchored PPO recipe: BC warm-start, 5M-step fine-tune with mixed opponents (selfplay/damagefirst/random), pool seeded with M2 + M5.5.
     - Bot evals (**sizes revised by Phase 1**): 20-checkpoint sweep vs Random at 500 battles each (~19 s/checkpoint — the old 150 was ±7pp and could not rank the sweep), then confirmations at **2,000 vs Random and 2,000 vs DamageFirst** on the top candidates (~75 s each, raw policy).
     - Pre-register (**revised by Phase 1**): if this checkpoint beats the M5.5 baseline by **≥+3pp vs Random at n=2,000/arm with the 95% CI on the difference excluding 0**, it's the Phase 3 ladder candidate; else use M5.5. Report DamageFirst alongside but do not gate on a ±2pp DamageFirst bar — that needs 4,948 games/arm. Raw-policy evals cost ~75 s per 2,000 battles, so run the full n.
     - Effort: ~8 hours training (home box, 8 envs, ~3 hours wall) + 2 hours eval.

  4. **2d: Opponent-pool saturation check** — ✅ **COMPLETE 2026-08-01, NULL
     RESULT. Replacing the entire self-play pool with human imitators changed
     nothing.**

     Implemented with one flag and no new code: `--selfplay-pool` pointed at a
     dedicated directory holding the two BC human-imitators, so the self-play
     half of every rollout faced a human-style opponent (one that actually
     *switches Pokémon*, unlike `Random` and `DamageFirst`, which by
     construction never do) for the entire run. Warm-start and anchor were the
     mixed BC per the 2c finding; everything else matched M7/m9seed.

     All arms n=2,000, one session, one machine:

     | arm | vs Random | vs DamageFirst |
     |---|---:|---:|
     | M7 | 70.5% | 58.0% |
     | **m9seed** (matched control) | **68.9%** | **57.0%** |
     | **m9p2d** (human sparring) | **67.9%** | **57.0%** |
     | 2c (randbats BC) | 62.8% | 50.7% |
     | M5.5 | 53.4% | 41.5% |

     **m9p2d − m9seed = −1.0pp [−3.9, +1.9] vs Random and +0.0pp [−3.1, +3.1]
     vs DamageFirst** — both CIs include 0, and the DamageFirst arms tie to the
     battle (1140/2000 each).

     **The carry-forward observation:** m9seed's pool escalates with the agent,
     m9p2d's is frozen and weak (~39%/34% vs Random), **and they tie**. Within
     this recipe the self-play pool's identity and strength appear to do very
     little work. That is evidence against opponent quality as the binding
     constraint and shifts weight toward **model capacity (151,187 parameters)
     and the format's intrinsic team luck**.

     **Limits, pre-registered before the result:** a win would have been
     unambiguous, a non-win is not — the agent likely outgrew the frozen
     partners. But the manipulation was strong (50% of every rollout, whole
     run) and returned exactly zero. And bot evals cannot speak to *human*
     play, which is the real target. The weaker follow-up (BC checkpoints as
     seeds inside the escalating pool) is therefore **low priority**.

  4b. **2d as originally scoped** — the design below was superseded by the
     cheaper implementation above, which needed no new sampler code.
     - Create a "ladder-like pool" by sampling highest-rated players from the scraped human replays.
     - Train a checkpoint with `--opponent-mix "human_randbats=0.7,damagefirst=0.2,random=0.1"` (human pool replaces selfplay).
     - Bot evals: compare vs M7 (which trained on bot descendants + heuristics).
     - If human-sampler checkpoint transfers better to ladder, opponent-pool diversity matters.
     - Effort: ~4 hours (build human-sampler opponent code, train, eval).

- **Deliverable:** Best Phase 2 checkpoint, pre-registered against M5.5 baseline.
- **Gate:** Phase 2 candidate beats M5.5 by ≥+2pp on both bot opponents (n=500), OR use M5.5 for Phase 3 (no failure; fallback is acceptable).

---

**Phase 3 (Ladder Validation with Methodology v2):** Test the Phase 2 candidate reliably

- **Scope:**
**Design constraint (corrects the original Phase 3 draft):** the gate must NOT
be "candidate GXE on a new account vs M7's historical 28.2%/32.9%." That repeats
the exact defect Phase 1 exists to fix — it compares a cumulative account
statistic at n≈100 against one at n≈506, across accounts, across months of
ladder drift. GXE only becomes meaningful again on fresh accounts compared at
**matched game counts in the same time window.**

  1. **3a: Paired concurrent A/B (the core design).** Run the M7 control and the
     Phase 2 candidate on **two fresh accounts, alternating, in the same
     sessions.** Both face the same ladder pool over the same period, so
     day-to-day drift — the confound that produced 42% → 27% on an unchanged
     checkpoint — cancels in the paired difference. Primary endpoint is the
     **difference in raw win rate between arms**, not either arm's absolute GXE.
  2. **3b: Report per-arm Elo/GXE as secondary,** valid only because the accounts
     are fresh and matched on n. Also report each arm's mean opponent Elo, so a
     win-rate difference driven by opponent-strength asymmetry is visible rather
     than hidden.
  3. **3c: Head-to-head bot A/B (n=1000)** — much cheaper than ladder; run first
     to decide whether the candidate is even worth ladder time.

- **Power, stated honestly.** For two arms at p≈0.30, 80% power, α=0.05:
  detecting **+5pp needs ~1,400 games per arm**; **+10pp needs ~350 per arm**;
  **+15pp needs ~160 per arm**. At the measured ~4 min/battle, 350 games/arm is
  **~23 hours of ladder per arm.** This is the real reason every prior milestone
  used an underpowered gate, and it does not go away by wishing.
  **Pre-registered choice: M9 powers for +10pp (~350/arm, ~2 days wall clock
  for both arms).** Effects smaller than 10pp are declared *not measurable by
  this project on the ladder*, and must be judged on bot evals instead.
- **Deliverable:** Paired ladder results on two fresh accounts, reported as a
  difference with a 95% CI, plus per-arm opponent-Elo distributions.
- **Gate (on the paired difference, not on absolute GXE):**
  - ✅ **Difference ≥ +10pp with CI excluding 0:** clear win.
  - 🟨 **CI includes 0:** inconclusive — and with 350/arm this now means the
    effect is genuinely <10pp, which is a *finding*, not a failure to measure.
  - ❌ **Difference ≤ −10pp with CI excluding 0:** regression; postmortem.

---

**Phase 4 (Stopping Decision):** Determine next direction

Context: M2 → M7 achieved 51% → 93% bot-eval improvement (+42pp) but only 23.9% → 28.2% GXE (+4.3pp). The 65pp gap between bot eval and ladder suggests format intrinsic variance or opponent-pool saturation may be the ceiling, not policy quality.

**Phase 1 note (2026-07-31):** the GXE thresholds below are the pre-Phase-1
framing and are retained only for their reasoning. Read them as the paired-
difference gate from Phase 3: succeeds = difference ≥+10pp with CI excluding 0;
inconclusive = CI includes 0; regresses = difference ≤−10pp with CI excluding 0.

- **Scope:**

  1. **If Phase 3 succeeds (GXE ≥35%):**
     - Declare M8/M9 a win; data/distribution was the binding constraint.
     - Recommend M10 direction: team-specific fine-tuning on ladder favorites, multi-format RL (gen1ou, gen1ou-teambuilder), or online learning (accumulate ladder data to retrain).

  2. **If Phase 3 is inconclusive (25–32% GXE):**
     - Measure 95% CI on Phase 3b result + Phase 3a replication baseline.
     - If CI upper bound > 32%: declare "statistically consistent with a real gain"; recommend continued ladder play (accept stalemate, accumulate games for higher n).
     - If CI ≤ 30%: declare gen1randombattle's intrinsic variance too high to resolve <±5pp effects; recommend pivot: (a) accept offline-only agent mode (no ladder validation), (b) switch to a format with less team luck (gen1ou with fixed teams, constructed), or (c) scale compute (AlphaZero-scale self-play if time/resources allow).

  3. **If Phase 3 regresses (<25% GXE):**
     - Postmortem: compare Phase 2 candidate bot evals vs M5.5/M7. If bot evals held, data-quality issue (replay corruption/misalignment). If bot evals degraded, Phase 2 candidate is weaker; use M5.5.
     - Recommendation: pivot to offline-only agent (accept ladder transfer is not fixable), scale compute, or stop project.

- **Deliverable:** Written postmortem + recommendation (continue M10 / accept stalemate / pivot direction / stop).
- **Effort:** 2–4 hours analysis + doc.

---

### Pre-Registered Success Criteria (M9 Global)

| Phase | Criterion | Pass | Fail | Status |
|---|---|---|---|---|
| 1 | Methodology v2 written + instrument capable of the Phase 3 design | Runbook + analysis tool + arm-labelled per-battle log | Doc incomplete | ✅ 2026-07-31 |
| 2a | Randbats-only BC hypothesis test | Randbats BC evals compared vs mixed | Not pursued | ✅ 2026-07-31 — **+5.6pp R / +4.6pp DF at n=5,000, both CIs excluding 0** |
| 2d | Opponent-pool saturation | Human-sampler pool beats bot-lineage pool | No measurable difference | 🟨 2026-08-01 — **NULL: −1.0pp R / +0.0pp DF vs matched control, both CIs including 0** |
| 2b | Richer replay corpus assembled | ≥50k games total, adapter shards created | ❌ closed — archive exhausted | Closed |
| 2c | Fine-tune beats M5.5 **and is not worse than M7** | ≥+3pp vs Random at n=2,000/arm, CI on the difference excluding 0, and not measurably below M7 | <+3pp, or below M7 with CI excluding 0 | ❌ 2026-07-31 — **+10.5pp vs M5.5 but −6.3pp vs M7, CI excluding 0. Phase 3 ladders M7.** |
| 3a | Paired ladder A/B run on two fresh accounts | ≥350 games/arm, arms alternated within sessions | Under-powered or unpaired | Protocol |
| 3b | Phase 2 candidate ladders | Paired difference ≥+10pp, CI excluding 0 | ≤−10pp, CI excluding 0 (regression) | Gate |
| 4 | Stopping decision made | Postmortem + recommendation written | Analysis incomplete | Deliverable |

**Gate 2c was amended again on 2026-07-31, before any 2c result existed.** The
Phase-1 rewrite fixed its sample size but left it scoring against **M5.5**,
while Phase 3's control arm is **M7** — the stronger agent, and the shipping
one. A candidate could therefore clear the gate against M5.5 while still losing
to M7, and Phase 3 would spend ~2 days of ladder on an arm already known to be
weaker than its own control. Both baselines are now measured at n=2,000 and the
full decision table lives in `IN-PROGRESS.md` → Phase 2. Nothing was loosened:
every arm is still judged on a difference with a CI.

**Gates 2c/3a/3b were rewritten by Phase 1 (2026-07-31).** The originals were
underpowered or measured the wrong quantity:
- 2c's "≥+2pp at n=500" needs **2,213 games/arm** vs Random and **4,948** vs
  DamageFirst to detect reliably — the same defect as M8's "+3pp at n=200".
  Raw-policy evals run at ~27 battles/s (200 battles in 7.4 s), so n=2,000/arm
  costs ~75 seconds; there was never a reason to run 500.
- 3a's "variance <±5pp GXE" is not measurable: GXE is account-cumulative and
  moved 32.9% → 32.9% over an entire 100-game run.
- 3b's "GXE ≥32%" compares a fresh account at n≈350 against a 506-game account —
  exactly the defect Phase 1 exists to fix. The endpoint is the paired
  difference in raw win rate.

See `docs/EVALUATION-METHODOLOGY.md` for the derivations and the runbook.

---

### Critical Dependencies & Parallelization

- **Phases 1 & 2 can overlap:** Phase 1 (methodology doc + replication baseline ~50 games = ~1–2 hours ladder) runs while Phase 2 training happens on home box (8-hour 5M PPO run).
- **Phase 3 depends on Phase 2:** Requires the Phase 2 candidate checkpoint (delivered at end of Phase 2 evals).
- **Phase 4 depends on Phase 3:** Requires Phase 3 ladder result.

---

### Effort & Duration

- **Phase 1:** 4–6 hours doc + 1–2 hours ladder = **5–8 hours total**.
- **Phase 2:** 3 (2a) + 3 (2b) + 8 (training) + 2 (evals) + 4 (2d) = **12–20 hours** (subset as time permits; 2c is mandatory).
- **Phase 3:** 2–8 hours ladder depending on win rate.
- **Phase 4:** 2–4 hours analysis + doc.
- **Total:** **25–40 hours over 3–4 days**, mostly parallelizable.

**Most likely path (if 2c is mandatory and 2a/2d are deferred):** Phase 1 (5h) + Phase 2c (10h) + Phase 3 (4h) + Phase 4 (3h) = **~22 hours**.

---

### Unblocks & Next Milestones

- **If M9 Phase 3 succeeds (GXE ≥35%):** M10 candidate directions: (a) team-specific fine-tuning (ladder favorites), (b) multi-format RL (gen1ou, gen1ou-teambuilder), (c) online learning (live ladder data retraining), or (d) AlphaZero-scale self-play if compute available.
- **If M9 Phase 3 inconclusive/regresses:** Decision to accept stalemate (gen1randombattle as-is with current agent), pivot to a different format, or invest in fundamental architectural upgrade (large models, online learning, or AlphaZero-scale).

---

## Architecture Reference

```
Battle state
    │
    ▼
extractFeaturesStructured()      [M2; v2 schema M3.4]
    │  (12 Pokémon tokens × 65 features each; 77 with --obs-v2)
    ▼
MLP-PPO shared trunk (flattened obs → 128-dim)  [M2; transformer retired per M3.2]
    ├──▶ Policy head (128 → 9 logits)
    ├──▶ Value head  (128 → 1 scalar)
    └──▶ Opp head   (128 → 9 logits)   [M5]
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
| M5: Opponent Modeling | ✅ | Thesis negative (head sampler = policy sampler, −2.2pp); side finding: **new best agent** — M5 ckpt + policy-sampler MCTS 72.6% DF / 86.0% R | M6 |
| M5.5: Human Replay Data + BC | ✅ | **Positive — new best agent.** BC on human replays → anchored PPO fine-tune → tuned MCTS = 90.6% R / 79.2% DF (prior best 86.0/72.6); h2h vs M5 best 78.4% (392/500) | M6 |
| M6: Server Integration | ✅ | Live ladder bot shipped (raw + MCTS via `BattleSim.fromTracked`); 100/100 clean rated battles, ≤579ms/move. **External read: Elo 1017 / GXE 23.9% — bottom of the human ladder** (MCTS 21/100 vs raw ~13%) | M7 |
| M7: Observation Schema v3 | ✅ | **New best agent (bot evals)** — tuned MCTS 93.0% R / 84.2% DF, +2.4/+5.0pp vs prior best. **Ladder: inconclusive** — Elo 1034.6 / GXE 28.2% (M6: 1017/23.9%), lands in the pre-registered 25–34% noise band; directionally up (+9pp raw win rate) but not a confirmed win. **50-game follow-up (2026-07-23): still inconclusive** — 21/50 raw (42%), Elo 1101.4 / GXE 32.9%, 2.1pp under the ≥35% win bar; trend over 150 games monotonic (23.9→28.2→32.9) | M8 |
| **M8: Value-Head Targeting + Ladder Infra** | ✅ COMPLETE 2026-07-31 — all bets negative or inconclusive | **Contingency escalation, both technical bets spent, neither paid.** Phase 0 (ladder-bot websocket reconnect) ✅. Phase 1A (obs refinement, speed-ratio dim) ❌ Criterion A failed (Random −3pp) → Phase 1B skipped. Phase 2 (AlphaZero value targeting) ❌ Criterion C failed (base 82.5% → fine-tuned 80.0%, −2.5pp vs a ≥+3pp bar) → Phase 3 skipped, and ❌ **failed again on replication 2026-07-30** at exactly 160/200 with targets re-collected against DamageFirst, eliminating distribution mismatch as the explanation. **Notable:** the fine-tune fixed the value head in-distribution both times (R² 0.00 → 0.20, then −0.13 → +0.19; the PPO-trained head scored *below* a constant predictor) and none of it transferred to play strength. Phase 4 (ladder validation on the unchanged M7 checkpoint) 🟨 **inconclusive 2026-07-31** — 27/100 raw, Elo 1084.0 / GXE 32.9%, third consecutive reading inside the 25–34% band. **Key finding is methodological:** the same checkpoint swung 42% → 27% raw between runs, so the prior "monotonic GXE trend" was an over-read and GXE (account-level, cumulative over 506 games) was never a valid per-run gate. | M9 |
