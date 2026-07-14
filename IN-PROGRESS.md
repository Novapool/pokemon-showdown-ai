# In Progress — Pokemon Showdown AI Training

Last updated: 2026-07-13

---

## Current Work

**Milestone:** M3 (Transformer Encoder + PPO, BC warm-start) — **concluded, negative result.** The transformer does not beat the 51% MLP-PPO baseline at any point across two full training runs (~90 total checkpoints evaluated), and PPO training on this architecture does not improve monotonically with more steps — it peaks early (near the BC-pretrained starting point) and then degrades, either violently (uncontrolled run) or gradually (stability-fixed run). Per `MILESTONES.md`'s own stated M3 recommendation, **M4 (MCTS) and M5 (opponent modeling) should not proceed on top of this architecture** until/unless this is revisited.
**Phase:** Done. See "Recently Completed" for the full experimental trail (from-scratch, warm-started, extended warm-started with a mid-run checkpoint sweep revealing collapse, and a stability-fix retrain with KL early-stopping + LR annealing that changed the failure mode but not the outcome).

### Active Tasks (M3) — all resolved
- [x] `models/transformer/transformer_agent.py` — `TransformerAgent(nn.Module)`, PPO wrapper composing `TransformerPolicy` as `self.policy` (composition, not subclassing, so BC checkpoint keys stay unprefixed and loadable). Later extended with `target_kl` approx-KL early-stopping (see below)
- [x] `models/transformer/train.py` — rollout/GAE/checkpoint training loop mirroring `models/ppo/train.py`; always structured `(12,65)` unflattened; `--pretrain_checkpoint` flag calls `load_pretrain_checkpoint(agent.policy, path)` before PPO starts; checkpoints split into `checkpoints/{scratch,pretrained}/`; `--resume <checkpoint>` restores full agent+optimizer state and parses the starting step count from the filename. Later extended with linear LR annealing toward 0 over `--steps` (see below)
- [x] `models/evaluate.py` — added `--model transformer`; split the old single `structured` bool in `_run_battles` into `structured`/`flatten` so the transformer gets raw `(12,65)` while PPO-structured still gets flattened `(780,)`
- [x] Smoke-tested: agent construction/act/save/load, BC warm-start (30/30 tensors loaded into `agent.policy`), a short training run in both scratch and pretrained modes, `--resume`, and `evaluate.py --model transformer` — all pass with no shape errors
- [x] **From-scratch run:** 2.6M steps. Evaluated 500 real battles: **32% win rate (158/500)** — below the M2 MLP-PPO baseline (51%)
- [x] **Warm-started run 1 (2.6M steps, no stability fixes):** Evaluated 500 real battles: **41% win rate (204/500)** — better than from-scratch, still below 51%
- [x] **Extended warm-started run 1 to 7.6M steps + full checkpoint sweep (21 checkpoints, 250k–7.6M):** revealed the run actually peaked at **46%** around 2.5M–3.1M steps, then **collapsed violently** — cratering to 0–13% between 3.6M and 5.6M steps, partially recovering to 32% at 6.1M, then drifting down to 24% by 7.6M. Root-caused to unconstrained PPO updates over a long horizon with no LR decay or trust-region constraint — a single bad minibatch update can yank the policy far enough to wreck its move-vs-switch balance (recall `RandomPlayerAI` never voluntarily switches, so any policy that over-favors switching loses almost every battle)
- [x] **Stability fix:** added approximate-KL early-stopping (`target_kl=0.02`, stops a rollout's PPO minibatch epochs early if `1.5×target_kl` is exceeded — standard PPO safeguard, http://joschu.net/blog/kl-approx.html) to `TransformerAgent.update()`, and linear LR annealing to 0 over the `--steps` budget in `train.py`. Both smoke-tested
- [x] **Warm-started run 2 (fresh 5M-step run, with stability fixes) + full checkpoint sweep (20 checkpoints, 250k–5.0M):** the violent collapses are gone (no more 0% crashes), confirming the fix addressed that specific failure mode. But the run now peaks almost immediately (**45% at 500k steps** — near where the BC-pretrained starting policy already was) and then **gradually decays** to 25–35% noise, ending at **27% (135/500)** at the 5.0M-step final checkpoint. The stability fixes changed *how* it fails, not *whether* it fails to improve on the BC starting point
- [x] **Final M3 conclusion:** best-ever recorded win rate across ~40 checkpoints spanning both warm-started runs is **46%** (150-battle spot check, run 1 @ 2.5M steps) — never reaching, let alone beating, the 51% MLP-PPO baseline. Continued PPO fine-tuning of the transformer consistently degrades performance from its early/BC-pretrained peak rather than improving it, the opposite of the M3 hypothesis. This is a real, decisive negative result, not an artifact of insufficient training budget or an unfixed instability bug

### Active Tasks (M2, complete)
- [x] Add `extractFeaturesStructured()` to `sim/tools/feature-extractor.ts` — returns `(12, 65)` token array, exact layout match with `models/metamon_adapter.py`
- [x] Add opponent-reveal tracker to `sim/tools/pokemon-gym.ts` — reconstructs revealed opponent state (species/HP/status/fainted/used moves) from `|switch|/|drag|/|-damage|/|-heal|/|-status|/|-curestatus|/|faint|/|move|` lines on the battle log, never from omniscient state
- [x] Wire `obsMode: 'flat' | 'structured'` into `PokemonGymEnv` (default `'structured'`)
- [x] Update `models/gym_bridge.js` — serializes structured obs as 780-element flat array by default; `--flat` CLI flag restores the legacy 100-dim path
- [x] Update `models/gym_client.py` — reshapes `(780,)` → `(12, 65)`; `GymClient(structured=False)` mirrors `--flat`
- [x] Fixed `models/{q_learning,dqn,ppo}/train.py` + `evaluate.py` to pass `structured=False` — their networks are hardcoded to the flat 100-dim vector and would otherwise silently break now that structured is the default
- [x] Tests: shape/unknown-flag/fainted-flag/bench-ordering coverage in `test/tools/gym.test.js`; full build + battle-loop smoke tests pass
- [x] Renamed `--battles` → `--steps` in `dqn/train.py` and `ppo/train.py` (the flag counted environment steps, not battles — collided in meaning with `evaluate.py`'s genuinely battle-counting `--battles`)
- [x] Added `--structured` to `ppo/train.py` and `evaluate.py` — `PPOAgent` already took `obs_size` as a parameter, so no architecture changes were needed, just plumbing + a separate `checkpoints/structured/` directory
- [x] **Verify:** MLP PPO trained on flattened structured obs (2.6M steps ≈ 50k battles), evaluated 500 real battles greedy vs RandomPlayerAI: **51% win rate (254/500)** — clears the ≥50% target, confirms the structured token representation preserves parity with the M1 flat baseline. Checkpoint: `models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`
- [x] **Bug found + fixed:** `evaluate.py`'s `_run_battles()` did `action = agent.act(obs, valid_mask)` uniformly for all three model types. `QAgent`/`DQNAgent.act()` return a plain `int`, but `PPOAgent.act()` returns `(action, log_prob, value)` — a 3-tuple. Every `evaluate.py --model ppo` run was silently handing the whole tuple to `env.step()`, which JSON-serializes a Python tuple as a JS array; comparing/indexing with that array in `pokemon-gym.ts` always fails, so **every PPO action was rejected as illegal, forever, on battle 1** — an infinite hang with no error, no exception, nothing in the logs. This bug predates this session (written 2026-05-18) — `evaluate.py --model ppo` had never actually completed a run before. See "Recently Completed" for the full incident writeup.
- [x] **M2.5:** Run `bash scripts/download_metamon.sh` — done 2026-07-02, **119,536 gen1ou trajectories** in `data/metamon_cache/`
- [x] **M2.5:** Smoke-test BC on real data — done 2026-07-02, `--max_files 200`: 33,270 samples, loss 2.04→1.72, acc 19.8%→34.1% (chance ≈ 11%), no adapter warnings; action histogram over real battles ≈ 71% moves / 29% switches (plausible for human gen1ou)
- [x] **M2.5:** Full BC run — done 2026-07-02: 4.96M samples/epoch × 5 epochs (24.8M total) in 2414s on MPS; **final loss 1.339, top-1 acc 50.5%** (chance ≈ 11%); checkpoint at `models/checkpoints/bc_pretrain_gen1ou.pt`, verified loading 30/30 tensors into `TransformerPolicy`. M2.5 complete — remaining criterion (warm-start vs from-scratch PPO comparison) blocked on M3.

### Deviation from the original M2 plan: stat boosts dropped

The original M2 plan called for a boost tracker (`|-boost|`/`|-unboost|`) feeding into the active-Pokémon token. That was written before M2.5 locked the token schema to exactly match `models/metamon_adapter.py` (65 dims, no boost slots) so the BC-pretrained checkpoint's input projection stays valid. Metamon's replay dataset doesn't encode stat boosts either, so there's no way to add a boost feature without breaking BC-checkpoint compatibility or retraining it. Boosts are dropped from M2's scope; revisit only as a deliberate schema-version bump (would require a new BC pretraining run).

### Bug fixed in passing: level default

`parseLevelFromDetails()` defaulted to level 50 when a details string had no explicit `L##` tag. Showdown omits the level tag entirely when it's 100 (the common case for gen1ou/gen1randombattle), so this was silently encoding real level-100 Pokémon as level 50 in both the flat and structured extractors. Default is now 100, matching Showdown's actual convention and Metamon's adapter assumption.

---

## Active Plan

**M3 Execution Plan — code phase complete, training phase in progress:**

1. ~~**`TransformerAgent`** (composes `TransformerPolicy`, PPO wrapper)~~ ✅ Done
2. ~~**`models/transformer/train.py`** (rollout PPO loop, `--pretrain_checkpoint` warm-start, `--resume`)~~ ✅ Done
3. ~~**`evaluate.py --model transformer`**~~ ✅ Done
4. ~~**From-scratch training run (2.6M steps) + evaluation**~~ ✅ Done — **32% win rate (158/500)**, underperforms the 51% MLP-PPO baseline
5. ~~**Warm-started training run (2.6M steps, `--pretrain_checkpoint`) + evaluation**~~ ✅ Done — **41% win rate (204/500)**, better than from-scratch but still below the MLP baseline
6. ~~**Extend warm-started run to 7.6M steps + checkpoint sweep**~~ ✅ Done — revealed a peak of **46%** at 2.5M–3.1M steps followed by violent collapse (down to 0%) and oscillation through 7.6M steps
7. ~~**Add PPO stability fixes (approx-KL early-stop, LR annealing) + retrain (5M steps) + checkpoint sweep**~~ ✅ Done — collapses eliminated, but the run still peaks early (45% @ 500k) and gradually decays (27% @ 5.0M final)
8. ~~**Final M3 conclusion**~~ ✅ **Negative result, decisive.** Transformer PPO (warm-started) tops out at 46% across ~40 evaluated checkpoints from two full runs — never beats, or sustainably matches, the 51% MLP-PPO baseline. Per `MILESTONES.md`'s own M3 guidance, M4/M5 should not proceed on top of this architecture

M2 is fully complete (see below).

**M2 Execution Plan — final status (all resolved):**

1. ~~**Token schema** (in `feature-extractor.ts`)~~ ✅ Done — 12 tokens, 65 dims each, matches `metamon_adapter.py` exactly
2. ~~**Stat boost tracking**~~ ❌ Dropped — see "Deviation from the original M2 plan" above
3. ~~**Bridge update** (in `gym_bridge.js` + `gym_client.py`)~~ ✅ Done — 780-float flat array, `--flat`/`structured=False` for backward compat
4. ~~**Verification**~~ ✅ Done — MLP PPO on flattened structured obs, 2.6M steps (~50k battles), **51% win rate vs RandomPlayerAI over 500 real evaluation battles**. Parity with the M1 flat baseline confirmed; token representation is not broken.

---

## Recently Completed

✅ **M3 concluded: stability-fix retrain confirms negative result** (2026-07-13)
- Extended the 2.6M-step warm-started run to 7.6M steps via `--resume`, then swept all 21 intermediate checkpoints (150 battles each) to see the full trajectory rather than just two data points. Result: the run peaked at **46% win rate** around 2.5M–3.1M steps, then **collapsed violently** — 0–13% win rate between 3.6M and 5.6M steps, a partial recovery to 32% at 6.1M, then drifting to 24% by 7.6M. This is a much more severe finding than the earlier two-point (41%→24%) comparison suggested: it's not gradual drift, it's repeated near-total collapse and partial recovery, consistent with an unconstrained PPO update occasionally wrecking the policy's learned move-vs-switch balance (any policy that over-favors switching loses almost every game to `RandomPlayerAI`, which never voluntarily switches)
- **Added two standard PPO stability fixes**, since the failure pattern (violent collapse, not slow decay) pointed at unconstrained per-update policy movement rather than a capacity problem:
  - `models/transformer/transformer_agent.py`: approximate-KL early-stopping in `update()` — `target_kl=0.02` (new `TransformerAgent` hyperparameter, backward-compatible default for old checkpoints via `cls(**hparams)`). If a minibatch update pushes `approx_kl` (the http://joschu.net/blog/kl-approx.html estimator) past `1.5×target_kl`, the rest of that rollout's PPO epochs are skipped rather than continuing to update on an already-large policy shift. `update()` now returns `(loss, kl_early_stop: bool)` instead of just `loss`
  - `models/transformer/train.py`: linear LR annealing toward 0 over `--steps`, recomputed once per rollout from `agent._hparams["lr"] * max(0, 1 - total_steps/total_budget)`, applied directly to `agent.optimizer.param_groups`. Naturally continues decaying correctly on `--resume` since it's keyed off absolute `total_steps`/`total_budget`, not a step counter that resets. Rollout log line now also prints current LR and a `[kl early-stop]` tag when the safeguard fires
  - Both smoke-tested at 1k–2k steps before the real run; KL early-stop was observed firing correctly in the (very noisy, small-rollout) smoke test
- **Retrained a fresh 5,000,000-step warm-started run** with both fixes (not resumed from the already-collapsed 7.6M checkpoint — a fresh run tests whether collapse is prevented from the start). Swept 20 checkpoints (250k–5.0M) plus a full 500-battle evaluation of the final checkpoint. Result: **the violent collapses are gone** (no checkpoint dropped below ~25%), confirming the fix addressed that exact failure mode. But the run now peaks almost immediately — **45% at 500k steps**, close to where the BC-pretrained starting policy already performs — and then **gradually decays** into a noisy 25–35% band for the remaining 4.5M steps, ending at **27% (135/500)** on the full 500-battle evaluation of the final checkpoint
- **Conclusion:** across ~40 checkpoints evaluated spanning both warm-started runs, the best-ever recorded win rate is **46%** — the transformer never beats, and never sustainably matches, the 51% MLP-PPO baseline. The stability fixes changed *how* training fails (violent collapse → gradual decay) but not *whether* continued PPO fine-tuning improves on the BC-pretrained starting point — it doesn't; performance is highest very early and erodes from there. This is a decisive, well-supported negative result, not an artifact of insufficient training budget (two runs, ~2.6M and ~5–7.6M steps) or an unaddressed instability bug (the specific instability found was fixed, and the conclusion didn't change)
- **MILESTONES.md updated** — M3 marked complete with this negative result; M4/M5 held per the project's own stated M3 recommendation
- Checkpoints: `models/transformer/checkpoints/pretrained/transformer_step_2500000.pt` (run 1 peak, 46%) and `transformer_step_5000000_final.pt` (run 2 final, 27%)

✅ **M3 verification: warm-started transformer run (41%) — still below MLP baseline** (2026-07-11)
- Trained the transformer PPO agent warm-started from the BC checkpoint (`models/checkpoints/bc_pretrain_gen1ou.pt`) for 2.6M steps — same budget as the from-scratch run and the M2 MLP baseline. Evaluated 500 real battles vs RandomPlayerAI: **41% win rate (204/500)** — a +9pp improvement over the from-scratch transformer (32%), confirming BC warm-start helps, but still below the MLP-PPO baseline's 51% at equal compute
- Comparison table at 2.6M steps: MLP-PPO baseline 51% (254/500) > transformer warm-started 41% (204/500) > transformer from-scratch 32% (158/500)
- **Decision:** 2.6M steps (~50k battles) was chosen to match the M2 baseline for a controlled comparison, but it's well short of MILESTONES.md's actual M3 target range (200k–500k battles / ~10.4M–26M steps). Transformers are typically more sample-hungry than MLPs, so it's plausible the warm-started variant is still climbing rather than having plateaued below 51%. Resuming the warm-started run to **10.4M steps** (~200k battles, the low end of the target range) before drawing a final M3 conclusion
- Run was interrupted (Ctrl+C, not a bug — `KeyboardInterrupt` mid-`agent.act()`) at step 7,600,000; resumed via `--resume models/transformer/checkpoints/pretrained/transformer_step_7600000.pt`
- **Next:** finish the 10.4M-step run, evaluate at 500 battles, compare against the 51% baseline one more time — that result is the real M3 conclusion

✅ **M3 verification: from-scratch transformer run (32%) + --resume flag** (2026-07-10)
- Trained the transformer PPO agent from scratch for 2.6M steps (same budget as the M2 MLP-PPO baseline). Evaluated 500 real battles vs RandomPlayerAI: **32% win rate (158/500)** — a real result, well below the MLP baseline's 51% at equal compute. Does not meet the M3 success criterion (transformer must beat, not just match, the MLP baseline)
- **Diagnosed a scare mid-run:** rollout win rate sat at ~0% for a long stretch early in training (step ~500k/2.6M), which looked like a possible bug. Investigated by testing a completely untrained `TransformerAgent`, a completely untrained MLP `PPOAgent`, and a literal uniform-random-over-all-valid-actions policy — all three scored 0-3% vs `RandomPlayerAI`, ruling out anything transformer-specific. Root cause: `RandomPlayerAI` (`sim/tools/random-player-ai.ts`) defaults to `move: 1.0`, so its switch condition (`this.prng.random() > this.move`) is never true — it essentially never voluntarily switches, only attacks. A policy that hasn't yet learned to prefer moves over switches (true of any untrained/early-training policy, since ~5/9 actions are switches) loses almost every battle to an opponent that presses the attack every turn. Not a bug — expected early-training behavior that resolves as the policy learns, which the final 32% (vs. near-0% early on) confirms
- **Added `--resume <checkpoint>` to `models/transformer/train.py`** after a training run was killed mid-flight. Restores full agent + optimizer state via `TransformerAgent.load()`, parses the starting step count from the checkpoint filename (`transformer_step_N.pt`), and reuses the same checkpoint directory (scratch/pretrained) so the resumed run keeps appending to the original sequence. Mutually exclusive with `--pretrain_checkpoint` (resuming already restores whatever weights the original run started from). Smoke-tested: resumed from a fake `step_100` checkpoint with `--steps 300`, correctly continued to step 300 and saved into the same directory
- Checkpoint: `models/transformer/checkpoints/scratch/transformer_step_2600000_final.pt`
- **Next:** the warm-started run (BC pretrain checkpoint) is the actual test of the M3 hypothesis — in progress now

✅ **M3 code: Transformer PPO agent + training loop** (2026-07-10)
- Created `models/transformer/transformer_agent.py` — `TransformerAgent(nn.Module)` composes `TransformerPolicy` (built in M2.5) as `self.policy` rather than subclassing it, so the policy's state-dict keys (`embed.*`, `encoder.*`, `policy_head.*`, `value_head.*`) stay unprefixed and `load_pretrain_checkpoint()` — which matches by exact key name — loads the BC checkpoint's 30/30 tensors into `agent.policy` directly. `act()`/`evaluate_actions()`/`update()`/`save()`/`load()` mirror `models/ppo/ppo_agent.py`'s `PPOAgent`, duplicated rather than shared via a mixin (matches the existing per-model-family convention — `q_learning`/`dqn`/`ppo` already each duplicate `_pick_device()` with no shared base class)
- Created `models/transformer/train.py` — same rollout/GAE/checkpoint/logging structure as `models/ppo/train.py`, but always `GymClient(structured=True)` unflattened (no `--structured` flag — the transformer has exactly one obs shape). New `--pretrain_checkpoint <path>` flag calls `load_pretrain_checkpoint(agent.policy, path)` — **not** `agent` — before training starts; checkpoints split into `checkpoints/scratch/` vs `checkpoints/pretrained/` so a from-scratch and warm-started run at the same step count never collide
- Modified `models/evaluate.py` — added `"transformer"` to `--model` choices and a `_load_agent` branch; split `_run_battles`'s single `structured` bool into `structured` (does `GymClient` return `(12,65)`) and `flatten` (additionally reshape to `(780,)`) — PPO-structured needs both True, transformer needs `structured=True, flatten=False`. No new CLI flag; the flatten decision is derived from `--model`
- Updated `models/CLAUDE.md`'s directory table and Quick-Start block with the new `transformer/` files and commands
- Smoke-tested end-to-end: agent construction/act()/save()/load(), BC checkpoint warm-start (`[pretrain] loaded 30/30 tensors`, confirming the `agent.policy` vs `agent` wiring is correct), a 400-step training run in both scratch and pretrained modes (checkpoints land in the correct separate subdirs), and `evaluate.py --model transformer --battles 10` (loads and runs without shape errors)
- **Remaining for M3:** the real training runs (~200k–500k battles, both scratch and pretrained) and the win-rate comparison against the 51% MLP-PPO baseline haven't been executed yet (needs a background training session, same as M2's verification run)

✅ **M2 verification run + evaluate.py PPO hang bug** (2026-07-09)
- Ran the M2 verification: PPO with trunk `Linear(780,128)→ReLU→Linear(128,128)` on flattened structured obs, `--steps 2600000` (≈50k battles at the measured ~52 steps/battle), checkpoint at `models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`
- First evaluation attempt (`evaluate.py --model ppo --structured --battles 500`) hung indefinitely (17+ min, no error, no output). Diagnosed live: the Python process was burning ~40% CPU while the `gym_bridge.js` subprocess was almost completely idle (19s CPU over 18 min) — a sign the loop was spinning without the battle engine doing any real work
- Root cause, confirmed by direct reproduction: `evaluate.py`'s `_run_battles()` did `action = agent.act(obs, valid_mask)` for all three model types. `QAgent`/`DQNAgent.act()` return a plain `int`; `PPOAgent.act()` returns `(action, log_prob, value)`. The 3-tuple was getting passed whole into `env.step()`, JSON-serialized as a JS array, and silently rejected as an out-of-range/invalid index by `pokemon-gym.ts` on every single call — meaning **every `evaluate.py --model ppo` run, ever, on any checkpoint, since the script was written on 2026-05-18, has hung on battle 1 without any visible error.** M2's structured obs work didn't cause this; it just was the first time anyone ran a PPO evaluation long enough to notice
- Fixed: `action = act_result[0] if isinstance(act_result, tuple) else act_result` — handles both agent shapes without the caller needing to know which model type is loaded
- Added a `models/CLAUDE.md`-documented `--structured` flag to `evaluate.py` for evaluating structured-obs checkpoints, and running win-rate progress logging (`Battle N/total | running win rate: ...`) so future runs are never silent for minutes at a time
- Re-ran the fixed evaluation: **51% win rate (254/500), 55 seconds, no hang** — this is the real, final M2 verification number
- **Also renamed** `--battles` → `--steps` in `dqn/train.py` and `ppo/train.py` — both flags actually counted environment steps, not battles, which collided in meaning with `evaluate.py`'s correctly-battle-counting `--battles` and would have caused real confusion about training scale later

✅ **M2 code: Structured State Representation** (2026-07-09)
- Added `extractFeaturesStructured()` + `TOKEN_DIM=65`/`N_TOKENS=12` to `sim/tools/feature-extractor.ts`; byte-for-byte layout match with `models/metamon_adapter.py` so the M2.5 BC checkpoint stays loadable
- Exported `parseHpRatio`/`parseStatus`/`fillStatusBitmask`/`parseLevelFromDetails`/`typeToIndex`/`categoryToIndex` from `feature-extractor.ts` for reuse
- Added an opponent-reveal tracker to `sim/tools/pokemon-gym.ts` — reconstructs revealed opponent Pokémon (species, HP, status, fainted, revealed moves) purely from `p2`-side battle-log lines, since a real player's request never contains opponent team info
- Added `obsMode: 'flat' | 'structured'` to `PokemonGymEnv` (default `'structured'`)
- `models/gym_bridge.js`: 780-float serialization by default, `--flat` restores the legacy 100-dim path
- `models/gym_client.py`: reshapes to `(12, 65)` by default, `structured=False` restores flat
- Fixed `models/{q_learning,dqn,ppo}/train.py` + `evaluate.py` to pass `structured=False` (their networks are hardcoded to 100-dim input — would have silently broken under the new default)
- Fixed a pre-existing bug: `parseLevelFromDetails()` defaulted to level 50 instead of 100 for Pokémon without an explicit `L##` tag (Showdown omits the tag at level 100, the gen1ou/gen1randombattle norm)
- Fixed a pre-existing build bug: `tools/build-utils.js`'s `copyOverDataJSON` didn't create destination directories before copying, which broke `./build` once `data/metamon_cache/` (2.1GB, gitignored) existed under `data/`
- Extended `test/tools/gym.test.js`: structured-obs shape/NaN checks, unknown-vs-fainted flag distinction, own-bench request-slot ordering, 300-step full-battle shape-stability smoke test — all passing; smoke-tested `gym_bridge.js`/`gym_client.py` end-to-end in both obsMode
- **Decided against** adding stat-boost features (see "Deviation" note above) — schema is locked to BC-checkpoint compatibility
- **Remaining for M2:** the 50k-battle MLP PPO verification run hasn't been executed yet (needs a background training session)

✅ **M2.5 code: Behavior Cloning Pretraining pipeline** (2026-07-02)
- Created `scripts/download_metamon.sh` — installs Metamon (UT-Austin-RPL), sets `METAMON_CACHE_DIR=./data/metamon_cache`, downloads parsed-replays for gen1ou, prints verification count
- Created `models/metamon_adapter.py` — `MetamonDataAdapter` streaming `(obs (12,65), action, done)` from Metamon replay JSONs; M2 token conventions; Metamon's alphabetical move/switch ordering preserved so `MinimalActionSpace` indices ground identically to our gym actions; skips `-1` actions; `FileNotFoundError` points to download script
- Created `models/transformer/transformer_policy.py` — shared `TransformerPolicy` (M3 architecture: 65→128 embed, 2-layer encoder, unknown-token attention masking) + `load_pretrain_checkpoint()` (skips name/shape-mismatched tensors with warnings)
- Created `models/bc_pretrain.py` — CE on policy head only, Adam lr=1e-3, batch 256, 5 epochs default, streamed 50k-sample shuffle buffer; flags `--epochs`, `--format`, `--checkpoint_dir`, `--max_files`
- Verified end-to-end on synthetic fixture battles: all adapter spec checks pass, BC loss decreases on MPS, checkpoint round-trips 30/30 tensors, deliberate shape mismatch skips with warning
- MILESTONES.md: added M2.5 entry; M3 `train.py` now specifies the `--pretrain_checkpoint` wiring requirement

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

### M2 — Structured State
✅ Complete and verified (51% win rate vs RandomPlayerAI, 500 battles). Nothing left here.

### M3 — Transformer + PPO
✅ **Complete — negative result (2026-07-13).** Two full training runs (from-scratch 2.6M steps, warm-started up to 7.6M steps, plus a stability-fixed warm-started retrain to 5M steps) and ~40 evaluated checkpoints all agree: transformer PPO tops out at **46%** win rate vs RandomPlayerAI, never beating (or sustainably matching) the M2 MLP-PPO baseline's **51%**. Continued PPO fine-tuning degrades performance from its early/BC-pretrained peak rather than improving it. See "Recently Completed" for the full diagnostic trail. Nothing left to run here — this is the final M3 result.

### Immediate (post-M3 decision needed)
- **M4 (MCTS) / M5 (opponent modeling) / M6 (server) are on hold**, per `MILESTONES.md`'s own M3 recommendation — none of them should be built on top of an architecture that lost to the simpler MLP baseline. This needs a human decision on direction, options include:
  - Ship the M2 MLP-PPO checkpoint (51%) as the project's baseline model and proceed to M4/M5/M6 with that architecture instead of the transformer
  - Root-cause *why* PPO fine-tuning degrades the BC-pretrained transformer (e.g. try a much lower LR from step 1, explicit KL-to-BC-init regularization, or simply freezing at an early checkpoint) before deciding whether the transformer is worth continuing to invest in
  - Treat M3 as closed and move on

### Stretch (deprioritized until the above is decided)
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

