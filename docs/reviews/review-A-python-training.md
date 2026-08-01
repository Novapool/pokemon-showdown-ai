# Review A — Python training path: hard-coded assumptions & silent-corruption surfaces

Scope: `models/ppo/{ppo_agent,train,trajectory_buffer}.py`, `models/bc_pretrain_mlp.py`,
`models/value_finetune.py`, `models/collect_value_data.py`, `models/{gym_client,vec_gym_client}.py`.
Read-only; nothing modified.

Ranked by severity. Findings 1–3 are the ones I'd act on before the width probe runs.

---

## 1. `train.py` has no `--lr`. The learning rate is unreachable from the trainer, and on a warm start it comes from whatever the *BC script* baked into the checkpoint. **TIME-CRITICAL for the width probe.**

**What is hard-coded.** `PPOAgent.__init__` defaults `lr=3e-4` (`models/ppo/ppo_agent.py:43`) and
builds `Adam` from it at `ppo_agent.py:95`. `train.py`'s argparse (lines 46–247) never exposes it,
and the from-scratch construction passes only four kwargs — `obs_size`, `hidden_size`, `device`,
`opp_coef` (`train.py:409-410`). On `--resume` / `--pretrain-checkpoint` the LR is *also* not
settable: `PPOAgent.load` does `cls(**hparams)` (`ppo_agent.py:426`), so the optimizer is rebuilt
at the LR stored in the checkpoint's hparams, and then `load_state_dict(checkpoint["optimizer"])`
(`ppo_agent.py:447`) overwrites the param-group LR with the stored one anyway. There is no code
path in this repo that trains PPO at any LR other than 3e-4.

Worse, the stored value is a fiction on BC checkpoints: `bc_pretrain_mlp.py` builds its **own**
optimizer at `--lr` (default 1e-3, `bc_pretrain_mlp.py:215,258`) and never touches
`agent.optimizer`, so `bc_mlp_gen1_v3*.pt` carries `hparams["lr"] = 3e-4` — a number that describes
neither the BC run nor any deliberate PPO choice.

**What it blocks.** The width probe. `models/checkpoints/bc_mlp_gen1_v3_h512.pt` already exists;
`h=512` is 801k params vs 151k at 128 (5.3×, per the module docstring). It will be fine-tuned at
3e-4 because there is no way to do otherwise. Adam at fixed LR on a 5.3× wider net takes a
materially larger step in function space; the standard expectation is that a wider net wants a
*smaller* LR. **A null or negative result from the width probe will be uninterpretable** — exactly
the "we ran out of width" / "it was an LR artifact" ambiguity the probe exists to remove. Same
applies at the BC stage: `bc_mlp_gen1_v3_h512.pt` was presumably trained at the default 1e-3 tuned
for 128 width, and nothing records what it actually used.

**Severity.** Bounds the *next* experiment, and would make its result non-conclusive. This is the
single highest-value fix on the list.

**Fix.** Add `--lr` to `train.py`; pass to `PPOAgent(...)` on the from-scratch path; on the
load paths, after `PPOAgent.load(...)`, do
`for g in agent.optimizer.param_groups: g["lr"] = args.lr` (must be *after* the optimizer
state load at `ppo_agent.py:447`, or it is silently reverted) and print the effective LR.
Then run the width probe as a 2×2 (128/512 × 3e-4/1e-4) or at minimum 512 at two LRs.
**Effort: ~15 lines, 20 minutes.** The extra GPU cost is one additional 70-minute arm — cheap
insurance against a seventh inconclusive milestone.

---

## 2. The training log's only progress metric is uninterpretable under `--opponent-mix`, and there is no diagnostic that would reveal a broken run.

**What is assumed.** `train.py:649-667` logs exactly `Step / Win rate (rollout) / Loss` (+ opp-label
coverage). But with `--opponent-mix` — the standard recipe for every run from M3.4 onward — the
opponent family is **resampled every rollout** (`train.py:478`) and **is not printed**. So
consecutive log lines report win rates against `random`, then `damagefirst`, then a self-play
checkpoint, with no label. The training curve is a mixture of three incomparable series. `Loss` is a
single scalar mixing surrogate + 0.5·value MSE + entropy bonus + opp CE + BC-anchor KL, so it moves
for five unrelated reasons.

Nothing logs: policy entropy, approximate KL to the old policy, clip fraction, value loss,
explained variance, the BC-anchor KL magnitude, gradient norm, or which self-play checkpoint was
sampled. `agent.update()` (`ppo_agent.py:271-389`) computes entropy, ratio and (when anchored) the
anchor KL, and throws all of them away, returning only the summed loss.

**What it distorts.** Every milestone. This is precisely the "trains successfully for 70 minutes
while silently doing the wrong thing" surface: an entropy collapse, a dead BC anchor, a
value head diverging, or a clip fraction pinned at 1.0 would all produce a log that looks normal.
Six hypotheses were adjudicated on end-of-run aggregate win rates with no mid-run instrumentation
capable of distinguishing "hypothesis is false" from "this run was pathological".

**Severity.** Does not by itself invalidate a past conclusion, but it means no past run can be
retrospectively diagnosed — the evidence was never recorded. High value, near-zero cost.

**Fix.** Have `update()` return a dict (entropy, approx_kl, clip_frac, value_loss, policy_loss,
anchor_kl); log it plus `family=` and `opp=<pool checkpoint basename>` on every line. Keep
`float(...)` back-compat if anything else consumes the return.
**Effort: ~40 lines across `ppo_agent.py` and `train.py`, ~1 hour.**

---

## 3. `--opponent-mix` silently discards a large fraction of episodes before they reach a terminal reward — and biases what survives toward short battles.

**What is structural.** `_switch_family` (`train.py:365-383`) resets **every** env whenever the
sampled family differs from the current one, "abandoning in-flight episodes". Family is resampled
per rollout (`train.py:478`). With the standard `selfplay=0.5,damagefirst=0.3,random=0.2`, the
probability the family *changes* each rollout is `1 − (0.25+0.09+0.04) = 0.62`.

Now the arithmetic: `--rollout-steps 512` is a total across envs, so at `--num-envs 8` each env gets
**64 steps per rollout** (`train.py:575-609`) against a ~50-step average battle. So roughly one
episode per env per rollout, and ~62% of rollouts end with every env's in-flight battle thrown away.

The reward is ~99% terminal: ±1.0 at `|win|`, ±0.01 per faint, +0.0001 per status
(`sim/tools/pokemon-gym.ts:1247-1259`). Truncated episodes contribute only the tiny shaped
component plus a bootstrap from the agent's own (untrusted) value head.

Two consequences, both invisible:
- **Signal density.** Mixed-opponent runs see materially fewer completed battles per environment
  step than fixed-`--opponent` runs, which never truncate (`_switch_family` is a no-op when the
  family can't change). Every comparison of an M3.4+ mixed-opponent run against an M2/M3
  fixed-opponent baseline is confounded by this, independent of opponent diversity.
- **Length bias.** Truncation is length-biased: long battles are disproportionately the ones
  destroyed. The surviving training signal over-represents short battles. In Gen 1, long battles
  are exactly the ones where switching and positioning decide the game.

Also note the logged rollout win rate counts only *completed* episodes, so this loss is invisible
in the log (see finding 2).

**Severity.** Plausibly distorts the whole mixed-opponent line of milestones. At minimum it means
`--opponent-mix` is not the "pure opponent-diversity" variable it is documented as.

**Fix (cheap diagnostic first).** Log `episodes_completed` and `transitions_truncated` per rollout —
that alone tells you the size of the effect for ~5 lines. Then either (a) sample the family per
*env* rather than globally and only reset an env at its own episode boundary, or (b) hold the family
fixed for K rollouts. Option (b) is ~5 lines and reduces the switch rate by 1/K.
**Effort: diagnostic 30 min; fix 1–2 hours.**

---

## 4. `--opp-coef` is accepted and silently ignored on `--resume`, while the startup banner reports the value you asked for.

`train.py:391` — on `--resume`, `agent = PPOAgent.load(...)` and `agent.opp_coef` comes from the
checkpoint's hparams. The `--pretrain-checkpoint` branch explicitly repairs this
(`train.py:406`: `agent.opp_coef = args.opp_coef`); the `--resume` branch does not.
`agent.update(merged)` is called with no `opp_coef` argument (`train.py:637`), so `self.opp_coef`
wins. Meanwhile `train.py:444` prints `opp_coef={args.opp_coef}` — **the banner and the training log
assert a value the run is not using.**

Compounding: `args.opp_coef != 0.0` also drives checkpoint-dir routing (`train.py:316-319`), the
already-recorded gotcha that sent the M8 Phase 1A v3-extended run into `checkpoints/opp/`
(`MILESTONES.md:1442`). One flag with two effects, one of which is a no-op on one code path.

**Severity.** Any resumed run where `--opp-coef` was passed trained with the checkpoint's λ and
logged the flag's λ. Whether that bit anything historically depends on whether any resume ever
changed λ — worth grepping the run logs.
**Fix.** Set `agent.opp_coef = args.opp_coef` in the resume branch too (1 line), or, better, make
the banner print `agent.opp_coef` rather than `args.opp_coef` everywhere.
**Effort: 5 minutes.**

---

## 5. Zero seeding anywhere in the trainer — so every A/B pays a variance penalty it does not need to pay.

`train.py:350` is `np.random.default_rng()` with no seed; there is no `torch.manual_seed`, no
`--seed` flag, and `GymClient` (`gym_client.py:129-141`) never passes a seed to the bridge. Contrast
`collect_value_data.py:345` and `value_finetune.py:125`, which both have `--seed` and use it.

The docs treat this as a feature ("re-running an identical command is an independent draw",
`docs/TRAINING-COMMANDS.md`), and it is fine for replication — but it also means **no two arms of an
A/B can share randomness**. Team draws, opponent-family sequences, self-play pool draws and weight
init are all independent between arms. Common random numbers is the standard variance-reduction
trick here and would directly attack the problem that has been forcing n≈4,300/arm evaluations
(`docs/EVALUATION-METHODOLOGY.md`).

**Severity.** Bounds every future comparison; makes small true effects (the +2pp gates these
milestones keep using) expensive or impossible to resolve.
**Fix.** Add `--seed`; seed `torch`, the numpy rng, and — the part with real leverage — plumb a
per-env seed through `GymClient`/`gym_bridge.js` so both arms of an A/B face the same team draws.
The bridge/TS side is another reviewer's file, but the Python side is ~10 lines.
**Effort: Python 30 min; end-to-end with bridge seeding, 2–3 hours.**

---

## 6. Every PPO hyperparameter except width is a constructor default that `train.py` cannot reach — and they are frozen into checkpoint lineages.

Never exposed by any flag: `clip_eps=0.2`, `value_coef=0.5`, `entropy_coef=0.01`,
`max_grad_norm=0.5`, `ppo_epochs=4`, `batch_size=64` (`ppo_agent.py:44-49`), and
`gamma=0.99`, `lam=0.95` (`trajectory_buffer.py:15`, constructed argument-free at `train.py:335`).
Network *depth* is also fixed: the trunk is literally two `Linear`s (`ppo_agent.py:67-72`) with no
`n_layers`; there is no LayerNorm, no orthogonal/small-gain init on the policy head (PyTorch default
`Linear` init throughout), no LR schedule, and no entropy annealing.

Specific sub-findings worth separating out:
- **Entropy is a fixed 0.01 for 5M steps with no anneal and no logging.** Combined with finding 2,
  an entropy collapse at, say, step 1M would be undetectable. "Is the agent still exploring at 5M
  steps?" has never been a question anyone could answer from the artifacts.
- **`--num-envs` is not a pure throughput knob.** `--rollout-steps` is a *total* across envs
  (`train.py:575-609` increments by `num_envs`), so doubling `--num-envs` halves the per-env GAE
  horizon (64 → 32 steps) and doubles the truncation rate. Two runs at different `--num-envs` are
  running different algorithms. Nothing warns.
- **Defaults are inherited, not re-applied.** Because hparams are stored and `load()` reconstructs
  from them (`ppo_agent.py:426`), changing a default in `ppo_agent.py` today would apply to
  from-scratch runs but *not* to resumed or BC-warm-started ones. Arms with different checkpoint
  ancestry can silently differ in hyperparameters. Nothing compares hparams across arms.

**Severity.** Bounds the experiment space in the same way the width literal did. None of these have
been varied in nine milestones, so "is the PPO recipe itself the constraint?" has never been
testable.
**Fix.** A single `--hp key=value,...` passthrough into `PPOAgent(**overrides)` plus
`--gamma`/`--gae-lambda` into `TrajectoryBuffer`, and (important) an override that also applies on
the load paths. **Effort: ~30 lines, 1 hour.** Depth (`--n-layers`) is a larger change (~1 hour)
because `trunk` shape must round-trip through hparams.

---

## 7. No run manifest is written. Device, argv and git SHA exist only in a tmux scrollback.

`PPOAgent.save` (`ppo_agent.py:395-407`) stores state dicts + hparams and — deliberately, and
correctly for portability — no device. But nothing else records it either: no `run_config.json`, no
argv, no git SHA, no torch version, no host. This is exactly the gap that let the `mps` vs `cuda`
confound reach a published A/B and cost an extra 70-minute run.

Note `collect_value_data.py:357-372` already does this right — it writes `meta.json` next to its
shards with checkpoint, opponent, sims, c_puct, seed, obs mode. `value_finetune.py:213-226` does the
same. The PPO trainer, the one that consumes the most GPU time, is the only script that doesn't.

**Severity.** Does not invalidate a past conclusion by itself, but it is the enabling condition for
the class of confound that already burned a run once.

**Fix.** At `train.py` startup, dump `vars(args)` + `str(agent.device)` + `torch.__version__` +
`platform.node()` + `git rev-parse HEAD` to `checkpoint_dir/run_config.json`. Optionally also stash
that dict inside `save()` under a `"run"` key so a checkpoint is self-describing.
**Effort: ~15 lines, 20 minutes.** Highest value-per-line item in this report after finding 1.

---

## 8. The self-play pool silently degrades to narcissistic self-play when the seed checkpoints aren't there.

`_sample_opponent` (`train.py:273-287`) globs `ppo_step_*.pt` from `pool_dir`; if the glob is empty
it falls back to a frozen copy of the *current* policy, with no warning. Pool seeds are `tier 2`
files that don't travel with `git push` (`docs/TRAINING-COMMANDS.md` explicitly warns: "rsync them
or the run silently trains from scratch with an empty pool"). The failure mode is documented in
prose and unguarded in code. Nothing logs the pool contents at startup or which checkpoint each
rollout drew.

Because the m9p2c-style A/Bs are described as "M7's recipe with only the warm-start swapped", an
arm whose seeds failed to rsync differs in a second variable — and nothing in the log would show it.

**Severity.** Could have silently un-matched an arm in any of the M9 comparisons. Cannot be checked
retrospectively because pool composition was never logged.
**Fix.** Print the resolved pool listing (count + basenames) at startup; add
`--require-pool N` that hard-fails if fewer than N seeds are present when `selfplay` is in the mix.
**Effort: ~10 lines, 20 minutes.**

---

## 9. The reward's stalling penalty is asymmetric by construction, and the MCTS forward model doesn't implement it at all.

The reward function is: ±0.01 per faint, +0.0001 per status inflicted on p2 (no symmetric penalty
for own status), ±1.0 on win/loss (`sim/tools/pokemon-gym.ts:1247-1259`), **minus `0.001 ×
turnCount` applied once at terminal** (`pokemon-gym.ts:706` single-seat, `:869` dual-seat), then
clipped to `[-1, +1]` two lines below each.

The clip makes the stalling penalty **one-sided**. On a win, terminal reward ≈ `+1.0 + 0.01 −
0.001·T`, which for any realistic T is below the clip and so the penalty is *fully applied*. On a
loss, terminal reward ≈ `−1.0 − 0.01 − 0.001·T`, which is always ≤ −1 and so is **clipped back to
−1.0 — the penalty vanishes entirely**. Net effect: the agent is punished for taking a long time to
*win* but not for taking a long time to *lose*. That is a direct gradient against stalling,
pivoting, and playing for position — the Gen 1 skill set — and it compounds with the length-biased
truncation in finding 3. Neither `0.001` nor the clip is reachable from any flag, so "does the
stalling penalty cap strategic play?" has never been a hypothesis anyone could test.

Second, `sim/tools/battle-sim.ts` — the forward model MCTS searches over — implements
`parseProgressLines` but applies **no** terminal turn penalty and **no** `[-1,1]` clip (grep for
`turnCount`/`0.001` in that file returns only bookkeeping). So MCTS optimizes a slightly different
objective than the policy/value head were trained on. This bears directly on the M8 Phase 2 null
result: `value_finetune.py --target mc` builds targets from *env* rewards (penalty included) while
`--target root` builds them from *sim* root Q (penalty excluded), and the two are presented as
interchangeable options on the same `--target` flag (`value_finetune.py:119`).

**Severity.** Bounds a hypothesis nobody has been able to state. The env/sim mismatch is a live
candidate explanation for M8 Phase 2. The reward files are TypeScript and belong to another
reviewer — flagging here because the *consequence* (an untestable reward hypothesis, and two
incommensurable `--target` options) lands in my scope.
**Fix.** Expose the shaping constants as gym-bridge flags so they are varyable; separately, make
`BattleSim` apply the identical terminal penalty + clip, or drop the penalty from both.
**Effort: TS side 1–2 hours; the "is shaping the constraint?" experiment is then a normal A/B.**

---

## 10. `bc_pretrain_mlp.py` never learned about schema v3-extended — so the M8 Phase 1A A/B could not have been warm-start-matched.

`bc_pretrain_mlp.py:51-52` hard-codes exactly two obs widths (`OBS_V2_SIZE = 924`,
`OBS_V3_SIZE = 1032`) and `--obs-v3` is the only schema flag (line 203). There is no 1044 path, and
`grep` finds no v3-extended handling in the file or in `replay_adapter_cli.js`. Meanwhile
`train.py --obs-v3-extended` exists and enforces `obs_size == 1044` on any `--pretrain-checkpoint`
(`train.py:398-405`). **Therefore the M8 Phase 1A speed-ratio arm could not have been BC
warm-started**, while the v3 control lineage (`bc_mlp_gen1_v3.pt` exists in
`models/checkpoints/`) could. If the two arms differed in warm-start, the recorded conclusion
("richer observations are not the constraint", `MILESTONES.md:1444-1455`) is confounded by
initialization, not just by the 128-width bottleneck already identified.

I could not confirm from the milestone text how the 2M-step v3-extended arm was actually launched —
**this is worth 10 minutes of checking `checkpoints/opp/` training logs before anyone re-uses that
conclusion.** The structural point stands regardless: the RL trainer gained a schema the BC trainer
never did, so schema experiments and warm-start experiments are silently coupled.

**Severity.** Potentially invalidates the M8 Phase 1A negative result (already independently
suspect for the width-bottleneck reason). Definitely blocks any future schema extension from being
tested on equal footing.
**Fix.** Replace the two `OBS_*_SIZE` constants with a single `--obs-schema {v2,v3,v3ext}` →
`12 * TOKEN_DIM` mapping shared with `gym_client.py` (which already has all four constants at
`gym_client.py:31-35`). Add a startup assertion that the shard width matches.
**Effort: ~20 lines, 30 minutes** (plus regenerating v3-ext shards, which is the real cost).

---

## Lower-severity, listed for completeness

- **`bc_pretrain_mlp.py:335` saves every epoch, overwriting.** The final checkpoint is always the
  *last* epoch, never the best-validation one, and validation is computed (line 328) but never used
  to select. Five epochs of BC on 3.9M records at 1e-3 with no weight decay and no early stopping —
  the shipped `bc_mlp_gen1_v3.pt` may be an overfit epoch. Cheap fix: keep the best-val checkpoint,
  or at least write `bc_..._epoch{N}.pt`. **~10 lines.**
- **`bc_pretrain_mlp.py:73` uses `shards[:val_shards]` as validation** — the *first* shards in
  sorted order, which for chronologically-emitted replay shards means the oldest games. The BC
  validation number is therefore an out-of-time estimate, not an i.i.d. one. Not wrong, but it isn't
  what "held-out accuracy" is usually read to mean. **~3 lines to randomize.**
- **`value_finetune.py:210` saves `agent.optimizer`, not the fine-tune optimizer.** The written
  checkpoint carries Adam moments matching the *pre*-fine-tune value head. Only matters if someone
  `--resume`s PPO from a `_valft` checkpoint. **1 line.**
- **`train.py:583-591` drops an errored env's transition but doesn't mark the episode boundary.**
  The env is auto-reset by `VecGymClient`, so the buffer's next push for that slot is from a fresh
  episode while the previous transition still has `done=False` — GAE will propagate value across the
  boundary. Rare (bridge errors only), but it is a silent GAE corruption rather than a loud one.
  **~5 lines: push a synthetic `done=True` on the prior transition, or clear that env's buffer.**
- **`merge_buffers` (`trajectory_buffer.py:195-196`) normalizes advantages over the whole batch**,
  which under `--opponent-mix` is a single family per update. So each family's gradient is rescaled
  to unit variance regardless of how informative it was; `--opponent-mix` weights control frequency
  only, and any natural difficulty weighting between opponents is normalized away. Defensible, but
  it is an unstated assumption.
- **`ppo_agent.py:371-376` re-runs `self.trunk(obs_b)` a second time** for the BC-anchor KL instead
  of reusing the `features` already computed in `evaluate_actions`. Pure waste (~15% of a step in
  anchored runs), no correctness impact.
- **Files with nothing significant to report:** `models/gym_client.py` and
  `models/vec_gym_client.py` are clean for this review's purpose — the schema constants are
  centralized (`gym_client.py:31-35`), the slicing contract is explicit and guarded
  (`gym_client.py:84-87`), and the mutual-exclusion checks are real. `models/collect_value_data.py`
  is the best-parameterized script in scope (every MCTS knob is a flag, it writes `meta.json`, it
  has a `--seed`); its only issue is the reward mismatch inherited from finding 9. It is worth
  holding up as the template the PPO trainer should follow.

---

## Recommended order of action

1. **Finding 1 (`--lr`)** — before the width probe launches; otherwise that run produces a fourth
   ambiguous result.
2. **Finding 7 (`run_config.json`)** — 20 minutes, permanently closes the class of confound that
   already cost a run.
3. **Finding 3's diagnostic** (log completed vs truncated episodes) and **finding 2's metrics** —
   both are logging-only, both retire large blind spots, together ~1.5 hours.
4. **Finding 10's check** — 10 minutes of log archaeology decides whether the M8 Phase 1A negative
   result should be re-opened alongside the width question.
5. Findings 4, 8 — a handful of lines each, remove two "flag lies to you" surfaces.
