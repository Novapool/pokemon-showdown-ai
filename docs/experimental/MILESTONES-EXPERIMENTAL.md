# Experiment Milestone Plan: Hierarchical Action Policy + Move Embeddings

**Branch:** `experiment/hierarchical-action-policy`
**Rescoped:** 2026-08-03
**Status:** 🧪 Ready to build

---

## Overview

Four phases (Phase 0–3), each with a pre-registered gate, kill criterion, and wall-clock estimate. Phase 2 (observation schema change) invalidates all checkpoints and triggers full retraining. See `docs/experimental/hierarchical-action-policy.md` for design rationale.

---

## Phase 0: Non-Greedy Decision Probe

**Status: ✅ COMPLETE (2026-08-03) — results in `phase0-results.md`.**
Gate 1 (accuracy gap) passes: +17.5pp randbats / +12.0pp gen1ou for BC,
+32.0 / +25.4 for PPO, battle-clustered CIs excluding 0. Gate 2 as literally
written does not pass; it was mis-specified and was amended after seeing the
data — the amendment is recorded in the results doc. Headline finding: **RL,
not the observation, is what destroys switching** (BC switches 26.8% on the
subset, PPO 6.0%, on the identical v3 observation).

**Goal:** Diagnostic baseline. Measure whether the agent's greedy play is actually a problem, before building anything.

**Scope:**
- Filter `data/replay_trajs/` BC shards for positions where a rated human chose a **0-BP status move or switch while a ≥90-BP damaging move was legal**.
- Score M7 v3 checkpoint on this set (greedy decoding) vs full set.
- Measure the agent's **greediness rate**: on the non-greedy subset, how often
  does it pick the high-BP attack the human declined?
- Estimate ceiling: what fraction of human decisions are recoverable from v3 observations?

> **Corrected 2026-08-03.** An earlier draft of this phase gated on "human
> agreement must exceed the agent's." That is incoherent — the human choice *is*
> the label, so human agreement is 100% by construction. It also named the
> Phase 1 heuristic bot as the non-neural baseline, inverting the phase order.
> Both are replaced by the greediness rate below, which needs nothing Phase 1
> builds. Note a max-base-power baseline is *also* useless here: it scores 0% on
> this subset by construction, since the subset is defined by the human
> declining the high-BP move.

**Deliverables:**
- CSV: {subset, top-1 accuracy, N, 95% CI}
- Greediness rate on the non-greedy subset, with CI
- Ceiling estimate (per-subset category breakout)

**Pre-Registered Gate:**
- **Accuracy gap (full set − non-greedy subset) must be ≥3pp**, CI excluding 0
  (target ≥500 subset positions; report achieved n and do not gate if underpowered).
- **Greediness rate must exceed 50%** — i.e. on positions where a rated human
  declined the big attack, the agent takes it more often than not. This is the
  direct measurement of the degeneracy the branch exists to fix.

**Kill Criterion:** Gap <3pp **or** greediness rate ≤50% → the agent is not
behaving like a max-BP picker and "it plays greedily" is not our problem.
Document and stop; the observation-poverty case would then need to rest on
something other than move selection.

**Wall-Clock:** ~2–3 hours (Mac, no training).

**Dependencies:** None. Runs on existing data and M7 checkpoint.

---

## Phase 1: Gen 1 Damage Calculator + HeuristicAI

**Goal:** Build an evaluation instrument that can discriminate observation improvements. DamageFirstAI never switches; a real heuristic bot will, enabling better measurement.

**Scope:**
- Closed-form Gen 1 damage formula (turn order, accuracy, KO check).
  - **Caveat:** Random DV range (0–15) creates a few-percent uncertainty band in randbats; gen1ou damage estimates are computable to similar precision.
  - No external library; use game rules directly from `Dex` and base stats.
- `HeuristicAI` strategy:
  - Max expected damage (damage × accuracy) vs each legal move.
  - Switch when incoming best-move damage exceeds a threshold *and* a resisting bench member exists.
  - Recover below HP threshold (e.g., ≤30%).
  - Avoid Hyper Beam if target dies anyway.
  - Falls back to max-damage move as tiebreaker.
- Measure M7 v3 agent vs HeuristicAI at **n=2,000 raw-policy battles** per opponent.

**Deliverables:**
- `sim/tools/heuristic-ai.ts` — HeuristicAI class (extends RandomPlayerAI).
- `sim/tools/damage-calculator.ts` — Gen 1 damage formula (exported utility).
- Bot-eval run: agent (M7 v3) vs HeuristicAI, n=2,000, greedy decoding, CI estimate via `bot_eval_ab.py`.

**No pass/fail gate — this phase calibrates the instrument.** The deliverable is
a number to compare against later, not a decision. Report the M7 v3 agent's
win rate vs HeuristicAI with CI (`scripts/bot_eval_ab.py`, n=2,000, greedy,
raw policy), and freeze the heuristic's parameters once measured so later
phases move the agent, not the baseline.

**Instrument-validity criterion (this is the real check):** HeuristicAI must be
*discriminating* — the agent's win rate against it should land meaningfully
below the ~93% it scores against Random/DamageFirst. If the agent also beats
HeuristicAI ~90%+, the heuristic is too weak to serve as the missing rung and
Phase 1 has not delivered its purpose; strengthen it (switching logic first —
that is what DamageFirst lacks) before proceeding.

**Kill Criterion:** HeuristicAI cannot be made discriminating without becoming a
search agent → drop it as an eval rung and note that bot evals stay uninformative
about human-facing gains, which raises the evidential weight Phase 4 must carry.

**Wall-Clock:** ~1 day (Mac for code, ~2–3 hours for bot eval).

**Dependencies:** Phase 0 must gate positive (or be run anyway).

---

## Phase 2: Observation Schema (v4)

**Goal:** Add damage-derived features, move type one-hot, move-id, and defensive type effectiveness. **This invalidates all checkpoints.**

**Scope:**

| Feature | Source | Dims | Per-slot |
|---|---|---:|:---:|
| Move identity | Dex.mod('gen1').moves LUT | 1 | ✅ |
| Move type (one-hot) | Replace `typeIdx/20` ordinal | 15 | ✅ |
| Base power + accuracy (existing) | Keep as-is | 2 | ✅ |
| Expected damage (as % of active's current HP) | `damage / current_hp` | 2 | ✅ |
| P(OHKO) | `accuracy` if OHKO move else 0 | 1 | ✅ |
| Turns-to-KO | `ceil(active_hp / expected_damage)` | 1 | ✅ |
| **Defensive type effectiveness** | Opponent active's best move vs this token's type | 4 | ✅ opponent slot |
| Incoming damage (on bench) | Best vs each bench slot | 5 | ✅ bench members |
| Speed order | `speed_ratio` with incoming best damage | 1 | active only |
| Species ID | Dex.mod('gen1').species LUT | 1 | ✅ |
| Base stats (5: HP, Atk, Def, SpA, Spe) | Species LUT | 5 | ✅ |
| Trapping flag | `maybeLocked` in state | 1 | ✅ |
| Recharge flag | `|-cant|…recharge` parser | 1 | ✅ |

**Design Decisions:**
- Unrevealed opponent moves: draw prior over movepool per species (already computable).
- Unrevealed bench moves: `<unknown>` token in move embedding (deferred to Phase 3 if needed).
- All damage estimates use turn-1 rolls (min/max roll separate cells if precision needed).

**Implementation Path:**
1. Schema design & dimension calculation: 1 day (Mac).
2. `sim/tools/feature-extractor.ts` v4 path + LUTs: 1 day (Mac).
3. `sim/tools/replay-adapter.ts` v4 path for BC retraining: 0.5 day (Mac).
4. Smoke test (4k-step BC training): 2–3 hours (Mac).
5. BC retraining:
   - Gen1OU corpus: ~2–3 hours (homebox, parallelizable with code work).
   - Gen1 randbats corpus: ~2–3 hours (homebox).

**Pre-Registered Gate — BC ablation, flat head throughout.**

This phase isolates *the observation's* contribution. The head does not change
here, so any movement is attributable to the schema. Same corpus, same
optimizer, same step count; only the features change:

| Arm | Change from v3 |
|---|---|
| baseline | current MLP on v3 obs (known: **55.5%** randbats val @ H=512) |
| +onehot | move type as 15-dim one-hot instead of `typeIdx/20` |
| +damagefeats | expected-damage / P(OHKO) / turns-to-KO / incoming-damage dims |
| +moveid | move-id dim (embedding concatenated to hand features) |
| v4-full | all three together |

Report top-1 / top-3 held-out accuracy on the **full** set *and* on Phase 0's
non-greedy subset, per arm.

- **Gate: `v4-full` must beat baseline by ≥2pp top-1 on the non-greedy subset.**
- Run the single-change arms regardless of whether the gate passes — attribution
  is the point. If `v4-full` wins but no single arm does, say so rather than
  crediting the most interesting one.

**Kill Criterion:** `v4-full` gains <2pp on the non-greedy subset → the schema
is not supplying decision-relevant information; do not spend Phase 3/4 compute.
Divergence or NaN is a bug, not a result — debug and re-run.

**Wall-Clock:** ~3–4 days total (code 2–3 days Mac; BC retraining 4–6 hours homebox, parallelizable).

**⚠️ COORDINATION WITH M11 v4:** Move-id dim (observation index + LUT) is a shared dependency. If M11 Phase 1 lands v4 on master first, coordinate on:
- LUT structure (how to encode move IDs).
- Dimension ordering (ensure move_id slot is compatible).
- BC re-conversion path in replay-adapter.ts.
- If M11 lands first, pull move-id design from there; otherwise, design it here and offer it upstream.

**Dependencies:** Phase 1 must complete (instrument ready). Phase 0 should gate positive.

---

## Phase 3: Pointer Head + Auxiliary Category Head

**Goal:** Replace `Linear(H, 9)` with action-conditioned scoring head. Add auxiliary category prediction (no hard routing).

**Scope:**
- Pointer head: `logit_i = ⟨W_q h(s), [e_i ; f_i]⟩` where:
  - `e_i` = move embedding (168×d, d=16 tentative).
  - `f_i` = move hand-features from v4 (6+10 dims per slot).
  - `W_q` projects trunk to query space.
  - Logits computed only over legal actions; mask as in original.
- Auxiliary head: category prediction (damage / status / switch) with λ=0.1 CE loss.
  - No hard routing; provides interpretability only.
- BC and PPO training on v4 observations with new heads.

**Implementation Path:**
1. Embedding table + pointer head: 1 day (models/ppo/ppo_agent.py).
2. Auxiliary head + loss weighting: 0.5 day (models/ppo/train.py).
3. BC training on v4 shards: 2–3 hours (homebox).

**No RL in this phase.** All RL lives in Phase 4 — running a 5M-step PPO arm
here and a 2M-step arm there would pay the RL bill twice for one question.

**Pre-Registered Gate — BC, `+pointer` vs Phase 2's `v4-full` flat head.**

The comparison is against the flat head **on the same v4 schema**, never against
the v3 baseline. Comparing to v3 would credit the head with the observation
win Phase 2 already bought — the misattribution this experiment is most exposed
to.

- **Gate: `+pointer` beats `v4-full` flat head by ≥2pp top-1 on Phase 0's
  non-greedy subset.**
- Also report top-1 on the full set, and per-slot accuracy — slot invariance is
  the mechanism claimed, so a win should show up as *reduced variance across
  the four move slots*, not just a higher mean. If the mean improves and the
  slot spread does not, the stated mechanism is not what is working.

**Kill Criterion:** `+pointer` gains <2pp on the non-greedy subset → the
action-conditioned head is not extracting anything the flat head could not.
Keep the v4 schema, drop the pointer head, stop.

**Wall-Clock:** ~2 days (code 1.5 days Mac; BC 2–3 hours homebox).

**Caveat:** Better BC accuracy does not predict RL outcome (M9 Phase 2c: +5.6pp BC → −8.3pp RL). This gate is a screen, not a decision.

**Dependencies:** Phase 2 must complete (v4 schema finalized, BC trained).

---

## Phase 4: RL Validation (Short Arms)

**Goal:** Quick RL signal check before committing to longer runs. **Pre-registered gates only.**

**Scope:**
- Two 2M-step arms (not 5M): v3 baseline vs v4+pointer.
- Same seed, M7 recipe (mixed opponent pool, etc.).
- Evaluate raw-policy greedy at n=2,000/opponent after training completes.

**Pre-Registered Gate (ladder not required):**
- v4+pointer must beat v3 by **≥+3pp vs Random, CI excluding 0** (n=2,000).
- Meets gate: escalate to full ladder validation (Phase 5, not scoped here).
- Fails gate: checkpoint invalidation cost is high; close hypothesis as "observation enrichment + pointer head alone does not transfer to RL."

**Kill Criterion:** Fails the gate → close the hypothesis. Do **not** reach for
"it was a seed issue" — training here reproduces within 0.6pp on an identical
recipe (M9), so a >3pp shortfall is a result, not noise. Replication is only
warranted if the two arms landed within ~1pp of each other, i.e. genuinely
indistinguishable rather than negative.

**Also report** each arm vs `HeuristicAI` from Phase 1, not just vs Random and
DamageFirst. That is the rung this branch built specifically because the bot
ladder has no discriminating opponent between "93% vs bots" and the humans.

**Wall-Clock:** ~1.5 days (PPO 2–3 hours per arm homebox, parallel; eval 2–3 hours Mac).

**Dependencies:** Phase 3 must gate positive (pointer head working in BC).

---

## Checkpoint Invalidation & Cost Summary

| Phase | Invalidates | Cost | When |
|---|---|---|---|
| 0 | None | Negligible | Before Phase 1 |
| 1 | None | ~1 day | Parallel with Phase 2 design |
| 2 | **All existing checkpoints** | ~4 days total | ~midway through experiment |
| 3 | None (v4 schema fixed) | ~2 days | After Phase 2 gates |
| 4 | None | ~1.5 days | After Phase 3 gates — **the only RL in the plan** |

**Total project length:** ~10–12 days wall-clock (parallelizable: Phases 1 & 2 overlap; homebox BC runs in parallel with Mac code work).

---

## Pre-Registration Checklist

Before each phase, complete:

- [ ] Effect size and n chosen (for bot-eval / ladder gates)
- [ ] Kill criterion stated
- [ ] Decision rule written
- [ ] Sample size checked against `docs/EVALUATION-METHODOLOGY.md` tables
- [ ] Fresh accounts registered (if ladder planned)
- [ ] Arm labels / `--run-id` decided

After each phase, report:

- [ ] Per-arm n, wins, rate, CI, mean opponent Elo (if ladder)
- [ ] Difference with CI
- [ ] Verdict against pre-registered gate
- [ ] Update IN-PROGRESS.md with result

---

## Success & Failure Cases

| Outcome | Path |
|---|---|
| Phase 0: gap <3pp, human ≈ agent | Stop. Motivation is weak. Document and close experiment. |
| Phase 1: HeuristicAI working | Continue to Phase 2. |
| Phase 2: v4 schema finalizes, BC trains | Continue to Phase 3. |
| Phase 3: v4+pointer ≥+3pp vs Random | Continue to Phase 4. |
| Phase 3: v4+pointer regresses vs v3 | Stop. Pointer head is the issue; keep schema, revert head. |
| Phase 4: v4+pointer ≥+3pp vs Random | Success. Escalate to full ladder validation (Phase 5, future). |
| Phase 4: regresses vs Phase 3 | Checkpoint invalidation cost not justified; close branch. |

---

## Deliverables Checklist

- [ ] `docs/experimental/hierarchical-action-policy.md` (design doc, rescoped)
- [ ] `docs/experimental/MILESTONES-EXPERIMENTAL.md` (this file, build plan)
- [ ] Phase 0 results: CSV + probe script
- [ ] Phase 1 results: damage formula, HeuristicAI, bot-eval report
- [ ] Phase 2 results: v4 schema, feature-extractor, replay-adapter, BC checkpoints
- [ ] Phase 3 results: pointer head, auxiliary head, bot-eval report
- [ ] Phase 4 results: short-arm RL evals, bot-eval report
- [ ] Updated `docs/experimental/hierarchical-action-policy.md` with any findings
