# In Progress — Pokemon Showdown AI Training

Last updated: 2026-08-03

---

## Where We Stand (2026-07-31)

**Shipping agent: still the unchanged M7 checkpoint** (`v3/ppo_step_5000002_final.pt`),
93.0% vs Random / 84.2% vs DamageFirst. Nothing since M7 has beaten it —
including M9 Phase 2c, which came in **6–7pp below M7** despite starting from a
demonstrably better BC checkpoint (see Phase 2 below). **M7 is the Phase 3
candidate as well as the control.**

**M7 re-measured at n=2,000 on 2026-07-31: 69.7% vs Random / 59.4% vs
DamageFirst as a raw policy** (the 93.0/84.2 figures are with tuned MCTS on
top). Its historical raw reading was 70.0% at n=500 — a clean replication, and
the first time any baseline in this project has been re-measured rather than
quoted.

**Four hypotheses have now been tested and closed:**

| Hypothesis | Verdict |
|---|---|
| (b) Obs richness is the constraint | ❌ M8 Phase 1A — Criterion A failed |
| (a) Value head miscalibration is the constraint | ❌ M8 Phase 2 — failed twice, −2.5pp each, distribution mismatch eliminated |
| More gen 1 human data can be scraped | ❌ M9 Phase 2b — archive exhausted, we hold ~all of it |
| Unrated tour games are an untapped pool | ❌ already in training, and 74% of it |

**(c) opponent-pool / data distribution: POSITIVE at the BC stage, NEGATIVE
after RL — and as of 2026-08-01 both halves are confirmed, not confounded.**
Phase 2a is a clean, well-powered win for format alignment; Phase 2c carried
that checkpoint through M7's exact 5M-step recipe and came out **8.3pp below**
a same-machine control. A seed replication (below) reproduced M7's recipe to
within **0.6pp**, ruling out training-seed and backend noise as the
explanation. **The finding is that a better imitator made a worse RL starting
point.** **Read 2a, 2c and the seed replication below before citing any of
them.** M9 Phase 2a trained a BC checkpoint on gen1randombattle replays
only, against the mixed gen1ou+randbats BC that every checkpoint from M5.5
through M7 was built on. Identical recipe, identical schema, only the corpus
differs. At n=5,000 per arm:

| | randbats-only BC | mixed BC | difference (95% CI) |
|---|---|---|---|
| vs Random | **39.5%** | 33.9% | **+5.6pp [+3.7, +7.5]** |
| vs DamageFirst | **33.2%** | 28.6% | **+4.6pp [+2.8, +6.4]** |
| randbats val acc | **54.3%** | 52.7% | +1.6pp (n=60,766) |

Both CIs exclude 0. **The gen1ou half of the BC corpus was diluting
gen1randombattle play, not enriching it** — the opposite of the 2026-07-16
project decision that put it there. Note this reverses no measurement; the
mixed-corpus choice was never A/B'd, it was assumed.

**The blocking problem was measurement, not modeling — and M9 Phase 1 has now
fixed it (✅ 2026-07-31).** Three consecutive inconclusive ladder readings, and
the same checkpoint "swung" 42% → 27% between runs. The re-analysis shows that
swing was **sample size, not drift** (±13pp CI at n=50; session heterogeneity
`phi=1.21, p=0.303`), and that the 42% was **the better of two unreported 50-game
sessions that day** (the other scored 24.0%). Pooling all 387 M7-era rated games
gives the number no single run could see: **M7 = 30.5% on ladder, 95% CI
[26.1, 35.3]** — **22.9% [18.7, 27.6] excluding opponent concessions**, which is
the figure to quote for play strength (see Recently Completed, 2026-08-03).

**👉 For a plain-language, one-screen orientation — what the agent can and
can't do, what's settled, and the live menu of options with a recommendation —
read `docs/WHERE-WE-ARE.md`.** It is deliberately not a timeline; this file is
the timeline. Surface it to the user after a context clear.

**Ground-truth docs — read these before re-proposing anything in the table above:**
- `docs/EVALUATION-METHODOLOGY.md` — **the protocol.** Required sample sizes, the
  runbook, the reporting template. Any result not produced this way doesn't count
- `docs/DATA-INVENTORY.md` — what data exists, what we hold, what BC consumes
- `docs/LADDER-MEASUREMENT.md` — the diagnosis of what went wrong in M6–M8

---

## Current Work

**👉 THE IMMEDIATE NEXT ACTION: run the M11 eval battery. Both arms finished
training 2026-08-02** — `m11_h128` at 5,000,003 steps, `m11_h512` at 5,000,006,
on the home box at `~/Projects/pokemon-showdown-ai/models/ppo/checkpoints/<arm>/`.
Nothing has been evaluated yet. Full runbook in **Next Steps §1**.

**Field observation (2026-08-02, user watching a live ladder game) — added to
M11 Phase 1 scope.** The agent switched a Water type in after a faint against an
Electric, then spent the following turn switching it out for a Normal. Checked
the encoder: **type effectiveness is encoded offensively only.** `fillV3MoveDims`
(`sim/tools/feature-extractor.ts`) fills dims 77–80 with each token's moves vs
`ctx.defenderTypes` — always the opponent's active. No dim anywhere encodes the
opponent's moves vs *this* token's typing, so resistant-switch-in selection has
to be learned from raw type one-hots while the attacking direction is
precomputed. Added to the v4 schema list in MILESTONES.md → M11 Phase 1
(swap the arguments to `computeTypeEffMultiplier`; no new type logic).
**Status: hypothesis from n=1 game plus a code read, not a measurement.** The
wasted turn specifically (double switch) is *not* explained by this alone.

**M11 SCOPED (2026-08-01): Observation Enrichment + Reward Asymmetry.**
A code review on 2026-08-01 identified the root cause of the 93%-vs-bots / 30%-vs-humans gap: the observation is information-poor in ways invisible against scripted bots and expensive against humans. The agent encodes moves without identity (Recover and Swords Dance are byte-identical), carries no species/stats/trapping, and cannot reason about damage. This is the first hypothesis that predicts the *shape* of the gap rather than proposing another knob.

**Scope:** Phase 0 (reward asymmetry fix, ~1 hour), Phase 1 (observation enrichment, ~4 days), Phase 2 (ladder validation if gates pass). New schema v4 invalidates all checkpoints; retraining from BC scratch. See MILESTONES.md → M11 for full plan.

**Blocker:** User approval to commit the retraining cost (BC + PPO + full eval
cycle). Note Phase 0 is already done — the blocker is Phase 1 (schema v4) only.

**✅ M11 Phase 0 (reward asymmetry) IS DONE AND SHIPPED** — `applyStallPenalty`
now clips before charging the duration cost, so it applies equally to wins and
losses. `stallPenalty` is settable (`--stall-penalty`), default 0.001 =
historical. `gamma` already discounts terminal reward symmetrically, so
`--stall-penalty 0` is a live experiment, not a fallback. Regression test added
that asserts win/loss duration costs are equal, with the old formula inlined —
the previous test asserted rewards stayed in `[-1,1]`, which **encoded the bug**.

**🔬 IN FLIGHT (launched 2026-08-01 ~11:50 local, home box, both `cuda`):**
two PPO arms, `models/ppo/checkpoints/m11_h128/` and `m11_h512/`.

| arm | params | warm-start + anchor |
|---|---:|---|
| `m11_h128` | 151,187 | `bc_mlp_gen1_v3.pt` |
| `m11_h512` | **801,299** | `bc_mlp_gen1_v3_h512.pt` |

Everything else identical: 5M steps, `selfplay=0.5,damagefirst=0.3,random=0.2`,
`--bc-anchor-coef 0.05`, 200k value warmup, same two pool seeds
(`ppo_step_0_seed_m2.pt`, `ppo_step_0_seed_m33best.pt`), same machine, same
commit, **same fixed reward**.

**Why two arms and not one.** The reward fix landed *before* the launch, so
`m9seed` stopped being a valid control — it trained under the buggy reward.
Running only the width arm would have moved width *and* reward together, the
exact confound the "same machine, same code" standing rule exists to prevent.
Two arms give two clean single-variable comparisons for one wall-clock cost:

- `m11_h128` vs **m9seed** → the **reward fix** (width held at 128)
- `m11_h512` vs **m11_h128** → **width** (reward held fixed)

**ETA ~5 h, not the ~70 min quoted elsewhere in this file.** Measured 16.5k
steps/min with two concurrent runs sharing 24 cores. Concurrency does not
threaten validity — 5M steps is 5M steps — it only costs wall-clock.

**Do not read the early rollout win rates as a result.** At step ~248k the
policy has only just unfrozen (value warmup ends at 200k) and the arms showed
0.33 (h128) vs 0.50 (h512). That is 5% of training, during a transient, on a
metric that does not record which opponent family it was against. **Gate on the
n=2,000 bot evals, all arms in one session on one machine, per
`docs/EVALUATION-METHODOLOGY.md`.**

**Pre-registered, written before any result exists:** width gate is **≥+3pp vs
Random at n=2,000/arm with the CI on the difference excluding 0**; DamageFirst
reported alongside but not gated (needs ~4,948/arm to resolve ±2pp). Expect an
**informative null on width** — the observation was measured to carry ~25
distinct values, and 5.3× params bought only +2.8pp BC val accuracy. A null now
closes the capacity question with a mechanism attached rather than a shrug.

---

## Recently Completed

- **Ladder log deep-dive + analysis tooling (2026-08-03).** Three findings from
  the 360-game `m7-greedy` run beyond the headline win rate:
  1. **~10% of ladder games are opponent concessions and they are ~35% of all
     our wins.** Across 747 rated games we have never lost one in under 16
     decisions; 74 wins came that fast. Contested-only: **greedy 19.3%
     (n=326)**, M7-era sampling **22.9% (n=341)** — the same −3.5pp difference,
     so no comparison changes, but every absolute ladder number this project has
     quoted is ~7pp high. Logged as correction 4 in MILESTONES.md.
  2. **Win rate decays with game length:** 31.2% (16–20 decisions) → 24.1%
     (21–25) → 18.8% (26–30) → **14.6% (31–40)**. First behavioral evidence
     matching the observation-poverty prediction — short games are damage races,
     long games are position games.
  3. **Win rate decays with opponent Elo:** 34.4% (1000–1099) → 15.3%
     (1200–1299), r=−0.143, t=−2.72, n=360. Own rating never left ~1000–1120 and
     hit the 1000 floor twice.

  `scripts/ladder_analysis.py` now reports the concession split and a
  win-rate-by-opponent-Elo table on every run, and takes `--min-decisions N`
  (use 16) to score contested games only. Both CSV schemas still parse.
  **Consequence: M10 is re-prioritised ahead of M11 Phase 1** — finding (2) is a
  correlation that M10 can turn into a mechanism without any retraining.
- **Greedy decoding shipped and confirmed offline (2026-08-01).** `--greedy` now
  exists on `models/evaluate.py`, `models/infer_server.py` and
  `tools/ladder-bot/ladder-bot.js`; `PPOAgent`/`TransformerAgent`
  `act()`/`act_batch()` take the masked argmax when `agent.greedy` is set,
  default off so rollouts still sample. `evaluate.py` prints the decision rule
  and refuses the flag where it would be inert (`--model mcts`, q_learning/dqn);
  the ladder bot's banner prints `policy=greedy|sampled`. **A/B on the M7 v3
  checkpoint, n=5,000/arm** (not 2,000 — the baseline is in the 0.3–0.7 band
  where the methodology requires 5,000), all four arms one session one machine:

  | vs | sampled | greedy | difference (Newcombe 95%) |
  |---|---|---|---|
  | Random | 69.9% (3497/5000) | **77.7%** (3887/5000) | **+7.8pp [+6.1, +9.5]** ✅ |
  | DamageFirst | 60.2% (3010/5000) | **65.2%** (3259/5000) | **+5.0pp [+3.1, +6.9]** ✅ |

  vs Random replicates the review's +7.8pp exactly at 12.5× the n. **The
  DamageFirst arm reverses the review's −2.0pp** (that reading was n=400 and
  underpowered), so greedy is *not* opponent-dependent — `docs/CODE-REVIEW-FINDINGS.md`
  §3 has been corrected. Side benefit: sampled-vs-Random came in at 69.9%
  against M7's recorded 69.7%, a data point on the §5f harness-reproducibility
  question. **`ladder-bot.js` now plays greedy by default** (`--sample` opts
  out). **Untested: the ladder**, where determinism against adapting humans is
  the one risk bot evals cannot speak to — costed at ~83 h and declined.
- **Code Review (2026-08-01):** Three independent read-only reviews of the training path, observation pipeline, and evaluation machinery. Identified root cause of the bot/human gap: **observation poverty**. The agent encodes moves without identity, carries no species/stats/trapping, cannot reason about damage. Also found: (1) MCTS confound (argmax vs sampling, ~+7.8pp vs Random), (2) M8 Phase 1A doubly confounded (trunk width fixed, no replay-adapter v3-extended path), (3) M8 Phase 2 new candidate cause (reward scale asymmetry in battle-sim), (4) M2 decision-rule confound (epsilon=0 for some arms, sampling for PPO), (5) "~3pp seat bias" unsupported by CI including 0.
<!-- Older entries (M9 phases 2a–2d, seed replication, observation-poverty
     scoping) now live in MILESTONES.md — Results Ledger and M11 respectively,
     per the three-item rule at the bottom of this file. -->

---

## Blockers

- **M11 approval:** Observation schema v4 invalidates all checkpoints (BC + PPO). Retraining cost: ~2–3h BC + ~2h PPO + ~2h full eval + ladder if gates pass. User approval required before committing to M11. **As of 2026-08-03 the recommendation is to run M10 first** — it needs no approval, no retraining, and should tell M11 which dims to prioritise. See MILESTONES.md → M10 Recommendation.
- **Width-512 probe: TRAINED 2026-08-02, UNMEASURED.** Both arms hit 5M steps.
  Not a blocker on anyone — it is the immediate next action. Runbook in Next
  Steps §1. Expectation is still an informative null on width.
- ~~Greedy decoding~~ — **resolved 2026-08-02.** Confirmed offline (+7.8pp /
  +5.0pp), adopted as the ladder default; the one ladder read was flat and
  diluted by `--mcts`. Closed. See Next Steps §0.

---

## Next Steps

0. **Greedy is the ladder default (2026-08-01); the ladder read was flat
   (2026-08-02).** Adopted on 10,000 offline battles (+7.8pp / +5.0pp, both CIs
   excluding 0).

   **Ladder run `m7-greedy`** (M7 `ppo_step_5000002_final.pt`, account
   `Novapool`, `--mcts`, greedy): **97/360 = 26.9% [22.6, 31.8], mean opponent
   Elo 1148.5.** Against the M7-era sampled baseline (118/387 = 30.5%):
   **−3.5pp, 95% CI [−10.0, +3.0] — inconclusive, CI includes 0.** Against the
   full 507-game pre-change pool (28.0%): −1.1pp [−7.0, +5.0].

   **This is not a refutation of the offline result, and must not be recorded as
   one.** Three reasons: (a) the run used `--mcts`, where search already returns
   an argmax, so greedy reached only the **~20% of decisions that fall back to
   the raw policy** (`ladder-bot.js:291`) — a ~5× diluted treatment; (b) it is a
   historical comparison, not a paired A/B, against July sessions weeks apart;
   (c) the pre-change CSV has **no `opp_rating` column**, so the opponent-Elo
   confound check the methodology mandates is impossible. Verdict: **greedy is
   not harmful in the shipping config at ±6.5pp.** Nothing stronger is
   supportable.

   **Decision: keep greedy on, stop testing it.** A real ladder A/B needs
   `--sample` as a paired control, no search on either arm, ~83 h — to answer a
   question M11 invalidates by changing the checkpoint.

   **Two findings worth keeping independent of greedy:**
   - **26.9% at n=360 (±4.6pp) is the most precise ladder number this project
     has**, and the first with opponent Elo recorded.
   - **~20% of ladder decisions never reach search.** Force-switches after a
     faint and locked states go to the raw policy. Those are precisely the
     "what do I bring in" decisions where the missing *defensive* type-matchup
     dim bites. Two independent lines now point at the same v4 item.

   ⚠️ **Ladder history is split across two CSVs.** The M9 schema change rotated
   the old file to `ladder_results.pre-m9.csv`, and `ladder_analysis.py` defaults
   to `ladder_results.csv` alone — so a bare invocation **silently drops all 507
   pre-2026-08-01 games**. Always pass both paths:
   ```bash
   python3 scripts/ladder_analysis.py \
     data/replays/self_ladder/ladder_results.pre-m9.csv \
     data/replays/self_ladder/ladder_results.csv --since 2026-07-16
   ```
   Pooled across both, all 867 rated games: **27.6% [24.7, 30.6]**.

1. **▶ M11 eval battery — BOTH ARMS TRAINED, NOTHING MEASURED YET.**
   Finished 2026-08-02: `m11_h128` (151,187 params) at 5,000,003 steps,
   `m11_h512` (801,299) at 5,000,006. Checkpoints live **on the home box only**
   at `~/Projects/pokemon-showdown-ai/models/ppo/checkpoints/<arm>/` — they do
   not sync via git (`docs/MULTI-MACHINE.md`).

   **Two pre-registered comparisons**, each single-variable:
   - `m11_h128` vs **`m9seed`** → the **reward fix** (width held at 128)
   - `m11_h512` vs **`m11_h128`** → **width** (reward held fixed)

   Pre-registered gate: **≥+3pp vs Random with the CI on the difference
   excluding 0**; DamageFirst reported alongside but not gated. Expectation was
   an informative null on width.

   **Runbook:** evaluate the **final** checkpoint of each arm (never a sweep
   pick — `docs/EVALUATION-METHODOLOGY.md`), **all arms in one session on one
   machine**, identical decision rule across arms, then
   `scripts/bot_eval_ab.py --arm base=W/N --arm cand=W/N --gate 3`.

   **Two open calls, flagged before any number exists:**
   - **n.** Pre-registration says 2,000. These arms will land near 0.6–0.7 like
     M7, where 2,000 resolves only ~±4.5pp against a +3pp gate — underpowered by
     the project's own rule (n=5,000 for anything in the 0.3–0.7 band).
     **Recommend running n=5,000 and recording plainly that the registration was
     widened**, rather than running a test known in advance to be underpowered.
   - **Decision rule.** The gate compares *training recipes*, so it only needs to
     be identical across arms. **Recommend `sampled`** for the gate — keeps these
     comparable to every historical number including M7's 69.7% — and add a
     `--greedy` read on the winner only.

   Then: if width gates ≥+3pp, capacity is live and M11 Phase 1 should budget a
   wider net; if null (expected), the capacity question closes with a mechanism
   attached and Phase 1 proceeds at width 128.

2. **M11 Phase 0 (reward asymmetry fix, ~1 hour):** ~3 lines in `battle-sim.ts`, 1 replication of M8 Phase 2's value fine-tune. If moves needle (≥+1pp), include before Phase 1. If not, proceed to Phase 1 directly.

3. **M11 Phase 1 (observation enrichment, ~4 days):** Scope v4 schema carefully; estimate BC/PPO/eval costs; pre-register gates. A/B at n=2,000/opponent, gate ≥+3pp CI excluding 0.

4. **Phase 2 (if Phase 1 gates pass):** Paired ladder A/B, n≥350/arm, power for +10pp.


---

## Active Plan

**M11: Observation Enrichment + Reward Asymmetry.** Full spec in
`MILESTONES.md` → M11.

```
Phase 0  reward asymmetry fix          ✅ SHIPPED (applyStallPenalty clips
                                          before charging duration cost)
  │
Width/reward probe  m11_h128, m11_h512  ✅ TRAINED 2026-08-02 (home box)
  │                                     ⏳ UNMEASURED ← immediate next action
  │      m11_h128 vs m9seed  → reward fix (width held)
  │      m11_h512 vs m11_h128 → width (reward held)
  │      gate: ≥+3pp vs Random, CI on the difference excluding 0
  ▼
Phase 1  observation schema v4          🚫 BLOCKED on user approval
         move identity, species/base stats, trapping, recharge,
         defensive type effectiveness
  │      invalidates every checkpoint → BC + PPO retrain from scratch
  │      gate: ≥+3pp vs Random at n=2,000/arm, CI excluding 0
  ▼
Phase 2  paired ladder A/B              (only if Phase 1 gates pass)
         n≥350/arm, powered for +10pp, M9 Phase 3 protocol
```

Instrumentation debt to pay alongside Phase 1 (~4 h, unblocks every future
A/B): `--seed` on both trainers, per-step entropy/KL/clip-fraction/value-loss
logging, and a `meta.json` run manifest (device, git SHA, argv, timestamp).

---

## Standing Rules

Learned the hard way; each one has a milestone behind it.

- **Both arms of an A/B on the same machine, in one session, same decision
  rule.** M9 Phase 2c moved BC corpus + seed + backend at once and cost a
  verdict.
- **Evaluate the final checkpoint, never a sweep pick.** M3.4's 62% sweep peaks
  regressed to 54% at n=500.
- **n=5,000 for anything in the 0.3–0.7 band**; n=2,000 only resolves ~±4.5pp
  against a +3pp gate. Full tables in `docs/EVALUATION-METHODOLOGY.md`.
- **GXE is account-level and cumulative — never a per-run gate.** This produced
  three milestones of over-read trends (M6–M8).
- **Pass both ladder CSVs to `ladder_analysis.py`** or it silently drops all 507
  pre-2026-08-01 games:
  ```bash
  python3 scripts/ladder_analysis.py \
    data/replays/self_ladder/ladder_results.pre-m9.csv \
    data/replays/self_ladder/ladder_results.csv --since 2026-07-16
  ```
- **Pre-register the gate before the number exists**, and record widenings
  plainly rather than quietly.

---

## Housekeeping

- **This file is the timeline; `docs/WHERE-WE-ARE.md` is the orientation.**
  Surface WHERE-WE-ARE after a context clear.
- **Keep `Recently Completed` to the last three items.** Older entries belong in
  the `MILESTONES.md` ledger; run-level detail belongs in
  `docs/MILESTONES-ARCHIVE.md`.
- Historical execution notes (M9 phase detail, M5 smoke battery, M7 job logs)
  were moved to the archive's appendix on 2026-08-03.
