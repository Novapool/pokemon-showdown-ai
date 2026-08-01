# In Progress — Pokemon Showdown AI Training

Last updated: 2026-07-31

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
[26.1, 35.3]**.

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

**M9 SCOPED (2026-07-31): Evaluation Methodology + Data Distribution Hypothesis.** M8 spent both its technical bets (obs richness Phase 1A, value-head targeting Phase 2) on the M7 checkpoint and both failed. Three methodological defects also came to light: (1) GXE is account-level cumulative (506 games M6-M8), not per-run; (2) same M7 checkpoint shows 42%→27% win-rate variance across runs (~1.8 SD); (3) +3pp gate at n=200 is underpowered (SE ~3.9pp; a true +3pp effect caught only ~1/3 of the time). The surviving hypothesis — opponent-pool/data distribution (constraint c) — was deprioritized on reasoning alone, never tested. **M9 addresses both:** Phase 1 fixes evaluation methodology (per-run GXE isolation, pre-registered sample sizes), Phase 2 tests the data/distribution hypothesis (randbats-only BC, richer replay corpus, opp-pool saturation), Phase 3 ladders with proper instrumentation, Phase 4 makes a stopping decision. See MILESTONES.md → M9 for full scope.

### M9 Phases (Pre-Registered Gates)

**Phase 1: Evaluation Methodology v2 — ✅ COMPLETE 2026-07-31**

Delivered:
- `docs/EVALUATION-METHODOLOGY.md` — the protocol: three standing rules, required
  sample sizes for ladder *and* bot evals, account/arm setup, run and analysis
  commands, the reporting template, interpretation rules, checklist.
- `scripts/ladder_analysis.py` — Wilson intervals, Newcombe CI on the paired
  difference, session segmentation, chi-square heterogeneity, power tables.
  Stdlib only. `--power` needs no data.
- `tools/ladder-bot/ladder-bot.js` — `ladder_results.csv` now records
  `run_id` (arm label), `account`, `checkpoint`, and pre-battle
  `opp_rating`/`own_rating` from the `|player|` lines. Pre-M9 rows retire
  automatically to `ladder_results.pre-m9.csv`. Covered by
  `test/tools/ladder-results.test.js` (5 tests; tools suite 111 passing).

**Findings that changed the plan:**
1. **No measurable ladder drift.** `chi2=4.85, df=4, p=0.303, phi=1.21` across
   the five M7-era sessions — pure binomial. The paired design's stated
   justification ("cancels drift") is unsupported; keep it (free, equalises the
   opponent pool) but it buys **no** reduction in required n.
2. **The 7/23 record was incomplete.** Two back-to-back 50-game sessions, same
   checkpoint: **24.0%** then **42.0%**. Only the 42% was written down.
3. **A well-powered M7 baseline already existed in the logs: 30.5% (118/387),
   95% CI [26.1, 35.3].** Already at the n≈350 target, so **the planned
   replication ladder run was not needed** — Phase 3's fresh-account control arm
   covers the design-validation purpose. Phase 1 cost ~0 ladder hours.
4. **The Phase 2 bot-eval gate was underpowered too** and has been rewritten:
   "+2pp at n=500" needs 2,213 games/arm vs Random and 4,948 vs DamageFirst.
   Measured throughput is ~27 battles/s raw-policy (200 in 7.4 s) and ~4.5
   s/battle for tuned MCTS — so **n=2,000/arm raw-policy costs 75 seconds** and
   the new gate is **≥+3pp vs Random at n=2,000/arm, CI excluding 0**.

**Phase 2: Data/Distribution Hypothesis Test**
- **2a: ✅ COMPLETE 2026-07-31 — POSITIVE.** Randbats-only BC beats the mixed
  BC by +5.6pp vs Random and +4.6pp vs DamageFirst at n=5,000/arm, both CIs
  excluding 0 (table under "Where We Stand"). Format alignment is real.
  Checkpoint: `models/checkpoints/bc_mlp_gen1_v3_rb5.pt` (v3 schema, 5 epochs,
  1.10M records/epoch, 201 s on the Mac). Raw counts:
  `models/checkpoints/m9p2a_ab_results.txt`; runner `run_m9p2a_ab.sh`.
- **2b: ❌ CLOSED 2026-07-31** — gen 1 replay archive exhausted, see Next Steps.
- **2c: ✅ COMPLETE 2026-07-31 — gate PASSES against M5.5, but the candidate
  is measurably WORSE than M7, so the pre-registered rule sends M7 to Phase 3.**
  5M steps in ~68 min on the RTX 3080, 20 checkpoints, clean finish.

  | arm | vs Random | vs DamageFirst |
  |---|---:|---:|
  | M5.5 (`bcft`, v2) | 52.8% | 43.2% |
  | **M7 (`v3`, the control)** | **69.7%** | **59.4%** |
  | 2c final (5.00M) | 63.3% | 51.9% |
  | 2c best-sweep (4.00M) | 62.1% | 53.7% |

  All arms n=2,000, same machine, same session. Differences:

  | comparison | vs Random | vs DamageFirst |
  |---|---|---|
  | 2c − M5.5 | **+10.5pp [+7.4, +13.5]** | **+8.7pp [+5.6, +11.8]** |
  | 2c − M7 | **−6.3pp [−9.2, −3.4]** | **−7.5pp [−10.5, −4.4]** |

  **Decision: Phase 3 ladders M7, not 2c** — decision table row 2, fixed in
  advance. **This is exactly the case the amendment was written for**: judged
  only against the literal M5.5 gate, 2c passes by a wide margin and would have
  been sent to a ~2-day paired ladder run while being 6–7pp weaker than its own
  control arm.

  **M7 replicated at 69.7% (n=2,000) against its historical 70.0% (n=500)** —
  independent evidence that the eval pipeline is sound and that the 2c gap is
  real rather than an artifact of the harness.

  **The substantive finding: 2a's BC-stage advantage did not merely fail to
  compound, it inverted.**

  | lineage | BC | after 5M PPO | RL gain |
  |---|---:|---:|---:|
  | mixed BC → M7 | 33.9% | 69.7% | **+35.8pp** |
  | randbats BC → 2c | 39.5% | 63.3% | **+23.8pp** |

  The format-aligned checkpoint was **the better imitator and the worse RL
  substrate.** The most plausible reading is that the gen1ou half of the corpus
  buys behavioural *breadth* that is useless for imitating randbats play but
  valuable as an exploration prior and KL anchor across 5M steps — a narrower
  anchor constrains the policy toward a smaller region. That is a hypothesis,
  not a measurement.

  **✅ BOTH CAVEATS BELOW WERE RESOLVED 2026-08-01 by the seed replication —
  read that entry (Next Steps) before acting on them.** A fresh run of M7's
  recipe reproduced M7 to within 0.6pp across a backend change, so neither
  training seed nor `mps`/`cuda` explains 2c's gap. In the properly controlled
  same-machine A/B the deficit is **−8.3pp vs Random [−11.1, −5.3]**. **2c's
  regression is real and the format-alignment effect reverses after RL.** The
  caveats are kept below as written because they were the correct things to
  doubt at the time.

  **⚠️ SECOND CONFOUND, found 2026-07-31 after the verdict was committed:
  M7 trained on `device=mps` (the Mac); 2c trained on `device=cuda` (the home
  box).** Both headers are in their `train.log`s. So the −6.3pp is not
  "BC corpus" — it is **BC corpus + training seed + backend**, three variables
  moving at once. Different kernels and accumulation orders compound chaotically
  over 5M PPO steps; there is no reason to think this favours either arm, but it
  is one more independent-draw term and it should have been listed. **The
  cross-device comparison was avoidable and should not be repeated: run both
  arms of any future A/B on the same machine.**

  **⚠️ CONFOUND, and it is not small: training-run-to-run variance has never
  been measured in this project, and both lineages are n=1.** The CIs above
  quantify **eval sampling noise only**. Comparing two checkpoints from two
  different training runs also contains a *training-seed* variance component
  that no amount of eval battles can shrink — running n=20,000 evals would
  narrow those CIs to nothing while leaving the attribution exactly as
  uncertain. **So "format-aligned BC hurts RL" is NOT established; what is
  established is that this particular 2c run is 6–7pp below M7.** Resolving it
  needs a second PPO run (≈70 min on the home box now that throughput is known)
  — ideally re-running M7's own recipe from the mixed BC with a fresh seed, to
  measure the spread directly. **Until that exists, no PPO A/B in this project
  — including M7-vs-M5.5 — can separate recipe effects from seed effects.**

  Artifacts: `models/ppo/checkpoints/m9p2c/{train.log,sweep_results.txt,
  confirm_results.txt,ppo_step_5000004_final.pt}`.

- **2c (superseded launch note): RUN 2026-07-31 18:02 local, home box.**
  Warm-starts 2a's randbats-only BC into the *exact* M5.5/M7 anchored-PPO
  recipe — same 5M steps, same `selfplay=0.5,damagefirst=0.3,random=0.2` mix,
  same `--bc-anchor-coef 0.05`, same 200k value warmup, same M2 + M3.3-best
  pool seeds. **The only variable versus M7's own v3 run is which corpus the BC
  saw**, which makes 2c a clean single-variable test of the distribution
  hypothesis rather than the M7 replication it would have been had 2a gone the
  other way. Checkpoints → `models/ppo/checkpoints/m9p2c/`.
- 2d (optional, now lower priority): opp-pool saturation via human-sampler
  opponents. 2a already produced positive evidence for constraint (c) on the
  *data* leg; 2d tests the *opponent* leg.
- **Gate (revised by Phase 1):** Phase 2 candidate beats the M5.5 baseline by
  **≥+3pp vs Random at n=2,000/arm with the 95% CI on the difference excluding
  0**, OR use M5.5 for Phase 3. Report DamageFirst alongside; do not gate on a
  ±2pp DamageFirst bar (needs ~4,948/arm). **Both baselines (M5.5 and M7) are
  re-measured at n=2,000 rather than quoted from their historical n=150/200/500
  readings** — comparing a fresh n=2,000 number against an old n=200 one is the
  defect Phase 1 exists to remove. Runner:
  `models/ppo/checkpoints/m9p2c/run_confirm.sh`.
- **⚠️ Defect in the pre-registered gate, recorded 2026-07-31 BEFORE the 2c
  results existed: the gate scores against M5.5, but Phase 3's control arm is
  M7.** Those are different agents, and M7 is the stronger one — it is the
  shipping agent precisely because it beat M5.5 (93.0%/84.2% vs 90.6%/79.2%
  under tuned MCTS). As written, the gate would pass a candidate that clears
  M5.5 by +3pp while still losing to M7, and Phase 3 would then spend ~2 days
  of ladder on an arm known in advance to be weaker than its own control.
  **Both baselines are therefore measured, and the decision rule is fixed now
  rather than after seeing the numbers:**

  | 2c vs M5.5 | 2c vs M7 | Phase 3 candidate |
  |---|---|---|
  | ≥+3pp, CI excl. 0 | ≥0 (CI may include 0) | **2c** — gate met and not worse than control |
  | ≥+3pp, CI excl. 0 | **<0, CI excl. 0** | **M7** — gate is met but the candidate is measurably worse than the control; laddering it would be spending two days to confirm a known regression |
  | <+3pp or CI includes 0 | any | **M7** (not M5.5 — the milestone text says "use M5.5 for Phase 3", but M7 has been the shipping agent since M7 closed and is the correct control) |

  This changes no measurement and is not a loosening — every arm is still
  judged at n=2,000 with a CI. It closes a hole where the literal gate and the
  Phase 3 design disagree about which agent is the incumbent.

**Phase 3: Ladder Validation (Methodology v2)** (~2 days ladder, both arms)
- **Paired concurrent A/B**: M7 control and Phase 2 candidate on **two fresh
  accounts, alternating within the same sessions**, so ladder drift — the
  confound that produced 42%→27% on an unchanged checkpoint — cancels in the
  paired difference. Primary endpoint is the **difference in raw win rate**, not
  either arm's absolute GXE.
- Elo/GXE reported as secondary only, valid because accounts are fresh and
  matched on n; report each arm's mean opponent Elo too.
- 3c (optional, run first): head-to-head bot A/B at n=1000 — far cheaper than
  ladder, decides whether the candidate earns ladder time at all.
- **Power (pre-registered):** at p≈0.30 and 80% power, +5pp needs ~1,400
  games/arm, +10pp ~350/arm, +15pp ~160/arm. At ~4 min/battle, 350/arm ≈ 23h per
  arm. **M9 powers for +10pp.** Smaller effects are declared not measurable on
  the ladder by this project and must be judged on bot evals.
- **Gate (on the paired difference):** ≥+10pp with CI excluding 0 = win; CI
  includes 0 = inconclusive, which at 350/arm now *means the effect is <10pp*
  rather than "we couldn't tell"; ≤−10pp with CI excluding 0 = regression.

**Phase 4: Stopping Decision** (2–4 hours analysis)
- Win: declare, recommend M10 direction (team-specific, multi-format, online learning)
- Inconclusive at adequate power: the effect is <10pp — record that as a
  *finding* and stop spending on this direction, don't re-run bigger
- Regression: postmortem (data quality vs checkpoint weakness); pivot or stop

---

### Next Steps

**M9 Phase 2 is COMPLETE (2026-07-31). The Phase 3 candidate is M7.** Phase 2
spent one bet (2a) that won and one (2c) that lost, and the net effect on the
shipping agent is zero — M7 still ships. What Phase 2 bought is three things
worth more than the checkpoint would have been: format alignment is now a
*measured* effect rather than an assumption, both baselines have well-powered
n=2,000 readings for the first time, and the gate defect was caught before it
could spend two days of ladder.

**✅ COMPLETE 2026-08-01 — the PPO seed replication settled both questions, and
the answer is good news twice over.**

All four arms re-measured at n=2,000 in **one session on one machine**:

| arm | vs Random | vs DamageFirst | difference vs M7 (R) |
|---|---:|---:|---|
| **M7** | 71.7% | 57.6% | — |
| **m9seed** (M7 recipe, fresh seed) | **71.0%** | **57.4%** | **−0.6pp [−3.4, +2.2]** ✅ |
| 2c (randbats BC) | 62.8% | 50.7% | −8.9pp [−11.7, −5.9] |
| M5.5 | 51.2% | 42.0% | −20.5pp [−23.4, −17.5] |

**1. PPO run-to-run spread is small — under a point.** Re-running M7's recipe
from the same mixed BC reproduced it to **−0.6pp vs Random and −0.3pp vs
DamageFirst, both CIs comfortably including 0** — and that is *across a backend
change too* (M7 on `mps`, m9seed on `cuda`), so neither seed nor device moved
the outcome detectably. **The alarm raised in the 2c write-up was warranted to
raise and is now answered: it does not fire.**

**2. Therefore 2c's regression is real.** The properly controlled A/B — both
runs `cuda`, same machine, same recipe, **BC corpus the only systematic
variable**:

| | m9seed (mixed BC) | 2c (randbats BC) | difference |
|---|---:|---:|---|
| vs Random | 71.0% | 62.8% | **−8.3pp [−11.1, −5.3]** |
| vs DamageFirst | 57.4% | 50.7% | **−6.7pp [−9.7, −3.6]** |

**So both Phase 2 findings stand, and they point in opposite directions:**
the format-aligned BC is a **better imitator** (2a: +5.6pp raw, +1.6pp
imitation accuracy) and a **worse RL substrate** (−8.3pp after 5M steps).
**Better imitation did not mean a better starting point for RL.** The standing
explanation — that the gen1ou half buys behavioural breadth which is useless
for imitating randbats but valuable as an exploration prior and KL anchor over
5M steps — is now the surviving hypothesis rather than one of several.

**What this rescues:** low run-to-run variance means the project's existing
single-run A/Bs are more trustworthy than the 2c write-up feared. M7-vs-M5.5
stands. So do the M8 negatives — Phase 1A's −3pp and Phase 2's −2.5pp are
unlikely to have been seed noise, which makes those closures firmer, not
weaker.

**Honest limit: this is ONE replication pair (n=2 runs).** It shows the two
runs agree to within eval noise, which bounds the spread as *small*; it does
not put a tight interval on the spread itself. A third run would. It is
adequate for the decision at hand — an 8.3pp gap against an observed 0.6pp
reproduction difference — and should not be quoted as "PPO variance is 0.6pp".

**Standing rule adopted: run both arms of any A/B on the same machine.** The
2c-vs-M7 comparison crossed `mps`/`cuda` unnecessarily. It turned out not to
matter, which is a fact now rather than an assumption.

Artifacts: `models/ppo/checkpoints/m9seed/{train.log,confirm_results.txt,
ppo_step_5000004_final.pt}`.

**Setup (superseded status note): launched 2026-07-31 ~19:45 local, home box,
68 min.** `models/ppo/checkpoints/m9seed/` — M7's exact recipe,
warm-started and anchored from the **mixed** BC (`bc_mlp_gen1_v3.pt`, md5
verified identical across machines), pool seeded with the same two files.
`train.py` performs **no seeding of any kind** (`manual_seed`/`np.random.seed`
appear nowhere in the trainer, agent, or gym clients), so re-running the
identical command is already an independent draw — no code change was needed.

**It answers two questions, and the second one is better than the experiment
was designed for:**

| comparison | holds constant | varies | what it measures |
|---|---|---|---|
| m9seed vs **M7** | recipe, BC corpus | seed **+ backend** (cuda vs mps) | run-to-run spread |
| m9seed vs **2c** | recipe, backend (**both cuda**), machine | **BC corpus** + seed | the format-alignment effect, *properly controlled* |

The second row is the A/B that 2c-vs-M7 was supposed to be and wasn't. If
m9seed lands near M7's 69.7%, the spread is small and 2c's regression is real.
If it lands near 2c's 63.3%, then **2c is inside noise and so is a share of the
project's existing record.**

**The original framing of this step, for the record:** a ~70-minute seed
replication re-running M7's exact recipe from the mixed BC to measure spread. Every A/B this project has ever run compared two
single training runs and attributed the whole difference to the recipe. That
includes M7-vs-M5.5, the comparison the entire current lineage rests on. If
PPO run-to-run spread turns out to be ±6pp, then 2c is inside noise, "format
alignment hurts RL" is unsupported, **and so is a share of the project's
existing conclusions.** If the spread is ±1pp, 2c's regression is real and the
breadth hypothesis is worth pursuing. It is the cheapest question with the
largest reach, and the M9 Phase 1 lesson (ladder readings were sample size, not
signal) is the same mistake one level up.

**✅ COMPLETE 2026-08-01 — M9 Phase 2d (human sparring): NULL RESULT. Swapping
the entire self-play opponent pool for human imitators changed nothing.**

All arms n=2,000, **one session, one machine**:

| arm | vs Random | vs DamageFirst |
|---|---:|---:|
| M7 | 70.5% | 58.0% |
| **m9seed** (matched control) | **68.9%** | **57.0%** |
| **m9p2d** (human sparring) | **67.9%** | **57.0%** |
| 2c (randbats BC) | 62.8% | 50.7% |
| M5.5 | 53.4% | 41.5% |

**Against its matched control (m9seed — identical warm-start, recipe and
machine; the self-play pool is the only difference):**

| | vs Random | vs DamageFirst |
|---|---|---|
| m9p2d − m9seed | **−1.0pp [−3.9, +1.9]** | **+0.0pp [−3.1, +3.1]** |

Both CIs include 0, and the DamageFirst arms are **identical to the battle**
(1140/2000 each). M7, m9seed and m9p2d all sit in one band; 2c is the only run
that separates from it.

**What this buys, stated carefully.** Per the pre-registration above, a win
would have been unambiguous and a non-win is not. The frozen-partner confound
stands: the agent almost certainly outgrew opponents stuck at ~39%/34% vs
Random. **But the manipulation was strong** — 50% of every rollout, for the
whole run — and it moved the result by *nothing*, which is more informative
than a weak manipulation returning a null.

**The observation worth carrying forward:** m9seed's pool *escalates* with the
agent, m9p2d's is frozen and weak, and **they tie**. So within this recipe the
identity and strength of the self-play pool appears not to be doing much work
at all. That is evidence against opponent quality being the binding constraint,
and it shifts weight toward **model capacity (151k parameters) and the format's
own luck** — options 4 and 3 in `docs/WHERE-WE-ARE.md`.

**Honest limit:** bot evals cannot speak to *human* play, which is the actual
target. A null here does not prove human sparring would not help on ladder; it
proves it did not help against Random and DamageFirst. The named follow-up (BC
checkpoints as seeds inside the escalating pool) is now **low priority** —
it is a strictly weaker manipulation than the one that just returned zero.

Artifacts: `models/ppo/checkpoints/m9p2d/{train.log,sweep_results.txt,
confirm_results.txt,ppo_step_5000002_final.pt}`.

**Setup (superseded status note): launched 2026-08-01, home box, ~70 min.**
`models/ppo/checkpoints/m9p2d/`. Everything matches
M7/m9seed except one flag: `--selfplay-pool models/ppo/checkpoints/human_pool`,
a dedicated directory holding the two BC human-imitators
(`ppo_step_0_bc_randbats.pt`, `ppo_step_0_bc_mixed.pt`). Because the run's own
checkpoints land in a *different* directory, **the self-play half of every
rollout faces a human-style opponent for the whole run** instead of a frozen
copy of the agent's own lineage. Warm-start and anchor are the mixed BC, per the
2c finding. No new code was needed.

**Pre-registered interpretation, written before any result exists** (the 2c gate
amendment is why this is now standard practice here):

- The manipulation swaps opponent *style* (human-like, and crucially they
  **switch Pokémon**) but also opponent *dynamics*: m9seed's pool escalates as
  the agent improves, while this pool is **frozen** at BC strength (~39.5% and
  ~33.9% vs Random). Late in training the agent will likely outgrow it.
- **A win is unambiguous** and directly actionable.
- **A loss is ambiguous** between "human-style sparring doesn't help" and
  "frozen sparring partners stop teaching once you pass them." The
  disambiguating follow-up is already identified: re-run with the BC
  checkpoints as *pool seeds inside the normal checkpoint dir* (replacing the
  `m2`/`m33best` seeds), which keeps the escalating self-play ladder but makes
  the manipulation much weaker. **That weakness is exactly why it isn't the
  first run** — a treatment that is ~4% of rollouts by end of training risks a
  null result that means nothing, which is this project's recurring failure
  mode.
- Early rollout win rate is **0.397** over the first 150k steps, against 0.527
  (m9seed) and 0.493 (2c) in the same window — so the human imitators are
  *harder* early opponents than a frozen copy of self, not weaker. The initial
  concern that the sparring partners were too soft is not supported.

**Phase 3 is ready to run whenever wanted** — accounts registered and verified
clean, protocol in `docs/EVALUATION-METHODOLOGY.md`, ~350 games/arm for +10pp
power. **The user runs live ladder sessions, not Claude.** But note: **Phase 3
still has no candidate.** 2c is rejected, and `m9seed` is statistically
indistinguishable from M7 (by construction — it is the same recipe), so pairing
them would measure nothing about the agent. Phase 3 needs a genuine candidate
first.

**Where a candidate could come from, now that 2a/2c/m9seed have narrowed it:**
1. **2d (opponent-pool saturation)** — the *opponent* leg of constraint (c),
   still untested. 2a/2c tested the *data* leg and found the effect reverses
   after RL, which makes the opponent leg more interesting, not less.
2. **Exploit the 2c finding directly.** The evidence says breadth in the BC
   corpus helps RL while alignment helps imitation. That suggests trying the
   opposite of 2a: a *broader* BC (add more gen1ou, or reweight toward it) as
   the RL warm-start, which no run has tried — every checkpoint to date used
   the 50/50 default.
3. Neither is a large commitment: ~70 min of GPU per PPO arm, now that
   throughput is known and the pipeline is shown to be reproducible.

0. **✅ DONE 2026-07-31 — M9 Phase 2b (replay backfill) is COMPLETE, and the
   answer is that there is no more gen 1 human data to get.** Both formats were
   mined to the end and the scraping lever is now closed.

   **`gen1ou`: archive fully exhausted.** `--formats gen1ou=1300 --backfill`
   ran to `no more results (page 2029); history exhausted`:
   `scanned=103,436 downloaded=634 skipped_existing=98,263 skipped_lowrated=4,539`
   → corpus 98,349 → **98,983**. Only **634 new** replays; the Metamon bootstrap
   already held 98,263 of the 103,436 replays that exist.

   **Full-archive census of gen1ou (every replay Showdown has ever kept):**

   | Rating band | Count | Share |
   |---|---|---|
   | ≥1300 | **10,674** | 10.3% |
   | <1300 | 34,462 | 33.3% |
   | unrated | **58,300** | 56.4% |
   | **total** | **103,436** | |

   **We now hold ~100% of the gen1ou ≥1300 archive.** Not "hard to get" — it
   does not exist. Any future plan that assumes more gen1ou high-rated human
   data can be scraped is wrong.

   **`gen1randombattle`: stopped early, same conclusion.** +1,387 at 10.4% yield
   (20,196 → 21,583), cursor 2018-04-05 → 2017-05-18. Filling a 30k cap would
   have needed ~5,900 pages of archive that does not exist.

   **Ceiling on gen 1 human data at the ≥1300 bar: ~32k replays total**
   (~10.7k gen1ou + ~21.6k randbats). That is the whole pool.

   **CORRECTION (2026-07-31, investigated): the "58,300 untapped unrated
   replays" claim was wrong — that harvest was already done, years ago.**
   `bc_pretrain_mlp.py:82` keeps any unrated record whose battle id is not plain
   `<format>-…`, which catches every `smogtours-` tournament game by default.
   Measured over all 99 gen1ou trajectory shards:

   | Category | records | kept by BC today |
   |---|---|---|
   | smogtours | 2,941,686 | ✅ all |
   | rated ≥1300 | 941,436 | ✅ all |
   | other-server | 66,706 | ✅ all |
   | rated <1300 | 2,380,136 | ❌ dropped |
   | unrated main-ladder | 2,121,732 | ❌ dropped |
   | **total** | **8,451,696** | **3,949,828 (46.7%)** |

   **Smogon tour games are already 74% of BC's gen1ou training data.** What is
   actually unused is the *casual* tier: 26,791 unrated main-ladder replays
   (2.12M records) plus 2.38M records of rated-<1300 play, both dropped
   deliberately as weak, not overlooked.

   **Residual option, costed:** ratings are absent on the casual tier, so the
   only available quality signal is player identity — keep games whose players
   also appear in the ≥1300 corpus (3,174 distinct players). Yield: both players
   strong **4,955 games (18.5%)**, one player strong 8,089 (30.2%), neither
   13,747 (51.3%). Seat-aware (keep only the strong seat in one-strong games)
   that is ~18,000 of 53,582 seat-trajectories ≈ **+720k records, +18%** on the
   3.95M already in use. **Not recommended as a priority:** it is a modest,
   uncertain-quality increment, and nothing in the project's evidence says data
   *volume* is the binding constraint — BC already trains on 3.95M
   tournament-dominated records and BC-only still scored 22% raw. Implement only
   if a data-scaling experiment is wanted for its own sake; ~30 lines in
   `ReplayShardDataset` plus a strong-player set built from `manifest.csv`.

   **A projection made during this run was wrong and is corrected here:** early
   yield of 14.9% was extrapolated to "~24,000 new replays, tripling the corpus."
   Realized: 634. The failure mode was assuming partial overlap with the Metamon
   corpus when overlap was near-total. Early-page yield on a backfill is not
   predictive when an unfiltered bootstrap corpus is already present.

   **The home box is authoritative for BOTH `data/replays/gen1randombattle`
   (21,583 logs) and `data/replays/gen1ou` (98,983 logs, ~400 MB)** — rsync back
   before editing either on the Mac.

   **Why it was repointed (2026-07-31).** The randbats backfill ran ~1h and was
   stopped: **20,196 → 21,583 (+1,387) at 10.4% yield**, cursor 2018-04-05 →
   2017-05-18. Extrapolating, filling a 30k cap would have needed ~5,900 pages ≈
   two decades of archive that does not exist. **Scraping is retired as a way to
   grow the randbats corpus** — that question is now answered empirically.

   **Format-volume census (replay API, 2026-07-31, 50-replay page → rate):**
   gen1randombattle 44/day, gen1ou 31/day, gen3rb 36/day, gen7rb 63/day,
   gen8rb 37/day, gen2rb 8/day — versus **gen9randombattle 1,873/day and
   gen9ou 2,407/day (~45×)**. Gen 9 is where the data and the ladder traffic
   are, and its ladder activity is the only thing that would make M9 Phase 3's
   ~350-games/arm power target cheap. **But it is not a pivot, it is a restart:**
   `sim/tools/feature-extractor.ts` hard-codes `Dex.mod('gen1')` in 5 places plus
   a 15-type table, a gen1 base-speed table and gen1 boost semantics, and gen 9
   would add abilities, items, tera, weather/terrain and ~1000 species — while
   every checkpoint M2→M8 and both replay corpora become worthless. Not
   recommended now; recorded so the option is costed rather than re-litigated.

   **Correction to a claim made earlier in this session:** gen1ou was described
   as "5× more human data than randbats." That compared an unfiltered count to a
   filtered one and is wrong. Of gen1ou's 98,349 replays only **10,101 are
   ≥1300** (55,861 unrated, 32,387 rated <1300), whereas all 21,583 randbats logs
   were scraped with the ≥1300 filter applied. At equal quality bar gen1ou had
   *less* usable data, which is exactly why the backfill above matters.
   **Untapped asset:** many of those 55,861 unrated gen1ou replays are Smogon
   tour games — unrated but high-level (noted in `models/bc_pretrain_mlp.py`'s
   docstring). The scraper *always* skips unrated entries, so they can only be
   harvested by a different selection rule, not by lowering `--min-rating`.
1. **✅ DONE 2026-07-31 — M9 Phase 1 (evaluation methodology) is COMPLETE.**
   See the Phase 1 block above. Doc + analysis script + instrumented result log,
   with three gates rewritten. No ladder time was spent.
2. **✅ DONE 2026-07-31 — Phase 3 ladder accounts registered.** `m9p3ctl` and
   `m9p3cand` both verified against `pokemonshowdown.com/users/<name>.json`:
   `"ratings":{}` on both, i.e. clean slates, which is the precondition the
   paired design needs. Credentials in `config/showdown_login_m9p3ctl.txt` /
   `..._m9p3cand.txt` (gitignored via `showdown_login_*.txt`).
   For contrast, `novapool` now reads 143W/363L over **506 games** (lifetime raw
   28.3%, GXE 32.9) spanning M6–M8 — the contamination that makes it unusable as
   a Phase 3 arm. The M7-era subset (30.5%, n=387) is the control prior.
3. **✅ DONE 2026-07-31 — the 2a-then-2c path was run, and 2a came back
   positive.** 2a cost ~25 min end to end, not the estimated 3 h (BC training
   201 s; four n=5,000 evals ~13 min).

   **Why 2a mattered more than its size suggests.** Had randbats-only BC lost,
   2c would have warm-started from the same mixed BC as M7 and been a
   seed-replication of the M7 run — informative about pipeline variance,
   useless as a test of the distribution hypothesis. 2a is what gives 2c a
   variable to test.

   **New tool: `scripts/bot_eval_ab.py`** — the bot-eval counterpart to
   `ladder_analysis.py`, importing its Wilson/Newcombe implementations. Reports
   arms as a *difference with a CI*, which is what the methodology doc requires
   and what no bot-eval A/B in this project had ever done.

   **Methodology refinement found while running 2a, worth carrying forward:
   the required n depends on the baseline win rate, and the doc's standing
   n=2,000 was derived at p=0.93.** BC checkpoints sit near p=0.35–0.45, where
   binomial variance is at its maximum and n=2,000 resolves only ~±4.5pp. At
   p=0.42 detecting +3pp needs **4,286** games/arm, versus 906 at p=0.93. 2a
   therefore ran at n=5,000 (~3 min/arm). Rule of thumb: **n=2,000 is right for
   strong agents near the ceiling; mid-range agents need 5,000.**
4. **✅ DONE 2026-07-31 — home box preflight green and 2c launched.**
   `device=cuda` (RTX 3080), node v22.20.0, `dist/` current, warm-start and
   KL-anchor both confirmed in the log header. Throughput ~21k steps/min ⇒ 5M
   steps ≈ 4 h.

   **Preflight initially FAILED and the cause is worth recording:** the home
   box's working tree was dirty with six untracked files from earlier sessions
   (five backfill/collection logs plus
   `ppo_v3_df_valft_outcome_valft.json`), so it refused to pull and would have
   run stale code against stale weights. The logs were **moved, not deleted**,
   to `logs/session-archive-2026-07/` (gitignored). The JSON was the M8 Phase 2
   run-2 metadata whose sibling on the Mac *is* tracked — it has now been
   committed, so the convention is consistent again. **Leaving stray files in
   the home-box tree silently blocks every future remote job**; clean up at the
   end of a remote session.

Collection ran on the MacBook in ~35 min (2000 games, 66,459 decisions, 84
shards, 6 workers) — data clean, obs finite at 1032 dims, seats balanced.
The `--target outcome` fine-tune worked *well* in-distribution: val MSE
0.7414 → 0.5907 against a constant-predictor baseline of 0.7396, i.e. **R²
0.00 → 0.20**, and sign agreement 0.713 → 0.800 against a base rate of 0.755.
**The PPO-trained value head was scoring below a constant predictor** —
hypothesis (a) of the M8 thesis confirmed outright.

**And none of it transferred.** That's the substantive finding: better leaf
evaluation did not produce better search outcomes. Two candidate causes were
carried forward — (1) MCTS uses *relative* leaf values and much of the MSE gain
was learning the base rate (+0.51), a constant offset; (2) the targets were
collected against a frozen raw-policy opponent (searcher won 75.5%) but the A/B
is against DamageFirst. **Run 2 (above) tested and eliminated (2).**

**Open decision (deferred, prior now weaker):** `--target mc` and `--target
root` reuse either dataset, ~1.5h for both including evals. `outcome` was the
strongest target a priori and has now failed twice, and cause (1) — the
surviving explanation — is a property of what MCTS does with leaf values, not of
which target the head is fit to, so it predicts `mc` and `root` fail too. If run
anyway, pre-commit that any arm clearing +3pp needs confirmation at 500
battles/arm. Otherwise go straight to Phase 4.

**Methodological note (project-wide): the +3pp gate at n=200 was underpowered**
— SE on the difference is ~3.9pp, so a true +3pp effect would be caught only
~1/3 of the time. Future A/Bs resolving ±3pp want ~500–800 battles per arm.
The −2.5pp delta here is inside noise, so the fine-tune is not *established* as
harmful; it simply fails a point gate.

**Multi-machine setup ✅ LIVE and verified 2026-07-29.** Tailscale + SSH
(`ssh homebox`) works from the Mac; `docs/MULTI-MACHINE.md` holds the inventory
and the three-tier sync model. Final checkpoints are committed to git
(`344def9ef`, 14 files / 24 MB), so evals and fine-tunes no longer strand on the
wrong machine.

**`scripts/homebox-preflight.sh` added 2026-07-29** — mandatory before any
remote job. Fast-forwards the home checkout (stale checkout = *wrong weights*
now that checkpoints are committed), asserts node ≥22 and `.venv` torch+CUDA,
rebuilds `dist/` when TS is newer, lists which tier-2 data dirs exist. Non-zero
exit ⇒ don't launch. **Two gotchas it encodes:** non-interactive
`ssh homebox '...'` gets system **node 18** (nvm loads only in a login shell, and
`./build` hard-rejects <22) — so wrap remote commands in `bash -lc "..."`; and
the home box's system `python3` has **no torch** — always `.venv/bin/python`.
Also fixed: MULTI-MACHINE's tmux/rsync recipes pointed at a nonexistent
`~/pokemon-showdown` (real path `~/Projects/pokemon-showdown-ai`) and a bare
`python`, so none of them would have run as written.

**Home box validated end-to-end 2026-07-29:** preflight exit 0, then
`train.py --obs-v3 --steps 400` ran on **`device=cuda` (RTX 3080)**, loss 0.230,
checkpoint saved (smoke dir deleted after). Node bridge + venv + GPU + checkpoint
path all confirmed, so a 5M-step run there needs no further setup.
**Data caveat:** the home box has almost no tier-2 data — `data/replays`,
`data/replay_trajs`, `data/metamon_cache`, `vendor/` are all absent. The one
exception as of 2026-07-30 is `data/value_targets/m8_v3_df` (4.7 MB, collected
there). Checkpoint-only jobs (PPO training, bot evals, MCTS collection) run
there immediately; BC pretrain does not (needs 1.7 GB of trajectories).
Value-target dirs are only ~5 MB each and rsync in seconds.

---

### Phase 2 infrastructure (built 2026-07-28, all of it exercised by the run above)

- `models/mcts/mcts_agent.py`: `act()` now records `last_root_visits` and
  `last_root_value` (root Q = mean backup return; NaN when no search ran at
  that decision). Search behaviour itself is untouched.
- `models/collect_value_data.py`: MCTS self-play collection. Tuned MCTS drives
  one seat (alternating per game), a frozen raw-policy checkpoint the other;
  each searched decision stores obs (sliced to the checkpoint's `obs_size`),
  root visits, root Q, the shaped reward accumulated to the next decision in
  the **searching seat's** perspective (p2 negated), and the game outcome.
  `.npz` shards flush every `--shard-games` games and re-running the same
  command resumes; `--workers N` runs N bridges in parallel. **Measured
  2026-07-29 on the M4 MacBook: ~1100 games/h per worker at `--sims 100`, so
  2000 games took ~35 min at `--workers 6`** (the earlier ~600 games/h estimate
  was conservative). Use `--workers ≈ physical cores − few`; each worker is a
  Python process *and* a Node bridge.
- `models/value_finetune.py`: value-head-only fine-tune. One dataset serves
  three targets (`--target outcome | mc | root`); split is by whole game;
  prints val MSE + sign agreement before/after. Trunk, policy head and opp head
  come out **bit-identical** (verified by a state-dict diff), so the search
  prior is unchanged and the Criterion C delta isolates leaf evaluation.

**Verified:** collection smokes (1 worker, 2 workers, resume-after-kill) —
reward signs match outcomes on both seats, obs all finite; fine-tune smokes on
all three targets; the fine-tuned checkpoint plays through
`evaluate.py --model mcts` end-to-end. All of this held up on the full run.

**Correction to the smoke-data reading:** the smoke sample suggested val sign
agreement of **0.08** against outcomes. The full 66k-decision dataset put it at
**0.794** (root Q vs eventual outcome), and the value head's *own* val sign
agreement at **0.713**. The 0.08 figure was small-sample noise and should not be
cited. The optimism it pointed at is real but far milder than implied: root Q
averages **+0.289 in games the searching seat went on to lose** (vs +0.683 in
wins) — miscalibrated specifically in losing positions, not globally broken.

**Carried gotcha (unfixed):** `train.py`'s checkpoint-dir routing checks
`elif args.opp_coef != 0.0` before the obs-schema branches, so any run without
an explicit `--opp-coef 0` lands in `checkpoints/opp/` regardless of schema.
Left alone deliberately (out of Phase 2 scope) — pass `--checkpoint-dir`
explicitly on M8 runs.


**M8 Phase 1A (obs refinement — speed-ratio dim) — ✅ COMPLETE 2026-07-28,
CRITERION A FAILED (negative result).** Built the `structured-v3-extended` schema
(87 dims/token, obs 1044): v3's 86 dims byte-identical, plus **dim 86 = own/opp
active base-speed ratio** placed identically on all 12 tokens (like the Sleep
Clause flag). Encoding: `min(ownBaseSpe/oppBaseSpe, 2) / 2` → [0,1], equal
speed = 0.5, ≥2x faster = 1.0, unknown opponent → neutral 0.5. Base speeds are
pure dex lookups (`Dex.mod('gen1').species.get().baseStats.spe`) — no gym
tracker state needed, so it's computed in the extractor from data it already
has (own = `request.side.pokemon[0]`, opp = revealed active). Rolled out across
the full v3 pipeline, mirroring the M7 pattern: `type-chart-v3.ts`
(TOKEN_DIM_V3_EXT=87, V3_SPEED_RATIO=86, `gen1BaseSpeed`/`encodeSpeedRatio`),
`feature-extractor.ts` (`V3Info.extended`, `fillSpeedRatio`),
`pokemon-gym.ts` + `battle-sim.ts` (obsMode `structured-v3-extended`),
`gym_bridge.js` (`--obs-v3-extended`), `gym_client.py`/`vec_gym_client.py`
(`obs_v3_extended`, per-token `slice_structured_obs` handles 1044→1032/924/780
automatically), `train.py` (`--obs-v3-extended` → checkpoints/v3-extended/),
`evaluate.py` (threaded through all five runners incl. cross-schema h2h),
`ladder-bot.js` (1044 → obsV3Ext auto-detect), `infer_server.py` (obs-agnostic,
docstring only). **Verified:** `./build` green; 99/99 tool tests pass (16 new
v3-extended tests — 7 extractor incl. dim 0–85 byte-equality + speed-ratio
math/placement/neutral-fallback, 2 gym reset/full-battle stability); Python
smokes clean — GymClient/VecGymClient → (12,87)/(2,12,87) finite + mutual-
exclusion guard; `train.py --obs-v3-extended` → obs_size 1044, checkpoint
saved, loss healthy; `evaluate.py` vec+serial both opponents + opp-head
accuracy; cross-schema h2h v3-extended(p1) vs v3(p2) slices 1044→1032 with no
reshape errors; MCTS eval on a v3-extended ckpt clean (~21ms latency — the
battle-sim obs path is correctly sized, no M7-style reshape bug).

**A/B run ✅ COMPLETE 2026-07-28 (user-run on home machine).** Trained
`--obs-v3-extended --steps 2000000` with the M7 opponent-mix recipe. **Gotcha
found:** the checkpoint landed in `checkpoints/opp/ppo_step_2000003_final.pt`,
not `checkpoints/v3-extended/` — `train.py`'s checkpoint-dir routing checks
`elif args.opp_coef != 0.0` (default 0.1) *before* the `--obs-v3-extended`
branch, so any run without an explicit `--opp-coef 0` gets routed to
`checkpoints/opp/` regardless of obs schema. Obs schema itself trained
correctly (1044-dim v3-extended); this is a path-naming quirk only, worth
fixing in `train.py` if the routing order ever bites again — reorder the
`elif` so `--obs-v3-extended`/`--obs-v3` schema routing takes priority over
the opp-coef default, or require an explicit `--checkpoint-dir` for the M8
A/B runs.

**Results (150 raw-policy battles each):**
| | v3-extended | v3 control |
|---|---|---|
| vs Random | 58% (87/150) | 61% (92/150, existing sweep data, no re-run) |
| vs DamageFirst | 61% (91/150) | ~55% (200/battles @ step 2500025 — nearest existing reading, not the exact 2M control step; transferring the control checkpoint to get an exact-step number was judged not worth it) |

**Verdict: Criterion A FAILED.** Random moved **−3pp** (wrong direction),
failing the "both opponents >+2pp" gate regardless of the (favorable but
inexact) DamageFirst comparison. Per the pre-registered rule: **Phase 1B
(full 5M v3-extended run) is skipped. Next: Phase 2 (AlphaZero-style
value-head targeting on the M7 checkpoint)** — see `MILESTONES.md` → M8 →
Phase 2 for the design (self-play + MCTS value targets, Criterion C gate:
≥+3pp vs DamageFirst on tuned MCTS, 200 battles).



**M8 Phase 0 (websocket reconnect) ✅ DONE 2026-07-22.** `tools/ladder-bot/
ladder-bot.js` now survives disconnects: exponential backoff (1s → 30s cap,
15 attempts, counter reset on successful login), re-login on the fresh
`challstr`, and rejoin of in-flight battles. Interrupted rooms are rebuilt
as fresh `BattleRoom`s from the server's full-log replay on `/join` (an
`|init|` frame for an existing room resets it), so trackers never process a
line twice. `updatesearch` games are joined when untracked — covers battles
the ladder matched while the bot was offline. `send()` is guarded while the
socket is down (dropped choices get re-solicited via the replayed request),
and the `[Invalid choice]` → `/choose default` fallback now ignores
"too late" errors (a stale replayed request would otherwise loop). Verified
via local bot-vs-bot smoke through a killable TCP proxy: normal 1-battle
flow unchanged; mid-battle connection kill → reconnect in 1s → re-login →
rejoin → battle completed cleanly (27 decisions, exit 0). Formal Phase 0
acceptance (one clean 100+ battle ladder session, zero manual restarts)
rides on the next long ladder run — the user's pre-authorized M7
Criterion C 50-game follow-up is the natural first exercise.

**M7 Criterion C 50-game follow-up ✅ RUN 2026-07-22/23 (user-run) — STILL
INCONCLUSIVE, but strongly directional.** 50/50 rated games in ONE clean
session, zero disconnects/manual restarts — the reconnect-enabled bot's
first real outing (nothing to reconnect from, so formal Phase 0 acceptance
still technically pending a session that actually drops). **Raw 21/50
(42%)**, up from the main run's 30%. Account after
(`pokemonshowdown.com/users/novapool.json`): **Elo 1101.4, GXE 32.9%**
(from 1034.6 / 28.2% pre-follow-up; M6 baseline 1017 / 23.9%). GXE 32.9%
is still inside the 25–34% noise band → per the pre-registered rule M7
Criterion C stays **inconclusive**, but the trend across 150 total games
is monotonic (23.9 → 28.2 → 32.9) and sits 2.1pp under the win bar. Log:
`data/replays/self_ladder/m7_ladder_followup.log`. Next: M8 Phase 1A.

**Ladder-bot `--run-id` resumable runs ✅ DONE 2026-07-23.** `--run-id
<name>` persists `{finished, wins, target}` to `<save-dir>/run_<id>.json`
after every battle; re-running the SAME command (with `--battles` as the
absolute target) resumes at the next battle — no more manual
`--battles <remaining>` arithmetic after a crash/kill. Re-running a
completed run prints a notice and exits 0. Also from the 50-game log:
deduped the doubled `/join` sends, and finished battles now retire their
base/`-suffixed` (password-alias) twin room so it can't linger as a
phantom. Verified locally: kill after battle 1 of a 2-battle `--run-id`
run → restart resumes at 2/2 → completion → rerun no-ops.

**Phase 6 (ladder, Criterion C) ✅ COMPLETE 2026-07-20 — INCONCLUSIVE.**
100/100 consecutive rated `gen1randombattle` games via
`tools/ladder-bot/ladder-bot.js --mcts` on the v3 checkpoint + tuned MCTS
(sims=100/det=1/c_puct=0.5), zero crashes, well under the 2s/move budget.
The run fragmented into three sessions due to the known (M6-documented)
lack of websocket reconnect logic in `ladder-bot.js` — two dropped
connections, no data loss, each resumed with `--battles <remaining>`. The
user ran the final 95-battle segment themselves on their own terminal after
the bot dropped twice under mine; it disconnected once more on their end
too, meaning the instability is generic (Showdown-side or the missing
reconnect logic), not specific to either execution environment.
- **Raw record: 30W–70L** (up from M6's 21W–79L).
- **Account state after the run** (`gen1randombattle`, via
  `pokemonshowdown.com/users/novapool.json`): **Elo 1034.6, GXE 28.2%**
  (M6 baseline: Elo 1017, GXE 23.9%).
- **Verdict per pre-registered bands:** GXE 28.2% falls in the 25–34%
  "noise band" — not the ≥35% needed for a clear win, though directionally
  up +4.3pp GXE / +9pp raw win rate over M6. Per the pre-committed rule,
  this is scored as **inconclusive**, not a win.
- **Contingency (not yet run):** an optional 50-game follow-up is
  pre-authorized to try to resolve the band one way or the other.
- Log: `data/replays/self_ladder/m7_ladder_run.log`.

**Job 5.2 / battle-sim.ts fix — committed 2026-07-20** (previously
uncommitted; folded into the M7 closeout commit along with the eval logs
and doc updates below).

**Phase 5 — Job 5.1 (PPO fine-tune on v3) ✅ COMPLETE 2026-07-19.** The 5M-step
run launched 2026-07-18 (see Job 5.1 entry below) finished cleanly:
`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt` (20 checkpoints + final),
loss healthy to the end, late rollout win rates 0.40–0.45 vs the mixed pool.

**Phase 5 — Job 5.2 (Criterion B evals) 🟡 IN PROGRESS 2026-07-19.**
- **Raw sweep (20 ckpts, 150 vs Random each, `v3/sweep_results.txt`):**
  43–73% band, rising through training, best = the **final 5M checkpoint at
  73%** — the v2 bcft sweep never exceeded 58%.
- **Raw confirmations (`v3/confirm_results.txt`):** final **70.0% (350/500) R
  / 52.0% (104/200) DF**; 4.75M 65.2%/58.5%; 2.5M 69.0%/55.0%. All three are
  the best raw-policy numbers in project history (prior best raw: 57% R
  M3.3 / 54.6% R bcft). MCTS base = the final checkpoint (best R, shipping
  convention; DF differences within ±7pp noise).
- **Bug found & fixed (uncommitted): `sim/tools/battle-sim.ts` had no v3
  support** — `_extractObsFor` sized non-v2 structured obs at TOKEN_DIM (780)
  and passed no v3Info, so every `--model mcts --obs-v3` eval crashed with
  "cannot reshape 780 into (12,86)". Job 3.1's "MCTS verified
  obs-shape-agnostic" claim covered only the Python side (`mcts_agent.py`);
  the Node sim host was never exercised under v3. Fixed to mirror
  `pokemon-gym.ts` (volatiles for v2+v3, `{sleepClause}` v3Info from the
  tracker). `./build` green, battle-sim 13/13 + gym 45/45, MCTS v3 smoke
  3/3 wins @ ~57ms mean search latency.
- **Tuned-MCTS battery ✅ COMPLETE 2026-07-19 — CRITERION B MET, NEW BEST
  AGENT** (`models/mcts/results/m7_v3_mcts_{random,damagefirst}.log`,
  tuned defaults sims=100/det=1/c_puct=0.5): **93.0% (465/500) vs Random /
  84.2% (421/500) vs DamageFirst** — both bars (≥80% R, ≥65% DF) cleared,
  and both ahead of the prior best (v2 bcft + tuned MCTS: 90.6%/79.2%;
  +2.4pp / +5.0pp). Search latency 56ms mean / 107ms p95. Per the
  pre-registered conditional rule, Phase 6 (≥100-game ladder run,
  Criterion C GXE bands) is mandatory next.

---

## Recently Completed

**M8: Value-Head Targeting + Ladder Infrastructure — ✅ COMPLETE 2026-07-31,
all bets negative or inconclusive**
Contingency-based escalation. Phase 0 (websocket reconnect) ✅ 2026-07-22.
Phase 1A (obs refinement, speed-ratio A/B) ❌ 2026-07-28 — Criterion A failed
(Random −3pp), Phase 1B skipped. Phase 2 (AlphaZero value targeting) ❌
2026-07-29, **replicated ❌ 2026-07-30** — infra built and fully exercised, but
Criterion C failed (82.5% → 80.0%, −2.5pp vs a ≥+3pp bar) and failed *again* at
exactly 160/200 when the targets were re-collected against DamageFirst to rule
out distribution mismatch, so Phase 3 is skipped. The two structural ideas
M8 was built to test have both come back negative, and the shipping agent is
still the unchanged M7 checkpoint. **Phase 4 (ladder validation on M7) 🟨
INCONCLUSIVE 2026-07-31** — 100 rated games, raw **27/100 (27.0%)**, account
`novapool` after: **Elo 1084.0, GXE 32.9%**, inside the 25–34% band for the
third consecutive reading.

**The finding that matters is about the instrument.** Phase 4 ran the *same
checkpoint* as the 7/23 follow-up, yet raw win rate went **42% → 27%** and Elo
fell 17 points. The model did not change. That retires the "monotonic
23.9 → 28.2 → 32.9" reading recorded on 7/23 — it was ladder drift, not
progress. Two defects, both now understood: GXE is an **account-level
cumulative** statistic over 506 games spanning M6–M8 (it did not move at all
this run: 32.9 → 32.9), so it was never a valid per-run gate; and raw win rate
is not comparable across runs at different Elo, since climbing means facing
stronger opponents. **Every M6/M7/M8 ladder conclusion rests on these two
statistics** — which is why M9 gates evaluation methodology ahead of any further
training spend. See `MILESTONES.md` → M8 → Phase 4.

**M7 (Observation Schema v3) — ✅ COMPLETE 2026-07-20**
Criterion A ✅, Criterion B ✅ (new best agent: 93.0% R / 84.2% DF), Criterion C 🟡 inconclusive (Elo 1034.6 / GXE 28.2% vs M6's 1017/23.9%, landed in 25–34% noise band). v3 adds type effectiveness, move-effect flags, Sleep Clause tracking — the highest-leverage obs features humans use every game. Bug fix: battle-sim.ts now supports v3 (obs-shape-agnostic MCTS now truly holds). Full results in `MILESTONES.md` → M7.

**Phase 0 — Job 0.1 (Type chart + move-effect flags + encoding) ✅ DONE
2026-07-17.** Created `sim/tools/type-chart-v3.ts` — the spec-lock layer Jobs
1.1/1.2 import. All lookups sourced from the engine's own gen1 dex
(`Dex.mod('gen1').getEffectiveness`/`getImmunity`; move-data fields), no
hand-transcribed tables or name lists. Exports: `TOKEN_DIM_V3 = 86`, dim
offsets (`V3_TYPE_EFF=77`, `V3_FLAG_RECHARGE/SELFKO/PRIORITY=81/82/83`,
`V3_INFLICTED_STATUS=84`, `V3_SLEEP_CLAUSE=85`), `computeTypeEffMultiplier`
/ `encodeTypeEff` (mult/4 → [0,1]; 0x immunity → 0.0) / `typeEffDim`,
`getMoveEffectFlags` + `MoveEffectFlags`, `MOVE_STATUS_ID`. Verified vs engine:
Ice→Dragonite 4x, Fire→Slowbro 0.5x, Explosion→Gengar 0x; Hyper Beam recharge,
Explosion self-KO, Quick Attack priority, Rest→status 0 (self-sleep excluded).
`./build` green. Next: Jobs 1.1 + 1.2.

**Phase 1 — Job 1.1 (Feature extractor v3 logic) ✅ DONE 2026-07-17.** Extended
`sim/tools/feature-extractor.ts`: added optional 4th param `v3Info` to
`extractFeaturesStructured(request, opponent, volatiles?, v3Info?)` — passing
it (shape `{ sleepClause: boolean }`, even when false) selects the 86-dim v3
schema (`TOKEN_DIM_V3`, re-exported). Fills, on every token, dims 77–80 (that
token's ≤4 move slots' type-eff vs the opponent active's types, via
`typeEffDim`; empty slot / unknown / 0x → 0.0), 81–83 (recharge/self-KO/priority
OR-aggregated over the move set), 84 (first inflicted-status id, normalised
`/MOVE_STATUS_ID_MAX` → [0,1]), and 85 (global Sleep Clause flag placed
identically on all 12 tokens). Dims 0–76 untouched — kept byte-identical to v2
(v3 fills the v2 prefix via the existing volatiles path when volatiles are
supplied alongside v3Info). No NaN/inf on unknown move/species/switch/empty
inputs. New tests: `test/tools/feature-extractor.test.js` (22 passing, incl.
v2/v3 and v1/v3 byte-equality on dims 0–76, 4x/0.5x/0x matchups, effect-flag and
inflicted-status cases, edge cases). `gym.test.js` + `replay-adapter.test.js`
still green (43 passing). `./build` green. Did NOT touch `pokemon-gym.ts`
(Job 1.2) or `gym.test.js` (Job 3.1).

**Phase 1 — Job 1.2 (Sleep Clause tracker + obsMode 'structured-v3') ✅ DONE
2026-07-17.** `sim/tools/pokemon-gym.ts` only. Added a per-side Sleep Clause
tracker to `ObservationTrackers` (`sleepInflicted: {p1,p2} Map<nickname,boolean>`
+ `sleepClauseFlagFor(viewer)`), driven from public log lines via a new
`_processSleepClauseLine`: `|-status|pXa: N|slp` sets the flag on that side
(EXCLUDING `move: Rest` self-sleep); `|-curestatus|...slp` and `|faint|` clear
it; NOT reset on switch/drag (bench-persistent). `sleepClauseFlagFor('p1')` reads
p2's map so the flag means "an opponent WE slept is still asleep". Wired
`obsMode: 'structured-v3'`: `_getVolatilesFor` now also serves v3 (keeps dims
65–76), new `_getV3InfoFor` supplies `{ sleepClause: boolean }` to Job 1.1's 4th
extractor param, and the terminal filler sizes v3 at `N_TOKENS * TOKEN_DIM_V3`
(1032). Consumes Job 1.1's landed `V3Info` type + `TOKEN_DIM_V3` (from
`type-chart-v3`). Sleep tracker state round-trips through snapshot/fromSnapshot
(BattleSim/M4-safe). New tests: `test/tools/gym.test.js` "Sleep Clause tracker"
suite (7 tests: set on opponent sleep, NOT on Rest, persist across switch, clear
on faint, clear on wake, second-sleeper, snapshot round-trip). `gym.test.js` 36
passing; `./build` green; `tsc` clean on `pokemon-gym.ts`; e2e smoke: v3 reset +
40 steps all 1032-dim, no NaN/inf.

**Phase 2 — Job 2.1 (Bridge / gym-client serialization for v3) ✅ DONE
2026-07-17.** Mirrored the existing `--obs-v2` pattern in exactly the three
files scoped: `models/gym_bridge.js` (recognizes `--obs-v3`, mutually
exclusive with `--flat`/`--obs-v2`, requests `obsMode: 'structured-v3'` — no
other bridge changes needed since obs arrays are forwarded length-agnostic),
`models/gym_client.py` (`TOKEN_DIM_V3 = 86`, `GymClient(obs_v3=True)` sets
`--obs-v3` + `_token_dim`, mutual-exclusion guard vs `obs_v2`;
`slice_structured_obs` needed no code change — it's already shape-generic, so
v3→v2 (86→77) and v3→v1 (86→65) per-token slicing work automatically, doc
comment updated to describe it explicitly), `models/vec_gym_client.py`
(`obs_v3` param passed straight through to each `GymClient`; no separate
obs-version-mixing logic exists today to update). Smoke-verified: single-seat
`GymClient(obs_v3=True)` reset+5 steps → (12,86), no NaN/inf; dual-seat
selfplay v3 → 30 steps, both seats (12,86), no NaN/inf; `VecGymClient(n_envs=2,
obs_v3=True)` → 300 steps with 9 auto-resets, (2,12,86), no NaN/inf; v2 path
(`GymClient`/`VecGymClient(obs_v2=True)`) still produces unchanged (12,77)/(2,12,77)
shapes; `obs_v2=True, obs_v3=True` together raises `ValueError`. `./build`
green (no TS changes this job). Did NOT touch `pokemon-gym.ts`,
`feature-extractor.ts`, `train.py`, `evaluate.py`, or `test/tools/*` (out of
scope per job spec). Next: Job 2.2 (replay-adapter regen) can proceed in
parallel; Job 3.1 depends on both 2.1 and 2.2.

**Phase 3 — Job 3.1 (Trainer/Eval/Infra wiring for v3 + smokes) ✅ DONE
2026-07-18.** Wired `--obs-v3` end-to-end across the training/eval/inference
scripts, mirroring the established `--obs-v2` conventions:
- `models/bc_pretrain_mlp.py`: `--obs-v3` (obs_size 1032, defaults `--traj-dir`
  to `data/replay_trajs/v3`); a first-batch shape assertion (924→1032);
  per-format val accuracy reporting unchanged.
- `models/ppo/train.py`: `--obs-v3` flag (implies `--structured`, mutually
  exclusive with `--obs-v2`), obs_size auto-inferred from the env (1032),
  checkpoints → `checkpoints/v3/`, `obs_v3` passed to `VecGymClient`.
- `models/evaluate.py`: `--obs-v3` threaded through all five `_run_battles*`
  runners → `VecGymClient`/`GymClient(obs_v3=)`. v3-vs-v2 h2h works via the
  existing per-token `slice_structured_obs` (v2 vs-checkpoint sliced 1032→924;
  dims 0–76 of a v3 token == a native v2 token). Mutually exclusive with
  `--obs-v2`; ppo/mcts only.
- `models/infer_server.py`: no code change needed — obs_size is read from the
  checkpoint hparams (1032 for v3) and act/act_tracked pass obs at that width;
  docstring updated to note v3.
- `models/mcts/mcts_agent.py`: VERIFIED obs-shape-agnostic (its `_fit_obs`
  slices to the checkpoint's obs_size via `slice_structured_obs`, works at
  1032). NOT modified.
- `tools/ladder-bot/ladder-bot.js`: the bot has never had an `--obs-v2` flag —
  it infers the schema from the checkpoint's `obs_size` at ping time. Extended
  that detection to v3 (1032 → `obsV3`), builds the obs with volatiles + the
  Sleep-Clause `v3Info` (`{ sleepClause: trackers.sleepClauseFlagFor(seat)===1 }`),
  and passes `obs_mode: 'structured-v3'` to `act_tracked`. (Deviation from the
  job's literal "forward --obs-v3 flag": faithful mirror of the existing
  auto-detect design, which has no obs flag.)
- `test/tools/gym.test.js`: +9 tests (36→45 passing). New "schema v3 (M7)"
  extractor block: v3 shape (1032, no NaN/inf), v2-prefix byte-equality
  (dims 0–76 of v3 == v2), type-eff matchups (Ice→Dragonite 4x=1.0,
  Flamethrower→Slowbro 0.5x=0.125, Explosion→Gengar 0x=0.0 + self-KO flag),
  Sleep-Clause dim-85 placement, and tracker→obs integration; plus gym-level
  structured-v3 reset shape and a full-battle v3 stability run (shape stable,
  no NaN/inf, all extension dims in [0,1]).
- Smokes (all clean): BC pretrain `--obs-v3 --max-shards 1 --epochs 1` →
  obs_size 1032, val acc randbats 0.408 / gen1ou 0.338, checkpoint saved;
  `train.py --obs-v3 --steps 400 --num-envs 2` → obs_mode=v3 (obs_size 1032),
  final checkpoint saved; `evaluate.py --model ppo --obs-v3` v3-vs-v2 h2h and
  single-opponent (vs Random) both ran without error; `infer_server.py` ping on
  the v3 checkpoint → `obs_size: 1032`, `act` on a 1032 obs → valid action.
  `gym.test.js` 45/45. `./build` green (no TS source changed this job).

**Previous: M5.5 (Human Replay Data + BC for the MLP) — ✅ COMPLETE
2026-07-16. POSITIVE RESULT, NEW BEST AGENT** (bcft final + tuned MCTS:
90.6% R / 79.2% DF / 78.4% h2h vs M5 best). Full verdict in
`MILESTONES.md` → M5.5. M6 shipped `bcft/ppo_step_5000000_final.pt`.

Project decision: competitive target stays **gen1randombattle**; training
data is multi-format — high-Elo (≥1300) randbats replays + high-level gen1ou
("how pros play with real teams"), per-format sampling weights as the knob.
Future follow-up (out of scope): fine-tune on one specific OU team.

### Active Tasks (M5.5)
- [x] Phase 1: `scripts/scrape_replays.py` — public replay-API scraper
      (top-up + `--backfill` modes, per-format rating floors, manifest,
      census). Smoke-verified all modes; committed `880eee5ff`.
- [x] Phase 1b: `scripts/bootstrap_gen1ou_replays.py` — 98,349 gen1ou raw
      logs bulk-imported from HF metamon-raw-replays (only parquet shards
      34–36 carry gen1* rows; 10.0k rated ≥1300, 55.9k unrated incl.
      tournament games). Committed `d2c5e0bf1`.
- [x] Phase 2: `sim/tools/replay-adapter.ts` + `models/replay_adapter_cli.js`
      + `test/tools/replay-adapter.test.js` (7 tests incl. byte-identical
      gym round-trip). Coverage 91% randbats / 86% gen1ou. Committed
      `5a670798a`.
- [x] Phase 3: `models/bc_pretrain_mlp.py` — smoke: 1 epoch on 2 shards →
      val acc 0.319 / opp-acc 0.300 (chance ~0.11); checkpoint plays via
      `evaluate.py --obs-v2` unchanged. Committed `875c1aa77`.
- [x] Data collection: full gen1ou→trajectory conversion done (98,349
      battles → 8.45M decisions, 0 parse errors, 746MB). **Randbats backfill
      completed 2026-07-16**: +12,051 new high-Elo logs → **20,194 on disk**
      (log `data/replays/gen1randombattle_backfill_20260716.log`; census of
      scanned pages: bulk of rated games sit at 1000–1299, below the 1300
      floor — the 1300+ tail is thin, ~3.3k per ~30k scanned).
      Re-converted: **20,133 battles → 1,161,658 decisions, 0 parse errors,
      90.7% label coverage, 21 shards** in
      `data/replay_trajs/gen1randombattle/`
      (log `data/replays/gen1randombattle_adapter_20260716.log`). A future
      BC re-run can now train on ~2.4x the randbats decisions.
- [x] Phase 4 (done 2026-07-16): **BC runs done.** Run 1 (policy+opp
      heads): val acc 49.7% randbats / 53.1% gen1ou (chance ~11%), opp-acc
      ~35% — but raw play only 20% vs Random / 20.5% vs DamageFirst; ~50%
      human-imitation accuracy does NOT transfer to raw play vs bots.
      Run 2 added `--value-bc-coef` (outcome-trained value head, 62/67%
      sign-acc — needed for search; plain policy-BC leaves the value head
      at init, crippling MCTS): raw unchanged (22%R / 14.5%DF).
      **MCTS over the BC net (final, 500+500): 56.2% vs Random / 45.0% vs
      DamageFirst** (`models/mcts/results/m55_bc_mcts_*.log`) — search lifts
      the BC net hugely (raw 22%/14.5%) but BC-only is decisively below the
      72.6%DF/86.0%R bars. BC-only verdict: negative; the fine-tune is the
      remaining shot.
      **Contingency triggered:** M3.2 fixes ported to `models/ppo/train.py`
      (`--pretrain-checkpoint`, `--value-warmup-steps`, `--bc-anchor`
      constant-coef) — 5M-step anchored fine-tune running
      (`models/ppo/checkpoints/bcft/`, pool seeded M2+M3.3-best, M5
      opponent-mix recipe). Externally stopped at 2.02M, resumed at user
      request (`--resume ppo_step_2000030.pt --bc-anchor ...`).
      **Interim 8-checkpoint sweep (150 battles vs Random):** 31% @ 250k →
      45/45 → 51% @ 1M → 53% @ 1.25M → 36/37 (dip) → 49% @ 2M. The anchored
      recipe lifts BC's 22% raw into the ~50% band — first BC→RL transfer
      in the project that improves rather than erodes.
      **Run complete 2026-07-16: 5,000,000/5,000,000 steps**
      (`ppo_step_5000000_final.pt`, 21 checkpoints incl. final; loss healthy
      to the end, late rollout win rates 0.4–0.9 band vs the mixed pool).
      **Full 20-checkpoint sweep (150 vs Random,
      `bcft/sweep_results.txt`): 35–58% band, no collapse** — the anchored
      recipe holds BC→RL gains across all 5M steps (BC raw was 22%).
      **Confirmations (`bcft/confirm_results.txt`, 500R/200DF):** best is the
      **final 5M checkpoint at 54.6% (273/500) vs Random / 42.0% (84/200) vs
      DamageFirst**; 3.75M best-DF at 46.0% (92/200) with 52.4% R (within
      noise); the 58% sweep reading @4.5M did not confirm (48.8%/34.5%).
      Raw verdict: fine-tune ≈ M2/M3.3-class peer (50–55% band), well above
      raw BC but not above the bot-trained lineage.
      **Tuned-MCTS battery on the final checkpoint
      (`models/mcts/results/m55_bcft_mcts_*.log`): 90.6% (453/500) vs
      Random / 79.2% (396/500) vs DamageFirst — BOTH pre-registered bars
      cleared decisively** (prior best, M5 ckpt + tuned MCTS: 86.0%/72.6%;
      +4.6pp and +6.6pp). Human-data BC → anchored PPO fine-tune → search
      is the first pipeline to beat the bot-trained lineage.
      **Seat-balanced h2h, MCTS(bcft final) vs the raw M5 best checkpoint:
      78.4% (392/500) combined — 77.6% (194/250) p1 / 79.2% (198/250) p2**
      (`m55_bcft_h2h_*.log`; the p1 arm crashed twice ~battle 130–150 on a
      Node `sim_fork` error and was completed as 2×125 fresh-process
      chunks — see MILESTONES → M5.5 → Known issue; root-cause TODO for M6).
      **VERDICT: M5.5 POSITIVE — new best agent** =
      `models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt` + tuned
      policy-sampler MCTS. All pre-registered criteria ✅; full write-up in
      `MILESTONES.md` → M5.5; ledger updated in `docs/MODEL-COMPARISON.md`.
      M6 ships this checkpoint (M5 ckpt becomes the ladder-A/B control).
      Side note: the human-data opp head reads DamageFirst at only ~19-24%
      (vs the bot-trained M5 head's 30-36%) — the M5 "mixture
      miscalibration" reading, confirmed from the other direction.
- [x] M6 Phase 1 build (committed `e85a6c5e5`): `tools/ladder-bot/` +
      `models/infer_server.py` per the revised M6 spec in MILESTONES.
      **Verified 2026-07-17:** local-server smoke (bot vs bot, 2 battles,
      bcft v2 checkpoint) clean — mirrored results, 38–41 decisions/battle,
      max latency 182ms cold / 2ms warm, zero invalid choices; saved logs
      are raw protocol streams (replay-adapter compatible). Added
      `--login-file` so credentials stay off argv/env; login lives at
      `config/showdown_login.txt` (gitignored via `/config/*` + explicit
      `showdown_login.txt` entry, chmod 600).
- [x] M6 Phase 1 official-ladder shakedown (account: Novapool, 2026-07-17):
      3 rated battles, 0/3 won but mechanically clean — no invalid choices,
      timeouts, or crashes; latency 4ms warm / 264ms cold; logs + CSV in
      `data/replays/self_ladder/`. Losses at fresh-account Elo are not yet
      a signal; Elo measurement is the criterion run.
- [x] M6 Phase 2 wiring (2026-07-17): `BattleSim.fromTracked` (core was
      already committed, `611c14d6c`) is now reachable from the ladder —
      `gym_bridge.js` `sim_from_tracked` command (no live env needed);
      `models/infer_server.py --mcts` hosts `MCTSAgent` over a
      `TrackedSimClient` (spawns a bridge subprocess as sim host; raw-policy
      fallback on any search error, reason reported); `ladder-bot.js --mcts
      [--sims/--determinizations/--c-puct]` searches clean move requests and
      answers force-switches/locked states with the raw policy per
      fromTracked's contract; per-battle `searched` count logged.
      **Verified:** `battle-sim-tracked` suite 7/7; local bot-vs-bot smoke
      (MCTS vs raw, 2 battles): 28/32 and 23/28 decisions searched, 0
      fallbacks/errors, ~100ms warm / 574ms cold — far inside the 2s budget.
- [x] M6 criterion run — ✅ **COMPLETE 2026-07-17, all three criteria
      met; M6 CLOSED.** First attempt (raw policy) externally stopped at
      2/15; full run with `--mcts` finished **21/100 won**, 100/100
      mechanically clean (zero invalid choices/timeouts/crashes, max
      latency 579ms cold / ≤540ms warm vs the 2s budget). **Final ladder
      read (the external measurement): Elo 1017 (floor 1000), GXE 23.9%,
      Glicko-1 1281 ± 37, account 23W–96L.** MCTS lift replicated
      externally (raw ~13% → 21%; early pace 29% before a mid-run losing
      streak). Honest note: 13/21 wins were ≤9-decision opponent
      forfeits; full-battle win rate ~9%. Verdict + qualitative failure
      analysis (no type effectiveness / stats / move effects in obs →
      obs v3 case) in `MILESTONES.md` → M6. All game logs in
      `data/replays/self_ladder/`. Known gap (carried): no websocket
      reconnect logic.

**Previous: M5 (Opponent Modeling Head) — ✅ COMPLETE 2026-07-16. Thesis negative,
new-best side finding. Full results + verdict in `MILESTONES.md` → M5.**
- **C3 (the thesis) failed:** under tuned MCTS, sampling opponent actions
  from the trained head is parity-to-worse vs the existing policy sampler
  (70.4% vs 72.6% on DamageFirst, −2.2pp against a ≥+3pp bar). C2/C4/C5 ✅,
  C1 ❌ (35.8% accuracy vs the 40% bar; above the 25% label-bug floor —
  reads as a partial-observability ceiling, labels were oracle-verified).
- **Side finding — new best agent:** the M5 checkpoint under standard
  policy-sampler tuned MCTS confirms **72.6% (363/500) vs DamageFirst /
  86.0% (430/500) vs Random** — nominally ahead of the prior best (v2:
  70.2%/82.6%) on both, within ~1σ. Consistent with the aux loss shaping
  the trunk (secondary hypothesis) even though the sampler didn't win.
  Checkpoint: `models/ppo/checkpoints/opp/ppo_step_5000001_final.pt`.
  `docs/MODEL-COMPARISON.md` ledger updated.
- **Decision:** `--opp-sampler policy` stays the default; head mode kept
  (it's cheaper, 57ms vs 59–64ms). The opp head stays in the training
  recipe. M6 ships the M5 checkpoint (or ladder-A/Bs it vs the v2 control).
- Post-run cleanup pass done (code-simplifier findings below). Remaining
  before M6: commit results/docs.

**Previous: M5 build (all phases complete, committed `ffd14e275`).**
- Auxiliary opponent-action-prediction head (`Linear(128, 9)`) on the
  MLP-PPO shared trunk, multi-task loss `PPO + λ·CE` (λ=0.1). Labels are
  the opponent's resolved choice each decision point, captured
  omnisciently in `pokemon-gym.ts` and mapped into the **opponent's own
  9-way action frame** (moves by opponent request-slot, switches by
  opponent bench order; label = -1 masked out of CE when no simultaneous
  choice exists or the choice is unmappable).
- **Primary payoff is search integration:** MCTS's opponent-action sampler
  currently assumes the opponent plays like us (base policy on the
  opponent's reveal-tracked obs); the head replaces that with the actual
  opponent-action distribution, evaluated on the searcher's own obs
  (`opp_sampler: 'policy'|'head'`, A/B'd under tuned search).
- **Design decisions made:** obs schema **v2** base (`--obs-v2` +
  opponent-mix recipe — boosts/volatiles are direct predictors of the
  opponent's next action, and the M3.4 v2 checkpoint becomes the λ=0
  control); **fresh 5M-step decision run**, not warm-started (avoids the
  M3.2-style random-head-wrecks-mature-trunk risk; a run is only ~2h);
  raw-policy gain explicitly *not* expected — no-regression is the gate,
  the head-vs-policy sampler A/B under tuned MCTS is the thesis (C1–C5
  pre-registered in `MILESTONES.md` → M5).
- **Phase 0 first: commit the post-M4 tuning working tree** (tuned
  `MCTSAgent`/`evaluate.py` defaults, mcts_agent cleanup, sweep/A/B logs,
  doc updates) so M5 code starts from a clean tree.

### Active Tasks (M5)
- [x] Phase 0: commit uncommitted post-M4 tuning changes (clean tree)
      (done 2026-07-16: the tuning code/logs were already in `84750027c`;
      the M5 plan docs committed as `0127bf3f1`)
- [x] Phase 1A: `sim/tools/pokemon-gym.ts` — opponent-frame `oppAction`
      labels in `info` (single-seat + dual-seat) + `test/tools/gym.test.js`
      coverage (move/switch mapping, force-switch masking, dual symmetry,
      locked-choice labeling)
      (done 2026-07-16: labels derived omnisciently from the battle engine's
      committed choice — `sides[opp].choice.actions[0]`: `moveSlot` → 0–3,
      `moveid` fallback for auto-completed locked moves (recharge/multi-turn,
      the M4 gotcha) → 0–3, switch `target` index in `side.pokemon` → 4–8;
      −1 for wait/team-preview requests, Struggle, no-op sleep/freeze, or
      unmappable choices. Single-seat: captured before submitting p1's choice
      via `_captureOppLabel('p2')` (bounded I/O flush so the opponent's async
      choice has committed). Dual-seat: symmetric `oppAction`/`oppActionP2`
      from the caller-supplied per-seat actions (already in each seat's own
      frame). Verified vs the opponent's actual submitted choice across
      DamageFirst/Random/self-play seeds: 0 mismatches. 6 new tests + `./build`
      clean + all 29 gym tests green.)
- [x] Phase 1B (parallel with 1A): `models/ppo/ppo_agent.py` opp head +
      masked aux CE in `update()` + two-way checkpoint compat;
      `models/ppo/trajectory_buffer.py` label storage + merge
      (done 2026-07-16: `opp_head = Linear(128,9)` off shared trunk; λ via
      `update(..., opp_coef)` + `self.opp_coef` default 0.0 ⇒ bit-for-bit
      old path, verified; masked CE with all-masked/all-λ=0 NaN guards;
      old ckpts load with fresh head + optimizer-reset warnings, new ckpts
      load clean; buffer `opp_action` (-1=masked) stored + carried by merge)
- [x] Phase 2C (needs 1A+1B): plumbing — `models/gym_bridge.js` (both
      protocols), `models/gym_client.py`, `models/vec_gym_client.py`,
      `models/ppo/train.py` `--opp-coef` (default 0.1; 0 reproduces old
      loss path; checkpoints → `checkpoints/opp/`)
      (done 2026-07-16: `gym_bridge.js` already forwarded the gym's `info`
      object verbatim in both single- and dual-seat `step` responses, so
      `oppAction`/`oppActionP2` needed no bridge code changes — just doc
      comments confirming the passthrough is intentional; `gym_client.py`
      adds a `with_opp_action()` helper surfacing the snake_case
      `opp_action` (int, -1 = masked) in `step()`/`step_dual()` info dicts,
      reused by `vec_gym_client.py`'s `step()`/`step_dual()`; `train.py`
      collects `opp_action` into every buffer push across all three rollout
      paths (single-seat, opponent-mix random/damagefirst, and selfplay
      dual-seat — the dual-seat `pending` transition is labeled from the
      SAME `step_dual()` call that submitted the p1 action, not a later
      one still accumulating reward), adds `--opp-coef` (default 0.1,
      passed into `PPOAgent(...)` construction so it's persisted in
      hparams), routes checkpoints to `checkpoints/opp/` by default when
      `--opp-coef != 0`, and logs per-rollout opp-label coverage (only when
      `--opp-coef != 0`, so the `--opp-coef 0` log output is byte-identical
      to the pre-M5 path). Verified: bridge smokes (single-seat + `--selfplay`,
      `--obs-v2`) show `opp_action` in `[-1,8]` with dual-seat labels
      matching the actual simultaneous opponent action 100% of the time;
      `VecGymClient` smoke same; `train.py` smokes at `--num-envs 2` —
      `--opp-coef 0.1` (finite loss, coverage 77–96%), `--opp-coef 0`
      (old log format, no coverage line), opponent-mix incl. selfplay at
      `--opp-coef 0.1` (finite loss, coverage logged across all three
      families), default checkpoint routing to `checkpoints/opp/`
      confirmed and cleaned up, checkpoint round-trip confirms `opp_coef`
      persisted in hparams and `opp_head` present after `PPOAgent.load()`)
- [x] Phase 2D (parallel with 2C; needs 1B, accuracy half needs 1A):
      `models/evaluate.py` opp-accuracy reporting + `--opp-sampler`;
      `models/mcts/mcts_agent.py` head-based opponent sampler with
      policy fallback
      (done 2026-07-16: `evaluate.py` — `_checkpoint_has_opp_head()` detects a
      headed ckpt by the saved `opp_head` key (robust to how the trainer
      propagates `opp_coef`); PPO single-opponent eval (vec + serial) reports
      opp-prediction top-1 accuracy = opp-head argmax vs `info["opp_action"]`
      over unmasked (label ≥ 0) steps, printed alongside win rate (n/a when no
      labels present, e.g. Job C plumbing absent); added `--opp-sampler
      head|policy` (default policy) forwarded to `MCTSAgent`. `mcts_agent.py` —
      `opp_sampler`/`has_opp_head` ctor args; `head` mode caches the opp head's
      masked+renormalized dist on each node from the SAME trunk forward that
      makes the PUCT prior (`_policy_value_opp` + `_head_opp_dist`), masking to
      the opponent's legal actions (`state[opp]["mask"]`, opponent's own frame —
      the frame the sim accepts, no remapping) and renormalizing; `_advance`
      consumes it for the node's first simultaneous opponent choice, falling
      back to the policy sampler for subsequent opponent-only force-switches;
      auto-falls-back to policy (with warning) when the ckpt has no trained head
      (via explicit `has_opp_head` from the file, else `opp_coef>0` hparam).
      Tuned defaults (sims=100, c_puct=0.5, det=1) and the locked-choice
      raw-policy root fallback preserved. Verified: both files compile; raw
      obs-v2 eval of the pre-M5 v2 ckpt still runs (no accuracy line, correctly
      no opp head). MCTS head/policy smoke on the v2 ckpt was PENDING (blocked
      by a harness outage) — now confirmed in Phase 3 below: headless ckpt
      warns + falls back to policy correctly; headed ckpt runs clean under
      both `--opp-sampler head` and `--opp-sampler policy`.)
- [x] Phase 3: smoke battery — TS/gym suites green, label spot-check vs
      DamageFirst, 4k-step train smoke (CE decreasing), `--opp-coef 0`
      regression smoke, MCTS head-sampler smoke; commit code
      (done 2026-07-16 by tester — see "Test Results — M5 Phase 3" below.
      All 6 steps PASS: `./build` clean; gym suite 29/29 incl. 6 new M5
      label tests, battle-sim 13/13; bridge smokes (single- + dual-seat
      `--obs-v2`) show `opp_action` in `[-1,8]`; 4k-step train smoke
      (`--opp-coef 0.1`, mixed opponents) finite loss + coverage 78–96%,
      checkpoints routed to `checkpoints/opp/`; `--opp-coef 0` smoke
      reproduces the old log format (no coverage line) and exercises
      checkpoint-compat (old ckpt loads into headed agent with warnings,
      no crash); DamageFirst-only label sanity — opp-accuracy 32.4%
      (266/822), well above the ~11–25% chance band; all four
      `evaluate.py` load paths pass, including the previously-unverified
      MCTS head-sampler fallback (headless ckpt warns + falls back) and
      head/policy runs on a headed ckpt. Smoke checkpoints cleaned up;
      code committed as `ffd14e275`.)
- [x] Phase 4: 5M-step decision run → sweep → confirmations → sampler A/B →
      verdict recorded (done 2026-07-16). Run trained stably to 5,000,001
      steps (externally stopped at 2.93M and resumed — `--resume` added to
      `models/ppo/train.py` for this, commit `589d8bf1e`; pool seeded with
      M2 + M3.3 best per the M3.4 recipe). 21-checkpoint sweep: 40→50–59%
      band, no collapse (`sweep_results.txt`). Confirmations
      (`confirm_results.txt`): best raw 57% (285/500) vs Random @ 5.0M final /
      44% (88/200) vs DamageFirst @ 3.0M; best opp accuracy 35.8% vs DF.
      Sampler A/B (tuned MCTS, 500/arm/opponent, `models/mcts/results/
      m5_ab_*.log`): head 70.4% DF / 85.8% R vs policy 72.6% DF / 86.0% R —
      **C3 thesis negative (−2.2pp, parity)**; C2/C4/C5 ✅, C1 ❌ (35.8% vs
      40% bar). Side finding: **M5 final + policy-sampler MCTS is the new
      best agent (72.6% DF / 86.0% R)**. Full verdict in `MILESTONES.md` → M5.
- [x] Post-run cleanup pass (code-simplifier review of `ffd14e275`,
      2026-07-16 — all 5 applied, verified end-to-end (single-opponent,
      h2h, and MCTS head/policy-sampler eval runs), `./build` +
      `test/tools/gym.test.js` green):
      (1) `PPOAgent.load()` now sets `agent.has_opp_head` directly (default
      `True` from `__init__`, so it's meaningful on non-loaded agents too);
      `evaluate.py` reads it instead of re-torch.loading the checkpoint via
      the now-deleted `_checkpoint_has_opp_head()`; (2) kept `oppActionP2`
      (cheap — already computed alongside `oppActionP1` — and covered by the
      dual-seat symmetry test) but added a `opp_action_p2` snake_case alias
      in `gym_client.with_opp_action()` for consistency, matching
      `opp_action`; (3) `_captureOppLabel()` now logs when the opponent's
      choice is still pending after 3 flushes (an unresolved timing race)
      instead of silently masking it the same as -1, and
      `_opponentActionLabel()`'s catch now logs unexpected errors before
      returning -1, distinguishing them from the many deliberate -1 returns;
      (4) merged `_policy_value_opp()` into `_policy_value(..., with_opp_logits=
      False)` in `mcts_agent.py`; (5) `evaluate.py`'s five `_run_battles*`
      functions now return a `BattleResult` NamedTuple (wins, total,
      opp_correct=0, opp_total=0) instead of growing positional tuples.

**Search-knob sweep (post-M4 step 1) — ✅ COMPLETE 2026-07-15, decisive.
New operating point: sims=100, c_puct=0.5, det=1.**
- Two-stage sweep (OFAT then combos, 200 battles/cell) + 500-battle
  confirmation on `ppo_step_4750059.pt`. Lower c_puct and fewer
  determinizations each beat their defaults on both opponents, and the
  effects stack: confirmed **81.2% (406/500) vs Random** and **67.2%
  (336/500) vs DamageFirst** — vs 66%/56% at the M4 defaults, at slightly
  *lower* latency (~85ms/move). The narrowly-missed M4 vs-Random criterion
  (+10pp over raw) is now cleared at +23.8pp. Lesson: one deep tree with a
  trusted prior beats a shallow determinization ensemble at fixed budget.
- Defaults updated in `MCTSAgent`/`evaluate.py` (uncommitted, with the
  earlier mcts_agent.py cleanup); full tables in `MILESTONES.md` → M4 →
  "Post-M4 knob sweep"; logs in `models/mcts/results/sweep/`.

**v2-checkpoint A/B (post-M4 step 2, pre-planned in M3.4) — ✅ COMPLETE
2026-07-16: a tie, v2 nominally ahead.**
- Tuned MCTS over `v2/ppo_step_2250032.pt`: **82.6% (413/500) vs Random,
  70.2% (351/500) vs DamageFirst** vs v1's 81.2%/67.2% — within noise
  (~±4pp), though the direction reversed (raw v2 trailed raw v1). Verdict
  in `MILESTONES.md` → M4 → "Post-M4 knob sweep": either checkpoint is a
  valid M5/M6 base; **v1 stays the default**; v2 is a live option for M5.
  Logs: `models/mcts/results/sweep/v2ab_*.log`.

**Post-M4 steps 1–2 (knob sweep, v2 A/B) — both complete above; M5 is now
scoped and active (see top of Current Work).** The open obs-schema question
from the A/B tie is resolved: M5 trains on v2 (decision + rationale in
`MILESTONES.md` → M5). The uncommitted post-M4 working tree (mcts_agent.py
cleanup + tuned defaults in `MCTSAgent`/`evaluate.py`, doc updates,
sweep/A/B logs) is M5 Phase 0 — commit it before any M5 code lands.

**M4 — ✅ complete 2026-07-15, POSITIVE RESULT. Next up: M5 (opponent modeling)
or search tuning.**
- Determinized UCT over the M3.3 best checkpoint (100 sims, 4 determinizations,
  88ms/move): **60.2% (602/1000)** seat-balanced h2h vs the same checkpoint
  without search, **56% (113/200) vs DamageFirst** (raw: 46%) — the first
  intervention since M2 that clearly improves play, and the first to move the
  DamageFirst number. **66% (330/500) vs Random** (+8.6pp) narrowly missed the
  pre-registered +10pp bar; 4 of 5 criteria met. Full tables + reading in
  `MILESTONES.md` → M4; logs in `models/mcts/results/`.
- Follow-ups worth considering before/alongside M5: search-knob sweep
  (`--sims`, `--c-puct`, determinization count — all untuned defaults), the
  pre-planned A/B on the v2 2.25M checkpoint (richer obs may matter more with
  search attached), and M5's opponent-model head plugs directly into the
  search's opponent-action sampler.

### Active Tasks (M4)
- [x] **Forward model** `sim/tools/battle-sim.ts` — clones the live gym battle via
  `State.serializeBattle`/`deserializeBattle` (0.8ms round-trip), snapshots the
  gym's reveal/volatile tracker state alongside (`ObservationTrackers`, extracted
  from `PokemonGymEnv` + new `snapshot()` API), steps both seats directly on the
  clone with gym-identical obs/mask/reward semantics, `fork()` for tree branching.
  **Key engine gotcha found and fixed:** a seat's choice can be auto-completed by
  the engine (`side.isChoiceDone()` — sleep/recharge/multi-turn locks), so
  `needsAction` must consult it; submitting a choice for an auto-done seat lands
  one decision point ahead and desyncs the battle (mask-legal moves get rejected)
- [x] **Determinizer** (Node-side, inside battle-sim — deviation from the original
  `models/mcts/determinizer.py` spec since team generation/legality live in Node):
  replaces the searcher's opponent's *unrevealed* slots with sets sampled from the
  gen1randombattle generator (seeded, species-clause-safe); `perspective: 'p1'|'p2'`
  so a p2-seated searcher determinizes p1. Documented approximation: revealed
  Pokémon keep their true full movesets (only unrevealed *slots* are resampled)
- [x] TS tests — `test/tools/battle-sim.test.js`, 13 tests: obs parity with the
  live gym (v1 + v2 byte-equality), determinization invariants (both perspectives,
  seed-reproducible, revealed mons survive), illegal-free playouts to terminal,
  fork consistency, snapshot-throws-when-done. All pass; gym suite still green
- [x] Bridge/`GymClient` sim protocol: `sim_clone` (determinize/seed/perspective),
  `sim_step`, `sim_fork`, `sim_free`, `sim_free_all`; sims cleared on env reset;
  measured 0.36ms/sim-step over the wire
- [x] `models/mcts/mcts_agent.py` — root-parallel determinized PUCT: policy head
  as prior, opponent actions sampled from the same policy on the opponent's
  reveal-tracked obs, value head + shaped path rewards at leaves, visit counts
  summed across determinizations, argmax-visits (Q tiebreak); `seat='p1'|'p2'`;
  falls back to the raw policy when search can't run (locked-choice roots)
- [x] `evaluate.py --model mcts` (`--sims/--determinizations/--c-puct/`
  `--no-determinize/--mcts-seat`, h2h via `--vs-checkpoint`) + latency: **88ms
  mean / 155ms max per searched decision @ 100 sims** — well under the 500ms
  budget. Early signal: **73% (22/30) vs Random** (raw checkpoint: 57%)
- [x] Code committed: `9da4d273a`
- [x] **Eval battery** (2026-07-15, ~2h): 500 vs Random → **66.0% (330/500)**
  (raw 57.4%, +8.6pp, criterion ❌ by 5 battles); 200 vs DamageFirst →
  **56.5% (113/200)** (raw 45.5%, +11.0pp ✅); seat-balanced h2h vs the raw
  checkpoint → **60.4% as p1 (302/500), 60.0% as p2 (300/500), 60.2%
  combined ✅**; latency 84–88ms mean / ~102ms p95 ✅ (one 909ms outlier
  across ~30k searched decisions). Logs: `models/mcts/results/*.log`
- [x] Results + verdict recorded in `MILESTONES.md` → M4 (4 of 5 criteria met);
  `docs/MODEL-COMPARISON.md` updated — **MCTS-wrapped M3.3 checkpoint is the
  new best agent**; M3.3/M3.4 rows backfilled into the ledger

**Previous: M3.4 — ✅ complete 2026-07-15, NEGATIVE RESULT.**
- Code for both parts (obs schema v2, mixed-opponent training) landed, smoke-verified, and committed (`7533e08d7`). The 5M-step decision run (externally killed at 4.77M steps — no crash; 19 checkpoints cover the trajectory) trained stably in a 42–62% sweep band with no collapse, but full-battle confirmations regress to an M2/M3.3 peer: best **54% (272/500) vs Random**, **46% (92/200) vs DamageFirst**, and **48.0% (480/1000) seat-balanced head-to-head vs the M3.3 best** — all three pre-registered criteria unmet. Full tables + reading in `MILESTONES.md` → M3.4.
- Takeaway: four independent 5M-step-class runs (fixed-opponent, transformer, self-play, v2+mix) all land in the same 51–57%-vs-Random band — the bottleneck is not the observation or the opponent distribution. M4 (MCTS lookahead) attacks the most plausible remaining lever and its criteria were already recalibrated to be relative to the base policy.
- **M4 base checkpoint (recommended):** `models/ppo/checkpoints/selfplay/ppo_step_4750059.pt` (M3.3 best, v1); A/B against `models/ppo/checkpoints/v2/ppo_step_2250032.pt` (v2 peer, richer obs) once the MCTS harness exists.

**Recently closed: M3.1 / M3.2 / M3.3 (2026-07-14)** — parallel training (~5x); transformer retired, MLP-PPO is the architecture; self-play fixed training stability but produced an M2 peer (57% vs Random / 46% vs DamageFirst / 52.4% h2h). Full trails in `MILESTONES.md`.

### Active Tasks (M3.4)
- [x] `sim/tools/feature-extractor.ts` — `TOKEN_DIM_V2=77`, `ActiveVolatiles`/`BOOST_ORDER`, optional `volatiles` param on `extractFeaturesStructured()` (token dim parameterized; v2 dims filled on tokens 0 and 6 only)
- [x] `sim/tools/pokemon-gym.ts` — per-side volatile tracker (`_processVolatileLine`: boosts incl. Haze `-clearallboost`, screens/Sub/Leech Seed `-start`/`-end`, toxic counter via `[from] psn` ticks; reset on switch/drag/faint); obsMode `'structured-v2'`
- [x] `models/gym_bridge.js` — `--obs-v2`; per-reset `{"opponent": ...}` override with per-env single/dual protocol switching
- [x] `models/gym_client.py` — `obs_v2=`, `set_opponent()`, `_reset_cmd()`, `slice_structured_obs()` (v2→v1 per-token view for cross-schema play)
- [x] `models/vec_gym_client.py` — `obs_v2=` passthrough, `set_opponent()`, `reset_all(opponent=)`, mode-aware auto-reset
- [x] `models/ppo/train.py` — `--obs-v2` (checkpoints → `checkpoints/v2/`), `--opponent-mix` per-rollout family sampling with `_switch_family()` env resets, cross-schema pool opponents via `_opp_view()`
- [x] `models/evaluate.py` — `--obs-v2`; `_run_battles_h2h` slices per-seat obs to each agent's `obs_size` (v2-vs-v1 head-to-head)
- [x] `test/tools/gym.test.js` — v2 shape (924), v1-prefix byte-equality, volatile-dim placement, full-battle v2 range/stability; 23/23 pass
- [x] Smokes: v2 bridge single+dual seat, live opponent switching, 4k-step mixed-opponent training run with v1 seed opponent, all 4 eval paths, v1 selfplay trainer regression
- [x] **5M-step decision run** — killed externally at 4.77M steps (no crash; last checkpoint 4.75M, 19 checkpoints total; `models/ppo/checkpoints/v2/train.log`)
- [x] Eval battery: 19-checkpoint sweep (42–62% band, no collapse; `sweep_results.txt`), confirmations on top 4 (`confirm_results.txt`), seat-balanced h2h vs M3.3 best (`h2h_results.txt`)
- [x] Results recorded vs pre-registered criteria in `MILESTONES.md` → M3.4 — **all three unmet** (54% vs Random / 46% vs DamageFirst / 48.0% h2h); winner unchanged, so `docs/MODEL-COMPARISON.md` needs no update

### Active Tasks (M3.1) — all resolved
- [x] `models/vec_gym_client.py` — `VecGymClient(n_envs, structured)`: N parallel `gym_bridge.js` subprocesses, pipelined write-all/read-all stepping, auto-reset on done, per-env errors reset in place and surface as `infos[i]["error"]`
- [x] `models/gym_client.py` — `_send()` split into `_write()`/`_read()` for pipelining
- [x] `models/{ppo,transformer}` agents — `act_batch()` batched inference; `update()` accepts merged tensor dict; `device=` override (excluded from checkpoint hparams so checkpoints stay Mac↔CUDA portable); `load(path, device=...)`
- [x] `models/ppo/trajectory_buffer.py` — `compute_advantages(normalize=...)` + `merge_buffers()` (per-env GAE, global advantage normalization)
- [x] Both trainers — `--num-envs` (default 8) + `--device`; vectorized rollout collection; checkpointing/`--resume`/LR-annealing unchanged (keyed off `total_steps`)
- [x] `models/evaluate.py` — `--num-envs` + `--device`; per-env battle quotas keep counts exact; q/dqn fall back to per-env `act()`
- [x] Verified: vec smoke (4 envs, 800 steps, auto-reset), 600-step train smokes (transformer + PPO both obs modes), checkpoint load round-trips, `--resume` continues at step 600 with LR intact, **M2 51% checkpoint evaluates 49% (49/100) through the parallel path in 4.8s**
- [x] Benchmarks: transformer eval 100 battles 20.2s → 5.6s at 8 envs (3.6x); training throughput table in `docs/ML-TRAINING.md`

### Active Tasks (M3, concluded) — all resolved
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

## Test Results — M5 Phase 3 Smoke Battery (2026-07-16)

Run by the tester agent (Job 3.1) after all four Phase 1–2 builder jobs
(A/B/C/D) completed. Nothing modified; all commands below run as-is against
the working tree. **Overall verdict: PASS.** No routing back to any builder
required.

**1. `./build`** — TypeScript compile.
Command: `./build`
Exit code: `0`. No output (clean compile).

**2. Gym test suite.**
Note: `.mocharc.json`'s `spec` glob is array-typed, so yargs merges it with
any CLI `--spec`/positional args instead of overriding — plain
`npx mocha test/tools/gym.test.js` silently ran the full 2354-test suite.
Used `--no-config` to scope correctly.
Command: `npx mocha --no-config --reporter dot test/tools/gym.test.js`
Exit code: `0`. Output: `29 passing (315ms)`. Spec listing confirms all 6
new M5 tests present and green under `PokemonGymEnv opponent action labels
(M5)`: move-slot mapping, deterministic oppAction sequence, locked-move
(recharge) labeling, dual-seat symmetry, switch bench-slot mapping (4-8),
and force-switch masking (-1).
Command: `npx mocha --no-config --reporter dot test/tools/battle-sim.test.js`
Exit code: `0`. Output: `13 passing (380ms)`.
Pre-existing unrelated failure (not counted, not in scope): full-suite run
hits `better-sqlite3` `NODE_MODULE_VERSION 137` vs required `141` in "SQLite
worker wrapper" — a native module ABI mismatch unrelated to M5.

**3. Bridge smokes (`--obs-v2`), single-seat and dual-seat.**
No `--obs-v2`/dual-seat coverage existed in the `__main__` smokes in
`gym_client.py`/`vec_gym_client.py`, so a throwaway script was written to
the scratchpad (not added to the repo) exercising `GymClient(obs_v2=True,
opponent="damagefirst")` (100 random-valid steps) and
`GymClient(obs_v2=True, selfplay=True)` via `reset_dual()`/`step_dual()`
(100 steps, "needs"-gated actions per seat).
Exit code: `0`.
Output:
```
=== single-seat obs_v2 ===
reset obs shape: (12, 77)
single-seat: 77 labeled, 23 masked, distinct values seen=[-1, 0, 1, 2, 3, 8]
single-seat close: OK
=== dual-seat (selfplay) obs_v2 ===
reset_dual obs shapes: (12, 77), (12, 77)
dual-seat: 91 labeled, 9 masked, distinct values seen=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
dual-seat close: OK
ALL SMOKE CHECKS PASSED
```
`opp_action` confirmed present in `info` and in `[-1, 8]` for both paths.

**4. Training smoke.**
Command: `python models/ppo/train.py --obs-v2 --opponent-mix
"selfplay=0.5,damagefirst=0.3,random=0.2" --opp-coef 0.1 --num-envs 2
--steps 4000 --rollout-steps 512 --checkpoint-every 2000`
Exit code: `0`. Loss finite every rollout (0.163–0.226); label coverage
logged every step, 0.78–0.96 (sane — the mixed-opponent smoke touches
random/damagefirst/selfplay families in one short run, and coverage
tracks family composition). Checkpoints saved to
`models/ppo/checkpoints/opp/` as specified (`ppo_step_2001.pt`,
`ppo_step_4001.pt`, `ppo_step_4001_final.pt`) — confirms default routing.
Note: `train.py` logs only the combined loss plus label coverage, not a
separately-broken-out CE term (`ppo_agent.update()` doesn't expose CE
apart from the combined loss — see the code comment at `train.py:514`).
This matches the task instructions given to this tester run ("assert
finite loss and sane label coverage logging") even though the broader
Phase-3 job description in `AGENT_JOBS.md` mentions "CE loss ... logged
and decreasing" — flagging as an observation, not a failure, since finite
loss + coverage was the explicit bar for this run.
Command (control): `python models/ppo/train.py --obs-v2 --opponent-mix
"selfplay=0.5,damagefirst=0.3,random=0.2" --opp-coef 0 --num-envs 2
--steps 500 --rollout-steps 256 --checkpoint-every 500`
Exit code: `0`. Confirmed: no "Opp label coverage" line in any log line
(old log format reproduced exactly). Bonus: this run's self-play opponent
sampling pulled a pre-M5 checkpoint from `checkpoints/v2/` and exercised
checkpoint two-way compatibility live —
`UserWarning: checkpoint has no 'opp_head'; using a freshly initialized
opponent-prediction head (expected for pre-M5 checkpoints)` and a
matching optimizer-reset warning, then continued without crashing.
All smoke checkpoints from both runs cleaned up afterward (see note at
end of this section).

**5. Label/head sanity check (DamageFirst-only).**
Command: `python models/ppo/train.py --opponent damagefirst --obs-v2
--opp-coef 0.1 --num-envs 2 --steps 6000 --rollout-steps 512
--checkpoint-every 6000 --checkpoint-dir <scratchpad dir>`
Exit code: `0`. Loss decreased monotonically-ish 0.259 → 0.125 across 12
rollouts; label coverage 0.78–0.85 throughout.
Command: `python models/evaluate.py --model ppo --obs-v2 --checkpoint
<smoke checkpoint>/ppo_step_6000_final.pt --battles 20 --opponent
damagefirst`
Exit code: `0`. Output: `Opp-prediction top-1 accuracy: 0.324 (266/822
unmasked steps)` — well above the ~11–25% chance band for a briefly
trained head, as expected since DamageFirst's argmax-damage choice is
independently predictable. This is also step **6a** (raw PPO eval on a
headed checkpoint) — the accuracy line printed correctly.

**6. Eval/MCTS load paths (previously UNVERIFIED — infra outage blocked
Job D from running these).**
- **6a** (raw PPO eval on headed checkpoint): satisfied by the command in
  step 5 above — accuracy line printed, exit 0.
- **6b** (`--model mcts --opp-sampler head` on the **headless** v2
  control checkpoint): `python models/evaluate.py --model mcts --obs-v2
  --checkpoint models/ppo/checkpoints/v2/ppo_step_2250032.pt --opp-sampler
  head --battles 2 --sims 20`. Exit code `0`.
  `UserWarning: --opp-sampler head requested but the checkpoint has no
  trained opponent head; falling back to the policy sampler` — correct
  WARN + fallback, no crash. Summary line: `opp_sampler=policy` (the
  effective, fallen-back value).
- **6c** (`--model mcts --opp-sampler head` on the **headed** smoke
  checkpoint from step 4, `checkpoints/opp/ppo_step_4001_final.pt`):
  same command shape. Exit code `0`. No warning, ran clean. Summary line:
  `opp_sampler=head`.
- **6d** (regression, `--opp-sampler policy` on the same headed
  checkpoint): exit code `0`, ran clean. Summary line:
  `opp_sampler=policy`.
All four eval/MCTS paths pass — this closes out the item Job D flagged as
blocked/unverified.

**Cleanup.** Smoke checkpoints created by this run were deleted afterward:
`models/ppo/checkpoints/opp/` (entire directory — did not exist before
this run) and `models/ppo/checkpoints/v2/ppo_step_501{,_final}.pt` (the
two files this run added to that pre-existing directory; all other files
in `checkpoints/v2/` are untouched). The step-5 label-sanity checkpoint
was trained straight into the scratchpad (`--checkpoint-dir`) to begin
with and never touched the repo. `git status` after cleanup shows no
untracked/modified files under `models/ppo/checkpoints/` — checkpoints
are gitignored, so this was filesystem hygiene only, not a git concern.

---

## Active Plan

**M5 Execution Plan (2026-07-16) — ACTIVE:**

Phase 0 (commit the post-M4 tuning tree) →
Phase 1 [**A:** gym opponent-frame labels + TS tests ∥ **B:** PPO opp head +
buffer] →
Phase 2 [**C:** bridge/clients/trainer plumbing ∥ **D:** evaluate.py
opp-accuracy + MCTS head sampler] →
Phase 3 (smoke battery + code commit) →
Phase 4 (5M-step v2 decision run → sweep → raw confirmations → tuned-MCTS
head-vs-policy sampler A/B → verdict vs pre-registered C1–C5).

Full spec — design decisions (v2 base, fresh run, label grounding in the
opponent's action frame, CE masking rules, sampler integration), file list,
protocol, and success criteria — in `MILESTONES.md` → M5. Task-level
checklist in Active Tasks (M5) above. Pre-registered contingencies: sweep
band < 45% vs Random → one rerun at λ=0.05; opp accuracy vs DamageFirst
< 25% → label bug, fix and rerun; nothing else.

---

**Post-M3 roadmap (decided 2026-07-14): M3.1 → M3.2 → M3.3, then resume M4+**

1. ~~**M3.1 Parallel training infrastructure**~~ ✅ Done (see Active Tasks above)
2. **M3.2 BC→PPO degradation fix** — next up. Store real action masks in the buffer and use them in `update()`; `--value-warmup-steps` (freeze embed/encoder/policy_head, train value head only); `--bc-anchor`/`--bc-anchor-coef` (KL to frozen BC policy); then a full warm-started decision run vs the 51% MLP baseline. Full spec in `MILESTONES.md` → M3.2.
3. **M3.3 Self-play + opponent pool** — heuristic DamageFirst-style opponent wired through gym/bridge/trainers/eval first, then dual-seat self-play against a frozen checkpoint pool. Full spec in `MILESTONES.md` → M3.3.

**M3 Execution Plan — concluded (negative result):**

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

✅ **M4: MCTS Integration — positive result, search clearly improves play** (2026-07-15)
- **Forward model** (`sim/tools/battle-sim.ts`): clones the live gym battle
  (engine state + tracker state) in 0.8ms, steps both seats with gym-identical
  obs/mask/reward semantics, forks for tree branching; determinizer resamples
  the searcher's opponent's unrevealed slots from the format generator (either
  perspective). Engine gotcha fixed: locked states auto-complete a seat's
  choice (`side.isChoiceDone()`) — submitting anyway desyncs the battle
- **Search** (`models/mcts/mcts_agent.py` + bridge `sim_*` protocol +
  `evaluate.py --model mcts`): root-parallel determinized PUCT, policy prior +
  value leaves, opponent modeled by the same policy on its reveal-tracked obs
- **Results** (100 sims / 4 det, base = M3.3 best): 60.2% (602/1000)
  seat-balanced h2h vs the raw checkpoint; 56% vs DamageFirst (raw 46%); 66%
  vs Random (raw 57%, +8.6pp — the one criterion narrowly missed, bar was
  +10pp); 84–88ms/move. 4 of 5 criteria ✅. Committed `9da4d273a` (code);
  13 new TS tests. Full trail in `MILESTONES.md` → M4
- **Reading:** confirms M3.4's hypothesis — lookahead was the binding
  constraint, not observations or opponent distribution. Gains grow with
  opponent strength (+8.6pp vs Random, +11pp vs DamageFirst, +10.2pp h2h)

✅ **M3.4: obs schema v2 + mixed-opponent training — negative result, all criteria unmet** (2026-07-15)
- **Part A:** `TOKEN_DIM_V2=77` — 7 boost stages + Reflect/Light Screen/Substitute/Leech Seed flags + toxic counter appended per token (v1 prefix byte-identical), tracked in `pokemon-gym.ts` from public log lines only, active tokens only; obsMode `'structured-v2'` wired through bridge (`--obs-v2`), clients (`obs_v2=`), trainer, and `evaluate.py`
- **Part B:** `--opponent-mix` per-rollout family sampling (selfplay/damagefirst/random) with per-reset opponent switching in the bridge (no respawn); pool seeded with M2 + M3.3-best; cross-schema play via `slice_structured_obs()` (v1 checkpoints act inside v2 envs, and `--vs-checkpoint` pairs v2 vs v1 directly)
- Verified: 23/23 gym tests, full smoke battery, v1 regressions; committed `7533e08d7`
- **Decision run:** 4.77M/5M steps (externally killed, no crash), stable 42–62% sweep band, **no collapse**. Confirmations: best **54% (272/500) vs Random**, **46% (92/200) vs DamageFirst**, **48.0% (480/1000)** h2h vs the M3.3 best — an M2/M3.3 peer. All three pre-registered criteria ❌. Artifacts: `models/ppo/checkpoints/v2/{train.log,sweep_results.txt,confirm_results.txt,h2h_results.txt}`
- **Reading:** 4 independent 5M-step-class runs now land in the same 51–57% band — observation richness and opponent distribution are ruled out as the bottleneck. M4 (MCTS) proceeds on the M3.3 best checkpoint with relative criteria

✅ **M3.3 closed (head-to-head eval) + M3.4 milestone defined + M4 criteria recalibrated** (2026-07-14)
- Added `--vs-checkpoint` to `models/evaluate.py` — checkpoint-vs-checkpoint head-to-head through the dual-seat self-play bridge path (`_run_battles_h2h`; PPO-only, both checkpoints share the obs mode; wins counted from seat p1)
- Ran the M3.3 head-to-head, seat-balanced: self-play 4.75M vs M2 checkpoint = 51% as p1 (254/500), 54% as p2 (270/500) → **52.4% combined (524/1000)** — inside the ±3.1pp CI, statistical parity. M3.3 marked ✅ complete with a mixed verdict: self-play fixed training stability but produced a peer of the M2 agent, not a stronger one
- Defined **M3.4 (raise the policy ceiling)** in `MILESTONES.md`: Part A obs schema v2 — boost stages + screens/Substitute/Leech Seed/toxic counter, a deliberate TOKEN_DIM bump now that the transformer retirement (M3.2) ended the BC-checkpoint schema freeze; Part B mixed-opponent training — seed the self-play pool with the M2 checkpoint (zero code) + `--opponent-mix` per-rollout sampling across selfplay/damagefirst/random. Rationale: best policy is 57% vs Random while M4's criteria assumed ≥90%; MCTS amplifies whatever policy quality it's given
- Recalibrated M4's success criteria (were transformer-based and assumed the ≥90% base): now relative to the base policy — MCTS must add ≥5pp vs DamageFirst and ≥10pp vs Random over the same checkpoint without search

✅ **M3.3 comparison run: 5M-step MLP-PPO self-play + evaluation sweep** (2026-07-14)
- Ran the full self-play training run (`--opponent selfplay`, own-checkpoint pool, 8 envs, 5M steps, ~1.9h). Rollout win rates oscillated 0.2–0.8 as expected (each rollout samples a different frozen pool opponent); loss stayed 0.03–0.09 throughout
- 20-checkpoint sweep (150 battles each vs Random): rises from 42% @ 250k into a stable 47–61% band — **the first run in this project whose eval strength improves over training instead of peaking early and decaying**. No collapse anywhere
- Confirmed top candidates at full battle counts (500 vs Random / 200 vs DamageFirst): best is `ppo_step_4750059.pt` at **57% (287/500) vs Random** — ~2σ above the M2 baseline's 51% — but **46% (91/200) vs DamageFirstAI** vs the M2 agent's 51%: self-play did *not* transfer better to the held-out heuristic (all top checkpoints landed 43–46%)
- Verdict per M3.3 criteria: sanity floor ✅ (no regression vs Random); the DamageFirst half of the main criterion ✗; head-to-head half still unrun (needs a checkpoint-vs-checkpoint mode in `evaluate.py`). Results + reading recorded in `MILESTONES.md` → M3.3

✅ **M3.2: BC→PPO degradation fix — transformer retired, MLP-PPO is the architecture** (2026-07-14)
- Implemented all three fixes: per-step `valid_mask` stored in `TrajectoryBuffer` and used in both agents' `update()` (updates previously pretended all 9 actions were legal); `--value-warmup-steps` (freeze embed/encoder/policy_head so the random value head fits the BC policy before full PPO gradients hit the shared encoder); `--bc-anchor`/`--bc-anchor-coef` (frozen BC policy as an annealed KL anchor). Plus `--checkpoint-dir` on both trainers so new runs can't overwrite old checkpoints. Verified freeze semantics (policy tensors bit-identical through a frozen update), anchor annealing, and a live warmup→unfreeze transition
- Fixed an MPS blocker found en route: `TransformerPolicy`'s encoder now uses `enable_nested_tensor=False` (the eval-mode fast path used aten ops unimplemented on MPS; numerically identical, checkpoint-compatible)
- **Decision run:** 5M steps warm-started with all fixes (`models/transformer/checkpoints/m32/`), 20-checkpoint sweep + 500-battle confirmations. The diagnosis was right — no collapse, the policy holds 44–55% for 3.5M steps, and decay resumes exactly as the anchor anneals to zero (35%→25% over the last 1.25M steps): unconstrained PPO erodes the BC policy at any LR. But the ceiling is the BC policy itself: best confirmed **53% (263/500) vs Random** (parity with MLP's 51%) and **39% (77/200) vs DamageFirstAI** (MLP: 51%)
- **Per the milestone's pre-registered criterion: transformer retired; M4 (MCTS) and M5 (opponent modeling) proceed on the M2 MLP-PPO baseline**

✅ **M3.3 code: DamageFirst opponent + dual-seat self-play** (2026-07-14)
- `sim/tools/damage-first-ai.ts` (`DamageFirstAI`, highest-base-power move) wired as `--opponent damagefirst` through gym → bridge → `GymClient`/`VecGymClient` → both trainers → `evaluate.py`. New baseline: M2 MLP scores **51% (101/200) vs DamageFirstAI**
- Dual-seat self-play: `PokemonGymEnv` `opponent: 'self'` + `resetDual()`/`stepDual()` (both seats' obs/mask/needsAction per decision point; freshness-counted requests; force-switches are single-seat decision points; reveal tracker runs for both sides so each seat sees only revealed info; reward stays p1-perspective). Bridge `--selfplay` dual protocol; `VecGymClient.reset_all_dual()/step_dual()` pipelined with auto-reset
- Both trainers: `--opponent selfplay` samples one frozen opponent per rollout from `--selfplay-pool` (default: own checkpoint dir; 50% newest / 50% uniform; frozen copy of the current policy until a first checkpoint exists). Pending-transition collection credits rewards from opponent-only steps to p1's open transition
- Smoke-verified at every layer: TS (episodes complete, force-switch points, no illegal moves), Python bridge, and short self-play training runs on both trainers. Comparison training runs remain (see Next Steps)

✅ **M3.1: Parallel training infrastructure** (2026-07-14)
- Root-caused training slowness: fully serial pipeline (one bridge subprocess, one battle at a time, batch-1 inference every step — GPU mostly idle). Built `models/vec_gym_client.py` (`VecGymClient`): N parallel `gym_bridge.js` subprocesses with pipelined stepping (write all N commands, then read all N responses — Python waits only for the slowest env) and auto-reset on episode end
- Batched inference (`act_batch()`) on both `PPOAgent` and `TransformerAgent`; per-env `TrajectoryBuffer`s with `merge_buffers()` doing global advantage normalization (GAE never crosses env streams); `--num-envs` (default 8) + `--device` on both trainers and `evaluate.py`
- Device facts recorded in `docs/ML-TRAINING.md`: CUDA auto-detected already (RTX 3080 box = clone + build + CUDA torch, zero code changes; checkpoints portable both directions); Apple Neural Engine is unreachable from PyTorch (CoreML-only) — MPS is the Mac backend and was already in use
- Verified end-to-end: vec smoke, train smokes (both trainers, both PPO obs modes), checkpoint round-trips, `--resume`, and the M2 51% MLP checkpoint reads 49% (49/100) through the parallel path — old checkpoints fully compatible
- Measured: transformer eval 100 battles 20.2s → 5.6s (8 envs); training throughput table in `docs/ML-TRAINING.md`
- Roadmap updated: new M3.1/M3.2/M3.3 milestones in `MILESTONES.md`; M4/M5 resume after M3.2's architecture decision

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

### M8: Value-Head Targeting + Ladder Infrastructure (scoped 2026-07-22)

**Immediate first action (after user approval to build):** parallelize Phase 0 + Phase 1A.
- **Phase 0 (30 min, can start immediately):** Websocket reconnect logic for ladder-bot (`tools/ladder-bot/ladder-bot.js`). De-risks all future ladder runs.
- **Phase 1A (4–6 hours, parallel machine):** Speed-ratio A/B run — quick 1–2M PPO steps to test whether base-stats/speed dims help. Decision gate at 2M: if >+2pp signal on both Random + DamageFirst, escalate to Phase 1B full run; else escalate to Phase 2 (value targeting).

**Full M8 contingency structure in MILESTONES.md → M8:** 6 phases with gates between them. **Most likely path:** Phase 1A → 1B or 2 → 4 = ~12–16 hours total. Worst case (full AlphaZero): ~48 hours over 3–4 days.

**Pre-registered criteria:** Bot evals repeat M7 baseline (93%/84%), ladder GXE ≥35% for win or <25% for clear regression (same ±35 noise band as M7).

**M7 follow-up decision (orthogonal to M8):** Criterion C landed inconclusive (GXE 28.2%, 25–34% noise band). An optional 50-game follow-up is pre-authorized to narrow the band, but does not block M8 scoping or build. User runs it independently if desired.

**Known gap carried forward:** M8 Phase 0 fixes the websocket reconnect issue, de-risking Phase 4 (ladder validation).

### M7 — Observation Schema v3
✅ **Complete 2026-07-20.** Criterion A pass, Criterion B pass (93.0% R /
84.2% DF, new best agent), Criterion C inconclusive (Elo 1034.6 / GXE 28.2%
vs M6's 1017/23.9%, landed in the pre-registered 25–34% noise band). Full
results in `MILESTONES.md` → M7.

---

### Previously Completed Milestones

#### M6 — Server Integration & Ladder
✅ **Complete 2026-07-17.** All three criteria met: 100/100 clean rated battles on the official ladder (MCTS config, ≤579ms/move), **Elo 1017 / GXE 23.9%** — the agent sits at the bottom of the human ladder despite dominating every project bot (external measurement delivered; the bot-relative ledger overstates absolute strength). Full results + qualitative failure analysis in `MILESTONES.md` → M6.

#### M5.5 — Human Replay Data + BC for the MLP
✅ **Complete 2026-07-16 — POSITIVE RESULT, NEW BEST AGENT.** BC on human replays → anchored PPO fine-tune → tuned MCTS = **90.6% (453/500) vs Random / 79.2% (396/500) vs DamageFirst** (prior best 86.0%/72.6%). Checkpoint: `models/ppo/checkpoints/bcft/ppo_step_5000000_final.pt`. See `MILESTONES.md` → M5.5.

#### M5 — Opponent Modeling Head
✅ **Complete 2026-07-16 — THESIS NEGATIVE, NEW-BEST SIDE FINDING.** Sampling opponent's action from the trained prediction head adds nothing over the policy sampler under tuned MCTS (−2.2pp vs DamageFirst, parity). But the run produced the project's best agent: M5 checkpoint under policy-sampler MCTS at 72.6%/86.0%. See `MILESTONES.md` → M5.

#### M4 — MCTS Integration
✅ **Complete 2026-07-15 — POSITIVE RESULT.** Determinized UCT beats the raw policy 60.2% h2h (seat-balanced), +11pp vs DamageFirst, 88ms/move (4 of 5 criteria ✅). Knob sweep post-M4 raised tuned MCTS to **81.2% Random / 67.2% DamageFirst** (vs 100/sims default 66%/56.5%). See `MILESTONES.md` → M4.

#### M3.1 / M3.2 / M3.3 / M3.4
✅ **All complete 2026-07-15.** M3 (transformer) was negative; M3.1 parallelized training; M3.2 retired the transformer (MLP-PPO is the architecture); M3.3 self-play fixed training stability; M3.4 obs schema v2 + opponent mix were both washes. See `MILESTONES.md` → M3–M3.4 and "Recently Completed" above for full detail.

#### M2.5 — Behavior Cloning Pretraining
✅ **Complete 2026-07-02.** Scraped and adapted 119k gen1ou + 20k+ high-Elo gen1randombattle human games. BC achieved 50.5% top-1 accuracy vs ~11% chance. See `MILESTONES.md` → M2.5.

#### M2 — Structured State Representation
✅ **Complete and verified (51% win rate vs RandomPlayerAI, 500 battles). Nothing left here.**

#### M1 & M0
✅ Complete. Foundation and environment + baselines working.

### Stretch (deprioritized)
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

## Test Results — M7 Job 3.2 / Criterion A

**Date:** 2026-07-18. **Scope:** full test/build suite over Phases 0–3.1 output, plus explicit Criterion A verification before Phase 4 (BC pretrain) training spend.

**Commands run (verbatim) and exit codes:**
1. `./build` → exit 0
2. `npx mocha --no-config --reporter dot test/tools/gym.test.js` → `45 passing (208ms)`, exit 0
3. `npx mocha --no-config --reporter dot test/tools/feature-extractor.test.js` → `22 passing (21ms)`, exit 0
4. `npx mocha --no-config --reporter dot test/tools/battle-sim.test.js` → `13 passing (287ms)`, exit 0
5. `npx mocha --no-config --reporter dot test/tools/replay-adapter.test.js` → `10 passing`, exit 0

No failures in any of the above. (Note: full-suite/`npm test` runs are known to separately hit a pre-existing, unrelated better-sqlite3 `NODE_MODULE_VERSION` mismatch in "SQLite worker wrapper" — out of scope per Job 3.2 instructions, not run here since targeted suites above cover everything Job 3.1 touched.)

**Criterion A coverage check** (MILESTONES.md: valid obs shape at every decision point, no NaN/inf in type-eff/new dims, Sleep Clause flag toggles correctly) — confirmed present in existing tests, no smoke script needed:
- v3 obs shape `(12, 86)` / flat `1032` at decision points: `test/tools/gym.test.js` lines 219–224 (`extractFeaturesStructured` direct call), 348–354 (`env.reset()` in `structured-v3` obsMode), 479–496 (shape re-checked at every step across a full battle, `TOKEN_DIM_V3`/`N_TOKENS` asserted each iteration).
- No NaN/inf in new dims: same tests assert `!isNaN(v) && isFinite(v)` over the full 86-dim vector, including the full-battle step loop (line 492) and a targeted loop over dims `TOKEN_DIM_V2..TOKEN_DIM_V3` (lines 495–496).
- v2-prefix byte-equality (dims 0–76 identical to v2): `test/tools/gym.test.js` line 227 ("should keep the first 77 dims of every token byte-identical to v2"); mirrored in `test/tools/feature-extractor.test.js` (22 tests, incl. v1/v3 and v2/v3 byte-equality per Job 1.1's summary).
- Sleep Clause flag toggling: `test/tools/gym.test.js` "Sleep Clause tracker" describe block (lines 810–902) — flag starts cleared, sets when opponent is put to sleep, Rest-sourced self-sleep excluded (line 828), bench-persistent across switch (not reset on switch/drag), clears on cure and on faint (line 858), multi-sleeper case, snapshot/restore preserves flag. Plus obs-level placement test (line 269) confirming the flag lands identically on all 12 tokens (dim 85) and a tracker→obs integration test (line 278).

All Criterion A sub-requirements have direct, passing test coverage — no gap requiring an ad hoc smoke script.

**Verdict: PASS.** Criterion A is met. Phase 4 (Job 4.1, BC pretrain on v3 data) is cleared to begin.

---

## Job 4.1 — BC Pretrain Run on v3 Data (2026-07-18)

**Command (verbatim):**
```
python3 models/bc_pretrain_mlp.py --epochs 5 --obs-v3 --out bc_mlp_gen1_v3.pt
```
Matches the M5.5 Run 2 recipe exactly (defaults: `--min-rating 1300`, equal
per-format sampling weights, `--opp-bc-coef 0.1`, `--value-bc-coef 0.5` —
the outcome-trained value head Run 2 added for MCTS), changing only
`--obs-v3` (obs_size 1032, `data/replay_trajs/v3/`: 99 gen1ou + 21
gen1randombattle shards) and `--out`. Ran via `nohup` to
`logs/bc_pretrain_v3_full.log`, monitored to completion (no polling).

**Result: completed cleanly, 5/5 epochs, 25,109,340 samples, wall time
1054s (~17.6 min — far under the ~2h estimate; v3's extra dims didn't
change dataset size or add meaningful per-step cost).**

Final held-out validation:
| format | policy acc | opp-head acc | value-sign acc | val samples |
|---|---|---|---|---|
| gen1randombattle | **52.7%** | 34.6% | 63.1% | 60,766 |
| gen1ou | **54.0%** | 34.4% | 68.6% | 28,852 |

vs. M5.5 v2 baseline (Run 2): 49.7% randbats / 53.1% gen1ou (policy),
62%/67% value-sign. v3 is **flat-to-slightly-up** on both formats and both
heads (+3.0pp randbats / +0.9pp gen1ou policy; +1.1pp / +1.6pp value-sign) —
not degenerate (chance ≈11%), no crash, no NaN.

**Checkpoint:** `models/checkpoints/bc_mlp_gen1_v3.pt` (kept distinct from
the v2 checkpoint `models/checkpoints/bc_mlp_gen1.pt`, which is untouched).

**Verdict: Phase 5 (PPO fine-tune) is cleared to begin** — gate was
"not degenerate/crashed," which is met with margin; v3's extra type-eff/
move-flag/Sleep-Clause signal is at minimum not hurting human-imitation
accuracy at the BC stage.

---

## Job 5.1 — PPO Fine-Tune Run on v3 Obs (started 2026-07-18)

**Command (verbatim):**
```
python3 models/ppo/train.py --obs-v3 --steps 5000000 --rollout-steps 512 \
  --num-envs 8 --checkpoint-every 250000 \
  --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2" \
  --checkpoint-dir models/ppo/checkpoints/v3 \
  --pretrain-checkpoint models/checkpoints/bc_mlp_gen1_v3.pt \
  --bc-anchor models/checkpoints/bc_mlp_gen1_v3.pt --bc-anchor-coef 0.05 \
  --value-warmup-steps 200000 --opp-coef 0.1
```
Matches the M5.5 `bcft` recipe exactly (verified against
`models/ppo/checkpoints/bcft/train.log`'s header line and MILESTONES.md's
M5.5 write-up), swapping only obs version and checkpoint paths: `--obs-v3`
(obs_size auto-inferred 1032) in place of `--obs-v2`, warm-start/anchor from
`bc_mlp_gen1_v3.pt` in place of `bc_mlp_gen1.pt`, checkpoints to
`models/ppo/checkpoints/v3/` (new dir, did not exist — no overwrite risk).
Pool seeded identically to bcft: copied `ppo_step_0_seed_m2.pt` and
`ppo_step_0_seed_m33best.pt` (M2 + M3.3-best, both v1/780-dim structured
checkpoints) from `models/ppo/checkpoints/bcft/` into the new v3 checkpoint
dir before launch, so the default self-play pool (= checkpoint dir) picks
them up. Cross-schema compatibility confirmed in `models/ppo/train.py`
before running: `_opp_view()` (line ~454) calls `slice_structured_obs()`
generically whenever `args.structured` is set — not v2-specific — and the
`--obs-v3` help text explicitly documents "v1/v2 checkpoints still play as
pool/h2h opponents (sliced per token)," so the M5.5 pool seeding works
unmodified under v3.

Launched via `nohup` (PID 52721) to `models/ppo/checkpoints/v3/train.log`,
monitored via a background watch (no busy-polling). Startup log line
confirms exact match to the bcft recipe modulo obs version:
`obs_mode=v3 (obs_size=1032) | rollout=512 steps | num_envs=8 | opponent=
selfplay=0.5,damagefirst=0.3,random=0.2 (pool: models/ppo/checkpoints/v3) |
checkpoint every 250000 steps | opp_coef=0.1`, `Value warmup: policy frozen
until step 200000`. Same two expected pre-M5-checkpoint warnings as bcft
(fresh opp_head / fresh optimizer state for the v1 seed checkpoints) — not
errors, identical to the bcft run's own log.

**Status: RUNNING as of this entry — result pending, will be appended when
the monitor reports completion or failure.**

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

