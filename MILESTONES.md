# Pokemon Showdown AI Training — Milestones

Goal: Build a genuinely intelligent Pokemon trainer AI.

**Current architecture** (after nine milestones of revision): human-replay BC →
anchored PPO fine-tune on an MLP policy/value net with structured per-Pokémon
token observations (v3 schema) → determinized MCTS at decision time. The
transformer was tried and retired (M3.2); MCTS is the only intervention with a
clearly positive result (M4, though see the correction below).

**Where the project actually stands:** ~93% vs training bots; on the
gen1randombattle ladder ~27% of games and **~19–23% of the games that get played
out** (see correction 4). Closing that gap is the whole remaining problem. Six
hypotheses have been directly tested and come back null or weak.

| File | Read when |
|---|---|
| `docs/WHERE-WE-ARE.md` | Session start — one screen, plain language |
| **This file** | Planning the next milestone; checking what a past one concluded |
| `docs/MILESTONES-ARCHIVE.md` | You need M0–M9 in full: original plans, build phasing, file manifests, complete results tables |
| `docs/EVALUATION-METHODOLOGY.md` | Before running or gating on **any** evaluation |
| `docs/CODE-REVIEW-FINDINGS.md` | Before trusting any pre-2026-08-01 comparison |

---

## Four corrections that invalidate earlier numbers

Read these before quoting any result from M2–M8, or any ladder win rate. The
first three landed 2026-08-01 from the code review and are recorded inline in
the archive too; the fourth landed 2026-08-03 from the ladder-log analysis.

1. **The MCTS-vs-raw comparisons are confounded** (`CODE-REVIEW-FINDINGS.md
   §3`). MCTS plays argmax over visit counts while every "raw checkpoint" arm
   sampled stochastically. Greedy decoding alone accounts for **+7.8pp vs
   Random and +5.0pp vs DamageFirst** (n=5,000/arm) — roughly the entire
   reported search gain. M4's "+8.6pp", the 60.2% head-to-head, the shipped
   "93.0% with MCTS vs 69.7% raw", and every ladder `--mcts` on/off comparison
   are affected. **Search's true contribution is unknown; the lookahead
   hypothesis is not closed.** Greedy decoding is now the ladder default.
2. **The "~3pp seat bias" was never measured** (`§5d`). It served as an error
   bar for nine milestones. It is a convention, not a measurement — don't quote
   it without that caveat.
3. **M8 Phase 1A's closure is doubly confounded and should be reopened**
   (`§1 & §6`). Its conclusion — "richer observations are not the constraint" —
   rests on two uncontrolled variables: (a) trunk width was held at 128 while
   the schema grew 77 → 87 dims, and (b) `replay-adapter.ts` has no
   v3-extended path, so the speed-ratio arm **could not have been BC
   warm-started** while the v3 control it was compared against was. The
   speed-ratio feature itself is sound and worth retesting against a proper
   warm-started baseline. **This one matters most right now:** M11's whole
   thesis is observation poverty, and M8 Phase 1A is the main evidence against
   it.
4. **Every ladder win rate is inflated ~7pp by opponent concessions** (found
   2026-08-03 analysing the 360-game greedy run). Across all 747 rated ladder
   games we have **never lost one in under 16 decisions**, but 74 wins came that
   fast — forfeits, timeouts and disconnects, ~10% of games and **~35% of every
   win we have recorded.** They are legitimate Elo but they measure the pool's
   quit rate, not our play. Restated: M7-era sampling **30.5% → 22.9%
   contested**; the greedy run **26.9% → 19.3% contested**. The contamination
   rate is the same in both eras, so no *comparison* between arms is
   invalidated — only the absolute levels, including the widely-quoted "~30% on
   ladder". `scripts/ladder_analysis.py` now prints the split on every run and
   takes `--min-decisions 16`.

---

## Results Ledger (M0–M9, all complete)

Condensed. Each entry: what was tested, what the numbers were, and what it
closed. Full plans, build phasing, and complete tables live in
`docs/MILESTONES-ARCHIVE.md`.

### M0: Foundation ✅
Build system (`./build` → `dist/`), docs suite, `RandomPlayerAI` verified as a
base class, `BattleStream` verified for programmatic parallel battles.

### M1: Environment & Baseline Agents ✅
`PokemonGymEnv` (reset/step/validActions), flat 100-dim feature extractor,
parallel evaluator, Python↔Node stdio bridge, and three baseline learners
(tabular Q — archived as a confirmed dead end; DQN; PPO). Action space fixed at
**Discrete(9)**: moves 0–3, switches 4–8, validity-masked every step. Reward:
±1 win/loss, ±0.01 per KO, +0.0001 per status, −0.001×turns, clipped to [−1,1].

### M2: Structured State Representation ✅ — parity
Replaced the flat vector with **12 per-Pokémon tokens × 65 dims**. Unrevealed
opponent slots get `unknown_flag=1, HP=1.0` (never a zero vector — that reads
as fainted). Added the opponent-reveal tracker that reconstructs opponent state
from public log lines only. **51% vs Random (254/500)** — parity with the flat
baseline, which was the bar.

> ⚠️ The architecture comparison behind this is confounded: `evaluate.py` ran
> q_learning/DQN greedily and PPO stochastically (`CODE-REVIEW-FINDINGS.md
> §5e`). The representation itself is sound; the relative-strength claim isn't.

### M2.5: Behavior Cloning Pretraining ✅
BC on 119,536 Metamon gen1ou human trajectories → **50.5% top-1** (vs ~11%
chance). Established the load-bearing action-grounding invariant that still
holds everywhere: **action k = move-slot k of the own active token; action 4+j =
bench token j+1.** Fed the transformer only; the MLP got human data in M5.5.

### M3: Transformer Encoder + PPO ❌ NEGATIVE
2-layer, 4-head, d_model=128 encoder over the tokens. Best-ever across ~40
checkpoints and three runs: **46%**, against the MLP's 51% at equal compute.
The uncontrolled warm-started run collapsed 46% → 0% between 2.5M and 4.6M
steps. Not a training-budget artifact.

### M3.1: Parallel Training Infrastructure ✅
`VecGymClient` (N Node subprocesses, pipelined steps, auto-reset), batched
inference, per-env GAE buffers, `--num-envs`/`--device`. ≥5× throughput at 8
envs. Device note kept so it isn't relitigated: the Apple Neural Engine is not
reachable from PyTorch; MPS is the Mac backend, and CPU can beat it at small
batch sizes.

### M3.2: BC→PPO Degradation Fix ✅ — TRANSFORMER RETIRED
Three fixes, all verified: real action masks in PPO updates, value-head warmup,
KL-anchor to the BC policy. **The diagnosis was right** — the model held at its
BC plateau for 3.5M steps instead of decaying, and decay resumed exactly as the
anchor annealed away. But the ceiling was still the BC policy itself: **53% vs
Random (parity), 39% vs DamageFirst (behind the MLP's 51%)**. Per this
milestone's own pre-registered rule, **the transformer is retired and
everything after builds on MLP-PPO**.

### M3.3: Self-Play + Opponent Pool ✅ — MIXED
Built `DamageFirstAI`, dual-seat self-play (`resetDual`/`stepDual`), and a
league sampler. **First run in the project whose eval strength trends up over
training instead of decaying** — that stability finding is the real deliverable.
Strength-wise a peer: 57% vs Random (best), 46% vs DamageFirst, 52.4% h2h over
1000 seat-balanced battles (inside noise). No transfer to the held-out
heuristic; the pool contained only its own descendants.

### M3.4: Raise the Policy Ceiling ❌ NEGATIVE
Two levers at once: **obs schema v2** (77 dims — 7 boost stages, screens/Sub/
Leech Seed, toxic counter, active tokens only, v1-byte-identical prefix) and
**mixed opponents** (`--opponent-mix selfplay=0.5,damagefirst=0.3,random=0.2`
over a seeded pool). Result: **54% R / 46% DF / 48% h2h** — all three criteria
unmet. Also the moment the 150-battle sweep peaks (62%) were shown to regress
to the mean at n=500, which should have been the methodology warning it wasn't.

**The pattern that emerged here and never broke:** four independent 5M-step
runs — fixed-opponent, transformer, self-play, v2+mix — all land in the same
**51–57% vs Random** band. Later runs made it five, then six.

### M4: MCTS Integration ✅ POSITIVE (but see correction)
Determinized root-parallel PUCT: policy head as prior, value head at leaves,
opponent modeled by sampling the same policy, visit-count argmax. Forward model
`BattleSim` clones the live battle via serialize/deserialize (0.8ms) plus
tracker state. Engine gotcha fixed and still relevant: locked states (sleep,
recharge, multi-turn) auto-complete a seat's choice, and submitting anyway
desyncs the battle — `needsAction` must consult `side.isChoiceDone()`.

Post-M4 sweep found a much better operating point — **sims=100, c_puct=0.5,
det=1** (concentrate the search; det=1 is also the fastest at ~85ms/move) —
taking it to **81.2% R / 67.2% DF**. That's still the shipped config.

> ⚠️ The headline "search beats raw policy" numbers are confounded by greedy
> vs sampling (see correction 1 above).

### M5: Opponent Modeling Head ❌ THESIS NEGATIVE, side finding positive
Aux head predicting the opponent's resolved action (λ=0.1 CE through the shared
trunk), with labels grounded in the *opponent's own* action frame. The head
learned real signal (30–36% top-1 vs DamageFirst, above the ~25% floor) but
**sampling from it inside search did nothing: −2.2pp vs the policy sampler**.
Plausible cause: it's trained on the opponent *mixture*, so it's miscalibrated
against any specific opponent, whereas "the opponent plays like us" is
adversarially robust. Head sampler retired as a default; the aux loss stayed
(it costs nothing and the trunk it shaped posted the best search numbers yet at
the time, 72.6% DF / 86.0% R).

### M5.5: Human Replay Data + BC for the MLP ✅ POSITIVE — biggest single win
Built the replay pipeline: public-API scraper, a 98,349-log gen1ou bulk import,
and `replay-adapter.ts`, which reuses the live gym's own trackers so replay obs
are identical to training obs by construction (verified by a byte-identical
round-trip test). Label coverage 91% randbats / 86% gen1ou; the gap is
`|cant|` turns, which are unknowable by design.

Results: **BC alone was decisively negative** (raw 22% vs Random). The M3.2
fixes — warm-start + value warmup + BC KL-anchor — are what converted the human
prior into **90.6% R / 79.2% DF** under tuned MCTS, and 78.4% h2h vs the prior
best. **Neither ingredient suffices alone.** This remains the single largest
improvement in project history and the only pathway that beat the bot-trained
lineage.

### M6: Server Integration & Ladder ✅ — the reality check
Shipped the live ladder bot (websocket client, `models/infer_server.py`,
`BattleSim.fromTracked` so search works without a local engine battle).
**100/100 clean rated battles, max 579ms/move.** And then the external
measurement the project had never had: **Elo 1017, GXE 23.9%, 23W–96L —
bottom of the human ladder.** Honest decomposition: 13 of 21 wins were ≤9
decisions (forfeits/disconnects); full-length win rate ~9%.

An agent at 90.6% vs its own training bots is a floor-adjacent ladder player.
**The bot-relative ledger drastically overstates absolute strength; the human
ladder is the primary evaluator from here on.**

### M7: Observation Schema v3 🟡 INCONCLUSIVE
Added the rules knowledge M6's live games showed missing: per-move **type
effectiveness** vs the opponent active, move-effect flags (recharge, self-KO,
priority, inflicted status), and a **Sleep Clause** flag — 86 dims/token, v2
prefix preserved. Motivated by observed blunders: Hypnosis into a sleeping
Pokémon, Explosion into a Ghost, Fire Blast into Slowbro.

**Best bot numbers in the project: 93.0% R / 84.2% DF.** Ladder: 30W–70L,
GXE 28.2% — inside the pre-registered 25–34% inconclusive band, so by the
pre-committed rule, not a win. A 50-game follow-up reached 32.9%, still 2.1pp
short. Bug worth remembering: `battle-sim.ts` had no v3 path at all, which
falsified M4/M5's "MCTS is obs-shape-agnostic" claim (only ever checked
Python-side).

### M8: Value-Head Targeting + Ladder Infra ❌ ALL BETS NEGATIVE
- **Phase 1A** (speed-ratio obs dim): failed, Random moved −3pp. **⚠️ This
  closure is doubly confounded and should be reopened** — see correction 3
  above. Do not cite it as evidence that observations aren't the constraint.
- **Phase 2** (AlphaZero-style value targets): failed at 80.0% vs an 82.5%
  base, then **failed again on replication** with targets re-collected against
  DamageFirst — which eliminates distribution mismatch as the explanation.
- **The striking part:** the fine-tune fixed the value head in-distribution
  both times (R² 0.00 → 0.20, then −0.13 → +0.19 — the PPO-trained head scored
  *below a constant predictor*), and none of it transferred to play strength.
- **Phase 4** ladder: 27/100, GXE 32.9%, third consecutive reading in the band.

**The durable finding is methodological.** The same checkpoint swung 42% → 27%
raw between runs, so the "monotonic GXE trend" (23.9 → 28.2 → 32.9) was an
over-read: GXE is account-level and cumulative over 506 games, and was never a
valid per-run gate.

### M9: Evaluation Methodology + Data Distribution ✅
- **Phase 1:** fixed measurement — `docs/EVALUATION-METHODOLOGY.md`,
  `scripts/ladder_analysis.py` (Wilson/Newcombe intervals, session
  segmentation, power tables), plus per-run ladder logging.
- **Phase 2a:** format alignment is real — randbats-only BC beats mixed-format
  BC by **+5.6pp, CI excludes 0**.
- **Phase 2c/2d:** sparring partners do nothing (−1.0pp / ±0pp despite
  genuinely doubled opponent quality), and a *better* BC checkpoint pushed
  through the exact M7 recipe **regressed 8.3pp** — pointing at capacity, not
  curriculum. **A better imitator made a worse RL substrate.**
  −8.3pp is the figure to quote: it comes from the same-machine controlled
  A/B. An earlier **−6.3pp** reading is superseded — it moved BC corpus, seed,
  and backend (`mps` vs `cuda`) at once. A seed replication put run-to-run
  spread under 1pp, which is what makes the effect real rather than noise.

---

## What has been closed

Six directly-tested hypotheses, all null or weak: richer observations (M3.4,
M7, and M8-1A — **but see correction 3; the M8-1A closure is confounded, so
this direction is weaker evidence than the count suggests**), value-head
targeting (M8-2, replicated), more data (M5.5's ceiling), unrated tournament
data, format alignment (small but real), and sparring partners. Plus one
architecture retired (transformer, M3.2) and one measurement regime rebuilt
(M9).

**Still live:** observation *poverty* of a specific kind — move identity,
species/base stats, trapping, and **defensive** type effectiveness (v3 encodes
only the attacking direction) — which is M11's thesis; model capacity, which
M9 Phase 2d pointed at; tactical blunders, which M10 will actually look for;
and team-luck variance in randbats, still untested.

---

## Architecture Reference

```
Battle state
    │
    ▼
extractFeaturesStructured()          [M2; v2 M3.4; v3 M7]
    │  12 Pokémon tokens × 86 dims (v3)
    │  0–64 v1 base │ 65–76 boosts/volatiles │ 77–85 type-eff, effect flags, Sleep Clause
    ▼
MLP-PPO shared trunk (flattened obs → 128-dim)   [transformer retired, M3.2]
    ├──▶ Policy head (128 → 9 logits)
    ├──▶ Value head  (128 → 1 scalar)
    └──▶ Opp head    (128 → 9 logits)             [M5; aux loss only, sampler retired]
    │
    ▼  [M4]
MCTS (determinized PUCT, sims=100, c_puct=0.5, det=1)
    │  policy as prior, value at leaves, argmax over visit counts
    ▼
Best action                          (greedy decoding is the ladder default)
```

Training recipe as of M7: randbats BC → PPO 5M steps warm-started with value
warmup 200k, BC KL-anchor 0.05, opponent-mix 0.5/0.3/0.2, `--opp-coef 0.1`.

---

## Live Milestones

## M10: Battle Log Analysis — Tactical Error Diagnosis ⏳ PLANNED

**Status:** ⏳ Planned (2026-08-01), **sharpened and re-prioritised 2026-08-03**
— now recommended *ahead of* M11 Phase 1. See `docs/WHERE-WE-ARE.md` →
Recommendation.

**Motivation & Context:** After nine milestones and six directly-tested hypotheses (richer obs, value targeting, more data, unrated tournaments, format alignment, sparring partners), all null or weak, we have high confidence in what does NOT work but zero insight into *what the agent actually does wrong* on the human ladder. The agent wins 93% vs bots but ~19% of contested ladder games — a 70+ point gap. Nobody has looked at the tactical decisions themselves, only aggregate win rates.

**Sharpened 2026-08-03 by the 360-game greedy run.** Three findings from that
analysis bear directly on this milestone's design:

1. **There is now a target signal to explain.** Win rate decays monotonically
   with game length across the 326 contested games: 31.2% at 16–20 decisions,
   24.1% at 21–25, 18.8% at 26–30, **14.6% at 31–40**. Whatever the agent does
   wrong, it compounds with turn count. **Every metric below must be reported
   stratified by game length**, not just wins vs losses — a blunder rate that is
   flat in length does not explain this curve, and one that climbs does.
2. **Stratify by opponent Elo too.** Win rate falls 34.4% → 15.3% from Elo
   1000–1099 to 1200–1299 (r=−0.143, t=−2.72, n=360). Error rates that rise with
   opponent strength mean the agent is being *punished* for a blunder it always
   made; rates that are flat mean stronger opponents simply play better.
3. **Exclude concessions from the loss/win comparison.** 34 of 97 wins in that
   run ended under 16 decisions with the opponent quitting. Including them
   contaminates the "wins" arm with games that were never played — precisely the
   arm this milestone compares losses against. Use
   `ladder_analysis.py --min-decisions 16` semantics.

**Data confirmed usable (2026-08-03):** the logs are not just public protocol —
they carry `|request|` lines with our exact stats, HP, PP and legal move set at
every decision, plus the opponent's full revealed state. Ground truth for damage
calculation and matchup evaluation is therefore already on disk; no re-simulation
is needed to score most Tier 1 metrics. 885 logs available (the plan below says
516; that count is stale).

**Core Hypothesis:** The agent makes specific, categorical tactical errors (legal but strategically bad moves) at rates significantly higher in its losses than in its wins, or at rates higher than humans. These errors are detectable through automated analysis of the full battle logs and can guide targeted fixes or investment decisions.

**Key Insight:** This is the first deep behavioral diagnostic. All prior work was aggregate-level (win rate), architecture-level (adding modules), or data-level (more games). This analysis looks at the game tree itself to ask: "What moves did the agent choose, and how often did it choose bad ones?"

---

### Scope & Plan

See `docs/BATTLE-LOG-ANALYSIS.md` for the comprehensive plan. Summary:

**Phase 1: Data & Parser** (2 days)
- Verify ladder battle logs on disk (**885** games, gzipped, full protocol
  including `|request|` state — verified 2026-08-03)
- Build `models/battle_log_parser.py` to extract game-state trees
- Acceptance: 10 logs parse correctly

**Phase 2: Metrics** (3 days)
- Implement Tier 1 error checkers:
  - Sleep clause violations (illegal move, wasted turn)
  - Freeze clause violations
  - High-damage when priority move secures KO
  - Attack into immunity/resistance
  - Switch into weakness
  - Fail to switch out of bad matchup
- Build `models/battle_log_analysis.py` to run all metrics
- Output: CSV with rates in agent-wins vs agent-losses, with 95% CIs (Wilson/Newcombe)
- **Every metric additionally stratified by game length (16–20 / 21–25 / 26–30 /
  31–40 / 41+) and opponent Elo band**, per the sharpening notes above. The
  length stratification is the primary endpoint: it is the one pattern we
  already know needs explaining.

**Phase 3: Human Baseline** (2 days)
- Sync human replay corpus from home box (if not already synced)
- Run same metrics on ~5k human games
- Compare agent vs human error rates with CIs

**Phase 4: Reporting & Interpretation** (1 day)
- Write `results/battle_log_analysis_results_*.md`
- Pre-registered interpretation: what each finding means and what action it triggers
- Update `docs/WHERE-WE-ARE.md` with any findings
- Update `IN-PROGRESS.md` with next steps

---

### Pre-Registered Interpretation (before any result is computed)

**Positive Finding:** A metric E shows error rate in losses >> wins (95% CI excludes 0) AND/OR agent >> human. 
- Meaning: E is a measurable driver of losses.
- Action: Document in `docs/TACTICAL-ERRORS.md`, assess fix difficulty, recommend intervention.
- Expected value: Single-digit pp points per error; 2–3 concurrent errors could explain 1–5pp of the gap.

**Null Finding:** Error rate same in losses vs wins (CI includes 0) or indistinguishable from human.
- Meaning: E is not a driver of losses; narrows down what matters.
- Action: Record that E was checked and found baseline. Do not pursue.
- Expected value: This is valuable — it closes directions. The project's prior failure mode was chasing weak manipulations; null findings are findings.

**Confounded Finding:** Sample too small, error doesn't occur, or metric untestable.
- Meaning: Evidence inconclusive for this data.
- Action: Flag for Phase 2 (larger sample or focused variant) or accept as out of scope.

---

### Confidence & Risk Assessment

**Success probability by outcome:**
- ~40% no actionable errors found at Tier 1 → points to observation state / value function / team luck / format difference as the constraint (obs and value already tested in M8; luck and format are new hypotheses)
- ~25% errors found but individually small (<1pp each) → multiple small errors might sum; documentation valuable
- ~20% success: find 2–3 measurable errors explaining 1–5pp of the gap → enables targeted fix or informs bigger-picture decision
- ~15% confounds or data issues → mitigated by stratification by opponent Elo

**Why this matters:** If this analysis yields actionable errors, the project has a concrete next target (e.g., "fix observation of status state," "add reward shaping for rule violations"). If it yields nulls across the board, it's strong evidence the gap is not driven by tactical blunders, which narrows the focus to higher-level constraints (model capacity, team luck, observation richness, or the format itself).

---

### Execution Plan

**Parallelization:** Phase 1 & 2 can overlap (start writing checkers while parser finalizes).

**Effort breakdown:**
- Phase 1: 2 days
- Phase 2: 3 days  
- Phase 3: 2 days (conditional on home-box sync)
- Phase 4: 1 day
- **Total: ~8 days if home-box sync exists; ~5 days if agent-only metrics suffice**

**Go/no-go decisions:**
- After Phase 1: If logs are too corrupted or incomplete, abort and document the data limitation.
- After Phase 2: If no Tier 1 metrics fire (all null), decide whether to pursue Tier 2 (exploratory) before Phase 3.

**Recommendation (updated 2026-08-03):** Execute **before M11 Phase 1**, after the
two trained width arms are evaluated. This is a diagnostic, not a solution, and
that is the point: M11 is a multi-day retraining justified by a hypothesis built
from a code read, one observed live game, and an aggregate length correlation.
M10 needs no retraining, runs on data already on disk, and its output survives
M11 invalidating every checkpoint. It should either concentrate M11's schema
work on the dims that actually cost games, or find something the code review
missed.

---

### Deliverables Checklist

- [ ] `docs/BATTLE-LOG-ANALYSIS.md` — plan document (complete)
- [ ] `models/battle_log_parser.py` — parse gzipped logs → game trees
- [ ] `models/battle_log_analysis.py` — apply Tier 1 rule checkers
- [ ] `models/analyze_ladder_logs.py` — CLI driver for agent analysis
- [ ] `models/analyze_human_replays.py` — CLI driver for human baseline (if Phase 3 runs)
- [ ] `results/battle_log_metrics.csv` — per-metric summary (n, rate, CI)
- [ ] `results/battle_log_analysis_results_*.md` — full report with interpretation
- [ ] `docs/WHERE-WE-ARE.md` — updated with any findings + next-step recommendations
- [ ] `IN-PROGRESS.md` — updated with results and blockers for M11+

---

### Unblocks

**If positive:** Concrete intervention targets; evidence that tactical fixes are high-ROI.

**If null/inconclusive:** Strong evidence to deprioritize tactical optimization; redirects to capacity (M11 → bigger-brain investigation) or team luck (format re-evaluation).

---

## M11: Observation Enrichment + Reward Asymmetry ⏳ SCOPED

**Status:** ⏳ Scoped 2026-08-01 (code review findings; pending user approval to build)

**Thesis:** The 93%-vs-bots / 30%-vs-humans gap is driven by observation poverty.
The agent encodes moves without identity, carries no species/stats/trapping, and
cannot reason about damage. Against `Random` and `DamageFirst`, which never
switch, base power and type suffice. Against humans, which switch constantly and
reason with stats and coverage, the agent is playing blind. This is the first
hypothesis that predicts the *shape* of the gap (`docs/CODE-REVIEW-FINDINGS.md
§1`) rather than another knob. Separately, the reward clip is asymmetric —
penalizing long losses but not long wins — which favors stalling when behind,
punishing the switching-heavy play this project has spent three milestones
trying to teach.

**⚠️ CRITICAL CAVEAT:** A new observation schema invalidates every existing
checkpoint (BC and PPO). M11 means retraining the lineage from scratch:
BC→PPO→eval→ladder. Estimate BC retraining (~2–3 hours), PPO training
(~2 hours for an A/B run on each opponent), full eval (RL evals + ladder if
gates pass). This is not a small cost and must be stated plainly, not buried.

---

### M11 Phases (contingency gates between them)

**Phase 0 (Reward asymmetry fix): ~3 lines, ~1 hour.** Quick smoke test of a
new hypothesis from the code review.

- **Goal:** Test whether `battle-sim.ts` (MCTS forward model) and `gym.ts` (env)
  should apply identical reward scaling, not just shaped-vs-unshapen. M8 Phase 2
  found value-head fine-tuning improved in-distribution calibration but didn't
  transfer to play strength; MCTS evaluates at leaves using the fine-tuned head,
  but the targets were collected on a different reward scale.
- **Design:** Apply both the turn penalty and the reward clip symmetrically in
  `battle-sim.ts` (currently applies neither). Run M8 Phase 2's value fine-tune
  pipeline on the M7 checkpoint with this fix; if the fine-tuned head transfers
  to +3pp bot eval, escalate to Phase 1 (obs richness). If still null, conclude
  the issue is deeper.
- **Effort:** 30 min code, 2 hours eval.
- **Acceptance:** Report whether the symmetry fix moved the needle on Phase 2's
  reproduction run. If yes (≥+1pp, not a hard gate), proceed. If no, the issue
  is not reward scale.

---

**Phase 1 (Observation enrichment):** Add move identity, species, stats, trapping, recharge.

- **Goal:** Close the observation-poverty hypothesis end-to-end. Each missing
  feature has been verified as critical and invisible to the current encoder:
  - Move identity: Recover/Swords Dance/OHKO moves are byte-identical
  - Species/stats: No damage estimate formable; bulk unknown; type coverage invisible
  - Trapping: `maybeLocked` never read; agent cannot learn to punish recharge
  - Hyper Beam recharge: Encodes as 0-BP physical Normal; agent structurally cannot punish
  - **Defensive type effectiveness is absent — offensive is not.** v3 dims 77–80
    encode *this* token's moves vs the opponent's active (`fillV3MoveDims`, the
    only defender ever passed is `ctx.defenderTypes`). Nothing encodes the
    reverse: the opponent's known moves/STAB vs *this* token's typing. So
    picking a resistant switch-in requires learning the 15×15 chart from two
    type one-hots, while the attacking direction is handed over precomputed —
    an asymmetry that bites hardest exactly where humans switch and bots don't.
- **Design:**
  - New observation schema `v4` (working name; dims TBD):
    - Per-move: add 1 dim for move_id (encoded 0–162 via `Dex.mod('gen1').moves` LUT)
    - Per token: 1–4 dims of **defensive** type effectiveness — the opponent
      active's revealed moves (and its STAB types, for the unrevealed case) vs
      this token's typing. Reuses `computeTypeEffMultiplier`/`encodeTypeEff`
      from `type-chart-v3.ts` with the arguments swapped; no new type logic.
      Cheapest item in the schema and the one that most directly targets
      switch-in selection.
    - Per active Pokémon: add species_id (1 dim, 0–151 LUT), base stats (5 dims:
      HP, Atk, Def, SpA, Spe), trapping-locked flag (1 dim)
    - Recharge state: Hyper Beam recharge and multi-turn moves tracked as
      `|-cant|…recharge` and `|-cant|…partiallytrapped` (parser already sees
      these); encode as a flag on the active token
  - Update `sim/tools/feature-extractor.ts` with v4 schema and LUTs
  - Add matching `sim/tools/replay-adapter.ts` v4 path for BC retraining
  - All other infrastructure (bridge, gym_client, schema dispatch) auto-detects
    new size and flows through unchanged (tested in M3–M9)
- **Implementation path:**
  1. Schema design: decide dims, LUT structure, naming (1 day design + test)
  2. Feature extractor + replay adapter + tests (1 day)
  3. Build green, smoke training (2–3 hours on Mac)
  4. BC retraining from raw replays (2–3 hours on home box per corpus)
- **A/B runs (parallel arms, n=2,000/opponent, gate: ≥+3pp CI excl. 0):**
  - Control: M7 checkpoint (`v3/ppo_step_5000002_final.pt`), v3 observation
  - Candidate: v4 BC (trained on mixed corpus) warm-start → PPO 5M steps (M7 recipe)
  - Report vs Random, vs DamageFirst, ladder candidate (if gates pass)
- **Effort:** ~3–4 days (schema + code + BC retraining + PPO training + eval). BC
  retraining is the parallelizable dependency (home-box task; can overlap with
  feature work on the Mac).

---

**Phase 2 (if Phase 1 gates positive): Full ladder validation.** Same protocol
as M9 Phase 3 — fresh accounts, paired concurrent A/B, power for +10pp, gate on
the paired difference.

---

### Instrumentation Debt (parallel to Phase 1)

The code review identified several defects that prevent future debugging:
- `train.py` has no seeding anywhere, so no run-to-run A/B can use common random
  numbers
- No entropy, KL, clip fraction, or value loss is ever logged (only win-rate +
  summed loss)
- No run manifest (device / argv / SHA)

**Fix alongside M11 Phase 1** (cheap, ~4 hours):
- Add `--seed` to both trainers and seed all RNGs before creating the policy
- Add histogram logging for entropy, KL, clip fraction, value loss per-step
- Write `meta.json` with device, git SHA, argv, timestamp (template: `collect_value_data.py`)

This cost is paid once and unblocks future A/Bs from confound creep.

---

### Success Criteria

- **Phase 0 (reward fix A/B):** Report MSE change and bot-eval deltas vs M7.
  If ≥+1pp and not regressed, proceed to Phase 1. If no movement, conclude
  reward scale is not the issue.

- **Phase 1 (obs enrichment A/B):**
  - ✅ v4 schema properly implements all five feature categories above
  - ✅ BC retraining completes and produces a checkpoint (v4 BC)
  - ✅ v4 arm trains to 5M steps on the M7 recipe
  - ✅ Raw-policy bot evals: ≥+3pp vs Random at n=2,000, CI excluding 0
  - ✅ Report vs DamageFirst alongside (no gate, but directional)

- **Phase 2 (if gates pass):**
  - Paired ladder A/B at n≥350/arm, gate on difference: ≥+10pp with CI excl. 0
  - Same protocol as M9 Phase 3

- **Instrumentation debt:**
  - ✅ Trainer has `--seed` and uses it
  - ✅ Logs include entropy, KL, clip fraction per step
  - ✅ `meta.json` written on every run

### Decision Rules (pre-registered)

**After Phase 0:** If the reward asymmetry fix doesn't move the needle (≤+0.5pp),
conclude the issue is deeper in the observation, and proceed to Phase 1 anyway
(this was the pre-registered hypothesis; a new candidate cause doesn't override
it). If it does move the needle (≥+1pp), it's a cheap fix to include before
Phase 1.

**After Phase 1:** If v4 beats v3 by ≥+3pp on both Random and DamageFirst at
n=2,000 with CIs excluding 0, escalate to Phase 2 (ladder). If it's negative or
inside noise, close the hypothesis as "observation enrichment alone does not
close the gap" and stop (a finding worth recording).

---

### Unblocks (if positive)

Concrete evidence that observation poverty was the binding constraint; actionable
directions for M12+ (other missing observations? bigger network now that the
agent can use it? ladder optimization?). If gates pass, M7 shipping agent is
replaced.

### Risks and Mitigations

**Risk 1: BC retraining is a bottleneck.** Run on home box in parallel with code
work on the Mac. Estimated 2–3 hours per corpus.

**Risk 2: Schema design choices cascade.** Design dims carefully; test shape
stability on a 4k-step smoke first. All infrastructure auto-detects size, so
shape mistakes are caught early.

**Risk 3: New obs schema still doesn't help.** This is a possible finding, not
a bug. Record it honestly and move to other directions (capacity, team luck, or
stopping).

