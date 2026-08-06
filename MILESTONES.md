# Pokemon Showdown AI Training — Milestones

Goal: Build a genuinely intelligent Pokemon trainer AI.

---

## 🏁 PROJECT STATUS: BOUNDED FINISH (decided 2026-08-05)

**The project is being wound down deliberately, on a fixed scope, with one
milestone left.** This is a decision to end on an *answer* rather than let the
work fade out mid-blocker. It is not a failure state — see "Why" below.

**The only remaining work is M12 (fixed-team Gen 1 OU), Phases 0–4.** Phase 5
(ladder) is optional and runs only if Phase 4 surprises upward. When Phase 4
reports, the project is done and gets archived.

| Milestone | Disposition |
|---|---|
| **M12 Phases 0–4** | ✅ **THE REMAINING WORK.** Roster → plumbing → BC → PPO → bot-eval gate |
| M12 Phase 5 (gen1ou ladder) | ⚪ Optional. Only if Phase 4 clears its gate with room to spare |
| M10 (battle log analysis) | ❌ **CLOSED, NOT PURSUED.** ~8 days for a diagnosis nothing downstream would act on |
| M11 Phase 1 (obs schema v4) | ❌ **CLOSED, NOT PURSUED.** Was already deferred; the pivot ends before it |
| M11 eval battery (h128/h512) | ⚪ Optional, off the critical path. Hours of compute once SSH is back; expected null; changes no decision |

**Why now.** Eight milestones, seven directly-tested hypotheses back null or
weak, every intervention sized at 1–5pp against a ~30pp gap. The agent wins
77.7% vs an opponent playing random legal moves and **19.3% of contested ladder
games** — mediocre across the board. The remaining ideas are real but small, and
the honest ceiling on M12 is "less bad in a narrower format," not "competitive."
Meanwhile the cost per result is rising: infrastructure (multi-machine sync, SSH,
WSL portproxy) now consumes more of a session than the ML does.

**What M12 is for.** Closure with a number. Fixed teams is the one idea that
shrinks the *problem* instead of adding another knob, and it deserves to be
tried rather than shelved as a hypothetical. Phase 4's bot-eval gate is a real
pre-registered go/no-go, so it terminates cleanly either way.

**What the project actually produced.** The agent is mediocre; the experimental
method is not. Pre-registered gates, recorded sample-size widenings, a
dead-ends table, four self-issued corrections invalidating the project's own
earlier numbers, and confounds caught in M4 and M8 by its own review. That is
the durable output, and it is written down in `docs/EVALUATION-METHODOLOGY.md`.

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

## M10: Battle Log Analysis — Tactical Error Diagnosis ❌ CLOSED — NOT PURSUED

**Status:** ❌ **Closed 2026-08-05 without execution**, as part of the project
bounded finish (see PROJECT STATUS at the top of this file). Planned 2026-08-01,
sharpened 2026-08-03, never built beyond the throwaway probe below.

**Why closed.** M10 costs ~8 days and produces a *diagnosis*, not a fix. In a
wind-down there is nothing downstream to act on it: M11's v4 schema is closed,
and M12 ends at its bot-eval gate. The one thing M10 would have justified —
targeted observation work — is exactly what the bounded finish drops.

**What survives.** The preliminary probe below is the project's only behavioral
(as opposed to aggregate) evidence, and it fires cleanly. It stays on record as a
lead for anyone who picks this up later. It is **not** a finding: it has no human
baseline, and causation runs either way (losing positions may *force* defensive
shuffling). Do not cite it as a demonstrated agent error.

The plan text below is preserved as-is for that future reader; **none of the
deliverables were built.**

---

<details>
<summary>Original M10 plan (unexecuted) — expand if picking this up later</summary>

**Status when closed:** ⏳ Planned (2026-08-01), sharpened and re-prioritised
2026-08-03.

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

**Preliminary probe (2026-08-03, ~30 min, 864 logs) — the first tactical error
metric already fires.** Definition: a *wasted-turn double switch* is a voluntary
switch-in that we voluntarily switch back out on the very next turn, with no
faint forcing either move — a free turn handed to the opponent, and the exact
pattern observed live on 2026-08-02.

| | double-switches / voluntary switches | rate | 95% CI |
|---|---|---:|---|
| our wins | 18 / 297 | 6.06% | [3.87, 9.38] |
| our losses | 230 / 1468 | **15.67%** | [13.90, 17.62] |
| difference | | **+9.61pp** | **[+5.85, +12.54]** ✅ excludes 0 |

And it climbs with game length, tracking the win-rate decay: **3.7%** (≤20
turns) → 13.9% (21–25) → 14.1% (26–30) → **17.1%** (31–40). Holding length
fixed at 26–30 turns it is still 14.9% in losses vs 8.8% in wins.

**This is a lead, not a finding.** Three caveats that Phase 2/3 must resolve:
(a) **direction of causation is unresolved** — losing positions may force
defensive shuffling, so this could be a symptom rather than a cause; (b) it
pools the greedy and sampling eras across all 864 logs on disk, not the clean
360-game run; (c) **it has no human baseline**, which is the only thing that
converts "we do this 16% of the time" into "this is a mistake." **Phase 3 is
therefore promoted from conditional to required** — the human comparison is
what makes this metric mean anything. Probe script was throwaway; Phase 1 should
reimplement it properly inside `models/battle_log_parser.py`.

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

**Phase 2: Metrics — Format-General Only** (3 days)
- Implement Tier 1 error checkers — **format-general tactics only** (re-scoped
  2026-08-03 for format pivot):
  - **Included (survive format change):**
    - Double-switch waste (voluntary switch back next turn, no faint)
    - High-damage when priority move secures KO
    - Attack into immunity (via type alone, not matchup)
    - Failure to switch out of known weakness
  - **Excluded (Gen1RandBats-specific, dropped):**
    - Species-specific blunders (e.g., bringing Electric into Water)
    - Matchup-specific errors (e.g., setup vs special walls)
    - Team-roster synergy failures
  - Rationale: Fixed-team Gen 1 OU completely changes the roster and
    matchup space; species-specific metrics will not transfer. Format-general
    tactics like "waste turns" or "miss priority KOs" transfer to any format.
- Build `models/battle_log_analysis.py` to run format-general metrics only
- Output: CSV with rates in agent-wins vs agent-losses, with 95% CIs (Wilson/Newcombe)
- **Every metric additionally stratified by game length (16–20 / 21–25 / 26–30 /
  31–40 / 41+) and opponent Elo band**, per the sharpening notes above. The
  length stratification is the primary endpoint: it is the one pattern we
  already know needs explaining.

**Phase 3: Human Baseline** (2 days) — **REQUIRED, not conditional** (promoted
2026-08-03: the double-switch probe shows a rate with no reference point, and
without the human number every Tier 1 metric has the same problem)
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
- Phase 3: 2 days (required as of 2026-08-03; needs the home-box replay sync)
- Phase 4: 1 day
- **Total: ~8 days.** The "~5 days if agent-only metrics suffice" variant is
  withdrawn — agent-only error rates have no reference point and cannot be
  interpreted.

**Go/no-go decisions:**
- After Phase 1: If logs are too corrupted or incomplete, abort and document the data limitation.
- After Phase 2: If no Tier 1 metrics fire (all null), decide whether to pursue Tier 2 (exploratory) before Phase 3.

**Recommendation (updated 2026-08-03):** Execute **after M11 eval battery is
evaluated (when home box becomes reachable) and before the pivot training**.
This is a diagnostic that transfers across format: its findings on format-general
blunders will apply to Gen 1 OU and will not be invalidated by the pivot.
M10's format-general metrics can be re-run on Gen 1 OU later to track whether
the agent's tactical profile improves post-pivot.

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

</details>

---

## M12: Fixed-Team Gen 1 OU Pivot 🏁 THE FINAL MILESTONE

**Status:** ⏳ Approved 2026-08-03. **Re-scoped 2026-08-05 as the project's last
milestone** — Phases 0–4 only, Phase 5 optional. When Phase 4 reports, the
project ends. See PROJECT STATUS at the top of this file.

**Terminal condition (pre-registered 2026-08-05, before any Phase 0 work):**

| Phase 4 outcome | What happens |
|---|---|
| Both gates pass (≥10% vs Random **and** DamageFirst) | Record the number. **Optionally** run Phase 5 for a gen1ou ladder read, then archive |
| Either gate fails | Record it as the finding. **Do not diagnose, do not iterate, do not try a second roster.** Archive |

The "try an alternative roster" mitigation under Risk 1 below is **withdrawn** —
it was written when M12 was a platform for future work. Re-rolling the roster
after seeing the gate result is exactly the sweep-picking this project banned in
its own standing rules. One roster, pre-registered, one result.

**Decision Context:** The project is **pivoting from gen1randombattle (random
teams) to fixed-team Gen 1 OU**. This reduces team-luck variance and allows
measuring whether the agent's observed weaknesses (vs. DamageFirst, team-
composition-driven performance variance) are inherent to random teams or persist
with a stable roster. Format move is orthogonal to observation enrichment (M11-
deferred below); the two are bundled to avoid confounding a format change with
observation changes. This is a deliberate choice recorded plainly: **fixed-team
and Gen 1 OU changes are packaged together. The design cannot separate their
effects.**

**Bundles together:**
1. Fixed roster (6 Pokémon, always the same lineup per battle format)
2. Format switch (gen1ou instead of gen1randombattle)

**Cannot be separated because:** Fixed-team plumbing (team selection, roster
encoding, permutations in BC preprocessing) requires format decision. Format
switch requires re-evaluation on a new ladder (slower per battle, smaller sample
sizes). Both must land together for a clean experimental story.

**Baseline reset:** Every bot-eval number and all ladder records are from
randbats with random teams. Gen 1 OU evals will be measured from scratch against
the same bots (Random, DamageFirst) on fixed Gen 1 OU teams. The 93.0% (Random)
/ 84.2% (DamageFirst) v3 numbers do not transfer — they are randbats-specific.

**Data note:** Gen 1 OU archive (~10.7k games at ≥1300 quality) is smaller than
randbats (~21.6k games) and lower-quality on average (24% of gen1ou are ≥1300 vs
100% of the scraped randbats). This is a trade-off: on-format data is more
relevant but smaller. BC will re-train on gen1ou + tournament games as the
primary corpus. See `docs/DATA-INVENTORY.md` for the full audit.

---

### M12 Phases (sequential, contingency gates between them)

**Phase 0 (Team and roster selection): ✅ DONE 2026-08-05**

Roster locked and pre-registered: **Tauros / Chansey / Snorlax / Exeggutor /
Starmie / Alakazam**, each slot on its modal human move set. Rank-#1 most-used
exact team across 16,743 revealed teams from 10,101 local replays rated ≥1300
(6.50%, vs the Rhydon variant's 6.10%). Packed team:
`config/rosters/gen1ou-standard.txt`. Full rationale, the usage table, the
tiebreak and two caveats: `docs/BATTLE-FORMATS.md` → THE FIXED ROSTER.
Reproducible via `scripts/mine_gen1ou_teams.py` / `mine_gen1ou_movesets.py`.
Design options 2 and 3 below (multi-roster averaging, permutations) were
**withdrawn** with the bounded finish — one roster, one result.

**Phase 1 (Plumbing): ✅ DONE 2026-08-05**

`sim/tools/roster.ts` is the single source of truth; `--format gen1ou` threads
the same packed team to both seats through the gym, evaluator, bridge, Python
clients, trainer and ladder bot (`/utm`). `./build` passes, 126/126 project tool
tests pass (14 new). **Found and fixed a real bug:** `Teams.getGenerator` does
not throw for `gen1ou` — it silently falls back to the gen1 *random* generator,
so MCTS would have searched against a bench the opponent cannot have. The
determinizer now fills from the roster; the ladder path still samples on purpose.
**BC needed no changes** (the v3 observation has no species or stats, so the
roster is invisible to the encoder). Measured: **no seat bias** (p1 49.2%, n=600,
CI [45.2, 53.2]) and **~6% draws / ~111-turn games** — Phase 4 must report draws.

**BC corpus pre-registered:** the mixed `gen1randombattle,gen1ou` default (M7's
recipe). Chosen for continuity, not from a result — see `IN-PROGRESS.md` → Next
Steps item 2 for why M9 Phase 2a/2c does not cleanly transfer to a gen1ou target.

**Phase 2 (BC retraining): ✅ DONE 2026-08-06**

Home box SSH restored (WSL). `data/replay_trajs/v3` regenerated there first — it
is tier-2 local-only data and was absent; `replay_adapter_cli.js --obs-v3` on
both formats reproduced the documented coverage exactly (**gen1ou 86.4%**,
**randbats 90.7%**; 98,961/98,985 + 21,520 battles, 9.74M decisions).

Command, verbatim — the M7 Job 4.1 recipe with only `--out` changed:
```
.venv/bin/python models/bc_pretrain_mlp.py --epochs 5 --obs-v3 --out bc_mlp_m12.pt
```
5/5 epochs, 25,397,540 samples, 973s. Checkpoint `models/checkpoints/bc_mlp_m12.pt`.

| format | policy acc | opp-head acc | value-sign acc | val samples |
|---|---|---|---|---|
| gen1randombattle | **53.1%** | 34.5% | 62.3% | 121,876 |
| gen1ou | **55.1%** | 35.6% | 67.2% | 36,434 |

**This is a replication, not an improvement.** Against M7 Job 4.1 (52.7% /
54.0%) the deltas are +0.4pp and +1.1pp — single arm, no CI, nothing controlled.
The supported claim is that the pipeline reproduces after a corpus regeneration
on a different machine. **Do not quote these deltas as evidence of anything.**

**Caveat on comparability:** shards are now battle-sized (50 gen1ou / 11
randbats vs the archive's 99 / 21), so `--val-shards 1` holds out ~2× the
validation data the archived run used. The accuracy figures rest on more data
but are not comparable to the archive at fine precision.

**No roster filtering was applied, and none is possible.** Phase 1 established
that the v3 observation carries no species or stats, so the fixed roster is
invisible to the encoder and there is no feature to filter the corpus on.

**Phase 3 (PPO training): ✅ DONE 2026-08-06**

```
.venv/bin/python models/ppo/train.py --format gen1ou --obs-v3 --steps 5000000 \
  --rollout-steps 512 --num-envs 8 \
  --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2" \
  --pretrain-checkpoint models/checkpoints/bc_mlp_m12.pt \
  --bc-anchor models/checkpoints/bc_mlp_m12.pt --bc-anchor-coef 0.05 \
  --value-warmup-steps 200000 --opp-coef 0.1 \
  --checkpoint-dir models/ppo/checkpoints/m12
```

Final: `ppo_step_5000005_final.pt` (hidden=128, 151,187 params, lr 3e-4, cuda).
Training-mix win rate by half-million steps — **stable, no collapse**:

```
0.0M–0.5M  0.616   1.5M–2.0M  0.751   3.0M–3.5M  0.744   4.5M–5.0M  0.737
0.5M–1.0M  0.731   2.0M–2.5M  0.729   3.5M–4.0M  0.739
1.0M–1.5M  0.735   2.5M–3.0M  0.748   4.0M–4.5M  0.732
```

**Not a strength number** — half the opponent mix is self-play, which converges
toward 0.5 by construction. The 4M steps of flat curve are consistent with this
project's repeated finding that RL extracts little more from the v3 schema.

**Deviation from M7, recorded:** `--selfplay-pool` was left at its default (this
run's own checkpoints) instead of being seeded with M2/M3.3-best. Those are
randbats-trained and would have sat in the p2 seat as off-format opponents for
the whole run. The three things Phase 3 specified as "the M7 recipe" —
opponent-mix, value warmup, KL-anchor — all match exactly.

**Phase 4 (bot-eval gate): ✅ PASSED 2026-08-06 🏁 TERMINAL**

Raw policy, **sampled** decoding, no search, fixed roster both sides,
n=5,000/opponent as pre-registered.

| opponent | win rate | 95% CI (Wilson) | draws | losses | gate ≥10% |
|---|---|---|---|---|---|
| RandomPlayerAI | **95.88%** (4794/5000) | [95.29, 96.40] | 0 | 206 | ✅ |
| DamageFirstAI | **92.20%** (4610/5000) | [91.42, 92.91] | 0 | 390 | ✅ |

Both lower bounds clear the bar by ~80pp. **What this establishes is that the
format pivot did not cause a catastrophic regression — which is what the gate
was explicitly designed to catch, and nothing more.** The comparison to M7's
69.7% / 59.4% is invalid: different format, mirror roster, and the mirror
removes the team-luck variance that dominated randbats. A strong team played by
RandomPlayerAI is a much easier opponent than a random team played randomly.
**Do not put these numbers in a table with M7's without this caveat.**

⚠️ **The 0-draw result is unconfirmed.** `evaluate.py` had no draw handling at
all — draws were silently counted as losses — so draw counting was added in the
same session (commit `712cf5670`) and then never observed to increment across
10,000 games. The branch is reachable and does not crash, but it has no positive
test. Phase 1 measured ~6% draws on this roster in mirror self-play, where both
sides stall competently; fast losses to bots are a plausible explanation for 0
here, but it is not verified. The gate is unaffected — `win_rate` divides by all
battles regardless.

Opp-head top-1 accuracy: 0.259 vs Random, **0.891 vs DamageFirst** (predicting a
deterministic heuristic). A sanity check on the aux head, not a strength signal.

**Phase 5 (gen1ou ladder): ⏳ RUNNING from 2026-08-06**

Optional, and user-authorized explicitly. tmux `m12ladder` on the home box,
account Novapool, `--run-id m12-ladder`, n=356 (~24 h) — the pre-registered
floor giving ±4.8pp single-arm precision at p≈0.30. **Decision rule is greedy**
(`ladder-bot.js` default since 2026-08-01), which differs from Phase 4's sampled
eval; greedy on the ladder has never been tested in this project.

**This produces a standalone number, not a comparison.** The 30.5% / 26.9%
baselines are the **gen1randombattle** ladder — a different pool, a different
rating distribution, and no concurrent control arm. Per `LADDER-MEASUREMENT.md`,
non-concurrent ladder readings do not support comparison. Read: `.venv/bin/python
scripts/ladder_analysis.py --run m12-ladder`.

<details>
<summary>Original Phase 0 + Phase 1 plans (superseded by the records above)</summary>

**Phase 0 (Team and roster selection): ~1–2 days**

- **Goal:** Pick a fixed 6-Pokémon Gen 1 OU roster to use for all training and
  evaluation. This is a one-time decision; the fixed roster is used for every
  battle from here forward (until the next format pivot).
- **Design options:**
  1. Use a single canonical "competitively strong" roster (e.g., Dragonite, Alakazam, Gengar, etc.)
  2. Use multiple roster permutations and average results (higher variance, more data)
  3. Sample from the top-1300 replay archive (represents human play)
- **Pre-register before the run:** which roster(s) will be used, and how many
  permutations (if multi-roster). This choice drives BC corpus composition and
  all downstream evals.
- **Effort:** ~1 day for roster curation + code changes; option 2 or 3 require more iteration.
- **Acceptance:** Roster decision recorded in `docs/BATTLE-FORMATS.md` with
  rationale. Code passes `./build`.

---

**Phase 1 (Plumbing & BC Corpus Decision): ~1 day**

- **Goal:** Integrate fixed teams into the gym, evaluator, ladder bot, and BC
  preprocessing. Verify that team plumbing works end-to-end.
- **Design:** The codebase already supports fixed teams (`tools/gym.ts` can
  toggle `--random-teams off`). Work needed:
  1. Encode the chosen roster in `config/` or a new `rosters/` directory
  2. Update `models/bc_pretrain_mlp.py` to accept gen1ou replays with fixed-team
     format (BC preprocessing may assume dynamic teams)
  3. Gym and evaluator test: verify battles run with fixed rosters
  4. Ladder bot integration: pass roster to the bot launcher
- **BC corpus decision:** Decide whether to re-train BC on gen1ou replays alone
  (smaller corpus, on-format, but noisier) or mixed gen1ou+tournament
  (current practice). Record this as a pre-registered choice.
  - **Recommendation:** Start with gen1ou+tournament (proven mixed corpus) to
    avoid confounding format + corpus changes. Can re-visit post-pivot if results
    warrant.
    > ⚠️ **Superseded and the wording was wrong.** Tournament (`smogtours-`)
    > games are already *inside* the gen1ou corpus (74% of it), so
    > "gen1ou+tournament" described gen1ou-**only**. The actual pre-registered
    > choice is the mixed `gen1randombattle,gen1ou` default — see the Phase 1
    > record above.
- **Effort:** ~1 day (plumbing + tests).
- **Acceptance:** `./build` passes; gym/evaluator/ladder-bot run 10 battles each with
  fixed roster; BC starts on the chosen corpus.

</details>

---

**Phase 2 (BC Retraining on Gen 1 OU): ~2–3 hours (home box)** ⏳ NEXT — blocked on home box SSH

- **Goal:** Train a fresh BC checkpoint on the gen1ou corpus with the fixed roster.
- **Design:**
  - Re-run `models/bc_pretrain_mlp.py` on `data/replays/gen1ou` with the
    fixed-roster encoding
  - Measure BC accuracy on a held-out validation set (randbats and OU both carry labels)
  - Report: action count, label coverage %, top-1 accuracy
- **Effort:** ~2–3 hours on home box (CPU-bound shard processing)
- **Acceptance:** BC checkpoint saved (`bc_mlp_gen1ou_fixed.pt`). Accuracy
  reported and compared to M7's v3 BC baseline.

---

**Phase 3 (PPO Training on Gen 1 OU): ~2–3 hours (home box or Mac)**

- **Goal:** Train a fresh policy on gen1ou with fixed teams, warm-started from
  the Phase 2 BC checkpoint.
- **Design:**
  - 5M-step PPO run identical to M7 recipe (opponent-mix, value warmup, KL-anchor),
    warm-started on gen1ou BC
  - Single arm (randbats-to-OU is a format pivot, not a comparative A/B)
  - Run on same machine as BC to avoid drift confounds (see EVALUATION-METHODOLOGY.md)
- **Effort:** ~2–3 hours training + overhead
- **Acceptance:** Checkpoint saved; training curve stable (no collapse); final step
  ready for eval.

---

**Phase 4 (Bot-Eval Gate on Fixed Roster): ~3–4 hours**

- **Goal:** Measure raw-policy performance against bots on the new fixed-roster
  agent. This is the go/no-go for ladder validation.
- **Design:**
  - **Pre-registered gate: n=5,000/opponent, baseline win rate ≥10% on both
    Random and DamageFirst.** (The rationale: randbats agent averaged 69.7%
    (Random) / 59.4% (DamageFirst); a fixed-team, on-format agent starting from
    BC may be weaker initially due to smaller corpus and roster unfamiliarity.
    Expect ≤50% vs Random as a plausible outcome. The gate is set to catch
    catastrophic regression, not to pass M7's randbats ceiling. Specific numbers
    can be re-visited after Phase 3 if BC accuracy is surprisingly high/low.)
  - Decision rule: `--model ppo`, no search, `sampled` decision (same as M7
    historical for comparability, though the format is different)
  - Both Random and DamageFirst reported with CIs; gate is a joint rejection rule
    (if either <10%, abort)
- **Effort:** ~3–4 hours (27 battles/s, 5k battles, ~50 min per opponent + overhead)
- **Acceptance criteria:**
  - ✅ ≥10% vs Random, 95% CI lower bound > 0 (non-zero performance)
  - ✅ ≥10% vs DamageFirst, 95% CI lower bound > 0
  - ✅ No anomalies in action distributions (e.g., stuck in a loop)
  - **If both pass:** Proceed to Phase 5 (ladder). If either fails, diagnose
    (likely: BC corpus mismatch, roster too weak, observation issue) and halt.

---

**Phase 5 (Ladder Validation on Gen 1 OU): ~24–48 hours — ⚪ OPTIONAL**

> **Optional as of 2026-08-05.** Phase 4 is the project's terminal gate. Run
> Phase 5 only if Phase 4 clears both gates with room to spare and you want a
> live-ladder number for closure. It establishes a baseline for future work that
> is not planned. User runs ladder sessions personally — hand over the command,
> do not launch it in a background task.

- **Goal:** Run on the live gen1ou ladder to establish a baseline for fixed-team,
  on-format play.
- **Design:**
  - Fresh account per arm, `--mcts` (determinized search as shipped)
  - Single arm only — this is a baseline establishment, not a comparative A/B.
  - Pre-register: **n=360 games, power for detecting a +10pp effect size
    (baseline ~30% expected from randbats transfer, but it's a format switch)**.
  - Decision rule: `--greedy` (current ladder default)
- **Pre-registered acceptance:** Report n, W/L, win rate with 95% CI, mean
  opponent Elo. No threshold gate — any non-zero win rate is a baseline. Success
  is having a measurable, interpretable number on the new format.
- **Effort:** ~24–48h wall-clock (360 games × ~4 min/game, accounting for queue
  and re-connects)
- **Acceptance:** Ladder run logged. If ≥5% (sanity check; catastrophic regression
  would read <1%), the format pivot is established and future improvements can
  be measured against this baseline.

---

### Pre-Registered Decision Rules

**After Phase 0 (roster selection):** Record the choice plainly. No gate.

**After Phase 1 (plumbing):** If tests pass, proceed. If plumbing has gaps or
BC preprocessing fails, address before Phase 2.

**After Phase 2 (BC retraining):** If BC accuracy is <30% on a held-out set,
investigate (may indicate corpus/format mismatch). Otherwise proceed.

**After Phase 3 (PPO training):** If training loss goes flat or collapses, halt
and diagnose. Otherwise proceed.

**After Phase 4 (bot-eval gate):** Gate is ≥10% on both Random and DamageFirst,
95% CI lower bound > 0. If either fails, halt and diagnose (likely BC mismatch
or roster weakness). If both pass, proceed to Phase 5.

**After Phase 5 (ladder):** Baseline established. No gate. Future comparisons
(e.g., observation enrichment on gen1ou, bigger models) will be measured against
this Phase 5 number, not against the randbats M7 baseline.

---

### What This Changes

- **Observation enrichment (M11-deferred below) is re-scoped post-pivot.** With a
  fixed roster, the observation schema calculus changes drastically: species
  identity drops from 151 values to 6 compile-time constants; base stats are
  static lookups instead of per-gym encodings. The v4 design should be
  re-derived after M12, not before. The current design is randbats-specific and
  must not ship to gen1ou unchanged.
- **Ladder numbers reset.** The 30.5% (M7 randbats) / 22.9% (contested) are no
  longer the active comparison baseline. Phase 5 establishes the gen1ou baseline.
- **Team luck is removed.** Every eval and ladder number going forward is on
  the same roster, so performance variance is agent-only, not roster-luck.

---

### Success Criteria

- ✅ Phase 0: Roster chosen and recorded
- ✅ Phase 1: Plumbing works; `./build` passes; 10-battle tests on fixed roster run clean
- ✅ Phase 2: BC checkpoint produced; accuracy measured
- ✅ Phase 3: Policy checkpoint produced; stable training curve
- ✅ Phase 4: Bot evals run; gate passed (≥10% on both opponents)
- ✅ Phase 5: Ladder baseline established on gen1ou; at least n≥100 games logged

### Risks and Mitigations

**Risk 1: Roster is too weak.** Mitigated by Phase 0 curation (use proven
competitive teams) — *before* the run, not after. ~~If Phase 4 gate fails, try an
alternative roster.~~ **Withdrawn 2026-08-05:** re-rolling the roster after
seeing the gate is sweep-picking. One roster, pre-registered, one result.

**Risk 2: BC accuracy drops on gen1ou corpus.** Mitigated by using a mixed
corpus (tournament games); if still weak, check corpus for parsing/encoding
errors.

**Risk 3: Ladder is much slower on gen1ou.** Expected (~4 min/game vs ~3 for
randbats). Phase 5 will take 24–48h. Plan accordingly.

**Risk 4: Performance degrades more than expected post-pivot.** This is a
possible finding, not a bug. Record it and proceed to assess whether observation
enrichment helps (M11-deferred).

### What M12 delivers

- A clean on-format number for Gen 1 OU with team-luck variance removed — an
  answer to "does shrinking the problem help?", which is the last open question
  this project intends to ask
- ~~A platform for future improvements~~ — **withdrawn 2026-08-05.** There is no
  planned work after M12. If the number is interesting, that is a reason to start
  something new, not to reopen this

---

## M11 (Deferred): Observation Enrichment — v4 Schema ❌ CLOSED — NOT PURSUED

> **Numbering note.** M11 is the *observation enrichment* milestone, scoped
> 2026-08-01. Its **Phase 0 (reward asymmetry) is DONE and shipped**; its
> Phase 1 (schema v4) is what is deferred here. The **pivot is M12** — when a
> doc says "Phase 2" or similar, check which milestone owns it.

**Status:** ❌ **Closed 2026-08-05 without execution.** Deferred 2026-08-03
pending M12; the bounded finish ends the project at M12 Phase 4, so the deferral
becomes a closure. Its **Phase 0 (reward asymmetry) shipped** 2026-08-01 and
carries into M12 — only the v4 schema work is closed.

**This is the project's best untested idea, and it is being left on the table
deliberately.** The evidence behind it is real: the observation encodes no move
identity (Recover and Swords Dance are byte-identical), no species or stats, and
type effectiveness in the attacking direction only. If anyone resumes this work,
**start here** — the rationale below still holds, and `docs/WHERE-WE-ARE.md` →
"What the observation poverty looks like" has the full diagnosis.

**Rationale for the original deferral:** The observation schema v4 design (move identity,
species, base stats, trapping, defensive type effectiveness) was scoped for
gen1randombattle with 151 possible species and dynamic rosters. Fixed-team Gen 1
OU changes the calculus drastically:

- **Species identity:** Fixed roster of 6 known species instead of 151 unknowns
  → move from a learned embedding (0–151 LUT) to compile-time constants. The
  value proposition of adding species_id vanishes; the real cost becomes *why
  each species is chosen* (matchup reasoning with base stats), which is
  different from identification.
- **Base stats:** In randbats, unknown active stats require damage estimation from
  an opaque bulk figure (HP × Def / Spe ratio). In OU with a fixed roster,
  active stats are determined by species alone and are static — a simple lookup.
- **Team roster synergy:** Randbats schema did not encode team synergy (6 unknown
  species at battle start). With a fixed roster, synergy is implicit (the same
  6 always, every battle). Schema should encode *matchup trees* (e.g., "which of
  my 6 beat their active?") instead of per-Pokémon properties.

**Consequence:** The v4 design should be **re-derived post-M11 from first
principles**, not ported as-is. The design work in the old v4 spec (move identity,
defensive type effectiveness, recharge tracking) contains real insights about
observation poverty; those findings carry forward. But the dimensionality,
encoding, and priority of each feature should be re-evaluated for fixed-team
play.

**If revisited (not planned):** The entry point would be a gen1ou ladder
baseline (M12 Phase 5) plus M10-style log analysis on OU logs, feeding a v4-OU
schema derived from first principles — **not** the v4-randbats spec ported over.

**Instrumentation debt — dropped 2026-08-05.** The `--seed` plumbing, per-step
logging and `meta.json` run manifest were "cheap insurance for all future A/Bs."
With M12 as the last milestone there are no future A/Bs to insure, and M12's
phases are single-arm. Skip it. (Listed here rather than silently deleted, since
it appears in `IN-PROGRESS.md` history.)

