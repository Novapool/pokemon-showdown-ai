# Pokemon Showdown AI Training — Milestones

Goal: Build a genuinely intelligent Pokemon trainer AI.

---

## 🏁 PROJECT STATUS: ML RESEARCH CLOSED — REPRODUCIBILITY & OPS PHASE OPEN (2026-09-24)

**ML research is closed.** The bounded finish decided 2026-08-05 ran to
completion: M12's terminal bot-eval gate passed on 2026-08-06, and M12 was
closed on 2026-09-24 with its optional Phase 5 ladder result unrecovered (the
home box is offline; no number is claimed). The agent is mediocre — 77.7% vs
Random as a raw greedy policy on randbats, 19.3% of contested ladder games — and
that is the recorded answer.

**Reopened for one bounded, non-ML milestone: M13, experiment tracking + CI.**
MLflow tracking in the training and eval entry points, the instrumentation debt
(`--seed`, per-step metrics, run metadata), and a GitHub Actions eval-smoke gate.

**🚫 Standing rule: do not add ML scope.** No new models, training arms,
hypotheses, observation-schema work, or "one more idea". M13 may *re-evaluate
existing checkpoints* and run training only as a logging smoke test; it may not
train a candidate or test a hypothesis. If M13 surfaces something interesting
about the agent, record it as a note — it is a reason to start a new project,
not to reopen this one.

| Milestone | Disposition |
|---|---|
| **M13 (tracking + CI)** | ⏳ **OPEN** — the only live work |
| M12 (fixed-team Gen 1 OU) | ✅ Closed 2026-09-24. Gate passed; Phase 5 unrecovered |
| M11 Phase 1 (obs schema v4) | ❌ Closed, not pursued — the best untested idea, left on the table |
| M11 eval battery (h128/h512) | ⚪ Checkpoints home-box only; eligible for M13 re-evaluation if it comes back, not required |
| M10 (battle log analysis) | ❌ Closed, not pursued |

**What the project actually produced.** The agent is mediocre; the experimental
method is not. Pre-registered gates, recorded sample-size widenings, a
dead-ends table, four self-issued corrections invalidating the project's own
earlier numbers, and confounds caught in M4 and M8 by its own review. That is
the durable output, written down in `docs/EVALUATION-METHODOLOGY.md`. M13 makes
it mechanical: tracked runs and a CI gate instead of disciplined note-taking.

**Current architecture:** human-replay BC → anchored PPO fine-tune on an MLP
policy/value net with structured per-Pokémon token observations (v3 schema) →
determinized MCTS at decision time. Transformer retired (M3.2).

| File | Read when |
|---|---|
| `docs/WHERE-WE-ARE.md` | Session start — one screen, plain language |
| **This file** | Planning the next milestone; checking what a past one concluded |
| `docs/MILESTONES-ARCHIVE.md` | You need M0–M12 in full: original plans, build phasing, file manifests, complete results tables |
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

## Results Ledger (M0–M12, all closed)

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

### M10: Battle Log Analysis ❌ CLOSED, NOT PURSUED (2026-08-05)
~8 days for a diagnosis nothing downstream would act on. Its one throwaway probe
fired: **wasted-turn double switches 15.7% in losses vs 6.1% in wins (+9.6pp
[+5.9, +12.5])**, climbing with game length. A lead, not a finding — no human
baseline, and causation may run the other way. Full plan in the archive.

### M11: Observation Enrichment ❌ PHASE 1 CLOSED, NOT PURSUED (2026-08-05)
**Phase 0 shipped (2026-08-01):** `applyStallPenalty` now charges the duration
cost equally to wins and losses (`--stall-penalty`, default 0.001). **Phase 1
(schema v4 — move identity, species/stats, defensive type effectiveness,
trapping) is the project's best untested idea and was left on the table
deliberately;** under a fixed roster it must be re-derived, not ported. Two
width arms (`m11_h128`, `m11_h512`) were trained 2026-08-02 on the home box and
**never evaluated**. Full text in the archive.

### M12: Fixed-Team Gen 1 OU Pivot ✅ GATE PASSED; Phase 5 unrecovered (closed 2026-09-24)
Fixed roster Tauros / Chansey / Snorlax / Exeggutor / Starmie / Alakazam
(rank-#1 exact team in 10,101 replays ≥1300; `config/rosters/gen1ou-standard.txt`),
same team both seats. Plumbing found a real bug: `Teams.getGenerator('gen1ou')`
silently falls back to the random generator, so MCTS would have searched against
an impossible bench. BC (M7 recipe) reproduced 53.1% / 55.1% val acc; PPO 5M
steps stable. **Terminal gate, raw sampled policy, n=5,000/opponent: 95.9% vs
Random [95.3, 96.4], 92.2% vs DamageFirst [91.4, 92.9]** (gate ≥10%). Read as
"no catastrophic regression from the pivot" — **not comparable to M7's 69.7% on
randbats** (mirror roster removes team-luck variance). The 0-draw count is
unconfirmed (draw branch never observed to fire). Seat bias measured for the
first time: none (p1 49.2%, n=600).

**Phase 5 (gen1ou ladder) — UNRECOVERED, no number claimed.** Launched on the
home box 2026-08-06 (tmux `m12ladder`, `--run-id m12-ladder`, n=356, greedy).
At close the home box was offline and no `m12-ladder` rows exist on the Mac, so
whether it finished is unknown. If the home box comes back:
`rsync` `data/replays/self_ladder/ladder_results.csv` → `python3
scripts/ladder_analysis.py --run m12-ladder` (and `--min-decisions 16` for the
contested figure), reported as a **standalone** number — it has no concurrent
control and does not compare to any randbats ladder figure.

---

## What has been closed

Six directly-tested hypotheses, all null or weak: richer observations (M3.4,
M7, and M8-1A — **but see correction 3; the M8-1A closure is confounded, so
this direction is weaker evidence than the count suggests**), value-head
targeting (M8-2, replicated), more data (M5.5's ceiling), unrated tournament
data, format alignment (small but real), and sparring partners. Plus one
architecture retired (transformer, M3.2) and one measurement regime rebuilt
(M9).

**Untested when research closed (not disproven — for a future project, not
this one):** observation *poverty* of a specific kind — move identity,
species/base stats, trapping, and **defensive** type effectiveness (v3 encodes
only the attacking direction) — which was M11's thesis; model capacity, which
M9 Phase 2d pointed at; and tactical blunders, where M10's probe left a lead.

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

## M13: Experiment Tracking + CI ⏳ OPEN (opened 2026-09-24)

**Scope — ops only.** Makes the evaluation discipline mechanical. No modeling:
see the standing rule in PROJECT STATUS.

1. **MLflow tracking** in the training (`models/ppo/train.py`,
   `models/bc_pretrain_mlp.py`) and eval (`models/evaluate.py`,
   `scripts/bot_eval_ab.py`) entry points: params, per-step metrics, git SHA,
   seed, checkpoint path + hash. One authoritative store on the Mac; home-box
   runs reach it over a reverse SSH tunnel.
2. **Instrumentation debt** (dropped 2026-08-05, reinstated here): `--seed`,
   per-step metrics (entropy, value loss, KL — currently computed and
   discarded), a run-metadata file next to checkpoints.
3. **`.github/workflows/eval-smoke.yml`**: CPU torch, the tracked M7
   checkpoint, raw policy vs Random, on every push/PR. Upstream `test.yml` is
   left alone.

**Gate (pre-registered 2026-09-24, before implementation):**
- **(a)** An MLflow tracking store holding **≥10 real runs**, each with params,
  metrics, git SHA, and checkpoint path. Re-evaluating existing tracked
  checkpoints counts; typing historical numbers from the ledger in as runs does
  **not**.
- **(b)** A GitHub Actions workflow that runs a bot-eval smoke test on every
  push/PR, with a pass threshold **pre-registered in
  `docs/EVALUATION-METHODOLOGY.md`**, derived from the measured M7 or M12
  baselines and a sample size CI can afford.
- **(c)** That workflow has **failed on a real regression and passed after the
  fix**, visible in Actions history. No manufactured failure: either CI catches
  a genuine regression during this work, or a documented, already-fixed bug from
  this repo is re-introduced on a PR and recorded explicitly as a **backtest**.

**Constraints known at open:** the home box is offline, so the ≥10 runs come
from Mac-local checkpoints (M7, `m9seed`, `m9p2c`, `m9p2d`, `v3_valft` — all
tracked in git); M12 and the M11 width arms are added only if it returns.
Upstream `test.yml` fails on every push (`npm ci`: `package-lock.json` out of
sync) — a pre-existing red check, not this milestone's.
