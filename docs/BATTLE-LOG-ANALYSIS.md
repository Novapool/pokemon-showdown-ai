# Battle Log Analysis Plan — Tactical Error Diagnosis

**Objective:** Identify specific, categorical tactical mistakes the agent makes against human opponents that drive the 93% (vs bots) → 30.5% (vs humans) performance gap.

**Constraint:** Automated, aggregate, statistical analysis only. No manual replay review.

**Deliverable:** A quantitative mapping of "reasons we lose" with confidence intervals, pre-registered interpretation, and actionable findings.

Last updated: 2026-08-01 (Plan, not executed)

---

## Part 1: Data Inventory

### 1.1 Agent ladder games — on disk, sufficient

**Location:** `/data/replays/self_ladder/`

**Volume:**
- `ladder_results.csv`: 517 rows (header + 516 battles, 2026-07-16 to 2026-07-31, M7 checkpoint only)
- Full battle logs: 516 gzipped battle protocol files (one per rated game)
- CSV schema: `timestamp,room,opponent,rated,result,decisions,max_latency_ms`
  - **Note:** M9-era runs will have richer schema with `run_id`, `account`, `checkpoint`, `opp_rating`, `own_rating`, but these pre-M9 rows use the legacy schema. Columns `run_id` and `checkpoint` will be derived from file metadata and the known M7 checkpoint.

**Data quality for analysis:** ✅ **SUFFICIENT**
- Each gzipped log contains the full battle protocol (confirmed via sample inspection)
- Protocol includes: team rosters, available moves at each decision point, the move the agent chose, and game state after
- Winner is recorded in both CSV and protocol (`|win|` line)
- Opponent identity and starting Elo are present in the protocol (`|player|` lines)

**Critical caveat:** The CSV does not yet have `opp_rating` / `own_rating` fields (M9 schema), so opponent strength must be extracted from the gzipped logs. This adds complexity but is fully doable.

**Gating decision:** Proceed. The data is sufficient for all planned metrics.

---

### 1.2 Human replay reference corpus — on home box, will need sync

**Location (authoritative):** `homebox:~/Projects/pokemon-showdown-ai/data/replays/gen1ou/` and `gen1randombattle/`

**Volume:**
- gen1ou: ~98,983 games (mixed rating tiers, unfiltered; contains ~10.7k high-rated and ~88k lower-rated)
- gen1randombattle: ~21,583 games (all ≥1300 rating, scraped with filter)
- Total: ~120k human replays

**Status on Mac:** Partially present (verified 07-31 note: "both extended on home box"). May not be fully synced. **Action required before analysis:** rsync down the human corpus or confirm sync status via homebox inspection.

**Data quality for analysis:** ✅ **SUFFICIENT** (once synced)
- Same protocol format as agent logs
- Contains full team rosters, move choices at each turn, and game outcomes
- Allows computation of human-play statistics (move selection frequencies, win rates by situation)

**Gating decision:** Do not block the plan on this. If the home-box corpus is not synced by the time detailed metric work begins, the analysis can proceed using only agent-vs-agent comparisons (agent wins vs agent losses) and scale to human comparisons in phase 2. However, human comparison is the most informative comparison (humans beat the agent; their patterns matter).

---

### 1.3 Summary: what can and cannot be analyzed

| Analysis | Data available? | Notes |
|---|---|---|
| Agent move selection in losses vs wins | ✅ Full logs on disk | 516 ladder battles |
| Agent tactical errors (e.g., sleep clause violation) | ✅ Full logs on disk | All info present to detect |
| Comparison: agent vs human move patterns | ⚠️ Partial | Requires home-box sync |
| Opponent-strength breakdown (strong vs weak opponents) | ⚠️ Partial | Elo in logs, but not yet in CSV; needs extraction |
| Checkpoint variation (was M7 noisy?) | ❌ Not applicable | Only M7 in ladder corpus |

**Decision rule:** Begin metric development and pilot analysis on agent ladder games (losses vs wins) immediately. Add human comparison layer once homebox sync is verified.

---

## Part 2: Metric Catalog — Tactical Errors to Measure

### Prioritization framework

For each candidate metric, rate on:
1. **Detectability** — how reliably can we find it in the logs? (1=hard, 5=trivial)
2. **Expected frequency** — what's a reasonable occurrence rate? (rough %)
3. **Win-impact** — plausibly how much win rate do we lose per instance? (rough points)
4. **Ease of implementation** — lines of parser code needed? (1=trivial, 5=major)

**Rank by:** frequency × win-impact / (implementation effort + analysis complexity)

### 2.1 Tier 1: High-priority, Gen 1 fundamental rules

#### 1.1 — Sleep clause violation
**Operational definition:** Agent attempts to apply sleep (via sleep-inducing move like Hypnosis, Leech Seed, etc.) when the opponent already has a Pokemon asleep on the field. This is illegal under Gen 1 Sleep Clause Mod.

**Detectability:** 5 (rules engine can track status; marked in protocol with `|-status|`)
**Expected frequency:** 0.5–2% (depends on how often agent faces sleep-vulnerable boards)
**Win-impact:** ~1–3pp (wasted turn, does nothing, but opponent still gets their turn)
**Implementation ease:** 5 (rule is unambiguous in the protocol)

**Specific metric:** 
- Count: agent attempts to sleep opponent when opponent already asleep
- Rate: (attempts in losses) vs (attempts in wins), with 95% CI on difference
- Control: human occurrence of same in our replay corpus

**Why it matters:** Sleep Clause is the most common hard rule in Gen 1. Violating it is strictly bad play.

---

#### 1.2 — Freeze clause violation  
**Operational definition:** Agent attempts to freeze the opponent when the opponent already has a frozen Pokemon.

**Detectability:** 5 (same protocol marker, `|-status|`)
**Expected frequency:** 0.2–1% (fewer moves induce freeze than sleep; less common)
**Win-impact:** ~1–2pp (wasted turn)
**Implementation ease:** 5

**Metric:** (Freeze clause violations in losses) vs (in wins)

**Note:** Less common than sleep but equally clear-cut.

---

#### 1.3 — High-damage move when priority move secures KO
**Operational definition:** Agent uses a high-damage non-priority move when a priority move (Quick Attack, Mach Punch, etc.) would have fainter the opponent's active Pokemon (calculated via damage calcs on current HP and Def).

**Detectability:** 3 (requires damage calculator; must parse all opp HP and atk stats; speed comparison)
**Expected frequency:** 2–8% (depends on whether agent has priority + target is low-HP)
**Win-impact:** ~2–5pp (missing secured KO leaves opponent alive; potential swing)
**Implementation ease:** 2 (needs damage calculator integration or approximation)

**Metric:** Rate of "secured KO available but not taken" in losses vs wins

**Why it matters:** In a format where speed and priority matter (Gen 1, random teams), missing guaranteed damage is a skill leak.

**Caveat:** Requires an accurate damage calculator. The codebase has `sim/dex.ts` for move/ability data; damage calculation would need to be added or the dex consulted. Alternatively, use a heuristic (e.g., "move was predicted to deal >50% and Pokemon was low-HP").

---

#### 1.4 — Hyper Beam recharge mismanagement
**Operational definition:** Agent uses Hyper Beam (or similar recharge-required move) when:
  - (a) the opponent's active Pokemon would survive the damage, OR  
  - (b) the agent's active Pokemon is low-HP and would faint before the turn after recharge

**Detectability:** 3 (HP tracking, recharge status tracking, damage calc)
**Expected frequency:** 0.5–2% (recharge moves are not common; agent would need to pick them)
**Win-impact:** ~2–4pp (recharge turn is a dead turn; opponent gets a free turn to switch or heal)
**Implementation ease:** 2

**Metric:** Rate of "recharge moves used in bad board states" in losses vs wins

**Why it matters:** Hyper Beam is a trap move for overconfident players; humans avoid it in uncertain states.

---

### 2.2 Tier 2: Intermediate priority, move selection patterns

#### 2.1 — Overkill (more damage than needed to KO)
**Operational definition:** Agent chose a move that would deal >110% of target's current HP when a weaker move dealing ≥100% was available.

**Detectability:** 3 (damage calc)
**Expected frequency:** 3–12% (high-damage moves are tempting even when overkill)
**Win-impact:** 0–1pp (not strictly wrong, but misses the point of "best move"; true cost is opportunity cost)
**Implementation ease:** 2 (damage calc again)

**Metric:** "Overkill rate" in losses vs wins

**Why it matters:** Pattern recognition: does the agent overshoot when it should diversify or switch?

---

#### 2.2 — Attacking into known resistor/immunity
**Operational definition:** Agent attacks a Pokemon with a move it knows (from protocol history or team preview) cannot damage it (e.g., Normal-type move vs Ghost).

**Detectability:** 4 (type chart + Dex; no damage calc needed)
**Expected frequency:** 1–4% (depends on team composition and agent awareness)
**Win-impact:** ~1–3pp (wasted turn, but not as bad as sleep clause since the *attempt* at least succeeded)
**Implementation ease:** 3 (type matching against Dex)

**Metric:** Rate of "attacking move type not super-effective or at least neutral" in losses vs wins

**Why it matters:** A sign the agent is not properly tracking type advantage — fundamental to Pokemon play.

---

#### 2.3 — Attacking into status-immune types
**Operational definition:** Agent attempts to apply status (burn, poison, etc.) to a Pokemon type that is immune (e.g., poison move on Poison-type, burn on Fire-type).

**Detectability:** 4 (type immunity rules)
**Expected frequency:** 0.5–2% (agent would need to pick a status move, which is rare)
**Win-impact:** ~1–2pp (wasted move)
**Implementation ease:** 3

**Metric:** Count of "status applied to immune type" in losses vs wins

---

#### 2.4 — Switch into predictable hard counter
**Operational definition:** Agent switches its active Pokemon to a Pokemon that:
  - (a) was weak to the opponent's active Pokemon's moves, AND
  - (b) the agent could have predicted this matchup (e.g., opponent's Pokemon and movepool are known)

**Detectability:** 2 (requires game tree tracking; type advantage calc; but protocol has full visibility into opponent's team)
**Expected frequency:** 2–8% (switching strategy is complex; easy to get wrong in random teams)
**Win-impact:** ~2–4pp (bad switch can cost a Pokemon)
**Implementation ease:** 1 (type matching only)

**Metric:** "Switch into weakness" rate in losses vs wins

**Why it matters:** Switching is the central skill in Gen 1. Misplaying switches is a major source of losses.

---

### 2.3 Tier 3: Exploratory, higher-level patterns

#### 3.1 — Fail to switch when active Pokemon has clear disadvantage
**Operational definition:** Agent's active Pokemon is:
  - Weak to opponent's moves, AND
  - Health is dropping, AND
  - A switch to a better matchup was available
  
But agent chose to keep attacking instead of switching. Measured as "did not switch when a better bench Pokemon was available."

**Detectability:** 2 (game tree, type advantage)
**Expected frequency:** 3–10% (complex heuristic; high variance by game state)
**Win-impact:** ~2–6pp (bad switches can decide games)
**Implementation ease:** 1 (type matching)

**Metric:** "Missed opportunity to switch out" in losses vs wins

**Caveat:** This metric is high-level and has confounds (sometimes staying in is correct). Use as exploratory only.

---

#### 3.2 — Move distribution anomaly
**Operational definition:** Agent's move selection frequencies deviate significantly from:
  - (a) Human move frequencies in similar board states, OR  
  - (b) Agent's own frequencies in winning games

**Detectability:** 2 (no individual-move detection needed; aggregate comparison)
**Expected frequency:** N/A (aggregate metric)
**Win-impact:** N/A (this is exploratory pattern detection, not a specific error)
**Implementation ease:** 1 (just build histograms)

**Metric:** Chi-square test of agent's move distribution in losses vs wins; compare to human dist

**Why it matters:** If we can't find specific errors, an aggregate distribution shift might point to which decisions are systematically different.

---

### 2.4 Summary: Recommended Tier 1 starting set

| Metric | Priority | Implementation cost |
|---|---|---|
| Sleep clause violation | High | 1 day |
| Freeze clause violation | High | 1 day |
| High-damage when priority secures KO | Medium | 2 days |
| Attack into immunity/resistance | Medium | 1 day |
| Switch into weakness | Medium | 1 day |
| Fail to switch out | Medium | 1 day |

**Effort estimate for Tier 1:** ~5–7 days to develop all metrics (including damage calculator stub, game-state tracker, output pipeline).

**Tier 2 (exploratory):** +2–3 days if Tier 1 is inconclusive.

---

## Part 3: Comparison Design — Baselines and Confounds

### 3.1 Required comparisons

For each metric, the minimum comparison set is:

1. **Agent losses vs agent wins** (does the error occur more often in losses?)
   - Sample: n_losses = ~155 (30% of 516), n_wins = ~361
   - Statistical test: two-proportion z-test with Wilson CIs
   - Interpretation: if an error is genuinely associated with losses, expect it to be significantly higher in loss games

2. **Agent vs human reference** (do humans make this error at different rates?)
   - Sample: extract same metric from human replay corpus
   - Requires: human corpus sync + same parser run on both datasets
   - Statistical test: difference of proportions with 95% CI
   - Interpretation: humans' error rate sets expectations. If agent error rate > human rate, it's actionable.

3. **Opponent strength stratification** (does error rate vary by opponent level?)
   - Stratify ladder games by opponent starting Elo (low <1100, med 1100–1150, high >1150)
   - Compute error rate per stratum
   - Interpretation: if error rate is constant across opponent strength, error is not "playing down to opponent level." If higher against stronger opponents, agent may be tilting or out-of-distribution.

### 3.2 Confounds to track

**Confound 1: Selection bias in available moves**
- Some errors can only occur if the agent had the relevant move in its team
- Example: "sleep clause violation" can only occur if the agent's team includes a sleep move
- **Mitigation:** Report numerator (violations) and denominator (games where sleep move was available separately). Report rate as "violations per game opportunity."

**Confound 2: Opponent passivity**
- A bad move may not cost a game if the opponent is also playing suboptimally
- Example: attacking a Pokémon you shouldn't might not matter if opponent doesn't punish it
- **Mitigation:** Compare against human play on the *same board states* if possible (requires game-state matching, not in scope for Phase 1). For now, just track the frequency; interpretation comes later.

**Confound 3: Recharge move frequency**
- If agent rarely uses recharge moves, low rate of "recharge mismanagement" doesn't mean the error is fixed — it means the move doesn't appear in losses
- **Mitigation:** Always report denominator (games with recharge move) separately.

---

## Part 4: Implementation Plan — Scripts and Architecture

### 4.1 Phased development

#### Phase 1: Parser & data extraction (2 days)

**Output:** A Python library that reads the gzipped battle logs and extracts structured game-state traces.

**Module: `models/battle_log_parser.py`**
- Input: path to a gzipped battle log
- Output: a Python object representing the full game tree
  - Teams for both players (species, moves, stats)
  - Per-turn snapshot: (active Pokemon, current HP, status, available moves, chosen move, damage dealt, game state after)
  - Final outcome (win/loss/tie)
  - Metadata: opponent username, opponent starting Elo, timestamp

**Dependencies:** 
- `gzip`, `json`, `re` (standard library)
- No external dependencies initially (unless we decide to integrate a damage calculator)

**Acceptance criteria:**
- Correctly parses 10 sample ladder logs
- Correctly identifies chosen move for the agent (the player marked as `p1` in most logs)
- Correctly extracts opponent metadata

#### Phase 2: Rule checkers & metrics (3 days)

**Modules: `models/battle_log_analysis.py`**

Submodules for each error class:

- `class SleepClauseChecker`: tracks sleep status, flags violations
- `class FreezeClauseChecker`: tracks freeze status, flags violations
- `class KOOpportunityChecker`: needs damage calc or heuristic
- `class TypeMatchupChecker`: type chart lookup + resistance checking
- `class SwitchAnalyzer`: tracks matchups before/after switch

**Output per metric:** 
- Counts in wins vs losses
- Counts by opponent Elo stratum
- Rates with 95% CI (using existing `scripts/bot_eval_ab.py` Wilson interval code)

**Key class: `BattleAnalyzer`**
- Wraps a parser output
- Applies all rule checkers in sequence
- Returns a summary dict: `{ metric_name: { count_total, count_wins, count_losses, ci_diff, ... }, ... }`

**Acceptance criteria:**
- Runs on all 516 ladder logs in <5 min
- Produces a CSV output (one row per metric, columns: metric, n_losses, rate_losses, ci_losses, n_wins, rate_wins, ci_wins, difference_ci)

#### Phase 3: Human baseline comparison (2 days)

**Prerequisite:** Homebox human corpus synced to local disk.

**Module: `models/battle_log_analysis.py` extended**

- Run the same `BattleAnalyzer` on a sample of human replays (e.g., 5,000 games)
- Output: same CSV format, labeled by dataset (agent vs human)
- Post-processing: compute difference in error rates with 95% CI

**Output format:**
```
metric,dataset,n_games,n_errors,rate,ci_lower,ci_upper
sleep_clause,agent,516,8,1.55%,[0.8%, 2.8%]
sleep_clause,human,5000,15,0.30%,[0.2%, 0.5%]
sleep_clause,diff,—,—,+1.25pp,[+0.5pp, +2.2pp]
```

**Acceptance criteria:**
- Runs on human corpus without crashing
- Produces believable baseline rates (humans should be roughly ≥ agent on most errors, or have a reasonable explanation for deviation)

#### Phase 4: Pre-registered output & interpretation (1 day)

**Output: `results/battle_log_analysis_results_20260801.md`**

A markdown report with:
1. Per-metric summary (rate in agent losses vs wins vs humans, with CIs)
2. Ranked by effect size (largest rate difference first)
3. Per-metric interpretation (see Part 5 below)
4. Executive summary: "Reasons we lose" in priority order

**Acceptance criteria:**
- Report includes every metric from Tier 1 catalog
- Each metric has n, rate, CI
- Interpretation is clear and actionable

---

### 4.2 Script organization

```
models/
  battle_log_parser.py          # Parses gzipped logs → game tree
  battle_log_analysis.py        # Applies checkers, computes metrics
  battle_log_metrics.py         # Per-metric checker classes
  analyze_ladder_logs.py         # CLI driver for Phase 1–2
  analyze_human_replays.py       # CLI driver for Phase 3
  
scripts/
  run_battle_analysis.sh         # Orchestration script (if needed)
  
data/replays/self_ladder/
  ladder_results.csv            # Input (exists)
  *.log.gz                       # Input (exist)
  
results/
  battle_log_analysis_results_*.md   # Output
  battle_log_metrics.csv             # CSV summary (output)
```

### 4.3 Reuse existing infrastructure

- **Confidence interval code:** `scripts/bot_eval_ab.py` already implements Wilson intervals + Newcombe CI. Reuse this for all metric comparisons.
- **CLI pattern:** Model after `scripts/ladder_analysis.py` (argparse, CSV I/O, clear help)
- **Damage calc:** If needed, can stub with a heuristic or integrate `sim/dex.ts` via Node subprocess (complex; avoid if possible)

---

## Part 5: Pre-Registered Interpretation

**Critical rule:** Write down BEFORE analyzing what each outcome means.

### 5.1 Positive finding: "We found a reason for the gap"

**Definition:** A metric shows that the agent makes error E at rate R_agent in losses, significantly higher than either:
- (a) R_agent in wins, OR
- (b) R_human in the same error

**Criteria:**
- Rate difference has 95% CI that excludes 0
- Effect size is plausibly large enough to matter (e.g., ≥0.5pp win-rate cost)

**Interpretation:** Error E is a measurable driver of losses. Candidate for intervention.

**Action:**
1. Document error in `docs/WHERE-WE-ARE.md` → add to "Tactical errors catalog" section
2. Assess fix difficulty (e.g., "teach the value head to penalize sleep-clause violations")
3. Recommend whether to attempt a fix or accept as noise
4. If fixing: design an experiment (e.g., reward shaping, data augmentation)

**Example:** "Sleep clause violations occur in 3.2% of losses vs 0.8% of wins (diff +2.4pp, 95% CI [+1.1, +3.7]) and humans violate it only 0.1% of the time. This suggests the agent is not properly tracking status state or is risk-blind to it."

### 5.2 Null finding: "Error rate is indistinguishable"

**Definition:** A metric shows no significant difference between:
- Error rate in losses vs wins, OR
- Agent error rate vs human error rate

**Criteria:** 95% CI for the difference includes 0, OR absolute difference is <0.5pp (below noise floor)

**Interpretation:** Error E is not driving losses. It occurs at baseline rate even when the agent wins.

**Action:** Record that error E was checked and found to be background noise. Do not pursue.

**Rationale:** This is a *finding*, not a failure. It narrows down what matters.

### 5.3 Confounded finding: "We can't tell"

**Definition:** Sample size is too small, or the error cannot occur in enough games to form a reliable rate.

**Example:** Recharge-move mismanagement occurs 0 times in the 516 games (agent doesn't have Hyper Beam in the random team subset).

**Interpretation:** The error may exist in principle but is not testable on this dataset.

**Action:** Flag for Phase 2 (larger sample or focused team composition). Do not draw conclusions.

---

### 5.4 Interpretation rules (project convention)

1. **Multiple comparisons correction:** This analysis tests ~10 metrics. Do not gate individual findings on p<0.05; instead report 95% CIs and let effect size speak. CIs that exclude 0 are highlighted; others are marked "inconclusive."

2. **Rate difference interpretation:** A 1pp difference in error rate is not automatically actionable. If agent makes sleep-clause violations 1% of the time and humans 0.8%, is that materially different? Only if the CI excludes 0 AND the absolute rate is not background noise. Use effect size judgment.

3. **Comparison against humans:** Human rates are reference, not gospel. If agent error rate > human, the gap is actionable. If agent error rate ≤ human, the agent is at least human-level at that error (the problem lies elsewhere).

4. **Negative result:** If all Tier 1 metrics are null/inconclusive, move to Tier 2 (higher-level patterns, exploratory). If Tier 2 is also null, the gap likely lies in:
   - Team composition luck (randomness in team draw, not skill)
   - Observation state (agent not seeing information it should)
   - Value function (agent knows the right move but evaluates it wrongly)
   - Format misalignment (random teams are fundamentally different from bot-eval format)

   Document this and prioritize investigating one of these higher-level hypotheses.

---

## Part 6: Honest Risk Assessment

### 6.1 What could go wrong

**Risk 1: No actionable errors found** (probability: ~40%)

All Tier 1 metrics come back null/inconclusive. Interpretation: the gap is not driven by rule-based tactical blunders, but by:
- Observation richness (agent doesn't see what humans see)
- Value calibration (agent sees moves but evaluates them wrong)
- Team luck (random team draws favor humans at this sample size)
- Format difference (random battles are categorically different from self-play)

**Mitigation:** This is still valuable — it narrows down what to investigate next. The project already tested observations (M8 Phase 1A, null) and value (M8 Phase 2, null). This analysis might add evidence to support investigating luck or observation-state mismatch more deeply.

**Risk 2: Found errors are too small to fix** (probability: ~25%)

We find that the agent violates sleep clause 2% of the time vs humans 0.2%, BUT: this only accounts for ~0.3pp of the 40pp gap (assuming each violation costs 1–2 turns). Too small to move the needle.

**Mitigation:** Multiple small errors can add up. If Tier 1 finds 5 errors averaging 0.5pp each, the total is 2.5pp — not large but measurable. Helps inform whether to invest in fixes.

**Risk 3: Confounds invalidate comparisons** (probability: ~15%)

Example: Agent's games are against a different opponent strength distribution than humans (agent fought stronger players, humans played weaker). Error rates shift with opponent strength, confounding the finding.

**Mitigation:** We control for this by stratifying by opponent Elo. The worst case is the comparison is invalid; in that case, report it and note that a properly matched game set would be needed for conclusive evidence.

**Risk 4: Data on disk is insufficient** (probability: ~5%, if homebox sync fails)

Human corpus is not synced, or ladder logs are corrupted, or parser fails. Analysis can proceed with agent-only metrics but loses the human baseline.

**Mitigation:** Begin with agent-loss vs agent-win comparison (fully doable on 516 games, no external data). Human baseline is secondary, not gating.

---

### 6.2 Success bar

**Minimum success:** Find at least 1 metric where:
- Agent error rate in losses > agent error rate in wins (95% CI excludes 0), OR
- Agent error rate > human error rate (95% CI excludes 0)

**Realistic success:** Find 2–3 such metrics, explaining 1–5pp of the gap collectively.

**Ambitious success:** Find a pattern that explains 5–10pp and points to a specific intervention (e.g., "agent doesn't track sleep status; fix X in the observation layer").

---

## Part 7: Deliverables Checklist

### For Phase 1 (Parser & extraction)
- [ ] `models/battle_log_parser.py` — reads gzipped logs, extracts game trees
- [ ] Sample parsing test (10 logs, all parsed correctly)
- [ ] Documentation of output schema

### For Phase 2 (Metrics)
- [ ] `models/battle_log_analysis.py` — metric checkers
- [ ] `models/analyze_ladder_logs.py` — CLI driver
- [ ] CSV output: `results/battle_log_metrics.csv` (one row per metric; n_losses, rate_losses, ci_losses, n_wins, rate_wins, ci_wins, diff_ci)
- [ ] Runtime: all 516 games analyzed in <5 min

### For Phase 3 (Human baseline)
- [ ] Human corpus synced to local (or confirmed status)
- [ ] `models/analyze_human_replays.py` — runs same metrics on human set
- [ ] Merged CSV: agent + human rates side by side
- [ ] Comparison report with effect sizes

### For Phase 4 (Interpretation & reporting)
- [ ] `results/battle_log_analysis_results_20260801.md` — full report with interpretation
- [ ] `docs/WHERE-WE-ARE.md` — updated with findings (if any) + recommendations
- [ ] `IN-PROGRESS.md` — updated with results and next steps

---

## Part 8: Timeline & Effort

**Phase 1:** 2 days (parser + 10-log validation)
**Phase 2:** 3 days (Tier 1 checkers + CSV output)
**Phase 3:** 2 days (human baseline, if corpus is synced)
**Phase 4:** 1 day (reporting + interpretation)

**Total:** ~8 days if human corpus is synced; ~5 days if agent-only.

**Parallelization:** Phase 1 and 2 can overlap (start writing checkers while parser is being finalized).

---

## Reference & Background

- `docs/EVALUATION-METHODOLOGY.md` — how to report numbers with confidence intervals
- `docs/DATA-FORMATS.md` — Section 1 covers the full battle protocol schema
- `scripts/ladder_analysis.py`, `scripts/bot_eval_ab.py` — existing CI infrastructure to reuse
- `docs/WHERE-WE-ARE.md` — context on why this investigation matters
- `sim/dex.ts` — Pokédex for type lookups and move data
