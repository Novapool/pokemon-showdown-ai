# Where We Are

**Plain-language orientation. Start here after a break, or when the other docs
assume too much.** No timeline and no history — for those, `IN-PROGRESS.md` has
the running log, `MILESTONES.md` has the results ledger plus the live plan,
and `docs/MILESTONES-ARCHIVE.md` has M0–M9 in full.

**Keep this current.** Update it whenever a milestone or experiment closes, and
prune anything that stops being true. It should stay roughly one screen.

Last updated: **2026-08-06**

---

## 🏁 Read this first: the project is finishing

**Decided 2026-08-05.** This is a deliberate, bounded wind-down — ending on an
answer rather than fading out mid-blocker.

**Phases 0–4 are all DONE as of 2026-08-06, and the terminal gate PASSED.**
On the fixed roster the agent wins **95.9% vs Random** [95.3, 96.4] and
**92.2% vs DamageFirst** [91.4, 92.9], n=5,000 each, raw sampled policy, no
search. The gate was ≥10% on both. **Read it as "no catastrophic regression from
the format pivot" — which is exactly what it was designed to test — and not as a
strength result.** It is a mirror-roster format against bots that cannot exploit
a strong team; it is not comparable to M7's 69.7% on randbats.

**Phase 5 (the optional gen1ou ladder) is running now** — 356 games, ~24 h, in
tmux `m12ladder` on the home box. When it lands, the project is done and gets
archived. No diagnosis pass, no second roster, no "one more idea."

**Closed without execution:** M10 (tactical error diagnosis, ~8 days — a
diagnosis with nothing downstream to act on) and M11 Phase 1 (observation schema
v4 — the best untested idea in the project, left on the table on purpose).

**Why.** Eight milestones, seven tested hypotheses back null or weak, every
intervention sized at 1–5pp against a ~30pp gap, and infrastructure now eating
more of a session than the ML. The honest ceiling on M12 is "less bad in a
narrower format," not "competitive." It is worth doing anyway because fixed teams
is the one idea that shrinks the *problem* instead of adding another knob — and
because a project that concludes reads differently than one that was abandoned.

**If you are picking this up later:** start at `MILESTONES.md` → M11, the
observation-poverty work. The section below tells you why.

---

## The one-paragraph version

We have a Pokémon-battling agent that is **mediocre**. The long-standing framing
— "great against bots, bad against humans" — was largely two measurement
artifacts, both now corrected. With MCTS it wins ~93% against our scripted
opponents, but the **raw greedy policy wins 77.7% against `Random`**, an
opponent that plays random legal moves, and MCTS's true contribution is
confounded and unknown. On the ladder it wins ~27% of games and **19.3% of the
games that actually get played out** — roughly a third of our "wins" are
opponents forfeiting in the first few turns.

The best available explanation for *why* is **observation poverty**: the agent
encodes moves without identity (Recover and Swords Dance are byte-identical),
carries no species or stats, and encodes type effectiveness offensively but
never defensively. It is well-evidenced from a 2026-08-01 code review and
predicts the shape of the gap — but it is **still an untested diagnosis**, and
seven hypotheses before it were tested and came back null or weak.

**As of 2026-08-03 the project is pivoting to fixed-team Gen 1 OU** — the first
change that shrinks the problem rather than adding another knob. See "The plan"
below.

## What the observation poverty looks like

The observation encodes a move in six numbers: base power, accuracy, type,
category, PP, disabled. **There is no move identity.** So to the agent:
- Recover and Swords Dance are byte-identical (0 base power, 100% accuracy, Normal status)
- Wrap, Fire Spin, Clamp (15 BP, 85% accuracy, Normal physical) are identical to each other
- Horn Drill and Fissure (0 BP, 30%, Normal physical) *look like the worst moves in the game* — they are actually OHKO moves that decide games

**Species and stats are completely absent.** The gym sends HP as a ratio;
absolute bulk is invisible. No damage estimate is formable.

**Type effectiveness is encoded in one direction only.** v3 dims 77–80 give each
token's moves vs the *opponent's active* — the attacking direction. There is no
dim for the defending direction (the opponent's moves vs this token's typing),
so choosing a switch-in that *resists* the incoming attack requires learning the
15×15 chart unaided from two type one-hots. Watching a live ladder game
(2026-08-02) showed the predicted shape: after a faint the agent brought in a
Water type against an Electric, then burned the next turn switching it out for a
Normal — a defensively better choice reached one turn late. One game is an
anecdote, and the wasted turn isn't fully explained by this dim; but the gap is
real and is the cheapest item in the v4 schema.

**Trapping and Hyper Beam recharge are invisible.** The agent cannot learn to
punish DamageFirst spamming Hyper Beam into recharge, even though DamageFirst
does this structurally every fifth move.

This explains the shape of the gap: the agent dominates bots that never switch
(93%) and struggles against humans who switch constantly and play for position
(~27%). The sparring-partner hypothesis was tested in M9 and failed entirely
(−1.0pp vs Random, 0.0 vs DamageFirst) — **it is the agent's perception that
is the constraint, not its training opponents.**

**The 360-game run gives this its first behavioral evidence.** Win rate by game
length, over the 326 contested games:

| game length (decisions) | W | L | win rate |
|---|---:|---:|---:|
| 16–20 | 5 | 11 | 31.2% |
| 21–25 | 13 | 41 | 24.1% |
| 26–30 | 18 | 78 | 18.8% |
| **31–40** | **19** | **111** | **14.6%** |

The agent decays as games get longer. Short games are won on raw damage, which
base power alone can serve; long games are switch-and-position games, which need
exactly the information the observation omits. This is a correlation, not a
mechanism — **M10 is the milestone that turns it into one.**

## What we know for sure

Each of these is measured, with confidence intervals, at sample sizes that can
actually support the claim. That was not true of this project before M9.

- **The agent is strong vs bots, weak vs humans.** 93% vs Random with search;
  **26.9% on ladder** (n=360, CI [22.6, 31.8]), and 30.5% (n=387, CI [26.1,
  35.3]) in the earlier sampling era.
- **About a third of our ladder wins are forfeits, and every historical ladder
  number is inflated by ~7pp.** Across 747 rated games we have **never** lost one
  in under 16 decisions, but 74 of our wins came that fast — opponents quitting,
  timing out or disconnecting. They are real Elo, but they measure the ladder
  pool's quit rate, not our play. **Contested-only: 19.3% (n=326) greedy, 22.9%
  (n=341) sampling.** `ladder_analysis.py` now reports this split on every run;
  `--min-decisions 16` scores contested games only. Both eras are contaminated at
  the same rate, so no past *comparison* is invalidated — only the absolute
  levels. Quote the contested figure for play strength, the pooled one for
  ladder standing, never both for the same claim.
- **The agent gets worse as opponents get better, and it is not noise.** 34.4%
  vs Elo 1000–1099, 25.6% vs 1100–1199, 15.3% vs 1200–1299 (r=−0.143, t=−2.72,
  n=360). Its own rating never escaped ~1000–1120 across the run and hit the
  1000 floor twice. **Caveat on any A/B against pre-M9 data:** those rows carry
  no `opp_rating`, and win rate moves ~19pp across this Elo range, so an
  unmatched pool could swamp any effect we are trying to measure.
- **Playing the argmax beats sampling it — offline.** +7.8pp vs Random, +5.0pp
  vs DamageFirst, n=5,000/arm. **It did not show up on the ladder** (26.9% at
  n=360 vs 30.5% before, −3.5pp [−10.0, +3.0]; contested-only 19.3% vs 22.9%,
  the same −3.5pp) — but that run was under
  `--mcts`, where search already argmaxes, so greedy reached only the ~20% of
  decisions that bypass search. Diluted treatment, underpowered test.
- **~20% of ladder decisions never reach search.** Force-switches after a faint
  and locked states fall back to the raw policy (`ladder-bot.js:291`). Those are
  the "what do I bring in" decisions, which is where the missing defensive
  type-effectiveness dim would bite hardest.
- **Our measuring instrument is now sound.** Early ladder reads were 50–100
  games where the noise (±13pp) was larger than any effect we could produce.
  Protocol now lives in `docs/EVALUATION-METHODOLOGY.md`.
- **Training is reproducible.** Re-running the identical 5M-step recipe lands
  within **0.6pp**, even across a Mac→GPU change. Differences >~3pp are real.
- **A better imitator can be a worse learner.** Training BC on only random
  battles made a better mimic (+5.6pp) but then finished 8.3pp worse after RL.
  The mixed corpus buys exploration breadth, not just imitation accuracy.
- **There is no more Gen 1 human data to get.** The replay archive is exhausted.
- **The observation is information-poor, measured directly.** ~25 distinct
  values per decision; 128 of 1044 dims non-zero; 667 dims non-zero in <1%.

## Dead ends — don't re-propose these without new evidence

| Idea | Why it's closed |
|---|---|
| Scrape more Gen 1 replays | Archive exhausted |
| Harvest "untapped" tournament games | Already 74% of training data |
| Fix the value head (AlphaZero-style) | Tested twice, both null (−2.5pp each) |
| Train BC on target format only | Tested — better imitation, worse RL substrate |
| Spar with human-like opponents | M9: no measurable effect at all (−1.0pp) |
| Pick the best checkpoint off a sweep | Sweep peak is luck-inflated; it regressed 4.2pp |
| Speed-ratio observation feature | M8 Phase 1A: −3pp, and doubly confounded (see `docs/MILESTONES-ARCHIVE.md` → M8) |

## The plan — the bounded finish, decided 2026-08-05

**M12 only, Phases 0–4.** Fixed 6-Pokémon roster, Gen 1 OU:

| Phase | What | Status |
|---|---|---|
| **0** | Roster locked, pre-registered in `docs/BATTLE-FORMATS.md` | ✅ 2026-08-05 |
| **1** | Fixed-team plumbing: gym, evaluator, ladder-bot, MCTS determinizer | ✅ 2026-08-05 |
| **2** | BC retrain, mixed corpus, M7 recipe | ✅ 2026-08-06 — 53.1% / 55.1% |
| **3** | PPO 5M steps, M7 recipe, single arm | ✅ 2026-08-06 — stable, no collapse |
| **4** | Bot-eval gate, n=5,000/opponent, ≥10% both | ✅ 🏁 **PASSED** — 95.9% / 92.2% |
| 5 | gen1ou ladder, n=356 | ⏳ running (~24 h, tmux `m12ladder`) |

Home box SSH was restored 2026-08-06 (WSL). `data/replay_trajs` had to be
regenerated there first — it is tier-2 local-only data and never syncs.

**The roster (locked 2026-08-05):** Tauros / Chansey / Snorlax / Exeggutor /
Starmie / Alakazam — the rank-#1 most-used exact team in 10,101 local replays
rated ≥1300, with each slot's modal human move set. Packed team at
`config/rosters/gen1ou-standard.txt`; full rationale and caveats in
`docs/BATTLE-FORMATS.md`.

**One roster, pre-registered, one result.** The old plan allowed multi-roster
averaging and "try another roster if the gate fails." Both are withdrawn — after
seeing the result, re-rolling is the sweep-picking this project banned in its own
standing rules.

**Also dropped:** M10, the v4 schema, and the instrumentation debt (`--seed`,
per-step logging, `meta.json` — insurance for future A/Bs that no longer exist).
The M11 h128/h512 eval battery is optional and off the critical path: hours of
compute if SSH is back, expected null, changes no decision.

## Two things to walk in knowing

**The bundle is declared.** "Fixed team" and "Gen 1 OU" are two changes.
Fixing the team requires the format switch; the format switch does not require
fixing the team. The design **cannot attribute** results between them. This was
accepted deliberately — the outcome matters more than the attribution here — but
it is written down, because this project has twice been burned by confounds that
were not (M8 Phase 1A, the M4 search comparisons).

**The data story is a tradeoff, not an upgrade.** The pivot puts more human data
on-format (~99k gen1ou vs ~21.6k randbats) but at **lower average quality** —
the randbats corpus was scraped at ≥1300, gen1ou mostly was not, so at equal
quality bar gen1ou is the *smaller* pile (~10.7k vs ~21.6k). Do not record this
as a clean win.

**The pivot also resets the measurement baseline.** Every bot-eval number and all
885 ladder logs are randbats. gen1ou starts from scratch.

## Left on the table (closed 2026-08-05, not disproven)

**M11 Phase 1 (observation schema v4) — the best untested idea here. Start here
if you resume.** The observation genuinely is broken —
no move identity, no species or stats, type effectiveness encoded offensively
only. But a fixed roster changes the calculus: species identity drops from 151
values to 6 compile-time constants and base stats become static lookups, so much
of v4 becomes cheap or moot and the rest should be **re-derived for the new
format after M12, not ported from randbats**. The findings behind it carry
forward; the design does not. (M11 Phase 0, the reward-asymmetry fix, shipped
2026-08-01 and carries into M12 unchanged.)

**Greedy decoding.** Confirmed offline (+7.8pp vs Random, +5.0pp vs
DamageFirst), is the ladder default, and its one ladder read was flat and
diluted by `--mcts`. Leave it on and stop testing it.

**M10 (tactical error diagnosis).** Its one probe fired: wasted-turn double
switches at 15.7% in losses vs 6.1% in wins (+9.6pp, CI excludes 0), climbing
with game length. That is a **lead, not a finding** — no human baseline, and
causation could run the other way (losing positions may force defensive
shuffling). Closed because it costs ~8 days and produces a diagnosis with nothing
downstream to act on.

## Honest expectation

The agent's raw greedy policy wins **77.7% vs Random** — an opponent that plays
random legal moves — and **19.3% of contested ladder games**. That is not "great
against bots, bad against humans"; it is mediocre across the board, with the
apparent split produced by two measurement artifacts (search-vs-sampling, and
forfeit inflation) that have now been corrected.

**19% → 50% is not a feature fix.** Every intervention this project has run was
sized at 1–5pp, and seven have come back null or weak. The pivot is worth doing
because it is the first change that shrinks the *problem* rather than adding
another knob — but a realistic good outcome is a meaningfully stronger agent in a
narrower format, not a competitive one.

**And either way, M12 Phase 4 is the end.** That was decided before the roster
was chosen, precisely so the result can't move the finish line. A failed gate is
a finding, not a reason to keep going.

**What this project actually produced.** The agent is mediocre. The method is
not: pre-registered gates, sample-size widenings recorded rather than quietly
applied, a dead-ends table, four self-issued corrections invalidating its own
earlier numbers, and confounds caught in M4 and M8 by its own review. Most of the
value here is in `docs/EVALUATION-METHODOLOGY.md`, and it transfers to problems
whose ceiling isn't set by a 1990s hidden-information game.
