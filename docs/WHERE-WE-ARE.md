# Where We Are

**Plain-language orientation. Start here after a break, or when the other docs
assume too much.** No timeline and no history — `IN-PROGRESS.md` is the running
log, `MILESTONES.md` the results ledger plus the live plan, and
`docs/MILESTONES-ARCHIVE.md` has M0–M12 in full.

**Keep this current.** Update it whenever a milestone closes, and prune anything
that stops being true. It should stay roughly one screen.

Last updated: **2026-09-24**

---

## Read this first: research is closed, the tooling is done

**The ML research is finished.** The bounded finish ran to its end: M12's
terminal gate passed, and M12 is closed. The answer on record is that the agent
is **mediocre**, and the method that measured it is the durable output.

**M13 (experiment tracking + CI) is also done, closed 2026-09-24 with its gate
passed.**
- Every training and eval run now lands in an MLflow store with its git SHA,
  seed and checkpoint hash.
- A GitHub Actions gate fails any push that regresses the shipping checkpoint.
  It was proven by re-introducing a real, previously fixed bug: the build went
  red, and the revert went green.

**🚫 Do not add ML scope.** No new models, arms, hypotheses or schema work.
Tracked re-evaluations of existing checkpoints are fine. Anything interesting
they turn up is a note for a future project, not a reason to reopen this one.

## The one-paragraph version

On random-team Gen 1 (randbats), the shipping M7 agent's raw greedy policy wins
**77.7% vs Random** (an opponent playing random legal moves). On the human
ladder it wins **19.3% of the games that actually get played out**. With MCTS it
wins ~93% against our scripted bots, but search's true contribution is
confounded and unknown. The best explanation for the ceiling is **observation
poverty**: moves are encoded without identity (Recover and Swords Dance are
byte-identical), there are no species or stats, and type effectiveness is
encoded in the attacking direction only. That diagnosis is well-evidenced but
was **never tested**.

## What we know for sure

Measured, with confidence intervals, at sample sizes that support the claim.

- **Fixed-team Gen 1 OU didn't break anything (M12).** Raw sampled policy,
  n=5,000 per opponent: **95.9% vs Random, 92.2% vs DamageFirst**. It is *not*
  comparable to randbats: the mirror roster removes team-luck variance. Its gen1ou
  ladder run (Phase 5) was never recovered, so no gen1ou ladder number exists.
- **About a third of our ladder wins are forfeits.** Every historical ladder
  number is ~7pp high. Contested only: **19.3% (n=326) greedy, 22.9% (n=341)
  sampling**. `ladder_analysis.py --min-decisions 16` scores contested games.
- **The agent gets worse as opponents and games get longer or stronger.** Win
  rate falls 34.4% → 15.3% across opponent Elo 1000 → 1299, and 31.2% → 14.6%
  from 16–20 to 31–40 decisions.
- **Greedy decoding beats sampling offline:** +7.8pp vs Random [+6.1, +9.5] and
  +5.0pp vs DamageFirst, n=5,000/arm. It is the ladder default. The CI gate
  caught exactly this bug when it was re-introduced as a backtest.
- **Training and eval are reproducible.** Re-running the 5M-step recipe lands
  within **0.6pp**, even across Mac → GPU. M13's tracked re-evaluations
  reproduced the ledger (M7 greedy 76.8% vs 77.7%; M9 2c −8.2pp vs −8.3pp), and
  CI on Linux reads 76.0–77.6% (six clean runs) against the Mac's 77.3%.
- **A better imitator can be a worse learner.** Randbats-only BC was a +5.6pp
  better mimic and finished 8.3pp worse after RL.
- **There is no more Gen 1 human data to get.** The replay archive is exhausted.

## Dead ends — don't re-propose these without new evidence

| Idea | Why it's closed |
|---|---|
| Scrape more Gen 1 replays | Archive exhausted |
| Harvest "untapped" tournament games | Already 74% of training data |
| Fix the value head (AlphaZero-style) | Tested twice, both null (−2.5pp each) |
| Train BC on target format only | Tested — better imitation, worse RL substrate |
| Spar with human-like opponents | M9: no measurable effect (−1.0pp) |
| Pick the best checkpoint off a sweep | Sweep peak is luck-inflated; it regressed 4.2pp |
| Speed-ratio observation feature | M8 Phase 1A: −3pp, doubly confounded |
| Re-roll the M12 roster after seeing the gate | Sweep-picking by another name |

## Left on the table (closed, not disproven)

These are for a *future* project; M13 does not touch them.

- **Observation schema v4 (M11 Phase 1)** is the best untested idea. Under a fixed
  roster much of it becomes cheap (species drops to 6 constants), so it must be
  re-derived for that format, not ported from the randbats spec.
- **M10's tactical-error lead.** Wasted-turn double switches run 15.7% in losses
  vs 6.1% in wins (+9.6pp). There's no human baseline, and causation may run
  the other way.
- **The M11 width arms (h128/h512)** were trained and never evaluated. They're
  home box only. If the box comes back, M13 may *re-evaluate* them as tracked runs.
- **M12 Phase 5.** If the home box comes back, the recovery recipe is in
  `MILESTONES.md` → M12. It gives a standalone number with no comparison.

## The live menu

**Nothing is planned, and that is the recommendation.** Optional, if the home
box comes back:
- Recover the M12 Phase 5 ladder number.
- Log tracked re-evaluations of the M12 and M11-width checkpoints. That is also
  the first real test of the reverse-tunnel recipe.

Neither changes any conclusion.

## What this project actually produced

The agent is mediocre. The method is not: pre-registered gates, sample-size
widenings recorded rather than quietly applied, a dead-ends table, four
self-issued corrections invalidating its own earlier numbers, and confounds
caught in M4 and M8 by its own review. Most of the value is in
`docs/EVALUATION-METHODOLOGY.md`, and M13 turned it into tooling: tracked runs
and a pre-registered CI gate.
