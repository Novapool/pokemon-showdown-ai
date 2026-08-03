# Where We Are

**Plain-language orientation. Start here after a break, or when the other docs
assume too much.** No timeline and no history — for those, `IN-PROGRESS.md` has
the running log, `MILESTONES.md` has the results ledger plus the live plan,
and `docs/MILESTONES-ARCHIVE.md` has M0–M9 in full.

**Keep this current.** Update it whenever a milestone or experiment closes, and
prune anything that stops being true. It should stay roughly one screen.

Last updated: **2026-08-03**

---

## The one-paragraph version

We have a Pokémon-battling agent that is **very good against simple bots and
mediocre against humans**. It wins ~93% against our scripted opponents but only
**~27% on the real ladder — and ~19–23% of the games that are actually played
out**, because roughly a third of our ladder wins are opponents forfeiting in
the first few turns. A code review on 2026-08-01 identified the root
cause: **the agent cannot see what it needs to play Gen 1 well.** Its observation
encodes moves without identity (Recover and Swords Dance are byte-identical),
and carries no species or stats at all, making damage estimation impossible.
This is invisible against `Random` and `DamageFirst`, which never switch and
mostly attack (base power suffices), and expensive against humans (which switch
constantly and reason with stats and coverage). **This is the first hypothesis
that predicts the shape of the gap rather than proposing another knob.** Six
other theories have been tested and ruled out since M7. The binding constraint
is now measured and understood.

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

## Options from here

**The new leading hypothesis is observation poverty.** This is the first lever
that predicts *why* the gap has this exact shape. It is not yet tested end-to-end,
and closing it requires retraining (new obs schema invalidates every existing
checkpoint). The honest options:

**1. Fix observations (M11, ~2–3 days training).** Add move identity, species,
stats, trapping, and Hyper Beam recharge. Requires new `obs_schema_v4`,
matching `replay-adapter.ts` path, BC retraining. Gates on raw-policy bot evals
at n=2,000/opponent with CI excluding 0 (currently ~+3pp bar). **This is the
highest-leverage option, and the first one worth spending time on.**

**2. Reward asymmetry — DONE, shipped 2026-08-01 (M11 Phase 0).**
`applyStallPenalty` now clips before charging the duration cost, so wins and
losses pay it equally, and `--stall-penalty` makes it settable (default 0.001 =
historical). The regression test that "passed" under the old formula encoded the
bug; it has been replaced. **Not yet re-tested end-to-end** — whether the fix
un-suppresses M8 Phase 2's value-head null is still an open question, and it
rides along free on the next training run.

**3. Greedy decoding — settled offline, unresolved on the ladder.** At
n=5,000/arm greedy is **+7.8pp vs Random [+6.1, +9.5]** and **+5.0pp vs
DamageFirst [+3.1, +6.9]**; the DamageFirst reading *reverses* the underpowered
n=400 −2.0pp in the code review. `ladder-bot.js` has been greedy by default
since 2026-08-01 (`--sample` opts out). **The one ladder run since (n=360,
`--mcts`) read 26.9%, −3.5pp [−10.0, +3.0] against the 30.5% baseline — no
detectable effect either way.** That run is not a refutation: under `--mcts`
search already argmaxes, so greedy reached only the ~20% of decisions that fall
back to the raw policy, and the comparison is against July sessions with no
recorded opponent Elo. Keeping the default on the offline evidence. A real
ladder test means `--sample` as a paired control arm, no search, ~83 h — **not
worth it before the observation fix**, since M11 invalidates the checkpoint.

**4. A bigger brain — trained, awaiting evaluation.** The shipping network is
151,187 parameters. Two PPO arms finished on the home box 2026-08-02:
`m11_h128` (151,187 params) and `m11_h512` (**801,299**), both 5M steps,
identical recipe, differing only in width. **Nothing has been evaluated yet —
running that battery is the immediate next action.** Note this is a capacity
test on the *old* v3 schema: if width helps while the agent is still playing
blind, that is informative; if it does nothing, it does not rule out width
helping after the observation fix. A larger transformer was tried in M3 and
retired for underperforming, so bigger is not automatic.

**5. Diagnose the losses directly (M10, ~5 days, no retraining).** We have 885
full ladder logs on disk, and they carry `|request|` lines — exact stats, HP and
the legal move set at every decision. That is enough to replay each decision
against ground truth and count *categorical* blunders: switching into a
weakness, failing to leave a losing matchup, using a big move where a priority
move already secures the KO, attacking into an immunity. Scored in wins vs
losses with CIs. **This is the only option that tells us what the agent does
wrong rather than inferring it**, and the long-game decay above is precisely the
pattern it would explain. It is also the one option whose value survives
whatever M11 does to the checkpoints. Fully planned in MILESTONES.md → M10.

**6. Fixed-team Gen 1 OU instead of random teams (~1 day).** Reduces team-luck
variance. The code already supports it; the missing piece is plumbing teams and
picking a roster. Downside: Gen 1 OU ladder is slower, so validation takes longer.

**7. Call it.** The agent is a solid club player. M8–M9 tried two structural
bets and both failed. The observation fix is well-motivated and measurable, but
it is a retraining — invalidates all checkpoints, costs time. **This is a
legitimate stopping point** if the resource/motivation calculation doesn't favor
continuing.

## Recommendation

**First, finish what is already paid for: run the eval battery on the two
trained arms (option 4).** They have been sitting untouched since 2026-08-02.
Costs hours, not days, and option 1's scoping depends on knowing whether width
does anything.

**Then do option 5 (M10) before option 1.** This is a change from the previous
recommendation, and the 360-game run is why. M11 is a multi-day retraining
justified by a hypothesis assembled from a code read, one live game, and an
aggregate correlation — we have never once looked at what the agent actually
does wrong. M10 needs no retraining, runs on data already on disk, and either
confirms the observation story concretely (blunder rates that track the missing
dims, concentrated in long games) or finds something else entirely. **Going
straight to M11 means spending the project's largest remaining budget on an
untested diagnosis when the test is five days and already planned.**

**Then option 1 (M11 Phase 1), scoped by what M10 finds.** The observation
hypothesis remains the best one we have and is well-evidenced from the code
review (`docs/CODE-REVIEW-FINDINGS.md`); M10 should sharpen which dims to
prioritise rather than replace it. The new schema invalidates every checkpoint,
so pre-register the gates and confirm the cost before starting.

**Option 3 is done as far as it is worth taking it** — greedy is confirmed
offline, is the ladder default, and its one ladder read was flat and diluted by
search. Resolving it properly costs ~83 h to answer a question M11 would
invalidate. Leave it on and move on.

**One free fix regardless:** the ladder bot should record `opp_rating` on every
run (it now does) and future ladder claims should quote contested win rate. Two
of the three numbers this project has argued over were partly measuring how
often strangers rage-quit.

**Expectation:** The realistic outcome is *understanding why the gap exists*,
not erasing it. Gen 1 random battles carry real team luck that puts a ceiling
below 100% no matter how good the player is. The project has produced a strong
bot-eval agent and an unusually rigorous measurement setup. If observation
fixes don't close the gap substantially, that is a finding worth recording.
