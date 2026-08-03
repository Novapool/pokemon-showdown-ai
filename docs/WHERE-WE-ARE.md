# Where We Are

**Plain-language orientation. Start here after a break, or when the other docs
assume too much.** No timeline and no history — for those, `IN-PROGRESS.md` has
the running log and `MILESTONES.md` has the full record with numbers.

**Keep this current.** Update it whenever a milestone or experiment closes, and
prune anything that stops being true. It should stay roughly one screen.

Last updated: **2026-08-02**

---

## The one-paragraph version

We have a Pokémon-battling agent that is **very good against simple bots and
mediocre against humans**. It wins ~93% against our scripted opponents but only
**~30% on the real ladder**. A code review on 2026-08-01 identified the root
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

This completely explains why the agent dominates bots that never switch (93%)
and struggles against humans who switch constantly and play for position
(30%). The sparring-partner hypothesis was tested in M9 and failed entirely
(−1.0pp vs Random, 0.0 vs DamageFirst) — **it is the agent's perception that
is the constraint, not its training opponents.**

## What we know for sure

Each of these is measured, with confidence intervals, at sample sizes that can
actually support the claim. That was not true of this project before M9.

- **The agent is strong vs bots, weak vs humans.** 93% vs Random with search;
  **30.5% on ladder** (n=387, CI [26.1, 35.3]) — scored while *sampling* the
  policy, which we now know costs ~5–8pp offline.
- **Playing the argmax beats sampling it — offline.** +7.8pp vs Random, +5.0pp
  vs DamageFirst, n=5,000/arm. **It did not show up on the ladder** (26.9% at
  n=360 vs 30.5% before, −3.5pp [−10.0, +3.0]) — but that run was under
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
| Speed-ratio observation feature | M8 Phase 1A: −3pp, and doubly confounded (see MILESTONES.md) |

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

**2. Also fix reward asymmetry (~3 lines, free to test).** `gym.ts` clips the
turn penalty but only when losing; `battle-sim.ts` (MCTS forward model) applies
neither penalty nor clip. M8 Phase 2's null may have been suppressed by this
asymmetry. Direct test: apply symmetric reward scaling and retest value
fine-tuning.

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

**4. A bigger brain (medium effort).** The network is 151,187 parameters — tiny.
At observation poverty's root, bigger capacity may help *after* the observation
fix. Before: the agent is playing blind; after: it might need more parameters
to use the information. **Caveat:** a larger transformer was tried in M3 and
retired for underperforming, so bigger is not automatic.

**5. Fixed-team Gen 1 OU instead of random teams (~1 day).** Reduces team-luck
variance. The code already supports it; the missing piece is plumbing teams and
picking a roster. Downside: Gen 1 OU ladder is slower, so validation takes longer.

**6. Call it.** The agent is a solid club player. M8–M9 tried two structural
bets and both failed. The observation fix is well-motivated and measurable, but
it is a retraining — invalidates all checkpoints, costs time. **This is a
legitimate stopping point** if the resource/motivation calculation doesn't favor
continuing.

## Recommendation

**Do option 1 next.** The observation poverty hypothesis is the first one that
predicts the gap's shape rather than proposing another knob. It is well-evidenced
from the code review (`docs/CODE-REVIEW-FINDINGS.md`), and M8 Phase 1A's
failure to help with speed ratio now reads as a second confound on top of a
fundamentally incomplete observation.

**Scope M11 carefully:** The new observation schema invalidates every
checkpoint. Estimate BC retraining time, the full training run, eval power.
Pre-register the gates. Only proceed if the costs are clear and agreed.

**Do option 2 as a quick smoke first** (reward fix is ~3 lines and an eval run).
It is a direct test of a new hypothesis from the code review, costs nothing
extra, and might move the dial.

**Option 3 is done as far as it is worth taking it** — greedy is confirmed
offline, is the ladder default, and its one ladder read was flat. Resolving it
properly on the ladder costs ~83 h to answer a question M11 would invalidate.
Leave it on and move on.

**Expectation:** The realistic outcome is *understanding why the gap exists*,
not erasing it. Gen 1 random battles carry real team luck that puts a ceiling
below 100% no matter how good the player is. The project has produced a strong
bot-eval agent and an unusually rigorous measurement setup. If observation
fixes don't close the gap substantially, that is a finding worth recording.
