# Where We Are

**Plain-language orientation. Start here after a break, or when the other docs
assume too much.** No timeline and no history — for those, `IN-PROGRESS.md` has
the running log and `MILESTONES.md` has the full record with numbers.

**Keep this current.** Update it whenever a milestone or experiment closes, and
prune anything that stops being true. It should stay roughly one screen.

Last updated: **2026-08-01**

---

## The one-paragraph version

We have a Pokémon-battling agent that is **very good against simple bots and
mediocre against humans**. It wins ~93% against our scripted opponents but only
**~30% on the real ladder**. Closing that gap has been the whole project since
M7, and five separate theories about why it exists have now been tested and
ruled out. The most likely remaining explanation is the one we are testing now:
**the agent has never really practised against anything that plays like a
person.**

## The core problem, as an analogy

Our agent trains against a **tennis ball machine**, then we're surprised it
loses to actual players.

Its three training opponents are `Random` (fires legal moves at random),
`DamageFirst` (always attacks with its strongest move), and self-play (frozen
copies of itself). The detail that makes this literal: **both scripted
opponents never voluntarily switch Pokémon** — it's in the header comment of
`sim/tools/damage-first-ai.ts`. Switching is *the* central skill in Gen 1.
So for half its training the agent faced opponents that structurally cannot do
the main thing real opponents do, and the other half it faced itself, which
shares all of its own blind spots.

## What we know for sure

Each of these is measured, with confidence intervals, at sample sizes that can
actually support the claim. That was not true of this project before M9.

- **The agent is strong vs bots, weak vs humans.** 93% vs Random with search;
  **30.5% on ladder** (n=387, CI [26.1, 35.3]).
- **Our measuring instrument used to be broken, and now isn't.** Early ladder
  reads were 50–100 games where the noise (±13pp) was larger than any effect we
  could produce — like judging a diet from daily weigh-ins. `GXE` was worse: a
  *lifetime* average that barely moves. Protocol now lives in
  `docs/EVALUATION-METHODOLOGY.md` and nothing counts unless it follows it.
- **Training is reproducible.** Re-running the identical 5M-step recipe lands
  within **0.6pp**, even across a Mac→GPU change. So differences bigger than
  ~3pp are real signal, not luck.
- **A better imitator can be a worse learner.** Training the copy-a-human stage
  on *only* the target format made a better mimic (+5.6pp) that then finished
  **8.3pp worse** after self-practice. Reading narrowly gave it fewer ideas to
  explore with. Practical upshot: use the **mixed** corpus checkpoint
  (`bc_mlp_gen1_v3.pt`) to start any RL run.
- **There is no more Gen 1 human data to get.** The replay archive is
  exhausted; we hold essentially all of it (`docs/DATA-INVENTORY.md`).

## Dead ends — don't re-propose these without new evidence

| Idea | Why it's closed |
|---|---|
| Scrape more Gen 1 replays | Archive exhausted, we hold ~all of it |
| Harvest the "untapped" unrated tournament games | Already 74% of training data |
| Richer observations (speed-ratio feature) | Tested, made it *worse* |
| Fix the value head (AlphaZero-style) | Tested twice, both negative |
| Train on the target format only | Tested — better mimic, worse learner |

---

## Options from here

**1. Spar with a human impersonator — 🟡 RUNNING NOW (~70 min/run).**
Our best human-imitator is the copy-a-human checkpoint. Instead of practising
against a ball machine and itself, the agent now practises against *that*,
including its switching behaviour. Cheap, needed almost no new code. This is
the direct answer to the core problem above. **Known limitation, written down
before the result:** the sparring partners are frozen, so the agent will
eventually outgrow them — a win is unambiguous, a loss doesn't distinguish
"human style doesn't help" from "frozen partners stop teaching." Follow-up for
that case is recorded in `IN-PROGRESS.md`.

**2. Widen the reading list (~2 h).** Our own result says variety in the
imitation stage helps learning. Every checkpoint ever built used a 50/50 split
and nobody has tried tilting further toward variety. Cheap, but expect a few
points, not a transformation.

**3. Fixed-team Gen 1 OU instead of random teams (~1 day).** Today every battle
deals random teams, so the agent must know ~150 Pokémon shallowly and a lot of
wins and losses come down to the draw. With a fixed team it would learn 6
Pokémon deeply, and results would reflect skill instead of the deal.

**Cheaper than it sounds, and the code was checked rather than guessed:**
`PokemonGymEnv` already takes a `format` option (`sim/tools/pokemon-gym.ts`
line ~535, defaulting to `gen1randombattle`), and it stays in Gen 1 so **the
observation layer needs no rewrite** — unlike a Gen 9 move, which is why that
one is a restart and this one isn't. The genuinely missing piece is teams: the
battle spec it builds is `{formatid, seed}` with no team attached, which is
fine for a random format and not fine for OU. So the work is plumbing a packed
team through gym → bridge → clients, plus picking a team.

**Two real catches.** The Gen 1 OU ladder is *less* busy than random battles
(31 vs 44 replays/day), so final human validation gets **slower**, not faster.
And every result M2→M9 is on random battles, so baselines would need
re-measuring in the new format before anything could be compared.

**4. A bigger brain (medium).** The network is **151,187 parameters** — tiny.
We keep asking "what should it study?" and never "can it hold what it studies?"
Its ~54% ceiling at predicting human moves might be capacity, not curriculum.
**Caveat:** a larger transformer was already tried in M3 and *retired* for
underperforming, so bigger is not automatically better here.

**5. Switch to Gen 9 (large).** ~45× the data and ladder traffic. But it's a
restart, not a pivot: abilities, items, terastallisation, ~1000 species, and
every checkpoint M2→M9 becomes worthless. Costed in `docs/DATA-INVENTORY.md`.
Not recommended.

**6. Call it.** The agent is a solid club player, not a tournament threat. M8
and M9 produced almost entirely negative results on "make it stronger," and
these levers are worth single digits against a ~40-point gap.

## Recommendation

**Option 1 is running.** If sparring against a human-like opponent doesn't move
the number, that is genuinely informative — it would mean opponent realism
isn't the binding constraint either, and the honest conversation becomes
options 3, 4 and 6.

**Then option 3** if we want to keep going seriously. Removing team luck
attacks a source of noise that no amount of skill overcomes, and it makes every
future experiment easier to measure. Option 2 is worth its two hours as a cheap
side quest.

**Expectation to hold onto:** the effects we measure are 5–8 points on bot
evals, and the gap to human-level play is much larger. Treat this as *learning
why the gap exists* rather than expecting to erase it. Gen 1 random battles
also carry real team luck, so there is a ceiling below 100% that no skill
reaches.
