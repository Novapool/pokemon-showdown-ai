# Review B — Observation & action-space domain correctness (Gen 1)

Scope: `sim/tools/{feature-extractor,pokemon-gym,replay-adapter,battle-sim,damage-first-ai,random-player-ai,type-chart-v3}.ts`.
Read-only. All measurements below were produced by running the compiled gym in this repo (scripts in the scratchpad).

## Bottom line

The observation is not a compressed view of the game — it is a *lossy* one, and what it drops is
almost exactly the set of things that separate a strong Gen 1 player from a move-clicker:
absolute stats, move identity, PP, trapping, recharge, sleep duration, confusion, and last move.
Against `Random`/`DamageFirst` none of it matters, because neither opponent ever switches
(measured: 0.6% of decision points, all forced replacements) and neither uses status, Wrap, or
recovery. Against humans all of it matters. That asymmetry is the shape of the 93% → 30.5% gap,
and it is present in the environment independently of model size or format luck.

Measured on the live gym (`structured-v3-extended`, 1044 dims, 258–449 decision points,
random legal policy):

| metric | value |
|---|---|
| non-zero dims per observation | **128.9 / 1044 (12.3%)** |
| distinct float values per observation | **25.3** |
| dims non-zero in <1% of decision points | **667 / 1044** |
| within-token dims that never fired at all | 21 of 87 |
| boost/volatile block (dims 65–76, both active tokens) non-zero | **3.8%** of decisions vs DamageFirst (25.4% vs Random) |
| opponent chose a switch | **0.6%** of decisions (both bots) |

So the "1032/1044-dim" observation carries on the order of **25 numbers of real content**. The
8:1 first-layer compression is a red herring; the input is the bottleneck, not the projection.
Widening the schema again without adding *information* will not help — v2→v3→v3-ext added 264
dims and the measured content barely moved, because the new blocks are ~never active in the
training distribution.

---

## S1 — No stats and no species identity in the observation (CRITICAL)

**What's missing.** The v1/v2/v3 token is: HP ratio, level/100, two 15-way type one-hots, 6 status
bits, active/unknown/fainted flags, 4×(basePower, accuracy, PP ratio, type/20, category/2)
(`feature-extractor.ts:446-455`, `590-656`). There is **no Attack/Defense/Special/Speed/HP stat and
no species identity** for any of the 12 tokens. HP is a *ratio*, so absolute bulk is invisible too.

The data is already in hand and is being thrown away. A live request carries exact stats per own
Pokémon — verified:

```
{ ident: "p1: Pinsir", details: "Pinsir, L75", condition: "143/252",
  stats: { atk: 262, def: 224, spa: 157, spd: 157, spe: 202 }, moves: [...] }
```

`extractFeaturesStructured` never reads `stats` (`feature-extractor.ts:744-769`). Opponent base
stats are public dex data (gen1randombattle has no hidden EVs/IVs beyond generator conventions)
and are equally available via `Dex.mod('gen1').species.get(...).baseStats`.

**Why it matters in Gen 1.** Every switch decision is a damage race: "does Tauros' Body Slam
2HKO my Starmie", "can Chansey take a Blizzard", "do I outspeed after the para". Without any
stat the agent cannot form a damage estimate at all — the closest proxy it has is
`basePower/250` and a type-effectiveness scalar. It also means the agent cannot tell a
level-75 Rhydon from a level-75 Nidoking on the bench beyond move types.

**Bot-invisible?** Yes, maximally. Against a non-switching opponent, "highest type-eff ×
basePower" is near-optimal and needs no stats. Against a human, every stay/switch/sack decision
is a stat comparison.

**Fix.** Schema v4: append 5–6 dims/token — own side from `request.side.pokemon[i].stats`
(normalise /512), opponent from level-scaled gen1 base stats. Optionally a species index or
learned embedding. ~60 LOC in `feature-extractor.ts` + a `gen1BaseStats()` next to
`gen1BaseSpeed()` in `type-chart-v3.ts`. **Effort: S to implement, L to retrain (and the BC
adapter must move in lockstep — see S11a).**

---

## S2 — Move identity is not in the observation; the encoding aliases the decisive Gen 1 moves (CRITICAL)

A move slot is only `(basePower/250, accuracy, PP ratio, type/20, category/2, disabled)`
(`feature-extractor.ts:625-653`). Verified collisions from the gen1 dex — these pairs are
**byte-identical** in the observation:

| collides | encoding |
|---|---|
| **Rest ≡ Reflect ≡ Amnesia ≡ Agility** | `0, 1.0, Psychic, Status` |
| **Recover ≡ Softboiled ≡ Substitute ≡ Swords Dance** | `0, 1.0, Normal, Status` |
| **Seismic Toss ≡ Counter** | `0.004, 1.0, Fighting, Physical` |

Snorlax's canonical randbats set is Body Slam / Reflect / Rest / Earthquake — the agent literally
cannot tell Rest from Reflect. Slowbro's Amnesia (the tier's premier win condition) is
indistinguishable from Rest on the same mon.

Further, the whole class of Gen 1-defining moves is encoded as junk:

- **Wrap / Bind / Fire Spin / Clamp** → 15–40 BP attacks with mediocre accuracy. Nothing marks
  them as trapping. A policy will never learn to click them and cannot recognise them.
- **Horn Drill / Fissure / Guillotine** → `bp 0, acc 0.30` — indistinguishable from a bad status
  move. No OHKO flag.
- **Slash / Razor Leaf / Crabhammer / Karate Chop** → `critRatio` exists in the dex (Slash = 2)
  and is not in the schema. Gen 1 crit rate is base-speed-keyed and high-crit moves are dominant;
  neither signal is present.
- **Recovery** (Recover/Softboiled/Rest) has no flag — see the collisions above.

The v3 dims that could have disambiguated (`V3_FLAG_RECHARGE/SELFKO/PRIORITY`,
`V3_INFLICTED_STATUS`) are **aggregated over the whole 4-move set into one value per token**
(`feature-extractor.ts:533-561`). So the observation says "this mon has a sleep move somewhere"
but not which of the four buttons it is. The agent must choose a slot; the slot-level features
don't identify the slot.

**Fix.** Move the v3 flags to per-slot (4× the bits) and add per-slot bits for
heal / boost / partial-trap / OHKO / high-crit / recharge / fixed-damage, all derivable from dex
fields (`move.heal`, `move.boosts`, `move.volatileStatus === 'partialtrappinglock'`,
`move.ohko`, `move.critRatio`, `move.flags.recharge`, `move.damage`). Or, better, a learned
per-slot move-id embedding (gen1 has ~165 moves). **Effort: M.**

---

## S3 — The reward function penalises long games, and the clip makes it one-sided (HIGH, cheapest fix in the review)

`pokemon-gym.ts:706` (and `:869`, `battle-sim.ts:443`):

```ts
if (done) { reward -= 0.001 * this._turnCount; ... }
reward = Math.max(-1, Math.min(1, reward));   // line 711
```

Per-faint shaping is ±0.01 and status-on-p2 is +0.0001 (`parseProgressLines`, `:1246-1266`).

Two problems:

1. **The turn penalty is applied only on the terminal step and then clipped.** A loss terminal is
   `-1 - 0.01·faints - 0.001·T`, which saturates at −1 for any T. A win terminal is
   `+1 + 0.01·faints - 0.001·T`, which does *not* saturate. Net effect: **long games are penalised
   only when you win; stalling when losing is free.** A 20-turn win scores ~0.98, a 100-turn win
   ~0.90.
2. Switching is the tempo-losing, game-lengthening move. It costs HP and a turn now for position
   later. The only explicit shaping term the environment has therefore points *against* the
   central Gen 1 skill. Against non-switching bots the agent wins fast anyway so the term is
   nearly free; against humans, games are long and patient, and the term is a persistent
   headwind.

Also asymmetric: +0.0001 for inflicting status on p2, **no penalty for statuses received**. Getting
slept/paralysed is free in the reward.

**Fix.** Remove `-0.001 * turnCount`, or convert it to a symmetric per-step living cost (~1e-4)
applied identically to wins and losses, and clip the shaping *before* adding the ±1 terminal so
the terminal signal is never eaten. Mirror in `battle-sim.ts:443`. **Effort: XS (3 lines + a
retrain).** This is the single cheapest experiment in the review and it is directly on the
switching hypothesis.

---

## S4 — The training opponents make switching worthless and leave most of the schema dead (HIGH)

`RandomPlayerAI` is constructed with `move` defaulting to 1.0 (`random-player-ai.ts:25`, gym at
`pokemon-gym.ts:562`) → never voluntarily switches. `DamageFirstAI` picks max base power
(`damage-first-ai.ts:35-47`), ignoring type effectiveness, and inherits `move: 1.0` → never
switches, never uses a status move (bp 0 always loses the argmax), never uses Wrap (15 BP),
never Recovers, and always picks the same move on a given Pokémon.

Measured consequences in the observation the agent actually trains on:

- opponent switch chosen on **0.6%** of decision points (forced post-KO replacements only);
- the entire boost/volatile block (dims 65–76 on both active tokens) non-zero on **3.8%** of
  decisions vs DamageFirst;
- Light Screen, Substitute, Leech Seed and the toxic counter dims **never fired once** in 258
  decision points;
- DamageFirst reveals **exactly one move per Pokémon for the whole game** — verified opponent
  tracker state: `Meowth → [thunderbolt]`, `Persian → [thunderbolt]`. So opponent move tokens
  are ~always "1 filled + 3 empty"; against a human they are 3–4 filled.

So the features that encode human play are approximately constant during training, and the
opponent-token statistics are out-of-distribution at eval time in a very literal sense.

**Fix.** Either (a) give DamageFirst a minimal competent policy (switch on bad type matchup,
use sleep/paralysis when the opponent is healthy, recover under 40%), or (b) make self-play
(`opponent: 'self'`, already implemented) the primary opponent so at least the switching
distribution is non-degenerate. **Effort: M.**

---

## S5 — Partial trapping is completely invisible in both directions (HIGH, human-only)

Gen 1 partial trap: the victim can still switch but every move choice is a no-op. In the engine
this sets `maybeLocked` on the victim (`data/mods/gen1/conditions.ts:205-207`,
`sim/pokemon.ts:1100-1102`) and pushes an empty move action (`sim/side.ts:673-680`).

- The victim's request **still lists all 4 moves**, so the observation on a trapped turn is
  indistinguishable from a normal turn — but all four move actions do nothing.
- `request.active[0].maybeLocked` / `maybeDisabled` / `trapped` are **never read by the
  extractor** — only `validActionsForRequest` reads `trapped` (`pokemon-gym.ts:1307`).
- The volatile tracker handles only Reflect / Light Screen / Substitute / Leech Seed
  (`pokemon-gym.ts:370-376`); `|cant|…partiallytrapped` and `|-activate|…move: Wrap` are dropped.
  So the agent also cannot see that *it* has the opponent trapped.

Wrap-stall (Dragonite/Tentacruel/Cloyster/Victreebel) is live ladder strategy. Neither bot ever
uses a trapping move, so this costs nothing in training and can cost whole games on the ladder.

**Fix.** Track `partiallyTrapped` per side in `ObservationTrackers._processVolatileLine` (the
`|cant|` line names it explicitly) and read `maybeLocked` off the request; 1 dim per active token.
**Effort: S.**

---

## S6 — Hyper Beam recharge is invisible for the opponent, and garbage for self (HIGH, human-only)

- On the agent's own recharge turn the request collapses to `moves: [{ move: 'Recharge', id:
  'recharge' }]`. Verified: `Dex.mod('gen1').moves.get('recharge').exists === false`, so
  `fillPokemonToken`'s `if (moveInfo.exists)` guard (`feature-extractor.ts:635`) falls through to
  the defaults — `bp 0, acc 1.0, type Normal, category **Physical**`. The recharge turn is
  encoded as a nonsense 0-BP physical Normal move plus 3 empty slots.
- v3 dim 81 (`V3_FLAG_RECHARGE`) reads `move.flags.recharge` over the *known move set*
  (`type-chart-v3.ts:216-233`), i.e. it means "this mon knows Hyper Beam", **not** "is recharging
  right now".
- For the opponent there is no recharge state at all. Nothing in `OpponentPokemonInfo`
  (`feature-extractor.ts:474-480`) or the tracker records it.

The free turn after an opponent's Hyper Beam — plus the Gen 1 rule that a KO skips the recharge —
is among the most exploitable patterns in the tier. DamageFirst *does* spam Hyper Beam (highest
BP), so the agent has seen thousands of them and still cannot learn to punish, because the state
isn't in the observation.

**Fix.** In the tracker: a `|move|…Hyper Beam` by side S with no accompanying `|faint|` sets
`rechargingNextTurn[S]`, cleared after the next `|move|`/`|cant|` from S. One dim per active
token. **Effort: S.**

---

## S7 — Sleep is one bit with no counter, and asleep/frozen erases the moveset (HIGH)

- No sleep-turn counter anywhere, for either side. Gen 1 sleep is 1–7 turns and counting is
  standard human play (when to switch the sleeper out, when to bring in the answer). The
  observation is a single frame with **no history and no turn number** — if the policy is not
  recurrent it has no way to recover this.
- While asleep or frozen, `getMoveRequestData` replaces the whole moveset with
  `[{ move: 'Fight', id: 'fight' }]` (`sim/pokemon.ts:1079-1083`), and `fight` does not exist in
  the dex, so the agent's entire moveset disappears from the observation at exactly the moment it
  must decide "ride it out or switch out my sleeper". The status bit is set, so it knows it's
  asleep; it just can't see what the mon does.
- The toxic counter *is* tracked (`V_TOXIC_COUNTER`) — good, and the right precedent for the
  sleep counter.

**Fix.** Track turns-since-`-status …slp` per nickname (bench-persistent, same lifecycle as the
existing `sleepInflicted` map) and add a dim; on locked/asleep requests fill move dims from
`request.side.pokemon[0].moves` instead of `active[0].moves`. **Effort: S.**

---

## S8 — No PP tracking for the opponent or for own bench; PP stalling is unplayable and unrecognisable (MEDIUM-HIGH)

`fillOpponentToken` builds `TokenMoveInput` with no `pp`/`maxpp` (`feature-extractor.ts:669`), and
`fillOwnBenchToken` does the same (`:662`); the PP ratio then defaults to **1.0 forever**
(`:648-649`). So:

- the agent never sees that the opponent's Recover/Softboiled (8 PP) is running out — the
  standard human stall win condition;
- it can't see that its own benched Chansey is out of Softboiled before switching to it;
- only the *own active* has real PP.

Note the asymmetry with BC: `replay-adapter.ts:247-248,352-356` *does* model own-side PP decay by
counting `|move|` lines. So the BC training distribution and the live gym disagree on this feature
for bench Pokémon.

**Fix.** Count `(nickname, moveid)` uses in `ObservationTrackers._processRevealLine` (the `move`
branch already parses both) and expose `pp/maxpp` for opponent + own-bench tokens. **Effort: S.**

---

## S9 — Confusion, last move, and several volatiles are untracked (MEDIUM)

`_processVolatileLine` (`pokemon-gym.ts:347-382`) handles only boosts, Haze, Reflect, Light
Screen, Substitute, Leech Seed and the toxic counter. Not tracked, either side:

- **confusion** (`|-start|…|confusion`) — Confuse Ray Gengar/Haunter, Psybeam; 50% self-hit;
  invisible to the agent about itself *and* the opponent;
- **the opponent's last move** — there is no last-move feature at all. A human always knows what
  just happened; it's the basis of prediction and of Counter;
- Disable applied to the opponent, Mist, Focus Energy, Bide, Mimic-modified movesets, Transform
  (a transformed opponent keeps its pre-Transform species in `revealRecords`).

Gen 1 Disable *on the agent* is handled correctly — the gen1 mod sets `moveSlots[i].disabled = true`
directly (`data/mods/gen1/moves.ts:246-249`), not `'hidden'`, so the request and the mask are right.

**Fix.** Add tracked flags per active token; each is one line in `_processVolatileLine` plus one
dim. **Effort: S each.**

---

## S10 — The one speed feature is wrong exactly when speed matters (MEDIUM)

`fillSpeedRatio` (`feature-extractor.ts:791-802`) writes a single scalar, replicated on all 12
tokens: `min(ownBaseSpe / oppBaseSpe, 2) / 2`, from `gen1BaseSpeed` = species **base** speed
(`type-chart-v3.ts:79-90`).

- **Level is ignored.** gen1randombattle levels vary widely (verified in one battle: L85, L73,
  L75) and level scales speed linearly — the ratio is simply wrong for most matchups.
- **Paralysis is ignored.** Gen 1 par is a ×0.25 speed drop; the single most consequential speed
  modifier in the tier flips turn order and the feature doesn't move.
- **Bench speeds are absent**, so revenge-kill switch planning has no basis.
- The agent's own true `stats.spe` is in the request and unused (see S1).
- Gen 1 crit rate is base-speed-keyed; that isn't represented either (see S2).

**Fix.** Use `request.side.pokemon[0].stats.spe` for own, level-scaled base speed for the
opponent, apply the par multiplier from the status bits, and emit per-token speed rather than one
global scalar. **Effort: S.**

---

## S11 — BC ↔ live-gym distribution mismatches in `replay-adapter.ts` (MEDIUM-HIGH; BC is the warm start for every RL run)

a. **No v3-extended support.** `ReplayObsVersion = 'v2' | 'v3'` (`:122`), and `_snapshot` passes
   `{ sleepClause }` with no `extended` (`:364`). The M8 87-dim schema has **no BC data path** —
   any v3-ext run is either un-warm-started or warm-started from a differently-shaped schema.
   **Effort: XS.**

b. **Different canonical orderings.** Replays order own move slots **alphabetically by move id**
   (`:332-333`) and own bench by **(speciesId, nickname)** (`:337`); the live gym uses request
   order and the engine's mutating `side.pokemon` order. The header's defence is that the policy
   learns a "slot-content invariant" — but per S1/S2 the slot content is only
   `(level, types, bp, acc, type, cat)`, which is weak and sometimes degenerate (Rest ≡ Reflect).
   The invariant the design depends on may not be learnable from the features provided. **This is
   the interaction that makes S1/S2 worse than they look in isolation.** **Effort: M** (pick one
   canonical ordering and apply it on both sides).

c. **Own move slots contain only moves revealed in that log** (`sortedMoves` over the pass-1
   revealed set, `:332`). A mon that only ever used 2 moves contributes 2 filled slots. The live
   gym always has 4. "Empty own move slot" is common in BC and impossible in the gym.

d. **`|cant|` turns are dropped** (`:267-274`). BC therefore has **zero coverage** of asleep /
   frozen / fully-paralysed / trapped states — which are exactly the states the gym *does* present
   for a decision (with the degenerate `fight` moveset, S7). The warm start has never seen them.

e. **Silent loss of pre-KO decisions.** A seat KO'd before it acts emits no `|move|`/`|switch|`,
   so its `pendingTurn` is overwritten at the next `|turn|` (`:230-233`) and is **not counted in
   `skipped`**. This systematically discards the decisions immediately preceding getting KO'd —
   i.e. the mispredictions, the highest-information examples for learning to switch. **Effort: S**
   (emit or at least count them).

---

## S12 — Minor leak / ladder mismatch: trackers are fed from the omniscient stream (LOW)

`_runOmniscientReader` feeds `this._trackers.processLine` from `streams.omniscient`
(`pokemon-gym.ts:1141-1152`), which carries *secret* health for both sides. Verified opponent
tracker state: `condition: "273/273"` — exact HP, where a ladder client sees `"100/100"`
percentages. The extractor only takes the ratio, so the numeric impact is sub-1% rounding, and
opponent max HP (present in the string) is never read. But it does mean the gym and the ladder
client build tracker records from differently-formatted inputs; worth confirming the ladder-side
tracker rounds identically, otherwise there is a small train/deploy shift on every HP dim.

---

## S13 — Action space and masking: correct, with one unguarded failure mode (LOW)

9 actions (4 moves + 5 switches) is the right frame for Gen 1 singles; no mechanic needs more.
Masking (`validActionsForRequest`, `pokemon-gym.ts:1287-1333`) checked against the engine:

- **forced switch after KO** — switch-only branch, fainted excluded. Correct.
- **Wrap / partial trap** — gen1's `partiallytrapped` sets `maybeLocked`, *not* `trapped`
  (`data/mods/gen1/conditions.ts:186-207`), so switching stays legal. Correct for Gen 1 (moves
  become no-ops, which the mask can't express — see S5).
- **Hyper Beam recharge / multi-turn lock** — request collapses to 1 move, only index 0 legal.
  Correct (but the *observation* is garbage, S6).
- **sleep / freeze** — request collapses to `[Fight]`, index 0 plus all switches legal. Correct.
- **Struggle** — `moves: [struggle]`, index 0 only. Correct.
- **Disable** — gen1 sets `disabled = true` (not `'hidden'`), so the mask is right.
- gen1 has no trapping abilities, so `trapped` is never set. Correct.

One gap: `GymPlayer` (`pokemon-gym.ts:431-494`) does **not** override `receiveError`, while
`RandomPlayerAI` does (`random-player-ai.ts:30-35`) and `BattlePlayer.receiveError` **throws**
(`sim/battle-stream.ts:354`). A rejected choice would throw inside the detached
`void this._gymPlayer.start()` loop; `step()` would then hang forever on `waitForRequest` rather
than surface an error. I found no gen1 path that produces one given the mask above, but it is an
unguarded hang. **Effort: XS (copy the 4-line override).**

---

## Recommended order (value ÷ effort)

1. **S3** — delete the terminal turn penalty / fix the clip order. Hours, and it is a direct test
   of "the reward discourages switching".
2. **S4** — a switching, status-using scripted opponent (or self-play as default). Without this,
   no observation fix can be *learned*, because the states never occur in training.
3. **S1 + S2** — schema v4: real stats + per-slot move identity/flags. These are the actual
   blindness. Must land together with S11a/S11b so BC and RL share a schema.
4. **S5, S6, S7** — trapping, recharge, sleep counter. Small, self-contained tracker additions,
   each individually human-only in effect.
5. **S8, S9, S10** — PP, confusion/last-move, correct speed.
6. **S11c–e, S12, S13** — BC fidelity and hygiene.

The consistent thread: every item in 3–5 is a mechanic that DamageFirst and Random *never
exercise*, so none of them could ever have shown up as a regression in bot win rate. A 93% bot
win rate is exactly what a strong move-clicker with no positional model scores, and that is what
this observation and this reward are specified to produce.
