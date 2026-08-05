# Battle Formats

Available battle formats in this Pokemon Showdown simulator and how to choose one for ML training.
Relevant source: `config/formats.ts`, `sim/TEAMS.md`, `simulate.js`.

---

## 🔒 THE FIXED ROSTER (M12 Phase 0 — pre-registered 2026-08-05)

**This is the project's final experimental configuration.** Format `gen1ou`, one
fixed 6-Pokémon team used by **both sides** in every training battle, bot eval and
ladder game from here forward. Locked before any code was written or any result
seen — see `MILESTONES.md` → M12 for the terminal Phase 4 gate.

**Do not change this roster.** If M12 Phase 4 fails its gate, that is the
finding. Re-rolling the roster after seeing the result is the sweep-picking
banned by the standing rules in `IN-PROGRESS.md`.

### The team

| Slot | Pokémon | Moves | Human set usage |
|---|---|---|---:|
| 1 | **Tauros** | Body Slam / Hyper Beam / Blizzard / Earthquake | 81.7% |
| 2 | **Chansey** | Ice Beam / Thunderbolt / Thunder Wave / Soft-Boiled | 34.3% |
| 3 | **Snorlax** | Body Slam / Earthquake / Reflect / Rest | 21.1% |
| 4 | **Exeggutor** | Psychic / Sleep Powder / Stun Spore / Explosion | 46.0% |
| 5 | **Starmie** | Psychic / Blizzard / Thunder Wave / Recover | 33.1% |
| 6 | **Alakazam** | Psychic / Seismic Toss / Thunder Wave / Recover | 65.6% |

Gen 1 has no items or abilities. All Pokémon are level 100 with maximum DVs and
stat experience (the Showdown default, and what `Teams.pack` emits below).

### Packed team string

Canonical copy lives at `config/rosters/gen1ou-standard.txt`. Validated with
`./pokemon-showdown validate-team gen1ou` (passes) and smoke-tested for 5/5
completed `gen1ou` battles via `BattleStream` + `RandomPlayerAI`.

```
Tauros||||BodySlam,HyperBeam,Blizzard,Earthquake||255,255,255,255,255,255|||||]Chansey||||IceBeam,Thunderbolt,ThunderWave,SoftBoiled||255,255,255,255,255,255|||||]Snorlax||||BodySlam,Earthquake,Reflect,Rest||255,255,255,255,255,255|||||]Exeggutor||||Psychic,SleepPowder,StunSpore,Explosion||255,255,255,255,255,255|||||]Starmie||||Psychic,Blizzard,ThunderWave,Recover||255,255,255,255,255,255|||||]Alakazam||||Psychic,SeismicToss,ThunderWave,Recover||255,255,255,255,255,255|||||
```

### How it was chosen

Mined from **10,101 local gen1ou replays rated ≥1300** (`data/replays/gen1ou/`,
filtered via `manifest.csv`), yielding 16,743 fully-revealed 6-Pokémon teams.

**Species usage across those teams:**

| Pokémon | % of teams | | Pokémon | % of teams |
|---|---:|---|---|---:|
| Tauros | 91.4% | | Alakazam | 43.9% |
| Chansey | 78.9% | | Jynx | 26.4% |
| Snorlax | 76.5% | | Rhydon | 24.1% |
| Exeggutor | 71.2% | | Zapdos | 21.9% |
| Starmie | 48.8% | | Gengar | 21.0% |

The top five are the metagame's core. For the sixth slot, the two candidates
were near-tied by usage as **exact** teams:

- **+Alakazam — 1,088 teams (6.50%), rank #1** ✅ chosen
- +Rhydon — 1,021 teams (6.10%), rank #2

**Tiebreak — three reasons, recorded before the run:**
1. **Rank #1 by usage.** An objective, pre-registerable rule rather than taste.
2. **Its value doesn't depend on perception the agent lacks.** Rhydon's main
   contribution is Ground typing (immunity to every Thunder Wave, and paralysis
   is Gen 1's dominant mechanic) — but that is *defensive positional* value, and
   v3 encodes type effectiveness **offensively only** (`docs/WHERE-WE-ARE.md`).
   We would be handing the agent a piece it cannot see. Alakazam's value is raw
   offensive pressure plus Thunder Wave/Recover.
3. **More BC signal.** 1,772 fully-revealed Alakazam sets vs 237 for Rhydon.

Move sets are the modal fully-revealed 4-move set per species in the same pool
(percentages in the table above), so every slot is the dominant human build.

### Running on the fixed roster (M12 Phase 1)

`--format gen1ou` is enough everywhere — it implies the pre-registered roster and
hands the **same** packed team to both seats. `--roster <file>` overrides which
file. Random-team formats are untouched: with no `--format`, everything behaves
exactly as it did before.

```bash
# Bot eval (Phase 4 shape; add --greedy for a decision-rule read)
python models/evaluate.py --model ppo --obs-v3 --format gen1ou \
    --checkpoint <ckpt> --opponent random      --battles 5000 --num-envs 8
python models/evaluate.py --model ppo --obs-v3 --format gen1ou \
    --checkpoint <ckpt> --opponent damagefirst --battles 5000 --num-envs 8

# BC (Phase 2) — mixed corpus, the pre-registered M7 recipe
python models/bc_pretrain_mlp.py --obs-v3 --formats gen1randombattle,gen1ou \
    --epochs 5 --out models/checkpoints/bc_mlp_gen1ou_fixed.pt

# PPO (Phase 3) — 5M steps, M7 recipe, single arm
python models/ppo/train.py --obs-v3 --format gen1ou --steps 5000000 \
    --num-envs 8 --opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2" \
    --bc-anchor <bc.pt> --value-warmup-steps 200000 \
    --checkpoint-dir models/ppo/checkpoints/m12

# Ladder (Phase 5, OPTIONAL) — uploads the team via /utm before each search
node tools/ladder-bot/ladder-bot.js --login-file config/showdown_login.txt \
    --format gen1ou --checkpoint <ckpt> --battles 360 --mcts
```

`evaluate.py` prints `Format: gen1ou | FIXED ROSTER (both sides)` in its results
header, and the ladder bot's banner prints `format=gen1ou (fixed roster)`, so no
run is ambiguous about what it played.

**MCTS determinization is roster-aware.** Under a mirror roster the opponent's
bench is not hidden information, so search fills it from the roster instead of
sampling. Without this, `gen1ou` silently falls through to the gen1 **random**
generator (`Teams.getGenerator` does not throw for non-random formats), and
search would model the opponent as arbitrary Gen 1 Pokémon. The ladder path
(`BattleSim.fromTracked`) still samples on purpose — a human opponent brings
their own team.

### ⚠️ Two caveats recorded with the selection

**Win rate by team is NOT usable, and was discarded.** Extracting a team requires
all 6 Pokémon to appear in the log, and winners frequently never bring in their
6th. Measured directly: P(win) is 44.5% given 6 revealed vs 81.2% given 5. The
"6 revealed" filter **selects for losing teams**, so any win rate computed this
way is biased downward and cannot rank rosters. Usage counts are unaffected by
this in any way that matters — they concern which species appear at all.

**Expect long mirror matches, and DRAWS.** Measured at n=600 (Phase 1,
`RandomPlayerAI` both sides): mean **111 turns** and **6.3% draws**. Chansey's
Soft-Boiled, Starmie/Alakazam's Recover and Snorlax's Rest on *both* sides make
this both slower and more drawish than randbats, where draws were negligible.

Two consequences:
- **Phase 4 must report draws explicitly.** Win rate + loss rate will not sum to
  1, and the ≥10% gate reads as a share of *all* games.
- **Fewer episodes per 5M PPO steps** in Phase 3, and longer ladder wall-clock if
  the optional Phase 5 runs.

Same run also gives a clean **seat-bias** reading, which a mirror roster makes
possible for the first time: p1 won **49.2%**, 95% CI [45.2, 53.2] — covers 50%,
no detectable bias.

---

## What a format ID is

A format ID is the lowercase, no-spaces version of the format name. It is passed to:
- `>start {"formatid":"..."}` in the battle stream input
- `Teams.generate(formatId)` to generate a random team
- `TeamValidator(formatId)` to validate a team
- `./pokemon-showdown generate-team <formatId>` on the command line

---

## Generation Random Battles

Recommended for ML training: teams are auto-generated, no team-building required. Use `Teams.generate(formatId)` to get a valid team for any random format.

| Format ID | Name | Notes |
|-----------|------|-------|
| `gen1randombattle` | [Gen 1] Random Battle | Simplest: no items, no abilities, no natures, DVs instead of IVs |
| `gen2randombattle` | [Gen 2] Random Battle | Adds held items; still no abilities or natures |
| `gen3randombattle` | [Gen 3] Random Battle | Adds abilities, natures, proper IVs/EVs |
| `gen4randombattle` | [Gen 4] Random Battle | Physical/Special split (moves have a category, not just type) |
| `gen5randombattle` | [Gen 5] Random Battle | Adds weather-centric metagame, new abilities |
| `gen6randombattle` | [Gen 6] Random Battle | Adds Mega Evolution |
| `gen7randombattle` | [Gen 7] Random Battle | Adds Z-Moves, Alolan forms |
| `gen8randombattle` | [Gen 8] Random Battle | Adds Dynamax/Gigantamax |
| `gen9randombattle` | [Gen 9] Random Battle | Adds Terastallization |

---

## Standard Competitive Formats (require team construction)

These formats require you to supply a valid team. Use `Teams.pack()` or `Teams.import()` to prepare the team string, and `TeamValidator` to confirm legality before battle.

| Format ID | Notes |
|-----------|-------|
| `gen1ou` | Gen 1 OU — no items, no abilities, DVs |
| `gen2ou` | Gen 2 OU — introduces held items |
| `gen3ou` | Gen 3 OU — abilities, natures |
| `gen4ou` | Gen 4 OU — Physical/Special split |
| `gen5ou` | Gen 5 OU |
| `gen6ou` | Gen 6 OU — Mega Evolution |
| `gen7ou` | Gen 7 OU — Z-Moves |
| `gen8ou` | Gen 8 OU — Dynamax |
| `gen9ou` | Gen 9 OU — Terastallization |

---

## Custom Game Formats (no restrictions)

These formats bypass legality checks — any Pokemon, any move, any item. Useful for scripted scenarios like `simulate.js` gym battles where teams are hand-constructed.

| Format ID | Gen | Notes |
|-----------|-----|-------|
| `gen1customgame` | 1 | Used in simulate.js for Gen 1 gym battles |
| `gen2customgame` | 2 | Used in simulate.js for Gen 2 gym battles |
| `gen3customgame` | 3 | Used in simulate.js for Gen 3 gym battles |
| `gen4customgame` | 4 | Used in simulate.js for Gen 4 gym battles |
| `gen5customgame` | 5 | Used in simulate.js for Gen 5 gym battles |
| `gen7customgame` | 7 | Often used as the "no rules" baseline |
| `gen9customgame` | 9 | Modern no-restrictions format |

`simulate.js` uses `gen1customgame` through `gen5customgame` for its gym leader battles since team legality is irrelevant — the teams are packed directly with `packMon()`.

---

## Listing all available formats

```javascript
const { Dex } = require('./dist/sim/index');
const formats = Dex.formats.all();
formats.forEach(f => console.log(f.id, f.name));
```

---

## Format object structure

```javascript
const { Dex } = require('./dist/sim/index');
const fmt = Dex.formats.get('gen1randombattle');

fmt.id           // 'gen1randombattle'
fmt.name         // '[Gen 1] Random Battle'
fmt.mod          // 'gen1' — which generation's ruleset to use
fmt.team         // 'random' for random formats; undefined for constructed-team formats
fmt.ruleset      // array of active rule strings, e.g. ['Species Clause', 'Sleep Clause Mod', ...]
fmt.banlist      // array of banned items/abilities/moves/Pokemon
fmt.gameType     // 'singles' (default), 'doubles', 'triples', etc.
```

---

## Custom format definition (from config/formats.ts)

Formats are defined in `config/formats.ts` as entries in the `Formats` array. To add your own, create `config/custom-formats.ts`:

```typescript
export const Formats: FormatList = [
  {
    name: "[Gen 1] My Custom Format",
    mod: 'gen1',
    ruleset: ['Obtainable', 'Species Clause', 'Sleep Clause Mod', 'Freeze Clause Mod'],
    banlist: ['Uber'],
  },
];
```

Key fields:
- `name` — display name; format ID is derived by lowercasing and removing spaces/brackets
- `mod` — generation module: `'gen1'` through `'gen9'`
- `team` — set to `'random'` to enable random team generation
- `ruleset` — active rule IDs; can inherit another format's rules by referencing its name (e.g., `'[Gen 9] OU'`)
- `banlist` — banned Pokemon (by name), moves, items, or abilities

---

## Recommended progression for ML training

**Step 1: Start with `gen1randombattle`**
- Fewest mechanics: no items, no abilities, no natures, no Physical/Special split
- Fastest battles (fewer decision points)
- State space is smallest — easiest policy to learn
- DVs max at 15; stat experience maxes at 252 per stat

**Step 2: Move to `gen3randombattle`**
- Adds abilities (passive effects, significant strategic depth)
- Adds natures (stat modifiers), proper IVs/EVs
- Physical/Special split not yet present (all moves still type-based)
- Substantially more interesting decisions without the complexity spike of Gen 4+

**Step 3: `gen7randombattle` or `gen9randombattle`**
- Gen 7 has the most community battle data available (Smogon, Showdown replays)
- Gen 9 is the current format with Terastallization as a major decision axis

---

## Generating teams for a format

```javascript
const { Teams } = require('./dist/sim/index');

// Generate a random team (for random battle formats)
const team = Teams.generate('gen1randombattle');
const packed = Teams.pack(team);  // convert to packed string for the battle stream

// With a fixed seed (for reproducible experiments)
const seededTeam = Teams.generate('gen3randombattle', { seed: [1, 2, 3, 4] });
```

Command-line equivalent:
```bash
./pokemon-showdown generate-team gen1randombattle
./pokemon-showdown generate-team gen3randombattle
```

---

## Using a format in a battle stream

```javascript
const { BattleStream, getPlayerStreams } = require('./dist/sim/index');
const { Teams } = require('./dist/sim/index');

const stream = new BattleStream();
const streams = getPlayerStreams(stream);

const formatId = 'gen1randombattle';
const p1Team = Teams.pack(Teams.generate(formatId));
const p2Team = Teams.pack(Teams.generate(formatId));

stream.write(`>start {"formatid":"${formatId}"}`);
stream.write(`>player p1 {"name":"Player1","team":"${p1Team}"}`);
stream.write(`>player p2 {"name":"Player2","team":"${p2Team}"}`);
```

For constructed teams (non-random formats), replace the `Teams.generate()` call with your own `PokemonSet[]` array passed through `Teams.pack()`.

---

## Gen-specific format naming pattern

All format IDs follow one of two patterns:

- `gen{N}randombattle` — random teams, no construction needed
- `gen{N}{tier}` — constructed teams, where tier is `ou`, `uu`, `ru`, `nu`, `pu`, `lc`, `ubers`, `customgame`, etc.

Modded formats (Stadium, Let's Go) use the corresponding gen number: Let's Go = `gen7`, Stadium = gen-appropriate number.
