# Battle Formats

Available battle formats in this Pokemon Showdown simulator and how to choose one for ML training.
Relevant source: `config/formats.ts`, `sim/TEAMS.md`, `simulate.js`.

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
