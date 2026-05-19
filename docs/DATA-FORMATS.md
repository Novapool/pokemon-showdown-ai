# Data Formats

All data formats you will encounter when working with this Pokemon Showdown simulator.
Relevant source: `sim/SIM-PROTOCOL.md`, `sim/TEAMS.md`, `simulate.js`.

---

## Section 1: Battle Protocol Output (pipe-delimited)

The battle stream emits newline-separated protocol messages. Each line starts with `|`. Messages are produced by `BattleStream` from `dist/sim/index`.

### Battle initialization sequence

```
|player|p1|PlayerName|60|1200
|player|p2|OpponentName|113|1300
|teamsize|p1|6
|teamsize|p2|6
|gametype|singles
|gen|1
|tier|[Gen 1] OU
|rule|Sleep Clause Mod: Limit one foe to sleep at a time
|start
```

Key initialization messages:

- `|player|PLAYER|USERNAME|AVATAR|RATING` — player identifiers; PLAYER is `p1` or `p2`
- `|teamsize|PLAYER|NUMBER` — how many Pokemon each side starts with
- `|gametype|GAMETYPE` — `singles`, `doubles`, `triples`, `multi`, or `freeforall`
- `|gen|GENNUM` — generation number 1–9
- `|tier|FORMATNAME` — human-readable format name
- `|rule|RULE: DESCRIPTION` — active rules (appears once per rule)
- `|start` — battle has begun

If the format uses Team Preview, `|clearpoke`, `|poke|PLAYER|DETAILS|ITEM`, and `|teampreview` will appear before `|start`.

### Identifying Pokemon in protocol messages

Pokemon are identified as `POSITION: NAME` where POSITION is `p1a`, `p2a`, etc. In singles, the position letter is always `a`. In doubles it can be `a` or `b`. In triples, `a`, `b`, or `c`.

DETAILS strings (used in `|switch|`, `|drag|`, `|detailschange|`) take the form:
```
SPECIES, L##, M/F, shiny
```
Level is omitted if 100. Gender is omitted if genderless. Example: `Sawsbuck, L50, F, shiny`.

### Major action messages

These are the most important protocol lines for ML feature extraction:

| Message | Format | Meaning |
|---------|--------|---------|
| `\|turn\|` | `\|turn\|NUMBER` | Turn counter incremented |
| `\|move\|` | `\|move\|POKEMON\|MOVE\|TARGET` | Pokemon used a move |
| `\|switch\|` | `\|switch\|POKEMON\|DETAILS\|HP STATUS` | Intentional switch |
| `\|drag\|` | `\|drag\|POKEMON\|DETAILS\|HP STATUS` | Forced switch (Whirlwind, Roar) |
| `\|faint\|` | `\|faint\|POKEMON` | Pokemon fainted |
| `\|cant\|` | `\|cant\|POKEMON\|REASON` | Pokemon could not move (paralysis, sleep, etc.) |
| `\|win\|` | `\|win\|USER` | Battle winner (USER matches the username from `\|player\|`) |
| `\|tie\|` | `\|tie` | Battle ended in a tie |

HP STATUS format: own Pokemon shows `CURRENT/MAX`, opponent shows percentage `/100` (if HP Percentage Mod is active) or `/48` otherwise. STATUS suffix can be blank, `slp`, `par`, `brn`, `frz`, `psn`, or `tox`.

If `|[miss]` appears after `|move|`, the move missed.

### Minor action messages

Minor actions describe the effects of moves and abilities:

| Message | Format | Meaning |
|---------|--------|---------|
| `\|-damage\|` | `\|-damage\|POKEMON\|HP STATUS` | Damage dealt |
| `\|-heal\|` | `\|-heal\|POKEMON\|HP STATUS` | HP restored |
| `\|-sethp\|` | `\|-sethp\|POKEMON\|HP` | HP set to exact value |
| `\|-status\|` | `\|-status\|POKEMON\|STATUS` | Status condition applied |
| `\|-curestatus\|` | `\|-curestatus\|POKEMON\|STATUS` | Status condition cured |
| `\|-boost\|` | `\|-boost\|POKEMON\|STAT\|AMOUNT` | Stat raised by AMOUNT stages |
| `\|-unboost\|` | `\|-unboost\|POKEMON\|STAT\|AMOUNT` | Stat lowered by AMOUNT stages |
| `\|-setboost\|` | `\|-setboost\|POKEMON\|STAT\|AMOUNT` | Stat set to AMOUNT stages |
| `\|-weather\|` | `\|-weather\|WEATHER` | Weather changed or continued |
| `\|-fieldstart\|` | `\|-fieldstart\|CONDITION` | Field condition began (Trick Room, Terrain) |
| `\|-fieldend\|` | `\|-fieldend\|CONDITION` | Field condition ended |
| `\|-sidestart\|` | `\|-sidestart\|SIDE\|CONDITION` | Side condition began (Stealth Rock, Reflect) |
| `\|-sideend\|` | `\|-sideend\|SIDE\|CONDITION` | Side condition ended |
| `\|-fail\|` | `\|-fail\|POKEMON\|ACTION` | Move failed due to its own mechanics |
| `\|-miss\|` | `\|-miss\|SOURCE\|TARGET` | Move missed |

Stat abbreviations used in boost messages: `atk`, `def`, `spa`, `spd`, `spe`, `accuracy`, `evasion`.

Status abbreviations: `brn` (burn), `par` (paralysis), `slp` (sleep), `frz` (freeze), `psn` (poison), `tox` (badly poisoned).

### Hidden info and split messages

When the simulator runs with hidden information (the default for competitive play), some messages appear in `|split|p1` blocks followed by two variants: one for the target player, one for the spectator. For omniscient data collection (ML training), use the omniscient stream or parse both variants. In `simulate.js`, the BattleStream is consumed directly, which gives the full omniscient view.

### Complete example (Gen 1 singles, first few turns)

```
|player|p1|Bulbasaur|60|
|player|p2|Brock|113|
|teamsize|p1|1
|teamsize|p2|2
|gametype|singles
|gen|1
|tier|[Gen 1] Custom Game
|start
|switch|p1a: Bulbasaur|Bulbasaur, L14|45/45
|switch|p2a: Geodude|Geodude, L12|40/40
|turn|1
|move|p1a: Bulbasaur|Vine Whip|p2a: Geodude
|-damage|p2a: Geodude|25/40
|move|p2a: Geodude|Rock Throw|p1a: Bulbasaur
|-damage|p1a: Bulbasaur|30/45
|turn|2
...
|faint|p2a: Geodude
|switch|p2a: Onix|Onix, L14|85/85
...
|win|Bulbasaur
```

---

## Section 2: Team Formats

Source: `sim/TEAMS.md`. Three representations exist; all can be converted through the `Teams` API.

### Packed format

Used in simulator API calls and command-line tools. Multiple Pokemon separated by `]`, each Pokemon encoded as:

```
NICKNAME|SPECIES|ITEM|ABILITY|MOVES|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|HAPPINESS,POKEBALL,HIDDENPOWERTYPE,GIGANTAMAX,DYNAMAXLEVEL,TERATYPE
```

Rules:
- `SPECIES` is blank if identical to `NICKNAME`
- `ABILITY` can be `0`, `1`, `H` (slot index) or an ability ID string
- `MOVES` is a comma-separated list of move IDs (lowercase, no spaces)
- `EVS` and `IVS` are comma-separated in order: HP, Atk, Def, SpA, SpD, Spe
- Blank EVs default to 0; blank IVs default to 31
- If all EVs or IVs are blank, all commas can be omitted
- `SHINY` is `S` for shiny, blank otherwise
- `LEVEL` is blank for level 100
- `HAPPINESS` is blank for 255 (max)
- If the trailing fields (POKEBALL through TERATYPE) are all blank, their commas are omitted

Example (Articuno):
```
Articuno||leftovers|pressure|icebeam,hurricane,substitute,roost|Modest|252,,,252,4,||,,,30,30,|||
```

### JSON format (PokemonSet)

Used internally throughout the codebase. TypeScript type is `PokemonSet[]`.

```json
{
  "name": "",
  "species": "Articuno",
  "gender": "",
  "item": "Leftovers",
  "ability": "Pressure",
  "evs": {"hp": 252, "atk": 0, "def": 0, "spa": 252, "spd": 4, "spe": 0},
  "nature": "Modest",
  "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 30, "spd": 30, "spe": 31},
  "moves": ["Ice Beam", "Hurricane", "Substitute", "Roost"]
}
```

### Export (human-readable) format

Used only by the client for teambuilder import/export. Example:

```
Articuno @ Leftovers
Ability: Pressure
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
IVs: 30 SpA / 30 SpD
- Ice Beam
- Hurricane
- Substitute
- Roost
```

### Converting between formats

```javascript
const { Teams } = require('./dist/sim/index');

Teams.unpack(packedStr)    // packed string → PokemonSet[]
Teams.pack(jsonArray)      // PokemonSet[] → packed string
Teams.import(anyStr)       // any string format → PokemonSet[]
Teams.export(jsonArray)    // PokemonSet[] → human-readable export string
Teams.exportSet(set)       // single PokemonSet → export string
Teams.generate('gen1randombattle')  // generate random valid team as PokemonSet[]
Teams.generate('gen3randombattle', { seed: [1, 2, 3, 4] })  // seeded generation
```

Round-trip example (packed to export):
```javascript
Teams.export(Teams.unpack(packedStr))
```

### Generation-specific quirks (important for simulate.js)

Gen 1:
- No held items — leave `ITEM` blank
- No abilities — leave `ABILITY` blank
- No natures — leave `NATURE` blank
- DVs instead of IVs: IVs are divided by 2 (rounded down) to produce DVs, so blank IVs (31) become 15 DVs (max)
- Stat experience instead of EVs: max is 252 per stat

Gen 2:
- Has held items, no abilities, no natures
- Same DV and stat experience system as Gen 1

Gen 3+:
- Full EVs (0–252 per stat, 508 total), IVs (0–31), abilities, natures
- Blank EVs = 0, blank IVs = 31

In `simulate.js`, the `g1mon()` and `g2mon()` helpers set `MAX_EVS = [252, 252, 252, 252, 252, 252]` and leave IVs null (blank = 15 DVs). The `gXmon()` helper leaves both EVs and IVs null for Gen 3–5 (0 EVs, 31 IVs).

### Team validation

```javascript
const { Teams, TeamValidator } = require('./dist/sim/index');
const validator = new TeamValidator('gen6nu');
const problems = validator.validateTeam(Teams.unpack(packedStr));
// problems is null if legal, or an array of problem strings
```

---

## Section 3: CSV Output Schema

`simulate.js` produces three CSV files in `output/` (path configured via `OUTPUT_DIR`). The simulation runs 1000 iterations × 5 starters × 8 gyms × 10 battles = 400,000 total battles.

### output/raw_battles.csv

One row per individual battle. Header:
```
iteration,starter_line,gym_number,battle_run,evolution_stage,gym_leader_name,starter_won,gym_leader_pokemon_defeated,gym_leader_pokemon_total,turns
```

| Column | Type | Description |
|--------|------|-------------|
| `iteration` | int (1–1000) | Experiment iteration index |
| `starter_line` | string | Starter identifier (e.g., `Bulbasaur`, `Cyndaquil`) |
| `gym_number` | int (1–8) | Gym index |
| `battle_run` | int (1–10) | Battle number within this gym for this iteration |
| `evolution_stage` | string | `Base`, `Middle`, or `Final` |
| `gym_leader_name` | string | Gym leader name (e.g., `Brock`, `Misty`) |
| `starter_won` | int (0 or 1) | 1 if starter's side won |
| `gym_leader_pokemon_defeated` | int | Number of gym leader's Pokemon KO'd by starter |
| `gym_leader_pokemon_total` | int | Total size of gym leader's team |
| `turns` | int | Battle duration in turns |

Example row:
```
1,Bulbasaur,1,1,Base,Brock,1,2,2,5
```

### output/summary_stats.csv

One row per starter line, aggregated across all 1000 iterations. Header:
```
starter_line,avg_ko_ratio_sum,avg_battle_duration_turns,avg_ranking_score
```

| Column | Type | Description |
|--------|------|-------------|
| `starter_line` | string | Starter identifier |
| `avg_ko_ratio_sum` | float | Mean of per-iteration KO ratio sums across all 8 gyms |
| `avg_battle_duration_turns` | float | Mean turns per battle across all gyms and iterations |
| `avg_ranking_score` | float | Composite ranking score (mean of per-iteration ranking points) |

### output/gym_rankings.csv

One row per (gym, starter) pair. Header:
```
gym_number,starter_line,avg_performance_metric,points_awarded
```

| Column | Type | Description |
|--------|------|-------------|
| `gym_number` | int (1–8) | Gym index |
| `starter_line` | string | Starter identifier |
| `avg_performance_metric` | float | Mean win rate (if any wins occurred) or mean KO ratio |
| `points_awarded` | int (1–5) | Ranking points: 5 = best, 1 = worst, ties share a rank |

The performance metric falls back to KO ratio when all battles were losses (win rate = 0), allowing partial-credit ranking even without wins.
