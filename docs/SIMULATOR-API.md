# Simulator API Reference

Node.js API for running Pokemon battles programmatically. Used for ML training data generation and agent evaluation.

> **Note:** This document covers the Node.js API for running battles programmatically — `BattleStream`, players, teams, and the Dex. It is the primary reference for ML training use. `sim/SIMULATOR.md` and `sim/SIM-PROTOCOL.md` contain the upstream protocol specification (raw stdin/stdout message format) — consult them for low-level protocol details not covered here.

## Build First

The simulator runs from compiled output. Always build before running:

```bash
node build
```

Compiled files land in `dist/`. All `require()` paths below point into `dist/`.

---

## Imports

```javascript
const { BattleStream, getPlayerStreams, Teams, Dex, PRNG } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');
```

---

## BattleStream

`BattleStream` is an `ObjectReadWriteStream<string>`. You write command strings to it and read protocol output strings from it. It manages one battle per instance.

```javascript
const battleStream = new BattleStream();
```

Constructor options (all optional):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `debug` | boolean | false | Emit verbose debug output |
| `noCatch` | boolean | false | Don't swallow errors from battle engine |
| `keepAlive` | boolean | false | Don't close stream when battle ends |
| `replay` | boolean or `'spectator'` | false | Emit replay-formatted output |

---

## getPlayerStreams

Splits a single `BattleStream` into six named sub-streams:

```javascript
const streams = getPlayerStreams(battleStream);
// streams.omniscient — sees all hidden info (exact HP, opponent team, etc.)
// streams.spectator  — public-only view (what a spectator sees)
// streams.p1         — player 1's private view + receives choice requests
// streams.p2         — player 2's private view + receives choice requests
// streams.p3         — player 3 (doubles/multi only)
// streams.p4         — player 4 (doubles/multi only)
```

For ML training, read from `streams.omniscient` to collect full state. For fair evaluation, read from `streams.p1` or `streams.p2` to simulate an agent with only its own knowledge.

The sub-streams are all `ObjectReadWriteStream<string>`. Writing to `streams.p1` automatically prepends `>p1 ` to each line before forwarding to the underlying `BattleStream`.

---

## Starting a Battle

Write initialization commands to `streams.omniscient` (or directly to the `BattleStream`). Commands must start with `>`.

```javascript
// Minimal: random teams, no seed
streams.omniscient.write(`>start {"formatid":"gen7randombattle"}`);
streams.omniscient.write(`>player p1 {"name":"Bot1"}`);
streams.omniscient.write(`>player p2 {"name":"Bot2"}`);

// With explicit team and reproducible seed
streams.omniscient.write(
  `>start ${JSON.stringify({ formatid: 'gen1customgame', seed: [1, 2, 3, 4] })}\n` +
  `>player p1 ${JSON.stringify({ name: 'Bot1', team: packedTeamString })}\n` +
  `>player p2 ${JSON.stringify({ name: 'Bot2', team: packedTeamString })}`
);
```

`>start` OPTIONS fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `formatid` | string | Yes | Format ID (e.g. `"gen1ou"`, `"gen7randombattle"`) |
| `seed` | number[4] | No | PRNG seed for reproducibility |
| `p1` | PLAYEROPTIONS | No | Inline player 1 options (skips `>player p1` step) |
| `p2` | PLAYEROPTIONS | No | Inline player 2 options |

`>player PLAYERID` PLAYEROPTIONS fields:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Display name |
| `avatar` | string | Avatar identifier |
| `team` | string or null | Packed team string. Omit or pass `null` for random formats |

> Note: teams are not validated by the simulator. Use `TeamValidator` before passing a custom team if correctness matters.

---

## Reading Output

Iterate the omniscient stream with `for await`. Each `chunk` is a newline-delimited string of PS battle protocol lines.

```javascript
for await (const chunk of streams.omniscient) {
  for (const line of chunk.split('\n')) {
    // Line format: |TYPE|ARG1|ARG2|...
    if (line.startsWith('|win|')) {
      const winner = line.slice(5).trim();  // player name
    }
    if (line.startsWith('|turn|')) {
      const turnNum = parseInt(line.split('|')[2]);
    }
    if (line.startsWith('|faint|p2')) {
      // An opposing Pokemon fainted
    }
    if (line.startsWith('|move|')) {
      // A move was used: |move|POKEMON|MOVE|TARGET
    }
  }
}
// Loop exits when the battle ends (stream closes)
```

Key protocol lines for training data extraction:

| Line pattern | Meaning |
|--------------|---------|
| `\|win\|NAME` | Battle over, NAME won |
| `\|turn\|N` | Turn N started |
| `\|move\|MON\|MOVE\|TARGET` | Move used |
| `\|switch\|MON\|SPECIES\|HP` | Pokemon switched in |
| `\|faint\|MON` | Pokemon fainted |
| `\|request\|JSON` | Choice request sent to player (via sideupdate) |
| `\|-damage\|MON\|HP` | Damage dealt |
| `\|-heal\|MON\|HP` | HP restored |

Full protocol documentation: `sim/SIM-PROTOCOL.md`

---

## Player Choices

Write choices to the player's sub-stream. The stream prepends `>p1 ` (or `>p2`) automatically.

```javascript
streams.p1.write('move 1');      // use move in slot 1
streams.p1.write('move 3');      // use move in slot 3
streams.p1.write('switch 2');    // switch to Pokemon in party slot 2
streams.p1.write('team 123456'); // team preview order (lead with slot 1, then 2, etc.)
```

Choice requests arrive on the player's stream as `|request|JSON`. Parse the JSON to see available moves and switches:

```javascript
for await (const chunk of streams.p1) {
  for (const line of chunk.split('\n')) {
    if (line.startsWith('|request|')) {
      const request = JSON.parse(line.slice(9));
      // request.active[0].moves — array of available moves
      // request.side.pokemon    — full party data
    }
  }
}
```

---

## BattlePlayer (Base Class for AI)

`BattlePlayer` is an abstract base class for implementing AI agents. Subclass it and implement `receiveRequest`:

```javascript
import { BattlePlayer } from './dist/sim/battle-stream';

class MyAI extends BattlePlayer {
  receiveRequest(request) {
    // request.active[0].moves = [{move, id, pp, maxpp, target, disabled}, ...]
    // request.side.pokemon = [{ident, details, condition, ...}, ...]
    if (request.active) {
      const moves = request.active[0].moves.filter(m => !m.disabled);
      const choice = moves[0];  // pick first legal move
      this.choose(`move ${choice.id}`);
    } else if (request.forceSwitch) {
      this.choose('switch 2');
    }
  }
}

const ai = new MyAI(streams.p1);
void ai.start();  // begins listening loop; resolves when stream closes
```

`RandomPlayerAI` (built-in) picks moves randomly with optional move/switch probability weighting:

```javascript
const ai = new RandomPlayerAI(streams.p1, {
  seed: [1, 2, 3, 4],   // PRNG seed for reproducibility
  move: 0.7,             // probability of choosing move vs. switch (default varies)
});
void ai.start();
```

---

## Teams API

```javascript
// Generate a random team for a format
const team = Teams.generate('gen7randombattle');  // returns PokemonSet[]

// Convert between formats
const packed = Teams.pack(team);           // PokemonSet[] → packed string
const unpacked = Teams.unpack(packed);     // packed string → PokemonSet[]
const imported = Teams.import(exportText); // PS export format → PokemonSet[]

// Packed format is what you pass to >player as "team"
streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'Bot', team: packed })}`);
```

Packed team string format (per-Pokemon, joined by `]`):

```
NICKNAME|SPECIES|ITEM|ABILITY|MOVE1,MOVE2,MOVE3,MOVE4|NATURE|EVS|GENDER|IVS|SHINY|LEVEL|
```

Manual packing for custom teams (used in `simulate.js`):

```javascript
function packMon(species, item, ability, moves, nature, evs, ivs, level) {
  const evsStr = evs ? evs.join(',') : '';
  const ivsStr = ivs ? ivs.join(',') : '';
  return `${species}||${item}|${ability}|${moves.join(',')}|${nature}|${evsStr}||${ivsStr}||${level}|`;
}
function packTeam(mons) { return mons.join(']'); }
```

---

## Dex API

Lookup game data for any generation:

```javascript
const dex = Dex.mod('gen1');   // gen-specific dex
// Also: 'gen2', 'gen3', 'gen4', 'gen5', 'gen6', 'gen7', 'gen8', 'gen9'
// Or use the default (latest gen): Dex directly

// Move data
const tbolt = dex.moves.get('thunderbolt');
// tbolt.basePower   → 90
// tbolt.type        → 'Electric'
// tbolt.category    → 'Special'
// tbolt.accuracy    → 100
// tbolt.pp          → 15

// Species data
const pikachu = dex.species.get('pikachu');
// pikachu.baseStats  → { hp: 35, atk: 55, def: 30, spa: 50, spd: 40, spe: 90 }
// pikachu.types      → ['Electric']
// pikachu.abilities  → { 0: 'Static', H: 'Lightning Rod' }

// Ability data
const intimidate = dex.abilities.get('intimidate');
// intimidate.desc → "..."

// Item data
const leftovers = dex.items.get('leftovers');
// leftovers.desc → "..."
```

The `DamageFirstAI` in `simulate.js` uses the Dex to choose highest base-power moves:

```javascript
const moveData = dex.moves.get(moveName);
const power = moveData ? (moveData.basePower || 0) : 0;
```

---

## PRNG

Seeded pseudo-random number generator. Seeds are 4-element integer arrays. Same seed always produces the same sequence.

```javascript
const prng = new PRNG([1, 2, 3, 4]);

prng.next()           // float in [0, 1)
prng.next(6)          // integer in [0, 6) — i.e. 0..5
prng.next(1, 6)       // integer in [1, 6)
prng.sample(array)    // random element from array
prng.getSeed()        // returns current seed as number[4]
```

Pass seeds to the battle start command and to AI constructors for fully reproducible runs:

```javascript
const seed = [iter, gymIdx, battleIdx, playerIdx];
// Battle engine uses this seed for all random events (crits, damage rolls, etc.)
streams.omniscient.write(`>start ${JSON.stringify({ formatid, seed })}`);
// AI uses a separate seed (XOR'd to avoid correlation)
const p1 = new RandomPlayerAI(streams.p1, { seed: [seed[0] ^ 0xAAAA, seed[1], seed[2], seed[3]] });
```

---

## Minimal Working Example

```javascript
const { BattleStream, getPlayerStreams, Teams } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');

async function runOneBattle() {
  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);

  const p1 = new RandomPlayerAI(streams.p1);
  const p2 = new RandomPlayerAI(streams.p2);
  void p1.start();
  void p2.start();

  void streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: 'gen7randombattle' })}\n` +
    `>player p1 ${JSON.stringify({ name: 'Bot1' })}\n` +
    `>player p2 ${JSON.stringify({ name: 'Bot2' })}`
  );

  let winner = null;
  let turns = 0;
  for await (const chunk of streams.omniscient) {
    for (const line of chunk.split('\n')) {
      if (line.startsWith('|win|')) winner = line.slice(5).trim();
      if (line.startsWith('|turn|')) turns = parseInt(line.split('|')[2]);
    }
  }
  return { winner, turns };
}

runOneBattle().then(console.log);
```

---

## Format IDs

Common format IDs for ML training:

| ID | Description |
|----|-------------|
| `gen1randombattle` | Gen 1 random teams |
| `gen1customgame` | Gen 1 custom teams (no validation) |
| `gen2randombattle` | Gen 2 random teams |
| `gen2customgame` | Gen 2 custom teams |
| `gen3randombattle` | Gen 3 random teams |
| `gen3customgame` | Gen 3 custom teams |
| `gen4randombattle` | Gen 4 random teams |
| `gen4customgame` | Gen 4 custom teams |
| `gen5randombattle` | Gen 5 random teams |
| `gen5customgame` | Gen 5 custom teams |
| `gen7randombattle` | Gen 7 random teams |
| `gen7customgame` | Gen 7 custom teams |
| `gen8randombattle` | Gen 8 random teams |
| `gen9randombattle` | Gen 9 random teams |

`customgame` formats skip team validation and Clause enforcement, making them ideal for scripted training scenarios.

See `sim/tools/multi-random-runner.ts` for the canonical list of tested random formats.
