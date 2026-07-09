# AI Players — Creating Custom Pokemon Battle Agents

## Overview

All AI players extend `RandomPlayerAI`, which handles the full request/response lifecycle with the battle stream. You only need to override the three decision methods. The base class lives at `sim/tools/random-player-ai.ts`; import from the compiled output at runtime.

```javascript
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');
```

## How the Base Class Works

`RandomPlayerAI` extends `BattlePlayer`, which reads from a player stream and calls `receiveRequest(request)` whenever the simulator needs a decision. `receiveRequest` handles three request types:

- `request.wait` — opponent is deciding; do nothing
- `request.forceSwitch` — one or more Pokemon fainted; must switch in replacements
- `request.teamPreview` — lead order selection before battle begins
- `request.active` — normal turn; choose moves or switches for each active Pokemon

After resolving which options are legal, `receiveRequest` delegates to `chooseMove`, `chooseSwitch`, or `chooseTeamPreview` and calls `this.choose(decisionString)` with the result.

**You never call `receiveRequest` yourself.** Override the three `choose*` methods and let the base class handle protocol details.

## Constructor

```javascript
new MyAI(playerStream, options)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `seed` | `PRNG \| PRNGSeed \| null` | random | PRNG seed; pass a fixed `[n,n,n,n]` array for reproducible runs |
| `move` | `number` | `1.0` | Probability of choosing a move vs switching. 0 = always switch, 1.0 = always move (0–1) |
| `mega` | `number` | `0` | Probability of mega-evolving / dynamaxing / terastallizing when eligible (0–1) |

`PRNGSeed` is a 4-element array of integers: `[a, b, c, d]`.

## Methods to Override

### `chooseMove(active, moves)`

Called when the Pokemon needs to select an attack or Z-move. Must return one of the `choice` strings from the `moves` array.

```typescript
protected chooseMove(
  active: AnyObject,          // the active Pokemon's request slice
  moves: { choice: string, move: AnyObject }[]
): string
```

`moves` is already filtered to legal, non-disabled options. Each element:

```
{
  choice: "move 1"           // string to return; may include target or "zmove" suffix
  move: {
    slot: number,            // 1-based move slot
    move: string,            // move id, e.g. "thunderbolt"
    target: string,          // "normal", "self", "adjacentAlly", etc.
    zMove: boolean,
  }
}
```

In doubles, the base class automatically appends a target slot to choice strings for moves that need one. You do not have to handle targeting.

Default behavior: `return this.prng.sample(moves).choice;`

### `chooseSwitch(active, switches)`

Called when voluntarily switching (move probability check) or when forced to switch. Must return a slot number.

```typescript
protected chooseSwitch(
  active: AnyObject | undefined,   // undefined on forced switch
  switches: { slot: number, pokemon: AnyObject }[]
): number
```

`switches` is pre-filtered: no fainted, no currently active, no already-chosen slots.

Each `pokemon` object in the array has the same shape as `request.side.pokemon[i]` — species, condition (HP string), status, moves, stats, etc.

Default behavior: `return this.prng.sample(switches).slot;`

### `chooseTeamPreview(team)`

Called once at the start if the format uses team preview. Must return an order string.

```typescript
protected chooseTeamPreview(team: AnyObject[]): string
```

`team` is `request.side.pokemon` — all six Pokemon with full data.

Return `"default"` to use the submitted order, or a string like `"1 2 3 4 5 6"` (1-based slots).

Default behavior: `return 'default';`

## Wiring an AI into a Battle

```javascript
const { BattleStream, getPlayerStreams } = require('./dist/sim/index');

const streams = getPlayerStreams(new BattleStream());

const p1 = new MyAI(streams.p1, { seed: [1, 2, 3, 4], move: 1.0 });
const p2 = new RandomPlayerAI(streams.p2, { seed: [5, 6, 7, 8] });

void p1.start();
void p2.start();

void streams.omniscient.write(
  `>start ${JSON.stringify({ formatid: 'gen1randombattle', seed: '1,2,3,4' })}\n` +
  `>player p1 ${JSON.stringify({ name: 'MyBot', team: packedTeamString })}\n` +
  `>player p2 ${JSON.stringify({ name: 'Opponent', team: opponentTeam })}`
);

for await (const chunk of streams.omniscient) {
  // parse battle protocol lines here
}
```

`streams.omniscient` is the god-view stream; both players' private streams are `streams.p1` and `streams.p2`. The AI constructors take the player-specific streams.

## DamageFirstAI — a Concrete Example

From `simulate.js`. Picks the move with the highest base power; ties fall back to random.

```javascript
const { Dex } = require('./dist/sim/index');

const GEN_DEX = {
  1: Dex.mod('gen1'),
  2: Dex.mod('gen2'),
  3: Dex.mod('gen3'),
};

class DamageFirstAI extends RandomPlayerAI {
  constructor(playerStream, options, genNum) {
    super(playerStream, options);
    this.dex = GEN_DEX[genNum] || Dex;
  }

  chooseMove(active, moves) {
    let bestChoice = null;
    let bestPower = -1;
    for (const m of moves) {
      const moveData = this.dex.moves.get(m.move.move);
      const power = moveData ? (moveData.basePower || 0) : 0;
      if (power > bestPower) {
        bestPower = power;
        bestChoice = m;
      }
    }
    return bestChoice ? bestChoice.choice : this.prng.sample(moves).choice;
  }
}
```

Key point: `m.move.move` is the move id string (e.g., `"thunderbolt"`). Pass it to `dex.moves.get()` to retrieve base power, type, category, PP, etc. Always use the gen-scoped Dex (`Dex.mod('gen1')`) when running older formats — base powers and move mechanics differ between generations.

## The `active` Object — What's Available

Inside `chooseMove`, `active` is the element of `request.active` for the current Pokemon. Key fields:

```
active.moves[]         // same as the raw move objects before filtering
active.canMegaEvo      // boolean
active.canDynamax      // boolean
active.canTerastallize // boolean
active.canZMove[]      // array or falsy; each element is null or a z-move descriptor
active.maxMoves        // present if the Pokemon can dynamax; has .maxMoves[]
active.trapped         // boolean; if true, switch options are removed by the base class
```

The Pokemon's current HP, status, and boosts are on `request.side.pokemon[i]`, not on `active`. To access them inside `chooseMove`, you need to reach into the request — either stash a reference in `receiveRequest` before calling super, or override `receiveRequest` entirely and call your own `chooseMove` with extra context.

## Error Handling

`receiveError` is called if the simulator rejects your choice string. The base class silently ignores `[Unavailable choice]` errors (the simulator will re-send the request). Any other error is re-thrown. You generally do not need to override this unless you want to log rejected choices during debugging.

## For RL Agents

`chooseMove` is the natural action hook. The full decision loop at each step:

1. `receiveRequest` fires with a new request object
2. Base class filters legal moves/switches and calls `chooseMove`
3. Your policy reads `active` + `moves` to build an observation vector
4. Policy outputs a logit over actions (masked to legal moves)
5. You return the winning `choice` string
6. Base class calls `this.choose(decision)` to submit to the simulator

Because `receiveRequest` runs synchronously inside the async stream loop, you can `await` inside `chooseMove` (it is declared `protected` not `protected readonly`, so you can make it async in a subclass). This lets you call an external model server without blocking the event loop permanently.

## Gym Wrapper (PokemonGymEnv)

The `PokemonGymEnv` is a step-based reinforcement learning environment that wraps `BattleStream` in a standard RL interface. It is compatible with training algorithms like DQN, PPO, and tabular Q-Learning.

### What the Gym Is

`PokemonGymEnv` abstracts away battle protocol details and exposes a simple interface:

- **reset()** — returns an observation (780-float structured token array by default, or 100-feature Float32Array with `obsMode: 'flat'`)
- **step(action)** — takes a discrete action (0–8) and returns (obs, reward, done, info)
- **validActions()** — returns a boolean mask of legal actions
- **destroy()** — cleans up streams and resources

Each environment instance runs a single battle and terminates on win, loss, or draw. Your agent always plays as player 1 (gym player); player 2 is a configurable opponent (default: RandomPlayerAI).

### Quick Start

```javascript
const { PokemonGymEnv } = require('./dist/sim/tools/pokemon-gym');

const env = new PokemonGymEnv({ format: 'gen1randombattle', seed: [1, 2, 3, 4] });
let obs = await env.reset();

for (let step = 0; step < 1000; step++) {
  const mask = env.validActions();
  const action = Math.floor(Math.random() * 9); // random valid action
  
  if (!mask[action]) continue; // skip illegal actions
  
  const { obs: nextObs, reward, done, info } = await env.step(action);
  console.log(`Step ${step}: reward=${reward}, done=${done}`);
  
  if (done) {
    console.log(`Battle ended: ${info.winner} won in ${info.turns} turns`);
    obs = await env.reset();
  } else {
    obs = nextObs;
  }
}

env.destroy();
```

### Observation Space

`PokemonGymEnv` supports two observation modes via the `obsMode` constructor option:

- **`'structured'` (default, M2)** — a per-Pokémon token observation from `extractFeaturesStructured()`, shape `(12, 65)` flattened to **780 floats**.
- **`'flat'` (legacy, M1)** — the original **100-feature Float32Array** from `extractFeatures()`, kept for MLP-baseline regression checks (`gym_bridge.js --flat`).

#### Structured tokens (12 × 65)

12 tokens, one per Pokémon:

| Token | Contents |
|-------|----------|
| `[0]` | Own active Pokémon |
| `[1–5]` | Own bench, in `side.pokemon[1..5]` **request-slot order** — action `4+j` grounds to bench token `1+j` |
| `[6]` | Opponent active Pokémon (from the reveal tracker, see below) |
| `[7–11]` | Opponent bench — revealed non-active Pokémon first (reveal order), then unrevealed slots |

Each 65-dim token: HP ratio (1), level/100 (1), type 1 one-hot (15), type 2 one-hot (15, duplicates type 1 if monotype), status one-hot (6: brn/frz/par/psn/slp/tox), active flag (1), unknown flag (1), fainted flag (1), 4 × move features (6 each: base_power/250, accuracy, PP ratio, type_idx/20, category/2, disabled).

Unrevealed opponent bench slots get `unknown_flag=1, HP_ratio=1.0` (never an all-zero vector — that would be indistinguishable from fainted). Fainted Pokémon get `fainted_flag=1, HP_ratio=0`, all other dims zero.

This layout intentionally matches `models/metamon_adapter.py` exactly so the M2.5 BC-pretrained checkpoint (`models/checkpoints/bc_pretrain_gen1ou.pt`) loads against live-gym observations without remapping. It has **no stat-boost dimensions** — boosts were dropped from the plan when the schema was locked to the BC-compatible layout (Metamon's replay dataset doesn't encode them either).

**Opponent reveal tracker:** since a real player's request never contains the opponent's team, `PokemonGymEnv` reconstructs opponent state purely from battle-log lines (`|switch|`, `|drag|`, `|-damage|`, `|-heal|`, `|-status|`, `|-curestatus|`, `|faint|`, `|move|` for `p2`) — never from omniscient state. Opponent moves populate only as they're used in battle; unused move slots stay zero.

#### Flat vector (legacy, `obsMode: 'flat'`)

| Indices | Feature | Description |
|---------|---------|-------------|
| 0–14 | Own active Pokémon | HP ratio, level, status (burn/freeze/paralysis/poison/sleep/toxic), stat boosts (placeholder) |
| 15–54 | Own moves ×4 | 10 features per move: base power, accuracy, PP ratio, type, category, disabled, padding |
| 55–59 | Switch mask | Boolean availability for bench slots 1–5 |
| 60–74 | Opponent active Pokémon | HP ratio, level, status, types, species index, padding |
| 75–99 | Padding | Zeros |

All features are normalized to approximate [0, 1] ranges. Refer to `sim/tools/feature-extractor.ts` for exact scaling.

### Action Space

Discrete space with 9 actions:

| Action | Meaning |
|--------|---------|
| 0–3 | Use move 1–4 |
| 4–8 | Switch to bench slot 1–5 |

The `validActions()` method returns a boolean array indicating which actions are legal given the current request. Illegal actions incur a −0.01 reward and do not advance the battle.

### Reward Function

Rewards are clipped to [−1, +1] and comprise:

- **Per-turn KOs:** +0.01 per opponent faint, −0.01 per own faint
- **Per-turn status:** +0.0001 when opponent gains status
- **Episode end:**
  - +1.0 if your gym player wins
  - −1.0 if opponent wins
- **Stalling penalty:** −0.001 × (turn count) applied when terminal
- **Illegal action:** −0.01 (no battle advance)

### Evaluator — Running Baseline Tests

The `evaluator.ts` module exports `evaluateVsRandom()` to benchmark your agents:

```javascript
const { evaluateVsRandom } = require('./dist/sim/tools/evaluator');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');

// Define your agent factory
function myAgentFactory(stream, seed) {
  return new MyCustomAI(stream, { seed });
}

// Run 1000 battles at concurrency 50
const result = await evaluateVsRandom(myAgentFactory, 1000, 50);

console.log(`Win rate: ${(result.p1WinRate * 100).toFixed(1)}%`);
console.log(`Avg turns: ${result.avgTurns.toFixed(1)}`);
console.log(`KO ratio: ${(result.p1KORatio * 100).toFixed(1)}%`);
console.log(`Duration: ${result.durationMs}ms`);
```

### Seeding

Pass a `seed` to the constructor for reproducible battles:

```javascript
const env = new PokemonGymEnv({
  format: 'gen1randombattle',
  seed: [42, 0, 0, 0],  // 4-element PRNG seed
});
```

Without a seed, the environment uses a random seed on each reset. Both the battle RNG and opponent team generation are seeded.
