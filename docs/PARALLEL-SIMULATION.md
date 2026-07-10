# Parallel Battle Simulation

How to run thousands of Pokemon battles concurrently for ML training data generation. Patterns are packaged in `docs/examples/parallel-training-skeleton.js`, a self-contained, runnable reference for this project's worker-pool concurrency and seeding approach.

---

## Why Parallel Works Without Worker Threads

`BattleStream` is fully async and non-blocking. Each battle is a self-contained Promise that only touches JavaScript when the event loop delivers I/O or timer callbacks. Node.js handles many concurrent battles on a single thread because the simulator is CPU-light per turn — it does arithmetic and state mutations, not file I/O or network calls, but yields control between turns via the async iteration protocol.

There is no shared mutable state between `BattleStream` instances. Spawning 50 concurrent battles means 50 independent state machines all making progress as the event loop schedules their continuations.

Worker threads add overhead (serialization, thread creation) and are not needed here.

---

## The Concurrency Pattern

`docs/examples/parallel-training-skeleton.js` uses a worker-pool pattern that keeps exactly `limit` promises in flight at once (`runConcurrent()`):

```javascript
async function runConcurrent(tasks, limit) {
  const results = new Array(tasks.length);
  let idx = 0;

  async function worker() {
    while (idx < tasks.length) {
      const i = idx++;
      results[i] = await tasks[i]();
    }
  }

  const workers = [];
  for (let i = 0; i < Math.min(limit, tasks.length); i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}
```

Each `worker` coroutine claims the next task by incrementing a shared index, runs it to completion, then immediately claims the next. This is simpler and more cache-friendly than the `Promise.race` approach because results are written to pre-allocated slots in order.

Usage:

```javascript
const CONCURRENCY = 50;

const tasks = battles.map(b => () => runBattle(b.format, b.p1Team, b.p2Team, b.seed, b.gen));
const results = await runConcurrent(tasks, CONCURRENCY);
```

Note that `tasks` is an array of **thunks** (zero-argument functions returning Promises), not Promises themselves. This ensures battles don't start until a worker slot is available.

---

## Running a Single Battle as a Promise

This is the unit of work that fills the task array. See `runBattle()` in `docs/examples/parallel-training-skeleton.js` for the runnable version:

```javascript
const { BattleStream, getPlayerStreams } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');

async function runBattle(format, starterTeam, gymTeam, seed, genNum) {
  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);

  // XOR the seed to de-correlate AI decisions from battle RNG
  const aiSeed1 = `${seed[0] ^ 0xAAAA},${seed[1]},${seed[2]},${seed[3]}`;
  const aiSeed2 = `${seed[0] ^ 0x5555},${seed[1]},${seed[2]},${seed[3]}`;

  const p1 = new DamageFirstAI(streams.p1, { seed: aiSeed1, move: 1.0 }, genNum);
  const p2 = new RandomPlayerAI(streams.p2, { seed: aiSeed2, move: 0.7 });

  void p1.start();
  void p2.start();

  // Write all three init commands as one string; newlines separate them
  const seedStr = seed.join(',');
  void streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: format, seed: seedStr })}\n` +
    `>player p1 ${JSON.stringify({ name: 'Starter', team: starterTeam })}\n` +
    `>player p2 ${JSON.stringify({ name: 'GymLeader', team: gymTeam })}`
  );

  let won = false;
  let koCount = 0;
  let turns = 0;
  const gymTotal = gymTeam.split(']').length;  // count mons in packed team

  try {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|faint|p2')) koCount++;
        if (line.startsWith('|win|')) won = (line.slice(5).trim() === 'Starter');
        if (line.startsWith('|turn|')) {
          const t = parseInt(line.split('|')[2]);
          if (!isNaN(t)) turns = t;
        }
      }
    }
  } catch (err) {
    // Battle crashed — treat as a loss; do not propagate
  }

  // Cleanly close the stream to release memory
  try { streams.omniscient.writeEnd(); } catch (_) {}

  return { won, koCount, gymTotal, turns };
}
```

Key points:
- `void p1.start()` and `void p2.start()` fire without awaiting — the AI loops run concurrently with the battle loop below.
- `void streams.omniscient.write(...)` similarly fires without awaiting — commands are buffered.
- The `for await` loop is the only awaited operation. It drives the battle to completion.
- Always catch errors inside the battle and return a sentinel result rather than letting a crash propagate and abort the whole pool.
- Call `writeEnd()` after the loop to signal the stream is done and allow GC.

---

## Seed Management for Reproducibility

Seeds are 4-element arrays of integers. Passing the same seed to `>start` always produces the same sequence of random events (damage rolls, crits, AI decisions if the AI also uses a seed).

Seed derivation strategy (see `docs/examples/parallel-training-skeleton.js` for the runnable version):

```javascript
// Top-level base seed per iteration
const baseSeed = [iter + 1, iter * 3 + 7, iter * 7 + 13, iter * 11 + 17];

// Unique seed per individual battle (derived from base + position offsets)
const seed = [
  (baseSeed[0] + starterIdx * 1000 + gymIdx * 100 + runIdx * 10) & 0xFFFF,
  (baseSeed[1] + starterIdx * 1001 + gymIdx * 101 + runIdx * 11) & 0xFFFF,
  (baseSeed[2] + starterIdx * 1003 + gymIdx * 103 + runIdx * 13) & 0xFFFF,
  (baseSeed[3] + starterIdx * 1007 + gymIdx * 107 + runIdx * 17) & 0xFFFF,
];

// Pass to the battle engine as a comma-separated string or array
const seedStr = seed.join(',');
streams.omniscient.write(`>start ${JSON.stringify({ formatid, seed: seedStr })}`);

// De-correlate AI seeds from battle seed using XOR
const aiSeed1 = `${seed[0] ^ 0xAAAA},${seed[1]},${seed[2]},${seed[3]}`;
const aiSeed2 = `${seed[0] ^ 0x5555},${seed[1]},${seed[2]},${seed[3]}`;
```

Using the `PRNG` class directly:

```javascript
const { PRNG } = require('./dist/sim/index');
const prng = new PRNG([iter, gymIdx, battleIdx, 0]);
const seed = prng.getSeed();  // number[4]
```

---

## Scale Reference

See `TOTAL_BATTLES = 10000` in `docs/examples/parallel-training-skeleton.js` for a baseline configuration. Expect roughly 2,000–5,000 battles/minute at CONCURRENCY=50 depending on battle length and format.

At CONCURRENCY=50, this completes in a few minutes on modern hardware. The bottleneck is pure JS execution in the battle engine, not I/O.

The simulation outputs results to CSV in real-time as battles finish, so partial results are usable if the run is interrupted.

---

## Tuning Concurrency

Start at 50 and increase. Watch Node.js heap:

```bash
node --max-old-space-size=4096 docs/examples/parallel-training-skeleton.js
```

| CONCURRENCY | When to use |
|-------------|-------------|
| 50 | Default; safe on 8 GB RAM |
| 100 | Good on 16 GB; ~2x throughput |
| 200 | Viable on 32 GB; watch for GC pressure |
| >200 | Usually yields diminishing returns; event loop itself becomes the bottleneck |

Each in-flight battle holds a `Battle` object in memory (~1-5 MB depending on format and turn count). At CONCURRENCY=50 that is 50-250 MB of live battle state, plus GC overhead for completed battles waiting to be collected.

If you see OOM errors, reduce `CONCURRENCY`. If GC pauses are long, reduce it or add `--expose-gc` and call `global.gc()` periodically between iteration batches.

---

## Structuring Batch Output

Write results to CSV as they complete rather than accumulating all results in memory:

```javascript
const fs = require('fs');
const rawStream = fs.createWriteStream('output/raw_battles.csv');
rawStream.write('iteration,starter,gym,run,won,ko_count,turns\n');

// After runConcurrent returns results for an iteration:
for (let t = 0; t < results.length; t++) {
  const { won, koCount, turns } = results[t];
  const { starterIdx, gymIdx, runIdx } = taskMeta[t];
  rawStream.write(`${iter},${starterIdx},${gymIdx},${runIdx},${won ? 1 : 0},${koCount},${turns}\n`);
}
```

Streaming writes avoid buffering gigabytes of CSV in memory for large runs.

---

## Custom AI for Training

Extend `RandomPlayerAI` or `BattlePlayer` to implement the agent being trained. Example (see `docs/AI-PLAYERS.md` → 'DamageFirstAI — a Concrete Example' for the full pattern):

```javascript
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');
const { Dex } = require('./dist/sim/index');

const GEN_DEX = {
  1: Dex.mod('gen1'),
  2: Dex.mod('gen2'),
  // ...
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

For a neural-network agent, override `receiveRequest` in `BattlePlayer`, extract features from the request JSON, run inference, and call `this.choose(choice)`. Keep the inference synchronous or use `async receiveRequest` — the stream loop handles backpressure automatically.

---

## Full Skeleton for a Training Run

The full runnable skeleton — worker-pool concurrency, seed derivation with XOR decorrelation, and streaming CSV output — lives in `docs/examples/parallel-training-skeleton.js`. Run it directly with `node docs/examples/parallel-training-skeleton.js` after `./build`.

---

## Format IDs for Training

| Format | Use case |
|--------|----------|
| `gen1randombattle` | Gen 1 baseline; no items, simpler mechanics |
| `gen1customgame` | Custom Gen 1 teams (gym leader scenarios) |
| `gen2customgame` | Custom Gen 2 teams |
| `gen3customgame` | Custom Gen 3 teams; abilities introduced |
| `gen4customgame` | Custom Gen 4 teams; Physical/Special split |
| `gen5customgame` | Custom Gen 5 teams |
| `gen7randombattle` | Gen 7 random; large and well-tested format |
| `gen8randombattle` | Gen 8 random; Dynamax available |
| `gen9randombattle` | Latest gen random |

`customgame` formats bypass clause enforcement and team validation — useful for scripted curriculum scenarios where you control both teams exactly.

For the full list of supported random formats, see `MultiRandomRunner.FORMATS` in `sim/tools/multi-random-runner.ts`.
