# Setup

## Prerequisites

- Node.js v22+ (engine minimum is v16, but v22 is recommended for ML workloads)
- Clone the repo: `git clone https://github.com/smogon/pokemon-showdown.git`

## Build

The TypeScript source must be compiled to `dist/` before anything runs.

```bash
./build              # Linux / macOS
node build           # Windows
```

The build step uses esbuild to transpile all TypeScript under `sim/`, `server/`, `lib/`, etc. into plain JS under `dist/`. This is fast (seconds, not minutes).

Force a clean rebuild if you see stale output or type errors:

```bash
./build --force
```

## Verify

```bash
npm test
```

Runs ESLint, mocha test suite, and a TypeScript type-check. All three must pass before trusting battle output.

## After Every `git pull`

```bash
./build
```

The compiled `dist/` is not committed. Any source change requires a rebuild or imports will use stale code.

## Runtime Imports

All programmatic imports come from compiled output, not TypeScript source:

```js
// Correct
const { BattleStream, Dex, PRNG } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');

// Wrong — Node cannot run .ts files directly
const { BattleStream } = require('./sim/index');
```

## ML Training Note

For ML training you only need the simulator, not the full PS web server. Running `./build` is sufficient — you do not need to `npm start` or set up a database. The `simulate.js` harness runs entirely in-process using `BattleStream`.
