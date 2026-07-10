Pokemon Showdown battle simulator — used here for training a Pokemon trainer ML model via parallel battle simulation.

## Quick Start

```
./build
```

## Key Entry Points

| Entry point | Purpose |
|---|---|
| `./pokemon-showdown` | CLI for team gen, validation, battle piping |
| `dist/sim/index.js` | Node.js API — import `BattleStream`, `Dex`, `PRNG` |

> **Note:** `simulate.js` in the repo root is an unrelated script (gym leader simulation for a separate project). It is a useful concurrency reference but is **not** part of this ML training project.

## Documentation Index

| File | Read when |
|---|---|
| `docs/SETUP.md` | Setting up the repo, rebuilding after pulls |
| `docs/CLI.md` | Using `./pokemon-showdown` commands (team gen, format inspection) |
| `docs/SIMULATOR-API.md` | Using `BattleStream` in Node.js for high-throughput simulation |
| `docs/PARALLEL-SIMULATION.md` | Running thousands of concurrent battles for ML training |
| `docs/AI-PLAYERS.md` | Creating custom Pokemon battle agents; extending `RandomPlayerAI` |
| `docs/ML-TRAINING.md` | Building and training a Pokemon trainer ML model (start here for ML) |
| `docs/DATA-FORMATS.md` | Battle protocol, team formats, CSV output schemas |
| `docs/BATTLE-FORMATS.md` | Available formats; choosing formats for training |
| `docs/MODEL-COMPARISON.md` | Model exploration results and winner selection (M2 results — fill after training) |
| `models/CLAUDE.md` | Working in `models/` — Python↔Node bridge architecture, obsMode requirements per model, quick-start training/evaluation commands |
| `sim/SIMULATOR.md` | Detailed battle protocol (stdin/stdout message format) |
| `sim/TEAMS.md` | Team formats: packed, JSON, export |
| `sim/DEX.md` | Accessing Pokedex/move/ability data via `Dex` |

`sim/README.md`, `sim/NONSTANDARD.md`, and `sim/SIM-PROTOCOL.md` exist as upstream reference material from the Pokemon Showdown project. They are not required reading for the ML training workflow — `docs/SIMULATOR-API.md` covers the subset of the API that matters here.

## Directory Overview

| Directory | Contents |
|---|---|
| `sim/` | Battle engine TypeScript source |
| `data/` | Game data (moves, abilities, items, pokedex) |
| `config/` | Format definitions and server config |
| `tools/` | AI helpers (`RandomPlayerAI`, etc.) |
| `dist/` | Compiled JS output — what Node.js actually runs |
| `output/` | Output from `simulate.js` (unrelated script — ignore for ML work) |
