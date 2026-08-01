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
| `docs/WHERE-WE-ARE.md` | **Start here at the beginning of any session, and surface it to the user when context is cleared.** One screen, plain language, no timeline: what the agent can and can't do, what's been settled, what's a dead end, and the live menu of options with a recommendation. Keep it updated as things close |
| `docs/DATA-INVENTORY.md` | **Read first before proposing any data-acquisition work.** What human data exists (the gen 1 archive is exhausted), what we hold, and what BC training actually consumes — vs what the manifest suggests |
| `docs/EVALUATION-METHODOLOGY.md` | **Read first before running or gating on any evaluation, ladder or bot.** The protocol: required sample sizes, account/arm setup, `scripts/ladder_analysis.py`, reporting template |
| `docs/LADDER-MEASUREMENT.md` | Why GXE/Elo readings from M6–M8 don't support per-run conclusions — the diagnosis behind the protocol above |
| `docs/SETUP.md` | Setting up the repo, rebuilding after pulls |
| `docs/MULTI-MACHINE.md` | **Read first when a file/checkpoint seems missing, or when a job needs the home machine.** Mac ↔ home-machine inventory, what syncs via git vs never, SSH/tmux recipes |

## Home machine (`homebox`)

Before **any** command on the home box, `git push` from here, then run the
preflight — it fast-forwards the remote checkout, verifies node 22 / `.venv`
torch+CUDA, and rebuilds `dist/` if stale. Non-zero exit ⇒ do not launch the job.

```
ssh homebox 'bash -lc "cd ~/Projects/pokemon-showdown-ai && scripts/homebox-preflight.sh"'
```

Every other remote command needs `bash -lc "..."` (node 22 lives in nvm; a plain
`ssh` gets node 18) and `.venv/bin/python` (system `python3` has no torch).
Details + machine inventory: `docs/MULTI-MACHINE.md`. **The home box is now
authoritative for both replay corpora** (`data/replays/gen1ou`,
`data/replays/gen1randombattle`) — rsync back before editing them on the Mac.
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

## Obsidian

Companion note: `/Users/laithassaf/Documents/Obsidian/nebula/1 Projects/pokemon-showdown.md`

When you make a large/architectural change or complete a milestone, update that
note's **Summary / Status / Next** sections to match (keep it concise). See the
"Obsidian Vault Sync" convention in `~/.claude/CLAUDE.md`.
