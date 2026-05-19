# CLI Reference

The `./pokemon-showdown` binary (or `node pokemon-showdown` on Windows) exposes several sub-commands. Run `./pokemon-showdown help` at any time to see the full list.

---

## Commands

### `start [--skip-build] [PORT]`

Starts a Pokemon Showdown web server.

```bash
./pokemon-showdown start 8000
```

Not needed for ML training — skip this.

---

### `generate-team [FORMAT-ID] [SEED]`

Generates a random team and writes it to stdout in packed team format.

```bash
./pokemon-showdown generate-team gen8randombattle
./pokemon-showdown generate-team gen1randombattle 12345  # deterministic with seed
```

Useful for building team pools for training data without running a full battle.

---

### `validate-team [FORMAT-ID]`

Reads a packed or JSON team from stdin. Exits `0` if valid, exits `1` and writes errors to stderr if invalid.

```bash
echo "<packed-team>" | ./pokemon-showdown validate-team gen8ou
```

---

### `simulate-battle`

Runs a single battle over stdin/stdout using the PS battle protocol. One subprocess handles one battle, then exits.

```bash
./pokemon-showdown simulate-battle
```

The stdin/stdout protocol is documented in `sim/SIMULATOR.md`.

Note: For ML training at scale, prefer the in-process Node.js `BattleStream` API (see `docs/SIMULATOR-API.md`) over this command. Spawning a subprocess per battle adds ~10–50ms of process startup overhead — at 50k battles that is meaningful. Use `simulate-battle` only for quick manual testing of a single battle.

---

### `json-team`

Reads a team in any format from stdin, writes unpacked JSON to stdout.

```bash
echo "<packed-team>" | ./pokemon-showdown json-team
```

---

### `pack-team`

Reads a team in any format (export/JSON) from stdin, writes packed format to stdout.

```bash
echo "<exported-team>" | ./pokemon-showdown pack-team
```

---

### `export-team`

Reads a team in any format from stdin, writes human-readable export format to stdout.

```bash
echo "<packed-team>" | ./pokemon-showdown export-team
```

---

### `help`

Prints the full command reference.

```bash
./pokemon-showdown help
./pokemon-showdown -h
```

---

## Piping Examples

Commands are composable via standard Unix pipes:

Generate a random gen8 team and display it in human-readable form:
```bash
./pokemon-showdown generate-team gen8randombattle | ./pokemon-showdown export-team
```

Generate a team and check whether it would be legal in gen8ou:
```bash
./pokemon-showdown generate-team gen8randombattle | ./pokemon-showdown validate-team gen8ou
```

Generate a team, convert to JSON, then pretty-print it:
```bash
./pokemon-showdown generate-team gen8randombattle | ./pokemon-showdown json-team | python3 -m json.tool
```

---

## ML Training Guidance

| Task | Use |
|---|---|
| Generate training team pools | `generate-team` — fast, scriptable |
| Inspect a format's legality rules | `validate-team` |
| Run one battle manually for debugging | `simulate-battle` |
| Run thousands of battles for training | `BattleStream` API in `simulate.js` — no subprocess overhead |

For high-throughput simulation, always use the `BattleStream` Node.js API. The CLI subprocess approach does not scale past a few hundred battles/sec on a single machine.
