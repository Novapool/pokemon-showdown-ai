# ML Training Guide — Building a Pokemon Battle AI

## Goal

Train an agent that wins Pokemon battles. The simulator provides a fast, deterministic, fully-observable environment with no network latency. You can run thousands of battles per minute on a single machine.

This guide is opinionated. It tells you what to do first, not every option that exists.

## Prerequisites

Build the simulator before doing anything:

```bash
npm install
npm run build
```

All runtime imports come from `dist/`, not `sim/`. See `docs/SETUP.md` for full build instructions.

## Start Here: Format and Teams

Use `gen1randombattle` for initial experiments.

**Why gen 1:**
- No items, no abilities, no held items, no weather
- No switching penalty mechanics
- Deterministic damage rolls (no critical-hit rate complexity in early gens for training purposes)
- Smallest action space: exactly 4 moves, no special mechanics layered on top

**Why randombattle:**
- The simulator generates legal random teams for you — no team-building required
- Consistent difficulty: neither player has a team-building advantage
- Both players face the same uncertainty about the opponent's team

```javascript
const { Teams, Dex } = require('./dist/sim/index');

// Generate a legal random team for gen1randombattle
const team = Teams.generate('gen1randombattle');
// team is a packed string ready for the battle stream
```

If `Teams.generate` is unavailable in your build, teams can be packed manually using the format `SPECIES||ITEM|ABILITY|MOVES|NATURE|EVS||IVS||LEVEL|`. `simulate.js`'s `packMon` helper is a reference implementation of this format (see root `CLAUDE.md` — it's an unrelated script, not part of this project, but the packing format itself is standard PS team format).

## M1 Gym Wrapper — Start Here for RL

The repo now includes a complete step-based RL environment. Use it instead of building the training loop manually.

```javascript
const { PokemonGymEnv } = require('./dist/sim/tools/pokemon-gym');
const { evaluateVsRandom } = require('./dist/sim/tools/evaluator');

// Step-based training loop
// 'structured' (shown here) is the default: (12,65) tokens flattened to 780 floats.
// q_learning/dqn/ppo's networks are hardcoded to the legacy 100-dim vector — pass
// obsMode: 'flat' if you're feeding one of those, not a new/transformer model.
const env = new PokemonGymEnv({ seed: [1, 2, 3, 4], obsMode: 'structured' });
let obs = await env.reset();
while (true) {
  const mask = env.validActions();           // boolean[9]
  const action = myPolicy(obs, mask);        // pick from legal actions
  const { obs: next, reward, done } = await env.step(action);
  obs = done ? await env.reset() : next;
}

// Evaluate win rate against RandomPlayerAI (1000 battles, 50 concurrent)
const result = await evaluateVsRandom(myAgentFactory, 1000, 50);
console.log(`Win rate: ${(result.p1WinRate * 100).toFixed(1)}%`);
```

See `docs/AI-PLAYERS.md` → **Gym Wrapper** section for the full API reference and observation space layout. The feature extractor is at `sim/tools/feature-extractor.ts` (`OBS_SIZE = 100`).

## Python-Node Bridge Protocol

`gym_client.py` spawns `gym_bridge.js` as a child process via `subprocess.Popen`. All communication is line-delimited JSON over stdin/stdout — Python writes one JSON command per line and reads one JSON response line back. The bridge processes commands sequentially (one at a time), so no request IDs or async multiplexing are needed.

### Message Types

| Direction | Command | Request | Response |
|-----------|---------|---------|----------|
| Python → Node | reset | `{"cmd":"reset"}` (optional `"opponent":"random"\|"damagefirst"\|"self"` — M3.4 per-episode override; `"self"` switches this env to the dual-seat protocol) | `{"obs":[...780 floats...],"mask":[...]}` (100 with `--flat`, 924 with `--obs-v2`; dual protocol when the episode's opponent is `self`) |
| Python → Node | step | `{"cmd":"step","action":<int 0–8>}` | `{"obs":[...],"reward":<float>,"done":<bool>,"info":{},"mask":[...]}` |
| Python → Node | valid_actions | `{"cmd":"valid_actions"}` | `{"mask":[<9 booleans>]}` |
| Python → Node | close | `{"cmd":"close"}` | `{"ok":true}` then process exits |

### Error Responses

Any command can return `{"error":"<message>"}` instead of the normal response. `GymClient._send()` raises `RuntimeError` on an error response. Common errors:

- `"not initialized"` — `step` or `valid_actions` called before the first `reset`
- `"unknown command: <cmd>"` — unrecognized command string
- `"invalid JSON: <message>"` — malformed JSON sent on stdin
- Any unhandled exception inside the gym is caught and returned as `{"error":"<message>"}` — the bridge keeps running after these

Always call `reset()` before starting a training episode.

### Troubleshooting

- **"Cannot find module '../dist/sim/tools/pokemon-gym'"** — run `npm run build` (or `./build`) first; the bridge requires compiled `dist/` output.
- **Python hangs on readline** — `gym_bridge.js` writes all unhandled exceptions to stdout as `{"error":"..."}`, so a hang usually means the Node process crashed before writing any response. Check `self._proc.stderr` for Node startup errors (stderr is captured via `subprocess.PIPE`).
- **Bridge stderr** — available via `self._proc.stderr` for debugging Node.js startup failures such as missing modules or syntax errors.

## Parallel Training (M3.1)

Training and evaluation run `--num-envs` parallel battle simulations (default 8). `models/vec_gym_client.py`'s `VecGymClient` manages N `gym_bridge.js` subprocesses — one Node event loop (and CPU core) per env — and pipelines each step: all N commands are written before any response is read, so Python waits only for the slowest env. Finished episodes auto-reset; the trainer never calls `reset()` after startup. Inference is batched over all envs' observations (`act_batch()` on `PPOAgent`/`TransformerAgent`), and each env keeps its own `TrajectoryBuffer` so GAE never crosses episode streams (`merge_buffers()` combines them and normalizes advantages globally before each PPO update).

```bash
python models/transformer/train.py --steps 2600000 --num-envs 8
python models/ppo/train.py --structured --steps 2600000 --num-envs 8
python models/evaluate.py --model transformer --checkpoint <ckpt> --battles 500 --num-envs 8
```

`--num-envs 1` uses the same code path and reproduces the old serial behavior.

### Measured throughput (Apple Silicon Mac, MPS, transformer PPO, 16,384 steps, rollout 512)

| num_envs | Wall time | Steps/sec | Speedup |
|---|---|---|---|
| 1 (old serial) | 58.5s | ~280 | 1x |
| 8 | 11.6s | ~1,410 | 5.0x (≈6x steady-state after ~2.5s process startup) |

At 8 envs, a 2.6M-step training run drops from ~2.6 hours to ~30 minutes. Evaluation: 100 transformer battles went from 20.2s to 5.6s. Gains flatten past 8 envs on this machine (16 envs measured only marginally faster); benchmark before assuming more envs helps.

### Devices: CUDA, MPS, and the Neural Engine

- `--device {cpu,mps,cuda}` overrides auto-detection (CUDA → MPS → CPU) on both trainers and `evaluate.py`. Device is a runtime choice, deliberately **not** stored in checkpoints — they are portable across machines in both directions (`torch.load(..., map_location="cpu")`).
- **Training on a CUDA machine (e.g. RTX 3080):** no code changes needed. Clone the repo, run `./build`, install a CUDA build of PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cu121`), and run the same commands — CUDA is auto-detected. Copy checkpoints back and forth freely.
- **Apple Neural Engine:** not usable. Apple exposes the ANE only through CoreML, which cannot train and which PyTorch does not target. MPS (the GPU) is the correct Mac backend and is what auto-detection picks on Apple Silicon.
- For these small models at small batch sizes, `--device cpu` can outperform MPS due to per-op dispatch overhead; the gap narrows as `--num-envs` (and thus inference batch size) grows. Benchmark a short run before committing to a long one.

## Opponents & Self-Play (M3.3)

Both trainers and `evaluate.py` accept `--opponent`:

- **`random`** (default) — `RandomPlayerAI`. Note its `move: 1.0` default means it never voluntarily switches; it presses a random attack every turn.
- **`damagefirst`** — `DamageFirstAI` (`sim/tools/damage-first-ai.ts`): always picks the legal move with the highest base power. A strictly stronger fixed benchmark. The M2 MLP baseline scores 51% (101/200) against it.
- **`selfplay`** (trainers only) — the p2 seat is a frozen past checkpoint of the agent itself, league-style.

Self-play runs the gym in dual-seat mode (`PokemonGymEnv` `opponent: 'self'`, bridge `--selfplay`): each `stepDual()` advances the battle to the next decision point and reports both seats' observations, masks, and a `needsAction` flag (only one seat acts during force-switches). The learner trains on the p1 seat only; rewards from opponent-only decision points accumulate into p1's open transition. Each rollout samples one frozen opponent from `--selfplay-pool` (default: the run's own checkpoint directory) — the newest checkpoint 50% of the time, otherwise uniform over the pool; until a first checkpoint exists the opponent is a frozen copy of the current policy. Both seats see only revealed information (the reveal tracker runs for both sides).

### Schema v2 & Mixed Opponents (M3.4)

- **Obs schema v2** — `--obs-v2` on `models/ppo/train.py` and `evaluate.py` (bridge `--obs-v2`, gym obsMode `'structured-v2'`): `(12, 77)` tokens = the v1 65 dims (byte-identical) + 7 boost stages (stage/6), Reflect/Light Screen/Substitute/Leech Seed flags, and a toxic counter, all tracked from public battle-log lines and non-zero only on the two active tokens (gen1 resets them on switch). v2 checkpoints live in `models/ppo/checkpoints/v2/` and are incompatible with v1 by design; the v1-prefix property means `gym_client.slice_structured_obs()` can hand any v1 checkpoint its native view of a v2 observation (used for cross-schema self-play opponents and `--vs-checkpoint` head-to-head).
- **Mixed opponents** — `--opponent-mix "selfplay=0.5,damagefirst=0.3,random=0.2"` (PPO trainer only, mutually exclusive with `--opponent`) samples one opponent family per rollout and resets the envs when the family changes (abandoned episodes are bootstrapped, the same truncation PPO applies at rollout ends). Seed the self-play pool by copying checkpoints into the run's checkpoint dir as `ppo_step_0_<name>.pt` — step 0 keeps them out of the "newest" half of the sampling; skip `ppo_step_0_*` files when sweeping checkpoints for evaluation.

```bash
python models/transformer/train.py --steps 5000000 --num-envs 8 --opponent selfplay \
    --checkpoint-dir models/transformer/checkpoints/selfplay \
    --pretrain_checkpoint models/checkpoints/bc_pretrain_gen1ou.pt \
    --bc-anchor models/checkpoints/bc_pretrain_gen1ou.pt --value-warmup-steps 200000

python models/evaluate.py --model transformer --checkpoint <ckpt> --battles 500 --opponent damagefirst
```

## The Training Loop (Manual — if not using the gym)

```
for each epoch:
  1. Generate N battle tasks (starter team vs opponent team, unique seed per battle)
  2. Run all battles concurrently up to CONCURRENCY limit
  3. For each completed battle, extract (observation, action, reward) tuples from the log
  4. Update model weights
  5. Swap in the updated model as one or both players
  6. Repeat
```

At 50 concurrent battles (`CONCURRENCY = 50`), expect roughly 2,000–5,000 battles per minute depending on battle length and hardware. Plan for 10,000 battles per training epoch as a starting budget.

## Running Battles

The core pattern (see `docs/examples/parallel-training-skeleton.js` for the full runnable version):

```javascript
const { BattleStream, getPlayerStreams } = require('./dist/sim/index');
const { RandomPlayerAI } = require('./dist/sim/tools/random-player-ai');

async function runBattle(format, p1Team, p2Team, seed) {
  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);

  const p1 = new MyAI(streams.p1, { seed: seed, move: 1.0 });
  const p2 = new RandomPlayerAI(streams.p2, { seed: [seed[0] ^ 0x5555, seed[1], seed[2], seed[3]] });

  void p1.start();
  void p2.start();

  void streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: format, seed: seed.join(',') })}\n` +
    `>player p1 ${JSON.stringify({ name: 'MyBot', team: p1Team })}\n` +
    `>player p2 ${JSON.stringify({ name: 'Opponent', team: p2Team })}`
  );

  let won = false;
  let opponentKOs = 0;
  let turns = 0;
  const opponentTeamSize = p2Team.split(']').length;

  for await (const chunk of streams.omniscient) {
    for (const line of chunk.split('\n')) {
      if (line.startsWith('|faint|p2')) opponentKOs++;
      if (line.startsWith('|win|')) won = line.slice(5).trim() === 'MyBot';
      if (line.startsWith('|turn|')) {
        const t = parseInt(line.split('|')[2]);
        if (!isNaN(t)) turns = t;
      }
    }
  }

  return { won, opponentKOs, opponentTeamSize, turns };
}
```

The `|faint|` line format is `|faint|POSITION` where `POSITION` is `p1a`, `p2a`, etc. Checking `startsWith('|faint|p2')` counts all opponent faints regardless of doubles position.

## Concurrency Pool

Run battles in parallel without hammering your process limit. This is `runConcurrent()` from `docs/examples/parallel-training-skeleton.js`:

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

// Usage:
const tasks = battles.map(b => () => runBattle(b.format, b.p1Team, b.p2Team, b.seed));
const results = await runConcurrent(tasks, 50);
```

Each `BattleStream` is an isolated in-process object — no shared state between concurrent battles. Safe to run hundreds in parallel.

## Reward Signal

Start with the simplest signal that gives the model a learning gradient.

**Tier 1 — sparse win/loss (simplest, slowest to learn):**
```javascript
const reward = won ? 1.0 : -1.0;
```

**Tier 2 — shaped with KO ratio (recommended starting point):**
```javascript
const koRatio = opponentTeamSize > 0 ? opponentKOs / opponentTeamSize : 0;
const reward = (won ? 1.0 : -1.0) + koRatio * 0.5;
```

This gives a gradient even in losses: knocking out 3 of 6 opponent Pokemon before losing is better than being swept.

**Tier 3 — turn efficiency bonus:**
```javascript
const koRatio = opponentKOs / opponentTeamSize;
const reward = (won ? 1.0 : -1.0) + koRatio * 0.5 - turns * 0.001;
```

Penalizes stalling and rewards decisive wins. Only add this after the model can already win reliably — otherwise it learns to rush into self-destructing.

**Tier 4 — combined (for self-play):**
```javascript
const reward = (won ? 1.0 : -1.0) * 1.0
             + (opponentKOs / opponentTeamSize) * 0.5
             - turns * 0.001;
```

## Observation Space

`PokemonGymEnv` provides observations via two modes:

- **`'structured'` (default):** A 12×65 per-Pokémon token representation flattened to 780 floats. Suitable for transformer-based models or new architectures.
- **`'flat'` (legacy):** A flat ~100-dimensional feature vector. Required by the baseline models `q_learning`, `dqn`, and `ppo`, which have networks hardcoded for this input shape.

For the exact layout of both observation modes (field-by-field specification of each mode's structure), see `docs/AI-PLAYERS.md` → "Gym Wrapper (PokemonGymEnv)" → "Observation Space".

## Action Space

Actions are discrete. In `gen1randombattle`:

- 4 move slots (indices 0–3)
- Up to 5 switch slots (indices 4–8), only legal when not trapped and bench Pokemon exist

Max action space: 9. In practice gen 1 singles rarely has more than 4 legal actions (moves only, switching down-prioritized by setting `move: 1.0` in options).

Always mask illegal actions before passing to your policy head:

```javascript
// legal_mask is a boolean/binary array of length = action_space_size
// Set logit to -inf for illegal actions before softmax
const logits = model.forward(obs);
logits[illegalActionIndices] = -Infinity;
const probs = softmax(logits);
const action = sample(probs);
```

## Three Training Approaches

### 1. Supervised Learning (Imitation) — Best First Step

Record battles where `DamageFirstAI` plays, then train your model to imitate the decisions.

```javascript
const fs = require('fs');
const stream = fs.createWriteStream('imitation_data.jsonl', { flags: 'a' });

class RecordingAI extends DamageFirstAI {
  chooseMove(active, moves) {
    const choice = super.chooseMove(active, moves);          // teacher's action
    const actionIndex = moves.findIndex(m => m.choice === choice);
    stream.write(JSON.stringify({
      obs: buildObs(active, moves, this.dex),
      action: actionIndex,
    }) + '\n');
    return choice;
  }
}
```

Collect 50,000–100,000 labeled decisions, then train a classifier to predict `action` from `obs`. This bootstraps a model that is not random — a much easier starting point for RL fine-tuning.

### 2. Reinforcement Learning (Self-Play) — Main Approach

Two instances of your model play each other. Use PPO (recommended) or DQN.

Self-play loop:
1. Snapshot current model weights as the "opponent" every N epochs
2. Train current model against the snapshot
3. Occasionally mix in `RandomPlayerAI` as opponent to prevent strategy collapse

The `move: 1.0` option in the constructor eliminates random switching from both agents, so all decisions flow through your `chooseMove` hook.

**Seeding for reproducibility:**

```javascript
// Derive unique seeds per battle to prevent correlation
const seed = [
  (iterNum + battleIdx * 1000) & 0xFFFF,
  (iterNum * 3 + 7) & 0xFFFF,
  (iterNum * 7 + 13) & 0xFFFF,
  (iterNum * 11 + 17) & 0xFFFF,
];
const p1Seed = seed;
const p2Seed = [seed[0] ^ 0xAAAA, seed[1], seed[2], seed[3]];
```

This matches the seed-derivation/XOR-decorrelation pattern in `docs/examples/parallel-training-skeleton.js` and guarantees different PRNG states for each player while keeping battles reproducible from the iteration number.

### 3. Evolutionary / Genetic — Simplest to Implement

Score AI variants by win rate over N battles, breed top performers.

```javascript
// Score each variant
const scores = await Promise.all(variants.map(v => evaluateVariant(v, 200)));
// Sort by win rate
scores.sort((a, b) => b.winRate - a.winRate);
// Keep top half, mutate weights, repeat
const survivors = scores.slice(0, scores.length / 2);
```

Useful for hyperparameter search or as a sanity check before investing in full RL infrastructure.

## Data Collection at Scale

Write results to CSV as battles complete — do not buffer in memory.

```javascript
const rawStream = fs.createWriteStream('training_data.csv', { flags: 'a' });
rawStream.write('iteration,won,opponent_kos,opponent_team_size,turns,reward\n');

function recordBattle(iter, result) {
  const { won, opponentKOs, opponentTeamSize, turns } = result;
  const koRatio = opponentTeamSize > 0 ? opponentKOs / opponentTeamSize : 0;
  const reward = (won ? 1.0 : -1.0) + koRatio * 0.5;
  rawStream.write(`${iter},${won ? 1 : 0},${opponentKOs},${opponentTeamSize},${turns},${reward.toFixed(4)}\n`);
}
```

For trajectory data needed by PPO, buffer per-battle step data in memory and flush when the battle ends.

## Reading the Battle Protocol

The `streams.omniscient` stream yields raw PS protocol lines. Key lines for ML:

| Line prefix | Meaning |
|-------------|---------|
| `\|turn\|N` | Turn N begins |
| `\|move\|POKEMON\|MOVE\|TARGET` | A Pokemon used a move |
| `\|switch\|POKEMON\|DETAILS\|HP STATUS` | A Pokemon switched in |
| `\|-damage\|POKEMON\|HP STATUS` | HP changed from damage |
| `\|-heal\|POKEMON\|HP STATUS` | HP changed from healing |
| `\|faint\|POKEMON` | A Pokemon fainted |
| `\|win\|USER` | Battle over; USER won |
| `\|tie` | Battle ended in a tie |

`HP STATUS` format is `CURRENT/MAX STATUS` for your own Pokemon (e.g., `150/350 par`) and a percentage for opponents when HP Percentage Mod is on (`62/100`). Parse with:

```javascript
function parseHP(conditionStr) {
  // e.g. "150/350 par" or "62/100" or "0 fnt"
  const [hpPart, status] = conditionStr.split(' ');
  const [current, max] = hpPart.split('/').map(Number);
  return { current: current || 0, max: max || 100, status: status || '' };
}
```

`POKEMON` position strings are `p1a`, `p2a` in singles; `p1a`, `p1b`, `p2a`, `p2b` in doubles.

## Milestones

Work through these sequentially. Do not skip ahead.

> **Current status:** M0 (foundation) and M1 (gym wrapper, evaluator, feature extractor) are complete. M2 (model exploration: tabular Q, DQN, PPO) is next. See `MILESTONES.md` for the full plan.

**Milestone 1 ✅:** Gym wrapper, feature extractor, and evaluation harness in place.

**Milestone 2:** Beat `RandomPlayerAI` with >80% win rate in `gen1randombattle`.

Baseline win rate of `DamageFirstAI` vs `RandomPlayerAI` in gen 1 is approximately 65–75% depending on team matchup. Your model should exceed this before moving on.

**Milestone 2:** Beat `DamageFirstAI` with >60% win rate.

`DamageFirstAI` is a credible opponent. Beating it requires the model to learn type matchups or at least exploit PP depletion and status moves.

**Milestone 3:** Self-play stability.

Run self-play for 50 epochs. Check that win rate against a held-out `RandomPlayerAI` opponent does not regress (strategy collapse). If it does, mix in 30% random opponents during training.

**Milestone 4:** Upgrade format to `gen3randombattle` or `gen7randombattle`.

These add held items, abilities, and weather — richer features but the same training infrastructure. Expand your observation space to include item and ability lookups via `dex.items.get()` and `dex.abilities.get()`.

## Common Pitfalls

**Move id vs move name:** `m.move.move` is the normalized id (`"thunderbolt"`), not the display name. Always pass the id to `dex.moves.get()`.

**Gen-scoped Dex:** Base powers differ between generations. Surf is 95 BP in gens 1-5 and 90 BP in gen 6+. Always use `Dex.mod('gen1')` when running gen 1 formats. See `docs/AI-PLAYERS.md` → 'DamageFirstAI — a Concrete Example' for the `GEN_DEX` pattern.

**Seed correlation:** If you use the same seed for both players in the same battle, their PRNG states will be correlated. XOR the seed for one player as shown above.

**Choice string format:** Return exactly what is in `m.choice` — the base class pre-formats targeting suffixes. Do not construct your own `"move 1"` string from scratch; use the provided choice values.

**Fainted Pokemon in switches:** The base class already filters fainted and active Pokemon from the `switches` array passed to `chooseSwitch`. You do not need to re-filter.
