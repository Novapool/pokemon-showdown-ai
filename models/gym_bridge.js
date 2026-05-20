'use strict';

// --flat flag: launch with --flat to get 100-element obs (backward compat with pre-M2 builds).
// Without this flag the bridge expects PokemonGymEnv to return 876-element observations
// (12 tokens × 73 features, flattened from the structured extractor).
// The bridge itself serializes obs generically via Array.from(), so no logic change is needed here —
// the obs size is entirely determined by what PokemonGymEnv returns internally.
const FLAT_MODE = process.argv.includes('--flat');
const OPPONENT = (() => {
  const idx = process.argv.indexOf('--opponent');
  return idx !== -1 ? process.argv[idx + 1] : 'random';
})();

/**
 * gym_bridge.js — Line-delimited JSON stdio server wrapping PokemonGymEnv.
 *
 * Reads one JSON command per line from stdin, writes one JSON response per
 * line to stdout. All gym methods are async; commands are processed
 * sequentially (one at a time) to preserve request/response ordering.
 *
 * Supported commands (M2+, default):
 *   {"cmd":"reset"}                   → {"obs":[...876 floats...],"mask":[...]}
 *   {"cmd":"step","action":3}         → {"obs":[...876 floats...],"reward":0.01,"done":false,"info":{},"mask":[...]}
 *   {"cmd":"valid_actions"}           → {"mask":[true,true,...]}  (length 9)
 *   {"cmd":"close"}                   → {"ok":true}  then process.exit(0)
 *
 * Launch flags:
 *   --flat                  Use 100-element obs (backward compat with pre-M2 builds)
 *   --opponent <name>       Opponent AI: 'random' (default) or 'damage-first'
 *
 * Examples:
 *   node gym_bridge.js --flat
 *   node gym_bridge.js --opponent damage-first
 */

const readline = require('readline');
const { PokemonGymEnv } = require('../dist/sim/tools/pokemon-gym');

// Env is recreated on every reset() to avoid stream/worker accumulation.
let env = null;
let initialized = false;

/**
 * Write a single JSON response line to stdout.
 * @param {object} obj
 */
function respond(obj) {
	process.stdout.write(JSON.stringify(obj) + '\n');
}

/**
 * Process one parsed command object. Returns a Promise that resolves when
 * the command is fully handled (including any async gym operations).
 * @param {object} command
 * @returns {Promise<void>}
 */
async function processCommand(command) {
	const { cmd } = command;

	if (cmd === 'reset') {
		if (env) env.destroy();
		env = new PokemonGymEnv({ opponent: OPPONENT });
		const obsFloat32 = await env.reset();
		initialized = true;
		respond({ obs: Array.from(obsFloat32), mask: env.validActions() });

	} else if (cmd === 'step') {
		if (!initialized) {
			respond({ error: 'not initialized' });
			return;
		}
		const action = command.action;
		const result = await env.step(action);
		respond({
			obs: Array.from(result.obs),
			reward: result.reward,
			done: result.done,
			info: result.info,
			mask: env.validActions(),
		});

	} else if (cmd === 'valid_actions') {
		if (!initialized) {
			respond({ error: 'not initialized' });
			return;
		}
		const mask = env.validActions();
		respond({ mask });

	} else if (cmd === 'close') {
		respond({ ok: true });
		env.destroy();
		process.exit(0);

	} else {
		respond({ error: `unknown command: ${cmd}` });
	}
}

// ---------------------------------------------------------------------------
// Readline loop — sequential: await each command before reading the next.
// ---------------------------------------------------------------------------

async function main() {
	const rl = readline.createInterface({
		input: process.stdin,
		crlfDelay: Infinity,
	});

	for await (const line of rl) {
		const trimmed = line.trim();
		if (!trimmed) continue;

		let command;
		try {
			command = JSON.parse(trimmed);
		} catch (parseErr) {
			respond({ error: `invalid JSON: ${parseErr.message}` });
			continue;
		}

		try {
			await processCommand(command);
		} catch (err) {
			respond({ error: err instanceof Error ? err.message : String(err) });
		}
	}
}

// Catch any unhandled exceptions — write to stdout and keep running.
process.on('uncaughtException', (err) => {
	respond({ error: err instanceof Error ? err.message : String(err) });
});

process.on('unhandledRejection', (reason) => {
	const msg = reason instanceof Error ? reason.message : String(reason);
	respond({ error: msg });
});

main().catch((err) => {
	respond({ error: err instanceof Error ? err.message : String(err) });
	process.exit(1);
});
