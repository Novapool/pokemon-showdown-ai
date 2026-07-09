'use strict';

/**
 * gym_bridge.js — Line-delimited JSON stdio server wrapping PokemonGymEnv.
 *
 * Reads one JSON command per line from stdin, writes one JSON response per
 * line to stdout. All gym methods are async; commands are processed
 * sequentially (one at a time) to preserve request/response ordering.
 *
 * Supported commands:
 *   {"cmd":"reset"}                   → {"obs":[...780 floats...]}  (or 100 with --flat)
 *   {"cmd":"step","action":3}         → {"obs":[...],"reward":0.01,"done":false,"info":{}}
 *   {"cmd":"valid_actions"}           → {"mask":[true,true,...]}  (length 9)
 *   {"cmd":"close"}                   → {"ok":true}  then process.exit(0)
 *
 * By default obs is the M2 structured (12, 65) token observation flattened
 * to 780 floats. Pass --flat on the command line to fall back to the legacy
 * 100-dim extractFeatures() vector (M1 MLP baseline regression checks).
 */

const readline = require('readline');
const { PokemonGymEnv } = require('../dist/sim/tools/pokemon-gym');

const obsMode = process.argv.includes('--flat') ? 'flat' : 'structured';

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
		env = new PokemonGymEnv({ obsMode });
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
