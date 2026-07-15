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
 * 100-dim extractFeatures() vector (M1 MLP baseline regression checks), or
 * --obs-v2 (M3.4) for the (12, 77) schema-v2 observation (boost stages +
 * volatile flags) flattened to 924 floats.
 *
 * Opponent selection (M3.3):
 *   --opponent random|damagefirst   picks the built-in p2 AI (default random)
 *   --selfplay                      dual-seat mode: p2 is a second externally
 *                                   driven seat. reset/step use the dual
 *                                   protocol:
 *     {"cmd":"reset"} → {"p1":{"obs":[...],"mask":[...],"needs":bool},
 *                        "p2":{"obs":[...],"mask":[...],"needs":bool}}
 *     {"cmd":"step","action":<int|null>,"opp_action":<int|null>}
 *       → {"p1":{...},"p2":{...},"reward":<float>,"done":<bool>,"info":{}}
 *   Pass an action for exactly the seats whose "needs" was true; reward is
 *   always from p1's perspective (self-play trains the p1 seat).
 *
 * Per-reset opponent override (M3.4, --opponent-mix training):
 *   {"cmd":"reset","opponent":"random"|"damagefirst"|"self"} recreates the
 *   env with that opponent for the new episode. "self" switches this env to
 *   the dual-seat protocol (and back) — the reset/step response shapes follow
 *   the CURRENT episode's opponent, not the spawn-time flags.
 *
 * Simulation / forward-model commands (M4, MCTS):
 *   {"cmd":"sim_clone","determinize":bool,"seed":<int?>}
 *     → {"sim":<id>,"p1":{obs,mask,needs},"p2":{...},"done":bool,"info":{...}}
 *     Clones the CURRENT live battle into an independent BattleSim.
 *     determinize=true replaces the opponent's unrevealed Pokémon with sets
 *     sampled from the format generator (seeded by "seed" for reproducibility).
 *   {"cmd":"sim_step","sim":<id>,"action":<int|null>,"opp_action":<int|null>}
 *     → {"p1":{...},"p2":{...},"reward":<float>,"done":bool,"info":{...}}
 *     Dual-seat semantics: pass an action for exactly the seats whose "needs"
 *     was true. Reward is from p1's perspective, gym-identical shaping.
 *   {"cmd":"sim_fork","sim":<id>}
 *     → {"sim":<newId>,"p1":{...},"p2":{...},"done":bool,"info":{...}}
 *     Independent copy of a sim's current state (tree branching).
 *   {"cmd":"sim_free","sim":<id>}    → {"ok":true}
 *   {"cmd":"sim_free_all"}           → {"ok":true,"freed":<count>}
 */

const readline = require('readline');
const { PokemonGymEnv } = require('../dist/sim/tools/pokemon-gym');
const { BattleSim } = require('../dist/sim/tools/battle-sim');

const flat = process.argv.includes('--flat');
const obsV2 = process.argv.includes('--obs-v2');
if (flat && obsV2) {
	process.stdout.write(JSON.stringify({ error: '--flat and --obs-v2 are mutually exclusive' }) + '\n');
	process.exit(1);
}
const obsMode = flat ? 'flat' : (obsV2 ? 'structured-v2' : 'structured');
const selfplay = process.argv.includes('--selfplay');
const opponentArgIdx = process.argv.indexOf('--opponent');
const defaultOpponent = selfplay
	? 'self'
	: (opponentArgIdx !== -1 ? process.argv[opponentArgIdx + 1] : 'random');

if (!['random', 'damagefirst', 'self'].includes(defaultOpponent)) {
	process.stdout.write(JSON.stringify({ error: `unknown --opponent: ${defaultOpponent}` }) + '\n');
	process.exit(1);
}

/** Serialize one DualSeatState for the wire. */
function seatToJSON(seat) {
	return { obs: Array.from(seat.obs), mask: seat.mask, needs: seat.needsAction };
}

// Env is recreated on every reset() to avoid stream/worker accumulation.
let env = null;
let initialized = false;
// Opponent of the CURRENT episode; 'self' selects the dual-seat protocol.
let currentOpponent = defaultOpponent;

// Live BattleSim instances (M4 MCTS), keyed by id. Freed explicitly via
// sim_free/sim_free_all and implicitly on every reset (stale sims reference
// the previous battle).
let sims = new Map();
let nextSimId = 1;

/** Serialize a sim's current state (no reward — sim_clone/sim_fork shape). */
function simStateToJSON(id, sim) {
	return {
		sim: id,
		p1: seatToJSON(sim.seatState('p1')),
		p2: seatToJSON(sim.seatState('p2')),
		done: sim.done,
		info: { winner: sim.winner, turns: sim.turn },
	};
}

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
		const requested = command.opponent ?? currentOpponent;
		if (!['random', 'damagefirst', 'self'].includes(requested)) {
			respond({ error: `unknown opponent: ${requested}` });
			return;
		}
		if (env) env.destroy();
		sims.clear(); // stale sims reference the previous battle
		currentOpponent = requested;
		env = new PokemonGymEnv({ obsMode, opponent: currentOpponent });
		if (currentOpponent === 'self') {
			const result = await env.resetDual();
			initialized = true;
			respond({ p1: seatToJSON(result.p1), p2: seatToJSON(result.p2) });
		} else {
			const obsFloat32 = await env.reset();
			initialized = true;
			respond({ obs: Array.from(obsFloat32), mask: env.validActions() });
		}

	} else if (cmd === 'step') {
		if (!initialized) {
			respond({ error: 'not initialized' });
			return;
		}
		if (currentOpponent === 'self') {
			const action = command.action ?? null;
			const oppAction = command.opp_action ?? null;
			const result = await env.stepDual(action, oppAction);
			respond({
				p1: seatToJSON(result.p1),
				p2: seatToJSON(result.p2),
				reward: result.reward,
				done: result.done,
				info: result.info,
			});
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

	} else if (cmd === 'sim_clone') {
		if (!initialized) {
			respond({ error: 'not initialized' });
			return;
		}
		const sim = BattleSim.fromSnapshot(env.snapshot(), {
			determinize: !!command.determinize,
			perspective: command.perspective === 'p2' ? 'p2' : 'p1',
			seed: command.seed,
		});
		const id = nextSimId++;
		sims.set(id, sim);
		respond(simStateToJSON(id, sim));

	} else if (cmd === 'sim_step') {
		const sim = sims.get(command.sim);
		if (!sim) {
			respond({ error: `unknown sim id: ${command.sim}` });
			return;
		}
		const result = sim.step(command.action ?? null, command.opp_action ?? null);
		respond({
			p1: seatToJSON(result.p1),
			p2: seatToJSON(result.p2),
			reward: result.reward,
			done: result.done,
			info: result.info,
		});

	} else if (cmd === 'sim_fork') {
		const sim = sims.get(command.sim);
		if (!sim) {
			respond({ error: `unknown sim id: ${command.sim}` });
			return;
		}
		const fork = sim.fork();
		const id = nextSimId++;
		sims.set(id, fork);
		respond(simStateToJSON(id, fork));

	} else if (cmd === 'sim_free') {
		respond({ ok: sims.delete(command.sim) });

	} else if (cmd === 'sim_free_all') {
		const freed = sims.size;
		sims.clear();
		respond({ ok: true, freed });

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
