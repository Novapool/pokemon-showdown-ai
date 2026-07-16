'use strict';

/**
 * Tests for the M5.5 replay adapter (sim/tools/replay-adapter.ts).
 *
 * The load-bearing checks:
 *  1. Round-trip vs the live gym: a battle played through PokemonGymEnv,
 *     re-parsed from its own omniscient log, must yield byte-identical
 *     opponent tokens + volatile dims at every matched decision point
 *     (the adapter and the gym share ObservationTrackers — this verifies
 *     the adapter's snapshot TIMING matches the gym's request timing).
 *  2. The label↔slot invariant: action k must ground to the move written in
 *     own-active move slot k of the SAME observation (action 4+j to bench
 *     token j+1) — the property that makes BC labels transfer to the gym's
 *     request-ordered action space (M2.5 invariant).
 *
 * Run: npx mocha --no-config test/tools/replay-adapter.test.js --timeout 60000
 */

const assert = require('../assert');
const { ReplayAdapter } = require('../../dist/sim/tools/replay-adapter');
const { PokemonGymEnv } = require('../../dist/sim/tools/pokemon-gym');
const { TOKEN_DIM_V2 } = require('../../dist/sim/tools/feature-extractor');
const { Dex } = require('../../dist/sim');

const dex = Dex.mod('gen1');
const T = TOKEN_DIM_V2; // 77

function randomValidAction(mask, rng) {
	const legal = [];
	for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
	return legal[Math.floor(rng() * legal.length)];
}

// Deterministic LCG so the test is reproducible.
function makeRng(seed) {
	let s = seed >>> 0;
	return () => {
		s = (s * 1664525 + 1013904223) >>> 0;
		return s / 2 ** 32;
	};
}

/** Play one seeded gym battle; return its omniscient log + per-turn p1 obs. */
async function playGymBattle(seed) {
	const env = new PokemonGymEnv({ seed, obsMode: 'structured-v2', opponent: 'damagefirst' });
	const rng = makeRng(seed[0] * 7919 + seed[3]);
	const obsByTurn = new Map(); // turn -> p1 obs at that turn's move request
	let obs = await env.reset();
	let done = false;
	let steps = 0;
	while (!done && steps < 500) {
		const mask = env.validActions();
		const turnCount = env._omniscientLines.filter(l => l.startsWith('|turn|')).length;
		const isMoveRequest = mask.slice(0, 4).some(m => m);
		if (isMoveRequest && !obsByTurn.has(turnCount)) obsByTurn.set(turnCount, obs);
		const result = await env.step(randomValidAction(mask, rng));
		obs = result.obs;
		done = result.done;
		steps++;
	}
	const log = env._omniscientLines.join('\n');
	env.destroy();
	return { log, obsByTurn };
}

describe('ReplayAdapter', function () {
	this.timeout(120000);

	let parsed, obsByTurn;
	before(async () => {
		const battle = await playGymBattle([21, 22, 23, 24]);
		obsByTurn = battle.obsByTurn;
		parsed = new ReplayAdapter().parse(battle.log);
	});

	it('parses a gym-generated omniscient log into two non-empty trajectories', () => {
		assert(parsed, 'parse returned null');
		assert(parsed.p1.decisions.length > 0, 'p1 trajectory empty');
		assert(parsed.p2.decisions.length > 0, 'p2 trajectory empty');
		assert.equal(parsed.p1.win + parsed.p2.win, 1);
		for (const traj of [parsed.p1, parsed.p2]) {
			for (const dec of traj.decisions) {
				assert.equal(dec.obs.length, 12 * T);
				assert(dec.action >= 0 && dec.action <= 8, `action out of range: ${dec.action}`);
				assert([...dec.obs].every(v => !isNaN(v)), 'obs contains NaN');
			}
		}
	});

	it('round-trips opponent tokens + volatile dims byte-identically vs the live gym', () => {
		// Match p1 move decisions to the gym's per-turn move-request obs. The
		// adapter and gym both see all lines up to |turn|N before snapshotting.
		let compared = 0;
		for (const dec of parsed.p1.decisions) {
			if (dec.action >= 4) continue; // switches can be force-switch (mid-turn state)
			const gymObs = obsByTurn.get(dec.turn);
			if (!gymObs) continue;
			// Opponent tokens (6..11) come purely from ObservationTrackers in
			// both systems; own-active volatile dims (65..76) likewise.
			for (let i = 6 * T; i < 12 * T; i++) {
				assert.equal(dec.obs[i], gymObs[i], `opp-token mismatch at dim ${i} (turn ${dec.turn})`);
			}
			for (let i = 65; i < T; i++) {
				assert.equal(dec.obs[i], gymObs[i], `own-volatile mismatch at dim ${i} (turn ${dec.turn})`);
			}
			compared++;
		}
		assert(compared >= 10, `only ${compared} decisions compared — battle too short for a meaningful test`);
	});

	it('grounds move labels to the matching own-active move slot (invariant)', () => {
		let checked = 0;
		for (const traj of [parsed.p1, parsed.p2]) {
			for (const dec of traj.decisions) {
				if (dec.action >= 4) continue;
				const move = dex.moves.get(dec.resolved);
				assert(move.exists, `resolved move missing from dex: ${dec.resolved}`);
				const base = 41 + dec.action * 6; // T_MOVES + action * T_MOVE_DIM in token 0
				const expectedPower = Math.min(1, Math.max(0, move.basePower / 250));
				assert.equal(dec.obs[base + 0], Math.fround(expectedPower),
					`move slot ${dec.action} basePower mismatch for ${dec.resolved}`);
				const rawAcc = move.accuracy;
				const expectedAcc = (rawAcc === true || rawAcc === undefined) ? 1.0 : rawAcc / 100;
				assert.equal(dec.obs[base + 1], Math.fround(expectedAcc),
					`move slot ${dec.action} accuracy mismatch for ${dec.resolved}`);
				checked++;
			}
		}
		assert(checked >= 10, `only ${checked} move labels checked`);
	});

	it('grounds switch labels to the matching bench token (invariant)', () => {
		let checked = 0;
		for (const traj of [parsed.p1, parsed.p2]) {
			for (const dec of traj.decisions) {
				if (dec.action < 4) continue;
				const species = dex.species.get(dec.resolved);
				assert(species.exists, `resolved species missing from dex: ${dec.resolved}`);
				const offset = (dec.action - 3) * T; // bench token j+1 for action 4+j
				assert.equal(dec.obs[offset + 40], 0, 'switch target marked fainted');
				assert.equal(dec.obs[offset + 39], 0, 'switch target marked unknown');
				// type1 one-hot at dims 2..16 must match the species' first type
				const typeIdx = {
					normal: 0, fire: 1, water: 2, electric: 3, grass: 4, ice: 5, fighting: 6,
					poison: 7, ground: 8, flying: 9, psychic: 10, bug: 11, rock: 12, ghost: 13, dragon: 14,
				}[species.types[0].toLowerCase()];
				assert.equal(dec.obs[offset + 2 + typeIdx], 1.0,
					`bench token type mismatch for ${dec.resolved}`);
				checked++;
			}
		}
		assert(checked >= 3, `only ${checked} switch labels checked`);
	});

	it('cross-links simultaneous turn decisions symmetrically (opp-head labels)', () => {
		for (const dec of parsed.p1.decisions) {
			if (dec.oppAction < 0) continue;
			const partner = parsed.p2.decisions.find(d => d.turn === dec.turn && d.oppAction >= 0);
			assert(partner, `p1 turn ${dec.turn} has oppAction but no p2 partner decision`);
			assert.equal(dec.oppAction, partner.action, `oppAction != partner action (turn ${dec.turn})`);
			assert.equal(partner.oppAction, dec.action, `partner oppAction != own action (turn ${dec.turn})`);
		}
		assert(parsed.p1.decisions.some(d => d.oppAction >= 0), 'no cross-linked decisions at all');
	});

	it('marks exactly the last decision of each trajectory done', () => {
		for (const traj of [parsed.p1, parsed.p2]) {
			const dones = traj.decisions.filter(d => d.done);
			assert.equal(dones.length, 1);
			assert.equal(traj.decisions[traj.decisions.length - 1].done, true);
		}
	});

	it('handles nicknamed teams (OU-style logs)', () => {
		const log = [
			'|player|p1|Alice|1|1500',
			'|player|p2|Bob|2|1490',
			'|gen|1',
			'|tier|[Gen 1] OU',
			'|start',
			'|switch|p1a: Fluffy|Tauros, L100|353/353',
			'|switch|p2a: Eggy|Exeggutor, L100|393/393',
			'|turn|1',
			'|move|p1a: Fluffy|Body Slam|p2a: Eggy',
			'|-damage|p2a: Eggy|250/393',
			'|move|p2a: Eggy|Sleep Powder|p1a: Fluffy',
			'|-status|p1a: Fluffy|slp',
			'|upkeep',
			'|turn|2',
			'|switch|p1a: Chansey Lady|Chansey, L100|703/703',
			'|move|p2a: Eggy|Psychic|p1a: Chansey Lady',
			'|-damage|p1a: Chansey Lady|500/703',
			'|upkeep',
			'|win|Bob',
		].join('\n');
		const result = new ReplayAdapter().parse(log);
		assert(result, 'nicknamed log failed to parse');
		assert.equal(result.winner, 'p2');
		assert.equal(result.p2.win, 1);
		// p1 turn 1: Body Slam is Fluffy's only known move -> action 0
		assert.equal(result.p1.decisions[0].action, 0);
		assert.equal(result.p1.decisions[0].resolved, 'bodyslam');
		// p1 turn 2: voluntary switch to the nicknamed Chansey; only bench mon -> action 4
		assert.equal(result.p1.decisions[1].action, 4);
		assert.equal(result.p1.decisions[1].resolved, 'chansey');
		// p2's turn-1 decision should cross-link with p1's
		assert.equal(result.p2.decisions[0].oppAction, 0);
	});
});
