'use strict';

/**
 * Tests for BattleSim.fromTracked (M6 P2) — reconstructing a searchable local
 * battle from what a remote (ladder) client knows: its own request JSON + the
 * reveal/volatile trackers. Ground truth comes from a live PokemonGymEnv
 * battle treated as if it were remote.
 *
 * Run: npx mocha --no-config test/tools/battle-sim-tracked.test.js --timeout 60000
 */

const assert = require('../assert');
const { BattleSim } = require('../../dist/sim/tools/battle-sim');
const { PokemonGymEnv } = require('../../dist/sim/tools/pokemon-gym');
const { TOKEN_DIM_V2, N_TOKENS } = require('../../dist/sim/tools/feature-extractor');

const T = TOKEN_DIM_V2;

function makeRng(seed) {
	let s = seed >>> 0;
	return () => {
		s = (s * 1664525 + 1013904223) >>> 0;
		return s / 2 ** 32;
	};
}

function randomLegal(mask, rng) {
	const legal = [];
	for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
	return legal[Math.floor(rng() * legal.length)];
}

/**
 * Drive a live gym battle to a mid-game MOVE-request decision point and
 * return everything a ladder client would know there, plus the gym's own obs
 * as ground truth.
 */
async function trackedInputFromLiveBattle(seed, minSteps) {
	const env = new PokemonGymEnv({ seed, obsMode: 'structured-v2', opponent: 'damagefirst' });
	const rng = makeRng(seed[0] * 104729 + seed[2]);
	let obs = await env.reset();
	for (let step = 0; step < 200; step++) {
		const mask = env.validActions();
		const request = env._gymPlayer._lastActionable;
		if (step >= minSteps && mask.slice(0, 4).some(m => m) && request && request.active &&
			!request.active[0].moves.some(m => m.disabled)) {
			const snap = env.snapshot();
			const input = {
				formatid: 'gen1randombattle',
				seat: 'p1',
				request,
				trackers: snap.trackers,
				turnCount: snap.turnCount,
				obsMode: 'structured-v2',
			};
			return { env, input, liveObs: obs, liveMask: mask };
		}
		const result = await env.step(randomLegal(mask, rng));
		obs = result.obs;
		if (result.done) {
			env.destroy();
			return trackedInputFromLiveBattle([seed[0] + 1, seed[1], seed[2], seed[3] + 7], minSteps);
		}
	}
	throw new Error('no usable decision point found');
}

/** Dims of token `t` excluding own-side PP ratios (unknowable remotely). */
function comparableDims(t) {
	const dims = [];
	for (let d = 0; d < T; d++) {
		const isPP = d >= 41 && (d - 41) % 6 === 2;
		if (!isPP) dims.push(t * T + d);
	}
	return dims;
}

describe('BattleSim.fromTracked', function () {
	this.timeout(120000);

	let env, input, liveObs, liveMask;
	before(async () => {
		({ env, input, liveObs, liveMask } = await trackedInputFromLiveBattle([31, 32, 33, 34], 8));
	});
	after(() => env && env.destroy());

	it('reconstructs a battle where the searcher must act with the live mask', () => {
		const sim = BattleSim.fromTracked(input, { seed: 1 });
		assert(sim.needsAction('p1'), 'searcher seat should need an action');
		const state = sim.seatState('p1');
		// Own side is fully known (species/levels/moves/faints), so legality
		// must match the remote battle exactly at a clean move request.
		assert.deepEqual(state.mask, liveMask, 'reconstructed mask != live mask');
	});

	it('reproduces the live observation (own + opponent tokens, modulo PP)', () => {
		const sim = BattleSim.fromTracked(input, { seed: 2 });
		const obs = sim.seatState('p1').obs;
		assert.equal(obs.length, N_TOKENS * T);
		// gen1randombattle sets share the generator's stat conventions, so own
		// HP ratios round-trip; opponent tokens come from the same trackers.
		for (let t = 0; t < N_TOKENS; t++) {
			for (const i of comparableDims(t)) {
				assert(Math.abs(obs[i] - liveObs[i]) < 1e-2,
					`obs dim ${i} (token ${t}): reconstructed ${obs[i]} vs live ${liveObs[i]}`);
			}
		}
	});

	it('preserves revealed opponent mons (identity + revealed moves) and species clause', () => {
		const sim = BattleSim.fromTracked(input, { seed: 3 });
		const battle = sim._battle;
		const oppPokemon = battle.sides[1].pokemon;
		assert.equal(oppPokemon.length, 6);
		const species = oppPokemon.map(p => p.species.id);
		assert.equal(new Set(species).size, 6, 'species clause violated');
		const records = env.snapshot().trackers.revealRecords.p2;
		for (const [nickname, record] of records) {
			const rebuilt = oppPokemon.find(p => p.set.name === nickname);
			assert(rebuilt, `revealed opponent ${nickname} missing from reconstruction`);
			for (const moveId of record.moves) {
				assert(rebuilt.set.moves.map(m => String(m).toLowerCase().replace(/[^a-z0-9]/g, ''))
					.includes(moveId), `revealed move ${moveId} missing from ${nickname}'s rebuilt set`);
			}
		}
	});

	it('plays random legal actions to terminal without illegal moves', () => {
		const rng = makeRng(0xbeef);
		const sim = BattleSim.fromTracked(input, { seed: 4 });
		let result = sim.state();
		let guard = 1000;
		while (!result.done && guard-- > 0) {
			const p1a = result.p1.needsAction ? randomLegal(result.p1.mask, rng) : null;
			const p2a = result.p2.needsAction ? randomLegal(result.p2.mask, rng) : null;
			result = sim.step(p1a, p2a);
			assert(!result.info.illegalMove, 'illegal move during playout');
		}
		assert(result.done, 'playout did not terminate');
	});

	it('patches boosts and volatile flags onto the reconstructed actives', () => {
		const tweaked = JSON.parse(JSON.stringify(input));
		tweaked.trackers.volatiles.p2.boosts = { atk: 2, spe: -1 };
		tweaked.trackers.volatiles.p2.reflect = true;
		tweaked.trackers.volatiles.p1.substitute = true;
		const sim = BattleSim.fromTracked(tweaked, { seed: 5 });
		const [p1active, p2active] = [sim._battle.sides[0].active[0], sim._battle.sides[1].active[0]];
		assert.equal(p2active.boosts.atk, 2);
		assert.equal(p2active.boosts.spe, -1);
		assert(p2active.volatiles['reflect'], 'reflect volatile missing');
		assert(p1active.volatiles['substitute'], 'substitute volatile missing');
		assert.equal(p1active.volatiles['substitute'].hp, Math.floor(p1active.maxhp / 4));
	});

	it('reflects tracked faints in the reconstructed bench and switch mask', () => {
		// Fabricate: mark the first own bench mon fainted in the request copy.
		const tweaked = JSON.parse(JSON.stringify(input));
		const bench = tweaked.request.side.pokemon.find(p => !p.active && !p.condition.endsWith('fnt'));
		if (!bench) return; // battle had no living bench mon — nothing to test
		bench.condition = '0 fnt';
		const sim = BattleSim.fromTracked(tweaked, { seed: 6 });
		const name = String(bench.ident).slice(String(bench.ident).indexOf(':') + 1).trim();
		const rebuilt = sim._battle.sides[0].pokemon.find(p => p.set.name === name);
		assert(rebuilt.fainted, 'tracked faint not applied');
		const mask = sim.seatState('p1').mask;
		const idx = tweaked.request.side.pokemon.indexOf(bench);
		assert.equal(mask[3 + idx], false, 'fainted bench mon still switchable');
	});

	it('rejects force-switch and wait requests', () => {
		assert.throws(() => BattleSim.fromTracked(
			{ ...input, request: { forceSwitch: [true], side: input.request.side } }));
		assert.throws(() => BattleSim.fromTracked(
			{ ...input, request: { wait: true, side: input.request.side } }));
	});
});
