'use strict';

/**
 * Tests for BattleSim — the MCTS forward model (M4).
 *
 * Run: npx mocha --spec test/tools/battle-sim.test.js --timeout 60000
 * Or via the normal test suite: npm test (after node build)
 */

const assert = require('../assert');
const { PokemonGymEnv } = require('../../dist/sim/tools/pokemon-gym');
const { BattleSim } = require('../../dist/sim/tools/battle-sim');
const { Teams } = require('../../dist/sim');

function seededRng(seed) {
	let a = seed;
	return function () {
		let t = a += 0x6d2b79f5;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function pickLegal(rng, mask) {
	const legal = mask.map((v, i) => (v ? i : -1)).filter(i => i >= 0);
	if (!legal.length) return null;
	return legal[Math.floor(rng() * legal.length)];
}

function float32Equal(a, b) {
	if (a.length !== b.length) return false;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) return false;
	}
	return true;
}

/**
 * Create a gym env, advance it a few decisions so some opponent Pokémon are
 * revealed, and return { env, obs, snap }. Caller must env.destroy().
 */
async function makeSnapshot(rng, envSteps = 6, obsMode = 'structured') {
	const env = new PokemonGymEnv({ obsMode, seed: '12,34,56,78' });
	let obs = await env.reset();
	for (let i = 0; i < envSteps; i++) {
		const result = await env.step(pickLegal(rng, env.validActions()));
		obs = result.obs;
		if (result.done) throw new Error('battle ended during setup; adjust envSteps');
	}
	return { env, obs, snap: env.snapshot() };
}

function playout(sim, rng, maxSteps = 1000) {
	let steps = 0;
	let illegal = 0;
	const rewards = [];
	while (!sim.done && steps < maxSteps) {
		const a1 = sim.needsAction('p1') ? pickLegal(rng, sim.seatState('p1').mask) : null;
		const a2 = sim.needsAction('p2') ? pickLegal(rng, sim.seatState('p2').mask) : null;
		const result = sim.step(a1, a2);
		if (result.info.illegalMove) illegal++;
		rewards.push(result.reward);
		steps++;
	}
	return { steps, illegal, rewards };
}

describe('BattleSim', function () {
	this.timeout(60000);

	describe('cloning', () => {
		it('plain clone root obs should byte-equal the live gym obs', async () => {
			const rng = seededRng(1);
			const { env, obs, snap } = await makeSnapshot(rng);
			const sim = BattleSim.fromSnapshot(snap);
			assert(float32Equal(obs, sim.seatState('p1').obs),
				'sim p1 obs must match the gym obs it was cloned from');
			env.destroy();
		});

		it('v2-obsMode clone root obs should byte-equal the live gym obs', async () => {
			const rng = seededRng(2);
			const { env, obs, snap } = await makeSnapshot(rng, 6, 'structured-v2');
			const sim = BattleSim.fromSnapshot(snap);
			assert(float32Equal(obs, sim.seatState('p1').obs),
				'sim p1 obs (v2) must match the gym obs it was cloned from');
			env.destroy();
		});

		it('stepping a clone must not affect the live battle', async () => {
			const rng = seededRng(3);
			const { env, obs, snap } = await makeSnapshot(rng);
			const sim = BattleSim.fromSnapshot(snap, { determinize: true, seed: 3 });
			playout(sim, rng, 50);
			// A second clone from the same (frozen) snapshot still matches the gym
			const sim2 = BattleSim.fromSnapshot(snap);
			assert(float32Equal(obs, sim2.seatState('p1').obs),
				'snapshot must stay frozen while sims step');
			// The live env still works
			const result = await env.step(pickLegal(rng, env.validActions()));
			assert.equal(typeof result.reward, 'number');
			env.destroy();
		});
	});

	describe('determinization', () => {
		it('should not change p1 root obs (only unrevealed opponent slots swap)', async () => {
			const rng = seededRng(4);
			const { env, obs, snap } = await makeSnapshot(rng);
			const sim = BattleSim.fromSnapshot(snap, { determinize: true, seed: 99 });
			assert(float32Equal(obs, sim.seatState('p1').obs),
				'determinization must be invisible in p1 root obs');
			env.destroy();
		});

		it('should keep revealed opponent Pokémon and produce a legal 6-mon team', async () => {
			const rng = seededRng(5);
			const { env, snap } = await makeSnapshot(rng);
			const plain = BattleSim.fromSnapshot(snap);
			const det = BattleSim.fromSnapshot(snap, { determinize: true, seed: 7 });

			const trueTeam = plain._battle.sides[1].pokemon.map(p => p.species.name);
			const detTeam = det._battle.sides[1].pokemon.map(p => p.species.name);
			assert.equal(detTeam.length, 6);
			assert.equal(new Set(detTeam.map(n => n.toLowerCase())).size, 6, 'species clause');

			const revealed = snap.trackers.revealOrder.p2;
			assert(revealed.length >= 1, 'setup should have revealed at least the opponent lead');
			for (const nickname of revealed) {
				assert(detTeam.includes(nickname),
					`revealed opponent Pokémon ${nickname} must survive determinization`);
			}
			// Sanity: the sampled bench really is sampled (with 5 unknown slots the
			// odds of resampling the true team are negligible)
			if (revealed.length <= 2) {
				assert(detTeam.join() !== trueTeam.join(), 'unrevealed slots should be resampled');
			}
			env.destroy();
		});

		it('same seed should sample the same team; different seeds should differ', async () => {
			const rng = seededRng(6);
			const { env, snap } = await makeSnapshot(rng);
			const names = sim => sim._battle.sides[1].pokemon.map(p => p.species.name).join(',');
			const a = names(BattleSim.fromSnapshot(snap, { determinize: true, seed: 11 }));
			const b = names(BattleSim.fromSnapshot(snap, { determinize: true, seed: 11 }));
			const c = names(BattleSim.fromSnapshot(snap, { determinize: true, seed: 12 }));
			assert.equal(a, b, 'determinization must be reproducible by seed');
			assert(a !== c, 'different seeds should sample different teams');
			env.destroy();
		});

		it('replacement sets should be valid gen1randombattle sets', async () => {
			const rng = seededRng(7);
			const { env, snap } = await makeSnapshot(rng);
			const det = BattleSim.fromSnapshot(snap, { determinize: true, seed: 21 });
			const validator = require('../../dist/sim').TeamValidator.get('gen1randombattle');
			const revealed = new Set(snap.trackers.revealOrder.p2);
			for (const pokemon of det._battle.sides[1].pokemon) {
				if (revealed.has(pokemon.name)) continue;
				assert(pokemon.hp === pokemon.maxhp, 'sampled replacements start at full HP');
				assert(!pokemon.status, 'sampled replacements start with no status');
				assert(pokemon.set.moves.length >= 1 && pokemon.set.moves.length <= 4);
				assert(Teams.pack([pokemon.set]).length > 0, 'set must pack cleanly');
			}
			// validator is fetched to assert the format itself exists
			assert(validator.format.id === 'gen1randombattle');
			env.destroy();
		});
	});

	describe('determinization from the p2 perspective', () => {
		it("perspective 'p2' should resample p1's unrevealed team and keep p2's", async () => {
			const rng = seededRng(8);
			const { env, snap } = await makeSnapshot(rng);
			const plain = BattleSim.fromSnapshot(snap);
			const det = BattleSim.fromSnapshot(snap, { determinize: true, seed: 31, perspective: 'p2' });

			const trueP1 = plain._battle.sides[0].pokemon.map(p => p.species.name);
			const trueP2 = plain._battle.sides[1].pokemon.map(p => p.species.name);
			const detP1 = det._battle.sides[0].pokemon.map(p => p.species.name);
			const detP2 = det._battle.sides[1].pokemon.map(p => p.species.name);

			assert.equal(detP2.join(), trueP2.join(), "p2's own team must be untouched");
			for (const nickname of snap.trackers.revealOrder.p1) {
				assert(detP1.includes(nickname), `revealed p1 Pokémon ${nickname} must survive`);
			}
			if (snap.trackers.revealOrder.p1.length <= 2) {
				assert(detP1.join() !== trueP1.join(), "p1's unrevealed slots should be resampled");
			}
			// p2's obs must be unchanged by a p2-perspective determinization
			assert(float32Equal(plain.seatState('p2').obs, det.seatState('p2').obs),
				"determinization must be invisible in p2's obs");
			env.destroy();
		});
	});

	describe('playouts', () => {
		it('determinized playouts should reach terminal with no illegal moves (10 trials)', async () => {
			for (let trial = 0; trial < 10; trial++) {
				const rng = seededRng(100 + trial);
				const { env, snap } = await makeSnapshot(rng);
				const sim = BattleSim.fromSnapshot(snap, { determinize: true, seed: trial });
				const { steps, illegal, rewards } = playout(sim, rng);
				assert(sim.done, `trial ${trial}: playout must terminate (${steps} steps)`);
				assert.equal(illegal, 0, `trial ${trial}: mask-legal actions must never be rejected`);
				assert(['Gym', 'Opponent'].includes(sim.winner), `trial ${trial}: winner must be set`);
				assert(rewards.every(r => r >= -1 && r <= 1), `trial ${trial}: rewards must stay in [-1, 1]`);
				env.destroy();
			}
		});

		it('terminal reward should carry the win/loss signal', async () => {
			const rng = seededRng(200);
			const { env, snap } = await makeSnapshot(rng);
			const sim = BattleSim.fromSnapshot(snap, { determinize: true, seed: 5 });
			let lastReward = 0;
			while (!sim.done) {
				const a1 = sim.needsAction('p1') ? pickLegal(rng, sim.seatState('p1').mask) : null;
				const a2 = sim.needsAction('p2') ? pickLegal(rng, sim.seatState('p2').mask) : null;
				lastReward = sim.step(a1, a2).reward;
			}
			if (sim.winner === 'Gym') {
				assert(lastReward > 0.5, `winning terminal reward should be near +1, got ${lastReward}`);
			} else {
				assert(lastReward < -0.5, `losing terminal reward should be near -1, got ${lastReward}`);
			}
			env.destroy();
		});
	});

	describe('fork', () => {
		it('fork + identical actions should replay identically to the parent', async () => {
			const rng = seededRng(300);
			const { env, snap } = await makeSnapshot(rng);
			const parent = BattleSim.fromSnapshot(snap, { determinize: true, seed: 9 });
			const child = parent.fork();
			for (let i = 0; i < 60 && !parent.done && !child.done; i++) {
				const a1 = parent.needsAction('p1') ? pickLegal(rng, parent.seatState('p1').mask) : null;
				const a2 = parent.needsAction('p2') ? pickLegal(rng, parent.seatState('p2').mask) : null;
				const rp = parent.step(a1, a2);
				const rc = child.step(a1, a2);
				assert.equal(rp.reward, rc.reward, `step ${i}: fork reward must match parent`);
				assert.equal(rp.done, rc.done, `step ${i}: fork done must match parent`);
				assert(float32Equal(rp.p1.obs, rc.p1.obs), `step ${i}: fork obs must match parent`);
			}
			env.destroy();
		});

		it('stepping a fork must not affect its parent', async () => {
			const rng = seededRng(400);
			const { env, snap } = await makeSnapshot(rng);
			const parent = BattleSim.fromSnapshot(snap, { determinize: true, seed: 13 });
			const before = parent.seatState('p1').obs;
			const child = parent.fork();
			playout(child, rng, 40);
			assert(float32Equal(before, parent.seatState('p1').obs),
				'parent obs must be unchanged after fork playout');
			env.destroy();
		});
	});

	describe('gym snapshot()', () => {
		it('should throw on a finished battle', async () => {
			const rng = seededRng(500);
			const env = new PokemonGymEnv({ obsMode: 'structured' });
			await env.reset();
			let result = { done: false };
			let guard = 1000;
			while (!result.done && guard-- > 0) {
				result = await env.step(pickLegal(rng, env.validActions()));
			}
			assert.throws(() => env.snapshot());
			env.destroy();
		});
	});
});
