'use strict';

/**
 * Tests for PokemonGymEnv and extractFeatures.
 *
 * Run: npx mocha test/tools/gym.test.js --timeout 60000
 * Or via the normal test suite: npm test (after node build)
 */

const assert = require('../assert');
const { extractFeatures, OBS_SIZE } = require('../../dist/sim/tools/feature-extractor');
const { PokemonGymEnv } = require('../../dist/sim/tools/pokemon-gym');

// ---------------------------------------------------------------------------
// Helpers: minimal mock request objects
// ---------------------------------------------------------------------------

function makeMoveRequest(conditionOverride) {
	return {
		active: [
			{
				moves: [
					{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
				],
			},
		],
		side: {
			name: 'TestPlayer',
			id: 'p1',
			pokemon: [
				{
					ident: 'p1: Bulbasaur',
					details: 'Bulbasaur, L50, M',
					condition: conditionOverride || '200/300',
					active: true,
					stats: { atk: 50, def: 50, spa: 50, spd: 50, spe: 50 },
					moves: ['tackle'],
					baseAbility: 'overgrow',
					item: '',
					pokeball: 'pokeball',
				},
			],
		},
	};
}

// ---------------------------------------------------------------------------
// Feature extractor tests
// ---------------------------------------------------------------------------

describe('extractFeatures', () => {
	describe('shape consistency', () => {
		it('should return a Float32Array of length OBS_SIZE with no NaN values', () => {
			const result = extractFeatures(makeMoveRequest(), null);
			assert.equal(result.length, OBS_SIZE);
			assert([...result].every(v => !isNaN(v)), 'Observation vector must not contain NaN');
		});
	});

	describe('HP ratio parsing', () => {
		it('should encode HP ratio correctly for "150/250"', () => {
			const result = extractFeatures(makeMoveRequest('150/250'), null);
			assert(Math.abs(result[0] - 0.6) < 0.001, `Expected obs[0] ~0.6, got ${result[0]}`);
		});

		it('should encode fainted pokemon HP as 0.0', () => {
			const result = extractFeatures(makeMoveRequest('0/250 fnt'), null);
			assert.equal(result[0], 0.0);
		});
	});
});

// ---------------------------------------------------------------------------
// PokemonGymEnv tests
// ---------------------------------------------------------------------------

describe('PokemonGymEnv', () => {
	describe('reset', () => {
		it('should return an observation of the correct shape with no NaN values', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				const obs = await env.reset();
				assert.equal(obs.length, OBS_SIZE);
				assert([...obs].every(v => !isNaN(v)), 'Observation from reset() must not contain NaN');
			} finally {
				env.destroy();
			}
		});
	});

	describe('validActions', () => {
		it('should return a boolean array of length 9 after reset', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				await env.reset();
				const mask = env.validActions();
				assert.equal(mask.length, 9);
				assert(mask.every(v => typeof v === 'boolean'), 'Every element of the action mask must be a boolean');
			} finally {
				env.destroy();
			}
		});
	});

	describe('step', () => {
		it('should accept a legal move and return an observation of correct shape', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				await env.reset();
				const mask = env.validActions();
				const legalAction = mask.findIndex(v => v);
				if (legalAction === -1) return;
				const result = await env.step(legalAction);
				assert(!result.info.illegalMove, 'Legal action should not be flagged as illegal');
				assert.equal(result.obs.length, OBS_SIZE);
			} finally {
				env.destroy();
			}
		});

		it('should flag an out-of-range action as illegal', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				await env.reset();
				const result = await env.step(9);
				assert(result.info.illegalMove, 'Out-of-range action should be flagged as illegal');
			} finally {
				env.destroy();
			}
		});
	});

	describe('full battle', () => {
		it('should terminate within 200 steps', async function () {
			this.timeout(60000);
			const env = new PokemonGymEnv({ seed: [5, 6, 7, 8] });
			let result;
			try {
				await env.reset();
				for (let i = 0; i < 200; i++) {
					const mask = env.validActions();
					const action = mask.findIndex(v => v);
					result = await env.step(action >= 0 ? action : 0);
					if (result.done) break;
				}
				assert(result && result.done, 'Battle must terminate within 200 steps');
				assert(typeof result.info.winner === 'string', 'Completed battle must report a winner string');
			} finally {
				env.destroy();
			}
		});

		it('should keep all step rewards in the range [-1, +1]', async function () {
			this.timeout(60000);
			for (let trial = 0; trial < 2; trial++) {
				const env = new PokemonGymEnv({ seed: [trial, trial + 1, trial + 2, trial + 3] });
				try {
					await env.reset();
					let done = false;
					while (!done) {
						const mask = env.validActions();
						const action = mask.findIndex(v => v);
						const result = await env.step(action >= 0 ? action : 0);
						assert(result.reward >= -1 && result.reward <= 1, `Reward ${result.reward} is outside [-1, +1]`);
						done = result.done;
					}
				} finally {
					env.destroy();
				}
			}
		});

		it('should produce the same winner with the same seed (determinism)', async function () {
			this.timeout(60000);
			const seed = [42, 42, 42, 42];
			const winners = [];
			for (let trial = 0; trial < 2; trial++) {
				const env = new PokemonGymEnv({ seed: seed.slice() });
				try {
					await env.reset();
					let result;
					let done = false;
					while (!done) {
						const mask = env.validActions();
						const action = mask.findIndex(v => v);
						result = await env.step(action >= 0 ? action : 0);
						done = result.done;
					}
					winners.push(result.info.winner);
				} finally {
					env.destroy();
				}
			}
			assert.equal(winners[0], winners[1], `Same seed should produce the same winner (got ${winners.join(', ')})`);
		});
	});
});
