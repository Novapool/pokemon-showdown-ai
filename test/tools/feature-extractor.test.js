'use strict';

/**
 * Extractor-level unit tests for Observation Schema v3 (M7, Job 1.1).
 *
 * Scope: exercises `extractFeaturesStructured`'s new v3 path directly (dims
 * 77–85) plus the v2/v3 byte-equality invariant on dims 0–76. Integration-level
 * gym/battle tests live in test/tools/gym.test.js (Job 3.1) — this file stays
 * at the pure-function extractor level to avoid overlapping that job's file.
 *
 * Run: npx mocha --no-config test/tools/feature-extractor.test.js --timeout 60000
 */

const assert = require('../assert');
const {
	extractFeaturesStructured,
	TOKEN_DIM,
	TOKEN_DIM_V2,
	TOKEN_DIM_V3,
	N_TOKENS,
} = require('../../dist/sim/tools/feature-extractor');
const {
	V3_TYPE_EFF,
	V3_FLAG_RECHARGE,
	V3_FLAG_SELFKO,
	V3_FLAG_PRIORITY,
	V3_INFLICTED_STATUS,
	V3_SLEEP_CLAUSE,
	MOVE_STATUS_ID_MAX,
} = require('../../dist/sim/tools/type-chart-v3');

// ---------------------------------------------------------------------------
// Helpers: minimal mock request objects (mirrors gym.test.js conventions)
// ---------------------------------------------------------------------------

function makeMoveRequest(moves, conditionOverride) {
	const moveList = moves || [
		{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
	];
	return {
		active: [{ moves: moveList }],
		side: {
			name: 'TestPlayer',
			id: 'p1',
			pokemon: [
				{
					ident: 'p1: Bulbasaur',
					details: 'Bulbasaur, L100, M',
					condition: conditionOverride || '200/300',
					active: true,
					stats: { atk: 50, def: 50, spa: 50, spd: 50, spe: 50 },
					moves: moveList.map(m => m.id),
					baseAbility: 'overgrow',
					item: '',
					pokeball: 'pokeball',
				},
			],
		},
	};
}

function neutralVolatiles() {
	return {
		boosts: [0, 0, 0, 0, 0, 0, 0],
		reflect: false,
		lightScreen: false,
		substitute: false,
		leechSeed: false,
		toxicCounter: 0,
	};
}

function bothVolatiles() {
	return { own: neutralVolatiles(), opp: neutralVolatiles() };
}

const OWN = 0; // own active token offset (dim 0)

// ---------------------------------------------------------------------------

describe('extractFeaturesStructured — schema v3 (M7)', () => {
	it('returns 12 × TOKEN_DIM_V3 (=86) floats with no NaN/inf when v3Info is passed', () => {
		const result = extractFeaturesStructured(makeMoveRequest(), [], bothVolatiles(), { sleepClause: false });
		assert.equal(TOKEN_DIM_V3, 86);
		assert.equal(result.length, N_TOKENS * TOKEN_DIM_V3);
		assert([...result].every(v => Number.isFinite(v)), 'v3 observation must not contain NaN/inf');
	});

	describe('dims 0–76 byte-equality with v2', () => {
		it('keeps every token dim 0–76 byte-identical between a v2 and a v3 call on the same input', () => {
			const request = makeMoveRequest(
				[{ move: 'Fire Blast', id: 'fireblast', pp: 5, maxpp: 5, target: 'normal', disabled: false }],
				'150/250 par'
			);
			const opponent = [
				{ details: 'Slowbro, L100', condition: '80/100', active: true, moves: ['surf'] },
				{ details: 'Gengar, L100', condition: '90/100', active: false, moves: ['lick'] },
			];
			const vol = bothVolatiles();
			const v2 = extractFeaturesStructured(request, opponent, vol);
			const v3 = extractFeaturesStructured(request, opponent, vol, { sleepClause: true });

			assert.equal(v2.length, N_TOKENS * TOKEN_DIM_V2);
			assert.equal(v3.length, N_TOKENS * TOKEN_DIM_V3);
			for (let t = 0; t < N_TOKENS; t++) {
				for (let d = 0; d < TOKEN_DIM_V2; d++) {
					assert.equal(
						v3[t * TOKEN_DIM_V3 + d], v2[t * TOKEN_DIM_V2 + d],
						`v3 token ${t} dim ${d} diverged from v2`
					);
				}
			}
		});

		it('keeps dims 0–64 identical to v1 as well (transitive through v2)', () => {
			const request = makeMoveRequest();
			const opponent = [{ details: 'Gengar, L100', condition: '80/100', active: true, moves: ['lick'] }];
			const v1 = extractFeaturesStructured(request, opponent);
			const v3 = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			for (let t = 0; t < N_TOKENS; t++) {
				for (let d = 0; d < TOKEN_DIM; d++) {
					assert.equal(
						v3[t * TOKEN_DIM_V3 + d], v1[t * TOKEN_DIM + d],
						`v3 token ${t} dim ${d} diverged from v1`
					);
				}
			}
		});
	});

	describe('dims 77–80 — per-move-slot type effectiveness vs opponent active', () => {
		it('encodes 4x (Ice Beam vs Dragonite [Dragon/Flying]) as 1.0', () => {
			const request = makeMoveRequest([
				{ move: 'Ice Beam', id: 'icebeam', pp: 10, maxpp: 10, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Dragonite, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert(Math.abs(result[OWN + V3_TYPE_EFF] - 1.0) < 1e-6, `expected 4x->1.0, got ${result[OWN + V3_TYPE_EFF]}`);
		});

		it('encodes 0.5x (Fire Blast vs Slowbro [Water/Psychic]) as 0.125', () => {
			const request = makeMoveRequest([
				{ move: 'Fire Blast', id: 'fireblast', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Slowbro, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert(Math.abs(result[OWN + V3_TYPE_EFF] - 0.125) < 1e-6, `expected 0.5x->0.125, got ${result[OWN + V3_TYPE_EFF]}`);
		});

		it('encodes 0x immunity (Explosion vs Gengar [Ghost/Poison]) as 0.0', () => {
			const request = makeMoveRequest([
				{ move: 'Explosion', id: 'explosion', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Gengar, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_TYPE_EFF], 0.0);
		});

		it('encodes empty move slots as 0.0 and never NaN', () => {
			const request = makeMoveRequest([
				{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Slowbro, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			// slots 1–3 have no move -> 0.0
			assert.equal(result[OWN + V3_TYPE_EFF + 1], 0.0);
			assert.equal(result[OWN + V3_TYPE_EFF + 2], 0.0);
			assert.equal(result[OWN + V3_TYPE_EFF + 3], 0.0);
			assert([...result].every(v => Number.isFinite(v)));
		});

		it('encodes each of 4 slots independently in order', () => {
			const request = makeMoveRequest([
				{ move: 'Ice Beam', id: 'icebeam', pp: 10, maxpp: 10, target: 'normal', disabled: false },
				{ move: 'Fire Blast', id: 'fireblast', pp: 5, maxpp: 5, target: 'normal', disabled: false },
				{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
				{ move: 'Explosion', id: 'explosion', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			// Dragonite [Dragon/Flying] per engine gen1 data: Ice 4x, Fire 0.5x, Normal 1x, Explosion(Normal) 1x
			const opponent = [{ details: 'Dragonite, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert(Math.abs(result[OWN + V3_TYPE_EFF + 0] - 1.0) < 1e-6);   // Ice 4x -> 1.0
			assert(Math.abs(result[OWN + V3_TYPE_EFF + 1] - 0.125) < 1e-6); // Fire 0.5x -> 0.125
			assert(Math.abs(result[OWN + V3_TYPE_EFF + 2] - 0.25) < 1e-6);  // Normal 1x -> 0.25
			assert(Math.abs(result[OWN + V3_TYPE_EFF + 3] - 0.25) < 1e-6);  // Explosion(Normal) 1x -> 0.25
		});

		it('defaults type-eff to 0.0 when the opponent active is unknown (no defender types)', () => {
			const request = makeMoveRequest([
				{ move: 'Ice Beam', id: 'icebeam', pp: 10, maxpp: 10, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_TYPE_EFF], 0.0);
		});
	});

	describe('dims 81–83 — aggregated move effect flags', () => {
		it('sets recharge=1 for a Hyper Beam move set', () => {
			const request = makeMoveRequest([
				{ move: 'Hyper Beam', id: 'hyperbeam', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_FLAG_RECHARGE], 1.0);
			assert.equal(result[OWN + V3_FLAG_SELFKO], 0.0);
		});

		it('sets self-KO=1 for Explosion/Selfdestruct', () => {
			const request = makeMoveRequest([
				{ move: 'Explosion', id: 'explosion', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_FLAG_SELFKO], 1.0);
		});

		it('sets priority=1 for Quick Attack', () => {
			const request = makeMoveRequest([
				{ move: 'Quick Attack', id: 'quickattack', pp: 30, maxpp: 30, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_FLAG_PRIORITY], 1.0);
		});

		it('leaves all flags 0 for a plain damaging move (Tackle)', () => {
			const request = makeMoveRequest([
				{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_FLAG_RECHARGE], 0.0);
			assert.equal(result[OWN + V3_FLAG_SELFKO], 0.0);
			assert.equal(result[OWN + V3_FLAG_PRIORITY], 0.0);
		});

		it('OR-aggregates flags across a mixed move set', () => {
			const request = makeMoveRequest([
				{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
				{ move: 'Quick Attack', id: 'quickattack', pp: 30, maxpp: 30, target: 'normal', disabled: false },
				{ move: 'Explosion', id: 'explosion', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_FLAG_PRIORITY], 1.0);
			assert.equal(result[OWN + V3_FLAG_SELFKO], 1.0);
			assert.equal(result[OWN + V3_FLAG_RECHARGE], 0.0);
		});
	});

	describe('dim 84 — inflicted-status id (normalised to [0,1])', () => {
		it('sets paralysis (id 3) for Thunder Wave, normalised by MOVE_STATUS_ID_MAX', () => {
			const request = makeMoveRequest([
				{ move: 'Thunder Wave', id: 'thunderwave', pp: 20, maxpp: 20, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert(Math.abs(result[OWN + V3_INFLICTED_STATUS] - 3 / MOVE_STATUS_ID_MAX) < 1e-6,
				`expected ${3 / MOVE_STATUS_ID_MAX}, got ${result[OWN + V3_INFLICTED_STATUS]}`);
		});

		it('sets sleep (id 5) for Hypnosis', () => {
			const request = makeMoveRequest([
				{ move: 'Hypnosis', id: 'hypnosis', pp: 20, maxpp: 20, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert(Math.abs(result[OWN + V3_INFLICTED_STATUS] - 5 / MOVE_STATUS_ID_MAX) < 1e-6);
		});

		it('leaves inflicted-status 0 for a non-status move (Tackle)', () => {
			const request = makeMoveRequest([
				{ move: 'Tackle', id: 'tackle', pp: 35, maxpp: 35, target: 'normal', disabled: false },
			]);
			const result = extractFeaturesStructured(request, [], bothVolatiles(), { sleepClause: false });
			assert.equal(result[OWN + V3_INFLICTED_STATUS], 0.0);
		});
	});

	describe('dim 85 — Sleep Clause flag', () => {
		it('places the flag value on every token when set', () => {
			const request = makeMoveRequest();
			const opponent = [{ details: 'Gengar, L100', condition: '80/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: true });
			for (let t = 0; t < N_TOKENS; t++) {
				assert.equal(result[t * TOKEN_DIM_V3 + V3_SLEEP_CLAUSE], 1.0, `token ${t} sleep-clause not set`);
			}
		});

		it('places 0.0 on every token when cleared', () => {
			const result = extractFeaturesStructured(makeMoveRequest(), [], bothVolatiles(), { sleepClause: false });
			for (let t = 0; t < N_TOKENS; t++) {
				assert.equal(result[t * TOKEN_DIM_V3 + V3_SLEEP_CLAUSE], 0.0);
			}
		});
	});

	describe('edge cases — no NaN/inf on degenerate input', () => {
		it('handles unknown move ids without NaN', () => {
			const request = makeMoveRequest([
				{ move: 'Bogusmove', id: 'bogusmove', pp: 5, maxpp: 5, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Slowbro, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert([...result].every(v => Number.isFinite(v)), 'unknown move must not produce NaN/inf');
			assert.equal(result[OWN + V3_TYPE_EFF], 0.0); // unknown type -> 0.0
		});

		it('handles an unknown opponent species without NaN', () => {
			const request = makeMoveRequest([
				{ move: 'Ice Beam', id: 'icebeam', pp: 10, maxpp: 10, target: 'normal', disabled: false },
			]);
			const opponent = [{ details: 'Notarealmon, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: false });
			assert([...result].every(v => Number.isFinite(v)));
			assert.equal(result[OWN + V3_TYPE_EFF], 0.0); // no defender types -> 0.0
		});

		it('handles a switch request (no active moves) without NaN', () => {
			const request = {
				forceSwitch: [true],
				side: {
					name: 'TestPlayer', id: 'p1',
					pokemon: [
						{
							ident: 'p1: Bulbasaur', details: 'Bulbasaur, L100, M', condition: '200/300',
							active: true, moves: ['tackle'], baseAbility: 'overgrow', item: '', pokeball: 'pokeball',
						},
					],
				},
			};
			const opponent = [{ details: 'Slowbro, L100', condition: '100/100', active: true, moves: [] }];
			const result = extractFeaturesStructured(request, opponent, bothVolatiles(), { sleepClause: true });
			assert.equal(result.length, N_TOKENS * TOKEN_DIM_V3);
			assert([...result].every(v => Number.isFinite(v)));
			// switch request -> own active has no moves -> type-eff all 0
			assert.equal(result[OWN + V3_TYPE_EFF], 0.0);
			// sleep clause still placed
			assert.equal(result[OWN + V3_SLEEP_CLAUSE], 1.0);
		});
	});
});
