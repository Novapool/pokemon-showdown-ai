'use strict';

/**
 * Tests for PokemonGymEnv and extractFeatures.
 *
 * Run: npx mocha test/tools/gym.test.js --timeout 60000
 * Or via the normal test suite: npm test (after node build)
 */

const assert = require('../assert');
const { extractFeatures, OBS_SIZE, extractFeaturesStructured, TOKEN_DIM, TOKEN_DIM_V2, N_TOKENS } =
	require('../../dist/sim/tools/feature-extractor');
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

describe('extractFeaturesStructured', () => {
	it('should return a Float32Array of length N_TOKENS * TOKEN_DIM with no NaN values', () => {
		const result = extractFeaturesStructured(makeMoveRequest(), []);
		assert.equal(TOKEN_DIM, 65);
		assert.equal(N_TOKENS, 12);
		assert.equal(result.length, N_TOKENS * TOKEN_DIM);
		assert([...result].every(v => !isNaN(v)), 'Structured observation must not contain NaN');
	});

	it('should mark unrevealed opponent bench tokens as unknown with HP ratio 1.0', () => {
		// No opponent info at all -> opponent active + all 5 bench slots are unknown.
		const result = extractFeaturesStructured(makeMoveRequest(), []);
		const activeOffset = 6 * TOKEN_DIM;
		assert.equal(result[activeOffset + 0], 1.0); // HP ratio
		assert.equal(result[activeOffset + 39], 1.0); // unknown flag
		for (let j = 0; j < 5; j++) {
			const offset = (7 + j) * TOKEN_DIM;
			assert.equal(result[offset + 0], 1.0);
			assert.equal(result[offset + 39], 1.0);
		}
	});

	it('should mark a fainted opponent Pokemon with fainted_flag=1 and HP=0', () => {
		const result = extractFeaturesStructured(makeMoveRequest(), [
			{ details: 'Gengar, L100', condition: '0 fnt', active: true, moves: [] },
		]);
		const activeOffset = 6 * TOKEN_DIM;
		assert.equal(result[activeOffset + 0], 0.0); // HP ratio
		assert.equal(result[activeOffset + 40], 1.0); // fainted flag
		assert.equal(result[activeOffset + 39], 0.0); // not unknown
	});

	it('should distinguish unknown (HP=1, unknown_flag=1) from fainted (HP=0, fainted_flag=1)', () => {
		const unknown = extractFeaturesStructured(makeMoveRequest(), []);
		const fainted = extractFeaturesStructured(makeMoveRequest(), [
			{ details: 'Gengar, L100', condition: '0 fnt', active: true, moves: [] },
		]);
		const activeOffset = 6 * TOKEN_DIM;
		assert.notEqual(unknown[activeOffset + 0], fainted[activeOffset + 0]);
	});

	describe('schema v2 (M3.4)', () => {
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

		it('should return 12 × TOKEN_DIM_V2 floats when volatiles are passed', () => {
			const result = extractFeaturesStructured(makeMoveRequest(), [], {
				own: neutralVolatiles(), opp: neutralVolatiles(),
			});
			assert.equal(TOKEN_DIM_V2, 77);
			assert.equal(result.length, N_TOKENS * TOKEN_DIM_V2);
			assert([...result].every(v => !isNaN(v)), 'v2 observation must not contain NaN');
		});

		it('should keep the first 65 dims of every token byte-identical to v1', () => {
			const request = makeMoveRequest('150/250 par');
			const opponent = [
				{ details: 'Gengar, L100', condition: '80/100', active: true, moves: ['lick'] },
			];
			const v1 = extractFeaturesStructured(request, opponent);
			const v2 = extractFeaturesStructured(request, opponent, {
				own: neutralVolatiles(), opp: neutralVolatiles(),
			});
			for (let t = 0; t < N_TOKENS; t++) {
				for (let d = 0; d < TOKEN_DIM; d++) {
					assert.equal(
						v2[t * TOKEN_DIM_V2 + d], v1[t * TOKEN_DIM + d],
						`v2 token ${t} dim ${d} diverged from v1`
					);
				}
			}
		});

		it('should write boost stages and volatile flags on the two active tokens only', () => {
			const own = neutralVolatiles();
			own.boosts = [2, 0, -1, 6, 0, 0, 6]; // atk +2, spe -1, spa/spd +6 (Amnesia ×3)
			own.reflect = true;
			own.toxicCounter = 4;
			const opp = neutralVolatiles();
			opp.substitute = true;
			opp.leechSeed = true;
			opp.lightScreen = true;
			const result = extractFeaturesStructured(makeMoveRequest(), [
				{ details: 'Gengar, L100', condition: '80/100', active: true, moves: [] },
			], { own, opp });

			const ownBase = 0;
			assert(Math.abs(result[ownBase + 65] - 2 / 6) < 1e-6);   // atk
			assert(Math.abs(result[ownBase + 67] - (-1 / 6)) < 1e-6); // spe
			assert.equal(result[ownBase + 68], 1.0);                  // spa capped at stage 6
			assert.equal(result[ownBase + 72], 1.0);                  // reflect
			assert.equal(result[ownBase + 73], 0.0);                  // no light screen
			assert(Math.abs(result[ownBase + 76] - 4 / 16) < 1e-6);   // toxic counter

			const oppBase = 6 * TOKEN_DIM_V2;
			assert.equal(result[oppBase + 73], 1.0); // light screen
			assert.equal(result[oppBase + 74], 1.0); // substitute
			assert.equal(result[oppBase + 75], 1.0); // leech seed

			// Bench tokens must stay neutral in every v2 dim
			for (const t of [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]) {
				for (let d = 65; d < TOKEN_DIM_V2; d++) {
					assert.equal(result[t * TOKEN_DIM_V2 + d], 0.0, `bench token ${t} dim ${d} not neutral`);
				}
			}
		});
	});

	it('should place own bench tokens in side.pokemon[1..5] request-slot order', () => {
		const request = makeMoveRequest();
		request.side.pokemon.push(
			{ ident: 'p1: Squirtle', details: 'Squirtle, L50, M', condition: '100/100', active: false, stats: {}, moves: ['tackle'], baseAbility: 'torrent', item: '', pokeball: 'pokeball' },
			{ ident: 'p1: Charmander', details: 'Charmander, L50, M', condition: '0 fnt', active: false, stats: {}, moves: ['scratch'], baseAbility: 'blaze', item: '', pokeball: 'pokeball' }
		);
		const result = extractFeaturesStructured(request, []);
		const slot1Offset = 1 * TOKEN_DIM;
		const slot2Offset = 2 * TOKEN_DIM;
		assert.equal(result[slot1Offset + 0], 1.0); // Squirtle alive, full HP
		assert.equal(result[slot2Offset + 40], 1.0); // Charmander fainted
	});
});

// ---------------------------------------------------------------------------
// PokemonGymEnv tests
// ---------------------------------------------------------------------------

describe('PokemonGymEnv', () => {
	describe('reset', () => {
		it('should return a flat observation of length OBS_SIZE in flat obsMode', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4], obsMode: 'flat' });
			try {
				const obs = await env.reset();
				assert.equal(obs.length, OBS_SIZE);
				assert([...obs].every(v => !isNaN(v)), 'Observation from reset() must not contain NaN');
			} finally {
				env.destroy();
			}
		});

		it('should return a structured observation of length N_TOKENS * TOKEN_DIM by default', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				const obs = await env.reset();
				assert.equal(obs.length, N_TOKENS * TOKEN_DIM);
				assert([...obs].every(v => !isNaN(v)), 'Structured observation from reset() must not contain NaN');
			} finally {
				env.destroy();
			}
		});

		it('should return a v2 observation of length N_TOKENS * TOKEN_DIM_V2 in structured-v2 obsMode', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4], obsMode: 'structured-v2' });
			try {
				const obs = await env.reset();
				assert.equal(obs.length, N_TOKENS * TOKEN_DIM_V2);
				assert([...obs].every(v => !isNaN(v)), 'v2 observation from reset() must not contain NaN');
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
		it('should accept a legal move and return a flat observation of correct shape', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4], obsMode: 'flat' });
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

		it('should accept a legal move and return a structured observation of correct shape', async function () {
			this.timeout(20000);
			const env = new PokemonGymEnv({ seed: [1, 2, 3, 4] });
			try {
				await env.reset();
				const mask = env.validActions();
				const legalAction = mask.findIndex(v => v);
				if (legalAction === -1) return;
				const result = await env.step(legalAction);
				assert(!result.info.illegalMove, 'Legal action should not be flagged as illegal');
				assert.equal(result.obs.length, N_TOKENS * TOKEN_DIM);
				assert([...result.obs].every(v => !isNaN(v)), 'Structured observation from step() must not contain NaN');
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
		it('should keep structured obs shape stable across move requests, switch requests, and end-of-episode', async function () {
			this.timeout(60000);
			const env = new PokemonGymEnv({ seed: [7, 8, 9, 10] });
			try {
				let obs = await env.reset();
				assert.equal(obs.length, N_TOKENS * TOKEN_DIM);
				let done = false;
				let steps = 0;
				while (!done && steps < 300) {
					const mask = env.validActions();
					const action = mask.findIndex(v => v);
					const result = await env.step(action >= 0 ? action : 0);
					assert.equal(result.obs.length, N_TOKENS * TOKEN_DIM, `obs shape changed at step ${steps}`);
					assert([...result.obs].every(v => !isNaN(v)), `NaN in obs at step ${steps}`);
					obs = result.obs;
					done = result.done;
					steps++;
				}
				assert(done, 'Battle must terminate within 300 steps');
			} finally {
				env.destroy();
			}
		});

		it('should keep v2 obs shape stable and all v2 dims in range across a full battle', async function () {
			this.timeout(60000);
			const env = new PokemonGymEnv({ seed: [11, 12, 13, 14], obsMode: 'structured-v2' });
			try {
				let obs = await env.reset();
				assert.equal(obs.length, N_TOKENS * TOKEN_DIM_V2);
				let done = false;
				let steps = 0;
				while (!done && steps < 300) {
					const mask = env.validActions();
					const action = mask.findIndex(v => v);
					const result = await env.step(action >= 0 ? action : 0);
					assert.equal(result.obs.length, N_TOKENS * TOKEN_DIM_V2, `v2 obs shape changed at step ${steps}`);
					assert([...result.obs].every(v => !isNaN(v)), `NaN in v2 obs at step ${steps}`);
					// v2 extension dims: boosts in [-1, 1], flags/counter in [0, 1]
					for (let t = 0; t < N_TOKENS; t++) {
						for (let d = 65; d < TOKEN_DIM_V2; d++) {
							const v = result.obs[t * TOKEN_DIM_V2 + d];
							assert(v >= -1 && v <= 1, `v2 dim out of range at step ${steps}, token ${t}, dim ${d}: ${v}`);
						}
					}
					obs = result.obs;
					done = result.done;
					steps++;
				}
				assert(done, 'Battle must terminate within 300 steps');
			} finally {
				env.destroy();
			}
		});

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

// ---------------------------------------------------------------------------
// M5: opponent action labels (info.oppAction)
// ---------------------------------------------------------------------------

describe('PokemonGymEnv opponent action labels (M5)', () => {
	// Map an opponent choice string ("move 2" / "switch 3") into the opponent's
	// own 9-way action frame — the inverse of actionToChoice. Ground-truth
	// oracle for what the opponent actually decided.
	function choiceToAction(str) {
		if (!str) return -1;
		const s = str.trim().split(',')[0].trim();
		let m = s.match(/^move (\d+)/);
		if (m) return parseInt(m[1], 10) - 1;         // "move 1" -> slot 0
		m = s.match(/^switch (\d+)/);
		if (m) return parseInt(m[1], 10) - 1 + 3;     // "switch 2" -> bench 1 -> action 4
		return -1;                                    // pass / default / etc.
	}

	// Run a single-seat battle, wrapping the built-in opponent so we capture the
	// exact choice it submitted each turn. Returns per-step
	// { oppAction, oppTruth } where oppTruth is the opponent's real choice in its
	// own frame (null only for turn 1, whose choice is made during reset()).
	async function runWithOracle(env) {
		const opp = env._opponentPlayer;
		let recvCount = 0;
		const origRecv = opp.receiveRequest.bind(opp);
		opp.receiveRequest = (req) => {
			if (!(req && req.wait)) recvCount++;
			return origRecv(req);
		};
		const truthByCount = new Map();
		const origChoose = opp.choose.bind(opp);
		opp.choose = (choice) => {
			truthByCount.set(recvCount, choiceToAction(choice));
			return origChoose(choice);
		};

		const rows = [];
		let done = false;
		let steps = 0;
		while (!done && steps < 300) {
			const mask = env.validActions();
			const action = mask.findIndex(v => v);
			const countBefore = recvCount;
			const r = await env.step(action >= 0 ? action : 0);
			rows.push({
				oppAction: r.info.oppAction,
				oppTruth: truthByCount.has(countBefore) ? truthByCount.get(countBefore) : null,
			});
			done = r.done;
			steps++;
		}
		return rows;
	}

	describe('single-seat', () => {
		it('maps each resolved opponent move to its own request-slot (0-3), matching the actual choice', async function () {
			this.timeout(60000);
			const env = new PokemonGymEnv({ seed: [11, 22, 33, 44], obsMode: 'structured-v2', opponent: 'damagefirst' });
			try {
				await env.reset();
				const rows = await runWithOracle(env);
				let checkedMoves = 0;
				for (const row of rows) {
					assert(
						row.oppAction === -1 || (row.oppAction >= 0 && row.oppAction <= 8),
						`oppAction ${row.oppAction} out of range [-1, 8]`
					);
					if (row.oppAction >= 0 && row.oppTruth !== null) {
						assert.equal(
							row.oppAction, row.oppTruth,
							`label ${row.oppAction} != opponent's actual choice ${row.oppTruth}`
						);
						if (row.oppAction <= 3) checkedMoves++;
					}
				}
				assert(checkedMoves >= 10, `expected >= 10 verified move labels, got ${checkedMoves}`);
			} finally {
				env.destroy();
			}
		});

		it('produces a deterministic oppAction sequence containing real move labels', async function () {
			this.timeout(60000);
			async function seq() {
				const env = new PokemonGymEnv({ seed: [11, 22, 33, 44], obsMode: 'structured-v2', opponent: 'damagefirst' });
				try {
					await env.reset();
					const labels = [];
					let done = false;
					let steps = 0;
					while (!done && steps < 300) {
						const mask = env.validActions();
						const a = mask.findIndex(v => v);
						const r = await env.step(a >= 0 ? a : 0);
						labels.push(r.info.oppAction);
						done = r.done;
						steps++;
					}
					return labels;
				} finally {
					env.destroy();
				}
			}
			const a = await seq();
			const b = await seq();
			assert.deepEqual(a, b, 'oppAction sequence must be deterministic for a fixed seed');
			assert(a.some(x => x >= 0 && x <= 3), 'expected at least one move label (0-3)');
		});

		it('labels an engine-auto-completed locked move (recharge) instead of masking it', async function () {
			this.timeout(60000);
			// Seed chosen so the opponent lands a recharge move (Hyper Beam) whose
			// choice the engine auto-completes: choice.actions carries a moveid but
			// no moveSlot (the M4 gotcha). The label must recover the slot, not mask.
			const env = new PokemonGymEnv({ seed: [13, 27, 40, 20], obsMode: 'structured-v2', opponent: 'random' });
			try {
				await env.reset();
				const battle = env._battleStream.battle;
				let sawLocked = false;
				let done = false;
				let steps = 0;
				while (!done && steps < 300) {
					const action = battle.sides[1].choice && battle.sides[1].choice.actions && battle.sides[1].choice.actions[0];
					const isLocked = !!(action && action.choice === 'move' &&
						typeof action.moveSlot !== 'number' && action.moveid && action.moveid !== 'struggle');
					let expectedSlot = -1;
					if (isLocked) {
						const req = battle.sides[1].activeRequest;
						const moves = (req && req.active && req.active[0] && req.active[0].moves) || [];
						expectedSlot = moves.findIndex(m => m.id === action.moveid);
					}
					const mask = env.validActions();
					const a = mask.findIndex(v => v);
					const r = await env.step(a >= 0 ? a : 0);
					if (isLocked) {
						sawLocked = true;
						assert(r.info.oppAction >= 0, `locked move must be labeled, got ${r.info.oppAction}`);
						assert.equal(r.info.oppAction, expectedSlot, 'locked-move label must match its request slot');
					}
					done = r.done;
					steps++;
				}
				assert(sawLocked, 'expected the scripted seed to reach a locked (recharge) turn');
			} finally {
				env.destroy();
			}
		});
	});

	describe('dual-seat (self-play)', () => {
		it('labels both seats symmetrically with the other seat\'s simultaneous choice', async function () {
			this.timeout(60000);
			const env = new PokemonGymEnv({ seed: [7, 7, 7, 7], obsMode: 'structured-v2', opponent: 'self' });
			try {
				let st = await env.resetDual();
				let sawBothAct = false;
				let steps = 0;
				while (!st.done && steps < 300) {
					const a1 = st.p1.needsAction ? st.p1.mask.findIndex(v => v) : null;
					const a2 = st.p2.needsAction ? st.p2.mask.findIndex(v => v) : null;
					st = await env.stepDual(a1, a2);
					const bothActed = a1 !== null && a2 !== null;
					if (bothActed) {
						sawBothAct = true;
						assert.equal(st.info.oppAction, a2, 'p1 label must equal p2\'s simultaneous action');
						assert.equal(st.info.oppActionP2, a1, 'p2 label must equal p1\'s simultaneous action');
					}
					steps++;
				}
				assert(sawBothAct, 'expected at least one step where both seats acted');
			} finally {
				env.destroy();
			}
		});

		it('maps an opponent switch to the 4-8 bench-slot frame', async function () {
			this.timeout(60000);
			// Turn 1: both seats have a legal switch. Submit a switch for the p2
			// seat and confirm p1's label lands in the switch band and equals it.
			const env = new PokemonGymEnv({ seed: [7, 7, 7, 7], obsMode: 'structured-v2', opponent: 'self' });
			try {
				const st0 = await env.resetDual();
				const p1Move = st0.p1.mask.findIndex(v => v);
				const p2Switch = st0.p2.mask.findIndex((v, i) => v && i >= 4);
				assert(p2Switch >= 4, 'p2 should have a legal switch on turn 1');
				const st = await env.stepDual(p1Move, p2Switch);
				assert.equal(st.info.oppAction, p2Switch, 'p1 label must equal the p2 switch action (4-8)');
				assert(st.info.oppAction >= 4 && st.info.oppAction <= 8, 'switch label must be in 4-8');
			} finally {
				env.destroy();
			}
		});

		it('masks (-1) when only one seat has a simultaneous choice (force-switch)', async function () {
			this.timeout(60000);
			// Find the first step where exactly one seat needs an action (a
			// post-faint force-switch). With no simultaneous opponent choice the
			// label must be masked.
			const env = new PokemonGymEnv({ seed: [7, 7, 7, 7], obsMode: 'structured-v2', opponent: 'self' });
			try {
				let st = await env.resetDual();
				let tested = false;
				let steps = 0;
				while (!st.done && steps < 300) {
					const oneSeat = st.p1.needsAction !== st.p2.needsAction;
					const a1 = st.p1.needsAction ? st.p1.mask.findIndex(v => v) : null;
					const a2 = st.p2.needsAction ? st.p2.mask.findIndex(v => v) : null;
					st = await env.stepDual(a1, a2);
					if (oneSeat) {
						assert.equal(st.info.oppAction, -1, 'one-seat step must mask the p1 label');
						assert.equal(st.info.oppActionP2, -1, 'one-seat step must mask the p2 label');
						tested = true;
						break;
					}
					steps++;
				}
				assert(tested, 'expected to reach a one-seat force-switch step');
			} finally {
				env.destroy();
			}
		});
	});
});
