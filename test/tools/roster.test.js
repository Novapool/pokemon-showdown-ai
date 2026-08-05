'use strict';

/**
 * Tests for the M12 fixed roster (sim/tools/roster.ts) and its two consumers:
 * the gym (same team to both seats) and the MCTS determinizer (a known bench
 * is filled from the roster instead of sampled).
 *
 * Run: npx mocha --no-config --spec test/tools/roster.test.js --timeout 60000
 */

const assert = require('../assert');
const {
	loadRoster, rosterSets, rosterForFormat, FIXED_ROSTER_FORMAT, DEFAULT_ROSTER_FILE,
} = require('../../dist/sim/tools/roster');
const { PokemonGymEnv } = require('../../dist/sim/tools/pokemon-gym');
const { BattleSim } = require('../../dist/sim/tools/battle-sim');
const { Teams } = require('../../dist/sim');
const { TeamValidator } = require('../../dist/sim/team-validator');

/** The pre-registered roster — see docs/BATTLE-FORMATS.md. Changing this is a milestone decision. */
const EXPECTED = ['Alakazam', 'Chansey', 'Exeggutor', 'Snorlax', 'Starmie', 'Tauros'];

function speciesOf(side) {
	return side.pokemon.map(p => p.species.name).sort();
}

describe('fixed roster (M12)', () => {
	it('loads the pre-registered six', () => {
		const sets = rosterSets();
		assert.equal(sets.length, 6);
		assert.deepEqual(sets.map(s => s.species).sort(), EXPECTED);
	});

	it('gives every Pokemon exactly four moves', () => {
		for (const set of rosterSets()) {
			assert.equal(set.moves.length, 4, `${set.species} has ${set.moves.length} moves`);
		}
	});

	it('is legal in gen1ou', () => {
		const problems = new TeamValidator(FIXED_ROSTER_FORMAT).validateTeam(rosterSets());
		assert.equal(problems, null, `validator complained: ${problems && problems.join('; ')}`);
	});

	it('round-trips species and moves through the packed format', () => {
		// NOTE: not a byte round-trip — Teams.unpack backfills each set's
		// `ability` from the dex, so pack(unpack(x)) reintroduces ability
		// names the file omits. Gen 1 has no abilities and the engine ignores
		// them, so only species/moves need to survive.
		const before = rosterSets();
		const after = Teams.unpack(Teams.pack(before));
		assert.deepEqual(
			after.map(s => [s.species, s.moves.join(',')]),
			before.map(s => [s.species, s.moves.join(',')])
		);
	});

	it('resolves relative to the repo, not process.cwd()', () => {
		const fromRepo = loadRoster();
		const cwd = process.cwd();
		try {
			process.chdir('/');
			assert.equal(loadRoster(DEFAULT_ROSTER_FILE), fromRepo);
		} finally {
			process.chdir(cwd);
		}
	});

	it('throws on a missing roster file rather than silently generating a team', () => {
		// A silent fallback would mean training on random teams while every doc
		// claims a fixed one — the confound this project has been burned by.
		assert.throws(() => loadRoster('config/rosters/does-not-exist.txt'), /not found/);
	});

	describe('rosterForFormat', () => {
		it('returns the roster for the fixed-roster format', () => {
			assert.equal(rosterForFormat(FIXED_ROSTER_FORMAT).length, 6);
		});

		it('returns undefined for random-team formats', () => {
			assert.equal(rosterForFormat('gen1randombattle'), undefined);
		});

		it('honours an explicit null (force generator sampling)', () => {
			assert.equal(rosterForFormat(FIXED_ROSTER_FORMAT, null), undefined);
		});
	});
});

describe('fixed roster — gym integration', () => {
	it('gives BOTH seats the same team', async () => {
		const env = new PokemonGymEnv({
			format: FIXED_ROSTER_FORMAT, team: loadRoster(),
			opponent: 'random', obsMode: 'structured-v3',
		});
		try {
			await env.reset();
			const battle = env.snapshot && env._battleStream ? env._battleStream.battle : null;
			assert(battle, 'expected a live battle');
			assert.deepEqual(speciesOf(battle.sides[0]), EXPECTED);
			assert.deepEqual(speciesOf(battle.sides[1]), EXPECTED);
		} finally {
			env.destroy();
		}
	});

	it('leaves random-team formats generating their own teams', async () => {
		const env = new PokemonGymEnv({ opponent: 'random', obsMode: 'structured-v3' });
		try {
			await env.reset();
			const battle = env._battleStream.battle;
			// Overwhelmingly unlikely to match the fixed six by chance.
			assert.notDeepEqual(speciesOf(battle.sides[0]), EXPECTED);
		} finally {
			env.destroy();
		}
	});
});

describe('fixed roster — MCTS determinizer', () => {
	/** Play a few turns so some of the opponent's team is revealed and some isn't. */
	async function snapshotMidBattle() {
		const env = new PokemonGymEnv({
			format: FIXED_ROSTER_FORMAT, team: loadRoster(),
			opponent: 'random', obsMode: 'structured-v3',
		});
		await env.reset();
		for (let i = 0; i < 6; i++) {
			const mask = env.validActions();
			const action = mask.findIndex(Boolean);
			if (action < 0) break;
			const result = await env.step(action);
			if (result.done) break;
		}
		const snap = env.snapshot();
		env.destroy();
		return snap;
	}

	it('fills the hidden bench from the roster, not the random generator', async () => {
		const snap = await snapshotMidBattle();
		const sim = BattleSim.fromSnapshot(snap, { determinize: true, perspective: 'p1', seed: 7 });
		assert.deepEqual(speciesOf(sim._battle.sides[1]), EXPECTED);
	});

	it('still samples when roster is explicitly disabled', async () => {
		const snap = await snapshotMidBattle();
		const sim = BattleSim.fromSnapshot(
			snap, { determinize: true, perspective: 'p1', seed: 7, roster: null }
		);
		// Without the roster this falls through to the gen1 RANDOM generator,
		// which hands search Pokemon the opponent cannot possibly have.
		assert.notDeepEqual(speciesOf(sim._battle.sides[1]), EXPECTED);
	});

	it('does not re-derive a roster for a fork of a roster-disabled sim', async () => {
		const snap = await snapshotMidBattle();
		const sim = BattleSim.fromSnapshot(
			snap, { determinize: true, perspective: 'p1', seed: 7, roster: null }
		);
		const child = sim.fork();
		assert.deepEqual(speciesOf(child._battle.sides[1]), speciesOf(sim._battle.sides[1]));
	});
});
