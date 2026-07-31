'use strict';

/**
 * Tests for the M9 Phase 1 ladder result log (tools/ladder-bot/ladder-bot.js).
 *
 * The per-battle CSV is the only measurement instrument M9 Phase 3 has — the
 * account JSON's Elo/GXE are account-level cumulative statistics and cannot be
 * used per-run (docs/LADDER-MEASUREMENT.md). So the load-bearing checks are:
 *  1. Every row carries the arm label (run_id), account, checkpoint and the
 *     pre-battle opponent/own Elo, so a paired A/B can be reconstructed and
 *     opponent-strength asymmetry between arms is visible.
 *  2. Rows stay well-formed when fields are absent (unrated games have no
 *     ratings) or hostile (usernames are user-controlled and may contain
 *     commas).
 *  3. A pre-M9 log is retired rather than appended to with a wider row, which
 *     would silently corrupt the M6-M8 record.
 *
 * Run: npx mocha --no-config test/tools/ladder-results.test.js
 */

const assert = require('../assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { BattleRoom, CSV_HEADER, csvField } = require('../../tools/ladder-bot/ladder-bot');

const COLUMNS = CSV_HEADER.split(',');
const PRE_M9_HEADER = 'timestamp,room,opponent,rated,result,decisions,max_latency_ms';

function makeRoom(bot, lines) {
	const room = new BattleRoom('battle-gen1randombattle-999', bot);
	for (const line of lines) room.receiveLine(line);
	return room;
}

function bot(overrides = {}) {
	return {
		username: 'novapool',
		verbose: false,
		args: {
			runId: 'm9p3-control',
			checkpoint: 'models/ppo/checkpoints/v3/ppo_step_5000002_final.pt',
			saveDir: null,
			...(overrides.args || {}),
		},
		obsV2: false, obsV3: true, obsV3Ext: false,
		...overrides,
	};
}

function readCsv(dir, name = 'ladder_results.csv') {
	return fs.readFileSync(path.join(dir, name), 'utf8').trimEnd().split('\n');
}

function cell(row, column) {
	return row.split(',')[COLUMNS.indexOf(column)];
}

describe('ladder-bot result log', () => {
	let dir;
	beforeEach(() => {
		dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ladder-results-'));
	});
	afterEach(() => {
		fs.rmSync(dir, { recursive: true, force: true });
	});

	it('records arm, account, checkpoint and both pre-battle Elos', () => {
		const room = makeRoom(bot(), [
			'|player|p1|novapool|1|1084',
			'|player|p2|Rival|265|1211',
			'|rated',
		]);
		room.result = 'win';
		room.decisions = 31;
		room.maxLatencyMs = 98;
		room.save(dir);

		const [header, row] = readCsv(dir);
		assert.equal(header, CSV_HEADER);
		assert.equal(row.split(',').length, COLUMNS.length);
		assert.equal(cell(row, 'run_id'), 'm9p3-control');
		assert.equal(cell(row, 'account'), 'novapool');
		assert.equal(cell(row, 'checkpoint'), 'ppo_step_5000002_final.pt');
		assert.equal(cell(row, 'opponent'), 'Rival');
		assert.equal(cell(row, 'opp_rating'), '1211');
		assert.equal(cell(row, 'own_rating'), '1084');
		assert.equal(cell(row, 'rated'), '1');
		assert.equal(cell(row, 'result'), 'win');
		assert.equal(cell(row, 'decisions'), '31');
	});

	it('leaves ratings empty on unrated games without shifting columns', () => {
		const room = makeRoom(bot({ args: { runId: null, checkpoint: null } }), [
			'|player|p1|novapool|1',
			'|player|p2|Rival|265',
		]);
		room.result = 'tie';
		room.save(dir);

		const row = readCsv(dir)[1];
		assert.equal(row.split(',').length, COLUMNS.length);
		assert.equal(cell(row, 'opp_rating'), '');
		assert.equal(cell(row, 'own_rating'), '');
		assert.equal(cell(row, 'run_id'), '');
		assert.equal(cell(row, 'rated'), '0');
		assert.equal(cell(row, 'result'), 'tie');
	});

	it('strips separators out of user-controlled usernames', () => {
		const room = makeRoom(bot(), [
			'|player|p1|novapool|1|1084',
			'|player|p2|a,b"c|265|1211',
			'|rated',
		]);
		room.result = 'loss';
		room.save(dir);

		const row = readCsv(dir)[1];
		assert.equal(row.split(',').length, COLUMNS.length);
		assert.equal(cell(row, 'opponent'), 'abc');
		assert.equal(csvField('x,y\nz"'), 'xyz');
	});

	it('appends to an existing M9 log without re-migrating it', () => {
		for (const runId of ['m9p3-control', 'm9p3-candidate']) {
			const room = makeRoom(bot({ args: { runId } }), [
				'|player|p1|novapool|1|1084',
				'|player|p2|Rival|265|1211',
				'|rated',
			]);
			room.result = 'win';
			room.save(dir);
		}
		const rows = readCsv(dir);
		assert.equal(rows.length, 3);
		assert.equal(cell(rows[1], 'run_id'), 'm9p3-control');
		assert.equal(cell(rows[2], 'run_id'), 'm9p3-candidate');
		assert.equal(fs.existsSync(path.join(dir, 'ladder_results.pre-m9.csv')), false);
	});

	it('retires a pre-M9 log byte-for-byte instead of appending wider rows', () => {
		const legacy = `${PRE_M9_HEADER}\n` +
			'2026-07-31T00:00:00.000Z,battle-gen1randombattle-1,Rival,1,win,20,100\n';
		fs.writeFileSync(path.join(dir, 'ladder_results.csv'), legacy);

		const room = makeRoom(bot(), [
			'|player|p1|novapool|1|1084',
			'|player|p2|Rival|265|1211',
			'|rated',
		]);
		room.result = 'win';
		room.save(dir);

		assert.equal(
			fs.readFileSync(path.join(dir, 'ladder_results.pre-m9.csv'), 'utf8'),
			legacy,
			'the M6-M8 record must survive the migration unchanged');
		const rows = readCsv(dir);
		assert.equal(rows.length, 2);
		assert.equal(rows[0], CSV_HEADER);
		assert.equal(cell(rows[1], 'run_id'), 'm9p3-control');
	});
});
