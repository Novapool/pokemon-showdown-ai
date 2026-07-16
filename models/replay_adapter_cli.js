'use strict';

/**
 * replay_adapter_cli.js — Batch-convert scraped replay logs (M5.5 Phase 1)
 * into per-decision BC training shards using the ReplayAdapter (M5.5 Phase 2).
 *
 * Reads data/replays/<format>/*.log.gz (+ manifest.csv for ratings), parses
 * each into two per-seat trajectories, and writes sharded gzipped JSONL under
 * data/replay_trajs/<format>/shard-NNNN.jsonl.gz. One JSON record per
 * decision:
 *   {b: battleId, f: format, r: rating|null, s: "p1"|"p2", w: 0|1,
 *    a: action 0-8, oa: oppAction -1..8, d: 0|1 (trajectory done),
 *    t: turn, o: base64(Float32Array LE, 924 floats — v2 obs)}
 *
 * Usage:
 *   node models/replay_adapter_cli.js --format gen1randombattle
 *   node models/replay_adapter_cli.js --format gen1ou --limit 500
 *   node models/replay_adapter_cli.js --format gen1ou --shard-size 2000
 *
 * Requires `./build` to have compiled dist/ first.
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { ReplayAdapter } = require('../dist/sim/tools/replay-adapter');

function parseArgs(argv) {
	const args = { format: null, limit: 0, shardSize: 2000, replaysDir: 'data/replays', outDir: 'data/replay_trajs' };
	for (let i = 2; i < argv.length; i++) {
		switch (argv[i]) {
		case '--format': args.format = argv[++i]; break;
		case '--limit': args.limit = parseInt(argv[++i], 10); break;
		case '--shard-size': args.shardSize = parseInt(argv[++i], 10); break;
		case '--replays-dir': args.replaysDir = argv[++i]; break;
		case '--out-dir': args.outDir = argv[++i]; break;
		default: throw new Error(`unknown arg: ${argv[i]}`);
		}
	}
	if (!args.format) throw new Error('--format is required (e.g. gen1randombattle, gen1ou)');
	return args;
}

function loadRatings(manifestPath) {
	const ratings = new Map();
	if (!fs.existsSync(manifestPath)) return ratings;
	const lines = fs.readFileSync(manifestPath, 'utf8').split('\n');
	for (const line of lines.slice(1)) {
		const cols = line.split(',');
		if (cols.length >= 3 && cols[0]) ratings.set(cols[0], cols[2] ? parseInt(cols[2], 10) : null);
	}
	return ratings;
}

async function main() {
	const args = parseArgs(process.argv);
	const inDir = path.join(args.replaysDir, args.format);
	const outDir = path.join(args.outDir, args.format);
	if (!fs.existsSync(inDir)) throw new Error(`no such directory: ${inDir} (run scripts/scrape_replays.py first)`);
	fs.mkdirSync(outDir, { recursive: true });

	const ratings = loadRatings(path.join(inDir, 'manifest.csv'));
	let files = fs.readdirSync(inDir).filter(f => f.endsWith('.log.gz')).sort();
	if (args.limit) files = files.slice(0, args.limit);

	const adapter = new ReplayAdapter();
	let shardIdx = 0;
	let shardBattles = 0;
	let gzip = null;
	const stats = { battles: 0, unusable: 0, decisions: 0, skipped: 0, errors: 0 };

	// Streamed gzip shards: battles are written as they parse, never buffered
	// whole-shard in memory (a 2000-battle shard is ~1GB of JSON pre-gzip).
	const openShard = () => {
		const file = path.join(outDir, `shard-${String(shardIdx).padStart(4, '0')}.jsonl.gz`);
		gzip = zlib.createGzip();
		gzip.pipe(fs.createWriteStream(file));
		return gzip;
	};
	const closeShard = () => {
		if (!gzip) return Promise.resolve();
		const g = gzip;
		gzip = null;
		shardIdx++;
		shardBattles = 0;
		return new Promise(resolve => {
			g.on('close', resolve);
			g.end();
		});
	};
	const writeLine = line => {
		if (!gzip) openShard();
		return gzip.write(line + '\n');
	};

	for (const file of files) {
		const battleId = file.slice(0, -'.log.gz'.length);
		let parsed;
		try {
			const log = zlib.gunzipSync(fs.readFileSync(path.join(inDir, file))).toString('utf8');
			parsed = adapter.parse(log);
		} catch (err) {
			stats.errors++;
			if (stats.errors <= 5) console.error(`[replay-adapter] ${battleId}: ${err.message}`);
			continue;
		}
		if (!parsed) {
			stats.unusable++;
			continue;
		}
		const rating = ratings.get(battleId) ?? null;
		let needDrain = false;
		for (const traj of [parsed.p1, parsed.p2]) {
			for (const dec of traj.decisions) {
				needDrain = !writeLine(JSON.stringify({
					b: battleId, f: args.format, r: rating, s: traj.seat, w: traj.win,
					a: dec.action, oa: dec.oppAction, d: dec.done ? 1 : 0, t: dec.turn,
					o: Buffer.from(dec.obs.buffer, dec.obs.byteOffset, dec.obs.byteLength).toString('base64'),
				})) || needDrain;
				stats.decisions++;
			}
		}
		if (needDrain) await new Promise(resolve => gzip.once('drain', resolve));
		stats.skipped += parsed.skipped;
		stats.battles++;
		shardBattles++;
		if (shardBattles >= args.shardSize) await closeShard();
		if (stats.battles % 2000 === 0) {
			console.error(`[replay-adapter] ${stats.battles}/${files.length} battles, ${stats.decisions} decisions...`);
		}
	}
	await closeShard();

	console.log(JSON.stringify({
		format: args.format,
		files: files.length,
		battles: stats.battles,
		unusable: stats.unusable,
		parse_errors: stats.errors,
		decisions: stats.decisions,
		skipped_unmappable: stats.skipped,
		label_coverage: stats.decisions / Math.max(1, stats.decisions + stats.skipped),
		shards: shardIdx,
	}, null, 2));
}

main().catch(err => {
	console.error(err);
	process.exit(1);
});
