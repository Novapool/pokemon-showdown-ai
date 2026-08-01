'use strict';

/**
 * ladder-bot.js — Pokemon Showdown ladder client for the trained agent (M6 P1).
 *
 * Connects to a Showdown server over websocket (Node's built-in WebSocket),
 * logs in, queues on the ladder (or challenges/accepts), and plays battles by
 * reusing the exact live-gym code path:
 *   |request| JSON  +  public log lines -> ObservationTrackers
 *       -> extractFeaturesStructured (v1/v2/v3/v3-extended, per checkpoint obs_size) -> obs
 *   validActionsForRequest(request)        -> mask
 *   models/infer_server.py (PPO checkpoint, stdio)  -> action
 *   actionToChoice(action)                 -> /choose string
 *
 * Every finished battle's log is saved to data/replays/self_ladder/ (same
 * layout as the M5.5 scraper) and appended to ladder_results.csv.
 *
 * Usage:
 *   # Local server smoke (start one first: node pokemon-showdown start --no-security)
 *   node tools/ladder-bot/ladder-bot.js --server ws://localhost:8000/showdown/websocket \
 *     --checkpoint models/ppo/checkpoints/opp/ppo_step_5000001_final.pt \
 *     --name testbot123 --battles 2 --challenge someuser
 *
 *   # Official ladder (registered account via PS_USERNAME/PS_PASSWORD env)
 *   PS_USERNAME=... PS_PASSWORD=... node tools/ladder-bot/ladder-bot.js \
 *     --checkpoint models/ppo/checkpoints/opp/ppo_step_5000001_final.pt --battles 10
 *
 *   # The policy plays its argmax by default (--sample restores sampling, the
 *   # pre-2026-08-01 behaviour and the control arm of a greedy A/B). Note that
 *   # under --mcts the search already argmaxes, so the decision rule only
 *   # reaches the raw-policy fallbacks (force switches / locked states).
 *
 *   # Resumable run: --run-id persists progress (finished/wins) to
 *   # <save-dir>/run_<id>.json after every battle. If the process dies for any
 *   # reason, re-running the SAME command continues at the next battle —
 *   # --battles stays the absolute target, no manual arithmetic.
 *
 * Requires `./build` (dist/) and Node >= 22 (built-in WebSocket).
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const { spawn } = require('child_process');
const readline = require('readline');

const { ObservationTrackers } = require('../../dist/sim/tools/pokemon-gym');
const { actionToChoice, validActionsForRequest } = require('../../dist/sim/tools/pokemon-gym');
const { extractFeaturesStructured } = require('../../dist/sim/tools/feature-extractor');

const DEFAULT_SERVER = 'wss://sim3.psim.us/showdown/websocket';
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30000;
const RECONNECT_MAX_ATTEMPTS = 15;
const LOGIN_API = 'https://play.pokemonshowdown.com/api/login';
const ASSERTION_API = 'https://play.pokemonshowdown.com/action.php';

function parseArgs(argv) {
	const args = {
		server: DEFAULT_SERVER,
		format: 'gen1randombattle',
		checkpoint: null,
		battles: 1,
		name: process.env.PS_USERNAME || null,
		password: process.env.PS_PASSWORD || null,
		challenge: null,   // send a challenge to this user instead of laddering
		acceptFrom: null,  // accept challenges from this user instead of laddering
		runId: null,       // named resumable run: progress persists in saveDir, so
		                   // re-running the same command continues where it stopped
		loginFile: null,   // "username: ...\npassword: ..." file (keeps creds off argv/env)
		saveDir: 'data/replays/self_ladder',
		device: 'cpu',
		verbose: false,
		// Play the policy's argmax, not a sample from it. Default ON since
		// 2026-08-01: greedy beat sampling by +7.8pp vs Random [+6.1, +9.5] and
		// +5.0pp vs DamageFirst [+3.1, +6.9] at n=5,000/arm. Every ladder result
		// before that date was scored while sampling. --sample restores the old
		// behaviour (it is the control arm of any greedy A/B).
		greedy: true,
		mcts: false,        // M6 P2: search clean move requests via act_tracked
		sims: 100,
		determinizations: 1,
		cPuct: 0.5,
	};
	for (let i = 2; i < argv.length; i++) {
		switch (argv[i]) {
		case '--server': args.server = argv[++i]; break;
		case '--format': args.format = argv[++i]; break;
		case '--checkpoint': args.checkpoint = argv[++i]; break;
		case '--battles': args.battles = parseInt(argv[++i], 10); break;
		case '--name': args.name = argv[++i]; break;
		case '--password': args.password = argv[++i]; break;
		case '--challenge': args.challenge = argv[++i]; break;
		case '--accept-from': args.acceptFrom = argv[++i]; break;
		case '--login-file': args.loginFile = argv[++i]; break;
		case '--run-id': args.runId = argv[++i]; break;
		case '--save-dir': args.saveDir = argv[++i]; break;
		case '--device': args.device = argv[++i]; break;
		case '--verbose': args.verbose = true; break;
		case '--greedy': args.greedy = true; break;   // default; kept so old commands still run
		case '--sample': args.greedy = false; break;
		case '--mcts': args.mcts = true; break;
		case '--sims': args.sims = parseInt(argv[++i], 10); break;
		case '--determinizations': args.determinizations = parseInt(argv[++i], 10); break;
		case '--c-puct': args.cPuct = parseFloat(argv[++i]); break;
		default: throw new Error(`unknown arg: ${argv[i]}`);
		}
	}
	if (args.loginFile) {
		for (const line of fs.readFileSync(args.loginFile, 'utf8').split('\n')) {
			const m = /^(username|password):\s*(.+?)\s*$/.exec(line);
			if (m) args[m[1] === 'username' ? 'name' : 'password'] = m[2];
		}
	}
	if (!args.checkpoint) throw new Error('--checkpoint is required');
	if (!args.name) throw new Error('--name (or PS_USERNAME env, or --login-file) is required');
	return args;
}

function toID(text) {
	return ('' + text).toLowerCase().replace(/[^a-z0-9]+/g, '');
}

// ---------------------------------------------------------------------------
// Inference subprocess (models/infer_server.py)
// ---------------------------------------------------------------------------

class InferServer {
	constructor(args) {
		const argv = [
			path.join(__dirname, '../../models/infer_server.py'),
			'--checkpoint', args.checkpoint, '--device', args.device,
		];
		if (args.greedy) argv.push('--greedy');
		if (args.mcts) {
			argv.push('--mcts', '--sims', String(args.sims),
				'--determinizations', String(args.determinizations),
				'--c-puct', String(args.cPuct));
		}
		this.proc = spawn('python3', argv, { stdio: ['pipe', 'pipe', 'inherit'] });
		this.rl = readline.createInterface({ input: this.proc.stdout });
		this.queue = [];
		this.rl.on('line', line => {
			const resolve = this.queue.shift();
			if (resolve) resolve(JSON.parse(line));
		});
	}

	_send(msg) {
		return new Promise(resolve => {
			this.queue.push(resolve);
			this.proc.stdin.write(JSON.stringify(msg) + '\n');
		});
	}

	async ping() {
		const resp = await this._send({ cmd: 'ping' });
		if (!resp.ok) throw new Error(`infer server ping failed: ${JSON.stringify(resp)}`);
		return resp;
	}

	async act(obs, mask) {
		const resp = await this._send({ cmd: 'act', obs: Array.from(obs), mask });
		if (resp.error) throw new Error(`infer server: ${resp.error}`);
		return resp.action;
	}

	/** M6 P2: determinized UCT over a reconstruction of the remote battle. */
	async actTracked(payload) {
		const resp = await this._send({ cmd: 'act_tracked', ...payload });
		if (resp.error) throw new Error(`infer server: ${resp.error}`);
		return resp; // {action, sims, fallback?}
	}

	close() {
		try {
			this.proc.stdin.write(JSON.stringify({ cmd: 'close' }) + '\n');
		} catch {}
		setTimeout(() => this.proc.kill(), 500).unref();
	}
}

// ---------------------------------------------------------------------------
// One battle room
// ---------------------------------------------------------------------------

// Per-battle result schema. `run_id` labels the arm of an A/B (M9 Phase 3
// convention: always pass --run-id), `opp_rating`/`own_rating` are the
// pre-battle ladder Elos. Analysed by scripts/ladder_analysis.py — never read a
// per-run win rate off the account JSON (see docs/LADDER-MEASUREMENT.md).
const CSV_HEADER = 'timestamp,run_id,account,checkpoint,room,opponent,' +
	'opp_rating,own_rating,rated,result,decisions,max_latency_ms';

/** Keep a value inside one unquoted CSV cell (usernames are user-controlled). */
function csvField(value) {
	return String(value).replace(/[,\r\n"]/g, '');
}

class BattleRoom {
	constructor(roomid, bot) {
		this.roomid = roomid;
		this.bot = bot;
		this.trackers = new ObservationTrackers();
		this.seat = null;          // 'p1' | 'p2'
		this.request = null;       // latest actionable request (unanswered)
		this.lines = [];           // full log for saving
		this.rated = false;
		this.opponent = '';
		// Pre-battle ladder ratings from the |player| lines. Only rated battles
		// carry them, and they are the ONLY per-battle record of who we faced —
		// M9 Phase 3 needs per-arm mean opponent Elo to show both arms met the
		// same pool. See docs/EVALUATION-METHODOLOGY.md.
		this.oppRating = null;
		this.ownRating = null;
		this.done = false;
		this.result = null;        // 'win' | 'loss' | 'tie'
		this.decisions = 0;
		this.maxLatencyMs = 0;
		this.turn = 0;             // latest |turn| number (BattleSim.fromTracked input)
		this.searched = 0;         // decisions answered by MCTS (not fallback)
	}

	/** Process one protocol line; returns true if state changed. */
	receiveLine(line) {
		this.lines.push(line);
		const parts = line.split('|');
		const type = parts[1] ?? '';

		if (type === 'player' && parts[2] && parts[3]) {
			// |player|pN|username|avatar|rating — rating present on rated games only.
			const rating = parseInt(parts[5] ?? '');
			if (toID(parts[3]) === toID(this.bot.username)) {
				this.seat = parts[2];
				if (Number.isFinite(rating)) this.ownRating = rating;
			} else {
				this.opponent = parts[3];
				if (Number.isFinite(rating)) this.oppRating = rating;
			}
		} else if (type === 'rated') {
			this.rated = true;
		} else if (type === 'turn') {
			this.turn = parseInt(parts[2], 10) || this.turn;
			this.trackers.processLine(line);
		} else if (type === 'request' && parts[2]) {
			try {
				const request = JSON.parse(parts.slice(2).join('|'));
				if (request && !request.wait) this.request = request;
			} catch (err) {
				console.error(`[${this.roomid}] bad request JSON: ${err.message}`);
			}
			return true;
		} else if (type === 'win') {
			this.done = true;
			const winnerIsMe = toID(parts[2] ?? '') === toID(this.bot.username);
			this.result = winnerIsMe ? 'win' : 'loss';
		} else if (type === 'tie') {
			this.done = true;
			this.result = 'tie';
		} else if (type === 'error') {
			// Invalid choice — fall back to the server-side default so the
			// battle never stalls; log it loudly (should not happen).
			console.error(`[${this.roomid}] ${line}`);
			// "too late" = a choice is already locked in (e.g. a stale
				// request replayed on rejoin); defaulting again would error in
				// a loop, so leave it alone.
				if (line.includes('[Invalid choice]') && !line.includes('too late')) {
				this.bot.send(`${this.roomid}|/choose default`);
			}
		} else {
			this.trackers.processLine(line);
		}
		return true;
	}

	/** Answer the pending request, if any (called at message-frame end). */
	async maybeAct() {
		if (this.done || !this.request || !this.seat) return;
		const request = this.request;
		this.request = null;
		const started = Date.now();

		const mask = validActionsForRequest(request);
		if (!mask.some(m => m)) return; // nothing legal (wait-like) — server decides

		// v2 and v3(+extended) all need the appended volatile dims; v3 additionally
		// needs the Sleep Clause flag (its dim 85) and v3-extended the speed-ratio
		// selector, all read from the public-log trackers exactly as the live gym does.
		const isV3 = this.bot.obsV3 || this.bot.obsV3Ext;
		const obs = extractFeaturesStructured(
			request,
			this.trackers.opponentInfoFor(this.seat),
			(this.bot.obsV2 || isV3) ? this.trackers.volatilesFor(this.seat) : null,
			isV3 ? {
				sleepClause: this.trackers.sleepClauseFlagFor(this.seat) === 1,
				extended: this.bot.obsV3Ext,
			} : undefined,
		);
		let action;
		try {
			// M6 P2: search clean move requests; raw policy answers everything
			// else (force switches / locked states — fromTracked's contract).
			if (this.bot.mctsReady && request.active && !request.forceSwitch && !request.wait) {
				const resp = await this.bot.infer.actTracked({
					obs: Array.from(obs), mask,
					formatid: this.bot.args.format,
					seat: this.seat,
					request,
					trackers: this.trackers.snapshot(),
					turn: this.turn,
					obs_mode: this.bot.obsV3Ext ? 'structured-v3-extended' : this.bot.obsV3 ? 'structured-v3' : this.bot.obsV2 ? 'structured-v2' : 'structured',
				});
				action = resp.action;
				if (resp.sims > 0) this.searched++;
				else if (this.bot.verbose) {
					console.log(`[${this.roomid}] search fell back to raw policy: ${resp.fallback}`);
				}
			} else {
				action = await this.bot.infer.act(obs, mask);
			}
		} catch (err) {
			console.error(`[${this.roomid}] inference failed (${err.message}); choosing default`);
			this.bot.send(`${this.roomid}|/choose default|${request.rqid ?? ''}`);
			return;
		}
		if (!mask[action]) {
			console.error(`[${this.roomid}] illegal action ${action} from model; choosing default`);
			this.bot.send(`${this.roomid}|/choose default|${request.rqid ?? ''}`);
			return;
		}
		const choice = actionToChoice(action);
		const rqid = request.rqid !== undefined ? `|${request.rqid}` : '';
		this.bot.send(`${this.roomid}|/choose ${choice}${rqid}`);
		this.decisions++;
		const ms = Date.now() - started;
		if (ms > this.maxLatencyMs) this.maxLatencyMs = ms;
		if (this.bot.verbose) console.log(`[${this.roomid}] turn choice: ${choice} (${ms}ms)`);
	}

	save(saveDir) {
		fs.mkdirSync(saveDir, { recursive: true });
		// Room ids are only unique per server session (local servers restart at
		// 1), and bot-vs-bot tests have two instances in the same room — key
		// the file by room + our username + start time.
		const file = path.join(
			saveDir,
			`${this.roomid.replace(/^battle-/, '')}-${toID(this.bot.username)}-${Date.now()}.log.gz`);
		fs.writeFileSync(file, zlib.gzipSync(this.lines.join('\n') + '\n'));
		const csv = path.join(saveDir, 'ladder_results.csv');
		if (!fs.existsSync(csv)) {
			fs.writeFileSync(csv, CSV_HEADER + '\n');
		} else {
			// M9 added run_id/account/checkpoint/opp_rating/own_rating. Appending
			// the wider row to a pre-M9 file would silently corrupt it, so retire
			// the old log under a dated name and start a clean one. The old file
			// stays readable by scripts/ladder_analysis.py.
			const existing = fs.readFileSync(csv, 'utf8').split('\n', 1)[0];
			if (existing !== CSV_HEADER) {
				const retired = path.join(saveDir, 'ladder_results.pre-m9.csv');
				if (!fs.existsSync(retired)) fs.renameSync(csv, retired);
				else fs.appendFileSync(retired, fs.readFileSync(csv, 'utf8').split('\n').slice(1).join('\n'));
				fs.writeFileSync(csv, CSV_HEADER + '\n');
				console.log(`ladder_results.csv used the pre-M9 schema; retired it to ${retired}`);
			}
		}
		fs.appendFileSync(csv, [
			new Date().toISOString(),
			csvField(this.bot.args.runId ?? ''),
			csvField(this.bot.username),
			csvField(this.bot.args.checkpoint ? path.basename(this.bot.args.checkpoint) : ''),
			this.roomid, csvField(this.opponent),
			this.oppRating ?? '', this.ownRating ?? '',
			this.rated ? 1 : 0, this.result, this.decisions, this.maxLatencyMs,
		].join(',') + '\n');
	}
}

// ---------------------------------------------------------------------------
// The bot
// ---------------------------------------------------------------------------

class LadderBot {
	constructor(args) {
		this.args = args;
		this.username = args.name;
		this.verbose = args.verbose;
		this.rooms = new Map();
		this.finishedRooms = new Set();
		this.finished = 0;
		this.wins = 0;
		this.searching = false;
		this.loggedIn = false;
		this.infer = new InferServer(args);
		this.obsV2 = null;     // set after ping (schema inferred from checkpoint obs_size)
		this.obsV3 = null;     // set after ping (M7: (12,86) schema-v3 checkpoint)
		this.obsV3Ext = null;  // set after ping (M8: (12,87) schema-v3-extended checkpoint)
		this.mctsReady = false; // set after ping (--mcts requested AND server confirms)
		this.shuttingDown = false;      // set when the run is complete (clean close)
		this.reconnectAttempts = 0;     // consecutive failed (re)connects; reset on login
		this.pendingRejoin = new Set(); // battles interrupted by a disconnect
		this.joinRequested = new Set(); // /join sent, room frame not yet seen (dedupe)
		this.runFile = args.runId ?
			path.join(args.saveDir, `run_${toID(args.runId)}.json`) : null;
		this._loadRunState();
	}

	/** --run-id: restore finished/wins so a restarted run continues in place. */
	_loadRunState() {
		if (!this.runFile || !fs.existsSync(this.runFile)) return;
		try {
			const state = JSON.parse(fs.readFileSync(this.runFile, 'utf8'));
			this.finished = state.finished || 0;
			this.wins = state.wins || 0;
			console.log(`resuming run '${this.args.runId}': ` +
				`${this.wins}/${this.finished} won so far, target ${this.args.battles}`);
		} catch (err) {
			throw new Error(`corrupt run file ${this.runFile}: ${err.message}`);
		}
	}

	_saveRunState() {
		if (!this.runFile) return;
		fs.mkdirSync(path.dirname(this.runFile), { recursive: true });
		fs.writeFileSync(this.runFile, JSON.stringify({
			finished: this.finished, wins: this.wins,
			target: this.args.battles, updated: new Date().toISOString(),
		}) + '\n');
	}

	send(message) {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			console.error(`send skipped (socket not open): ${message.split('\n')[0]}`);
			return;
		}
		if (this.verbose && !message.includes('/choose')) console.log(`>> ${message}`);
		this.ws.send(message);
	}

	async start() {
		if (this.finished >= this.args.battles) {
			console.log(`run '${this.args.runId}' already complete: ` +
				`${this.wins}/${this.finished} won — raise --battles to extend it`);
			this.infer.close();
			return;
		}
		const pong = await this.infer.ping();
		const obsSize = pong.obs_size;
		this.obsV3Ext = obsSize === 12 * 87;
		this.obsV3 = obsSize === 12 * 86;
		this.obsV2 = obsSize === 12 * 77;
		if (!this.obsV3Ext && !this.obsV3 && !this.obsV2 && obsSize !== 12 * 65) {
			throw new Error(`unsupported checkpoint obs_size ${obsSize} (need 780, 924, 1032, or 1044)`);
		}
		const schema = this.obsV3Ext ? 'v3-extended' : this.obsV3 ? 'v3' : this.obsV2 ? 'v2' : 'v1';
		this.mctsReady = this.args.mcts && !!pong.mcts;
		console.log(`inference ready: obs_size=${obsSize} (${schema} schema)` +
			`, policy=${this.args.greedy ? 'greedy' : 'sampled'}` +
			(this.mctsReady ? `, MCTS on (sims=${this.args.sims}, det=${this.args.determinizations}, ` +
				`c_puct=${this.args.cPuct})` : ''));

		this._connect();
	}

	_connect() {
		this.ws = new WebSocket(this.args.server);
		this.ws.addEventListener('open', () => console.log(`connected: ${this.args.server}`));
		this.ws.addEventListener('message', event => {
			this.receive(String(event.data)).catch(err => console.error('receive error:', err));
		});
		this.ws.addEventListener('close', () => {
			if (this.shuttingDown) {
				console.log('connection closed');
				process.exit(this.finished >= this.args.battles ? 0 : 1);
			}
			this._scheduleReconnect();
		});
		this.ws.addEventListener('error', err => console.error('websocket error:', err.message ?? err));
	}

	/**
	 * Reconnect with exponential backoff after an unexpected disconnect.
	 * In-flight battles are remembered and rejoined after re-login; their
	 * room state is discarded so the server's full-log replay on /join
	 * rebuilds trackers from scratch (no double-processing).
	 */
	_scheduleReconnect() {
		this.reconnectAttempts++;
		if (this.reconnectAttempts > RECONNECT_MAX_ATTEMPTS) {
			console.error(`giving up after ${RECONNECT_MAX_ATTEMPTS} reconnect attempts ` +
				`(${this.finished}/${this.args.battles} battles finished)`);
			this.infer.close();
			process.exit(1);
		}
		for (const roomid of this.rooms.keys()) {
			this.pendingRejoin.add(roomid);
			this.rooms.delete(roomid);
		}
		this.loggedIn = false;
		this.searching = false;
		this.joinRequested.clear();
		const delay = Math.min(RECONNECT_MAX_DELAY_MS,
			RECONNECT_BASE_DELAY_MS * 2 ** (this.reconnectAttempts - 1));
		console.log(`disconnected — reconnecting in ${Math.round(delay / 1000)}s ` +
			`(attempt ${this.reconnectAttempts}/${RECONNECT_MAX_ATTEMPTS}` +
			(this.pendingRejoin.size ? `, ${this.pendingRejoin.size} battle(s) to rejoin` : '') + ')');
		setTimeout(() => this._connect(), delay);
	}

	async login(challstr) {
		const userid = toID(this.username);
		let assertion = '';
		try {
			if (this.args.password) {
				const resp = await fetch(LOGIN_API, {
					method: 'POST',
					headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
					body: new URLSearchParams({ name: this.username, pass: this.args.password, challstr }),
				});
				const text = await resp.text();
				const data = JSON.parse(text.replace(/^\]/, ''));
				if (!data.assertion) throw new Error(`login rejected: ${text.slice(0, 200)}`);
				assertion = data.assertion;
			} else {
				const resp = await fetch(
					`${ASSERTION_API}?act=getassertion&userid=${userid}&challstr=${encodeURIComponent(challstr)}`);
				assertion = await resp.text();
				if (assertion === ';') throw new Error(`name '${this.username}' is registered — set PS_PASSWORD`);
				if (assertion.startsWith(';;')) throw new Error(`login failed: ${assertion.slice(2)}`);
			}
		} catch (err) {
			// Local --no-security servers accept a trn with no assertion.
			console.log(`assertion unavailable (${err.message}); trying unauthenticated /trn (local server)`);
			assertion = '';
		}
		this.send(`|/trn ${this.username},0,${assertion}`);
	}

	onLoggedIn() {
		if (this.loggedIn) return;
		this.loggedIn = true;
		this.reconnectAttempts = 0;
		console.log(`logged in as ${this.username}`);
		this.send('|/cmd rooms'); // keepalive-ish no-op
		if (this.pendingRejoin.size) {
			for (const roomid of this.pendingRejoin) {
				console.log(`rejoining interrupted battle: ${roomid}`);
				this.send(`|/join ${roomid}`);
			}
			this.pendingRejoin.clear();
			return; // resume searching only after the rejoined battle(s) finish
		}
		if (this.args.challenge) {
			this.send(`|/challenge ${this.args.challenge}, ${this.args.format}`);
			console.log(`challenged ${this.args.challenge} to ${this.args.format}`);
		} else if (!this.args.acceptFrom) {
			this.searchNext();
		} else {
			console.log(`waiting for challenges from ${this.args.acceptFrom}...`);
		}
	}

	searchNext() {
		if (this.finished >= this.args.battles) return;
		if (this.searching) return;
		this.searching = true;
		this.send(`|/search ${this.args.format}`);
		console.log(`searching ladder: ${this.args.format} ` +
			`(battle ${this.finished + 1}/${this.args.battles})`);
	}

	async receive(message) {
		const lines = message.split('\n');
		let roomid = '';
		if (lines[0].startsWith('>')) {
			roomid = lines.shift().slice(1);
		}
		const touched = new Set();

		for (const line of lines) {
			if (!line.startsWith('|')) continue;
			const parts = line.split('|');
			const type = parts[1] ?? '';

			if (!roomid) {
				// Global messages
				if (type === 'challstr') {
					await this.login(parts.slice(2).join('|'));
				} else if (type === 'updateuser') {
					const named = parts[3] === '1';
					if (named && toID(parts[2]) === toID(this.username)) this.onLoggedIn();
				} else if (type === 'pm' && parts[4] && parts[4].startsWith('/challenge')) {
					const from = parts[2].replace(/^[^a-zA-Z0-9]/, '');
					if (this.args.acceptFrom && toID(from) === toID(this.args.acceptFrom)) {
						this.send(`|/accept ${from}`);
						console.log(`accepted challenge from ${from}`);
					}
				} else if (type === 'updatesearch') {
					// After a reconnect, battles that started while we were
					// offline (ladder matched us mid-disconnect) only show up
					// here — join any game we aren't already tracking.
					try {
						const data = JSON.parse(parts.slice(2).join('|'));
						for (const gameid of Object.keys(data.games || {})) {
							if (gameid.startsWith('battle-') &&
								!this.rooms.has(gameid) && !this.finishedRooms.has(gameid) &&
								!this.joinRequested.has(gameid)) {
								this.joinRequested.add(gameid);
								console.log(`joining untracked battle: ${gameid}`);
								this.send(`|/join ${gameid}`);
							}
						}
					} catch {}
				} else if (type === 'popup' || type === 'nametaken') {
					console.error(`server: ${line}`);
					// A rejoin target that expired while we were offline —
					// don't stall the run waiting for it.
					if (line.includes("does not exist") && !this.rooms.size &&
						!this.args.challenge && !this.args.acceptFrom) {
						setTimeout(() => this.searchNext(), 3000);
					}
				}
				continue;
			}

			if (roomid.startsWith('battle-')) {
				if (this.finishedRooms.has(roomid)) continue; // late frames after /leave
				if (type === 'init' && this.rooms.has(roomid)) {
					// The server resends the full log on a rejoin/replayed join —
					// rebuild the room from scratch so trackers never see a line
					// twice.
					console.log(`re-initializing room from replay: ${roomid}`);
					this.rooms.delete(roomid);
				}
				let room = this.rooms.get(roomid);
				if (!room) {
					room = new BattleRoom(roomid, this);
					this.rooms.set(roomid, room);
					this.searching = false;
					console.log(`battle started: ${roomid}`);
					this.send(`${roomid}|/timer on`);
				}
				room.receiveLine(line);
				touched.add(roomid);
			}
		}

		// Frame end: act on any pending requests, finalize done battles.
		for (const id of touched) {
			const room = this.rooms.get(id);
			if (!room) continue;
			await room.maybeAct();
			if (room.done && room.result) {
				this.finished++;
				if (room.result === 'win') this.wins++;
				console.log(`battle finished: ${id} — ${room.result} vs ${room.opponent} ` +
					`(${room.decisions} decisions${this.mctsReady ? `, ${room.searched} searched` : ''}, ` +
					`max ${room.maxLatencyMs}ms) [${this.wins}/${this.finished} won]`);
				room.save(this.args.saveDir);
				this._saveRunState();
				this.rooms.delete(id);
				this.finishedRooms.add(id);
				this.send(`${id}|/leave`);
				// Rated battles are joined under both the base roomid and a
				// -suffixed (password) alias; retire the twin so it doesn't
				// linger as a phantom room.
				for (const otherId of this.rooms.keys()) {
					if (id.startsWith(otherId) || otherId.startsWith(id)) {
						this.rooms.delete(otherId);
						this.finishedRooms.add(otherId);
						this.send(`${otherId}|/leave`);
					}
				}
				if (this.finished >= this.args.battles) {
					console.log(`done: ${this.wins}/${this.finished} battles won`);
					this.shuttingDown = true;
					this.infer.close();
					this.ws.close();
				} else if (this.args.challenge) {
					setTimeout(() => {
						this.send(`|/challenge ${this.args.challenge}, ${this.args.format}`);
						console.log(`re-challenged ${this.args.challenge} ` +
							`(battle ${this.finished + 1}/${this.args.battles})`);
					}, 3000);
				} else if (!this.args.acceptFrom) {
					setTimeout(() => this.searchNext(), 5000);
				}
			}
		}
	}
}

async function main() {
	const args = parseArgs(process.argv);
	const bot = new LadderBot(args);
	await bot.start();
}

// Guarded so test/tools/ladder-results.test.js can require the result-logging
// pieces without connecting to a server.
if (require.main === module) {
	main().catch(err => {
		console.error(err);
		process.exit(1);
	});
}

exports.BattleRoom = BattleRoom;
exports.CSV_HEADER = CSV_HEADER;
exports.csvField = csvField;
