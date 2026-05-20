/**
 * Pokemon Gym Environment
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Step-based RL interface over a single Pokemon battle.
 * reset() returns an observation, step(action) returns (obs, reward, done, info).
 *
 * @license MIT
 */

import { BattleStream, getPlayerStreams, BattlePlayer } from '../battle-stream';
import { PRNG } from '../prng';
import type { PRNGSeed } from '../prng';
import { RandomPlayerAI } from './random-player-ai';
import { DamageFirstAI } from './damage-first-ai';
import { extractFeatures, extractFeaturesStructured } from './feature-extractor';
import type { BoostData } from './feature-extractor';
import type { ChoiceRequest, MoveRequest, SwitchRequest } from '../side';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface GymStepResult {
	obs: Float32Array;
	reward: number;
	done: boolean;
	info: { winner?: string; turns?: number; illegalMove?: boolean };
}

// ---------------------------------------------------------------------------
// GymPlayer — BattlePlayer that exposes a Promise-based request interface
// ---------------------------------------------------------------------------

class GymPlayer extends BattlePlayer {
	private _requestResolve: ((request: ChoiceRequest | null) => void) | null = null;
	private _requestPromise: Promise<ChoiceRequest | null> | null = null;

	_currentRequest: ChoiceRequest | null = null;

	constructor(playerStream: import('../../lib/streams').ObjectReadWriteStream<string>) {
		super(playerStream, false);
	}

	override async start() {
		await super.start();
		// Stream closed (battle ended) — unblock any pending waitForRequest
		if (this._requestResolve) {
			const resolve = this._requestResolve;
			this._requestResolve = null;
			this._requestPromise = null;
			resolve(null);
		}
	}

	/**
	 * Returns a Promise that resolves the next time an actionable request arrives.
	 * WaitRequests are skipped — the promise stays pending until a real request comes.
	 * Resolves null if the player stream closes (battle ended).
	 */
	waitForRequest(): Promise<ChoiceRequest | null> {
		this._requestPromise = new Promise<ChoiceRequest | null>(resolve => {
			this._requestResolve = resolve;
		});
		return this._requestPromise;
	}

	override receiveRequest(request: ChoiceRequest): void {
		this._currentRequest = request;
		// Skip WaitRequests — keep the promise pending until a real request arrives
		if ('wait' in request && request.wait) return;
		if (this._requestResolve) {
			const resolve = this._requestResolve;
			this._requestResolve = null;
			this._requestPromise = null;
			resolve(request);
		}
	}

	/**
	 * Send a choice string to the battle engine.
	 */
	submitChoice(choice: string): void {
		this.choose(choice);
	}
}

// ---------------------------------------------------------------------------
// PokemonGymEnv
// ---------------------------------------------------------------------------

export class PokemonGymEnv {
	private readonly _format: string;
	private readonly _seed: PRNGSeed | undefined;
	private readonly _opponent: string;

	private _battleStream!: BattleStream;
	private _streams!: ReturnType<typeof getPlayerStreams>;
	private _gymPlayer!: GymPlayer;
	private _randomOpponent!: RandomPlayerAI;

	private _turnCount = 0;
	private _done = false;
	private _currentRequest: ChoiceRequest | null = null;

	// Background omniscient reader state
	private _omniscientLines: string[] = [];
	private _omniscientNotify: (() => void) | null = null;
	private _omniscientDone = false;
	private _omniscientTask: Promise<void> | null = null;

	// Track win result from omniscient stream
	private _winResult: string | null = null;
	private _winResolve: ((winner: string) => void) | null = null;
	private _winPromise: Promise<string> | null = null;

	// Stat boost accumulators (reset on switch)
	private _ownBoosts: Record<string, number> = {};
	private _oppBoosts: Record<string, number> = {};

	constructor(options: { seed?: PRNGSeed; format?: string; opponent?: 'random' | 'damage-first' } = {}) {
		this._format = options.format ?? 'gen1randombattle';
		this._seed = options.seed;
		this._opponent = options.opponent ?? 'random';
	}

	// -------------------------------------------------------------------------
	// reset
	// -------------------------------------------------------------------------

	async reset(): Promise<Float32Array> {
		this.destroy();

		this._battleStream = new BattleStream();
		this._streams = getPlayerStreams(this._battleStream);

		this._gymPlayer = new GymPlayer(this._streams.p1);
		this._randomOpponent = this._opponent === 'damage-first'
			? new DamageFirstAI(this._streams.p2, { seed: this._seed })
			: new RandomPlayerAI(this._streams.p2, { seed: this._seed });

		// Reset background reader state
		this._omniscientLines = [];
		this._omniscientNotify = null;
		this._omniscientDone = false;
		this._winResult = null;
		this._winResolve = null;
		this._winPromise = null;

		// Reset boost accumulators
		this._ownBoosts = {};
		this._oppBoosts = {};

		// Start players (they loop reading their streams)
		void this._gymPlayer.start();
		void this._randomOpponent.start();

		// Start background omniscient reader
		this._omniscientTask = this._runOmniscientReader();

		// Build seeds for the start message
		const prng = new PRNG(this._seed ?? null);
		const spec = { formatid: this._format, seed: prng.getSeed() };
		const p1spec = { name: 'Gym', seed: prng.getSeed() };
		const p2spec = { name: 'Opponent', seed: prng.getSeed() };

		const initMessage =
			`>start ${JSON.stringify(spec)}\n` +
			`>player p1 ${JSON.stringify(p1spec)}\n` +
			`>player p2 ${JSON.stringify(p2spec)}`;

		// Set up the first waitForRequest BEFORE writing start so we don't miss it
		const firstRequest = this._gymPlayer.waitForRequest();
		void this._streams.omniscient.write(initMessage);

		// Wait for the first request to arrive
		const request = await firstRequest;
		this._currentRequest = request;
		this._turnCount = 0;
		this._done = false;

		return extractFeaturesStructured(this._currentRequest!, null, this._currentBoosts());
	}

	// -------------------------------------------------------------------------
	// step
	// -------------------------------------------------------------------------

	async step(action: number): Promise<GymStepResult> {
		if (this._done) {
			throw new Error('Battle already done; call reset()');
		}

		// Validate action range
		if (action < 0 || action > 8) {
			return {
				obs: extractFeaturesStructured(this._currentRequest!, null, this._currentBoosts()),
				reward: -0.01,
				done: false,
				info: { illegalMove: true },
			};
		}

		// Check validActions mask
		const mask = this.validActions();
		if (!mask[action]) {
			return {
				obs: extractFeaturesStructured(this._currentRequest!, null, this._currentBoosts()),
				reward: -0.01,
				done: false,
				info: { illegalMove: true },
			};
		}

		// Convert action index to choice string
		const choiceStr = actionToChoice(action);

		// Register a new request waiter before submitting so we don't miss it
		const nextRequestPromise = this._gymPlayer.waitForRequest();
		const winPromise = this._waitForWin();

		// Submit the choice — battle engine will process it and emit lines
		this._gymPlayer.submitChoice(choiceStr);

		// Snapshot current omniscient line count before we wait
		const linesBefore = this._omniscientLines.length;

		// Race: either we get a new request (next turn) or the battle ends
		type RequestEvent = { type: 'request'; request: ChoiceRequest | null };
		type WinEvent = { type: 'win'; winner: string };

		const result = await Promise.race<RequestEvent | WinEvent>([
			nextRequestPromise.then(r => ({ type: 'request' as const, request: r })),
			winPromise.then(w => ({ type: 'win' as const, winner: w })),
		]);

		// Yield to I/O callbacks so the omniscient reader can flush any pending lines
		await new Promise<void>(resolve => setImmediate(resolve));
		const newLines = this._omniscientLines.slice(linesBefore);

		// Parse reward from omniscient lines
		let reward = 0;
		let winner: string | undefined;
		let done = false;

		for (const line of newLines) {
			if (line.startsWith('|faint|p2a:') || line.startsWith('|faint|p2b:')) {
				reward += 0.01;
			} else if (line.startsWith('|faint|p1a:') || line.startsWith('|faint|p1b:')) {
				reward -= 0.01;
			} else if (line.startsWith('|-status|p2')) {
				reward += 0.0001;
			} else if (line.startsWith('|win|Gym')) {
				reward += 1.0;
				done = true;
				winner = 'Gym';
			} else if (line.startsWith('|win|Opponent')) {
				reward -= 1.0;
				done = true;
				winner = 'Opponent';
			} else if (line.startsWith('|turn|')) {
				const turnStr = line.slice('|turn|'.length).trim();
				const turnNum = parseInt(turnStr, 10);
				if (!isNaN(turnNum)) this._turnCount = turnNum;
			}
		}

		// Capture win from race result
		if (result.type === 'win' && !done) {
			done = true;
			winner = result.winner;
			if (winner === 'Gym') {
				reward += 1.0;
			} else {
				reward -= 1.0;
			}
		}

		// Fallback: check _winResult in case the |win| line arrived after the race resolved
		if (!done && this._winResult !== null) {
			done = true;
			winner = this._winResult;
			reward += (winner === 'Gym') ? 1.0 : -1.0;
		}

		// Fallback: null request means player stream closed (battle ended without win line)
		if (!done && result.type === 'request' && result.request === null) {
			done = true;
			winner = this._winResult ?? undefined;
		}

		// Stalling penalty (applied on terminal)
		if (done) {
			reward -= 0.001 * this._turnCount;
			this._done = true;
		}

		// Clip reward to [-1, +1]
		reward = Math.max(-1, Math.min(1, reward));

		// Update current request if we got a real (non-null) one
		if (result.type === 'request' && result.request !== null) {
			this._currentRequest = result.request;
		}

		const obs = extractFeaturesStructured(this._currentRequest!, null, this._currentBoosts());

		return {
			obs,
			reward,
			done,
			info: {
				winner,
				turns: this._turnCount,
			},
		};
	}

	// -------------------------------------------------------------------------
	// validActions
	// -------------------------------------------------------------------------

	/**
	 * Returns a boolean array of length 9:
	 *   [move0, move1, move2, move3, switch1, switch2, switch3, switch4, switch5]
	 */
	validActions(): boolean[] {
		const result: boolean[] = new Array(9).fill(false);

		const request = this._currentRequest;
		if (!request || request.wait || request.teamPreview) {
			return result;
		}

		if (isMoveRequest(request)) {
			// Moves 0-3 only — no voluntary switches.
			// RandomPlayerAI never switches voluntarily (move=1.0), so including
			// switch actions in the move-turn mask gives the gym agent a ~55% switch
			// rate against a never-switching opponent, producing a ~4% win rate baseline
			// instead of ~50%.  Force-switches (after KOs) are still handled below.
			const activeSlot = request.active[0];
			if (activeSlot) {
				for (let i = 0; i < 4; i++) {
					const move = activeSlot.moves[i];
					if (move && !move.disabled) {
						result[i] = true;
					}
				}
			}
		} else if (isSwitchRequest(request)) {
			// Force-switch: only switches are available
			const pokemon = request.side.pokemon;
			for (let slot = 1; slot <= 5; slot++) {
				const poke = pokemon[slot];
				if (!poke) continue;
				const isFainted = poke.condition.endsWith(' fnt') || poke.condition === '0 fnt';
				if (!isFainted) {
					result[slot + 3] = true;
				}
			}
		}

		return result;
	}

	// -------------------------------------------------------------------------
	// destroy
	// -------------------------------------------------------------------------

	destroy(): void {
		this._done = true;

		if (this._streams) {
			try {
				this._streams.omniscient.pushEnd();
			} catch {
				// Ignore errors during cleanup — stream may already be closed
			}
		}

		// Resolve any pending win watchers so they don't hang
		if (this._winResolve) {
			this._winResolve('__destroyed__');
			this._winResolve = null;
		}

		this._omniscientDone = true;
		if (this._omniscientNotify) {
			this._omniscientNotify();
			this._omniscientNotify = null;
		}
	}

	// -------------------------------------------------------------------------
	// Internal helpers
	// -------------------------------------------------------------------------

	private async _runOmniscientReader(): Promise<void> {
		try {
			for await (const chunk of this._streams.omniscient) {
				for (const line of chunk.split('\n')) {
					if (!line) continue;
					this._omniscientLines.push(line);

					// Notify any pending waiters that new data arrived
					if (this._omniscientNotify) {
						const notify = this._omniscientNotify;
						this._omniscientNotify = null;
						notify();
					}

					// Check for win and resolve win promise
					if (line.startsWith('|win|')) {
						const winnerName = line.slice('|win|'.length).trim();
						this._winResult = winnerName;
						if (this._winResolve) {
							const resolve = this._winResolve;
							this._winResolve = null;
							this._winPromise = null;
							resolve(winnerName);
						}
					}

					// Track stat boosts: |-boost|p1a:<name>|<stat>|<amount>
					// and switches: |switch|p1a: resets boosts for that side
					if (line.startsWith('|-boost|') || line.startsWith('|-unboost|')) {
						// split on '|': ['', '-boost'/'−unboost', 'p1a:...', 'stat', 'amount']
						const parts = line.split('|');
						const eventType = parts[1]; // '-boost' or '-unboost'
						const pokemonIdent = parts[2] ?? ''; // e.g. 'p1a:Pikachu'
						const stat = parts[3];
						const amount = parseInt(parts[4], 10);
						if (stat && !isNaN(amount)) {
							if (pokemonIdent.startsWith('p1a:')) {
								const cur = this._ownBoosts[stat] ?? 0;
								this._ownBoosts[stat] = Math.max(-6, Math.min(6,
									eventType === '-boost' ? cur + amount : cur - amount));
							} else if (pokemonIdent.startsWith('p2a:')) {
								const cur = this._oppBoosts[stat] ?? 0;
								this._oppBoosts[stat] = Math.max(-6, Math.min(6,
									eventType === '-boost' ? cur + amount : cur - amount));
							}
						}
					} else if (line.startsWith('|switch|p1a:')) {
						this._ownBoosts = {};
					} else if (line.startsWith('|switch|p2a:')) {
						this._oppBoosts = {};
					}
				}
			}
		} catch {
			// Stream closed or destroyed — ignore
		} finally {
			this._omniscientDone = true;
			// Resolve any hanging win waiters
			if (this._winResolve) {
				this._winResolve('__ended__');
				this._winResolve = null;
			}
			if (this._omniscientNotify) {
				this._omniscientNotify();
				this._omniscientNotify = null;
			}
		}
	}

	/**
	 * Returns a snapshot of current boost data for both sides.
	 */
	private _currentBoosts(): BoostData {
		return {
			ownActive: { ...this._ownBoosts },
			oppActive: { ...this._oppBoosts },
		};
	}

	/**
	 * Returns a Promise that resolves when a |win| line is seen.
	 * If a win was already seen, resolves immediately.
	 */
	private _waitForWin(): Promise<string> {
		if (this._winResult !== null) {
			return Promise.resolve(this._winResult);
		}
		if (this._omniscientDone) {
			return Promise.resolve('__ended__');
		}
		// Reuse existing win promise if one is pending
		if (this._winPromise) {
			return this._winPromise;
		}
		this._winPromise = new Promise<string>(resolve => {
			this._winResolve = resolve;
		});
		return this._winPromise;
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function isMoveRequest(r: ChoiceRequest): r is MoveRequest {
	return 'active' in r && !('forceSwitch' in r);
}

function isSwitchRequest(r: ChoiceRequest): r is SwitchRequest {
	return 'forceSwitch' in r;
}

/**
 * Convert a 0-indexed action (0-8) to a Pokemon Showdown choice string.
 *   0-3  → "move 1" through "move 4"
 *   4-8  → "switch 2" through "switch 6"
 */
function actionToChoice(action: number): string {
	if (action <= 3) {
		return `move ${action + 1}`;
	}
	// action 4 → switch 2 (bench slot 1 = pokemon index 1)
	return `switch ${action - 2}`;
}
