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
import { State } from '../state';
import { toID } from '../dex-data';
import { RandomPlayerAI } from './random-player-ai';
import { DamageFirstAI } from './damage-first-ai';
import { extractFeatures, extractFeaturesStructured, BOOST_ORDER, N_TOKENS, TOKEN_DIM, TOKEN_DIM_V2 } from './feature-extractor';
import type { OpponentPokemonInfo, ActiveVolatiles, V3Info } from './feature-extractor';
import { TOKEN_DIM_V3, TOKEN_DIM_V3_EXT } from './type-chart-v3';
import type { ChoiceRequest, MoveRequest, SwitchRequest } from '../side';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * 'structured' (default) returns the (12, 65) per-Pokémon token observation
 * from extractFeaturesStructured(), flattened to 780 floats. 'flat' returns
 * the legacy 100-dim extractFeatures() vector for MLP-baseline regression
 * checks against the M1 result. 'structured-v2' (M3.4) returns the (12, 77)
 * schema-v2 observation — v1 tokens plus boost stages and volatile-condition
 * flags on the two active tokens — flattened to 924 floats. 'structured-v3'
 * (M7) returns the (12, 86) schema-v3 observation — v2's 77 dims byte-identical
 * plus per-move type-effectiveness, move-effect flags, inflicted-status id, and
 * the log-derived Sleep Clause flag — flattened to 1032 floats.
 * 'structured-v3-extended' (M8 Phase 1A) returns the (12, 87) schema — v3's 86
 * dims byte-identical plus the own/opp active base-speed ratio appended at dim
 * 86 on every token — flattened to 1044 floats.
 */
export type ObsMode = 'flat' | 'structured' | 'structured-v2' | 'structured-v3' | 'structured-v3-extended';

/**
 * Who sits in the p2 seat (M3.3):
 *   'random'      — RandomPlayerAI (legacy default; never voluntarily switches)
 *   'damagefirst' — DamageFirstAI heuristic (always picks the highest-base-power move)
 *   'self'        — a second GymPlayer seat, driven externally via the dual-seat
 *                   API (resetDual/stepDual) for self-play training
 */
export type OpponentKind = 'random' | 'damagefirst' | 'self';

export interface GymStepResult {
	obs: Float32Array;
	reward: number;
	done: boolean;
	/**
	 * `oppAction` (M5): the opponent's resolved simultaneous choice at this
	 * decision point, mapped into the opponent's OWN 9-way action frame
	 * (0–3 = moves in the opponent's request-slot order, 4–8 = switches in the
	 * opponent's bench order). -1 when masked — no simultaneous opponent choice
	 * (e.g. a p1-only force-switch), Struggle / no-op sleep-freeze, or an
	 * unmappable choice. See `_opponentActionLabel`.
	 */
	info: { winner?: string; turns?: number; illegalMove?: boolean; oppAction?: number };
}

/** Per-seat state returned by the dual-seat (self-play) API. */
export interface DualSeatState {
	obs: Float32Array;
	mask: boolean[];
	/** True if this seat must supply an action on the next stepDual call. */
	needsAction: boolean;
}

export interface DualStepResult {
	p1: DualSeatState;
	p2: DualSeatState;
	/** Reward from p1's perspective — self-play trains the p1 seat only. */
	reward: number;
	done: boolean;
	/**
	 * `oppAction` (M5) labels are symmetric in dual-seat self-play: each seat's
	 * transition is labeled with the OTHER seat's simultaneous choice, in that
	 * seat's opponent's own 9-way frame. `oppAction` is p1's label (= the p2
	 * seat's simultaneous action); `oppActionP2` is p2's label (= the p1 seat's
	 * simultaneous action). Both are -1 when there is no simultaneous choice by
	 * both seats this step. Because the dual-seat actions are supplied by the
	 * caller already in each seat's own frame, they are the labels directly.
	 */
	info: { winner?: string; turns?: number; illegalMove?: boolean; oppAction?: number; oppActionP2?: number };
}

/** Internal opponent-tracker record — reconstructed from the battle log. */
interface OpponentRecord {
	details: string;
	condition: string;
	active: boolean;
	moves: string[];
}

/**
 * Serializable snapshot of an ObservationTrackers instance (plain JSON —
 * safe to deep-copy or ship over the bridge).
 */
export interface TrackerSnapshot {
	revealOrder: { p1: string[], p2: string[] };
	revealRecords: { p1: Array<[string, OpponentRecord]>, p2: Array<[string, OpponentRecord]> };
	volatiles: { p1: SideVolatiles, p2: SideVolatiles };
	/**
	 * Per-side opponent-inflicted-sleep state (M7 Sleep Clause). Keyed by
	 * Pokémon nickname; `true` while that mon is asleep from a sleep move the
	 * OTHER side used (Rest-sourced self-sleep is excluded). Bench-persistent —
	 * survives switch-out, cleared only on wake (`-curestatus`) or faint.
	 */
	sleepInflicted: { p1: Array<[string, boolean]>, p2: Array<[string, boolean]> };
}

/**
 * Everything needed to reconstruct a live battle inside BattleSim (M4 MCTS
 * forward model): the engine state plus the gym's log-derived tracker state.
 * Produced by PokemonGymEnv.snapshot(); consumed by BattleSim.fromSnapshot().
 */
export interface GymSnapshot {
	/** State.serializeBattle() output, JSON deep-copied (owns no live refs). */
	battleState: AnyObject;
	trackers: TrackerSnapshot;
	obsMode: ObsMode;
	turnCount: number;
}

/**
 * Internal per-side volatile tracker (M3.4, obsMode 'structured-v2') —
 * boost stages and volatile conditions of the side's ACTIVE Pokémon,
 * reconstructed purely from public battle-log lines. Everything resets on
 * switch/drag/faint, matching gen1 mechanics (boosts, screens, Substitute,
 * Leech Seed, and the toxic counter are all lost on switch-out).
 */
interface SideVolatiles {
	boosts: Record<string, number>;
	reflect: boolean;
	lightScreen: boolean;
	substitute: boolean;
	leechSeed: boolean;
	toxicCounter: number;
}

function freshSideVolatiles(): SideVolatiles {
	return {
		boosts: {},
		reflect: false,
		lightScreen: false,
		substitute: false,
		leechSeed: false,
		toxicCounter: 0,
	};
}

function toActiveVolatiles(v: SideVolatiles): ActiveVolatiles {
	return {
		boosts: BOOST_ORDER.map(stat => v.boosts[stat] ?? 0),
		reflect: v.reflect,
		lightScreen: v.lightScreen,
		substitute: v.substitute,
		leechSeed: v.leechSeed,
		toxicCounter: v.toxicCounter,
	};
}

// ---------------------------------------------------------------------------
// ObservationTrackers — log-derived observation state, shared by the live gym
// and the BattleSim forward model (M4)
// ---------------------------------------------------------------------------

/**
 * Reveal + volatile trackers reconstructed purely from battle-log lines a
 * real player could see — never omniscient state. Both sides are tracked:
 * p2's records feed p1's opponent tokens and vice versa.
 *
 * Extracted from PokemonGymEnv (M4) so BattleSim can snapshot this state
 * alongside the serialized battle and keep processing new log lines with
 * identical semantics inside simulated rollouts.
 */
export class ObservationTrackers {
	revealOrder: { p1: string[], p2: string[] } = { p1: [], p2: [] };
	revealRecords: { p1: Map<string, OpponentRecord>, p2: Map<string, OpponentRecord> } = {
		p1: new Map(), p2: new Map(),
	};
	volatiles: { p1: SideVolatiles, p2: SideVolatiles } = {
		p1: freshSideVolatiles(), p2: freshSideVolatiles(),
	};

	/**
	 * Per-side Sleep Clause state (M7). For side S, maps a nickname → `true`
	 * while that Pokémon is asleep from a sleep move the OTHER side inflicted.
	 * Rest-sourced (self-induced) sleep is deliberately NOT recorded here, since
	 * it does not count toward the Sleep Clause. Bench-persistent: never reset on
	 * switch/drag, only on wake (`-curestatus ... slp`) or `faint`. Derived
	 * entirely from public log lines (a slept mon was necessarily active when
	 * slept), so no omniscient state is used.
	 */
	sleepInflicted: { p1: Map<string, boolean>, p2: Map<string, boolean> } = {
		p1: new Map(), p2: new Map(),
	};

	/** Update all trackers from one battle-log line. */
	processLine(line: string): void {
		this._processRevealLine(line);
		this._processVolatileLine(line);
		this._processSleepClauseLine(line);
	}

	/**
	 * Sleep Clause flag for the `viewer` seat's v3 observation (dim 85): 1 iff
	 * any opponent Pokémon that `viewer` put to sleep (non-Rest) is still asleep.
	 * From p1's view the opponent is p2, so this reads p2's inflicted-sleep map.
	 */
	sleepClauseFlagFor(viewer: 'p1' | 'p2'): number {
		const opp = viewer === 'p1' ? 'p2' : 'p1';
		for (const asleep of this.sleepInflicted[opp].values()) {
			if (asleep) return 1;
		}
		return 0;
	}

	/** The `viewer` seat's revealed knowledge of the OTHER side's team. */
	opponentInfoFor(viewer: 'p1' | 'p2'): OpponentPokemonInfo[] {
		const side = viewer === 'p1' ? 'p2' : 'p1';
		return this.revealOrder[side].map(nickname => {
			const record = this.revealRecords[side].get(nickname)!;
			return {
				details: record.details,
				condition: record.condition,
				active: record.active,
				moves: record.moves,
			};
		});
	}

	/** The `viewer` seat's volatile view (own side + opponent side). */
	volatilesFor(viewer: 'p1' | 'p2'): { own: ActiveVolatiles, opp: ActiveVolatiles } {
		const other = viewer === 'p1' ? 'p2' : 'p1';
		return {
			own: toActiveVolatiles(this.volatiles[viewer]),
			opp: toActiveVolatiles(this.volatiles[other]),
		};
	}

	/** Nicknames of `side`'s Pokémon revealed so far (BattleSim determinizer). */
	revealedNicknames(side: 'p1' | 'p2'): Set<string> {
		return new Set(this.revealOrder[side]);
	}

	snapshot(): TrackerSnapshot {
		return JSON.parse(JSON.stringify({
			revealOrder: this.revealOrder,
			revealRecords: {
				p1: Array.from(this.revealRecords.p1.entries()),
				p2: Array.from(this.revealRecords.p2.entries()),
			},
			volatiles: this.volatiles,
			sleepInflicted: {
				p1: Array.from(this.sleepInflicted.p1.entries()),
				p2: Array.from(this.sleepInflicted.p2.entries()),
			},
		}));
	}

	static fromSnapshot(snap: TrackerSnapshot): ObservationTrackers {
		const copy: TrackerSnapshot = JSON.parse(JSON.stringify(snap));
		const trackers = new ObservationTrackers();
		trackers.revealOrder = copy.revealOrder;
		trackers.revealRecords = {
			p1: new Map(copy.revealRecords.p1),
			p2: new Map(copy.revealRecords.p2),
		};
		trackers.volatiles = copy.volatiles;
		// Backward-compatible with pre-M7 snapshots that predate this field.
		trackers.sleepInflicted = {
			p1: new Map(copy.sleepInflicted?.p1 ?? []),
			p2: new Map(copy.sleepInflicted?.p2 ?? []),
		};
		return trackers;
	}

	private _processRevealLine(line: string): void {
		const parts = line.split('|');
		const type = parts[1];
		const ident = parts[2] ?? '';
		const side = ident.startsWith('p2') ? 'p2' as const : ident.startsWith('p1') ? 'p1' as const : null;
		if (!side) return;
		const records = this.revealRecords[side];
		const nickname = extractNickname(ident);

		if (type === 'switch' || type === 'drag') {
			for (const record of records.values()) record.active = false;

			const details = parts[3] ?? '';
			const condition = parts[4] ?? '';
			const existing = records.get(nickname);
			if (existing) {
				existing.details = details;
				existing.condition = condition;
				existing.active = true;
			} else {
				records.set(nickname, { details, condition, active: true, moves: [] });
				this.revealOrder[side].push(nickname);
			}
		} else if (type === '-damage' || type === '-heal') {
			const record = records.get(nickname);
			if (record) record.condition = parts[3] ?? record.condition;
		} else if (type === '-status') {
			const record = records.get(nickname);
			if (record) {
				const statusToken = parts[3] ?? '';
				record.condition = `${hpFraction(record.condition)} ${statusToken}`.trim();
			}
		} else if (type === '-curestatus') {
			const record = records.get(nickname);
			if (record) record.condition = hpFraction(record.condition);
		} else if (type === 'faint') {
			const record = records.get(nickname);
			if (record) {
				record.condition = '0 fnt';
				record.active = false;
			}
		} else if (type === 'move') {
			const record = records.get(nickname);
			if (record && record.moves.length < 4) {
				const moveId = toID(parts[3] ?? '');
				if (moveId && !record.moves.includes(moveId)) record.moves.push(moveId);
			}
		}
	}

	/**
	 * Update the per-side volatile trackers (boosts, screens, Substitute,
	 * Leech Seed, toxic counter) from one battle-log line. All of these are
	 * public information. Gen1 line inventory:
	 *   |-boost|p1a: X|atk|1     |-unboost|...        (stat stages)
	 *   |-clearallboost|[silent]                      (Haze — both sides)
	 *   |-start|p1a: X|Reflect / Light Screen / Substitute / move: Leech Seed
	 *   |-end|p1a: X|<same>                           (incl. Haze's [silent] ends)
	 *   |-status|p1a: X|tox / |-curestatus|           (toxic-counter lifecycle)
	 *   |-damage|p1a: X|HP|[from] psn                 (one poison tick)
	 *   |switch/|drag/|faint                          (everything resets)
	 */
	private _processVolatileLine(line: string): void {
		const parts = line.split('|');
		const type = parts[1];

		if (type === '-clearallboost') {
			this.volatiles.p1.boosts = {};
			this.volatiles.p2.boosts = {};
			return;
		}

		const ident = parts[2] ?? '';
		const side = ident.startsWith('p2') ? 'p2' as const : ident.startsWith('p1') ? 'p1' as const : null;
		if (!side) return;
		const vol = this.volatiles[side];

		if (type === 'switch' || type === 'drag' || type === 'faint') {
			this.volatiles[side] = freshSideVolatiles();
		} else if (type === '-boost' || type === '-unboost') {
			const stat = parts[3] ?? '';
			const amount = parseInt(parts[4] ?? '', 10);
			if (!stat || isNaN(amount)) return;
			const delta = type === '-boost' ? amount : -amount;
			vol.boosts[stat] = Math.max(-6, Math.min(6, (vol.boosts[stat] ?? 0) + delta));
		} else if (type === '-start' || type === '-end') {
			const condition = (parts[3] ?? '').replace(/^move: /, '');
			const value = type === '-start';
			if (condition === 'Reflect') vol.reflect = value;
			else if (condition === 'Light Screen') vol.lightScreen = value;
			else if (condition === 'Substitute') vol.substitute = value;
			else if (condition === 'Leech Seed') vol.leechSeed = value;
		} else if (type === '-status' || type === '-curestatus') {
			vol.toxicCounter = 0;
		} else if (type === '-damage' && line.includes('[from] psn')) {
			vol.toxicCounter++;
		}
	}

	/**
	 * Maintain the per-side Sleep Clause state (M7) from one public log line.
	 * Only three transitions matter, and none of them is a switch/drag — sleep
	 * persists on the bench:
	 *   |-status|pXa: Nick|slp            → opponent-inflicted sleep starts
	 *   |-status|pXa: Nick|slp|[from] move: Rest → self-sleep, EXCLUDED (Rest)
	 *   |-curestatus|pXa: Nick|slp        → woke up, clear the flag
	 *   |faint|pXa: Nick                  → fainted, clear the flag
	 * The recorded side is the side the status is ON; `sleepClauseFlagFor` reads
	 * the OPPONENT side's map so the flag reflects "a mon WE slept".
	 */
	private _processSleepClauseLine(line: string): void {
		const parts = line.split('|');
		const type = parts[1];
		if (type !== '-status' && type !== '-curestatus' && type !== 'faint') return;

		const ident = parts[2] ?? '';
		const side = ident.startsWith('p2') ? 'p2' as const : ident.startsWith('p1') ? 'p1' as const : null;
		if (!side) return;
		const nickname = extractNickname(ident);
		const map = this.sleepInflicted[side];

		if (type === 'faint') {
			map.delete(nickname);
			return;
		}

		// -status / -curestatus: only the sleep status is relevant.
		if ((parts[3] ?? '') !== 'slp') return;

		if (type === '-curestatus') {
			// Woke up (or otherwise cured of sleep) — no longer counts.
			map.delete(nickname);
			return;
		}

		// type === '-status' with slp. Rest-sourced self-sleep is excluded: it
		// does not count toward the Sleep Clause.
		if (line.includes('move: Rest')) return;
		map.set(nickname, true);
	}
}

// ---------------------------------------------------------------------------
// GymPlayer — BattlePlayer that exposes a Promise-based request interface
// ---------------------------------------------------------------------------

class GymPlayer extends BattlePlayer {
	private _requestResolve: ((request: ChoiceRequest | null) => void) | null = null;
	private _requestPromise: Promise<ChoiceRequest | null> | null = null;

	_currentRequest: ChoiceRequest | null = null;

	/**
	 * Count of actionable (non-wait) requests received. The dual-seat
	 * self-play mode compares this against a consumed-count to tell fresh
	 * requests (seat must act) from stale ones (already acted on).
	 */
	requestCount = 0;

	/** Last actionable request — unlike _currentRequest, never a wait request. */
	_lastActionable: ChoiceRequest | null = null;

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
		this.requestCount++;
		this._lastActionable = request;
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

/**
 * Per-turn cost applied to the terminal reward. Historically 0.001 and
 * unreachable from any flag; kept as the default so existing recipes reproduce.
 * Pass `stallPenalty: 0` to disable it — see `applyStallPenalty` for why that
 * is worth testing.
 */
export const DEFAULT_STALL_PENALTY = 0.001;

/**
 * Apply the duration cost to a terminal reward.
 *
 * **This clips first and penalises second, which is a fix.** The original order
 * (penalise, then clip to [-1,1]) made the penalty one-sided: a loss is already
 * at or past -1 before the penalty lands, so the clip absorbed it entirely,
 * while a win felt it in full. Measured over the real range:
 *
 * ```
 *   turns      WIN     LOSS      <- pre-fix
 *      10    1.000   -1.000
 *     120    0.890   -1.000      loss column is flat; the penalty never applies
 * ```
 *
 * So the agent was punished for taking a long time to *win* but not for taking
 * a long time to *lose* — a standing gradient against patient, positional play.
 * Switching costs tempo and lengthens games, and switching is the central skill
 * in Gen 1, so this pushed directly against the behaviour we want.
 *
 * Clipping first makes duration cost the same in both directions. Note the
 * result is no longer bounded below by -1 (it reaches about -1.2 in a long
 * game); that is intended — an unbounded-below cost is what makes it symmetric.
 *
 * Worth knowing: PPO's own `gamma` already discounts distant terminal reward,
 * and does so symmetrically, so this penalty is arguably redundant. That is why
 * it is now settable — `stallPenalty: 0` is a live experiment, not a fallback.
 */
export function applyStallPenalty(reward: number, turns: number, penalty: number): number {
	return Math.max(-1, Math.min(1, reward)) - penalty * turns;
}

export class PokemonGymEnv {
	private readonly _format: string;
	private readonly _seed: PRNGSeed | undefined;
	private readonly _obsMode: ObsMode;

	private _battleStream!: BattleStream;
	private _streams!: ReturnType<typeof getPlayerStreams>;
	private _gymPlayer!: GymPlayer;
	private _opponentPlayer: RandomPlayerAI | null = null;
	/** Second gym seat, only in opponent: 'self' (dual-seat self-play) mode. */
	private _p2Player: GymPlayer | null = null;
	private readonly _opponent: OpponentKind;
	private readonly _stallPenalty: number;
	/** requestCount high-water marks already acted on, per seat (dual mode). */
	private _consumedCount = { p1: 0, p2: 0 };

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

	// Reveal + volatile trackers (structured obs modes) — see
	// ObservationTrackers. Snapshot-able for the BattleSim forward model.
	private _trackers = new ObservationTrackers();

	constructor(options: {
		seed?: PRNGSeed; format?: string; obsMode?: ObsMode; opponent?: OpponentKind;
		stallPenalty?: number;
	} = {}) {
		this._format = options.format ?? 'gen1randombattle';
		this._seed = options.seed;
		this._obsMode = options.obsMode ?? 'structured';
		this._opponent = options.opponent ?? 'random';
		this._stallPenalty = options.stallPenalty ?? DEFAULT_STALL_PENALTY;
	}

	// -------------------------------------------------------------------------
	// reset
	// -------------------------------------------------------------------------

	async reset(): Promise<Float32Array> {
		this.destroy();

		this._battleStream = new BattleStream();
		this._streams = getPlayerStreams(this._battleStream);

		this._gymPlayer = new GymPlayer(this._streams.p1);
		this._opponentPlayer = null;
		this._p2Player = null;
		if (this._opponent === 'self') {
			this._p2Player = new GymPlayer(this._streams.p2);
		} else if (this._opponent === 'damagefirst') {
			this._opponentPlayer = new DamageFirstAI(this._streams.p2, {
				seed: this._seed,
				format: this._format,
			});
		} else {
			this._opponentPlayer = new RandomPlayerAI(this._streams.p2, {
				seed: this._seed,
			});
		}
		this._consumedCount = { p1: 0, p2: 0 };

		// Reset background reader state
		this._omniscientLines = [];
		this._omniscientNotify = null;
		this._omniscientDone = false;
		this._winResult = null;
		this._winResolve = null;
		this._winPromise = null;

		// Reset reveal + volatile trackers
		this._trackers = new ObservationTrackers();

		// Start players (they loop reading their streams)
		void this._gymPlayer.start();
		if (this._p2Player) {
			void this._p2Player.start();
		} else {
			void this._opponentPlayer!.start();
		}

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

		return this._extractObs();
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
				obs: this._extractObs(),
				reward: -0.01,
				done: false,
				info: { illegalMove: true },
			};
		}

		// Check validActions mask
		const mask = this.validActions();
		if (!mask[action]) {
			return {
				obs: this._extractObs(),
				reward: -0.01,
				done: false,
				info: { illegalMove: true },
			};
		}

		// M5: capture the opponent's simultaneous committed choice (its own-frame
		// action label) BEFORE we submit p1's choice — submitting commits the
		// turn and clears the engine's choice buffers. The opponent (p2) has
		// already responded to this turn's request via its async loop.
		const oppAction = await this._captureOppLabel('p2');

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
		const parsed = this._parseProgressLines(newLines);
		let { reward, winner, done } = parsed;

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

		// Stalling penalty (applied on terminal). Clips before penalising so the
		// cost is symmetric across win/loss — see applyStallPenalty.
		if (done) {
			this._done = true;
			reward = applyStallPenalty(reward, this._turnCount, this._stallPenalty);
		} else {
			reward = Math.max(-1, Math.min(1, reward));
		}

		// Update current request if we got a real (non-null) one
		if (result.type === 'request' && result.request !== null) {
			this._currentRequest = result.request;
		}

		const obs = this._extractObs();

		return {
			obs,
			reward,
			done,
			info: {
				winner,
				turns: this._turnCount,
				oppAction,
			},
		};
	}

	// -------------------------------------------------------------------------
	// Dual-seat (self-play) API — requires opponent: 'self'
	// -------------------------------------------------------------------------

	/**
	 * Reset in dual-seat mode. Both seats' first requests arrive in the same
	 * engine batch; a microtask flush after the p1 request guarantees p2's has
	 * been delivered too.
	 */
	async resetDual(): Promise<DualStepResult> {
		if (this._opponent !== 'self') {
			throw new Error("resetDual() requires opponent: 'self'");
		}
		await this.reset();
		await new Promise<void>(resolve => setImmediate(resolve));
		return {
			p1: this._seatState('p1'),
			p2: this._seatState('p2'),
			reward: 0,
			done: false,
			info: { turns: this._turnCount },
		};
	}

	/**
	 * Advance the battle to the next decision point. The caller must pass an
	 * action for exactly the seats whose needsAction flag was true in the
	 * previous result (both on turn 1; only one during force-switches), and
	 * null for the others. Reward is from p1's perspective — self-play trains
	 * the p1 seat, the p2 seat runs a frozen opponent.
	 */
	async stepDual(p1Action: number | null, p2Action: number | null): Promise<DualStepResult> {
		if (!this._p2Player) {
			throw new Error("stepDual() requires opponent: 'self'");
		}
		if (this._done) {
			throw new Error('Battle already done; call resetDual()');
		}

		const p1Needs = this._needsAction('p1');
		const p2Needs = this._needsAction('p2');
		if (p1Needs !== (p1Action !== null) || p2Needs !== (p2Action !== null)) {
			throw new Error(
				`stepDual action/seat mismatch: needsAction=(${p1Needs},${p2Needs}) ` +
				`but got actions=(${p1Action},${p2Action})`
			);
		}

		// Validate both actions BEFORE submitting either, so an illegal action
		// never leaves one seat half-committed.
		const illegal =
			(p1Action !== null && (p1Action < 0 || p1Action > 8 || !this._validActionsFor('p1')[p1Action])) ||
			(p2Action !== null && (p2Action < 0 || p2Action > 8 || !this._validActionsFor('p2')[p2Action]));
		if (illegal) {
			return {
				p1: this._seatState('p1'),
				p2: this._seatState('p2'),
				reward: -0.01,
				done: false,
				info: { illegalMove: true, turns: this._turnCount },
			};
		}

		// M5: symmetric opponent-action labels. In dual-seat mode both seats'
		// choices are supplied by the caller already in each seat's own 9-way
		// frame, so a seat's label IS the other seat's simultaneous action. A
		// label is only defined when BOTH seats act this step (otherwise there
		// is no simultaneous opponent choice — e.g. a one-seat force-switch).
		const bothActed = p1Action !== null && p2Action !== null;
		const oppActionP1 = bothActed ? p2Action! : -1;
		const oppActionP2 = bothActed ? p1Action! : -1;

		let cursor = this._omniscientLines.length;
		let reward = 0;
		let done = false;
		let winner: string | undefined;

		// Register waiters before submitting so request events can't be missed
		let p1Wait = this._gymPlayer.waitForRequest();
		let p2Wait = this._p2Player.waitForRequest();

		if (p1Action !== null) {
			this._consumedCount.p1 = this._gymPlayer.requestCount;
			this._gymPlayer.submitChoice(actionToChoice(p1Action));
		}
		if (p2Action !== null) {
			this._consumedCount.p2 = this._p2Player.requestCount;
			this._p2Player.submitChoice(actionToChoice(p2Action));
		}

		type SeatEvent = { type: 'request', request: ChoiceRequest | null };
		type WinEvent = { type: 'win', winner: string };

		// Wait until the battle ends or at least one seat has a fresh
		// actionable request. Usually one race iteration; the loop guards
		// against wake-ups where the other seat's request hasn't landed yet.
		for (;;) {
			const result = await Promise.race<SeatEvent | WinEvent>([
				p1Wait.then(r => ({ type: 'request' as const, request: r })),
				p2Wait.then(r => ({ type: 'request' as const, request: r })),
				this._waitForWin().then(w => ({ type: 'win' as const, winner: w })),
			]);

			// Let the omniscient reader and any sibling request callbacks flush
			await new Promise<void>(resolve => setImmediate(resolve));

			const parsed = this._parseProgressLines(this._omniscientLines.slice(cursor));
			cursor = this._omniscientLines.length;
			reward += parsed.reward;
			if (parsed.done) {
				done = true;
				winner = parsed.winner;
			}
			if (!done && result.type === 'win') {
				done = true;
				winner = result.winner;
				reward += (winner === 'Gym') ? 1.0 : -1.0;
			}
			if (!done && this._winResult !== null) {
				done = true;
				winner = this._winResult;
				reward += (winner === 'Gym') ? 1.0 : -1.0;
			}
			// A null request means a player stream closed — battle over
			if (!done && result.type === 'request' && result.request === null) {
				done = true;
				winner = this._winResult ?? undefined;
			}

			if (done || this._needsAction('p1') || this._needsAction('p2')) break;

			// Spurious wake-up (e.g. stale waiter) — re-arm and wait again
			p1Wait = this._gymPlayer.waitForRequest();
			p2Wait = this._p2Player.waitForRequest();
		}

		if (done) {
			this._done = true;
			reward = applyStallPenalty(reward, this._turnCount, this._stallPenalty);
		} else {
			reward = Math.max(-1, Math.min(1, reward));
		}

		return {
			p1: this._seatState('p1'),
			p2: this._seatState('p2'),
			reward,
			done,
			info: { winner, turns: this._turnCount, oppAction: oppActionP1, oppActionP2 },
		};
	}

	private _needsAction(seat: 'p1' | 'p2'): boolean {
		const player = seat === 'p1' ? this._gymPlayer : this._p2Player;
		return !!player && player.requestCount > this._consumedCount[seat];
	}

	// -------------------------------------------------------------------------
	// M5: opponent-action labels
	// -------------------------------------------------------------------------

	/**
	 * True while the opponent seat has an actionable request this turn but has
	 * not yet committed a choice — i.e. its async player loop hasn't run yet.
	 * Used to bound the flush in `_captureOppLabel`.
	 */
	private _oppChoicePending(oppSeat: 'p1' | 'p2'): boolean {
		const battle = this._battleStream?.battle;
		const side = battle?.sides[oppSeat === 'p1' ? 0 : 1];
		if (!side) return false;
		const req = side.activeRequest as ChoiceRequest | null;
		if (!req || req.wait || req.teamPreview) return false;
		return !side.choice?.actions?.length;
	}

	/**
	 * Flush pending I/O so the opponent's async player loop can commit this
	 * turn's choice, then read its opponent-frame action label. Only yields
	 * when the choice is still pending, so the common (already-committed) case
	 * costs nothing.
	 */
	private async _captureOppLabel(oppSeat: 'p1' | 'p2'): Promise<number> {
		for (let i = 0; i < 3 && this._oppChoicePending(oppSeat); i++) {
			await new Promise<void>(resolve => setImmediate(resolve));
		}
		if (this._oppChoicePending(oppSeat)) {
			// The opponent's async choice still hasn't committed after 3 flushes —
			// an unresolved timing race, not a legitimate "no simultaneous choice"
			// case. Log it so label-coverage drops stay diagnosable instead of
			// silently blending into the masked (-1) bucket below.
			console.error(
				`[pokemon-gym] _captureOppLabel(${oppSeat}): opponent choice still ` +
				`pending after 3 flushes; masking label as -1`
			);
		}
		return this._opponentActionLabel(oppSeat);
	}

	/**
	 * Map the opponent seat's committed engine choice to its OWN 9-way gym
	 * action frame (omniscient — read straight off the battle engine):
	 *   • Move — the move's slot in the opponent's active request → 0–3.
	 *     A normal or paralysis-blocked player choice carries an explicit
	 *     `moveSlot`; an engine-auto-completed locked move (recharge / Wrap /
	 *     Thrash / Petal Dance) carries only a `moveid`, matched against the
	 *     request's move list (the M4 gotcha: these are labeled, not masked).
	 *   • Switch — the switched-in Pokémon's bench index → 4–8.
	 *   • -1 (masked) — no simultaneous choice (no request / wait / team
	 *     preview / empty choice), Struggle, a no-op sleep/freeze/trap turn
	 *     (empty engine action with no move), or any choice that doesn't map to
	 *     a slot (pass / revival / shift).
	 */
	private _opponentActionLabel(oppSeat: 'p1' | 'p2'): number {
		try {
			const battle = this._battleStream?.battle;
			const side = battle?.sides[oppSeat === 'p1' ? 0 : 1];
			if (!side) return -1;

			const req = side.activeRequest as ChoiceRequest | null;
			// No simultaneous opponent choice at this decision point.
			if (!req || req.wait || req.teamPreview) return -1;

			const action = side.choice?.actions?.[0];
			if (!action) return -1;

			if (action.choice === 'move') {
				// Explicit slot from a submitted (or paralysis-blocked) choice.
				if (typeof action.moveSlot === 'number' && action.moveSlot >= 0 && action.moveSlot <= 3) {
					return action.moveSlot;
				}
				// Auto-completed locked move: recover the slot from its move ID.
				// Struggle and no-op sleep/freeze/trap turns carry no usable ID.
				const moveid = action.moveid;
				if (moveid && moveid !== 'struggle' && isMoveRequest(req)) {
					const moves = req.active[0]?.moves ?? [];
					const idx = moves.findIndex(m => m.id === moveid);
					if (idx >= 0 && idx <= 3) return idx;
				}
				return -1;
			}

			if (action.choice === 'switch' || action.choice === 'instaswitch') {
				const target = action.target;
				if (!target) return -1;
				const idx = side.pokemon.indexOf(target);
				// Bench slots 1–5 map to action indices 4–8 (slot 0 is active).
				if (idx >= 1 && idx <= 5) return idx + 3;
				return -1;
			}

			// pass / revivalblessing / shift / team — no clean slot mapping.
			return -1;
		} catch (err) {
			// Genuinely unexpected (e.g. a malformed battle/request shape) — distinct
			// from the deliberate -1 returns above, which cover every known
			// no-simultaneous-choice case without needing to throw. Log so this
			// doesn't silently blend into legitimate masked labels.
			console.error(`[pokemon-gym] _opponentActionLabel(${oppSeat}) threw, masking label as -1:`, err);
			return -1;
		}
	}

	private _validActionsFor(seat: 'p1' | 'p2'): boolean[] {
		const player = seat === 'p1' ? this._gymPlayer : this._p2Player;
		return validActionsForRequest(player?._lastActionable ?? null);
	}

	private _seatState(seat: 'p1' | 'p2'): DualSeatState {
		const needsAction = !this._done && this._needsAction(seat);
		return {
			obs: this._extractObsFor(seat),
			mask: needsAction ? this._validActionsFor(seat) : new Array(9).fill(false),
			needsAction,
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
		return validActionsForRequest(this._currentRequest);
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

	private _extractObs(): Float32Array {
		if (this._obsMode === 'flat') {
			return extractFeatures(this._currentRequest!, null);
		}
		return extractFeaturesStructured(
			this._currentRequest!, this._getOpponentInfo('p1'), this._getVolatilesFor('p1'),
			this._getV3InfoFor('p1'),
		);
	}

	/** Obs for a specific seat (dual-seat self-play mode). */
	private _extractObsFor(seat: 'p1' | 'p2'): Float32Array {
		const player = seat === 'p1' ? this._gymPlayer : this._p2Player!;
		const request = player._lastActionable ?? player._currentRequest;
		if (!request) {
			// Stream closed before any actionable request — terminal filler
			const size = this._obsMode === 'flat' ? 100 :
				N_TOKENS * (
					this._obsMode === 'structured-v3-extended' ? TOKEN_DIM_V3_EXT :
					this._obsMode === 'structured-v3' ? TOKEN_DIM_V3 :
					this._obsMode === 'structured-v2' ? TOKEN_DIM_V2 : TOKEN_DIM
				);
			return new Float32Array(size);
		}
		if (this._obsMode === 'flat') {
			return extractFeatures(request, null);
		}
		return extractFeaturesStructured(
			request, this._getOpponentInfo(seat), this._getVolatilesFor(seat), this._getV3InfoFor(seat),
		);
	}

	/**
	 * The `viewer` seat's volatile view (own side + opponent side), or null in v1
	 * mode. Both v2 and v3 keep dims 65–76 (boosts/volatiles), so both need this.
	 */
	private _getVolatilesFor(viewer: 'p1' | 'p2'): { own: ActiveVolatiles, opp: ActiveVolatiles } | null {
		if (this._obsMode !== 'structured-v2' && this._obsMode !== 'structured-v3' &&
			this._obsMode !== 'structured-v3-extended') return null;
		return this._trackers.volatilesFor(viewer);
	}

	/**
	 * The `viewer` seat's v3-only info (dim 85 Sleep Clause flag), or undefined
	 * outside v3 mode so the extractor's v2/v1 code path is unchanged. Job 1.1's
	 * extractor derives dims 77–84 itself from `request`/`opponent`; only the
	 * Sleep Clause flag is log-tracked here and must be handed in.
	 */
	private _getV3InfoFor(viewer: 'p1' | 'p2'): V3Info | undefined {
		if (this._obsMode !== 'structured-v3' && this._obsMode !== 'structured-v3-extended') return undefined;
		return {
			sleepClause: this._trackers.sleepClauseFlagFor(viewer) === 1,
			extended: this._obsMode === 'structured-v3-extended',
		};
	}

	/** The `viewer` seat's revealed knowledge of the OTHER side's team. */
	private _getOpponentInfo(viewer: 'p1' | 'p2'): OpponentPokemonInfo[] {
		return this._trackers.opponentInfoFor(viewer);
	}

	/**
	 * Parse a batch of new omniscient lines into (reward, done, winner),
	 * updating the turn counter as a side effect. Reward is always from p1's
	 * perspective. Shared between step() and stepDual().
	 */
	private _parseProgressLines(lines: string[]): { reward: number, done: boolean, winner?: string } {
		const parsed = parseProgressLines(lines);
		if (parsed.lastTurn !== undefined) this._turnCount = parsed.lastTurn;
		return parsed;
	}

	/**
	 * Snapshot the live battle for the BattleSim forward model (M4 MCTS):
	 * serialized engine state + tracker state, deep-copied so the snapshot
	 * stays frozen while the real battle advances.
	 */
	snapshot(): GymSnapshot {
		const battle = this._battleStream?.battle;
		if (!battle || this._done) {
			throw new Error('snapshot() requires an active, unfinished battle');
		}
		return {
			battleState: JSON.parse(JSON.stringify(State.serializeBattle(battle))),
			trackers: this._trackers.snapshot(),
			obsMode: this._obsMode,
			turnCount: this._turnCount,
		};
	}

	private async _runOmniscientReader(): Promise<void> {
		try {
			for await (const chunk of this._streams.omniscient) {
				for (const line of chunk.split('\n')) {
					if (!line) continue;
					this._omniscientLines.push(line);
					this._trackers.processLine(line);

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

/** "p2a: Gengar" -> "Gengar". Falls back to the raw ident if unparseable. */
function extractNickname(ident: string): string {
	const colonIdx = ident.indexOf(': ');
	return colonIdx === -1 ? ident : ident.slice(colonIdx + 2);
}

/** "97/100 brn" -> "97/100"; "0 fnt" -> "0". Strips any status/fnt suffix. */
function hpFraction(condition: string): string {
	const spaceIdx = condition.indexOf(' ');
	return spaceIdx === -1 ? condition : condition.slice(0, spaceIdx);
}

/**
 * Parse a batch of battle-log lines (omniscient view) into progress info.
 * Reward is always from p1's ("Gym") perspective: ±0.01 per faint, +0.0001
 * per status inflicted on p2, ±1.0 on |win|. `lastTurn` is the highest
 * |turn| number seen (undefined if none). Shared by PokemonGymEnv and
 * BattleSim so live and simulated rewards stay identical.
 */
export function parseProgressLines(
	lines: string[],
): { reward: number, done: boolean, winner?: string, lastTurn?: number } {
	let reward = 0;
	let winner: string | undefined;
	let done = false;
	let lastTurn: number | undefined;

	for (const line of lines) {
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
			if (!isNaN(turnNum)) lastTurn = turnNum;
		}
	}
	return { reward, done, winner, lastTurn };
}

/**
 * Convert a 0-indexed action (0-8) to a Pokemon Showdown choice string.
 *   0-3  → "move 1" through "move 4"
 *   4-8  → "switch 2" through "switch 6"
 */
export function actionToChoice(action: number): string {
	if (action <= 3) {
		return `move ${action + 1}`;
	}
	// action 4 → switch 2 (bench slot 1 = pokemon index 1)
	return `switch ${action - 2}`;
}

/**
 * Compute the legal-action mask for a request:
 *   [move0, move1, move2, move3, switch1, switch2, switch3, switch4, switch5]
 */
export function validActionsForRequest(request: ChoiceRequest | null): boolean[] {
	const result: boolean[] = new Array(9).fill(false);

	if (!request || request.wait || request.teamPreview) {
		return result;
	}

	if (isMoveRequest(request)) {
		// Moves 0-3
		const activeSlot = request.active[0];
		if (activeSlot) {
			for (let i = 0; i < 4; i++) {
				const move = activeSlot.moves[i];
				if (move && !move.disabled) {
					result[i] = true;
				}
			}
		}

		// Switches 4-8 — blocked when the active Pokemon is trapped
		if (!activeSlot?.trapped) {
			const pokemon = request.side.pokemon;
			for (let slot = 1; slot <= 5; slot++) {
				const poke = pokemon[slot];
				if (!poke) continue;
				const isFainted = poke.condition.endsWith(' fnt') || poke.condition === '0 fnt';
				const isActive = poke.active === true;
				if (!isFainted && !isActive) {
					result[slot + 3] = true; // slot 1 -> index 4, slot 5 -> index 8
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
