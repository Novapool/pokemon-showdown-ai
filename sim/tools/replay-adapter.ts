/**
 * Replay adapter (M5.5) — converts a raw Pokemon Showdown battle log (the
 * spectator protocol lines stored by replay.pokemonshowdown.com and the
 * metamon-raw-replays dump) into per-decision (obs, action) training records
 * for BOTH seats of the battle.
 *
 * Observations are built with the exact same code the live gym uses —
 * ObservationTrackers (reveal + volatile tracking from public log lines,
 * including the Sleep Clause tracker) and extractFeaturesStructured — so
 * token conventions are identical to PokemonGymEnv by construction, not by
 * reimplementation. Defaults to schema v2 ((12, 77) tokens); construct with
 * `new ReplayAdapter('v3')` for schema v3 ((12, 86) tokens, M7) — see the
 * constructor doc below.
 *
 * Action grounding (the M2.5 invariant): replays carry no |request| JSON, so
 * request-slot order is unknowable. Instead, own move slots are written in
 * ALPHABETICAL move-id order and own bench tokens in alphabetical
 * (species id, nickname) order, and labels index those same orderings:
 *   action k   = the move in own-active move slot k        (k = 0..3)
 *   action 4+j = the Pokémon in own bench token j+1        (j = 0..4)
 * The policy learns the slot-content invariant, which holds identically in
 * the live gym (where slots follow request order) — proven to transfer by the
 * M2.5 BC result. Unmappable choices are skipped, mirroring metamon_adapter.
 *
 * Documented approximations (vs live-gym observations):
 *  - Own rosters/movesets are reconstructed from the FULL log (two-pass): the
 *    real player knew their team from turn 0, so whole-game knowledge is the
 *    faithful reconstruction. Mons/moves never revealed in the log stay
 *    unknown (unknown token / missing move slot).
 *  - Own-side PP ratio is approximated as (base PP - observed uses) / base PP
 *    (no PP Ups, no true request data); `disabled` is always 0.
 *  - Fully-paralyzed / asleep / trapped turns (|cant|) hide the player's
 *    actual choice — the decision is dropped.
 *  - Transform can push a moveset past 4; only the first 4 alphabetical move
 *    ids keep slots, later ones label as -1 (rare, gen1 Ditto only).
 *
 * @license MIT
 */

import { Dex } from '..';
import { toID } from '../dex-data';
import type { ChoiceRequest } from '../side';
import { ObservationTrackers } from './pokemon-gym';
import { extractFeaturesStructured } from './feature-extractor';

type Seat = 'p1' | 'p2';

export interface ReplayDecision {
	seat: Seat;
	/** Flattened (12, 77) schema-v2 observation, or (12, 86) schema-v3 when the
	 * adapter was constructed with `obsVersion: 'v3'`. */
	obs: Float32Array;
	/** 0–8 in the adapter's alphabetical action frame (see header). */
	action: number;
	/**
	 * The other seat's simultaneous same-turn action in ITS own frame
	 * (M5 opp-head label), -1 when none/unmappable.
	 */
	oppAction: number;
	/**
	 * What the label grounds to — the chosen move id (action 0–3) or the
	 * switch target's species id (action 4–8). For tests/debugging: lets the
	 * label↔obs-slot invariant be verified against dex data.
	 */
	resolved: string;
	turn: number;
	/** True for the trajectory's final decision. */
	done: boolean;
}

export interface ReplayTrajectory {
	seat: Seat;
	/** 1 = this seat won, 0 = lost. */
	win: number;
	decisions: ReplayDecision[];
}

export interface ParsedReplay {
	p1: ReplayTrajectory;
	p2: ReplayTrajectory;
	winner: Seat;
	/** Decisions dropped because the action couldn't be mapped (label -1). */
	skipped: number;
}

interface RosterEntry {
	nickname: string;
	speciesId: string;
	details: string;
	/** All move ids this mon revealed across the whole log (chosen moves only). */
	moves: Set<string>;
}

/** A decision point awaiting its resolving action line. */
interface PendingDecision {
	obs: Float32Array;
	/** Alphabetical move ids in the own-active token's slots at snapshot time. */
	moveIds: string[];
	/** Bench nicknames in own bench-token order at snapshot time. */
	benchNicks: string[];
	turn: number;
}

function nicknameOf(ident: string): string {
	// '|switch|p1a: Snorlax|...' -> ident 'p1a: Snorlax' -> 'Snorlax'
	return ident.slice(ident.indexOf(':') + 1).trim();
}

function seatOf(ident: string): Seat | null {
	if (ident.startsWith('p1')) return 'p1';
	if (ident.startsWith('p2')) return 'p2';
	return null;
}

/** True for |move| lines that are engine-called continuations/copies
 * (Wrap follow-ups, Metronome/Mirror Move calls) rather than choices. */
function isCalledMove(parts: string[]): boolean {
	return parts.some(p => p.startsWith('[from]'));
}

/** Selects the obs schema the adapter emits (see `ReplayAdapter`'s constructor). */
export type ReplayObsVersion = 'v2' | 'v3';

export class ReplayAdapter {
	private readonly _dex = Dex.mod('gen1');
	private readonly _obsVersion: ReplayObsVersion;

	/**
	 * @param obsVersion 'v2' (default) emits the existing (12, 77) schema.
	 *   'v3' emits (12, 86): dims 0–76 are byte-identical to v2 (same
	 *   `extractFeaturesStructured` prefix), with dims 77–85 (type-eff,
	 *   move-effect flags, inflicted-status, Sleep Clause) appended per Job
	 *   1.1/1.2. The Sleep Clause flag is read from the same
	 *   `ObservationTrackers` instance used for reveal/volatile tracking, so
	 *   it reflects the adapter's own two-pass replay of the log rather than
	 *   any live-gym state.
	 */
	constructor(obsVersion: ReplayObsVersion = 'v2') {
		this._obsVersion = obsVersion;
	}

	/**
	 * Parse one raw battle log into two per-seat trajectories.
	 * Returns null for unusable logs (no winner, no battle content).
	 */
	parse(log: string): ParsedReplay | null {
		const lines = log.split('\n');

		// ---- Pass 1: full-game rosters + movesets + winner ------------------
		const roster: Record<Seat, Map<string, RosterEntry>> = { p1: new Map(), p2: new Map() };
		const playerNames: Record<string, Seat> = {};
		let winner: Seat | null = null;

		for (const line of lines) {
			const parts = line.split('|');
			const type = parts[1];
			if (type === 'player' && parts[2] && parts[3]) {
				playerNames[parts[3]] = parts[2] as Seat;
			} else if (type === 'switch' || type === 'drag') {
				const seat = seatOf(parts[2] ?? '');
				if (!seat) continue;
				const nickname = nicknameOf(parts[2]);
				if (!roster[seat].has(nickname)) {
					const details = parts[3] ?? '';
					roster[seat].set(nickname, {
						nickname,
						speciesId: toID(details.split(',')[0]),
						details,
						moves: new Set(),
					});
				}
			} else if (type === 'move' && !isCalledMove(parts)) {
				const seat = seatOf(parts[2] ?? '');
				if (!seat) continue;
				const entry = roster[seat].get(nicknameOf(parts[2]));
				const moveId = toID(parts[3] ?? '');
				if (entry && moveId && moveId !== 'struggle') entry.moves.add(moveId);
			} else if (type === 'win') {
				winner = playerNames[parts[2] ?? ''] ?? null;
			}
		}

		if (!winner || !roster.p1.size || !roster.p2.size) return null;

		// ---- Pass 2: replay through the live-gym trackers -------------------
		const trackers = new ObservationTrackers();
		const active: Record<Seat, string | null> = { p1: null, p2: null };
		const ppUsed: Record<Seat, Map<string, number>> = { p1: new Map(), p2: new Map() };
		const pendingTurn: Record<Seat, PendingDecision | null> = { p1: null, p2: null };
		const pendingFaint: Record<Seat, PendingDecision | null> = { p1: null, p2: null };
		// Per-turn labels for the oppAction cross-link (index into decisions[]).
		const turnDecisionIdx: Record<Seat, number> = { p1: -1, p2: -1 };
		const decisions: Record<Seat, ReplayDecision[]> = { p1: [], p2: [] };
		let skipped = 0;
		let turn = 0;

		const emit = (seat: Seat, pending: PendingDecision, action: number, isTurnDecision: boolean, resolved: string) => {
			if (action < 0) {
				skipped++;
				if (isTurnDecision) turnDecisionIdx[seat] = -1;
				return;
			}
			decisions[seat].push({
				seat, obs: pending.obs, action, oppAction: -1, resolved, turn: pending.turn, done: false,
			});
			if (isTurnDecision) turnDecisionIdx[seat] = decisions[seat].length - 1;
		};

		const crossLinkTurn = () => {
			// Both seats acted this turn: each seat's opp-head label is the other
			// seat's own-frame action (exactly the M5 dual-seat convention).
			const i1 = turnDecisionIdx.p1;
			const i2 = turnDecisionIdx.p2;
			if (i1 >= 0 && i2 >= 0) {
				decisions.p1[i1].oppAction = decisions.p2[i2].action;
				decisions.p2[i2].oppAction = decisions.p1[i1].action;
			}
			turnDecisionIdx.p1 = turnDecisionIdx.p2 = -1;
		};

		for (const line of lines) {
			const parts = line.split('|');
			const type = parts[1];

			if (type === 'turn') {
				crossLinkTurn();
				turn = parseInt(parts[2] ?? '0', 10) || turn + 1;
				// Both players face a fresh decision at turn start; snapshot the
				// state each of them sees BEFORE any of this turn's lines resolve.
				for (const seat of ['p1', 'p2'] as Seat[]) {
					pendingTurn[seat] = this._snapshot(seat, roster, active, ppUsed, trackers, turn, false);
					pendingFaint[seat] = null;
				}
				continue;
			}

			if (type === 'move' && !isCalledMove(parts)) {
				const seat = seatOf(parts[2] ?? '');
				if (seat) {
					const nickname = nicknameOf(parts[2]);
					const moveId = toID(parts[3] ?? '');
					const pending = pendingTurn[seat];
					if (pending) {
						pendingTurn[seat] = null;
						emit(seat, pending, pending.moveIds.indexOf(moveId), true, moveId);
					}
					const key = `${nickname}|${moveId}`;
					ppUsed[seat].set(key, (ppUsed[seat].get(key) ?? 0) + 1);
				}
			} else if (type === 'switch' || type === 'drag') {
				const seat = seatOf(parts[2] ?? '');
				if (seat) {
					const nickname = nicknameOf(parts[2]);
					if (type === 'switch') {
						const pending = pendingTurn[seat] ?? pendingFaint[seat];
						if (pending) {
							const isTurnDecision = pendingTurn[seat] !== null;
							pendingTurn[seat] = null;
							pendingFaint[seat] = null;
							const benchIdx = pending.benchNicks.indexOf(nickname);
							emit(seat, pending, benchIdx >= 0 ? 4 + benchIdx : -1, isTurnDecision,
								toID((parts[3] ?? '').split(',')[0]));
						}
					}
					active[seat] = nickname;
				}
			} else if (type === 'cant') {
				const seat = seatOf(parts[2] ?? '');
				// The player may well have made a choice (sleep/para/trap hides
				// it) — unknowable, so the decision is dropped.
				if (seat && pendingTurn[seat]) {
					pendingTurn[seat] = null;
					skipped++;
				}
			}

			// Trackers consume every line AFTER the decision logic: a decision's
			// obs reflects the state before its own action line resolves.
			trackers.processLine(line);

			if (type === 'faint') {
				const seat = seatOf(parts[2] ?? '');
				if (seat) {
					active[seat] = null;
					// Snapshot after the faint is processed — the replacement
					// choice is made seeing the fainted mon's aftermath.
					pendingFaint[seat] = this._snapshot(seat, roster, active, ppUsed, trackers, turn, true, nicknameOf(parts[2]));
				}
			}
		}
		crossLinkTurn();

		for (const seat of ['p1', 'p2'] as Seat[]) {
			const traj = decisions[seat];
			if (traj.length) traj[traj.length - 1].done = true;
		}

		return {
			p1: { seat: 'p1', win: winner === 'p1' ? 1 : 0, decisions: decisions.p1 },
			p2: { seat: 'p2', win: winner === 'p2' ? 1 : 0, decisions: decisions.p2 },
			winner,
			skipped,
		};
	}

	/**
	 * Build seat `seat`'s observation at the current tracker state, plus the
	 * move/bench orderings its action labels index into.
	 */
	private _snapshot(
		seat: Seat,
		roster: Record<Seat, Map<string, RosterEntry>>,
		active: Record<Seat, string | null>,
		ppUsed: Record<Seat, Map<string, number>>,
		trackers: ObservationTrackers,
		turn: number,
		forceSwitch: boolean,
		faintedNick?: string,
	): PendingDecision | null {
		const own = roster[seat];
		const activeNick = active[seat] ?? faintedNick;
		if (!activeNick || !own.has(activeNick)) return null;

		const records = trackers.revealRecords[seat];
		const conditionOf = (nickname: string): string => {
			const record = records.get(nickname);
			// Own mons not yet revealed in the log are untouched: full HP. The
			// live player would see exact values; 100/100 is ratio-identical.
			return record ? record.condition : '100/100';
		};

		const sortedMoves = (entry: RosterEntry): string[] =>
			Array.from(entry.moves).sort().slice(0, 4);

		const benchEntries = Array.from(own.values())
			.filter(e => e.nickname !== activeNick)
			.sort((a, b) => a.speciesId.localeCompare(b.speciesId) || a.nickname.localeCompare(b.nickname));

		const activeEntry = own.get(activeNick)!;
		const activeMoveIds = sortedMoves(activeEntry);

		const pokemonData = [activeEntry, ...benchEntries].map(entry => ({
			details: entry.details,
			condition: forceSwitch && entry.nickname === activeNick ? '0 fnt' : conditionOf(entry.nickname),
			moves: sortedMoves(entry),
		}));

		let request: AnyObject;
		if (forceSwitch) {
			request = { forceSwitch: [true], side: { pokemon: pokemonData } };
		} else {
			const moves = activeMoveIds.map(id => {
				const maxpp = this._dex.moves.get(id).pp || 1;
				const used = ppUsed[seat].get(`${activeNick}|${id}`) ?? 0;
				return { id, pp: Math.max(0, maxpp - used), maxpp };
			});
			request = { active: [{ moves }], side: { pokemon: pokemonData } };
		}

		const obs = extractFeaturesStructured(
			request as ChoiceRequest,
			trackers.opponentInfoFor(seat),
			trackers.volatilesFor(seat),
			this._obsVersion === 'v3' ? { sleepClause: trackers.sleepClauseFlagFor(seat) === 1 } : null,
		);

		return {
			obs,
			moveIds: activeMoveIds,
			benchNicks: benchEntries.map(e => e.nickname),
			turn,
		};
	}
}
