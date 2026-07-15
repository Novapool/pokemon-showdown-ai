/**
 * BattleSim — forward model for MCTS (M4)
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Wraps a deserialized clone of a live gym battle so search can step
 * hypothetical action sequences without touching the real battle. Built from
 * a PokemonGymEnv.snapshot(): engine state (State.serializeBattle) plus the
 * gym's log-derived tracker state, so simulated observations, masks, and
 * rewards reproduce the gym's semantics exactly.
 *
 * Imperfect information is handled by determinization: fromSnapshot() can
 * replace the opponent's *unrevealed* Pokémon with sets sampled from the
 * format's random-team generator. Revealed Pokémon keep their true engine
 * state. Known approximation (documented in MILESTONES.md → M4): revealed
 * Pokémon keep their full true movesets, not just the revealed moves.
 *
 * @license MIT
 */

import { Battle, extractChannelMessages } from '../battle';
import { Pokemon } from '../pokemon';
import { PRNG } from '../prng';
import type { PRNGSeed } from '../prng';
import { State } from '../state';
import { Teams } from '../teams';
import { toID } from '../dex-data';
import type { ChoiceRequest } from '../side';
import type { PokemonSet } from '../teams';
import {
	ObservationTrackers, actionToChoice, parseProgressLines, validActionsForRequest,
} from './pokemon-gym';
import type { GymSnapshot, ObsMode } from './pokemon-gym';
import {
	extractFeatures, extractFeaturesStructured, N_TOKENS, TOKEN_DIM, TOKEN_DIM_V2,
} from './feature-extractor';

export interface SimSeatState {
	obs: Float32Array;
	mask: boolean[];
	/** True if this seat must supply an action on the next step() call. */
	needsAction: boolean;
}

export interface SimStepResult {
	p1: SimSeatState;
	p2: SimSeatState;
	/** Reward from p1's perspective, same shaping/clipping as the gym. */
	reward: number;
	done: boolean;
	info: { winner?: string, turns?: number, illegalMove?: boolean };
}

export interface SimOptions {
	/**
	 * Replace the searcher's OPPONENT's unrevealed Pokémon with sets sampled
	 * from the format's random-team generator (fair imperfect-information
	 * search). When false, the sim keeps the true hidden team (omniscient /
	 * cheating search — useful as an upper-bound diagnostic only).
	 */
	determinize?: boolean;
	/**
	 * Which seat the searcher occupies (default 'p1'). Determinization
	 * resamples the OTHER seat's unrevealed Pokémon.
	 */
	perspective?: 'p1' | 'p2';
	/**
	 * Reseed the cloned battle's RNG (and the determinizer's sampler) so
	 * different clones explore different chance outcomes. Any integer;
	 * omitted → nondeterministic fresh seed.
	 */
	seed?: number;
}

/** Deterministic gen5-style PRNG seed from an arbitrary integer. */
function seedFromInt(n: number): PRNGSeed {
	// Mix so consecutive ints don't produce correlated LCG streams.
	const h = (k: number) => {
		let x = (n + k * 0x9e3779b9) >>> 0;
		x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0;
		x = Math.imul(x ^ (x >>> 16), 0x45d9f3b) >>> 0;
		return ((x ^ (x >>> 16)) % 0xffff) >>> 0;
	};
	return `${h(1)},${h(2)},${h(3)},${h(4)}` as PRNGSeed;
}

export class BattleSim {
	private readonly _battle: Battle;
	private readonly _trackers: ObservationTrackers;
	private readonly _obsMode: ObsMode;
	private _logCursor: number;
	private _turnCount: number;
	private _done = false;
	private _winner: string | undefined;
	/** Last actionable (non-wait) request per seat — obs source at wait/terminal points. */
	private _lastActionable: { p1: ChoiceRequest | null, p2: ChoiceRequest | null } = { p1: null, p2: null };

	private constructor(battle: Battle, trackers: ObservationTrackers, obsMode: ObsMode, turnCount: number) {
		this._battle = battle;
		this._trackers = trackers;
		this._obsMode = obsMode;
		this._turnCount = turnCount;
		this._logCursor = battle.log.length;
		this._done = battle.ended;
		this._winner = battle.winner || undefined;
		this._syncLastActionable();
	}

	/**
	 * Build a sim from a gym snapshot. The snapshot is deep-copied, so one
	 * snapshot can seed many independent sims (one per determinization).
	 */
	static fromSnapshot(snap: GymSnapshot, options: SimOptions = {}): BattleSim {
		const state = JSON.parse(JSON.stringify(snap.battleState));
		const battle = State.deserializeBattle(state);
		const sim = new BattleSim(
			battle, ObservationTrackers.fromSnapshot(snap.trackers), snap.obsMode, snap.turnCount,
		);
		const seed = options.seed !== undefined ? seedFromInt(options.seed) : PRNG.generateSeed();
		battle.prng = new PRNG(seed);
		if (options.determinize) {
			const hiddenSide = (options.perspective ?? 'p1') === 'p1' ? 'p2' : 'p1';
			sim._determinize(
				hiddenSide,
				new PRNG(options.seed !== undefined ? seedFromInt(options.seed + 0x5f356495) : null),
			);
		}
		return sim;
	}

	/**
	 * Independent copy of this sim's current state (tree branching). The
	 * child continues from the parent's exact engine + tracker + RNG state,
	 * so identical action sequences give identical outcomes.
	 */
	fork(): BattleSim {
		const snap: GymSnapshot = {
			battleState: State.serializeBattle(this._battle),
			trackers: this._trackers.snapshot(),
			obsMode: this._obsMode,
			turnCount: this._turnCount,
		};
		// No JSON round-trip needed on battleState here: fromSnapshot deep-copies.
		const sim = BattleSim.fromSnapshot(snap);
		// fromSnapshot reseeded the RNG; restore the parent's stream instead.
		sim._battle.prng = new PRNG(this._battle.prng.getSeed());
		return sim;
	}

	get done(): boolean {
		return this._done;
	}

	get winner(): string | undefined {
		return this._winner;
	}

	get turn(): number {
		return this._turnCount;
	}

	needsAction(seat: 'p1' | 'p2'): boolean {
		if (this._done) return false;
		const side = this._battle.sides[seat === 'p1' ? 0 : 1];
		// A seat needs input only if the engine is actually waiting on its
		// choice. An actionable-looking request is NOT enough: locked states
		// (sleep, recharge, multi-turn moves) auto-complete the side's choice
		// (side.isChoiceDone() auto-passes), and submitting for such a seat
		// would land one decision point ahead and desync the battle.
		const request = side.activeRequest;
		if (!request || (request as AnyObject).wait) return false;
		return !side.isChoiceDone();
	}

	seatState(seat: 'p1' | 'p2'): SimSeatState {
		const needsAction = this.needsAction(seat);
		return {
			obs: this._extractObsFor(seat),
			mask: needsAction ? this._validActionsFor(seat) : new Array(9).fill(false),
			needsAction,
		};
	}

	state(): SimStepResult {
		return {
			p1: this.seatState('p1'),
			p2: this.seatState('p2'),
			reward: 0,
			done: this._done,
			info: { winner: this._winner, turns: this._turnCount },
		};
	}

	/**
	 * Advance the battle to the next decision point. Mirrors the gym's
	 * stepDual(): pass an action for exactly the seats whose needsAction was
	 * true, null for the others. Reward is from p1's perspective with the
	 * gym's exact shaping (faints, status, win/loss, stalling penalty, clip).
	 */
	step(p1Action: number | null, p2Action: number | null): SimStepResult {
		if (this._done) {
			throw new Error('BattleSim: battle already done');
		}
		const p1Needs = this.needsAction('p1');
		const p2Needs = this.needsAction('p2');
		if (!p1Needs && !p2Needs) {
			// The engine guarantees at least one pending choice per phase
			// (makeRequest asserts it) — reaching here means a desync bug.
			throw new Error('BattleSim.step: no seat needs an action but the battle is not over');
		}
		if (p1Needs !== (p1Action !== null) || p2Needs !== (p2Action !== null)) {
			throw new Error(
				`BattleSim.step action/seat mismatch: needsAction=(${p1Needs},${p2Needs}) ` +
				`but got actions=(${p1Action},${p2Action})`
			);
		}

		// Validate both actions BEFORE submitting either (mirrors stepDual).
		const illegal =
			(p1Action !== null && (p1Action < 0 || p1Action > 8 || !this._validActionsFor('p1')[p1Action])) ||
			(p2Action !== null && (p2Action < 0 || p2Action > 8 || !this._validActionsFor('p2')[p2Action]));
		if (illegal) {
			const result = this.state();
			result.reward = -0.01;
			result.info.illegalMove = true;
			return result;
		}

		if (p1Action !== null && !this._battle.choose('p1', actionToChoice(p1Action))) {
			const result = this.state();
			result.reward = -0.01;
			result.info.illegalMove = true;
			return result;
		}
		if (p2Action !== null && !this._battle.choose('p2', actionToChoice(p2Action))) {
			const result = this.state();
			result.reward = -0.01;
			result.info.illegalMove = true;
			return result;
		}

		// The engine ran synchronously; consume new log lines the same way the
		// gym's omniscient reader does (channel -1 = secret/full view).
		const newLog = this._battle.log.slice(this._logCursor);
		this._logCursor = this._battle.log.length;
		const lines = newLog.length ? extractChannelMessages(newLog.join('\n'), [-1])[-1] : [];
		for (const line of lines) this._trackers.processLine(line);

		const parsed = parseProgressLines(lines);
		let reward = parsed.reward;
		let winner = parsed.winner;
		if (parsed.lastTurn !== undefined) this._turnCount = parsed.lastTurn;
		let done = parsed.done;

		// battle.ended is authoritative (covers ties/forced ends without a
		// parsed |win| credit — same as the gym's null-request fallback).
		if (!done && this._battle.ended) {
			done = true;
			winner = this._battle.winner || undefined;
		}

		if (done) {
			reward -= 0.001 * this._turnCount;
			this._done = true;
			this._winner = winner;
		}
		reward = Math.max(-1, Math.min(1, reward));

		this._syncLastActionable();

		return {
			p1: this.seatState('p1'),
			p2: this.seatState('p2'),
			reward,
			done,
			info: { winner, turns: this._turnCount },
		};
	}

	// -------------------------------------------------------------------------
	// Internals
	// -------------------------------------------------------------------------

	private _syncLastActionable(): void {
		for (const seat of ['p1', 'p2'] as const) {
			const request = this._battle.sides[seat === 'p1' ? 0 : 1].activeRequest;
			if (request && !(request as AnyObject).wait) {
				this._lastActionable[seat] = request as ChoiceRequest;
			}
		}
	}

	private _validActionsFor(seat: 'p1' | 'p2'): boolean[] {
		const request = this._battle.sides[seat === 'p1' ? 0 : 1].activeRequest;
		return validActionsForRequest((request as ChoiceRequest | null) ?? null);
	}

	private _extractObsFor(seat: 'p1' | 'p2'): Float32Array {
		const side = this._battle.sides[seat === 'p1' ? 0 : 1];
		const request = this._lastActionable[seat] ?? (side.activeRequest as ChoiceRequest | null);
		if (!request) {
			const size = this._obsMode === 'flat' ? 100 :
				N_TOKENS * (this._obsMode === 'structured-v2' ? TOKEN_DIM_V2 : TOKEN_DIM);
			return new Float32Array(size);
		}
		if (this._obsMode === 'flat') {
			return extractFeatures(request, null);
		}
		const volatiles = this._obsMode === 'structured-v2' ? this._trackers.volatilesFor(seat) : null;
		return extractFeaturesStructured(request, this._trackers.opponentInfoFor(seat), volatiles);
	}

	/**
	 * Replace `hiddenSide`'s unrevealed Pokémon with sets sampled from the
	 * format's random-team generator. Unrevealed mons have never been on the
	 * field (gen1 has no entry hazards), so they carry no battle state —
	 * swapping the Pokemon object wholesale is safe. Revealed mons are untouched.
	 */
	private _determinize(hiddenSide: 'p1' | 'p2', samplePrng: PRNG): void {
		const side = this._battle.sides[hiddenSide === 'p1' ? 0 : 1];
		const revealed = this._trackers.revealedNicknames(hiddenSide);
		const targets: number[] = [];
		for (const [i, pokemon] of side.pokemon.entries()) {
			if (!revealed.has(pokemon.name)) targets.push(i);
		}
		if (!targets.length) return;

		// Species already on the team (kept mons) — species clause.
		const keepSpecies = new Set<string>();
		for (const [i, pokemon] of side.pokemon.entries()) {
			if (!targets.includes(i)) keepSpecies.add(toID(pokemon.species.name));
		}

		// Sample replacement sets from the format's own generator.
		const formatid = this._battle.format.id;
		const sampled: PokemonSet[] = [];
		const sampledSpecies = new Set<string>();
		let guard = 20;
		while (sampled.length < targets.length && guard-- > 0) {
			const team = Teams.getGenerator(formatid, samplePrng).getTeam();
			for (const set of team) {
				if (sampled.length >= targets.length) break;
				const speciesId = toID(set.species);
				if (keepSpecies.has(speciesId) || sampledSpecies.has(speciesId)) continue;
				sampled.push(set);
				sampledSpecies.add(speciesId);
			}
		}
		if (sampled.length < targets.length) {
			throw new Error(`BattleSim determinizer: could not sample ${targets.length} replacement sets`);
		}

		for (const [k, idx] of targets.entries()) {
			const old = side.pokemon[idx];
			const replacement = new Pokemon(sampled[k], side);
			replacement.position = idx;
			side.pokemon[idx] = replacement;
			// Keep side.team consistent so this sim can itself be serialized (fork()).
			const teamIdx = side.team.indexOf(old.set);
			if (teamIdx >= 0) side.team[teamIdx] = replacement.set;
		}

		// Bench composition changed — regenerate active requests so the hidden
		// side's request data (and switch targets) reflect the sampled bench.
		// Mirrors what State.deserializeBattle does, preserving null tombstones.
		if (!this._battle.ended && this._battle.requestState) {
			const requests = this._battle.getRequests(this._battle.requestState);
			for (const [i, s] of this._battle.sides.entries()) {
				if (s.activeRequest !== null) s.activeRequest = requests[i];
			}
		}
		this._syncLastActionable();
	}
}
