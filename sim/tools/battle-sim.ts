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
import type { GymSnapshot, ObsMode, TrackerSnapshot } from './pokemon-gym';
import {
	extractFeatures, extractFeaturesStructured, N_TOKENS, TOKEN_DIM, TOKEN_DIM_V2, TOKEN_DIM_V3,
	TOKEN_DIM_V3_EXT,
	parseHpRatio, parseStatus, parseLevelFromDetails, BOOST_ORDER,
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

/**
 * Everything a remote (ladder) client knows about its battle — enough to
 * reconstruct a local engine battle for search (M6 P2). Unlike GymSnapshot
 * there is no engine state: the battle lives on the server.
 */
export interface TrackedBattleInput {
	formatid: string;
	/** The searcher's seat in the remote battle. */
	seat: 'p1' | 'p2';
	/** The searcher's latest actionable MOVE request (not a force-switch). */
	request: ChoiceRequest;
	/** The client's tracker snapshot (reveal + volatiles, both sides). */
	trackers: TrackerSnapshot;
	turnCount: number;
	obsMode?: ObsMode;
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
	 * Build a sim by RECONSTRUCTING a battle the client only observes remotely
	 * (M6 P2 — search on the ladder, where no local engine battle exists).
	 *
	 * The searcher's own side is rebuilt from its request JSON (true species/
	 * levels/movesets/conditions); the opponent's side from the tracker's
	 * revealed records, with generator-sampled sets filling unrevealed slots
	 * and generator movesets backfilling revealed mons' unseen moves. Tracked
	 * state (HP ratios, statuses, faints, boosts, screens/Sub/Leech Seed,
	 * toxic counter) is then patched onto the fresh battle.
	 *
	 * Documented approximations, beyond M4's determinization ones:
	 *  - Stats/EVs come from the format generator's conventions, so absolute
	 *    HP can differ from the remote battle; HP *ratios* are preserved,
	 *    which is all the obs schema encodes.
	 *  - Hidden counters are defaulted (sleep turns ~ mid-range, Substitute
	 *    at full sub HP); PP is full except the searcher's active (request).
	 *  - Locked states (Wrap/Hyper Beam recharge/multi-turn moves) are NOT
	 *    reproduced — callers should answer locked/force-switch requests with
	 *    the raw policy and only search clean move requests (`request` must
	 *    be a move request; anything else throws).
	 */
	static fromTracked(input: TrackedBattleInput, options: SimOptions = {}): BattleSim {
		const request = input.request as AnyObject;
		if (!request || request.wait || request.forceSwitch || !request.active) {
			throw new Error('BattleSim.fromTracked: `request` must be an actionable move request');
		}
		const obsMode = input.obsMode ?? 'structured-v2';
		const trackers = ObservationTrackers.fromSnapshot(input.trackers);
		const mySide = input.seat;
		const oppSide = mySide === 'p1' ? 'p2' : 'p1';
		const seed = options.seed !== undefined ? seedFromInt(options.seed) : PRNG.generateSeed();
		const samplePrng = new PRNG(
			options.seed !== undefined ? seedFromInt(options.seed + 0x5f356495) : null);
		const generator = Teams.getGenerator(input.formatid, samplePrng) as AnyObject;

		const setFor = (species: string, level: number, name: string, moves?: string[]): PokemonSet => {
			const sampled = generator.randomSet(species);
			const set = { ...sampled, name, level } as unknown as PokemonSet;
			if (moves?.length) {
				const merged = [...moves];
				for (const move of sampled.moves as string[]) {
					if (merged.length >= 4) break;
					if (!merged.includes(move)) merged.push(move);
				}
				set.moves = merged.slice(0, 4);
			}
			return set;
		};

		// --- Own team: true species/levels/movesets from the request ---------
		const ownConditions = new Map<string, string>();
		const ownTeam: PokemonSet[] = (request.side.pokemon as AnyObject[]).map(poke => {
			const nickname = String(poke.ident).slice(String(poke.ident).indexOf(':') + 1).trim();
			const species = String(poke.details).split(',')[0].trim();
			const level = parseLevelFromDetails(poke.details);
			ownConditions.set(nickname, poke.condition);
			return setFor(species, level, nickname, (poke.moves as string[]).slice(0, 4));
		});

		// --- Opponent team: revealed records + sampled unrevealed slots ------
		const oppRecords = trackers.revealRecords[oppSide];
		const oppOrder = trackers.revealOrder[oppSide];
		const oppTeam: PokemonSet[] = [];
		const usedSpecies = new Set<string>();
		// Active revealed mon first, so the initial switch-in matches the field.
		const orderedNicks = [...oppOrder].sort((a, b) =>
			Number(oppRecords.get(b)!.active) - Number(oppRecords.get(a)!.active));
		for (const nickname of orderedNicks) {
			const record = oppRecords.get(nickname)!;
			const species = record.details.split(',')[0].trim();
			oppTeam.push(setFor(species, parseLevelFromDetails(record.details), nickname, record.moves));
			usedSpecies.add(toID(species));
		}
		let guard = 20;
		while (oppTeam.length < 6 && guard-- > 0) {
			for (const set of generator.getTeam() as PokemonSet[]) {
				if (oppTeam.length >= 6) break;
				if (usedSpecies.has(toID(set.species))) continue;
				oppTeam.push(set);
				usedSpecies.add(toID(set.species));
			}
		}
		if (oppTeam.length < 6) {
			throw new Error('BattleSim.fromTracked: could not fill the opponent team');
		}

		// --- Fresh local battle (initial switch-ins fire immediately) --------
		const battle = new Battle({
			formatid: input.formatid as any,
			seed,
			[mySide]: { name: 'searcher', team: ownTeam },
			[oppSide]: { name: 'opponent', team: oppTeam },
		} as any);

		// --- Patch tracked per-Pokémon state ---------------------------------
		const conditionFor = (seat: 'p1' | 'p2', name: string): string | undefined => {
			if (seat === mySide) return ownConditions.get(name);
			return oppRecords.get(name)?.condition;
		};
		for (const seat of ['p1', 'p2'] as const) {
			const side = battle.sides[seat === 'p1' ? 0 : 1];
			for (const pokemon of side.pokemon) {
				const condition = conditionFor(seat, pokemon.set.name);
				if (condition === undefined) continue; // unrevealed: untouched
				const ratio = parseHpRatio(condition);
				if (ratio <= 0) {
					pokemon.hp = 0;
					pokemon.fainted = true;
					pokemon.status = '' as any;
					side.pokemonLeft--;
					continue;
				}
				pokemon.hp = Math.max(1, Math.round(ratio * pokemon.maxhp));
				const status = parseStatus(condition);
				if (status) {
					pokemon.status = status as any;
					// Mid-range residual sleep counter; exact remote value is hidden.
					pokemon.statusState = { id: status, target: pokemon, ...(status === 'slp' ? { time: 3 } : {}) } as any;
				}
			}
			// Active-only volatile state (everything resets on switch in gen1).
			const active = side.active[0];
			const vol = trackers.volatiles[seat];
			if (active && !active.fainted && vol) {
				for (const stat of BOOST_ORDER) {
					const stage = vol.boosts[stat] ?? 0;
					if (stage && stat in active.boosts) (active.boosts as AnyObject)[stat] = stage;
				}
				if (vol.reflect) active.addVolatile('reflect');
				if (vol.lightScreen) active.addVolatile('lightscreen');
				if (vol.leechSeed) active.addVolatile('leechseed');
				if (vol.substitute) {
					active.addVolatile('substitute');
					if (active.volatiles['substitute']) {
						active.volatiles['substitute'].hp = Math.floor(active.maxhp / 4);
					}
				}
				if (vol.toxicCounter > 0 && active.status === 'tox' as any) {
					active.addVolatile('residualdmg');
					if (active.volatiles['residualdmg']) {
						active.volatiles['residualdmg'].counter = vol.toxicCounter;
					}
				}
			}
		}

		battle.turn = input.turnCount;

		// Bench/HP/status changed under the engine — regenerate both requests
		// so masks and request JSONs reflect the patched state (same pattern
		// as _determinize()).
		if (!battle.ended && battle.requestState) {
			const requests = battle.getRequests(battle.requestState);
			for (const [i, s] of battle.sides.entries()) {
				if (s.activeRequest !== null) s.activeRequest = requests[i];
			}
		}

		return new BattleSim(battle, trackers, obsMode, input.turnCount);
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
		const isV3 = this._obsMode === 'structured-v3' || this._obsMode === 'structured-v3-extended';
		if (!request) {
			const size = this._obsMode === 'flat' ? 100 :
				N_TOKENS * (this._obsMode === 'structured-v3-extended' ? TOKEN_DIM_V3_EXT :
					this._obsMode === 'structured-v3' ? TOKEN_DIM_V3 :
					this._obsMode === 'structured-v2' ? TOKEN_DIM_V2 : TOKEN_DIM);
			return new Float32Array(size);
		}
		if (this._obsMode === 'flat') {
			return extractFeatures(request, null);
		}
		// v3(+extended) keeps the v2 volatile dims (65–76), so all need volatiles;
		// v3 additionally hands in the log-tracked Sleep Clause flag, and
		// v3-extended additionally selects the appended speed-ratio dim.
		const volatiles = this._obsMode === 'structured-v2' || isV3 ?
			this._trackers.volatilesFor(seat) : null;
		const v3Info = isV3 ?
			{
				sleepClause: this._trackers.sleepClauseFlagFor(seat) === 1,
				extended: this._obsMode === 'structured-v3-extended',
			} : undefined;
		return extractFeaturesStructured(request, this._trackers.opponentInfoFor(seat), volatiles, v3Info);
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
