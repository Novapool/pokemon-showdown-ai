/**
 * Feature extractor for Pokemon battle observations.
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Converts a ChoiceRequest into a fixed-size Float32Array suitable
 * for ML model consumption.
 *
 * @license MIT
 */

import { Dex } from '..';
import type { ChoiceRequest, MoveRequest, SwitchRequest, PokemonSwitchRequestData, MoveRequestData } from '../side';
import {
	TOKEN_DIM_V3,
	V3_TYPE_EFF,
	V3_MOVE_SLOTS,
	V3_FLAG_RECHARGE,
	V3_FLAG_SELFKO,
	V3_FLAG_PRIORITY,
	V3_INFLICTED_STATUS,
	V3_SLEEP_CLAUSE,
	TOKEN_DIM_V3_EXT,
	V3_SPEED_RATIO,
	MOVE_STATUS_ID_MAX,
	typeEffDim,
	getMoveEffectFlags,
	gen1BaseSpeed,
	encodeSpeedRatio,
} from './type-chart-v3';

export { TOKEN_DIM_V3, TOKEN_DIM_V3_EXT } from './type-chart-v3';

export const OBS_SIZE = 100;

// Static mapping of Gen 1 types to numeric indices (0-based).
// Gen 1 has 15 types. We normalise by dividing by 20.
const TYPE_INDEX: Record<string, number> = {
	normal: 0,
	fire: 1,
	water: 2,
	electric: 3,
	grass: 4,
	ice: 5,
	fighting: 6,
	poison: 7,
	ground: 8,
	flying: 9,
	psychic: 10,
	bug: 11,
	rock: 12,
	ghost: 13,
	dragon: 14,
};

// ---- Helpers ----------------------------------------------------------------

function isMoveRequest(r: ChoiceRequest): r is MoveRequest {
	return 'active' in r;
}

function isSwitchRequest(r: ChoiceRequest): r is SwitchRequest {
	return 'forceSwitch' in r;
}

/**
 * Parse an HP condition string of the form "NNN/MMM" or "NNN/MMM STS"
 * or the fainted shorthand "0 fnt".
 * Returns the ratio current/max clamped to [0,1].
 */
export function parseHpRatio(condition: string): number {
	if (!condition || condition === '0 fnt') return 0;

	// "NNN/MMM" or "NNN/MMM STS"
	const slashIdx = condition.indexOf('/');
	if (slashIdx === -1) {
		// No slash – could be bare "0 fnt" already handled above or some edge case
		return 0;
	}

	const current = parseInt(condition.slice(0, slashIdx), 10);
	// After slash: digits optionally followed by " STS"
	const afterSlash = condition.slice(slashIdx + 1);
	const spaceIdx = afterSlash.indexOf(' ');
	const maxStr = spaceIdx === -1 ? afterSlash : afterSlash.slice(0, spaceIdx);
	const max = parseInt(maxStr, 10);

	if (!max || isNaN(current) || isNaN(max)) return 0;
	return Math.min(1, Math.max(0, current / max));
}

/**
 * Parse a status string out of a condition like "200/300 brn".
 * Returns the status token in lowercase, or '' if none.
 */
export function parseStatus(condition: string): string {
	if (!condition) return '';
	const slashIdx = condition.indexOf('/');
	if (slashIdx === -1) {
		// Could be "0 fnt"
		const spaceIdx = condition.indexOf(' ');
		return spaceIdx !== -1 ? condition.slice(spaceIdx + 1).toLowerCase() : '';
	}
	const afterSlash = condition.slice(slashIdx + 1);
	const spaceIdx = afterSlash.indexOf(' ');
	return spaceIdx !== -1 ? afterSlash.slice(spaceIdx + 1).toLowerCase() : '';
}

/**
 * Fill bytes [offset+0 .. offset+5] with a 6-element status bitmask.
 * Order: burn, freeze, paralysis, poison, sleep, toxic (each 0.0 or 1.0).
 * Reads the status token from the condition string.
 */
export function fillStatusBitmask(obs: Float32Array, offset: number, condition: string): void {
	const status = parseStatus(condition);
	obs[offset + 0] = status === 'brn' ? 1.0 : 0.0;  // burn
	obs[offset + 1] = status === 'frz' ? 1.0 : 0.0;  // freeze
	obs[offset + 2] = status === 'par' ? 1.0 : 0.0;  // paralysis
	obs[offset + 3] = status === 'psn' ? 1.0 : 0.0;  // poison
	obs[offset + 4] = status === 'slp' ? 1.0 : 0.0;  // sleep
	obs[offset + 5] = status === 'tox' ? 1.0 : 0.0;  // toxic
}

/**
 * Parse the level from a details string like "Pikachu, L50, M".
 * Showdown omits the level tag entirely when it's 100 (`Pokemon#details`),
 * so the default here must be 100, not 50 — gen1ou/gen1randombattle mons
 * are level 100 by convention, and Metamon's adapter assumes the same
 * default for parity with the BC-pretrained checkpoint.
 */
export function parseLevelFromDetails(details: string): number {
	const match = details.match(/L(\d+)/);
	if (match) {
		const lvl = parseInt(match[1], 10);
		if (!isNaN(lvl)) return lvl;
	}
	return 100;
}

/**
 * Map a type name string to its numeric index (0–14).
 */
export function typeToIndex(typeName: string): number {
	return TYPE_INDEX[typeName.toLowerCase()] ?? 0;
}

/**
 * Map a move category to a numeric index.
 *   Physical -> 0, Special -> 1, Status -> 2
 */
export function categoryToIndex(category: string): number {
	if (category === 'Physical') return 0;
	if (category === 'Special') return 1;
	return 2; // Status (or unknown)
}

// ---- Own active Pokémon (indices 0–14) -------------------------------------

function fillOwnActive(obs: Float32Array, condition: string, details: string): void {
	// [0] HP ratio
	obs[0] = parseHpRatio(condition);

	// [1] Level /100
	const lvl = parseLevelFromDetails(details);
	obs[1] = lvl / 100;

	// [2–7] Status bitmask (burn, freeze, paralysis, poison, sleep, toxic)
	fillStatusBitmask(obs, 2, condition);

	// [8–14] Stat boosts: set to 0.5 (neutral) — not available in Gen 1 request
	for (let i = 8; i <= 14; i++) {
		obs[i] = 0.5;
	}
}

// ---- Own moves (indices 15–54) ---------------------------------------------

function fillMoves(obs: Float32Array, moves: MoveRequestData[]): void {
	const dex = Dex.mod('gen1');
	const BASE = 15;

	for (let i = 0; i < 4; i++) {
		const base = BASE + i * 10;
		const moveData = moves[i];

		if (!moveData) {
			// Pad with zeros for missing move slots
			for (let j = 0; j < 10; j++) obs[base + j] = 0;
			continue;
		}

		// Look up move from Dex
		const moveId = moveData.id;
		let basePower = 0;
		let accuracy = 1.0;
		let typeIdx = 0;
		let catIdx = 0;

		try {
			const moveInfo = dex.moves.get(moveId);
			if (moveInfo.exists) {
				basePower = Math.min(1, Math.max(0, moveInfo.basePower / 250));

				const rawAccuracy = moveInfo.accuracy;
				if (rawAccuracy === true || rawAccuracy === undefined) {
					accuracy = 1.0;
				} else {
					accuracy = (rawAccuracy as number) / 100;
				}

				typeIdx = typeToIndex(moveInfo.type);
				catIdx = categoryToIndex(moveInfo.category);
			}
		} catch {
			// Safe defaults already set
		}

		// [base+0] basePower/250 clamped 0–1
		obs[base + 0] = basePower;
		// [base+1] accuracy/100 (1.0 if always-hits)
		obs[base + 1] = accuracy;
		// [base+2] PP ratio
		const pp = moveData.pp;
		const maxpp = moveData.maxpp;
		obs[base + 2] = (pp !== undefined && maxpp !== undefined && maxpp > 0) ? pp / maxpp : 1.0;
		// [base+3] type index /20
		obs[base + 3] = typeIdx / 20;
		// [base+4] category index /2
		obs[base + 4] = catIdx / 2;
		// [base+5] disabled
		obs[base + 5] = moveData.disabled ? 1.0 : 0.0;
		// [base+6–9] padding zeros
		obs[base + 6] = 0;
		obs[base + 7] = 0;
		obs[base + 8] = 0;
		obs[base + 9] = 0;
	}
}

// ---- Switch options (indices 55–59) ----------------------------------------

/**
 * For each bench slot 1–5 (0-indexed indices 1–5 in side.pokemon),
 * set 1.0 if the Pokémon is alive and not the currently active one, else 0.0.
 */
function fillSwitchMask(obs: Float32Array, pokemon: PokemonSwitchRequestData[]): void {
	const BASE = 55;
	const activeIdent = pokemon[0]?.ident ?? '';

	for (let slot = 1; slot <= 5; slot++) {
		const poke = pokemon[slot];
		if (!poke) {
			obs[BASE + slot - 1] = 0.0;
			continue;
		}
		const isFainted = poke.condition.endsWith(' fnt') || poke.condition === '0 fnt';
		const isActive = poke.ident === activeIdent;
		obs[BASE + slot - 1] = (!isFainted && !isActive) ? 1.0 : 0.0;
	}
}

// ---- Opponent active Pokémon (indices 60–74) --------------------------------

function fillOpponent(obs: Float32Array, opponentRequest: ChoiceRequest | null): void {
	const BASE = 60;

	if (!opponentRequest || opponentRequest.wait) {
		// No info available — use neutral defaults
		obs[BASE + 0] = 0.5; // HP ratio
		obs[BASE + 1] = 0.5; // Level
		for (let i = 2; i <= 14; i++) obs[BASE + i] = 0;
		return;
	}

	const pokemon = opponentRequest.side.pokemon;
	const activePoke = pokemon[0];

	if (!activePoke) {
		obs[BASE + 0] = 0.5;
		obs[BASE + 1] = 0.5;
		for (let i = 2; i <= 14; i++) obs[BASE + i] = 0;
		return;
	}

	// [60] HP ratio
	obs[BASE + 0] = parseHpRatio(activePoke.condition);

	// [61] Level /100
	const lvl = parseLevelFromDetails(activePoke.details);
	obs[BASE + 1] = lvl / 100;

	// [62–67] Status bitmask
	fillStatusBitmask(obs, BASE + 2, activePoke.condition);

	// [68–69] Type indices /20 from species
	const dex = Dex.mod('gen1');
	const speciesName = activePoke.details.split(',')[0].trim();
	let type1Idx = 0;
	let type2Idx = 0;

	try {
		const speciesInfo = dex.species.get(speciesName);
		if (speciesInfo.exists && speciesInfo.types.length > 0) {
			type1Idx = typeToIndex(speciesInfo.types[0]);
			type2Idx = speciesInfo.types.length > 1 ? typeToIndex(speciesInfo.types[1]) : type1Idx;
		}
	} catch {
		// Defaults remain 0
	}

	obs[BASE + 8] = type1Idx / 20;
	obs[BASE + 9] = type2Idx / 20;

	// [70] Species index normalized
	let speciesNum = 0;
	try {
		const speciesInfo = dex.species.get(speciesName);
		if (speciesInfo.exists) {
			speciesNum = speciesInfo.num;
		}
	} catch {
		// Default 0
	}
	obs[BASE + 10] = speciesNum / 200;

	// [71–74] padding zeros
	obs[BASE + 11] = 0;
	obs[BASE + 12] = 0;
	obs[BASE + 13] = 0;
	obs[BASE + 14] = 0;
}

// ---- Main export -----------------------------------------------------------

/**
 * Extract a fixed-size (OBS_SIZE=100) observation vector from a battle request.
 *
 * Layout:
 *   [0–14]  Own active Pokémon features
 *   [15–54] Own move features (4 moves × 10 features)
 *   [55–59] Switch availability mask (bench slots 1–5)
 *   [60–74] Opponent active Pokémon features
 *   [75–99] Padding zeros
 */
export function extractFeatures(
	request: ChoiceRequest,
	opponentRequest: ChoiceRequest | null,
): Float32Array {
	const obs = new Float32Array(OBS_SIZE); // initialised to 0

	// WaitRequest and TeamPreviewRequest: return zero vector
	if (request.wait || request.teamPreview) {
		return obs;
	}

	if (isSwitchRequest(request)) {
		// Force-switch: extract own Pokémon info from side.pokemon[0]
		const activePoke = request.side.pokemon[0];
		if (activePoke) {
			fillOwnActive(obs, activePoke.condition, activePoke.details);
		}
		// No move features for a switch request — indices 15–54 remain 0
		fillSwitchMask(obs, request.side.pokemon);
	} else if (isMoveRequest(request)) {
		// Move request: full feature extraction
		const activePoke = request.side.pokemon[0];
		if (activePoke) {
			fillOwnActive(obs, activePoke.condition, activePoke.details);
		}

		// Own moves from request.active[0]
		const activeMoveData = request.active[0];
		if (activeMoveData) {
			fillMoves(obs, activeMoveData.moves);
		}

		fillSwitchMask(obs, request.side.pokemon);
	}

	// Opponent features (indices 60–74)
	fillOpponent(obs, opponentRequest);

	// Indices 75–99 remain 0 (padding)

	return obs;
}

// ---- Structured per-Pokémon token extraction (M2) --------------------------
//
// Token layout mirrors models/metamon_adapter.py exactly (offsets, one-hot
// order, unknown/fainted conventions) so the M2.5 BC-pretrained checkpoint's
// 65-dim input projection stays valid against live-gym observations.

export const TOKEN_DIM = 65;
export const N_TOKENS = 12;

// ---- Schema v2 (M3.4) -------------------------------------------------------
//
// The v1 65-dim token was frozen to match the Metamon BC checkpoint's input
// projection. The M3.2 decision retired the transformer/BC path, so v2 is a
// deliberate schema bump: 12 extra dims are APPENDED to every token (dims
// 0–64 stay byte-identical to v1, so a v1 observation is exactly the first 65
// dims of each v2 token). The new dims are only ever non-zero on the two
// active-Pokémon tokens ([0] own active, [6] opponent active) — gen1 boosts
// and volatiles all reset on switch, so bench tokens are correctly neutral.

export const TOKEN_DIM_V2 = 77;

const V_BOOSTS = 65;        // 7 dims: atk, def, spe, spa, accuracy, evasion, spd — each stage/6 in [-1, 1]
const V_REFLECT = 72;
const V_LIGHTSCREEN = 73;
const V_SUBSTITUTE = 74;
const V_LEECHSEED = 75;
const V_TOXIC_COUNTER = 76; // poison damage ticks since status applied, min(n,16)/16

/** Boost-slot order for ActiveVolatiles.boosts (gen1: spa/spd are both "special"). */
export const BOOST_ORDER = ['atk', 'def', 'spe', 'spa', 'accuracy', 'evasion', 'spd'] as const;

/**
 * Boost stages and volatile conditions of one ACTIVE Pokémon, tracked by the
 * gym from public battle-log lines (see pokemon-gym.ts). All of it resets on
 * switch/faint, matching gen1 mechanics.
 */
export interface ActiveVolatiles {
	/** 7 stages in BOOST_ORDER, each an integer in [-6, 6]. */
	boosts: number[];
	reflect: boolean;
	lightScreen: boolean;
	substitute: boolean;
	leechSeed: boolean;
	/** Poison/toxic damage ticks taken since the status was applied. */
	toxicCounter: number;
}

function fillVolatileDims(obs: Float32Array, offset: number, vol: ActiveVolatiles): void {
	for (let i = 0; i < 7; i++) {
		const stage = vol.boosts[i] ?? 0;
		obs[offset + V_BOOSTS + i] = Math.max(-1, Math.min(1, stage / 6));
	}
	obs[offset + V_REFLECT] = vol.reflect ? 1.0 : 0.0;
	obs[offset + V_LIGHTSCREEN] = vol.lightScreen ? 1.0 : 0.0;
	obs[offset + V_SUBSTITUTE] = vol.substitute ? 1.0 : 0.0;
	obs[offset + V_LEECHSEED] = vol.leechSeed ? 1.0 : 0.0;
	obs[offset + V_TOXIC_COUNTER] = Math.min(vol.toxicCounter, 16) / 16;
}

const T_HP = 0;
const T_LEVEL = 1;
const T_TYPE1 = 2;
const T_TYPE2 = 17;
const T_STATUS = 32;
const T_ACTIVE_FLAG = 38;
const T_UNKNOWN_FLAG = 39;
const T_FAINTED_FLAG = 40;
const T_MOVES = 41;
const T_MOVE_DIM = 6;

/**
 * One move slot's input for token building. `pp`/`maxpp` are omitted for
 * Pokémon whose current PP isn't observable (bench, opponent) — PP ratio
 * then defaults to 1.0, matching metamon_adapter.py's assumption.
 */
interface TokenMoveInput {
	id: string;
	pp?: number;
	maxpp?: number;
	disabled?: boolean;
}

/**
 * One revealed opponent Pokémon, reconstructed from the battle log by the
 * gym's opponent tracker (see pokemon-gym.ts). Contains only information a
 * real player could have observed — no hidden state.
 */
export interface OpponentPokemonInfo {
	details: string;
	condition: string;
	active: boolean;
	/** Revealed move ids, in the order first observed (max 4 kept). */
	moves: string[];
}

// ---- Schema v3 (M7) ---------------------------------------------------------
//
// v3 appends dims 77–85 to every token; dims 0–76 stay byte-identical to v2.
// The lookups/encoders live in type-chart-v3.ts (Job 0.1) — this file only
// places their outputs. Per-slot type effectiveness (77–80) is that token's
// move set vs the OPPONENT ACTIVE Pokémon's type(s); effect flags (81–83) and
// inflicted-status (84) are aggregated over the token's ≤4 moves; the Sleep
// Clause flag (85) is a global battle signal fed by the gym tracker (Job 1.2)
// and is placed identically on every token.

/**
 * v3 information supplied by the gym (Job 1.2's `structured-v3` obsMode).
 * Currently just the Sleep Clause flag — everything else in dims 77–84 is
 * derived from move/species data the extractor already has. Passing this
 * (even with `sleepClause: false`) is what selects the 86-dim v3 schema.
 */
export interface V3Info {
	/** 1 while an opponent Pokémon WE put to sleep is still asleep (bench-persistent). */
	sleepClause: boolean;
	/**
	 * M8 Phase 1A: select the v3-extended schema (87 dims/token) — appends the
	 * own/opp active base-speed ratio at dim 86. When absent/false the v3 (86)
	 * schema is produced. Dims 0–85 are byte-identical either way.
	 */
	extended?: boolean;
}

/** Internal per-call v3 context threaded into token filling. */
interface V3FillContext {
	/** Opponent active Pokémon's type(s) — the defender for dims 77–80. */
	defenderTypes: string[];
}

/**
 * Fill dims 77–84 for one token from its move set. Dim 85 (Sleep Clause) is set
 * separately across all tokens. Empty move slots encode type-eff 0.0 (shared
 * with 0x immunity — documented, disambiguated by the existing PP/disabled dims).
 * All values are bounded in [0, 1]; unknown moves/types degrade to safe defaults.
 */
function fillV3MoveDims(
	obs: Float32Array,
	offset: number,
	moves: TokenMoveInput[],
	ctx: V3FillContext,
): void {
	const dex = Dex.mod('gen1');
	let anyRecharge = false;
	let anySelfKO = false;
	let anyPriority = false;
	let statusId = 0;

	for (let i = 0; i < V3_MOVE_SLOTS; i++) {
		const move = moves[i];
		if (!move) {
			obs[offset + V3_TYPE_EFF + i] = 0.0;
			continue;
		}

		let moveType = '';
		try {
			const moveInfo = dex.moves.get(move.id);
			if (moveInfo.exists) moveType = moveInfo.type;
		} catch {
			// leave moveType empty -> type-eff 0.0
		}
		obs[offset + V3_TYPE_EFF + i] = (moveType && ctx.defenderTypes.length > 0) ?
			typeEffDim(moveType, ctx.defenderTypes) : 0.0;

		const flags = getMoveEffectFlags(move.id);
		if (flags.recharge) anyRecharge = true;
		if (flags.selfKO) anySelfKO = true;
		if (flags.priority) anyPriority = true;
		if (statusId === 0 && flags.inflictedStatusId > 0) statusId = flags.inflictedStatusId;
	}

	obs[offset + V3_FLAG_RECHARGE] = anyRecharge ? 1.0 : 0.0;
	obs[offset + V3_FLAG_SELFKO] = anySelfKO ? 1.0 : 0.0;
	obs[offset + V3_FLAG_PRIORITY] = anyPriority ? 1.0 : 0.0;
	// Normalise the 0–6 status id into [0, 1] to match every other dim's scale.
	obs[offset + V3_INFLICTED_STATUS] = statusId / MOVE_STATUS_ID_MAX;
}

/** Resolve the opponent active Pokémon's gen1 type(s) for the type-eff defender. */
function opponentActiveTypes(opponent: OpponentPokemonInfo[]): string[] {
	const active = opponent.find(p => p.active);
	if (!active) return [];
	const speciesName = active.details.split(',')[0].trim();
	try {
		const speciesInfo = Dex.mod('gen1').species.get(speciesName);
		if (speciesInfo.exists && speciesInfo.types.length > 0) return speciesInfo.types;
	} catch {
		// fall through to empty
	}
	return [];
}

function fillUnknownToken(obs: Float32Array, offset: number): void {
	// Unrevealed opponent bench: assumed full HP, everything else zero.
	// Do NOT use an all-zero vector — that would be indistinguishable from
	// a fainted Pokémon (HP=0, fainted_flag=1).
	obs[offset + T_HP] = 1.0;
	obs[offset + T_UNKNOWN_FLAG] = 1.0;
}

function fillFaintedToken(obs: Float32Array, offset: number): void {
	obs[offset + T_FAINTED_FLAG] = 1.0;
}

function fillPokemonToken(
	obs: Float32Array,
	offset: number,
	details: string,
	condition: string,
	active: boolean,
	moves: TokenMoveInput[],
	v3ctx?: V3FillContext,
): void {
	if (parseHpRatio(condition) <= 0) {
		fillFaintedToken(obs, offset);
		return;
	}

	const dex = Dex.mod('gen1');

	obs[offset + T_HP] = parseHpRatio(condition);
	obs[offset + T_LEVEL] = parseLevelFromDetails(details) / 100;

	const speciesName = details.split(',')[0].trim();
	try {
		const speciesInfo = dex.species.get(speciesName);
		if (speciesInfo.exists && speciesInfo.types.length > 0) {
			const type1Idx = typeToIndex(speciesInfo.types[0]);
			const type2Idx = speciesInfo.types.length > 1 ? typeToIndex(speciesInfo.types[1]) : type1Idx;
			obs[offset + T_TYPE1 + type1Idx] = 1.0;
			obs[offset + T_TYPE2 + type2Idx] = 1.0;
		}
	} catch {
		// Leave type one-hots zero
	}

	fillStatusBitmask(obs, offset + T_STATUS, condition);
	obs[offset + T_ACTIVE_FLAG] = active ? 1.0 : 0.0;

	for (let i = 0; i < Math.min(4, moves.length); i++) {
		const move = moves[i];
		const base = offset + T_MOVES + i * T_MOVE_DIM;

		let basePower = 0;
		let accuracy = 1.0;
		let typeIdx = 0;
		let catIdx = 0;
		try {
			const moveInfo = dex.moves.get(move.id);
			if (moveInfo.exists) {
				basePower = Math.min(1, Math.max(0, moveInfo.basePower / 250));
				const rawAccuracy = moveInfo.accuracy;
				accuracy = (rawAccuracy === true || rawAccuracy === undefined) ? 1.0 : (rawAccuracy as number) / 100;
				typeIdx = typeToIndex(moveInfo.type);
				catIdx = categoryToIndex(moveInfo.category);
			}
		} catch {
			// Safe defaults already set
		}

		obs[base + 0] = basePower;
		obs[base + 1] = accuracy;
		obs[base + 2] = (move.pp !== undefined && move.maxpp !== undefined && move.maxpp > 0) ?
			move.pp / move.maxpp : 1.0;
		obs[base + 3] = typeIdx / 20;
		obs[base + 4] = catIdx / 2;
		obs[base + 5] = move.disabled ? 1.0 : 0.0;
	}

	if (v3ctx) fillV3MoveDims(obs, offset, moves, v3ctx);
}

function fillOwnBenchToken(
	obs: Float32Array, offset: number, poke: PokemonSwitchRequestData, v3ctx?: V3FillContext
): void {
	// Bench Pokémon only expose choosable move ids (no live PP/disabled info).
	const moves: TokenMoveInput[] = (poke.moves || []).slice(0, 4).map(id => ({ id }));
	fillPokemonToken(obs, offset, poke.details, poke.condition, false, moves, v3ctx);
}

function fillOpponentToken(
	obs: Float32Array, offset: number, info: OpponentPokemonInfo, v3ctx?: V3FillContext
): void {
	const moves: TokenMoveInput[] = info.moves.slice(0, 4).map(id => ({ id }));
	fillPokemonToken(obs, offset, info.details, info.condition, info.active, moves, v3ctx);
}

function fillOpponentTokens(
	obs: Float32Array, opponent: OpponentPokemonInfo[], tokenDim: number, v3ctx?: V3FillContext
): void {
	const activeOffset = 6 * tokenDim;
	const active = opponent.find(p => p.active);
	if (active) {
		fillOpponentToken(obs, activeOffset, active, v3ctx);
	} else {
		// Between switch resolution and the next reveal there's a brief window
		// with no known active opponent — fall back to unknown rather than crash.
		fillUnknownToken(obs, activeOffset);
	}

	const bench = opponent.filter(p => !p.active);
	for (let j = 0; j < 5; j++) {
		const offset = (7 + j) * tokenDim;
		if (j < bench.length) {
			fillOpponentToken(obs, offset, bench[j], v3ctx);
		} else {
			fillUnknownToken(obs, offset);
		}
	}
}

/**
 * Extract a (12, 65) structured per-Pokémon token observation, flattened to
 * a single Float32Array of length N_TOKENS * TOKEN_DIM = 780.
 *
 * Token order: [0] own active, [1–5] own bench (side.pokemon[1..5] order —
 * action k grounds to bench token k-4, matching PokemonGymEnv's fixed-slot
 * action space), [6] opponent active, [7–11] opponent bench (revealed
 * non-active first, then unknown, then fainted).
 *
 * `opponent` must be built from what a real player could observe (see the
 * opponent tracker in pokemon-gym.ts) — never from omniscient/hidden state.
 *
 * Passing `volatiles` (M3.4) switches to schema v2: 77-dim tokens whose first
 * 65 dims are identical to v1, with boost stages and volatile-condition flags
 * appended on the two active tokens ([0] own, [6] opponent).
 *
 * Passing `v3Info` (M7) switches to schema v3: 86-dim tokens whose first 77 dims
 * are byte-identical to v2 (so `volatiles` must be supplied alongside `v3Info`
 * for a full v2-parity prefix), with dims 77–85 appended to every token:
 * per-move-slot type effectiveness vs the opponent active (77–80), aggregated
 * recharge/self-KO/priority flags (81–83), inflicted-status id (84), and the
 * global Sleep Clause flag (85). Dims 0–76 are never altered by the v3 path.
 *
 * Passing `v3Info.extended` (M8 Phase 1A) switches to schema v3-extended:
 * 87-dim tokens whose first 86 dims are byte-identical to v3, with the global
 * own/opp active base-speed ratio appended at dim 86 (all 12 tokens).
 */
export function extractFeaturesStructured(
	request: ChoiceRequest,
	opponent: OpponentPokemonInfo[],
	volatiles?: { own: ActiveVolatiles, opp: ActiveVolatiles } | null,
	v3Info?: V3Info | null,
): Float32Array {
	const v3Ext = !!(v3Info && v3Info.extended);
	const tokenDim = v3Ext ? TOKEN_DIM_V3_EXT : (v3Info ? TOKEN_DIM_V3 : (volatiles ? TOKEN_DIM_V2 : TOKEN_DIM));
	const obs = new Float32Array(N_TOKENS * tokenDim);
	const v3ctx: V3FillContext | undefined = v3Info ?
		{ defenderTypes: opponentActiveTypes(opponent) } : undefined;

	if (request.wait || request.teamPreview) {
		fillOpponentTokens(obs, opponent, tokenDim, v3ctx);
		if (volatiles) fillVolatileDims(obs, 6 * tokenDim, volatiles.opp);
		if (v3Info) fillSleepClause(obs, tokenDim, v3Info);
		if (v3Ext) fillSpeedRatio(obs, tokenDim, request, opponent);
		return obs;
	}

	const pokemon = request.side.pokemon;
	const ownActive = pokemon[0];

	if (ownActive) {
		if (isSwitchRequest(request)) {
			fillPokemonToken(obs, 0, ownActive.details, ownActive.condition, true, [], v3ctx);
		} else if (isMoveRequest(request)) {
			const activeMoveData = request.active[0];
			const moves: TokenMoveInput[] = activeMoveData ?
				activeMoveData.moves.slice(0, 4).map(m => (
					{ id: m.id, pp: m.pp, maxpp: m.maxpp, disabled: !!m.disabled }
				)) :
				[];
			fillPokemonToken(obs, 0, ownActive.details, ownActive.condition, true, moves, v3ctx);
		}
	}

	for (let slot = 1; slot <= 5; slot++) {
		const offset = slot * tokenDim;
		const poke = pokemon[slot];
		if (!poke) {
			fillFaintedToken(obs, offset);
			continue;
		}
		fillOwnBenchToken(obs, offset, poke, v3ctx);
	}

	fillOpponentTokens(obs, opponent, tokenDim, v3ctx);

	if (volatiles) {
		fillVolatileDims(obs, 0, volatiles.own);
		fillVolatileDims(obs, 6 * tokenDim, volatiles.opp);
	}

	if (v3Info) fillSleepClause(obs, tokenDim, v3Info);
	if (v3Ext) fillSpeedRatio(obs, tokenDim, request, opponent);

	return obs;
}

/**
 * Place the own/opp active base-speed ratio (dim 86) identically on every token
 * — a battle-wide scalar, like the Sleep Clause flag, so token-pooling models
 * see it regardless of which token they attend to. Own active is the requesting
 * side's `pokemon[0]`; opp active is the revealed opponent active. Either side
 * unknown → neutral 0.5 (see `encodeSpeedRatio`). v3-extended only.
 */
function fillSpeedRatio(
	obs: Float32Array, tokenDim: number, request: ChoiceRequest, opponent: OpponentPokemonInfo[]
): void {
	const ownActive = request.side?.pokemon?.[0];
	const ownSpe = ownActive ? gen1BaseSpeed(ownActive.details.split(',')[0].trim()) : null;
	const oppActive = opponent.find(p => p.active);
	const oppSpe = oppActive ? gen1BaseSpeed(oppActive.details.split(',')[0].trim()) : null;
	const value = encodeSpeedRatio(ownSpe, oppSpe);
	for (let t = 0; t < N_TOKENS; t++) {
		obs[t * tokenDim + V3_SPEED_RATIO] = value;
	}
}

/**
 * Place the global Sleep Clause flag (dim 85) identically on every token — a
 * battle-wide signal, so token-pooling models see it regardless of which token
 * they attend to. No-op unless v3.
 */
function fillSleepClause(obs: Float32Array, tokenDim: number, v3Info: V3Info): void {
	const value = v3Info.sleepClause ? 1.0 : 0.0;
	for (let t = 0; t < N_TOKENS; t++) {
		obs[t * tokenDim + V3_SLEEP_CLAUSE] = value;
	}
}
