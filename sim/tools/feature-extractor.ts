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

const GEN1_TYPE_CHART: Record<string, Record<string, number>> = {
	normal:   { rock: 0.5, ghost: 0 },
	fire:     { fire: 0.5, water: 0.5, rock: 0.5, grass: 2, ice: 2, bug: 2 },
	water:    { water: 0.5, grass: 0.5, dragon: 0.5, fire: 2, ground: 2, rock: 2 },
	electric: { electric: 0.5, grass: 0.5, dragon: 0.5, ground: 0, water: 2, flying: 2 },
	grass:    { fire: 0.5, grass: 0.5, poison: 0.5, flying: 0.5, dragon: 0.5, bug: 0.5, water: 2, ground: 2, rock: 2 },
	ice:      { water: 0.5, grass: 2, ground: 2, flying: 2, dragon: 2 },
	fighting: { normal: 2, ice: 2, rock: 2, poison: 0.5, bug: 0.5, psychic: 0.5, flying: 0.5, ghost: 0 },
	poison:   { grass: 2, bug: 2, poison: 0.5, ground: 0.5, rock: 0.5, ghost: 0 },
	ground:   { fire: 2, electric: 2, poison: 2, rock: 2, grass: 0.5, bug: 0.5, flying: 0 },
	flying:   { grass: 2, fighting: 2, bug: 2, electric: 0.5, rock: 0.5 },
	psychic:  { fighting: 2, poison: 2, psychic: 0.5 },
	bug:      { grass: 2, poison: 2, psychic: 2, fire: 0.5, fighting: 0.5, flying: 0.5, ghost: 0.5 },
	rock:     { fire: 2, ice: 2, flying: 2, bug: 2, fighting: 0.5, ground: 0.5 },
	ghost:    { ghost: 0, psychic: 0, normal: 0 },
	dragon:   { dragon: 2 },
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
function parseHpRatio(condition: string): number {
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
function parseStatus(condition: string): string {
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
function fillStatusBitmask(obs: Float32Array, offset: number, condition: string): void {
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
 * Defaults to 50 if not found.
 */
function parseLevelFromDetails(details: string): number {
	const match = details.match(/L(\d+)/);
	if (match) {
		const lvl = parseInt(match[1], 10);
		if (!isNaN(lvl)) return lvl;
	}
	return 50;
}

/**
 * Map a type name string to its numeric index (0–14).
 */
function typeToIndex(typeName: string): number {
	return TYPE_INDEX[typeName.toLowerCase()] ?? 0;
}

/**
 * Map a move category to a numeric index.
 *   Physical -> 0, Special -> 1, Status -> 2
 */
function categoryToIndex(category: string): number {
	if (category === 'Physical') return 0;
	if (category === 'Special') return 1;
	return 2; // Status (or unknown)
}

function computeEffectiveness(moveType: string, defType1: string, defType2: string): number {
	const chart = GEN1_TYPE_CHART[moveType] ?? {};
	const e1 = defType1 ? (chart[defType1] ?? 1.0) : 1.0;
	const e2 = (defType2 && defType2 !== defType1) ? (chart[defType2] ?? 1.0) : 1.0;
	const raw = e1 * e2;
	if (raw === 0) return 0.0;
	return (Math.log2(raw) + 2) / 4;
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

// ---- Structured feature extraction (M2) ------------------------------------

/** Number of Pokémon tokens in the structured observation. */
export const N_TOKENS = 12;

/** Number of feature dimensions per token. */
export const TOKEN_DIM = 73;

/**
 * Optional stat-boost data for the active Pokémon on each side.
 * Boost levels are in the range −6 to +6.
 */
export interface BoostData {
	ownActive: Record<string, number>;  // stat name → boost level (−6 to +6)
	oppActive: Record<string, number>;
}

/**
 * Fill one token (TOKEN_DIM=69 floats) starting at `tokenOffset` in `obs`.
 *
 * Layout per token:
 *   [0]     HP ratio
 *   [1]     level/100
 *   [2–16]  type1 one-hot (15 Gen1 types)
 *   [17–31] type2 one-hot (15 Gen1 types)
 *   [32–37] status one-hot: brn, frz, par, psn, slp, tox
 *   [38]    active_flag
 *   [39]    unknown_flag
 *   [40]    fainted_flag
 *   [41–47] move 1 features (base_power/250, accuracy/100, pp_ratio, type_idx/15, category_idx/2, disabled, effectiveness/4)
 *   [48–54] move 2 features
 *   [55–61] move 3 features
 *   [62–68] move 4 features
 */
function fillUnknownToken(obs: Float32Array, tokenOffset: number): void {
	// unknown_flag = 1, HP_ratio = 1, all other dims 0
	obs[tokenOffset + 0] = 1.0;  // HP ratio = 1.0 for unknown
	// dims [1..38] stay 0
	obs[tokenOffset + 39] = 1.0; // unknown_flag
	// dims [40..68] stay 0
}

function fillFaintedToken(obs: Float32Array, tokenOffset: number): void {
	// fainted_flag = 1, HP_ratio = 0, all other dims 0
	// obs[tokenOffset + 0] already 0 (HP ratio = 0)
	obs[tokenOffset + 40] = 1.0; // fainted_flag
	// all other dims stay 0
}

function fillPokemonToken(
	obs: Float32Array,
	tokenOffset: number,
	condition: string,
	details: string,
	isActive: boolean,
	moves: MoveRequestData[] | null,
	oppType1: string,
	oppType2: string,
	boostMap: Record<string, number> = {},
): void {
	const dex = Dex.mod('gen1');

	const hpRatio = parseHpRatio(condition);

	// [0] HP ratio
	obs[tokenOffset + 0] = hpRatio;

	// [1] level/100
	obs[tokenOffset + 1] = parseLevelFromDetails(details) / 100;

	// [2–16] type1 one-hot, [17–31] type2 one-hot
	const speciesName = details.split(',')[0].trim();
	let type1Idx = 0;
	let type2Idx = 0;
	try {
		const speciesInfo = dex.species.get(speciesName);
		if (speciesInfo.exists && speciesInfo.types.length > 0) {
			type1Idx = typeToIndex(speciesInfo.types[0]);
			type2Idx = speciesInfo.types.length > 1 ? typeToIndex(speciesInfo.types[1]) : type1Idx;
		}
	} catch {
		// Defaults stay 0
	}
	obs[tokenOffset + 2 + type1Idx] = 1.0;
	obs[tokenOffset + 17 + type2Idx] = 1.0;

	// [32–37] status one-hot
	fillStatusBitmask(obs, tokenOffset + 32, condition);

	// [38] active_flag
	obs[tokenOffset + 38] = isActive ? 1.0 : 0.0;
	// [39] unknown_flag stays 0
	// [40] fainted_flag stays 0

	// [41–68] move features (4 moves × 7 dims each)
	if (moves !== null) {
		for (let i = 0; i < 4; i++) {
			const moveBase = tokenOffset + 41 + i * 7;
			const moveData = moves[i];

			if (!moveData) {
				// All zeros for missing slot (already initialised to 0)
				continue;
			}

			let basePower = 0;
			let accuracy = 1.0;
			let typeIdx = 0;
			let catIdx = 0;
			let effectiveness = 0.5; // default: neutral 1× maps to (log2(1)+2)/4 = 0.5

			try {
				const moveInfo = dex.moves.get(moveData.id);
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
					effectiveness = computeEffectiveness(moveInfo.type.toLowerCase(), oppType1, oppType2);
				}
			} catch {
				// Safe defaults already set
			}

			obs[moveBase + 0] = basePower;
			obs[moveBase + 1] = accuracy;
			const pp = moveData.pp;
			const maxpp = moveData.maxpp;
			obs[moveBase + 2] = (pp !== undefined && maxpp !== undefined && maxpp > 0) ? pp / maxpp : 1.0;
			obs[moveBase + 3] = typeIdx / 15;
			obs[moveBase + 4] = catIdx / 2;
			obs[moveBase + 5] = moveData.disabled ? 1.0 : 0.0;
			obs[moveBase + 6] = effectiveness;
		}
	}
	// For opponent tokens (moves === null), move dims stay 0

	// [69–72] boost encoding (atk, def, spe, spc), normalised to [-1, 1] via /6
	const boostStats = ['atk', 'def', 'spe', 'spc'];
	for (let bi = 0; bi < 4; bi++) {
		obs[tokenOffset + 69 + bi] = (boostMap[boostStats[bi]] ?? 0) / 6;
	}
}

function fillOpponentPokemonToken(
	obs: Float32Array,
	tokenOffset: number,
	condition: string,
	details: string,
	isActive: boolean,
	boostMap: Record<string, number> = {},
): void {
	// Opponent tokens: no move data visible
	const dex = Dex.mod('gen1');

	const hpRatio = parseHpRatio(condition);

	obs[tokenOffset + 0] = hpRatio;
	obs[tokenOffset + 1] = parseLevelFromDetails(details) / 100;

	const speciesName = details.split(',')[0].trim();
	let type1Idx = 0;
	let type2Idx = 0;
	try {
		const speciesInfo = dex.species.get(speciesName);
		if (speciesInfo.exists && speciesInfo.types.length > 0) {
			type1Idx = typeToIndex(speciesInfo.types[0]);
			type2Idx = speciesInfo.types.length > 1 ? typeToIndex(speciesInfo.types[1]) : type1Idx;
		}
	} catch {
		// Defaults stay 0
	}
	obs[tokenOffset + 2 + type1Idx] = 1.0;
	obs[tokenOffset + 17 + type2Idx] = 1.0;

	fillStatusBitmask(obs, tokenOffset + 32, condition);

	obs[tokenOffset + 38] = isActive ? 1.0 : 0.0;
	// unknown_flag (39) and fainted_flag (40) stay 0
	// move dims [41–68] stay 0 (opponent moves not visible)

	// [69–72] boost encoding (atk, def, spe, spc), normalised to [-1, 1] via /6
	const boostStats = ['atk', 'def', 'spe', 'spc'];
	for (let bi = 0; bi < 4; bi++) {
		obs[tokenOffset + 69 + bi] = (boostMap[boostStats[bi]] ?? 0) / 6;
	}
}

/**
 * Extract a structured (N_TOKENS=12, TOKEN_DIM=65) observation from a battle request.
 *
 * Token layout:
 *   Token 0:   own active Pokémon
 *   Token 1–5: own bench slots 1–5
 *   Token 6:   opponent active Pokémon
 *   Token 7–11: opponent bench slots 1–5
 *
 * Returns a flat Float32Array of length N_TOKENS * TOKEN_DIM = 828.
 */
export function extractFeaturesStructured(
	request: ChoiceRequest,
	opponentRequest: ChoiceRequest | null,
	boosts?: BoostData,
): Float32Array {
	const obs = new Float32Array(N_TOKENS * TOKEN_DIM); // initialised to 0

	// WaitRequest and TeamPreviewRequest: return zero vector
	if (request.wait || request.teamPreview) {
		return obs;
	}

	const ownPokemon = request.side.pokemon;

	let oppType1 = '';
	let oppType2 = '';
	if (opponentRequest && !(opponentRequest as any).wait) {
		const oppActivePoke = opponentRequest.side.pokemon[0];
		if (oppActivePoke) {
			const oppSpeciesName = oppActivePoke.details.split(',')[0].trim();
			try {
				const dexG1 = Dex.mod('gen1');
				const oppSpeciesInfo = dexG1.species.get(oppSpeciesName);
				if (oppSpeciesInfo.exists) {
					oppType1 = oppSpeciesInfo.types[0]?.toLowerCase() ?? '';
					oppType2 = (oppSpeciesInfo.types[1]?.toLowerCase()) ?? oppType1;
				}
			} catch {
				// leave as ''
			}
		}
	}

	// --- Tokens 0–5: own side ---
	// Token 0: own active (index 0)
	const ownActive = ownPokemon[0];
	if (ownActive) {
		const isFainted = ownActive.condition === '0 fnt' || ownActive.condition.endsWith(' fnt');
		if (isFainted) {
			fillFaintedToken(obs, 0 * TOKEN_DIM);
		} else {
			// For move request, get move data from request.active[0].moves
			let moves: MoveRequestData[] | null = null;
			if (isMoveRequest(request) && request.active[0]) {
				moves = request.active[0].moves;
			}
			fillPokemonToken(obs, 0 * TOKEN_DIM, ownActive.condition, ownActive.details, true, moves, oppType1, oppType2, boosts?.ownActive ?? {});
		}
	} else {
		fillUnknownToken(obs, 0 * TOKEN_DIM);
	}

	// Tokens 1–5: own bench slots 1–5
	for (let slot = 1; slot <= 5; slot++) {
		const tokenOffset = slot * TOKEN_DIM;
		const poke = ownPokemon[slot];
		if (!poke) {
			fillUnknownToken(obs, tokenOffset);
		} else {
			const isFainted = poke.condition === '0 fnt' || poke.condition.endsWith(' fnt');
			if (isFainted) {
				fillFaintedToken(obs, tokenOffset);
			} else {
				fillPokemonToken(obs, tokenOffset, poke.condition, poke.details, false, null, '', '');
			}
		}
	}

	// --- Tokens 6–11: opponent side ---
	const allOpponentUnknown = !opponentRequest || (opponentRequest as any).wait === true;

	if (allOpponentUnknown) {
		// All 6 opponent tokens are unknown
		for (let t = 6; t <= 11; t++) {
			fillUnknownToken(obs, t * TOKEN_DIM);
		}
	} else {
		const oppPokemon = opponentRequest.side.pokemon;

		// Token 6: opponent active (index 0)
		const oppActive = oppPokemon[0];
		if (oppActive) {
			const isFainted = oppActive.condition === '0 fnt' || oppActive.condition.endsWith(' fnt');
			if (isFainted) {
				fillFaintedToken(obs, 6 * TOKEN_DIM);
			} else {
				fillOpponentPokemonToken(obs, 6 * TOKEN_DIM, oppActive.condition, oppActive.details, true, boosts?.oppActive ?? {});
			}
		} else {
			fillUnknownToken(obs, 6 * TOKEN_DIM);
		}

		// Tokens 7–11: opponent bench slots 1–5
		for (let slot = 1; slot <= 5; slot++) {
			const tokenOffset = (6 + slot) * TOKEN_DIM;
			const poke = oppPokemon[slot];
			if (!poke) {
				fillUnknownToken(obs, tokenOffset);
			} else {
				const isFainted = poke.condition === '0 fnt' || poke.condition.endsWith(' fnt');
				if (isFainted) {
					fillFaintedToken(obs, tokenOffset);
				} else {
					fillOpponentPokemonToken(obs, tokenOffset, poke.condition, poke.details, false);
				}
			}
		}
	}

	return obs;
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
