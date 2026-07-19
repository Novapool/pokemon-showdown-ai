/**
 * Observation Schema v3 — type-effectiveness lookup + move-effect-flag mapping.
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Job 0.1 (M7): the design/spec-lock layer that Job 1.1's extractor logic
 * consumes. All values here are sourced from the engine's OWN gen1 dex data
 * (`Dex.mod('gen1')`) — the type chart via `getEffectiveness`/`getImmunity`,
 * the move-effect flags via move-data fields — so nothing is hand-transcribed
 * and everything generalises to the full dex, not a known-example list.
 *
 * Dim layout (v3 = 86 dims/token; dims 0–76 stay byte-identical to v2):
 *   Dims 0–64  : v1 per-Pokémon token (unchanged)
 *   Dims 65–76 : v2 boosts/volatiles (unchanged, active tokens only)
 *   Dims 77–80 : per-move-slot type-effectiveness vs opponent active (4 dims)
 *   Dims 81–83 : move effect-flag bits (recharge, self-KO, priority)
 *   Dim  84    : inflicted-status id (0 = none, 1–6 = brn/frz/par/psn/slp/tox)
 *   Dim  85    : Sleep Clause flag (1 = an opponent mon WE slept is still asleep)
 *
 * This module intentionally contains NO dims-77–85 extraction/placement logic
 * (that is Job 1.1) — only the reusable lookups, encoders, and offset constants.
 *
 * @license MIT
 */

import { Dex } from '..';

// ---- Dim layout constants ---------------------------------------------------

/** Full per-token width of a v3 observation. v2 was 77 (see feature-extractor.ts). */
export const TOKEN_DIM_V3 = 86;

/** First of the 4 per-move-slot type-effectiveness dims (77,78,79,80). */
export const V3_TYPE_EFF = 77;
/** Number of move slots / type-eff dims. */
export const V3_MOVE_SLOTS = 4;

/** Effect-flag bit dims. */
export const V3_FLAG_RECHARGE = 81;
export const V3_FLAG_SELFKO = 82;
export const V3_FLAG_PRIORITY = 83;

/** Inflicted-status id dim (single scalar, see MOVE_STATUS_ID). */
export const V3_INFLICTED_STATUS = 84;

/** Sleep Clause flag dim (fed by the tracker in pokemon-gym.ts, Job 1.2). */
export const V3_SLEEP_CLAUSE = 85;

// ---- Type effectiveness -----------------------------------------------------
//
// The engine's `getEffectiveness(sourceType, targetTyping)` returns a summed
// type MODIFIER (each defender type contributes +1 super / -1 resist / 0
// neutral), NOT a multiplier — and it does NOT account for 0x immunity
// (that lives in `getImmunity`). The real gen1 damage multiplier is therefore
//   immune            -> 0
//   otherwise         -> 2 ** modifier
// which yields exactly the gen1 value set {0, 0.25, 0.5, 1, 2, 4}. Dual-typed
// defenders stack because getEffectiveness/getImmunity fold over both types:
//   Ice  vs Dragonite[Dragon,Flying]  -> +1 +1 = +2 -> 4x
//   Fire vs Slowbro  [Water,Psychic]  -> -1 +0 = -1 -> 0.5x
//   Normal vs Gengar [Ghost,Poison]   -> immune to Ghost -> 0x
// All three verified against the engine's own gen1 data.

const GEN1_DEX = Dex.mod('gen1');

/** The six discrete gen1 damage multipliers, for reference/validation. */
export const GEN1_TYPE_MULTIPLIERS = [0, 0.25, 0.5, 1, 2, 4] as const;

/** Divisor used to scale a multiplier into [0, 1] (max multiplier is 4x). */
export const TYPE_EFF_SCALE = 4;

/**
 * Combined gen1 type-effectiveness multiplier of an attacking move type vs a
 * (possibly dual-typed) defender. Type names are the capitalised dex forms
 * ("Ice", "Dragon", ...) exactly as exposed by `move.type` and `species.types`.
 *
 * Returns one of {0, 0.25, 0.5, 1, 2, 4}. 0 means immune (e.g. Normal→Ghost).
 * Never returns NaN/inf: an unknown/empty type simply contributes a 0 modifier.
 */
export function computeTypeEffMultiplier(moveType: string, defenderTypes: string[]): number {
	if (!moveType || !defenderTypes || defenderTypes.length === 0) return 1;
	if (!GEN1_DEX.getImmunity(moveType, defenderTypes)) return 0;
	const modifier = GEN1_DEX.getEffectiveness(moveType, defenderTypes);
	return Math.pow(2, modifier);
}

/**
 * Scale a raw multiplier to the encoded dim value in [0, 1] via `mult / 4`.
 * Round-trip:
 *   0x    -> 0.0     0.25x -> 0.0625   0.5x -> 0.125
 *   1x    -> 0.25    2x    -> 0.5      4x   -> 1.0
 * 0x immunity encodes to exactly 0.0. NOTE: an empty/absent move slot should
 * ALSO be written as 0.0 by Job 1.1's extractor, so 0.0 is shared by "immune"
 * and "no move" — the disabled/PP dims already distinguish empty slots, so this
 * collision is acceptable and documented (see MILESTONES.md v3 spec).
 */
export function encodeTypeEff(multiplier: number): number {
	return multiplier / TYPE_EFF_SCALE;
}

/**
 * Convenience: encoded [0,1] type-effectiveness of a move type vs a defender's
 * type(s), ready to drop straight into dims 77–80.
 */
export function typeEffDim(moveType: string, defenderTypes: string[]): number {
	return encodeTypeEff(computeTypeEffMultiplier(moveType, defenderTypes));
}

// ---- Move effect flags ------------------------------------------------------
//
// Derived from move-data fields on the gen1 dex, NOT a name list:
//   recharge -> move.flags.recharge === 1   (Hyper Beam; self mustrecharge)
//   self-KO  -> !!move.selfdestruct          ('always'/'ifHit': Explosion, Selfdestruct)
//   priority -> move.priority > 0            (Quick Attack = +1)
//   status   -> move.status ?? move.secondary?.status
//               (Thunder Wave/Hypnosis set move.status; Body Slam/Ice Beam set
//                a secondary.status. Rest has NO move.status — its self-sleep is
//                handled by its own onHit, so it correctly maps to "none" here
//                and never counts as an opponent-inflicted status.)

/**
 * Inflicted-status id for dim 84. 0 = none; 1–6 follow the same order as the
 * v1/v2 status bitmask (burn, freeze, paralysis, poison, sleep, toxic) shifted
 * by +1 so that 0 can mean "inflicts no status" (the common case).
 *
 * NOTE / deviation from MILESTONES.md's literal "0–5": the spec's 0–5 conflated
 * "none" with a real status. A dedicated 0 = none sentinel is required because
 * most moves inflict nothing, so the range here is 0–6. Job 1.1 may normalise
 * (e.g. `/ 6`) for placement; the raw id is exposed for clarity.
 */
export const MOVE_STATUS_ID: Record<string, number> = {
	brn: 1,
	frz: 2,
	par: 3,
	psn: 4,
	slp: 5,
	tox: 6,
};

/** Max status id, for optional `/ MOVE_STATUS_ID_MAX` normalisation in Job 1.1. */
export const MOVE_STATUS_ID_MAX = 6;

export interface MoveEffectFlags {
	/** Hyper Beam-style must-recharge turn. */
	recharge: boolean;
	/** Explosion/Selfdestruct — user faints. */
	selfKO: boolean;
	/** Positive-priority move (Quick Attack). */
	priority: boolean;
	/** Status this move inflicts on the target: 0 = none, 1–6 per MOVE_STATUS_ID. */
	inflictedStatusId: number;
}

const EMPTY_FLAGS: MoveEffectFlags = { recharge: false, selfKO: false, priority: false, inflictedStatusId: 0 };

/**
 * Effect flags for a move id (or name). Unknown/empty moves return all-false /
 * status 0 — never NaN. Generalises to every gen1 move via dex fields.
 */
export function getMoveEffectFlags(moveId: string): MoveEffectFlags {
	if (!moveId) return { ...EMPTY_FLAGS };
	let move;
	try {
		move = GEN1_DEX.moves.get(moveId);
	} catch {
		return { ...EMPTY_FLAGS };
	}
	if (!move || !move.exists) return { ...EMPTY_FLAGS };

	const statusName = (move.status || (move.secondary && move.secondary.status) || '') as string;
	return {
		recharge: move.flags?.recharge === 1,
		selfKO: !!move.selfdestruct,
		priority: (move.priority || 0) > 0,
		inflictedStatusId: MOVE_STATUS_ID[statusName] ?? 0,
	};
}
