/**
 * Fixed-Roster Loader (M12)
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * The M12 pivot runs every battle — training, bot eval and ladder — on ONE
 * fixed Gen 1 OU team, used by both sides. This module is the single source of
 * truth for that team so the gym, the evaluator, the MCTS determinizer and the
 * ladder bot cannot drift apart.
 *
 * The roster itself is pre-registered and must not be changed; see
 * `docs/BATTLE-FORMATS.md` → THE FIXED ROSTER and `MILESTONES.md` → M12.
 *
 * @license MIT
 */

import * as fs from 'fs';
import * as path from 'path';
import { Teams, type PokemonSet } from '../teams';

/** Repo-root-relative path of the pre-registered M12 roster. */
export const DEFAULT_ROSTER_FILE = 'config/rosters/gen1ou-standard.txt';

/** The format the fixed roster is legal in and M12 is run on. */
export const FIXED_ROSTER_FORMAT = 'gen1ou';

/**
 * Repo root, resolved from this file's location rather than `process.cwd()` —
 * the gym bridge and eval scripts are launched from several directories.
 * `dist/sim/tools/roster.js` and `sim/tools/roster.ts` are both three levels
 * below the root.
 */
function repoRoot(): string {
	return path.resolve(__dirname, '..', '..', '..');
}

const packedCache = new Map<string, string>();

/**
 * Load a packed team string. `file` may be absolute or repo-root-relative;
 * it defaults to the pre-registered M12 roster.
 *
 * Throws rather than falling back to a generated team: a silent fallback would
 * mean training on random teams while every doc says otherwise, which is
 * exactly the kind of confound this project has been burned by twice.
 */
export function loadRoster(file: string = DEFAULT_ROSTER_FILE): string {
	const resolved = path.isAbsolute(file) ? file : path.join(repoRoot(), file);
	const cached = packedCache.get(resolved);
	if (cached !== undefined) return cached;

	let raw: string;
	try {
		raw = fs.readFileSync(resolved, 'utf8');
	} catch {
		throw new Error(`Fixed roster not found: ${resolved} (looked up from '${file}')`);
	}

	const packed = raw.trim();
	if (!packed) throw new Error(`Fixed roster file is empty: ${resolved}`);

	const sets = Teams.unpack(packed);
	if (!sets) throw new Error(`Fixed roster is not a valid packed team: ${resolved}`);
	if (sets.length !== 6) {
		throw new Error(`Fixed roster must have 6 Pokemon, got ${sets.length}: ${resolved}`);
	}

	packedCache.set(resolved, packed);
	return packed;
}

/** The roster as `PokemonSet`s. Returns deep copies — callers mutate sets. */
export function rosterSets(file: string = DEFAULT_ROSTER_FILE): PokemonSet[] {
	const sets = Teams.unpack(loadRoster(file));
	if (!sets) throw new Error(`Fixed roster is not a valid packed team: ${file}`);
	return sets.map(set => JSON.parse(JSON.stringify(set)) as PokemonSet);
}

/**
 * The roster a battle in `formatid` is being played with, or `undefined` for
 * random-team formats.
 *
 * This exists so the MCTS determinizer defaults to the *right* behaviour: a
 * fixed-roster format has no hidden team composition to sample, and falling
 * through to the random generator would hand search a bench the opponent cannot
 * have. `override` takes precedence — pass a packed team to use a different one,
 * or `null` to force generator sampling.
 */
export function rosterForFormat(
	formatid: string, override?: string | null
): PokemonSet[] | undefined {
	if (override === null) return undefined;
	if (override) {
		const sets = Teams.unpack(override);
		if (!sets) throw new Error('rosterForFormat: override is not a valid packed team');
		return sets;
	}
	if (formatid !== FIXED_ROSTER_FORMAT) return undefined;
	try {
		return rosterSets();
	} catch {
		// No roster file (e.g. a bare upstream checkout). Fall back to sampling
		// rather than breaking gen1ou battles outright.
		return undefined;
	}
}
