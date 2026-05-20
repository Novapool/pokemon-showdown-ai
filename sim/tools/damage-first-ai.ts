/**
 * DamageFirstAI — always picks the highest base-power available move.
 * Extends RandomPlayerAI; inherits all switch/force-switch/team-preview logic.
 * Ties broken randomly using this.prng.sample().
 *
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * @license MIT
 */

import { Dex } from '..';
import { RandomPlayerAI } from './random-player-ai';

function getBasePower(moveName: string): number {
	try {
		const bp = Dex.mod('gen1').moves.get(moveName).basePower;
		return Number.isFinite(bp) ? bp : 0;
	} catch {
		return 0;
	}
}

export class DamageFirstAI extends RandomPlayerAI {
	protected override chooseMove(
		active: AnyObject,
		moves: { choice: string; move: AnyObject }[]
	): string {
		let best: { choice: string; move: AnyObject }[] = [];
		let bestPower = -1;

		for (const m of moves) {
			const bp = getBasePower(m.move.move);
			if (bp > bestPower) {
				bestPower = bp;
				best = [m];
			} else if (bp === bestPower) {
				best.push(m);
			}
		}

		// Fallback: if best is somehow empty, pick randomly from all moves
		if (best.length === 0) return this.prng.sample(moves).choice;
		return this.prng.sample(best).choice;
	}
}
