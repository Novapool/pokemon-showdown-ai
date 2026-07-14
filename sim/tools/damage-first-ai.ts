/**
 * DamageFirst AI
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Heuristic opponent for gym training/evaluation (M3.3): always attacks, and
 * always picks the legal move with the highest base power. Never voluntarily
 * switches (move: 1.0), like RandomPlayerAI's default, but strictly stronger
 * than random move selection — a more meaningful benchmark than an opponent
 * that picks Growl as happily as Hyper Beam.
 *
 * Ported from the DamageFirstAI reference in the repo-root simulate.js
 * (an unrelated project's script, used here as a concurrency/AI reference).
 *
 * @license MIT
 */

import type { ObjectReadWriteStream } from '../../lib/streams';
import { Dex } from '..';
import type { ModdedDex } from '../dex';
import { PRNG, type PRNGSeed } from '../prng';
import { RandomPlayerAI } from './random-player-ai';

export class DamageFirstAI extends RandomPlayerAI {
	private readonly _dex: ModdedDex;

	constructor(
		playerStream: ObjectReadWriteStream<string>,
		options: { move?: number, mega?: number, seed?: PRNG | PRNGSeed | null, format?: string } = {},
		debug = false
	) {
		super(playerStream, { move: 1.0, ...options }, debug);
		this._dex = options.format ? Dex.forFormat(options.format) : Dex;
	}

	protected override chooseMove(active: AnyObject, moves: { choice: string, move: AnyObject }[]): string {
		let bestChoice: string | null = null;
		let bestPower = -1;
		for (const m of moves) {
			const moveData = this._dex.moves.get(m.move.move);
			const power = moveData?.basePower || 0;
			if (power > bestPower) {
				bestPower = power;
				bestChoice = m.choice;
			}
		}
		return bestChoice ?? super.chooseMove(active, moves);
	}
}
