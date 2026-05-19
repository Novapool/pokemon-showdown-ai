/**
 * Battle Evaluation Harness
 * Pokemon Showdown - http://pokemonshowdown.com/
 *
 * Runs N battles between two AI factories and returns aggregate stats.
 * Supports up to 50 concurrent battles.
 *
 * @license MIT
 */

import { BattleStream, getPlayerStreams } from '../battle-stream';
import { PRNG } from '../prng';
import type { PRNGSeed } from '../prng';
import { RandomPlayerAI } from './random-player-ai';
import type { ObjectReadWriteStream } from '../../lib/streams';

export type AIFactory = (stream: ObjectReadWriteStream<string>, seed: PRNGSeed) => RandomPlayerAI;

export interface EvalOptions {
	numBattles: number;
	concurrency: number;
	format: string;
	p1Factory: AIFactory;
	p2Factory: AIFactory;
	seed?: PRNGSeed;
}

export interface EvalResult {
	/** 0–1 */
	p1WinRate: number;
	/** 0–1 */
	p2WinRate: number;
	drawRate: number;
	avgTurns: number;
	/** avg fraction of opponent team KO'd by P1 (opponent fainted / 6) */
	p1KORatio: number;
	/** avg fraction of opponent team KO'd by P2 (opponent fainted / 6) */
	p2KORatio: number;
	totalBattles: number;
	durationMs: number;
}

interface BattleStats {
	winner: 'p1' | 'p2' | 'draw';
	turns: number;
	/** number of P2's Pokemon fainted (P1 KO'd them) */
	p1KOs: number;
	/** number of P1's Pokemon fainted (P2 KO'd them) */
	p2KOs: number;
}

/**
 * Runs at most `concurrency` tasks at a time and resolves with all results
 * in the same order as `tasks`.
 */
async function runPool<T>(tasks: Array<() => Promise<T>>, concurrency: number): Promise<T[]> {
	const results: T[] = new Array(tasks.length);
	let nextIndex = 0;

	async function worker(): Promise<void> {
		while (nextIndex < tasks.length) {
			const index = nextIndex++;
			results[index] = await tasks[index]();
		}
	}

	const workers: Array<Promise<void>> = [];
	const poolSize = Math.min(concurrency, tasks.length);
	for (let i = 0; i < poolSize; i++) {
		workers.push(worker());
	}
	await Promise.all(workers);
	return results;
}

/** Creates a Gen5-compatible PRNGSeed string from four numbers. */
function makeSeed(a: number, b: number, c: number, d: number): PRNGSeed {
	return `${a},${b},${c},${d}` as PRNGSeed;
}

/** Derives a per-battle seed from a base PRNG, advancing it by 4 numbers per call. */
function deriveSeed(prng: PRNG): PRNGSeed {
	return [
		prng.random(2 ** 16),
		prng.random(2 ** 16),
		prng.random(2 ** 16),
		prng.random(2 ** 16),
	].join(',') as PRNGSeed;
}

async function runSingleBattle(
	format: string,
	p1Factory: AIFactory,
	p2Factory: AIFactory,
	battleSeed: PRNGSeed,
	p1AISeed: PRNGSeed,
	p2AISeed: PRNGSeed,
	p1TeamSeed: PRNGSeed,
	p2TeamSeed: PRNGSeed,
): Promise<BattleStats> {
	const battleStream = new BattleStream();
	const streams = getPlayerStreams(battleStream);

	const p1 = p1Factory(streams.p1, p1AISeed);
	const p2 = p2Factory(streams.p2, p2AISeed);

	void p1.start();
	void p2.start();

	const initMessage =
		`>start ${JSON.stringify({ formatid: format, seed: battleSeed })}\n` +
		`>player p1 ${JSON.stringify({ name: 'Bot1', seed: p1TeamSeed })}\n` +
		`>player p2 ${JSON.stringify({ name: 'Bot2', seed: p2TeamSeed })}`;

	void streams.omniscient.write(initMessage);

	let winner: 'p1' | 'p2' | 'draw' = 'draw';
	let turns = 0;
	let p1KOs = 0;
	let p2KOs = 0;

	for await (const chunk of streams.omniscient) {
		for (const line of chunk.split('\n')) {
			if (line.startsWith('|win|')) {
				const winnerName = line.slice('|win|'.length).trim();
				if (winnerName === 'Bot1') {
					winner = 'p1';
				} else if (winnerName === 'Bot2') {
					winner = 'p2';
				} else {
					winner = 'draw';
				}
			} else if (line.startsWith('|turn|')) {
				const turnStr = line.slice('|turn|'.length).trim();
				const n = parseInt(turnStr, 10);
				if (!isNaN(n) && n > turns) turns = n;
			} else if (line.startsWith('|faint|')) {
				const target = line.slice('|faint|'.length).trim();
				if (target.startsWith('p1a:')) {
					// P1's Pokemon fainted → P2 KO'd it
					p2KOs++;
				} else if (target.startsWith('p2a:')) {
					// P2's Pokemon fainted → P1 KO'd it
					p1KOs++;
				}
			}
		}
	}

	await streams.omniscient.writeEnd();

	return { winner, turns, p1KOs, p2KOs };
}

export async function evaluate(options: EvalOptions): Promise<EvalResult> {
	const {
		numBattles,
		concurrency,
		format,
		p1Factory,
		p2Factory,
		seed = makeSeed(1, 2, 3, 4),
	} = options;

	const startMs = Date.now();
	const masterPRNG = new PRNG(seed);

	const tasks: Array<() => Promise<BattleStats>> = [];
	for (let i = 0; i < numBattles; i++) {
		// Derive 5 seeds per battle: battleSeed, p1AISeed, p2AISeed, p1TeamSeed, p2TeamSeed
		const battleSeed = deriveSeed(masterPRNG);
		const p1AISeed = deriveSeed(masterPRNG);
		const p2AISeed = deriveSeed(masterPRNG);
		const p1TeamSeed = deriveSeed(masterPRNG);
		const p2TeamSeed = deriveSeed(masterPRNG);

		tasks.push(() =>
			runSingleBattle(
				format,
				p1Factory,
				p2Factory,
				battleSeed,
				p1AISeed,
				p2AISeed,
				p1TeamSeed,
				p2TeamSeed,
			)
		);
	}

	const results = await runPool(tasks, concurrency);
	const durationMs = Date.now() - startMs;

	let p1Wins = 0;
	let p2Wins = 0;
	let draws = 0;
	let totalTurns = 0;
	let totalP1KOs = 0;
	let totalP2KOs = 0;

	for (const r of results) {
		if (r.winner === 'p1') p1Wins++;
		else if (r.winner === 'p2') p2Wins++;
		else draws++;
		totalTurns += r.turns;
		totalP1KOs += r.p1KOs;
		totalP2KOs += r.p2KOs;
	}

	const total = results.length;
	return {
		p1WinRate: p1Wins / total,
		p2WinRate: p2Wins / total,
		drawRate: draws / total,
		avgTurns: totalTurns / total,
		p1KORatio: totalP1KOs / (total * 6),
		p2KORatio: totalP2KOs / (total * 6),
		totalBattles: total,
		durationMs,
	};
}

export async function evaluateVsRandom(
	agentFactory: AIFactory,
	numBattles = 1000,
	concurrency = 50,
): Promise<EvalResult> {
	return evaluate({
		numBattles,
		concurrency,
		format: 'gen1randombattle',
		p1Factory: agentFactory,
		p2Factory: (stream, seed) => new RandomPlayerAI(stream, { seed }),
		seed: makeSeed(1, 2, 3, 4),
	});
}
