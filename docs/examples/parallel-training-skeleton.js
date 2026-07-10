/**
 * Canonical parallel-battle-simulation reference for the Pokemon trainer ML training project.
 *
 * This file packages the `runConcurrent()` worker-pool pattern and the seed-derivation/XOR-decorrelation
 * pattern for reproducible, high-throughput battle simulation. Both patterns were originally demonstrated
 * in the repo-root `simulate.js`, which is an unrelated script for a separate project (gym-leader simulation;
 * see root CLAUDE.md). This file supersedes it as the reference for this ML training project's concurrency approach.
 */

// Run from repo root after ./build: node docs/examples/parallel-training-skeleton.js

'use strict';
const { BattleStream, getPlayerStreams, Teams, PRNG } = require('../../dist/sim/index');
const { RandomPlayerAI } = require('../../dist/sim/tools/random-player-ai');
const path = require('path');
const fs = require('fs');

const CONCURRENCY = 50;
const TOTAL_BATTLES = 10000;
const FORMAT = 'gen7randombattle';

async function runBattle(seed) {
  const battleStream = new BattleStream();
  const streams = getPlayerStreams(battleStream);

  const p1 = new RandomPlayerAI(streams.p1, { seed: [seed[0] ^ 0xAAAA, seed[1], seed[2], seed[3]] });
  const p2 = new RandomPlayerAI(streams.p2, { seed: [seed[0] ^ 0x5555, seed[1], seed[2], seed[3]] });
  void p1.start();
  void p2.start();

  void streams.omniscient.write(
    `>start ${JSON.stringify({ formatid: FORMAT, seed })}\n` +
    `>player p1 ${JSON.stringify({ name: 'p1' })}\n` +
    `>player p2 ${JSON.stringify({ name: 'p2' })}`
  );

  let winner = null, turns = 0;
  try {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|win|')) winner = line.slice(5).trim();
        if (line.startsWith('|turn|')) turns = parseInt(line.split('|')[2]);
      }
    }
  } catch (_) {}
  try { streams.omniscient.writeEnd(); } catch (_) {}

  return { winner, turns, seed };
}

async function runConcurrent(tasks, limit) {
  const results = new Array(tasks.length);
  let idx = 0;
  async function worker() {
    while (idx < tasks.length) { const i = idx++; results[i] = await tasks[i](); }
  }
  const workers = [];
  for (let i = 0; i < Math.min(limit, tasks.length); i++) workers.push(worker());
  await Promise.all(workers);
  return results;
}

async function main() {
  const outPath = path.join(__dirname, 'battles.csv');
  const out = fs.createWriteStream(outPath);
  out.write('battle_id,winner,turns\n');

  const tasks = Array.from({ length: TOTAL_BATTLES }, (_, i) => {
    const seed = [i + 1, (i * 3 + 7) & 0xFFFF, (i * 7 + 13) & 0xFFFF, (i * 11 + 17) & 0xFFFF];
    return () => runBattle(seed);
  });

  const results = await runConcurrent(tasks, CONCURRENCY);
  for (let i = 0; i < results.length; i++) {
    const { winner, turns } = results[i];
    out.write(`${i + 1},${winner ?? 'tie'},${turns}\n`);
  }
  out.end();
  console.log(`Done: ${TOTAL_BATTLES} battles → ${outPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
