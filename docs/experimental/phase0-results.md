# Phase 0 Results — Non-Greedy Decision Probe

**Run:** 2026-08-03 · Mac · ~5 s · no training
**Script:** `scripts/phase0_nongreedy_probe.py` · **CSV:** `phase0_results.csv`
**Checkpoints:** `models/checkpoints/bc_mlp_gen1_v3_h512.pt` (BC, H=512),
`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt` (M7 PPO, H=128)

Both checkpoints read the **same v3 observation**. That is what makes the
comparison below informative.

## Method

Held-out validation shards only (`shard-0000` per format, the same split
`bc_pretrain_mlp.py` holds out), same record filter as BC training
(rating ≥ 1300, tournaments kept). **Non-greedy subset:** positions where a
rated human chose a 0-BP status move or a switch while a ≥90 BP damaging move
was legal. Legal-action mask is derived from the observation (move present via
the accuracy dim, not disabled, PP > 0; bench token alive and not fainted) —
without it an unmasked argmax can "choose" impossible actions and deflate
accuracy. Sanity check: BC scores **55.20%** on the full randbats set against
the **55.5%** on record, so the pipeline reproduces the known number.

## Headline numbers

| | full top-1 | non-greedy subset top-1 | gap (battle-clustered CI) |
|---|---:|---:|---|
| **randbats** BC | 55.20% (n=60,766) | 37.71% (n=19,078) | **+17.50pp** [+16.71, +18.33] |
| **randbats** PPO | 43.50% | 11.51% | **+31.99pp** [+31.29, +32.72] |
| **gen1ou** BC | 55.48% (n=28,852) | 43.45% (n=9,938) | **+12.03pp** [+10.75, +13.41] |
| **gen1ou** PPO | 37.29% | 11.91% | **+25.37pp** [+24.26, +26.43] |

CIs resample whole **battles**, not decisions — decisions within a battle are
correlated, so a Wilson interval on decisions understates width. It made almost
no difference here (effects are large), but the clustered interval is the one
to quote.

## Gate outcome

**Gate 1 — accuracy gap ≥3pp, CI excluding 0: PASSES,** by a wide margin on
both formats and both checkpoints.

**Gate 2 — greediness rate >50%: DOES NOT PASS as literally written.**

| pre-registered metric (agent picks *the single highest-BP* slot) | randbats | gen1ou |
|---|---:|---:|
| BC | 24.69% | 18.65% |
| PPO | **50.57%** [49.86, 51.28] | **45.79%** [44.82, 46.77] |

PPO on gen1ou is 45.8%, below the 50% bar; randbats is 50.6% with a CI that
straddles 50. By the letter of the pre-registration, the kill criterion fires.

### ⚠️ Amendment, recorded because it was made after seeing the data

The metric was **mis-specified**, and I am the one who specified it. "Greedy"
does not mean "picks the single highest-BP move" — when three ≥90 BP moves are
legal, an agent that always attacks but spreads across them scores low on that
metric while being maximally greedy in the sense that matters. The broader
measures, **post-hoc and labelled as such**:

| on the non-greedy subset | randbats BC | randbats PPO | gen1ou BC | gen1ou PPO |
|---|---:|---:|---:|---:|
| picked *any* ≥90 BP move | 31.61% | **63.60%** | 24.70% | **59.41%** |
| picked *any* move (vs switching) | 73.25% | **93.97%** | 68.73% | **92.05%** |
| switched | 26.75% | **6.03%** | 31.27% | **7.95%** |

On positions where a rated human switched or used a status move, the PPO agent
attacks **~93%** of the time and switches **~6–8%**. The phenomenon Gate 2 was
written to detect is present and overwhelming; the estimator was wrong, not the
hypothesis. **Recommendation: proceed, with the amendment on the record.**
Anyone auditing this should know the gate was changed after the data was seen.

## The finding that was not being looked for

**RL training, not the observation, is what destroys switching.**

BC and PPO consume the *identical* v3 observation. On the non-greedy subset,
BC switches **26.8%** of the time (randbats); after 5M PPO steps on that same
observation, the agent switches **6.0%**. PPO is also a 11.7pp *worse* imitator
overall (43.50% vs 55.20% full-set top-1).

This is a problem for the strong form of the observation-poverty hypothesis.
Observation poverty predicts a ceiling on how well *any* policy on v3 can play.
It does not predict that two policies on the same v3 observation differ by
20.7pp in switch rate on exactly the positions where humans switch. The
information needed to switch more often is evidently *present enough* in v3 for
BC to use it; RL then trains it away.

Consistent with the standing evidence rather than contradicting it: the
opponent pool is `Random` and `DamageFirst`, **neither of which ever switches**
(`sim/tools/damage-first-ai.ts` sets `move: 1.0`). Against opponents that never
switch, attacking is close to optimal, so PPO is correctly optimising a
training distribution that does not resemble the ladder. That is the same
mechanism M9's sparring-partner experiment tested and found null — but M9
changed *who the opponent was*, not whether the reward rewarded switching.

## What this implies for sequencing

Phase 1 (damage calculator + a heuristic bot that *does* switch) is now the
highest-value next step, and its justification is stronger than when written:
it is not only the missing eval rung, it is a training opponent that punishes
never-switching. The case for Phase 2's full v4 schema retraining is
**weaker** than it was this morning — v3 is demonstrably not the binding
constraint on switch rate.

This also bears on the main line's M10-vs-M11 question: it is independent
evidence for doing M10 (behavioural diagnosis, no retraining) before M11
Phase 1 (schema change, invalidates every checkpoint).

**Caveats.** Top-1 agreement with humans is not play strength — a policy can
diverge from humans and win more. Held-out shards, but a single shard per
format. And this measures the *raw policy*; the ladder bot runs greedy
decoding, which these numbers do reflect, but not MCTS.
