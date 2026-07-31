# Ladder Measurement — what our ladder numbers do and don't mean

**Read this before quoting, comparing, or gating on any ladder result.**

Three milestones (M6, M7, M8) were graded on ladder statistics that could not
support the conclusions drawn from them. This documents what went wrong so it
isn't repeated. The *fix* — the protocol future runs should follow — is M9
Phase 1's deliverable; this file is the diagnosis and the standing warnings.

Last verified: **2026-07-31**.

---

## The readings, in order

| Run | Raw win rate | Elo | GXE |
|---|---|---|---|
| M6 baseline | ~13% raw / 21% MCTS | 1017 | 23.9% |
| M7 main (n=100) | 30% | 1034.6 | 28.2% |
| M7 follow-up (n=50, 7/23) | 42% | 1101.4 | 32.9% |
| **M8 Phase 4 (n=100, 7/31)** | **27%** | **1084.0** | **32.9%** |

All four were taken on the **same account** (`novapool`), cumulative.

## Defect 1 — GXE is an account-level cumulative statistic

GXE is computed from the account's Glicko rating over its **entire** history —
506+ games spanning M6, M7 and M8. It is not a per-run measurement and never
was.

- It has enormous inertia. M8 Phase 4 played 100 games and GXE **did not move at
  all**: 32.9% → 32.9%.
- Every reading is contaminated by every prior run, including runs of *different
  checkpoints*.
- Comparing a fresh account's GXE at n≈100 against this account's GXE at n≈506 is
  not a valid comparison. (An M9 Phase 3 draft proposed exactly that and was
  corrected before it shipped.)

**Rule: never gate on GXE from a shared or long-lived account.** GXE is only
meaningful on fresh accounts compared at matched game counts in the same period.

## Defect 2 — raw win rate isn't comparable across runs

The ladder is self-correcting: climbing to Elo 1101 means facing stronger
opponents. A falling win rate at higher Elo can mean the ladder is working, not
that the agent regressed. Any win-rate comparison across runs at different Elo
needs the opponent-strength distribution reported alongside it.

## Defect 3 — the instrument is far noisier than assumed

**M8 Phase 4 ran the same M7 checkpoint as the 7/23 follow-up.** The model did
not change. Results:

```
7/23 follow-up : 42% raw (n=50),  Elo 1101.4
7/31 Phase 4   : 27% raw (n=100), Elo 1084.0   ← same checkpoint
```

A **15pp swing and a 17-point Elo drop at zero true effect** (~1.8 SD). This
retires the "monotonic 23.9 → 28.2 → 32.9 GXE trend" that was recorded on
2026-07-23 as evidence of progress. It was ladder drift.

**Any design assuming ±4–5pp ladder noise is already falsified.** Use the
measured swing.

## Defect 4 — the gates were underpowered

The bot-eval gate (+3pp at n=200) has SE ≈ 3.9pp on the difference: a true +3pp
effect would be detected only ~1/3 of the time. Both M8 Phase 2 arms "failed" a
gate they could not have reliably passed. Their −2.5pp deltas are *inside noise*
— the fine-tune is not established as harmful, it simply failed a point gate.

### Power, for two arms at p ≈ 0.30, 80% power, α = 0.05

| Effect to detect | Games per arm | Ladder time per arm (~4 min/battle) |
|---|---:|---|
| +5pp | ~1,400 | ~4 days |
| **+10pp** | **~350** | **~23 hours** |
| +15pp | ~160 | ~11 hours |

This is the real reason every prior milestone used an underpowered gate. It does
not go away by wishing. **M9 pre-registers +10pp**; effects below that are
declared not measurable on the ladder by this project and must be judged on bot
evals instead.

## What a valid comparison looks like

The design M9 Phase 3 adopts, and the shape any future ladder A/B should take:

1. **Two fresh accounts**, one per arm — never a shared or reused account.
2. **Alternate arms within the same sessions**, so both face the same ladder pool
   over the same period and day-to-day drift cancels in the paired difference.
3. **Primary endpoint = difference in raw win rate between arms**, with a 95% CI.
   Not either arm's absolute GXE.
4. **Report per-arm mean opponent Elo**, so a difference driven by
   opponent-strength asymmetry is visible rather than hidden.
5. **Size for the effect you intend to detect** (table above) and say so in
   advance.
6. Run the **cheap bot-eval A/B first** (n=1000) to decide whether a candidate
   earns ladder time at all.

An inconclusive result at adequate power is a *finding* — "the effect is under
10pp" — not a failure to measure, and should be recorded as such rather than
re-run bigger.

## Operational notes

- **Have the user run live ladder sessions.** The bot has reconnect logic (M8
  Phase 0) but has repeatedly dropped when launched from Claude's sandboxed
  background execution. Local sim work is unaffected. See
  `TRAINING-COMMANDS.md` → "Running the ladder bot".
- Account rating is read from `pokemonshowdown.com/users/<name>.json`
  (`elo`, `gxe`, `rpr`, `rprd`, `w`, `l`) — `ladder-bot.js` does not parse or
  record it, so capture it manually at run boundaries.
- Per-battle results land in `data/replays/self_ladder/ladder_results.csv`
  (`timestamp,room,opponent,rated,result,decisions,max_latency_ms`). This is the
  only per-run record that isn't contaminated by account history — **prefer it
  over the account JSON for per-run win rates**, filtering by timestamp and
  `rated=1`.
