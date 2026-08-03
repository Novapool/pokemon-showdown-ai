# Evaluation Methodology v2

**The rules for measuring whether a checkpoint is better. Follow this or the
result does not count.**

M9 Phase 1 deliverable. `docs/LADDER-MEASUREMENT.md` is the *diagnosis* — what
went wrong in M6–M8. This is the *protocol*. Read that one first if you have not.

Last verified: **2026-07-31** (all numbers below reproduced on that date).

---

## The three rules

1. **Every ladder run gets a `--run-id`.** It is the arm label. Without it the
   run cannot be separated from the rest of the log and is unusable for an A/B.
2. **Every A/B uses fresh, dedicated accounts — one per arm.** Never reuse an
   account across arms or milestones. Account-level statistics (Elo, GXE) are
   cumulative and contaminate everything after them.
3. **Win rates come from `scripts/ladder_analysis.py`, never from the account
   JSON.** The per-battle CSV is the only per-run record we have.

Anything gated on a number that did not come out of that script, at a sample
size pre-registered *before* the run, is not evidence.

---

## Part 1 — What the instrument actually does

### The corrected diagnosis: it was sample size, not drift

`LADDER-MEASUREMENT.md` reported the same M7 checkpoint swinging **42% (n=50) →
27% (n=100)** and attributed it to "ladder drift." Re-analysing the per-battle
CSV in M9 Phase 1 shows that reading was wrong in two ways, and the correction
changes what the protocol needs to defend against.

**First, the run history is not what was recorded.** Segmenting
`ladder_results.csv` on wall-clock gaps recovers the actual sessions:

```
$ python3 scripts/ladder_analysis.py --since 2026-07-20

07-20 04:10 -> 10:35   n= 95   29W   30.5%   95% CI [22.2, 40.4]
07-22 18:34 -> 00:47   n= 89   29W   32.6%   95% CI [23.7, 42.9]
07-23 02:04 -> 05:39   n= 50   12W   24.0%   95% CI [14.3, 37.4]   <-- never reported
07-23 06:40 -> 10:04   n= 50   21W   42.0%   95% CI [29.4, 55.8]   <-- the cited "follow-up"
07-30 21:58 -> 04:34   n=100   27W   27.0%   95% CI [19.3, 36.4]
```

There were **two** 50-game sessions on 7/23, back-to-back, same checkpoint: one
scored 24%, the next 42%. Only the 42% entered the milestone record, where it
was read as evidence of progress. That is a selection effect layered on top of a
sampling problem, and it is the single most damaging measurement failure in the
project's history.

**Second, there is no measurable drift term.** Testing the five M7-era sessions
for heterogeneity:

```
session heterogeneity: chi2=4.85 df=4 p=0.303  phi=1.21
```

Session-to-session variation is **fully consistent with binomial sampling noise**
(p=0.30, and φ≈1.2 is not distinguishable from 1.0 at this sample size). There is
no extra day-to-day variance component to design against. The 42%→27% swing did
not need a drift explanation — at n=50 the 95% CI is **±13pp wide on its own**.

**What this changes:** the paired-concurrent design (M9 Phase 3) was justified as
"cancelling ladder drift." That justification is not supported by our data. Keep
the design anyway — it is free, it guards against drift we cannot yet rule out at
this n, and it equalises the opponent pool — but **do not expect it to reduce the
required sample size.** The binding constraint is n, and it always was.

### Established M7 baseline

Pooling every rated game the M7 checkpoint has played on the ladder:

```
$ python3 scripts/ladder_analysis.py --since 2026-07-20
all games   n=387   118W   30.5%   95% CI [26.1, 35.3]
```

**M7's true ladder win rate in gen1randombattle is 30.5%, ±4.6pp.** This is
already at the n≈350 target and is the best-powered ladder number the project
has. Use it as the standing control prior — but note it was collected on a
long-lived account across a rating climb, so it is a prior, not a substitute for
a concurrent control arm.

### Precision, so nobody has to guess again

95% CI half-width for a single arm at p≈0.30:

| n | ± |
|---:|---:|
| 50 | ±12.3pp |
| 100 | ±8.8pp |
| 200 | ±6.3pp |
| 350 | ±4.8pp |
| 500 | ±4.0pp |
| 1000 | ±2.8pp |

**Every ladder run in project history before M9 was n≤100, i.e. ±9pp or worse.**
No conclusion about a sub-10pp effect was ever available from them.

---

## Part 2 — Required sample sizes

Two arms, 80% power, α=0.05 two-sided. Regenerate any time with
`python3 scripts/ladder_analysis.py --power`.

### Ladder (p₁ ≈ 0.30, ~4 min/battle)

| Effect to detect | Games per arm | Hours per arm |
|---|---:|---:|
| +2pp | 8,394 | 560 |
| +3pp | 3,763 | 251 |
| +5pp | 1,377 | 92 |
| +7.5pp | 623 | 42 |
| **+10pp** | **356** | **24** |
| +15pp | 163 | 11 |
| +20pp | 93 | 6 |

**M9 pre-registers +10pp (~350/arm).** Anything smaller costs days of ladder per
arm and is declared **not measurable on the ladder by this project** — such
effects must be judged on bot evals instead.

### Bot evals — and a correction to the M9 Phase 2 gate

Measured throughput on the M4 MacBook, 2026-07-31:

| Eval type | Rate | 2,000 battles |
|---|---|---|
| Raw policy (`--model ppo`, 8 envs) | ~27 battles/s | **~75 seconds** |
| Tuned MCTS (`--model mcts --sims 100 --determinizations 1`) | ~4.5 s/battle | **~2.5 hours** |

Games per arm to detect an effect on bot opponents:

| Effect | vs Random (p₁=0.93) | vs DamageFirst (p₁=0.842) |
|---|---:|---:|
| +2pp | 2,213 | 4,948 |
| +3pp | 906 | 2,137 |
| +5pp | 269 | 723 |

**The M9 Phase 2 gate as written in `docs/MILESTONES-ARCHIVE.md` — "≥+2pp on both bot opponents
at n=500" — is underpowered by 4–10×**, the same defect as M8's "+3pp at n=200".
It cannot pass reliably even if the candidate is genuinely +2pp better.

**Corrected standing gate for bot-eval A/Bs:**

- **Raw-policy evals: use n = 2,000 per arm.** It costs 75 seconds. There is no
  justification for ever running fewer. This resolves +3pp vs Random and about
  +4pp vs DamageFirst.
- **MCTS evals: use n = 1,000 per arm** (~75 min/arm), resolving ~+4pp vs Random
  and ~+6pp vs DamageFirst. Go to 2,000 (~2.5 h/arm) when the decision is
  expensive — e.g. deciding whether to spend two days of ladder.
- **Report the difference with a CI**, not two point estimates compared by eye.
  Use `scripts/bot_eval_ab.py` — the bot-eval counterpart to
  `ladder_analysis.py`, sharing its Wilson/Newcombe implementations:
  ```bash
  python3 scripts/bot_eval_ab.py --arm base=1697/5000 --arm cand=1977/5000 --gate 3
  ```
- A ±2pp gate on DamageFirst is not reachable at any sample size we would
  actually run (4,948/arm ≈ 6 h MCTS). Do not pre-register one.

#### The required n depends on the baseline win rate — n=2,000 is not universal

The n=2,000 above is derived at **p₁=0.93**, the M7 operating point, where
binomial variance is small. Variance is maximised at p=0.5, so a *mid-range*
agent needs far more games to resolve the same effect:

| Baseline p₁ | n/arm for +3pp | n/arm for +5pp |
|---|---:|---:|
| 0.93 (M7 vs Random) | 906 | 269 |
| 0.84 (M7 vs DamageFirst) | 2,137 | 723 |
| 0.55 | 4,286 | 1,534 |
| **0.42** (a BC checkpoint) | **4,286** | **1,550** |

At p₁≈0.42, n=2,000 resolves only about ±4.5pp — so an A/B on BC checkpoints
run at the "standing" n=2,000 would be underpowered in exactly the way this
document exists to prevent. **M9 Phase 2a ran at n=5,000/arm** (~3 min/arm at
27 battles/s) for this reason.

**Rule:** look up n against the arm's *actual* win rate before running, with
`python3 scripts/bot_eval_ab.py --power --baseline-p <p>`. n=2,000 for strong
agents near the ceiling; **n=5,000 for anything in the 0.3–0.7 band.**

#### Don't ship the sweep maximum — it is biased upward

Picking the best of 20 noisy checkpoint readings is a multiple-comparisons
trap: the winner is partly *lucky*, so its next measurement regresses. Measured
on M9 2d, 2026-08-01:

| checkpoint | sweep (n=500) | confirmation (n=2,000) |
|---|---:|---:|
| 4.50M — the sweep's best | **69.0%** | **64.8%** (−4.2pp) |
| 5.00M final | 63.0% | **67.9%** |

The sweep's top pick lost 4.2pp on re-measurement and ended up **3.2pp below
the final checkpoint** [+0.2, +6.1] — i.e. the sweep ranked them backwards.

**Rule:** treat the sweep as a *coarse filter and a training-curve sanity
check*, never as a selection mechanism. Confirm the **final** checkpoint (the
shipping convention) plus any candidate with an independent reason to be
better, and compare them at n=2,000. If a sweep pick must be used, its sweep
number is not a valid estimate of its strength — re-measure it.

#### Eval CIs do not cover training-run variance — but that variance is small

A CI from `bot_eval_ab.py` covers **eval sampling noise only**. When the two
arms are checkpoints from two *different training runs*, the comparison also
contains a training-seed term that no number of eval battles can shrink.

**Measured 2026-08-01.** `train.py` does no seeding at all (no `manual_seed`,
no `np.random.seed`, anywhere in the trainer, agent or gym clients), so
re-running an identical command is an independent draw. Re-running M7's 5M-step
recipe reproduced it to **−0.6pp vs Random [−3.4, +2.2]** and **−0.3pp vs
DamageFirst** — *and across a backend change* (`mps` → `cuda`). Run-to-run
spread is therefore **under a point**, and single-run A/Bs in this project are
sound for effects of roughly 3pp and up.

Caveats worth keeping: this is **one replication pair**, which bounds the
spread as small without putting a tight interval on it — do not quote "PPO
variance is 0.6pp". A ~70-minute replication is cheap insurance before
declaring any *surprising* result, and it is what turned M9 Phase 2c from
"confounded three ways" into a finding.

**Standing rule: run both arms of an A/B on the same machine.** M9 Phase 2c
compared an `mps`-trained checkpoint against a `cuda`-trained one for no
reason. It turned out not to matter — which is now a measurement rather than
an assumption, but the comparison was avoidable.

---

## Part 3 — Runbook

### 3.1 Set up an arm

One fresh account per arm, registered before the run:

1. Register the account on play.pokemonshowdown.com. Naming convention:
   `<milestone><phase><arm>`, e.g. `m9p3ctl`, `m9p3cand`.
2. Write credentials to `config/showdown_login_<arm>.txt`:
   ```
   username: m9p3ctl
   password: <password>
   ```
   (This file is gitignored; it keeps credentials off argv and out of the
   process list.)
3. Record the account's starting state (should be a clean slate) from
   `https://pokemonshowdown.com/users/<name>.json`.

**Never point two arms at one account, and never reuse `novapool`** — it carries
506+ games of M6–M8 history.

### 3.2 Run an arm

**The user runs live ladder sessions, not Claude.** The bot has reconnect logic
(M8 Phase 0), but it has repeatedly dropped when launched from Claude's sandboxed
background execution. Claude hands over the command.

```bash
node tools/ladder-bot/ladder-bot.js \
  --login-file config/showdown_login_m9p3ctl.txt \
  --checkpoint models/ppo/checkpoints/v3/ppo_step_5000002_final.pt \
  --run-id m9p3-control \
  --battles 360 \
  --mcts
```

360, not 350: the +10pp target needs **356** games per arm, and running exactly
350 leaves the analysis reporting that it could only resolve 15pp.

⚠️ **Decision rule (2026-08-01):** the bot now plays the policy's **argmax** by
default; `--sample` restores the pre-2026-08-01 sampling. Every ladder number
recorded before that date — 30.5%, 28.0%, all per-session reads — was scored
while sampling. **Do not pool across the change**, and state the decision rule
in the writeup. The banner prints `policy=greedy|sampled` at startup; if you are
unsure what an arm ran, that line is the record.

⚠️ **`--mcts` masks the decision rule.** Search returns an argmax over visit
counts, so `--greedy`/`--sample` only reach the ~20% of decisions that fall back
to the raw policy (force switches, locked states — `ladder-bot.js:291`). **Any
A/B on the decision rule must run both arms without `--mcts`**, or it tests a
fifth of the change and reads as a null for the wrong reason. This is not
hypothetical: run `m7-greedy` (2026-08-02) did exactly that.

⚠️ **Ladder history lives in two CSVs.** `ladder_analysis.py` defaults to
`ladder_results.csv` only, and the M9 schema change rotated everything before it
into `ladder_results.pre-m9.csv`. A bare invocation silently drops 507 games.
Pass both paths. The old file also has **no `opp_rating` column**, so the
opponent-Elo confound check cannot be run against pre-M9 arms — a historical
comparison to them is confounded in a way no analysis can repair.

- `--run-id` is **mandatory** — it is the arm label written to every CSV row.
- `--battles` is the **absolute** target. The run is resumable: re-running the
  identical command after a crash picks up where it left off (progress lives in
  `data/replays/self_ladder/run_<id>.json`).
- For a paired A/B, alternate the two arms in blocks within the same sessions
  (e.g. 25 games control, 25 games candidate, repeat) so both meet the same
  ladder pool over the same hours. Do not run one arm to completion and then the
  other.

### 3.3 Analyse

```bash
# per-session breakdown, pooled estimate, heterogeneity check
python3 scripts/ladder_analysis.py --run m9p3-control

# the primary endpoint: paired difference with 95% CI
python3 scripts/ladder_analysis.py --arm m9p3-control --arm m9p3-candidate
```

The paired output gives the difference, its CI, the pre-registered verdict, an
honest power statement for the n actually collected, and each arm's mean opponent
Elo. **If the two arms' mean opponent Elo differ materially, the difference is
confounded** — say so in the writeup rather than burying it.

### 3.4 Report

Every ladder result gets reported in this shape, and no other:

> **Arm A** (`m9p3-control`, M7 `ppo_step_5000002_final.pt`, account `m9p3ctl`):
> 108/360 = 30.0% [25.5, 35.0], mean opponent Elo 1082.
> **Arm B** (`m9p3-candidate`, …): 137/360 = 38.1% [33.3, 43.1], mean opponent
> Elo 1079.
> **Difference: +8.1pp, 95% CI [+1.1, +14.9].** Pre-registered gate was
> +10pp. The CI excludes 0, so a real difference is indicated, but the point
> estimate is under the bar: **inconclusive at the pre-registered effect size.**

Required elements: per-arm n, wins, rate, CI, mean opponent Elo; the difference
with its CI; the pre-registered gate; the verdict against that gate.

---

## Part 4 — Interpretation rules

**Pre-register before the run, not after.** Write the effect size, the sample
size, and the decision rule into IN-PROGRESS.md *before* the first battle. A gate
chosen after seeing the data is not a gate.

**An inconclusive result at adequate power is a finding.** At 350/arm,
"CI includes 0" means *the effect is smaller than 10pp*. Record that and stop
spending on the direction. Do not re-run bigger hoping for a different answer —
that is how three milestones produced three inconclusive readings.

**Never compare across runs.** Only within a paired design, or against the
pooled M7 baseline with both CIs stated.

**Report every session you ran.** The 7/23 failure was not a statistics error —
one of two sessions simply was not written down. If a session is excluded, the
exclusion rule must have been stated in advance.

**Do not gate on GXE or Elo.** They are secondary, reportable only when accounts
are fresh and matched on n in the same period. GXE moved 32.9% → 32.9% across an
entire 100-game run; it has no resolving power at our sample sizes.

### The account JSON

`https://pokemonshowdown.com/users/<name>.json` gives `elo`, `gxe`, `rpr`,
`rprd`, `w`, `l`. Capture it at run boundaries for the record. `ladder-bot.js`
does not read it. It is context, never an endpoint.

---

## Part 5 — The instrument

### Per-battle CSV

`data/replays/self_ladder/ladder_results.csv`, one row per finished battle:

```
timestamp,run_id,account,checkpoint,room,opponent,opp_rating,own_rating,rated,result,decisions,max_latency_ms
```

`opp_rating`/`own_rating` are the pre-battle ladder Elos from the battle's
`|player|` lines (present on rated games only). `run_id` is the arm label.

The pre-M9 log (M6–M8, 517 rows, no arm labels or ratings) is retired
automatically to `ladder_results.pre-m9.csv` the first time the bot writes a new
row. `ladder_analysis.py` reads both schemas.

### Analysis script

`scripts/ladder_analysis.py` — Wilson intervals, Newcombe interval on the
difference, session segmentation, chi-square heterogeneity/overdispersion, power
tables. No dependencies beyond the standard library.

```bash
python3 scripts/ladder_analysis.py --help
python3 scripts/ladder_analysis.py --power          # sample sizes, no data needed
python3 scripts/ladder_analysis.py --since 2026-07-20
python3 scripts/ladder_analysis.py --arm A --arm B  # paired endpoint
```

Regression tests for the logging path: `test/tools/ladder-results.test.js`
(`npx mocha --no-config test/tools/ladder-results.test.js`).

---

## Checklist

Use the `ml-experiment-reviewer` agent (`.claude/agents/ml-experiment-reviewer.md`)
to check a pre-registration or result against this document before trusting it.

Before a run:
- [ ] Fresh account per arm, credentials in `config/showdown_login_<arm>.txt`
- [ ] Effect size, n/arm, and decision rule written into IN-PROGRESS.md
- [ ] n/arm checked against the table in Part 2
- [ ] `--run-id` chosen for each arm
- [ ] Alternating block schedule agreed if paired

After a run:
- [ ] `ladder_analysis.py --arm … --arm …` output pasted into the writeup
- [ ] Every session reported, including the bad ones
- [ ] Mean opponent Elo compared across arms
- [ ] Verdict stated against the *pre-registered* gate
- [ ] Account JSON captured for the record
