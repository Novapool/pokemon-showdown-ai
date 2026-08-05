# In Progress — Pokemon Showdown AI Training

Last updated: 2026-08-05

---

## 🏁 THE PROJECT IS IN A BOUNDED FINISH (decided 2026-08-05)

**One milestone left: M12 (fixed-team Gen 1 OU), Phases 0–4.** Phase 5 is
optional. When Phase 4 reports its number, the project is done and gets archived
— pass or fail. Full rationale and disposition table: `MILESTONES.md` → PROJECT
STATUS.

**Closed without execution:** M10 (battle log analysis), M11 Phase 1
(observation schema v4), instrumentation debt (`--seed`, per-step logging,
`meta.json`). **Off the critical path:** the M11 h128/h512 eval battery — run it
only if SSH is back and it is genuinely cheap; it changes no decision.

**The rule for this phase: do not add scope.** If M12 Phase 4 fails its gate,
that is the finding — record it and stop. No second roster, no diagnosis pass,
no "one more idea." This project's own standing rules call re-rolling after
seeing the result what it is.

---

## Where We Stand (2026-07-31)

**Shipping agent: still the unchanged M7 checkpoint** (`v3/ppo_step_5000002_final.pt`),
93.0% vs Random / 84.2% vs DamageFirst. Nothing since M7 has beaten it —
including M9 Phase 2c, which came in **6–7pp below M7** despite starting from a
demonstrably better BC checkpoint (see Phase 2 below). **M7 is the Phase 3
candidate as well as the control.**

**M7 re-measured at n=2,000 on 2026-07-31: 69.7% vs Random / 59.4% vs
DamageFirst as a raw policy** (the 93.0/84.2 figures are with tuned MCTS on
top). Its historical raw reading was 70.0% at n=500 — a clean replication, and
the first time any baseline in this project has been re-measured rather than
quoted.

**Four hypotheses have now been tested and closed:**

| Hypothesis | Verdict |
|---|---|
| (b) Obs richness is the constraint | ❌ M8 Phase 1A — Criterion A failed |
| (a) Value head miscalibration is the constraint | ❌ M8 Phase 2 — failed twice, −2.5pp each, distribution mismatch eliminated |
| More gen 1 human data can be scraped | ❌ M9 Phase 2b — archive exhausted, we hold ~all of it |
| Unrated tour games are an untapped pool | ❌ already in training, and 74% of it |

**(c) opponent-pool / data distribution: POSITIVE at the BC stage, NEGATIVE
after RL — and as of 2026-08-01 both halves are confirmed, not confounded.**
Phase 2a is a clean, well-powered win for format alignment; Phase 2c carried
that checkpoint through M7's exact 5M-step recipe and came out **8.3pp below**
a same-machine control. A seed replication (below) reproduced M7's recipe to
within **0.6pp**, ruling out training-seed and backend noise as the
explanation. **The finding is that a better imitator made a worse RL starting
point.** **Read 2a, 2c and the seed replication below before citing any of
them.** M9 Phase 2a trained a BC checkpoint on gen1randombattle replays
only, against the mixed gen1ou+randbats BC that every checkpoint from M5.5
through M7 was built on. Identical recipe, identical schema, only the corpus
differs. At n=5,000 per arm:

| | randbats-only BC | mixed BC | difference (95% CI) |
|---|---|---|---|
| vs Random | **39.5%** | 33.9% | **+5.6pp [+3.7, +7.5]** |
| vs DamageFirst | **33.2%** | 28.6% | **+4.6pp [+2.8, +6.4]** |
| randbats val acc | **54.3%** | 52.7% | +1.6pp (n=60,766) |

Both CIs exclude 0. **The gen1ou half of the BC corpus was diluting
gen1randombattle play, not enriching it** — the opposite of the 2026-07-16
project decision that put it there. Note this reverses no measurement; the
mixed-corpus choice was never A/B'd, it was assumed.

**The blocking problem was measurement, not modeling — and M9 Phase 1 has now
fixed it (✅ 2026-07-31).** Three consecutive inconclusive ladder readings, and
the same checkpoint "swung" 42% → 27% between runs. The re-analysis shows that
swing was **sample size, not drift** (±13pp CI at n=50; session heterogeneity
`phi=1.21, p=0.303`), and that the 42% was **the better of two unreported 50-game
sessions that day** (the other scored 24.0%). Pooling all 387 M7-era rated games
gives the number no single run could see: **M7 = 30.5% on ladder, 95% CI
[26.1, 35.3]** — **22.9% [18.7, 27.6] excluding opponent concessions**, which is
the figure to quote for play strength (see Recently Completed, 2026-08-03).

**👉 For a plain-language, one-screen orientation — what the agent can and
can't do, what's settled, and the live menu of options with a recommendation —
read `docs/WHERE-WE-ARE.md`.** It is deliberately not a timeline; this file is
the timeline. Surface it to the user after a context clear.

**Ground-truth docs — read these before re-proposing anything in the table above:**
- `docs/EVALUATION-METHODOLOGY.md` — **the protocol.** Required sample sizes, the
  runbook, the reporting template. Any result not produced this way doesn't count
- `docs/DATA-INVENTORY.md` — what data exists, what we hold, what BC consumes
- `docs/LADDER-MEASUREMENT.md` — the diagnosis of what went wrong in M6–M8

---

## Current Work

**✅ M12 Phase 0 COMPLETE (2026-08-05) — the roster is locked and
pre-registered.** Full record in `docs/BATTLE-FORMATS.md` → THE FIXED ROSTER.

**Tauros / Chansey / Snorlax / Exeggutor / Starmie / Alakazam**, modal human
move sets, packed team at `config/rosters/gen1ou-standard.txt`. Validated
(`validate-team gen1ou` passes) and smoke-tested 5/5 completed battles.

Chosen from **10,101 local gen1ou replays rated ≥1300** → 16,743 revealed teams.
This is the **rank-#1 most-used exact team** (1,088 teams, 6.50%); the Rhydon
variant was rank #2 (6.10%). Tiebreak: usage rank, plus Rhydon's value is
*defensive positional* (Ground = Thunder Wave immunity) which v3 cannot see since
type effectiveness is encoded offensively only, plus 1,772 vs 237 revealed sets
for BC. Reproducible via `scripts/mine_gen1ou_teams.py` and
`scripts/mine_gen1ou_movesets.py`.

**Two caveats recorded with the selection:**
- **Team win rate was computed, found invalid, and discarded.** Extracting a team
  needs all 6 revealed, and winners often never bring in their 6th: P(win)=44.5%
  given 6 revealed vs 81.2% given 5. The filter selects for *losing* teams. Usage
  counts are unaffected; win rates by roster are not usable and were not used.
- **Expect long mirror matches.** Smoke test ran 66–151 turns (Soft-Boiled /
  Recover / Rest on both sides). Fewer episodes per 5M PPO steps in Phase 3, and
  longer ladder wall-clock if Phase 5 runs.

**Next: Phase 1 (plumbing).** Mostly doable on the Mac. Everything from Phase 2
on waits for home box SSH (see Blockers), which the user restores when home.

**Closed, not deferred:** observation enrichment (v4 schema) and M10. The
earlier note here said v4 was "deferred post-pivot" for re-derivation on OU —
that is now a closure, because the project ends at M12 Phase 4. See
`MILESTONES.md` → M11 for what a future reader should pick up first.

**✅ M11 Phase 0 (reward asymmetry) IS DONE AND SHIPPED** — `applyStallPenalty`
now clips before charging the duration cost, so it applies equally to wins and
losses. `stallPenalty` is settable (`--stall-penalty`), default 0.001 =
historical. `gamma` already discounts terminal reward symmetrically, so
`--stall-penalty 0` is a live experiment, not a fallback. Regression test added
that asserts win/loss duration costs are equal, with the old formula inlined —
the previous test asserted rewards stayed in `[-1,1]`, which **encoded the bug**.
This is the one part of M11 that shipped; it carries forward into M12 unchanged.

**🔬 TRAINED 2026-08-02, UNMEASURED — the two PPO arms.**
`models/ppo/checkpoints/m11_h128/` (151,187 params, 5,000,003 steps) and
`m11_h512/` (**801,299**, 5,000,006), home box, both `cuda`. Identical
otherwise: 5M steps, `selfplay=0.5,damagefirst=0.3,random=0.2`,
`--bc-anchor-coef 0.05`, 200k value warmup, same two pool seeds, same machine,
same commit, **same fixed reward**.

**Why two arms and not one.** The reward fix landed *before* the launch, so
`m9seed` stopped being a valid control — it trained under the buggy reward.
Running only the width arm would have moved width *and* reward together, the
exact confound the "same machine, same code" standing rule exists to prevent.
Two arms give two clean single-variable comparisons for one wall-clock cost:

- `m11_h128` vs **m9seed** → the **reward fix** (width held at 128)
- `m11_h512` vs **m11_h128** → **width** (reward held fixed)

⚠️ **Verify `m9seed`'s final checkpoint still exists on the box before starting.**
Without it the reward comparison is dead and only the width comparison survives.

**Pre-registered gate, with one deliberate widening.** Width gate is **≥+3pp vs
Random with the CI on the difference excluding 0**; DamageFirst reported
alongside but not gated. **n widened from the registered 2,000 to 5,000**
(2026-08-03): these arms land near 0.6–0.7, where 2,000 resolves only ~±4.5pp
against a +3pp gate — underpowered by the project's own rule. Recorded here
rather than quietly changed. Decision rule for the gate is **sampled** (keeps
these comparable to every historical number including M7's 69.7%), with a
`--greedy` read on the winner only.

Expect an **informative null on width** — the observation carries ~25 distinct
values, and 5.3× params bought only +2.8pp BC val accuracy. Either way this does
not gate M12: **the pivot happens regardless**, and the width result only decides
whether M12 budgets a wider net or stays at 128.

**Evidence that carries into the deferred v4 work (M11), not into M12 directly.**
Field observation 2026-08-02, user watching a live ladder game: the agent
switched a Water type in after a faint against an Electric, then spent the
following turn switching it out for a Normal. The encoder explains the shape —
**type effectiveness is encoded offensively only.** `fillV3MoveDims`
(`sim/tools/feature-extractor.ts`) fills dims 77–80 with each token's moves vs
`ctx.defenderTypes`, always the opponent's active; no dim encodes the opponent's
moves vs *this* token's typing. **Status: hypothesis from n=1 game plus a code
read, not a measurement**, and the wasted turn itself is not fully explained by
it. Keep for whenever v4 is re-derived for the fixed-team format.


---

## Recently Completed

- **Ladder log deep-dive + analysis tooling (2026-08-03).** Three findings from
  the 360-game `m7-greedy` run beyond the headline win rate:
  1. **~10% of ladder games are opponent concessions and they are ~35% of all
     our wins.** Across 747 rated games we have never lost one in under 16
     decisions; 74 wins came that fast. Contested-only: **greedy 19.3%
     (n=326)**, M7-era sampling **22.9% (n=341)** — the same −3.5pp difference,
     so no comparison changes, but every absolute ladder number this project has
     quoted is ~7pp high. Logged as correction 4 in MILESTONES.md.
  2. **Win rate decays with game length:** 31.2% (16–20 decisions) → 24.1%
     (21–25) → 18.8% (26–30) → **14.6% (31–40)**. First behavioral evidence
     matching the observation-poverty prediction — short games are damage races,
     long games are position games.
  3. **Win rate decays with opponent Elo:** 34.4% (1000–1099) → 15.3%
     (1200–1299), r=−0.143, t=−2.72, n=360. Own rating never left ~1000–1120 and
     hit the 1000 floor twice.

  `scripts/ladder_analysis.py` now reports the concession split and a
  win-rate-by-opponent-Elo table on every run, and takes `--min-decisions N`
  (use 16) to score contested games only. Both CSV schemas still parse.
  **Consequence: M10 is re-prioritised ahead of M11 Phase 1** — finding (2) is a
  correlation that M10 can turn into a mechanism without any retraining.
- **Greedy decoding shipped and confirmed offline (2026-08-01).** `--greedy` now
  exists on `models/evaluate.py`, `models/infer_server.py` and
  `tools/ladder-bot/ladder-bot.js`; `PPOAgent`/`TransformerAgent`
  `act()`/`act_batch()` take the masked argmax when `agent.greedy` is set,
  default off so rollouts still sample. `evaluate.py` prints the decision rule
  and refuses the flag where it would be inert (`--model mcts`, q_learning/dqn);
  the ladder bot's banner prints `policy=greedy|sampled`. **A/B on the M7 v3
  checkpoint, n=5,000/arm** (not 2,000 — the baseline is in the 0.3–0.7 band
  where the methodology requires 5,000), all four arms one session one machine:

  | vs | sampled | greedy | difference (Newcombe 95%) |
  |---|---|---|---|
  | Random | 69.9% (3497/5000) | **77.7%** (3887/5000) | **+7.8pp [+6.1, +9.5]** ✅ |
  | DamageFirst | 60.2% (3010/5000) | **65.2%** (3259/5000) | **+5.0pp [+3.1, +6.9]** ✅ |

  vs Random replicates the review's +7.8pp exactly at 12.5× the n. **The
  DamageFirst arm reverses the review's −2.0pp** (that reading was n=400 and
  underpowered), so greedy is *not* opponent-dependent — `docs/CODE-REVIEW-FINDINGS.md`
  §3 has been corrected. Side benefit: sampled-vs-Random came in at 69.9%
  against M7's recorded 69.7%, a data point on the §5f harness-reproducibility
  question. **`ladder-bot.js` now plays greedy by default** (`--sample` opts
  out). **Untested: the ladder**, where determinism against adapting humans is
  the one risk bot evals cannot speak to — costed at ~83 h and declined.
- **Code Review (2026-08-01):** Three independent read-only reviews of the training path, observation pipeline, and evaluation machinery. Identified root cause of the bot/human gap: **observation poverty**. The agent encodes moves without identity, carries no species/stats/trapping, cannot reason about damage. Also found: (1) MCTS confound (argmax vs sampling, ~+7.8pp vs Random), (2) M8 Phase 1A doubly confounded (trunk width fixed, no replay-adapter v3-extended path), (3) M8 Phase 2 new candidate cause (reward scale asymmetry in battle-sim), (4) M2 decision-rule confound (epsilon=0 for some arms, sampling for PPO), (5) "~3pp seat bias" unsupported by CI including 0.
<!-- Older entries (M9 phases 2a–2d, seed replication, observation-poverty
     scoping) now live in MILESTONES.md — Results Ledger and M11 respectively,
     per the three-item rule at the bottom of this file. -->

---

## Blockers

- **Home box SSH unreachable (2026-08-03).** Tailscale reports the host
  **online and directly reachable** (`active; direct`), but port 22 **times out**
  rather than refusing — a drop, not a closed port. `home-pc` is Windows and the
  repo/sshd live in WSL2, which has its own network namespace: sshd binds the WSL
  VM's IP, not the Windows host's tailscale IP, and **WSL2's IP changes on every
  boot**, so any previously-working `netsh portproxy` rule now points at a stale
  address. Diagnosis path on the box (elevated PowerShell): `wsl hostname -I` for
  the current IP, `netsh interface portproxy show all` to see the stale rule, then
  re-add `listenaddress=0.0.0.0 listenport=22 → connectaddress=<new WSL IP>`, plus
  a firewall rule allowing inbound 22 (Windows Firewall drops silently, which is
  why this times out instead of refusing). Inside WSL: `sudo service ssh start`.
  Blocks: **M12 Phases 2–4** (BC/PPO training and therefore the terminal gate),
  plus the optional M11 eval battery (both trained arms live only on the home
  box). **User will restore this when back home — record as blocked, not
  failed.** M10 Phase 3 is no longer affected; M10 is closed.
- **M12 Phase 0 (roster selection):** Not blocked by SSH — a design decision made
  and pre-registered on the Mac. This is the work that survives the outage, and
  it is what's in progress now.
- ~~M12 approval~~ — **APPROVED 2026-08-03.** User has committed to the pivot.
- ~~Width-512 probe~~ — **TRAINED 2026-08-02, UNMEASURED** due to home box SSH
  blocker (see above).

---

## Next Steps

**This is the complete remaining task list for the project.** Nothing else is
planned. Items 2–5 wait on home box SSH.

1. ~~**M12 Phase 0 — roster selection.**~~ ✅ **DONE 2026-08-05.** See Current
   Work above and `docs/BATTLE-FORMATS.md`.

2. **M12 Phase 1 — plumbing (~1 day). ← NEXT.** Load
   `config/rosters/gen1ou-standard.txt` in gym, evaluator, ladder-bot and BC
   preprocessing; format `gen1ou`, same team both sides. `./build` passes; 10
   battles per subsystem on the fixed roster. Mostly doable on the Mac.
   **Also decide and pre-register the BC corpus** (recommendation on record:
   gen1ou + tournament, to avoid confounding format and corpus at once).

3. **M12 Phase 2 — BC retraining (~2–3 h, home box).** gen1ou corpus, fixed
   roster. Report checkpoint path + held-out accuracy.

4. **M12 Phase 3 — PPO training (~2–3 h, home box).** 5M-step warm start from
   Phase 2, M7 recipe, single arm.

5. **M12 Phase 4 — bot-eval gate (~3–4 h). 🏁 TERMINAL.** n=5,000/opponent,
   sampled, no search. Gate ≥10% on both Random and DamageFirst.
   **Pass → record the number, optionally run Phase 5, archive.**
   **Fail → record the number as the finding, archive.** Either way the project
   ends here. Do not diagnose, do not re-roll the roster.

6. **(Optional) M12 Phase 5 — gen1ou ladder, ~24–48 h.** Only if Phase 4 clears
   with room to spare. User runs ladder sessions personally — hand over the
   command, don't launch it in a background task.

7. **(Optional, off critical path) M11 eval battery.** `m11_h128` vs `m9seed`
   (reward fix) and `m11_h512` vs `m11_h128` (width), n=5,000/opponent. Only if
   SSH is back and it's cheap. Expected null; changes no decision.

**Then: wind-down.** Final numbers into `MILESTONES.md`, `docs/WHERE-WE-ARE.md`
rewritten as a closing summary, Obsidian note updated to Complete.

---

## Active Plan

**M12: Fixed-Team Gen 1 OU Pivot — the final milestone.** Full spec in
`MILESTONES.md` → M12.

```
Phase 0  roster selection & curation   ✅ DONE 2026-08-05
         Tauros/Chansey/Snorlax/Exeggutor/Starmie/Alakazam
         config/rosters/gen1ou-standard.txt — locked
         │
Phase 1  plumbing & corpus decision    ⏳ NEXT (Mac, no blocker)
         fixed-team encoding into gym/eval/ladder, BC corpus choice
         │
Phase 2  BC retraining (home box)       ⏳ BLOCKED (home box SSH)
         gen1ou corpus, fixed roster, report accuracy
         │
Phase 3  PPO training (home box)        ⏳ BLOCKED (home box SSH)
         5M steps, M7 recipe, warm-start from Phase 2 BC
         │
Phase 4  bot-eval gate (~3–4h)          ⏳ BLOCKED (needs Phase 3)
         n=5,000/opponent, gate ≥10% on both Random/DamageFirst
         🏁 TERMINAL — project ends here, pass or fail
         │
Phase 5  ladder baseline (24–48h)       ⚪ OPTIONAL (only if Phase 4 clears well)
```

**Dropped as part of the bounded finish (2026-08-05):**
- M10 battle log analysis (~8 days) — diagnosis with nothing downstream to act on
- M11 Phase 1 observation schema v4 — the best untested idea, left on the table
  deliberately; it is where a future reader should start
- Instrumentation debt (`--seed`, per-step logging, `meta.json`) — insurance for
  future A/Bs that are no longer planned; M12's phases are single-arm

---

## Standing Rules

Learned the hard way; each one has a milestone behind it.

- **Both arms of an A/B on the same machine, in one session, same decision
  rule.** M9 Phase 2c moved BC corpus + seed + backend at once and cost a
  verdict.
- **Evaluate the final checkpoint, never a sweep pick.** M3.4's 62% sweep peaks
  regressed to 54% at n=500.
- **n=5,000 for anything in the 0.3–0.7 band**; n=2,000 only resolves ~±4.5pp
  against a +3pp gate. Full tables in `docs/EVALUATION-METHODOLOGY.md`.
- **GXE is account-level and cumulative — never a per-run gate.** This produced
  three milestones of over-read trends (M6–M8).
- **Pass both ladder CSVs to `ladder_analysis.py`** or it silently drops all 507
  pre-2026-08-01 games:
  ```bash
  python3 scripts/ladder_analysis.py \
    data/replays/self_ladder/ladder_results.pre-m9.csv \
    data/replays/self_ladder/ladder_results.csv --since 2026-07-16
  ```
- **Pre-register the gate before the number exists**, and record widenings
  plainly rather than quietly.

---

## Housekeeping

- **This file is the timeline; `docs/WHERE-WE-ARE.md` is the orientation.**
  Surface WHERE-WE-ARE after a context clear.
- **Keep `Recently Completed` to the last three items.** Older entries belong in
  the `MILESTONES.md` ledger; run-level detail belongs in
  `docs/MILESTONES-ARCHIVE.md`.
- Historical execution notes (M9 phase detail, M5 smoke battery, M7 job logs)
  were moved to the archive's appendix on 2026-08-03.
