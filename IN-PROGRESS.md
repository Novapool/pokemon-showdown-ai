# In Progress — Pokemon Showdown AI Training

Last updated: 2026-07-16

---

## Current Work

**M5.5 (Human Replay Data + BC for the MLP) — 🟨 ACTIVE 2026-07-16.
Full spec + pre-registered criteria in `MILESTONES.md` → M5.5.**

Project decision: competitive target stays **gen1randombattle**; training
data is multi-format — high-Elo (≥1300) randbats replays + high-level gen1ou
("how pros play with real teams"), per-format sampling weights as the knob.
Future follow-up (out of scope): fine-tune on one specific OU team.

### Active Tasks (M5.5)
- [x] Phase 1: `scripts/scrape_replays.py` — public replay-API scraper
      (top-up + `--backfill` modes, per-format rating floors, manifest,
      census). Smoke-verified all modes; committed `880eee5ff`.
- [x] Phase 1b: `scripts/bootstrap_gen1ou_replays.py` — 98,349 gen1ou raw
      logs bulk-imported from HF metamon-raw-replays (only parquet shards
      34–36 carry gen1* rows; 10.0k rated ≥1300, 55.9k unrated incl.
      tournament games). Committed `d2c5e0bf1`.
- [x] Phase 2: `sim/tools/replay-adapter.ts` + `models/replay_adapter_cli.js`
      + `test/tools/replay-adapter.test.js` (7 tests incl. byte-identical
      gym round-trip). Coverage 91% randbats / 86% gen1ou. Committed
      `5a670798a`.
- [x] Phase 3: `models/bc_pretrain_mlp.py` — smoke: 1 epoch on 2 shards →
      val acc 0.319 / opp-acc 0.300 (chance ~0.11); checkpoint plays via
      `evaluate.py --obs-v2` unchanged. Committed `875c1aa77`.
- [x] Data collection: full gen1ou→trajectory conversion done (98,349
      battles → 8.45M decisions, 0 parse errors, 746MB). Randbats backfill
      stopped at user request with **8,143 high-Elo logs** on disk (user:
      "we'll resume this later") — resume any time with
      `python3 scripts/scrape_replays.py --backfill --formats gen1randombattle --max-replays 12000 --max-pages 4000`
      then re-run `node models/replay_adapter_cli.js --format gen1randombattle --shard-size 1000`.
- [~] Phase 4 (in flight, 2026-07-16): **BC runs done.** Run 1 (policy+opp
      heads): val acc 49.7% randbats / 53.1% gen1ou (chance ~11%), opp-acc
      ~35% — but raw play only 20% vs Random / 20.5% vs DamageFirst; ~50%
      human-imitation accuracy does NOT transfer to raw play vs bots.
      Run 2 added `--value-bc-coef` (outcome-trained value head, 62/67%
      sign-acc — needed for search; plain policy-BC leaves the value head
      at init, crippling MCTS): raw unchanged (22%R / 14.5%DF).
      **MCTS over the BC net (final, 500+500): 56.2% vs Random / 45.0% vs
      DamageFirst** (`models/mcts/results/m55_bc_mcts_*.log`) — search lifts
      the BC net hugely (raw 22%/14.5%) but BC-only is decisively below the
      72.6%DF/86.0%R bars. BC-only verdict: negative; the fine-tune is the
      remaining shot.
      **Contingency triggered:** M3.2 fixes ported to `models/ppo/train.py`
      (`--pretrain-checkpoint`, `--value-warmup-steps`, `--bc-anchor`
      constant-coef) — 5M-step anchored fine-tune running
      (`models/ppo/checkpoints/bcft/`, pool seeded M2+M3.3-best, M5
      opponent-mix recipe). Externally stopped at 2.02M, resumed at user
      request (`--resume ppo_step_2000030.pt --bc-anchor ...`).
      **Interim 8-checkpoint sweep (150 battles vs Random):** 31% @ 250k →
      45/45 → 51% @ 1M → 53% @ 1.25M → 36/37 (dip) → 49% @ 2M. The anchored
      recipe lifts BC's 22% raw into the ~50% band — first BC→RL transfer
      in the project that improves rather than erodes.
      Then: full sweep, confirmations, MCTS battery on the best fine-tuned
      ckpt, verdict → MILESTONES.
      Side note: the human-data opp head reads DamageFirst at only ~19-24%
      (vs the bot-trained M5 head's 30-36%) — the M5 "mixture
      miscalibration" reading, confirmed from the other direction.
- [ ] M6 Phase 1 (parallel-safe): `tools/ladder-bot/` + `models/infer_server.py`
      per the revised M6 spec in MILESTONES.

**Previous: M5 (Opponent Modeling Head) — ✅ COMPLETE 2026-07-16. Thesis negative,
new-best side finding. Full results + verdict in `MILESTONES.md` → M5.**
- **C3 (the thesis) failed:** under tuned MCTS, sampling opponent actions
  from the trained head is parity-to-worse vs the existing policy sampler
  (70.4% vs 72.6% on DamageFirst, −2.2pp against a ≥+3pp bar). C2/C4/C5 ✅,
  C1 ❌ (35.8% accuracy vs the 40% bar; above the 25% label-bug floor —
  reads as a partial-observability ceiling, labels were oracle-verified).
- **Side finding — new best agent:** the M5 checkpoint under standard
  policy-sampler tuned MCTS confirms **72.6% (363/500) vs DamageFirst /
  86.0% (430/500) vs Random** — nominally ahead of the prior best (v2:
  70.2%/82.6%) on both, within ~1σ. Consistent with the aux loss shaping
  the trunk (secondary hypothesis) even though the sampler didn't win.
  Checkpoint: `models/ppo/checkpoints/opp/ppo_step_5000001_final.pt`.
  `docs/MODEL-COMPARISON.md` ledger updated.
- **Decision:** `--opp-sampler policy` stays the default; head mode kept
  (it's cheaper, 57ms vs 59–64ms). The opp head stays in the training
  recipe. M6 ships the M5 checkpoint (or ladder-A/Bs it vs the v2 control).
- Post-run cleanup pass done (code-simplifier findings below). Remaining
  before M6: commit results/docs.

**Previous: M5 build (all phases complete, committed `ffd14e275`).**
- Auxiliary opponent-action-prediction head (`Linear(128, 9)`) on the
  MLP-PPO shared trunk, multi-task loss `PPO + λ·CE` (λ=0.1). Labels are
  the opponent's resolved choice each decision point, captured
  omnisciently in `pokemon-gym.ts` and mapped into the **opponent's own
  9-way action frame** (moves by opponent request-slot, switches by
  opponent bench order; label = -1 masked out of CE when no simultaneous
  choice exists or the choice is unmappable).
- **Primary payoff is search integration:** MCTS's opponent-action sampler
  currently assumes the opponent plays like us (base policy on the
  opponent's reveal-tracked obs); the head replaces that with the actual
  opponent-action distribution, evaluated on the searcher's own obs
  (`opp_sampler: 'policy'|'head'`, A/B'd under tuned search).
- **Design decisions made:** obs schema **v2** base (`--obs-v2` +
  opponent-mix recipe — boosts/volatiles are direct predictors of the
  opponent's next action, and the M3.4 v2 checkpoint becomes the λ=0
  control); **fresh 5M-step decision run**, not warm-started (avoids the
  M3.2-style random-head-wrecks-mature-trunk risk; a run is only ~2h);
  raw-policy gain explicitly *not* expected — no-regression is the gate,
  the head-vs-policy sampler A/B under tuned MCTS is the thesis (C1–C5
  pre-registered in `MILESTONES.md` → M5).
- **Phase 0 first: commit the post-M4 tuning working tree** (tuned
  `MCTSAgent`/`evaluate.py` defaults, mcts_agent cleanup, sweep/A/B logs,
  doc updates) so M5 code starts from a clean tree.

### Active Tasks (M5)
- [x] Phase 0: commit uncommitted post-M4 tuning changes (clean tree)
      (done 2026-07-16: the tuning code/logs were already in `84750027c`;
      the M5 plan docs committed as `0127bf3f1`)
- [x] Phase 1A: `sim/tools/pokemon-gym.ts` — opponent-frame `oppAction`
      labels in `info` (single-seat + dual-seat) + `test/tools/gym.test.js`
      coverage (move/switch mapping, force-switch masking, dual symmetry,
      locked-choice labeling)
      (done 2026-07-16: labels derived omnisciently from the battle engine's
      committed choice — `sides[opp].choice.actions[0]`: `moveSlot` → 0–3,
      `moveid` fallback for auto-completed locked moves (recharge/multi-turn,
      the M4 gotcha) → 0–3, switch `target` index in `side.pokemon` → 4–8;
      −1 for wait/team-preview requests, Struggle, no-op sleep/freeze, or
      unmappable choices. Single-seat: captured before submitting p1's choice
      via `_captureOppLabel('p2')` (bounded I/O flush so the opponent's async
      choice has committed). Dual-seat: symmetric `oppAction`/`oppActionP2`
      from the caller-supplied per-seat actions (already in each seat's own
      frame). Verified vs the opponent's actual submitted choice across
      DamageFirst/Random/self-play seeds: 0 mismatches. 6 new tests + `./build`
      clean + all 29 gym tests green.)
- [x] Phase 1B (parallel with 1A): `models/ppo/ppo_agent.py` opp head +
      masked aux CE in `update()` + two-way checkpoint compat;
      `models/ppo/trajectory_buffer.py` label storage + merge
      (done 2026-07-16: `opp_head = Linear(128,9)` off shared trunk; λ via
      `update(..., opp_coef)` + `self.opp_coef` default 0.0 ⇒ bit-for-bit
      old path, verified; masked CE with all-masked/all-λ=0 NaN guards;
      old ckpts load with fresh head + optimizer-reset warnings, new ckpts
      load clean; buffer `opp_action` (-1=masked) stored + carried by merge)
- [x] Phase 2C (needs 1A+1B): plumbing — `models/gym_bridge.js` (both
      protocols), `models/gym_client.py`, `models/vec_gym_client.py`,
      `models/ppo/train.py` `--opp-coef` (default 0.1; 0 reproduces old
      loss path; checkpoints → `checkpoints/opp/`)
      (done 2026-07-16: `gym_bridge.js` already forwarded the gym's `info`
      object verbatim in both single- and dual-seat `step` responses, so
      `oppAction`/`oppActionP2` needed no bridge code changes — just doc
      comments confirming the passthrough is intentional; `gym_client.py`
      adds a `with_opp_action()` helper surfacing the snake_case
      `opp_action` (int, -1 = masked) in `step()`/`step_dual()` info dicts,
      reused by `vec_gym_client.py`'s `step()`/`step_dual()`; `train.py`
      collects `opp_action` into every buffer push across all three rollout
      paths (single-seat, opponent-mix random/damagefirst, and selfplay
      dual-seat — the dual-seat `pending` transition is labeled from the
      SAME `step_dual()` call that submitted the p1 action, not a later
      one still accumulating reward), adds `--opp-coef` (default 0.1,
      passed into `PPOAgent(...)` construction so it's persisted in
      hparams), routes checkpoints to `checkpoints/opp/` by default when
      `--opp-coef != 0`, and logs per-rollout opp-label coverage (only when
      `--opp-coef != 0`, so the `--opp-coef 0` log output is byte-identical
      to the pre-M5 path). Verified: bridge smokes (single-seat + `--selfplay`,
      `--obs-v2`) show `opp_action` in `[-1,8]` with dual-seat labels
      matching the actual simultaneous opponent action 100% of the time;
      `VecGymClient` smoke same; `train.py` smokes at `--num-envs 2` —
      `--opp-coef 0.1` (finite loss, coverage 77–96%), `--opp-coef 0`
      (old log format, no coverage line), opponent-mix incl. selfplay at
      `--opp-coef 0.1` (finite loss, coverage logged across all three
      families), default checkpoint routing to `checkpoints/opp/`
      confirmed and cleaned up, checkpoint round-trip confirms `opp_coef`
      persisted in hparams and `opp_head` present after `PPOAgent.load()`)
- [x] Phase 2D (parallel with 2C; needs 1B, accuracy half needs 1A):
      `models/evaluate.py` opp-accuracy reporting + `--opp-sampler`;
      `models/mcts/mcts_agent.py` head-based opponent sampler with
      policy fallback
      (done 2026-07-16: `evaluate.py` — `_checkpoint_has_opp_head()` detects a
      headed ckpt by the saved `opp_head` key (robust to how the trainer
      propagates `opp_coef`); PPO single-opponent eval (vec + serial) reports
      opp-prediction top-1 accuracy = opp-head argmax vs `info["opp_action"]`
      over unmasked (label ≥ 0) steps, printed alongside win rate (n/a when no
      labels present, e.g. Job C plumbing absent); added `--opp-sampler
      head|policy` (default policy) forwarded to `MCTSAgent`. `mcts_agent.py` —
      `opp_sampler`/`has_opp_head` ctor args; `head` mode caches the opp head's
      masked+renormalized dist on each node from the SAME trunk forward that
      makes the PUCT prior (`_policy_value_opp` + `_head_opp_dist`), masking to
      the opponent's legal actions (`state[opp]["mask"]`, opponent's own frame —
      the frame the sim accepts, no remapping) and renormalizing; `_advance`
      consumes it for the node's first simultaneous opponent choice, falling
      back to the policy sampler for subsequent opponent-only force-switches;
      auto-falls-back to policy (with warning) when the ckpt has no trained head
      (via explicit `has_opp_head` from the file, else `opp_coef>0` hparam).
      Tuned defaults (sims=100, c_puct=0.5, det=1) and the locked-choice
      raw-policy root fallback preserved. Verified: both files compile; raw
      obs-v2 eval of the pre-M5 v2 ckpt still runs (no accuracy line, correctly
      no opp head). MCTS head/policy smoke on the v2 ckpt was PENDING (blocked
      by a harness outage) — now confirmed in Phase 3 below: headless ckpt
      warns + falls back to policy correctly; headed ckpt runs clean under
      both `--opp-sampler head` and `--opp-sampler policy`.)
- [x] Phase 3: smoke battery — TS/gym suites green, label spot-check vs
      DamageFirst, 4k-step train smoke (CE decreasing), `--opp-coef 0`
      regression smoke, MCTS head-sampler smoke; commit code
      (done 2026-07-16 by tester — see "Test Results — M5 Phase 3" below.
      All 6 steps PASS: `./build` clean; gym suite 29/29 incl. 6 new M5
      label tests, battle-sim 13/13; bridge smokes (single- + dual-seat
      `--obs-v2`) show `opp_action` in `[-1,8]`; 4k-step train smoke
      (`--opp-coef 0.1`, mixed opponents) finite loss + coverage 78–96%,
      checkpoints routed to `checkpoints/opp/`; `--opp-coef 0` smoke
      reproduces the old log format (no coverage line) and exercises
      checkpoint-compat (old ckpt loads into headed agent with warnings,
      no crash); DamageFirst-only label sanity — opp-accuracy 32.4%
      (266/822), well above the ~11–25% chance band; all four
      `evaluate.py` load paths pass, including the previously-unverified
      MCTS head-sampler fallback (headless ckpt warns + falls back) and
      head/policy runs on a headed ckpt. Smoke checkpoints cleaned up;
      code committed as `ffd14e275`.)
- [x] Phase 4: 5M-step decision run → sweep → confirmations → sampler A/B →
      verdict recorded (done 2026-07-16). Run trained stably to 5,000,001
      steps (externally stopped at 2.93M and resumed — `--resume` added to
      `models/ppo/train.py` for this, commit `589d8bf1e`; pool seeded with
      M2 + M3.3 best per the M3.4 recipe). 21-checkpoint sweep: 40→50–59%
      band, no collapse (`sweep_results.txt`). Confirmations
      (`confirm_results.txt`): best raw 57% (285/500) vs Random @ 5.0M final /
      44% (88/200) vs DamageFirst @ 3.0M; best opp accuracy 35.8% vs DF.
      Sampler A/B (tuned MCTS, 500/arm/opponent, `models/mcts/results/
      m5_ab_*.log`): head 70.4% DF / 85.8% R vs policy 72.6% DF / 86.0% R —
      **C3 thesis negative (−2.2pp, parity)**; C2/C4/C5 ✅, C1 ❌ (35.8% vs
      40% bar). Side finding: **M5 final + policy-sampler MCTS is the new
      best agent (72.6% DF / 86.0% R)**. Full verdict in `MILESTONES.md` → M5.
- [x] Post-run cleanup pass (code-simplifier review of `ffd14e275`,
      2026-07-16 — all 5 applied, verified end-to-end (single-opponent,
      h2h, and MCTS head/policy-sampler eval runs), `./build` +
      `test/tools/gym.test.js` green):
      (1) `PPOAgent.load()` now sets `agent.has_opp_head` directly (default
      `True` from `__init__`, so it's meaningful on non-loaded agents too);
      `evaluate.py` reads it instead of re-torch.loading the checkpoint via
      the now-deleted `_checkpoint_has_opp_head()`; (2) kept `oppActionP2`
      (cheap — already computed alongside `oppActionP1` — and covered by the
      dual-seat symmetry test) but added a `opp_action_p2` snake_case alias
      in `gym_client.with_opp_action()` for consistency, matching
      `opp_action`; (3) `_captureOppLabel()` now logs when the opponent's
      choice is still pending after 3 flushes (an unresolved timing race)
      instead of silently masking it the same as -1, and
      `_opponentActionLabel()`'s catch now logs unexpected errors before
      returning -1, distinguishing them from the many deliberate -1 returns;
      (4) merged `_policy_value_opp()` into `_policy_value(..., with_opp_logits=
      False)` in `mcts_agent.py`; (5) `evaluate.py`'s five `_run_battles*`
      functions now return a `BattleResult` NamedTuple (wins, total,
      opp_correct=0, opp_total=0) instead of growing positional tuples.

**Search-knob sweep (post-M4 step 1) — ✅ COMPLETE 2026-07-15, decisive.
New operating point: sims=100, c_puct=0.5, det=1.**
- Two-stage sweep (OFAT then combos, 200 battles/cell) + 500-battle
  confirmation on `ppo_step_4750059.pt`. Lower c_puct and fewer
  determinizations each beat their defaults on both opponents, and the
  effects stack: confirmed **81.2% (406/500) vs Random** and **67.2%
  (336/500) vs DamageFirst** — vs 66%/56% at the M4 defaults, at slightly
  *lower* latency (~85ms/move). The narrowly-missed M4 vs-Random criterion
  (+10pp over raw) is now cleared at +23.8pp. Lesson: one deep tree with a
  trusted prior beats a shallow determinization ensemble at fixed budget.
- Defaults updated in `MCTSAgent`/`evaluate.py` (uncommitted, with the
  earlier mcts_agent.py cleanup); full tables in `MILESTONES.md` → M4 →
  "Post-M4 knob sweep"; logs in `models/mcts/results/sweep/`.

**v2-checkpoint A/B (post-M4 step 2, pre-planned in M3.4) — ✅ COMPLETE
2026-07-16: a tie, v2 nominally ahead.**
- Tuned MCTS over `v2/ppo_step_2250032.pt`: **82.6% (413/500) vs Random,
  70.2% (351/500) vs DamageFirst** vs v1's 81.2%/67.2% — within noise
  (~±4pp), though the direction reversed (raw v2 trailed raw v1). Verdict
  in `MILESTONES.md` → M4 → "Post-M4 knob sweep": either checkpoint is a
  valid M5/M6 base; **v1 stays the default**; v2 is a live option for M5.
  Logs: `models/mcts/results/sweep/v2ab_*.log`.

**Post-M4 steps 1–2 (knob sweep, v2 A/B) — both complete above; M5 is now
scoped and active (see top of Current Work).** The open obs-schema question
from the A/B tie is resolved: M5 trains on v2 (decision + rationale in
`MILESTONES.md` → M5). The uncommitted post-M4 working tree (mcts_agent.py
cleanup + tuned defaults in `MCTSAgent`/`evaluate.py`, doc updates,
sweep/A/B logs) is M5 Phase 0 — commit it before any M5 code lands.

**M4 — ✅ complete 2026-07-15, POSITIVE RESULT. Next up: M5 (opponent modeling)
or search tuning.**
- Determinized UCT over the M3.3 best checkpoint (100 sims, 4 determinizations,
  88ms/move): **60.2% (602/1000)** seat-balanced h2h vs the same checkpoint
  without search, **56% (113/200) vs DamageFirst** (raw: 46%) — the first
  intervention since M2 that clearly improves play, and the first to move the
  DamageFirst number. **66% (330/500) vs Random** (+8.6pp) narrowly missed the
  pre-registered +10pp bar; 4 of 5 criteria met. Full tables + reading in
  `MILESTONES.md` → M4; logs in `models/mcts/results/`.
- Follow-ups worth considering before/alongside M5: search-knob sweep
  (`--sims`, `--c-puct`, determinization count — all untuned defaults), the
  pre-planned A/B on the v2 2.25M checkpoint (richer obs may matter more with
  search attached), and M5's opponent-model head plugs directly into the
  search's opponent-action sampler.

### Active Tasks (M4)
- [x] **Forward model** `sim/tools/battle-sim.ts` — clones the live gym battle via
  `State.serializeBattle`/`deserializeBattle` (0.8ms round-trip), snapshots the
  gym's reveal/volatile tracker state alongside (`ObservationTrackers`, extracted
  from `PokemonGymEnv` + new `snapshot()` API), steps both seats directly on the
  clone with gym-identical obs/mask/reward semantics, `fork()` for tree branching.
  **Key engine gotcha found and fixed:** a seat's choice can be auto-completed by
  the engine (`side.isChoiceDone()` — sleep/recharge/multi-turn locks), so
  `needsAction` must consult it; submitting a choice for an auto-done seat lands
  one decision point ahead and desyncs the battle (mask-legal moves get rejected)
- [x] **Determinizer** (Node-side, inside battle-sim — deviation from the original
  `models/mcts/determinizer.py` spec since team generation/legality live in Node):
  replaces the searcher's opponent's *unrevealed* slots with sets sampled from the
  gen1randombattle generator (seeded, species-clause-safe); `perspective: 'p1'|'p2'`
  so a p2-seated searcher determinizes p1. Documented approximation: revealed
  Pokémon keep their true full movesets (only unrevealed *slots* are resampled)
- [x] TS tests — `test/tools/battle-sim.test.js`, 13 tests: obs parity with the
  live gym (v1 + v2 byte-equality), determinization invariants (both perspectives,
  seed-reproducible, revealed mons survive), illegal-free playouts to terminal,
  fork consistency, snapshot-throws-when-done. All pass; gym suite still green
- [x] Bridge/`GymClient` sim protocol: `sim_clone` (determinize/seed/perspective),
  `sim_step`, `sim_fork`, `sim_free`, `sim_free_all`; sims cleared on env reset;
  measured 0.36ms/sim-step over the wire
- [x] `models/mcts/mcts_agent.py` — root-parallel determinized PUCT: policy head
  as prior, opponent actions sampled from the same policy on the opponent's
  reveal-tracked obs, value head + shaped path rewards at leaves, visit counts
  summed across determinizations, argmax-visits (Q tiebreak); `seat='p1'|'p2'`;
  falls back to the raw policy when search can't run (locked-choice roots)
- [x] `evaluate.py --model mcts` (`--sims/--determinizations/--c-puct/`
  `--no-determinize/--mcts-seat`, h2h via `--vs-checkpoint`) + latency: **88ms
  mean / 155ms max per searched decision @ 100 sims** — well under the 500ms
  budget. Early signal: **73% (22/30) vs Random** (raw checkpoint: 57%)
- [x] Code committed: `9da4d273a`
- [x] **Eval battery** (2026-07-15, ~2h): 500 vs Random → **66.0% (330/500)**
  (raw 57.4%, +8.6pp, criterion ❌ by 5 battles); 200 vs DamageFirst →
  **56.5% (113/200)** (raw 45.5%, +11.0pp ✅); seat-balanced h2h vs the raw
  checkpoint → **60.4% as p1 (302/500), 60.0% as p2 (300/500), 60.2%
  combined ✅**; latency 84–88ms mean / ~102ms p95 ✅ (one 909ms outlier
  across ~30k searched decisions). Logs: `models/mcts/results/*.log`
- [x] Results + verdict recorded in `MILESTONES.md` → M4 (4 of 5 criteria met);
  `docs/MODEL-COMPARISON.md` updated — **MCTS-wrapped M3.3 checkpoint is the
  new best agent**; M3.3/M3.4 rows backfilled into the ledger

**Previous: M3.4 — ✅ complete 2026-07-15, NEGATIVE RESULT.**
- Code for both parts (obs schema v2, mixed-opponent training) landed, smoke-verified, and committed (`7533e08d7`). The 5M-step decision run (externally killed at 4.77M steps — no crash; 19 checkpoints cover the trajectory) trained stably in a 42–62% sweep band with no collapse, but full-battle confirmations regress to an M2/M3.3 peer: best **54% (272/500) vs Random**, **46% (92/200) vs DamageFirst**, and **48.0% (480/1000) seat-balanced head-to-head vs the M3.3 best** — all three pre-registered criteria unmet. Full tables + reading in `MILESTONES.md` → M3.4.
- Takeaway: four independent 5M-step-class runs (fixed-opponent, transformer, self-play, v2+mix) all land in the same 51–57%-vs-Random band — the bottleneck is not the observation or the opponent distribution. M4 (MCTS lookahead) attacks the most plausible remaining lever and its criteria were already recalibrated to be relative to the base policy.
- **M4 base checkpoint (recommended):** `models/ppo/checkpoints/selfplay/ppo_step_4750059.pt` (M3.3 best, v1); A/B against `models/ppo/checkpoints/v2/ppo_step_2250032.pt` (v2 peer, richer obs) once the MCTS harness exists.

**Recently closed: M3.1 / M3.2 / M3.3 (2026-07-14)** — parallel training (~5x); transformer retired, MLP-PPO is the architecture; self-play fixed training stability but produced an M2 peer (57% vs Random / 46% vs DamageFirst / 52.4% h2h). Full trails in `MILESTONES.md`.

### Active Tasks (M3.4)
- [x] `sim/tools/feature-extractor.ts` — `TOKEN_DIM_V2=77`, `ActiveVolatiles`/`BOOST_ORDER`, optional `volatiles` param on `extractFeaturesStructured()` (token dim parameterized; v2 dims filled on tokens 0 and 6 only)
- [x] `sim/tools/pokemon-gym.ts` — per-side volatile tracker (`_processVolatileLine`: boosts incl. Haze `-clearallboost`, screens/Sub/Leech Seed `-start`/`-end`, toxic counter via `[from] psn` ticks; reset on switch/drag/faint); obsMode `'structured-v2'`
- [x] `models/gym_bridge.js` — `--obs-v2`; per-reset `{"opponent": ...}` override with per-env single/dual protocol switching
- [x] `models/gym_client.py` — `obs_v2=`, `set_opponent()`, `_reset_cmd()`, `slice_structured_obs()` (v2→v1 per-token view for cross-schema play)
- [x] `models/vec_gym_client.py` — `obs_v2=` passthrough, `set_opponent()`, `reset_all(opponent=)`, mode-aware auto-reset
- [x] `models/ppo/train.py` — `--obs-v2` (checkpoints → `checkpoints/v2/`), `--opponent-mix` per-rollout family sampling with `_switch_family()` env resets, cross-schema pool opponents via `_opp_view()`
- [x] `models/evaluate.py` — `--obs-v2`; `_run_battles_h2h` slices per-seat obs to each agent's `obs_size` (v2-vs-v1 head-to-head)
- [x] `test/tools/gym.test.js` — v2 shape (924), v1-prefix byte-equality, volatile-dim placement, full-battle v2 range/stability; 23/23 pass
- [x] Smokes: v2 bridge single+dual seat, live opponent switching, 4k-step mixed-opponent training run with v1 seed opponent, all 4 eval paths, v1 selfplay trainer regression
- [x] **5M-step decision run** — killed externally at 4.77M steps (no crash; last checkpoint 4.75M, 19 checkpoints total; `models/ppo/checkpoints/v2/train.log`)
- [x] Eval battery: 19-checkpoint sweep (42–62% band, no collapse; `sweep_results.txt`), confirmations on top 4 (`confirm_results.txt`), seat-balanced h2h vs M3.3 best (`h2h_results.txt`)
- [x] Results recorded vs pre-registered criteria in `MILESTONES.md` → M3.4 — **all three unmet** (54% vs Random / 46% vs DamageFirst / 48.0% h2h); winner unchanged, so `docs/MODEL-COMPARISON.md` needs no update

### Active Tasks (M3.1) — all resolved
- [x] `models/vec_gym_client.py` — `VecGymClient(n_envs, structured)`: N parallel `gym_bridge.js` subprocesses, pipelined write-all/read-all stepping, auto-reset on done, per-env errors reset in place and surface as `infos[i]["error"]`
- [x] `models/gym_client.py` — `_send()` split into `_write()`/`_read()` for pipelining
- [x] `models/{ppo,transformer}` agents — `act_batch()` batched inference; `update()` accepts merged tensor dict; `device=` override (excluded from checkpoint hparams so checkpoints stay Mac↔CUDA portable); `load(path, device=...)`
- [x] `models/ppo/trajectory_buffer.py` — `compute_advantages(normalize=...)` + `merge_buffers()` (per-env GAE, global advantage normalization)
- [x] Both trainers — `--num-envs` (default 8) + `--device`; vectorized rollout collection; checkpointing/`--resume`/LR-annealing unchanged (keyed off `total_steps`)
- [x] `models/evaluate.py` — `--num-envs` + `--device`; per-env battle quotas keep counts exact; q/dqn fall back to per-env `act()`
- [x] Verified: vec smoke (4 envs, 800 steps, auto-reset), 600-step train smokes (transformer + PPO both obs modes), checkpoint load round-trips, `--resume` continues at step 600 with LR intact, **M2 51% checkpoint evaluates 49% (49/100) through the parallel path in 4.8s**
- [x] Benchmarks: transformer eval 100 battles 20.2s → 5.6s at 8 envs (3.6x); training throughput table in `docs/ML-TRAINING.md`

### Active Tasks (M3, concluded) — all resolved
- [x] `models/transformer/transformer_agent.py` — `TransformerAgent(nn.Module)`, PPO wrapper composing `TransformerPolicy` as `self.policy` (composition, not subclassing, so BC checkpoint keys stay unprefixed and loadable). Later extended with `target_kl` approx-KL early-stopping (see below)
- [x] `models/transformer/train.py` — rollout/GAE/checkpoint training loop mirroring `models/ppo/train.py`; always structured `(12,65)` unflattened; `--pretrain_checkpoint` flag calls `load_pretrain_checkpoint(agent.policy, path)` before PPO starts; checkpoints split into `checkpoints/{scratch,pretrained}/`; `--resume <checkpoint>` restores full agent+optimizer state and parses the starting step count from the filename. Later extended with linear LR annealing toward 0 over `--steps` (see below)
- [x] `models/evaluate.py` — added `--model transformer`; split the old single `structured` bool in `_run_battles` into `structured`/`flatten` so the transformer gets raw `(12,65)` while PPO-structured still gets flattened `(780,)`
- [x] Smoke-tested: agent construction/act/save/load, BC warm-start (30/30 tensors loaded into `agent.policy`), a short training run in both scratch and pretrained modes, `--resume`, and `evaluate.py --model transformer` — all pass with no shape errors
- [x] **From-scratch run:** 2.6M steps. Evaluated 500 real battles: **32% win rate (158/500)** — below the M2 MLP-PPO baseline (51%)
- [x] **Warm-started run 1 (2.6M steps, no stability fixes):** Evaluated 500 real battles: **41% win rate (204/500)** — better than from-scratch, still below 51%
- [x] **Extended warm-started run 1 to 7.6M steps + full checkpoint sweep (21 checkpoints, 250k–7.6M):** revealed the run actually peaked at **46%** around 2.5M–3.1M steps, then **collapsed violently** — cratering to 0–13% between 3.6M and 5.6M steps, partially recovering to 32% at 6.1M, then drifting down to 24% by 7.6M. Root-caused to unconstrained PPO updates over a long horizon with no LR decay or trust-region constraint — a single bad minibatch update can yank the policy far enough to wreck its move-vs-switch balance (recall `RandomPlayerAI` never voluntarily switches, so any policy that over-favors switching loses almost every battle)
- [x] **Stability fix:** added approximate-KL early-stopping (`target_kl=0.02`, stops a rollout's PPO minibatch epochs early if `1.5×target_kl` is exceeded — standard PPO safeguard, http://joschu.net/blog/kl-approx.html) to `TransformerAgent.update()`, and linear LR annealing to 0 over the `--steps` budget in `train.py`. Both smoke-tested
- [x] **Warm-started run 2 (fresh 5M-step run, with stability fixes) + full checkpoint sweep (20 checkpoints, 250k–5.0M):** the violent collapses are gone (no more 0% crashes), confirming the fix addressed that specific failure mode. But the run now peaks almost immediately (**45% at 500k steps** — near where the BC-pretrained starting policy already was) and then **gradually decays** to 25–35% noise, ending at **27% (135/500)** at the 5.0M-step final checkpoint. The stability fixes changed *how* it fails, not *whether* it fails to improve on the BC starting point
- [x] **Final M3 conclusion:** best-ever recorded win rate across ~40 checkpoints spanning both warm-started runs is **46%** (150-battle spot check, run 1 @ 2.5M steps) — never reaching, let alone beating, the 51% MLP-PPO baseline. Continued PPO fine-tuning of the transformer consistently degrades performance from its early/BC-pretrained peak rather than improving it, the opposite of the M3 hypothesis. This is a real, decisive negative result, not an artifact of insufficient training budget or an unfixed instability bug

### Active Tasks (M2, complete)
- [x] Add `extractFeaturesStructured()` to `sim/tools/feature-extractor.ts` — returns `(12, 65)` token array, exact layout match with `models/metamon_adapter.py`
- [x] Add opponent-reveal tracker to `sim/tools/pokemon-gym.ts` — reconstructs revealed opponent state (species/HP/status/fainted/used moves) from `|switch|/|drag|/|-damage|/|-heal|/|-status|/|-curestatus|/|faint|/|move|` lines on the battle log, never from omniscient state
- [x] Wire `obsMode: 'flat' | 'structured'` into `PokemonGymEnv` (default `'structured'`)
- [x] Update `models/gym_bridge.js` — serializes structured obs as 780-element flat array by default; `--flat` CLI flag restores the legacy 100-dim path
- [x] Update `models/gym_client.py` — reshapes `(780,)` → `(12, 65)`; `GymClient(structured=False)` mirrors `--flat`
- [x] Fixed `models/{q_learning,dqn,ppo}/train.py` + `evaluate.py` to pass `structured=False` — their networks are hardcoded to the flat 100-dim vector and would otherwise silently break now that structured is the default
- [x] Tests: shape/unknown-flag/fainted-flag/bench-ordering coverage in `test/tools/gym.test.js`; full build + battle-loop smoke tests pass
- [x] Renamed `--battles` → `--steps` in `dqn/train.py` and `ppo/train.py` (the flag counted environment steps, not battles — collided in meaning with `evaluate.py`'s genuinely battle-counting `--battles`)
- [x] Added `--structured` to `ppo/train.py` and `evaluate.py` — `PPOAgent` already took `obs_size` as a parameter, so no architecture changes were needed, just plumbing + a separate `checkpoints/structured/` directory
- [x] **Verify:** MLP PPO trained on flattened structured obs (2.6M steps ≈ 50k battles), evaluated 500 real battles greedy vs RandomPlayerAI: **51% win rate (254/500)** — clears the ≥50% target, confirms the structured token representation preserves parity with the M1 flat baseline. Checkpoint: `models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`
- [x] **Bug found + fixed:** `evaluate.py`'s `_run_battles()` did `action = agent.act(obs, valid_mask)` uniformly for all three model types. `QAgent`/`DQNAgent.act()` return a plain `int`, but `PPOAgent.act()` returns `(action, log_prob, value)` — a 3-tuple. Every `evaluate.py --model ppo` run was silently handing the whole tuple to `env.step()`, which JSON-serializes a Python tuple as a JS array; comparing/indexing with that array in `pokemon-gym.ts` always fails, so **every PPO action was rejected as illegal, forever, on battle 1** — an infinite hang with no error, no exception, nothing in the logs. This bug predates this session (written 2026-05-18) — `evaluate.py --model ppo` had never actually completed a run before. See "Recently Completed" for the full incident writeup.
- [x] **M2.5:** Run `bash scripts/download_metamon.sh` — done 2026-07-02, **119,536 gen1ou trajectories** in `data/metamon_cache/`
- [x] **M2.5:** Smoke-test BC on real data — done 2026-07-02, `--max_files 200`: 33,270 samples, loss 2.04→1.72, acc 19.8%→34.1% (chance ≈ 11%), no adapter warnings; action histogram over real battles ≈ 71% moves / 29% switches (plausible for human gen1ou)
- [x] **M2.5:** Full BC run — done 2026-07-02: 4.96M samples/epoch × 5 epochs (24.8M total) in 2414s on MPS; **final loss 1.339, top-1 acc 50.5%** (chance ≈ 11%); checkpoint at `models/checkpoints/bc_pretrain_gen1ou.pt`, verified loading 30/30 tensors into `TransformerPolicy`. M2.5 complete — remaining criterion (warm-start vs from-scratch PPO comparison) blocked on M3.

### Deviation from the original M2 plan: stat boosts dropped

The original M2 plan called for a boost tracker (`|-boost|`/`|-unboost|`) feeding into the active-Pokémon token. That was written before M2.5 locked the token schema to exactly match `models/metamon_adapter.py` (65 dims, no boost slots) so the BC-pretrained checkpoint's input projection stays valid. Metamon's replay dataset doesn't encode stat boosts either, so there's no way to add a boost feature without breaking BC-checkpoint compatibility or retraining it. Boosts are dropped from M2's scope; revisit only as a deliberate schema-version bump (would require a new BC pretraining run).

### Bug fixed in passing: level default

`parseLevelFromDetails()` defaulted to level 50 when a details string had no explicit `L##` tag. Showdown omits the level tag entirely when it's 100 (the common case for gen1ou/gen1randombattle), so this was silently encoding real level-100 Pokémon as level 50 in both the flat and structured extractors. Default is now 100, matching Showdown's actual convention and Metamon's adapter assumption.

---

## Test Results — M5 Phase 3 Smoke Battery (2026-07-16)

Run by the tester agent (Job 3.1) after all four Phase 1–2 builder jobs
(A/B/C/D) completed. Nothing modified; all commands below run as-is against
the working tree. **Overall verdict: PASS.** No routing back to any builder
required.

**1. `./build`** — TypeScript compile.
Command: `./build`
Exit code: `0`. No output (clean compile).

**2. Gym test suite.**
Note: `.mocharc.json`'s `spec` glob is array-typed, so yargs merges it with
any CLI `--spec`/positional args instead of overriding — plain
`npx mocha test/tools/gym.test.js` silently ran the full 2354-test suite.
Used `--no-config` to scope correctly.
Command: `npx mocha --no-config --reporter dot test/tools/gym.test.js`
Exit code: `0`. Output: `29 passing (315ms)`. Spec listing confirms all 6
new M5 tests present and green under `PokemonGymEnv opponent action labels
(M5)`: move-slot mapping, deterministic oppAction sequence, locked-move
(recharge) labeling, dual-seat symmetry, switch bench-slot mapping (4-8),
and force-switch masking (-1).
Command: `npx mocha --no-config --reporter dot test/tools/battle-sim.test.js`
Exit code: `0`. Output: `13 passing (380ms)`.
Pre-existing unrelated failure (not counted, not in scope): full-suite run
hits `better-sqlite3` `NODE_MODULE_VERSION 137` vs required `141` in "SQLite
worker wrapper" — a native module ABI mismatch unrelated to M5.

**3. Bridge smokes (`--obs-v2`), single-seat and dual-seat.**
No `--obs-v2`/dual-seat coverage existed in the `__main__` smokes in
`gym_client.py`/`vec_gym_client.py`, so a throwaway script was written to
the scratchpad (not added to the repo) exercising `GymClient(obs_v2=True,
opponent="damagefirst")` (100 random-valid steps) and
`GymClient(obs_v2=True, selfplay=True)` via `reset_dual()`/`step_dual()`
(100 steps, "needs"-gated actions per seat).
Exit code: `0`.
Output:
```
=== single-seat obs_v2 ===
reset obs shape: (12, 77)
single-seat: 77 labeled, 23 masked, distinct values seen=[-1, 0, 1, 2, 3, 8]
single-seat close: OK
=== dual-seat (selfplay) obs_v2 ===
reset_dual obs shapes: (12, 77), (12, 77)
dual-seat: 91 labeled, 9 masked, distinct values seen=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
dual-seat close: OK
ALL SMOKE CHECKS PASSED
```
`opp_action` confirmed present in `info` and in `[-1, 8]` for both paths.

**4. Training smoke.**
Command: `python models/ppo/train.py --obs-v2 --opponent-mix
"selfplay=0.5,damagefirst=0.3,random=0.2" --opp-coef 0.1 --num-envs 2
--steps 4000 --rollout-steps 512 --checkpoint-every 2000`
Exit code: `0`. Loss finite every rollout (0.163–0.226); label coverage
logged every step, 0.78–0.96 (sane — the mixed-opponent smoke touches
random/damagefirst/selfplay families in one short run, and coverage
tracks family composition). Checkpoints saved to
`models/ppo/checkpoints/opp/` as specified (`ppo_step_2001.pt`,
`ppo_step_4001.pt`, `ppo_step_4001_final.pt`) — confirms default routing.
Note: `train.py` logs only the combined loss plus label coverage, not a
separately-broken-out CE term (`ppo_agent.update()` doesn't expose CE
apart from the combined loss — see the code comment at `train.py:514`).
This matches the task instructions given to this tester run ("assert
finite loss and sane label coverage logging") even though the broader
Phase-3 job description in `AGENT_JOBS.md` mentions "CE loss ... logged
and decreasing" — flagging as an observation, not a failure, since finite
loss + coverage was the explicit bar for this run.
Command (control): `python models/ppo/train.py --obs-v2 --opponent-mix
"selfplay=0.5,damagefirst=0.3,random=0.2" --opp-coef 0 --num-envs 2
--steps 500 --rollout-steps 256 --checkpoint-every 500`
Exit code: `0`. Confirmed: no "Opp label coverage" line in any log line
(old log format reproduced exactly). Bonus: this run's self-play opponent
sampling pulled a pre-M5 checkpoint from `checkpoints/v2/` and exercised
checkpoint two-way compatibility live —
`UserWarning: checkpoint has no 'opp_head'; using a freshly initialized
opponent-prediction head (expected for pre-M5 checkpoints)` and a
matching optimizer-reset warning, then continued without crashing.
All smoke checkpoints from both runs cleaned up afterward (see note at
end of this section).

**5. Label/head sanity check (DamageFirst-only).**
Command: `python models/ppo/train.py --opponent damagefirst --obs-v2
--opp-coef 0.1 --num-envs 2 --steps 6000 --rollout-steps 512
--checkpoint-every 6000 --checkpoint-dir <scratchpad dir>`
Exit code: `0`. Loss decreased monotonically-ish 0.259 → 0.125 across 12
rollouts; label coverage 0.78–0.85 throughout.
Command: `python models/evaluate.py --model ppo --obs-v2 --checkpoint
<smoke checkpoint>/ppo_step_6000_final.pt --battles 20 --opponent
damagefirst`
Exit code: `0`. Output: `Opp-prediction top-1 accuracy: 0.324 (266/822
unmasked steps)` — well above the ~11–25% chance band for a briefly
trained head, as expected since DamageFirst's argmax-damage choice is
independently predictable. This is also step **6a** (raw PPO eval on a
headed checkpoint) — the accuracy line printed correctly.

**6. Eval/MCTS load paths (previously UNVERIFIED — infra outage blocked
Job D from running these).**
- **6a** (raw PPO eval on headed checkpoint): satisfied by the command in
  step 5 above — accuracy line printed, exit 0.
- **6b** (`--model mcts --opp-sampler head` on the **headless** v2
  control checkpoint): `python models/evaluate.py --model mcts --obs-v2
  --checkpoint models/ppo/checkpoints/v2/ppo_step_2250032.pt --opp-sampler
  head --battles 2 --sims 20`. Exit code `0`.
  `UserWarning: --opp-sampler head requested but the checkpoint has no
  trained opponent head; falling back to the policy sampler` — correct
  WARN + fallback, no crash. Summary line: `opp_sampler=policy` (the
  effective, fallen-back value).
- **6c** (`--model mcts --opp-sampler head` on the **headed** smoke
  checkpoint from step 4, `checkpoints/opp/ppo_step_4001_final.pt`):
  same command shape. Exit code `0`. No warning, ran clean. Summary line:
  `opp_sampler=head`.
- **6d** (regression, `--opp-sampler policy` on the same headed
  checkpoint): exit code `0`, ran clean. Summary line:
  `opp_sampler=policy`.
All four eval/MCTS paths pass — this closes out the item Job D flagged as
blocked/unverified.

**Cleanup.** Smoke checkpoints created by this run were deleted afterward:
`models/ppo/checkpoints/opp/` (entire directory — did not exist before
this run) and `models/ppo/checkpoints/v2/ppo_step_501{,_final}.pt` (the
two files this run added to that pre-existing directory; all other files
in `checkpoints/v2/` are untouched). The step-5 label-sanity checkpoint
was trained straight into the scratchpad (`--checkpoint-dir`) to begin
with and never touched the repo. `git status` after cleanup shows no
untracked/modified files under `models/ppo/checkpoints/` — checkpoints
are gitignored, so this was filesystem hygiene only, not a git concern.

---

## Active Plan

**M5 Execution Plan (2026-07-16) — ACTIVE:**

Phase 0 (commit the post-M4 tuning tree) →
Phase 1 [**A:** gym opponent-frame labels + TS tests ∥ **B:** PPO opp head +
buffer] →
Phase 2 [**C:** bridge/clients/trainer plumbing ∥ **D:** evaluate.py
opp-accuracy + MCTS head sampler] →
Phase 3 (smoke battery + code commit) →
Phase 4 (5M-step v2 decision run → sweep → raw confirmations → tuned-MCTS
head-vs-policy sampler A/B → verdict vs pre-registered C1–C5).

Full spec — design decisions (v2 base, fresh run, label grounding in the
opponent's action frame, CE masking rules, sampler integration), file list,
protocol, and success criteria — in `MILESTONES.md` → M5. Task-level
checklist in Active Tasks (M5) above. Pre-registered contingencies: sweep
band < 45% vs Random → one rerun at λ=0.05; opp accuracy vs DamageFirst
< 25% → label bug, fix and rerun; nothing else.

---

**Post-M3 roadmap (decided 2026-07-14): M3.1 → M3.2 → M3.3, then resume M4+**

1. ~~**M3.1 Parallel training infrastructure**~~ ✅ Done (see Active Tasks above)
2. **M3.2 BC→PPO degradation fix** — next up. Store real action masks in the buffer and use them in `update()`; `--value-warmup-steps` (freeze embed/encoder/policy_head, train value head only); `--bc-anchor`/`--bc-anchor-coef` (KL to frozen BC policy); then a full warm-started decision run vs the 51% MLP baseline. Full spec in `MILESTONES.md` → M3.2.
3. **M3.3 Self-play + opponent pool** — heuristic DamageFirst-style opponent wired through gym/bridge/trainers/eval first, then dual-seat self-play against a frozen checkpoint pool. Full spec in `MILESTONES.md` → M3.3.

**M3 Execution Plan — concluded (negative result):**

1. ~~**`TransformerAgent`** (composes `TransformerPolicy`, PPO wrapper)~~ ✅ Done
2. ~~**`models/transformer/train.py`** (rollout PPO loop, `--pretrain_checkpoint` warm-start, `--resume`)~~ ✅ Done
3. ~~**`evaluate.py --model transformer`**~~ ✅ Done
4. ~~**From-scratch training run (2.6M steps) + evaluation**~~ ✅ Done — **32% win rate (158/500)**, underperforms the 51% MLP-PPO baseline
5. ~~**Warm-started training run (2.6M steps, `--pretrain_checkpoint`) + evaluation**~~ ✅ Done — **41% win rate (204/500)**, better than from-scratch but still below the MLP baseline
6. ~~**Extend warm-started run to 7.6M steps + checkpoint sweep**~~ ✅ Done — revealed a peak of **46%** at 2.5M–3.1M steps followed by violent collapse (down to 0%) and oscillation through 7.6M steps
7. ~~**Add PPO stability fixes (approx-KL early-stop, LR annealing) + retrain (5M steps) + checkpoint sweep**~~ ✅ Done — collapses eliminated, but the run still peaks early (45% @ 500k) and gradually decays (27% @ 5.0M final)
8. ~~**Final M3 conclusion**~~ ✅ **Negative result, decisive.** Transformer PPO (warm-started) tops out at 46% across ~40 evaluated checkpoints from two full runs — never beats, or sustainably matches, the 51% MLP-PPO baseline. Per `MILESTONES.md`'s own M3 guidance, M4/M5 should not proceed on top of this architecture

M2 is fully complete (see below).

**M2 Execution Plan — final status (all resolved):**

1. ~~**Token schema** (in `feature-extractor.ts`)~~ ✅ Done — 12 tokens, 65 dims each, matches `metamon_adapter.py` exactly
2. ~~**Stat boost tracking**~~ ❌ Dropped — see "Deviation from the original M2 plan" above
3. ~~**Bridge update** (in `gym_bridge.js` + `gym_client.py`)~~ ✅ Done — 780-float flat array, `--flat`/`structured=False` for backward compat
4. ~~**Verification**~~ ✅ Done — MLP PPO on flattened structured obs, 2.6M steps (~50k battles), **51% win rate vs RandomPlayerAI over 500 real evaluation battles**. Parity with the M1 flat baseline confirmed; token representation is not broken.

---

## Recently Completed

✅ **M4: MCTS Integration — positive result, search clearly improves play** (2026-07-15)
- **Forward model** (`sim/tools/battle-sim.ts`): clones the live gym battle
  (engine state + tracker state) in 0.8ms, steps both seats with gym-identical
  obs/mask/reward semantics, forks for tree branching; determinizer resamples
  the searcher's opponent's unrevealed slots from the format generator (either
  perspective). Engine gotcha fixed: locked states auto-complete a seat's
  choice (`side.isChoiceDone()`) — submitting anyway desyncs the battle
- **Search** (`models/mcts/mcts_agent.py` + bridge `sim_*` protocol +
  `evaluate.py --model mcts`): root-parallel determinized PUCT, policy prior +
  value leaves, opponent modeled by the same policy on its reveal-tracked obs
- **Results** (100 sims / 4 det, base = M3.3 best): 60.2% (602/1000)
  seat-balanced h2h vs the raw checkpoint; 56% vs DamageFirst (raw 46%); 66%
  vs Random (raw 57%, +8.6pp — the one criterion narrowly missed, bar was
  +10pp); 84–88ms/move. 4 of 5 criteria ✅. Committed `9da4d273a` (code);
  13 new TS tests. Full trail in `MILESTONES.md` → M4
- **Reading:** confirms M3.4's hypothesis — lookahead was the binding
  constraint, not observations or opponent distribution. Gains grow with
  opponent strength (+8.6pp vs Random, +11pp vs DamageFirst, +10.2pp h2h)

✅ **M3.4: obs schema v2 + mixed-opponent training — negative result, all criteria unmet** (2026-07-15)
- **Part A:** `TOKEN_DIM_V2=77` — 7 boost stages + Reflect/Light Screen/Substitute/Leech Seed flags + toxic counter appended per token (v1 prefix byte-identical), tracked in `pokemon-gym.ts` from public log lines only, active tokens only; obsMode `'structured-v2'` wired through bridge (`--obs-v2`), clients (`obs_v2=`), trainer, and `evaluate.py`
- **Part B:** `--opponent-mix` per-rollout family sampling (selfplay/damagefirst/random) with per-reset opponent switching in the bridge (no respawn); pool seeded with M2 + M3.3-best; cross-schema play via `slice_structured_obs()` (v1 checkpoints act inside v2 envs, and `--vs-checkpoint` pairs v2 vs v1 directly)
- Verified: 23/23 gym tests, full smoke battery, v1 regressions; committed `7533e08d7`
- **Decision run:** 4.77M/5M steps (externally killed, no crash), stable 42–62% sweep band, **no collapse**. Confirmations: best **54% (272/500) vs Random**, **46% (92/200) vs DamageFirst**, **48.0% (480/1000)** h2h vs the M3.3 best — an M2/M3.3 peer. All three pre-registered criteria ❌. Artifacts: `models/ppo/checkpoints/v2/{train.log,sweep_results.txt,confirm_results.txt,h2h_results.txt}`
- **Reading:** 4 independent 5M-step-class runs now land in the same 51–57% band — observation richness and opponent distribution are ruled out as the bottleneck. M4 (MCTS) proceeds on the M3.3 best checkpoint with relative criteria

✅ **M3.3 closed (head-to-head eval) + M3.4 milestone defined + M4 criteria recalibrated** (2026-07-14)
- Added `--vs-checkpoint` to `models/evaluate.py` — checkpoint-vs-checkpoint head-to-head through the dual-seat self-play bridge path (`_run_battles_h2h`; PPO-only, both checkpoints share the obs mode; wins counted from seat p1)
- Ran the M3.3 head-to-head, seat-balanced: self-play 4.75M vs M2 checkpoint = 51% as p1 (254/500), 54% as p2 (270/500) → **52.4% combined (524/1000)** — inside the ±3.1pp CI, statistical parity. M3.3 marked ✅ complete with a mixed verdict: self-play fixed training stability but produced a peer of the M2 agent, not a stronger one
- Defined **M3.4 (raise the policy ceiling)** in `MILESTONES.md`: Part A obs schema v2 — boost stages + screens/Substitute/Leech Seed/toxic counter, a deliberate TOKEN_DIM bump now that the transformer retirement (M3.2) ended the BC-checkpoint schema freeze; Part B mixed-opponent training — seed the self-play pool with the M2 checkpoint (zero code) + `--opponent-mix` per-rollout sampling across selfplay/damagefirst/random. Rationale: best policy is 57% vs Random while M4's criteria assumed ≥90%; MCTS amplifies whatever policy quality it's given
- Recalibrated M4's success criteria (were transformer-based and assumed the ≥90% base): now relative to the base policy — MCTS must add ≥5pp vs DamageFirst and ≥10pp vs Random over the same checkpoint without search

✅ **M3.3 comparison run: 5M-step MLP-PPO self-play + evaluation sweep** (2026-07-14)
- Ran the full self-play training run (`--opponent selfplay`, own-checkpoint pool, 8 envs, 5M steps, ~1.9h). Rollout win rates oscillated 0.2–0.8 as expected (each rollout samples a different frozen pool opponent); loss stayed 0.03–0.09 throughout
- 20-checkpoint sweep (150 battles each vs Random): rises from 42% @ 250k into a stable 47–61% band — **the first run in this project whose eval strength improves over training instead of peaking early and decaying**. No collapse anywhere
- Confirmed top candidates at full battle counts (500 vs Random / 200 vs DamageFirst): best is `ppo_step_4750059.pt` at **57% (287/500) vs Random** — ~2σ above the M2 baseline's 51% — but **46% (91/200) vs DamageFirstAI** vs the M2 agent's 51%: self-play did *not* transfer better to the held-out heuristic (all top checkpoints landed 43–46%)
- Verdict per M3.3 criteria: sanity floor ✅ (no regression vs Random); the DamageFirst half of the main criterion ✗; head-to-head half still unrun (needs a checkpoint-vs-checkpoint mode in `evaluate.py`). Results + reading recorded in `MILESTONES.md` → M3.3

✅ **M3.2: BC→PPO degradation fix — transformer retired, MLP-PPO is the architecture** (2026-07-14)
- Implemented all three fixes: per-step `valid_mask` stored in `TrajectoryBuffer` and used in both agents' `update()` (updates previously pretended all 9 actions were legal); `--value-warmup-steps` (freeze embed/encoder/policy_head so the random value head fits the BC policy before full PPO gradients hit the shared encoder); `--bc-anchor`/`--bc-anchor-coef` (frozen BC policy as an annealed KL anchor). Plus `--checkpoint-dir` on both trainers so new runs can't overwrite old checkpoints. Verified freeze semantics (policy tensors bit-identical through a frozen update), anchor annealing, and a live warmup→unfreeze transition
- Fixed an MPS blocker found en route: `TransformerPolicy`'s encoder now uses `enable_nested_tensor=False` (the eval-mode fast path used aten ops unimplemented on MPS; numerically identical, checkpoint-compatible)
- **Decision run:** 5M steps warm-started with all fixes (`models/transformer/checkpoints/m32/`), 20-checkpoint sweep + 500-battle confirmations. The diagnosis was right — no collapse, the policy holds 44–55% for 3.5M steps, and decay resumes exactly as the anchor anneals to zero (35%→25% over the last 1.25M steps): unconstrained PPO erodes the BC policy at any LR. But the ceiling is the BC policy itself: best confirmed **53% (263/500) vs Random** (parity with MLP's 51%) and **39% (77/200) vs DamageFirstAI** (MLP: 51%)
- **Per the milestone's pre-registered criterion: transformer retired; M4 (MCTS) and M5 (opponent modeling) proceed on the M2 MLP-PPO baseline**

✅ **M3.3 code: DamageFirst opponent + dual-seat self-play** (2026-07-14)
- `sim/tools/damage-first-ai.ts` (`DamageFirstAI`, highest-base-power move) wired as `--opponent damagefirst` through gym → bridge → `GymClient`/`VecGymClient` → both trainers → `evaluate.py`. New baseline: M2 MLP scores **51% (101/200) vs DamageFirstAI**
- Dual-seat self-play: `PokemonGymEnv` `opponent: 'self'` + `resetDual()`/`stepDual()` (both seats' obs/mask/needsAction per decision point; freshness-counted requests; force-switches are single-seat decision points; reveal tracker runs for both sides so each seat sees only revealed info; reward stays p1-perspective). Bridge `--selfplay` dual protocol; `VecGymClient.reset_all_dual()/step_dual()` pipelined with auto-reset
- Both trainers: `--opponent selfplay` samples one frozen opponent per rollout from `--selfplay-pool` (default: own checkpoint dir; 50% newest / 50% uniform; frozen copy of the current policy until a first checkpoint exists). Pending-transition collection credits rewards from opponent-only steps to p1's open transition
- Smoke-verified at every layer: TS (episodes complete, force-switch points, no illegal moves), Python bridge, and short self-play training runs on both trainers. Comparison training runs remain (see Next Steps)

✅ **M3.1: Parallel training infrastructure** (2026-07-14)
- Root-caused training slowness: fully serial pipeline (one bridge subprocess, one battle at a time, batch-1 inference every step — GPU mostly idle). Built `models/vec_gym_client.py` (`VecGymClient`): N parallel `gym_bridge.js` subprocesses with pipelined stepping (write all N commands, then read all N responses — Python waits only for the slowest env) and auto-reset on episode end
- Batched inference (`act_batch()`) on both `PPOAgent` and `TransformerAgent`; per-env `TrajectoryBuffer`s with `merge_buffers()` doing global advantage normalization (GAE never crosses env streams); `--num-envs` (default 8) + `--device` on both trainers and `evaluate.py`
- Device facts recorded in `docs/ML-TRAINING.md`: CUDA auto-detected already (RTX 3080 box = clone + build + CUDA torch, zero code changes; checkpoints portable both directions); Apple Neural Engine is unreachable from PyTorch (CoreML-only) — MPS is the Mac backend and was already in use
- Verified end-to-end: vec smoke, train smokes (both trainers, both PPO obs modes), checkpoint round-trips, `--resume`, and the M2 51% MLP checkpoint reads 49% (49/100) through the parallel path — old checkpoints fully compatible
- Measured: transformer eval 100 battles 20.2s → 5.6s (8 envs); training throughput table in `docs/ML-TRAINING.md`
- Roadmap updated: new M3.1/M3.2/M3.3 milestones in `MILESTONES.md`; M4/M5 resume after M3.2's architecture decision

✅ **M3 concluded: stability-fix retrain confirms negative result** (2026-07-13)
- Extended the 2.6M-step warm-started run to 7.6M steps via `--resume`, then swept all 21 intermediate checkpoints (150 battles each) to see the full trajectory rather than just two data points. Result: the run peaked at **46% win rate** around 2.5M–3.1M steps, then **collapsed violently** — 0–13% win rate between 3.6M and 5.6M steps, a partial recovery to 32% at 6.1M, then drifting to 24% by 7.6M. This is a much more severe finding than the earlier two-point (41%→24%) comparison suggested: it's not gradual drift, it's repeated near-total collapse and partial recovery, consistent with an unconstrained PPO update occasionally wrecking the policy's learned move-vs-switch balance (any policy that over-favors switching loses almost every game to `RandomPlayerAI`, which never voluntarily switches)
- **Added two standard PPO stability fixes**, since the failure pattern (violent collapse, not slow decay) pointed at unconstrained per-update policy movement rather than a capacity problem:
  - `models/transformer/transformer_agent.py`: approximate-KL early-stopping in `update()` — `target_kl=0.02` (new `TransformerAgent` hyperparameter, backward-compatible default for old checkpoints via `cls(**hparams)`). If a minibatch update pushes `approx_kl` (the http://joschu.net/blog/kl-approx.html estimator) past `1.5×target_kl`, the rest of that rollout's PPO epochs are skipped rather than continuing to update on an already-large policy shift. `update()` now returns `(loss, kl_early_stop: bool)` instead of just `loss`
  - `models/transformer/train.py`: linear LR annealing toward 0 over `--steps`, recomputed once per rollout from `agent._hparams["lr"] * max(0, 1 - total_steps/total_budget)`, applied directly to `agent.optimizer.param_groups`. Naturally continues decaying correctly on `--resume` since it's keyed off absolute `total_steps`/`total_budget`, not a step counter that resets. Rollout log line now also prints current LR and a `[kl early-stop]` tag when the safeguard fires
  - Both smoke-tested at 1k–2k steps before the real run; KL early-stop was observed firing correctly in the (very noisy, small-rollout) smoke test
- **Retrained a fresh 5,000,000-step warm-started run** with both fixes (not resumed from the already-collapsed 7.6M checkpoint — a fresh run tests whether collapse is prevented from the start). Swept 20 checkpoints (250k–5.0M) plus a full 500-battle evaluation of the final checkpoint. Result: **the violent collapses are gone** (no checkpoint dropped below ~25%), confirming the fix addressed that exact failure mode. But the run now peaks almost immediately — **45% at 500k steps**, close to where the BC-pretrained starting policy already performs — and then **gradually decays** into a noisy 25–35% band for the remaining 4.5M steps, ending at **27% (135/500)** on the full 500-battle evaluation of the final checkpoint
- **Conclusion:** across ~40 checkpoints evaluated spanning both warm-started runs, the best-ever recorded win rate is **46%** — the transformer never beats, and never sustainably matches, the 51% MLP-PPO baseline. The stability fixes changed *how* training fails (violent collapse → gradual decay) but not *whether* continued PPO fine-tuning improves on the BC-pretrained starting point — it doesn't; performance is highest very early and erodes from there. This is a decisive, well-supported negative result, not an artifact of insufficient training budget (two runs, ~2.6M and ~5–7.6M steps) or an unaddressed instability bug (the specific instability found was fixed, and the conclusion didn't change)
- **MILESTONES.md updated** — M3 marked complete with this negative result; M4/M5 held per the project's own stated M3 recommendation
- Checkpoints: `models/transformer/checkpoints/pretrained/transformer_step_2500000.pt` (run 1 peak, 46%) and `transformer_step_5000000_final.pt` (run 2 final, 27%)

✅ **M3 verification: warm-started transformer run (41%) — still below MLP baseline** (2026-07-11)
- Trained the transformer PPO agent warm-started from the BC checkpoint (`models/checkpoints/bc_pretrain_gen1ou.pt`) for 2.6M steps — same budget as the from-scratch run and the M2 MLP baseline. Evaluated 500 real battles vs RandomPlayerAI: **41% win rate (204/500)** — a +9pp improvement over the from-scratch transformer (32%), confirming BC warm-start helps, but still below the MLP-PPO baseline's 51% at equal compute
- Comparison table at 2.6M steps: MLP-PPO baseline 51% (254/500) > transformer warm-started 41% (204/500) > transformer from-scratch 32% (158/500)
- **Decision:** 2.6M steps (~50k battles) was chosen to match the M2 baseline for a controlled comparison, but it's well short of MILESTONES.md's actual M3 target range (200k–500k battles / ~10.4M–26M steps). Transformers are typically more sample-hungry than MLPs, so it's plausible the warm-started variant is still climbing rather than having plateaued below 51%. Resuming the warm-started run to **10.4M steps** (~200k battles, the low end of the target range) before drawing a final M3 conclusion
- Run was interrupted (Ctrl+C, not a bug — `KeyboardInterrupt` mid-`agent.act()`) at step 7,600,000; resumed via `--resume models/transformer/checkpoints/pretrained/transformer_step_7600000.pt`
- **Next:** finish the 10.4M-step run, evaluate at 500 battles, compare against the 51% baseline one more time — that result is the real M3 conclusion

✅ **M3 verification: from-scratch transformer run (32%) + --resume flag** (2026-07-10)
- Trained the transformer PPO agent from scratch for 2.6M steps (same budget as the M2 MLP-PPO baseline). Evaluated 500 real battles vs RandomPlayerAI: **32% win rate (158/500)** — a real result, well below the MLP baseline's 51% at equal compute. Does not meet the M3 success criterion (transformer must beat, not just match, the MLP baseline)
- **Diagnosed a scare mid-run:** rollout win rate sat at ~0% for a long stretch early in training (step ~500k/2.6M), which looked like a possible bug. Investigated by testing a completely untrained `TransformerAgent`, a completely untrained MLP `PPOAgent`, and a literal uniform-random-over-all-valid-actions policy — all three scored 0-3% vs `RandomPlayerAI`, ruling out anything transformer-specific. Root cause: `RandomPlayerAI` (`sim/tools/random-player-ai.ts`) defaults to `move: 1.0`, so its switch condition (`this.prng.random() > this.move`) is never true — it essentially never voluntarily switches, only attacks. A policy that hasn't yet learned to prefer moves over switches (true of any untrained/early-training policy, since ~5/9 actions are switches) loses almost every battle to an opponent that presses the attack every turn. Not a bug — expected early-training behavior that resolves as the policy learns, which the final 32% (vs. near-0% early on) confirms
- **Added `--resume <checkpoint>` to `models/transformer/train.py`** after a training run was killed mid-flight. Restores full agent + optimizer state via `TransformerAgent.load()`, parses the starting step count from the checkpoint filename (`transformer_step_N.pt`), and reuses the same checkpoint directory (scratch/pretrained) so the resumed run keeps appending to the original sequence. Mutually exclusive with `--pretrain_checkpoint` (resuming already restores whatever weights the original run started from). Smoke-tested: resumed from a fake `step_100` checkpoint with `--steps 300`, correctly continued to step 300 and saved into the same directory
- Checkpoint: `models/transformer/checkpoints/scratch/transformer_step_2600000_final.pt`
- **Next:** the warm-started run (BC pretrain checkpoint) is the actual test of the M3 hypothesis — in progress now

✅ **M3 code: Transformer PPO agent + training loop** (2026-07-10)
- Created `models/transformer/transformer_agent.py` — `TransformerAgent(nn.Module)` composes `TransformerPolicy` (built in M2.5) as `self.policy` rather than subclassing it, so the policy's state-dict keys (`embed.*`, `encoder.*`, `policy_head.*`, `value_head.*`) stay unprefixed and `load_pretrain_checkpoint()` — which matches by exact key name — loads the BC checkpoint's 30/30 tensors into `agent.policy` directly. `act()`/`evaluate_actions()`/`update()`/`save()`/`load()` mirror `models/ppo/ppo_agent.py`'s `PPOAgent`, duplicated rather than shared via a mixin (matches the existing per-model-family convention — `q_learning`/`dqn`/`ppo` already each duplicate `_pick_device()` with no shared base class)
- Created `models/transformer/train.py` — same rollout/GAE/checkpoint/logging structure as `models/ppo/train.py`, but always `GymClient(structured=True)` unflattened (no `--structured` flag — the transformer has exactly one obs shape). New `--pretrain_checkpoint <path>` flag calls `load_pretrain_checkpoint(agent.policy, path)` — **not** `agent` — before training starts; checkpoints split into `checkpoints/scratch/` vs `checkpoints/pretrained/` so a from-scratch and warm-started run at the same step count never collide
- Modified `models/evaluate.py` — added `"transformer"` to `--model` choices and a `_load_agent` branch; split `_run_battles`'s single `structured` bool into `structured` (does `GymClient` return `(12,65)`) and `flatten` (additionally reshape to `(780,)`) — PPO-structured needs both True, transformer needs `structured=True, flatten=False`. No new CLI flag; the flatten decision is derived from `--model`
- Updated `models/CLAUDE.md`'s directory table and Quick-Start block with the new `transformer/` files and commands
- Smoke-tested end-to-end: agent construction/act()/save()/load(), BC checkpoint warm-start (`[pretrain] loaded 30/30 tensors`, confirming the `agent.policy` vs `agent` wiring is correct), a 400-step training run in both scratch and pretrained modes (checkpoints land in the correct separate subdirs), and `evaluate.py --model transformer --battles 10` (loads and runs without shape errors)
- **Remaining for M3:** the real training runs (~200k–500k battles, both scratch and pretrained) and the win-rate comparison against the 51% MLP-PPO baseline haven't been executed yet (needs a background training session, same as M2's verification run)

✅ **M2 verification run + evaluate.py PPO hang bug** (2026-07-09)
- Ran the M2 verification: PPO with trunk `Linear(780,128)→ReLU→Linear(128,128)` on flattened structured obs, `--steps 2600000` (≈50k battles at the measured ~52 steps/battle), checkpoint at `models/ppo/checkpoints/structured/ppo_step_2600000_final.pt`
- First evaluation attempt (`evaluate.py --model ppo --structured --battles 500`) hung indefinitely (17+ min, no error, no output). Diagnosed live: the Python process was burning ~40% CPU while the `gym_bridge.js` subprocess was almost completely idle (19s CPU over 18 min) — a sign the loop was spinning without the battle engine doing any real work
- Root cause, confirmed by direct reproduction: `evaluate.py`'s `_run_battles()` did `action = agent.act(obs, valid_mask)` for all three model types. `QAgent`/`DQNAgent.act()` return a plain `int`; `PPOAgent.act()` returns `(action, log_prob, value)`. The 3-tuple was getting passed whole into `env.step()`, JSON-serialized as a JS array, and silently rejected as an out-of-range/invalid index by `pokemon-gym.ts` on every single call — meaning **every `evaluate.py --model ppo` run, ever, on any checkpoint, since the script was written on 2026-05-18, has hung on battle 1 without any visible error.** M2's structured obs work didn't cause this; it just was the first time anyone ran a PPO evaluation long enough to notice
- Fixed: `action = act_result[0] if isinstance(act_result, tuple) else act_result` — handles both agent shapes without the caller needing to know which model type is loaded
- Added a `models/CLAUDE.md`-documented `--structured` flag to `evaluate.py` for evaluating structured-obs checkpoints, and running win-rate progress logging (`Battle N/total | running win rate: ...`) so future runs are never silent for minutes at a time
- Re-ran the fixed evaluation: **51% win rate (254/500), 55 seconds, no hang** — this is the real, final M2 verification number
- **Also renamed** `--battles` → `--steps` in `dqn/train.py` and `ppo/train.py` — both flags actually counted environment steps, not battles, which collided in meaning with `evaluate.py`'s correctly-battle-counting `--battles` and would have caused real confusion about training scale later

✅ **M2 code: Structured State Representation** (2026-07-09)
- Added `extractFeaturesStructured()` + `TOKEN_DIM=65`/`N_TOKENS=12` to `sim/tools/feature-extractor.ts`; byte-for-byte layout match with `models/metamon_adapter.py` so the M2.5 BC checkpoint stays loadable
- Exported `parseHpRatio`/`parseStatus`/`fillStatusBitmask`/`parseLevelFromDetails`/`typeToIndex`/`categoryToIndex` from `feature-extractor.ts` for reuse
- Added an opponent-reveal tracker to `sim/tools/pokemon-gym.ts` — reconstructs revealed opponent Pokémon (species, HP, status, fainted, revealed moves) purely from `p2`-side battle-log lines, since a real player's request never contains opponent team info
- Added `obsMode: 'flat' | 'structured'` to `PokemonGymEnv` (default `'structured'`)
- `models/gym_bridge.js`: 780-float serialization by default, `--flat` restores the legacy 100-dim path
- `models/gym_client.py`: reshapes to `(12, 65)` by default, `structured=False` restores flat
- Fixed `models/{q_learning,dqn,ppo}/train.py` + `evaluate.py` to pass `structured=False` (their networks are hardcoded to 100-dim input — would have silently broken under the new default)
- Fixed a pre-existing bug: `parseLevelFromDetails()` defaulted to level 50 instead of 100 for Pokémon without an explicit `L##` tag (Showdown omits the tag at level 100, the gen1ou/gen1randombattle norm)
- Fixed a pre-existing build bug: `tools/build-utils.js`'s `copyOverDataJSON` didn't create destination directories before copying, which broke `./build` once `data/metamon_cache/` (2.1GB, gitignored) existed under `data/`
- Extended `test/tools/gym.test.js`: structured-obs shape/NaN checks, unknown-vs-fainted flag distinction, own-bench request-slot ordering, 300-step full-battle shape-stability smoke test — all passing; smoke-tested `gym_bridge.js`/`gym_client.py` end-to-end in both obsMode
- **Decided against** adding stat-boost features (see "Deviation" note above) — schema is locked to BC-checkpoint compatibility
- **Remaining for M2:** the 50k-battle MLP PPO verification run hasn't been executed yet (needs a background training session)

✅ **M2.5 code: Behavior Cloning Pretraining pipeline** (2026-07-02)
- Created `scripts/download_metamon.sh` — installs Metamon (UT-Austin-RPL), sets `METAMON_CACHE_DIR=./data/metamon_cache`, downloads parsed-replays for gen1ou, prints verification count
- Created `models/metamon_adapter.py` — `MetamonDataAdapter` streaming `(obs (12,65), action, done)` from Metamon replay JSONs; M2 token conventions; Metamon's alphabetical move/switch ordering preserved so `MinimalActionSpace` indices ground identically to our gym actions; skips `-1` actions; `FileNotFoundError` points to download script
- Created `models/transformer/transformer_policy.py` — shared `TransformerPolicy` (M3 architecture: 65→128 embed, 2-layer encoder, unknown-token attention masking) + `load_pretrain_checkpoint()` (skips name/shape-mismatched tensors with warnings)
- Created `models/bc_pretrain.py` — CE on policy head only, Adam lr=1e-3, batch 256, 5 epochs default, streamed 50k-sample shuffle buffer; flags `--epochs`, `--format`, `--checkpoint_dir`, `--max_files`
- Verified end-to-end on synthetic fixture battles: all adapter spec checks pass, BC loss decreases on MPS, checkpoint round-trips 30/30 tensors, deliberate shape mismatch skips with warning
- MILESTONES.md: added M2.5 entry; M3 `train.py` now specifies the `--pretrain_checkpoint` wiring requirement

✅ **Job 2.4: evaluate.py + MODEL-COMPARISON.md** (2026-05-18)
- Created `models/evaluate.py` — CLI evaluation script (`--model`, `--checkpoint`, `--battles`); loads agent from checkpoint, sets epsilon=0.0 for q_learning/dqn greedy eval, runs N battles via `GymClient`, reports win rate vs RandomPlayerAI
- Created `docs/MODEL-COMPARISON.md` — results comparison template with Overview, Models Evaluated, Results table (all TBD), Analysis placeholder, Winner Selection placeholder, Next Steps pointing to M3

✅ **Job 2.3: ppo/** (2026-05-18)
- Created `models/ppo/trajectory_buffer.py` — `TrajectoryBuffer` with GAE advantage computation, normalized advantages, `push()`/`compute_advantages()`/`get_tensors()`/`clear()` API
- Created `models/ppo/ppo_agent.py` — `PPOAgent` with shared trunk (100→128→128), policy head (128→9), value head (128→1), `act()` with valid_mask, `evaluate_actions()`, `update()` with PPO clipped surrogate + value loss + entropy bonus, `save()`/`load()` classmethod
- Created `models/ppo/train.py` — rollout-based loop collecting `rollout_steps` transitions across episode boundaries, per-rollout win-rate and loss logging, checkpoints to `models/ppo/checkpoints/ppo_step_{N}.pt`, `env.close()` in finally
- Created `models/ppo/checkpoints/` directory

✅ **Job 2.2: dqn/** (2026-05-18)
- Created `models/dqn/replay_buffer.py` — `ReplayBuffer` with deque(maxlen), `push()` converts to numpy, `sample()` returns 5 CPU torch tensors with correct dtypes
- Created `models/dqn/dqn_agent.py` — `QNetwork` MLP (100→128→128→9), `DQNAgent` with policy/target nets, ε-greedy `act()` with valid_mask, `learn()` with MSE loss, `update_target()`, `decay_epsilon()`, `save()`/`load()` classmethod
- Created `models/dqn/train.py` — step-based loop (not episode-based), rolling 500-episode win-rate/loss logging, checkpoints to `models/dqn/checkpoints/dqn_step_{N}.pt`, `env.close()` in finally
- Created `models/dqn/checkpoints/` directory

✅ **Job 2.1: q_learning/** (2026-05-18)
- Created `models/q_learning/q_agent.py` — `QAgent` with defaultdict Q-table, 5-element state discretization, epsilon-greedy `act()`, TD(0) `update()`, `decay_epsilon()`, pickle `save()`/`load()`
- Created `models/q_learning/train.py` — episode loop with `GymClient`, rolling 500-episode win rate logging, Q-table saved to `qtable.pkl` after training, `env.close()` in finally block
- Created `models/q_learning/README.md` — overview, architecture, training instructions, hyperparameters table, placeholder Results and Analysis sections

✅ **Job 1.2: gym_client.py** (2026-05-18)
- Created `models/gym_client.py` — `GymClient` class spawning `gym_bridge.js` via subprocess
- Implements `reset()`, `step()`, `valid_actions()`, `close()` with correct numpy dtypes
- Line-delimited JSON protocol: one command written per `_send()` call, one response line read back
- Error responses (`{"error":"..."}`) raise `RuntimeError`
- `__main__` block for smoke testing
- No external deps beyond `subprocess`, `json`, `numpy`, `pathlib`

✅ **Job 1.1: gym_bridge.js** (2026-05-18)
- Created `models/gym_bridge.js` — line-delimited JSON stdio server wrapping `PokemonGymEnv`
- Supports `reset`, `step`, `valid_actions`, `close` commands
- Sequential async processing preserves request/response ordering
- `obs` returned as plain Array (not Float32Array) for JSON serialization
- Unhandled exceptions written to stdout; process never crashes silently
- `models/` directory created

✅ **M1: Environment & Baseline Agent** (2026-05-10)
- **Job 3.1:** Created `sim/tools/pokemon-gym.ts` — `PokemonGymEnv` class with `reset()`, `step(action)`, `validActions()`, `destroy()`. Background omniscient reader, reward parsing, valid-action masking.
- **Job 3.2a:** Created `sim/tools/feature-extractor.ts` — 100-feature fixed-size extraction (own active, moves, switch mask, opponent, padding).
- **Job 3.2b:** Created `sim/tools/evaluator.ts` — Parallel battle runner with `evaluate()`, `evaluateVsRandom()`, up to 50 concurrent battles.
- **Job 3.1 (revised):** Added unit tests in `test/tools/gym.test.js` — reset/step validation, legal move masking, reward bounds, observation consistency, determinism check. Written as plain JS (matches project test convention) in `test/tools/` so mocharc picks it up automatically.
- **Job 3.4:** Updated `docs/AI-PLAYERS.md` — new "Gym Wrapper (PokemonGymEnv)" section covering what the gym is, quick start, observation/action space, reward function, evaluator usage, seeding.
- All code compiles under strict TypeScript with zero errors. Gym tested for 100 battles without crashes.

✅ **Job 2.2: pokemon-gym.ts** (2026-05-10)
- Created `sim/tools/pokemon-gym.ts`
- Exports `PokemonGymEnv` with `reset()`, `step(action)`, `validActions()`, `destroy()`
- Exports `GymStepResult` interface
- GymPlayer extends BattlePlayer with Promise-based request handoff
- Background omniscient reader buffers all battle lines for reward parsing
- Reward: +0.01 per opponent KO, -0.01 per own KO, +0.0001 status, ±1.0 win/loss, stalling penalty
- Compiles clean under strict TypeScript (zero errors)

✅ **Job 2.1: feature-extractor.ts** (2026-05-10)
- Created `sim/tools/feature-extractor.ts`
- Exports `extractFeatures(request, opponentRequest): Float32Array` and `OBS_SIZE = 100`
- 100-feature layout: own active (0–14), moves (15–54), switch mask (55–59), opponent (60–74), padding (75–99)
- Compiles under strict TypeScript with no errors

✅ **M0: Foundation**
- Build system (`./build`) working, `dist/` populated
- Documentation suite created (8 reference docs in `docs/`)
- Verified: `RandomPlayerAI` base class at `sim/tools/random-player-ai.ts`
- Verified: `BattleStream` API works for programmatic parallel battles

> **Note:** `simulate.js` in the repo root is an unrelated script (gym leader battles for a separate project). It is a useful code reference but is not a deliverable of this ML project. Its output in `output/` is similarly unrelated.

---

## Blockers

None. All components verified and working.

### Assumptions Made
- Python bridge for training models: new code (not integrating existing ML code)
- PyTorch preferred for neural networks (fast iteration)
- Gen1 format chosen for training (smallest state space, fastest convergence)

---

## Next Steps

### M2 — Structured State
✅ Complete and verified (51% win rate vs RandomPlayerAI, 500 battles). Nothing left here.

### M3 — Transformer + PPO
✅ **Complete — negative result (2026-07-13).** Two full training runs (from-scratch 2.6M steps, warm-started up to 7.6M steps, plus a stability-fixed warm-started retrain to 5M steps) and ~40 evaluated checkpoints all agree: transformer PPO tops out at **46%** win rate vs RandomPlayerAI, never beating (or sustainably matching) the M2 MLP-PPO baseline's **51%**. Continued PPO fine-tuning degrades performance from its early/BC-pretrained peak rather than improving it. See "Recently Completed" for the full diagnostic trail. Nothing left to run here — this is the final M3 result.

### M3.1 / M3.2 — Complete
✅ Both done (2026-07-14). `--num-envs 8` is the default everywhere; **the project architecture is MLP-PPO** (M3.2 decision — transformer retired after the fixes-run confirmed its ceiling is the BC policy).

### M5 — Complete
✅ Done (2026-07-16). Thesis negative (head sampler = policy sampler under
search); side finding: **new best agent = tuned MCTS (policy sampler) over
`models/ppo/checkpoints/opp/ppo_step_5000001_final.pt`** (72.6% DF / 86.0% R).
Verdict + reading in `MILESTONES.md` → M5.

### Immediate (post-M5)
1. ~~**Post-run cleanup pass**~~ — ✅ done 2026-07-16 (all 5 code-simplifier
   findings from the `ffd14e275` review applied; details in Active Tasks
   above).
2. **M6 (server integration & ladder)** — ship the M5 checkpoint with tuned
   policy-sampler MCTS (~60ms/move, well inside M6's 2s budget); optionally
   ladder-A/B vs the v2 control checkpoint.
3. Longer term (unchanged): AlphaZero-style fine-tuning on MCTS-played games
   (MILESTONES → M4 "Unblocks") — now with a stronger search agent to
   generate data.

### Stretch (deprioritized)
- Attention weight visualization to confirm non-uniform attention
- Start scoping MCTS determinizer (opponent team sampling from gen1 usage data)

---

## Key Metrics to Track

**M1 Success Indicators:**
- Gym env runs 100 battles without crashes
- Observation shape: consistent (same size every turn)
- Reward bounds: all rewards in [−1, +1]
- Baseline win rates:
  - RandomPlayerAI: ~50% (sanity check, should beat coin flip)
  - DamageFirstAI: ~60% (heuristic baseline)
- Evaluation runtime: < 5 min for 1500 battles

**M2 Success Indicators (later):**
- Model A (Tabular Q): ≥ 50% vs Random, < 70% vs DamageFirst (confirms limitation)
- Model B (DQN): ≥ 80% vs Random, ≥ 60% vs DamageFirst (primary hypothesis)
- Model C (PPO): ≥ 85% vs Random, ≥ 65% vs DamageFirst (stability)

---

## Dependencies & Ordering

**Hard blocking:** M1 must complete before M2 (gym needed by all models)

**Can overlap:**
- M2 & M3: If M2 winner is clear at 100k battles, can extrapolate training while exploring other models
- M3 & M4: Checkpoints from M3 can be loaded into M4 inference during training
- M4 & M5: Start human playtest while model still training (use stable checkpoints)

---

## Files & Directories

### Core Simulation (M0, already done)
- `sim/tools/random-player-ai.ts` — base AI class (extend for custom policies)
- `dist/sim/index.js` — compiled simulator API (`BattleStream`, `Dex`, `PRNG`)
- **Reference only (unrelated project):** `simulate.js`, `output/raw_battles.csv`

### M1 (Gym & Baseline)
- **New:** `sim/tools/pokemon-gym.ts` (gym wrapper)
- **New:** `sim/tools/feature-extractor.ts` (observation building)
- **New:** `sim/tools/evaluator.ts` (eval harness)
- **New:** `test/tools/gym.test.js` (unit tests)
- **Modify:** `docs/AI-PLAYERS.md` (gym usage section)

### M2 (Model Exploration)
- **New:** `models/` directory (root)
- **New:** `models/q_learning/` (tabular Q)
- **New:** `models/dqn/` (DQN)
- **New:** `models/ppo/` (PPO)
- **New:** `models/` base: `train_env.py`, `evaluate.py`
- **New:** `docs/MODEL-COMPARISON.md` (results & winner selection)

### M3 (Scale Training)
- **Modify:** `models/{dqn|ppo}/train.py` (checkpointing, self-play)
- **New:** `models/{dqn|ppo}/TRAINING-RESULTS.md` (logs & analysis)

### M4 (Showdown Integration)
- **New:** `server/bot-client.ts` (WebSocket)
- **New:** `server/battle-handler.ts` (inference)
- **New:** `docs/DEPLOYMENT.md` (setup guide)

### M5 (Evaluation)
- **New:** `server/elo-ladder.ts` (rating tracking)
- **New:** `docs/FUTURE-WORK.md` (research directions)

---

## Checklist for Session Start

When starting a new session:

- [ ] Review this file for current work status
- [ ] Check MILESTONES.md for milestone definitions
- [ ] If blocked, resolve blockers (see Blockers section)
- [ ] If continuing M1: check last day's progress
- [ ] Update "Recently Completed" when tasks finish
- [ ] Update "Active Tasks" when starting new work
- [ ] Commit this file after major progress

---

## Session Log

### Session 3 (2026-05-18)
- **Action:** Verified Job 3.1 — gym_bridge.js and gym_client.py integration tests
- **Status:** All three checks PASS
  - Check 1 (syntax): `node --check gym_bridge.js` → exit 0 ✓
  - Check 2 (bridge reset): reset returns valid JSON with obs array of length 100 ✓
  - Check 3 (Python client): gym_client.py smoke test runs without errors, obs shape (100,), valid_actions works, step returns correct tuple ✓
- **Notes:** numpy was not installed initially; installed via pip, then all tests passed
- **Next:** Ready to begin M2 phase training (Model A/B/C) or continue verification tasks

### Session 2 (2026-05-10)
- **Action:** Completed M1 gym wrapper, feature extractor, evaluator, unit tests, docs update
- **Status:** M1 complete. All code compiles under strict TypeScript. Gym tested 100+ battles without crashes.
- **Deliverables:** `pokemon-gym.ts`, `feature-extractor.ts`, `evaluator.ts`, `test/tools/gym.test.js`, `docs/AI-PLAYERS.md` (new Gym Wrapper section)
- **Next:** M2 model exploration — Python bridge, Model A (Tabular Q), Models B & C (DQN/PPO) in parallel

### Session 1 (2026-05-10)
- **Action:** Created docs suite (8 reference docs), MILESTONES.md, IN-PROGRESS.md
- **Status:** Project director defined 6 milestones (M0–M5), M0 complete, M1–M5 scoped
- **Clarified:** `simulate.js` is an unrelated script (not an ML project artifact)
- **Next:** Begin M1 gym wrapper implementation

