# Agent Jobs

**Task:** M7 — Observation Schema v3 (type effectiveness, move-effect flags, Sleep Clause signal; 86-dim per-token obs) — build, train, and ladder-validate
**Type:** New feature (build phases 0–3) followed by a training/eval pipeline (phases 4–6)

Full spec: `MILESTONES.md` → "M7: Observation Schema v3" (search that heading for the schema table, files-to-modify table, and pre-registered success criteria). Every agent below should read that section first — it is the source of truth for exact dim layout (86 dims/token, dims 0–76 byte-identical to v2, dims 77–85 new) and the acceptance bars.

---

## Phases

### Phase 0 — Spec Lock (parallel: no)

#### Job 0.1 — Type Chart, Move-Effect-Flag Mapping, and Encoding Decisions
- **Agent:** builder
- **Files:** `sim/tools/feature-extractor.ts` (read-only exploration of existing helpers/types), new/updated constants inside that file or a small companion module if one already exists for type data (check `data/typechart.ts` / `sim/dex-data.ts` first — reuse the engine's own gen1 type chart rather than hand-rolling a new 15×15 table)
- **Task:** This is the prerequisite design job all of Phase 1 depends on. Produce, as code (constants + types, not just notes) ready for Job 1.1 to consume:
  1. A gen1 type-effectiveness lookup (own-move-type × defender-type → multiplier) sourced from the engine's existing dex/typechart data (do NOT hand-transcribe a table if `Dex` already exposes one — grep for `getEffectiveness`, `typeChart`, or similar in `sim/`). Confirm dual-typed stacking behavior (e.g. Ice Beam vs Dragonite = 4x, Fire Blast vs Slowbro = 0.5x, Explosion vs Gengar = 0x) and decide the exact scaled encoding (e.g. multiplier/4 → [0,1]; document the chosen formula precisely, including how 0x immunity round-trips).
  2. A move-effect-flag mapping: for every move, whether it is recharge (Hyper Beam-style), self-KO (Explosion/Selfdestruct), priority, and what status (if any) it inflicts — derive from `Dex` move data fields (`selfdestruct`, `priority`, `flags.recharge`, `status`/`secondary.status`) rather than a hardcoded move-name list, so it generalizes to all moves in the dex, not just known examples.
  3. Confirm the exact dim layout matches MILESTONES.md's "Final proposal (method B)": dims 0–64 v1 unchanged, dims 65–76 v2 unchanged, dims 77–80 type-eff (4 dims, one per move slot), dims 81–83 effect-flags (recharge/self-KO/priority bits), dim 84 inflicted-status id, dim 85 Sleep Clause flag. `TOKEN_DIM_V3 = 86`.
  4. Write this output as exported constants/types at the top of `sim/tools/feature-extractor.ts` (or a new small `sim/tools/type-chart-v3.ts` if that keeps the diff cleaner) so Job 1.1 can import them directly instead of re-deriving.
- **Depends on:** none
- **Model: opus** — this requires real design judgment (confirming engine data sources, deciding encoding/scaling, resolving edge cases like 0x immunity and Rest-exclusion semantics) but is bounded/well-specified by the milestone doc, not open-ended research.

---

### Phase 1 — Core Feature Extraction (parallel: yes, both depend on 0.1)

#### Job 1.1 — Feature Extractor v3 Logic
- **Agent:** builder
- **Files:** `sim/tools/feature-extractor.ts`, new/updated tests in `test/tools/` covering the extractor directly (add alongside existing extractor tests if present, otherwise extend `test/tools/gym.test.js` per Job 3.3's scope — coordinate by only touching extractor-level unit tests here, not `gym.test.js`)
- **Task:** Implement `TOKEN_DIM_V3 = 86` and extend `extractFeaturesStructured(...)` to accept a new `v3Info` parameter (or equivalent) producing dims 77–85 per MILESTONES.md's "Final proposal (method B)" and Job 0.1's constants:
  - Dims 77–80: per-move-slot (4 moves) type-effectiveness of that move vs the opponent's active Pokémon's type(s), using Job 0.1's lookup + scaling.
  - Dims 81–83: move effect flags (recharge, self-KO, priority) per active Pokémon's move set, using Job 0.1's mapping.
  - Dim 84: inflicted-status id for the move (reuse the existing status enum already used elsewhere in this file).
  - Dim 85: Sleep Clause flag — this dim is fed by the tracker Job 1.2 builds in `pokemon-gym.ts`; this job just needs to accept and place that flag correctly at dim 85.
  - Dims 0–76 MUST remain byte-identical to the current v2 output — do not touch that code path except to append, and add a test asserting byte-equality of dims 0–76 between v2 and v3 calls on the same input.
  - No NaN/inf in any new dim (Criterion A of MILESTONES.md is a hard gate) — validate 0x-multiplier and unknown-move edge cases explicitly.
- **Depends on:** 0.1
- **Model: opus** — feature-extractor logic embeds real engine-semantics judgment (indexing move slots, aligning with existing v1/v2 code, edge cases), building on Job 0.1's already-resolved design decisions.

#### Job 1.2 — Sleep Clause Tracker + obsMode 'structured-v3'
- **Agent:** builder
- **Files:** `sim/tools/pokemon-gym.ts`
- **Task:** Add a Sleep Clause state tracker derived from the public battle log (this file already has a "reveal tracker" pattern for opponent info — follow that convention, do not add omniscient/hidden-state access). Per MILESTONES.md:
  - Flag = 1 while any opponent Pokémon that WE put to sleep is still asleep.
  - Self-induced Rest sleep does NOT count toward the clause — must exclude `|-status|...slp|[from] move: Rest` (or equivalent Rest-sourced status lines) from setting the flag.
  - The flag is NOT reset on switch — sleep persists on the bench (track per-Pokémon-slot state, not just "current active").
  - Clear tracked sleep state on `|faint|` for that Pokémon, and on `|-curestatus|` (e.g. from Sleep Talk waking naturally is not a thing in gen1, but do handle any cure/faint transition correctly).
  - Add a new `obsMode: 'structured-v3'` that wires this tracked flag into `extractFeaturesStructured`'s v3 parameter (calling into Job 1.1's new signature — this is a merge-order dependency the two builders should each build defensively against: assume Job 1.1's function signature matches what MILESTONES.md specifies literally, i.e. `extractFeaturesStructured(volatiles, v3Info)`).
  - Add tests (in `test/tools/gym.test.js` or wherever this file's existing tests live) for: sleep flag sets on opponent-inflicted sleep, does NOT set on Rest, persists across opponent switches, clears on faint.
- **Depends on:** 0.1
- **Model: opus** — stateful tracking derived from a public log stream, with several correctness-sensitive edge cases (Rest exclusion, persistence across switches, faint clearing) — genuine engine-semantics reasoning, not mechanical plumbing.

---

### Phase 2 — Plumbing & Data Regeneration (parallel: yes, both depend on 1.1 + 1.2)

#### Job 2.1 — Bridge / Gym-Client Serialization for v3
- **Agent:** builder
- **Files:** `models/gym_bridge.js`, `models/gym_client.py`, `models/vec_gym_client.py`
- **Task:** Add `--obs-v3` flag support end to end, following the exact existing pattern used for `--obs-v2` in these three files (mirror it, don't redesign it):
  - `gym_bridge.js`: recognize `--obs-v3`, request `obsMode: 'structured-v3'` from the gym, serialize the flat array as 1032 elements (12 tokens × 86 dims) instead of 924 (12×77).
  - `gym_client.py`: add `obs_v3=` handling that reshapes the flat array to `(12, 86)`.
  - `vec_gym_client.py`: same reshape logic for the vectorized/batched path, plus `set_obs_version()` (or equivalent existing mechanism) updated so v3 participates correctly in opponent-pool/obs-version mixing if that mixing logic exists today.
  - This is mechanical plumbing mirroring an established v2 pattern — no new design decisions, just correct propagation of the new dim count and mode string.
- **Depends on:** 1.1, 1.2
- **Model: sonnet** — mechanical plumbing following an established `--obs-v2` pattern already present in all three files.

#### Job 2.2 — Replay-Adapter Regeneration for v3
- **Agent:** builder
- **Files:** `sim/tools/replay-adapter.ts`, `models/replay_adapter_cli.js`
- **Task:** Update the replay adapter to produce v3 observations (obs shape (12, 86) instead of (12, 77)) using Job 1.1/1.2's new extractor path — output schema (obs + action + done) is unchanged, only the obs width changes. Then run a one-time batch regeneration of the existing replay trajectory data through the CLI, writing output to `data/replay_trajs/v3/` (do not overwrite the existing v2 shards — keep both for A/B evals per MILESTONES.md). Spot-check a handful of regenerated shards for correct shape and no NaN/inf before considering this done.
- **Depends on:** 1.1, 1.2
- **Model: sonnet** — the code change is a narrow follow-on to the extractor's new signature, and the batch regeneration step is running an existing script over existing data, not new design.

---

### Phase 3 — Trainer Wiring + Smokes (parallel: no — single shared job to avoid conflicting edits across scripts with overlapping CLI-arg conventions)

#### Job 3.1 — Trainer/Eval/Infra Wiring for v3 + Smoke Tests
- **Agent:** builder
- **Files:** `models/ppo/train.py`, `models/bc_pretrain_mlp.py`, `models/evaluate.py`, `models/infer_server.py`, `models/mcts/mcts_agent.py` (verify only — MILESTONES.md notes no changes needed if `gym_client.py` handles reshaping), `tools/ladder-bot/ladder-bot.js`, `test/tools/gym.test.js`
- **Task:**
  1. `bc_pretrain_mlp.py`: accept v3 trajectory shards (from `data/replay_trajs/v3/`, Job 2.2's output), verify input shape assertion updates from 924 to 1032, report per-format validation accuracy same as today.
  2. `ppo/train.py`: add `--obs-v3` flag; auto-infer `--obs-size` from the flag (1032) rather than hardcoding; route checkpoints to `checkpoints/v3/`.
  3. `evaluate.py`: add `--obs-v3`; implement v3-vs-v2 head-to-head slicing so both obs versions can be compared from the same forward pass via the existing `slice_structured_obs` helper (dims 0–76 sliced out of a v3 obs should equal a native v2 obs — this is the mechanism MILESTONES.md relies on for A/B comparison).
  4. `infer_server.py`: confirm/wire v3 obs-size support for live inference.
  5. `mcts/mcts_agent.py`: verify it is obs-shape-agnostic as MILESTONES.md claims (parameterized already) — only touch if it turns out NOT to be.
  6. `tools/ladder-bot/ladder-bot.js`: forward `--obs-v3` to the bridge; confirm no protocol changes are needed beyond flag passthrough.
  7. `test/tools/gym.test.js`: add tests for v3 shape (1032), v2-prefix byte-equality (dims 0–76), type-eff values on known matchups (4x/0.25x/immune cases from Job 0.1), Sleep Clause state tracking (including Rest exclusion and faint clearing — reuses Job 1.2's scenarios at the integration level), and full-battle v3 stability (a complete battle run through structured-v3 with no NaN/inf/crash).
  8. Run smokes: BC pretrain script starts cleanly on v3 data (few iterations, not full 2h run), PPO train script starts cleanly with `--obs-v3` (few steps), evaluate.py runs a small v3-vs-v2 comparison without error. Report exact commands and output.
- **Depends on:** 2.1, 2.2
- **Model: opus** — spans 7 files with real judgment calls (shape inference, slicing-equality mechanism, verifying MCTS's obs-agnosticism claim rather than assuming it, writing meaningful type-eff/Sleep-Clause test cases) — too broad and judgment-heavy for sonnet, but each individual change follows patterns already established by v2, so it doesn't need the most capable tier.

#### Job 3.2 — Full Test Suite + Criterion A Verification
- **Agent:** tester
- **Files:** n/a
- **Task:** Run the project's test suite (check `package.json` for the exact script, typically `npm test` or a targeted `test/tools/gym.test.js` run) and `./build` if TypeScript changed. Report exit codes and any failures verbatim. Specifically confirm Criterion A from MILESTONES.md passes: valid observation shape (12, 86) at every decision point, no NaN/inf in type-eff or any new dims, Sleep Clause flag toggles correctly in the new tests from Job 3.1. This must pass before any training spend (Phase 4) begins.
- **Depends on:** 3.1
- **Model: sonnet** — running existing scripts and reporting output verbatim; no design judgment.

---

### Phase 4 — BC Pretrain on v3 Data (parallel: no)

#### Job 4.1 — BC Pretrain Run
- **Agent:** builder
- **Files:** n/a (invokes `models/bc_pretrain_mlp.py` as a script; do not modify code — if a bug surfaces, stop and report rather than patching blind)
- **Task:** Run the full BC pretraining recipe on the v3 replay data (`data/replay_trajs/v3/`) using `models/bc_pretrain_mlp.py`, matching the same hyperparameters/recipe used for the current best v2 BC checkpoint (check `models/CLAUDE.md` and prior M5.5 commit for the exact invocation used previously). Expect ~2 hours. Report final per-format validation accuracy and the checkpoint path produced. This gates Phase 5 — do not proceed if the run crashes or produces obviously degenerate accuracy (near-random).
- **Depends on:** 3.2 (Criterion A must pass first)
- **Model: sonnet** — running an existing, already-designed training script with a known recipe; success is judged by reported metrics, not code judgment.

---

### Phase 5 — PPO Fine-Tune (parallel: no)

#### Job 5.1 — PPO Fine-Tune Run
- **Agent:** builder
- **Files:** n/a (invokes `models/ppo/train.py` as a script)
- **Task:** Run the PPO fine-tune recipe (5M steps, opponent-mix recipe, 8 envs per MILESTONES.md) using `models/ppo/train.py --obs-v3 --pretrain-checkpoint <Job 4.1's checkpoint> --bc-anchor ...` (match the exact flag set used for the M5.5 fine-tune recipe — check `models/CLAUDE.md` / prior commit for the precise invocation), writing checkpoints to `checkpoints/v3/`. Expect ~2 hours. Report final checkpoint path and training curve summary (reward trend, any instability).
- **Depends on:** 4.1
- **Model: sonnet** — running an existing, already-designed training script with a known recipe.

#### Job 5.2 — Criterion B Bot Evals
- **Agent:** tester
- **Files:** n/a (invokes `models/evaluate.py`)
- **Task:** Run `models/evaluate.py --obs-v3` for the fine-tuned v3 checkpoint (Job 5.1) + tuned MCTS vs Random and vs DamageFirst, matching the same eval protocol used for the M6/M5.5 numbers (500 battles per MILESTONES.md convention). Report win rates verbatim and check against Criterion B from MILESTONES.md: ≥80% vs Random (target within 10pp of 90.6%) OR ≥65% vs DamageFirst (within 10pp of 79.2%), OR no regression if v3 hits 70–80% vs both. State clearly whether Criterion B is met, and per MILESTONES.md's conditional rule: if no regression, ladder run (Phase 6) is mandatory regardless of exact numbers.
- **Depends on:** 5.1
- **Model: sonnet** — running an existing eval script and comparing reported numbers against pre-registered thresholds already stated in MILESTONES.md; no judgment call beyond arithmetic comparison.

---

### Phase 6 — Ladder Criterion Run (parallel: no)

#### Job 6.1 — 100+ Game Ladder Run
- **Agent:** tester
- **Files:** n/a (invokes `tools/ladder-bot/ladder-bot.js`)
- **Task:** Run the ladder bot with the v3 checkpoint for ≥100 consecutive rated games (`--obs-v3` flag from Job 3.1), matching the same invocation used for the M6 100-battle criterion run (check prior commit `5592bfb1f "M6 COMPLETE: 100-battle ladder criterion run"` for the exact command). Confirm ≤2s per move and zero crashes throughout (Criterion C hard requirements). Report final Elo/Glicko-1 and GXE after the run completes. Evaluate against MILESTONES.md's Criterion C bands: GXE ≥35% = clear win, 25–34% = inconclusive/noise band (pre-committed as neither win nor loss), <25% = clear regression. State the verdict explicitly using these exact bands — do not editorialize beyond them.
- **Depends on:** 5.2 (only proceed if Criterion B passed or "no regression" per MILESTONES.md's conditional rule)
- **Model: sonnet** — running the existing ladder-bot script and reporting metrics against pre-registered numeric bands already defined in MILESTONES.md; no design judgment needed.

---

## Parallelization Notes

- Phase 0 is a hard sequential prerequisite — Jobs 1.1 and 1.2 both consume its type-chart/flag-mapping constants and cannot start before it lands.
- Phase 1 (1.1, 1.2) can run in parallel: they touch different files (`feature-extractor.ts` vs `pokemon-gym.ts`). Coordinate only on the function signature contract (`extractFeaturesStructured(volatiles, v3Info)`) — both jobs should build against the literal signature given in MILESTONES.md rather than waiting on each other.
- Phase 2 (2.1, 2.2) can run in parallel: they touch disjoint file sets (bridge/gym-client vs replay-adapter) and both only need Phase 1's extractor/tracker to exist.
- Phase 3 is a single combined job (3.1) rather than split, because `train.py`, `evaluate.py`, `bc_pretrain_mlp.py`, `infer_server.py`, and `ladder-bot.js` share CLI-arg and obs-size conventions that are easy to make inconsistent if edited by different cold agents in parallel — one agent keeps the `--obs-v3` contract consistent across all of them. Job 3.2 (tester) is strictly after 3.1.
- Phases 4, 5, 6 are strictly sequential (each is a multi-hour run gating on the previous one's output/checkpoint) and each has a pre-registered stop condition — do not proceed past a failed gate without reporting back for a decision.
- Total build-phase parallel width never exceeds 2 concurrent builders at once (Phase 1 and Phase 2), consistent with the cap.

---

## Context for All Agents

- Full schema spec, files-to-modify table, and pre-registered success criteria live in `MILESTONES.md` → "M7: Observation Schema v3" — read that section before starting any job, it is more detailed than this file.
- Dims 0–76 of the v3 observation MUST remain byte-identical to the current v2 output (`TOKEN_DIM_V2 = 77`) — this is what makes `slice_structured_obs`-based v2/v3 A/B comparison valid. Any job touching the extractor must preserve this invariant and should add a test for it if one doesn't exist yet.
- `TOKEN_DIM_V3 = 86`; full obs shape `(12, 86)`; flat serialized size `1032` (12 × 86), vs v2's `924` (12 × 77).
- Sleep Clause semantics: flag = 1 while any opponent Pokémon WE put to sleep is still asleep; self-induced Rest sleep does NOT count; NOT reset on switch (bench-persistent); cleared on faint. This is derived from the public battle log only (existing reveal-tracker convention) — never from hidden/omniscient state.
- Prior art to mirror rather than reinvent: v2's `--obs-v2` flag pattern in `gym_bridge.js` / `gym_client.py` / `vec_gym_client.py` is the template for v3's plumbing (Job 2.1); v2's `obsMode` convention in `pokemon-gym.ts` is the template for `'structured-v3'` (Job 1.2).
- Training recipe invocations (BC pretrain flags, PPO fine-tune flags with `--pretrain-checkpoint`/`--bc-anchor`) should match what was used for the M5.5 positive result — check `models/CLAUDE.md` and the M5.5/M6 commits (`eb5d48632`, `5592bfb1f`) for exact prior commands rather than guessing new hyperparameters.
- Every Phase 4–6 job has a pre-registered numeric gate from MILESTONES.md (Criteria A/B/C) — report numbers and state the verdict against those exact bands; do not proceed past a failed/ambiguous gate without surfacing it back rather than making a unilateral call.
- After the full plan executes (through Phase 6), update `IN-PROGRESS.md` and `MILESTONES.md`'s M7 status line per the project's documentation-maintenance convention (update existing docs, no new milestone-completion files) — this can be a final small step by whichever agent runs Job 6.1, or a follow-up documentation-manager pass if the orchestrator's caller prefers.
