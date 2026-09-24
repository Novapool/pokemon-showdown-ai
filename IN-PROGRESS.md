# In Progress — Pokemon Showdown AI Training

Last updated: 2026-09-24

---

## Where we are

**ML research is closed. M13 (experiment tracking + CI) closed 2026-09-24 with
its gate passed. There is no live milestone.** Status:
`MILESTONES.md` → PROJECT STATUS. Orientation: `docs/WHERE-WE-ARE.md`.

**🚫 Do not add ML scope.** No new models, training arms, hypotheses or schema
work. Tracked re-evaluations of existing checkpoints are fine. Anything
interesting they surface is a note, not a reason to reopen research.

**Shipping agent: unchanged M7 checkpoint** (`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt`,
tracked in git). Raw greedy vs Random: **77.26%** (7,726/10,000 pooled). That
number is the CI gate's baseline.

---

## How the tooling works now (M13)

- **Every train/BC/eval run is tracked** in `mlflow.db` at the repo root
  (Mac-local, gitignored). Browse it with
  `.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db`. Check it with
  `.venv/bin/python scripts/mlflow_runs.py`. `--no-track` opts out, and every
  script runs without mlflow installed.
- **Always pass `--seed`** for anything you might want to repeat. It covers
  python/numpy/torch. The Node battle RNG is not seeded, so evals are
  independent draws by design.
- **A/Bs from tracked runs:** `.venv/bin/python scripts/bot_eval_ab.py --arm base=mlflow:<id> --arm cand=mlflow:<id>`.
- **Home-box runs** log to the Mac's store over a reverse tunnel
  (`docs/MULTI-MACHINE.md` → MLflow). That route hasn't been exercised yet.
- **CI:** `eval-smoke.yml` fails any push or PR where M7 greedy vs Random drops
  below 3,751/5,000. The gate is pre-registered in `EVALUATION-METHODOLOGY.md`
  Part 6. Change the doc first, dated, before touching the workflow or
  `scripts/eval_smoke_gate.py`.

---

## Current Work

None. M13 is merged and closed.

---

## Blockers

- **Home box offline (since at least 2026-09-24).** It blocks only optional work:
  M12 Phase 5 recovery, tracked re-evaluation of the M12 / M11-width
  checkpoints, and the first end-to-end test of the tunnel. Before any home-box
  command, follow the preflight in `CLAUDE.md`.
- **Upstream `test.yml` is red on every push** (`npm ci`: `package-lock.json`
  out of sync, missing `pg@8.22.0`). Pre-existing and left alone by decision.
  `eval-smoke.yml` uses `npm install`, so it's unaffected.

---

## Next Steps

Nothing is planned. Optional, if the home box comes back:
1. Recover M12 Phase 5 (recipe in `MILESTONES.md` → M12). This gives a
   standalone number.
2. Run tracked n=5,000 re-evaluations of `m12/ppo_step_5000005_final.pt`
   (`--format gen1ou`) and `m11_h128` / `m11_h512`, which exercises the tunnel
   recipe.

---

## Recently Completed

- **M13 — experiment tracking + CI (✅ 2026-09-24, PR #1).** Gate passed:
  - **(a)** 12 tracked n=5,000 re-evaluations, which replicate the ledger (M7
    greedy vs Random 76.8% vs 77.7%; m9p2c −8.2pp vs the ledger's −8.3pp).
  - **(b)** The CI gate was pre-registered before it was enabled.
  - **(c)** Backtest on PR #2: re-introducing the greedy-decoding bug went
    **red** (70.72% / 72.20%), and the revert went **green**.

  Record: `MILESTONES.md` → M13. Full detail: archive.
- **M12 closed (2026-09-24).** Phase 5 is recorded as **unrecovered, no number
  claimed**. Project status became "ML research closed; reproducibility & ops
  phase open".
- **M12 Phases 3 + 4 — PPO and the terminal gate (✅ 2026-08-06, home box).**
  Gate PASSED: 95.88% vs Random, 92.20% vs DamageFirst, n=5,000, raw sampled.
  Not comparable to M7's randbats numbers (mirror roster). The checkpoint is
  home box only.

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

---

## Housekeeping

- **This file is the timeline; `docs/WHERE-WE-ARE.md` is the orientation.**
  Surface WHERE-WE-ARE after a context clear.
- **Keep `Recently Completed` to the last three items.** Older entries belong in
  the `MILESTONES.md` ledger; run-level detail belongs in
  `docs/MILESTONES-ARCHIVE.md`.
