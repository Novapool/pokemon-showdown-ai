# In Progress — Pokemon Showdown AI Training

Last updated: 2026-09-24

---

## Where we are

**ML research is closed. One bounded ops milestone is open: M13, experiment
tracking + CI.** M12 closed 2026-09-24 (gate passed 2026-08-06; Phase 5 ladder
result unrecovered because the home box is offline). Full status:
`MILESTONES.md` → PROJECT STATUS. Orientation: `docs/WHERE-WE-ARE.md`.

**🚫 Do not add ML scope.** No new models, training arms, hypotheses or schema
work. M13 re-evaluates existing checkpoints and runs training only as a logging
smoke test. Anything interesting it surfaces about the agent is a note, not a
reason to reopen research.

**Shipping agent: unchanged M7 checkpoint** (`models/ppo/checkpoints/v3/ppo_step_5000002_final.pt`,
tracked in git). Raw policy on randbats, n=5,000: **77.7% vs Random greedy,
69.9% sampled**. That greedy number is the CI gate's baseline.

---

## Active Plan — M13: Experiment Tracking + CI

Gate (a)/(b)/(c) is pre-registered in `MILESTONES.md` → M13. Work happens on
branch `infra/tracking-ci`, merged by PR.

```
Step 1  close M12, refocus docs (master)            ✅ 2026-09-24
        │
2.1     MLflow store: SQLite on the Mac (mlflow.db, gitignored);
        home box logs via reverse SSH tunnel to `mlflow server` on the Mac
        │
2.2     models/tracking.py (no-op without mlflow / with --no-track)
        --seed + per-step metrics + run_meta.json in train.py / bc_pretrain_mlp.py
        PPOAgent.update() returns a metrics dict
        evaluate.py: tracked run + --json-out;  bot_eval_ab.py: --from-mlflow
        │
2.3     .github/workflows/eval-smoke.yml   (test.yml untouched)
        1. workflow_dispatch timing run, n=500 → throughput only
        2. threshold from the ledger baseline only (77.7%, n=5,000):
           α≤0.1% false-fail, ≥99% power vs −7.8pp → planned n≈1,000, ~73.6%
        3. pre-register in EVALUATION-METHODOLOGY.md, THEN enable the gate
        │
2.4     gate (c) BACKTEST: re-introduce the greedy-decoding bug (414966b14)
        on a draft PR → red; revert → green; close unmerged; record URLs
        │
2.5     ≥10 tracked re-evaluations, Mac-local, raw policy, n=2,000:
        {M7, m9seed, m9p2c, m9p2d, v3_valft} × {Random, DamageFirst} sampled
        + M7 greedy × 2  (+ one ~20k-step train.py logging smoke)
        │
2.6     merge PR; M13 ledger entry; docs + Obsidian note; code-simplifier pass
```

---

## Current Work

Step 1 done: M12 closed, M10–M12 full text moved to `docs/MILESTONES-ARCHIVE.md`,
M13 opened with its gate. Next: branch `infra/tracking-ci` and build 2.1–2.2.

---

## Blockers

- **Home box offline (2026-09-24)** — SSH times out, Tailscale stopped on the
  Mac. **Not blocking M13**: the ≥10 runs use Mac-local tracked checkpoints and
  CI runs on GitHub. It blocks only the M12 Phase 5 recovery and the optional
  M12 / M11-width re-evaluations. Before any home-box command, follow the
  preflight in `CLAUDE.md`.
- **Upstream `test.yml` is red on every push** (`npm ci`: `package-lock.json`
  out of sync — missing `pg@8.22.0`). Pre-existing, left alone by decision;
  `eval-smoke.yml` uses `npm install` instead so it doesn't inherit it.

---

## Next Steps

1. Branch `infra/tracking-ci`; add `mlflow` dependency; `models/tracking.py`.
2. Wire `--seed`, per-step metrics, `run_meta.json` into `train.py` and
   `bc_pretrain_mlp.py`; tracked runs + `--json-out` in `evaluate.py`;
   `--from-mlflow` in `bot_eval_ab.py`.
3. Log the ≥10 re-evaluations (2.5); check M7 sampled vs Random lands within CI
   of the ledger's 69.9%.
4. `eval-smoke.yml`: timing run → pre-register n + threshold → enable gate →
   green on the PR.
5. Backtest PR (2.4) → red → green. Record URLs.
6. Merge; close M13 per the lifecycle rule.

---

## Recently Completed

- **M12 closed (2026-09-24).** Phase 5 recorded as **unrecovered, no number
  claimed**. It ran on the home box from 2026-08-06 (`--run-id m12-ladder`,
  n=356 target), but the box is offline and no `m12-ladder` rows exist on the
  Mac. Recovery recipe in `MILESTONES.md` → M12. Project status changed from
  "bounded finish / archive" to "ML research closed; reproducibility & ops
  phase open".
- **M12 Phases 3 + 4 — PPO and the terminal gate (✅ 2026-08-06, home box).**
  5M steps, M7 recipe, warm-started from `bc_mlp_m12.pt`; training-mix win rate
  flat ~0.74 from 1M steps, no collapse. **Gate PASSED**, raw sampled policy,
  n=5,000/opponent: **95.88% vs Random [95.29, 96.40]; 92.20% vs DamageFirst
  [91.42, 92.91]**. Not comparable to M7's randbats numbers (mirror roster).
  The 0-draw count is unconfirmed. Checkpoint
  `models/ppo/checkpoints/m12/ppo_step_5000005_final.pt` is **home box only**.
- **M12 Phase 2 — BC retrain (✅ 2026-08-06, home box).** M7 recipe verbatim;
  53.1% randbats / 55.1% gen1ou val acc. A replication of M7 Job 4.1, not an
  improvement. `bc_mlp_m12.pt` is home box only.

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
