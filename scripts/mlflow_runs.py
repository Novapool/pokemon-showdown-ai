#!/usr/bin/env python3
"""
mlflow_runs.py — list the tracked runs and check each carries the M13 gate (a)
minimum: params, metrics, git SHA, and checkpoint path.

Exits 1 if any listed run is missing one of them, so the gate count is
"runs that pass", not "rows in the store".

Usage:
    .venv/bin/python scripts/mlflow_runs.py                 # all runs
    .venv/bin/python scripts/mlflow_runs.py --kind eval     # one kind
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_TAGS = ("git_sha", "checkpoint_path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--kind", default=None, help="train | bc | eval | ab")
    args = parser.parse_args()

    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI")
                            or f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
    runs = mlflow.search_runs(experiment_names=["pokemon-showdown"],
                              filter_string=f"tags.kind = '{args.kind}'" if args.kind else "",
                              output_format="list", order_by=["attributes.start_time ASC"])

    bad = 0
    for r in runs:
        tags, data = r.data.tags, r.data
        missing = [t for t in REQUIRED_TAGS if not tags.get(t)]
        if not data.params:
            missing.append("params")
        if not data.metrics:
            missing.append("metrics")
        bad += bool(missing)
        wr = data.metrics.get("win_rate")
        print(f"{r.info.run_id[:8]}  {tags.get('kind', '?'):5}  {r.info.status:8}  "
              f"{tags.get('git_sha', '?')[:9]}  "
              f"{tags.get('checkpoint_path', '-'):58}  "
              f"{tags.get('opponent', ''):14} {tags.get('decision_rule', ''):8} "
              f"{'' if wr is None else f'{wr:.3f}':>6}  n={int(data.metrics.get('battles', 0)) or '-'}"
              + (f"  MISSING {missing}" if missing else ""))

    print(f"\n{len(runs) - bad}/{len(runs)} runs carry params, metrics, git SHA and checkpoint path")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
